from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from negmas import SAOMechanism, make_issue, TimeBasedConcedingNegotiator
from negmas.sao import SAONegotiator, ResponseType


class PriceQuantityUFun:
    """
    Two-issue utility: price (0..max_price) and quantity (1..max_qty).
    Buyer: low price is good, high quantity is good.
    Seller: high price is good, low quantity is good.
    """

    def __init__(
        self,
        max_price: int,
        max_qty: int,
        prefer_low_price: bool,
        price_weight: float = 0.6,
        reserved_value: float = 0.3,
    ):
        self.max_price = max_price
        self.max_qty = max_qty
        self.prefer_low_price = prefer_low_price
        self.price_weight = price_weight
        self.qty_weight = 1.0 - price_weight
        self.reserved_value = reserved_value

    def __call__(self, outcome) -> float:
        if outcome is None:
            return self.reserved_value
        price = int(outcome[0])
        qty = int(outcome[1])
        u_price = (
            (1.0 - price / self.max_price)
            if self.prefer_low_price
            else (price / self.max_price)
        )
        u_qty = (
            (qty / self.max_qty)
            if self.prefer_low_price
            else (1.0 - qty / self.max_qty)
        )
        return float(self.price_weight * u_price + self.qty_weight * u_qty)


def issue_values(issue) -> List[Any]:
    if hasattr(issue, "all") and issue.all is not None:
        vals = list(issue.all)
        if len(vals) > 0:
            return vals

    v = getattr(issue, "values", None)
    if isinstance(v, int):
        return list(range(v))
    if isinstance(v, (list, tuple, range)):
        vals = list(v)
        if len(vals) > 0:
            return vals

    mn = getattr(issue, "min_value", None)
    mx = getattr(issue, "max_value", None)
    if isinstance(mn, int) and isinstance(mx, int) and mx >= mn:
        return list(range(mn, mx + 1))

    n = getattr(issue, "n_values", None) or getattr(issue, "cardinality", None)
    if isinstance(n, int):
        return list(range(n))

    raise TypeError(
        f"Cannot infer discrete values for issue {getattr(issue, 'name', issue)}"
    )


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class MicroConfig:
    reserved_value: float = 0.30
    power: float = 2.5
    accept_slack: float = 0.02
    debug_every: int = 10


class MicroNegotiator(SAONegotiator):
    def __init__(self, name: str, cfg: Optional[MicroConfig] = None):
        super().__init__(name=name)
        self.cfg = cfg or MicroConfig()
        self._scored_desc: List[Tuple[float, Tuple[Any, ...]]] = []
        self._printed = False

    def on_preferences_changed(self, changes=None):
        self._scored_desc = []
        self._printed = False

    def _t(self, state) -> float:
        t = getattr(state, "relative_time", None)
        return clamp01(float(t)) if t is not None else 0.0

    def _rv(self) -> float:
        return float(self.cfg.reserved_value)

    def _ensure_cache(self):
        if self._scored_desc or self.ufun is None or self.nmi is None:
            return

        issues = getattr(self.nmi, "issues", None) or []
        vals_per_issue = [issue_values(iss) for iss in issues]

        if not self._printed:
            counts = [len(v) for v in vals_per_issue]
            print(f"[{self.name}] extracted_values_per_issue={counts}")
            self._printed = True

        scored: List[Tuple[float, Tuple[Any, ...]]] = []
        for o in itertools.product(*vals_per_issue):
            ot = tuple(o)
            scored.append((float(self.ufun(ot)), ot))

        scored.sort(key=lambda x: x[0], reverse=True)
        self._scored_desc = scored

    def _acceptable_prefix_len(self) -> int:
        rv = self._rv()
        k = 0
        for u, _ in self._scored_desc:
            if u >= rv:
                k += 1
            else:
                break
        return k

    def _target(self, t: float) -> float:
        self._ensure_cache()
        rv = self._rv()
        maxu = self._scored_desc[0][0] if self._scored_desc else 1.0
        return rv + (maxu - rv) * ((1.0 - t) ** float(self.cfg.power))

    def _planned_offer(self, t: float) -> Tuple[Tuple[Any, ...], float]:
        self._ensure_cache()
        if not self._scored_desc:
            return (0, 1), 0.0

        k = self._acceptable_prefix_len()
        if k <= 0:
            u, o = self._scored_desc[0]
            return o, float(u)

        c = 1.0 - (1.0 - t) ** float(self.cfg.power)
        idx = int(round(c * (k - 1)))
        idx = max(0, min(k - 1, idx))
        u, o = self._scored_desc[idx]
        return o, float(u)

    def _extract_offer(self, state, offer):
        if offer is None:
            offer = getattr(state, "current_offer", None)
        if isinstance(offer, list):
            offer = tuple(offer)
        return offer if isinstance(offer, tuple) else None

    def respond(self, state, offer=None, source=None, *args, **kwargs):
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer = self._extract_offer(state, offer)
        if offer is None:
            return ResponseType.REJECT_OFFER

        t = self._t(state)
        rv = self._rv()
        u_offer = float(self.ufun(offer))
        if u_offer < rv:
            return ResponseType.REJECT_OFFER

        target = max(rv, self._target(t))
        floor = max(rv, target - float(self.cfg.accept_slack))

        step = getattr(state, "step", 0)
        if self.cfg.debug_every and step % self.cfg.debug_every == 0:
            planned_o, planned_u = self._planned_offer(t)
            print(
                f"[{self.name}.respond] step={step:03d} t={t:.2f} offer={offer} u={u_offer:.3f} "
                f"target={target:.3f} floor={floor:.3f} planned={planned_o} planned_u={planned_u:.3f}"
            )

        return (
            ResponseType.ACCEPT_OFFER if u_offer >= floor else ResponseType.REJECT_OFFER
        )

    def propose(self, state, dest=None, *args, **kwargs):
        if self.ufun is None or self.nmi is None:
            return None
        t = self._t(state)
        o, u = self._planned_offer(t)

        step = getattr(state, "step", 0)
        if self.cfg.debug_every and step % self.cfg.debug_every == 0:
            print(f"[{self.name}.propose] step={step:03d} t={t:.2f} -> {o} u={u:.3f}")
        return o


if __name__ == "__main__":
    MAX_PRICE = 99
    MAX_QTY = 10
    issues = [
        make_issue(name="price", values=MAX_PRICE + 1),
        make_issue(name="quantity", values=list(range(1, MAX_QTY + 1))),
    ]
    session = SAOMechanism(issues=issues, n_steps=100)

    buyer_ufun = PriceQuantityUFun(max_price=MAX_PRICE, max_qty=MAX_QTY, prefer_low_price=True)
    seller_ufun = PriceQuantityUFun(max_price=MAX_PRICE, max_qty=MAX_QTY, prefer_low_price=False)

    micro_buyer = MicroNegotiator(
        "micro_buyer",
        MicroConfig(reserved_value=0.30, power=2.5, accept_slack=0.02, debug_every=10),
    )
    baseline_seller = TimeBasedConcedingNegotiator(name="baseline_seller")

    session.add(micro_buyer, ufun=buyer_ufun)
    session.add(baseline_seller, ufun=seller_ufun)

    result = session.run()
    print("Agreement:", result.agreement, "Rounds:", result.step)
    session.plot()
