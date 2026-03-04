from __future__ import annotations
import itertools
import math
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


@dataclass
class TimeBasedWithOutcomeEnumConfig:
    reserved_value: float = 0.40
    power: float = 6.0
    accept_slack: float = 0.01
    debug_every: int = 5


class TimeBasedWithOutcomeEnumNegotiator(SAONegotiator):
    def __init__(self, name: str, cfg: Optional[TimeBasedWithOutcomeEnumConfig] = None):
        super().__init__(name=name)
        self.cfg = cfg or TimeBasedWithOutcomeEnumConfig()
        self._sorted_outcomes: List[Tuple[float, Tuple[Any, ...]]] = []

    def _t(self, state) -> float:
        t = getattr(state, "relative_time", None)
        return max(0.0, min(1.0, float(t))) if t is not None else 0.0

    def _ensure_outcome_cache(self):
        if self._sorted_outcomes or self.ufun is None:
            return

        vals_per_issue = []
        for issue in self.nmi.issues:
            if hasattr(issue, "all"):
                vals = list(issue.all)
            elif hasattr(issue, "values"):
                v = issue.values
                if isinstance(v, int):
                    vals = list(range(v))
                else:
                    vals = list(v)
            else:
                vals = list(range(int(issue.min_value), int(issue.max_value) + 1))
            vals_per_issue.append(vals)

        scored = []
        for combo in itertools.product(*vals_per_issue):
            u = float(self.ufun(combo))
            if u >= self.cfg.reserved_value:
                scored.append((u, combo))

        self._sorted_outcomes = sorted(scored, key=lambda x: x[0], reverse=True)

    def _get_target(self, t: float) -> float:
        self._ensure_outcome_cache()
        if not self._sorted_outcomes:
            return self.cfg.reserved_value
        max_u = self._sorted_outcomes[0][0]
        rv = self.cfg.reserved_value
        return rv + (max_u - rv) * math.pow((1.0 - t), self.cfg.power)

    def respond(self, state, offer=None, source=None):
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        current_offer = offer if offer is not None else state.current_offer
        if current_offer is None:
            return ResponseType.REJECT_OFFER

        u_offer = float(self.ufun(current_offer))
        t = self._t(state)
        target = self._get_target(t)

        if u_offer >= (target - self.cfg.accept_slack):
            return ResponseType.ACCEPT_OFFER

        if self.cfg.debug_every and state.step % self.cfg.debug_every == 0:
            print(
                f"[Step {state.step:03}] T: {t:.2f} | Offer: {current_offer} | U(Offer): {u_offer:.3f} | Target: {target:.3f}"
            )

        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        self._ensure_outcome_cache()
        t = self._t(state)
        target = self._get_target(t)

        for u, o in reversed(self._sorted_outcomes):
            if u >= target:
                return o
        return self._sorted_outcomes[0][1]


if __name__ == "__main__":
    MAX_PRICE = 99
    MAX_QTY = 10
    issues = [
        make_issue(values=MAX_PRICE + 1, name="price"),
        make_issue(values=list(range(1, MAX_QTY + 1)), name="quantity"),
    ]
    session = SAOMechanism(issues=issues, n_steps=100)

    buyer_ufun = PriceQuantityUFun(max_price=MAX_PRICE, max_qty=MAX_QTY, prefer_low_price=True)
    seller_ufun = PriceQuantityUFun(max_price=MAX_PRICE, max_qty=MAX_QTY, prefer_low_price=False)

    timebased_buyer = TimeBasedWithOutcomeEnumNegotiator(
        "timebased_buyer", TimeBasedWithOutcomeEnumConfig(power=6.0, reserved_value=0.40)
    )
    baseline_seller = TimeBasedConcedingNegotiator(name="baseline_seller")

    session.add(timebased_buyer, ufun=buyer_ufun)
    session.add(baseline_seller, ufun=seller_ufun)

    result = session.run()
    print("-" * 30)
    if result.agreement:
        print(f"SUCCESS: {result.agreement} | Round: {result.step}")
        print(f"Buyer Utility: {buyer_ufun(result.agreement):.3f}")
    else:
        print("FAILED: No agreement (Timed out).")
    print("-" * 30)
    session.plot()
