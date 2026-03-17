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


@dataclass
class ReactiveT4TConfig:
    concession_threshold: float = 0.03
    time_pressure: float = 0.85
    reserved_value: float = 0.2


class ReactiveT4TNegotiator(SAONegotiator):
    def __init__(self, name: str, cfg: Optional[ReactiveT4TConfig] = None):
        super().__init__(name=name)
        self.cfg = cfg or ReactiveT4TConfig()
        self._rv = self.cfg.reserved_value
        self._sorted_outcomes: List[Tuple[float, Any]] = []
        self._current_index = 0
        self._last_opponent_u = -1.0

    def _ensure_cache(self):
        if self._sorted_outcomes or not self.nmi:
            return

        vals_per_issue = []
        for issue in self.nmi.issues:
            if hasattr(issue, "all"):
                vals = list(issue.all)
            elif isinstance(issue.values, int):
                vals = list(range(issue.values))
            else:
                vals = list(issue.values)
            vals_per_issue.append(vals)

        scored = []
        for combo in itertools.product(*vals_per_issue):
            u = float(self.ufun(combo))
            if u >= self._rv:
                scored.append((u, combo))

        self._sorted_outcomes = sorted(scored, key=lambda x: x[0], reverse=True)

    def respond(self, state, offer=None, *args, **kwargs):
        self._ensure_cache()

        actual_offer = (
            offer if offer is not None else getattr(state, "current_offer", None)
        )
        if actual_offer is None:
            return ResponseType.REJECT_OFFER

        u_off = float(self.ufun(actual_offer))

        if u_off < self._rv:
            return ResponseType.REJECT_OFFER

        if self._last_opponent_u != -1.0:
            improvement = u_off - self._last_opponent_u
            if improvement > self.cfg.concession_threshold:
                self._current_index = min(
                    len(self._sorted_outcomes) - 1, self._current_index + 1
                )

        self._last_opponent_u = u_off

        my_current_target_u = self._sorted_outcomes[self._current_index][0]
        if u_off >= my_current_target_u:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state, *args, **kwargs):
        self._ensure_cache()

        t = state.relative_time
        if t > self.cfg.time_pressure:
            progress = (t - self.cfg.time_pressure) / (1.0 - self.cfg.time_pressure)
            target_idx = int(progress * (len(self._sorted_outcomes) - 1))
            self._current_index = max(self._current_index, target_idx)

        return self._sorted_outcomes[self._current_index][1]


if __name__ == "__main__":
    MAX_PRICE = 99
    MAX_QTY = 10
    issues = [
        make_issue(values=MAX_PRICE + 1, name="price"),
        make_issue(values=list(range(1, MAX_QTY + 1)), name="quantity"),
    ]
    session = SAOMechanism(issues=issues, n_steps=100)

    b_ufun = PriceQuantityUFun(MAX_PRICE, MAX_QTY, prefer_low_price=True, reserved_value=0.3)
    s_ufun = PriceQuantityUFun(MAX_PRICE, MAX_QTY, prefer_low_price=False, reserved_value=0.3)

    session.add(ReactiveT4TNegotiator("reactivet4t"), ufun=b_ufun)
    session.add(TimeBasedConcedingNegotiator(name="baseline_seller"), ufun=s_ufun)

    result = session.run()
    if result.agreement:
        print(f"Agreement: {result.agreement} at step {result.step}")
    else:
        print("Negotiation failed (timeout).")
    session.plot()
