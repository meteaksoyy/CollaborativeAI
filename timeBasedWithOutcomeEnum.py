# microAgent_high_performance.py
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from negmas import SAOMechanism, make_issue, TimeBasedConcedingNegotiator
from negmas.sao import SAONegotiator, ResponseType


# ----------------------------
# 1. Authentic Price Utility
# ----------------------------
class PriceUFun:
    """
    Calculates linear utility based on price.
    Buyer: 1.0 at Price 0, 0.0 at Price 99.
    """

    def __init__(self, max_price: int, prefer_low: bool):
        self.max_price = int(max_price)
        self.prefer_low = bool(prefer_low)

    def __call__(self, outcome) -> float:
        if outcome is None:
            return 0.0
        p = int(outcome[0])
        x = p / self.max_price if self.max_price > 0 else 0.0
        u = (1.0 - x) if self.prefer_low else x
        return max(0.0, min(1.0, float(u)))


# ----------------------------
# 2. Competitive MiCRO Agent
# ----------------------------
@dataclass
class TimeBasedWithOutcomeEnumConfig:
    reserved_value: float = 0.40  # Raised floor: prevents getting squeezed to 0.30
    power: float = 6.0  # Extreme Boulware: stays at 1.0 utility for ~70% of time
    accept_slack: float = 0.01  # Minimal slack to ensure high-quality deals
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
        """Authentic MiCRO: Pre-calculates and sorts the utility of all outcomes."""
        if self._sorted_outcomes or self.ufun is None:
            return

        issue = self.nmi.issues[0]
        # Dynamically handle different NegMAS versions/issue types
        if hasattr(issue, "all"):
            vals = list(issue.all)
        elif hasattr(issue, "values"):
            vals = list(issue.values)
        else:
            vals = list(range(int(issue.min_value), int(issue.max_value) + 1))

        scored = []
        for v in vals:
            u = float(self.ufun((v,)))
            # Only cache outcomes that are better than our walk-away value
            if u >= self.cfg.reserved_value:
                scored.append((u, (v,)))

        # Sort best to worst
        self._sorted_outcomes = sorted(scored, key=lambda x: x[0], reverse=True)

    def _get_target(self, t: float) -> float:
        """Calculates the current 'Minimum Acceptable Utility' based on time."""
        self._ensure_outcome_cache()
        if not self._sorted_outcomes:
            return self.cfg.reserved_value

        max_u = self._sorted_outcomes[0][0]
        rv = self.cfg.reserved_value

        # Boulware Formula: Target stays high, then drops to RV at the very end
        # Target = RV + (MaxU - RV) * (1 - t)^power
        return rv + (max_u - rv) * math.pow((1.0 - t), self.cfg.power)

    def respond(self, state, offer=None, source=None):
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        # Pull offer from state if missing (avoids 'Round 2' logic errors)
        current_offer = offer if offer is not None else state.current_offer
        if current_offer is None:
            return ResponseType.REJECT_OFFER

        u_offer = float(self.ufun(current_offer))
        t = self._t(state)
        target = self._get_target(t)

        # Acceptance logic: Is the offer better than our current target?
        if u_offer >= (target - self.cfg.accept_slack):
            return ResponseType.ACCEPT_OFFER

        # Logging to track the 'Target vs Offer' battle
        if self.cfg.debug_every and state.step % self.cfg.debug_every == 0:
            print(
                f"[Step {state.step:03}] T: {t:.2f} | Offer: {current_offer[0]} | U(Offer): {u_offer:.3f} | Target: {target:.3f}"
            )

        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        self._ensure_outcome_cache()
        t = self._t(state)
        target = self._get_target(t)

        # Propose the 'worst' outcome we still find acceptable.
        # This signals flexibility to the opponent without actually giving up utility.
        for u, o in reversed(self._sorted_outcomes):
            if u >= target:
                return o
        return self._sorted_outcomes[0][1]


# ----------------------------
# 3. Execution & Validation
# ----------------------------
if __name__ == "__main__":
    MAX_PRICE = 99
    issues = [make_issue(values=MAX_PRICE + 1, name="price")]
    session = SAOMechanism(issues=issues, n_steps=100)

    buyer_ufun = PriceUFun(max_price=MAX_PRICE, prefer_low=True)
    seller_ufun = PriceUFun(max_price=MAX_PRICE, prefer_low=False)

    # micro_buyer is now highly competitive (power=6.0)
    micro_buyer = TimeBasedWithOutcomeEnumNegotiator(
        "micro_buyer", TimeBasedWithOutcomeEnumConfig(power=6.0, reserved_value=0.40)
    )
    baseline_seller = TimeBasedConcedingNegotiator(name="baseline_seller")

    session.add(micro_buyer, ufun=buyer_ufun)
    session.add(baseline_seller, ufun=seller_ufun)

    result = session.run()

    print("-" * 30)
    if result.agreement:
        print(f"SUCCESS: Price {result.agreement[0]} | Round: {result.step}")
        print(f"Buyer Utility: {buyer_ufun(result.agreement):.3f}")
    else:
        print("FAILED: No agreement (Timed out).")
    print("-" * 30)

    session.plot()
