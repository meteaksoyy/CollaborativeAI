# authentic_micro_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from negmas import SAOMechanism, make_issue, TimeBasedConcedingNegotiator
from negmas.sao import SAONegotiator, ResponseType


class PriceUFun:
    def __init__(self, max_price: int, prefer_low: bool, reserved_value: float = 0.3):
        self.max_price = max_price
        self.prefer_low = prefer_low
        self.reserved_value = reserved_value

    def __call__(self, outcome) -> float:
        if outcome is None:
            return self.reserved_value
        p = int(outcome[0])
        x = p / self.max_price
        u = (1.0 - x) if self.prefer_low else x
        return float(u)


@dataclass
class ReactiveT4TConfig:
    # Behavioral threshold: only concede if opponent moves this much
    concession_threshold: float = 0.03
    # Hard time limit to start 'panic' concessions
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
        issue = self.nmi.issues[0]

        # Proper value extraction for different Issue types
        if hasattr(issue, "all"):
            vals = list(issue.all)
        elif isinstance(issue.values, int):
            vals = list(range(issue.values))
        else:
            vals = list(issue.values)

        scored = []
        for v in vals:
            u = float(self.ufun((v,)))
            if u >= self._rv:
                scored.append((u, (v,)))

        # Ranked outcomes: Heart of the MiCRO strategy
        self._sorted_outcomes = sorted(scored, key=lambda x: x[0], reverse=True)

    def respond(self, state, offer=None, *args, **kwargs):
        self._ensure_cache()

        # Robust offer extraction from either argument or state
        actual_offer = (
            offer if offer is not None else getattr(state, "current_offer", None)
        )
        if actual_offer is None:
            return ResponseType.REJECT_OFFER

        u_off = float(self.ufun(actual_offer))

        # 1. Immediate rejection if below walk-away value
        if u_off < self._rv:
            return ResponseType.REJECT_OFFER

        # 2. Behavioral Update: Concede only if opponent did
        if self._last_opponent_u != -1.0:
            improvement = u_off - self._last_opponent_u
            if improvement > self.cfg.concession_threshold:
                # Increment index to move down our sorted list (concede)
                self._current_index = min(
                    len(self._sorted_outcomes) - 1, self._current_index + 1
                )

        self._last_opponent_u = u_off

        # 3. Acceptance Region Check
        my_current_target_u = self._sorted_outcomes[self._current_index][0]
        if u_off >= my_current_target_u:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state, *args, **kwargs):
        self._ensure_cache()

        # Secondary time-based pressure to avoid timeout failure
        t = state.relative_time
        if t > self.cfg.time_pressure:
            progress = (t - self.cfg.time_pressure) / (1.0 - self.cfg.time_pressure)
            target_idx = int(progress * (len(self._sorted_outcomes) - 1))
            self._current_index = max(self._current_index, target_idx)

        return self._sorted_outcomes[self._current_index][1]


if __name__ == "__main__":
    MAX_PRICE = 99
    issues = [make_issue(values=MAX_PRICE + 1, name="price")]
    session = SAOMechanism(issues=issues, n_steps=100)

    # Standard utilities for Buyer and Seller
    b_ufun = PriceUFun(MAX_PRICE, prefer_low=True, reserved_value=0.3)
    s_ufun = PriceUFun(MAX_PRICE, prefer_low=False, reserved_value=0.3)

    session.add(ReactiveT4TNegotiator("reactivet4t"), ufun=b_ufun)
    session.add(TimeBasedConcedingNegotiator(name="baseline_seller"), ufun=s_ufun)

    result = session.run()
    if result.agreement:
        print(f"Agreement reached: {result.agreement} at step {result.step}")
    else:
        print("Negotiation failed (timeout).")
    session.plot()
