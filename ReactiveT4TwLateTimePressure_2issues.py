from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from negmas import SAOMechanism, make_issue, TimeBasedConcedingNegotiator
from negmas.sao import SAONegotiator, ResponseType

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

