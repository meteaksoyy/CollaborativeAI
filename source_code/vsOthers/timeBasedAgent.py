import math

from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState


class TimeBasedAgent(SAONegotiator):
    """Time-based conceding negotiator for NegMAS SAO sessions."""

    def __init__(
        self,
        *args,
        reservation_ratio: float = 0.4,
        beta: float = 1.0,
        concession_curve: str = "poly",
        reverse_log_k: float = 9.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio
        self.beta = beta
        self.concession_curve = concession_curve
        self.reverse_log_k = reverse_log_k

    def _relative_time(self, state: SAOState | None = None) -> float:
        if state is not None:
            t = getattr(state, "relative_time", None)
            if t is not None:
                return float(t)
        if self.nmi is not None:
            t = getattr(self.nmi.state, "relative_time", None)
            if t is not None:
                return float(t)
        return 0.0

    def _active_ufun(self):
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            ufun = getattr(self, "preferences", None)
        return ufun

    def _utility_range(self):
        ufun = self._active_ufun()
        if ufun is None:
            return 0.0, 0.0
        u_max = float(ufun.max())
        u_worst = float(ufun.min())
        u_min = u_worst + self.reservation_ratio * (u_max - u_worst)
        return u_min, u_max

    def _target_utility(self, t: float) -> float:
        u_min, u_max = self._utility_range()
        t = max(0.0, min(1.0, float(t)))

        if self.concession_curve == "reverse_log":
            k = max(float(self.reverse_log_k), 1e-9)
            progress = 1.0 - math.log1p(k * (1.0 - t)) / math.log1p(k)
        else:
            progress = t**self.beta

        return u_max - (u_max - u_min) * progress

    def _best_offer_above(self, threshold: float) -> Outcome | None:
        if self.nmi is None:
            return None
        ufun = self._active_ufun()
        if ufun is None:
            return None

        best_gap, best_u, best_o = float("inf"), -1e9, None
        for outcome in self.nmi.outcomes:
            u = float(ufun(outcome))
            if u < threshold:
                continue
            gap = u - threshold
            if gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and u > best_u):
                best_gap, best_u, best_o = gap, u, outcome

        return best_o if best_o is not None else ufun.best()

    def propose(self, state: SAOState) -> Outcome | None:
        t = self._relative_time(state)
        target = self._target_utility(t)
        return self._best_offer_above(target)

    def respond(self, state: SAOState) -> ResponseType:
        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))
        t = self._relative_time(state)
        target = self._target_utility(t)

        if offered_u >= target:
            return ResponseType.ACCEPT_OFFER

        u_min, _ = self._utility_range()
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
