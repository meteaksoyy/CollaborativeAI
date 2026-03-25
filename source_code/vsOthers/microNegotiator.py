from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState

class MicroNegotiator(SAONegotiator):
    """MiCRO: simple step-down negotiator for NegMAS SAO sessions."""

    def __init__(self, *args, reservation_ratio: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio

        self._sorted_outcomes: list[Outcome] = []
        self._step: int = 0
        self._initialized: bool = False

    def _active_ufun(self):
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            ufun = getattr(self, "preferences", None)
        return ufun

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

    def _utility_range(self):
        ufun = self._active_ufun()
        if ufun is None:
            return 0.0, 0.0
        u_max = float(ufun.max())
        u_worst = float(ufun.min())
        u_min = u_worst + self.reservation_ratio * (u_max - u_worst)
        return u_min, u_max

    def _init_outcomes(self):
        if self._initialized or self.nmi is None:
            return
        ufun = self._active_ufun()
        if ufun is None:
            return

        self._sorted_outcomes = sorted(
            self.nmi.outcomes,
            key=lambda o: float(ufun(o)),
            reverse=True,
        )
        self._initialized = True

    def _current_offer(self) -> Outcome | None:
        if not self._sorted_outcomes:
            return None
        idx = min(self._step, len(self._sorted_outcomes) - 1)
        return self._sorted_outcomes[idx]

    def propose(self, state: SAOState) -> Outcome | None:
        self._init_outcomes()
        offer = self._current_offer()
        self._step += 1
        return offer

    def respond(self, state: SAOState) -> ResponseType:
        self._init_outcomes()

        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))

        current = self._current_offer()
        if current is not None and offered_u >= float(ufun(current)):
            return ResponseType.ACCEPT_OFFER

        u_min, _ = self._utility_range()
        t = self._relative_time(state)
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
