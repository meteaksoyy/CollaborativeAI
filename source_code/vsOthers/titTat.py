from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState


class TitForTatAgent(SAONegotiator):
    """Tit-for-Tat negotiator for NegMAS SAO sessions."""

    def __init__(
        self,
        *args,
        reservation_ratio: float = 0.4,
        alpha: float = 1.0,
        opening_utility: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio
        self.alpha = alpha
        self.opening_utility = opening_utility

        self._opponent_history = []
        self._my_current_target = None

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._opponent_history = []
        _, u_max = self._utility_range()
        if self.opening_utility is not None:
            self._my_current_target = self.opening_utility
        else:
            self._my_current_target = u_max

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

    def respond(self, state: SAOState) -> ResponseType:
        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))
        self._opponent_history.append(offered_u)

        if self._my_current_target is None:
            self.on_negotiation_start(state)

        if offered_u >= self._my_current_target:
            return ResponseType.ACCEPT_OFFER

        u_min, _ = self._utility_range()
        t = float(getattr(state, "relative_time", 0.0))
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state: SAOState) -> Outcome | None:
        u_min, u_max = self._utility_range()

        if self._my_current_target is None:
            self.on_negotiation_start(state)

        self._my_current_target -= (u_max - u_min) / (self.nmi.n_steps * 2)

        if len(self._opponent_history) >= 2:
            latest_u = self._opponent_history[-1]
            previous_u = self._opponent_history[-2]

            concession = latest_u - previous_u

            if concession > 0:
                self._my_current_target -= (self.alpha * concession)

        self._my_current_target = max(u_min, min(u_max, self._my_current_target))

        return self._best_offer_above(self._my_current_target)
