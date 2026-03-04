# annaAgent_super_simple.py
# Super simple full runnable NegMAS SAO example:
# - Utility functions (no ComplexUtility class)
# - VerySimpleNegotiator (time-based accept + random propose)
# - SAOMechanism run

from negmas import SAOMechanism, make_issue
from negmas.sao import SAONegotiator, ResponseType
import math


# =====================================================
# Domain (issues)
# =====================================================

issues = [
    make_issue(name="price", values=10),     # 0..9
    make_issue(name="quantity", values=5),   # 0..4
]


# =====================================================
# Simple utility functions (callables with reserved_value)
# =====================================================

class BuyerUtility:
    reserved_value = 0.55  # minimum acceptable utility for the buyer

    def __call__(self, offer) -> float:
        p, q = offer  # offer is a tuple (price, quantity)

        # buyer prefers low price 
        price_u = 1.0 - (p / 9.0)

        # buyer prefers higher quantity with diminishing returns
        qty_u = math.sqrt(q / 4.0) if q > 0 else 0.0

        # weighted sum
        u = 0.6 * price_u + 0.4 * qty_u

        # penalty if total cost too high
        budget = 18
        total_cost = p * q
        if total_cost > budget:
            u -= 0.03 * (total_cost - budget)

        # clamp to [0,1]
        return max(0.0, min(1.0, u))


class SellerUtility:
    reserved_value = 0.60  # minimum acceptable utility for the seller

    def __call__(self, offer) -> float:
        p, q = offer

        # seller likes high price, scaled to [0,1]
        price_u = math.sqrt(p / 9.0) if p > 0 else 0.0

        # seller prefers low quantity, scaled to [0,1]
        qty_u = 1.0 - (q / 4.0)

        # weighted sum
        u = 0.7 * price_u + 0.3 * qty_u

        # penalty for low price with high quantity
        if q >= 3 and p <= 3:
            u -= 0.25

        # small bonus for revenue
        u += 0.1 * ((p * q) / (9.0 * 4.0))

        return max(0.0, min(1.0, u))


buyer_ufun = BuyerUtility()
seller_ufun = SellerUtility()


# =====================================================
# Very simple negotiator
# =====================================================

class VerySimpleNegotiator(SAONegotiator):
    """
    - Accept if offer utility >= a threshold that decreases over time.
    - Propose random outcomes (no search, no opponent model).
    """

    def __init__(self, name: str, start_thr: float = 0.95, end_thr: float = 0.60):
        super().__init__(name=name)
        self.start_thr = start_thr
        self.end_thr = end_thr

    def _t(self, state) -> float:
        t = getattr(state, "relative_time", 0.0)
        return max(0.0, min(1.0, float(t)))

    def _threshold(self, state) -> float:
        # linear concession
        t = self._t(state)
        return self.start_thr + (self.end_thr - self.start_thr) * t

    def respond(self, state, offer=None, source=None):
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        if offer is None:
            offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER

        u = float(self.ufun(offer))
        rv = float(getattr(self.ufun, "reserved_value", 0.0))
        thr = max(rv, self._threshold(state))

        return ResponseType.ACCEPT_OFFER if u >= thr else ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        if self.nmi is None:
            return None
        return tuple(self.nmi.random_outcome())


# =====================================================
# Negotiation
# =====================================================

buyer = VerySimpleNegotiator("buyer_simple", start_thr=0.95, end_thr=0.55)
seller = VerySimpleNegotiator("seller_simple", start_thr=0.95, end_thr=0.60)

session = SAOMechanism(issues=issues, n_steps=100)
session.add(buyer, ufun=buyer_ufun)
session.add(seller, ufun=seller_ufun)

result = session.run()
print(f"Agreement: {result.agreement}, Rounds: {result.step}")

try:
    session.plot()
except Exception as e:
    print("Plot skipped:", e)