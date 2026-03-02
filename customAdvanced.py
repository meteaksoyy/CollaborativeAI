# customAdvanced_fixed.py
# Fixes:
# 1) Make respond() signature compatible with NegMAS SAO variants by accepting an explicit offer.
# 2) Make propose() signature permissive (works whether dest is passed or not).
# 3) Add small defensive handling for initial/None offers.

from negmas import SAOMechanism, make_issue
from negmas.sao import SAONegotiator, ResponseType
from typing import Callable, Dict, Sequence, Any, Optional, List, Tuple
import math
import random


# =====================================================
# Utility: nonlinear per-issue + interactions + normalization
# =====================================================

class ComplexUtility:
    def __init__(
        self,
        issues: Sequence[Any],
        per_issue: Dict[str, Callable[[Any], float]],
        weights: Optional[Dict[str, float]] = None,
        interactions: Optional[List[Callable[[Dict[str, Any]], float]]] = None,
        base: Optional[Callable[[Dict[str, Any]], float]] = None,
        reserved_value: float = 0.0,
        normalize: bool = True,
        umin: float = 0.0,
        umax: float = 1.0,
    ):
        self.issues = list(issues)
        self.per_issue = per_issue
        self.weights = weights or {iss.name: 1.0 for iss in self.issues}
        self.interactions = interactions or []
        self.base = base or (lambda _: 0.0)
        self.reserved_value = reserved_value
        self.normalize = normalize
        self.umin = umin
        self.umax = umax

    def __call__(self, outcome) -> float:
        values = {self.issues[i].name: outcome[i] for i in range(len(self.issues))}

        u = float(self.base(values))

        for iss in self.issues:
            name = iss.name
            w = float(self.weights.get(name, 1.0))
            u += w * float(self.per_issue[name](values[name]))

        for g in self.interactions:
            u += float(g(values))

        if self.normalize:
            if self.umax <= self.umin:
                raise ValueError("Invalid normalization bounds: umax must be > umin")
            u = (u - self.umin) / (self.umax - self.umin)
            u = max(0.0, min(1.0, u))

        return u


# =====================================================
# Custom negotiator: adaptive aspiration + true counter-offers
# =====================================================

def _issue_values(issue) -> List[int]:
    """
    Supports make_issue(name, values=int) and make_issue(name, values=list).
    Returns discrete values.
    """
    v = getattr(issue, "values", None)
    if isinstance(v, int):
        return list(range(v))
    if isinstance(v, (list, tuple, range)):
        return list(v)
    n = getattr(issue, "n_values", None) or getattr(issue, "cardinality", None)
    if isinstance(n, int):
        return list(range(n))
    raise TypeError(f"Cannot infer values for issue {getattr(issue, 'name', issue)}")


def _normalized_l1_distance(a: Tuple[int, ...], b: Tuple[int, ...], max_steps: List[int]) -> float:
    s = 0.0
    for i in range(len(a)):
        denom = max(1, max_steps[i])
        s += abs(a[i] - b[i]) / denom
    return s / max(1, len(a))


class AdaptiveCounterNegotiator(SAONegotiator):
    """
    - Acceptance threshold (aspiration) declines over time towards RV.
    - Concession speed adapts based on whether opponent is improving offers to us.
    - Proposals are true counter-offers: local search around the opponent's last offer.
    """

    def __init__(
        self,
        name: str,
        max_aspiration: float = 1.0,
        base_power: float = 2.0,            # larger => slower early concession
        adapt_strength: float = 0.8,        # how much opponent behavior changes concession speed
        accept_slack: float = 0.02,         # accept slightly below threshold to reduce deadlocks
        local_steps: int = 200,             # local search iterations per proposal
        explore_prob: float = 0.10,         # occasional random move to escape local traps
        distance_penalty_end: float = 0.50  # how much we value staying close to last offer at end
    ):
        super().__init__(name=name)
        self.max_aspiration = max_aspiration
        self.base_power = base_power
        self.adapt_strength = adapt_strength
        self.accept_slack = accept_slack
        self.local_steps = local_steps
        self.explore_prob = explore_prob
        self.distance_penalty_end = distance_penalty_end

        self._issue_vals: List[List[int]] = []
        self._max_steps: List[int] = []

        # Opponent concession tracking (utility-to-us trend)
        self._last_u: Optional[float] = None
        self._ema_improve: float = 0.0

    def on_preferences_changed(self, changes=None):
        self._last_u = None
        self._ema_improve = 0.0

    def _t(self, state) -> float:
        t = getattr(state, "relative_time", None)
        if t is None:
            return 0.0
        return max(0.0, min(1.0, float(t)))

    def _ensure_issue_cache(self):
        if self._issue_vals:
            return
        if self.nmi is None:
            return
        issues = getattr(self.nmi, "issues", None)
        if issues is None:
            return
        self._issue_vals = [_issue_values(iss) for iss in issues]
        self._max_steps = [max(vals) - min(vals) if vals else 1 for vals in self._issue_vals]

    def _update_opponent_behavior(self, offer):
        if offer is None or self.ufun is None:
            return
        u = float(self.ufun(offer))
        if self._last_u is None:
            self._last_u = u
            return
        improve = u - self._last_u
        self._last_u = u

        # Track positive improvements (ignore/decay negatives)
        if improve > 0:
            self._ema_improve = 0.85 * self._ema_improve + 0.15 * improve
        else:
            self._ema_improve = 0.95 * self._ema_improve

    def _concession_power(self) -> float:
        c = max(0.0, min(1.0, self._ema_improve * 8.0))
        mult = 1.0 + self.adapt_strength * (2.0 * c - 1.0)
        return max(0.5, self.base_power * mult)

    def _target(self, t: float) -> float:
        rv = float(getattr(self.ufun, "reserved_value", 0.0))
        p = self._concession_power()
        frac = (1.0 - t) ** p
        return rv + (self.max_aspiration - rv) * frac

    # -------- FIX #1: accept explicit offer in signature and logic --------
    def respond(self, state, offer=None, source=None):
        """
        Compatible with SAO variants that pass offer explicitly.
        Falls back to state.current_offer if offer is not provided (older variants).
        """
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        if offer is None:
            offer = getattr(state, "current_offer", None)

        self._update_opponent_behavior(offer)

        if offer is None:
            return ResponseType.REJECT_OFFER

        t = self._t(state)
        u = float(self.ufun(offer))
        rv = float(getattr(self.ufun, "reserved_value", 0.0))
        thr = max(rv, self._target(t) - self.accept_slack)

        return ResponseType.ACCEPT_OFFER if u >= thr else ResponseType.REJECT_OFFER

    # -------- FIX #2: keep propose signature permissive --------
    def propose(self, state, dest=None):
        if self.ufun is None or self.nmi is None:
            return None

        self._ensure_issue_cache()

        t = self._t(state)
        target = self._target(t)

        last_offer = getattr(state, "current_offer", None)
        if last_offer is None:
            return self._best_of_random(k=3000)

        lam = self.distance_penalty_end * t
        best = tuple(last_offer)
        best_score = self._proposal_score(best, last_offer, lam)

        cur = best
        for _ in range(self.local_steps):
            if random.random() < self.explore_prob:
                cand = self.nmi.random_outcome()
            else:
                cand = self._neighbor(cur)

            if cand is None:
                continue

            u = float(self.ufun(cand))
            if u >= target:
                # among those meeting target, return the first found (fast), but it is still a local counter
                return cand

            sc = self._proposal_score(cand, last_offer, lam)
            if sc > best_score:
                best, best_score = cand, sc
                cur = cand

        return best

    def _proposal_score(self, offer, last_offer, lam: float) -> float:
        u = float(self.ufun(offer))
        d = _normalized_l1_distance(tuple(offer), tuple(last_offer), self._max_steps) if self._max_steps else 0.0
        return u - lam * d

    def _neighbor(self, offer):
        if not self._issue_vals:
            return self.nmi.random_outcome()

        o = list(offer)
        i = random.randrange(len(o))
        vals = self._issue_vals[i]

        try:
            idx = vals.index(o[i])
        except ValueError:
            idx = min(range(len(vals)), key=lambda k: abs(vals[k] - o[i]))

        step = random.choice([-1, 1])
        nidx = idx + step
        if 0 <= nidx < len(vals):
            o[i] = vals[nidx]
        return tuple(o)

    def _best_of_random(self, k: int):
        best_o, best_u = None, -1e9
        for _ in range(k):
            o = self.nmi.random_outcome()
            u = float(self.ufun(o))
            if u > best_u:
                best_o, best_u = o, u
        return best_o


# =====================================================
# Domain
# =====================================================

issues = [
    make_issue(name="price", values=10),     # 0..9
    make_issue(name="quantity", values=5),   # 0..4
]

P_MAX = 9
Q_MAX = 4


# =====================================================
# Buyer utility
# =====================================================

def buyer_price_nonlinear(p: int) -> float:
    x = p / P_MAX
    return 1.0 - (x ** 2)

def buyer_quantity_diminishing(q: int) -> float:
    return math.sqrt(q / Q_MAX) if Q_MAX > 0 else 0.0

def buyer_budget_penalty(vals: Dict[str, Any]) -> float:
    p = vals["price"]
    q = vals["quantity"]
    total_cost = p * q
    budget = 18
    if total_cost <= budget:
        return 0.0
    return -0.03 * (total_cost - budget) ** 1.2

buyer_ufun = ComplexUtility(
    issues=issues,
    per_issue={"price": buyer_price_nonlinear, "quantity": buyer_quantity_diminishing},
    weights={"price": 0.6, "quantity": 0.4},
    interactions=[buyer_budget_penalty],
    reserved_value=0.55,
    normalize=True,
    umin=-0.7,
    umax=1.0,
)


# =====================================================
# Seller utility
# =====================================================

def seller_price_like(p: int) -> float:
    x = p / P_MAX
    return math.sqrt(x) if x > 0 else 0.0

def seller_quantity_dislike(q: int) -> float:
    x = q / Q_MAX
    return 1.0 - (x ** 2)

def seller_margin_penalty(vals: Dict[str, Any]) -> float:
    p = vals["price"]
    q = vals["quantity"]
    if q >= 3 and p <= 3:
        return -0.25
    return 0.0

def seller_revenue_bonus(vals: Dict[str, Any]) -> float:
    p = vals["price"]
    q = vals["quantity"]
    rev = p * q
    max_rev = P_MAX * Q_MAX
    return 0.3 * (rev / max_rev)

seller_ufun = ComplexUtility(
    issues=issues,
    per_issue={"price": seller_price_like, "quantity": seller_quantity_dislike},
    weights={"price": 0.7, "quantity": 0.3},
    interactions=[seller_margin_penalty, seller_revenue_bonus],
    reserved_value=0.60,
    normalize=True,
    umin=-0.4,
    umax=1.2,
)


# =====================================================
# Feasibility check: is agreement even possible given RVs?
# =====================================================

feasible = []
for p in range(10):
    for q in range(5):
        o = (p, q)
        bu = buyer_ufun(o)
        su = seller_ufun(o)
        if bu >= buyer_ufun.reserved_value and su >= seller_ufun.reserved_value:
            feasible.append((o, bu, su))

print("Feasible outcomes meeting both RVs:", len(feasible))
if feasible:
    print("Example feasible:", sorted(feasible, key=lambda x: x[1] + x[2], reverse=True)[:3])


# =====================================================
# Negotiation
# =====================================================

buyer = AdaptiveCounterNegotiator(
    name="buyer_adaptive",
    base_power=2.5,
    adapt_strength=0.8,
    accept_slack=0.03,
    local_steps=300,
    explore_prob=0.10,
    distance_penalty_end=0.60,
)

seller = AdaptiveCounterNegotiator(
    name="seller_adaptive",
    base_power=2.0,
    adapt_strength=0.8,
    accept_slack=0.03,
    local_steps=300,
    explore_prob=0.10,
    distance_penalty_end=0.60,
)

session = SAOMechanism(issues=issues, n_steps=100)
session.add(buyer, ufun=buyer_ufun)
session.add(seller, ufun=seller_ufun)

result = session.run()
print(f"Agreement: {result.agreement}, Rounds: {result.step}")

session.plot()