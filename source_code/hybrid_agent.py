from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios
from negmas.sao import SAONegotiator, ResponseType
import itertools
import tqdm
import pickle
import random
from collections import defaultdict

# -------------------------
# Hybrid Negotiator
# -------------------------
class HybridNegotiator(SAONegotiator):
    def __init__(self, gamma=1.0, reserved_value=0.2, memory=5, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self._rv = reserved_value
        self.memory = memory
        self.alpha = alpha # utility weight self vs opponent

        self.outcomes = []
        self.u_max = 1.0
        self.opponent_history = []

        # opponent model with frequency count
        self.opp_model = defaultdict(lambda: defaultdict(int))

    # generates all possible deals to find maximum utility
    def on_preferences_changed(self, changes):
        self.outcomes = self._all_outcomes()
        self.u_max = max(self.ufun(o) for o in self.outcomes)

    def _all_outcomes(self):
        vals = []
        for issue in self.nmi.issues:
            if hasattr(issue, "all") and issue.all is not None:
                vals.append(list(issue.all))
            elif isinstance(getattr(issue, "values", None), (list, tuple, range)):
                vals.append(list(issue.values))
            elif isinstance(getattr(issue, "values", None), int):
                vals.append(list(range(issue.values)))
            else:
                vals.append(list(range(int(issue.min_value), int(issue.max_value) + 1)))
        return list(itertools.product(*vals))
    
    # opponent modelling on what values are proposed
    def update_opponent_model(self, offer):
        for i, issue in enumerate(self.nmi.issues):
            value = offer[i]
            self.opp_model[issue.name][value] += 1

    # give a higher score to more frequently proposed values
    def estimate_opponent_utility(self, offer):
        score = 0

        for i, issue in enumerate(self.nmi.issues):
            value = offer[i]
            freq = self.opp_model[issue.name][value]
            total = sum(self.opp_model[issue.name].values()) + 1e-6
            score += freq / total

        return score / len(self.nmi.issues)
    
    def propose(self, state, *args, **kwargs):
        t = state.relative_time

        # adaptive gamma
        if len(self.opponent_history) >= 2:
            # average utility for recent offers (concede faster for better offers)
            recent = self.opponent_history[-self.memory:]
            avg_opp = sum(self.ufun(o) for o in recent) / len(recent)

            # T4T trend (concede more for better trend)
            last = self.ufun(self.opponent_history[-1])
            prev = self.ufun(self.opponent_history[-2])
            trend = last - prev

            # adjust quality
            if avg_opp < self.u_max * 0.5:
                self.gamma = min(self.gamma * 1.1, 3.0)
            else:
                self.gamma = max(self.gamma * 0.9, 0.5)

            # adjust trend
            if trend > 0:
                # conceding opponent
                self.gamma = max(self.gamma * 0.9, 0.5)
            else:
                # stubborn opponent
                self.gamma = min(self.gamma * 1.1, 3.0)
        
        # time-based target utility
        u_min, u_max = self._rv, self.u_max
        target = u_max - (u_max - u_min) * (t ** self.gamma)

        # candidate offers that are above our target
        candidates = [o for o in self.outcomes if self.ufun(o) >= target]

        if not candidates:
            return self.nmi.random_outcome()
        
        # hybrid scoring with weighted average of our and opponent's utility for offer
        scored = []
        for o in candidates:
            self_u = self.ufun(o)
            opp_u = self.estimate_opponent_utility(o)
            score = self.alpha * self_u + (1 - self.alpha) * opp_u
            scored.append((o, score))

        # probabilistic selection based on scores
        offers, scores = zip(*scored)
        return random.choices(offers, weights=scores, k=1)[0]
    
    def respond(self, state, offer=None, *args, **kwargs):
        offer = offer or state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        
        # update opponent info
        self.opponent_history.append(offer)
        self.update_opponent_model(offer)

        t = state.relative_time
        target = self.u_max - (self.u_max - self._rv) * (t ** self.gamma)

        # accept if good enough or better than what we expect to propose
        if self.ufun(offer) >= target:
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER
    
# -------------------------
# Evaluation class
# -------------------------
class HybridEvaluation:

    @staticmethod
    def run_scenarios(scenarios, gamma, sessions, buyer_class, seller_class):
        results = []

        for name, [issue, u_buyer, u_seller] in scenarios.items():
            agreements = 0
            buyer_utils, seller_utils, rounds = [], [], []
            all_buyer_points, all_seller_points = [], []

            for _ in tqdm.tqdm(range(sessions), desc=f"{name}"):
                mechanism = SAOMechanism(issues=issue, n_steps=100)

                if buyer_class is HybridNegotiator:
                    mechanism.add(
                        buyer_class(name="buyer", ufun=u_buyer, gamma=gamma, alpha=0.7, memory=5),
                        ufun=u_buyer,
                    )
                else:
                    mechanism.add(
                        buyer_class(name="buyer", ufun=u_buyer),
                        ufun=u_buyer,
                    )

                if seller_class is HybridNegotiator:
                    mechanism.add(
                        seller_class(name="seller", ufun=u_seller, gamma=gamma, alpha=0.7, memory=5),
                        ufun=u_seller,
                    )
                else:
                    mechanism.add(
                        seller_class(name="seller", ufun=u_seller),
                        ufun=u_seller,
                    )

                result = mechanism.run()

                if result.agreement:
                    agreements += 1
                    buyer_utils.append(u_buyer(result.agreement))
                    seller_utils.append(u_seller(result.agreement))
                    all_buyer_points.append(u_buyer(result.agreement))
                    all_seller_points.append(u_seller(result.agreement))

                rounds.append(mechanism.state.step)

            results.append({
                "scenario": name,
                "buyer_class": buyer_class.__name__,
                "seller_class": seller_class.__name__,
                "gamma": gamma,
                "agreement_rate": agreements / sessions,
                "average_buyer_utility": sum(buyer_utils)/len(buyer_utils) if buyer_utils else 0,
                "average_seller_utility": sum(seller_utils)/len(seller_utils) if seller_utils else 0,
                "average_rounds": sum(rounds)/len(rounds),
                "all_buyer_points": all_buyer_points,
                "all_seller_points": all_seller_points,
            })

        return results
    
    @staticmethod
    def run_and_store_results():
        scenarios = {
            "Single Issue": EvaluationScenarios.getSingleIssue(),
            "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
            "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
            "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
            "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
        }

        gammas = [0.5]
        alphas = [0.7]
        memories = [5]

        # other negotiators
        from negmas import AspirationNegotiator, RandomNegotiator
        other_agents = [AspirationNegotiator, RandomNegotiator]

        for gamma in gammas:
            for alpha in alphas:
                for memory in memories:
                    for other_class in other_agents:

                        # Hybrid as buyer
                        results = HybridEvaluation.run_scenarios(
                            scenarios,
                            gamma=gamma,
                            sessions=30,
                            buyer_class=HybridNegotiator,
                            seller_class=other_class,
                        )

                        filename = f"results_Hybrid_vs_{other_class.__name__}_g{gamma}_a{alpha}_m{memory}.pkl"
                        with open(filename, "wb") as f:
                            pickle.dump(results, f)
                        print(f"Saved {filename}")

                        # Hybrid as seller
                        results = HybridEvaluation.run_scenarios(
                            scenarios,
                            gamma=gamma,
                            sessions=30,
                            buyer_class=other_class,
                            seller_class=HybridNegotiator,
                        )

                        filename = f"results_{other_class.__name__}_vs_Hybrid_g{gamma}_a{alpha}_m{memory}.pkl"
                        with open(filename, "wb") as f:
                            pickle.dump(results, f)
                        print(f"Saved {filename}")

if __name__ == "__main__":
    HybridEvaluation.run_and_store_results()