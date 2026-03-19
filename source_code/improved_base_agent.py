from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios
from negmas.sao import SAONegotiator, ResponseType
import itertools
import tqdm
import pickle
import random

# -------------------------
# Improved Base Time-Based Agent
# -------------------------
class ImprovedBaseNegotiator(SAONegotiator):
    def __init__(self, gamma=1.0, reserved_value=0.2, memory = 5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma # starting consession speed
        self._rv = reserved_value
        self.outcomes = []
        self.u_max = 1.0

        self.opponent_history = [] # track last N offers
        self.memory = memory

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

    def propose(self, state, *args, **kwargs):
        t = state.relative_time

        # add adaptive gamma on how the opponent is acting
        if self.opponent_history:

            # calculate average utility for us on opponents offers
            avg_opp = sum(self.ufun(o) for o in self.opponent_history[-self.memory:]) / len(self.opponent_history[-self.memory:])            
            
            # if opponent is giving better deals, lower the gamma, otherwise increase
            if avg_opp < self.u_max * 0.5:
                self.gamma = min(self.gamma * 1.2, 3.0)
            else:
                self.gamma = max(self.gamma * 0.8, 0.5)

        else:
            # else do not change
            self.gamma = self.gamma

        u_min, u_max = self._rv, self.u_max
        target = u_max - (u_max - u_min) * (t ** self.gamma)

        # find best offer above target
        cand = [offer for offer in self.outcomes if self.ufun(offer) >= target]

        if not cand:
            return self.nmi.random_outcome()
        
        utils = [self.ufun(offer) for offer in cand]
        total = sum(utils)
        probs = [u / total for u in utils] # calculate probabilites based on how high
        return random.choices(cand, weights=probs, k=1)[0]


    def respond(self, state, offer=None, *args, **kwargs):
        offer = offer or state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        
        # save opponent offer
        self.opponent_history.append(offer)
        t = state.relative_time
        target = self.u_max - (self.u_max - self._rv) * (t ** self.gamma)
        return ResponseType.ACCEPT_OFFER if self.ufun(offer) >= target else ResponseType.REJECT_OFFER


# -------------------------
# Evaluation class
# -------------------------
class ImprovedEvaluation:

    @staticmethod
    def run_scenarios(scenarios, gamma, sessions, buyer_class, seller_class):
        results = []

        for name, [issue, u_buyer, u_seller] in scenarios.items():
            agreements = 0
            buyer_utils, seller_utils, rounds = [], [], []
            all_buyer_points, all_seller_points = [], []

            for _ in tqdm.tqdm(range(sessions), desc=f"{name}"):
                mechanism = SAOMechanism(issues=issue, n_steps=100)

                # only pass reserved_value and gamma to BaseNegotiator
                if buyer_class is ImprovedBaseNegotiator:
                    mechanism.add(
                        buyer_class(name="buyer", ufun=u_buyer, reserved_value=0.2, gamma=gamma),
                        ufun=u_buyer,
                    )
                else:
                    mechanism.add(
                        buyer_class(name="buyer", ufun=u_buyer),
                        ufun=u_buyer,
                    )

                if seller_class is ImprovedBaseNegotiator:
                    mechanism.add(
                        seller_class(name="seller", ufun=u_seller, reserved_value=0.2, gamma=gamma),
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

        gammas = [2.0]

        # other negotiators
        from negmas import AspirationNegotiator, RandomNegotiator
        other_agents = [AspirationNegotiator, RandomNegotiator]

        for gamma in gammas:
            for other_class in other_agents:
                # BaseNegotiator as buyer
                results = ImprovedEvaluation.run_scenarios(
                    scenarios, gamma=gamma, sessions=30,
                    buyer_class=ImprovedBaseNegotiator, seller_class=other_class
                )
                filename = f"results_ImprovedBaseNegotiator_vs_{other_class.__name__}_gamma{gamma}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved results to {filename}")

                # BaseNegotiator as seller
                results = ImprovedEvaluation.run_scenarios(
                    scenarios, gamma=gamma, sessions=30,
                    buyer_class=other_class, seller_class=ImprovedBaseNegotiator
                )
                filename = f"results_{other_class.__name__}_vs_ImprovedBaseNegotiator_gamma{gamma}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved results to {filename}")


if __name__ == "__main__":
    ImprovedEvaluation.run_and_store_results()