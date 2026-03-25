from __future__ import annotations
from copyreg import pickle
import itertools
from dataclasses import dataclass
import pickle
from typing import Any, List, Optional, Tuple

from negmas import SAOMechanism, make_issue, TimeBasedConcedingNegotiator
from negmas.sao import AspirationNegotiator, AspirationNegotiator, RandomNegotiator, SAONegotiator, ResponseType
from tqdm import tqdm

from evaluation_scenarios import EvaluationScenarios

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


# -------------------------
# Evaluation class
# -------------------------
class ReactiveEvaluation:

    @staticmethod
    def run_scenarios(scenarios, sessions, buyer_class, seller_class):
        results = []

        for name, [issue, u_buyer, u_seller] in scenarios.items():
            agreements = 0
            buyer_utils, seller_utils, rounds = [], [], []
            all_buyer_points, all_seller_points = [], []

            for _ in tqdm(range(sessions), desc=f"{name}"):
                mechanism = SAOMechanism(issues=issue, n_steps=100)

                if buyer_class is ReactiveT4TNegotiator:
                    mechanism.add(ReactiveT4TNegotiator(name="buyer", cfg=ReactiveT4TConfig()), ufun=u_buyer)
                else:
                    mechanism.add(buyer_class(name="buyer", ufun=u_buyer), ufun=u_buyer)

                if seller_class is ReactiveT4TNegotiator:
                    mechanism.add(ReactiveT4TNegotiator(name="seller", cfg=ReactiveT4TConfig()), ufun=u_seller)
                else:
                    mechanism.add(seller_class(name="seller", ufun=u_seller), ufun=u_seller)

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
                "agreement_rate": agreements / sessions,
                "average_buyer_utility": sum(buyer_utils) / len(buyer_utils) if buyer_utils else 0,
                "average_seller_utility": sum(seller_utils) / len(seller_utils) if seller_utils else 0,
                "average_rounds": sum(rounds) / len(rounds),
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

        for other_class in [AspirationNegotiator, RandomNegotiator]:
            results = ReactiveEvaluation.run_scenarios(scenarios, sessions=30,
                                    buyer_class=ReactiveT4TNegotiator, seller_class=other_class)
            filename = f"results_ReactiveT4T_vs_{other_class.__name__}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved {filename}")

            results = ReactiveEvaluation.run_scenarios(scenarios, sessions=30,
                                    buyer_class=other_class, seller_class=ReactiveT4TNegotiator)
            filename = f"results_{other_class.__name__}_vs_ReactiveT4T.pkl"
            with open(filename, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved {filename}")



if __name__ == "__main__":

    ReactiveEvaluation.run_and_store_results()
