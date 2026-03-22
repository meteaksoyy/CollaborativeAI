import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negmas import SAOMechanism, AspirationNegotiator, RandomNegotiator
from evaluation_scenarios import EvaluationScenarios
from ReactiveT4TwLateTimePressure_2issues import ReactiveT4TNegotiator, ReactiveT4TConfig
import tqdm
import pickle


def run_scenarios(scenarios, sessions, buyer_class, seller_class):
    results = []

    for name, [issue, u_buyer, u_seller] in scenarios.items():
        agreements = 0
        buyer_utils, seller_utils, rounds = [], [], []
        all_buyer_points, all_seller_points = [], []

        for _ in tqdm.tqdm(range(sessions), desc=f"{name}"):
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


if __name__ == "__main__":
    scenarios = {
        "Single Issue": EvaluationScenarios.getSingleIssue(),
        "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
        "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
        "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
        "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
    }

    for other_class in [AspirationNegotiator, RandomNegotiator]:
        results = run_scenarios(scenarios, sessions=30,
                                buyer_class=ReactiveT4TNegotiator, seller_class=other_class)
        filename = f"results_ReactiveT4T_vs_{other_class.__name__}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {filename}")

        results = run_scenarios(scenarios, sessions=30,
                                buyer_class=other_class, seller_class=ReactiveT4TNegotiator)
        filename = f"results_{other_class.__name__}_vs_ReactiveT4T.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {filename}")
