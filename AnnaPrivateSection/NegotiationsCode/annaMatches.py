import pickle
import tqdm
import os
from negmas import SAOMechanism, AspirationNegotiator, RandomNegotiator

from evaluation_scenarios import EvaluationScenarios
from base_agent import BaseNegotiator
from improved_base_agent import ImprovedBaseNegotiator
from reactive_agent import ReactiveT4TNegotiator, ReactiveT4TConfig
from hybrid_agent import HybridNegotiator


# -------------------------
# Create YOUR personal save folder
# -------------------------
SAVE_DIR = "anna_saves"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Agent factory
# -------------------------
def make_agent(agent_class, role_name, ufun):
    if agent_class is BaseNegotiator:
        return BaseNegotiator(name=role_name, ufun=ufun, reserved_value=0.2, gamma=1.0)

    elif agent_class is ImprovedBaseNegotiator:
        return ImprovedBaseNegotiator(name=role_name, ufun=ufun, reserved_value=0.2, gamma=2.0, memory=5)

    elif agent_class is ReactiveT4TNegotiator:
        return ReactiveT4TNegotiator(name=role_name, cfg=ReactiveT4TConfig())

    elif agent_class is HybridNegotiator:
        return HybridNegotiator(name=role_name, ufun=ufun, reserved_value=0.2, gamma=0.5, alpha=0.7, memory=5)

    else:
        return agent_class(name=role_name, ufun=ufun)


# -------------------------
# Run matches
# -------------------------
def run_matches_one_scenario(
    scenario_name,
    scenario_data,
    buyer_class,
    seller_class,
    sessions=30,
    n_steps=100
):
    issues, u_buyer, u_seller = scenario_data
    results = []

    agreements = 0
    buyer_utils = []
    seller_utils = []
    rounds = []

    for match_id in tqdm.tqdm(
        range(sessions),
        desc=f"{buyer_class.__name__} vs {seller_class.__name__} [{scenario_name}]"
    ):
        mechanism = SAOMechanism(issues=issues, n_steps=n_steps)

        buyer = make_agent(buyer_class, "buyer", u_buyer)
        seller = make_agent(seller_class, "seller", u_seller)

        mechanism.add(buyer, ufun=u_buyer)
        mechanism.add(seller, ufun=u_seller)

        result = mechanism.run()
        agreed = result.agreement is not None

        match_result = {
            "match_id": match_id,
            "scenario": scenario_name,
            "buyer_class": buyer_class.__name__,
            "seller_class": seller_class.__name__,
            "agreement": agreed,
            "rounds": mechanism.state.step,
            "buyer_utility": u_buyer(result.agreement) if agreed else 0,
            "seller_utility": u_seller(result.agreement) if agreed else 0,
            "final_agreement": result.agreement,
        }

        results.append(match_result)

        if agreed:
            agreements += 1
            buyer_utils.append(match_result["buyer_utility"])
            seller_utils.append(match_result["seller_utility"])

        rounds.append(match_result["rounds"])

    summary = {
        "scenario": scenario_name,
        "buyer_class": buyer_class.__name__,
        "seller_class": seller_class.__name__,
        "sessions": sessions,
        "agreement_rate": agreements / sessions,
        "average_buyer_utility": sum(buyer_utils) / len(buyer_utils) if buyer_utils else 0,
        "average_seller_utility": sum(seller_utils) / len(seller_utils) if seller_utils else 0,
        "average_rounds": sum(rounds) / len(rounds) if rounds else 0,
        "matches": results,
    }

    return summary


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    scenario_name = "Single Issue"
    scenario_data = EvaluationScenarios.getSingleIssue()

    # -------------------------
    # Random vs Reactive
    # -------------------------
    summary1 = run_matches_one_scenario(
        scenario_name=scenario_name,
        scenario_data=scenario_data,
        buyer_class=RandomNegotiator,
        seller_class=ReactiveT4TNegotiator,
        sessions=30
    )

    filename1 = os.path.join(
        SAVE_DIR,
        "results_RandomNegotiator_vs_ReactiveT4T_single_issue.pkl"
    )

    with open(filename1, "wb") as f:
        pickle.dump(summary1, f)

    print(f"Saved {filename1}")

    # -------------------------
    # Reactive vs Random
    # -------------------------
    summary2 = run_matches_one_scenario(
        scenario_name=scenario_name,
        scenario_data=scenario_data,
        buyer_class=ReactiveT4TNegotiator,
        seller_class=RandomNegotiator,
        sessions=30
    )

    filename2 = os.path.join(
        SAVE_DIR,
        "results_ReactiveT4T_vs_RandomNegotiator_single_issue.pkl"
    )

    with open(filename2, "wb") as f:
        pickle.dump(summary2, f)

    print(f"Saved {filename2}")