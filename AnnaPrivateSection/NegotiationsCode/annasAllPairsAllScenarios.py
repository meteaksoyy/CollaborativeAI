import os
import pickle
import tqdm
from negmas import SAOMechanism

from evaluation_scenarios import EvaluationScenarios
from base_agent import BaseNegotiator
from improved_base_agent import ImprovedBaseNegotiator
from reactive_agent import ReactiveT4TNegotiator, ReactiveT4TConfig
from hybrid_agent import HybridNegotiator


# -------------------------
# Save folder
# -------------------------
SAVE_DIR = "anna_saves"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# All your agents
# -------------------------
AGENTS = [
    BaseNegotiator,
    ImprovedBaseNegotiator,
    ReactiveT4TNegotiator,
    HybridNegotiator,
]


# -------------------------
# Agent factory
# -------------------------
def make_agent(agent_class, role_name, ufun):
    if agent_class is BaseNegotiator:
        return BaseNegotiator(
            name=role_name,
            ufun=ufun,
            reserved_value=0.2,
            gamma=1.0,
        )

    elif agent_class is ImprovedBaseNegotiator:
        return ImprovedBaseNegotiator(
            name=role_name,
            ufun=ufun,
            reserved_value=0.2,
            gamma=2.0,
            memory=5,
        )

    elif agent_class is ReactiveT4TNegotiator:
        return ReactiveT4TNegotiator(
            name=role_name,
            cfg=ReactiveT4TConfig(),
        )

    elif agent_class is HybridNegotiator:
        return HybridNegotiator(
            name=role_name,
            ufun=ufun,
            reserved_value=0.2,
            gamma=0.5,
            alpha=0.7,
            memory=5,
        )

    else:
        return agent_class(name=role_name, ufun=ufun)


# -------------------------
# Run one ordered pair on one scenario
# -------------------------
def run_one_scenario(
    scenario_name,
    scenario_data,
    buyer_class,
    seller_class,
    sessions=20,
    n_steps=100,
):
    issues, u_buyer, u_seller = scenario_data

    agreements = 0
    buyer_utils = []
    seller_utils = []
    rounds = []
    matches = []

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

        buyer_utility = u_buyer(result.agreement) if agreed else 0
        seller_utility = u_seller(result.agreement) if agreed else 0

        if agreed:
            agreements += 1
            buyer_utils.append(buyer_utility)
            seller_utils.append(seller_utility)

        rounds.append(mechanism.state.step)

        matches.append({
            "match_id": match_id,
            "scenario": scenario_name,
            "buyer_class": buyer_class.__name__,
            "seller_class": seller_class.__name__,
            "agreement": agreed,
            "rounds": mechanism.state.step,
            "buyer_utility": buyer_utility,
            "seller_utility": seller_utility,
            "final_agreement": result.agreement,
        })

    return {
        "scenario": scenario_name,
        "buyer_class": buyer_class.__name__,
        "seller_class": seller_class.__name__,
        "sessions": sessions,
        "agreement_rate": agreements / sessions,
        "average_buyer_utility": sum(buyer_utils) / len(buyer_utils) if buyer_utils else 0,
        "average_seller_utility": sum(seller_utils) / len(seller_utils) if seller_utils else 0,
        "average_rounds": sum(rounds) / len(rounds) if rounds else 0,
        "matches": matches,
    }


# -------------------------
# Run one ordered pair on all scenarios
# -------------------------
def run_all_scenarios(agent_a, agent_b, sessions=20):
    scenarios = {
        "Single Issue": EvaluationScenarios.getSingleIssue(),
        "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
        "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
        "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
        "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
    }

    results = []

    for scenario_name, scenario_data in scenarios.items():
        summary = run_one_scenario(
            scenario_name=scenario_name,
            scenario_data=scenario_data,
            buyer_class=agent_a,
            seller_class=agent_b,
            sessions=sessions,
        )
        results.append(summary)

    return results


# -------------------------
# Main: run every ordered pair across all scenarios
# -------------------------
if __name__ == "__main__":
    for agent_a in AGENTS:
        for agent_b in AGENTS:
            if agent_a is agent_b:
                continue

            print(f"\nRunning {agent_a.__name__} vs {agent_b.__name__} on all scenarios...")

            results = run_all_scenarios(agent_a, agent_b, sessions=20)

            filename = os.path.join(
                SAVE_DIR,
                f"{agent_a.__name__}_vs_{agent_b.__name__}_all_scenarios_50matches.pkl"
            )

            with open(filename, "wb") as f:
                pickle.dump(results, f)

            print(f"Saved {filename}")