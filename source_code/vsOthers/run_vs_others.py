import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import pickle
import tqdm
from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios

from reactive_agent import ReactiveT4TNegotiator, ReactiveT4TConfig
from hybrid_agent import HybridNegotiator

from adaptive_agent import AdaptiveNegotiator
from timeBasedAgent import TimeBasedAgent
from titTat import TitForTatAgent
from microNegotiator import MicroNegotiator

our_agents = [ReactiveT4TNegotiator, HybridNegotiator]
their_agents = [AdaptiveNegotiator, TimeBasedAgent, TitForTatAgent, MicroNegotiator]

scenarios = {
    "Single Issue": EvaluationScenarios.getSingleIssue(),
    "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
    "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
    "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
    "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
}


def instantiate(agent_class, name):
    if agent_class is ReactiveT4TNegotiator:
        return ReactiveT4TNegotiator(name=name, cfg=ReactiveT4TConfig())
    return agent_class(name=name)


def run_matchup(agent_a, agent_b, scenarios, sessions=30):
    results = []

    for name, (issue, u_a, u_b) in scenarios.items():
        agreements = 0
        a_utils, b_utils, rounds = [], [], []

        for _ in tqdm.tqdm(range(sessions), desc=f"{agent_a.__name__} vs {agent_b.__name__} ({name})"):
            mechanism = SAOMechanism(issues=issue, n_steps=100)
            mechanism.add(instantiate(agent_a, agent_a.__name__), ufun=u_a)
            mechanism.add(instantiate(agent_b, agent_b.__name__), ufun=u_b)

            result = mechanism.run()

            if result.agreement:
                agreements += 1
                a_utils.append(u_a(result.agreement))
                b_utils.append(u_b(result.agreement))

            rounds.append(mechanism.state.step)

        results.append({
            "agent_A": agent_a.__name__,
            "agent_B": agent_b.__name__,
            "scenario": name,
            "agreement_rate": agreements / sessions,
            "avg_A_utility": sum(a_utils) / len(a_utils) if a_utils else 0,
            "avg_B_utility": sum(b_utils) / len(b_utils) if b_utils else 0,
            "average_rounds": sum(rounds) / len(rounds),
        })

    return results


if __name__ == "__main__":
    os.makedirs("saved_results", exist_ok=True)

    for our in our_agents:
        for their in their_agents:
            for agent_a, agent_b in [(our, their), (their, our)]:
                results = run_matchup(agent_a, agent_b, scenarios, sessions=30)
                filename = f"saved_results/results_{agent_a.__name__}_vs_{agent_b.__name__}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved {filename}")
