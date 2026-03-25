import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import pickle
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def compute_scenario_benchmarks(issues, u_a, u_b):
    """Compute Pareto frontier, Nash bargaining point, and max welfare for a scenario."""
    outcomes = list(SAOMechanism(issues=issues, n_steps=100).outcomes)
    utils = [(float(u_a(o)), float(u_b(o))) for o in outcomes]

    # Pareto frontier: outcomes not dominated by any other
    pareto = []
    for i, (ua, ub) in enumerate(utils):
        dominated = any(
            ua2 >= ua and ub2 >= ub and (ua2 > ua or ub2 > ub)
            for j, (ua2, ub2) in enumerate(utils) if i != j
        )
        if not dominated:
            pareto.append((ua, ub))

    # Nash bargaining point: maximises u_a * u_b (disagreement point = 0,0)
    nash_point = max(pareto, key=lambda x: x[0] * x[1])

    # Max welfare point: maximises u_a + u_b
    max_welfare_point = max(pareto, key=lambda x: x[0] + x[1])
    max_welfare = max_welfare_point[0] + max_welfare_point[1]

    return {
        "pareto_frontier": sorted(pareto, key=lambda x: x[0]),
        "nash_point": nash_point,
        "max_nash_product": nash_point[0] * nash_point[1],
        "max_welfare_point": max_welfare_point,
        "max_welfare": max_welfare,
    }


def instantiate(agent_class, name):
    if agent_class is ReactiveT4TNegotiator:
        return ReactiveT4TNegotiator(name=name, cfg=ReactiveT4TConfig())
    return agent_class(name=name)


def run_matchup(agent_a, agent_b, scenarios, benchmarks, sessions=30):
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

        bench = benchmarks[name]
        avg_ua = sum(a_utils) / len(a_utils) if a_utils else 0
        avg_ub = sum(b_utils) / len(b_utils) if b_utils else 0
        welfare = avg_ua + avg_ub
        nash_product = avg_ua * avg_ub

        results.append({
            "agent_A": agent_a.__name__,
            "agent_B": agent_b.__name__,
            "scenario": name,
            "agreement_rate": agreements / sessions,
            "avg_A_utility": avg_ua,
            "avg_B_utility": avg_ub,
            "average_rounds": sum(rounds) / len(rounds),
            # Welfare metrics
            "welfare": welfare,
            "welfare_ratio": welfare / bench["max_welfare"] if bench["max_welfare"] > 0 else 0,
            # Nash metrics
            "nash_product": nash_product,
            "nash_product_ratio": nash_product / bench["max_nash_product"] if bench["max_nash_product"] > 0 else 0,
            # Scenario reference points
            "nash_point": bench["nash_point"],
            "max_welfare_point": bench["max_welfare_point"],
            "pareto_frontier": bench["pareto_frontier"],
        })

    return results


def save_pareto_plots(benchmarks, output_dir):
    """Save one Pareto frontier PNG per scenario."""
    print("Saving Pareto frontier plots...")
    for name, bench in benchmarks.items():
        pareto = bench["pareto_frontier"]
        xs, ys = zip(*sorted(pareto))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(xs, ys, "k-o", markersize=4, label="Pareto frontier")
        ax.scatter(*bench["nash_point"], color="red", zorder=5, s=80,
                   label=f"Nash ({bench['nash_point'][0]:.2f}, {bench['nash_point'][1]:.2f})")
        ax.scatter(*bench["max_welfare_point"], color="blue", zorder=5, s=80,
                   label=f"Max welfare ({bench['max_welfare_point'][0]:.2f}, {bench['max_welfare_point'][1]:.2f})")
        ax.set_xlabel("Agent A utility")
        ax.set_ylabel("Agent B utility")
        ax.set_title(f"Pareto Frontier — {name}")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = os.path.join(output_dir, f"pareto_{name.replace(' ', '_')}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    os.makedirs("saved_results", exist_ok=True)

    # Precompute benchmarks once (same for all matchups)
    print("Computing scenario benchmarks...")
    benchmarks = {}
    for name, (issue, u_a, u_b) in scenarios.items():
        benchmarks[name] = compute_scenario_benchmarks(issue, u_a, u_b)
        b = benchmarks[name]
        print(f"  {name}: Nash={b['nash_point']}, MaxWelfare={b['max_welfare']:.3f}")

    for our in our_agents:
        for their in their_agents:
            for agent_a, agent_b in [(our, their), (their, our)]:
                results = run_matchup(agent_a, agent_b, scenarios, benchmarks, sessions=30)
                filename = f"saved_results/results_{agent_a.__name__}_vs_{agent_b.__name__}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved {filename}")

    save_pareto_plots(benchmarks, "saved_results")
