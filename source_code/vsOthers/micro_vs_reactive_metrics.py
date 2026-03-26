import sys
import os
import math
import statistics
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios

from reactive_agent import ReactiveT4TNegotiator, ReactiveT4TConfig
from microNegotiator import MicroNegotiator


SCENARIOS = {
    "Single Issue": EvaluationScenarios.getSingleIssue(),
    "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
    "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
    "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
    "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
}


def euclidean_distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def compute_scenario_benchmarks(issues, u_a, u_b):
    """Compute Pareto frontier, Nash point, Kalai point, and KS point for one scenario."""
    outcomes = list(SAOMechanism(issues=issues, n_steps=100).outcomes)
    utils = [(float(u_a(o)), float(u_b(o))) for o in outcomes]

    pareto = []
    for i, (ua, ub) in enumerate(utils):
        dominated = any(
            ua2 >= ua and ub2 >= ub and (ua2 > ua or ub2 > ub)
            for j, (ua2, ub2) in enumerate(utils)
            if i != j
        )
        if not dominated:
            pareto.append((ua, ub))

    pareto = sorted(set(pareto), key=lambda x: x[0])

    nash_point = max(pareto, key=lambda x: x[0] * x[1])

    max_a = max(x[0] for x in pareto)
    max_b = max(x[1] for x in pareto)

    # KS 
    def ks_gap(x):
        ra = x[0] / max_a if max_a > 0 else 0.0
        rb = x[1] / max_b if max_b > 0 else 0.0
        return abs(ra - rb)

    ks_point = min(pareto, key=ks_gap)

    # Kalai: maximize the utility of the worse-off agent
    kalai_point = max(pareto, key=lambda x: min(x[0], x[1]))

    return {
        "pareto_frontier": pareto,
        "nash_point": nash_point,
        "kalai_point": kalai_point,
        "ks_point": ks_point,
    }


def closest_pareto_distance(point, frontier):
    closest = min(frontier, key=lambda x: euclidean_distance(point, x))
    return euclidean_distance(point, closest)


def instantiate_reactive(name):
    return ReactiveT4TNegotiator(name=name, cfg=ReactiveT4TConfig())


def instantiate_micro(name):
    return MicroNegotiator(name=name)


def instantiate_agent(agent_class, name):
    if agent_class == ReactiveT4TNegotiator:
        return instantiate_reactive(name)
    elif agent_class == MicroNegotiator:
        return instantiate_micro(name)
    else:
        return agent_class(name=name)


def run_sessions(agent_a_class, agent_b_class, issues, u_a, u_b, sessions=30):
    """Run repeated matches and return average outcome point plus raw session utilities."""
    a_utils = []
    b_utils = []

    for _ in range(sessions):
        mechanism = SAOMechanism(issues=issues, n_steps=100)

        agent_a = instantiate_agent(agent_a_class, agent_a_class.__name__)
        agent_b = instantiate_agent(agent_b_class, agent_b_class.__name__)

        mechanism.add(agent_a, ufun=u_a)
        mechanism.add(agent_b, ufun=u_b)

        result = mechanism.run()

        if result.agreement is not None:
            a_utils.append(float(u_a(result.agreement)))
            b_utils.append(float(u_b(result.agreement)))

    avg_a = statistics.mean(a_utils) if a_utils else 0.0
    avg_b = statistics.mean(b_utils) if b_utils else 0.0
    agreement_rate = len(a_utils) / sessions if sessions > 0 else 0.0

    return {
        "avg_utility_a": avg_a,
        "avg_utility_b": avg_b,
        "agreement_rate": agreement_rate,
        "raw_utilities_a": a_utils,
        "raw_utilities_b": b_utils,
        "num_agreements": len(a_utils),
        "num_sessions": sessions,
        "avg_outcome_point": (avg_a, avg_b),
    }


def evaluate_direction(agent_a_class, agent_b_class, direction_name, sessions=30):
    print(f"\n=== {direction_name} ===")

    scenario_results = {}

    pareto_distances = []
    nash_distances = []
    kalai_distances = []
    ks_distances = []
    agreement_rates = []
    avg_utils_a = []
    avg_utils_b = []

    for scenario_name, (issues, u_a, u_b) in SCENARIOS.items():
        benchmarks = compute_scenario_benchmarks(issues, u_a, u_b)
        session_results = run_sessions(
            agent_a_class, agent_b_class, issues, u_a, u_b, sessions=sessions
        )

        actual_point = session_results["avg_outcome_point"]
        agreement_rate = session_results["agreement_rate"]

        pareto_d = closest_pareto_distance(actual_point, benchmarks["pareto_frontier"])
        nash_d = euclidean_distance(actual_point, benchmarks["nash_point"])
        kalai_d = euclidean_distance(actual_point, benchmarks["kalai_point"])
        ks_d = euclidean_distance(actual_point, benchmarks["ks_point"])

        pareto_distances.append(pareto_d)
        nash_distances.append(nash_d)
        kalai_distances.append(kalai_d)
        ks_distances.append(ks_d)
        agreement_rates.append(agreement_rate)
        avg_utils_a.append(session_results["avg_utility_a"])
        avg_utils_b.append(session_results["avg_utility_b"])

        scenario_results[scenario_name] = {
            "agent_a": agent_a_class.__name__,
            "agent_b": agent_b_class.__name__,
            "avg_utility_a": session_results["avg_utility_a"],
            "avg_utility_b": session_results["avg_utility_b"],
            "agreement_rate": agreement_rate,
            "num_agreements": session_results["num_agreements"],
            "num_sessions": session_results["num_sessions"],
            "raw_utilities_a": session_results["raw_utilities_a"],
            "raw_utilities_b": session_results["raw_utilities_b"],
            "avg_outcome_point": actual_point,
            "pareto_distance": pareto_d,
            "nash_distance": nash_d,
            "kalai_distance": kalai_d,
            "ks_distance": ks_d,
            "nash_point": benchmarks["nash_point"],
            "kalai_point": benchmarks["kalai_point"],
            "ks_point": benchmarks["ks_point"],
            "pareto_frontier": benchmarks["pareto_frontier"],
        }

        print(f"\nScenario: {scenario_name}")
        print(f"  Agreement rate: {agreement_rate:.2f}")
        print(f"  Avg outcome point: ({actual_point[0]:.3f}, {actual_point[1]:.3f})")
        print(f"  Pareto Distance: {pareto_d:.3f}")
        print(f"  Nash Distance:   {nash_d:.3f}")
        print(f"  Kalai Distance:  {kalai_d:.3f}")
        print(f"  KS Distance:     {ks_d:.3f}")

    summary = {
        "agent_a": agent_a_class.__name__,
        "agent_b": agent_b_class.__name__,
        "direction_name": direction_name,
        "avg_pareto_distance": statistics.mean(pareto_distances) if pareto_distances else 0.0,
        "avg_nash_distance": statistics.mean(nash_distances) if nash_distances else 0.0,
        "avg_kalai_distance": statistics.mean(kalai_distances) if kalai_distances else 0.0,
        "avg_ks_distance": statistics.mean(ks_distances) if ks_distances else 0.0,
        "avg_agreement_rate": statistics.mean(agreement_rates) if agreement_rates else 0.0,
        "avg_utility_a_across_scenarios": statistics.mean(avg_utils_a) if avg_utils_a else 0.0,
        "avg_utility_b_across_scenarios": statistics.mean(avg_utils_b) if avg_utils_b else 0.0,
    }

    print("\n--- Average across all scenarios ---")
    print(f"Pareto Distance: {summary['avg_pareto_distance']:.3f}")
    print(f"Nash Distance:   {summary['avg_nash_distance']:.3f}")
    print(f"Kalai Distance:  {summary['avg_kalai_distance']:.3f}")
    print(f"KS Distance:     {summary['avg_ks_distance']:.3f}")
    print(f"Agreement Rate:  {summary['avg_agreement_rate']:.3f}")
    print(f"Avg Utility A:   {summary['avg_utility_a_across_scenarios']:.3f}")
    print(f"Avg Utility B:   {summary['avg_utility_b_across_scenarios']:.3f}")

    return {
        "direction_name": direction_name,
        "agent_a": agent_a_class.__name__,
        "agent_b": agent_b_class.__name__,
        "sessions_per_scenario": sessions,
        "scenarios": scenario_results,
        "summary": summary,
    }


if __name__ == "__main__":
    SESSIONS = 30

    results = {}

    # ReactiveT4T as  buyer, Micro as  seller
    forward_key = f"{ReactiveT4TNegotiator.__name__}_vs_{MicroNegotiator.__name__}"
    results[forward_key] = evaluate_direction(
        ReactiveT4TNegotiator,
        MicroNegotiator,
        "ReactiveT4TNegotiator vs MicroNegotiator",
        sessions=SESSIONS,
    )

    # Micro as  buyer, ReactiveT4T as seler 
    reverse_key = f"{MicroNegotiator.__name__}_vs_{ReactiveT4TNegotiator.__name__}"
    results[reverse_key] = evaluate_direction(
        MicroNegotiator,
        ReactiveT4TNegotiator,
        "MicroNegotiator vs ReactiveT4TNegotiator",
        sessions=SESSIONS,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "saved_results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "final_results12.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved results to: {output_file}")