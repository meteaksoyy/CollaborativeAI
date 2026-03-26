# mic.py
# Run: python3 mic.py

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios

from reactive_agent import ReactiveT4TNegotiator, ReactiveT4TConfig
from microNegotiator import MicroNegotiator

N_STEPS = 100


def instantiate_reactive(name: str):
    return ReactiveT4TNegotiator(name=name, cfg=ReactiveT4TConfig())


def instantiate_micro(name: str):
    return MicroNegotiator(name=name)


def run_match(agent_a, agent_b, ufun_a, ufun_b, issues, title: str, plot: bool = True):
    print("=" * 60)
    print(title)
    print("=" * 60)

    mechanism = SAOMechanism(issues=issues, n_steps=N_STEPS)
    mechanism.add(agent_a, ufun=ufun_a)
    mechanism.add(agent_b, ufun=ufun_b)

    result = mechanism.run()

    print(f"Agreement: {result.agreement}")
    print(f"Rounds: {mechanism.state.step}")

    if result.agreement is not None:
        print(f"{agent_a.name} utility: {float(ufun_a(result.agreement)):.3f}")
        print(f"{agent_b.name} utility: {float(ufun_b(result.agreement)):.3f}")
    else:
        print("No agreement reached.")

    if plot:
        mechanism.plot()

    print()
    return result


if __name__ == "__main__":
    # 2-issue equal weights
    issues, u_buyer, u_seller = EvaluationScenarios.getDoubleIssueA()

    run_match(
        instantiate_micro("micro_buyer"),
        instantiate_reactive("reactive_seller"),
        u_buyer,
        u_seller,
        issues,
        title="Match 1: Micro (buyer) vs ReactiveT4T (seller)",
        plot=True,
    )

    # reload scenario
    issues, u_buyer, u_seller = EvaluationScenarios.getDoubleIssueA()

    run_match(
        instantiate_reactive("reactive_buyer"),
        instantiate_micro("micro_seller"),
        u_buyer,
        u_seller,
        issues,
        title="Match 2: ReactiveT4T (buyer) vs Micro (seller)",
        plot=True,
    )