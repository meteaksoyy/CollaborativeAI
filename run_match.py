# run_match.py
# Run:
#   python run_match.py
#
# Notes:
# - Agents are imported from your files and pitted against each other.
# - Utilities are defined ONCE here to avoid mismatched reserved_value handling between files.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from negmas import SAOMechanism, make_issue
from negmas.sao import SAONegotiator

# Import agents
from microAgent import MicroNegotiator, MicroConfig as MicroCfg1
from ReactiveT4TwLateTimePressure import (
    ReactiveT4TNegotiator,
    ReactiveT4TConfig as rct4tCfg,
)
from timeBasedWithOutcomeEnum import (
    TimeBasedWithOutcomeEnumNegotiator,
    TimeBasedWithOutcomeEnumConfig as toeCfg,
)


# ----------------------------
# Shared utility for the match
# ----------------------------


class PriceUFun:
    """Linear price utility for a single-issue price domain: outcome=(price,)"""

    def __init__(self, max_price: int, prefer_low: bool):
        self.max_price = int(max_price)
        self.prefer_low = bool(prefer_low)

    def __call__(self, outcome) -> float:
        if outcome is None:
            return 0.0
        p = int(outcome[0])
        x = p / self.max_price if self.max_price > 0 else 0.0
        u = (1.0 - x) if self.prefer_low else x
        return max(0.0, min(1.0, float(u)))


# ----------------------------
# Match runner
# ----------------------------


def run_match(
    agent_a: SAONegotiator,
    agent_b: SAONegotiator,
    ufun_a,
    ufun_b,
    *,
    max_price: int = 99,
    n_steps: int = 100,
    plot: bool = True,
):
    issues = [make_issue(name="price", values=max_price + 1)]  # 0..max_price
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    session.add(agent_a, ufun=ufun_a)
    session.add(agent_b, ufun=ufun_b)

    result = session.run()
    print(f"Agreement: {result.agreement}, Rounds: {result.step}")

    if result.agreement is not None:
        print(f"AgentA utility: {ufun_a(result.agreement):.3f}")
        print(f"AgentB utility: {ufun_b(result.agreement):.3f}")

    if plot:
        session.plot()

    return result


if __name__ == "__main__":
    MAX_PRICE = 99
    N_STEPS = 100

    # Decide which agent plays "buyer" vs "seller"
    # Buyer prefers low price; Seller prefers high price.
    buyer_ufun = PriceUFun(MAX_PRICE, prefer_low=True)
    seller_ufun = PriceUFun(MAX_PRICE, prefer_low=False)
    seller_ufun_t4t = PriceUFun(MAX_PRICE, prefer_low=False)

    # Instantiate agents from your files
    # microAgent.py agent (time/aspiration index type)
    micro_buyer = MicroNegotiator(
        "micro_buyer",
        MicroCfg1(reserved_value=0.30, power=2.5, accept_slack=0.02, debug_every=10),
    )

    # ReactiveT4TwLateTimePressure.py agent (reactive/tit-for-tat-like)
    # IMPORTANT: this agent reads `self.ufun.reserved_value` internally.
    # Your runner utility above does NOT provide reserved_value, so to avoid internal errors,
    # we attach the attribute to the utility object:
    seller_ufun_t4t.reserved_value = 0.30  # required by T4TNegotiatior's code path

    reactive_seller = ReactiveT4TNegotiator(
        "reactive_seller",
        rct4tCfg(concession_threshold=0.03, time_pressure=0.85),
    )
    reactive_buyer = ReactiveT4TNegotiator(
        "reactive_buyer",
        rct4tCfg(concession_threshold=0.03, time_pressure=0.85),
    )

    timebased_seller = TimeBasedWithOutcomeEnumNegotiator(
        "Time_Based_Outcome_Enumerator_Seller",
        toeCfg(
            reserved_value=0.40,  # walk-away utility
            power=6.0,  # very boulware (slow concession)
            accept_slack=0.01,  # tight acceptance margin
            debug_every=0,  # print every 10 steps (or 0 to disable)
        ),
    )

    # Run the match: MicroNegotiator (buyer) vs ReactiveT4TNegotiator (seller)
    run_match(
        micro_buyer,
        reactive_seller,
        buyer_ufun,
        seller_ufun_t4t,
        max_price=MAX_PRICE,
        n_steps=N_STEPS,
        plot=True,
    )
    run_match(
        micro_buyer,
        timebased_seller,
        buyer_ufun,
        seller_ufun,
        max_price=MAX_PRICE,
        n_steps=N_STEPS,
        plot=True,
    )
    run_match(
        reactive_buyer,
        timebased_seller,
        buyer_ufun,
        seller_ufun,
        max_price=MAX_PRICE,
        n_steps=N_STEPS,
        plot=True,
    )
