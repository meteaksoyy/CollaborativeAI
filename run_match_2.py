# run_match_2.py
# Two-issue (price + quantity) negotiation comparison.
# Run: python run_match_2.py

from __future__ import annotations

from negmas import SAOMechanism, make_issue
from negmas.sao import SAONegotiator

from microAgent_2issues import MicroNegotiator, MicroConfig as MicroCfg2

from base_agent_simona import BaseNegotiator

from ReactiveT4TwLateTimePressure_2issues import (
    ReactiveT4TNegotiator,
    ReactiveT4TConfig as rct4tCfg2,
    PriceQuantityUFun,
)
from timeBasedWithOutcomeEnum_2issues import (
    TimeBasedWithOutcomeEnumNegotiator,
    TimeBasedWithOutcomeEnumConfig as toeCfg2,
)

MAX_PRICE = 99
MAX_QTY = 10
N_STEPS = 100


def make_ufuns(price_weight: float = 0.6, reserved_value: float = 0.3):
    buyer = PriceQuantityUFun(
        MAX_PRICE,
        MAX_QTY,
        prefer_low_price=True,
        price_weight=price_weight,
        reserved_value=reserved_value,
    )
    seller = PriceQuantityUFun(
        MAX_PRICE,
        MAX_QTY,
        prefer_low_price=False,
        price_weight=price_weight,
        reserved_value=reserved_value,
    )
    return buyer, seller


def run_match(
    agent_a: SAONegotiator,
    agent_b: SAONegotiator,
    ufun_a,
    ufun_b,
    *,
    plot: bool = True,
):
    issues = [
        make_issue(name="price", values=list(range(MAX_PRICE + 1))),
        make_issue(name="quantity", values=list(range(1, MAX_QTY + 1))),
    ]
    session = SAOMechanism(issues=issues, n_steps=N_STEPS)
    session.add(agent_a, ufun=ufun_a)
    session.add(agent_b, ufun=ufun_b)

    result = session.run()
    print(f"Agreement: {result.agreement}, Rounds: {result.step}")
    if result.agreement is not None:
        print(f"  {agent_a.name} utility: {ufun_a(result.agreement):.3f}")
        print(f"  {agent_b.name} utility: {ufun_b(result.agreement):.3f}")
    else:
        print("  No agreement reached.")

    if plot:
        session.plot()

    return result


if __name__ == "__main__":
    # --------------------------------------------------
    # Match 1: MicroNegotiator (buyer) vs ReactiveT4T (seller)
    # --------------------------------------------------
    print("=" * 50)
    print("Match 1: Micro (buyer) vs ReactiveT4T (seller)")
    print("=" * 50)
    buyer_ufun, seller_ufun = make_ufuns()
    run_match(
        MicroNegotiator(
            "micro_buyer",
            MicroCfg2(
                reserved_value=0.30, power=2.5, accept_slack=0.02, debug_every=10
            ),
        ),
        ReactiveT4TNegotiator(
            "reactive_seller",
            rct4tCfg2(concession_threshold=0.03, time_pressure=0.85),
        ),
        buyer_ufun,
        seller_ufun,
    )

    # --------------------------------------------------
    # Match 2: MicroNegotiator (buyer) vs TimeBasedOutcomeEnum (seller)
    # --------------------------------------------------
    print("=" * 50)
    print("Match 2: Micro (buyer) vs TimeBased (seller)")
    print("=" * 50)
    buyer_ufun, seller_ufun = make_ufuns()
    run_match(
        MicroNegotiator(
            "micro_buyer",
            MicroCfg2(
                reserved_value=0.30, power=2.5, accept_slack=0.02, debug_every=10
            ),
        ),
        TimeBasedWithOutcomeEnumNegotiator(
            "timebased_seller",
            toeCfg2(reserved_value=0.40, power=6.0, accept_slack=0.01, debug_every=0),
        ),
        buyer_ufun,
        seller_ufun,
    )

    # --------------------------------------------------
    # Match 3: ReactiveT4T (buyer) vs TimeBasedOutcomeEnum (seller)
    # --------------------------------------------------
    print("=" * 50)
    print("Match 3: ReactiveT4T (buyer) vs TimeBased (seller)")
    print("=" * 50)
    buyer_ufun, seller_ufun = make_ufuns()
    run_match(
        ReactiveT4TNegotiator(
            "reactive_buyer",
            rct4tCfg2(concession_threshold=0.03, time_pressure=0.85),
        ),
        TimeBasedWithOutcomeEnumNegotiator(
            "timebased_seller",
            toeCfg2(reserved_value=0.40, power=6.0, accept_slack=0.01, debug_every=0),
        ),
        buyer_ufun,
        seller_ufun,
    )

    # --------------------------------------------------
    # Match 4: ReactiveT4T (buyer) vs BaseNegotiator/Simona (seller)
    # --------------------------------------------------
    print("=" * 50)
    print("Match 4: ReactiveT4T (buyer) vs Simona/Base (seller)")
    print("=" * 50)
    buyer_ufun, seller_ufun = make_ufuns()
    run_match(
        ReactiveT4TNegotiator(
            "reactive_buyer",
            rct4tCfg2(concession_threshold=0.03, time_pressure=0.85),
        ),
        BaseNegotiator(name="simona_seller", gamma=1.0, reserved_value=0.4),
        buyer_ufun,
        seller_ufun,
    )
