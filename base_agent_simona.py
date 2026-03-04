from negmas import SAOMechanism, AspirationNegotiator, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAONegotiator, ResponseType

# making a linear time-based agent
class BaseNegotiator(SAONegotiator):

    def __init__(self, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        # concession speed
        self.gamma = gamma    

    def propose(self, state):
        # normalized time 0-1
        t = state.relative_time

        u_max = 1.0    # max utility
        u_min = self.ufun.reserved_value or 0.0                 # reservation value

        # linear concession
        target = u_max - (u_max - u_min) * (t ** self.gamma)    # current utility the agent aims for

        # chosing the best bid over the target
        best_offer = None
        best_util = 10

        for outcome in self.nmi.outcomes:                # all possible combinations of issues
            util = self.ufun(outcome)
            if util >= target and util < best_util:      # pick the smallest utility above the target to avoid over-demanding
                best_offer = outcome
                best_util = util

        if best_offer is None:                           # if nothing meets the target, pick a random outcome
            best_offer = self.nmi.random_outcome()

        return best_offer
    
    def respond(self, state):
        offer = state.current_offer
        
        if offer is None:
            return ResponseType.REJECT_OFFER
        
        t = state.relative_time
        u_max = 1.0
        u_min = self.ufun.reserved_value or 0.0
        target = u_max - (u_max - u_min) * (t ** self.gamma)

        if self.ufun(offer) >= target:                     # accept if offered utility is better than the target
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER
    
# defining issues
price = make_issue(name="price", values=11)
max_price = max(price.values)

quality = make_issue(name="quality", values=6)
max_quality = max(quality.values)

issues = [price, quality]

# defining utility functions for both agents
ufun1 = LinearAdditiveUtilityFunction(
    weights=[0.7, 0.3],
    issues=issues,
    # buyer prefers low price and high quality
    values={
        "price": lambda x: 1.0 - x / max_price,
        "quality": lambda x: x / max_quality,
    },
)
ufun1.reserved_value = 0.2

ufun2 = LinearAdditiveUtilityFunction(
    weights=[0.8, 0.2],
    issues=issues,
    # seller prefers high price and low quality
    values={
        "price": lambda x: x / max_price,
        "quality": lambda x: 1.0 - x / max_quality,
    },
)
ufun2.reserved_value = 0.15

# creating a mechanism
mechanism = SAOMechanism(issues=issues, n_steps=100)

# adding the agents to the mechanism
agent1 = BaseNegotiator(name="Agent 1 - Buyer", ufun=ufun1, gamma=1.2)  # buyer consedes slower
mechanism.add(agent1)
agent2 = BaseNegotiator(name="Agent 2 - Seller", ufun=ufun2, gamma=0.8) # seller consedes faster
mechanism.add(agent2)

# running the negotiation
agreement = mechanism.run()
print("Agreement: ", agreement)
mechanism.plot()
