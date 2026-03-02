from negmas import SAOMechanism, TimeBasedConcedingNegotiator, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

# Define what we're negotiating about
issues = [make_issue(name="price", values=100)]

# Create negotiation session (Stacked Alternating Offers)
session = SAOMechanism(issues=issues, n_steps=50)

buyer = TimeBasedConcedingNegotiator(name="buyer")
seller = TimeBasedConcedingNegotiator(name="seller")

buyer_ufun = LUFun.random(issues=issues, reserved_value=0.0)
seller_ufun = LUFun.random(issues=issues, reserved_value=0.0)

session.add(buyer, ufun=buyer_ufun)
session.add(seller, ufun=seller_ufun)

print(buyer_ufun((0,)))
print(buyer_ufun((99,)))
print(seller_ufun((0,)))
print(seller_ufun((99,)))

# Run and get result
result = session.run()
print(f"Agreement: {result.agreement}, Rounds: {result.step}")
