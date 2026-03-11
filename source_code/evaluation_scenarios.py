from negmas.preferences import LinearAdditiveUtilityFunction
from negmas import SAOMechanism, AspirationNegotiator, make_issue


class EvaluationScenarios:

    # method to get the single issue negotation scenario
    def getSingleIssue():

        issues = []
        u_seller = None


        price = make_issue(name="price", values=11)
        max_price = max(price.values)
        issues = [price]

        u_buyer = LinearAdditiveUtilityFunction(
            weights=[1.0],  
            issues=issues,
            values={
                "price": lambda x: 1.0 - x / max_price,  # buyer prefers low price
            },
        ).scale_max(1.0)

        u_seller = LinearAdditiveUtilityFunction(
            weights=[1.0],  
            issues=issues,
            values={
                "price": lambda x: x / max_price,  # seller prefers high price
            },
        ).scale_max(1.0)

        return issues, u_buyer, u_seller


    # get double issue equal weight scenario
    def getDoubleIssueA():

        price = make_issue(name="price", values=11)
        location = make_issue(name="location", values=["A", "B", "C"])
        issues = [price, location]

        u_buyer = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: 1.0 - x / max(price.values),  # buyer prefers low price
                "location": lambda x: 1.0 if x == "A" else 0.5 if x == "B" else 0.0,  # buyer prefers location A > B > C
            },
            weights={ # add equal weights 
                "price": 0.5,   
                "location": 0.5,
            },
        ).scale_max(1.0)

        u_seller = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: x / max(price.values),  # seller prefers high price
                "location": lambda x: 1.0 if x == "C" else 0.5 if x == "B" else 0.0,  # seller prefers location C > B > A
            },  
            weights={ # add equal weights 
                "price": 0.5,   
                "location": 0.5,
            },
        ).scale_max(1.0)

        return issues, u_buyer, u_seller

    # get double issue different weight scenario
    def getDoubleIssueB():

        price = make_issue(name="price", values=11)
        location = make_issue(name="location", values=["A", "B", "C"])
        issues = [price, location]

        u_buyer = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: 1.0 - x / max(price.values),  # buyer prefers low price
                "location": lambda x: 1.0 if x == "A" else 0.5 if x == "B" else 0.0,  # buyer prefers location A > B > C
            },
            weights={ # add different weights 
                "price": 0.6,   
                "location": 0.4,
            },
        ).scale_max(1.0)

        u_seller = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: x / max(price.values),  # seller prefers high price
                "location": lambda x: 1.0 if x == "C" else 0.5 if x == "B" else 0.0,  # seller prefers location C > B > A
            },  
            weights={ # add different weights 
                "price": 0.8,   
                "location": 0.2,
            },
        ).scale_max(1.0)

        return issues, u_buyer, u_seller

    # get multiple issue scenario
    def getMultipleIssueA():

        # define our issues
        price = make_issue(name="price", values=11)
        location = make_issue(name="location", values=["A", "B", "C"])
        energylabel = make_issue(name="energy_label", values=["A", "B", "C", "D"])
        size = make_issue(name="size", values=range(50, 100))
        propertycondition = make_issue(name="property_condition", values=["new", "old", "renovated"])

        issues = [price, location, energylabel, size, propertycondition]

        u_buyer = LinearAdditiveUtilityFunction(
            issues=issues,
            values={    
                "price": lambda x: 1.0 - x / max(price.values),  # buyer prefers low price
                "location": lambda x: 1.0 if x == "A" else 0.5 if x == "B" else 0.0,  # buyer prefers location A > B > C
                "energy_label": lambda x: 1.0 if x == "A" else 0.75 if x == "B" else 0.5 if x == "C" else 0.25,  # buyer prefers energy label A > B > C > D
                "size": lambda x: (x - min(size.values)) / (max(size.values) - min(size.values)),  # buyer prefers larger size
                "property_condition": lambda x: 1.0 if x == "new" else 0.5 if x == "renovated" else 0.0,  # buyer prefers new > renovated > used
            },
            weights={
                "price": 0.2,   
                "location": 0.2,
                "energy_label": 0.2,
                "size": 0.2,
                "property_condition": 0.2,
            },  
        ).scale_max(1.0)

        u_seller = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: x / max(price.values),  # seller prefers high price
                "location": lambda x: 1.0 if x == "C" else 0.5 if x == "B" else 0.0,  # seller prefers location C > B > A
                "energy_label": lambda x: 1.0 if x == "D" else 0.75 if x == "C" else 0.5 if x == "B" else 0.25,  # seller prefers energy label D > C > B > A
                "size": lambda x: (x - min(size.values)) / (max(size.values) - min(size.values)),  # seller also prefers larger size
                "property_condition": lambda x: 1.0 if x == "used" else 0.5 if x == "renovated" else 0.0,  # seller prefers used > renovated > new
            },
            weights={
                "price": 0.2,
                "location": 0.2,
                "energy_label": 0.2,
                "size": 0.2,
                "property_condition": 0.2,
            },
        ).scale_max(1.0)

        return issues, u_buyer, u_seller


    def getMultipleIssueB():

        # define our issues
        price = make_issue(name="price", values=11)
        location = make_issue(name="location", values=["A", "B", "C"])
        energylabel = make_issue(name="energy_label", values=["A", "B", "C", "D"])
        size = make_issue(name="size", values=range(50, 100))
        propertycondition = make_issue(name="property_condition", values=["new", "old", "renovated"])

        issues = [price, location, energylabel, size, propertycondition]

        u_buyer = LinearAdditiveUtilityFunction(
            issues=issues,
            values={    
                "price": lambda x: 1.0 - x / max(price.values),  # buyer prefers low price
                "location": lambda x: 1.0 if x == "A" else 0.5 if x == "B" else 0.0,  # buyer prefers location A > B > C
                "energy_label": lambda x: 1.0 if x == "A" else 0.75 if x == "B" else 0.5 if x == "C" else 0.25,  # buyer prefers energy label A > B > C > D
                "size": lambda x: (x - min(size.values)) / (max(size.values) - min(size.values)),  # buyer prefers larger size
                "property_condition": lambda x: 1.0 if x == "new" else 0.5 if x == "renovated" else 0.0,  # buyer prefers new > renovated > used
            },
            weights={
                "price": 0.35,   
                "location": 0.2,
                "energy_label": 0.15,
                "size": 0.15,
                "property_condition": 0.15,
            },  
        ).scale_max(1.0)

        u_seller = LinearAdditiveUtilityFunction(
            issues=issues,
            values={
                "price": lambda x: x / max(price.values),  # seller prefers high price
                "location": lambda x: 1.0 if x == "C" else 0.5 if x == "B" else 0.0,  # seller prefers location C > B > A
                "energy_label": lambda x: 1.0 if x == "D" else 0.75 if x == "C" else 0.5 if x == "B" else 0.25,  # seller prefers energy label D > C > B > A
                "size": lambda x: (x - min(size.values)) / (max(size.values) - min(size.values)),  # seller also prefers larger size
                "property_condition": lambda x: 1.0 if x == "used" else 0.5 if x == "renovated" else 0.0,  # seller prefers used > renovated > new
            },
            weights={
                "price": 0.5,
                "location": 0.15,
                "energy_label": 0.1,
                "size": 0.1,
                "property_condition": 0.15,
            },
        ).scale_max(1.0)

        return issues, u_buyer, u_seller


                