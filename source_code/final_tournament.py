from base_agent import BaseNegotiator
from improved_base_agent import ImprovedBaseNegotiator
from reactive_agent import ReactiveT4TNegotiator
from hybrid_agent import HybridNegotiator
import itertools
import tqdm
from negmas import SAOMechanism
from evaluation_scenarios import EvaluationScenarios
# load in all the agents
agents = [BaseNegotiator, ImprovedBaseNegotiator, ReactiveT4TNegotiator, HybridNegotiator]
import pickle
class TournamentEvaluation:


    @staticmethod       
    def run_tournament(scenarios, agents, sessions=30):

        results = []


        # permutate all agent combinations
        for agent_a, agent_b in itertools.permutations(agents, 2):

            # for every scenario
            for name, (issue, u_a, u_b) in scenarios.items():

                agreements = 0

                a_utils, b_utils = [], []

                for _ in tqdm.tqdm(range(sessions), desc=f"{agent_a.__name__} vs {agent_b.__name__} ({name})"):

                    mechanism = SAOMechanism(issues=issue, n_steps = 100)

                    mechanism.add(agent_a(name = agent_a.__name__), ufun=u_a)
                    mechanism.add(agent_b(name = agent_b.__name__), ufun=u_b)

                    result = mechanism.run()

                    if result.agreement:

                        agreements += 1

                        a_utils.append(u_a(result.agreement))
                        b_utils.append(u_b(result.agreement))

                    # if _ == 0:
                    #     mechanism.plot()

                results.append({
                    "agent_A": agent_a.__name__,
                    "agent_B": agent_b.__name__,
                    "scenario": name,
                    "agreement_rate": agreements / sessions,
                    "avg_A_utility": sum(a_utils)/len(a_utils) if a_utils else 0,
                    "avg_B_utility": sum(b_utils)/len(b_utils) if b_utils else 0,
                })

        return results

scenarios = {
            "Single Issue": EvaluationScenarios.getSingleIssue(),
            "Double Issue Equal": EvaluationScenarios.getDoubleIssueA(),
            "Double Issue Unequal": EvaluationScenarios.getDoubleIssueB(),
            "Multi Issue Equal": EvaluationScenarios.getMultipleIssueA(),
            "Multi Issue Unequal": EvaluationScenarios.getMultipleIssueB(),
        }
res = TournamentEvaluation.run_tournament(scenarios, agents, 10)

filename = f"final_results"
with open(filename, "wb") as f:
    pickle.dump(res, f)
print(f"Saved {filename}")