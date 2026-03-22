import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open(r"C:\Users\HP\Personal Projects\NegotatingAgent\folder\CollaborativeAI\source_code\final_results", "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(results)


def compare_agents_bidirectional(df, agent1, agent2, title):
    """
    df: DataFrame with tournament results
    agent1, agent2: names of the agents to compare
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Filter results for both directions
    results1 = df[(df["agent_A"] == agent1) & (df["agent_B"] == agent2)].sort_values("scenario")
    results2 = df[(df["agent_A"] == agent2) & (df["agent_B"] == agent1)].sort_values("scenario")

    scenarios = results1["scenario"].tolist()
    buyer1 = results1["avg_A_utility"].tolist()
    seller1 = results1["avg_B_utility"].tolist()
    buyer2 = results2["avg_B_utility"].tolist()
    seller2 = results2["avg_A_utility"].tolist()

    x = np.arange(len(scenarios))
    width = 0.18
    gap = 0.25

    plt.figure(figsize=(12,6))
    plt.bar(x - width - gap/2, buyer1, width, label=f"{agent1} Buyer", color="#b3d1ff")
    plt.bar(x - gap/2, seller1, width, label=f"{agent2} Seller", color="#ffe0b3")
    plt.bar(x + gap/2, buyer2, width, label=f"{agent2} Buyer", color="#ff9900")
    plt.bar(x + width + gap/2, seller2, width, label=f"{agent1} Seller", color="#0052cc")

    plt.xticks(x, scenarios, rotation=40, ha="right")
    plt.ylabel("Average Utility")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

print(df)

compare_agents_bidirectional(df, "HybridNegotiator", "ReactiveT4TNegotiator", "")