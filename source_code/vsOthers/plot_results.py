import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

folder = "saved_results"
scenarios = ["Single Issue", "Double Issue Equal", "Double Issue Unequal", "Multi Issue Equal", "Multi Issue Unequal"]

our_agents = ["ReactiveT4TNegotiator", "HybridNegotiator"]
their_agents = ["AdaptiveNegotiator", "TimeBasedAgent", "TitForTatAgent", "MicroNegotiator"]

# Load all data into a dict keyed by (agent_A, agent_B)
data = {}
for fname in os.listdir(folder):
    if not fname.endswith(".pkl"):
        continue
    with open(os.path.join(folder, fname), "rb") as f:
        records = pickle.load(f)
    for r in records:
        key = (r["agent_A"], r["agent_B"], r["scenario"])
        data[key] = r

def get(agent_a, agent_b, scenario, field):
    return data.get((agent_a, agent_b, scenario), {}).get(field, 0)


# ── Figure 1: Average utility heatmaps (our agents as rows, their agents as cols) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Average Utility: Our Agents vs Their Agents\n(averaged across all scenarios)", fontsize=14, fontweight="bold")

for row, our in enumerate(our_agents):
    for col, role in enumerate(["as Agent A (proposer first)", "as Agent B (proposer second)"]):
        ax = axes[row][col]
        matrix = np.zeros((len(their_agents), len(scenarios)))

        for i, their in enumerate(their_agents):
            for j, sc in enumerate(scenarios):
                if col == 0:  # our agent is A
                    matrix[i][j] = get(our, their, sc, "avg_A_utility")
                else:          # our agent is B
                    matrix[i][j] = get(their, our, sc, "avg_B_utility")

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=8)
        ax.set_yticks(range(len(their_agents)))
        ax.set_yticklabels(their_agents, fontsize=9)
        ax.set_title(f"{our}\n{role}", fontsize=9)

        for i in range(len(their_agents)):
            for j in range(len(scenarios)):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < matrix[i][j] < 0.8 else "white")

        plt.colorbar(im, ax=ax, fraction=0.03)

plt.tight_layout()
plt.savefig("plot_utility_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved plot_utility_heatmap.png")


# ── Figure 2: Head-to-head bar chart — our utility vs their utility per matchup ──
matchups = [(our, their) for our in our_agents for their in their_agents]
n = len(matchups)
x = np.arange(n)
width = 0.35

fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 4 * len(scenarios)))
fig.suptitle("Head-to-Head Utility: Our (blue) vs Their (orange)\n(our agent as Agent A)", fontsize=13, fontweight="bold")

for ax, sc in zip(axes, scenarios):
    our_utils = []
    their_utils = []
    labels = []

    for our, their in matchups:
        our_utils.append(get(our, their, sc, "avg_A_utility"))
        their_utils.append(get(our, their, sc, "avg_B_utility"))
        labels.append(f"{our.replace('Negotiator','').replace('Agent','')}\nvs\n{their.replace('Negotiator','').replace('Agent','')}")

    bars_our = ax.bar(x - width/2, our_utils, width, label="Our agent (A)", color="steelblue")
    bars_their = ax.bar(x + width/2, their_utils, width, label="Their agent (B)", color="darkorange")

    ax.set_title(sc, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg Utility")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)

    for bar in bars_our:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_their:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("plot_head_to_head.png", dpi=150, bbox_inches="tight")
print("Saved plot_head_to_head.png")


# ── Figure 3: Average rounds per matchup ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Average Rounds to Agreement", fontsize=13, fontweight="bold")

for ax, our in zip(axes, our_agents):
    avg_rounds = []
    labels = []
    for their in their_agents:
        vals = [get(our, their, sc, "average_rounds") for sc in scenarios]
        avg_rounds.append(np.mean(vals))
        labels.append(their.replace("Negotiator", "").replace("Agent", ""))

    bars = ax.bar(labels, avg_rounds, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_title(f"{our}", fontweight="bold")
    ax.set_ylabel("Avg Rounds (out of 100)")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="red", linestyle="--", linewidth=0.8, label="Max (100)")
    ax.legend(fontsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plot_rounds.png", dpi=150, bbox_inches="tight")
print("Saved plot_rounds.png")


# ── Figure 4: Overall win rate summary ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Overall Average Utility — Our Agents vs Each Opponent\n(both roles averaged)", fontsize=13, fontweight="bold")

x = np.arange(len(their_agents))
width = 0.35

for i, our in enumerate(our_agents):
    avg_utils = []
    for their in their_agents:
        as_a = np.mean([get(our, their, sc, "avg_A_utility") for sc in scenarios])
        as_b = np.mean([get(their, our, sc, "avg_B_utility") for sc in scenarios])
        avg_utils.append((as_a + as_b) / 2)

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, avg_utils, width, label=our)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([t.replace("Negotiator","").replace("Agent","") for t in their_agents])
ax.set_ylabel("Avg Utility (both roles)")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.legend()

plt.tight_layout()
plt.savefig("plot_summary.png", dpi=150, bbox_inches="tight")
print("Saved plot_summary.png")
