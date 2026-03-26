import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/annaglodek/Documents/CAI project/CollaborativeAI/source_code/vsOthers/saved_results/final_results12.pkl"
output_dir = "/Users/annaglodek/Documents/CAI project/CollaborativeAI/source_code/vsOthers/saved_results"

with open(file_path, "rb") as f:
    results = pickle.load(f)

rows = []

for matchup_name, matchup_data in results.items():
    agent_a = matchup_data["agent_a"]
    agent_b = matchup_data["agent_b"]

    for scenario_name, scenario_data in matchup_data["scenarios"].items():
        rows.append({
            "Matchup": matchup_name,
            "Agent A": agent_a,
            "Agent B": agent_b,
            "Scenario": scenario_name,
            "Avg Utility A": scenario_data["avg_utility_a"],
            "Avg Utility B": scenario_data["avg_utility_b"],
            "Agreement Rate": scenario_data["agreement_rate"],
            "Pareto Distance": scenario_data["pareto_distance"],
            "Nash Distance": scenario_data["nash_distance"],
            "Kalai Distance": scenario_data["kalai_distance"],
            "KS Distance": scenario_data["ks_distance"],
        })

df = pd.DataFrame(rows)

print("\nScenario-level results:\n")
print(df)

output_csv = os.path.join(output_dir, "scenario_results_table11.csv")
df.to_csv(output_csv, index=False)
print(f"\nSaved table to: {output_csv}")

summary_rows = []

for matchup_name, matchup_data in results.items():
    summary = matchup_data["summary"]
    summary_rows.append({
        "Matchup": matchup_name,
        "Agent A": summary["agent_a"],
        "Agent B": summary["agent_b"],
        "Avg Pareto Distance": summary["avg_pareto_distance"],
        "Avg Nash Distance": summary["avg_nash_distance"],
        "Avg Kalai Distance": summary["avg_kalai_distance"],
        "Avg KS Distance": summary["avg_ks_distance"],
        "Avg Agreement Rate": summary["avg_agreement_rate"],
        "Avg Utility A Across Scenarios": summary["avg_utility_a_across_scenarios"],
        "Avg Utility B Across Scenarios": summary["avg_utility_b_across_scenarios"],
    })

summary_df = pd.DataFrame(summary_rows)

print("\nSummary results:\n")
print(summary_df)

summary_csv = os.path.join(output_dir, "summary_results_table11.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary table to: {summary_csv}")


# final plot for comparing agents 
forward_key = "ReactiveT4TNegotiator_vs_MicroNegotiator"
reverse_key = "MicroNegotiator_vs_ReactiveT4TNegotiator"

if forward_key in results and reverse_key in results:
    forward = results[forward_key]
    reverse = results[reverse_key]

    scenario_order = [
        "Single Issue",
        "Double Issue Equal",
        "Double Issue Unequal",
        "Multi Issue Equal",
        "Multi Issue Unequal",
    ]

    def short_label(agent_name):
        mapping = {
            "ReactiveT4TNegotiator": "ReactiveT4T",
            "MicroNegotiator": "Micro",
        }
        return mapping.get(
            agent_name,
            agent_name.replace("Negotiator", "").replace("Agent", "")
        )

    left_agent_name = short_label(forward["agent_a"])
    right_agent_name = short_label(forward["agent_b"])

    left_agent_buyer = []
    right_agent_seller = []
    left_agent_seller = []
    right_agent_buyer = []

    for scenario in scenario_order:
        forward_scenario = forward["scenarios"][scenario]
        reverse_scenario = reverse["scenarios"][scenario]

        # forward: left agent is buyer, right agent is seller
        left_agent_buyer.append(forward_scenario["avg_utility_a"])
        right_agent_seller.append(forward_scenario["avg_utility_b"])

        # reverse: left agent is seller, right agent is buyer
        left_agent_seller.append(reverse_scenario["avg_utility_b"])
        right_agent_buyer.append(reverse_scenario["avg_utility_a"])

    x = np.arange(len(scenario_order))
    width = 0.18
    gap = 0.25

    plt.figure(figsize=(12, 6))

    # First direction
    plt.bar(
        x - width - gap / 2,
        left_agent_buyer,
        width,
        label=f"{left_agent_name} Buyer",
        color="#b3d1ff"
    )
    plt.bar(
        x - gap / 2,
        right_agent_seller,
        width,
        label=f"{right_agent_name} Seller",
        color="#ffe0b3"
    )

    # Flipped direction
    plt.bar(
        x + gap / 2,
        left_agent_seller,
        width,
        label=f"{left_agent_name} Seller",
        color="#0052cc"
    )
    plt.bar(
        x + width + gap / 2,
        right_agent_buyer,
        width,
        label=f"{right_agent_name} Buyer",
        color="#ff9900"
    )

    plt.xticks(x, scenario_order, rotation=40, ha="right")
    plt.xlabel("Scenario")
    plt.ylabel("Average Utility")
    plt.title(f"{left_agent_name} vs {right_agent_name} Average Utilities")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(
        output_dir,
        f"{left_agent_name}_vs_{right_agent_name}_role_based_avg_utilities.png"
    )
    plt.savefig(combined_plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {combined_plot_path}")

else:
    print("\nSkipped combined role-based plot because both matchup directions were not found in the pickle.")