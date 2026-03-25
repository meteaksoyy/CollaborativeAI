import pickle
import matplotlib.pyplot as plt
import numpy as np


def compare_bidirectional(results1, results2, title):

    scenarios = [r["scenario"] for r in results1]

    buyer1 = [r["average_buyer_utility"] for r in results1]
    seller1 = [r["average_seller_utility"] for r in results1]

    buyer2 = [r["average_buyer_utility"] for r in results2]
    seller2 = [r["average_seller_utility"] for r in results2]

    x = np.arange(len(scenarios))
    width = 0.18
    gap = 0.25   # controls the space between directions

    plt.figure(figsize=(12,6))

    # Base → Opponent
    plt.bar(x - width - gap/2, buyer1, width, label="Hybrid Buyer", color="#b3d1ff")
    plt.bar(x - gap/2, seller1, width, label="Aspirational Seller", color="#ffe0b3")

    # Opponent → Base
    plt.bar(x + width + gap/2, seller2, width, label="Hybrid Seller", color="#0052cc")
    plt.bar(x + gap/2, buyer2, width, label="Aspirational Buyer", color="#ff9900")

    plt.xticks(x, scenarios, rotation=40, ha="right")
    plt.ylabel("Average Utility")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # γ = 0.5


    # # with open("results_AspirationNegotiator_vs_ImprovedBaseNegotiator_gamma1.0.pkl", "rb") as f:
    # #     results12 = pickle.load(f)
    # with open("results_ImprovedBaseNegotiator_vs_RandomNegotiator_gamma2.0.pkl", "rb") as f:
    #     results13 = pickle.load(f)

    # # calculate mean of agreement rates across scenarios for each gamma
    # agreement_rates_05 = [r["agreement_rate"] for r in results1]
    # # agreement_rates_10 = [r["agreement_rate"] for r in results12]
    # agreement_rates_20 = [r["agreement_rate"] for r in results13]
    # print(f"Mean agreement rate for γ=0.5: {np.mean(agreement_rates_05):.2f}")
    # print(f"Mean agreement rate for γ=2.0: {np.mean(agreement_rates_20):.2f}")

    with open("/Users/annaglodek/Documents/CAI project/CollaborativeAI/source_code/saved_results/results_RandomNegotiator_vs_ReactiveT4T.pkl", "rb") as f:
        results1 = pickle.load(f)

    with open("/Users/annaglodek/Documents/CAI project/CollaborativeAI/source_code/saved_results/results_ReactiveT4T_vs_RandomNegotiator.pkl", "rb") as f:
        results2 = pickle.load(f)

    a = ([r["agreement_rate"] for r in results1])
    print(f"Mean agreement rate for ReactiveT4T vs Random: {np.mean(a):.2f}")
    b = ([r["agreement_rate"] for r in results2])
    print(f"Mean agreement rate for Random vs ReactiveT4T: {np.mean(b):.2f}")

    # Base vs Aspiration
    compare_bidirectional(
        results1,
        results2,
        "Hybrid vs Aspirational Average Utilities for γ = 0.5, α = 0.7 "
    )

    # # Base vs Random
    # compare_bidirectional(
    #     results3,
    #     results4,
    #     "Base vs Random Average Utilities (γ = 0.5)"
    # )