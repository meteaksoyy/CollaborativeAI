# CollaborativeAI

CollaborativeAI is a research-oriented automated negotiation framework built using **NegMAS**.  
The project implements and evaluates multiple negotiating agents in a shared discrete bargaining domain, allowing direct comparison of different negotiation strategies.

---

## Overview

This repository includes:

- Multiple custom negotiation agents (MiCRO-style, time-based, reactive variants)
- A centralized match runner (`run_match.py`) for controlled experiments
- Configurable negotiation parameters (deadline, concession power, reservation values)
- Linear utility functions for price-based negotiation
- Built-in visualization of negotiation traces and Pareto outcomes

The primary objective is to experimentally evaluate how different agent strategies perform against each other under identical negotiation conditions.

---

## Negotiation Domain

Current experiments use a **single-issue price negotiation**:

- Price range: `0–99`
- Buyer prefers lower prices
- Seller prefers higher prices
- Linear utility functions
- Discrete SAO (Stacked Alternating Offers) protocol

Utilities are private to each agent, consistent with standard negotiation assumptions.

---

## Implemented Agent Types

### MiCRO-Style Agents
- Full outcome enumeration
- Utility-sorted outcome space
- Time-based aspiration function (Boulware-like concession)

### Time-Based Agents
- Concession driven by relative time
- Adjustable concession power parameter

### Reactive Agents
- Concede only when opponent improves
- Optional late-stage time pressure mechanisms

All agents can be matched directly against each other.

---

## Running a Match

To execute a negotiation match:

```bash
python run_match.py
