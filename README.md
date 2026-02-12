---

# Train Platform Optimization using Constraint Programming (CSP)

This project studies the optimization of train platform allocation using Constraint Programming (CSP) with **Google OR-Tools CP-SAT**.

The objective is to assign trains to the minimum number of platforms such that no temporal overlap occurs.

---

## Solver

* Google OR-Tools
* CP-SAT (SAT-based Constraint Programming solver)
* Module: `ortools.sat.python.cp_model`

---

## Problem Description

Each train is defined by:

* Arrival time
* Departure time

A safety margin is enforced between trains.

**Goal:**
Minimize the number of platforms `K` such that:

If two trains overlap (including safety margin), they cannot be assigned to the same platform.

This naturally produces a **phase transition behavior** around the minimal feasible `K`.

---

## Modeling Approaches

### 1. Decomposed Model (Graph Coloring)

* Precompute conflict graph
* Add binary constraints: `Xi != Xj`
* Fast propagation
* Highly sensitive to variable ordering heuristics

### 2. Global Scheduling Model

* Uses optional interval variables
* Uses a global non-overlapping constraint per platform
* Stronger filtering
* More robust to poor heuristics

---

## Heuristics Compared

* `lexico` — naive ordering
* `maxdeg` — highest conflict degree first (fail-first principle)
* `portfolio` — CP-SAT internal portfolio search

---

## Experimental Setup

* Instance type: Rush Hour
* Number of trains: 130
* Safety margin: 2
* Time limit per run: 5 seconds
* K varied around the phase transition region

---

## Repository Structure

```
Train-Platform-Optimization-CSP/
│
├── src/
│   └── train_platform_csp.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── decomposed_time.png
│   ├── global_time.png
│   └── results.csv
│
├── requirements.txt
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed manually:

```bash
pip install ortools pandas matplotlib
```

---

## Run the Project

From the repository root:

```bash
python src/train_platform_csp.py
```

The following files will be generated:

* `results/results.csv`
* `results/decomposed_time.png`
* `results/global_time.png`

---

## Key Observations

* The decomposed model combined with `maxdeg` provides the best raw performance.
* The global model is more stable across heuristics.
* A clear phase transition appears near the minimal feasible number of platforms.
* Heuristic choice dramatically impacts solving time near the critical region.

---

This project illustrates how modeling decisions and search heuristics strongly influence performance in Constraint Programming optimization problems.

---

