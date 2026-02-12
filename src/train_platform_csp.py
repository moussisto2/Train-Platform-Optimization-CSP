import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from ortools.sat.python import cp_model


# ==========================================================
# 1) INSTANCE (RUSH HOUR)
# ==========================================================

@dataclass(frozen=True)
class Train:
    id: int
    arr: int
    dep: int


def generate_hard_instance(n_trains: int, seed: int = 42) -> List[Train]:
    rng = random.Random(seed)
    horizon = n_trains * 3
    mean_dur = 45
    trains = []

    for i in range(n_trains):
        if rng.random() < 0.8:
            base = horizon // 2 + rng.randint(-horizon // 8, horizon // 8)
        else:
            base = rng.randint(0, max(0, horizon - mean_dur - 1))

        dur = max(20, mean_dur + rng.randint(-15, 25))
        dep = base + dur
        trains.append(Train(i, base, dep))

    return trains


def check_overlap(t1: Train, t2: Train, safety: int = 2) -> bool:
    return not (t1.dep + safety <= t2.arr or t2.dep + safety <= t1.arr)


def compute_conflicts(trains: List[Train], safety: int = 2) -> Dict[int, List[int]]:
    adj = {t.id: [] for t in trains}
    for i in range(len(trains)):
        for j in range(i + 1, len(trains)):
            if check_overlap(trains[i], trains[j], safety):
                adj[trains[i].id].append(trains[j].id)
                adj[trains[j].id].append(trains[i].id)
    return adj


# ==========================================================
# 2) SOLVEURS
# ==========================================================

def _configure_solver(strategy: str, tlimit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tlimit
    solver.parameters.num_search_workers = 1

    if strategy == "portfolio":
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    else:
        solver.parameters.search_branching = cp_model.FIXED_SEARCH

    return solver


def solve_decomposed(
    n_trains: int,
    k_quais: int,
    adj: Dict[int, List[int]],
    strategy: str,
    symmetry: bool,
    tlimit: float,
) -> Dict:
    model = cp_model.CpModel()

    x = [model.NewIntVar(1, k_quais, f"t{i}") for i in range(n_trains)]

    for i in range(n_trains):
        for j in adj[i]:
            if i < j:
                model.Add(x[i] != x[j])

    if symmetry and k_quais > 0:
        model.Add(x[0] == 1)

    if strategy == "lexico":
        model.AddDecisionStrategy(x, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    elif strategy == "maxdeg":
        degs = [(len(adj[i]), i) for i in range(n_trains)]
        degs.sort(key=lambda p: p[0], reverse=True)
        ordered = [x[idx] for _, idx in degs]
        model.AddDecisionStrategy(ordered, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    solver = _configure_solver(strategy, tlimit)

    start = time.time()
    status = solver.Solve(model)
    end = time.time()

    return {
        "model": "decomposed",
        "k": k_quais,
        "strategy": strategy,
        "symmetry": symmetry,
        "status": solver.StatusName(status),
        "time": end - start,
        "conflicts": solver.NumConflicts(),
    }


def solve_global_nooverlap(
    trains: List[Train],
    k_quais: int,
    safety: int,
    strategy: str,
    symmetry: bool,
    tlimit: float,
) -> Dict:
    model = cp_model.CpModel()

    intervals_by_q = [[] for _ in range(k_quais)]
    assign = []

    for i, t in enumerate(trains):
        # safety incluse dans l'occupation
        dur = (t.dep - t.arr) + safety
        end = t.dep + safety

        row = []
        for q in range(k_quais):
            b = model.NewBoolVar(f"b[{i},{q}]")
            itv = model.NewOptionalIntervalVar(t.arr, dur, end, b, f"itv[{i},{q}]")
            intervals_by_q[q].append(itv)
            row.append(b)

        model.AddExactlyOne(row)
        assign.append(row)

    for q in range(k_quais):
        model.AddNoOverlap(intervals_by_q[q])

    if symmetry and k_quais > 0:
        model.Add(assign[0][0] == 1)

    solver = _configure_solver(strategy, tlimit)

    start = time.time()
    status = solver.Solve(model)
    end = time.time()

    return {
        "model": "global",
        "k": k_quais,
        "strategy": strategy,
        "symmetry": symmetry,
        "status": solver.StatusName(status),
        "time": end - start,
        "conflicts": solver.NumConflicts(),
    }


# ==========================================================
# 3) BENCHMARK
# ==========================================================

def run_benchmark():
    N_TRAINS = 130
    T_LIMIT = 5.0
    SAFETY = 2

    print(f"--- Instance (rush hour) : {N_TRAINS} trains ---")
    trains = generate_hard_instance(N_TRAINS, seed=42)
    adj = compute_conflicts(trains, safety=SAFETY)

    max_degree = max(len(v) for v in adj.values()) if adj else 0
    print(f"Max degree (densité locale) = {max_degree}")
    print("Les INFEASIBLE au début sont normales (K trop petit).")
    print("-" * 85)

    configs = [
        ("decomposed", "lexico"),
        ("decomposed", "maxdeg"),
        ("decomposed", "portfolio"),
        ("global", "lexico"),
        ("global", "fixed"),
        ("global", "portfolio"),
    ]

    start_k = max(1, int(max_degree * 0.6))
    end_k = max_degree + 8

    results = []
    solved_at = {(m, s): None for (m, s) in configs}

    print(f"{'Modèle':<10} | {'Stratégie':<10} | {'K':<3} | {'Statut':<10} | {'Temps(s)':<9} | {'Conflits':<9}")
    print("-" * 85)

    for k in range(start_k, end_k + 1):
        if all(v is not None and k > v + 2 for v in solved_at.values()):
            print("--> Tout le monde a convergé, fin.")
            break

        for model_name, strat in configs:
            if solved_at[(model_name, strat)] is not None and k > solved_at[(model_name, strat)] + 2:
                continue

            if model_name == "decomposed":
                res = solve_decomposed(
                    n_trains=N_TRAINS,
                    k_quais=k,
                    adj=adj,
                    strategy=strat,
                    symmetry=True,
                    tlimit=T_LIMIT,
                )

            else:
                # global : lexico/fixed => fixed search (sans DS imposée)
                s = strat if strat == "portfolio" else "fixed"
                res = solve_global_nooverlap(
                    trains=trains,
                    k_quais=k,
                    safety=SAFETY,
                    strategy=s,
                    symmetry=True,
                    tlimit=T_LIMIT,
                )
                res["strategy"] = strat  # garder le label demandé

            results.append(res)

            stat_disp = res["status"]
            if stat_disp == "UNKNOWN":
                stat_disp = "TIMEOUT"

            print(f"{res['model']:<10} | {res['strategy']:<10} | {k:<3} | {stat_disp:<10} | {res['time']:<9.4f} | {res['conflicts']:<9}")

            if res["status"] in ["OPTIMAL", "FEASIBLE"]:
                if solved_at[(model_name, strat)] is None:
                    solved_at[(model_name, strat)] = k

    df = pd.DataFrame(results)

    print("\n--- Résumé (K* observé = premier K satisfiable < timeout) ---")
    for (model_name, strat), kstar in solved_at.items():
        print(f"{model_name:10s} / {strat:10s} -> K* observé ~= {kstar}")

    return df, N_TRAINS, T_LIMIT


# ==========================================================
# 4) PLOTS : TIME + CONFLICTS
# ==========================================================

def plot_results(df: pd.DataFrame, n_trains: int, tlimit: float):

    if df.empty:
        print("Aucun résultat.")
        return

    df_plot = df.copy()

    # ===============================
    # FIGURE 1 : MODELE DECOMPOSE
    # ===============================
    fig1 = plt.figure(figsize=(11, 6))

    df_dec = df_plot[df_plot["model"] == "decomposed"]
    for strat in df_dec["strategy"].unique():
        data = df_dec[df_dec["strategy"] == strat].sort_values("k")
        plt.plot(data["k"], data["time"], marker="o", linewidth=2, label=strat)

    plt.title(f"Modèle Décomposé — Temps (N={n_trains})")
    plt.xlabel("Nombre de quais (K)")
    plt.ylabel("Temps (s)")
    plt.yscale("log")
    plt.axhline(y=tlimit, linestyle="--", alpha=0.6, label="Timeout")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1.savefig("decomposed_time.png", dpi=200)


    # ===============================
    # FIGURE 2 : MODELE GLOBAL
    # ===============================
    fig2 = plt.figure(figsize=(11, 6))

    df_glob = df_plot[df_plot["model"] == "global"]
    for strat in df_glob["strategy"].unique():
        data = df_glob[df_glob["strategy"] == strat].sort_values("k")
        plt.plot(data["k"], data["time"], marker="o", linewidth=2, label=strat)

    plt.title(f"Modèle Global — Temps (N={n_trains})")
    plt.xlabel("Nombre de quais (K)")
    plt.ylabel("Temps (s)")
    plt.yscale("log")
    plt.axhline(y=tlimit, linestyle="--", alpha=0.6, label="Timeout")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2.savefig("global_time.png", dpi=200)


    # ===============================
    # AFFICHAGE SIMULTANE
    # ===============================
    plt.show()

    print("Graphes sauvegardés :")
    print(" - decomposed_time.png")
    print(" - global_time.png")



# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    df, n, tlim = run_benchmark()

    # CSV : si Excel l'a ouvert, change de nom
    try:
        df.to_csv("results.csv", index=False)
        print("CSV sauvegardé : results.csv")
    except PermissionError:
        df.to_csv("results_out.csv", index=False)
        print("Permission refusée sur results.csv (souvent Excel ouvert).")
        print("CSV sauvegardé : results_out.csv")

    plot_results(df, n_trains=n, tlimit=tlim)
