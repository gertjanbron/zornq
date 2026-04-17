#!/usr/bin/env python3
"""
B156: Lasserre / Sum-of-Squares level-2 SDP Bound voor MaxCut.

Level-1 van de Lasserre-hiërarchie komt overeen met de Goemans-Williamson
SDP (zie B60). Level-2 is strikt strenger en gebruikt pseudo-momenten van
graad tot 4. Voor elke S ⊆ [n] met |S| ≤ 4 krijgen we één variabele y_S;
de moment-matrix M_2 geïndexeerd door monomialen van graad ≤ 2 moet PSD
zijn, waarbij [M_2]_{S,T} = y_{S △ T} (omdat x_i² = 1 in {-1,+1}-MaxCut).

Formulering:
  max  (1/2) Σ_{(u,v) ∈ E} w_uv (1 − y_{{u,v}})
  s.t. y_∅ = 1
       M_2[y] ⪰ 0

De formulering levert automatisch triangle-inequalities en hogere-orde
facet-ongelijkheden op, wat SoS-2 ≤ SoS-1 = GW garandeert (bij gelijk
optimum alleen als GW zelf al scherp was).

Schaalbaarheid: aantal pseudo-momenten is O(n^4), moment-matrix is
(n+1)(n+2)/2 × (n+1)(n+2)/2 = O(n^2) × O(n^2). Praktisch tot n ≈ 25-30
met SCS op CPU; daarboven wordt het geheugen- of tijdkritisch.

Gebruik:
  python b156_sos2_sdp.py --n 6            # K_6
  python b156_sos2_sdp.py --cycle 8        # cykel C_8
  python b156_sos2_sdp.py --petersen       # Petersen graaf
  python b156_sos2_sdp.py --random 10      # random 3-regular n=10
  python b156_sos2_sdp.py --compare        # SoS-1 vs SoS-2 vergelijking
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import combinations
from typing import Any

import numpy as np
import cvxpy as cp

# Hergebruik SimpleGraph en helpers uit B60
from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    gw_sdp_bound,
    random_3regular,
    random_erdos_renyi,
)


# ============================================================
# Graaf-helpers
# ============================================================

def complete_graph(n: int) -> SimpleGraph:
    g = SimpleGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def cycle_graph(n: int) -> SimpleGraph:
    g = SimpleGraph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def path_graph(n: int) -> SimpleGraph:
    g = SimpleGraph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def petersen_graph() -> SimpleGraph:
    g = SimpleGraph(10)
    # Buiten-cykel 0-1-2-3-4-0
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    # Binnen-ster 5-7-9-6-8-5 (pentagram)
    for i in range(5):
        g.add_edge(5 + i, 5 + (i + 2) % 5)
    # Spaken
    for i in range(5):
        g.add_edge(i, 5 + i)
    return g


def complete_bipartite(a: int, b: int) -> SimpleGraph:
    g = SimpleGraph(a + b)
    for i in range(a):
        for j in range(b):
            g.add_edge(i, a + j)
    return g


# ============================================================
# SoS Level-2 SDP
# ============================================================

def _basis_monomials(n: int) -> list[frozenset[int]]:
    """Monomialen van graad ≤ 2 over x_0, ..., x_{n-1}.

    Representatie: frozenset van indices (we werken in {-1,+1} dus x_i² = 1).
    Aantal: 1 + n + C(n, 2) = (n+1)(n+2)/2.
    """
    mons: list[frozenset[int]] = [frozenset()]
    mons.extend(frozenset({i}) for i in range(n))
    mons.extend(frozenset({i, j}) for i in range(n) for j in range(i + 1, n))
    return mons


def _pseudomoment_keys(basis_mons: list[frozenset[int]]) -> list[frozenset[int]]:
    """Alle unieke sleutels y_S die in M_2 voorkomen.

    [M_2]_{S,T} = y_{S △ T}, dus we verzamelen alle symmetric differences.
    Dit geeft S ⊆ [n] met |S| ≤ 4.
    """
    keys: set[frozenset[int]] = set()
    for S in basis_mons:
        for T in basis_mons:
            keys.add(frozenset(S ^ T))  # XOR op frozensets = symm. diff.
    return sorted(keys, key=lambda s: (len(s), sorted(s)))


def sos2_sdp_bound(
    graph: SimpleGraph,
    verbose: bool = True,
    max_n: int = 30,
    solver: str = "SCS",
    eps: float = 1e-6,
    max_iters: int = 20000,
) -> dict[str, Any]:
    """Bereken Lasserre level-2 SDP bovengrens voor MaxCut(graph).

    Parameters
    ----------
    graph : SimpleGraph
        Ongewogen of gewogen graaf.
    verbose : bool
        Print samenvatting.
    max_n : int
        Weiger te solven als n > max_n (bescherming tegen geheugenexplosie).
    solver : str
        'SCS' of 'CLARABEL' (open-source) of 'MOSEK' (licensie).
    eps, max_iters : float, int
        SCS/CLARABEL tolerantie en max iteraties.

    Returns
    -------
    dict met keys:
        sos2_bound         : float, level-2 bound (None bij falen)
        moment_matrix_size : int, N = (n+1)(n+2)/2
        num_pseudo_moments : int, aantal y_S variabelen
        total_edge_weight  : float, Σ w
        status             : cvxpy solver status
        solve_time         : float, wall time in seconden
        n, n_edges         : graaf-maten
    """
    n = graph.n
    t0 = time.time()

    if n > max_n:
        return {
            "sos2_bound": None,
            "status": f"SKIPPED: n={n} > max_n={max_n}",
            "solve_time": 0.0,
            "n": n,
            "n_edges": graph.n_edges,
        }

    # Basis-monomialen en pseudo-moment-sleutels
    basis_mons = _basis_monomials(n)
    N = len(basis_mons)
    var_sets = _pseudomoment_keys(basis_mons)
    var_idx = {s: i for i, s in enumerate(var_sets)}
    num_vars = len(var_sets)

    # Variabele y ∈ R^num_vars
    y = cp.Variable(num_vars)

    # Bouw moment-matrix M via 2D fancy-indexing op y
    idx_mat = np.zeros((N, N), dtype=np.int64)
    for i, S in enumerate(basis_mons):
        for j, T in enumerate(basis_mons):
            idx_mat[i, j] = var_idx[frozenset(S ^ T)]

    # cvxpy: reshape(y[flat_idx], (N, N)) werkt voor scalar expressie-matrix
    M = cp.reshape(y[idx_mat.flatten()], (N, N), order="C")

    # Constraints
    constraints = [
        M >> 0,                                 # M_2[y] PSD
        y[var_idx[frozenset()]] == 1,           # y_∅ = 1
    ]

    # Objective: (1/2) Σ_{(u,v) ∈ E} w_uv (1 − y_{{u,v}})
    total_weight = graph.total_weight()
    obj_expr: Any = 0
    for u, v, w in graph.edges:
        pair_key = frozenset({u, v})
        obj_expr = obj_expr + w * (1 - y[var_idx[pair_key]]) / 2

    prob = cp.Problem(cp.Maximize(obj_expr), constraints)

    try:
        if solver.upper() == "SCS":
            prob.solve(solver=cp.SCS, verbose=False,
                       max_iters=max_iters, eps=eps)
        elif solver.upper() == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=False)
        elif solver.upper() == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(verbose=False)
    except Exception as e:
        return {
            "sos2_bound": None,
            "status": f"FAILED: {type(e).__name__}: {e}",
            "solve_time": time.time() - t0,
            "moment_matrix_size": N,
            "num_pseudo_moments": num_vars,
            "n": n,
            "n_edges": graph.n_edges,
        }

    solve_time = time.time() - t0

    if prob.value is None:
        return {
            "sos2_bound": None,
            "status": prob.status,
            "solve_time": solve_time,
            "moment_matrix_size": N,
            "num_pseudo_moments": num_vars,
            "n": n,
            "n_edges": graph.n_edges,
        }

    sos2 = float(prob.value)

    result = {
        "sos2_bound": sos2,
        "sos2_ratio": sos2 / graph.n_edges if graph.n_edges > 0 else 0.0,
        "moment_matrix_size": N,
        "num_pseudo_moments": num_vars,
        "total_edge_weight": total_weight,
        "status": prob.status,
        "solve_time": solve_time,
        "n": n,
        "n_edges": graph.n_edges,
        "solver": solver,
    }

    if verbose:
        print("  SoS-2 (Lasserre level-2) bound:")
        print(f"    Moment-matrix: {N} x {N}  (monomialen van graad ≤ 2)")
        print(f"    Pseudo-momenten: {num_vars} (|S| ≤ 4)")
        print(f"    SoS-2 bovengrens: {sos2:.6f} / {graph.n_edges} edges "
              f"(ratio {result['sos2_ratio']:.6f})")
        print(f"    Solver: {solver}, status={prob.status}, "
              f"tijd={solve_time:.2f}s")

    return result


# ============================================================
# Vergelijking SoS-1 (GW) vs SoS-2
# ============================================================

def compare_bounds(
    graph: SimpleGraph,
    verbose: bool = True,
    name: str = "graph",
) -> dict[str, Any]:
    """Draai GW en SoS-2 op dezelfde graaf en rapporteer gap + tightening."""
    opt = brute_force_maxcut(graph) if graph.n <= 20 else None

    t0 = time.time()
    gw = gw_sdp_bound(graph, verbose=False)
    gw_time = time.time() - t0

    sos2 = sos2_sdp_bound(graph, verbose=False)

    result: dict[str, Any] = {
        "name": name,
        "n": graph.n,
        "n_edges": graph.n_edges,
        "opt": opt,
        "gw": gw.get("sdp_bound"),
        "gw_time": gw_time,
        "sos2": sos2.get("sos2_bound"),
        "sos2_time": sos2.get("solve_time"),
        "moment_matrix_size": sos2.get("moment_matrix_size"),
        "num_pseudo_moments": sos2.get("num_pseudo_moments"),
    }

    if result["gw"] is not None and result["sos2"] is not None:
        tightening = (result["gw"] - result["sos2"]) / result["gw"]
        result["tightening_pct"] = tightening * 100.0
        if opt is not None:
            result["gw_gap_pct"] = (result["gw"] - opt) / opt * 100.0
            result["sos2_gap_pct"] = (result["sos2"] - opt) / opt * 100.0

    if verbose:
        print(f"\n=== {name} (n={graph.n}, m={graph.n_edges}) ===")
        if opt is not None:
            print(f"  OPT (brute-force): {opt}")
        if result["gw"] is not None:
            print(f"  GW (SoS-1):        {result['gw']:.6f}  ({gw_time:.2f}s)")
        if result["sos2"] is not None:
            print(f"  SoS-2:             {result['sos2']:.6f}  "
                  f"({result['sos2_time']:.2f}s)")
        if "tightening_pct" in result:
            print(f"  SoS-2 is {result['tightening_pct']:.3f}% strakker dan GW")
        if "gw_gap_pct" in result and "sos2_gap_pct" in result:
            print(f"  Gap tot OPT:  GW={result['gw_gap_pct']:.3f}%, "
                  f"SoS-2={result['sos2_gap_pct']:.3f}%")

    return result


# ============================================================
# CLI
# ============================================================

def _build_graph(args: argparse.Namespace) -> tuple[SimpleGraph, str]:
    if args.n is not None:
        return complete_graph(args.n), f"K_{args.n}"
    if args.cycle is not None:
        return cycle_graph(args.cycle), f"C_{args.cycle}"
    if args.path is not None:
        return path_graph(args.path), f"P_{args.path}"
    if args.petersen:
        return petersen_graph(), "Petersen"
    if args.bipartite is not None:
        a, b = args.bipartite
        return complete_bipartite(a, b), f"K_{a},{b}"
    if args.random is not None:
        return random_3regular(args.random, seed=args.seed), f"3reg_n{args.random}"
    if args.erdos is not None:
        n = args.erdos
        return (random_erdos_renyi(n, p=args.p, seed=args.seed),
                f"ER_n{n}_p{args.p}")
    # Default: K_5
    return complete_graph(5), "K_5"


def main() -> None:
    parser = argparse.ArgumentParser(description="B156 Lasserre SoS level-2 SDP")
    parser.add_argument("--n", type=int, help="Complete graph K_n")
    parser.add_argument("--cycle", type=int, help="Cycle C_n")
    parser.add_argument("--path", type=int, help="Path P_n")
    parser.add_argument("--petersen", action="store_true")
    parser.add_argument("--bipartite", type=int, nargs=2,
                        metavar=("A", "B"), help="K_{a,b} complete bipartite")
    parser.add_argument("--random", type=int, help="random 3-regulier n")
    parser.add_argument("--erdos", type=int, help="Erdős-Rényi n")
    parser.add_argument("--p", type=float, default=0.3,
                        help="ER edge probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true",
                        help="Vergelijk GW (SoS-1) en SoS-2")
    parser.add_argument("--solver", default="SCS",
                        choices=["SCS", "CLARABEL", "MOSEK"])
    args = parser.parse_args()

    graph, name = _build_graph(args)

    if args.compare:
        compare_bounds(graph, name=name)
    else:
        sos2_sdp_bound(graph, solver=args.solver)


if __name__ == "__main__":
    main()
