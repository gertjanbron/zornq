#!/usr/bin/env python3
"""
B159: ILP-Oracle voor MaxCut — Certifieerbaar Exacte Ceiling.

Doel: certificeer OPT voor grafen waar BKS (Best Known Solution) onbekend is,
en geef een harde ceiling-kolom voor de paper-tabellen. Waar brute-force stopt
bij n~20 en GW/SoS-2 bovengrenzen leveren, levert de ILP-oracle een harde
OPT-waarde met certificaat tot n~80-120 (afhankelijk van dichtheid/time-budget).

Standaard 0/1-formulering (signed-safe -- forceert y_uv = |x_u - x_v|):
    max  sum_{(u,v) in E} w_uv * y_uv
    s.t. y_uv <= x_u + x_v              voor elke edge (u,v)   (UB1)
         y_uv <= 2 - x_u - x_v          voor elke edge (u,v)   (UB2)
         y_uv >= x_u - x_v              voor elke edge (u,v)   (LB1)
         y_uv >= x_v - x_u              voor elke edge (u,v)   (LB2)
         x_v, y_uv in {0,1}

UB1+UB2 dwingen y_uv = 0 als x_u = x_v (edge niet gesneden); LB1+LB2 dwingen
y_uv = 1 als x_u != x_v (edge wel gesneden). Zonder LB-paar zou de solver bij
*negatieve* w_uv de y_uv vrij op 0 zetten (max-positive-cut i.p.v. signed
MaxCut). We breken de reflectie-symmetrie door x_0 = 0 te fixeren.

Solvers:
1. HiGHS (via `scipy.optimize.milp`) -- default, altijd beschikbaar.
2. SCIP (via `pyscipopt`)            -- optioneel, vaak sneller op hard problems.
3. Gurobi (via `gurobipy`)           -- optioneel, snelst maar commercieel.

Retourneert `{"opt_value", "cut_bits", "certified", "gap_abs", "wall_time", ...}`.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    random_3regular,
    random_erdos_renyi,
)


# ============================================================
# HiGHS via scipy.optimize.milp
# ============================================================

def maxcut_ilp_highs(
    graph: SimpleGraph,
    time_limit: float | None = None,
    break_symmetry: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Los MaxCut op via HiGHS MILP (signed-safe)."""
    from scipy.optimize import milp, LinearConstraint, Bounds

    n = graph.n
    edges = list(graph.edges)
    m = len(edges)

    # Vars: x_0..x_{n-1}, y_e0..y_e{m-1}   (alle binair)
    N = n + m
    c = np.zeros(N)
    for e_idx, (_u, _v, w) in enumerate(edges):
        c[n + e_idx] = -float(w)      # minimize -sum w y

    # 4m rijen (signed-safe, forceert y_uv = |x_u - x_v|):
    #   UB1:  y - x_u - x_v <= 0
    #   UB2:  y + x_u + x_v <= 2
    #   LB1:  x_u - x_v - y <= 0      (y >= x_u - x_v)
    #   LB2:  x_v - x_u - y <= 0      (y >= x_v - x_u)
    rows: list[np.ndarray] = []
    rhs_ub: list[float] = []
    for e_idx, (u, v, _w) in enumerate(edges):
        row1 = np.zeros(N)
        row1[u] = -1.0
        row1[v] = -1.0
        row1[n + e_idx] = 1.0
        rows.append(row1)
        rhs_ub.append(0.0)

        row2 = np.zeros(N)
        row2[u] = 1.0
        row2[v] = 1.0
        row2[n + e_idx] = 1.0
        rows.append(row2)
        rhs_ub.append(2.0)

        # LB1: x_u - x_v - y_e <= 0   (forceert y_e >= x_u - x_v)
        row3 = np.zeros(N)
        row3[u] = 1.0
        row3[v] = -1.0
        row3[n + e_idx] = -1.0
        rows.append(row3)
        rhs_ub.append(0.0)

        # LB2: x_v - x_u - y_e <= 0   (forceert y_e >= x_v - x_u)
        row4 = np.zeros(N)
        row4[u] = -1.0
        row4[v] = 1.0
        row4[n + e_idx] = -1.0
        rows.append(row4)
        rhs_ub.append(0.0)

    A = np.array(rows) if rows else np.zeros((0, N))
    lb_ub = np.full(len(rhs_ub), -np.inf)
    rhs_ub_arr = np.asarray(rhs_ub)
    constr = LinearConstraint(A, lb_ub, rhs_ub_arr) if rows else None

    lb_x = np.zeros(N)
    ub_x = np.ones(N)
    if break_symmetry and n > 0:
        ub_x[0] = 0.0  # fix x_0 = 0
    bounds = Bounds(lb=lb_x, ub=ub_x)
    integrality = np.ones(N)

    options: dict[str, Any] = {"disp": verbose}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    t0 = time.time()
    constraints = constr if constr is not None else ()
    res = milp(c, constraints=constraints, bounds=bounds,
                integrality=integrality, options=options)
    wall = time.time() - t0

    certified = bool(res.success) and getattr(res, "status", -1) == 0
    if res.x is not None:
        x_vals = res.x[:n]
        cut_bits = "".join("1" if xi > 0.5 else "0" for xi in x_vals)
        if certified:
            opt_value = float(-res.fun)
        else:
            opt_value = 0.0
            for u, v, w in edges:
                if cut_bits[u] != cut_bits[v]:
                    opt_value += float(w)
    else:
        opt_value = None
        cut_bits = None

    gap_abs = 0.0 if certified else None

    return {
        "opt_value": opt_value,
        "cut_bits": cut_bits,
        "certified": certified,
        "gap_abs": gap_abs,
        "wall_time": wall,
        "n_vars": N,
        "n_constrs": len(rhs_ub),
        "solver": "HiGHS",
        "status": getattr(res, "message", "unknown"),
    }


# ============================================================
# SCIP via pyscipopt (optioneel)
# ============================================================

def maxcut_ilp_scip(
    graph: SimpleGraph,
    time_limit: float | None = None,
    break_symmetry: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """SCIP-variant. Vereist `pip install pyscipopt`."""
    try:
        from pyscipopt import Model
    except ImportError:
        return {
            "opt_value": None,
            "certified": False,
            "solver": "SCIP",
            "status": "SKIPPED_PYSCIPOPT_NOT_INSTALLED",
            "wall_time": 0.0,
        }

    n = graph.n
    edges = list(graph.edges)
    m = len(edges)

    model = Model("MaxCut_ILP")
    if not verbose:
        model.hideOutput()
    if time_limit is not None:
        model.setParam("limits/time", float(time_limit))

    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(n)]
    y = [model.addVar(vtype="B", name=f"y_{i}") for i in range(m)]

    if break_symmetry and n > 0:
        model.addCons(x[0] == 0)

    for e_idx, (u, v, _w) in enumerate(edges):
        model.addCons(y[e_idx] <= x[u] + x[v])
        model.addCons(y[e_idx] <= 2 - x[u] - x[v])
        model.addCons(y[e_idx] >= x[u] - x[v])
        model.addCons(y[e_idx] >= x[v] - x[u])

    obj = 0
    for e_idx, (_u, _v, w) in enumerate(edges):
        obj += float(w) * y[e_idx]
    model.setObjective(obj, sense="maximize")

    t0 = time.time()
    model.optimize()
    wall = time.time() - t0

    status = model.getStatus()
    certified = status == "optimal"

    if model.getNSols() > 0:
        opt_value = float(model.getObjVal())
        sol = model.getBestSol()
        cut_bits = "".join("1" if model.getSolVal(sol, xi) > 0.5 else "0"
                            for xi in x)
    else:
        opt_value = None
        cut_bits = None

    try:
        gap_abs = float(model.getGap()) * abs(opt_value) if opt_value else None
    except Exception:
        gap_abs = None

    return {
        "opt_value": opt_value,
        "cut_bits": cut_bits,
        "certified": certified,
        "gap_abs": 0.0 if certified else gap_abs,
        "wall_time": wall,
        "n_vars": n + m,
        "n_constrs": 4 * m + (1 if break_symmetry else 0),
        "solver": "SCIP",
        "status": status,
    }


# ============================================================
# Gurobi via gurobipy (optioneel)
# ============================================================

def maxcut_ilp_gurobi(
    graph: SimpleGraph,
    time_limit: float | None = None,
    break_symmetry: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Gurobi-variant. Vereist `pip install gurobipy` + licentie."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return {
            "opt_value": None,
            "certified": False,
            "solver": "Gurobi",
            "status": "SKIPPED_GUROBIPY_NOT_INSTALLED",
            "wall_time": 0.0,
        }

    n = graph.n
    edges = list(graph.edges)
    m_e = len(edges)

    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("MaxCut_ILP", env=env)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(m_e, vtype=GRB.BINARY, name="y")

    if break_symmetry and n > 0:
        model.addConstr(x[0] == 0)
    for e_idx, (u, v, _w) in enumerate(edges):
        model.addConstr(y[e_idx] <= x[u] + x[v])
        model.addConstr(y[e_idx] <= 2 - x[u] - x[v])
        model.addConstr(y[e_idx] >= x[u] - x[v])
        model.addConstr(y[e_idx] >= x[v] - x[u])

    model.setObjective(
        gp.quicksum(float(w) * y[e_idx]
                     for e_idx, (_u, _v, w) in enumerate(edges)),
        GRB.MAXIMIZE,
    )

    t0 = time.time()
    model.optimize()
    wall = time.time() - t0

    certified = model.status == GRB.OPTIMAL
    if model.SolCount > 0:
        opt_value = float(model.ObjVal)
        cut_bits = "".join("1" if x[i].X > 0.5 else "0" for i in range(n))
    else:
        opt_value = None
        cut_bits = None
    try:
        gap_abs = float(model.MIPGap) * abs(opt_value) if opt_value else None
    except Exception:
        gap_abs = None

    return {
        "opt_value": opt_value,
        "cut_bits": cut_bits,
        "certified": certified,
        "gap_abs": 0.0 if certified else gap_abs,
        "wall_time": wall,
        "n_vars": n + m_e,
        "n_constrs": 4 * m_e + (1 if break_symmetry else 0),
        "solver": "Gurobi",
        "status": int(model.status),
    }


# ============================================================
# Unified dispatcher
# ============================================================

SOLVERS = {
    "highs": maxcut_ilp_highs,
    "scip": maxcut_ilp_scip,
    "gurobi": maxcut_ilp_gurobi,
}


def maxcut_ilp(
    graph: SimpleGraph,
    solver: str = "highs",
    time_limit: float | None = None,
    break_symmetry: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Unified entry point. solver in {highs,scip,gurobi}."""
    fn = SOLVERS.get(solver.lower())
    if fn is None:
        raise ValueError(
            f"Onbekende solver: {solver!r}. Kies uit {list(SOLVERS)}.")
    return fn(graph, time_limit=time_limit,
               break_symmetry=break_symmetry, verbose=verbose)


# ============================================================
# Helper: verify cut-value on SimpleGraph
# ============================================================

def evaluate_cut(graph: SimpleGraph, bits: str) -> float:
    """Bereken cut-gewicht voor een gegeven 0/1-partitie."""
    total = 0.0
    for u, v, w in graph.edges:
        if bits[u] != bits[v]:
            total += float(w)
    return total


# ============================================================
# CLI
# ============================================================

def _build_graph(args: argparse.Namespace) -> tuple[SimpleGraph, str]:
    if args.petersen:
        from b156_sos2_sdp import petersen_graph
        return petersen_graph(), "Petersen"
    if args.cycle is not None:
        from b156_sos2_sdp import cycle_graph
        return cycle_graph(args.cycle), f"C_{args.cycle}"
    if args.kn is not None:
        from b156_sos2_sdp import complete_graph
        return complete_graph(args.kn), f"K_{args.kn}"
    if args.random is not None:
        return random_3regular(args.random, seed=args.seed),                f"3reg_n{args.random}"
    if args.erdos is not None:
        return random_erdos_renyi(args.erdos, p=args.p, seed=args.seed),                f"ER_n{args.erdos}_p{args.p}"
    from b156_sos2_sdp import complete_graph
    return complete_graph(4), "K_4"


def main() -> None:
    parser = argparse.ArgumentParser(description="B159 ILP-Oracle voor MaxCut")
    parser.add_argument("--kn", type=int)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--petersen", action="store_true")
    parser.add_argument("--random", type=int)
    parser.add_argument("--erdos", type=int)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", default="highs",
                        choices=list(SOLVERS))
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    graph, name = _build_graph(args)
    print(f"Graaf: {name}  (n={graph.n}, m={graph.n_edges})")
    res = maxcut_ilp(graph, solver=args.solver,
                      time_limit=args.time_limit,
                      verbose=args.verbose)

    print(f"  Solver:      {res['solver']}")
    print(f"  Status:      {res['status']}")
    print(f"  OPT:         {res['opt_value']}")
    print(f"  Certified:   {res['certified']}")
    print(f"  Gap (abs):   {res.get('gap_abs')}")
    print(f"  Cut:         {res.get('cut_bits')}")
    print(f"  n_vars:      {res['n_vars']}, n_constrs: {res['n_constrs']}")
    print(f"  Wall-time:   {res['wall_time']:.3f}s")

    if res["opt_value"] is not None and graph.n <= 18:
        bf = brute_force_maxcut(graph)
        print(f"  Brute force: {bf}  (match: {abs(res['opt_value'] - bf) < 1e-6})")
    if res["cut_bits"] is not None:
        check = evaluate_cut(graph, res["cut_bits"])
        print(f"  Recompute:   {check}")


if __name__ == "__main__":
    main()
