#\!/usr/bin/env python3
"""sb_solver.py - B95 Simulated Bifurcation (SBA) MaxCut Solver"""

import numpy as np
import time
import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqaoa import WeightedGraph


def _build_J_matrix(n_nodes, edges):
    J = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j, w in edges:
        J[i, j] -= w
        J[j, i] -= w
    return J


def _sb_step(x, y, J, p, dt, c):
    Jx = J @ x
    y_new = y + dt * ((p - 1.0) * x - x**3 + c * Jx)
    x_new = x + dt * y_new
    return x_new, y_new


def _sb_step_discrete(x, y, J, p, dt, c):
    Jx = J @ x
    y_new = y + dt * ((p - 1.0) * x - x**3 + c * Jx)
    x_new = x + dt * y_new
    x_new = np.clip(x_new, -1.0, 1.0)
    return x_new, y_new


def _eval_cut_array(assign, ei, ej, ew):
    return float(np.sum(ew[assign[ei] != assign[ej]]))


def _local_search(assign, ei, ej, ew, n_nodes, max_passes=50):
    best_cut = _eval_cut_array(assign, ei, ej, ew)
    improved = True
    total_flips = 0
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for node in range(n_nodes):
            assign[node] = 1 - assign[node]
            new_cut = _eval_cut_array(assign, ei, ej, ew)
            if new_cut > best_cut:
                best_cut = new_cut
                improved = True
                total_flips += 1
            else:
                assign[node] = 1 - assign[node]
    return assign, best_cut, total_flips


def sb_maxcut(n_nodes, edges, n_restarts=30, max_iter=2000,
              dt=0.05, p_max=3.0, c=None, variant="ballistic",
              local_search=True, verbose=False):
    t_start = time.time()
    if not edges:
        return {"best_cut": 0, "best_assignment": np.zeros(n_nodes, dtype=np.int32),
                "n_edges": 0, "ratio": 1.0, "solve_time": 0.0, "history": []}
    J = _build_J_matrix(n_nodes, edges)
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float64)
    n_edges = len(edges)
    total_weight = float(np.sum(ew))
    if c is None:
        max_row_sum = np.max(np.sum(np.abs(J), axis=1))
        c = 1.0 / max(max_row_sum, 1e-10)
    step_fn = _sb_step if variant == "ballistic" else _sb_step_discrete
    best_cut = -1.0
    best_assign = None
    history = []
    for restart in range(n_restarts):
        x = np.random.randn(n_nodes) * 0.01
        y = np.zeros(n_nodes)
        for step in range(max_iter):
            p = p_max * step / max_iter
            x, y = step_fn(x, y, J, p, dt, c)
        assign = (np.sign(x) < 0).astype(np.int32)
        cut = _eval_cut_array(assign, ei, ej, ew)
        if local_search:
            assign, cut, nf = _local_search(assign, ei, ej, ew, n_nodes)
        if cut > best_cut:
            best_cut = cut
            best_assign = assign.copy()
        history.append(float(cut))
        if verbose:
            ratio = cut / total_weight if total_weight > 0 else 0
            print("  SB restart %d/%d: cut=%.1f ratio=%.4f" % (restart+1, n_restarts, cut, ratio))
    solve_time = time.time() - t_start
    ratio = best_cut / total_weight if total_weight > 0 else 1.0
    return {"best_cut": best_cut, "best_assignment": best_assign,
            "n_edges": n_edges, "ratio": ratio,
            "solve_time": solve_time, "history": history}


def sb_maxcut_grid(Lx, Ly, **kwargs):
    g = WeightedGraph.grid(Lx, Ly)
    edges = [(i, j, w) for i, j, w in g.edges()]
    return sb_maxcut(g.n_nodes, edges, **kwargs)


if __name__ == "__main__":
    print("=== Simulated Bifurcation Quick Test ===")
    for Lx, Ly in [(4, 3), (6, 3), (10, 3)]:
        r = sb_maxcut_grid(Lx, Ly, n_restarts=20, verbose=False)
        print("Grid %dx%d: cut=%d/%d  ratio=%.4f  [%.3fs]" % (Lx, Ly, int(r["best_cut"]), r["n_edges"], r["ratio"], r["solve_time"]))
    print("Done.")
