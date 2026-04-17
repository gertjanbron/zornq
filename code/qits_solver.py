#\!/usr/bin/env python3
"""
qits_solver.py - B93 Quantum Imaginary-Time Solver (QITS) voor MaxCut

ITE-QAOA: vervang unitaire gates door imaginary-time evolution.
State-vector simulatie voor n <= 22, met tau-annealing en local search.
"""

import numpy as np
import time
import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqaoa import WeightedGraph


def _precompute_cuts_fast(n_nodes, edges):
    N = 1 << n_nodes
    cuts = np.zeros(N, dtype=np.float64)
    z = np.arange(N, dtype=np.int64)
    for i, j, w in edges:
        bi = (z >> i) & 1
        bj = (z >> j) & 1
        cuts += w * (bi ^ bj).astype(np.float64)
    return cuts


def _apply_cost_ite(psi, cuts, tau):
    psi *= np.exp(tau * cuts)
    psi /= np.linalg.norm(psi)
    return psi


def _apply_mixer_ite_fast(psi, n_nodes, tau):
    ch = np.cosh(tau)
    sh = np.sinh(tau)
    N = len(psi)
    for q in range(n_nodes):
        stride = 1 << q
        shape = (N // (2 * stride), 2, stride)
        p = psi.reshape(shape)
        p0 = p[:, 0, :].copy()
        p1 = p[:, 1, :].copy()
        p[:, 0, :] = ch * p0 - sh * p1
        p[:, 1, :] = -sh * p0 + ch * p1
    psi /= np.linalg.norm(psi)
    return psi


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


def qits_maxcut(n_nodes, edges, n_layers=10, tau_cost=0.3, tau_mix=0.1,
                anneal=True, n_restarts=5, local_search=True,
                n_samples=10, verbose=False):
    t_start = time.time()
    if n_nodes > 22:
        raise ValueError("QITS state-vector beperkt tot n<=22")
    if not edges:
        return {"best_cut": 0, "best_assignment": np.zeros(n_nodes, dtype=np.int32),
                "n_edges": 0, "ratio": 1.0, "solve_time": 0.0, "history": []}
    N = 1 << n_nodes
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float64)
    n_edges = len(edges)
    total_weight = float(np.sum(ew))
    cuts = _precompute_cuts_fast(n_nodes, edges)
    best_cut = -1.0
    best_assign = None
    history = []
    tau_schedules = []
    for r in range(n_restarts):
        scale = 0.5 + 1.5 * r / max(n_restarts - 1, 1)
        tau_schedules.append((tau_cost * scale, tau_mix * scale))
    for restart, (tc, tm) in enumerate(tau_schedules):
        psi = np.ones(N, dtype=np.float64) / np.sqrt(N)
        for layer in range(n_layers):
            if anneal:
                frac = (layer + 1) / n_layers
                tc_l = tc * frac
                tm_l = tm * (1.0 - 0.5 * frac)
            else:
                tc_l = tc
                tm_l = tm
            psi = _apply_cost_ite(psi, cuts, tc_l)
            psi = _apply_mixer_ite_fast(psi, n_nodes, tm_l)
        probs = psi ** 2
        top_indices = np.argsort(probs)[-n_samples:]
        for idx in top_indices:
            assign = np.array([(idx >> q) & 1 for q in range(n_nodes)], dtype=np.int32)
            cut = _eval_cut_array(assign, ei, ej, ew)
            if local_search:
                assign, cut, _ = _local_search(assign.copy(), ei, ej, ew, n_nodes)
            if cut > best_cut:
                best_cut = cut
                best_assign = assign.copy()
        history.append(float(best_cut))
        if verbose:
            ratio = best_cut / total_weight if total_weight > 0 else 0
            max_prob = np.max(probs)
            print("  QITS restart %d/%d (tc=%.3f tm=%.3f): cut=%.1f ratio=%.4f max_prob=%.4f" %
                  (restart+1, n_restarts, tc, tm, best_cut, ratio, max_prob))
    solve_time = time.time() - t_start
    ratio = best_cut / total_weight if total_weight > 0 else 1.0
    return {"best_cut": best_cut, "best_assignment": best_assign,
            "n_edges": n_edges, "ratio": ratio,
            "solve_time": solve_time, "history": history}


def qits_maxcut_grid(Lx, Ly, **kwargs):
    g = WeightedGraph.grid(Lx, Ly)
    edges = [(i, j, w) for i, j, w in g.edges()]
    return qits_maxcut(g.n_nodes, edges, **kwargs)


if __name__ == "__main__":
    print("=== QITS Quick Test ===")
    for Lx, Ly in [(4, 3), (6, 3)]:
        r = qits_maxcut_grid(Lx, Ly, n_restarts=5, n_layers=15, verbose=False)
        print("Grid %dx%d: cut=%d/%d  ratio=%.4f  [%.3fs]" %
              (Lx, Ly, int(r["best_cut"]), r["n_edges"], r["ratio"], r["solve_time"]))
    print("Done.")
