#\!/usr/bin/env python3
"""
qw_solver.py - B82 QW-QAOA: Szegedy Quantum Walk Mixer voor MaxCut

Graaf-bewuste mixer via XY-swap Hamiltoniaan H_mix = sum_{ij} (XiXj+YiYj)/2.
Amplitude wandelt langs edges — discrete quantum walk op de graaf.
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
        cuts += w * (((z >> i) & 1) ^ ((z >> j) & 1)).astype(np.float64)
    return cuts


def _apply_cost_ite(psi, cuts, tau):
    psi *= np.exp(tau * cuts)
    psi /= np.linalg.norm(psi)
    return psi


def _apply_cost_unitary(psi, cuts, gamma):
    psi *= np.exp(-1j * gamma * cuts)
    return psi


def _apply_xy_gate_vec(psi, qi, qj, param, is_ite=True):
    """Vectorized XY-gate op qubits qi, qj.
    Raakt alleen |01> <-> |10> subspace (bi!=bj)."""
    N = len(psi)
    si = 1 << qi
    sj = 1 << qj
    z = np.arange(N, dtype=np.int64)
    bi = (z >> qi) & 1
    bj = (z >> qj) & 1
    # Selecteer indices waar bi=0, bj=1
    mask01 = (bi == 0) & (bj == 1)
    idx01 = z[mask01]
    idx10 = idx01 ^ si ^ sj  # flip both bits
    a01 = psi[idx01].copy()
    a10 = psi[idx10].copy()
    if is_ite:
        ch = np.cosh(param)
        sh = np.sinh(param)
        psi[idx01] = ch * a01 - sh * a10
        psi[idx10] = -sh * a01 + ch * a10
    else:
        cb = np.cos(param)
        sb = np.sin(param)
        psi[idx01] = cb * a01 - 1j * sb * a10
        psi[idx10] = -1j * sb * a01 + cb * a10
    return psi


def _apply_qw_mixer(psi, n_nodes, edges, param, is_ite=True):
    """Trotterized QW mixer: product van XY-gates langs alle edges."""
    for i, j, w in edges:
        psi = _apply_xy_gate_vec(psi, i, j, param * w, is_ite)
    if is_ite:
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


def qw_maxcut(n_nodes, edges, n_layers=10, mode="ite",
              tau_cost=0.3, tau_mix=0.15, anneal=True,
              n_restarts=5, local_search=True, n_samples=10,
              verbose=False):
    t_start = time.time()
    if n_nodes > 20:
        raise ValueError("QW state-vector beperkt tot n<=20")
    if not edges:
        return {"best_cut": 0, "best_assignment": np.zeros(n_nodes, dtype=np.int32),
                "n_edges": 0, "ratio": 1.0, "solve_time": 0.0, "history": []}
    N = 1 << n_nodes
    ei_arr = np.array([e[0] for e in edges], dtype=np.int32)
    ej_arr = np.array([e[1] for e in edges], dtype=np.int32)
    ew_arr = np.array([e[2] for e in edges], dtype=np.float64)
    n_edges = len(edges)
    total_weight = float(np.sum(ew_arr))
    cuts = _precompute_cuts_fast(n_nodes, edges)
    is_ite = (mode == "ite")
    use_complex = not is_ite
    best_cut = -1.0
    best_assign = None
    history = []
    tau_schedules = []
    for r in range(n_restarts):
        scale = 0.5 + 1.5 * r / max(n_restarts - 1, 1)
        tau_schedules.append((tau_cost * scale, tau_mix * scale))
    for restart, (tc, tm) in enumerate(tau_schedules):
        if use_complex:
            psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        else:
            psi = np.ones(N, dtype=np.float64) / np.sqrt(N)
        for layer in range(n_layers):
            if anneal:
                frac = (layer + 1) / n_layers
                tc_l = tc * frac
                tm_l = tm * (1.0 - 0.5 * frac)
            else:
                tc_l = tc
                tm_l = tm
            if is_ite:
                psi = _apply_cost_ite(psi, cuts, tc_l)
            else:
                psi = _apply_cost_unitary(psi, cuts, tc_l)
            psi = _apply_qw_mixer(psi, n_nodes, edges, tm_l, is_ite)
        probs = np.abs(psi) ** 2
        top_indices = np.argsort(probs)[-n_samples:]
        for idx in top_indices:
            assign = np.array([(idx >> q) & 1 for q in range(n_nodes)], dtype=np.int32)
            cut = _eval_cut_array(assign, ei_arr, ej_arr, ew_arr)
            if local_search:
                assign, cut, _ = _local_search(assign.copy(), ei_arr, ej_arr, ew_arr, n_nodes)
            if cut > best_cut:
                best_cut = cut
                best_assign = assign.copy()
        history.append(float(best_cut))
        if verbose:
            ratio = best_cut / total_weight if total_weight > 0 else 0
            print("  QW restart %d/%d (tc=%.3f tm=%.3f): cut=%.1f ratio=%.4f" %
                  (restart+1, n_restarts, tc, tm, best_cut, ratio))
    solve_time = time.time() - t_start
    ratio = best_cut / total_weight if total_weight > 0 else 1.0
    return {"best_cut": best_cut, "best_assignment": best_assign,
            "n_edges": n_edges, "ratio": ratio,
            "solve_time": solve_time, "history": history}


def qw_maxcut_grid(Lx, Ly, **kwargs):
    g = WeightedGraph.grid(Lx, Ly)
    edges = [(i, j, w) for i, j, w in g.edges()]
    return qw_maxcut(g.n_nodes, edges, **kwargs)


if __name__ == "__main__":
    print("=== QW-QAOA Quick Test ===")
    for Lx, Ly in [(4, 3), (5, 3)]:
        for mode in ["ite", "unitary"]:
            r = qw_maxcut_grid(Lx, Ly, mode=mode, n_restarts=3, n_layers=12, verbose=False)
            print("%dx%d (%s): cut=%d/%d  ratio=%.4f  [%.3fs]" %
                  (Lx, Ly, mode, int(r["best_cut"]), r["n_edges"], r["ratio"], r["solve_time"]))
    print("Done.")
