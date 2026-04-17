#!/usr/bin/env python3
"""
kuramoto_solver.py - B92: Anti-Kuramoto MaxCut Solver.

Negatieve Kuramoto-koppeling op de graaf:
  d theta_i / dt = -sum_j A_ij * sin(theta_i - theta_j)

Oscillatoren proberen uit fase te lopen met buren -> convergeert
naar 2-partitie ~ MaxCut. Simpele ODE, een numpy vectoroperatie
per tijdstap.

Gebruik:
  python kuramoto_solver.py --Lx 8 --Ly 3
  python kuramoto_solver.py --Lx 20 --Ly 4 --restarts 50
  python kuramoto_solver.py --Lx 100 --Ly 3 --compare
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# Anti-Kuramoto ODE Solver
# ================================================================

def _fiedler_init(n, ei, ej, ew):
    """Spectrale initialisatie via Fiedler-vector."""
    L = np.zeros((n, n))
    for k in range(len(ei)):
        i, j, w = ei[k], ej[k], ew[k]
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Grootste eigenvector van L = MaxCut spectrale relaxatie
    # (Fiedler = 2e kleinste = min-cut, niet MaxCut)
    fiedler = eigenvectors[:, -1]
    theta = np.where(fiedler >= 0,
                     np.random.uniform(0, 0.3, n),
                     np.random.uniform(np.pi - 0.3, np.pi + 0.3, n))
    return theta


def _multi_angle_round(theta, ei, ej, ew, n):
    """Probeer meerdere rounding-hoeken en kies de beste cut."""
    best_cut = -1
    best_assign = None
    for angle_offset in np.linspace(0, np.pi, 12, endpoint=False):
        shifted = (theta + angle_offset) % (2 * np.pi)
        assign = (shifted >= np.pi).astype(np.int32)
        cut = _eval_cut_array(assign, ei, ej, ew)
        if cut > best_cut:
            best_cut = cut
            best_assign = assign.copy()
    return best_assign, best_cut


def _compute_dtheta(theta, ei, ej, ew, n):
    """Bereken d theta / dt = -sum_j w_ij sin(theta_i - theta_j)."""
    dtheta = np.zeros(n)
    diff_ij = theta[ei] - theta[ej]
    sin_diff = np.sin(diff_ij)
    np.add.at(dtheta, ei, -ew * sin_diff)
    np.add.at(dtheta, ej,  ew * sin_diff)
    return dtheta


def _eval_cut_array(assign, ei, ej, ew):
    """Bereken cut-waarde."""
    mask = (assign[ei] != assign[ej])
    return float(np.sum(ew[mask]))


def _local_search(assign, ei, ej, ew, n):
    """Greedy local search: flip node als dat cut verbetert."""
    adj_w = [[] for _ in range(n)]
    for k in range(len(ei)):
        adj_w[ei[k]].append((ej[k], ew[k]))
        adj_w[ej[k]].append((ei[k], ew[k]))
    n_flips = 0
    improved = True
    while improved:
        improved = False
        for v in range(n):
            gain = 0.0
            for u, w in adj_w[v]:
                if assign[v] == assign[u]:
                    gain += w
                else:
                    gain -= w
            if gain > 1e-10:
                assign[v] = 1 - assign[v]
                improved = True
                n_flips += 1
    cut = _eval_cut_array(assign, ei, ej, ew)
    return assign, cut, n_flips


def kuramoto_maxcut(n_nodes, edges, n_restarts=20, max_iter=2000,
                    dt=0.1, tol=1e-6, anneal=True, verbose=False):
    """Anti-Kuramoto MaxCut solver.

    Integreert de anti-Kuramoto ODE met RK4 en rondt af naar
    een bipartitie. Multi-start met local search.

    Verbeteringen t.o.v. naief Kuramoto:
    - Spectrale (Fiedler) initialisatie voor 1e restart
    - Multi-angle rounding (12 drempels i.p.v. alleen pi)
    - RK4 integratie met annealing
    - Greedy local search na rounding
    """
    n = n_nodes
    ne = len(edges)

    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float64)

    t0 = time.time()
    best_cut = -1
    best_assign = None
    history = []

    # Restart 0: directe Fiedler rounding (geen ODE) als snelle baseline
    if n <= 2000:
        fiedler_theta = _fiedler_init(n, ei, ej, ew)
        assign, cut = _multi_angle_round(fiedler_theta, ei, ej, ew, n)
        assign, cut, n_flips = _local_search(assign, ei, ej, ew, n)
        history.append(cut)
        if cut > best_cut:
            best_cut = cut
            best_assign = assign.copy()
            if verbose:
                print("  fiedler:    cut=%d/%d (%.4f) [direct rounding, %d flips]" %
                      (cut, ne, cut/ne, n_flips))

    for restart in range(n_restarts):
        if restart == 0 and n <= 2000:
            theta = _fiedler_init(n, ei, ej, ew)
        elif restart % 4 == 0 and n <= 2000:
            theta = _fiedler_init(n, ei, ej, ew)
            theta += np.random.uniform(-0.8, 0.8, n)
        else:
            theta = np.random.uniform(0, 2 * np.pi, n)

        cur_dt = dt
        converged = False

        for step in range(max_iter):
            dtheta = _compute_dtheta(theta, ei, ej, ew, n)

            if step < 100 and step % 20 == 0:
                noise_scale = 0.1 * (1.0 - step / 100.0)
                dtheta += np.random.normal(0, noise_scale, n)

            k1 = dtheta
            k2 = _compute_dtheta(theta + 0.5 * cur_dt * k1, ei, ej, ew, n)
            k3 = _compute_dtheta(theta + 0.5 * cur_dt * k2, ei, ej, ew, n)
            k4 = _compute_dtheta(theta + cur_dt * k3, ei, ej, ew, n)
            theta = theta + (cur_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            theta = theta % (2 * np.pi)

            max_deriv = np.max(np.abs(dtheta))
            if max_deriv < tol:
                converged = True
                break

            if anneal and step > 0 and step % 200 == 0:
                cur_dt *= 0.7

        assign, cut = _multi_angle_round(theta, ei, ej, ew, n)
        assign, cut, n_flips = _local_search(assign, ei, ej, ew, n)
        history.append(cut)

        if cut > best_cut:
            best_cut = cut
            best_assign = assign.copy()
            if verbose:
                print("  restart %d: cut=%d/%d (%.4f) %s [dt=%.4f, %d steps, %d flips]" %
                      (restart, cut, ne, cut/ne,
                       "CONVERGED" if converged else "MAX_ITER",
                       cur_dt, step+1, n_flips))

    solve_time = time.time() - t0

    if verbose:
        print("  Beste: cut=%d/%d (%.6f) in %.2fs (%d restarts)" %
              (best_cut, ne, best_cut/ne, solve_time, n_restarts))

    return {
        "best_cut": best_cut,
        "best_assignment": best_assign,
        "n_edges": ne,
        "ratio": best_cut / ne if ne > 0 else 0,
        "history": history,
        "solve_time": solve_time,
    }


# ================================================================
# Grid convenience
# ================================================================

def kuramoto_maxcut_grid(Lx, Ly, **kwargs):
    """Anti-Kuramoto solver op Lx x Ly grid."""
    from rqaoa import WeightedGraph
    g = WeightedGraph.grid(Lx, Ly)
    n = g.n_nodes
    edges = [(i, j, w) for i, j, w in g.edges()]
    return kuramoto_maxcut(n, edges, **kwargs)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B92: Anti-Kuramoto MaxCut Solver")
    parser.add_argument("--Lx", type=int, default=8)
    parser.add_argument("--Ly", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--compare", action="store_true",
                        help="Vergelijk met BM en brute force")
    args = parser.parse_args()

    from rqaoa import WeightedGraph, brute_force_maxcut

    Lx, Ly = args.Lx, args.Ly
    g = WeightedGraph.grid(Lx, Ly)
    n = g.n_nodes
    edges = [(i, j, w) for i, j, w in g.edges()]
    ne = len(edges)

    print("=" * 60)
    print("  B92: Anti-Kuramoto MaxCut Solver")
    print("  Grid: %dx%d (%d nodes, %d edges)" % (Lx, Ly, n, ne))
    print("=" * 60)
    print()

    result = kuramoto_maxcut(n, edges, n_restarts=args.restarts,
                             max_iter=args.max_iter, dt=args.dt,
                             verbose=True)
    print()
    print("Kuramoto: cut=%d/%d  ratio=%.6f  [%.2fs]" %
          (result["best_cut"], ne, result["ratio"], result["solve_time"]))

    if args.compare:
        print()
        print("--- Vergelijking ---")
        try:
            from bm_solver import bm_sdp_solve_fast
            bm_edges = [(i, j, w) for i, j, w in edges]
            t0 = time.time()
            bm = bm_sdp_solve_fast(n, bm_edges, n_restarts=5, verbose=False)
            bm_t = time.time() - t0
            print("BM:       cut=%d/%d  ratio=%.6f  [%.2fs]" %
                  (bm["best_cut"], ne, bm["best_cut"]/ne, bm_t))
        except ImportError:
            print("BM: niet beschikbaar")
        if n <= 24:
            t0 = time.time()
            bf_cut, bf_assign = brute_force_maxcut(g)
            bf_t = time.time() - t0
            print("Exact:    cut=%d/%d  ratio=%.6f  [%.2fs]" %
                  (bf_cut, ne, bf_cut/ne, bf_t))
            print("Gap:      Kuramoto %+d vs exact" %
                  (result["best_cut"] - bf_cut))


if __name__ == "__main__":
    main()
