#!/usr/bin/env python3
"""
bm_solver.py - B91: Burer-Monteiro Warm-Start voor QAOA MaxCut.

Vervangt cvxpy SDP (O(n^2) geheugen, O(n^3) tijd) door Burer-Monteiro
lage-rang factorizatie (O(n*k) geheugen, veel sneller).

Idee: In plaats van de volle n x n SDP matrix X te optimaliseren,
parametriseer X = V V^T met V in R^{n x k}, k = ceil(sqrt(2n)).
Elke rij v_i is een eenheidsvector op de bol.

Maximaliseer: sum_{(i,j) in E} w_ij (1 - v_i . v_j) / 2
Constraint:   ||v_i|| = 1  (project na elke stap)

GW rounding: random hyperplane op V, zelfde als origineel.

Drop-in vervanging voor gw_sdp_solve() uit ws_qaoa.py:
  van: sdp = gw_sdp_solve(n, edges) met cvxpy
  naar: sdp = bm_sdp_solve(n, edges) zonder cvxpy

Gebruik:
  python bm_solver.py --Lx 8 --Ly 3
  python bm_solver.py --Lx 50 --Ly 4 --compare
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Burer-Monteiro SDP Solver
# =====================================================================

def bm_sdp_solve(n_nodes, edges, k=None, n_restarts=5,
                 max_iter=500, lr=0.05, tol=1e-6, verbose=False):
    """Burer-Monteiro low-rank SDP solver voor MaxCut.

    Maximaliseert: sum w_ij (1 - v_i . v_j) / 2
    met ||v_i|| = 1 voor alle i.

    Args:
        n_nodes: aantal nodes
        edges: lijst van (u, v, weight) tuples
        k: rang van V (default: ceil(sqrt(2*n)))
        n_restarts: aantal random initialisaties
        max_iter: max iteraties per restart
        lr: learning rate (wordt adaptief geschaald)
        tol: convergentie-tolerantie
        verbose: print voortgang

    Returns:
        dict met:
          vectors: V matrix (n x k), elke rij is eenheidsvector
          best_assignment: beste binary assignment (n-array, 0/1)
          best_cut: cut-waarde van beste assignment
          sdp_bound: BM bovengrens (≈ SDP bound)
          solve_time: totale tijd
    """
    n = n_nodes
    if n == 0:
        return {
            'vectors': np.zeros((0, 1)),
            'best_assignment': np.array([], dtype=int),
            'best_cut': 0.0,
            'sdp_bound': 0.0,
            'solve_time': 0.0,
        }

    if k is None:
        k = max(2, int(np.ceil(np.sqrt(2 * n))))
    k = min(k, n)  # k kan niet groter zijn dan n

    t0 = time.time()

    # Bouw sparse edge structuur voor snelle gradient
    # adj[i] = [(j, w), ...]
    adj = [[] for _ in range(n)]
    total_weight = 0.0
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        total_weight += w

    # Objective: f(V) = sum w_ij (1 - v_i . v_j) / 2
    #                  = total_weight/2 - sum w_ij (v_i . v_j) / 2
    # Maximaliseer = minimaliseer sum w_ij (v_i . v_j) / 2

    # Gradient van inner product term t.o.v. V:
    # d/dV_i [sum_j w_ij v_i . v_j] = sum_j w_ij v_j
    # Dus gradient = W @ V  (W = gewogen adjacency matrix, sparse)

    def _objective(V):
        """BM objective: sum w_ij (1 - v_i . v_j) / 2"""
        obj = 0.0
        for u, v, w in edges:
            obj += w * (1 - np.dot(V[u], V[v])) / 2
        return obj

    def _gradient(V):
        """Gradient van -objective (we maximaliseren, dus minimaliseer -f).

        d(-f)/dV_i = sum_j w_ij v_j / 2
        """
        grad = np.zeros_like(V)
        for u, v, w in edges:
            grad[u] += w * V[v] / 2
            grad[v] += w * V[u] / 2
        return grad

    def _project_to_sphere(V):
        """Projecteer elke rij op de eenheidsbol."""
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return V / norms

    best_V = None
    best_obj = -1e9

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + restart)

        # Random initialisatie op de bol
        V = rng.randn(n, k)
        V = _project_to_sphere(V)

        # Riemannian gradient ascent met momentum
        velocity = np.zeros_like(V)
        momentum = 0.9
        current_lr = lr
        prev_obj = _objective(V)

        for it in range(max_iter):
            # Euclidean gradient van de inner product termen
            grad = _gradient(V)

            # Riemannian gradient: projecteer weg van de normaalrichting
            # grad_R_i = grad_i - (grad_i . v_i) v_i
            dots = np.sum(grad * V, axis=1, keepdims=True)
            riem_grad = grad - dots * V

            # Update met momentum
            velocity = momentum * velocity - current_lr * riem_grad
            V = V + velocity
            V = _project_to_sphere(V)

            # Check convergentie
            if (it + 1) % 20 == 0:
                obj = _objective(V)
                improvement = obj - prev_obj
                if abs(improvement) < tol * max(1.0, abs(obj)):
                    break
                if improvement < 0:
                    # Overshoot: halveer learning rate
                    current_lr *= 0.5
                prev_obj = obj

        obj = _objective(V)
        if obj > best_obj:
            best_obj = obj
            best_V = V.copy()

    if verbose:
        print("  [B91] BM-SDP: obj=%.4f (%.4f per edge), k=%d, %d restarts" %
              (best_obj, best_obj / len(edges) if edges else 0, k, n_restarts))

    # GW hyperplane rounding
    rng = np.random.RandomState(42)
    best_assignment = np.zeros(n, dtype=int)
    best_cut = -1.0
    n_rounds = max(100, 10 * k)  # meer trials bij hogere rang

    for trial in range(n_rounds):
        r = rng.randn(k)
        r /= np.linalg.norm(r)
        assignment = (best_V @ r >= 0).astype(int)

        cut = sum(w * (assignment[u] != assignment[v])
                  for u, v, w in edges)
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment.copy()

    solve_time = time.time() - t0

    if verbose:
        n_edges = len(edges)
        print("    Rounding: best_cut=%.0f/%d (%.4f), %d trials [%.2fs]" %
              (best_cut, n_edges, best_cut / n_edges if n_edges else 0,
               n_rounds, solve_time))

    return {
        'vectors': best_V,
        'best_assignment': best_assignment,
        'best_cut': best_cut,
        'sdp_bound': best_obj,
        'solve_time': solve_time,
    }


# =====================================================================
# Vectorized versie (numpy matrix-multiply, veel sneller)
# =====================================================================

def bm_sdp_solve_fast(n_nodes, edges, k=None, n_restarts=5,
                      max_iter=500, lr=0.05, tol=1e-6, verbose=False):
    """Snellere BM solver met numpy vectorized operaties.

    Bouwt sparse adjacency als dense gewichtenmatrix W.
    Gradient = W @ V (matrix-multiply, O(n*n*k) maar gecached).
    Voor n < 5000 is dit sneller dan sparse; daarboven: gebruik sparse.
    """
    n = n_nodes
    if n == 0:
        return bm_sdp_solve(n, edges, verbose=verbose)

    if k is None:
        k = max(2, int(np.ceil(np.sqrt(2 * n))))
    k = min(k, n)

    t0 = time.time()

    # Bouw gewichtenmatrix W
    W = np.zeros((n, n), dtype=np.float64)
    total_weight = 0.0
    for u, v, w in edges:
        W[u, v] += w
        W[v, u] += w
        total_weight += w

    def _objective_vec(V):
        # sum w_ij (1 - v_i . v_j) / 2
        # = total_weight/2 - trace(V^T W V) / 4
        # (factor 1/4 want W telt elke edge dubbel)
        inner = np.sum((V @ V.T) * W) / 4
        return total_weight / 2 - inner

    def _project(V):
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.maximum(norms, 1e-10)

    best_V = None
    best_obj = -1e9

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + restart)
        V = _project(rng.randn(n, k))
        velocity = np.zeros_like(V)
        momentum = 0.9
        current_lr = lr
        prev_obj = _objective_vec(V)

        for it in range(max_iter):
            # Gradient van inner product: d/dV = W @ V / 2
            grad = W @ V / 2

            # Riemannian: verwijder normaal-component
            dots = np.sum(grad * V, axis=1, keepdims=True)
            riem_grad = grad - dots * V

            velocity = momentum * velocity - current_lr * riem_grad
            V = _project(V + velocity)

            if (it + 1) % 20 == 0:
                obj = _objective_vec(V)
                improvement = obj - prev_obj
                if abs(improvement) < tol * max(1.0, abs(obj)):
                    break
                if improvement < 0:
                    current_lr *= 0.5
                prev_obj = obj

        obj = _objective_vec(V)
        if obj > best_obj:
            best_obj = obj
            best_V = V.copy()

    if verbose:
        print("  [B91] BM-SDP (fast): obj=%.4f, k=%d" % (best_obj, k))

    # Rounding
    rng = np.random.RandomState(42)
    best_assignment = np.zeros(n, dtype=int)
    best_cut = -1.0
    n_rounds = max(100, 10 * k)

    for trial in range(n_rounds):
        r = rng.randn(k)
        r /= np.linalg.norm(r)
        assignment = (best_V @ r >= 0).astype(int)
        cut = sum(w * (assignment[u] != assignment[v])
                  for u, v, w in edges)
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment.copy()

    solve_time = time.time() - t0

    if verbose:
        n_edges = len(edges)
        print("    Rounding: best_cut=%.0f/%d (%.4f) [%.2fs]" %
              (best_cut, n_edges, best_cut / n_edges if n_edges else 0,
               solve_time))

    return {
        'vectors': best_V,
        'best_assignment': best_assignment,
        'best_cut': best_cut,
        'sdp_bound': best_obj,
        'solve_time': solve_time,
    }


# =====================================================================
# BM Warm-Start: drop-in vervanging voor sdp_warm_start
# =====================================================================

def bm_warm_start(Lx, Ly, edges=None, epsilon=0.25, mode='binary',
                  verbose=False):
    """Burer-Monteiro warm-start hoeken voor QAOA.

    Drop-in vervanging voor ws_qaoa.sdp_warm_start() — werkt zonder cvxpy.

    Args:
        Lx, Ly: grid dimensies
        edges: optioneel, lijst van (u, v, w). Default: cylinder grid.
        epsilon: tilt parameter (0 = fully classical, pi/4 = cold)
        mode: 'binary' of 'continuous'
        verbose: print info

    Returns:
        angles: (Lx, Ly) array van hoeken theta in [0, pi]
    """
    n = Lx * Ly
    if edges is None:
        edges = _cylinder_edges(Lx, Ly)

    # Gebruik fast variant voor grids tot ~2000 nodes, daarboven sparse
    if n <= 2000:
        sdp = bm_sdp_solve_fast(n, edges, verbose=verbose)
    else:
        sdp = bm_sdp_solve(n, edges, verbose=verbose)

    if mode == 'binary':
        a = sdp['best_assignment']
        angles = np.zeros((Lx, Ly))
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                if a[node] == 0:
                    angles[x, y] = 2 * epsilon
                else:
                    angles[x, y] = np.pi - 2 * epsilon
        return angles

    elif mode == 'continuous':
        V = sdp['vectors']
        rng = np.random.RandomState(42)
        projections = np.zeros(n)
        for _ in range(20):
            r = rng.randn(V.shape[1])
            r /= np.linalg.norm(r)
            projections += V @ r
        projections /= 20

        pmax = np.max(np.abs(projections))
        if pmax > 1e-10:
            projections /= pmax

        angles = np.zeros((Lx, Ly))
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                c = projections[node]
                theta = np.pi / 2 + c * (np.pi / 2 - 2 * epsilon)
                angles[x, y] = np.clip(theta, 2 * epsilon,
                                        np.pi - 2 * epsilon)
        return angles

    else:
        raise ValueError("Unknown mode: %s" % mode)


def _cylinder_edges(Lx, Ly):
    """Genereer edges voor Lx x Ly cilinder-grid (OBC)."""
    edges = []
    for x in range(Lx):
        for y in range(Ly - 1):
            u = x * Ly + y
            v = x * Ly + y + 1
            edges.append((u, v, 1.0))
    for x in range(Lx - 1):
        for y in range(Ly):
            u = x * Ly + y
            v = (x + 1) * Ly + y
            edges.append((u, v, 1.0))
    return edges


# =====================================================================
# CLI: vergelijking BM vs cvxpy SDP
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B91: BM-QAOA — Burer-Monteiro Warm-Start')
    parser.add_argument('--Lx', type=int, default=8)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--mode', choices=['binary', 'continuous'],
                        default='binary')
    parser.add_argument('--compare', action='store_true',
                        help='Vergelijk met cvxpy SDP')
    parser.add_argument('--large', action='store_true',
                        help='Test op grote graaf (100x4)')
    args = parser.parse_args()

    Lx, Ly = args.Lx, args.Ly
    if args.large:
        Lx, Ly = 100, 4

    n = Lx * Ly
    edges = _cylinder_edges(Lx, Ly)
    n_edges = len(edges)

    print("=" * 60)
    print("  B91: BM-QAOA op %dx%d grid (%d nodes, %d edges)" %
          (Lx, Ly, n, n_edges))
    print("=" * 60)

    # BM solve
    print("\n--- Burer-Monteiro ---")
    t0 = time.time()
    if n <= 2000:
        bm = bm_sdp_solve_fast(n, edges, verbose=True)
    else:
        bm = bm_sdp_solve(n, edges, verbose=True)
    bm_time = time.time() - t0
    bm_ratio = bm['best_cut'] / n_edges
    print("  BM: cut=%d/%d ratio=%.6f bound=%.2f [%.2fs]" %
          (bm['best_cut'], n_edges, bm_ratio, bm['sdp_bound'], bm_time))

    # BM warm-start angles
    print("\n--- BM Warm-Start ---")
    angles = bm_warm_start(Lx, Ly, epsilon=args.epsilon,
                           mode=args.mode, verbose=True)
    print("  Angles: mean=%.3f std=%.3f" %
          (np.mean(angles), np.std(angles)))

    # Vergelijking met cvxpy
    if args.compare:
        print("\n--- cvxpy SDP (vergelijking) ---")
        try:
            from ws_qaoa import gw_sdp_solve
            t1 = time.time()
            sdp = gw_sdp_solve(n, edges, verbose=True)
            sdp_time = time.time() - t1
            sdp_ratio = sdp['best_cut'] / n_edges
            print("  SDP: cut=%d/%d ratio=%.6f bound=%.2f [%.2fs]" %
                  (sdp['best_cut'], n_edges, sdp_ratio,
                   sdp['sdp_bound'], sdp_time))
            print("\n--- Vergelijking ---")
            print("  BM:  ratio=%.6f  tijd=%.2fs" % (bm_ratio, bm_time))
            print("  SDP: ratio=%.6f  tijd=%.2fs" % (sdp_ratio, sdp_time))
            print("  Speedup: %.1fx" % (sdp_time / bm_time
                                         if bm_time > 0 else float('inf')))
        except ImportError:
            print("  cvxpy niet beschikbaar — skip vergelijking")

    # Test met TransverseQAOA warm-start
    print("\n--- QAOA met BM warm-start ---")
    try:
        from transverse_contraction import TransverseQAOA

        tc = TransverseQAOA(Lx, Ly, verbose=False)

        # Cold start
        cold_ratio, _, _, cold_info = tc.optimize(
            1, n_gamma=12, n_beta=12, refine=True)
        print("  Cold QAOA-1: ratio=%.6f [%.1fs]" %
              (cold_ratio, cold_info['total_time']))

        # Warm start met BM angles
        warm_ratio, _, _, warm_info = tc.optimize(
            1, n_gamma=12, n_beta=12, refine=True,
            warm_angles=angles)
        print("  Warm QAOA-1 (BM): ratio=%.6f [%.1fs]" %
              (warm_ratio, warm_info['total_time']))
        print("  Delta: %+.6f (%+.2f%%)" %
              (warm_ratio - cold_ratio,
               100 * (warm_ratio - cold_ratio) / cold_ratio
               if cold_ratio > 0 else 0))
    except Exception as e:
        print("  TransverseQAOA test overgeslagen: %s" % e)


if __name__ == '__main__':
    main()
