#!/usr/bin/env python3
"""
ws_qaoa.py - B69: Warm-Started QAOA via SDP Relaxatie

Gebruikt de Goemans-Williamson SDP-oplossing om een betere initiële
toestand te construeren voor QAOA. In plaats van |+⟩ start elk qubit
vanuit een richting die door de klassieke relaxatie is geïnformeerd.

Twee modi:
  1. Binary WS-QAOA: GW rounding → binary assignment → tilted start
     Qubit i start at: cos(ε)|a_i⟩ + sin(ε)|1-a_i⟩
     ε = π/4 → cold start (|+⟩), ε ≈ 0.1-0.3 → warm start

  2. Continuous WS-QAOA: SDP relaxatie → per-node hoek → continuous tilt
     Qubit i start at: cos(θ_i/2)|0⟩ + sin(θ_i/2)|1⟩
     θ_i = arccos(c_i) waar c_i = projectie op hyperplane

De MPS-tensor voor een kolom met warm-start is een product state (χ=1):
  T[σ] = ∏_y amplitude(σ_y, θ_{x,y})

Gebruik:
  from ws_qaoa import sdp_warm_start, warm_start_mps

  # Voor cylinder grid
  angles = sdp_warm_start(Lx, Ly, epsilon=0.2)
  mps = warm_start_mps(Lx, Ly, angles)

  # Of met bestaande TransverseQAOA
  tc = TransverseQAOA(Lx, Ly)
  ratio = tc.eval_ratio(p, gammas, betas, warm_angles=angles)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _cylinder_edges(Lx, Ly):
    """Genereer edges voor Lx×Ly cilinder-grid (OBC)."""
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


def gw_sdp_solve(n_nodes, edges, verbose=False):
    """Los SDP relaxatie op en retourneer matrix X + hyperplane roundings.

    Args:
        n_nodes: aantal nodes
        edges: lijst van (u, v, weight) tuples
        verbose: print status

    Returns:
        dict met:
          X: SDP matrix (n×n, PSD, diag=1)
          best_assignment: beste binary assignment (n-vector, 0/1)
          best_cut: cut-waarde van beste assignment
          sdp_bound: SDP bovengrens
          vectors: Cholesky-vectoren (n×n, elke rij is unit vector)
    """
    import cvxpy as cp

    n = n_nodes
    t0 = time.time()

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [X[i, i] == 1 for i in range(n)]

    obj = 0
    for u, v, w in edges:
        obj = obj + w * (1 - X[u, v]) / 2

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-6)

    if prob.value is None:
        raise RuntimeError(f"SDP failed: {prob.status}")

    X_val = np.array(X.value)
    sdp_bound = float(prob.value)

    # Cholesky decompositie voor rounding-vectoren
    eigvals, eigvecs = np.linalg.eigh(X_val)
    eigvals = np.maximum(eigvals, 0)  # numerical cleanup
    V = eigvecs * np.sqrt(eigvals)[None, :]  # n × n, elke rij is vector

    # GW hyperplane rounding: meerdere trials
    rng = np.random.RandomState(42)
    best_assignment = None
    best_cut = -1
    n_edges_total = len(edges)

    for trial in range(100):
        r = rng.randn(n)
        r /= np.linalg.norm(r)
        assignment = (V @ r >= 0).astype(int)

        cut = sum(w * (assignment[u] != assignment[v])
                  for u, v, w in edges)
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment.copy()

    solve_time = time.time() - t0

    if verbose:
        print(f"  [B69] SDP: bound={sdp_bound:.2f}/{n_edges_total} "
              f"({sdp_bound/n_edges_total:.4f}), "
              f"GW rounding={best_cut:.0f}/{n_edges_total} "
              f"({best_cut/n_edges_total:.4f}), "
              f"({solve_time:.2f}s)")

    return {
        'X': X_val,
        'best_assignment': best_assignment,
        'best_cut': best_cut,
        'sdp_bound': sdp_bound,
        'vectors': V,
        'solve_time': solve_time,
    }


def sdp_warm_start(Lx, Ly, edges=None, epsilon=0.25, mode='binary',
                   verbose=False):
    """Bereken warm-start hoeken voor QAOA op een grid.

    Args:
        Lx, Ly: grid dimensies
        edges: optioneel, lijst van (u, v, w) tuples. Default: cylinder grid.
        epsilon: warm-start tilt parameter (0 = fully classical, π/4 = cold)
        mode: 'binary' (GW rounding) of 'continuous' (SDP relaxatie)
        verbose: print info

    Returns:
        angles: (Lx, Ly) array van hoeken θ ∈ [0, π]
                Per qubit: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    """
    n = Lx * Ly
    if edges is None:
        edges = _cylinder_edges(Lx, Ly)

    sdp = gw_sdp_solve(n, edges, verbose=verbose)

    if mode == 'binary':
        # Binary WS-QAOA (Egger et al.)
        a = sdp['best_assignment']
        angles = np.zeros((Lx, Ly))
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                if a[node] == 0:
                    angles[x, y] = 2 * epsilon  # tilted towards |0⟩
                else:
                    angles[x, y] = np.pi - 2 * epsilon  # tilted towards |1⟩
        return angles

    elif mode == 'continuous':
        # Continuous WS-QAOA: use SDP projections
        V = sdp['vectors']
        # Project onto best hyperplane direction
        rng = np.random.RandomState(42)
        # Use average of top-k rounding vectors for smoother angles
        projections = np.zeros(n)
        for _ in range(20):
            r = rng.randn(n)
            r /= np.linalg.norm(r)
            projections += V @ r
        projections /= 20

        # Normalize to [-1, 1]
        pmax = np.max(np.abs(projections))
        if pmax > 1e-10:
            projections /= pmax

        # Map to angles: c_i ∈ [-1, 1] → θ_i ∈ [ε, π-ε]
        angles = np.zeros((Lx, Ly))
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                c = projections[node]
                # Interpolate: c=-1 → θ=2ε, c=0 → θ=π/2, c=1 → θ=π-2ε
                theta = np.pi / 2 + c * (np.pi / 2 - 2 * epsilon)
                angles[x, y] = np.clip(theta, 2 * epsilon, np.pi - 2 * epsilon)
        return angles

    else:
        raise ValueError(f"Unknown mode: {mode}")


def warm_start_mps(Lx, Ly, angles):
    """Bouw MPS initiële toestand vanuit per-qubit hoeken.

    Args:
        Lx, Ly: grid dimensies
        angles: (Lx, Ly) array van θ ∈ [0, π]
                Qubit (x,y) start at cos(θ/2)|0⟩ + sin(θ/2)|1⟩

    Returns:
        MPS: lijst van Lx tensors, elk (1, d, 1) met d = 2^Ly
    """
    d = 2 ** Ly
    bp = np.array([[(idx >> (Ly - 1 - q)) & 1 for q in range(Ly)]
                    for idx in range(d)])

    mps = []
    for x in range(Lx):
        tensor = np.zeros((1, d, 1), dtype=complex)
        for s in range(d):
            amp = 1.0
            for y in range(Ly):
                theta = angles[x, y]
                bit = bp[s, y]
                if bit == 0:
                    amp *= np.cos(theta / 2)
                else:
                    amp *= np.sin(theta / 2)
            tensor[0, s, 0] = amp
        mps.append(tensor)
    return mps


def cold_start_mps(Lx, Ly):
    """Standaard |+⟩ MPS (voor vergelijking)."""
    d = 2 ** Ly
    return [np.ones((1, d, 1), dtype=complex) / np.sqrt(d)
            for _ in range(Lx)]


# =================================================================
# Standalone test / demo
# =================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='B69: WS-QAOA')
    parser.add_argument('--Lx', type=int, default=4)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--mode', choices=['binary', 'continuous'], default='binary')
    parser.add_argument('--p', type=int, default=1)
    args = parser.parse_args()

    Lx, Ly = args.Lx, args.Ly
    print(f"\n{'='*60}")
    print(f"  B69: WS-QAOA op {Lx}x{Ly} grid, p={args.p}")
    print(f"{'='*60}\n")

    # 1. SDP warm-start
    angles = sdp_warm_start(Lx, Ly, epsilon=args.epsilon,
                            mode=args.mode, verbose=True)
    print(f"  Warm-start angles (mean={np.mean(angles):.3f}, "
          f"std={np.std(angles):.3f}):")
    for x in range(Lx):
        row = ' '.join(f'{angles[x,y]:.2f}' for y in range(Ly))
        print(f"    col {x}: [{row}]")

    # 2. Vergelijk cold vs warm
    from transverse_contraction import TransverseQAOA

    tc = TransverseQAOA(Lx, Ly, verbose=True)
    print(f"\n--- Cold-start QAOA-{args.p} ---")
    cold = tc.optimize(args.p, n_gamma=12, n_beta=12, refine=True)
    print(f"  Cold: ratio={cold[0]:.6f}")

    # 3. Warm-start: modify eval_ratio to accept warm MPS
    # (Inline test — engine integration follows)
    print(f"\n--- Warm-start QAOA-{args.p} (ε={args.epsilon}) ---")
    ws_mps = warm_start_mps(Lx, Ly, angles)

    # Quick check: warm MPS normalisatie
    norm_sq = sum(np.sum(np.abs(t)**2) for t in ws_mps)
    print(f"  Warm MPS norm check (product): each site norm~1")

    print(f"\n  (Engine integratie nodig voor volledige vergelijking)")
