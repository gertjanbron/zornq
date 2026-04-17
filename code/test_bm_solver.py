#!/usr/bin/env python3
"""
test_bm_solver.py - Tests voor B91: Burer-Monteiro Warm-Start.

Tests:
  1. BM op bipartite grid (4x3): vindt optimale cut
  2. BM op random graaf: cut >= 50% van edges
  3. BM vs cvxpy SDP: vergelijkbare kwaliteit
  4. BM warm-start angles: correct bereik en structuur
  5. BM warm-start + TransverseQAOA: ratio >= cold start
  6. Schaalbaarheid: 50x4 (200 nodes) in < 10s
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bm_solver import (bm_sdp_solve, bm_sdp_solve_fast,
                        bm_warm_start, _cylinder_edges)


def test_bipartite_grid():
    """Test 1: BM op 4x3 bipartite grid — moet optimum vinden."""
    Lx, Ly = 4, 3
    n = Lx * Ly
    edges = _cylinder_edges(Lx, Ly)
    n_edges = len(edges)

    result = bm_sdp_solve_fast(n, edges, n_restarts=3, verbose=False)
    ratio = result['best_cut'] / n_edges
    print("    4x3 grid: cut=%d/%d ratio=%.6f" %
          (result['best_cut'], n_edges, ratio))

    # Bipartite grid: optimale cut = alle edges
    assert result['best_cut'] == n_edges, \
        "Bipartite grid: verwacht optimale cut %d, got %d" % (
            n_edges, result['best_cut'])
    assert ratio == 1.0, "Verwacht ratio=1.0, got %.4f" % ratio

    print("  [PASS] test_bipartite_grid")


def test_random_graph():
    """Test 2: BM op random graaf — cut >= 50% van edges."""
    rng = np.random.RandomState(42)
    n = 20
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.3:
                edges.append((i, j, 1.0))
    n_edges = len(edges)

    result = bm_sdp_solve_fast(n, edges, n_restarts=5, verbose=False)
    ratio = result['best_cut'] / n_edges
    print("    Random(20, p=0.3): cut=%d/%d ratio=%.6f" %
          (result['best_cut'], n_edges, ratio))

    assert ratio >= 0.5, "Random graaf: verwacht ratio >= 0.5, got %.4f" % ratio
    assert result['best_assignment'].shape == (n,), \
        "Assignment shape mismatch"
    assert set(np.unique(result['best_assignment'])).issubset({0, 1}), \
        "Assignment moet 0/1 zijn"

    print("  [PASS] test_random_graph")


def test_bm_vs_cvxpy():
    """Test 3: BM vs cvxpy SDP — vergelijkbare kwaliteit."""
    try:
        from ws_qaoa import gw_sdp_solve
    except ImportError:
        print("  [SKIP] test_bm_vs_cvxpy — cvxpy niet beschikbaar")
        return

    Lx, Ly = 6, 3
    n = Lx * Ly
    edges = _cylinder_edges(Lx, Ly)
    n_edges = len(edges)

    # cvxpy SDP
    t0 = time.time()
    sdp = gw_sdp_solve(n, edges, verbose=False)
    sdp_time = time.time() - t0

    # BM
    t1 = time.time()
    bm = bm_sdp_solve_fast(n, edges, n_restarts=5, verbose=False)
    bm_time = time.time() - t1

    sdp_ratio = sdp['best_cut'] / n_edges
    bm_ratio = bm['best_cut'] / n_edges

    print("    6x3: SDP cut=%d (%.4f, %.2fs), BM cut=%d (%.4f, %.2fs)" %
          (sdp['best_cut'], sdp_ratio, sdp_time,
           bm['best_cut'], bm_ratio, bm_time))

    # BM mag max 5% slechter zijn dan SDP
    assert bm_ratio >= sdp_ratio - 0.05, \
        "BM (%.4f) significant slechter dan SDP (%.4f)" % (bm_ratio, sdp_ratio)

    print("  [PASS] test_bm_vs_cvxpy")


def test_warm_start_angles():
    """Test 4: BM warm-start angles — correct bereik."""
    Lx, Ly = 4, 3
    angles = bm_warm_start(Lx, Ly, epsilon=0.25, mode='binary', verbose=False)

    assert angles.shape == (Lx, Ly), \
        "Shape mismatch: verwacht (%d,%d), got %s" % (Lx, Ly, angles.shape)

    # Alle hoeken moeten in [2*epsilon, pi-2*epsilon]
    eps = 0.25
    assert np.all(angles >= 2 * eps - 0.01), \
        "Hoeken te klein: min=%.4f" % np.min(angles)
    assert np.all(angles <= np.pi - 2 * eps + 0.01), \
        "Hoeken te groot: max=%.4f" % np.max(angles)

    # Bipartite grid: hoeken moeten twee clusters vormen
    unique_vals = np.unique(np.round(angles, 4))
    assert len(unique_vals) == 2, \
        "Bipartite grid verwacht 2 unieke hoeken, got %d: %s" % (
            len(unique_vals), unique_vals)

    print("    Angles: %s" % unique_vals)

    # Test continuous mode
    angles_c = bm_warm_start(Lx, Ly, epsilon=0.25, mode='continuous',
                              verbose=False)
    assert angles_c.shape == (Lx, Ly)
    assert np.all(angles_c >= 2 * eps - 0.01)
    assert np.all(angles_c <= np.pi - 2 * eps + 0.01)

    print("  [PASS] test_warm_start_angles")


def test_warm_start_qaoa():
    """Test 5: BM warm-start + TransverseQAOA."""
    try:
        from transverse_contraction import TransverseQAOA
    except ImportError:
        print("  [SKIP] test_warm_start_qaoa — TransverseQAOA niet beschikbaar")
        return

    Lx, Ly = 4, 3

    angles = bm_warm_start(Lx, Ly, epsilon=0.2, mode='binary', verbose=False)
    tc = TransverseQAOA(Lx, Ly, verbose=False)

    # Cold start
    cold_ratio, _, _, _ = tc.optimize(1, n_gamma=12, n_beta=12, refine=True)

    # Warm start
    warm_ratio, _, _, _ = tc.optimize(1, n_gamma=12, n_beta=12, refine=True,
                                       warm_angles=angles)

    print("    4x3 Cold: %.6f, Warm(BM): %.6f, delta: %+.6f" %
          (cold_ratio, warm_ratio, warm_ratio - cold_ratio))

    # Warm start moet minstens zo goed zijn als cold
    assert warm_ratio >= cold_ratio - 0.01, \
        "Warm (%.4f) significant slechter dan cold (%.4f)" % (
            warm_ratio, cold_ratio)

    print("  [PASS] test_warm_start_qaoa")


def test_scalability():
    """Test 6: Schaalbaarheid — 50x4 (200 nodes) in < 10s."""
    Lx, Ly = 50, 4
    n = Lx * Ly
    edges = _cylinder_edges(Lx, Ly)
    n_edges = len(edges)

    t0 = time.time()
    result = bm_sdp_solve_fast(n, edges, n_restarts=3,
                                max_iter=200, verbose=False)
    elapsed = time.time() - t0

    ratio = result['best_cut'] / n_edges
    print("    50x4 (%d nodes): cut=%d/%d ratio=%.6f [%.2fs]" %
          (n, result['best_cut'], n_edges, ratio, elapsed))

    assert elapsed < 10, "Te traag: %.1fs > 10s" % elapsed
    assert ratio >= 0.5, "Ratio te laag: %.4f" % ratio

    print("  [PASS] test_scalability")


if __name__ == '__main__':
    print("=== B91 BM-QAOA Tests ===\n")

    test_bipartite_grid()
    test_random_graph()
    test_bm_vs_cvxpy()
    test_warm_start_angles()
    test_warm_start_qaoa()
    test_scalability()

    print("\n=== Alle tests PASSED ===")
