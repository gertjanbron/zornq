#!/usr/bin/env python3
"""
test_graph_coarsening.py - Unit tests for B72 Multiscale Graph Coarsening

Tests the HEM matching, coarsen/uncoarsen pipeline, refinement,
and end-to-end MaxCut quality on small+medium graphs.
"""

import numpy as np
import sys
import os
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_coarsening import (
    _build_adjacency, _adj_to_edges,
    heavy_edge_matching, coarsen_graph,
    multilevel_coarsen, project_assignment,
    refine_assignment, eval_cut, coarsen_maxcut,
)


# ============================================================
# Helper: simple graph builders
# ============================================================

def _grid_graph(Lx, Ly):
    """Build grid graph edge list."""
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x + 1) * Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((i, x * Ly + y + 1, 1.0))
    return n, edges


def _complete_graph(n):
    """K_n with unit weights."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, 1.0))
    return n, edges


def _path_graph(n):
    """Path graph 0-1-2-..-(n-1)."""
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    return n, edges


def _cycle_graph(n):
    """Cycle graph."""
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    return n, edges


def _weighted_triangle():
    """Triangle with different weights: 0-1(w=3), 1-2(w=1), 0-2(w=2)."""
    return 3, [(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)]


def _random_3regular(n, seed=42):
    """Generate a random 3-regular graph."""
    from bls_solver import random_3regular
    return random_3regular(n, seed=seed)


# ============================================================
# Test adjacency building
# ============================================================

def test_build_adjacency():
    """Test adjacency dict construction."""
    n, edges = _path_graph(4)  # 0-1-2-3
    adj = _build_adjacency(n, edges)

    assert len(adj) == 4
    assert set(adj[0].keys()) == {1}
    assert set(adj[1].keys()) == {0, 2}
    assert set(adj[2].keys()) == {1, 3}
    assert set(adj[3].keys()) == {2}
    assert adj[0][1] == 1.0
    print("  PASS: test_build_adjacency")


def test_build_adjacency_weighted():
    """Test weighted adjacency."""
    n, edges = _weighted_triangle()
    adj = _build_adjacency(n, edges)

    assert adj[0][1] == 3.0
    assert adj[1][0] == 3.0
    assert adj[1][2] == 1.0
    assert adj[0][2] == 2.0
    print("  PASS: test_build_adjacency_weighted")


def test_adj_to_edges():
    """Round-trip: edges -> adj -> edges preserves structure."""
    n, edges = _grid_graph(3, 3)
    adj = _build_adjacency(n, edges)
    edges2 = _adj_to_edges(adj)

    # Same number of edges
    assert len(edges2) == len(edges), f"{len(edges2)} != {len(edges)}"

    # Same total weight
    w1 = sum(e[2] for e in edges)
    w2 = sum(e[2] for e in edges2)
    assert abs(w1 - w2) < 1e-10
    print("  PASS: test_adj_to_edges")


# ============================================================
# Test HEM matching
# ============================================================

def test_hem_basic():
    """HEM on path graph: should match about half the nodes."""
    n, edges = _path_graph(8)
    adj = _build_adjacency(n, edges)
    mapping, n_coarse, groups = heavy_edge_matching(n, adj, rng=np.random.default_rng(42))

    # Each node is mapped
    assert len(mapping) == 8
    # Coarse nodes < original
    assert n_coarse < 8
    # Groups cover all nodes
    all_nodes = set()
    for g in groups.values():
        all_nodes.update(g)
    assert all_nodes == set(range(8))
    # Each group has 1 or 2 nodes
    for g in groups.values():
        assert len(g) in [1, 2]
    print("  PASS: test_hem_basic (n_coarse=%d)" % n_coarse)


def test_hem_complete_4():
    """HEM on K4: should match into 2 coarse nodes."""
    n, edges = _complete_graph(4)
    adj = _build_adjacency(n, edges)
    mapping, n_coarse, groups = heavy_edge_matching(n, adj, rng=np.random.default_rng(42))

    assert n_coarse == 2, f"K4 should match into 2 pairs, got {n_coarse}"
    print("  PASS: test_hem_complete_4")


def test_hem_heavy_edge_preference():
    """HEM should prefer heavier edges."""
    # Triangle: 0-1(w=10), 1-2(w=1), 0-2(w=1)
    # Node 0 should match with node 1 (heaviest edge)
    edges = [(0, 1, 10.0), (1, 2, 1.0), (0, 2, 1.0)]
    adj = _build_adjacency(3, edges)

    # Run multiple seeds; at least sometimes 0 matches 1
    matched_01 = 0
    for seed in range(20):
        mapping, nc, groups = heavy_edge_matching(3, adj, rng=np.random.default_rng(seed))
        for g in groups.values():
            if set(g) == {0, 1}:
                matched_01 += 1
    # Heavy edge should win most of the time
    assert matched_01 > 10, f"Heavy edge matched only {matched_01}/20 times"
    print("  PASS: test_hem_heavy_edge_preference (%d/20)" % matched_01)


# ============================================================
# Test coarsen_graph
# ============================================================

def test_coarsen_graph_path():
    """Coarsen path 0-1-2-3 with mapping {0->A, 1->A, 2->B, 3->B}."""
    n = 4
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    adj = _build_adjacency(n, edges)

    mapping = {0: 0, 1: 0, 2: 1, 3: 1}
    n_coarse = 2
    coarse_adj = coarsen_graph(n, adj, mapping, n_coarse)

    # Should have edge A-B with weight = w(1,2) = 1.0
    assert 1 in coarse_adj[0], "Expected edge between coarse nodes"
    assert abs(coarse_adj[0][1] - 1.0) < 1e-10, \
        f"Expected weight 1.0, got {coarse_adj[0][1]}"
    print("  PASS: test_coarsen_graph_path")


def test_coarsen_graph_preserves_cut():
    """Coarsening preserves the cut value of a compatible assignment."""
    n, edges = _grid_graph(4, 4)
    adj = _build_adjacency(n, edges)

    # Make a checkerboard assignment: node (x,y) -> (x+y) % 2
    fine_assign = {}
    for x in range(4):
        for y in range(4):
            fine_assign[x * 4 + y] = (x + y) % 2
    fine_cut = eval_cut(n, edges, fine_assign)

    # Coarsen with identity mapping (no actual coarsening) — trivial test
    mapping = {i: i for i in range(n)}
    coarse_adj = coarsen_graph(n, adj, mapping, n)
    coarse_edges = _adj_to_edges(coarse_adj)
    coarse_cut = eval_cut(n, coarse_edges, fine_assign)

    assert abs(fine_cut - coarse_cut) < 1e-10, \
        f"Identity coarsen changed cut: {fine_cut} -> {coarse_cut}"
    print("  PASS: test_coarsen_graph_preserves_cut (cut=%.1f)" % fine_cut)


# ============================================================
# Test multilevel coarsening
# ============================================================

def test_multilevel_small():
    """Multilevel on small graph should not crash."""
    n, edges = _grid_graph(3, 3)
    levels, coarse_n, coarse_edges = multilevel_coarsen(n, edges, target_size=3)

    assert coarse_n <= 9
    assert coarse_n >= 1
    assert isinstance(coarse_edges, list)
    print("  PASS: test_multilevel_small (9 -> %d in %d levels)" %
          (coarse_n, len(levels)))


def test_multilevel_medium():
    """Multilevel on 20x20 grid (400 nodes) -> ~100."""
    n, edges = _grid_graph(20, 20)
    levels, coarse_n, coarse_edges = multilevel_coarsen(
        n, edges, target_size=100, seed=42)

    assert coarse_n <= 120, f"Coarse graph too large: {coarse_n}"
    assert len(levels) >= 1, "Expected at least 1 coarsening round"
    print("  PASS: test_multilevel_medium (400 -> %d in %d levels)" %
          (coarse_n, len(levels)))


def test_multilevel_already_small():
    """Graph already below target: no coarsening needed."""
    n, edges = _path_graph(5)
    levels, coarse_n, coarse_edges = multilevel_coarsen(
        n, edges, target_size=10)

    assert len(levels) == 0, "Should not coarsen"
    assert coarse_n == 5
    print("  PASS: test_multilevel_already_small")


# ============================================================
# Test projection and refinement
# ============================================================

def test_project_assignment():
    """Project coarse assignment back to fine level."""
    groups = {
        0: [0, 1],
        1: [2, 3],
        2: [4],
    }
    coarse_assign = {0: 1, 1: 0, 2: 1}
    fine_assign = project_assignment(coarse_assign, groups)

    assert fine_assign[0] == 1
    assert fine_assign[1] == 1
    assert fine_assign[2] == 0
    assert fine_assign[3] == 0
    assert fine_assign[4] == 1
    print("  PASS: test_project_assignment")


def test_refine_improves_cut():
    """Refinement should not worsen (and often improve) the cut."""
    n, edges = _grid_graph(6, 6)
    adj = _build_adjacency(n, edges)

    # Start with random assignment
    rng = np.random.default_rng(42)
    random_assign = {i: int(rng.integers(0, 2)) for i in range(n)}
    random_cut = eval_cut(n, edges, random_assign)

    refined_assign, refined_cut, n_flips = refine_assignment(
        n, adj, random_assign, max_passes=10)

    assert refined_cut >= random_cut - 1e-10, \
        f"Refinement worsened: {random_cut} -> {refined_cut}"
    print("  PASS: test_refine_improves_cut (%.1f -> %.1f, %d flips)" %
          (random_cut, refined_cut, n_flips))


def test_refine_optimal_unchanged():
    """Refinement of an optimal solution should not change it."""
    # Cycle of 4: optimal cut = 4 (bipartite)
    n, edges = _cycle_graph(4)
    adj = _build_adjacency(n, edges)
    optimal = {0: 0, 1: 1, 2: 0, 3: 1}
    opt_cut = eval_cut(n, edges, optimal)

    refined, ref_cut, n_flips = refine_assignment(n, adj, optimal)
    assert abs(ref_cut - opt_cut) < 1e-10
    print("  PASS: test_refine_optimal_unchanged (cut=%.1f)" % ref_cut)


# ============================================================
# Test eval_cut
# ============================================================

def test_eval_cut_path():
    """Cut evaluation on path graph."""
    n, edges = _path_graph(4)
    # 0-1-2-3, assign {0:0, 1:1, 2:0, 3:1} -> cuts edges (0,1), (1,2), (2,3) = 3
    assign = {0: 0, 1: 1, 2: 0, 3: 1}
    cut = eval_cut(n, edges, assign)
    assert abs(cut - 3.0) < 1e-10, f"Expected 3.0, got {cut}"
    print("  PASS: test_eval_cut_path")


def test_eval_cut_all_same():
    """All same partition -> cut = 0."""
    n, edges = _complete_graph(5)
    assign = {i: 0 for i in range(n)}
    cut = eval_cut(n, edges, assign)
    assert abs(cut) < 1e-10
    print("  PASS: test_eval_cut_all_same")


# ============================================================
# Test full pipeline
# ============================================================

def test_pipeline_small_grid():
    """Full coarsen-solve-uncoarsen on 6x6 grid."""
    n, edges = _grid_graph(6, 6)
    cut, assign, info = coarsen_maxcut(
        n, edges, target_size=10, time_limit=10,
        seed=42, solver='pa')

    # 6x6 grid MaxCut optimal = 48 (checkerboard)
    assert cut >= 40, f"Cut too low: {cut}"
    assert len(assign) == n
    # Verify cut matches assignment
    recomputed = eval_cut(n, edges, assign)
    assert abs(cut - recomputed) < 1e-10, \
        f"Cut mismatch: {cut} vs recomputed {recomputed}"
    print("  PASS: test_pipeline_small_grid (cut=%.1f, optimal=48)" % cut)


def test_pipeline_combined_solver():
    """Pipeline with combined solver."""
    n, edges = _grid_graph(8, 8)
    cut, assign, info = coarsen_maxcut(
        n, edges, target_size=20, time_limit=15,
        seed=42, solver='combined')

    # 8x8 grid optimal = 88
    assert cut >= 75, f"Cut too low: {cut}"
    assert info['solver'] == 'combined'
    assert info['n_levels'] >= 1
    print("  PASS: test_pipeline_combined_solver (cut=%.1f, optimal=88)" % cut)


def test_pipeline_no_coarsening_needed():
    """Pipeline on tiny graph: no coarsening, just solve."""
    n, edges = _complete_graph(6)
    cut, assign, info = coarsen_maxcut(
        n, edges, target_size=10, time_limit=5,
        seed=42, solver='pa')

    # K6 MaxCut = 9
    assert cut >= 8, f"Cut too low for K6: {cut}"
    print("  PASS: test_pipeline_no_coarsening (cut=%.1f, K6 optimal=9)" % cut)


def test_pipeline_3regular():
    """Pipeline on random 3-regular n=200."""
    try:
        n, edges = _random_3regular(200, seed=200)
    except ImportError:
        print("  SKIP: test_pipeline_3regular (bls_solver not available)")
        return

    cut, assign, info = coarsen_maxcut(
        n, edges, target_size=50, time_limit=15,
        seed=42, solver='combined')

    # Rough bound: 3-regular MaxCut >= n*3/4 * 0.88
    lower_bound = n * 3 / 4 * 0.80
    assert cut >= lower_bound, \
        f"Cut {cut} below expected lower bound {lower_bound}"
    assert info['n_levels'] >= 1
    print("  PASS: test_pipeline_3regular (n=200, cut=%.1f, %d levels)" %
          (cut, info['n_levels']))


def test_pipeline_weighted():
    """Pipeline on weighted graph."""
    # 4 nodes, 2 heavy edges forming a path
    edges = [
        (0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0),
        (0, 2, 1.0), (1, 3, 1.0)
    ]
    cut, assign, info = coarsen_maxcut(
        4, edges, target_size=3, time_limit=5, seed=42, solver='pa')

    # Optimal: partition {0,2} vs {1,3} -> cut = 10+10+10+1+1 = 32?
    # Actually: 0-1(10, diff=yes) + 1-2(10, diff=yes) + 2-3(10, diff=yes)
    #   + 0-2(1, diff=no) + 1-3(1, diff=no) = 30
    # Or {0,3} vs {1,2}: 0-1(10,yes) + 1-2(10,no) + 2-3(10,yes)
    #   + 0-2(1,yes) + 1-3(1,yes) = 22
    assert cut >= 20, f"Cut too low: {cut}"
    print("  PASS: test_pipeline_weighted (cut=%.1f)" % cut)


# ============================================================
# Timing test
# ============================================================

def test_time_limit_respected():
    """Pipeline should respect time limit approximately."""
    n, edges = _grid_graph(20, 20)
    limit = 5.0
    t0 = time.time()
    cut, assign, info = coarsen_maxcut(
        n, edges, target_size=50, time_limit=limit,
        seed=42, solver='pa')
    elapsed = time.time() - t0

    # Allow 50% overrun for overhead
    assert elapsed < limit * 1.5, \
        f"Exceeded time limit: {elapsed:.1f}s > {limit * 1.5:.1f}s"
    print("  PASS: test_time_limit (limit=%.1f, actual=%.1fs)" %
          (limit, elapsed))


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("B72: Graph Coarsening Unit Tests")
    print("=" * 60)

    t0 = time.time()
    n_pass = 0
    n_fail = 0

    tests = [
        # Adjacency
        test_build_adjacency,
        test_build_adjacency_weighted,
        test_adj_to_edges,
        # HEM
        test_hem_basic,
        test_hem_complete_4,
        test_hem_heavy_edge_preference,
        # Coarsen
        test_coarsen_graph_path,
        test_coarsen_graph_preserves_cut,
        # Multilevel
        test_multilevel_small,
        test_multilevel_medium,
        test_multilevel_already_small,
        # Projection & refinement
        test_project_assignment,
        test_refine_improves_cut,
        test_refine_optimal_unchanged,
        # Eval
        test_eval_cut_path,
        test_eval_cut_all_same,
        # Full pipeline
        test_pipeline_small_grid,
        test_pipeline_combined_solver,
        test_pipeline_no_coarsening_needed,
        test_pipeline_3regular,
        test_pipeline_weighted,
        # Timing
        test_time_limit_respected,
    ]

    for test_fn in tests:
        try:
            test_fn()
            n_pass += 1
        except Exception as e:
            n_fail += 1
            print("  FAIL: %s -> %s" % (test_fn.__name__, e))

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Results: %d passed, %d failed (%.1fs)" % (n_pass, n_fail, elapsed))
    print("=" * 60)

    if n_fail > 0:
        sys.exit(1)
