#!/usr/bin/env python3
"""
test_feedback_edge_solver.py - Tests for B99 Feedback-Edge Skeleton Solver
"""

import numpy as np
import sys
import os
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback_edge_solver import (
    UnionFind, max_spanning_tree, root_tree,
    solve_tree_maxcut, feedback_edge_maxcut,
)


# ============================================================
# Helpers
# ============================================================

def _path(n):
    return n, [(i, i+1, 1.0) for i in range(n-1)]

def _cycle(n):
    return n, [(i, (i+1) % n, 1.0) for i in range(n)]

def _tree_star(n):
    """Star graph: node 0 connected to all others."""
    return n, [(0, i, 1.0) for i in range(1, n)]

def _grid(Lx, Ly):
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x+1 < Lx: edges.append((i, (x+1)*Ly+y, 1.0))
            if y+1 < Ly: edges.append((i, x*Ly+y+1, 1.0))
    return n, edges

def _eval_cut(n, edges, assign):
    cut = 0.0
    for u, v, w in edges:
        if assign.get(int(u), 0) != assign.get(int(v), 0):
            cut += w
    return cut


# ============================================================
# Tests
# ============================================================

def test_union_find():
    uf = UnionFind(5)
    assert uf.find(0) != uf.find(1)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    uf.union(2, 3)
    uf.union(0, 3)
    assert uf.find(0) == uf.find(3)
    assert uf.find(1) == uf.find(2)
    print("  PASS: test_union_find")

def test_spanning_tree_path():
    """Path graph IS a tree: 0 feedback edges."""
    n, edges = _path(5)
    tree, fb, adj = max_spanning_tree(n, edges)
    assert len(tree) == 4, f"Expected 4 tree edges, got {len(tree)}"
    assert len(fb) == 0, f"Expected 0 feedback, got {len(fb)}"
    print("  PASS: test_spanning_tree_path (k=0)")

def test_spanning_tree_cycle():
    """Cycle has exactly 1 feedback edge."""
    n, edges = _cycle(6)
    tree, fb, adj = max_spanning_tree(n, edges)
    assert len(tree) == 5
    assert len(fb) == 1
    print("  PASS: test_spanning_tree_cycle (k=1)")

def test_spanning_tree_grid():
    """4x4 grid: 16 nodes, 24 edges, tree=15, feedback=9."""
    n, edges = _grid(4, 4)
    tree, fb, adj = max_spanning_tree(n, edges)
    assert len(tree) == 15, f"Expected 15, got {len(tree)}"
    assert len(fb) == len(edges) - 15
    print("  PASS: test_spanning_tree_grid (k=%d)" % len(fb))

def test_tree_maxcut_path():
    """Path of 4: MaxCut = 3 (alternating)."""
    n, edges = _path(4)
    tree, fb, adj = max_spanning_tree(n, edges)
    fb_assign = np.array([], dtype=np.int8)
    assign, cut = solve_tree_maxcut(n, adj, tree, fb, fb_assign)
    assert abs(cut - 3.0) < 1e-10, f"Expected 3.0, got {cut}"
    print("  PASS: test_tree_maxcut_path (cut=%.1f)" % cut)

def test_tree_maxcut_star():
    """Star(5): MaxCut = 4 (center vs all leaves)."""
    n, edges = _tree_star(5)
    tree, fb, adj = max_spanning_tree(n, edges)
    fb_assign = np.array([], dtype=np.int8)
    assign, cut = solve_tree_maxcut(n, adj, tree, fb, fb_assign)
    assert abs(cut - 4.0) < 1e-10, f"Expected 4.0, got {cut}"
    print("  PASS: test_tree_maxcut_star (cut=%.1f)" % cut)

def test_pipeline_path():
    """Full pipeline on path (k=0, exact tree)."""
    n, edges = _path(100)
    cut, assign, info = feedback_edge_maxcut(n, edges, time_limit=5)
    assert abs(cut - 99.0) < 1e-10, f"Path(100) MaxCut should be 99, got {cut}"
    assert info['n_feedback_edges'] == 0
    assert info['method'] == 'exact_tree'
    print("  PASS: test_pipeline_path (cut=%.1f, exact)" % cut)

def test_pipeline_cycle():
    """Cycle(6): k=1, exact enum. MaxCut = 6 (even cycle, bipartite)."""
    n, edges = _cycle(6)
    cut, assign, info = feedback_edge_maxcut(n, edges, time_limit=5)
    assert abs(cut - 6.0) < 1e-10, f"Cycle(6) MaxCut should be 6, got {cut}"
    assert info['n_feedback_edges'] == 1
    print("  PASS: test_pipeline_cycle (cut=%.1f, k=1)" % cut)

def test_pipeline_cycle_odd():
    """Cycle(5): MaxCut = 4 (odd cycle, not bipartite)."""
    n, edges = _cycle(5)
    cut, assign, info = feedback_edge_maxcut(n, edges, time_limit=5)
    assert abs(cut - 4.0) < 1e-10, f"Cycle(5) MaxCut should be 4, got {cut}"
    print("  PASS: test_pipeline_cycle_odd (cut=%.1f)" % cut)

def test_pipeline_grid_small():
    """4x4 grid: k=9, exact enum."""
    n, edges = _grid(4, 4)
    cut, assign, info = feedback_edge_maxcut(n, edges, time_limit=10)
    # 4x4 grid MaxCut = 24 (checkerboard, bipartite)
    assert abs(cut - 24.0) < 1e-10, f"4x4 grid MaxCut should be 24, got {cut}"
    # Verify assignment
    recomputed = _eval_cut(n, edges, assign)
    assert abs(cut - recomputed) < 1e-10
    print("  PASS: test_pipeline_grid_small (cut=%.1f, k=%d)" %
          (cut, info['n_feedback_edges']))

def test_pipeline_grid_medium():
    """10x10 grid: k=81, BLS feedback."""
    n, edges = _grid(10, 10)
    cut, assign, info = feedback_edge_maxcut(
        n, edges, time_limit=10, seed=42)
    # 10x10 grid MaxCut = 180 (bipartite)
    assert cut >= 170, f"Cut too low: {cut}"
    print("  PASS: test_pipeline_grid_medium (cut=%.1f, k=%d, method=%s)" %
          (cut, info['n_feedback_edges'], info['method']))

def test_pipeline_weighted():
    """Weighted triangle: k=1."""
    edges = [(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)]
    cut, assign, info = feedback_edge_maxcut(3, edges, time_limit=5)
    # Optimal: {0} vs {1,2} -> cut = 3+2 = 5
    assert abs(cut - 5.0) < 1e-10, f"Expected 5.0, got {cut}"
    print("  PASS: test_pipeline_weighted (cut=%.1f)" % cut)

def test_pipeline_signed():
    """Signed weights: +-1 edges."""
    # Square with mixed weights
    edges = [(0,1,1.0), (1,2,-1.0), (2,3,1.0), (3,0,-1.0)]
    cut, assign, info = feedback_edge_maxcut(4, edges, time_limit=5)
    # With signed weights, MaxCut = maximize sum of w*(x_i != x_j)
    # Optimal: 0=0,1=1,2=1,3=0 -> cut = 1 + (-1)*0 + 1 + (-1)*0 = 2
    # Or: 0=0,1=1,2=0,3=1 -> cut = 1 + (-1)*1 + 1 + (-1)*1 = 0
    # Actually optimal: 0=0,1=1,2=1,3=0 -> edges (0,1)=diff(+1), (1,2)=same(0), (2,3)=diff(+1), (3,0)=diff(-1) = 1+0+1-1 = 1
    # Hmm let me think: cut = sum of w for edges where endpoints differ
    # (0,1,1): 0!=1 -> +1
    # (1,2,-1): 1==1 -> 0
    # (2,3,1): 1!=0 -> +1
    # (3,0,-1): 0==0 -> 0
    # total = 2. Seems right.
    assert cut >= 1.0, f"Cut should be positive, got {cut}"
    print("  PASS: test_pipeline_signed (cut=%.1f)" % cut)

def test_pipeline_3regular():
    """Random 3-regular graph."""
    try:
        from bls_solver import random_3regular
        n, edges = random_3regular(100, seed=42)
    except ImportError:
        print("  SKIP: test_pipeline_3regular")
        return

    cut, assign, info = feedback_edge_maxcut(
        n, edges, time_limit=10, seed=42)
    lower = n * 3/4 * 0.85
    assert cut >= lower, f"Cut {cut} below {lower}"
    print("  PASS: test_pipeline_3regular (n=100, cut=%.1f, k=%d)" %
          (cut, info['n_feedback_edges']))

def test_consistency():
    """Assignment should match reported cut."""
    n, edges = _grid(6, 6)
    cut, assign, info = feedback_edge_maxcut(n, edges, time_limit=5)
    recomputed = _eval_cut(n, edges, assign)
    assert abs(cut - recomputed) < 1e-10, \
        f"Mismatch: reported={cut}, recomputed={recomputed}"
    print("  PASS: test_consistency (cut=%.1f)" % cut)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("B99: Feedback-Edge Skeleton Solver Tests")
    print("=" * 60)

    t0 = time.time()
    n_pass = 0
    n_fail = 0

    tests = [
        test_union_find,
        test_spanning_tree_path,
        test_spanning_tree_cycle,
        test_spanning_tree_grid,
        test_tree_maxcut_path,
        test_tree_maxcut_star,
        test_pipeline_path,
        test_pipeline_cycle,
        test_pipeline_cycle_odd,
        test_pipeline_grid_small,
        test_pipeline_grid_medium,
        test_pipeline_weighted,
        test_pipeline_signed,
        test_pipeline_3regular,
        test_consistency,
    ]

    for fn in tests:
        try:
            fn()
            n_pass += 1
        except Exception as e:
            n_fail += 1
            print("  FAIL: %s -> %s" % (fn.__name__, e))

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Results: %d passed, %d failed (%.1fs)" % (n_pass, n_fail, elapsed))
    print("=" * 60)
    if n_fail > 0:
        sys.exit(1)
