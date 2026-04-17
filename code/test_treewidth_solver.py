#!/usr/bin/env python3
"""Tests voor treewidth_solver.py (B42)."""
import sys, unittest
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from treewidth_solver import (
    eval_cut, min_degree_ordering, build_elimination_tree,
    dp_maxcut, treewidth_maxcut, treewidth_estimate
)


def make_grid(Lx, Ly, signed=False, seed=42):
    rng = np.random.default_rng(seed)
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                w = rng.choice([-1.0, 1.0]) if signed else 1.0
                edges.append((i, (x+1)*Ly + y, w))
            if y + 1 < Ly:
                w = rng.choice([-1.0, 1.0]) if signed else 1.0
                edges.append((i, x*Ly + y+1, w))
    return n, edges


def make_path(length):
    edges = [(i, i+1, 1.0) for i in range(length)]
    return length + 1, edges


def make_cycle(length):
    edges = [(i, (i+1) % length, 1.0) for i in range(length)]
    return length, edges


def make_star(k):
    edges = [(0, i, 1.0) for i in range(1, k+1)]
    return k + 1, edges


def make_complete(k):
    edges = []
    for i in range(k):
        for j in range(i+1, k):
            edges.append((i, j, 1.0))
    return k, edges


class TestMinDegreeOrdering(unittest.TestCase):
    def test_path(self):
        n, edges = make_path(5)
        ordering, bags, tw = min_degree_ordering(n, edges)
        assert tw == 1, f"Path treewidth should be 1, got {tw}"
        assert len(ordering) == n

    def test_cycle(self):
        n, edges = make_cycle(6)
        ordering, bags, tw = min_degree_ordering(n, edges)
        assert tw == 2, f"Cycle treewidth should be 2, got {tw}"

    def test_grid_4x4(self):
        n, edges = make_grid(4, 4)
        ordering, bags, tw = min_degree_ordering(n, edges)
        assert tw <= 5, f"4x4 grid tw should be <=5, got {tw}"

    def test_star(self):
        n, edges = make_star(5)
        ordering, bags, tw = min_degree_ordering(n, edges)
        assert tw <= 5

    def test_isolated(self):
        # Graph with isolated vertex
        edges = [(0, 1, 1.0)]
        ordering, bags, tw = min_degree_ordering(3, edges)
        assert tw <= 1
        assert len(ordering) == 3

    def test_complete_4(self):
        n, edges = make_complete(4)
        ordering, bags, tw = min_degree_ordering(n, edges)
        assert tw == 3, f"K4 treewidth should be 3, got {tw}"


class TestBuildTree(unittest.TestCase):
    def test_path(self):
        n, edges = make_path(5)
        ordering, bags, tw = min_degree_ordering(n, edges)
        children, root = build_elimination_tree(ordering, bags)
        assert root == len(bags) - 1

    def test_disconnected(self):
        # Two separate edges
        edges = [(0, 1, 1.0), (2, 3, 1.0)]
        ordering, bags, tw = min_degree_ordering(4, edges)
        children, root = build_elimination_tree(ordering, bags)
        # Should still have a valid tree
        assert root == len(bags) - 1


class TestDPMaxcut(unittest.TestCase):
    def test_single_edge(self):
        edges = [(0, 1, 1.0)]
        result = treewidth_maxcut(2, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        assert abs(cut - 1.0) < 1e-6

    def test_single_edge_negative(self):
        edges = [(0, 1, -1.0)]
        result = treewidth_maxcut(2, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # MaxCut of negative edge: best is same partition (cut=-1) vs same (cut=0)
        # Actually MaxCut maximizes sum of w where endpoints differ
        # w=-1, so cutting gives -1, not cutting gives 0. Max is 0.
        assert cut >= -0.001, f"Expected >= 0, got {cut}"

    def test_path_5(self):
        n, edges = make_path(5)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # Path of 5 vertices: max cut = 5 (alternating)
        # Actually path has 5 edges, all weight 1, max cut = 5
        assert abs(cut - 5.0) < 1e-6, f"Expected 5, got {cut}"
        assert info['treewidth'] == 1

    def test_cycle_even(self):
        n, edges = make_cycle(6)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # Even cycle: max cut = n (all edges cut with alternating)
        assert abs(cut - 6.0) < 1e-6, f"Expected 6, got {cut}"

    def test_cycle_odd(self):
        n, edges = make_cycle(5)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # Odd cycle: max cut = n-1 = 4
        assert abs(cut - 4.0) < 1e-6, f"Expected 4, got {cut}"

    def test_star_5(self):
        n, edges = make_star(5)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # Star: max cut = 5 (center in one partition, leaves in other)
        assert abs(cut - 5.0) < 1e-6

    def test_complete_4(self):
        n, edges = make_complete(4)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # K4 max cut = 4 (split 2-2, each pair cuts 4 edges)
        assert abs(cut - 4.0) < 1e-6, f"Expected 4, got {cut}"

    def test_grid_4x4(self):
        n, edges = make_grid(4, 4)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # 4x4 grid, bipartite -> max cut = all edges = 24
        assert abs(cut - 24.0) < 1e-6, f"Expected 24, got {cut}"

    def test_grid_8x4(self):
        n, edges = make_grid(8, 4)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # 8x4 bipartite grid -> max cut = all 52 edges
        expected = 7*4 + 8*3  # horizontal + vertical = 28+24 = 52
        assert abs(cut - expected) < 1e-6, f"Expected {expected}, got {cut}"

    def test_grid_8x4_signed(self):
        n, edges = make_grid(8, 4, signed=True)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        # Signed grid: cut should be positive and verified
        assert info['verified']
        verified = eval_cut(n, edges, assign)
        assert abs(cut - verified) < 1e-6

    def test_grid_20x4(self):
        n, edges = make_grid(20, 4)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        expected = 19*4 + 20*3  # 76 + 60 = 136
        assert abs(cut - expected) < 1e-6, f"Expected {expected}, got {cut}"

    def test_grid_100x4(self):
        n, edges = make_grid(100, 4)
        result = treewidth_maxcut(n, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        expected = 99*4 + 100*3  # 396 + 300 = 696
        assert abs(cut - expected) < 1e-6, f"Expected {expected}, got {cut}"
        assert info['time'] < 30, f"Too slow: {info['time']:.1f}s"

    def test_disconnected(self):
        edges = [(0, 1, 1.0), (2, 3, 1.0)]
        result = treewidth_maxcut(4, edges, verbose=False)
        assert result is not None
        cut, assign, info = result
        assert abs(cut - 2.0) < 1e-6

    def test_empty_graph(self):
        result = treewidth_maxcut(3, [], verbose=False)
        assert result is not None
        cut, assign, info = result
        assert abs(cut) < 1e-6

    def test_max_tw_exceeded(self):
        n, edges = make_complete(10)
        result = treewidth_maxcut(n, edges, max_tw=5, verbose=False)
        assert result is None


class TestTreewidthEstimate(unittest.TestCase):
    def test_path(self):
        n, edges = make_path(10)
        tw = treewidth_estimate(n, edges)
        assert tw == 1

    def test_grid(self):
        n, edges = make_grid(10, 4)
        tw = treewidth_estimate(n, edges)
        assert tw <= 5, f"Expected <=5, got {tw}"


class TestVerification(unittest.TestCase):
    def test_assignment_complete(self):
        n, edges = make_grid(6, 4)
        result = treewidth_maxcut(n, edges, verbose=False)
        cut, assign, info = result
        for v in range(n):
            assert v in assign, f"Vertex {v} missing from assignment"

    def test_assignment_valid(self):
        n, edges = make_grid(6, 4, signed=True, seed=123)
        result = treewidth_maxcut(n, edges, verbose=False)
        cut, assign, info = result
        # Check assignment only contains 0/1
        for v, s in assign.items():
            assert s in (0, 1), f"Invalid assignment {s} for vertex {v}"
        # Check verified
        assert info['verified']


if __name__ == '__main__':
    unittest.main(verbosity=2)
