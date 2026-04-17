#!/usr/bin/env python3
"""Tests voor schur_complement.py (B119)."""
import sys, unittest
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from schur_complement import ReducedGraph, schur_maxcut, find_bfs_separator


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


def eval_cut(n, edges, assignment):
    cut = 0.0
    for u, v, w in edges:
        if assignment.get(int(u), 0) != assignment.get(int(v), 0):
            cut += float(w)
    return cut


class TestReducedGraph(unittest.TestCase):
    def test_init(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        rg = ReducedGraph(4, edges)
        assert rg.n_alive == 4
        assert rg.n_edges == 3

    def test_degree(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        rg = ReducedGraph(4, edges)
        assert rg.degree(0) == 1
        assert rg.degree(1) == 2
        assert rg.degree(3) == 1

    def test_eliminate_leaf(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        rg = ReducedGraph(4, edges)
        assert rg.eliminate_leaf(0) == True
        assert rg.n_alive == 3
        assert abs(rg.offset - 1.0) < 1e-10
        # Node 1 now has degree 1, so it can also be a leaf
        assert rg.eliminate_leaf(1) == True

    def test_eliminate_leaf_signed(self):
        edges = [(0, 1, -1.0)]
        rg = ReducedGraph(2, edges)
        assert rg.eliminate_leaf(0) == True
        assert abs(rg.offset - 1.0) < 1e-10

    def test_eliminate_degree2(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        rg = ReducedGraph(3, edges)
        assert rg.eliminate_degree2(1) == True
        assert rg.n_alive == 2
        assert abs(rg.offset - 1.0) < 1e-10
        assert 2 in rg.adj[0]
        assert abs(rg.adj[0][2] - 1.0) < 1e-10

    def test_eliminate_degree2_signed(self):
        edges = [(0, 1, 1.0), (1, 2, -1.0)]
        rg = ReducedGraph(3, edges)
        assert rg.eliminate_degree2(1) == True
        assert abs(rg.offset - 1.0) < 1e-10
        assert abs(rg.adj[0][2] - (-1.0)) < 1e-10

    def test_eliminate_degree2_same_sign(self):
        edges = [(0, 1, 2.0), (1, 2, 3.0)]
        rg = ReducedGraph(3, edges)
        assert rg.eliminate_degree2(1) == True
        assert abs(rg.offset - 3.0) < 1e-10
        assert abs(rg.adj[0][2] - 2.0) < 1e-10


class TestReduceIterative(unittest.TestCase):
    def test_path(self):
        edges = [(i, i+1, 1.0) for i in range(4)]
        rg = ReducedGraph(5, edges)
        rg.reduce_iterative()
        assert rg.n_alive <= 2

    def test_cycle(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        rg = ReducedGraph(4, edges)
        rg.reduce_iterative()
        assert rg.n_alive < 4

    def test_grid_positive(self):
        n, edges = make_grid(6, 4)
        rg = ReducedGraph(n, edges)
        rg.reduce_iterative(verbose=False)
        assert rg.n_alive < n

    def test_grid_signed(self):
        n, edges = make_grid(8, 4, signed=True)
        rg = ReducedGraph(n, edges)
        rg.reduce_iterative(verbose=False)
        assert rg.n_alive < n

    def test_star_graph(self):
        edges = [(0, i, 1.0) for i in range(1, 6)]
        rg = ReducedGraph(6, edges)
        rg.reduce_iterative(max_degree=1)
        assert rg.n_alive <= 1
        assert abs(rg.offset - 5.0) < 1e-10


class TestReconstruction(unittest.TestCase):
    def test_leaf_reconstruction(self):
        edges = [(0, 1, 1.0)]
        rg = ReducedGraph(2, edges)
        rg.eliminate_leaf(0)
        full = rg.reconstruct_assignment({1: 0})
        assert full[0] != full[1]

    def test_leaf_reconstruction_negative(self):
        edges = [(0, 1, -1.0)]
        rg = ReducedGraph(2, edges)
        rg.eliminate_leaf(0)
        full = rg.reconstruct_assignment({1: 0})
        assert full[0] == full[1]

    def test_chain_reconstruction(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        rg = ReducedGraph(3, edges)
        rg.eliminate_degree2(1)
        full = rg.reconstruct_assignment({0: 0, 2: 1})
        assert full[1] in [0, 1]

    def test_path_full(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        rg = ReducedGraph(4, edges)
        rg.reduce_iterative()
        remaining = {v: 0 for v in rg.alive}
        full = rg.reconstruct_assignment(remaining)
        for i in range(4):
            assert i in full

    def test_cut_quality(self):
        n, edges = make_grid(6, 4)
        rg = ReducedGraph(n, edges)
        rg.reduce_iterative()
        remaining = {}
        Ly = 4
        for v in rg.alive:
            x, y = divmod(v, Ly)
            remaining[v] = (x + y) % 2
        full = rg.reconstruct_assignment(remaining)
        cut = eval_cut(n, edges, full)
        max_cut = len(edges)
        assert cut / max_cut > 0.7


class TestSchurMaxcut(unittest.TestCase):
    def test_basic(self):
        from feedback_edge_solver import _multi_tree_ensemble
        n, edges = make_grid(8, 4)
        def solver(nn, ee):
            return _multi_tree_ensemble(nn, ee, time_limit=2, seed=42, n_trees=5)
        cut, assign, info = schur_maxcut(n, edges, solver, verbose=False)
        assert cut > 0
        assert len(assign) == n
        assert info['reduction_ratio'] > 0

    def test_signed(self):
        from feedback_edge_solver import _multi_tree_ensemble
        n, edges = make_grid(8, 4, signed=True)
        def solver(nn, ee):
            return _multi_tree_ensemble(nn, ee, time_limit=2, seed=42, n_trees=5)
        cut, assign, info = schur_maxcut(n, edges, solver, verbose=False)
        assert info['n_reduced'] > 0


class TestBfsSeparator(unittest.TestCase):
    def test_grid(self):
        n, edges = make_grid(8, 4)
        sep, interior, exterior = find_bfs_separator(n, edges)
        total = len(sep) + len(interior) + len(exterior)
        assert total == n

    def test_path(self):
        edges = [(i, i+1, 1.0) for i in range(9)]
        sep, interior, exterior = find_bfs_separator(10, edges)
        assert len(sep) > 0


if __name__ == '__main__':
    unittest.main(verbosity=2)
