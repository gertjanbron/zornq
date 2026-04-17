#!/usr/bin/env python3
"""Tests voor cut_sparsifier.py (B118)."""
import sys, unittest
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from cut_sparsifier import (
    effective_resistance_sparsify, degree_weighted_sparsify,
    weight_threshold_sparsify, sparsify, sparsified_maxcut
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


def eval_cut(n, edges, assignment):
    cut = 0.0
    for u, v, w in edges:
        if assignment.get(int(u), 0) != assignment.get(int(v), 0):
            cut += float(w)
    return cut


class TestWeightThreshold(unittest.TestCase):
    def test_no_removal(self):
        n, edges = make_grid(4, 4)
        sparse, info = weight_threshold_sparsify(n, edges, threshold=0.5)
        self.assertEqual(len(sparse), len(edges))

    def test_full_removal(self):
        n, edges = make_grid(4, 4)
        sparse, info = weight_threshold_sparsify(n, edges, threshold=2.0)
        self.assertEqual(len(sparse), 0)

    def test_partial_removal(self):
        edges = [(0, 1, 1.0), (1, 2, 0.5), (2, 3, 0.1)]
        sparse, info = weight_threshold_sparsify(4, edges, threshold=0.3)
        self.assertEqual(len(sparse), 2)

    def test_signed_edges(self):
        edges = [(0, 1, -1.0), (1, 2, 0.5), (2, 3, -0.1)]
        sparse, info = weight_threshold_sparsify(4, edges, threshold=0.3)
        self.assertEqual(len(sparse), 2)


class TestDegreeWeighted(unittest.TestCase):
    def test_basic(self):
        n, edges = make_grid(6, 4)
        sparse, info = degree_weighted_sparsify(n, edges, epsilon=0.5, seed=42)
        self.assertGreater(len(sparse), 0)
        self.assertLessEqual(len(sparse), len(edges))
        self.assertEqual(info['method'], 'degree_weighted')

    def test_small_epsilon_keeps_more(self):
        n, edges = make_grid(10, 4)
        sparse_tight, _ = degree_weighted_sparsify(n, edges, epsilon=0.1, seed=42)
        sparse_loose, _ = degree_weighted_sparsify(n, edges, epsilon=0.8, seed=42)
        self.assertGreaterEqual(len(sparse_tight), len(sparse_loose))

    def test_empty_graph(self):
        sparse, info = degree_weighted_sparsify(5, [], epsilon=0.3)
        self.assertEqual(len(sparse), 0)

    def test_signed(self):
        n, edges = make_grid(8, 4, signed=True)
        sparse, info = degree_weighted_sparsify(n, edges, epsilon=0.5, seed=42)
        self.assertGreater(len(sparse), 0)


class TestEffectiveResistance(unittest.TestCase):
    def test_basic(self):
        n, edges = make_grid(5, 4)
        sparse, info = effective_resistance_sparsify(n, edges, epsilon=0.3, seed=42)
        self.assertGreater(len(sparse), 0)
        self.assertEqual(info['method'], 'effective_resistance')

    def test_path_graph(self):
        edges = [(i, i+1, 1.0) for i in range(9)]
        sparse, info = effective_resistance_sparsify(10, edges, epsilon=0.3, seed=42)
        self.assertGreaterEqual(len(sparse), 5)

    def test_complete_graph(self):
        n = 10
        edges = [(i, j, 1.0) for i in range(n) for j in range(i+1, n)]
        sparse, info = effective_resistance_sparsify(n, edges, epsilon=0.8, seed=42)
        self.assertLessEqual(len(sparse), len(edges))

    def test_signed(self):
        n, edges = make_grid(5, 4, signed=True)
        sparse, info = effective_resistance_sparsify(n, edges, epsilon=0.3, seed=42)
        self.assertGreater(len(sparse), 0)


class TestSparsifyAuto(unittest.TestCase):
    def test_small_uses_er(self):
        n, edges = make_grid(5, 4)
        sparse, info = sparsify(n, edges, epsilon=0.3, method='auto')
        self.assertEqual(info['method'], 'effective_resistance')

    def test_method_override(self):
        n, edges = make_grid(5, 4)
        sparse, info = sparsify(n, edges, epsilon=0.3, method='dw')
        self.assertEqual(info['method'], 'degree_weighted')


class TestSparsifiedMaxcut(unittest.TestCase):
    def test_with_greedy_solver(self):
        n, edges = make_grid(6, 4)

        def simple_solver(n_nodes, sparse_edges):
            assign = {}
            adj = {}
            for u, v, w in sparse_edges:
                adj.setdefault(u, []).append((v, w))
                adj.setdefault(v, []).append((u, w))
            for node in range(n_nodes):
                if node not in assign:
                    assign[node] = 0
                    queue = [node]
                    while queue:
                        curr = queue.pop(0)
                        for nb, w in adj.get(curr, []):
                            if nb not in assign:
                                assign[nb] = 1 - assign[curr]
                                queue.append(nb)
            cut = eval_cut(n_nodes, sparse_edges, assign)
            return cut, assign

        cut, assign, info = sparsified_maxcut(
            n, edges, simple_solver, epsilon=0.3, verbose=False)
        self.assertGreater(cut, 0)
        self.assertEqual(len(assign), n)

    def test_with_b99v2(self):
        from feedback_edge_solver import _multi_tree_ensemble
        n, edges = make_grid(8, 4, signed=True)

        def b99_solver(n_nodes, sparse_edges):
            cut, assign = _multi_tree_ensemble(
                n_nodes, sparse_edges, time_limit=2, seed=42, n_trees=5)
            return cut, assign

        cut_sparse, assign, info = sparsified_maxcut(
            n, edges, b99_solver, epsilon=0.3, verbose=False)
        cut_direct, _ = _multi_tree_ensemble(n, edges, time_limit=2, seed=42, n_trees=5)
        self.assertGreater(cut_sparse, cut_direct * 0.8)


class TestCutPreservation(unittest.TestCase):
    def test_bipartite_grid(self):
        n, edges = make_grid(6, 4)
        max_cut = len(edges)
        assign = {}
        Ly = 4
        for x in range(6):
            for y in range(Ly):
                assign[x * Ly + y] = (x + y) % 2
        orig_cut = eval_cut(n, edges, assign)
        self.assertEqual(orig_cut, max_cut)
        sparse_edges, info = sparsify(n, edges, epsilon=0.3)
        sparse_cut = eval_cut(n, sparse_edges, assign)
        ratio = sparse_cut / max_cut
        self.assertGreater(ratio, 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
