#!/usr/bin/env python3
"""Tests voor B73: Quantum-Guided Branch-and-Bound."""

import unittest
import numpy as np
from quantum_branch_bound import (
    eval_cut,
    eval_partial_cut,
    greedy_extend,
    compute_upper_bound_greedy,
    branching_order_degree,
    branching_order_quantum,
    branching_order_hybrid,
    quantum_branch_bound,
    qbb_maxcut,
    BnBResult,
)


def triangle_graph():
    """Driehoek: n=3, edges met w=1."""
    return 3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]


def path_graph(n):
    """Pad-graaf: n nodes, n-1 edges."""
    edges = [(i, i+1, 1.0) for i in range(n-1)]
    return n, edges


def complete_graph(n):
    """Complete graaf K_n met w=1."""
    edges = [(i, j, 1.0) for i in range(n) for j in range(i+1, n)]
    return n, edges


def grid_graph(Lx, Ly):
    """Grid Lx x Ly."""
    n = Lx * Ly
    edges = []
    for r in range(Ly):
        for c in range(Lx):
            node = r * Lx + c
            if c + 1 < Lx:
                edges.append((node, node + 1, 1.0))
            if r + 1 < Ly:
                edges.append((node, node + Lx, 1.0))
    return n, edges


class TestEvalFunctions(unittest.TestCase):
    """Test cut-evaluatie functies."""

    def test_eval_cut_triangle(self):
        n, edges = triangle_graph()
        # Optimaal: 2 van 3 edges gesneden
        cut = eval_cut(n, edges, {0: 0, 1: 1, 2: 0})
        self.assertEqual(cut, 2.0)

    def test_eval_cut_all_same(self):
        n, edges = triangle_graph()
        cut = eval_cut(n, edges, {0: 0, 1: 0, 2: 0})
        self.assertEqual(cut, 0.0)

    def test_eval_partial_cut(self):
        n, edges = triangle_graph()
        cut = eval_partial_cut(edges, {0: 0, 1: 1})
        self.assertEqual(cut, 1.0)

    def test_eval_partial_empty(self):
        n, edges = triangle_graph()
        cut = eval_partial_cut(edges, {})
        self.assertEqual(cut, 0.0)

    def test_eval_cut_weighted(self):
        edges = [(0, 1, 3.0), (1, 2, 2.0), (0, 2, 1.0)]
        cut = eval_cut(3, edges, {0: 0, 1: 1, 2: 0})
        self.assertEqual(cut, 5.0)  # 3.0 + 2.0


class TestGreedyExtend(unittest.TestCase):
    """Test greedy assignment extension."""

    def test_greedy_from_empty(self):
        n, edges = triangle_graph()
        assign = greedy_extend(n, edges, {})
        self.assertEqual(len(assign), n)
        cut = eval_cut(n, edges, assign)
        self.assertGreater(cut, 0)

    def test_greedy_from_partial(self):
        n, edges = triangle_graph()
        assign = greedy_extend(n, edges, {0: 0})
        self.assertEqual(len(assign), n)
        self.assertEqual(assign[0], 0)

    def test_greedy_from_full(self):
        n, edges = triangle_graph()
        full = {0: 0, 1: 1, 2: 0}
        assign = greedy_extend(n, edges, full)
        self.assertEqual(assign, full)

    def test_greedy_path(self):
        n, edges = path_graph(6)
        assign = greedy_extend(n, edges, {})
        cut = eval_cut(n, edges, assign)
        # Optimaal voor pad: alternating → n-1 = 5
        self.assertGreaterEqual(cut, 3)


class TestUpperBound(unittest.TestCase):
    """Test upper bound berekeningen."""

    def test_greedy_bound_triangle(self):
        n, edges = triangle_graph()
        ub = compute_upper_bound_greedy(n, edges, {})
        self.assertEqual(ub, 3.0)  # 3 edges, alle w=1

    def test_greedy_bound_with_fixed(self):
        n, edges = triangle_graph()
        ub = compute_upper_bound_greedy(n, edges, {0: 0, 1: 0})
        # edge (0,1): same → 0, edges (0,2) en (1,2): max(0,1)=1 elk
        self.assertEqual(ub, 2.0)

    def test_bound_geq_optimal(self):
        """Upper bound moet >= optimale cut zijn."""
        n, edges = triangle_graph()
        ub = compute_upper_bound_greedy(n, edges, {})
        # Optimaal = 2
        self.assertGreaterEqual(ub, 2.0)


class TestBranchingOrder(unittest.TestCase):
    """Test branching heuristieken."""

    def test_degree_order(self):
        n, edges = triangle_graph()
        order = branching_order_degree(n, edges, {})
        self.assertEqual(len(order), n)
        self.assertEqual(set(order), {0, 1, 2})

    def test_degree_with_fixed(self):
        n, edges = triangle_graph()
        order = branching_order_degree(n, edges, {0: 0})
        self.assertEqual(len(order), 2)
        self.assertNotIn(0, order)

    def test_quantum_order(self):
        n, edges = triangle_graph()
        zz = {(0, 1): 0.9, (0, 2): 0.1, (1, 2): 0.5}
        order = branching_order_quantum(n, edges, zz, {})
        self.assertEqual(len(order), n)
        # Node 2 raakt edges (0,2)=0.1 en (1,2)=0.5 → avg=0.3 (laagste)
        # Moet eerst komen
        self.assertEqual(order[0], 2)

    def test_hybrid_order(self):
        n, edges = triangle_graph()
        zz = {(0, 1): 0.9, (0, 2): 0.1, (1, 2): 0.5}
        order = branching_order_hybrid(n, edges, zz, {}, alpha=0.5)
        self.assertEqual(len(order), n)


class TestBranchAndBound(unittest.TestCase):
    """Test de B&B engine."""

    def test_triangle_exact(self):
        n, edges = triangle_graph()
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 2.0)
        self.assertTrue(result.is_exact)

    def test_path_exact(self):
        n, edges = path_graph(6)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 5.0)  # alternating
        self.assertTrue(result.is_exact)

    def test_k4_exact(self):
        n, edges = complete_graph(4)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 4.0)  # K4 MaxCut = 4
        self.assertTrue(result.is_exact)

    def test_grid_2x3(self):
        n, edges = grid_graph(2, 3)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 7.0)  # bipartiet → alle edges
        self.assertTrue(result.is_exact)

    def test_quantum_branching(self):
        """Quantum branching met mock ZZ correlaties."""
        n, edges = complete_graph(4)
        zz = {}
        for u, v, w in edges:
            zz[(u, v)] = np.random.uniform(-1, 1)
        result = quantum_branch_bound(n, edges, zz_dict=zz,
                                       branching='quantum',
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 4.0)

    def test_hybrid_branching(self):
        n, edges = complete_graph(4)
        zz = {(u, v): 0.5 for u, v, w in edges}
        result = quantum_branch_bound(n, edges, zz_dict=zz,
                                       branching='hybrid', alpha=0.7,
                                       time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 4.0)

    def test_warm_start(self):
        n, edges = triangle_graph()
        warm = {0: 0, 1: 1, 2: 0}
        result = quantum_branch_bound(n, edges, warm_start=warm,
                                       branching='degree',
                                       time_limit=5, verbose=False)
        self.assertGreaterEqual(result.best_cut, 2.0)

    def test_time_limit(self):
        """Met extreme time limit moet het snel stoppen."""
        n, edges = complete_graph(20)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=0.01, verbose=False)
        self.assertGreater(result.best_cut, 0)
        self.assertLess(result.time_s, 1.0)

    def test_max_nodes(self):
        n, edges = complete_graph(15)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       max_nodes=100, time_limit=5,
                                       verbose=False)
        self.assertLessEqual(result.nodes_explored, 110)

    def test_result_fields(self):
        n, edges = triangle_graph()
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=5, verbose=False)
        self.assertIsInstance(result, BnBResult)
        self.assertIsInstance(result.best_cut, float)
        self.assertIsInstance(result.assignment, dict)
        self.assertIsInstance(result.is_exact, bool)
        self.assertIsInstance(result.nodes_explored, int)
        self.assertIsInstance(result.nodes_pruned, int)
        self.assertGreater(result.time_s, 0)

    def test_weighted_edges(self):
        """Test met gewogen edges."""
        edges = [(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)]
        result = quantum_branch_bound(3, edges, branching='degree',
                                       time_limit=5, verbose=False)
        # Optimaal: snij (0,1)=5 en (1,2)=3 → 8
        self.assertEqual(result.best_cut, 8.0)
        self.assertTrue(result.is_exact)

    def test_negative_weights(self):
        """Test met +-1 Ising gewichten."""
        edges = [(0, 1, 1.0), (1, 2, -1.0), (0, 2, 1.0)]
        result = quantum_branch_bound(3, edges, branching='degree',
                                       time_limit=5, verbose=False)
        # Optimaal: {0:0, 1:1, 2:1} → snij (0,1)=1 + (0,2)=1, miss (1,2) → 2
        # Of {0:0, 1:0, 2:1} → snij (0,2)=1, miss (0,1) en (1,2)=-1 bijdraagt 0 → 1
        # Eigenlijk: MaxCut = max Σ w_ij * (x_i != x_j)
        # {0:0,1:1,2:1}: 1+0+1=2, {0:0,1:1,2:0}: 1+(-1)+1=1, {0:0,1:0,2:1}: 0+(-1)+1=0
        self.assertGreaterEqual(result.best_cut, 2.0)


class TestQBBPipeline(unittest.TestCase):
    """Test de volledige QBB pipeline (met QAOA als beschikbaar)."""

    def test_qbb_maxcut_small(self):
        n, edges = triangle_graph()
        result = qbb_maxcut(n, edges, branching='degree',
                            time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 2.0)

    def test_qbb_maxcut_fallback(self):
        """Als QAOA niet beschikbaar, fallback naar degree."""
        n, edges = complete_graph(4)
        result = qbb_maxcut(n, edges, branching='hybrid',
                            time_limit=5, verbose=False)
        self.assertEqual(result.best_cut, 4.0)


class TestExactness(unittest.TestCase):
    """Verifieer dat B&B exact is voor kleine instanties."""

    def test_exact_k5(self):
        """K5: MaxCut = n*(n-1)/4 = 5 (afgerond) → eigenlijk floor(n^2/4)=6."""
        n, edges = complete_graph(5)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=10, verbose=False)
        self.assertTrue(result.is_exact)
        # K5 MaxCut = 6 (partitie 2-3)
        self.assertEqual(result.best_cut, 6.0)

    def test_exact_k6(self):
        n, edges = complete_graph(6)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=10, verbose=False)
        self.assertTrue(result.is_exact)
        # K6 MaxCut = 9 (partitie 3-3)
        self.assertEqual(result.best_cut, 9.0)

    def test_exact_grid_3x3(self):
        n, edges = grid_graph(3, 3)
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=10, verbose=False)
        self.assertTrue(result.is_exact)
        self.assertEqual(result.best_cut, 12.0)  # bipartiet

    def test_exact_agrees_with_brute_force(self):
        """BnB moet overeenkomen met brute force op n=10."""
        np.random.seed(42)
        n = 10
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if np.random.random() < 0.3:
                    edges.append((i, j, 1.0))

        # Brute force
        best_bf = 0
        for mask in range(2**n):
            assign = {i: (mask >> i) & 1 for i in range(n)}
            cut = eval_cut(n, edges, assign)
            best_bf = max(best_bf, cut)

        # B&B
        result = quantum_branch_bound(n, edges, branching='degree',
                                       time_limit=10, verbose=False)
        self.assertTrue(result.is_exact)
        self.assertEqual(result.best_cut, best_bf)


if __name__ == '__main__':
    unittest.main()
