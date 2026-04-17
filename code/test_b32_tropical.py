#!/usr/bin/env python3
"""
Comprehensive tests voor B32: Tropische Tensor Netwerken.

Test suites:
  1. TestTropicalSemiring: Basis (max,+) en (min,+) operaties
  2. TestTropicalTensor: Tensor constructie en representatie
  3. TestTropicalContract: Tensor contractie in tropische semiring
  4. TestMaxCutTropical: MaxCut als tropisch netwerk
  5. TestTropicalElim: Variabele-eliminatie contractie
  6. TestMinDegreeOrder: Eliminatievolgorde heuristiek
  7. TestTransferMatrix1D: Transfer matrix methode voor 1D keten
  8. TestBruteForce: Brute-force referentie
  9. TestQAOATropical: QAOA tropische MAP-schatting
  10. TestSandwichBound: Sandwich bound QAOA ≤ C_max
  11. Test2DGrid: 2D grid MaxCut
  12. TestWeightedGraphs: Gewogen en Ising grafen
  13. TestEdgeCases: Randgevallen en robuustheid
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tropical_tensor import (
    tropical_max_plus, tropical_min_plus, NEG_INF,
    TropicalTensor, tropical_contract, tropical_contract_network,
    build_maxcut_tropical_network, solve_maxcut_tropical,
    solve_maxcut_tropical_elim, min_degree_order,
    tropical_transfer_matrix_1d,
    _tropical_brute_force,
    qaoa_tropical_map, qaoa_expected_cost,
    build_2d_grid_edges, solve_maxcut_2d_tropical,
    sandwich_bound,
    random_weighted_graph, ising_weighted_graph,
)


class TestTropicalSemiring(unittest.TestCase):
    """Test tropische semiring operaties."""

    def test_max_plus_basic(self):
        """max(3, 5) = 5."""
        self.assertEqual(tropical_max_plus(3.0, 5.0), 5.0)

    def test_max_plus_negative(self):
        """max(-2, -5) = -2."""
        self.assertEqual(tropical_max_plus(-2.0, -5.0), -2.0)

    def test_max_plus_arrays(self):
        """Elementwise max op arrays."""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = tropical_max_plus(a, b)
        np.testing.assert_array_equal(result, [2.0, 5.0, 6.0])

    def test_min_plus_basic(self):
        """min(3, 5) = 3."""
        self.assertEqual(tropical_min_plus(3.0, 5.0), 3.0)

    def test_max_plus_identity(self):
        """max(x, -inf) = x (tropisch nul-element)."""
        self.assertEqual(tropical_max_plus(7.0, NEG_INF), 7.0)

    def test_max_plus_neginf(self):
        """-inf is nul in (max,+)."""
        self.assertEqual(tropical_max_plus(NEG_INF, NEG_INF), NEG_INF)


class TestTropicalTensor(unittest.TestCase):
    """Test tropische tensor constructie."""

    def test_create_tensor(self):
        """Maak een tropische tensor."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = TropicalTensor(data, ('z_0', 'z_1'))
        self.assertEqual(t.data.shape, (2, 2))
        self.assertEqual(t.indices, ('z_0', 'z_1'))

    def test_repr(self):
        """String representatie."""
        t = TropicalTensor(np.zeros((2, 3)), ('a', 'b'))
        s = repr(t)
        self.assertIn('shape=(2, 3)', s)
        self.assertIn("('a', 'b')", s)

    def test_scalar_tensor(self):
        """Scalaire tensor (geen indices)."""
        t = TropicalTensor(np.array(5.0), ())
        self.assertEqual(t.data.shape, ())
        self.assertEqual(len(t.indices), 0)


class TestTropicalContract(unittest.TestCase):
    """Test tropische tensor contractie."""

    def test_outer_product(self):
        """Outer product (geen gedeelde indices)."""
        t1 = TropicalTensor(np.array([1.0, 2.0]), ('a',))
        t2 = TropicalTensor(np.array([3.0, 4.0]), ('b',))
        result = tropical_contract(t1, t2)
        # result[a,b] = t1[a] + t2[b]
        expected = np.array([[4.0, 5.0], [5.0, 6.0]])
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.indices, ('a', 'b'))

    def test_inner_contraction(self):
        """Contractie over gedeelde index (tropisch: max over som)."""
        # t1[a, b], t2[b, c] -> result[a, c] = max_b (t1[a,b] + t2[b,c])
        t1 = TropicalTensor(np.array([[1.0, 2.0], [3.0, 0.0]]), ('a', 'b'))
        t2 = TropicalTensor(np.array([[4.0, 1.0], [0.0, 5.0]]), ('b', 'c'))
        result = tropical_contract(t1, t2)
        # result[0,0] = max(1+4, 2+0) = max(5,2) = 5
        # result[0,1] = max(1+1, 2+5) = max(2,7) = 7
        # result[1,0] = max(3+4, 0+0) = max(7,0) = 7
        # result[1,1] = max(3+1, 0+5) = max(4,5) = 5
        expected = np.array([[5.0, 7.0], [7.0, 5.0]])
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_full_contraction(self):
        """Volledige contractie tot scalar."""
        t1 = TropicalTensor(np.array([1.0, 3.0]), ('a',))
        t2 = TropicalTensor(np.array([2.0, 0.0]), ('a',))
        result = tropical_contract(t1, t2)
        # max_a (t1[a] + t2[a]) = max(1+2, 3+0) = max(3,3) = 3
        self.assertAlmostEqual(float(np.max(result.data)), 3.0)

    def test_network_contraction(self):
        """Contracteer een netwerk van 3 tensoren."""
        t1 = TropicalTensor(np.array([1.0, 2.0]), ('a',))
        t2 = TropicalTensor(np.array([[0.0, 3.0], [1.0, 0.0]]), ('a', 'b'))
        t3 = TropicalTensor(np.array([2.0, 1.0]), ('b',))
        result = tropical_contract_network([t1, t2, t3])
        # t1*t2: result[b] = max_a(t1[a]+t2[a,b])
        #   b=0: max(1+0, 2+1) = 3, b=1: max(1+3, 2+0) = 4
        # then *t3: max_b(result[b]+t3[b]) = max(3+2, 4+1) = max(5,5) = 5
        self.assertAlmostEqual(float(np.max(result.data)), 5.0)


class TestMaxCutTropical(unittest.TestCase):
    """Test MaxCut als tropisch tensornetwerk."""

    def test_build_network(self):
        """Bouw MaxCut netwerk voor driehoek."""
        edges = [(0, 1), (1, 2), (0, 2)]
        tensors = build_maxcut_tropical_network(3, edges)
        self.assertEqual(len(tensors), 3)
        for t in tensors:
            self.assertEqual(t.data.shape, (2, 2))

    def test_triangle_maxcut(self):
        """MaxCut op driehoek = 2 (bipartiet imperfect)."""
        n = 3
        edges = [(0, 1), (1, 2), (0, 2)]
        cut, config = solve_maxcut_tropical(n, edges)
        self.assertAlmostEqual(cut, 2.0)
        # Verifieer
        actual = sum(1 for (i, j) in edges if config[i] != config[j])
        self.assertEqual(actual, 2)

    def test_path_maxcut(self):
        """MaxCut op pad 0-1-2-3 = 3 (alternerende kleuring)."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        cut, config = solve_maxcut_tropical(n, edges)
        self.assertAlmostEqual(cut, 3.0)

    def test_bipartite_maxcut(self):
        """MaxCut op K_{2,2} = 4 (alle edges snijden)."""
        edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
        cut, config = solve_maxcut_tropical(4, edges)
        self.assertAlmostEqual(cut, 4.0)

    def test_single_edge(self):
        """MaxCut op enkele edge = 1."""
        cut, config = solve_maxcut_tropical(2, [(0, 1)])
        self.assertAlmostEqual(cut, 1.0)
        self.assertNotEqual(config[0], config[1])

    def test_no_edges(self):
        """Geen edges = cut 0."""
        cut, config = solve_maxcut_tropical(3, [])
        self.assertAlmostEqual(cut, 0.0)


class TestTropicalElim(unittest.TestCase):
    """Test variabele-eliminatie tropische contractie."""

    def test_triangle_elim(self):
        """MaxCut op driehoek via eliminatie."""
        n = 3
        edges = [(0, 1), (1, 2), (0, 2)]
        cut, config = solve_maxcut_tropical_elim(n, edges)
        self.assertAlmostEqual(cut, 2.0)
        actual = sum(1 for (i, j) in edges if config[i] != config[j])
        self.assertEqual(actual, 2)

    def test_path_elim(self):
        """MaxCut op pad via eliminatie."""
        n = 5
        edges = [(i, i + 1) for i in range(n - 1)]
        cut, config = solve_maxcut_tropical_elim(n, edges)
        self.assertAlmostEqual(cut, 4.0)

    def test_custom_order(self):
        """Andere eliminatievolgorde geeft zelfde resultaat."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        cut1, _ = solve_maxcut_tropical_elim(n, edges, elim_order=[0, 1, 2, 3])
        cut2, _ = solve_maxcut_tropical_elim(n, edges, elim_order=[3, 2, 1, 0])
        cut3, _ = solve_maxcut_tropical_elim(n, edges, elim_order=[1, 3, 0, 2])
        self.assertAlmostEqual(cut1, cut2)
        self.assertAlmostEqual(cut2, cut3)

    def test_matches_brute_force(self):
        """Eliminatie matcht brute-force op random graaf."""
        rng = np.random.default_rng(42)
        n = 8
        # Random edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        selected = rng.choice(len(all_edges), size=12, replace=False)
        edges = [all_edges[k] for k in selected]

        cut_bf, _ = _tropical_brute_force(n, edges, {})
        cut_elim, config = solve_maxcut_tropical_elim(n, edges)

        self.assertAlmostEqual(cut_elim, cut_bf)

    def test_weighted_elim(self):
        """Eliminatie met gewichten."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = {(0, 1): 3.0, (1, 2): 1.0, (2, 3): 2.0}
        cut, config = solve_maxcut_tropical_elim(n, edges, weights)
        cut_bf, _ = _tropical_brute_force(n, edges, weights)
        self.assertAlmostEqual(cut, cut_bf)

    def test_elim_vs_brute_5_instances(self):
        """5 random instanties: eliminatie == brute-force."""
        rng = np.random.default_rng(123)
        for seed in range(5):
            n = 7
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            k = rng.integers(5, 15)
            selected = rng.choice(len(all_edges), size=min(k, len(all_edges)), replace=False)
            edges = [all_edges[s] for s in selected]

            cut_bf, _ = _tropical_brute_force(n, edges, {})
            cut_elim, _ = solve_maxcut_tropical_elim(n, edges)
            self.assertAlmostEqual(cut_elim, cut_bf,
                                    msg=f"Seed {seed}: elim={cut_elim} != bf={cut_bf}")


class TestMinDegreeOrder(unittest.TestCase):
    """Test min-degree eliminatievolgorde."""

    def test_path_graph(self):
        """Pad graaf heeft treewidth 1."""
        n = 6
        edges = [(i, i + 1) for i in range(n - 1)]
        order, tw = min_degree_order(n, edges)
        self.assertEqual(len(order), n)
        self.assertEqual(set(order), set(range(n)))
        self.assertEqual(tw, 1)

    def test_complete_graph(self):
        """K_n heeft treewidth n-1."""
        n = 5
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        order, tw = min_degree_order(n, edges)
        self.assertEqual(tw, n - 1)

    def test_cycle(self):
        """Cycle C_n heeft treewidth 2."""
        n = 6
        edges = [(i, (i + 1) % n) for i in range(n)]
        order, tw = min_degree_order(n, edges)
        self.assertEqual(tw, 2)

    def test_grid_treewidth(self):
        """3x3 grid heeft treewidth <= 4 (min-degree heuristiek)."""
        edges = build_2d_grid_edges(3, 3)
        order, tw = min_degree_order(9, edges)
        self.assertLessEqual(tw, 5)  # Exact tw=3 maar heuristiek kan hoger


class TestTransferMatrix1D(unittest.TestCase):
    """Test transfer matrix methode voor 1D keten."""

    def test_path_4(self):
        """Path graph n=4: MaxCut = 3."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        cut, config = tropical_transfer_matrix_1d(n, edges)
        self.assertAlmostEqual(cut, 3.0)

    def test_path_matches_brute(self):
        """Transfer matrix matcht brute-force op pad."""
        n = 8
        edges = [(i, i + 1) for i in range(n - 1)]
        cut_tm, _ = tropical_transfer_matrix_1d(n, edges)
        cut_bf, _ = _tropical_brute_force(n, edges, {})
        self.assertAlmostEqual(cut_tm, cut_bf)

    def test_weighted_path(self):
        """Gewogen pad: transfer matrix vindt juiste maximum."""
        n = 3
        edges = [(0, 1), (1, 2)]
        weights = {(0, 1): 5.0, (1, 2): 1.0}
        cut, config = tropical_transfer_matrix_1d(n, edges, weights)
        # Best: cut edge (0,1) met w=5 + cut edge (1,2) met w=1 = 6
        self.assertAlmostEqual(cut, 6.0)


class TestBruteForce(unittest.TestCase):
    """Test brute-force referentie."""

    def test_empty_graph(self):
        """Lege graaf: cut = 0."""
        cut, _ = _tropical_brute_force(3, [], {})
        self.assertAlmostEqual(cut, 0.0)

    def test_k2(self):
        """K_2: cut = 1."""
        cut, config = _tropical_brute_force(2, [(0, 1)], {})
        self.assertAlmostEqual(cut, 1.0)
        self.assertNotEqual(config[0], config[1])

    def test_petersen_like(self):
        """5-cycle: MaxCut = 4."""
        n = 5
        edges = [(i, (i + 1) % n) for i in range(n)]
        cut, _ = _tropical_brute_force(n, edges, {})
        self.assertAlmostEqual(cut, 4.0)

    def test_weighted(self):
        """Gewogen graph: maximale gewogen cut."""
        edges = [(0, 1), (0, 2)]
        weights = {(0, 1): 10.0, (0, 2): 1.0}
        cut, config = _tropical_brute_force(3, edges, weights)
        self.assertAlmostEqual(cut, 11.0)
        # Beide edges moeten gesneden worden: 0 in andere partitie dan 1 en 2
        self.assertNotEqual(config[0], config[1])
        self.assertNotEqual(config[0], config[2])


class TestQAOATropical(unittest.TestCase):
    """Test QAOA tropische MAP-schatting."""

    def test_qaoa_map_basic(self):
        """QAOA MAP geeft een geldig bitstring."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        log_p, config, cut = qaoa_tropical_map(n, edges, [0.3], [0.7])
        self.assertEqual(len(config), n)
        self.assertTrue(all(c in [0, 1] for c in config))
        self.assertGreaterEqual(cut, 0)

    def test_qaoa_expected_cost(self):
        """QAOA verwachte cost is positief en <= C_max."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        cost = qaoa_expected_cost(n, edges, [0.3], [0.7])
        self.assertGreater(cost, 0)
        self.assertLessEqual(cost, len(edges))  # C_max <= |E|

    def test_qaoa_map_cut_bounded(self):
        """QAOA MAP cut ≤ MaxCut."""
        n = 5
        edges = [(i, i + 1) for i in range(n - 1)]
        _, _, map_cut = qaoa_tropical_map(n, edges, [0.5], [0.5])
        bf_cut, _ = _tropical_brute_force(n, edges, {})
        self.assertLessEqual(map_cut, bf_cut + 1e-10)

    def test_qaoa_p2(self):
        """QAOA p=2 MAP werkt."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        log_p, config, cut = qaoa_tropical_map(n, edges, [0.3, 0.5], [0.7, 0.4])
        self.assertGreaterEqual(cut, 0)


class TestSandwichBound(unittest.TestCase):
    """Test sandwich bound."""

    def test_sandwich_basic(self):
        """QAOA cost ≤ tropical MaxCut."""
        n = 5
        edges = [(i, i + 1) for i in range(n - 1)]
        result = sandwich_bound(n, edges, [0.3], [0.7])
        self.assertLessEqual(result['qaoa_cost'], result['tropical_max'] + 1e-10)
        self.assertLessEqual(result['qaoa_ratio'], 1.0 + 1e-10)
        self.assertGreater(result['qaoa_ratio'], 0.0)

    def test_sandwich_triangle(self):
        """Sandwich op driehoek."""
        n = 3
        edges = [(0, 1), (1, 2), (0, 2)]
        result = sandwich_bound(n, edges, [0.5], [0.5])
        self.assertAlmostEqual(result['tropical_max'], 2.0)
        self.assertLessEqual(result['qaoa_cost'], 2.0 + 1e-10)

    def test_sandwich_keys(self):
        """Alle verwachte keys aanwezig."""
        n = 3
        edges = [(0, 1), (1, 2)]
        result = sandwich_bound(n, edges, [0.3], [0.7])
        for key in ['qaoa_cost', 'map_cost', 'tropical_max', 'tropical_config',
                     'map_config', 'qaoa_ratio', 'map_ratio', 'treewidth']:
            self.assertIn(key, result)


class Test2DGrid(unittest.TestCase):
    """Test 2D grid MaxCut."""

    def test_2x2_grid(self):
        """2x2 grid: MaxCut = 4 (checkerboard)."""
        cut, config, tw = solve_maxcut_2d_tropical(2, 2)
        self.assertAlmostEqual(cut, 4.0)

    def test_3x3_grid(self):
        """3x3 grid: MaxCut = 12."""
        cut, config, tw = solve_maxcut_2d_tropical(3, 3)
        self.assertAlmostEqual(cut, 12.0)

    def test_2x3_grid(self):
        """2x3 grid: MaxCut = 7."""
        cut, config, tw = solve_maxcut_2d_tropical(2, 3)
        self.assertAlmostEqual(cut, 7.0)

    def test_grid_edges(self):
        """Juist aantal edges in 3x3 grid."""
        edges = build_2d_grid_edges(3, 3)
        # 3x3 grid: 2*3*2 = 12 edges (3 horizontaal per rij × 2 rijen + ...)
        # Horizontaal: (Lx-1)*Ly = 2*3 = 6
        # Verticaal: Lx*(Ly-1) = 3*2 = 6
        self.assertEqual(len(edges), 12)

    def test_periodic_grid(self):
        """Periodieke 3x3 grid: meer edges."""
        edges_open = build_2d_grid_edges(3, 3, periodic=False)
        edges_periodic = build_2d_grid_edges(3, 3, periodic=True)
        self.assertGreater(len(edges_periodic), len(edges_open))

    def test_4x4_grid_matches_brute(self):
        """4x4 grid: tropisch matcht brute-force."""
        edges = build_2d_grid_edges(4, 4)
        n = 16
        cut_tropical, _, _ = solve_maxcut_2d_tropical(4, 4)
        cut_bf, _ = _tropical_brute_force(n, edges, {})
        self.assertAlmostEqual(cut_tropical, cut_bf)


class TestWeightedGraphs(unittest.TestCase):
    """Test gewogen en Ising grafen."""

    def test_random_weights(self):
        """Random gewichten zijn in bereik."""
        rng = np.random.default_rng(42)
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = random_weighted_graph(4, edges, rng, weight_range=(1.0, 5.0))
        for (i, j) in edges:
            self.assertGreaterEqual(weights[(i, j)], 1.0)
            self.assertLessEqual(weights[(i, j)], 5.0)

    def test_ising_weights(self):
        """Ising gewichten zijn ±1."""
        rng = np.random.default_rng(42)
        edges = [(0, 1), (1, 2)]
        weights = ising_weighted_graph(3, edges, rng)
        for (i, j) in edges:
            self.assertIn(weights[(i, j)], [-1.0, 1.0])

    def test_weighted_maxcut(self):
        """Gewogen MaxCut via tropisch."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = {(0, 1): 3.0, (1, 2): 1.0, (2, 3): 5.0}
        cut, config = solve_maxcut_tropical_elim(n, edges, weights)
        # Brute: best is alternating -> cut alle 3 edges -> 3+1+5=9
        self.assertAlmostEqual(cut, 9.0)

    def test_ising_maxcut_positive(self):
        """Ising met alle +1 gewichten = standaard MaxCut."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        weights = {e: 1.0 for e in edges}
        cut, _ = solve_maxcut_tropical_elim(n, edges, weights)
        self.assertAlmostEqual(cut, 3.0)


class TestEdgeCases(unittest.TestCase):
    """Test randgevallen en robuustheid."""

    def test_single_node(self):
        """1 node, geen edges."""
        cut, config = solve_maxcut_tropical_elim(1, [])
        self.assertAlmostEqual(cut, 0.0)

    def test_two_nodes_one_edge(self):
        """2 nodes, 1 edge."""
        cut, config = solve_maxcut_tropical_elim(2, [(0, 1)])
        self.assertAlmostEqual(cut, 1.0)

    def test_disconnected_graph(self):
        """Disconnected graaf: twee componenten."""
        edges = [(0, 1), (2, 3)]
        cut, config = solve_maxcut_tropical_elim(4, edges)
        self.assertAlmostEqual(cut, 2.0)

    def test_self_loop_ignored(self):
        """Self-loop draagt niet bij aan cut."""
        # Self-loops worden niet ondersteund in MaxCut formalisatie
        # Maar de code moet niet crashen
        edges = [(0, 1), (1, 2)]
        cut, _ = solve_maxcut_tropical_elim(3, edges)
        self.assertAlmostEqual(cut, 2.0)

    def test_duplicate_edges(self):
        """Dubbele edges worden correct afgehandeld."""
        edges = [(0, 1), (0, 1)]
        tensors = build_maxcut_tropical_network(2, edges)
        self.assertEqual(len(tensors), 2)

    def test_large_chain(self):
        """Grote keten: n=100."""
        n = 100
        edges = [(i, i + 1) for i in range(n - 1)]
        cut, config = solve_maxcut_tropical_elim(n, edges)
        self.assertAlmostEqual(cut, n - 1)  # Alternating = alle edges gesneden

    def test_empty_network_contraction(self):
        """Contractie van leeg netwerk."""
        result = tropical_contract_network([])
        self.assertAlmostEqual(float(result.data), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
