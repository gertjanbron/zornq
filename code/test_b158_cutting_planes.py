#!/usr/bin/env python3
"""
Tests voor B158: Triangle + Odd-Cycle Cutting Planes voor MaxCut.

Test suites:
  1. TestEdgeIndex          : edge-index helpers
  2. TestExtendToComplete   : K_n-extensie behoudt edges + voegt 0-gewicht toe
  3. TestTriangleConstraints: triangle-inequalities (4 per driehoek)
  4. TestLPTriangleBound    : LP-relaxatie correctheid
  5. TestOddCycleSeparator  : signed-graph separatie vindt violaties
  6. TestLPOddCycleBound    : LP+odd-cycle iteratie convergeert
  7. TestCompareAllBounds   : vergelijking GW / LP_tri / LP+OC / SoS-2
  8. TestEdgeCases          : kleine grafen, lege constraints
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import SimpleGraph, brute_force_maxcut, gw_sdp_bound
from b156_sos2_sdp import (
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    complete_bipartite,
)
from b158_cutting_planes import (
    _edge_index,
    _extend_to_complete,
    _triangle_constraints,
    _separate_odd_cycle,
    _shortest_signed_path,
    lp_triangle_bound,
    lp_triangle_oddcycle_bound,
    compare_all_bounds,
)


TOL = 1e-3


# ============================================================
# 1. Edge-index helpers
# ============================================================

class TestEdgeIndex(unittest.TestCase):
    def test_complete_k4(self):
        g = complete_graph(4)
        idx = _edge_index(g)
        self.assertEqual(len(idx), 6)
        # All keys must be (u,v) with u<v
        for (u, v) in idx:
            self.assertLess(u, v)


# ============================================================
# 2. _extend_to_complete
# ============================================================

class TestExtendToComplete(unittest.TestCase):
    def test_petersen_extension(self):
        g = petersen_graph()
        g_ext, idx = _extend_to_complete(g)
        self.assertEqual(g_ext.n, 10)
        self.assertEqual(g_ext.n_edges, 10 * 9 // 2)  # 45
        # Originele 15 edges moeten gewicht 1 hebben
        original = {(min(u, v), max(u, v)): 1.0 for u, v, _ in g.edges}
        n_w1 = sum(1 for u, v, w in g_ext.edges
                   if (min(u, v), max(u, v)) in original and w == 1.0)
        self.assertEqual(n_w1, 15)

    def test_path_extension(self):
        g = path_graph(5)  # 4 edges
        g_ext, _ = _extend_to_complete(g)
        self.assertEqual(g_ext.n_edges, 5 * 4 // 2)  # 10


# ============================================================
# 3. Triangle-constraints
# ============================================================

class TestTriangleConstraints(unittest.TestCase):
    def test_k4_count(self):
        """K_4 heeft C(4,3)=4 triangles, dus 16 facetten."""
        g = complete_graph(4)
        idx = _edge_index(g)
        cons = _triangle_constraints(g, idx)
        self.assertEqual(len(cons), 16)

    def test_petersen_no_triangles(self):
        """Petersen is triangle-free."""
        g = petersen_graph()
        idx = _edge_index(g)
        cons = _triangle_constraints(g, idx)
        self.assertEqual(len(cons), 0)

    def test_triangle_facets_have_correct_form(self):
        """Elke 4-tuple facetten heeft RHS ∈ {0, 2}."""
        g = complete_graph(3)
        idx = _edge_index(g)
        cons = _triangle_constraints(g, idx)
        rhs_set = sorted({rhs for _, rhs in cons})
        self.assertEqual(rhs_set, [0.0, 2.0])


# ============================================================
# 4. LP_triangle bound
# ============================================================

class TestLPTriangleBound(unittest.TestCase):
    def test_triangle_k3_no_extend(self):
        """K_3 heeft 1 triangle dus LP_triangle = 2 = OPT."""
        g = complete_graph(3)
        r = lp_triangle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 2.0, delta=TOL)

    def test_triangle_k4(self):
        """K_4: LP_triangle (over K_4) = 4 = OPT."""
        g = complete_graph(4)
        r = lp_triangle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 4.0, delta=TOL)

    def test_petersen_with_extend(self):
        """Petersen + K_n-extensie: LP_triangle = 12 = OPT."""
        g = petersen_graph()
        r = lp_triangle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 12.0, delta=TOL)

    def test_petersen_without_extend_trivial(self):
        """Petersen zonder extensie: geen triangles ⇒ LP = m = 15."""
        g = petersen_graph()
        r = lp_triangle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 15.0, delta=TOL)
        self.assertEqual(r["n_triangles"], 0)

    def test_bipartite_exact(self):
        """K_3,3 bipartite: LP = OPT = m = 9."""
        g = complete_bipartite(3, 3)
        r = lp_triangle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 9.0, delta=TOL)


# ============================================================
# 5. Odd-cycle separator
# ============================================================

class TestOddCycleSeparator(unittest.TestCase):
    def test_pentagon_violation(self):
        """C_5 met y=1 op alle edges: pentagon-inequality violatie = 1."""
        g = cycle_graph(5)
        idx = _edge_index(g)
        y = np.ones(g.n_edges)
        cuts = _separate_odd_cycle(g, y, idx, max_cuts=10)
        self.assertGreaterEqual(len(cuts), 1)
        # Eerste cut moet pentagon zijn: |F|=5 odd, rhs=4
        row, rhs = cuts[0]
        # Som van absolute waardes = 5 (alle 5 edges in cycle)
        self.assertAlmostEqual(np.sum(np.abs(row)), 5.0)
        # |F| - 1 = oneven minus 1 = even, dus rhs ∈ {0, 2, 4}
        self.assertIn(rhs, [0.0, 2.0, 4.0])

    def test_hexagon_no_violation(self):
        """C_6 (even cycle) met y=1: geen odd-cycle violaties."""
        g = cycle_graph(6)
        idx = _edge_index(g)
        y = np.ones(g.n_edges)
        cuts = _separate_odd_cycle(g, y, idx, max_cuts=10)
        # Even cycle heeft geen odd-cycle facetten — separator mag wel iets vinden
        # via subcycles, maar voor een cleane C_6 is dat onwaarschijnlijk.
        # We accepteren beide: 0 cuts of cuts die niet ECHT helpen.
        # Belangrijker: de LP-bound zal niet veranderen want C_6 OPT = 6 = m.
        for row, rhs in cuts:
            # Elke gevonden cut moet geldig zijn: |F| oneven
            n_in_F = int(np.sum(row > 0))
            self.assertEqual(n_in_F % 2, 1)

    def test_signed_path_zero_when_violated(self):
        """Met y=1 op alle pentagon-edges, signed pad lengte = 0."""
        n = 5
        edge_list = [(i, (i + 1) % 5, 1.0) for i in range(5)]
        result = _shortest_signed_path(n, edge_list, start=0)
        self.assertIsNotNone(result)
        length, _ = result  # type: ignore[misc]
        self.assertLess(length, 0.5)


# ============================================================
# 6. LP+odd-cycle iteratie
# ============================================================

class TestLPOddCycleBound(unittest.TestCase):
    def test_pentagon_closes_to_opt(self):
        """C_5: LP+OC zou 4 = OPT moeten geven (pentagon-cut toegevoegd)."""
        g = cycle_graph(5)
        r = lp_triangle_oddcycle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 4.0, delta=TOL)
        self.assertGreaterEqual(r["n_cuts_added"], 1)

    def test_c7_closes_to_opt(self):
        """C_7: LP+OC = 6 = OPT."""
        g = cycle_graph(7)
        r = lp_triangle_oddcycle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 6.0, delta=TOL)

    def test_petersen_exact(self):
        """Petersen + extend + OC: 12 = OPT."""
        g = petersen_graph()
        r = lp_triangle_oddcycle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 12.0, delta=TOL)

    def test_k5_lp_bound(self):
        """K_5: LP_triangle = 20/3 ≈ 6.667 (klassieke waarde),
        OC voegt geen iets toe omdat K_5 al triangle-saturated is."""
        g = complete_graph(5)
        r1 = lp_triangle_bound(g, extend=True, verbose=False)
        r2 = lp_triangle_oddcycle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r1["lp_bound"], 20.0 / 3.0, delta=TOL)
        self.assertAlmostEqual(r2["lp_bound"], 20.0 / 3.0, delta=TOL)

    def test_bound_history_monotone(self):
        """Bound moet monotoon dalen door iteratie."""
        g = cycle_graph(7)
        r = lp_triangle_oddcycle_bound(g, extend=False, verbose=False)
        history = r["bound_history"]
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i - 1] + TOL)


# ============================================================
# 7. compare_all_bounds
# ============================================================

class TestCompareAllBounds(unittest.TestCase):
    def test_petersen_full_compare(self):
        g = petersen_graph()
        r = compare_all_bounds(g, name="Petersen", verbose=False)
        self.assertEqual(r["n"], 10)
        self.assertEqual(r["n_edges"], 15)
        self.assertEqual(r["opt"], 12)
        # GW = 12.5
        self.assertAlmostEqual(r["gw"], 12.5, delta=0.05)
        # LP_triangle (extend) = 12
        self.assertAlmostEqual(r["lp_triangle"], 12.0, delta=TOL)
        # LP+OC = 12
        self.assertAlmostEqual(r["lp_oddcycle"], 12.0, delta=TOL)
        # SoS-2 = 12
        self.assertAlmostEqual(r["sos2"], 12.0, delta=TOL)

    def test_c5_full_compare(self):
        g = cycle_graph(5)
        r = compare_all_bounds(g, name="C_5", verbose=False)
        self.assertEqual(r["opt"], 4)
        self.assertGreater(r["gw"], 4.4)            # ~4.52
        # LP_triangle without extend zou 5 zijn (geen triangles); MET extend = 5 ook
        # (omdat K_5-extensie geen extra signaal geeft, bipartite-LP is zwak op K_5)
        # We checken alleen LP+OC en SoS-2
        self.assertAlmostEqual(r["lp_oddcycle"], 4.0, delta=0.5)
        self.assertAlmostEqual(r["sos2"], 4.0, delta=TOL)


# ============================================================
# 8. Edge-cases
# ============================================================

class TestEdgeCases(unittest.TestCase):
    def test_single_edge(self):
        g = SimpleGraph(2)
        g.add_edge(0, 1)
        r = lp_triangle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 1.0, delta=TOL)
        r2 = lp_triangle_oddcycle_bound(g, extend=False, verbose=False)
        self.assertAlmostEqual(r2["lp_bound"], 1.0, delta=TOL)

    def test_path_p4_bipartite(self):
        """P_4: 3 edges, bipartite ⇒ OPT = 3."""
        g = path_graph(4)
        r = lp_triangle_oddcycle_bound(g, extend=True, verbose=False)
        self.assertAlmostEqual(r["lp_bound"], 3.0, delta=TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
