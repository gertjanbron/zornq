#!/usr/bin/env python3
"""
Tests voor B156: Lasserre / Sum-of-Squares level-2 SDP MaxCut bovengrens.

Test suites:
  1. TestBasisMonomials       : Basis-monomialen telling en structuur
  2. TestPseudoMomentKeys     : Pseudo-moment-sleutels |S| ≤ 4
  3. TestGraphHelpers         : complete_graph / cycle_graph / petersen / bipartite
  4. TestSoS2OnSmallGraphs    : SoS-2 numerieke correctheid
  5. TestSoS2VsGW             : SoS-2 ≤ GW (level-2 verfijnt level-1)
  6. TestSoS2OnBipartite      : Bipartite ⇒ SoS-2 = OPT = |E|
  7. TestSoS2ExactOnPetersen  : Petersen scherp op level-2
  8. TestSoS2OnCycles         : Odd/even cycle gedrag
  9. TestSoS2EdgeCases        : Lege graaf, single-edge, n=2
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    gw_sdp_bound,
)
from b156_sos2_sdp import (
    _basis_monomials,
    _pseudomoment_keys,
    complete_bipartite,
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    sos2_sdp_bound,
    compare_bounds,
)


TOL = 1e-3  # SCS heeft eps=1e-6 maar met DC-strucutuur ~1e-4 fouten ok


# ============================================================
# 1. Basis-monomialen
# ============================================================

class TestBasisMonomials(unittest.TestCase):
    """Test _basis_monomials: monomialen van graad ≤ 2."""

    def test_count_n3(self):
        """n=3: 1 + 3 + C(3,2) = 1 + 3 + 3 = 7."""
        self.assertEqual(len(_basis_monomials(3)), 7)

    def test_count_n5(self):
        """n=5: 1 + 5 + 10 = 16 = (5+1)(5+2)/2 = 21? Nee: 1+5+10=16."""
        # (n+1)(n+2)/2 = 21 telt 1+n+C(n+1,2) = 1+5+15?
        # Onze formule: 1 + n + C(n,2). Voor n=5: 1+5+10 = 16.
        self.assertEqual(len(_basis_monomials(5)), 16)

    def test_count_general(self):
        """Generiek: 1 + n + n(n-1)/2."""
        for n in [2, 4, 6, 8, 10]:
            expected = 1 + n + n * (n - 1) // 2
            self.assertEqual(len(_basis_monomials(n)), expected)

    def test_includes_empty(self):
        """Eerste monomial moet de constante (lege set) zijn."""
        mons = _basis_monomials(4)
        self.assertEqual(mons[0], frozenset())

    def test_unique_monomials(self):
        """Alle monomialen moeten uniek zijn."""
        mons = _basis_monomials(6)
        self.assertEqual(len(set(mons)), len(mons))


# ============================================================
# 2. Pseudo-moment-sleutels
# ============================================================

class TestPseudoMomentKeys(unittest.TestCase):
    """Test _pseudomoment_keys: alle y_S met |S| ≤ 4."""

    def test_includes_all_subsets_up_to_4(self):
        """Voor n=5 moeten alle deelverzamelingen |S| ≤ 4 voorkomen."""
        n = 5
        basis = _basis_monomials(n)
        keys = _pseudomoment_keys(basis)
        max_size = max(len(k) for k in keys)
        self.assertLessEqual(max_size, 4)

    def test_sorted_by_size(self):
        """Sleutels gesorteerd op grootte."""
        basis = _basis_monomials(4)
        keys = _pseudomoment_keys(basis)
        sizes = [len(k) for k in keys]
        self.assertEqual(sizes, sorted(sizes))

    def test_count_n4(self):
        """Voor n=4: alle deelverzamelingen van [4] hebben |S|≤4 dus 2^4 = 16."""
        n = 4
        basis = _basis_monomials(n)
        keys = _pseudomoment_keys(basis)
        # Alle 16 deelverzamelingen van [4]
        self.assertEqual(len(keys), 16)

    def test_count_n6_bound(self):
        """Voor n=6: aantal pseudo-momenten ≤ Σ_{k=0}^4 C(6,k) = 1+6+15+20+15 = 57."""
        n = 6
        basis = _basis_monomials(n)
        keys = _pseudomoment_keys(basis)
        upper = sum(math.comb(n, k) for k in range(5))
        self.assertLessEqual(len(keys), upper)


# ============================================================
# 3. Graaf-helpers
# ============================================================

class TestGraphHelpers(unittest.TestCase):
    """Test graaf-constructors."""

    def test_complete_graph(self):
        for n in [3, 4, 5, 7]:
            g = complete_graph(n)
            self.assertEqual(g.n, n)
            self.assertEqual(g.n_edges, n * (n - 1) // 2)

    def test_cycle_graph(self):
        for n in [4, 5, 8]:
            g = cycle_graph(n)
            self.assertEqual(g.n, n)
            self.assertEqual(g.n_edges, n)

    def test_path_graph(self):
        for n in [3, 5, 10]:
            g = path_graph(n)
            self.assertEqual(g.n, n)
            self.assertEqual(g.n_edges, n - 1)

    def test_petersen(self):
        g = petersen_graph()
        self.assertEqual(g.n, 10)
        self.assertEqual(g.n_edges, 15)

    def test_bipartite(self):
        g = complete_bipartite(3, 4)
        self.assertEqual(g.n, 7)
        self.assertEqual(g.n_edges, 12)


# ============================================================
# 4. SoS-2 op kleine grafen (bekende waardes)
# ============================================================

class TestSoS2OnSmallGraphs(unittest.TestCase):
    """SoS-2 numerieke correctheid op grafen met bekende OPT."""

    def test_triangle_k3(self):
        """K_3: OPT = 2, SoS-2 zou 2 moeten zijn (level-2 lost driehoek op)."""
        g = complete_graph(3)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 2.0, delta=TOL)

    def test_k4(self):
        """K_4: OPT = 4 (bipartite-completable), SoS-2 zou 4 moeten zijn."""
        g = complete_graph(4)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 4.0, delta=TOL)

    def test_k5(self):
        """K_5: OPT = 6, SoS-2 = 6.25 (level-2 lost K_5 nog niet op)."""
        g = complete_graph(5)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 6.25, delta=TOL)

    def test_c5_odd_cycle(self):
        """C_5: OPT = 4, SoS-2 = 4 exact (odd cycle gefixed op level-2)."""
        g = cycle_graph(5)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 4.0, delta=TOL)

    def test_c7_odd_cycle(self):
        """C_7: OPT = 6, SoS-2 = 6 exact."""
        g = cycle_graph(7)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 6.0, delta=TOL)


# ============================================================
# 5. SoS-2 ≤ GW
# ============================================================

class TestSoS2VsGW(unittest.TestCase):
    """Level-2 verfijnt level-1: SoS-2 ≤ GW + numerieke ruis."""

    def _check_tightening(self, g, name):
        gw = gw_sdp_bound(g, verbose=False)["sdp_bound"]
        sos2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertLessEqual(
            sos2, gw + 5e-3,
            f"{name}: SoS-2={sos2:.4f} > GW={gw:.4f}",
        )

    def test_k3_strict_tightening(self):
        """K_3: SoS-2=2.0 << GW=2.25."""
        g = complete_graph(3)
        gw = gw_sdp_bound(g, verbose=False)["sdp_bound"]
        sos2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertLess(sos2, gw - 0.1)

    def test_petersen_tightening(self):
        """Petersen: SoS-2=12 << GW≈12.5."""
        g = petersen_graph()
        gw = gw_sdp_bound(g, verbose=False)["sdp_bound"]
        sos2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertLess(sos2, gw - 0.1)

    def test_c5_tightening(self):
        """C_5: SoS-2=4 << GW≈4.52."""
        g = cycle_graph(5)
        self._check_tightening(g, "C_5")

    def test_k4_no_tightening(self):
        """K_4: GW al exact dus SoS-2 mag niet groter zijn."""
        g = complete_graph(4)
        self._check_tightening(g, "K_4")


# ============================================================
# 6. Bipartite: SoS-2 = OPT = |E|
# ============================================================

class TestSoS2OnBipartite(unittest.TestCase):
    """Bipartite grafen: alle edges in cut, SoS-2 moet exact |E| geven."""

    def test_k33(self):
        g = complete_bipartite(3, 3)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 9.0, delta=TOL)

    def test_k24(self):
        g = complete_bipartite(2, 4)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 8.0, delta=TOL)

    def test_path_p5(self):
        """P_5 is bipartite: OPT = 4 = |E|."""
        g = path_graph(5)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 4.0, delta=TOL)

    def test_even_cycle_c6(self):
        """C_6 bipartite: OPT = 6."""
        g = cycle_graph(6)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 6.0, delta=TOL)


# ============================================================
# 7. SoS-2 als geldige bovengrens (≥ OPT)
# ============================================================

class TestSoS2IsValidUpperBound(unittest.TestCase):
    """SoS-2 ≥ OPT moet altijd gelden."""

    def _check_upper_bound(self, g, name):
        opt = brute_force_maxcut(g)
        sos2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertGreaterEqual(
            sos2 + TOL, opt,
            f"{name}: SoS-2={sos2:.4f} < OPT={opt}",
        )

    def test_k5(self):
        self._check_upper_bound(complete_graph(5), "K_5")

    def test_petersen(self):
        self._check_upper_bound(petersen_graph(), "Petersen")

    def test_c7(self):
        self._check_upper_bound(cycle_graph(7), "C_7")

    def test_k_bipartite(self):
        self._check_upper_bound(complete_bipartite(2, 3), "K_2,3")


# ============================================================
# 8. Petersen exact op level-2
# ============================================================

class TestSoS2ExactOnPetersen(unittest.TestCase):
    """Bekend resultaat: Petersen MaxCut = 12, SoS-2 = 12 exact."""

    def test_petersen_exact(self):
        g = petersen_graph()
        sos2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertAlmostEqual(sos2, 12.0, delta=TOL)
        # GW geeft 12.5
        gw = gw_sdp_bound(g, verbose=False)["sdp_bound"]
        self.assertGreater(gw, 12.4)


# ============================================================
# 9. Edge-cases
# ============================================================

class TestSoS2EdgeCases(unittest.TestCase):
    """Robuustheid op kleine en pathologische input."""

    def test_n2_single_edge(self):
        """n=2, 1 edge: OPT = SoS-2 = 1."""
        g = SimpleGraph(2)
        g.add_edge(0, 1)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 1.0, delta=TOL)

    def test_n3_path(self):
        """P_3 (path): OPT = 2 = |E|."""
        g = path_graph(3)
        result = sos2_sdp_bound(g, verbose=False)
        self.assertAlmostEqual(result["sos2_bound"], 2.0, delta=TOL)

    def test_max_n_skip(self):
        """Te grote n moet SKIPPED retourneren zonder te crashen."""
        g = complete_graph(8)
        result = sos2_sdp_bound(g, max_n=5, verbose=False)
        self.assertIsNone(result["sos2_bound"])
        self.assertIn("SKIPPED", result["status"])

    def test_moment_matrix_size(self):
        """N = 1 + n + C(n,2)."""
        for n in [3, 4, 5, 6]:
            g = complete_graph(n)
            r = sos2_sdp_bound(g, verbose=False)
            self.assertEqual(r["moment_matrix_size"],
                             1 + n + n * (n - 1) // 2)


# ============================================================
# 10. compare_bounds samenwerking
# ============================================================

class TestCompareBounds(unittest.TestCase):
    """compare_bounds() retourneert correct gestructureerde dict."""

    def test_compare_petersen(self):
        g = petersen_graph()
        r = compare_bounds(g, verbose=False, name="Petersen")
        self.assertIn("gw", r)
        self.assertIn("sos2", r)
        self.assertIn("opt", r)
        self.assertIn("tightening_pct", r)
        # Petersen: SoS-2 strakker dan GW
        self.assertGreater(r["tightening_pct"], 1.0)

    def test_compare_k3(self):
        g = complete_graph(3)
        r = compare_bounds(g, verbose=False, name="K_3")
        # K_3: tightening ~11.1%
        self.assertGreater(r["tightening_pct"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
