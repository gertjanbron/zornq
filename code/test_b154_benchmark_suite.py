#!/usr/bin/env python3
"""
Tests voor B154: BiqMac + DIMACS Benchmark Suite.

Test suites:
  A. TestRudyParser          : rudy file/text-format parser correctheid
  B. TestRudyRoundTrip       : write_rudy → parse_rudy reproduceert graaf
  C. TestBiqMacGenerators    : synthetic generators geven juiste structuur
  D. TestBiqMacSpec          : spec-parser (spinglass2d_5, pm1s_100, ...)
  E. TestBiqMacBKS           : BKS-DB is consistent (n, m velden redelijk)
  F. TestDimacsParser        : DIMACS p/c/e-lines correct geparseerd
  G. TestDimacsFixtures      : ingebouwde fixtures correct leesbaar
  H. TestDimacsChromaticDB   : chromatic-DB consistent
  I. TestMaxKCutBounds       : UB + Frieze-Jerrum LB sanity checks
  J. TestCrossFormat         : rudy + DIMACS kunnen dezelfde graaf weergeven
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph

from b154_biqmac_loader import (
    parse_rudy,
    write_rudy,
    biqmac_spinglass_2d,
    biqmac_spinglass_3d,
    biqmac_torus_2d,
    biqmac_pm1s,
    biqmac_pm1d,
    biqmac_w01,
    biqmac_g05,
    generate_from_spec,
    BIQMAC_GENERATORS,
    BIQMAC_BKS,
)
from b154_dimacs_loader import (
    parse_dimacs,
    write_dimacs,
    load_fixture,
    DIMACS_FIXTURES,
    DIMACS_CHROMATIC,
    max_k_cut_upper_bound,
    frieze_jerrum_lower_bound,
    dimacs_to_maxcut,
)


TOL = 1e-9


# ============================================================
# A. Rudy-format parser
# ============================================================

class TestRudyParser(unittest.TestCase):
    def test_minimal_rudy_text(self):
        text = "3 2\n1 2 1.0\n2 3 1.0\n"
        g, n_decl, m_decl = parse_rudy(text, from_text=True)
        self.assertEqual(n_decl, 3)
        self.assertEqual(m_decl, 2)
        self.assertEqual(g.n_nodes, 3)
        self.assertEqual(g.n_edges, 2)

    def test_one_indexed_conversion(self):
        """Rudy 1-indexed → 0-indexed: vertex '1 2' maakt edge (0, 1)."""
        text = "2 1\n1 2 5.5\n"
        g, _, _ = parse_rudy(text, from_text=True)
        edges = list(g.edges())
        self.assertEqual(edges, [(0, 1, 5.5)])

    def test_negative_weights_allowed(self):
        """BiqMac spinglass heeft ±1; negatieve gewichten moeten werken."""
        text = "3 2\n1 2 -1.0\n2 3 1.0\n"
        g, _, _ = parse_rudy(text, from_text=True)
        edges = {(i, j): w for i, j, w in g.edges()}
        self.assertAlmostEqual(edges[(0, 1)], -1.0, delta=TOL)
        self.assertAlmostEqual(edges[(1, 2)], 1.0, delta=TOL)

    def test_no_weight_defaults_to_one(self):
        text = "2 1\n1 2\n"
        g, _, _ = parse_rudy(text, from_text=True)
        _, _, w = next(g.edges())
        self.assertAlmostEqual(w, 1.0, delta=TOL)

    def test_comments_ignored(self):
        text = "# this is a comment\nc another comment\n3 2\n1 2 1.0\n2 3 1.0\n"
        g, _, _ = parse_rudy(text, from_text=True)
        self.assertEqual(g.n_edges, 2)

    def test_self_loop_skipped(self):
        text = "3 3\n1 1 1.0\n1 2 1.0\n2 3 1.0\n"
        g, _, _ = parse_rudy(text, from_text=True)
        self.assertEqual(g.n_edges, 2)

    def test_isolated_nodes_preserved(self):
        """Als N=5 maar alleen edges (1,2): node 3,4,5 moeten bestaan."""
        text = "5 1\n1 2 1.0\n"
        g, _, _ = parse_rudy(text, from_text=True)
        self.assertEqual(g.n_nodes, 5)


# ============================================================
# B. Rudy round-trip
# ============================================================

class TestRudyRoundTrip(unittest.TestCase):
    def test_roundtrip_spinglass(self):
        g1 = biqmac_spinglass_2d(L=4, seed=17)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rud", delete=False) as f:
            path = f.name
        try:
            write_rudy(g1, path)
            g2, _, _ = parse_rudy(path)
            self.assertEqual(g1.n_nodes, g2.n_nodes)
            self.assertEqual(g1.n_edges, g2.n_edges)
            e1 = sorted(g1.edges())
            e2 = sorted(g2.edges())
            for (i1, j1, w1), (i2, j2, w2) in zip(e1, e2):
                self.assertEqual(i1, i2)
                self.assertEqual(j1, j2)
                self.assertAlmostEqual(w1, w2, delta=TOL)
        finally:
            os.unlink(path)

    def test_roundtrip_weighted(self):
        g1 = biqmac_w01(n=10, p=0.5, seed=5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rud", delete=False) as f:
            path = f.name
        try:
            write_rudy(g1, path)
            g2, _, _ = parse_rudy(path)
            self.assertEqual(g1.n_edges, g2.n_edges)
            for (i, j, w1), (i2, j2, w2) in zip(sorted(g1.edges()), sorted(g2.edges())):
                self.assertAlmostEqual(w1, w2, delta=1e-6)
        finally:
            os.unlink(path)


# ============================================================
# C. BiqMac generators — structuur
# ============================================================

class TestBiqMacGenerators(unittest.TestCase):
    def test_spinglass2d_edges(self):
        """L×L 2D spin-glass heeft (L−1)·L + L·(L−1) = 2L(L−1) edges."""
        for L in (3, 5, 7):
            g = biqmac_spinglass_2d(L, seed=0)
            self.assertEqual(g.n_nodes, L * L)
            self.assertEqual(g.n_edges, 2 * L * (L - 1))

    def test_spinglass3d_edges(self):
        """L³ 3D spin-glass: 3·L²·(L−1) edges."""
        L = 3
        g = biqmac_spinglass_3d(L, seed=0)
        self.assertEqual(g.n_nodes, L ** 3)
        self.assertEqual(g.n_edges, 3 * L * L * (L - 1))

    def test_torus2d_edges(self):
        """L×L torus: 2·L² edges (elk vertex 2 unieke uitgaande edges)."""
        L = 4
        g = biqmac_torus_2d(L, seed=0)
        self.assertEqual(g.n_nodes, L * L)
        self.assertEqual(g.n_edges, 2 * L * L)

    def test_pm1_weights_are_plusminus_one(self):
        g = biqmac_spinglass_2d(4, seed=42)
        for _, _, w in g.edges():
            self.assertIn(w, (-1.0, 1.0))

    def test_pm1d_is_complete(self):
        """Dense pm1d op n nodes heeft n(n−1)/2 edges."""
        n = 8
        g = biqmac_pm1d(n, seed=3)
        self.assertEqual(g.n_edges, n * (n - 1) // 2)
        for _, _, w in g.edges():
            self.assertIn(w, (-1.0, 1.0))

    def test_w01_weights_in_range(self):
        g = biqmac_w01(20, p=0.6, seed=9)
        for _, _, w in g.edges():
            self.assertGreaterEqual(w, -1.0 - TOL)
            self.assertLessEqual(w, 1.0 + TOL)

    def test_g05_is_unweighted(self):
        g = biqmac_g05(15, seed=7)
        for _, _, w in g.edges():
            self.assertAlmostEqual(w, 1.0, delta=TOL)

    def test_seed_reproducibility(self):
        g1 = biqmac_spinglass_2d(5, seed=42)
        g2 = biqmac_spinglass_2d(5, seed=42)
        self.assertEqual(list(g1.edges()), list(g2.edges()))


# ============================================================
# D. Spec-parser
# ============================================================

class TestBiqMacSpec(unittest.TestCase):
    def test_spinglass_spec(self):
        g, name = generate_from_spec("spinglass2d_5", seed=0)
        self.assertEqual(g.n_nodes, 25)
        self.assertIn("L5", name)

    def test_torus_spec(self):
        g, name = generate_from_spec("torus2d_4", seed=0)
        self.assertEqual(g.n_nodes, 16)
        self.assertEqual(g.n_edges, 32)

    def test_pm1s_with_p(self):
        g, name = generate_from_spec("pm1s_50_0.3", seed=1)
        self.assertEqual(g.n_nodes, 50)
        self.assertIn("p0.3", name)

    def test_pm1d_spec(self):
        g, _ = generate_from_spec("pm1d_20", seed=2)
        self.assertEqual(g.n_edges, 190)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            generate_from_spec("foo_99", seed=0)


# ============================================================
# E. BKS database integriteit
# ============================================================

class TestBiqMacBKS(unittest.TestCase):
    def test_bks_entries_valid(self):
        for name, (n, m, bks) in BIQMAC_BKS.items():
            self.assertGreater(n, 0, f"n = {n} for {name}")
            self.assertGreaterEqual(m, 0, f"m = {m} for {name}")
            # BKS voor MaxCut altijd ≤ m (ongewogen) of ≤ totaal gewicht,
            # voor ±1-grafen kan BKS tot m gaan
            self.assertLessEqual(abs(bks), m, f"|bks| > m voor {name}")

    def test_has_spinglass_torus_pm1_entries(self):
        fams = set()
        for name in BIQMAC_BKS:
            fams.add(name.split("_")[0])
        self.assertIn("spinglass2d", fams)
        self.assertIn("torus2d", fams)
        self.assertIn("pm1s", fams)
        self.assertIn("pm1d", fams)
        self.assertIn("g05", fams)


# ============================================================
# F. DIMACS parser
# ============================================================

class TestDimacsParser(unittest.TestCase):
    def test_minimal_dimacs(self):
        text = "c comment\np edge 3 2\ne 1 2\ne 2 3\n"
        g, n_decl, m_decl = parse_dimacs(text, from_text=True)
        self.assertEqual(n_decl, 3)
        self.assertEqual(m_decl, 2)
        self.assertEqual(g.n_nodes, 3)
        self.assertEqual(g.n_edges, 2)

    def test_one_indexed(self):
        text = "p edge 2 1\ne 1 2\n"
        g, _, _ = parse_dimacs(text, from_text=True)
        edges = list(g.edges())
        self.assertEqual(edges, [(0, 1, 1.0)])

    def test_weighted_dimacs_extension(self):
        """Onze extensie: 'e i j w' ondersteunt edge-gewicht."""
        text = "p edge 3 2\ne 1 2 3.5\ne 2 3 −\n"  # 2e edge invalid; zou moeten crashen of worden geskipt
        # We testen alleen de eerste regel:
        text = "p edge 3 2\ne 1 2 3.5\ne 2 3 1.0\n"
        g, _, _ = parse_dimacs(text, from_text=True)
        edges = {(i, j): w for i, j, w in g.edges()}
        self.assertAlmostEqual(edges[(0, 1)], 3.5, delta=TOL)

    def test_e_before_p_raises(self):
        text = "e 1 2\np edge 2 1\n"
        with self.assertRaises(ValueError):
            parse_dimacs(text, from_text=True)

    def test_col_alias_works(self):
        """Sommige DIMACS-files gebruiken 'p col' ipv 'p edge'."""
        text = "p col 3 2\ne 1 2\ne 2 3\n"
        g, _, _ = parse_dimacs(text, from_text=True)
        self.assertEqual(g.n_edges, 2)


# ============================================================
# G. DIMACS fixtures
# ============================================================

class TestDimacsFixtures(unittest.TestCase):
    def test_petersen_fixture(self):
        g, n, m = load_fixture("petersen")
        self.assertEqual(g.n_nodes, 10)
        self.assertEqual(g.n_edges, 15)
        # 3-regulier
        for node in g.nodes:
            self.assertEqual(len(g.adj[node]), 3)

    def test_myciel3_fixture(self):
        g, _, _ = load_fixture("myciel3")
        self.assertEqual(g.n_nodes, 11)
        self.assertEqual(g.n_edges, 20)

    def test_k4_fixture(self):
        g, _, _ = load_fixture("k4")
        self.assertEqual(g.n_nodes, 4)
        self.assertEqual(g.n_edges, 6)

    def test_c6_fixture(self):
        g, _, _ = load_fixture("c6")
        self.assertEqual(g.n_nodes, 6)
        self.assertEqual(g.n_edges, 6)

    def test_all_fixtures_parse(self):
        for name in DIMACS_FIXTURES:
            g, n_decl, m_decl = load_fixture(name)
            self.assertEqual(g.n_nodes, n_decl,
                             f"fixture {name}: n mismatch")

    def test_unknown_fixture_raises(self):
        with self.assertRaises(ValueError):
            load_fixture("nonexistent")


# ============================================================
# H. DIMACS chromatic-DB
# ============================================================

class TestDimacsChromaticDB(unittest.TestCase):
    def test_db_entries_valid(self):
        for name, (n, m, chi) in DIMACS_CHROMATIC.items():
            self.assertGreater(n, 0)
            self.assertGreater(m, 0)
            self.assertGreaterEqual(chi, 2, f"{name}: χ < 2")
            # Theoretische bovengrens: χ ≤ max degree + 1 ≤ n
            self.assertLessEqual(chi, n, f"{name}: χ > n")

    def test_petersen_chi_equals_3(self):
        self.assertEqual(DIMACS_CHROMATIC["petersen"][2], 3)

    def test_myciel_family(self):
        """Mycielski: χ(M_k) = k+1."""
        for k, expected_chi in [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]:
            name = f"myciel{k}"
            self.assertIn(name, DIMACS_CHROMATIC)
            self.assertEqual(DIMACS_CHROMATIC[name][2], expected_chi)


# ============================================================
# I. Max-k-Cut bounds
# ============================================================

class TestMaxKCutBounds(unittest.TestCase):
    def test_ub_equals_total_weight(self):
        g, _, _ = load_fixture("petersen")
        ub = max_k_cut_upper_bound(g, k=2)
        self.assertAlmostEqual(ub, g.total_weight(), delta=TOL)

    def test_frieze_jerrum_lb_below_ub(self):
        g, _, _ = load_fixture("k4")
        for k in (2, 3, 4, 5):
            ub = max_k_cut_upper_bound(g, k)
            lb = frieze_jerrum_lower_bound(g, k)
            self.assertLess(lb, ub + TOL, f"k={k}: LB {lb} > UB {ub}")

    def test_alpha_2_is_gw_constant(self):
        """Frieze-Jerrum α_2 ≈ GW-constante 0.87856."""
        g, _, _ = load_fixture("k4")
        lb = frieze_jerrum_lower_bound(g, k=2)
        expected = 0.87856 * g.total_weight()
        self.assertAlmostEqual(lb, expected, delta=1e-3)

    def test_k_goes_infty_lb_to_total(self):
        """Voor grote k: (1 − 1/k) · m → m."""
        g, _, _ = load_fixture("petersen")
        lb_large = frieze_jerrum_lower_bound(g, k=100)
        self.assertGreater(lb_large, 0.98 * g.total_weight())

    def test_dimacs_to_maxcut_identity(self):
        g, _, _ = load_fixture("petersen")
        g2 = dimacs_to_maxcut(g)
        self.assertEqual(g.n_nodes, g2.n_nodes)
        self.assertEqual(g.n_edges, g2.n_edges)


# ============================================================
# J. Cross-format
# ============================================================

class TestCrossFormat(unittest.TestCase):
    def test_petersen_rudy_then_dimacs_roundtrip(self):
        """Petersen → rudy → DIMACS-parse dezelfde graaf."""
        g_dimacs, _, _ = load_fixture("petersen")
        # Schrijf als rudy, lees terug
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rud", delete=False) as f:
            path = f.name
        try:
            write_rudy(g_dimacs, path)
            g_rudy, _, _ = parse_rudy(path)
            self.assertEqual(g_dimacs.n_nodes, g_rudy.n_nodes)
            self.assertEqual(g_dimacs.n_edges, g_rudy.n_edges)
            # Zelfde edge-set
            e_d = set((i, j) for i, j, _ in g_dimacs.edges())
            e_r = set((i, j) for i, j, _ in g_rudy.edges())
            self.assertEqual(e_d, e_r)
        finally:
            os.unlink(path)

    def test_spinglass_dimacs_write_read(self):
        """Schrijf BiqMac-gen spinglass als DIMACS en parse terug
        (met weighted=True voor behoud ±1-gewichten)."""
        g1 = biqmac_spinglass_2d(4, seed=11)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".col", delete=False) as f:
            path = f.name
        try:
            write_dimacs(g1, path, weighted=True)
            g2, _, _ = parse_dimacs(path)
            self.assertEqual(g1.n_nodes, g2.n_nodes)
            self.assertEqual(g1.n_edges, g2.n_edges)
            # Zelfde gewichten
            w1 = {(i, j): w for i, j, w in g1.edges()}
            w2 = {(i, j): w for i, j, w in g2.edges()}
            for key in w1:
                self.assertAlmostEqual(w1[key], w2[key], delta=1e-6)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
