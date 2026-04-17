#!/usr/bin/env python3
"""
Tests voor B159: ILP-Oracle voor MaxCut.

Test suites:
  1. TestHighsOptimality        : HiGHS-pad match brute force op kleine grafen
  2. TestHighsCutValidity       : teruggave-bitstring reproduceert OPT
  3. TestHighsCertification     : certified / gap_abs velden correct gevuld
  4. TestHighsTimeLimit         : time-limit levert feasible + uncertified result
  5. TestHighsEdgeCases         : lege graaf, single edge, disconnected
  6. TestHighsWeighted          : niet-uniforme edge-weights
  7. TestDispatcher             : maxcut_ilp() dispatcher werkt
  8. TestOptionalSolvers        : SCIP/Gurobi gracefully SKIPPED als absent
  9. TestAgainstBounds          : ILP-OPT <= alle upper bounds (GW, SoS-2, LP+OC)
 10. TestSignedInstancesDag8b   : signed-safe formulering regression tests
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    gw_sdp_bound,
    random_3regular,
)
from b156_sos2_sdp import (
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    complete_bipartite,
    sos2_sdp_bound,
)
from b158_cutting_planes import lp_triangle_oddcycle_bound
from b159_ilp_oracle import (
    maxcut_ilp,
    maxcut_ilp_highs,
    maxcut_ilp_scip,
    maxcut_ilp_gurobi,
    evaluate_cut,
    SOLVERS,
)


TOL = 1e-6


# ============================================================
# 1. HiGHS optimality on known graphs
# ============================================================

class TestHighsOptimality(unittest.TestCase):
    def _check(self, g: SimpleGraph, expected: float, name: str = ""):
        res = maxcut_ilp_highs(g)
        self.assertIsNotNone(res["opt_value"], f"{name}: OPT is None")
        self.assertAlmostEqual(res["opt_value"], expected, delta=TOL,
                                msg=f"{name}: got {res['opt_value']}")
        self.assertTrue(res["certified"], f"{name}: not certified")

    def test_k3(self):
        self._check(complete_graph(3), 2.0, "K_3")

    def test_k4(self):
        self._check(complete_graph(4), 4.0, "K_4")

    def test_k5(self):
        self._check(complete_graph(5), 6.0, "K_5")

    def test_c5(self):
        self._check(cycle_graph(5), 4.0, "C_5")

    def test_c7(self):
        self._check(cycle_graph(7), 6.0, "C_7")

    def test_c8_even(self):
        self._check(cycle_graph(8), 8.0, "C_8")

    def test_p8_path(self):
        self._check(path_graph(8), 7.0, "P_8")

    def test_petersen(self):
        self._check(petersen_graph(), 12.0, "Petersen")

    def test_k33_bipartite(self):
        self._check(complete_bipartite(3, 3), 9.0, "K_3,3")

    def test_3reg_n10(self):
        g = random_3regular(10, seed=42)
        bf = brute_force_maxcut(g)
        self._check(g, float(bf), "3-reg n=10")


# ============================================================
# 2. Cut-bitstring validity
# ============================================================

class TestHighsCutValidity(unittest.TestCase):
    def test_bits_length_matches_n(self):
        g = petersen_graph()
        res = maxcut_ilp_highs(g)
        self.assertEqual(len(res["cut_bits"]), g.n)

    def test_bits_reproduce_opt(self):
        for g in [complete_graph(4), cycle_graph(5), petersen_graph(),
                   path_graph(6), complete_bipartite(2, 4)]:
            res = maxcut_ilp_highs(g)
            recomputed = evaluate_cut(g, res["cut_bits"])
            self.assertAlmostEqual(recomputed, res["opt_value"], delta=TOL,
                                    msg=f"Bit-string reproduction failed op n={g.n}")

    def test_symmetry_break_x0_is_0(self):
        g = complete_graph(4)
        res = maxcut_ilp_highs(g, break_symmetry=True)
        self.assertEqual(res["cut_bits"][0], "0")


# ============================================================
# 3. Certification fields
# ============================================================

class TestHighsCertification(unittest.TestCase):
    def test_certified_on_small(self):
        res = maxcut_ilp_highs(complete_graph(5))
        self.assertTrue(res["certified"])
        self.assertEqual(res["gap_abs"], 0.0)
        self.assertEqual(res["solver"], "HiGHS")
        self.assertGreater(res["wall_time"], 0.0)

    def test_n_vars_matches_formula(self):
        """Signed-safe formulering heeft 4 rijen per edge (UB1+UB2+LB1+LB2).
        HiGHS telt de x_0-bound als bounds (niet als rij), dus 4m rijen."""
        g = petersen_graph()
        res = maxcut_ilp_highs(g)
        self.assertEqual(res["n_vars"], g.n + g.n_edges)
        self.assertEqual(res["n_constrs"], 4 * g.n_edges)


# ============================================================
# 4. Time-limit behavior
# ============================================================

class TestHighsTimeLimit(unittest.TestCase):
    def test_tiny_time_limit_still_feasible(self):
        import random
        random.seed(11)
        g = SimpleGraph(25)
        for u in range(25):
            for v in range(u + 1, 25):
                if random.random() < 0.5:
                    g.add_edge(u, v)
        res = maxcut_ilp_highs(g, time_limit=0.01)
        if res["cut_bits"] is not None:
            val = evaluate_cut(g, res["cut_bits"])
            self.assertAlmostEqual(val, res["opt_value"], delta=TOL)


# ============================================================
# 5. Edge cases
# ============================================================

class TestHighsEdgeCases(unittest.TestCase):
    def test_single_edge(self):
        g = SimpleGraph(2)
        g.add_edge(0, 1)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 1.0, delta=TOL)
        self.assertTrue(res["certified"])

    def test_no_edges(self):
        g = SimpleGraph(5)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 0.0, delta=TOL)

    def test_disconnected_two_triangles(self):
        g = SimpleGraph(6)
        g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(0, 2)
        g.add_edge(3, 4); g.add_edge(4, 5); g.add_edge(3, 5)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 4.0, delta=TOL)


# ============================================================
# 6. Weighted edges
# ============================================================

class TestHighsWeighted(unittest.TestCase):
    def test_weighted_k3_unit_weights_gives_2(self):
        g = SimpleGraph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, 1.0)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 2.0, delta=TOL)

    def test_weighted_k3_asymmetric(self):
        g = SimpleGraph(3)
        g.add_edge(0, 1, 5.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, 1.0)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 6.0, delta=TOL)

    def test_weighted_negative_weight_still_works(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, 1.0)
        g.add_edge(0, 3, -10.0)
        res = maxcut_ilp_highs(g)
        self.assertAlmostEqual(res["opt_value"], 2.0, delta=TOL)


# ============================================================
# 7. Dispatcher
# ============================================================

class TestDispatcher(unittest.TestCase):
    def test_dispatcher_highs(self):
        res = maxcut_ilp(complete_graph(4), solver="highs")
        self.assertAlmostEqual(res["opt_value"], 4.0, delta=TOL)

    def test_dispatcher_unknown_raises(self):
        with self.assertRaises(ValueError):
            maxcut_ilp(complete_graph(4), solver="foo")

    def test_solver_registry(self):
        self.assertIn("highs", SOLVERS)
        self.assertIn("scip", SOLVERS)
        self.assertIn("gurobi", SOLVERS)


# ============================================================
# 8. Optional solvers (SCIP / Gurobi)
# ============================================================

class TestOptionalSolvers(unittest.TestCase):
    def test_scip_either_works_or_skipped(self):
        res = maxcut_ilp_scip(complete_graph(4))
        self.assertIn(res["solver"], ("SCIP",))
        if res["opt_value"] is not None:
            self.assertAlmostEqual(res["opt_value"], 4.0, delta=TOL)
        else:
            self.assertIn("SKIPPED", str(res["status"]))

    def test_gurobi_either_works_or_skipped(self):
        res = maxcut_ilp_gurobi(complete_graph(4))
        self.assertIn(res["solver"], ("Gurobi",))
        if res["opt_value"] is not None:
            self.assertAlmostEqual(res["opt_value"], 4.0, delta=TOL)
        else:
            self.assertIn("SKIPPED", str(res["status"]))


# ============================================================
# 9. ILP-OPT matches brute + lies below upper bounds
# ============================================================

class TestAgainstBounds(unittest.TestCase):
    def test_ilp_matches_brute_small(self):
        for g_name, g in [
            ("K_4", complete_graph(4)),
            ("C_5", cycle_graph(5)),
            ("Petersen", petersen_graph()),
            ("K_3,3", complete_bipartite(3, 3)),
        ]:
            ilp = maxcut_ilp_highs(g)
            bf = brute_force_maxcut(g)
            self.assertAlmostEqual(ilp["opt_value"], float(bf), delta=TOL,
                                    msg=f"ILP-OPT vs brute op {g_name}")

    def test_ilp_below_gw_upper_bound(self):
        g = petersen_graph()
        ilp = maxcut_ilp_highs(g)["opt_value"]
        gw = gw_sdp_bound(g, verbose=False)["sdp_bound"]
        self.assertLessEqual(ilp, gw + 1e-3)

    def test_ilp_below_sos2_bound(self):
        g = petersen_graph()
        ilp = maxcut_ilp_highs(g)["opt_value"]
        s2 = sos2_sdp_bound(g, verbose=False)["sos2_bound"]
        self.assertLessEqual(ilp, s2 + 1e-3)

    def test_ilp_below_lpoc_bound(self):
        g = petersen_graph()
        ilp = maxcut_ilp_highs(g)["opt_value"]
        lpoc = lp_triangle_oddcycle_bound(g, verbose=False)["lp_bound"]
        self.assertLessEqual(ilp, lpoc + 1e-3)


# ============================================================
# 10. Signed-instance regression (Dag-8b fix: ILP sign-bug)
# ============================================================

class TestSignedInstancesDag8b(unittest.TestCase):
    """Regressie-tests voor de Dag-8b ILP-sign-bug.

    Historie: tot Dag-8b gebruikte maxcut_ilp_* alleen upper-bound constraints
    y_uv <= x_u+x_v en y_uv <= 2-x_u-x_v. Voor negatief-gewogen edges kon de
    solver y_uv = 0 kiezen op een gesneden edge om de negatieve bijdrage te
    ontlopen, zodat de gerapporteerde OPT gelijk werd aan max-positive-cut in
    plaats van de echte signed MaxCut. Met de toegevoegde lower bounds
    y_uv >= |x_u - x_v| (LB1 + LB2) wordt y_uv gelijk aan de cut-indicator,
    onafhankelijk van het teken.
    """

    def _check_against_brute(self, g, name):
        ilp = maxcut_ilp_highs(g)
        bf = brute_force_maxcut(g)
        self.assertTrue(ilp["certified"], f"{name}: niet gecertificeerd")
        self.assertAlmostEqual(ilp["opt_value"], float(bf), delta=TOL,
                                msg=f"{name}: ILP-OPT vs brute")
        recomputed = evaluate_cut(g, ilp["cut_bits"])
        self.assertAlmostEqual(recomputed, ilp["opt_value"], delta=TOL,
                                msg=f"{name}: bits reproduceren OPT niet")

    def test_path_with_negative_backedge(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 2, 10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(0, 3, -10.0)
        res = maxcut_ilp_highs(g)
        self.assertTrue(res["certified"])
        self.assertAlmostEqual(res["opt_value"], 20.0, delta=TOL,
                                msg="Bug-formulering zou 30 geven; fix geeft 20.")
        self._check_against_brute(g, "P_4 + neg-backedge")

    def test_c5_alternating_signs(self):
        g = SimpleGraph(5)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 2, -10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(3, 4, -10.0)
        g.add_edge(0, 4, 10.0)
        res = maxcut_ilp_highs(g)
        self.assertTrue(res["certified"])
        self.assertLess(res["opt_value"], 30.0 - TOL,
                         "Fix moet < 30 rapporteren; bug-formulering geeft 30.")
        self._check_against_brute(g, "C_5 alt-signs")

    def test_triangle_with_negative_edge(self):
        g = SimpleGraph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, -1.0)
        self._check_against_brute(g, "K_3 +,+,-")

    def test_all_negative_triangle(self):
        g = SimpleGraph(3)
        g.add_edge(0, 1, -1.0)
        g.add_edge(1, 2, -1.0)
        g.add_edge(0, 2, -1.0)
        res = maxcut_ilp_highs(g)
        self.assertTrue(res["certified"])
        self.assertAlmostEqual(res["opt_value"], 0.0, delta=TOL)
        self._check_against_brute(g, "K_3 all-neg")

    def test_k4_mixed_signs_bipartite_structure(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 3, 10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(0, 2, 10.0)
        g.add_edge(0, 3, -100.0)
        g.add_edge(1, 2, -100.0)
        res = maxcut_ilp_highs(g)
        self.assertTrue(res["certified"])
        self.assertAlmostEqual(res["opt_value"], 40.0, delta=TOL)
        self._check_against_brute(g, "K_4 cycle+diag")

    def test_dispatcher_routes_signed_correctly(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 2, 10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(0, 3, -10.0)
        res = maxcut_ilp(g, solver="highs")
        self.assertAlmostEqual(res["opt_value"], 20.0, delta=TOL)

    def test_scip_matches_on_signed_if_available(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 2, 10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(0, 3, -10.0)
        res = maxcut_ilp_scip(g)
        if res["opt_value"] is not None:
            self.assertAlmostEqual(res["opt_value"], 20.0, delta=TOL)
        else:
            self.assertIn("SKIPPED", str(res["status"]))

    def test_gurobi_matches_on_signed_if_available(self):
        g = SimpleGraph(4)
        g.add_edge(0, 1, 10.0)
        g.add_edge(1, 2, 10.0)
        g.add_edge(2, 3, 10.0)
        g.add_edge(0, 3, -10.0)
        res = maxcut_ilp_gurobi(g)
        if res["opt_value"] is not None:
            self.assertAlmostEqual(res["opt_value"], 20.0, delta=TOL)
        else:
            self.assertIn("SKIPPED", str(res["status"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
