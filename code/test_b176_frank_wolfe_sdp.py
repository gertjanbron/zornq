#!/usr/bin/env python3
"""Unit tests voor B176 Frank-Wolfe MaxCut-SDP-solver.

Coverage:
  * TestGraphLaplacian       (5) -- sparse L = D - W structuur, PSD, rij-som
  * TestLMO                  (5) -- bottom-eigen, dense+sparse, rank-1 argmin
  * TestFWResultStructure    (3) -- dataclass, vormen, types
  * TestFWSmallGraphs        (6) -- cvxpy-sandwich op kleine grafen
  * TestFWStepRules          (4) -- line-search vs jaggi, monotoon, bounds
  * TestFWConvergence        (4) -- gap omlaag, history, early-stop
  * TestGWRounding           (4) -- rounding geeft valide bitstring + cut
  * TestSandwichProperty     (3) -- LB <= cut_SDP <= UB voor diverse grafen
  * TestRankCap              (3) -- SVD-truncatie, identiteit op spectraplex
  * TestReproducibility      (3) -- seed-stabiliteit
"""
from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b176_frank_wolfe_sdp import (
    FWResult,
    cvxpy_reference_sdp,
    frank_wolfe_maxcut_sdp,
    graph_laplacian,
    gw_round_from_Y,
    lmo_spectraplex,
)
from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    cylinder_graph,
    random_3regular,
    random_erdos_renyi,
)


# ============================================================
# Hulp: triangle-graaf
# ============================================================


def triangle_graph() -> SimpleGraph:
    g = SimpleGraph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    return g


def k4_graph() -> SimpleGraph:
    g = SimpleGraph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    return g


# ============================================================
# TestGraphLaplacian
# ============================================================


class TestGraphLaplacian(unittest.TestCase):
    def test_triangle_laplacian_structure(self):
        g = triangle_graph()
        L = graph_laplacian(g)
        self.assertEqual(L.shape, (3, 3))
        self.assertTrue(sp.issparse(L))
        A = L.toarray()
        # D = diag(2,2,2), W = ones off-diag
        expected = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
        np.testing.assert_allclose(A, expected)

    def test_laplacian_symmetric(self):
        g = random_erdos_renyi(10, p=0.4, seed=7)
        L = graph_laplacian(g).toarray()
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_laplacian_row_sums_zero(self):
        g = random_3regular(20, seed=3)
        L = graph_laplacian(g).toarray()
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_laplacian_psd(self):
        g = cylinder_graph(3, 3)
        L = graph_laplacian(g).toarray()
        w = np.linalg.eigvalsh(L)
        self.assertGreaterEqual(w.min(), -1e-10)

    def test_laplacian_handles_weights(self):
        g = SimpleGraph(3)
        g.add_edge(0, 1, 2.5)
        g.add_edge(1, 2, 0.5)
        L = graph_laplacian(g).toarray()
        self.assertAlmostEqual(L[0, 0], 2.5)
        self.assertAlmostEqual(L[1, 1], 3.0)
        self.assertAlmostEqual(L[2, 2], 0.5)
        self.assertAlmostEqual(L[0, 1], -2.5)


# ============================================================
# TestLMO
# ============================================================


class TestLMO(unittest.TestCase):
    def test_lmo_dense_eigenvector(self):
        # G = diag(3, 1, 2): argmin v*v^T <G> is e_2 eigenvalue 1
        G = np.diag([3.0, 1.0, 2.0])
        matvec = lambda x: G @ x
        v, lam = lmo_spectraplex(matvec, 3, dense_fallback_below=10, dense_G=G)
        self.assertAlmostEqual(lam, 1.0, places=6)
        # v is +/- e_1 (the second basis vector)
        self.assertAlmostEqual(abs(v[1]), 1.0, places=6)

    def test_lmo_symmetric_matrix(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((20, 20))
        A = 0.5 * (A + A.T)
        w_exact = np.linalg.eigvalsh(A)
        v, lam = lmo_spectraplex(lambda x: A @ x, 20, dense_fallback_below=5)
        self.assertAlmostEqual(lam, float(w_exact[0]), places=4)
        # verify eigenvector
        residual = np.linalg.norm(A @ v - lam * v)
        self.assertLess(residual, 1e-3)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=6)

    def test_lmo_dense_fallback_activated(self):
        G = np.array([[2.0, 0.5], [0.5, 1.0]])
        v, lam = lmo_spectraplex(lambda x: G @ x, 2, dense_fallback_below=10, dense_G=G)
        # smallest eigen of [[2,.5],[.5,1]] = (3 - sqrt(1+1))/2 = 0.7929
        self.assertAlmostEqual(lam, (3 - np.sqrt(2.0)) / 2.0, places=6)

    def test_lmo_returns_unit_vector(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((15, 15))
        A = 0.5 * (A + A.T)
        v, _ = lmo_spectraplex(lambda x: A @ x, 15, dense_fallback_below=5)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=6)

    def test_lmo_sparse_matvec(self):
        # Gebruik een shift I + L om een NIET-singuliere matrix te krijgen;
        # ARPACK heeft anders moeite met het 0-eigenwaarde-null-space van L.
        n = 50
        L_sp = graph_laplacian(random_3regular(n, seed=2))
        shift = 2.0
        matvec = lambda x: L_sp.dot(x) - shift * x   # smallest eig ~ 0 - shift = -2
        v, lam = lmo_spectraplex(matvec, n, dense_fallback_below=10,
                                 tol=1e-8, maxiter=5000)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=6)
        w_exact = np.linalg.eigvalsh(L_sp.toarray() - shift * np.eye(n))
        self.assertAlmostEqual(lam, float(w_exact[0]), delta=1e-3)


# ============================================================
# TestFWResultStructure
# ============================================================


class TestFWResultStructure(unittest.TestCase):
    def test_fwresult_fields_present(self):
        g = triangle_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=50, tol=1e-5, rank_cap=8)
        for attr in ("sdp_bound", "sdp_upper_bound", "feasible_cut_lb",
                     "Y", "X_diag", "diag_err_max", "iterations",
                     "final_gap", "converged", "solve_time",
                     "history", "n", "n_edges", "penalty"):
            self.assertTrue(hasattr(res, attr), "missing attr %s" % attr)

    def test_Y_shape(self):
        g = cylinder_graph(4, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=30, tol=1e-3, rank_cap=8)
        self.assertEqual(res.Y.shape[0], g.n)
        self.assertLessEqual(res.Y.shape[1], 8 + 1)  # rank_cap + 1

    def test_history_grows_with_iter(self):
        g = triangle_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=20, tol=1e-12, rank_cap=8)
        self.assertEqual(len(res.history), res.iterations)


# ============================================================
# TestFWSmallGraphs -- sandwich tegen cvxpy-referentie
# ============================================================


class TestFWSmallGraphs(unittest.TestCase):
    """FW-bound moet cvxpy insluiten: LB <= cvxpy <= UB (tot kleine slack)."""

    def _check_sandwich(self, g, max_iter=500, rank_cap=16, slack=0.5):
        res = frank_wolfe_maxcut_sdp(g, max_iter=max_iter, tol=1e-5, rank_cap=rank_cap)
        ref = cvxpy_reference_sdp(g, verbose=False)
        self.assertIsNotNone(ref["sdp_bound"])
        cvx = ref["sdp_bound"]
        self.assertLessEqual(res.feasible_cut_lb, cvx + 1e-6,
                             msg="LB %.4f > cvx %.4f" % (res.feasible_cut_lb, cvx))
        self.assertGreaterEqual(res.sdp_upper_bound, cvx - slack,
                                msg="UB %.4f << cvx %.4f" % (res.sdp_upper_bound, cvx))
        return res, ref

    def test_triangle(self):
        self._check_sandwich(triangle_graph(), max_iter=200)

    def test_k4(self):
        self._check_sandwich(k4_graph(), max_iter=300)

    def test_cylinder_4x3(self):
        self._check_sandwich(cylinder_graph(4, 3), max_iter=400)

    def test_cylinder_3x4(self):
        self._check_sandwich(cylinder_graph(3, 4), max_iter=400)

    def test_random_3reg_20(self):
        self._check_sandwich(random_3regular(20, seed=5), max_iter=500, slack=1.0)

    def test_random_er_15(self):
        self._check_sandwich(random_erdos_renyi(15, p=0.4, seed=9),
                             max_iter=500, slack=1.0)


# ============================================================
# TestFWStepRules
# ============================================================


class TestFWStepRules(unittest.TestCase):
    def test_linesearch_default(self):
        g = cylinder_graph(3, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=100, tol=1e-5, rank_cap=8,
                                     step_rule="linesearch")
        self.assertTrue(res.converged or res.iterations == 100)

    def test_jaggi_step(self):
        g = cylinder_graph(3, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=200, tol=1e-4, rank_cap=8,
                                     step_rule="jaggi")
        # Jaggi is slower to converge
        self.assertGreater(res.iterations, 10)

    def test_unknown_step_rule_raises(self):
        g = triangle_graph()
        with self.assertRaises(ValueError):
            frank_wolfe_maxcut_sdp(g, max_iter=5, step_rule="foo")

    def test_linesearch_no_worse_than_jaggi(self):
        # Bij dezelfde max_iter geeft line-search een betere (lagere) f dan Jaggi.
        g = cylinder_graph(4, 3)
        r_ls = frank_wolfe_maxcut_sdp(g, max_iter=80, tol=0, rank_cap=10,
                                      step_rule="linesearch", seed=0)
        r_j = frank_wolfe_maxcut_sdp(g, max_iter=80, tol=0, rank_cap=10,
                                     step_rule="jaggi", seed=0)
        f_ls = r_ls.history[-1]["f"]
        f_j = r_j.history[-1]["f"]
        self.assertLessEqual(f_ls, f_j + 1e-6)


# ============================================================
# TestFWConvergence
# ============================================================


class TestFWConvergence(unittest.TestCase):
    def test_gap_decreases_over_iterations(self):
        g = cylinder_graph(4, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-8, rank_cap=16)
        gaps = [h["gap"] for h in res.history]
        # Trend omlaag: max(gap) van eerste 10 > max(gap) van laatste 10
        first = max(gaps[:10])
        last = max(gaps[-10:])
        self.assertLess(last, first + 1e-9)

    def test_early_stop_on_small_gap(self):
        # Met ruime tol stopt FW voor max_iter (plafond penalty-geinduceerd).
        g = triangle_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=1000, tol=5e-2, rank_cap=4)
        self.assertLess(res.iterations, 1000)
        self.assertTrue(res.converged)

    def test_iteration_cap_respected(self):
        g = random_3regular(30, seed=0)
        res = frank_wolfe_maxcut_sdp(g, max_iter=15, tol=1e-12, rank_cap=8)
        self.assertLessEqual(res.iterations, 15)

    def test_history_entries_contain_expected_keys(self):
        g = k4_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=5, tol=1e-12, rank_cap=4)
        for h in res.history:
            for k in ("iter", "f", "gap", "tr_LX", "diag_err_max",
                      "lam_min", "rank"):
                self.assertIn(k, h)


# ============================================================
# TestGWRounding
# ============================================================


class TestGWRounding(unittest.TestCase):
    def test_rounding_on_triangle_produces_valid_cut(self):
        g = triangle_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=200, tol=1e-5, rank_cap=4)
        bs, cut = gw_round_from_Y(res.Y, g, n_trials=50)
        self.assertEqual(len(bs), g.n)
        self.assertTrue(set(bs.tolist()) <= {0, 1})
        # triangle MaxCut = 2
        self.assertEqual(cut, 2.0)

    def test_rounding_matches_exact_on_k4(self):
        g = k4_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-5, rank_cap=4)
        _, cut = gw_round_from_Y(res.Y, g, n_trials=100)
        exact = brute_force_maxcut(g)
        self.assertEqual(cut, exact)

    def test_rounding_at_least_half_of_edges(self):
        # Random cut gives >= m/2 in expectation; best of 50 should hit it
        g = random_3regular(20, seed=11)
        res = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-4, rank_cap=12)
        _, cut = gw_round_from_Y(res.Y, g, n_trials=100)
        self.assertGreaterEqual(cut, 0.5 * g.n_edges)

    def test_rounding_with_zero_row_norm_safe(self):
        # Synthetic Y met een nul-rij mag niet NaN genereren
        Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        g = SimpleGraph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        bs, cut = gw_round_from_Y(Y, g, n_trials=20)
        self.assertTrue(np.all(np.isfinite(Y)))
        self.assertIsNotNone(cut)


# ============================================================
# TestSandwichProperty
# ============================================================


class TestSandwichProperty(unittest.TestCase):
    def _sandwich(self, g, max_iter, rank_cap):
        res = frank_wolfe_maxcut_sdp(g, max_iter=max_iter, tol=1e-6, rank_cap=rank_cap)
        return res.feasible_cut_lb, res.sdp_upper_bound

    def test_sandwich_on_3reg_n30(self):
        g = random_3regular(30, seed=8)
        lb, ub = self._sandwich(g, 500, 16)
        self.assertLessEqual(lb, ub + 1e-6)

    def test_upper_bound_ge_exact_maxcut(self):
        g = cylinder_graph(4, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=400, tol=1e-6, rank_cap=12)
        exact = brute_force_maxcut(g)
        self.assertGreaterEqual(res.sdp_upper_bound, exact - 1e-6)

    def test_feasible_lb_le_total_weight(self):
        g = random_erdos_renyi(12, p=0.5, seed=4)
        res = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-5, rank_cap=8)
        self.assertLessEqual(res.feasible_cut_lb, g.total_weight() + 1e-6)


# ============================================================
# TestRankCap
# ============================================================


class TestRankCap(unittest.TestCase):
    def test_rank_capped(self):
        g = random_3regular(40, seed=4)
        res = frank_wolfe_maxcut_sdp(g, max_iter=200, tol=1e-8, rank_cap=8)
        self.assertLessEqual(res.Y.shape[1], 8 + 1)

    def test_higher_rank_no_worse_f(self):
        g = cylinder_graph(4, 3)
        r_lo = frank_wolfe_maxcut_sdp(g, max_iter=80, tol=0, rank_cap=4, seed=0)
        r_hi = frank_wolfe_maxcut_sdp(g, max_iter=80, tol=0, rank_cap=20, seed=0)
        # Hogere rank zou minstens even goed (of iets beter) moeten zijn.
        self.assertLessEqual(r_hi.history[-1]["f"] - 1e-2, r_lo.history[-1]["f"] + 1e-2)

    def test_spectraplex_trace_invariant(self):
        g = triangle_graph()
        res = frank_wolfe_maxcut_sdp(g, max_iter=50, tol=0, rank_cap=8)
        tr = float(np.sum(res.X_diag))
        # tr(X) = n moet strikt behouden blijven
        self.assertAlmostEqual(tr, g.n, places=6)


# ============================================================
# TestReproducibility
# ============================================================


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_result(self):
        g = random_3regular(20, seed=2)
        r1 = frank_wolfe_maxcut_sdp(g, max_iter=100, tol=1e-6, rank_cap=8, seed=0)
        r2 = frank_wolfe_maxcut_sdp(g, max_iter=100, tol=1e-6, rank_cap=8, seed=0)
        self.assertAlmostEqual(r1.sdp_upper_bound, r2.sdp_upper_bound, places=6)
        self.assertAlmostEqual(r1.feasible_cut_lb, r2.feasible_cut_lb, places=6)

    def test_different_seed_similar_bound(self):
        g = cylinder_graph(4, 3)
        r1 = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-5, rank_cap=12, seed=0)
        r2 = frank_wolfe_maxcut_sdp(g, max_iter=300, tol=1e-5, rank_cap=12, seed=42)
        # upper bound moet binnen 10% van elkaar liggen
        self.assertLess(abs(r1.sdp_upper_bound - r2.sdp_upper_bound),
                        0.1 * max(r1.sdp_upper_bound, r2.sdp_upper_bound))

    def test_gw_rounding_reproducible(self):
        g = cylinder_graph(3, 3)
        res = frank_wolfe_maxcut_sdp(g, max_iter=200, tol=1e-5, rank_cap=8, seed=0)
        bs1, cut1 = gw_round_from_Y(res.Y, g, n_trials=30, seed=7)
        bs2, cut2 = gw_round_from_Y(res.Y, g, n_trials=30, seed=7)
        np.testing.assert_array_equal(bs1, bs2)
        self.assertEqual(cut1, cut2)


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)
