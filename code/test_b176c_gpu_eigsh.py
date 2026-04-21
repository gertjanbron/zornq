#!/usr/bin/env python3
"""Unit tests voor B176c GPU-eigsh module.

Coverage:
  * TestAvailableBackends       (2) -- advertising + dispatch-logica
  * TestDenseFallback           (2) -- kleine n altijd dense eigh
  * TestCorrectnessScipyArpack  (3) -- eigvals matchen numpy.linalg.eigh
  * TestCorrectnessScipyLobpcg  (3) -- idem voor LOBPCG-backend
  * TestWarmStartSpeedup        (2) -- v0 reduceert matvec-count
  * TestAutoDispatch            (2) -- 'auto' kiest juiste backend
  * TestInputValidation         (2) -- shape-mismatches raisen
  * TestLmoSpectraplexWrapper   (2) -- backwards-compat API
  * TestCupyBackends            (skip-if-no-GPU) -- CuPy-paden
"""
from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b176c_gpu_eigsh import (
    EigshResult,
    _dense_smallest,
    available_backends,
    clear_gpu_cache,
    cupy_available,
    gpu_eigsh_smallest,
    lmo_spectraplex_warm,
)


# ============================================================
# Hulp: synthetische Laplacian-bouwsels
# ============================================================


def random_sparse_laplacian(n: int, density: float = 0.01, seed: int = 0) -> sp.csr_matrix:
    """Bouw een random symm. Laplacian (L = D - A) van een sparse graaf."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, random_state=rng, dtype=float).tocoo()
    # Symmetriseer
    A = A + A.T
    A.data = np.ones_like(A.data)  # unweighted
    A = sp.csr_matrix(A)
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    return L.tocsr()


def ground_truth_lam_min(
    L: sp.spmatrix, z: np.ndarray, coef_L: float = -0.25,
) -> tuple[float, np.ndarray]:
    G = coef_L * L.toarray() + np.diag(z)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    return float(w[0]), V[:, 0].copy()


# ============================================================
# TestAvailableBackends
# ============================================================


class TestAvailableBackends(unittest.TestCase):
    def test_scipy_backends_always_listed(self) -> None:
        av = available_backends()
        self.assertIn("scipy_arpack", av)
        self.assertIn("scipy_lobpcg", av)

    def test_cupy_listed_iff_available(self) -> None:
        av = available_backends()
        has_cupy = ("cupy_lanczos" in av) and ("cupy_lobpcg" in av)
        # Must match the helper
        self.assertEqual(has_cupy, cupy_available())


# ============================================================
# TestDenseFallback
# ============================================================


class TestDenseFallback(unittest.TestCase):
    def test_small_n_uses_dense(self) -> None:
        L = random_sparse_laplacian(20, density=0.2, seed=1)
        z = np.zeros(20)
        res = gpu_eigsh_smallest(L, z, backend="auto")
        self.assertEqual(res.info["backend"], "dense_eigh")

    def test_dense_matches_ground_truth(self) -> None:
        L = random_sparse_laplacian(15, density=0.3, seed=2)
        z = 0.1 * np.arange(15, dtype=float)
        gt_lam, gt_v = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="dense")
        self.assertAlmostEqual(res.lam, gt_lam, places=10)
        # Eigenvector up to sign
        v_dot = abs(float(res.v @ gt_v))
        self.assertAlmostEqual(v_dot, 1.0, places=8)


# ============================================================
# TestCorrectnessScipyArpack
# ============================================================


class TestCorrectnessScipyArpack(unittest.TestCase):
    def test_arpack_small_laplacian(self) -> None:
        L = random_sparse_laplacian(100, density=0.05, seed=3)
        z = np.zeros(100)
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="scipy_arpack", tol=1e-10)
        self.assertAlmostEqual(res.lam, gt_lam, places=6)

    def test_arpack_with_diagonal_shift(self) -> None:
        rng = np.random.default_rng(4)
        L = random_sparse_laplacian(150, density=0.03, seed=4)
        z = rng.standard_normal(150) * 0.5
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="scipy_arpack", tol=1e-10)
        self.assertAlmostEqual(res.lam, gt_lam, places=6)

    def test_arpack_returns_eigenvector(self) -> None:
        L = random_sparse_laplacian(80, density=0.08, seed=5)
        z = np.ones(80) * 0.01
        res = gpu_eigsh_smallest(L, z, backend="scipy_arpack")
        # ||v|| = 1
        self.assertAlmostEqual(float(np.linalg.norm(res.v)), 1.0, places=6)
        # Rayleigh-quotient approximates lam
        G = -0.25 * L.toarray() + np.diag(z)
        rq = float(res.v @ (G @ res.v))
        self.assertAlmostEqual(rq, res.lam, places=5)


# ============================================================
# TestCorrectnessScipyLobpcg
# ============================================================


class TestCorrectnessScipyLobpcg(unittest.TestCase):
    def test_lobpcg_matches_ground_truth(self) -> None:
        L = random_sparse_laplacian(100, density=0.05, seed=6)
        z = 0.05 * np.arange(100, dtype=float)
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="scipy_lobpcg", tol=1e-10)
        # LOBPCG is iets losser dan ARPACK in de regel
        self.assertAlmostEqual(res.lam, gt_lam, places=4)

    def test_lobpcg_on_zero_diag(self) -> None:
        L = random_sparse_laplacian(120, density=0.04, seed=7)
        z = np.zeros(120)
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="scipy_lobpcg", tol=1e-10)
        self.assertAlmostEqual(res.lam, gt_lam, places=4)

    def test_lobpcg_eigvec_unit_norm(self) -> None:
        L = random_sparse_laplacian(80, density=0.06, seed=8)
        z = np.ones(80) * 0.02
        res = gpu_eigsh_smallest(L, z, backend="scipy_lobpcg")
        self.assertAlmostEqual(float(np.linalg.norm(res.v)), 1.0, places=4)


# ============================================================
# TestWarmStartSpeedup
# ============================================================


class TestWarmStartSpeedup(unittest.TestCase):
    """Warm-start met v0 = v_{k-1} moet zichtbaar matvec-count reduceren
    als z maar lichtjes drift."""

    def test_warm_start_reduces_matvec_lobpcg(self) -> None:
        n = 400
        L = random_sparse_laplacian(n, density=0.01, seed=9)
        rng = np.random.default_rng(9)
        z0 = rng.standard_normal(n) * 0.1

        # Cold start
        res_cold = gpu_eigsh_smallest(
            L, z0, backend="scipy_lobpcg", tol=1e-6)
        # Warm start met cold-eigvec op lichtjes geperturbeerde z
        z1 = z0 + 1e-4 * rng.standard_normal(n)
        res_warm = gpu_eigsh_smallest(
            L, z1, backend="scipy_lobpcg", v0=res_cold.v, tol=1e-6)
        # Warm-start levert typisch <= cold-matvecs en dezelfde oplossing
        self.assertTrue(res_warm.info["warm_start"])
        self.assertLessEqual(res_warm.info["n_matvec"], res_cold.info["n_matvec"])

    def test_warm_start_propagates_through_info(self) -> None:
        L = random_sparse_laplacian(100, density=0.03, seed=10)
        z = np.zeros(100)
        v0 = np.ones(100) / np.sqrt(100)
        res = gpu_eigsh_smallest(L, z, v0=v0, backend="scipy_arpack")
        self.assertTrue(res.info["warm_start"])


# ============================================================
# TestAutoDispatch
# ============================================================


class TestAutoDispatch(unittest.TestCase):
    def test_auto_below_dense_threshold(self) -> None:
        L = random_sparse_laplacian(30, density=0.2, seed=11)
        z = np.zeros(30)
        res = gpu_eigsh_smallest(L, z, backend="auto", dense_fallback_below=40)
        self.assertEqual(res.info["backend"], "dense_eigh")

    def test_auto_picks_expected_backend(self) -> None:
        # n=500 in auto-mode: CPU -> scipy_arpack; GPU -> cupy_lobpcg
        L = random_sparse_laplacian(500, density=0.01, seed=12)
        z = np.zeros(500)
        res = gpu_eigsh_smallest(L, z, backend="auto", gpu_threshold=500)
        expected = "cupy_lobpcg" if cupy_available() else "scipy_arpack"
        self.assertEqual(res.info["backend"], expected)


# ============================================================
# TestInputValidation
# ============================================================


class TestInputValidation(unittest.TestCase):
    def test_z_shape_mismatch(self) -> None:
        L = random_sparse_laplacian(50, seed=13)
        z_bad = np.zeros(49)
        with self.assertRaises(ValueError):
            gpu_eigsh_smallest(L, z_bad)

    def test_unknown_backend_raises(self) -> None:
        L = random_sparse_laplacian(100, seed=14)
        z = np.zeros(100)
        with self.assertRaises(ValueError):
            gpu_eigsh_smallest(L, z, backend="nonsense")


# ============================================================
# TestLmoSpectraplexWrapper
# ============================================================


class TestLmoSpectraplexWrapper(unittest.TestCase):
    def test_wrapper_returns_v_lam_info(self) -> None:
        L = random_sparse_laplacian(100, density=0.05, seed=15)
        z = 0.01 * np.arange(100, dtype=float)
        v, lam, info = lmo_spectraplex_warm(L, z, backend="scipy_arpack")
        self.assertEqual(v.shape, (100,))
        self.assertIsInstance(lam, float)
        self.assertIsInstance(info, dict)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_wrapper_matches_ground_truth(self) -> None:
        L = random_sparse_laplacian(80, density=0.08, seed=16)
        z = np.ones(80) * 0.02
        gt_lam, _ = ground_truth_lam_min(L, z)
        _, lam, _ = lmo_spectraplex_warm(L, z, backend="scipy_arpack", tol=1e-10)
        self.assertAlmostEqual(lam, gt_lam, places=6)


# ============================================================
# TestCupyBackends  (skip if no GPU)
# ============================================================


@unittest.skipUnless(cupy_available(), "CuPy/CUDA niet beschikbaar")
class TestCupyBackends(unittest.TestCase):
    def setUp(self) -> None:
        clear_gpu_cache()

    def tearDown(self) -> None:
        clear_gpu_cache()

    def test_cupy_lanczos_matches_ground_truth(self) -> None:
        L = random_sparse_laplacian(500, density=0.01, seed=100)
        z = 0.01 * np.arange(500, dtype=float)
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="cupy_lanczos", tol=1e-8)
        self.assertAlmostEqual(res.lam, gt_lam, places=4)

    def test_cupy_lobpcg_matches_ground_truth(self) -> None:
        L = random_sparse_laplacian(500, density=0.01, seed=101)
        z = np.zeros(500)
        gt_lam, _ = ground_truth_lam_min(L, z)
        res = gpu_eigsh_smallest(L, z, backend="cupy_lobpcg", tol=1e-6)
        self.assertAlmostEqual(res.lam, gt_lam, places=3)

    def test_cupy_cache_hit(self) -> None:
        """Zelfde L twee keer: tweede call moet cache-hit geven (geen re-upload)."""
        from b176c_gpu_eigsh import _GPU_L_CACHE
        L = random_sparse_laplacian(300, density=0.01, seed=102)
        z = np.zeros(300)
        clear_gpu_cache()
        self.assertEqual(len(_GPU_L_CACHE), 0)
        gpu_eigsh_smallest(L, z, backend="cupy_lobpcg")
        self.assertEqual(len(_GPU_L_CACHE), 1)
        gpu_eigsh_smallest(L, z, backend="cupy_lobpcg")
        self.assertEqual(len(_GPU_L_CACHE), 1)  # cache hit, geen groei


# ============================================================
# TestCupyFallback (runs whether or not CuPy is present)
# ============================================================


class TestCupyFallback(unittest.TestCase):
    def test_cupy_request_without_cupy_falls_back(self) -> None:
        if cupy_available():
            self.skipTest("CuPy actually available; fallback-pad niet testbaar")
        L = random_sparse_laplacian(100, density=0.05, seed=200)
        z = np.zeros(100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = gpu_eigsh_smallest(L, z, backend="cupy_lobpcg")
        self.assertTrue(any("fallback" in str(x.message).lower() for x in w))
        # Should still return a valid result via fallback
        self.assertIsInstance(res, EigshResult)
        self.assertEqual(res.info["backend"], "scipy_lobpcg")


if __name__ == "__main__":
    unittest.main()
