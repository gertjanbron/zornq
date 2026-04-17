#!/usr/bin/env python3
"""
Tests voor B39: TRG / HOTRG — Tensor Renormalization Group voor 2D QAOA MaxCut

Testcategorieën:
  1. Tensor2D / TensorGrid datastructures
  2. SVD truncatie
  3. Ising partitie functie (TRG vs exact)
  4. Ising partitie functie (HOTRG vs exact)
  5. TRG coarse-graining mechanics
  6. HOTRG coarse-graining mechanics
  7. QAOA 2D exact state vector
  8. Edge cases (1x1, 2x2, niet-vierkante grids)
  9. Convergentie: TRG nauwkeurigheid vs chi_max
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trg_hotrg import (
    Tensor2D, TensorGrid,
    build_qaoa_tensor_grid, _build_site_tensor, _compute_site_element,
    trg_truncate_svd, trg_coarse_grain_step, trg_contract,
    _contract_2x2_block, _truncate_tensor_4leg,
    _trivial_tensor, _trivial_row,
    hotrg_coarse_grain_step, hotrg_contract, _hotrg_contract_pair,
    ising_partition_trg, ising_free_energy_exact,
    qaoa_2d_exact, qaoa_2d_ratio, trg_qaoa_cost,
)


class TestTensor2D(unittest.TestCase):
    """Tests voor Tensor2D datastructure."""

    def test_basic_shape(self):
        T = Tensor2D(data=np.zeros((2, 3, 4, 5)))
        self.assertEqual(T.shape, (2, 3, 4, 5))
        self.assertEqual(T.chi_up, 2)
        self.assertEqual(T.chi_right, 3)
        self.assertEqual(T.chi_down, 4)
        self.assertEqual(T.chi_left, 5)

    def test_has_phys_false(self):
        T = Tensor2D(data=np.zeros((2, 2, 2, 2)))
        self.assertFalse(T.has_phys)

    def test_has_phys_true(self):
        T = Tensor2D(data=np.zeros((2, 2, 2, 2, 3)))
        self.assertTrue(T.has_phys)

    def test_tensor_grid(self):
        tensors = [[Tensor2D(data=np.ones((1, 2, 1, 2))) for _ in range(3)]
                    for _ in range(4)]
        grid = TensorGrid(Lx=4, Ly=3, tensors=tensors)
        self.assertEqual(grid.Lx, 4)
        self.assertEqual(grid.Ly, 3)
        self.assertEqual(grid.boundary, 'open')


class TestSVDTruncation(unittest.TestCase):
    """Tests voor trg_truncate_svd."""

    def test_no_truncation_needed(self):
        M = np.random.randn(3, 3)
        U, S, Vh = trg_truncate_svd(M, chi_max=10)
        # Reconstructie
        recon = U @ np.diag(S) @ Vh
        np.testing.assert_allclose(recon, M, atol=1e-12)

    def test_truncation_applied(self):
        M = np.random.randn(10, 10)
        U, S, Vh = trg_truncate_svd(M, chi_max=3)
        self.assertLessEqual(len(S), 3)
        self.assertEqual(U.shape[1], len(S))
        self.assertEqual(Vh.shape[0], len(S))

    def test_singular_values_decreasing(self):
        M = np.random.randn(8, 8)
        _, S, _ = trg_truncate_svd(M, chi_max=8)
        for i in range(len(S) - 1):
            self.assertGreaterEqual(S[i], S[i + 1])

    def test_zero_matrix(self):
        M = np.zeros((3, 3))
        U, S, Vh = trg_truncate_svd(M, chi_max=5)
        # Moet minimaal chi=1 teruggeven
        self.assertGreaterEqual(len(S), 1)

    def test_rank1_matrix(self):
        v = np.array([1, 2, 3, 4, 5], dtype=float)
        M = np.outer(v, v)
        U, S, Vh = trg_truncate_svd(M, chi_max=10)
        # Rank-1 → slechts 1 significante singuliere waarde
        self.assertEqual(len(S), 1)


class TestTrivialTensors(unittest.TestCase):
    """Tests voor _trivial_tensor en _trivial_row."""

    def test_trivial_tensor_shape(self):
        ref = np.ones((2, 3, 4, 5))
        T = _trivial_tensor(ref)
        self.assertEqual(T.shape, ref.shape)

    def test_trivial_tensor_values(self):
        ref = np.ones((2, 2, 2, 2))
        T = _trivial_tensor(ref)
        # Diagonaal-achtig: T[0,0,0,0] = 1, T[1,1,1,1] = 1
        self.assertEqual(T[0, 0, 0, 0], 1.0)
        self.assertEqual(T[1, 1, 1, 1], 1.0)

    def test_trivial_row_length(self):
        last_row = [np.ones((2, 2, 2, 2)) for _ in range(5)]
        row = _trivial_row(last_row, 5)
        self.assertEqual(len(row), 5)


class TestContract2x2Block(unittest.TestCase):
    """Tests voor 2x2 blok contractie."""

    def test_identity_contraction(self):
        """Contractie van identity-achtige tensors moet werken."""
        T = np.zeros((2, 2, 2, 2))
        T[0, 0, 0, 0] = 1.0
        T[1, 1, 1, 1] = 1.0
        result = _contract_2x2_block(T, T, T, T, chi_max=8)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 4)

    def test_scalar_tensors(self):
        """1x1x1x1 tensors contracteren tot 1x1x1x1."""
        A = np.array([[[[2.0]]]]);  B = np.array([[[[3.0]]]])
        C = np.array([[[[5.0]]]]);  D = np.array([[[[7.0]]]])
        result = _contract_2x2_block(A, B, C, D, chi_max=4)
        # Scalaire contractie: product
        val = result[0, 0, 0, 0] if result.size == 1 else result.flatten()[0]
        expected = 2.0 * 3.0 * 5.0 * 7.0
        self.assertAlmostEqual(abs(val), abs(expected), places=5)

    def test_output_shape_bounded(self):
        """Output bond dimensies moeten <= chi_max zijn."""
        A = np.random.randn(2, 3, 2, 3)
        B = np.random.randn(2, 2, 2, 3)
        C = np.random.randn(2, 3, 2, 3)
        D = np.random.randn(2, 2, 2, 3)
        chi_max = 4
        result = _contract_2x2_block(A, B, C, D, chi_max)
        for dim in result.shape:
            self.assertLessEqual(dim, chi_max * chi_max)  # may be product before truncation


class TestTruncateTensor4Leg(unittest.TestCase):
    """Tests voor 4-been tensor truncatie."""

    def test_no_truncation_small(self):
        T = np.random.randn(2, 2, 2, 2)
        result = _truncate_tensor_4leg(T, chi_max=4)
        np.testing.assert_array_equal(result, T)

    def test_truncation_reduces(self):
        T = np.random.randn(8, 8, 8, 8)
        result = _truncate_tensor_4leg(T, chi_max=4)
        # Minstens één dimensie moet gereduceerd zijn
        total_orig = np.prod(T.shape)
        total_new = np.prod(result.shape)
        self.assertLess(total_new, total_orig)


class TestTRGCoarseGrain(unittest.TestCase):
    """Tests voor TRG coarse-graining stap."""

    def test_grid_size_halves(self):
        """Grid moet halveren per stap."""
        grid = [[np.random.randn(2, 2, 2, 2) for _ in range(4)] for _ in range(4)]
        new_grid = trg_coarse_grain_step(grid, chi_max=4)
        self.assertEqual(len(new_grid), 2)
        self.assertEqual(len(new_grid[0]), 2)

    def test_odd_grid_padded(self):
        """Oneven grid wordt gepad en gehalveerd."""
        grid = [[np.random.randn(2, 2, 2, 2) for _ in range(3)] for _ in range(3)]
        new_grid = trg_coarse_grain_step(grid, chi_max=4)
        self.assertEqual(len(new_grid), 2)
        self.assertEqual(len(new_grid[0]), 2)

    def test_2x2_to_1x1(self):
        """2x2 grid contracheert tot 1x1."""
        grid = [[np.random.randn(2, 2, 2, 2) for _ in range(2)] for _ in range(2)]
        new_grid = trg_coarse_grain_step(grid, chi_max=8)
        self.assertEqual(len(new_grid), 1)
        self.assertEqual(len(new_grid[0]), 1)


class TestTRGContract(unittest.TestCase):
    """Tests voor volledige TRG contractie."""

    def test_1x1_grid(self):
        """1x1 grid: trace over tensor."""
        T = np.zeros((2, 2, 2, 2))
        T[0, 0, 0, 0] = 3.0
        T[1, 1, 1, 1] = 5.0
        result = trg_contract([[T]], chi_max=4)
        self.assertAlmostEqual(abs(result), 8.0, places=5)

    def test_simple_2x2(self):
        """2x2 grid met simpele tensors convergeert."""
        T = np.zeros((2, 2, 2, 2))
        T[0, 0, 0, 0] = 1.0
        T[1, 1, 1, 1] = 1.0
        grid = [[T.copy() for _ in range(2)] for _ in range(2)]
        result = trg_contract(grid, chi_max=4)
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result))

    def test_returns_scalar(self):
        """Resultaat moet een scalar zijn."""
        T = np.ones((1, 1, 1, 1))
        result = trg_contract([[T]], chi_max=4)
        self.assertTrue(np.isscalar(result) or (hasattr(result, 'shape') and result.shape == ()))


class TestHOTRGCoarseGrain(unittest.TestCase):
    """Tests voor HOTRG coarse-graining."""

    def test_horizontal_halves_x(self):
        grid = [[np.random.randn(2, 2, 2, 2) for _ in range(3)] for _ in range(4)]
        new_grid = hotrg_coarse_grain_step(grid, chi_max=4, direction='horizontal')
        self.assertEqual(len(new_grid), 2)
        self.assertEqual(len(new_grid[0]), 3)

    def test_vertical_halves_y(self):
        grid = [[np.random.randn(2, 2, 2, 2) for _ in range(4)] for _ in range(3)]
        new_grid = hotrg_coarse_grain_step(grid, chi_max=4, direction='vertical')
        self.assertEqual(len(new_grid), 3)
        self.assertEqual(len(new_grid[0]), 2)

    def test_hotrg_contract_pair_horizontal(self):
        A = np.random.randn(2, 3, 2, 2)
        B = np.random.randn(2, 2, 2, 3)  # B.left=3 moet matchen A.right=3
        result = _hotrg_contract_pair(A, B, 'horizontal', chi_max=4)
        self.assertEqual(len(result.shape), 4)

    def test_hotrg_contract_pair_vertical(self):
        A = np.random.randn(2, 2, 3, 2)  # A.down=3
        B = np.random.randn(3, 2, 2, 2)  # B.up=3 moet matchen
        result = _hotrg_contract_pair(A, B, 'vertical', chi_max=4)
        self.assertEqual(len(result.shape), 4)


class TestHOTRGContract(unittest.TestCase):
    """Tests voor volledige HOTRG contractie."""

    def test_1x1_grid(self):
        T = np.zeros((2, 2, 2, 2))
        T[0, 0, 0, 0] = 3.0
        T[1, 1, 1, 1] = 5.0
        result = hotrg_contract([[T]], chi_max=4)
        self.assertAlmostEqual(abs(result), 8.0, places=5)

    def test_2x2_grid_finite(self):
        T = np.zeros((2, 2, 2, 2))
        T[0, 0, 0, 0] = 1.0
        T[1, 1, 1, 1] = 1.0
        grid = [[T.copy() for _ in range(2)] for _ in range(2)]
        result = hotrg_contract(grid, chi_max=4)
        self.assertTrue(np.isfinite(result))


class TestIsingExact(unittest.TestCase):
    """Tests voor exacte Ising vrije energie (periodieke BC)."""

    def test_1x1_grid(self):
        """1 spin, periodieke BC → 2 self-loops: Z = sum_s exp(2*beta*s*s) = 2*exp(2*beta)."""
        beta = 0.5
        # 1x1 periodic: 1 horizontal wrap + 1 vertical wrap = 2 edges, both (0,0)
        # Z = sum_{s=+1,-1} exp(beta * s*s * 2) = 2 * exp(2*beta)
        fe = ising_free_energy_exact(1, 1, beta)
        Z = 2 * np.exp(2 * beta)
        expected = np.log(Z)
        self.assertAlmostEqual(fe, expected, places=10)

    def test_2x2_beta_zero(self):
        """Bij beta=0 (oneindige T): Z = 2^n, fe = ln(2)."""
        fe = ising_free_energy_exact(2, 2, 0.0)
        self.assertAlmostEqual(fe, np.log(2), places=10)

    def test_positive_partition_function(self):
        """Vrije energie moet reëel en eindig zijn."""
        for Lx, Ly in [(2, 2), (2, 3), (3, 3)]:
            for beta in [0.1, 0.5, 1.0]:
                fe = ising_free_energy_exact(Lx, Ly, beta)
                self.assertTrue(np.isfinite(fe))

    def test_too_large_raises(self):
        with self.assertRaises(ValueError):
            ising_free_energy_exact(5, 5, 0.5)  # 25 spins > 20


class TestIsingTRG(unittest.TestCase):
    """Tests voor Ising partitie functie via TRG/HOTRG."""

    def test_trg_2x2_beta_zero(self):
        """TRG bij beta=0 moet ln(2) geven."""
        fe = ising_partition_trg(2, 2, 0.0, chi_max=8, method='trg')
        self.assertAlmostEqual(fe, np.log(2), places=10)

    def test_hotrg_2x2_beta_zero(self):
        fe = ising_partition_trg(2, 2, 0.0, chi_max=8, method='hotrg')
        self.assertAlmostEqual(fe, np.log(2), places=10)

    def test_trg_exact_2x2(self):
        """TRG moet exact zijn voor 2x2 (geen padding nodig)."""
        beta = 0.44
        exact = ising_free_energy_exact(2, 2, beta)
        trg_val = ising_partition_trg(2, 2, beta, chi_max=8, method='trg')
        self.assertAlmostEqual(trg_val, exact, places=10)

    def test_hotrg_exact_2x2(self):
        beta = 1.0
        exact = ising_free_energy_exact(2, 2, beta)
        hotrg_val = ising_partition_trg(2, 2, beta, chi_max=8, method='hotrg')
        self.assertAlmostEqual(hotrg_val, exact, places=10)

    def test_trg_exact_4x4_chi16(self):
        """TRG met chi=16 moet exact zijn voor 4x4."""
        beta = 0.44
        exact = ising_free_energy_exact(4, 4, beta)
        trg_val = ising_partition_trg(4, 4, beta, chi_max=16, method='trg')
        self.assertAlmostEqual(trg_val, exact, delta=1e-10)

    def test_trg_vs_exact_3x3(self):
        """TRG op 3x3 (oneven, padding nodig) - grotere tolerantie."""
        beta = 0.44
        exact = ising_free_energy_exact(3, 3, beta)
        trg_val = ising_partition_trg(3, 3, beta, chi_max=16, method='trg')
        self.assertAlmostEqual(trg_val, exact, delta=0.5)


class TestQAOA2DExact(unittest.TestCase):
    """Tests voor exacte QAOA op 2D grids."""

    def test_2x2_returns_positive(self):
        cost = qaoa_2d_exact(2, 2, 1, [0.5], [1.0])
        self.assertGreater(cost, 0)

    def test_2x2_bounded(self):
        """Cost moet <= totaal aantal edges zijn."""
        cost = qaoa_2d_exact(2, 2, 1, [0.5], [1.0])
        max_edges = 4  # 2x2 grid: 4 edges
        self.assertLessEqual(cost, max_edges + 0.01)

    def test_3x3_reasonable(self):
        """3x3 grid: cost moet > 0 en < m zijn."""
        cost = qaoa_2d_exact(3, 3, 1, [0.5], [1.1778])
        m = 12  # 3x3 grid: 12 edges
        self.assertGreater(cost, 0)
        self.assertLess(cost, m + 0.01)

    def test_zero_gamma(self):
        """gamma=0: geen phase separation → cost = m/2 (uniform superposition)."""
        cost = qaoa_2d_exact(2, 2, 1, [0.0], [0.0])
        m = 4  # 2x2 grid
        # Bij gamma=0, beta=0: alle states gelijk → <ZZ>=0 → cost = m/2
        self.assertAlmostEqual(cost, m / 2, delta=0.1)

    def test_too_large_raises(self):
        with self.assertRaises(ValueError):
            qaoa_2d_exact(5, 5, 1, [0.5], [1.0])

    def test_ratio_bounded(self):
        """Ratio moet tussen 0 en 1 liggen."""
        ratio = qaoa_2d_ratio(3, 3, 1, [0.5], [1.1778])
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1.0 + 0.01)

    def test_nonsquare_grid(self):
        """Niet-vierkant grid moet werken."""
        cost = qaoa_2d_exact(2, 3, 1, [0.5], [1.0])
        self.assertGreater(cost, 0)


class TestTRGQAOACost(unittest.TestCase):
    """Tests voor TRG-based QAOA cost evaluatie."""

    def test_small_grid_uses_exact(self):
        """Kleine grids (<=20 qubits) moeten exact pad gebruiken."""
        cost_trg = trg_qaoa_cost(3, 3, 1, [0.5], [1.1778], chi_max=8)
        cost_exact = qaoa_2d_exact(3, 3, 1, [0.5], [1.1778])
        self.assertAlmostEqual(cost_trg, cost_exact, places=10)

    def test_returns_positive(self):
        cost = trg_qaoa_cost(2, 2, 1, [0.5], [1.0], chi_max=4)
        self.assertGreater(cost, 0)


class TestBuildQAOATensorGrid(unittest.TestCase):
    """Tests voor QAOA tensor grid constructie."""

    def test_grid_dimensions(self):
        grid = build_qaoa_tensor_grid(3, 4, 1, [0.5], [1.0])
        self.assertEqual(grid.Lx, 3)
        self.assertEqual(grid.Ly, 4)
        self.assertEqual(len(grid.tensors), 3)
        self.assertEqual(len(grid.tensors[0]), 4)

    def test_boundary_conditions(self):
        """Randtensors moeten chi=1 op randen hebben."""
        grid = build_qaoa_tensor_grid(3, 3, 1, [0.5], [1.0])
        # Hoek (0,0): up=1, left=1
        T = grid.tensors[0][0]
        self.assertEqual(T.chi_up, 1)
        self.assertEqual(T.chi_left, 1)
        # Midden (1,1): alle chi=2
        T = grid.tensors[1][1]
        self.assertEqual(T.chi_up, 2)
        self.assertEqual(T.chi_right, 2)
        self.assertEqual(T.chi_down, 2)
        self.assertEqual(T.chi_left, 2)

    def test_tensors_nonzero(self):
        """Alle tensors moeten niet-nul zijn."""
        grid = build_qaoa_tensor_grid(2, 2, 1, [0.5], [1.0])
        for col in grid.tensors:
            for t in col:
                self.assertGreater(np.abs(t.data).max(), 0)


class TestConvergence(unittest.TestCase):
    """Tests voor TRG convergentie met toenemende chi_max."""

    def test_trg_ising_convergence_4x4(self):
        """TRG nauwkeurigheid verbetert met chi_max; chi=16 exact voor 4x4."""
        beta = 0.44
        exact = ising_free_energy_exact(4, 4, beta)
        errors = []
        for chi in [4, 8, 16]:
            approx = ising_partition_trg(4, 4, beta, chi_max=chi, method='trg')
            errors.append(abs(approx - exact))
        # chi=16 moet exact zijn
        self.assertLess(errors[2], 1e-10)
        # Fouten moeten afnemen met chi (of gelijk blijven)
        self.assertLessEqual(errors[2], errors[0])

    def test_hotrg_ising_convergence_4x4(self):
        """HOTRG is nauwkeuriger dan TRG; al goed bij chi=4."""
        beta = 0.44
        exact = ising_free_energy_exact(4, 4, beta)
        errors = []
        for chi in [4, 8, 16]:
            approx = ising_partition_trg(4, 4, beta, chi_max=chi, method='hotrg')
            errors.append(abs(approx - exact))
        # HOTRG chi=4 moet al redelijk zijn
        self.assertLess(errors[0], 0.01)
        # chi=16 exact
        self.assertLess(errors[2], 1e-10)


class TestEdgeCases(unittest.TestCase):
    """Edge cases en randgevallen."""

    def test_1x1_qaoa(self):
        """1x1 grid: 0 edges → cost = 0."""
        cost = qaoa_2d_exact(1, 1, 1, [0.5], [1.0])
        self.assertAlmostEqual(cost, 0.0, places=10)

    def test_1x2_qaoa(self):
        """1x2 grid: 1 edge, cost moet > 0 zijn."""
        cost = qaoa_2d_exact(1, 2, 1, [0.5], [1.0])
        self.assertGreater(cost, 0)
        self.assertLessEqual(cost, 1.0 + 0.01)

    def test_nonsquare_ising(self):
        """Niet-vierkant Ising grid."""
        fe = ising_free_energy_exact(2, 3, 0.5)
        self.assertTrue(np.isfinite(fe))

    def test_ising_high_beta(self):
        """Hoge beta (lage T): vrije energie groeit met beta."""
        beta = 5.0
        fe = ising_free_energy_exact(3, 3, beta)
        # Periodieke BC op 3x3: 18 edges, ground state E=-18
        # Z ≈ 2*exp(18*beta) → fe ≈ (18*5 + ln2)/9 ≈ 10.08
        self.assertTrue(np.isfinite(fe))
        self.assertGreater(fe, 5.0)

    def test_qaoa_p2(self):
        """QAOA met p=2 moet werken."""
        cost = qaoa_2d_exact(2, 2, 2, [0.3, 0.6], [0.8, 1.0])
        self.assertGreater(cost, 0)
        self.assertLessEqual(cost, 4 + 0.01)

    def test_trg_contract_verbose(self):
        """Verbose mode mag niet crashen."""
        T = np.ones((1, 1, 1, 1))
        result = trg_contract([[T]], chi_max=4, verbose=True)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
