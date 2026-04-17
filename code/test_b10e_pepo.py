#!/usr/bin/env python3
"""Tests voor B10e PEPO / PEPS 2D tensor netwerk.

Suites:
  1. Pauli / gate primitives
  2. PEPS2D constructors + bookkeeping
  3. Single-site gate applicatie
  4. Two-site gate applicatie (horizontaal + verticaal)
  5. Boundary-MPO expectation values (trivially)
  6. PEPS vs exact state-vector (rigoreus)
  7. 2D MaxCut QAOA correctness
  8. Bond-dimensie controle (chi_max truncatie)
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b10e_pepo import (
    PEPS2D, Rx, ZZg, I2, Z_MAT, X_MAT, H_MAT,
    expectation_value, exact_plus_state, exact_qaoa_maxcut,
    apply_single_sv, apply_zz_sv,
    peps_qaoa_maxcut, grid_edges, grid_edges_flat,
)


# ============================================================
# Suite 1: Primitives
# ============================================================

class TestPrimitives(unittest.TestCase):

    def test_rx_is_unitary(self):
        for theta in [0.0, 0.3, 1.2, np.pi, 2 * np.pi]:
            U = Rx(theta)
            self.assertTrue(np.allclose(U @ U.conj().T, np.eye(2)))

    def test_rx_at_zero_is_identity(self):
        self.assertTrue(np.allclose(Rx(0.0), np.eye(2)))

    def test_zzg_is_diagonal_unitary(self):
        g = ZZg(0.4)
        g4 = g.reshape(4, 4)
        # Diagonaal
        off_diag = g4 - np.diag(np.diag(g4))
        self.assertTrue(np.allclose(off_diag, 0.0))
        # Unitair
        self.assertTrue(np.allclose(g4 @ g4.conj().T, np.eye(4)))

    def test_zzg_at_zero_is_identity(self):
        self.assertTrue(np.allclose(ZZg(0.0).reshape(4, 4), np.eye(4)))


# ============================================================
# Suite 2: Constructors
# ============================================================

class TestConstructors(unittest.TestCase):

    def test_plus_state_shape(self):
        p = PEPS2D.plus_state(3, 2)
        self.assertEqual(p.Lx, 3)
        self.assertEqual(p.Ly, 2)
        for x in range(3):
            for y in range(2):
                self.assertEqual(p.T[x][y].shape, (1, 1, 1, 1, 2))

    def test_zero_state_amplitude(self):
        p = PEPS2D.zero_state(2, 2)
        # Alle tensors moeten op |0⟩ staan
        for x in range(2):
            for y in range(2):
                v = p.T[x][y].reshape(2)
                self.assertAlmostEqual(abs(v[0]), 1.0)
                self.assertAlmostEqual(abs(v[1]), 0.0)

    def test_from_product_custom_vec(self):
        v = np.array([0.6, 0.8j], dtype=complex)
        p = PEPS2D.from_product_vec(2, 2, v)
        for x in range(2):
            for y in range(2):
                out = p.T[x][y].reshape(2)
                self.assertTrue(np.allclose(out, v))

    def test_copy_independence(self):
        p = PEPS2D.plus_state(2, 2)
        q = p.copy()
        q.T[0][0] *= 2.0
        self.assertFalse(np.allclose(p.T[0][0], q.T[0][0]))

    def test_max_bond_dim_initial(self):
        p = PEPS2D.plus_state(3, 3)
        self.assertEqual(p.max_bond_dim(), 1)


# ============================================================
# Suite 3: Single-site gates
# ============================================================

class TestSingleSiteGates(unittest.TestCase):

    def test_hadamard_on_zero_yields_plus(self):
        p = PEPS2D.zero_state(2, 2)
        p.apply_single(0, 0, H_MAT)
        v = p.T[0][0].reshape(2)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        self.assertTrue(np.allclose(v, expected))

    def test_x_gate_flips(self):
        p = PEPS2D.zero_state(2, 2)
        p.apply_single(1, 1, X_MAT)
        v = p.T[1][1].reshape(2)
        self.assertTrue(np.allclose(v, np.array([0, 1], dtype=complex)))

    def test_single_gate_does_not_touch_other_sites(self):
        p = PEPS2D.zero_state(2, 2)
        p.apply_single(0, 0, X_MAT)
        for (x, y) in [(0, 1), (1, 0), (1, 1)]:
            v = p.T[x][y].reshape(2)
            self.assertTrue(np.allclose(v, np.array([1, 0], dtype=complex)))


# ============================================================
# Suite 4: Two-site gates
# ============================================================

class TestTwoSiteGates(unittest.TestCase):

    def test_horizontal_identity_preserves_state(self):
        p = PEPS2D.plus_state(3, 2, chi_max=4)
        pbefore = [[p.T[x][y].copy() for y in range(2)] for x in range(3)]
        I4 = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
        p.apply_two_horizontal(0, 0, I4)
        # Bond-dim blijft 1 omdat identity rank-1 is over de gedeelde bond
        self.assertEqual(p.max_bond_dim(), 1)
        # Norm blijft gelijk
        norm_before = expectation_value(
            PEPS2D.plus_state(3, 2), ops=None, chi_b=8)
        norm_after = expectation_value(p, ops=None, chi_b=8)
        self.assertAlmostEqual(abs(norm_before - norm_after), 0.0, places=10)

    def test_vertical_identity_preserves_state(self):
        p = PEPS2D.plus_state(3, 3, chi_max=4)
        I4 = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
        p.apply_two_vertical(1, 0, I4)
        self.assertEqual(p.max_bond_dim(), 1)

    def test_horizontal_zz_increases_bond_dim(self):
        """ZZ(γ=0.4) op |+⟩|+⟩ zou bond-dim naar 2 moeten tillen."""
        p = PEPS2D.plus_state(2, 1, chi_max=4)
        zz = ZZg(0.4)
        p.apply_two_horizontal(0, 0, zz)
        # De bond tussen (0,0) en (1,0) moet nu chi=2 zijn
        self.assertEqual(p.T[0][0].shape[1], 2)
        self.assertEqual(p.T[1][0].shape[0], 2)

    def test_horizontal_out_of_bounds_raises(self):
        p = PEPS2D.plus_state(2, 2, chi_max=4)
        zz = ZZg(0.4)
        with self.assertRaises(ValueError):
            p.apply_two_horizontal(1, 0, zz)  # x+1=2 > Lx-1

    def test_vertical_out_of_bounds_raises(self):
        p = PEPS2D.plus_state(2, 2, chi_max=4)
        zz = ZZg(0.4)
        with self.assertRaises(ValueError):
            p.apply_two_vertical(0, 1, zz)  # y+1=2 > Ly-1


# ============================================================
# Suite 5: Boundary MPO expectation
# ============================================================

class TestBoundaryMPO(unittest.TestCase):

    def test_plus_state_norm_is_one(self):
        """⟨ψ|ψ⟩ met ψ = |+⟩^⊗n."""
        for Lx, Ly in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            p = PEPS2D.plus_state(Lx, Ly)
            val = expectation_value(p, ops=None, chi_b=8)
            self.assertAlmostEqual(val.real, 1.0, places=9)
            self.assertAlmostEqual(val.imag, 0.0, places=9)

    def test_zero_state_z_expectation(self):
        """⟨0...0|Z_i|0...0⟩ = 1 voor elke i."""
        p = PEPS2D.zero_state(2, 2)
        for x in range(2):
            for y in range(2):
                val = expectation_value(p, ops={(x, y): Z_MAT}, chi_b=8)
                self.assertAlmostEqual(val.real, 1.0, places=9)

    def test_plus_state_z_expectation_is_zero(self):
        """⟨+|Z|+⟩ = 0 op elke site."""
        p = PEPS2D.plus_state(3, 2)
        for x in range(3):
            for y in range(2):
                val = expectation_value(p, ops={(x, y): Z_MAT}, chi_b=8)
                self.assertAlmostEqual(val.real, 0.0, places=9)

    def test_plus_state_x_expectation_is_one(self):
        """⟨+|X|+⟩ = 1."""
        p = PEPS2D.plus_state(2, 3)
        for x in range(2):
            for y in range(3):
                val = expectation_value(p, ops={(x, y): X_MAT}, chi_b=8)
                self.assertAlmostEqual(val.real, 1.0, places=9)


# ============================================================
# Suite 6: PEPS vs exact state-vector
# ============================================================

class TestPepsVsExact(unittest.TestCase):

    def _check(self, Lx, Ly, gammas, betas, chi_max=8, chi_b=16,
              tol=1e-6):
        edges_flat = grid_edges_flat(Lx, Ly)
        _, E_exact = exact_qaoa_maxcut(Lx, Ly, edges_flat, gammas, betas)
        _, E_peps = peps_qaoa_maxcut(Lx, Ly, gammas, betas,
                                     chi_max=chi_max, chi_b=chi_b)
        self.assertAlmostEqual(E_peps, E_exact, delta=tol,
                               msg=f"{Lx}x{Ly} p={len(gammas)}: "
                                   f"exact={E_exact} peps={E_peps}")

    def test_2x2_p1(self):
        self._check(2, 2, [0.4], [0.3])

    def test_2x2_p2(self):
        self._check(2, 2, [0.4, 0.2], [0.3, 0.1])

    def test_3x2_p1(self):
        self._check(3, 2, [0.4], [0.3])

    def test_3x2_p2(self):
        self._check(3, 2, [0.4, 0.2], [0.3, 0.1])

    def test_3x3_p1(self):
        self._check(3, 3, [0.4], [0.3], chi_max=4, chi_b=16)

    def test_zero_gamma_keeps_plus_state_energy(self):
        """Met γ=0 is de state |+⟩^⊗n en ⟨H_C⟩=|E|/2 voor MaxCut."""
        Lx, Ly = 3, 3
        _, E_peps = peps_qaoa_maxcut(Lx, Ly, [0.0], [0.3],
                                     chi_max=2, chi_b=8)
        # |+⟩: ⟨Z_i Z_j⟩ = 0 voor i≠j → ⟨H_C⟩ = |E| * 0.5
        n_edges = len(grid_edges(Lx, Ly))
        self.assertAlmostEqual(E_peps, 0.5 * n_edges, places=5)


# ============================================================
# Suite 7: State-vector reference sanity
# ============================================================

class TestStateVectorReference(unittest.TestCase):

    def test_apply_single_is_unitary(self):
        n = 3
        state = exact_plus_state(n)
        state2 = apply_single_sv(state, n, 1, X_MAT)
        self.assertAlmostEqual(np.linalg.norm(state2), 1.0, places=10)

    def test_zz_phase_on_product_state(self):
        """e^{-iγZZ}|0..0⟩ krijgt factor e^{-iγ} (parity 0)."""
        n = 3
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
        gamma = 0.3
        state2 = apply_zz_sv(state, n, 0, 1, gamma)
        self.assertAlmostEqual(state2[0], np.exp(-1j * gamma))

    def test_plus_state_is_superposition(self):
        n = 2
        state = exact_plus_state(n)
        self.assertTrue(np.allclose(state, [0.5, 0.5, 0.5, 0.5]))

    def test_exact_qaoa_runs(self):
        Lx, Ly = 2, 2
        edges_flat = grid_edges_flat(Lx, Ly)
        _, E = exact_qaoa_maxcut(Lx, Ly, edges_flat, [0.3], [0.2])
        self.assertTrue(0 <= E <= len(edges_flat))


# ============================================================
# Suite 8: Bond dimension control
# ============================================================

class TestBondDimensionControl(unittest.TestCase):

    def test_chi_max_is_respected_horizontal(self):
        p = PEPS2D.plus_state(2, 1, chi_max=2)
        zz = ZZg(0.4)
        # Herhaalde ZZ zou in theorie chi kunnen opdrijven, maar met chi_max=2
        # moet het geklemd blijven.
        for _ in range(5):
            p.apply_two_horizontal(0, 0, zz)
        self.assertLessEqual(p.T[0][0].shape[1], 2)

    def test_chi_max_is_respected_vertical(self):
        p = PEPS2D.plus_state(1, 2, chi_max=2)
        zz = ZZg(0.4)
        for _ in range(5):
            p.apply_two_vertical(0, 0, zz)
        self.assertLessEqual(p.T[0][0].shape[3], 2)

    def test_identity_never_grows_bond(self):
        p = PEPS2D.plus_state(3, 3, chi_max=4)
        I4 = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
        for y in range(3):
            for x in range(2):
                p.apply_two_horizontal(x, y, I4)
        for y in range(2):
            for x in range(3):
                p.apply_two_vertical(x, y, I4)
        self.assertEqual(p.max_bond_dim(), 1)


# ============================================================
# Suite 9: Grid edge helpers
# ============================================================

class TestGridEdgeHelpers(unittest.TestCase):

    def test_edge_count_2x2(self):
        edges = grid_edges(2, 2)
        # 2 horizontale + 2 verticale = 4
        self.assertEqual(len(edges), 4)

    def test_edge_count_3x3(self):
        edges = grid_edges(3, 3)
        # 2 horizontaal per rij × 3 + 2 verticaal per kolom × 3 = 12
        self.assertEqual(len(edges), 12)

    def test_flat_edges_match_shape(self):
        Lx, Ly = 3, 2
        edges_2d = grid_edges(Lx, Ly)
        edges_flat = grid_edges_flat(Lx, Ly)
        self.assertEqual(len(edges_2d), len(edges_flat))

    def test_flat_edges_snake_index_consistent(self):
        """q = y*Lx + x moet matchen tussen 2D en flat edges."""
        Lx, Ly = 3, 2
        edges_2d = grid_edges(Lx, Ly)
        edges_flat = grid_edges_flat(Lx, Ly)
        set_flat = set(edges_flat)
        for (x1, y1, x2, y2, _) in edges_2d:
            q1 = y1 * Lx + x1
            q2 = y2 * Lx + x2
            self.assertIn(tuple(sorted([q1, q2])),
                          {tuple(sorted(e)) for e in set_flat})


# ============================================================
# Suite 10: Integration / smoke
# ============================================================

class TestIntegrationSmoke(unittest.TestCase):

    def test_full_pipeline_runs(self):
        """End-to-end: PEPS QAOA 2×2 p=1 moet een redelijke energie geven."""
        _, E = peps_qaoa_maxcut(2, 2, [0.4], [0.3], chi_max=2, chi_b=8)
        # Voor 2×2 MaxCut (4 edges), OPT=4 → 0≤E≤4
        self.assertGreaterEqual(E, 0.0)
        self.assertLessEqual(E, 4.0)

    def test_exact_vs_peps_on_nontrivial_angles(self):
        """PEPS moet ook met andere angles matchen."""
        Lx, Ly = 3, 2
        for gamma in [0.1, 0.5, 1.0]:
            for beta in [0.05, 0.4, 0.7]:
                edges_flat = grid_edges_flat(Lx, Ly)
                _, Eex = exact_qaoa_maxcut(Lx, Ly, edges_flat,
                                           [gamma], [beta])
                _, Ep = peps_qaoa_maxcut(Lx, Ly, [gamma], [beta],
                                         chi_max=4, chi_b=16)
                self.assertAlmostEqual(Ep, Eex, delta=1e-6,
                                       msg=f"γ={gamma} β={beta}: "
                                           f"ex={Eex} pe={Ep}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
