#!/usr/bin/env python3
"""Tests voor b12_octonion_spinor — octonion-spinor correspondentie."""
from __future__ import annotations

import itertools
import unittest

import numpy as np

from b12_octonion_spinor import (
    BASIS_NAMES,
    L_matrix,
    R_matrix,
    anticommutator,
    associator,
    basis_vector,
    clifford_algebra_dim,
    clifford_generators_7d,
    clifford_metric,
    commutator,
    fock_annihilation,
    fock_creation,
    fock_number,
    moufang_identity_left,
    moufang_identity_right,
    phi_bijection,
    triality_discrepancy,
    zorn_conjugate,
    zorn_identity,
    zorn_mul,
    zorn_norm,
    zorn_trace,
)


TOL = 1e-10


# ============================================================
#  1. Zorn algebra basics
# ============================================================


class TestZornAlgebra(unittest.TestCase):

    def test_identity(self) -> None:
        one = zorn_identity()
        for k in range(8):
            v = basis_vector(k)
            self.assertTrue(np.allclose(zorn_mul(one, v), v))
            self.assertTrue(np.allclose(zorn_mul(v, one), v))

    def test_conjugate_involution(self) -> None:
        # (A*)* = A
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = rng.normal(size=8)
            self.assertTrue(np.allclose(zorn_conjugate(zorn_conjugate(A)), A))

    def test_conjugate_antihom(self) -> None:
        # (AB)* = B* A*
        rng = np.random.default_rng(7)
        for _ in range(10):
            A = rng.normal(size=8); B = rng.normal(size=8)
            lhs = zorn_conjugate(zorn_mul(A, B))
            rhs = zorn_mul(zorn_conjugate(B), zorn_conjugate(A))
            self.assertTrue(np.allclose(lhs, rhs))

    def test_norm_multiplicative(self) -> None:
        # N(AB) = N(A)·N(B)  — kernmerk van composition-algebras
        rng = np.random.default_rng(11)
        for _ in range(10):
            A = rng.normal(size=8); B = rng.normal(size=8)
            self.assertAlmostEqual(
                zorn_norm(zorn_mul(A, B)),
                zorn_norm(A) * zorn_norm(B),
                delta=1e-8)

    def test_norm_signature_split(self) -> None:
        # Signatuur (4,4): 4 positieve, 4 negatieve eigenwaarden in N-kwadraat-vorm
        # N(a,α,β,b) = ab - α·β ⇒ bilinear form matrix:
        #   [[0, 0, 0, 0, 0, 0, 0, 1/2],
        #    [0, -1/2, 0, 0, ... voor α·β],  etc.
        # Maar simpeler: build symm. bilinear form en tel signs.
        B = np.zeros((8, 8))
        B[0, 7] = B[7, 0] = 0.5
        for i in range(3):
            B[1 + i, 4 + i] = B[4 + i, 1 + i] = -0.5
        w = np.linalg.eigvalsh(B)
        pos = int(np.sum(w > 1e-9))
        neg = int(np.sum(w < -1e-9))
        self.assertEqual(pos, 4)
        self.assertEqual(neg, 4)

    def test_trace(self) -> None:
        self.assertAlmostEqual(zorn_trace(zorn_identity()), 2.0)
        self.assertAlmostEqual(zorn_trace(basis_vector(0)), 1.0)
        self.assertAlmostEqual(zorn_trace(basis_vector(7)), 1.0)
        for k in range(1, 7):
            self.assertAlmostEqual(zorn_trace(basis_vector(k)), 0.0)


# ============================================================
#  2. Peirce-decompositie (orthogonale idempotenten)
# ============================================================


class TestPeirceDecomposition(unittest.TestCase):

    def test_idempotents(self) -> None:
        e0 = basis_vector(0); e7 = basis_vector(7)
        self.assertTrue(np.allclose(zorn_mul(e0, e0), e0))
        self.assertTrue(np.allclose(zorn_mul(e7, e7), e7))

    def test_orthogonal(self) -> None:
        e0 = basis_vector(0); e7 = basis_vector(7)
        self.assertTrue(np.allclose(zorn_mul(e0, e7), np.zeros(8)))
        self.assertTrue(np.allclose(zorn_mul(e7, e0), np.zeros(8)))

    def test_sum_is_identity(self) -> None:
        self.assertTrue(np.allclose(
            basis_vector(0) + basis_vector(7), zorn_identity()))

    def test_norms(self) -> None:
        # Idempotente elementen zijn isotroop: N(e0) = N(e7) = 0
        # (want a·b = 0 voor (1,0,0,0) en (0,0,0,1))
        self.assertAlmostEqual(zorn_norm(basis_vector(0)), 0.0)
        self.assertAlmostEqual(zorn_norm(basis_vector(7)), 0.0)


# ============================================================
#  3. Nilpotente imaginairen en fermion-achtige relaties
# ============================================================


class TestNilpotentImaginaries(unittest.TestCase):

    def test_all_six_squares_zero(self) -> None:
        for k in range(1, 7):
            jk = basis_vector(k)
            self.assertTrue(np.allclose(zorn_mul(jk, jk), np.zeros(8)),
                            f"j{k}² moet 0 zijn in 𝕆_s")

    def test_alpha_beta_mode_pair_anticommutator_is_one(self) -> None:
        # {j_i, j_{i+3}} = 1 (identity)  voor i=1,2,3 — de fermionische mode-pair
        one = zorn_identity()
        for i in (1, 2, 3):
            ji = basis_vector(i)
            ji3 = basis_vector(i + 3)
            ac = zorn_mul(ji, ji3) + zorn_mul(ji3, ji)
            self.assertTrue(np.allclose(ac, one),
                            f"(j{i} j{i+3} + j{i+3} j{i}) moet identity zijn")

    def test_same_subspace_anticommutator_is_zero_vector(self) -> None:
        # {j_i, j_j} = 0 voor i≠j binnen dezelfde α- of β-kolom
        for i, j in itertools.combinations((1, 2, 3), 2):
            ac = zorn_mul(basis_vector(i), basis_vector(j)) + \
                 zorn_mul(basis_vector(j), basis_vector(i))
            self.assertTrue(np.allclose(ac, np.zeros(8)))
        for i, j in itertools.combinations((4, 5, 6), 2):
            ac = zorn_mul(basis_vector(i), basis_vector(j)) + \
                 zorn_mul(basis_vector(j), basis_vector(i))
            self.assertTrue(np.allclose(ac, np.zeros(8)))

    def test_alpha_cross_product_is_beta(self) -> None:
        # j_i · j_j = -j_{k+3} voor (i,j,k) Levi-Civita
        pairs = [(1, 2, 6), (2, 3, 4), (3, 1, 5)]
        for i, j, kbeta in pairs:
            prod = zorn_mul(basis_vector(i), basis_vector(j))
            expect = -basis_vector(kbeta)
            self.assertTrue(np.allclose(prod, expect),
                            f"j{i}·j{j} moet -j{kbeta} zijn")


# ============================================================
#  4. Non-associativity (associator)
# ============================================================


class TestAssociator(unittest.TestCase):

    def test_associative_on_same_element(self) -> None:
        # [A, A, A] = 0 automatisch; [A, A, B] = 0 (alternatief)
        rng = np.random.default_rng(3)
        for _ in range(5):
            A = rng.normal(size=8); B = rng.normal(size=8)
            self.assertTrue(np.allclose(associator(A, A, B), np.zeros(8)),
                            "alternatief: [A,A,B]=0")
            self.assertTrue(np.allclose(associator(A, B, B), np.zeros(8)),
                            "alternatief: [A,B,B]=0")

    def test_associator_antisymmetric(self) -> None:
        # [A,B,C] is totaal antisymmetrisch in drie argumenten
        rng = np.random.default_rng(5)
        A = rng.normal(size=8); B = rng.normal(size=8); C = rng.normal(size=8)
        ass_ABC = associator(A, B, C)
        ass_BAC = associator(B, A, C)
        self.assertTrue(np.allclose(ass_ABC, -ass_BAC))
        ass_ACB = associator(A, C, B)
        self.assertTrue(np.allclose(ass_ABC, -ass_ACB))

    def test_genuinely_non_associative(self) -> None:
        # Er bestaan triples met [A,B,C] ≠ 0
        j1 = basis_vector(1); j2 = basis_vector(2); j3 = basis_vector(3)
        a = associator(j1, j2, j3)
        self.assertGreater(np.linalg.norm(a), 0.1)
        # Specifiek: (j1 j2) j3 = -e7, j1 (j2 j3) = -e0 ⇒ [j1,j2,j3] = e0 - e7
        expected = basis_vector(0) - basis_vector(7)
        self.assertTrue(np.allclose(a, expected))


# ============================================================
#  5. Moufang-identiteiten (alternatief-algebra bewijs)
# ============================================================


class TestMoufangIdentities(unittest.TestCase):

    def test_left_moufang_basis(self) -> None:
        # A(B(AC)) = ((AB)A)C  op alle basis-triples
        for (i, j, k) in itertools.product(range(8), repeat=3):
            r = moufang_identity_left(
                basis_vector(i), basis_vector(j), basis_vector(k))
            self.assertLess(r, TOL, f"left-Moufang faalt bij ({i},{j},{k})")

    def test_right_moufang_basis(self) -> None:
        for (i, j, k) in itertools.product(range(8), repeat=3):
            r = moufang_identity_right(
                basis_vector(i), basis_vector(j), basis_vector(k))
            self.assertLess(r, TOL, f"right-Moufang faalt bij ({i},{j},{k})")

    def test_moufang_random(self) -> None:
        rng = np.random.default_rng(13)
        for _ in range(10):
            A, B, C = rng.normal(size=(3, 8))
            self.assertLess(moufang_identity_left(A, B, C), 1e-9)
            self.assertLess(moufang_identity_right(A, B, C), 1e-9)


# ============================================================
#  6. Links- en rechts-multiplication matrices
# ============================================================


class TestLRMatrices(unittest.TestCase):

    def test_L_applied_equals_product(self) -> None:
        rng = np.random.default_rng(17)
        for _ in range(5):
            a = rng.normal(size=8); x = rng.normal(size=8)
            self.assertTrue(np.allclose(L_matrix(a) @ x, zorn_mul(a, x)))

    def test_R_applied_equals_product(self) -> None:
        rng = np.random.default_rng(19)
        for _ in range(5):
            a = rng.normal(size=8); x = rng.normal(size=8)
            self.assertTrue(np.allclose(R_matrix(a) @ x, zorn_mul(x, a)))

    def test_L_alpha_beta_pair_anticommutator_is_I(self) -> None:
        # {L_{j_i}, L_{j_{i+3}}} = I_8 exact
        for i in (1, 2, 3):
            L1 = L_matrix(basis_vector(i))
            L2 = L_matrix(basis_vector(i + 3))
            AC = anticommutator(L1, L2)
            self.assertTrue(np.allclose(AC, np.eye(8)),
                            f"{{L_j{i}, L_j{i+3}}} moet I_8 zijn")


# ============================================================
#  7. Clifford gamma-matrices Cl(4,3)
# ============================================================


class TestCliffordGenerators(unittest.TestCase):

    def setUp(self) -> None:
        self.gens = clifford_generators_7d()

    def test_count_and_shape(self) -> None:
        self.assertEqual(len(self.gens), 7)
        for g in self.gens:
            self.assertEqual(g.shape, (8, 8))

    def test_metric_signature(self) -> None:
        eta = clifford_metric(self.gens)
        diag = np.diag(eta)
        # Verwachting: (+1,+1,+1,-1,-1,-1,+1) = signatuur (4,3)
        np.testing.assert_array_equal(diag.astype(int), [1, 1, 1, -1, -1, -1, 1])

    def test_metric_diagonal(self) -> None:
        # Alle off-diagonalen exact 0 ⇒ orthogonale Clifford-basis
        eta = clifford_metric(self.gens)
        off = eta - np.diag(np.diag(eta))
        self.assertEqual(np.max(np.abs(off)), 0.0)

    def test_anticommutators_scalar(self) -> None:
        # Voor elk paar moet {γ_μ, γ_ν} = 2 η_μν I_8 gelden
        eta = clifford_metric(self.gens)
        for i in range(7):
            for j in range(7):
                AC = anticommutator(self.gens[i], self.gens[j])
                self.assertTrue(np.allclose(AC, 2 * eta[i, j] * np.eye(8)))


# ============================================================
#  8. Fermionische Fock-ruimte F_3
# ============================================================


class TestFockAlgebra(unittest.TestCase):

    def test_creation_squares_zero(self) -> None:
        for i in (1, 2, 3):
            c = fock_creation(i)
            self.assertTrue(np.allclose(c @ c, np.zeros((8, 8))))

    def test_annihilation_squares_zero(self) -> None:
        for i in (1, 2, 3):
            c = fock_annihilation(i)
            self.assertTrue(np.allclose(c @ c, np.zeros((8, 8))))

    def test_canonical_anticommutator(self) -> None:
        # {c_i, c_j†} = δ_ij · I
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                AC = fock_annihilation(i) @ fock_creation(j) + \
                     fock_creation(j) @ fock_annihilation(i)
                expected = (1.0 if i == j else 0.0) * np.eye(8)
                self.assertTrue(np.allclose(AC, expected))

    def test_creation_anticommute(self) -> None:
        # {c_i†, c_j†} = 0
        for i, j in itertools.combinations((1, 2, 3), 2):
            AC = fock_creation(i) @ fock_creation(j) + \
                 fock_creation(j) @ fock_creation(i)
            self.assertTrue(np.allclose(AC, np.zeros((8, 8))))

    def test_number_operator(self) -> None:
        # n_i |σ⟩ = 1·|σ⟩ als i∈σ anders 0
        from b12_octonion_spinor import FOCK_BASIS, FOCK_INDEX
        for i in (1, 2, 3):
            n = fock_number(i)
            for col, sigma in enumerate(FOCK_BASIS):
                v = np.zeros(8); v[col] = 1.0
                r = n @ v
                if i in sigma:
                    self.assertTrue(np.allclose(r, v))
                else:
                    self.assertTrue(np.allclose(r, np.zeros(8)))


# ============================================================
#  9. Φ bijection: lineaire iso, GEEN algebra-iso
# ============================================================


class TestPhiBijection(unittest.TestCase):

    def test_is_orthogonal(self) -> None:
        Phi = phi_bijection()
        self.assertTrue(np.allclose(Phi @ Phi.T, np.eye(8)))
        self.assertAlmostEqual(abs(np.linalg.det(Phi)), 1.0)

    def test_not_module_morphism_for_creation(self) -> None:
        # Φ is een VECTOR-RUIMTE iso; het is GEEN module-iso.
        # Bewijs: Φ · L_{j_1} ≠ c_1† · Φ
        Phi = phi_bijection()
        L1 = L_matrix(basis_vector(1))
        c1d = fock_creation(1)
        discrep = np.linalg.norm(Phi @ L1 - c1d @ Phi)
        self.assertGreater(discrep, 1e-3,
                           "Φ-intertwiner zou moeten falen (non-associativiteit)")

    def test_vacuum_mapped_to_fock_vacuum(self) -> None:
        # Φ(e_0) = |∅⟩ (Fock index 0)
        Phi = phi_bijection()
        e0 = basis_vector(0)
        expected = np.zeros(8); expected[0] = 1.0
        self.assertTrue(np.allclose(Phi @ e0, expected))

    def test_e7_mapped_to_top_fock_with_sign(self) -> None:
        # Φ(e_7) = -|123⟩
        Phi = phi_bijection()
        e7 = basis_vector(7)
        expected = np.zeros(8); expected[7] = -1.0
        self.assertTrue(np.allclose(Phi @ e7, expected))


# ============================================================
#  10. Falsificatie van Cl(4,4) ≅ 𝕆_s
# ============================================================


class TestClaimFalsifications(unittest.TestCase):

    def test_dim_cl44_vs_octonion(self) -> None:
        # Cl(4,4) heeft dim 2^8 = 256; split-octonionen zijn 8-dim
        self.assertEqual(clifford_algebra_dim(4, 4), 256)
        self.assertNotEqual(clifford_algebra_dim(4, 4), 8)

    def test_correct_is_cl43_spinor_module(self) -> None:
        # Cl(4,3) heeft dim 2^7 = 128; zijn spinor-module heeft dim 2^⌊7/2⌋ = 8.
        self.assertEqual(clifford_algebra_dim(4, 3), 128)
        spinor_dim = 2 ** (7 // 2)
        self.assertEqual(spinor_dim, 8)
        # En: we hebben expliciet 7 gamma-matrices op 𝕆_s (dim 8) geconstrueerd
        gens = clifford_generators_7d()
        self.assertEqual(len(gens), 7)
        self.assertEqual(gens[0].shape, (8, 8))


# ============================================================
#  11. Triality-indicatie: L ≠ R
# ============================================================


class TestTrialityIndicator(unittest.TestCase):

    def test_L_R_differ_for_imaginary(self) -> None:
        # Voor imaginaire basiselementen moeten L en R verschillen
        diffs = triality_discrepancy()
        self.assertGreater(diffs["__total_L_minus_R__"], 0.1,
                           "L en R moeten verschillen op Im(𝕆_s)")

    def test_L_R_transpose_related_for_some_elements(self) -> None:
        # Een bekende structurele regel: L(a) + L(a)^T ≠ R(a) + R(a)^T
        # Voor idempotenten zijn L en R sowieso zelf-geadjungeerd op hun subruimte.
        # Voor nilpotente j's verwachten we duidelijke verschillen.
        L1 = L_matrix(basis_vector(1))
        R1 = R_matrix(basis_vector(1))
        self.assertGreater(np.linalg.norm(L1 - R1), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
