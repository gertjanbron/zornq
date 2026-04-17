#!/usr/bin/env python3
"""Tests voor B160 QSVT / block-encoding framework."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b160_qsvt import (
    BlockEncoding,
    PauliSum,
    bessel_j,
    block_encode_pauli_sum,
    chebyshev_T_matrix,
    chebyshev_T_phases,
    hamiltonian_simulation_qsvt,
    jacobi_anger_truncation,
    pauli_matrix,
    prepare_unitary,
    qsp_polynomial_values,
    qsp_signal,
    qsp_unitary,
    rz,
    select_unitary,
    trotter_reference,
    verify_block_encoding,
)


# ============================================================
# 1. Pauli primitives
# ============================================================

class TestPauli(unittest.TestCase):

    def test_single_pauli(self) -> None:
        X = pauli_matrix("X")
        self.assertTrue(np.allclose(X, np.array([[0, 1], [1, 0]])))

    def test_two_qubit(self) -> None:
        XZ = pauli_matrix("XZ")
        expected = np.kron(
            np.array([[0, 1], [1, 0]]),
            np.array([[1, 0], [0, -1]]),
        )
        self.assertTrue(np.allclose(XZ, expected))

    def test_identity(self) -> None:
        II = pauli_matrix("II")
        self.assertTrue(np.allclose(II, np.eye(4)))

    def test_ham_from_pauli_sum(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[(1.0, "ZZ"), (-0.5, "IX")])
        H = ps.to_matrix()
        self.assertTrue(np.allclose(H, H.conj().T))  # Hermitisch
        self.assertEqual(H.shape, (4, 4))
        # alpha
        self.assertAlmostEqual(ps.alpha(), 1.5)

    def test_wrong_label_length_raises(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[(1.0, "X")])
        with self.assertRaises(ValueError):
            ps.to_matrix()


# ============================================================
# 2. Block-encoding correctness
# ============================================================

class TestBlockEncoding(unittest.TestCase):

    def test_single_term(self) -> None:
        ps = PauliSum(n_qubits=1, terms=[(1.0, "X")])
        be = block_encode_pauli_sum(ps)
        H = ps.to_matrix()
        self.assertTrue(verify_block_encoding(be, H))

    def test_two_term(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[(0.7, "XX"), (0.3, "ZZ")])
        be = block_encode_pauli_sum(ps)
        H = ps.to_matrix()
        self.assertTrue(verify_block_encoding(be, H))

    def test_three_term_mixed(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[
            (1.0, "XX"), (0.5, "ZZ"), (-0.3, "IX")
        ])
        be = block_encode_pauli_sum(ps)
        H = ps.to_matrix()
        self.assertTrue(verify_block_encoding(be, H))
        # alpha
        self.assertAlmostEqual(be.alpha, 1.8)

    def test_negative_coefs(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[(-1.0, "ZI"), (-0.5, "IX")])
        be = block_encode_pauli_sum(ps)
        H = ps.to_matrix()
        self.assertTrue(verify_block_encoding(be, H))

    def test_unitarity(self) -> None:
        ps = PauliSum(n_qubits=2, terms=[
            (1.0, "XX"), (0.5, "YY"), (0.3, "ZZ"), (-0.2, "IZ")
        ])
        be = block_encode_pauli_sum(ps)
        dim = be.U.shape[0]
        err = np.linalg.norm(be.U @ be.U.conj().T - np.eye(dim))
        self.assertLess(err, 1e-10)

    def test_ancilla_count(self) -> None:
        # 3 termen → ceil(log2(3)) = 2 ancillas
        ps = PauliSum(n_qubits=1, terms=[(1, "X"), (1, "Y"), (1, "Z")])
        be = block_encode_pauli_sum(ps)
        self.assertEqual(be.m_ancilla, 2)

    def test_top_left_block(self) -> None:
        ps = PauliSum(n_qubits=1, terms=[(2.0, "Z"), (1.0, "X")])
        be = block_encode_pauli_sum(ps)
        block = be.top_left_block()
        expected = ps.to_matrix() / ps.alpha()
        self.assertTrue(np.allclose(block, expected))

    def test_empty_raises(self) -> None:
        ps = PauliSum(n_qubits=1, terms=[])
        with self.assertRaises(ValueError):
            block_encode_pauli_sum(ps)

    def test_maxcut_hamiltonian(self) -> None:
        # H = -ZZ voor een enkele edge (triviale MaxCut)
        ps = PauliSum(n_qubits=2, terms=[(-0.5, "ZZ"), (0.5, "II")])
        be = block_encode_pauli_sum(ps)
        H = ps.to_matrix()
        self.assertTrue(verify_block_encoding(be, H))


# ============================================================
# 3. QSP correctness
# ============================================================

class TestQSP(unittest.TestCase):

    def test_signal_unitary(self) -> None:
        for x in [-0.7, 0.0, 0.5]:
            W = qsp_signal(x)
            self.assertTrue(np.allclose(W @ W.conj().T, np.eye(2)))

    def test_rz_unitary(self) -> None:
        for phi in [0, 0.1, np.pi / 3]:
            U = rz(phi)
            self.assertTrue(np.allclose(U @ U.conj().T, np.eye(2)))

    def test_qsp_d0(self) -> None:
        # Φ = [0]: U = Rz(0) = I, top-left = 1 voor alle x (constant polynoom)
        for x in [-0.9, 0.0, 0.9]:
            U = qsp_unitary(x, [0.0])
            self.assertAlmostEqual(U[0, 0], 1.0)

    def test_qsp_d1_identity(self) -> None:
        # Φ = [0, 0]: U = Rz(0) · W(x) · Rz(0) = W(x), top-left = x
        for x in [-0.9, -0.3, 0.0, 0.5, 0.9]:
            U = qsp_unitary(x, [0.0, 0.0])
            self.assertAlmostEqual(U[0, 0].real, x, places=10)
            self.assertAlmostEqual(U[0, 0].imag, 0.0, places=10)

    def test_qsp_chebyshev_T2(self) -> None:
        phases = chebyshev_T_phases(2)
        for x in [-0.9, -0.5, -0.1, 0.0, 0.4, 0.8]:
            U = qsp_unitary(x, phases)
            t2 = 2 * x * x - 1  # T_2(x)
            self.assertAlmostEqual(U[0, 0].real, t2, places=10,
                                   msg=f"T_2 at x={x}")

    def test_qsp_chebyshev_T3(self) -> None:
        phases = chebyshev_T_phases(3)
        for x in [-0.9, -0.5, 0.0, 0.4, 0.8]:
            U = qsp_unitary(x, phases)
            t3 = 4 * x ** 3 - 3 * x  # T_3(x)
            self.assertAlmostEqual(U[0, 0].real, t3, places=10,
                                   msg=f"T_3 at x={x}")

    def test_qsp_chebyshev_T4(self) -> None:
        phases = chebyshev_T_phases(4)
        for x in [-0.8, -0.3, 0.0, 0.5, 0.9]:
            U = qsp_unitary(x, phases)
            t4 = 8 * x ** 4 - 8 * x * x + 1  # T_4(x)
            self.assertAlmostEqual(U[0, 0].real, t4, places=9,
                                   msg=f"T_4 at x={x}")

    def test_qsp_unitary_property(self) -> None:
        phases = [0.1, 0.3, -0.2, 0.5]
        for x in [-0.5, 0.0, 0.7]:
            U = qsp_unitary(x, phases)
            self.assertTrue(np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10))

    def test_qsp_signal_x_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            qsp_signal(1.5)

    def test_qsp_empty_phases(self) -> None:
        with self.assertRaises(ValueError):
            qsp_unitary(0.5, [])


# ============================================================
# 4. Chebyshev matrix-polynoom
# ============================================================

class TestChebyshevMatrix(unittest.TestCase):

    def test_T0_is_identity(self) -> None:
        A = np.array([[0.5, 0.1], [0.1, -0.3]], dtype=complex)
        T = chebyshev_T_matrix(0, A)
        self.assertTrue(np.allclose(T, np.eye(2)))

    def test_T1_is_A(self) -> None:
        A = np.array([[0.5, 0.1], [0.1, -0.3]], dtype=complex)
        T = chebyshev_T_matrix(1, A)
        self.assertTrue(np.allclose(T, A))

    def test_T2_scalar(self) -> None:
        # T_2(x) = 2x² - 1 op eigenwaarden
        A = np.diag([0.5, -0.3]).astype(complex)
        T = chebyshev_T_matrix(2, A)
        expected = np.diag([2 * 0.25 - 1, 2 * 0.09 - 1])
        self.assertTrue(np.allclose(T, expected))

    def test_T3_pauli_X(self) -> None:
        # T_3(X) = 4 X³ - 3 X = 4 X - 3 X = X (want X² = I)
        X = pauli_matrix("X")
        T = chebyshev_T_matrix(3, X)
        self.assertTrue(np.allclose(T, X))

    def test_Tk_eigenspectrum(self) -> None:
        # Voor Hermitische A met ||A||<=1: T_k(A) heeft eigenwaarden T_k(λ_i)
        A = 0.5 * (pauli_matrix("X") + pauli_matrix("Z"))  # ‖A‖ = 1/√2
        A = A / np.linalg.norm(A, 2)  # normaliseer naar ||A||=1
        eigvals = np.linalg.eigvalsh((A + A.conj().T) / 2)
        for k in range(5):
            T = chebyshev_T_matrix(k, A)
            T_eigs = np.linalg.eigvalsh((T + T.conj().T) / 2)
            from numpy.polynomial.chebyshev import Chebyshev
            Tk_poly = Chebyshev([0] * k + [1])
            expected = sorted(Tk_poly(ev) for ev in eigvals)
            self.assertTrue(np.allclose(sorted(T_eigs.real), expected, atol=1e-8))


# ============================================================
# 5. Jacobi-Anger Hamiltonian-simulatie
# ============================================================

class TestJacobiAnger(unittest.TestCase):

    def test_bessel_j0_zero(self) -> None:
        self.assertAlmostEqual(bessel_j(0, 0.0), 1.0)

    def test_bessel_symmetries(self) -> None:
        # J_k(-x) = (-1)^k J_k(x)
        for k in range(5):
            self.assertAlmostEqual(
                bessel_j(k, -1.5), ((-1) ** k) * bessel_j(k, 1.5), places=8
            )

    def test_truncation_grows_with_tau(self) -> None:
        K1 = jacobi_anger_truncation(1.0, 1e-10)
        K2 = jacobi_anger_truncation(10.0, 1e-10)
        self.assertGreater(K2, K1)

    def test_hamiltonian_sim_single_pauli(self) -> None:
        # e^{-iXt} exact = cos(t) I - i sin(t) X
        from scipy.linalg import expm
        X = pauli_matrix("X")
        t = 0.8
        U_ja = hamiltonian_simulation_qsvt(X, t, alpha=1.0)
        U_exact = expm(-1j * X * t)
        err = np.linalg.norm(U_ja - U_exact, 2)
        self.assertLess(err, 1e-10)

    def test_hamiltonian_sim_two_qubit(self) -> None:
        from scipy.linalg import expm
        ps = PauliSum(n_qubits=2, terms=[
            (1.0, "XX"), (0.5, "ZZ"), (-0.3, "IX")
        ])
        H = ps.to_matrix()
        alpha = ps.alpha()
        for t in [0.1, 0.5, 1.5, 3.0]:
            U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha)
            U_exact = expm(-1j * H * t)
            err = np.linalg.norm(U_ja - U_exact, 2)
            self.assertLess(err, 1e-9, f"t={t}: err={err:.2e}")

    def test_hamiltonian_sim_heisenberg(self) -> None:
        from scipy.linalg import expm
        # Heisenberg XXX op 3 qubits
        terms = []
        for (i, j) in [(0, 1), (1, 2)]:
            label_x = ["I"] * 3; label_x[i] = "X"; label_x[j] = "X"
            label_y = ["I"] * 3; label_y[i] = "Y"; label_y[j] = "Y"
            label_z = ["I"] * 3; label_z[i] = "Z"; label_z[j] = "Z"
            terms.extend([(1.0, "".join(label_x)),
                          (1.0, "".join(label_y)),
                          (1.0, "".join(label_z))])
        ps = PauliSum(n_qubits=3, terms=terms)
        H = ps.to_matrix()
        alpha = ps.alpha()  # = 6
        for t in [0.2, 1.0]:
            U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha)
            U_exact = expm(-1j * H * t)
            err = np.linalg.norm(U_ja - U_exact, 2)
            self.assertLess(err, 1e-9, f"Heisenberg t={t}: err={err:.2e}")

    def test_hamiltonian_sim_unitarity(self) -> None:
        X = pauli_matrix("X")
        U = hamiltonian_simulation_qsvt(X, 2.0, alpha=1.0)
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10))


# ============================================================
# 6. Trotter reference + vergelijking
# ============================================================

class TestTrotterReference(unittest.TestCase):

    def test_single_term_exact(self) -> None:
        from scipy.linalg import expm
        X = pauli_matrix("X")
        U = trotter_reference([(1.0, X)], t=0.5, steps=1, order=1)
        self.assertTrue(np.allclose(U, expm(-1j * 0.5 * X)))

    def test_trotter2_improves(self) -> None:
        # 2 niet-commuterende termen; order=2 moet beter zijn dan order=1 bij
        # gelijk aantal stappen
        from scipy.linalg import expm
        X = pauli_matrix("X")
        Z = pauli_matrix("Z")
        H = X + 0.5 * Z
        t = 0.6
        steps = 4
        U_exact = expm(-1j * H * t)
        U1 = trotter_reference([(1.0, X), (0.5, Z)], t, steps, order=1)
        U2 = trotter_reference([(1.0, X), (0.5, Z)], t, steps, order=2)
        err1 = np.linalg.norm(U1 - U_exact, 2)
        err2 = np.linalg.norm(U2 - U_exact, 2)
        self.assertLess(err2, err1, f"trotter2 ({err2:.2e}) moet < trotter1 ({err1:.2e})")


# ============================================================
# 7. Hamiltonian-integratie
# ============================================================

class TestHamiltonianIntegration(unittest.TestCase):

    def test_from_hamiltonian_compiler(self) -> None:
        """Check dat PauliSum.from_hamiltonian correct converteert."""
        try:
            from hamiltonian_compiler import Hamiltonian
        except ImportError:
            self.skipTest("hamiltonian_compiler niet beschikbaar")
        H_obj = Hamiltonian.heisenberg_xxx(n=3, J=1.0)
        ps = PauliSum.from_hamiltonian(H_obj)
        # Matrix moet overeenkomen (modulo constante termen, die we skippen)
        # We controleren dimensie en Hermiticiteit
        M = ps.to_matrix()
        self.assertEqual(M.shape, (8, 8))
        self.assertTrue(np.allclose(M, M.conj().T))

    def test_ham_sim_beats_trotter1(self) -> None:
        """Jacobi-Anger moet bij vergelijkbare kosten beter zijn dan order-1 Trotter."""
        from scipy.linalg import expm
        X = pauli_matrix("X")
        Z = pauli_matrix("Z")
        # H = X + 0.8 Z op 1 qubit
        H = X + 0.8 * Z
        alpha = 1.8
        t = 1.0
        U_exact = expm(-1j * H * t)
        # Trotter-1 met 50 stappen
        U_tr = trotter_reference([(1.0, X), (0.8, Z)], t, 50, order=1)
        err_tr = np.linalg.norm(U_tr - U_exact, 2)
        # Jacobi-Anger met auto K
        U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha)
        err_ja = np.linalg.norm(U_ja - U_exact, 2)
        self.assertLess(err_ja, err_tr,
                        f"JA ({err_ja:.2e}) moet << Trotter1 met 50 stappen ({err_tr:.2e})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
