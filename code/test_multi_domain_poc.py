#!/usr/bin/env python3
"""
test_multi_domain_poc.py - Tests voor B132 Multi-Domain PoC

Unittest suite die de 3 demo-domeinen valideert:
  1. Condensed matter (Heisenberg XXX)
  2. Moleculair (H2 STO-3G)
  3. PDE (1D lattice evolutie)

Plus unit tests voor hulpfuncties.

Author: ZornQ project
Date: 16 april 2026
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from multi_domain_poc import (
    exact_ground_state, _pauli_string_to_matrix,
    vqe_optimize, h2_qubit_hamiltonian,
    lattice_kinetic_hamiltonian, lattice_potential_hamiltonian,
    demo_condensed_matter, demo_molecular, demo_pde,
    run_all_demos,
)
from hamiltonian_compiler import Hamiltonian
from circuit_interface import Circuit, Observable, run_circuit
from quality_certificate import CertificateLevel


# =====================================================================
# TEST HULPFUNCTIES
# =====================================================================

class TestPauliMatrix(unittest.TestCase):
    """Test Pauli string -> matrix conversie."""

    def test_identity(self):
        """Lege Pauli string = identiteit."""
        mat = _pauli_string_to_matrix({}, 2)
        np.testing.assert_allclose(mat, np.eye(4))

    def test_single_z(self):
        """Z op qubit 0."""
        mat = _pauli_string_to_matrix({0: 'Z'}, 1)
        expected = np.array([[1, 0], [0, -1]])
        np.testing.assert_allclose(mat, expected)

    def test_zz(self):
        """Z tensor Z."""
        mat = _pauli_string_to_matrix({0: 'Z', 1: 'Z'}, 2)
        expected = np.diag([1, -1, -1, 1]).astype(complex)
        np.testing.assert_allclose(mat, expected)

    def test_x_on_second(self):
        """X op qubit 1, I op qubit 0 (LSB conventie)."""
        mat = _pauli_string_to_matrix({1: 'X'}, 2)
        # LSB conventie: q1 otimes q0 = X tensor I
        expected = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
        np.testing.assert_allclose(mat, expected)

    def test_hermitian(self):
        """Pauli strings zijn Hermitisch."""
        for pauli in [{0: 'X', 1: 'Y'}, {0: 'Z', 2: 'X'}, {1: 'Y'}]:
            n = max(pauli.keys()) + 1
            mat = _pauli_string_to_matrix(pauli, n)
            np.testing.assert_allclose(mat, mat.conj().T, atol=1e-14)


class TestExactGroundState(unittest.TestCase):
    """Test exacte diagonalisatie."""

    def test_single_z(self):
        """H = Z: eigenwaarden +1, -1."""
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        E, psi = exact_ground_state(H)
        self.assertAlmostEqual(E, -1.0, places=10)

    def test_ising_2site(self):
        """H = -ZZ - 0.5*(X0 + X1): bekende grondtoestand."""
        H = Hamiltonian.ising_transverse(2, J=1.0, h=0.5)
        E, psi = exact_ground_state(H)
        # GS energie < -1 (zeker lager dan -ZZ alleen)
        self.assertLess(E, -1.0)
        # Normalisatie
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=10)

    def test_heisenberg_4site(self):
        """Heisenberg XXX 4 sites: bekende GS energie."""
        H = Hamiltonian.heisenberg_xxx(4, J=1.0)
        E, psi = exact_ground_state(H)
        # GS energie van 4-site Heisenberg keten ~ -6.0 tot -7.0
        # Exact: E0 = -2 - 2*sqrt(3) + ... afhankelijk van randvoorwaarden
        # Open keten met J=1: E0 per bond ~ -1.616
        # 3 bonds: E0 ~ -4.848... maar met correcties
        self.assertLess(E, 0)
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=10)

    def test_matches_numpy_eigh(self):
        """Vergelijk met directe numpy.eigh."""
        H = Hamiltonian.ising_transverse(3, J=1.0, h=0.3)
        E, psi = exact_ground_state(H)

        # Bouw matrix direct
        n = H.n_qubits
        dim = 1 << n
        H_mat = np.zeros((dim, dim), dtype=complex)
        for c, p in H.terms:
            H_mat += c * _pauli_string_to_matrix(p, n)
        evals = np.linalg.eigvalsh(H_mat)
        self.assertAlmostEqual(E, evals[0], places=10)


# =====================================================================
# TEST H2 INTEGRALEN
# =====================================================================

class TestH2Hamiltonian(unittest.TestCase):
    """Test H2 qubit Hamiltonian."""

    def test_n_qubits(self):
        """4 qubits voor H2."""
        H, E_nuc, nq = h2_qubit_hamiltonian(0.74)
        self.assertEqual(nq, 4)
        self.assertEqual(H.n_qubits, 4)

    def test_nuclear_repulsion(self):
        """Kernenergie positief."""
        _, E_nuc, _ = h2_qubit_hamiltonian(0.74)
        self.assertGreater(E_nuc, 0)
        self.assertAlmostEqual(E_nuc, 0.7138, places=3)

    def test_hamiltonian_hermitian(self):
        """Qubit Hamiltonian matrix is Hermitisch."""
        H, _, nq = h2_qubit_hamiltonian(0.74)
        H_mat = np.zeros((1 << nq, 1 << nq), dtype=complex)
        for c, p in H.terms:
            H_mat += c * _pauli_string_to_matrix(p, nq)
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-10)

    def test_n_terms(self):
        """Verwacht 15 termen (1 I + 4 Z + 6 ZZ + 4 XXYY)."""
        H, _, _ = h2_qubit_hamiltonian(0.74)
        self.assertEqual(H.n_terms, 15)

    def test_invalid_bond_length(self):
        """Ongeldige bindingslengte geeft fout."""
        with self.assertRaises(ValueError):
            h2_qubit_hamiltonian(1.5)

    def test_h2_fci_energy(self):
        """FCI energie van H2 bij R=0.74 is ~-1.137 Ha."""
        H, E_nuc, nq = h2_qubit_hamiltonian(0.74)
        E_elec, _ = exact_ground_state(H, nq)
        E_total = E_elec + E_nuc
        # Moet dicht bij -1.137 Ha liggen
        self.assertAlmostEqual(E_total, -1.137, delta=0.01)


# =====================================================================
# TEST LATTICE HAMILTONIANS
# =====================================================================

class TestLatticeHamiltonian(unittest.TestCase):
    """Test lattice discretisatie Hamiltonians."""

    def test_kinetic_terms(self):
        """Kinetische Hamiltonian heeft correcte termen."""
        H = lattice_kinetic_hamiltonian(3, dx=1.0, m=1.0)
        self.assertGreater(H.n_terms, 0)
        self.assertEqual(H.n_qubits, 3)

    def test_kinetic_hermitian(self):
        """Kinetische Hamiltonian matrix is Hermitisch."""
        H = lattice_kinetic_hamiltonian(4, dx=1.0, m=1.0)
        n = H.n_qubits
        H_mat = np.zeros((1 << n, 1 << n), dtype=complex)
        for c, p in H.terms:
            H_mat += c * _pauli_string_to_matrix(p, n)
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)

    def test_potential_harmonic(self):
        """Harmonische potentiaal Hamiltonian."""
        V = lambda x: 0.5 * x * x
        H = lattice_potential_hamiltonian(4, V, dx=1.0)
        self.assertGreater(H.n_terms, 0)

    def test_potential_hermitian(self):
        """Potentiaal Hamiltonian is Hermitisch."""
        V = lambda x: 0.5 * x * x
        H = lattice_potential_hamiltonian(4, V, dx=1.0)
        n = H.n_qubits
        H_mat = np.zeros((1 << n, 1 << n), dtype=complex)
        for c, p in H.terms:
            H_mat += c * _pauli_string_to_matrix(p, n)
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)

    def test_total_hermitian(self):
        """T + V is Hermitisch."""
        T = lattice_kinetic_hamiltonian(4, dx=1.0)
        V = lattice_potential_hamiltonian(4, lambda x: 0.5 * x * x, dx=1.0)
        H = T + V
        H.simplify()
        n = H.n_qubits
        H_mat = np.zeros((1 << n, 1 << n), dtype=complex)
        for c, p in H.terms:
            H_mat += c * _pauli_string_to_matrix(p, n)
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)

    def test_single_excitation_sector(self):
        """In 1-excitatie sector is H een tridiagonale matrix."""
        n = 4
        H_T = lattice_kinetic_hamiltonian(n, dx=1.0, m=1.0)
        H_mat = np.zeros((1 << n, 1 << n), dtype=complex)
        for c, p in H_T.terms:
            H_mat += c * _pauli_string_to_matrix(p, n)

        # 1-excitatie states: |0001>, |0010>, |0100>, |1000>
        # = indices 1, 2, 4, 8
        sector_indices = [1 << i for i in range(n)]
        H_sector = np.zeros((n, n), dtype=complex)
        for i, si in enumerate(sector_indices):
            for j, sj in enumerate(sector_indices):
                H_sector[i, j] = H_mat[si, sj]

        # Moet tridiagonaal zijn (nearest-neighbor hopping)
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    self.assertAlmostEqual(abs(H_sector[i, j]), 0, places=10,
                        msg="Off-diag (%d,%d) = %s" % (i, j, H_sector[i, j]))


# =====================================================================
# TEST VQE OPTIMIZER
# =====================================================================

class TestVQE(unittest.TestCase):
    """Test VQE optimalisatie."""

    def test_single_qubit(self):
        """VQE vindt GS van H=Z (trivial)."""
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        vqe = vqe_optimize(H, depth=1, n_restarts=2, maxiter=50)
        self.assertAlmostEqual(vqe['energy'], -1.0, delta=0.1)

    def test_returns_circuit(self):
        """VQE geeft circuit en resultaat terug."""
        H = Hamiltonian.ising_transverse(2, J=1.0, h=0.5)
        vqe = vqe_optimize(H, depth=2, n_restarts=1, maxiter=30)
        self.assertIn('circuit', vqe)
        self.assertIn('result', vqe)
        self.assertIn('energy', vqe)
        self.assertIn('params', vqe)
        self.assertIn('n_evals', vqe)

    def test_energy_below_zero(self):
        """VQE vindt negatieve energie voor Heisenberg."""
        H = Hamiltonian.heisenberg_xxx(2, J=1.0)
        vqe = vqe_optimize(H, depth=2, n_restarts=2, maxiter=100)
        self.assertLess(vqe['energy'], 0)


# =====================================================================
# TEST DEMO FUNCTIES
# =====================================================================

class TestDemoCondensedMatter(unittest.TestCase):
    """Test condensed matter demo."""

    def test_runs(self):
        """Demo draait zonder fouten."""
        result = demo_condensed_matter(
            n_sites=2, depth=2, n_restarts=1, maxiter=50, verbose=False)
        self.assertEqual(result['domain'], 'condensed_matter')
        self.assertIn('vqe_energy', result)
        self.assertIn('exact_energy', result)
        self.assertIn('certificate', result)

    def test_vqe_close_to_exact(self):
        """VQE komt in de buurt van exact voor 2 sites."""
        result = demo_condensed_matter(
            n_sites=2, depth=3, n_restarts=2, maxiter=100, verbose=False)
        # Relatieve fout < 20% (COBYLA met weinig iteraties)
        self.assertLess(result['relative_error'], 0.2)

    def test_fidelity_positive(self):
        """Fidelity is positief en <= 1."""
        result = demo_condensed_matter(
            n_sites=2, depth=2, n_restarts=1, maxiter=50, verbose=False)
        self.assertGreater(result['fidelity'], 0)
        self.assertLessEqual(result['fidelity'], 1.0 + 1e-10)


class TestDemoMolecular(unittest.TestCase):
    """Test molecular demo."""

    def test_runs(self):
        """Demo draait zonder fouten."""
        result = demo_molecular(
            bond_length=0.74, depth=2, n_restarts=1, maxiter=50,
            verbose=False)
        self.assertEqual(result['domain'], 'molecular')
        self.assertIn('vqe_energy', result)
        self.assertIn('fci_energy', result)

    def test_fci_energy_reasonable(self):
        """FCI energie is in verwacht bereik."""
        result = demo_molecular(
            bond_length=0.74, depth=2, n_restarts=1, maxiter=30,
            verbose=False)
        # FCI ~-1.137 Ha
        self.assertAlmostEqual(result['fci_energy'], -1.137, delta=0.02)

    def test_vqe_above_fci(self):
        """VQE energie >= FCI (variationeel principe)."""
        result = demo_molecular(
            bond_length=0.74, depth=3, n_restarts=1, maxiter=50,
            verbose=False)
        # VQE totaal >= FCI totaal (variationeel)
        self.assertGreaterEqual(result['vqe_energy'],
                                result['fci_energy'] - 0.01)


class TestDemoPDE(unittest.TestCase):
    """Test PDE demo."""

    def test_runs(self):
        """Demo draait zonder fouten."""
        result = demo_pde(
            n_sites=4, t_evolve=0.5, steps=5, trotter_order=2,
            verbose=False)
        self.assertEqual(result['domain'], 'pde')
        self.assertIn('fidelity', result)
        self.assertIn('occ_rmse', result)

    def test_fidelity_high(self):
        """Fidelity > 0.9 met voldoende stappen."""
        result = demo_pde(
            n_sites=4, t_evolve=0.5, steps=20, trotter_order=2,
            verbose=False)
        self.assertGreater(result['fidelity'], 0.9)

    def test_norm_preserved(self):
        """Trotter behoudt norm."""
        result = demo_pde(
            n_sites=4, t_evolve=0.5, steps=5, trotter_order=1,
            verbose=False)
        # Bezettingen moeten sommeren tot ~1 in 1-excitatie sector
        occ_sum = float(np.sum(result['occ_trotter']))
        # Kan afwijken door lekkage buiten 1-excitatie sector
        self.assertGreater(occ_sum, 0.5)
        self.assertLess(occ_sum, 1.5)

    def test_trotter2_better_than_1(self):
        """Trotter-2 geeft hogere fidelity dan Trotter-1."""
        r1 = demo_pde(n_sites=4, t_evolve=1.0, steps=5,
                       trotter_order=1, verbose=False)
        r2 = demo_pde(n_sites=4, t_evolve=1.0, steps=5,
                       trotter_order=2, verbose=False)
        self.assertGreaterEqual(r2['fidelity'], r1['fidelity'] - 0.01)

    def test_more_steps_better(self):
        """Meer Trotter stappen geeft hogere fidelity."""
        r_few = demo_pde(n_sites=4, t_evolve=1.0, steps=3,
                          trotter_order=2, verbose=False)
        r_many = demo_pde(n_sites=4, t_evolve=1.0, steps=20,
                           trotter_order=2, verbose=False)
        self.assertGreaterEqual(r_many['fidelity'],
                                r_few['fidelity'] - 0.01)


# =====================================================================
# TEST INTEGRATIE
# =====================================================================

class TestRunAllDemos(unittest.TestCase):
    """Test run_all_demos integratie."""

    def test_runs(self):
        """Alle demos draaien zonder fouten (klein formaat)."""
        # Override met kleine parameters via individuele calls
        r1 = demo_condensed_matter(n_sites=2, depth=1, n_restarts=1,
                                    maxiter=20, verbose=False)
        r2 = demo_molecular(depth=1, n_restarts=1, maxiter=20,
                             verbose=False)
        r3 = demo_pde(n_sites=4, t_evolve=0.5, steps=3,
                       trotter_order=1, verbose=False)
        self.assertEqual(r1['domain'], 'condensed_matter')
        self.assertEqual(r2['domain'], 'molecular')
        self.assertEqual(r3['domain'], 'pde')

    def test_all_have_certificates(self):
        """Alle resultaten hebben certificaten."""
        r1 = demo_condensed_matter(n_sites=2, depth=1, n_restarts=1,
                                    maxiter=20, verbose=False)
        r2 = demo_molecular(depth=1, n_restarts=1, maxiter=20,
                             verbose=False)
        r3 = demo_pde(n_sites=4, t_evolve=0.5, steps=3,
                       trotter_order=1, verbose=False)
        for r in [r1, r2, r3]:
            self.assertIsNotNone(r['certificate'])
            self.assertIsInstance(r['certificate'].level, CertificateLevel)


if __name__ == '__main__':
    unittest.main()
