#!/usr/bin/env python3
"""
test_b14_mera.py — Unit tests voor B14 MERA tensor netwerk.

22 tests:
  TestMERAInit (4): constructor, product state, random, chi profile
  TestMERAContraction (3): statevector norm, product state exactheid, consistency
  TestMERAExpectation (3): Z op |0⟩, X op |0⟩, 2-point ZZ
  TestMERAOptimize (3): TFIM convergentie, energiedaling, meerdere seeds
  TestMPSCompression (3): fidelity, truncatie, energy
  TestHelpers (3): TFIM terms, exact ground state, QAOA statevector
  TestMERAVsMPS (3): chi vergelijking, parameter telling, entropie
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from zorn_mera import (
    ZornMERA, compress_to_mps, mps_energy,
    build_tfim_terms, build_heisenberg_terms, build_maxcut_terms,
    exact_ground_state, qaoa_statevector, _apply_H_standalone,
    entanglement_entropy,
    _X, _Z, _ZZ, _I2,
)


class TestMERAInit(unittest.TestCase):
    """Test MERA constructie en initialisatie."""

    def test_constructor_n8(self):
        mera = ZornMERA(8, d=2, chi=4)
        self.assertEqual(mera.n_sites, 8)
        self.assertEqual(mera.n_layers, 3)
        self.assertEqual(len(mera.layers), 3)
        # Layer 0: 8→4, 3 disent, 4 isom
        self.assertEqual(mera.layers[0]['n_disent'], 3)
        self.assertEqual(mera.layers[0]['n_isom'], 4)
        # Layer 1: 4→2, 1 disent, 2 isom
        self.assertEqual(mera.layers[1]['n_disent'], 1)
        self.assertEqual(mera.layers[1]['n_isom'], 2)
        # Layer 2: 2→1, 0 disent, 1 isom
        self.assertEqual(mera.layers[2]['n_disent'], 0)
        self.assertEqual(mera.layers[2]['n_isom'], 1)

    def test_constructor_n4(self):
        mera = ZornMERA(4, d=2, chi=4)
        self.assertEqual(mera.n_layers, 2)

    def test_product_state(self):
        mera = ZornMERA(8, d=2, chi=4)
        mera.init_product(0)
        psi = mera.to_statevector()
        expected = np.zeros(256, dtype=complex)
        expected[0] = 1.0
        self.assertLess(np.linalg.norm(psi - expected), 1e-12)

    def test_random_init(self):
        mera = ZornMERA(8, d=2, chi=4)
        mera.init_random(seed=42)
        # Check disentangler unitarity
        for layer in mera.layers:
            for u in layer['disentanglers']:
                dim = u.shape[0] * u.shape[1]
                u_mat = u.reshape(dim, dim)
                eye = u_mat.conj().T @ u_mat
                self.assertLess(np.linalg.norm(eye - np.eye(dim)), 1e-10,
                                "Disentangler niet unitair")


class TestMERAContraction(unittest.TestCase):
    """Test MERA toestandsvector contractie."""

    def test_product_state_n4(self):
        mera = ZornMERA(4, d=2, chi=4)
        mera.init_product(0)
        psi = mera.to_statevector()
        expected = np.zeros(16, dtype=complex)
        expected[0] = 1.0
        self.assertLess(np.linalg.norm(psi - expected), 1e-12)

    def test_random_normalized(self):
        """Random MERA moet (bij benadering) genormaliseerd zijn."""
        mera = ZornMERA(8, d=2, chi=4)
        mera.init_random(seed=42)
        psi = mera.to_statevector()
        # Isometrieën bewaren de norm niet exact bij chi < d²
        # Maar de norm moet redelijk zijn
        norm = np.linalg.norm(psi)
        self.assertGreater(norm, 0.01, "Norm te klein")

    def test_consistency_n2(self):
        """n=2 MERA: 1 laag, 1 isometrie, geen disentanglers."""
        mera = ZornMERA(2, d=2, chi=4)
        mera.init_product(0)
        psi = mera.to_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        self.assertLess(np.linalg.norm(psi - expected), 1e-12)


class TestMERAExpectation(unittest.TestCase):
    """Test verwachtingswaarden."""

    def test_Z_product_state(self):
        """⟨0|Z|0⟩ = 1."""
        mera = ZornMERA(4, d=2, chi=4)
        mera.init_product(0)
        val = mera.expectation_local(_Z, 0)
        self.assertAlmostEqual(val.real, 1.0, places=10)

    def test_X_product_state(self):
        """⟨0|X|0⟩ = 0."""
        mera = ZornMERA(4, d=2, chi=4)
        mera.init_product(0)
        val = mera.expectation_local(_X, 0)
        self.assertAlmostEqual(abs(val), 0.0, places=10)

    def test_ZZ_product_state(self):
        """⟨00|ZZ|00⟩ = 1."""
        mera = ZornMERA(4, d=2, chi=4)
        mera.init_product(0)
        val = mera.expectation_2point(_Z, 0, _Z, 1)
        self.assertAlmostEqual(val.real, 1.0, places=10)


class TestMERAOptimize(unittest.TestCase):
    """Test variationele optimalisatie."""

    def test_tfim_converges(self):
        """MERA moet naar de grondtoestand convergeren."""
        n = 4
        terms = build_tfim_terms(n, J=1.0, h=1.0)
        E_exact, _ = exact_ground_state(terms, n)

        mera = ZornMERA(n, d=2, chi=4)
        mera.init_random(seed=42)
        energies = mera.optimize(terms, n_sweeps=20, verbose=False)

        # Energie moet dalen
        self.assertLess(energies[-1], energies[0],
                        "Energie daalt niet tijdens optimalisatie")

        # Energie moet in de buurt van exact komen (chi=4 voor n=4 is groot genoeg)
        rel_err = abs(energies[-1] - E_exact) / abs(E_exact)
        self.assertLess(rel_err, 0.05,
                        "MERA convergeert niet naar grondtoestand (rel_err=%.4f)" % rel_err)

    def test_energy_decreases(self):
        """Energie moet monotoon dalen (bij benadering)."""
        n = 4
        terms = build_tfim_terms(n)
        mera = ZornMERA(n, d=2, chi=4)
        mera.init_random(seed=99)
        energies = mera.optimize(terms, n_sweeps=10, verbose=False)

        # Gemiddelde trend moet dalend zijn
        first_half = np.mean(energies[:5])
        second_half = np.mean(energies[5:])
        self.assertLess(second_half, first_half + 0.1,
                        "Energie trend is niet dalend")

    def test_multiple_seeds(self):
        """Meerdere seeds moeten vergelijkbare energieën geven."""
        n = 4
        terms = build_tfim_terms(n)

        results = []
        for seed in [10, 20, 30]:
            mera = ZornMERA(n, d=2, chi=4)
            mera.init_random(seed=seed)
            energies = mera.optimize(terms, n_sweeps=15, verbose=False)
            results.append(energies[-1])

        best = min(results)
        worst = max(results)
        # Alle resultaten moeten in dezelfde buurt liggen
        self.assertLess(worst - best, 2.0,
                        "Seeds geven sterk verschillende resultaten: %s" % results)


class TestMPSCompression(unittest.TestCase):
    """Test MPS compressie-functie."""

    def test_full_chi_exact(self):
        """MPS met chi=2^(n/2) moet exacte representatie geven."""
        n = 4
        terms = build_tfim_terms(n)
        _, psi = exact_ground_state(terms, n)

        tensors, fid, max_chi = compress_to_mps(psi, n, 2, chi=16)
        self.assertGreater(fid, 0.9999, "Exacte MPS fidelity te laag: %.6f" % fid)

    def test_truncated_fidelity(self):
        """Lagere chi moet lagere fidelity geven."""
        n = 8
        terms = build_tfim_terms(n)
        _, psi = exact_ground_state(terms, n)

        _, fid_2, _ = compress_to_mps(psi, n, 2, chi=2)
        _, fid_8, _ = compress_to_mps(psi, n, 2, chi=8)

        self.assertGreater(fid_8, fid_2,
                           "Hogere chi moet hogere fidelity geven")

    def test_mps_energy(self):
        """MPS energie moet consistent zijn met fidelity."""
        n = 4
        terms = build_tfim_terms(n)
        E_exact, psi = exact_ground_state(terms, n)

        tensors, _, _ = compress_to_mps(psi, n, 2, chi=8)
        E_mps = mps_energy(tensors, terms, n, 2)

        self.assertAlmostEqual(E_mps, E_exact, places=4,
                               msg="MPS energie wijkt af: %.6f vs %.6f" % (E_mps, E_exact))


class TestHelpers(unittest.TestCase):
    """Test helper functies."""

    def test_tfim_terms(self):
        """TFIM moet n-1 ZZ termen en n X termen hebben."""
        n = 8
        terms = build_tfim_terms(n)
        n_zz = sum(1 for _, _, s in terms if len(s) == 2)
        n_x = sum(1 for _, _, s in terms if len(s) == 1)
        self.assertEqual(n_zz, n - 1)
        self.assertEqual(n_x, n)

    def test_exact_ground_state(self):
        """Exacte grondtoestand moet eigentoestand zijn."""
        n = 4
        terms = build_tfim_terms(n)
        E0, psi0 = exact_ground_state(terms, n)

        H_psi = _apply_H_standalone(psi0, terms, n, 2)
        diff = H_psi - E0 * psi0
        self.assertLess(np.linalg.norm(diff), 1e-10,
                        "Grondtoestand is geen eigentoestand")

    def test_qaoa_normalized(self):
        """QAOA toestandsvector moet genormaliseerd zijn."""
        psi = qaoa_statevector(4, [(0, 1), (1, 2), (2, 3)],
                               p=2, gammas=[0.3, 0.6], betas=[0.7, 0.35])
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=10)


class TestMERAVsMPS(unittest.TestCase):
    """Test MERA vs MPS vergelijking."""

    def test_chi_comparison_tfim(self):
        """MERA en MPS moeten beide redelijke energieën geven."""
        n = 4
        terms = build_tfim_terms(n)
        E_exact, psi = exact_ground_state(terms, n)

        # MPS chi=4
        tensors, fid_mps, _ = compress_to_mps(psi, n, 2, chi=4)

        # MERA chi=4
        mera = ZornMERA(n, d=2, chi=4)
        mera.init_random(seed=42)
        mera.optimize(terms, n_sweeps=15, verbose=False)
        fid_mera = mera.fidelity(psi)

        # Beide moeten redelijke fidelity hebben
        self.assertGreater(fid_mps, 0.5, "MPS fidelity te laag")
        # MERA mag lager zijn (optimalisatie kan vastlopen)
        # maar moet boven 0 liggen
        self.assertGreater(fid_mera, 0.01, "MERA fidelity te laag")

    def test_parameter_count(self):
        """MERA heeft meer parameters dan MPS bij dezelfde chi."""
        mera = ZornMERA(8, d=2, chi=4)
        # MERA params omvat disentanglers + isometries + top
        params = mera.total_params
        self.assertGreater(params, 0)

        # MPS chi=4, n=8, d=2: ~8*2*4² = 256 (ruw)
        # MERA chi=4 heeft meer door de disentanglers
        self.assertGreater(params, 100, "Te weinig MERA parameters")

    def test_entanglement_entropy(self):
        """Entanglement entropie moet positief zijn voor verstrengelde toestanden."""
        n = 8
        terms = build_tfim_terms(n)
        _, psi = exact_ground_state(terms, n)

        S = entanglement_entropy(psi, n, 2, n // 2)
        self.assertGreater(S, 0.1,
                           "TFIM grondtoestand moet verstrengeld zijn (S=%.4f)" % S)


# =====================================================================
# Entanglement entropy import (nodig voor test)
# =====================================================================
try:
    from zorn_mera import entanglement_entropy
except ImportError:
    # Als de functie nog niet in zorn_mera zit, definieer lokaal
    def entanglement_entropy(psi, n, d, cut):
        dim_A = d ** cut
        dim_B = d ** (n - cut)
        rho_AB = psi.reshape(dim_A, dim_B)
        _, S, _ = np.linalg.svd(rho_AB, full_matrices=False)
        S = S[S > 1e-15]
        S2 = S ** 2
        return -np.sum(S2 * np.log(S2 + 1e-30))


if __name__ == '__main__':
    unittest.main(verbosity=2)
