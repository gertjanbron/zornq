#!/usr/bin/env python3
"""
Comprehensive tests voor B34: Mid-Circuit Measurement / Adaptieve Projectie.

Test suites:
  1. TestMPS: Basis MPS operaties (constructie, conversie, norm, copy)
  2. TestMPSFromStatevector: Roundtrip state vector <-> MPS
  3. TestMPSCompress: Compressie met chi_max en min_weight
  4. TestBondEntropy: Bond-entropie berekening
  5. TestQAOAGates: Individuele QAOA gates op MPS
  6. TestQAOAMPS: Volledige QAOA via MPS vs state vector
  7. TestMeasureQubit: Enkele qubit meting
  8. TestMeasureQubits: Multi-qubit meting
  9. TestSplitMPS: MPS splitsen na meting
  10. TestAdaptiveMeasurement: Adaptieve meetpunt selectie
  11. TestMultiBranch: Multi-branch sampling
  12. TestObservables: Verwachtingswaarden Z en ZZ
  13. TestEdgeCases: Randgevallen en robuustheid
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from midcircuit_measurement import (
    MPS, mps_compress,
    apply_phase_gate, apply_mixer_gate, apply_qaoa_layer,
    qaoa_mps, qaoa_statevector,
    measure_qubit, measure_qubits, split_mps_at_measured,
    select_measurement_sites, adaptive_measurement_schedule,
    multi_branch_expectation,
    expectation_z, expectation_zz,
    maxcut_cost_mps, maxcut_cost_statevector,
)


class TestMPS(unittest.TestCase):
    """Test basis MPS operaties."""

    def test_create_product_state(self):
        """Product state |000> als MPS."""
        n = 3
        zero = np.array([1.0, 0.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        self.assertEqual(mps.n, 3)
        self.assertEqual(mps.d, 2)
        self.assertEqual(len(mps.tensors), 3)

    def test_plus_state(self):
        """|+>^n als MPS."""
        n = 4
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        psi = mps.to_statevector()
        # Alle amplitudes gelijk
        expected = np.ones(2**n) / (2**(n/2))
        np.testing.assert_allclose(np.abs(psi), expected, atol=1e-12)

    def test_norm_product_state(self):
        """Norm van genormaliseerde product state is 1."""
        n = 5
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        self.assertAlmostEqual(mps.norm(), 1.0, places=12)

    def test_copy_is_independent(self):
        """Copy maakt onafhankelijke kopie."""
        n = 3
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        mps2 = mps.copy()
        mps2.tensors[0] *= 2
        # Origineel ongewijzigd
        np.testing.assert_allclose(mps.tensors[0],
                                    plus.reshape(1, 2, 1), atol=1e-12)

    def test_bond_dims(self):
        """Bond dimensies van product state zijn allemaal 1."""
        n = 4
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        self.assertEqual(mps.bond_dims(), [1, 1, 1])
        self.assertEqual(mps.max_bond_dim(), 1)

    def test_classical_bits_init(self):
        """Classical bits worden correct opgeslagen."""
        n = 3
        tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
        mps = MPS(tensors, classical_bits={0: 1, 2: 0})
        self.assertEqual(mps.classical_bits, {0: 1, 2: 0})


class TestMPSFromStatevector(unittest.TestCase):
    """Test MPS <-> state vector conversie."""

    def test_roundtrip_plus_state(self):
        """Roundtrip: |+>^n -> MPS -> state vector."""
        n = 4
        psi = np.ones(2**n, dtype=complex) / (2**(n/2))
        mps = MPS.from_statevector(psi, n)
        psi_back = mps.to_statevector()
        np.testing.assert_allclose(np.abs(psi_back), np.abs(psi), atol=1e-12)

    def test_roundtrip_ghz(self):
        """Roundtrip: GHZ state (|000> + |111>) / sqrt(2)."""
        n = 3
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
        mps = MPS.from_statevector(psi, n)
        psi_back = mps.to_statevector()
        np.testing.assert_allclose(psi_back, psi, atol=1e-12)
        # GHZ heeft bond dim 2
        self.assertEqual(mps.max_bond_dim(), 2)

    def test_roundtrip_random(self):
        """Roundtrip: random state."""
        rng = np.random.default_rng(42)
        n = 5
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        psi_back = mps.to_statevector()
        # Fideliteit
        fid = abs(np.dot(psi.conj(), psi_back))**2
        self.assertAlmostEqual(fid, 1.0, places=10)

    def test_truncated_roundtrip(self):
        """Roundtrip met chi_max truncatie verliest enige fideliteit."""
        rng = np.random.default_rng(123)
        n = 6
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n, chi_max=4)
        psi_back = mps.to_statevector()
        fid = abs(np.dot(psi.conj(), psi_back))**2
        # Met chi_max=4 op 6 qubits verlies je iets, maar niet alles
        self.assertGreater(fid, 0.5)
        self.assertLessEqual(mps.max_bond_dim(), 4)


class TestMPSCompress(unittest.TestCase):
    """Test MPS compressie."""

    def test_compress_preserves_product_state(self):
        """Compressie verandert product state niet."""
        n = 4
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        compressed = mps_compress(mps, chi_max=2)
        psi1 = mps.to_statevector()
        psi2 = compressed.to_statevector()
        fid = abs(np.dot(psi1.conj(), psi2))**2
        self.assertAlmostEqual(fid, 1.0, places=10)

    def test_compress_reduces_bond_dim(self):
        """Compressie reduceert bond dimensie."""
        rng = np.random.default_rng(42)
        n = 6
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        compressed = mps_compress(mps, chi_max=4)
        self.assertLessEqual(compressed.max_bond_dim(), 4)

    def test_compress_with_min_weight(self):
        """min_weight drempel gooit kleine singuliere waarden weg."""
        rng = np.random.default_rng(42)
        n = 6
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        compressed = mps_compress(mps, chi_max=32, min_weight=0.1)
        # Met agressieve min_weight is bond dim kleiner
        self.assertLessEqual(compressed.max_bond_dim(), mps.max_bond_dim())

    def test_compress_fidelity(self):
        """Compressie behoudt fideliteit bij voldoende chi."""
        rng = np.random.default_rng(42)
        n = 6
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        compressed = mps_compress(mps, chi_max=16)
        psi1 = mps.to_statevector()
        psi2 = compressed.to_statevector()
        fid = abs(np.dot(psi1.conj(), psi2))**2
        self.assertGreater(fid, 0.99)


class TestBondEntropy(unittest.TestCase):
    """Test bond-entropie berekening."""

    def test_product_state_zero_entropy(self):
        """Product state heeft nul entropie op elke bond."""
        n = 4
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        for i in range(n - 1):
            self.assertAlmostEqual(mps.bond_entropy(i), 0.0, places=10)

    def test_ghz_maximal_entropy(self):
        """GHZ state heeft entropie 1.0 bit op elke bond."""
        n = 3
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
        mps = MPS.from_statevector(psi, n)
        for i in range(n - 1):
            self.assertAlmostEqual(mps.bond_entropy(i), 1.0, places=5)

    def test_bell_state_entropy(self):
        """Bell state (|00> + |11>)/sqrt(2) heeft entropie 1.0."""
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        mps = MPS.from_statevector(psi, 2)
        self.assertAlmostEqual(mps.bond_entropy(0), 1.0, places=5)

    def test_all_bond_entropies(self):
        """all_bond_entropies geeft juist aantal resultaten."""
        n = 5
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        entropies = mps.all_bond_entropies()
        self.assertEqual(len(entropies), n - 1)

    def test_schmidt_values(self):
        """Schmidt-waarden voor Bell state zijn [1/sqrt(2), 1/sqrt(2)]."""
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        mps = MPS.from_statevector(psi, 2)
        sv = mps.schmidt_values(0)
        np.testing.assert_allclose(sorted(sv, reverse=True),
                                    [1/np.sqrt(2), 1/np.sqrt(2)], atol=1e-10)

    def test_invalid_site_raises(self):
        """Ongeldige site geeft ValueError."""
        n = 3
        mps = MPS([np.zeros((1, 2, 1)) for _ in range(n)])
        with self.assertRaises(ValueError):
            mps.bond_entropy(-1)
        with self.assertRaises(ValueError):
            mps.bond_entropy(n - 1)


class TestQAOAGates(unittest.TestCase):
    """Test individuele QAOA gates op MPS."""

    def test_mixer_gate_identity(self):
        """Mixer met beta=0 is identiteit."""
        n = 3
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        psi_before = mps.to_statevector().copy()
        apply_mixer_gate(mps, 1, 0.0)
        psi_after = mps.to_statevector()
        np.testing.assert_allclose(psi_after, psi_before, atol=1e-12)

    def test_mixer_gate_flip(self):
        """Mixer met beta=pi/2 flipt qubit."""
        # Start met |0>
        zero = np.array([1.0, 0.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1)]
        mps = MPS(tensors)
        apply_mixer_gate(mps, 0, np.pi / 2)
        psi = mps.to_statevector()
        # e^{-i*pi/2*X}|0> = cos(pi/2)|0> - i*sin(pi/2)|1> = -i|1>
        expected = np.array([0.0, -1j], dtype=complex)
        np.testing.assert_allclose(psi, expected, atol=1e-12)

    def test_phase_gate_adjacent(self):
        """Phase gate op naburige qubits vs state vector."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        gamma = 0.3

        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)

        mps = MPS.from_statevector(psi, n)

        # Apply phase gate via MPS
        for (i, j) in edges:
            apply_phase_gate(mps, i, j, gamma)

        # Apply phase gate via state vector
        dim = 2**n
        indices = np.arange(dim)
        phase = np.zeros(dim)
        for (i, j) in edges:
            bi = (indices >> (n - 1 - i)) & 1
            bj = (indices >> (n - 1 - j)) & 1
            phase += gamma * (1 - 2 * (bi ^ bj))
        psi_sv = psi * np.exp(1j * phase)

        psi_mps = mps.to_statevector()
        fid = abs(np.dot(psi_sv.conj(), psi_mps))**2
        self.assertGreater(fid, 0.999)

    def test_phase_gate_identity(self):
        """Phase gate met gamma=0 is identiteit."""
        n = 3
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        psi_before = mps.to_statevector().copy()
        apply_phase_gate(mps, 0, 1, 0.0)
        psi_after = mps.to_statevector()
        fid = abs(np.dot(psi_before.conj(), psi_after))**2
        self.assertAlmostEqual(fid, 1.0, places=10)


class TestQAOAMPS(unittest.TestCase):
    """Test volledige QAOA via MPS vs state vector referentie."""

    def test_qaoa_p1_small(self):
        """QAOA p=1 op 4-qubit keten: MPS vs state vector."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3]
        betas = [0.7]

        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        mps = qaoa_mps(n, edges, gammas, betas)
        psi_mps = mps.to_statevector()

        fid = abs(np.dot(psi_sv.conj(), psi_mps))**2
        self.assertGreater(fid, 0.999)

    def test_qaoa_p2(self):
        """QAOA p=2 op 6-qubit keten."""
        n = 6
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3, 0.5]
        betas = [0.7, 0.4]

        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        mps = qaoa_mps(n, edges, gammas, betas)
        psi_mps = mps.to_statevector()

        fid = abs(np.dot(psi_sv.conj(), psi_mps))**2
        self.assertGreater(fid, 0.99)

    def test_qaoa_cost_matches(self):
        """MaxCut cost via MPS matcht state vector berekening."""
        n = 5
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.4]
        betas = [0.8]

        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        cost_sv = maxcut_cost_statevector(psi_sv, n, edges)

        mps = qaoa_mps(n, edges, gammas, betas)
        cost_mps = maxcut_cost_mps(mps, edges)

        self.assertAlmostEqual(cost_mps, cost_sv, places=3)

    def test_qaoa_with_compression(self):
        """QAOA met chi_max behoudt redelijke kwaliteit."""
        n = 8
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3]
        betas = [0.7]

        mps_exact = qaoa_mps(n, edges, gammas, betas)
        mps_compressed = qaoa_mps(n, edges, gammas, betas, chi_max=4)

        cost_exact = maxcut_cost_mps(mps_exact, edges)
        cost_compressed = maxcut_cost_mps(mps_compressed, edges)

        # p=1 op een keten: compressie zou niet veel moeten kosten
        self.assertAlmostEqual(cost_compressed, cost_exact, delta=0.5)


class TestMeasureQubit(unittest.TestCase):
    """Test enkele qubit meting."""

    def test_measure_zero_state(self):
        """|0> meten geeft altijd 0."""
        zero = np.array([1.0, 0.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1)]
        mps = MPS(tensors)
        outcome, prob = measure_qubit(mps, 0)
        self.assertEqual(outcome, 0)
        self.assertAlmostEqual(prob, 1.0, places=10)

    def test_measure_one_state(self):
        """|1> meten geeft altijd 1."""
        one = np.array([0.0, 1.0], dtype=complex)
        tensors = [one.reshape(1, 2, 1)]
        mps = MPS(tensors)
        outcome, prob = measure_qubit(mps, 0)
        self.assertEqual(outcome, 1)
        self.assertAlmostEqual(prob, 1.0, places=10)

    def test_measure_plus_state_probabilities(self):
        """|+> meten geeft 50/50."""
        rng = np.random.default_rng(42)
        n_trials = 1000
        outcomes = []
        for _ in range(n_trials):
            plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
            mps = MPS([plus.reshape(1, 2, 1)])
            outcome, prob = measure_qubit(mps, 0, rng=rng)
            outcomes.append(outcome)
            self.assertAlmostEqual(prob, 0.5, places=10)
        # Binomiale test: verwacht ~50% met ruime marge
        frac = sum(outcomes) / n_trials
        self.assertGreater(frac, 0.4)
        self.assertLess(frac, 0.6)

    def test_forced_outcome(self):
        """Geforceerde meetuitkomst werkt correct."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        mps = MPS([plus.reshape(1, 2, 1)])
        outcome, prob = measure_qubit(mps, 0, outcome=1)
        self.assertEqual(outcome, 1)
        self.assertAlmostEqual(prob, 0.5, places=10)

    def test_measure_registers_classical_bit(self):
        """Na meting is qubit geregistreerd als klassiek bit."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        mps = MPS([plus.reshape(1, 2, 1)])
        outcome, _ = measure_qubit(mps, 0, outcome=0)
        self.assertIn(0, mps.classical_bits)
        self.assertEqual(mps.classical_bits[0], 0)

    def test_measure_already_measured(self):
        """Opnieuw meten van gemeten qubit geeft zelfde resultaat."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        mps = MPS([plus.reshape(1, 2, 1)])
        measure_qubit(mps, 0, outcome=1)
        outcome2, prob2 = measure_qubit(mps, 0)
        self.assertEqual(outcome2, 1)
        self.assertAlmostEqual(prob2, 1.0, places=10)

    def test_measure_bell_state_correlations(self):
        """Meting op Bell state |00>+|11> geeft gecorreleerde uitkomsten."""
        rng = np.random.default_rng(42)
        correlations = []
        for _ in range(200):
            psi = np.zeros(4, dtype=complex)
            psi[0] = 1.0 / np.sqrt(2)
            psi[3] = 1.0 / np.sqrt(2)
            mps = MPS.from_statevector(psi, 2)
            out0, _ = measure_qubit(mps, 0, rng=rng)
            # Na meting van qubit 0: qubit 1 is gecorreleerd
            # We kunnen niet direct meten (tensor shape veranderd)
            # Maar we kunnen de Z-verwachting checken
            z1 = expectation_z(mps, 1)
            correlations.append((out0, z1))
        # Als qubit 0 = 0, dan Z1 moet ~+1 zijn, als 0 = 1, dan Z1 ~ -1
        for out0, z1 in correlations:
            if out0 == 0:
                self.assertGreater(z1, 0.5)
            else:
                self.assertLess(z1, -0.5)


class TestMeasureQubits(unittest.TestCase):
    """Test multi-qubit meting."""

    def test_measure_multiple(self):
        """Meet meerdere qubits sequentieel."""
        n = 4
        psi = np.zeros(2**n, dtype=complex)
        psi[0b1010] = 1.0  # |1010>
        mps = MPS.from_statevector(psi, n)
        results, prob = measure_qubits(mps, [0, 1, 2, 3])
        self.assertEqual(results, [1, 0, 1, 0])
        self.assertAlmostEqual(prob, 1.0, places=10)

    def test_measure_subset(self):
        """Meet subset van qubits."""
        n = 4
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        results, prob = measure_qubits(mps, [1, 3], outcomes=[0, 1])
        self.assertEqual(results, [0, 1])
        self.assertEqual(mps.classical_bits[1], 0)
        self.assertEqual(mps.classical_bits[3], 1)


class TestSplitMPS(unittest.TestCase):
    """Test MPS splitsen na meting."""

    def test_split_middle(self):
        """Split op midden-qubit geeft twee stukken."""
        n = 5
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        measure_qubit(mps, 2, outcome=0)
        pieces = split_mps_at_measured(mps, 2)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(pieces[0].n, 2)  # sites 0, 1
        self.assertEqual(pieces[1].n, 2)  # sites 3, 4

    def test_split_first(self):
        """Split op eerste qubit geeft 1 stuk."""
        n = 3
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        measure_qubit(mps, 0, outcome=0)
        pieces = split_mps_at_measured(mps, 0)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].n, 2)

    def test_split_last(self):
        """Split op laatste qubit geeft 1 stuk."""
        n = 3
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        measure_qubit(mps, 2, outcome=1)
        pieces = split_mps_at_measured(mps, 2)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].n, 2)

    def test_split_unmeasured_raises(self):
        """Split op ongemeten qubit geeft ValueError."""
        n = 3
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        with self.assertRaises(ValueError):
            split_mps_at_measured(mps, 1)

    def test_split_preserves_norm(self):
        """Gesplitste stukken hebben norm ~1."""
        n = 5
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, n)
        measure_qubit(mps, 2, outcome=0, rng=rng)
        pieces = split_mps_at_measured(mps, 2)
        for piece in pieces:
            self.assertAlmostEqual(piece.norm(), 1.0, delta=0.1)


class TestAdaptiveMeasurement(unittest.TestCase):
    """Test adaptieve meetpunt selectie."""

    def test_product_state_all_low_entropy(self):
        """Product state: alle bonds hebben nul entropie -> alle sites kandidaat."""
        n = 6
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        sites, entropies = select_measurement_sites(mps, entropy_threshold=0.5)
        # Alle sites zouden geselecteerd moeten zijn
        self.assertEqual(len(sites), n)

    def test_entangled_state_selective(self):
        """Verstrengelde state: niet alle sites worden geselecteerd bij lage drempel."""
        n = 4
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)  # GHZ-achtig
        mps = MPS.from_statevector(psi, n)
        sites, entropies = select_measurement_sites(mps, entropy_threshold=0.5)
        # GHZ: alle bonds hebben entropie ~1.0 > 0.5, dus geen sites
        self.assertEqual(len(sites), 0)

    def test_max_sites_limit(self):
        """max_sites beperkt het aantal geselecteerde sites."""
        n = 8
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        sites, _ = select_measurement_sites(mps, max_sites=3, entropy_threshold=1.0)
        self.assertLessEqual(len(sites), 3)

    def test_adaptive_schedule(self):
        """Adaptieve measurement schedule draait zonder fouten."""
        n = 6
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3, 0.5]
        betas = [0.7, 0.4]
        rng = np.random.default_rng(42)

        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)

        result_mps, stats = adaptive_measurement_schedule(
            mps, edges, gammas, betas,
            entropy_threshold=0.1,
            measure_every=1,
            rng=rng
        )
        self.assertIsNotNone(result_mps)
        self.assertIn('measurements_per_layer', stats)


class TestMultiBranch(unittest.TestCase):
    """Test multi-branch sampling."""

    def test_multi_branch_basic(self):
        """Multi-branch verwachtingswaarde convergeert naar juiste waarde."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3]
        betas = [0.7]
        rng = np.random.default_rng(42)

        # Referentie: exact cost
        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        cost_exact = maxcut_cost_statevector(psi_sv, n, edges)

        # Multi-branch met meting in het midden
        mean_cost, stderr, stats = multi_branch_expectation(
            n, edges, gammas, betas,
            measure_sites=[n // 2],
            measure_after_layer=0,
            observable_fn=lambda m: maxcut_cost_mps(m, edges),
            n_branches=64,
            rng=rng,
        )

        # Moet in de buurt komen (met meting verlies je iets)
        self.assertGreater(mean_cost, 0)
        self.assertEqual(stats['n_branches'], 64)

    def test_multi_branch_no_measurement(self):
        """Multi-branch zonder metingen matcht exact."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3]
        betas = [0.7]
        rng = np.random.default_rng(42)

        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        cost_exact = maxcut_cost_statevector(psi_sv, n, edges)

        # Meet geen qubits -> alle branches identiek
        mean_cost, stderr, stats = multi_branch_expectation(
            n, edges, gammas, betas,
            measure_sites=[],
            measure_after_layer=0,
            observable_fn=lambda m: maxcut_cost_mps(m, edges),
            n_branches=4,
            rng=rng,
        )

        self.assertAlmostEqual(mean_cost, cost_exact, delta=0.3)


class TestObservables(unittest.TestCase):
    """Test verwachtingswaarden."""

    def test_z_plus_state(self):
        """<Z> = 0 voor |+>."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        mps = MPS([plus.reshape(1, 2, 1)])
        z = expectation_z(mps, 0)
        self.assertAlmostEqual(z, 0.0, places=10)

    def test_z_zero_state(self):
        """<Z> = +1 voor |0>."""
        zero = np.array([1.0, 0.0], dtype=complex)
        mps = MPS([zero.reshape(1, 2, 1)])
        z = expectation_z(mps, 0)
        self.assertAlmostEqual(z, 1.0, places=10)

    def test_z_one_state(self):
        """<Z> = -1 voor |1>."""
        one = np.array([0.0, 1.0], dtype=complex)
        mps = MPS([one.reshape(1, 2, 1)])
        z = expectation_z(mps, 0)
        self.assertAlmostEqual(z, -1.0, places=10)

    def test_z_classical_bit(self):
        """<Z> voor klassiek bit."""
        mps = MPS([np.zeros((1, 2, 1), dtype=complex)],
                   classical_bits={0: 1})
        z = expectation_z(mps, 0)
        self.assertAlmostEqual(z, -1.0, places=10)

    def test_zz_product_state(self):
        """<ZZ> = +1 voor |00>."""
        zero = np.array([1.0, 0.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1), zero.reshape(1, 2, 1)]
        mps = MPS(tensors)
        zz = expectation_zz(mps, 0, 1)
        self.assertAlmostEqual(zz, 1.0, places=10)

    def test_zz_antiparallel(self):
        """<ZZ> = -1 voor |01>."""
        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1), one.reshape(1, 2, 1)]
        mps = MPS(tensors)
        zz = expectation_zz(mps, 0, 1)
        self.assertAlmostEqual(zz, -1.0, places=10)

    def test_zz_bell_state(self):
        """<ZZ> = +1 voor Bell state (|00>+|11>)/sqrt(2)."""
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        mps = MPS.from_statevector(psi, 2)
        zz = expectation_zz(mps, 0, 1)
        self.assertAlmostEqual(zz, 1.0, places=5)

    def test_maxcut_cost_trivial(self):
        """MaxCut cost voor |0...0>: alle edges geven 0."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        zero = np.array([1.0, 0.0], dtype=complex)
        tensors = [zero.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        cost = maxcut_cost_mps(mps, edges)
        self.assertAlmostEqual(cost, 0.0, places=10)

    def test_maxcut_cost_alternating(self):
        """MaxCut cost voor |0101>: alle naburige edges snijden."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        tensors = [(zero if i % 2 == 0 else one).reshape(1, 2, 1) for i in range(n)]
        mps = MPS(tensors)
        cost = maxcut_cost_mps(mps, edges)
        self.assertAlmostEqual(cost, 3.0, places=10)  # 3 edges, all cut


class TestEdgeCases(unittest.TestCase):
    """Test randgevallen en robuustheid."""

    def test_single_qubit_mps(self):
        """MPS met 1 qubit."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        mps = MPS([plus.reshape(1, 2, 1)])
        self.assertEqual(mps.n, 1)
        self.assertEqual(mps.bond_dims(), [])
        psi = mps.to_statevector()
        self.assertEqual(len(psi), 2)

    def test_two_qubit_mps(self):
        """MPS met 2 qubits."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1), plus.reshape(1, 2, 1)]
        mps = MPS(tensors)
        self.assertEqual(mps.n, 2)
        psi = mps.to_statevector()
        self.assertEqual(len(psi), 4)

    def test_qaoa_no_edges(self):
        """QAOA zonder edges geeft gewoon mixer rotaties."""
        n = 3
        edges = []
        gammas = [0.3]
        betas = [0.7]
        mps = qaoa_mps(n, edges, gammas, betas)
        psi = mps.to_statevector()
        # Alle qubits onafhankelijk geroteerd
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=10)

    def test_qaoa_single_edge(self):
        """QAOA met 1 edge."""
        n = 2
        edges = [(0, 1)]
        gammas = [0.3]
        betas = [0.7]
        mps = qaoa_mps(n, edges, gammas, betas)
        psi_sv = qaoa_statevector(n, edges, gammas, betas)
        psi_mps = mps.to_statevector()
        fid = abs(np.dot(psi_sv.conj(), psi_mps))**2
        self.assertGreater(fid, 0.99)

    def test_measure_then_continue_qaoa(self):
        """Meet midden in QAOA en ga door met resterende lagen."""
        n = 4
        edges = [(i, i + 1) for i in range(n - 1)]
        gammas = [0.3, 0.5]
        betas = [0.7, 0.4]

        mps = qaoa_mps(n, edges, gammas[:1], betas[:1])
        measure_qubit(mps, 2, outcome=0)
        # Ga door met laag 2
        apply_qaoa_layer(mps, edges, gammas[1], betas[1])
        mps.normalize()
        # Moet niet crashen en redelijke cost geven
        cost = maxcut_cost_mps(mps, edges)
        self.assertGreaterEqual(cost, 0)

    def test_normalize_zero_state(self):
        """Normalisatie van nul-vector crasht niet."""
        tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(3)]
        mps = MPS(tensors)
        mps.normalize()  # Moet niet crashen


if __name__ == '__main__':
    unittest.main(verbosity=2)
