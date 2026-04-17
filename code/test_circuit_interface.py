#!/usr/bin/env python3
"""
test_circuit_interface.py - Tests voor B128 Circuit Interface

Tests:
- Gate library (unitariteit, bekende waarden)
- Circuit constructie (append, depth, summary)
- State vector backend (Bell state, GHZ, bekende QAOA)
- Observable meting (Z, ZZ, MaxCut cost)
- Circuit constructors (qaoa_maxcut, hardware_efficient, trotter)
- Pauli rotation helper

Author: ZornQ project
Date: 15 april 2026
"""

import unittest
import numpy as np
import sys

from circuit_interface import (
    Gates, GateOp, Circuit, Observable,
    _run_statevector, _apply_1q_sv, _apply_2q_sv,
    _measure_observable_sv, _append_pauli_rotation,
    run_circuit,
)


class TestGates(unittest.TestCase):
    """Test gate library: unitariteit, bekende eigenwaarden, relaties."""

    def _assert_unitary(self, U, msg=""):
        prod = U.conj().T @ U
        np.testing.assert_allclose(prod, np.eye(U.shape[0]), atol=1e-12, err_msg=msg)

    def test_pauli_unitary(self):
        for name, gate_fn in [('I', Gates.I), ('X', Gates.X), ('Y', Gates.Y), ('Z', Gates.Z)]:
            self._assert_unitary(gate_fn(), msg=name)

    def test_pauli_squares(self):
        """Pauli gates kwadrateren tot identiteit."""
        I = Gates.I()
        for name, gate_fn in [('X', Gates.X), ('Y', Gates.Y), ('Z', Gates.Z)]:
            G = gate_fn()
            np.testing.assert_allclose(G @ G, I, atol=1e-12, err_msg=name)

    def test_pauli_anticommute(self):
        """XY = iZ, YZ = iX, ZX = iY."""
        X, Y, Z = Gates.X(), Gates.Y(), Gates.Z()
        np.testing.assert_allclose(X @ Y, 1j * Z, atol=1e-12)
        np.testing.assert_allclose(Y @ Z, 1j * X, atol=1e-12)
        np.testing.assert_allclose(Z @ X, 1j * Y, atol=1e-12)

    def test_hadamard(self):
        H = Gates.H()
        self._assert_unitary(H)
        np.testing.assert_allclose(H @ H, np.eye(2), atol=1e-12, err_msg="H^2 = I")

    def test_s_t_gates(self):
        self._assert_unitary(Gates.S())
        self._assert_unitary(Gates.T())
        S = Gates.S()
        np.testing.assert_allclose(S @ S, Gates.Z(), atol=1e-12, err_msg="S^2 = Z")

    def test_rotation_gates(self):
        for theta in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
            self._assert_unitary(Gates.RX(theta), msg="RX(%.2f)" % theta)
            self._assert_unitary(Gates.RY(theta), msg="RY(%.2f)" % theta)
            self._assert_unitary(Gates.RZ(theta), msg="RZ(%.2f)" % theta)

    def test_rx_pi_is_minus_iX(self):
        """RX(pi) = -iX."""
        rx = Gates.RX(np.pi)
        expected = -1j * Gates.X()
        np.testing.assert_allclose(rx, expected, atol=1e-12)

    def test_ry_pi_is_minus_iY(self):
        ry = Gates.RY(np.pi)
        expected = -1j * Gates.Y()
        np.testing.assert_allclose(ry, expected, atol=1e-12)

    def test_rz_zero_is_identity(self):
        np.testing.assert_allclose(Gates.RZ(0), Gates.I(), atol=1e-12)

    def test_2q_gates_unitary(self):
        for name, gate_fn in [('CNOT', Gates.CNOT), ('CZ', Gates.CZ), ('SWAP', Gates.SWAP)]:
            self._assert_unitary(gate_fn(), msg=name)

    def test_rzz_unitary(self):
        for theta in [0, 0.5, np.pi]:
            self._assert_unitary(Gates.RZZ(theta), msg="RZZ(%.2f)" % theta)

    def test_rzz_diagonal(self):
        """RZZ diag vector moet overeenkomen met diagonaal van RZZ matrix."""
        theta = 0.7
        diag = Gates.RZZ_diag(theta)
        full = Gates.RZZ(theta)
        np.testing.assert_allclose(np.diag(full), diag, atol=1e-12)

    def test_rxx_ryy_unitary(self):
        for theta in [0, 0.5, np.pi]:
            self._assert_unitary(Gates.RXX(theta))
            self._assert_unitary(Gates.RYY(theta))

    def test_xxplusyy(self):
        self._assert_unitary(Gates.XXplusYY(0.5, 0.3))

    def test_custom_1q(self):
        H = Gates.custom_1q(Gates.H())
        np.testing.assert_allclose(H, Gates.H(), atol=1e-12)

    def test_custom_2q(self):
        CNOT = Gates.custom_2q(Gates.CNOT())
        np.testing.assert_allclose(CNOT, Gates.CNOT(), atol=1e-12)


class TestCircuit(unittest.TestCase):
    """Test Circuit constructie en methodes."""

    def test_empty_circuit(self):
        qc = Circuit(4)
        self.assertEqual(qc.n_qubits, 4)
        self.assertEqual(len(qc), 0)
        self.assertEqual(qc.depth(), 0)

    def test_append_gates(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        self.assertEqual(len(qc), 2)
        self.assertEqual(qc.ops[0].name, 'H')
        self.assertEqual(qc.ops[1].name, 'CNOT')

    def test_depth_calculation(self):
        # H(0), H(1) parallel -> depth 1
        # CNOT(0,1) -> depth 2
        qc = Circuit(2)
        qc.h(0).h(1).cx(0, 1)
        self.assertEqual(qc.depth(), 2)

    def test_chaining(self):
        qc = Circuit(3)
        result = qc.h(0).x(1).cx(0, 2).rz(1, 0.5)
        self.assertIs(result, qc)
        self.assertEqual(len(qc), 4)

    def test_summary(self):
        qc = Circuit(2, name="test")
        qc.h(0).h(1).cx(0, 1)
        s = qc.summary()
        self.assertEqual(s['name'], 'test')
        self.assertEqual(s['n_qubits'], 2)
        self.assertEqual(s['n_gates'], 3)
        self.assertEqual(s['gate_counts']['H'], 2)
        self.assertEqual(s['gate_counts']['CNOT'], 1)

    def test_all_1q_gates(self):
        qc = Circuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.rx(0, 0.5).ry(0, 0.5).rz(0, 0.5)
        self.assertEqual(len(qc), 9)

    def test_all_2q_gates(self):
        qc = Circuit(2)
        qc.cx(0, 1).cz(0, 1).swap(0, 1)
        qc.rzz(0, 1, 0.5).rxx(0, 1, 0.5).ryy(0, 1, 0.5)
        self.assertEqual(len(qc), 6)

    def test_custom_gate(self):
        qc = Circuit(2)
        qc.apply_1q(0, Gates.H(), name="MyH")
        qc.apply_2q(0, 1, Gates.CNOT(), name="MyCX")
        self.assertEqual(qc.ops[0].name, "MyH")
        self.assertEqual(qc.ops[1].name, "MyCX")


class TestStateVector(unittest.TestCase):
    """Test state vector backend met bekende circuits."""

    def test_identity_circuit(self):
        """Leeg circuit geeft |0...0>."""
        qc = Circuit(3)
        state, _ = _run_statevector(qc)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-12)

    def test_x_gate(self):
        """X|0> = |1>."""
        qc = Circuit(1)
        qc.x(0)
        state, _ = _run_statevector(qc)
        np.testing.assert_allclose(abs(state[1]), 1.0, atol=1e-12)

    def test_hadamard_superposition(self):
        """H|0> = |+> = (|0> + |1>)/sqrt(2)."""
        qc = Circuit(1)
        qc.h(0)
        state, _ = _run_statevector(qc)
        s = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(state, [s, s], atol=1e-12)

    def test_bell_state(self):
        """H(0), CNOT(0,1) -> Bell state (|00> + |11>)/sqrt(2)."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        state, _ = _run_statevector(qc)
        s = 1.0 / np.sqrt(2)
        expected = np.array([s, 0, 0, s], dtype=np.complex128)
        np.testing.assert_allclose(state, expected, atol=1e-12)

    def test_ghz_state(self):
        """3-qubit GHZ: (|000> + |111>)/sqrt(2)."""
        qc = Circuit(3)
        qc.h(0).cx(0, 1).cx(1, 2)
        state, _ = _run_statevector(qc)
        s = 1.0 / np.sqrt(2)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = s
        expected[7] = s
        np.testing.assert_allclose(state, expected, atol=1e-12)

    def test_rz_phase(self):
        """RZ(theta)|1> = exp(i*theta/2)|1>."""
        theta = 0.7
        qc = Circuit(1)
        qc.x(0).rz(0, theta)
        state, _ = _run_statevector(qc)
        np.testing.assert_allclose(abs(state[0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(abs(state[1]), 1.0, atol=1e-12)
        np.testing.assert_allclose(state[1], np.exp(1j * theta / 2), atol=1e-12)

    def test_swap_gate(self):
        """SWAP|10> = |01>."""
        qc = Circuit(2)
        qc.x(0).swap(0, 1)  # |10> -> |01>
        state, _ = _run_statevector(qc)
        # |01> in little-endian: qubit 0 = 0, qubit 1 = 1 -> index 2
        np.testing.assert_allclose(abs(state[2]), 1.0, atol=1e-12)

    def test_cz_gate(self):
        """CZ|11> = -|11>."""
        qc = Circuit(2)
        qc.x(0).x(1).cz(0, 1)
        state, _ = _run_statevector(qc)
        np.testing.assert_allclose(state[3], -1.0, atol=1e-12)

    def test_normalization(self):
        """State vector blijft genormaliseerd na willekeurig circuit."""
        np.random.seed(42)
        qc = Circuit(4)
        for _ in range(10):
            q = np.random.randint(4)
            qc.h(q)
        for _ in range(5):
            q1, q2 = np.random.choice(4, 2, replace=False)
            qc.cx(int(q1), int(q2))
        state, _ = _run_statevector(qc)
        norm = np.linalg.norm(state)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_2q_gate_qubit_order(self):
        """CNOT(1,0) moet correct werken (control=1, target=0)."""
        qc = Circuit(2)
        qc.x(1).cx(1, 0)  # |10> -> CNOT met control=1 -> |11>
        state, _ = _run_statevector(qc)
        np.testing.assert_allclose(abs(state[3]), 1.0, atol=1e-12)


class TestObservable(unittest.TestCase):
    """Test Observable constructie en meting."""

    def test_z_observable(self):
        """<0|Z|0> = 1, <1|Z|1> = -1."""
        # |0>
        qc0 = Circuit(1)
        obs = {'z0': Observable.z(0)}
        _, results = _run_statevector(qc0, obs)
        self.assertAlmostEqual(results['z0'], 1.0, places=10)

        # |1>
        qc1 = Circuit(1)
        qc1.x(0)
        _, results = _run_statevector(qc1, obs)
        self.assertAlmostEqual(results['z0'], -1.0, places=10)

    def test_z_superposition(self):
        """<+|Z|+> = 0."""
        qc = Circuit(1)
        qc.h(0)
        _, results = _run_statevector(qc, {'z': Observable.z(0)})
        self.assertAlmostEqual(results['z'], 0.0, places=10)

    def test_zz_observable(self):
        """Bell state: <ZZ> = 1 (altijd gecorreleerd)."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        obs = {'zz': Observable.zz(0, 1)}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['zz'], 1.0, places=10)

    def test_xx_observable(self):
        """Bell state |Phi+>: <XX> = 1."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        obs = {'xx': Observable.xx(0, 1)}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['xx'], 1.0, places=10)

    def test_maxcut_cost_simple(self):
        """MaxCut cost voor |01> op 1 edge (0,1,1): C = 1*(1 - (-1))/2 = 1."""
        qc = Circuit(2)
        qc.x(1)  # |01>
        edges = [(0, 1, 1.0)]
        obs = {'cost': Observable.maxcut_cost(edges)}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['cost'], 1.0, places=10)

    def test_maxcut_cost_same_partition(self):
        """MaxCut cost voor |00> op 1 edge: C = 0."""
        qc = Circuit(2)
        edges = [(0, 1, 1.0)]
        obs = {'cost': Observable.maxcut_cost(edges)}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['cost'], 0.0, places=10)

    def test_maxcut_cost_triangle(self):
        """MaxCut op driehoek: |010> snijdt edges (0,1) en (1,2), cost = 2."""
        qc = Circuit(3)
        qc.x(1)
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        obs = {'cost': Observable.maxcut_cost(edges)}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['cost'], 2.0, places=10)

    def test_heisenberg_observable(self):
        obs = Observable.heisenberg(3, J=1.0)
        # 2 edges * 3 terms = 6 terms
        self.assertEqual(len(obs.terms), 6)

    def test_ising_transverse(self):
        obs = Observable.ising_transverse(4, J=1.0, h=0.5)
        # 3 ZZ terms + 4 X terms = 7
        self.assertEqual(len(obs.terms), 7)


class TestQAOACircuit(unittest.TestCase):
    """Test QAOA MaxCut circuit constructie en resultaten."""

    def test_qaoa_construction(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        qc = Circuit.qaoa_maxcut(3, edges, p=1, gammas=[0.5], betas=[0.3])
        self.assertEqual(qc.n_qubits, 3)
        self.assertGreater(len(qc), 0)
        self.assertEqual(qc.metadata['type'], 'qaoa')
        self.assertEqual(qc.metadata['p'], 1)

    def test_qaoa_p2(self):
        edges = [(0, 1, 1.0)]
        qc = Circuit.qaoa_maxcut(2, edges, p=2, gammas=[0.5, 0.3], betas=[0.4, 0.2])
        s = qc.summary()
        # 2 H + 2*(1 RZZ + 2 RX) = 2 + 2*3 = 8
        self.assertEqual(s['n_gates'], 8)

    def test_qaoa_maxcut_value(self):
        """QAOA p=1 op enkel edge: optimale gamma, beta geeft cut > 0."""
        edges = [(0, 1, 1.0)]
        # Optimale p=1 voor enkel edge met 2*gamma*w RZZ conventie
        gamma = 0.74 * np.pi
        beta = 0.12 * np.pi
        qc = Circuit.qaoa_maxcut(2, edges, p=1, gammas=[gamma], betas=[beta])
        obs = {'cost': Observable.maxcut_cost(edges)}
        _, results = _run_statevector(qc, obs)
        # Moet significant > 0 zijn (optimaal ~1.0 voor enkel edge)
        self.assertGreater(results['cost'], 0.9)

    def test_qaoa_normalization(self):
        """QAOA state blijft genormaliseerd."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        qc = Circuit.qaoa_maxcut(3, edges, p=2, gammas=[0.5, 0.3], betas=[0.4, 0.2])
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestHardwareEfficient(unittest.TestCase):
    """Test hardware-efficient VQE ansatz."""

    def test_construction(self):
        qc = Circuit.hardware_efficient(4, depth=2)
        self.assertEqual(qc.n_qubits, 4)
        self.assertGreater(len(qc), 0)
        self.assertEqual(qc.metadata['type'], 'vqe')

    def test_param_count(self):
        n, d = 4, 2
        expected = n * 2 * (d + 1) + (n - 1) * d
        qc = Circuit.hardware_efficient(n, depth=d)
        self.assertEqual(qc.metadata['n_params'], expected)

    def test_deterministic_with_params(self):
        params = np.zeros(20)  # n=3, d=2: 3*2*3 + 2*2 = 22... let me compute
        n, d = 3, 2
        n_params = n * 2 * (d + 1) + (n - 1) * d
        params = np.zeros(n_params)
        qc = Circuit.hardware_efficient(n, depth=d, params=params)
        state, _ = _run_statevector(qc)
        # Met alle params=0: RY(0)=I, RZ(0)=I, dus alleen CNOT ladders
        # Start met |000>, CNOT ladders don't change |000>
        np.testing.assert_allclose(abs(state[0]), 1.0, atol=1e-12)

    def test_normalization(self):
        np.random.seed(123)
        qc = Circuit.hardware_efficient(5, depth=3)
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestTrotterEvolution(unittest.TestCase):
    """Test Trotter tijdsevolutie."""

    def test_single_z_term(self):
        """exp(-iZt) op |0> geeft phase."""
        t = 0.5
        terms = [(1.0, {0: 'Z'})]
        qc = Circuit.trotter_evolution(1, terms, t=t, n_steps=1)
        state, _ = _run_statevector(qc)
        # |0> eigenvector van Z met eigenvalue +1
        # exp(-i*1*Z*0.5)|0> = exp(-i*0.5)|0>
        expected_phase = np.exp(-1j * t)
        np.testing.assert_allclose(state[0], expected_phase, atol=1e-12)

    def test_trotter_normalization(self):
        terms = [(1.0, {0: 'Z', 1: 'Z'}), (0.5, {0: 'X'})]
        qc = Circuit.trotter_evolution(2, terms, t=1.0, n_steps=5)
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)

    def test_trotter_construction(self):
        terms = [(1.0, {0: 'Z', 1: 'Z'})]
        qc = Circuit.trotter_evolution(2, terms, t=1.0, n_steps=10)
        self.assertEqual(qc.metadata['type'], 'trotter')
        # 10 steps * 1 RZZ per step = 10 gates
        self.assertEqual(len(qc), 10)


class TestPauliRotation(unittest.TestCase):
    """Test generieke Pauli-string rotatie."""

    def test_single_z_rotation(self):
        """Pauli rotation met enkele Z = RZ."""
        theta = 0.3
        qc1 = Circuit(1)
        qc1.h(0)
        _append_pauli_rotation(qc1, theta, {0: 'Z'})
        state1, _ = _run_statevector(qc1)

        qc2 = Circuit(1)
        qc2.h(0).rz(0, 2 * theta)
        state2, _ = _run_statevector(qc2)

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_xx_rotation(self):
        """Pauli rotation met XX moet gelijk zijn aan RXX."""
        theta = 0.4
        qc1 = Circuit(2)
        qc1.h(0).h(1)
        _append_pauli_rotation(qc1, theta, {0: 'X', 1: 'X'})
        state1, _ = _run_statevector(qc1)

        qc2 = Circuit(2)
        qc2.h(0).h(1)
        qc2.rxx(0, 1, 2 * theta)
        state2, _ = _run_statevector(qc2)

        # Globale fase mag verschillen, check overlap
        overlap = abs(np.vdot(state1, state2))
        self.assertAlmostEqual(overlap, 1.0, places=10)


class TestRunCircuit(unittest.TestCase):
    """Test unified run_circuit interface."""

    def test_auto_selects_statevector(self):
        qc = Circuit(4)
        qc.h(0).cx(0, 1)
        result = run_circuit(qc, backend='auto')
        self.assertEqual(result['backend'], 'statevector')

    def test_statevector_with_observable(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        obs = {'zz': Observable.zz(0, 1)}
        result = run_circuit(qc, observables=obs, backend='statevector')
        self.assertAlmostEqual(result['observables']['zz'], 1.0, places=10)
        self.assertIn('time', result)
        self.assertIn('state', result)

    def test_invalid_backend(self):
        qc = Circuit(2)
        with self.assertRaises(ValueError):
            run_circuit(qc, backend='invalid')


class TestQAOAFromHamiltonian(unittest.TestCase):
    """Test QAOA met willekeurige Hamiltonian."""

    def test_zz_hamiltonian_matches_maxcut(self):
        """QAOA from ZZ Hamiltonian moet zelfde resultaat geven als MaxCut QAOA."""
        edges = [(0, 1, 1.0)]
        gamma, beta = 0.5, 0.3

        qc1 = Circuit.qaoa_maxcut(2, edges, p=1, gammas=[gamma], betas=[beta])
        state1, _ = _run_statevector(qc1)

        terms = [(1.0, {0: 'Z', 1: 'Z'})]
        qc2 = Circuit.qaoa_from_hamiltonian(2, terms, p=1, gammas=[gamma], betas=[beta])
        state2, _ = _run_statevector(qc2)

        # RZZ(2*gamma*w) voor maxcut vs RZZ(2*gamma*coeff) voor ham
        # Met w=1, coeff=1 moeten ze gelijk zijn
        overlap = abs(np.vdot(state1, state2))
        self.assertAlmostEqual(overlap, 1.0, places=10)

    def test_mixed_hamiltonian(self):
        """QAOA met gemengde ZZ + X termen."""
        terms = [(1.0, {0: 'Z', 1: 'Z'}), (0.5, {0: 'X'})]
        qc = Circuit.qaoa_from_hamiltonian(2, terms, p=1, gammas=[0.5], betas=[0.3])
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestEdgeCases(unittest.TestCase):
    """Edge cases en error handling."""

    def test_too_many_qubits_statevector(self):
        qc = Circuit(30)
        with self.assertRaises(ValueError):
            _run_statevector(qc)

    def test_no_observables(self):
        qc = Circuit(2)
        qc.h(0)
        state, results = _run_statevector(qc)
        self.assertEqual(results, {})

    def test_empty_observable(self):
        """Observable met alleen constante term."""
        qc = Circuit(1)
        obs = {'const': Observable([(3.14, {})])}
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['const'], 3.14, places=10)

    def test_rzz_symmetry(self):
        """RZZ(q1,q2) = RZZ(q2,q1) in effect."""
        qc1 = Circuit(3)
        qc1.h(0).h(1).h(2).rzz(0, 2, 0.5)
        state1, _ = _run_statevector(qc1)

        qc2 = Circuit(3)
        qc2.h(0).h(1).h(2).rzz(2, 0, 0.5)
        state2, _ = _run_statevector(qc2)

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_multiple_observables(self):
        """Meerdere observables tegelijk meten."""
        qc = Circuit(2)
        qc.x(0)  # |10>
        obs = {
            'z0': Observable.z(0),
            'z1': Observable.z(1),
            'zz': Observable.zz(0, 1),
        }
        _, results = _run_statevector(qc, obs)
        self.assertAlmostEqual(results['z0'], -1.0, places=10)  # qubit 0 in |1>
        self.assertAlmostEqual(results['z1'], 1.0, places=10)   # qubit 1 in |0>
        self.assertAlmostEqual(results['zz'], -1.0, places=10)  # anti-correlated


if __name__ == '__main__':
    unittest.main()
