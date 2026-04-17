#!/usr/bin/env python3
"""
test_hamiltonian_compiler.py - Tests voor B129 Hamiltonian Compiler
"""

import unittest
import numpy as np
import sys

from hamiltonian_compiler import (
    Hamiltonian, CircuitOptimizer, compile_hamiltonian,
    jordan_wigner_number, jordan_wigner_hopping,
    jordan_wigner_interaction, jordan_wigner_two_body,
    _multiply_single_paulis,
)
from circuit_interface import (
    Circuit, Gates, Observable,
    _run_statevector, run_circuit,
)


class TestHamiltonianBasics(unittest.TestCase):
    """Test Hamiltonian constructie en basis operaties."""

    def test_empty(self):
        H = Hamiltonian()
        self.assertEqual(H.n_terms, 0)
        self.assertEqual(H.n_qubits, 0)

    def test_add_term(self):
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, {0: 'Z', 1: 'Z'})
        H.add_term(0.5, {0: 'X'})
        self.assertEqual(H.n_terms, 2)
        self.assertEqual(H.n_qubits, 2)

    def test_addition(self):
        H1 = Hamiltonian([(1.0, {0: 'Z'})], 2, "H1")
        H2 = Hamiltonian([(0.5, {1: 'X'})], 2, "H2")
        H3 = H1 + H2
        self.assertEqual(H3.n_terms, 2)
        self.assertEqual(H3.n_qubits, 2)

    def test_scalar_multiply(self):
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        H2 = 2.0 * H
        self.assertAlmostEqual(H2.terms[0][0], 2.0)
        H3 = H * 3.0
        self.assertAlmostEqual(H3.terms[0][0], 3.0)

    def test_simplify(self):
        H = Hamiltonian([
            (1.0, {0: 'Z'}),
            (2.0, {0: 'Z'}),
            (0.5, {1: 'X'}),
        ], 2)
        H.simplify()
        self.assertEqual(H.n_terms, 2)
        # Z term should be combined to 3.0
        for c, p in H.terms:
            if 0 in p and p[0] == 'Z':
                self.assertAlmostEqual(c, 3.0)

    def test_simplify_removes_zeros(self):
        H = Hamiltonian([
            (1.0, {0: 'Z'}),
            (-1.0, {0: 'Z'}),
        ], 1)
        H.simplify()
        self.assertEqual(H.n_terms, 0)

    def test_norm(self):
        H = Hamiltonian([(1.0, {0: 'Z'}), (-2.0, {1: 'X'})], 2)
        self.assertAlmostEqual(H.norm(), 3.0)

    def test_to_observable(self):
        H = Hamiltonian([(1.0, {0: 'Z', 1: 'Z'})], 2)
        obs = H.to_observable()
        self.assertEqual(len(obs.terms), 1)

    def test_is_diagonal(self):
        H_diag = Hamiltonian([(1.0, {0: 'Z'}), (0.5, {0: 'Z', 1: 'Z'})], 2)
        self.assertTrue(H_diag.is_diagonal())
        H_not = Hamiltonian([(1.0, {0: 'X'})], 1)
        self.assertFalse(H_not.is_diagonal())

    def test_locality(self):
        H = Hamiltonian([
            (1.0, {0: 'Z'}),
            (1.0, {0: 'Z', 1: 'Z', 2: 'Z'}),
        ], 3)
        self.assertEqual(H.locality(), 3)

    def test_pauli_weight(self):
        H = Hamiltonian([
            (1.0, {0: 'Z'}),           # weight 1
            (1.0, {0: 'Z', 1: 'Z'}),   # weight 2
        ], 2)
        self.assertAlmostEqual(H.pauli_weight(), 1.5)


class TestModelConstructors(unittest.TestCase):
    """Test Hamiltonian model constructors."""

    def test_ising_transverse(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        # 3 ZZ terms + 4 X terms = 7
        self.assertEqual(H.n_terms, 7)
        self.assertEqual(H.n_qubits, 4)

    def test_ising_periodic(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5, periodic=True)
        # 4 ZZ terms + 4 X terms = 8
        self.assertEqual(H.n_terms, 8)

    def test_heisenberg_xxx(self):
        H = Hamiltonian.heisenberg_xxx(4, J=1.0)
        # 3 bonds * 3 terms = 9
        self.assertEqual(H.n_terms, 9)
        self.assertEqual(H.n_qubits, 4)

    def test_heisenberg_xxz(self):
        H = Hamiltonian.heisenberg_xxz(4, Jxy=1.0, Jz=0.5)
        self.assertEqual(H.n_terms, 9)

    def test_heisenberg_xxz_field(self):
        H = Hamiltonian.heisenberg_xxz(4, Jxy=1.0, Jz=0.5, hz=0.3)
        # 9 bond terms + 4 field terms
        self.assertEqual(H.n_terms, 13)

    def test_maxcut(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        H = Hamiltonian.maxcut(3, edges)
        # 2 edges * 2 terms = 4
        self.assertEqual(H.n_terms, 4)

    def test_hubbard_1d(self):
        H = Hamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        self.assertEqual(H.n_qubits, 4)
        self.assertGreater(H.n_terms, 0)

    def test_hubbard_larger(self):
        H = Hamiltonian.hubbard_1d(4, t=1.0, U=2.0)
        self.assertEqual(H.n_qubits, 8)
        self.assertGreater(H.n_terms, 0)

    def test_custom_string(self):
        H = Hamiltonian.custom(3, [
            (1.0, "XZI"),
            (-0.5, "ZZZ"),
        ])
        self.assertEqual(H.n_terms, 2)
        self.assertEqual(H.n_qubits, 3)
        # Check XZI -> {0:'X', 1:'Z'}
        self.assertEqual(H.terms[0][1], {0: 'X', 1: 'Z'})

    def test_custom_dict(self):
        H = Hamiltonian.custom(2, [
            (1.0, {0: 'Z', 1: 'Z'}),
        ])
        self.assertEqual(H.n_terms, 1)

    def test_from_openfermion_str(self):
        s = "0.5 [X0 Z1 Y2] + -0.3 [Z0 Z1]"
        H = Hamiltonian.from_openfermion_str(s)
        self.assertEqual(H.n_terms, 2)
        self.assertEqual(H.n_qubits, 3)
        self.assertAlmostEqual(H.terms[0][0], 0.5)

    def test_molecular_simple(self):
        # Simpele 2-qubit molecuul (H2 minimal)
        one_body = {(0, 0): -1.0, (1, 1): -0.5}
        two_body = {(0, 0, 1, 1): 0.25}
        H = Hamiltonian.molecular(one_body, two_body, n_qubits=2)
        self.assertEqual(H.n_qubits, 2)
        self.assertGreater(H.n_terms, 0)

    def test_molecular_array(self):
        n = 2
        ob = np.array([[-1.0, 0.1], [0.1, -0.5]])
        tb = np.zeros((n, n, n, n))
        tb[0, 0, 1, 1] = 0.25
        H = Hamiltonian.molecular(ob, tb)
        self.assertGreater(H.n_terms, 0)


class TestCommutingGroups(unittest.TestCase):
    """Test commuterende groep decompositie."""

    def test_all_commute_zz(self):
        """Alle ZZ termen commuteren."""
        H = Hamiltonian([
            (1.0, {0: 'Z', 1: 'Z'}),
            (1.0, {2: 'Z', 3: 'Z'}),
            (1.0, {0: 'Z', 3: 'Z'}),
        ], 4)
        groups = H.commuting_groups()
        # Alle ZZ commuteren -> 1 groep
        self.assertEqual(len(groups), 1)

    def test_anticommuting(self):
        """X en Z op zelfde qubit anticommuteren."""
        H = Hamiltonian([
            (1.0, {0: 'X'}),
            (1.0, {0: 'Z'}),
        ], 1)
        groups = H.commuting_groups()
        self.assertEqual(len(groups), 2)

    def test_ising_groups(self):
        """Ising: ZZ commuteren, X anticommuteert met Z op zelfde qubit."""
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        groups = H.commuting_groups()
        # ZZ termen commuteren onderling, X termen commuteren onderling
        # Maar ZZ en X op overlappende qubits niet
        self.assertGreater(len(groups), 1)
        self.assertLessEqual(len(groups), 4)

    def test_heisenberg_groups(self):
        """Heisenberg XXX heeft meer groepen nodig."""
        H = Hamiltonian.heisenberg_xxx(4)
        groups = H.commuting_groups()
        self.assertGreater(len(groups), 1)


class TestJordanWigner(unittest.TestCase):
    """Test Jordan-Wigner transformatie."""

    def test_number_operator(self):
        """n_p = (I - Z_p)/2."""
        terms = jordan_wigner_number(0)
        self.assertEqual(len(terms), 2)
        # Check: sum should give n_p
        # On |0>: n=0, on |1>: n=1
        # (I - Z)/2 on |0>: (1 - 1)/2 = 0 ✓
        # (I - Z)/2 on |1>: (1 - (-1))/2 = 1 ✓

    def test_hopping_terms(self):
        """Hopping c†_0 c_1 + h.c. geeft XX + YY termen."""
        terms = jordan_wigner_hopping(0, 1, 2)
        self.assertEqual(len(terms), 2)
        # Should have XX and YY
        paulis = [tuple(sorted(p.items())) for _, p in terms]
        self.assertIn(((0, 'X'), (1, 'X')), paulis)
        self.assertIn(((0, 'Y'), (1, 'Y')), paulis)

    def test_hopping_with_string(self):
        """Hopping over meerdere sites geeft Z-string."""
        terms = jordan_wigner_hopping(0, 3, 4)
        # Moet Z op sites 1, 2 hebben
        for _, pauli in terms:
            self.assertIn(1, pauli)
            self.assertIn(2, pauli)
            self.assertEqual(pauli[1], 'Z')
            self.assertEqual(pauli[2], 'Z')

    def test_interaction(self):
        """n_p * n_q geeft 4 termen."""
        terms = jordan_wigner_interaction(0, 1)
        self.assertEqual(len(terms), 4)

    def test_number_expectation(self):
        """<1|n_0|1> = 1 via state vector."""
        n_terms = jordan_wigner_number(0)
        obs = Observable([(float(c.real), p) for c, p in n_terms])

        # |1> state
        qc = Circuit(1)
        qc.x(0)
        _, results = _run_statevector(qc, {'n': obs})
        self.assertAlmostEqual(results['n'], 1.0, places=10)

        # |0> state
        qc0 = Circuit(1)
        _, results0 = _run_statevector(qc0, {'n': obs})
        self.assertAlmostEqual(results0['n'], 0.0, places=10)


class TestPauliMultiply(unittest.TestCase):
    """Test Pauli string vermenigvuldiging."""

    def test_xx_equals_i(self):
        c, p = _multiply_single_paulis(1.0, {0: 'X'}, {0: 'X'})
        self.assertAlmostEqual(c, 1.0)
        self.assertEqual(len(p), 0)

    def test_xy_equals_iz(self):
        c, p = _multiply_single_paulis(1.0, {0: 'X'}, {0: 'Y'})
        self.assertAlmostEqual(c, 1j)
        self.assertEqual(p, {0: 'Z'})

    def test_yz_equals_ix(self):
        c, p = _multiply_single_paulis(1.0, {0: 'Y'}, {0: 'Z'})
        self.assertAlmostEqual(c, 1j)
        self.assertEqual(p, {0: 'X'})

    def test_zx_equals_iy(self):
        c, p = _multiply_single_paulis(1.0, {0: 'Z'}, {0: 'X'})
        self.assertAlmostEqual(c, 1j)
        self.assertEqual(p, {0: 'Y'})

    def test_different_qubits(self):
        c, p = _multiply_single_paulis(1.0, {0: 'X'}, {1: 'Z'})
        self.assertAlmostEqual(c, 1.0)
        self.assertEqual(p, {0: 'X', 1: 'Z'})


class TestTrotterCompilation(unittest.TestCase):
    """Test Trotter compilatie naar circuits."""

    def test_trotter1_ising(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc = H.trotter(t=1.0, steps=5, order=1)
        self.assertEqual(qc.n_qubits, 4)
        self.assertGreater(len(qc), 0)
        self.assertEqual(qc.metadata['order'], 1)

    def test_trotter2_ising(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc = H.trotter(t=1.0, steps=5, order=2)
        self.assertGreater(len(qc), 0)
        self.assertEqual(qc.metadata['order'], 2)

    def test_trotter4_ising(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc = H.trotter(t=1.0, steps=3, order=4)
        self.assertGreater(len(qc), 0)

    def test_trotter2_more_gates(self):
        """Trotter-2 heeft meer gates dan Trotter-1 (dubbele doorloop)."""
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc1 = H.trotter(t=1.0, steps=1, order=1)
        qc2 = H.trotter(t=1.0, steps=1, order=2)
        self.assertGreater(len(qc2), len(qc1))

    def test_trotter_grouped(self):
        H = Hamiltonian.ising_transverse(4)
        qc = H.trotter_grouped(t=1.0, steps=5)
        self.assertGreater(len(qc), 0)

    def test_trotter_normalization(self):
        """State blijft genormaliseerd na Trotter evolutie."""
        H = Hamiltonian.heisenberg_xxx(4)
        qc = H.trotter(t=0.5, steps=3, order=2)
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)

    def test_trotter_order_convergence(self):
        """Hogere orde Trotter convergeert sneller.

        Voor korte tijd en veel stappen moeten T1, T2, T4 naar dezelfde state.
        """
        H = Hamiltonian.ising_transverse(3, J=0.5, h=0.3)
        t = 0.1

        qc1 = H.trotter(t, steps=20, order=1)
        qc2 = H.trotter(t, steps=20, order=2)

        state1, _ = _run_statevector(qc1)
        state2, _ = _run_statevector(qc2)

        # Bij veel stappen moeten ze dicht bij elkaar liggen
        overlap = abs(np.vdot(state1, state2))
        self.assertGreater(overlap, 0.99)


class TestQAOACompilation(unittest.TestCase):
    """Test QAOA compilatie."""

    def test_qaoa_maxcut(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        H = Hamiltonian.maxcut(3, edges)
        qc = H.qaoa(p=1, gammas=[0.5], betas=[0.3])
        self.assertEqual(qc.n_qubits, 3)
        self.assertGreater(len(qc), 0)

    def test_qaoa_ising(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc = H.qaoa(p=2, gammas=[0.5, 0.3], betas=[0.4, 0.2])
        self.assertEqual(qc.n_qubits, 4)

    def test_qaoa_normalization(self):
        H = Hamiltonian.heisenberg_xxx(4)
        qc = H.qaoa(p=2, gammas=[0.5, 0.3], betas=[0.4, 0.2])
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)

    def test_qaoa_xy_mixer(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        H = Hamiltonian.maxcut(3, edges)
        qc = H.qaoa(p=1, gammas=[0.5], betas=[0.3], mixer='XY')
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)

    def test_qaoa_custom_mixer(self):
        H = Hamiltonian([(1.0, {0: 'Z', 1: 'Z'})], 2)
        mixer = Hamiltonian([(1.0, {0: 'X'}), (1.0, {1: 'X'})], 2)
        qc = H.qaoa(p=1, gammas=[0.5], betas=[0.3], mixer=mixer)
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestCircuitOptimizer(unittest.TestCase):
    """Test gate optimalisaties."""

    def test_merge_rz(self):
        qc = Circuit(1)
        qc.rz(0, 0.3).rz(0, 0.5)
        opt = CircuitOptimizer.merge_rotations(qc)
        self.assertEqual(len(opt), 1)
        self.assertAlmostEqual(opt.ops[0].params[0], 0.8)

    def test_merge_rx(self):
        qc = Circuit(1)
        qc.rx(0, 0.2).rx(0, 0.4)
        opt = CircuitOptimizer.merge_rotations(qc)
        self.assertEqual(len(opt), 1)

    def test_cancel_hh(self):
        qc = Circuit(1)
        qc.h(0).h(0)
        opt = CircuitOptimizer.cancel_inverses(qc)
        self.assertEqual(len(opt), 0)

    def test_cancel_xx(self):
        qc = Circuit(1)
        qc.h(0).x(0).x(0).h(0)
        opt = CircuitOptimizer.cancel_inverses(qc)
        # H X X H -> H H -> (nothing after second pass)
        self.assertEqual(len(opt), 2)  # H H remains after one pass

    def test_cancel_cnot_cnot(self):
        qc = Circuit(2)
        qc.cx(0, 1).cx(0, 1)
        opt = CircuitOptimizer.cancel_inverses(qc)
        self.assertEqual(len(opt), 0)

    def test_remove_small_rotations(self):
        qc = Circuit(1)
        qc.rz(0, 1e-15).rx(0, 0.5)
        opt = CircuitOptimizer.remove_small_rotations(qc)
        self.assertEqual(len(opt), 1)
        self.assertEqual(opt.ops[0].name, 'RX')

    def test_optimize_full(self):
        qc = Circuit(2)
        qc.h(0).h(0).rz(0, 0.3).rz(0, 0.5).cx(0, 1).cx(0, 1)
        opt = CircuitOptimizer.optimize(qc)
        # H H cancels, RZ merges, CNOT CNOT cancels
        self.assertEqual(len(opt), 1)
        self.assertEqual(opt.ops[0].name, 'RZ')

    def test_optimize_preserves_state(self):
        """Optimalisatie mag de state niet veranderen."""
        qc = Circuit(2)
        qc.h(0).rz(0, 0.3).rz(0, 0.2).cx(0, 1)
        opt = CircuitOptimizer.merge_rotations(qc)

        state1, _ = _run_statevector(qc)
        state2, _ = _run_statevector(opt)
        np.testing.assert_allclose(state1, state2, atol=1e-12)


class TestCompileHamiltonian(unittest.TestCase):
    """Test convenience compile_hamiltonian functie."""

    def test_trotter(self):
        H = Hamiltonian.ising_transverse(4)
        qc = compile_hamiltonian(H, mode='trotter', t=1.0, steps=5)
        self.assertGreater(len(qc), 0)

    def test_trotter2(self):
        H = Hamiltonian.ising_transverse(4)
        qc = compile_hamiltonian(H, mode='trotter2', t=1.0, steps=5)
        self.assertGreater(len(qc), 0)

    def test_trotter4(self):
        H = Hamiltonian.ising_transverse(4)
        qc = compile_hamiltonian(H, mode='trotter4', t=1.0, steps=3)
        self.assertGreater(len(qc), 0)

    def test_qaoa(self):
        H = Hamiltonian.maxcut(3, [(0,1,1.0)])
        qc = compile_hamiltonian(H, mode='qaoa', p=1, gammas=[0.5], betas=[0.3])
        self.assertGreater(len(qc), 0)

    def test_vqe(self):
        H = Hamiltonian.ising_transverse(4)
        qc = compile_hamiltonian(H, mode='vqe', depth=2)
        self.assertGreater(len(qc), 0)

    def test_invalid_mode(self):
        H = Hamiltonian.ising_transverse(4)
        with self.assertRaises(ValueError):
            compile_hamiltonian(H, mode='invalid')


class TestHubbardPhysics(unittest.TestCase):
    """Test dat Hubbard model fysisch correcte resultaten geeft."""

    def test_hubbard_hermitian(self):
        """Hubbard Hamiltonian moet hermitisch zijn (reele coefficienten)."""
        H = Hamiltonian.hubbard_1d(3, t=1.0, U=4.0)
        for coeff, _ in H.terms:
            self.assertAlmostEqual(coeff.imag, 0.0, places=10,
                                   msg="Hubbard term niet reeel: %s" % coeff)

    def test_hubbard_trotter_normalization(self):
        """Trotter evolutie van Hubbard behoudt norm."""
        H = Hamiltonian.hubbard_1d(2, t=1.0, U=2.0)
        qc = H.trotter(t=0.5, steps=5, order=2)
        state, _ = _run_statevector(qc)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestTrotterAccuracy(unittest.TestCase):
    """Test Trotter nauwkeurigheid door vergelijking met exact."""

    def test_single_z_exact(self):
        """exp(-iZt)|0> = e^{-it}|0> (exact, 1 term)."""
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        t = 0.7
        qc = H.trotter(t, steps=1, order=1)
        state, _ = _run_statevector(qc)
        expected = np.exp(-1j * t)
        self.assertAlmostEqual(state[0], expected, places=10)

    def test_trotter2_vs_trotter1(self):
        """Trotter-2 moet nauwkeuriger zijn dan Trotter-1 voor zelfde stappen."""
        H = Hamiltonian.ising_transverse(3, J=1.0, h=0.5)
        t = 0.3
        steps = 5

        # Reference: heel veel stappen Trotter-1
        qc_ref = H.trotter(t, steps=100, order=1)
        state_ref, _ = _run_statevector(qc_ref)

        qc1 = H.trotter(t, steps=steps, order=1)
        qc2 = H.trotter(t, steps=steps, order=2)
        state1, _ = _run_statevector(qc1)
        state2, _ = _run_statevector(qc2)

        err1 = 1.0 - abs(np.vdot(state_ref, state1))
        err2 = 1.0 - abs(np.vdot(state_ref, state2))

        self.assertLess(err2, err1,
                        "Trotter-2 (err=%.2e) niet beter dan Trotter-1 (err=%.2e)" % (err2, err1))


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests: Hamiltonian -> Circuit -> Observable meting."""

    def test_ising_energy(self):
        """Bereken Ising ground state energy via Trotter + meting."""
        H = Hamiltonian.ising_transverse(3, J=1.0, h=0.5)
        qc = H.trotter(t=2.0, steps=20, order=2)
        obs = H.to_observable()
        result = run_circuit(qc, observables={'E': obs}, backend='statevector')
        # Energie moet eindig zijn
        E = result['observables']['E']
        self.assertTrue(np.isfinite(E))

    def test_maxcut_qaoa_pipeline(self):
        """MaxCut QAOA pipeline via Hamiltonian compiler."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        H = Hamiltonian.maxcut(3, edges)
        qc = H.qaoa(p=2, gammas=[0.7, 0.5], betas=[0.3, 0.2])
        obs = H.to_observable()
        result = run_circuit(qc, observables={'cost': obs}, backend='statevector')
        cost = result['observables']['cost']
        self.assertGreater(cost, 0)  # Moet positief zijn

    def test_heisenberg_energy(self):
        """Heisenberg XXX ground state approximatie."""
        H = Hamiltonian.heisenberg_xxx(4, J=1.0)
        qc = H.trotter(t=3.0, steps=30, order=2)
        obs = H.to_observable()
        result = run_circuit(qc, observables={'E': obs}, backend='statevector')
        self.assertTrue(np.isfinite(result['observables']['E']))


if __name__ == '__main__':
    unittest.main()
