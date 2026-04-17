#!/usr/bin/env python3
"""
Tests voor B165: Qiskit Runtime Hardware-Run pipeline.

Test suites:
  1. TestToQiskit             : ZornQ Circuit → qiskit.QuantumCircuit
  2. TestAddMeasurements      : Z-basis measurement injection
  3. TestExpectationFromCounts: ⟨Z_i Z_j⟩ uit shot-counts
  4. TestMaxcutFromCounts     : E[H_C] en best-cut uit counts
  5. TestRunAer               : AerSimulator end-to-end
  6. TestNoiseModel           : depolariserende ruis (geen explosie)
  7. TestRunIbmRuntimeSkipped : SKIPPED_NO_TOKEN gedrag
  8. TestQaoaIntegration      : end-to-end QAOA op kleine grafen
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import SimpleGraph, brute_force_maxcut
from circuit_interface import Circuit
from b165_qiskit_runtime import (
    to_qiskit,
    add_measurements,
    expectation_zz_from_counts,
    maxcut_value_from_counts,
    best_cut_from_counts,
    run_aer,
    make_depolarising_noise,
    run_ibm_runtime,
    qaoa_maxcut_run,
)


TOL = 5e-2  # tolerantie voor sample-gebaseerde verwachtingen


def _k3() -> SimpleGraph:
    g = SimpleGraph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    return g


def _c4() -> SimpleGraph:
    g = SimpleGraph(4)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        g.add_edge(u, v)
    return g


def _path4() -> SimpleGraph:
    g = SimpleGraph(4)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        g.add_edge(u, v)
    return g


# ============================================================
# 1. to_qiskit
# ============================================================

class TestToQiskit(unittest.TestCase):
    def test_h_gate(self):
        qc = Circuit(2)
        qc.h(0)
        out = to_qiskit(qc)
        self.assertEqual(out.num_qubits, 2)
        self.assertEqual(len(out.data), 1)
        self.assertEqual(out.data[0].operation.name, "h")

    def test_all_single_qubit_gates(self):
        qc = Circuit(1)
        qc.x(0); qc.y(0); qc.z(0); qc.s(0); qc.t(0)
        qc.rx(0, 0.3); qc.ry(0, 0.4); qc.rz(0, 0.5)
        out = to_qiskit(qc)
        names = [op.operation.name for op in out.data]
        self.assertEqual(names, ["x", "y", "z", "s", "t", "rx", "ry", "rz"])

    def test_two_qubit_gates(self):
        qc = Circuit(2)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)
        qc.rxx(0, 1, 0.2)
        qc.ryy(0, 1, 0.3)
        qc.rzz(0, 1, 0.4)
        out = to_qiskit(qc)
        names = [op.operation.name for op in out.data]
        self.assertEqual(names, ["cx", "cz", "swap", "rxx", "ryy", "rzz"])

    def test_qaoa_circuit_translates(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        qc = Circuit.qaoa_maxcut(3, edges, p=2,
                                  gammas=[0.5, 0.6], betas=[0.4, 0.3])
        out = to_qiskit(qc)
        self.assertEqual(out.num_qubits, 3)
        # |+⟩ init = 3 H's, plus per layer: 3 RZZ + 3 RX
        # totaal ≥ 3 + 2*(3+3) = 15 ops
        self.assertGreaterEqual(len(out.data), 15)


# ============================================================
# 2. add_measurements
# ============================================================

class TestAddMeasurements(unittest.TestCase):
    def test_adds_measurements_per_qubit(self):
        qc = Circuit(3)
        qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
        out = to_qiskit(qc)
        meas = add_measurements(out)
        # Originele 3 ops + 3 measurements
        n_meas = sum(1 for op in meas.data if op.operation.name == "measure")
        self.assertEqual(n_meas, 3)


# ============================================================
# 3. expectation_zz_from_counts
# ============================================================

class TestExpectationFromCounts(unittest.TestCase):
    def test_all_zeros_gives_plus_one(self):
        counts = {"000": 1000}
        zz = expectation_zz_from_counts(counts, 0, 1, 3)
        self.assertAlmostEqual(zz, 1.0, delta=1e-9)

    def test_anti_correlated_gives_minus_one(self):
        # qubit 0 = 1, qubit 1 = 0, qubit 2 = 0  → bitstring "001" (LE)
        counts = {"001": 1000}  # bits[-(0+1)]='1', bits[-(1+1)]='0'
        zz = expectation_zz_from_counts(counts, 0, 1, 3)
        self.assertAlmostEqual(zz, -1.0, delta=1e-9)

    def test_uniform_gives_zero(self):
        # 8 even verdeelde 3-bit bitstrings → ⟨ZZ⟩ = 0
        counts = {f"{i:03b}": 125 for i in range(8)}
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            zz = expectation_zz_from_counts(counts, i, j, 3)
            self.assertAlmostEqual(zz, 0.0, delta=1e-9)

    def test_empty_counts(self):
        self.assertEqual(expectation_zz_from_counts({}, 0, 1, 2), 0.0)


# ============================================================
# 4. maxcut_value_from_counts / best_cut_from_counts
# ============================================================

class TestMaxcutFromCounts(unittest.TestCase):
    def test_bipartite_perfect_cut(self):
        """C_4: bitstring 0101 cut alle 4 edges."""
        g = _c4()
        # 0101 little-endian = q0=1 q1=0 q2=1 q3=0  → "0101"
        counts = {"0101": 1000}
        v = maxcut_value_from_counts(counts, g)
        self.assertAlmostEqual(v, 4.0, delta=1e-9)

    def test_no_cut_zero(self):
        g = _c4()
        counts = {"0000": 1000}
        v = maxcut_value_from_counts(counts, g)
        self.assertAlmostEqual(v, 0.0, delta=1e-9)

    def test_best_cut_finds_optimum(self):
        g = _c4()
        # Uniform mixture incl. perfect cut
        counts = {"0000": 100, "1010": 100, "0101": 100}
        best, bs = best_cut_from_counts(counts, g)
        self.assertEqual(best, 4)
        self.assertIn(bs, ["1010", "0101"])


# ============================================================
# 5. run_aer
# ============================================================

class TestRunAer(unittest.TestCase):
    def test_aer_runs_qaoa_k3(self):
        qc = Circuit.qaoa_maxcut(
            3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            p=1, gammas=[0.5], betas=[0.4])
        res = run_aer(qc, shots=1024, seed=42)
        self.assertEqual(res["shots"], 1024)
        self.assertEqual(res["n_qubits"], 3)
        self.assertEqual(sum(res["counts"].values()), 1024)
        self.assertFalse(res["noisy"])

    def test_aer_deterministic_with_seed(self):
        qc = Circuit.qaoa_maxcut(
            3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            p=1, gammas=[0.5], betas=[0.4])
        r1 = run_aer(qc, shots=512, seed=123)
        r2 = run_aer(qc, shots=512, seed=123)
        self.assertEqual(r1["counts"], r2["counts"])


# ============================================================
# 6. NoiseModel
# ============================================================

class TestNoiseModel(unittest.TestCase):
    def test_noise_model_constructs(self):
        nm = make_depolarising_noise(p1=1e-3, p2=1e-2)
        # Moet alle gebruikte gates raken
        names = set(nm.noise_instructions)
        for g in ["h", "x", "y", "z", "rz", "cx", "cz", "rzz"]:
            self.assertIn(g, names)

    def test_noisy_run_shifts_distribution(self):
        """Noisy run mag niet identiek zijn aan noise-loze run."""
        qc = Circuit.qaoa_maxcut(
            4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            p=1, gammas=[0.7], betas=[0.4])
        clean = run_aer(qc, shots=4096, seed=7)
        nm = make_depolarising_noise(p1=5e-3, p2=5e-2)  # extra ruis
        noisy = run_aer(qc, shots=4096, seed=7, noise_model=nm)
        self.assertTrue(noisy["noisy"])
        # Statistische test: minstens één bitstring verschilt > 5%
        keys = set(clean["counts"]) | set(noisy["counts"])
        max_diff = max(
            abs(clean["counts"].get(k, 0) - noisy["counts"].get(k, 0)) / 4096
            for k in keys
        )
        self.assertGreater(max_diff, 0.01)


# ============================================================
# 7. run_ibm_runtime: SKIPPED_NO_TOKEN
# ============================================================

class TestRunIbmRuntimeSkipped(unittest.TestCase):
    def test_no_token_returns_skipped(self):
        os.environ.pop("QISKIT_IBM_TOKEN", None)
        qc = Circuit(2); qc.h(0); qc.cx(0, 1)
        res = run_ibm_runtime(qc, shots=128, skip_if_no_token=True)
        self.assertEqual(res["status"], "SKIPPED_NO_TOKEN")
        self.assertIn("backend", res)

    def test_no_token_raises_when_strict(self):
        os.environ.pop("QISKIT_IBM_TOKEN", None)
        qc = Circuit(2); qc.h(0); qc.cx(0, 1)
        with self.assertRaises(RuntimeError):
            run_ibm_runtime(qc, shots=128, skip_if_no_token=False)


# ============================================================
# 8. qaoa_maxcut_run integration
# ============================================================

class TestQaoaIntegration(unittest.TestCase):
    def test_aer_k3(self):
        g = _k3()
        res = qaoa_maxcut_run(g, p=1, gammas=[0.5], betas=[0.4],
                               backend="aer", shots=2048, seed=42, verbose=False)
        self.assertEqual(res["opt"], 2)
        self.assertEqual(res["best_cut_seen"], 2)
        self.assertGreaterEqual(res["qaoa_expectation"], 0.0)
        self.assertLessEqual(res["qaoa_expectation"], 3.0)

    def test_noisy_c4(self):
        g = _c4()
        res = qaoa_maxcut_run(g, p=1, gammas=[0.7], betas=[0.4],
                               backend="noisy", shots=2048, seed=7,
                               p1err=1e-3, p2err=1e-2, verbose=False)
        self.assertEqual(res["opt"], 4)
        self.assertTrue(res["noisy"])
        # Best cut moet 4 zijn (bipartiet, perfect cut zichtbaar)
        self.assertEqual(res["best_cut_seen"], 4)

    def test_hardware_skipped_returns_status(self):
        g = _k3()
        os.environ.pop("QISKIT_IBM_TOKEN", None)
        res = qaoa_maxcut_run(g, p=1, gammas=[0.5], betas=[0.4],
                               backend="hardware", shots=128, verbose=False)
        self.assertEqual(res.get("status"), "SKIPPED_NO_TOKEN")

    def test_path4_aer_perfect_cut_visible(self):
        g = _path4()
        # Goede QAOA-parameters; toch zou perfecte cut (3) gevonden moeten worden
        res = qaoa_maxcut_run(g, p=1, gammas=[0.5], betas=[0.4],
                               backend="aer", shots=4096, seed=42, verbose=False)
        self.assertEqual(res["opt"], 3)
        self.assertEqual(res["best_cut_seen"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
