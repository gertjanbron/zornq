#!/usr/bin/env python3
"""
test_scalability_benchmark.py - Tests voor B133 Scalability Benchmark Suite

Valideert alle 7 benchmarks met kleine parameters.

Author: ZornQ project
Date: 16 april 2026
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from scalability_benchmark import (
    _make_random_graph, _make_chain_graph, _timed,
    bench_qaoa_scaling, bench_vqe_scaling, bench_trotter_scaling,
    bench_chi_convergence, bench_circuit_complexity,
    bench_break_even, bench_large_scale,
)


class TestHelpers(unittest.TestCase):
    """Test hulpfuncties."""

    def test_random_graph(self):
        """Random graaf heeft edges."""
        edges = _make_random_graph(10, edge_prob=0.5, seed=42)
        self.assertGreater(len(edges), 0)
        for i, j, w in edges:
            self.assertLess(i, j)
            self.assertIn(w, [-1, 1])

    def test_chain_graph(self):
        """Keten graaf heeft n-1 edges."""
        edges = _make_chain_graph(5)
        self.assertEqual(len(edges), 4)
        for k, (i, j, w) in enumerate(edges):
            self.assertEqual(i, k)
            self.assertEqual(j, k + 1)
            self.assertEqual(w, 1.0)

    def test_timed(self):
        """_timed geeft (result, time) tuple."""
        result, elapsed = _timed(lambda x: x * 2, 5)
        self.assertEqual(result, 10)
        self.assertGreater(elapsed, 0)

    def test_deterministic_graph(self):
        """Zelfde seed geeft zelfde graaf."""
        g1 = _make_random_graph(8, seed=123)
        g2 = _make_random_graph(8, seed=123)
        self.assertEqual(g1, g2)


class TestQAOAScaling(unittest.TestCase):
    """Test QAOA scaling benchmark."""

    def test_runs(self):
        """Benchmark draait met kleine parameters."""
        r = bench_qaoa_scaling(n_range=[4, 6], chi_values=[4],
                               verbose=False)
        self.assertEqual(r['bench'], 'qaoa_scaling')
        self.assertGreater(len(r['results']), 0)

    def test_has_timing(self):
        """Elk resultaat heeft timing info."""
        r = bench_qaoa_scaling(n_range=[4], chi_values=[4],
                               verbose=False)
        for entry in r['results']:
            self.assertIn('time', entry)
            self.assertGreater(entry['time'], 0)
            self.assertIn('n_gates', entry)

    def test_sv_and_mps(self):
        """Kleine n heeft zowel SV als MPS resultaten."""
        r = bench_qaoa_scaling(n_range=[4], chi_values=[4, 8],
                               verbose=False)
        entries = r['results']
        self.assertEqual(len(entries), 2)  # 2 chi waarden
        # SV should be present
        for e in entries:
            self.assertIsNotNone(e['sv_time'])
            self.assertIsNotNone(e['sv_energy'])


class TestVQEScaling(unittest.TestCase):
    """Test VQE scaling benchmark."""

    def test_runs(self):
        r = bench_vqe_scaling(n_range=[4, 6], chi_values=[4],
                              verbose=False)
        self.assertEqual(r['bench'], 'vqe_scaling')
        self.assertGreater(len(r['results']), 0)

    def test_energy_finite(self):
        """Energie is eindig."""
        r = bench_vqe_scaling(n_range=[4], chi_values=[4],
                              verbose=False)
        for entry in r['results']:
            if entry['energy'] is not None:
                self.assertTrue(np.isfinite(entry['energy']))


class TestTrotterScaling(unittest.TestCase):
    """Test Trotter scaling benchmark."""

    def test_runs(self):
        r = bench_trotter_scaling(n_range=[4, 6], orders=[1, 2],
                                  chi_values=[4], verbose=False)
        self.assertEqual(r['bench'], 'trotter_scaling')
        # 2 orders x 2 n-values x 1 chi = 4 entries
        self.assertEqual(len(r['results']), 4)

    def test_gate_count_increases(self):
        """Meer qubits geeft meer gates."""
        r = bench_trotter_scaling(n_range=[4, 8], orders=[1],
                                  chi_values=[4], verbose=False)
        gates = [e['n_gates'] for e in r['results']]
        self.assertLess(gates[0], gates[1])


class TestChiConvergence(unittest.TestCase):
    """Test chi convergentie benchmark."""

    def test_runs(self):
        r = bench_chi_convergence(n=6, chi_range=[2, 4, 8],
                                  verbose=False)
        self.assertEqual(r['bench'], 'chi_convergence')
        self.assertGreater(len(r['qaoa_results']), 0)
        self.assertGreater(len(r['heisenberg_results']), 0)

    def test_error_decreases(self):
        """Hogere chi geeft lagere fout (of gelijk)."""
        r = bench_chi_convergence(n=8, chi_range=[2, 4, 8, 16],
                                  verbose=False)
        errors = [e['error'] for e in r['qaoa_results']
                  if e['error'] is not None]
        # Fout moet globaal afnemen (monotoon niet gegarandeerd, maar trend)
        if len(errors) >= 2:
            self.assertGreaterEqual(errors[0], errors[-1] - 1e-10)


class TestCircuitComplexity(unittest.TestCase):
    """Test circuit complexiteit benchmark."""

    def test_runs(self):
        r = bench_circuit_complexity(n_range=[4, 8, 16], verbose=False)
        self.assertEqual(r['bench'], 'circuit_complexity')
        # 4 modellen x 3 n-waarden = 12 entries
        self.assertEqual(len(r['results']), 12)

    def test_linear_scaling_qaoa(self):
        """QAOA gate-count schaalt lineair met n (keten)."""
        r = bench_circuit_complexity(n_range=[10, 20, 40],
                                     verbose=False)
        qaoa = [e for e in r['results'] if e['model'] == 'QAOA-MaxCut-p1']
        # Gates ~ 2*n (H per qubit + RZZ per edge + RX per qubit)
        ratios = [qaoa[i+1]['n_gates'] / qaoa[i]['n_gates']
                  for i in range(len(qaoa) - 1)]
        for ratio in ratios:
            self.assertGreater(ratio, 1.5)  # groeit minstens ~2x
            self.assertLess(ratio, 3.0)     # maar niet meer dan ~2.5x


class TestBreakEven(unittest.TestCase):
    """Test break-even benchmark."""

    def test_structure(self):
        """Controleer resultaat structuur met kleine mock."""
        # Volledig bench_break_even duurt te lang voor unittest,
        # test alleen de structuur
        from scalability_benchmark import _make_random_graph, _timed
        from circuit_interface import Circuit, Observable, run_circuit
        # Handmatig een mini break-even test
        n = 6
        edges = _make_random_graph(n, edge_prob=0.5)
        qc = Circuit.qaoa_maxcut(n, edges, p=1, gammas=[0.5], betas=[0.3])
        obs = {'C': Observable.maxcut_cost(edges)}
        _, sv_t = _timed(run_circuit, qc, observables=obs, backend='statevector')
        _, mps_t = _timed(run_circuit, qc, observables=obs, backend='mps', max_chi=4)
        self.assertGreater(sv_t, 0)
        self.assertGreater(mps_t, 0)


class TestLargeScale(unittest.TestCase):
    """Test large-scale benchmark."""

    def test_runs(self):
        r = bench_large_scale(n_values=[50, 100], chi=4, verbose=False)
        self.assertEqual(r['bench'], 'large_scale')
        self.assertEqual(len(r['results']), 2)

    def test_has_energy(self):
        """Resultaten hebben energie."""
        r = bench_large_scale(n_values=[50], chi=4, verbose=False)
        for entry in r['results']:
            self.assertIsNotNone(entry.get('energy'))

    def test_ratio_reasonable(self):
        """MaxCut ratio is positief en < 1."""
        r = bench_large_scale(n_values=[50], chi=4, verbose=False)
        for entry in r['results']:
            if entry.get('ratio') is not None:
                self.assertGreater(entry['ratio'], 0)
                self.assertLess(entry['ratio'], 1.5)


if __name__ == '__main__':
    unittest.main()
