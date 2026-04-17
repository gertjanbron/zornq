#!/usr/bin/env python3
"""
test_quality_certificate.py - Tests voor B131 Kwaliteitsgarantie
"""

import unittest
import numpy as np

from quality_certificate import (
    CertificateLevel, QualityCertificate,
    TrotterErrorEstimator, FidelityEstimator, ObservableVerifier,
    MaxCutVerifier, certify_circuit_result, certify_maxcut,
    certify_energy, certify_observable, certify_chi_convergence,
    certify_batch, _paulis_commute,
    certify_maxcut_from_fw, certify_maxcut_from_ilp,
)
from circuit_interface import Circuit, Observable, run_circuit, _run_statevector
from hamiltonian_compiler import Hamiltonian


class TestCertificateLevel(unittest.TestCase):

    def test_ordering(self):
        self.assertTrue(CertificateLevel.UNKNOWN < CertificateLevel.APPROXIMATE)
        self.assertTrue(CertificateLevel.APPROXIMATE < CertificateLevel.BOUNDED)
        self.assertTrue(CertificateLevel.BOUNDED < CertificateLevel.NEAR_EXACT)
        self.assertTrue(CertificateLevel.NEAR_EXACT < CertificateLevel.EXACT)


class TestQualityCertificate(unittest.TestCase):

    def test_defaults(self):
        cert = QualityCertificate()
        self.assertEqual(cert.level, CertificateLevel.UNKNOWN)
        self.assertFalse(cert.is_exact)
        self.assertFalse(cert.is_reliable)

    def test_exact(self):
        cert = QualityCertificate(level=CertificateLevel.EXACT)
        self.assertTrue(cert.is_exact)
        self.assertTrue(cert.is_reliable)

    def test_summary(self):
        cert = QualityCertificate(
            level=CertificateLevel.BOUNDED,
            gap=2.5,
            fidelity=0.998,
            wall_time=0.05
        )
        s = cert.summary()
        self.assertIn("BOUNDED", s)
        self.assertIn("2.50%", s)
        self.assertIn("0.998", s)

    def test_to_dict(self):
        cert = QualityCertificate(level=CertificateLevel.EXACT, value=42.0)
        d = cert.to_dict()
        self.assertEqual(d['level'], 'EXACT')
        self.assertEqual(d['value'], 42.0)

    def test_repr(self):
        cert = QualityCertificate(level=CertificateLevel.NEAR_EXACT, gap=0.5)
        self.assertIn("NEAR_EXACT", repr(cert))


class TestTrotterErrorEstimator(unittest.TestCase):

    def test_commuting_hamiltonian_zero_error(self):
        """Commuterende termen hebben nul Trotter-fout."""
        H = Hamiltonian([(1.0, {0: 'Z'}), (1.0, {1: 'Z'})], 2)
        err = TrotterErrorEstimator.estimate(H, t=1.0, steps=5, order=1)
        self.assertAlmostEqual(err, 0.0, places=10)

    def test_noncommuting_positive_error(self):
        """Niet-commuterende termen geven positieve fout."""
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        err = TrotterErrorEstimator.estimate(H, t=1.0, steps=5, order=1)
        self.assertGreater(err, 0)

    def test_more_steps_less_error(self):
        H = Hamiltonian.ising_transverse(4)
        e5 = TrotterErrorEstimator.estimate(H, t=1.0, steps=5, order=1)
        e20 = TrotterErrorEstimator.estimate(H, t=1.0, steps=20, order=1)
        self.assertLess(e20, e5)

    def test_higher_order_less_error(self):
        H = Hamiltonian.ising_transverse(4)
        e1 = TrotterErrorEstimator.estimate(H, t=0.5, steps=10, order=1)
        e2 = TrotterErrorEstimator.estimate(H, t=0.5, steps=10, order=2)
        self.assertLess(e2, e1)

    def test_order4(self):
        H = Hamiltonian.ising_transverse(4)
        e4 = TrotterErrorEstimator.estimate(H, t=0.5, steps=10, order=4)
        self.assertIsNotNone(e4)
        self.assertGreater(e4, 0)


class TestFidelityEstimator(unittest.TestCase):

    def test_chi_series(self):
        vals = {8: 1.0, 16: 1.05, 32: 1.08, 64: 1.09}
        extrap, fid = FidelityEstimator.from_chi_series(vals)
        self.assertIsNotNone(extrap)
        self.assertIsNotNone(fid)

    def test_chi_series_converged(self):
        vals = {8: 5.0, 16: 5.0, 32: 5.0}
        extrap, fid = FidelityEstimator.from_chi_series(vals)
        self.assertAlmostEqual(extrap, 5.0, places=5)

    def test_truncation_error_zero(self):
        fid = FidelityEstimator.from_truncation_error(0.0)
        self.assertAlmostEqual(fid, 1.0)

    def test_truncation_error_small(self):
        fid = FidelityEstimator.from_truncation_error(0.01)
        self.assertGreaterEqual(fid, 0.99)

    def test_state_comparison_identical(self):
        state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        fid = FidelityEstimator.from_state_comparison(state, state)
        self.assertAlmostEqual(fid, 1.0, places=10)

    def test_state_comparison_orthogonal(self):
        s1 = np.array([1, 0, 0, 0], dtype=complex)
        s2 = np.array([0, 1, 0, 0], dtype=complex)
        fid = FidelityEstimator.from_state_comparison(s1, s2)
        self.assertAlmostEqual(fid, 0.0, places=10)


class TestMaxCutVerifier(unittest.TestCase):

    def test_verify_assignment(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        cut, valid = MaxCutVerifier.verify_assignment(3, edges, [0, 1, 0])
        self.assertTrue(valid)
        self.assertAlmostEqual(cut, 2.0)

    def test_verify_assignment_invalid(self):
        cut, valid = MaxCutVerifier.verify_assignment(3, [(0,1,1.0)], [0])
        self.assertFalse(valid)

    def test_trivial_bounds(self):
        edges = [(0, 1, 1.0), (1, 2, 2.0)]
        lower, upper = MaxCutVerifier.trivial_bounds(3, edges)
        self.assertAlmostEqual(upper, 3.0)
        self.assertAlmostEqual(lower, 1.5)

    def test_bipartite_check_path(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        is_bip, exact = MaxCutVerifier.bipartite_check(3, edges)
        self.assertTrue(is_bip)
        self.assertAlmostEqual(exact, 2.0)

    def test_bipartite_check_triangle(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        is_bip, _ = MaxCutVerifier.bipartite_check(3, edges)
        self.assertFalse(is_bip)

    def test_brute_force_triangle(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        cut, assign = MaxCutVerifier.brute_force(3, edges)
        self.assertAlmostEqual(cut, 2.0)
        self.assertIsNotNone(assign)

    def test_brute_force_too_large(self):
        cut, _ = MaxCutVerifier.brute_force(25, [], max_n=20)
        self.assertIsNone(cut)

    def test_brute_force_weighted(self):
        edges = [(0, 1, 3.0), (1, 2, 1.0)]
        cut, _ = MaxCutVerifier.brute_force(3, edges)
        self.assertAlmostEqual(cut, 4.0)


class TestObservableVerifier(unittest.TestCase):

    def test_check_bounds(self):
        obs = Observable([(1.0, {0: 'Z'})])
        ok, bound = ObservableVerifier.check_bounds(0.5, obs, 1)
        self.assertTrue(ok)
        self.assertAlmostEqual(bound, 1.0)

    def test_check_bounds_violation(self):
        obs = Observable([(1.0, {0: 'Z'})])
        ok, _ = ObservableVerifier.check_bounds(1.5, obs, 1)
        self.assertFalse(ok)


class TestCertifyCircuitResult(unittest.TestCase):

    def test_statevector_bell(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        result = run_circuit(qc, backend='statevector')
        cert = certify_circuit_result(result)
        self.assertEqual(cert.level, CertificateLevel.EXACT)
        self.assertAlmostEqual(cert.fidelity, 1.0)

    def test_statevector_with_reference(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        result = run_circuit(qc, backend='statevector')

        ref = result['state'].copy()
        cert = certify_circuit_result(result, reference_state=ref)
        self.assertAlmostEqual(cert.fidelity, 1.0)

    def test_trotter_with_hamiltonian(self):
        H = Hamiltonian.ising_transverse(4)
        qc = H.trotter(t=0.5, steps=5, order=2)
        result = run_circuit(qc, backend='statevector')
        cert = certify_circuit_result(result, hamiltonian=H, circuit=qc)
        self.assertIsNotNone(cert.trotter_error_bound)
        self.assertGreater(cert.trotter_error_bound, 0)


class TestCertifyMaxCut(unittest.TestCase):

    def test_exact_bipartite(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        cert = certify_maxcut(2.0, 3, edges, [0, 1, 0])
        self.assertEqual(cert.level, CertificateLevel.EXACT)
        self.assertEqual(cert.verification, "bipartite exact")

    def test_exact_brute_force(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        cert = certify_maxcut(2.0, 3, edges, [0, 1, 0])
        self.assertEqual(cert.level, CertificateLevel.EXACT)

    def test_suboptimal(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        cert = certify_maxcut(1.0, 3, edges)  # suboptimal
        self.assertNotEqual(cert.level, CertificateLevel.EXACT)
        self.assertIsNotNone(cert.gap)
        self.assertGreater(cert.gap, 0)

    def test_with_gw_bound(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        cert = certify_maxcut(1.8, 3, edges, gw_bound=2.0)
        self.assertLessEqual(cert.upper_bound, 2.0)

    def test_assignment_verification(self):
        edges = [(0, 1, 1.0)]
        cert = certify_maxcut(1.0, 2, edges, [0, 1])
        self.assertIn("assignment verified", cert.checks[0])

    def test_large_graph_no_brute_force(self):
        n = 25
        edges = [(i, (i+1)%n, 1.0) for i in range(n)]
        cert = certify_maxcut(20.0, n, edges)
        # Moet niet crashen op grote graaf
        self.assertIsNotNone(cert.level)


class TestCertifyEnergy(unittest.TestCase):

    def test_with_exact_gs(self):
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        cert = certify_energy(-1.0, H, exact_gs=-1.0)
        self.assertEqual(cert.level, CertificateLevel.EXACT)

    def test_approximate_energy(self):
        H = Hamiltonian.ising_transverse(4)
        cert = certify_energy(-2.0, H)
        self.assertEqual(cert.level, CertificateLevel.APPROXIMATE)

    def test_near_exact(self):
        H = Hamiltonian([(1.0, {0: 'Z'})], 1)
        cert = certify_energy(-0.995, H, exact_gs=-1.0)
        self.assertEqual(cert.level, CertificateLevel.NEAR_EXACT)

    def test_operator_norm_bound(self):
        H = Hamiltonian([(1.0, {0: 'Z'}), (0.5, {0: 'X'})], 1)
        cert = certify_energy(0.5, H)
        self.assertIsNotNone(cert.upper_bound)
        self.assertIsNotNone(cert.lower_bound)


class TestCertifyObservable(unittest.TestCase):

    def test_within_bounds(self):
        obs = Observable([(1.0, {0: 'Z'})])
        cert = certify_observable(0.5, obs)
        self.assertIn("within operator norm", cert.checks[0])

    def test_with_state(self):
        obs = Observable([(1.0, {0: 'Z'})])
        state = np.array([1, 0], dtype=complex)  # |0>
        cert = certify_observable(1.0, obs, state=state, n_qubits=1)
        self.assertEqual(cert.level, CertificateLevel.BOUNDED)
        self.assertGreaterEqual(len(cert.checks), 2)


class TestChiConvergence(unittest.TestCase):

    def test_converged(self):
        vals = {8: -3.0, 16: -3.1, 32: -3.110, 64: -3.1105}
        cert = certify_chi_convergence(vals)
        self.assertIn(cert.level, (CertificateLevel.NEAR_EXACT, CertificateLevel.BOUNDED))
        self.assertIsNotNone(cert.chi_extrapolated)

    def test_not_converged(self):
        vals = {8: -1.0, 16: -3.0}
        cert = certify_chi_convergence(vals)
        self.assertEqual(cert.level, CertificateLevel.APPROXIMATE)

    def test_single_chi(self):
        cert = certify_chi_convergence({32: -5.0})
        self.assertEqual(cert.level, CertificateLevel.APPROXIMATE)


class TestBatchCertify(unittest.TestCase):

    def test_maxcut_batch(self):
        results = [
            {'type': 'maxcut', 'cut': 2.0, 'n': 3,
             'edges': [(0,1,1.0),(1,2,1.0)], 'assignment': [0,1,0]},
            {'type': 'maxcut', 'cut': 1.0, 'n': 3,
             'edges': [(0,1,1.0),(1,2,1.0),(0,2,1.0)]},
        ]
        certs, summary = certify_batch(results)
        self.assertEqual(len(certs), 2)
        self.assertEqual(summary['total'], 2)
        self.assertGreater(summary['exact'], 0)


class TestPaulisCommute(unittest.TestCase):

    def test_same_pauli_commutes(self):
        self.assertTrue(_paulis_commute({0: 'Z'}, {0: 'Z'}))

    def test_different_pauli_anticommutes(self):
        self.assertFalse(_paulis_commute({0: 'X'}, {0: 'Z'}))

    def test_different_qubits_commute(self):
        self.assertTrue(_paulis_commute({0: 'X'}, {1: 'Z'}))

    def test_even_anticommutations(self):
        # XY and YX on qubits 0,1: anticommute on both -> commute
        self.assertTrue(_paulis_commute({0: 'X', 1: 'Y'}, {0: 'Y', 1: 'X'}))


class TestEndToEnd(unittest.TestCase):
    """End-to-end: Hamiltonian -> Circuit -> Run -> Certify."""

    def test_ising_trotter_pipeline(self):
        H = Hamiltonian.ising_transverse(4, J=1.0, h=0.5)
        qc = H.trotter(t=0.5, steps=10, order=2)
        result = run_circuit(qc, observables={'E': H.to_observable()},
                            backend='statevector')
        cert = certify_circuit_result(result, hamiltonian=H, circuit=qc)
        self.assertIsNotNone(cert.trotter_error_bound)
        self.assertTrue(cert.is_reliable or cert.level == CertificateLevel.APPROXIMATE)

    def test_maxcut_qaoa_pipeline(self):
        edges = [(0,1,1.0),(1,2,1.0),(0,2,1.0)]
        H = Hamiltonian.maxcut(3, edges)
        qc = H.qaoa(p=2, gammas=[0.7,0.5], betas=[0.3,0.2])
        result = run_circuit(qc, observables={'cost': H.to_observable()},
                            backend='statevector')
        cost = result['observables']['cost']
        cert = certify_maxcut(cost, 3, edges)
        self.assertIsNotNone(cert.level)
        self.assertIsNotNone(cert.gap)


class _MockFWResult:
    """Lichtgewicht mock voor B176 FWResult (attribute-access)."""

    def __init__(self, sdp_upper_bound, feasible_cut_lb,
                 iterations=50, final_gap=1e-4,
                 diag_err_max=1e-10, penalty=1.0, converged=True):
        self.sdp_upper_bound = sdp_upper_bound
        self.feasible_cut_lb = feasible_cut_lb
        self.iterations = iterations
        self.final_gap = final_gap
        self.diag_err_max = diag_err_max
        self.penalty = penalty
        self.converged = converged


class TestCertifyMaxCutFromFW(unittest.TestCase):
    """B176 Frank-Wolfe SDP-sandwich -> LEVEL 2 certificaat."""

    def test_tight_sandwich_exact(self):
        """Gap ~ 0 => EXACT."""
        fw = _MockFWResult(sdp_upper_bound=10.0,
                           feasible_cut_lb=10.0 - 1e-7,
                           final_gap=1e-9)
        cert = certify_maxcut_from_fw(fw)
        self.assertEqual(cert.level, CertificateLevel.EXACT)
        self.assertAlmostEqual(cert.upper_bound, 10.0)
        self.assertAlmostEqual(cert.lower_bound, 10.0 - 1e-7, places=6)
        self.assertEqual(cert.method, "b176_frank_wolfe_sdp")
        self.assertEqual(cert.verification, "fw_duality_sandwich")

    def test_near_exact_gap_under_one_percent(self):
        fw = _MockFWResult(sdp_upper_bound=100.0,
                           feasible_cut_lb=99.5,
                           final_gap=1e-4)
        cert = certify_maxcut_from_fw(fw)
        self.assertEqual(cert.level, CertificateLevel.NEAR_EXACT)
        self.assertLess(cert.gap, 1.0)

    def test_bounded_gap(self):
        """Gap ~5%: BOUNDED."""
        fw = _MockFWResult(sdp_upper_bound=100.0,
                           feasible_cut_lb=95.0,
                           final_gap=5e-2)
        cert = certify_maxcut_from_fw(fw)
        self.assertEqual(cert.level, CertificateLevel.BOUNDED)
        self.assertAlmostEqual(cert.gap, 5.0, places=4)

    def test_approximate_wide_gap(self):
        """Gap ~50%: APPROXIMATE."""
        fw = _MockFWResult(sdp_upper_bound=100.0,
                           feasible_cut_lb=50.0,
                           final_gap=0.5)
        cert = certify_maxcut_from_fw(fw)
        self.assertEqual(cert.level, CertificateLevel.APPROXIMATE)
        self.assertAlmostEqual(cert.gap, 50.0, places=4)

    def test_cut_value_strengthens_lb(self):
        """Incumbent > feasible_cut_lb pusht LB omhoog."""
        fw = _MockFWResult(sdp_upper_bound=10.0, feasible_cut_lb=8.0)
        cert = certify_maxcut_from_fw(fw, cut_value=9.0)
        self.assertAlmostEqual(cert.lower_bound, 9.0)
        self.assertAlmostEqual(cert.value, 9.0)

    def test_cut_value_below_feasible_lb_ignored(self):
        """Incumbent < feasible_cut_lb laat LB op feasible_cut_lb staan."""
        fw = _MockFWResult(sdp_upper_bound=10.0, feasible_cut_lb=8.0)
        cert = certify_maxcut_from_fw(fw, cut_value=5.0)
        self.assertAlmostEqual(cert.lower_bound, 8.0)

    def test_not_converged_warning(self):
        fw = _MockFWResult(sdp_upper_bound=10.0, feasible_cut_lb=9.0,
                           converged=False)
        cert = certify_maxcut_from_fw(fw)
        self.assertTrue(any("converge" in w.lower() for w in cert.warnings))

    def test_assignment_verification(self):
        """Optionele assignment-verificatie wordt meegenomen in checks."""
        fw = _MockFWResult(sdp_upper_bound=3.0, feasible_cut_lb=3.0)
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        assignment = [0, 1, 0]
        cert = certify_maxcut_from_fw(fw, n=3, edges=edges,
                                      cut_value=2.0,
                                      assignment=assignment)
        self.assertTrue(any("assignment verified" in c for c in cert.checks))

    def test_bounds_sanity(self):
        """lower_bound <= upper_bound altijd."""
        for ub, lb in [(10.0, 9.0), (100.0, 50.0), (5.0, 5.0)]:
            fw = _MockFWResult(sdp_upper_bound=ub, feasible_cut_lb=lb)
            cert = certify_maxcut_from_fw(fw)
            self.assertLessEqual(cert.lower_bound, cert.upper_bound + 1e-9)

    def test_checks_include_fw_diagnostics(self):
        fw = _MockFWResult(sdp_upper_bound=10.0, feasible_cut_lb=9.0,
                           iterations=123, final_gap=1e-5,
                           diag_err_max=1e-8, penalty=2.5)
        cert = certify_maxcut_from_fw(fw)
        text = " ".join(cert.checks)
        self.assertIn("FW iterations: 123", text)
        self.assertIn("FW duality gap", text)
        self.assertIn("diag error max", text)
        self.assertIn("penalty", text)
        self.assertIn("sandwich", text)


class TestCertifyMaxCutFromILP(unittest.TestCase):
    """B159 ILP-oracle -> LEVEL 1 certificaat (certified EXACT)."""

    def test_certified_optimal_exact(self):
        ilp = {
            'certified': True,
            'opt_value': 42.0,
            'gap_abs': 0.0,
            'solver': 'HiGHS',
            'wall_time': 0.12,
            'status': 'Optimal',
        }
        cert = certify_maxcut_from_ilp(ilp)
        self.assertEqual(cert.level, CertificateLevel.EXACT)
        self.assertEqual(cert.method, "b159_ilp_oracle")
        self.assertEqual(cert.verification, "ilp_certified_optimal")
        self.assertAlmostEqual(cert.upper_bound, 42.0)
        self.assertAlmostEqual(cert.lower_bound, 42.0)

    def test_certified_matching_incumbent_exact(self):
        ilp = {'certified': True, 'opt_value': 42.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.05, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp, cut_value=42.0)
        self.assertEqual(cert.level, CertificateLevel.EXACT)
        self.assertAlmostEqual(cert.lower_bound, 42.0)
        self.assertAlmostEqual(cert.upper_bound, 42.0)

    def test_certified_mismatching_incumbent_bounded(self):
        """Certified ILP + lagere user-cut => BOUNDED."""
        ilp = {'certified': True, 'opt_value': 100.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.01, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp, cut_value=90.0)
        self.assertEqual(cert.level, CertificateLevel.BOUNDED)
        self.assertAlmostEqual(cert.upper_bound, 100.0)
        self.assertAlmostEqual(cert.lower_bound, 90.0)
        self.assertAlmostEqual(cert.gap, 10.0, places=4)

    def test_certified_near_exact_small_gap(self):
        """Certified ILP + user-cut met gap <1% => NEAR_EXACT."""
        ilp = {'certified': True, 'opt_value': 100.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.01, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp, cut_value=99.5)
        self.assertEqual(cert.level, CertificateLevel.NEAR_EXACT)
        self.assertLess(cert.gap, 1.0)

    def test_not_certified_bounded(self):
        """certified=False + gap => BOUNDED."""
        ilp = {'certified': False, 'opt_value': 100.0, 'gap_abs': 5.0,
               'solver': 'HiGHS', 'wall_time': 60.0,
               'status': 'TimeLimit'}
        cert = certify_maxcut_from_ilp(ilp, cut_value=95.0)
        self.assertEqual(cert.verification, "ilp_incumbent_only")
        self.assertIn(cert.level,
                      (CertificateLevel.BOUNDED,
                       CertificateLevel.APPROXIMATE,
                       CertificateLevel.NEAR_EXACT))
        self.assertAlmostEqual(cert.upper_bound, 100.0)
        self.assertAlmostEqual(cert.lower_bound, 95.0)

    def test_bounds_consistency(self):
        """upper_bound = opt_value bij certified run."""
        ilp = {'certified': True, 'opt_value': 77.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.02, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp)
        self.assertAlmostEqual(cert.upper_bound, 77.0)

    def test_user_cut_exceeds_opt_warns(self):
        """User cut > ILP opt zou onmogelijk moeten zijn => warning."""
        ilp = {'certified': True, 'opt_value': 10.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.01, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp, cut_value=15.0)
        self.assertTrue(any("user cut" in w.lower() for w in cert.warnings))

    def test_checks_include_solver_metadata(self):
        ilp = {'certified': True, 'opt_value': 10.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.456, 'status': 'Optimal'}
        cert = certify_maxcut_from_ilp(ilp)
        text = " ".join(cert.checks)
        self.assertIn("HiGHS", text)
        self.assertIn("Optimal", text)
        self.assertIn("0.456", text)

    def test_assignment_verification(self):
        ilp = {'certified': True, 'opt_value': 2.0, 'gap_abs': 0.0,
               'solver': 'HiGHS', 'wall_time': 0.01, 'status': 'Optimal'}
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        cert = certify_maxcut_from_ilp(ilp, n=3, edges=edges,
                                       cut_value=2.0,
                                       assignment=[0, 1, 0])
        self.assertTrue(any("assignment verified" in c for c in cert.checks))

    def test_missing_opt_value_graceful(self):
        """Ontbrekende opt_value => geen crash, APPROXIMATE/BOUNDED."""
        ilp = {'certified': False, 'opt_value': None, 'gap_abs': None,
               'solver': 'HiGHS', 'wall_time': 0.0, 'status': 'Infeasible'}
        cert = certify_maxcut_from_ilp(ilp)
        self.assertIn(cert.level,
                      (CertificateLevel.APPROXIMATE,
                       CertificateLevel.BOUNDED,
                       CertificateLevel.UNKNOWN))


if __name__ == '__main__':
    unittest.main()
