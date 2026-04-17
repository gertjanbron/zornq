"""Tests voor B10g CT-scan."""
import numpy as np
import unittest
import sys
sys.path.insert(0, '.')


class TestPauliBasis(unittest.TestCase):
    def test_basis_size(self):
        from b10g_ctscan import PAULI_3Q, PAULI_NAMES
        self.assertEqual(len(PAULI_3Q), 64)
        self.assertEqual(len(PAULI_NAMES), 64)
        self.assertEqual(PAULI_3Q.shape, (64, 8, 8))

    def test_orthogonality(self):
        from b10g_ctscan import PAULI_3Q
        for i in range(64):
            for j in range(64):
                tr = np.trace(PAULI_3Q[i].conj().T @ PAULI_3Q[j]) / 8
                if i == j:
                    self.assertAlmostEqual(abs(tr), 1.0, places=10)
                else:
                    self.assertAlmostEqual(abs(tr), 0.0, places=10)

    def test_roundtrip(self):
        from b10g_ctscan import pauli_decompose, pauli_reconstruct
        M = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
        coeffs = pauli_decompose(M)
        M_recon = pauli_reconstruct(coeffs)
        self.assertLess(np.linalg.norm(M - M_recon), 1e-10)

    def test_identity(self):
        from b10g_ctscan import pauli_decompose, PAULI_NAMES
        coeffs = pauli_decompose(np.eye(8, dtype=complex))
        idx_III = PAULI_NAMES.index('III')
        self.assertAlmostEqual(abs(coeffs[idx_III]), 1.0, places=10)
        for i, c in enumerate(coeffs):
            if i != idx_III:
                self.assertAlmostEqual(abs(c), 0.0, places=10)


class TestZornLBasis(unittest.TestCase):
    def test_L_matrix_shape(self):
        from b10g_ctscan import Zorn
        L = Zorn.basis(0).L_matrix()
        self.assertEqual(L.shape, (8, 8))

    def test_zorn_identity(self):
        """Zorn-identiteit is (1,0,0,0,0,0,0,1) = e_0 + e_7."""
        from b10g_ctscan import Zorn
        ident = Zorn(np.array([1,0,0,0,0,0,0,1], dtype=complex))
        L = ident.L_matrix()
        np.testing.assert_allclose(L, np.eye(8), atol=1e-12)

    def test_8_L_matrices_independent(self):
        from b10g_ctscan import Zorn
        mats = []
        for k in range(8):
            mats.append(Zorn.basis(k).L_matrix().ravel())
        A = np.array(mats).T
        rank = np.linalg.matrix_rank(A, tol=1e-10)
        self.assertEqual(rank, 8)


class TestPerspectiveCoverage(unittest.TestCase):
    def test_perspectives_cover_many_paulis(self):
        """Fano-perspectief met zmul_perspective dekt breed: 46-64 Paulis."""
        from b10g_ctscan import build_perspective_basis
        for d in range(7):
            _, coverage = build_perspective_basis(d)
            self.assertGreaterEqual(len(coverage), 40,
                "Perspectief %d: slechts %d Paulis" % (d, len(coverage)))

    def test_total_coverage_64(self):
        """Alle 7 perspectieven samen dekken alle 64 Paulis."""
        from b10g_ctscan import analyze_perspective_coverage
        cov = analyze_perspective_coverage()
        self.assertEqual(cov['total_covered'], 64)
        self.assertEqual(len(cov['uncovered']), 0)

    def test_overlap_matrix_symmetric(self):
        from b10g_ctscan import analyze_perspective_coverage
        cov = analyze_perspective_coverage()
        M = cov['overlap_matrix']
        self.assertEqual(M.shape, (7, 7))
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-12)


class TestPerspectiveProjection(unittest.TestCase):
    def test_perspective_L_element_low_residual(self):
        """L-matrix gebouwd via zmul_perspective(e_k, e_j, d) moet
        goed passen in de L-projectie van perspectief d."""
        from b10g_ctscan import build_perspective_basis, perspective_project
        _, coverage = build_perspective_basis(0)
        # Bouw L-matrix voor perspectief 0, basis element 1
        from b10g_ctscan import zmul_perspective
        M = np.zeros((8, 8), dtype=complex)
        e_k = np.zeros(8); e_k[1] = 1.0
        for j in range(8):
            e_j = np.zeros(8); e_j[j] = 1.0
            M[:, j] = zmul_perspective(e_k, e_j, 0)
        _, resid, _ = perspective_project(M, 0)
        self.assertLess(resid, 1e-10)

    def test_random_matrix_has_residual(self):
        from b10g_ctscan import perspective_project
        np.random.seed(42)
        M = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
        _, resid, _ = perspective_project(M, 0)
        self.assertGreater(resid, 0.1)


class TestMPOChi(unittest.TestCase):
    def test_diagonal_chi(self):
        from b10g_ctscan import mpo_effective_chi
        # Z-operator: diag pattern, rank depends on unique values
        Z8 = np.diag([1,-1,1,-1,1,-1,1,-1.0])
        W = Z8.reshape(1, 8, 8, 1)
        chi = mpo_effective_chi(W)
        self.assertGreater(chi, 0)

    def test_identity_chi_8(self):
        """Identity in (bra, ket) format is rank 8."""
        from b10g_ctscan import mpo_effective_chi
        W_id = np.eye(8).reshape(1, 8, 8, 1)
        self.assertEqual(mpo_effective_chi(W_id), 8)

    def test_rank_2_tensor(self):
        from b10g_ctscan import mpo_effective_chi
        W = np.zeros((2, 8, 8, 2), dtype=complex)
        W[0, :, :, 0] = np.eye(8)
        W[1, :, :, 1] = np.diag([1, -1, 1, -1, 1, -1, 1, -1])
        chi = mpo_effective_chi(W)
        self.assertGreaterEqual(chi, 2)


class TestCTScan1D(unittest.TestCase):
    """CT-scan op 1D keten (d=2, Zorn-perspectief = passthrough)."""

    def test_1d_p1_runs(self):
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=4, Ly=1, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=16, verbose=False)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIn('full_chi', r)
        self.assertEqual(len(r['perspectives']), 7)

    def test_1d_zz_finite(self):
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=4, Ly=1, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=16, verbose=False)
        r = results[0]
        self.assertTrue(np.isfinite(r['full_zz']))
        self.assertTrue(np.isfinite(r['zz_avg']))

    def test_1d_chi_bounded(self):
        """1D p=1 chi moet laag zijn."""
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=6, Ly=1, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=16, verbose=False)
        self.assertLessEqual(results[0]['full_chi'], 8)


class TestCTScan2D(unittest.TestCase):
    """CT-scan op 2D grid."""

    def test_2d_p1_runs(self):
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=4, Ly=2, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=32, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['full_chi'], 0)

    def test_2d_perspective_count(self):
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=4, Ly=2, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=32, verbose=False)
        self.assertEqual(len(results[0]['perspectives']), 7)


class TestCTScanD8(unittest.TestCase):
    def test_d8_p1_runs(self):
        from b10g_ctscan import ct_scan_chi_comparison
        results = ct_scan_chi_comparison(
            Lx=4, Ly=3, p_values=[1],
            gamma=0.3, beta=0.7, max_chi=32, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['full_chi'], 0)
        self.assertEqual(len(results[0]['perspectives']), 7)


class TestRecompression(unittest.TestCase):
    def test_recompress_reduces_chi(self):
        from b10g_ctscan import perspective_recompress
        import numpy as np
        np.random.seed(42)
        W = np.random.randn(4, 8, 8, 4) + 1j * np.random.randn(4, 8, 8, 4)
        W_comp, chi, S = perspective_recompress(W, d=0)
        self.assertLessEqual(chi, 32)
        self.assertGreater(chi, 0)
        self.assertEqual(W_comp.shape, W.shape)


if __name__ == '__main__':
    unittest.main()
