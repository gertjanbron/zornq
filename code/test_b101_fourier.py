#!/usr/bin/env python3
"""
Tests voor B101: Symbolische Fourier Cost Compiler

Test coverage:
  - FourierTerm / FourierExpansion basisklassen
  - QAOA-1 analytische formule vs state-vector
  - Gradient correctheid (finite difference)
  - Grid search en optimalisatie
  - Numerieke compilatie
  - compile_and_optimize pipeline
  - Edge cases
"""

import numpy as np
import unittest
import time

from fourier_cost_compiler import (
    FourierTerm, FourierExpansion, _QAOA1Expansion,
    compile_qaoa1_graph, compile_qaoa_numerical,
    compile_and_optimize, landscape_scan, parameter_sensitivity,
)


# ===================== Helpers =====================

def triangle_graph():
    """Driehoek met eenheidsgewichten."""
    return 3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]

def path_graph(n=4):
    """Pad P_n."""
    return n, [(i, i+1, 1.0) for i in range(n-1)]

def grid_4x4():
    """4x4 grid."""
    n = 16
    edges = []
    for r in range(4):
        for c in range(4):
            node = r * 4 + c
            if c + 1 < 4:
                edges.append((node, node + 1, 1.0))
            if r + 1 < 4:
                edges.append((node, node + 4, 1.0))
    return n, edges

def weighted_triangle():
    """Driehoek met niet-uniforme gewichten."""
    return 3, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 0.5)]

def state_vector_qaoa1_cost(n, edges, gamma, beta):
    """Brute-force state-vector QAOA-1 evaluatie als referentie."""
    dim = 2 ** n
    # Phase Hamiltoniaan (ZZ terms)
    H_phase = np.zeros(dim)
    for u, v, w in edges:
        for s in range(dim):
            zu = 1 - 2 * ((s >> u) & 1)
            zv = 1 - 2 * ((s >> v) & 1)
            H_phase[s] += w * zu * zv

    # MaxCut cost Hamiltoniaan
    H_cost = np.zeros(dim)
    for u, v, w in edges:
        for s in range(dim):
            zu = 1 - 2 * ((s >> u) & 1)
            zv = 1 - 2 * ((s >> v) & 1)
            H_cost[s] += w / 2.0 * (1 - zu * zv)

    # |+>
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Phase separating
    psi *= np.exp(-1j * gamma * H_phase)

    # Mixer (e^{-i*beta*X} op elke qubit)
    for q in range(n):
        psi_r = psi.reshape(2**(n-q-1), 2, 2**q)
        c = np.cos(beta)
        s_val = -1j * np.sin(beta)
        new0 = c * psi_r[:, 0, :] + s_val * psi_r[:, 1, :]
        new1 = s_val * psi_r[:, 0, :] + c * psi_r[:, 1, :]
        psi_r[:, 0, :] = new0
        psi_r[:, 1, :] = new1

    return np.real(np.dot(np.conj(psi), H_cost * psi))


# ===================== Tests =====================

class TestFourierTermBasics(unittest.TestCase):
    """Test FourierTerm en FourierExpansion basis."""

    def test_constant_term(self):
        """Constante term: coeff * cos(0) * cos(0) = coeff."""
        term = FourierTerm(coeff=3.14, gamma_modes=[(0, 'cos')],
                           beta_modes=[(0, 'cos')])
        exp = FourierExpansion(p=1, n_qubits=2, n_edges=1, terms=[term])
        val = exp.evaluate(np.array([0.5]), np.array([0.3]))
        self.assertAlmostEqual(val, 3.14, places=10)

    def test_sin_term(self):
        """Enkel sin-term."""
        term = FourierTerm(coeff=1.0, gamma_modes=[(2, 'sin')],
                           beta_modes=[(0, 'cos')])
        exp = FourierExpansion(p=1, n_qubits=2, n_edges=1, terms=[term])
        val = exp.evaluate(np.array([0.5]), np.array([0.3]))
        expected = np.sin(2 * 0.5)
        self.assertAlmostEqual(val, expected, places=10)

    def test_sum_terms(self):
        """Som van twee termen."""
        t1 = FourierTerm(coeff=1.0, gamma_modes=[(0, 'cos')],
                         beta_modes=[(0, 'cos')])
        t2 = FourierTerm(coeff=2.0, gamma_modes=[(1, 'sin')],
                         beta_modes=[(2, 'cos')])
        exp = FourierExpansion(p=1, n_qubits=2, n_edges=1, terms=[t1, t2])
        g, b = 0.7, 0.4
        val = exp.evaluate(np.array([g]), np.array([b]))
        expected = 1.0 + 2.0 * np.sin(g) * np.cos(2 * b)
        self.assertAlmostEqual(val, expected, places=10)

    def test_n_terms(self):
        exp = FourierExpansion(p=1, n_qubits=2, n_edges=1,
                               terms=[FourierTerm(1, [(0,'cos')], [(0,'cos')])] * 5)
        self.assertEqual(exp.n_terms, 5)


class TestQAOA1AnalyticalVsStateVector(unittest.TestCase):
    """Vergelijk analytische QAOA-1 formule met state-vector simulatie."""

    def _compare(self, n, edges, tol=1e-10):
        expansion = compile_qaoa1_graph(n, edges)
        gammas = np.linspace(0.1, 2.5, 8)
        betas = np.linspace(0.1, 1.2, 6)
        for g in gammas:
            for b in betas:
                fourier = expansion.evaluate(np.array([g]), np.array([b]))
                sv = state_vector_qaoa1_cost(n, edges, g, b)
                self.assertAlmostEqual(fourier, sv, places=8,
                    msg=f"Mismatch at gamma={g:.3f}, beta={b:.3f}: "
                        f"Fourier={fourier:.10f} vs SV={sv:.10f}")

    def test_triangle(self):
        self._compare(*triangle_graph())

    def test_path_4(self):
        self._compare(*path_graph(4))

    def test_path_6(self):
        self._compare(*path_graph(6))

    def test_weighted_triangle(self):
        self._compare(*weighted_triangle())

    def test_single_edge(self):
        self._compare(2, [(0, 1, 1.0)])

    def test_single_edge_weighted(self):
        self._compare(2, [(0, 1, 2.5)])

    def test_star_graph(self):
        """Stergraaf: centraal node verbonden met 4 bladeren."""
        edges = [(0, i, 1.0) for i in range(1, 5)]
        self._compare(5, edges)

    def test_complete_4(self):
        """K4."""
        edges = [(i, j, 1.0) for i in range(4) for j in range(i+1, 4)]
        self._compare(4, edges)

    @unittest.skip("SV op 16 qubits te traag voor CI")
    def test_grid_4x4(self):
        """4x4 grid — 16 qubits."""
        self._compare(*grid_4x4())

    def test_negative_weights(self):
        """Negatieve gewichten."""
        self._compare(3, [(0, 1, -1.0), (1, 2, 1.0), (0, 2, -0.5)])

    def test_mixed_weights(self):
        """Mix van grote en kleine gewichten."""
        self._compare(4, [(0,1,0.01), (1,2,100.0), (2,3,1.0), (0,3,-5.0)])


class TestGradient(unittest.TestCase):
    """Test gradient via finite differences."""

    def _check_gradient(self, n, edges, gamma, beta, delta=1e-6, tol=1e-4):
        expansion = compile_qaoa1_graph(n, edges)
        g = np.array([gamma])
        b = np.array([beta])
        dg, db = expansion.gradient(g, b)

        # Finite difference gamma
        c_plus = expansion.evaluate(np.array([gamma + delta]), b)
        c_minus = expansion.evaluate(np.array([gamma - delta]), b)
        fd_g = (c_plus - c_minus) / (2 * delta)

        # Finite difference beta
        c_plus = expansion.evaluate(g, np.array([beta + delta]))
        c_minus = expansion.evaluate(g, np.array([beta - delta]))
        fd_b = (c_plus - c_minus) / (2 * delta)

        self.assertAlmostEqual(dg[0], fd_g, places=3,
            msg=f"dC/dgamma: analytical={dg[0]:.8f} vs FD={fd_g:.8f}")
        self.assertAlmostEqual(db[0], fd_b, places=3,
            msg=f"dC/dbeta: analytical={db[0]:.8f} vs FD={fd_b:.8f}")

    def test_gradient_triangle(self):
        n, edges = triangle_graph()
        self._check_gradient(n, edges, 0.5, 0.3)

    def test_gradient_path(self):
        n, edges = path_graph(4)
        self._check_gradient(n, edges, 1.0, 0.7)

    def test_gradient_weighted(self):
        n, edges = weighted_triangle()
        self._check_gradient(n, edges, 0.8, 0.5)

    def test_gradient_at_origin(self):
        """Gradient bij gamma=0, beta=0 moet nul zijn (saddle point)."""
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        dg, db = expansion.gradient(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(dg[0], 0.0, places=10)
        self.assertAlmostEqual(db[0], 0.0, places=10)

    def test_gradient_multiple_points(self):
        """Gradient klopt op een raster van punten."""
        n, edges = triangle_graph()
        for gamma in [0.3, 0.8, 1.5, 2.5]:
            for beta in [0.2, 0.5, 1.0]:
                self._check_gradient(n, edges, gamma, beta)


class TestGridSearch(unittest.TestCase):
    """Test grid search functionaliteit."""

    def test_grid_returns_valid_cost(self):
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        g, b, val = expansion.grid_search(50, 50)
        self.assertGreater(val, 0)
        self.assertEqual(len(g), 1)
        self.assertEqual(len(b), 1)

    def test_grid_matches_evaluate(self):
        """Grid search best waarde is consistent met evaluate."""
        n, edges = path_graph(4)
        expansion = compile_qaoa1_graph(n, edges)
        g, b, val = expansion.grid_search(100, 100)
        val_check = expansion.evaluate(g, b)
        self.assertAlmostEqual(val, val_check, places=10)

    def test_grid_finds_good_ratio(self):
        """Grid search vindt minstens 50% van max cut."""
        n, edges = triangle_graph()
        total_w = sum(w for _, _, w in edges)
        expansion = compile_qaoa1_graph(n, edges)
        _, _, val = expansion.grid_search(100, 100)
        self.assertGreater(val / total_w, 0.5)


class TestOptimize(unittest.TestCase):
    """Test L-BFGS-B optimalisatie."""

    def test_optimize_triangle(self):
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        g, b, val = expansion.optimize(n_restarts=5)
        total_w = sum(w for _, _, w in edges)
        self.assertGreater(val / total_w, 0.5)

    def test_optimize_beats_random(self):
        """Optimized waarde moet beter zijn dan random punt."""
        n, edges = path_graph(6)
        expansion = compile_qaoa1_graph(n, edges)
        _, _, opt_val = expansion.optimize(n_restarts=3)
        random_val = expansion.evaluate(np.array([0.1]), np.array([0.1]))
        self.assertGreaterEqual(opt_val, random_val)

    def test_optimize_reproduceerbaarheid(self):
        """Zelfde seed geeft zelfde resultaat."""
        n, edges = path_graph(4)
        expansion = compile_qaoa1_graph(n, edges)
        _, _, val1 = expansion.optimize(n_restarts=3, seed=123)
        _, _, val2 = expansion.optimize(n_restarts=3, seed=123)
        self.assertAlmostEqual(val1, val2, places=10)


class TestNumericalCompilation(unittest.TestCase):
    """Test numerieke compilatie voor p≥1."""

    def test_numerical_p1_triangle(self):
        """Numerieke compilatie p=1 geeft vergelijkbaar resultaat als analytisch."""
        n, edges = triangle_graph()
        analytical = compile_qaoa1_graph(n, edges)
        numerical = compile_qaoa_numerical(n, edges, p=1, n_samples=100)

        g, b = 0.5, 0.3
        val_a = analytical.evaluate(np.array([g]), np.array([b]))
        val_n = numerical.evaluate(np.array([g]), np.array([b]))
        # Interpolatie is niet exact, maar moet close zijn
        self.assertAlmostEqual(val_a, val_n, places=1)

    def test_numerical_grid_search(self):
        """Grid search op numerieke expansie."""
        n, edges = triangle_graph()
        expansion = compile_qaoa_numerical(n, edges, p=1, n_samples=50)
        g, b, val = expansion.grid_search()
        total_w = sum(w for _, _, w in edges)
        self.assertGreater(val / total_w, 0.3)

    def test_numerical_compile_time(self):
        """Compilatie duurt niet te lang."""
        n, edges = triangle_graph()
        t0 = time.time()
        compile_qaoa_numerical(n, edges, p=1, n_samples=30)
        dt = time.time() - t0
        self.assertLess(dt, 10.0)  # ruim budget


class TestCompileAndOptimize(unittest.TestCase):
    """Test de volledige pipeline."""

    def test_pipeline_p1(self):
        n, edges = triangle_graph()
        result = compile_and_optimize(n, edges, p=1)
        self.assertIn('gammas', result)
        self.assertIn('betas', result)
        self.assertIn('cost', result)
        self.assertIn('ratio', result)
        self.assertGreater(result['cost'], 0)
        self.assertGreater(result['ratio'], 0)

    def test_pipeline_returns_method(self):
        n, edges = triangle_graph()
        result = compile_and_optimize(n, edges, p=1)
        self.assertEqual(result['method'], 'analytical_fourier')

    def test_pipeline_timing(self):
        n, edges = path_graph(6)
        result = compile_and_optimize(n, edges, p=1)
        self.assertGreater(result['compile_time'], 0)
        self.assertGreater(result['total_time'], 0)

    def test_pipeline_grid_ratio(self):
        """Pipeline vindt redelijke ratio."""
        n, edges = path_graph(6)  # kleiner dan grid_4x4 voor snelheid
        result = compile_and_optimize(n, edges, p=1, n_restarts=2)
        self.assertGreater(result['ratio'], 0.4)


class TestAnalysisFunctions(unittest.TestCase):
    """Test landscape_scan en parameter_sensitivity."""

    def test_landscape_shape(self):
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        L = landscape_scan(expansion, n_gamma=20, n_beta=15)
        self.assertEqual(L.shape, (20, 15))

    def test_landscape_nonnegative(self):
        """Cost is niet-negatief voor positieve gewichten."""
        n, edges = path_graph(4)
        expansion = compile_qaoa1_graph(n, edges)
        L = landscape_scan(expansion, n_gamma=15, n_beta=15)
        self.assertTrue(np.all(L >= -1e-10))

    def test_sensitivity_keys(self):
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        sens = parameter_sensitivity(expansion, np.array([0.5]), np.array([0.3]))
        self.assertIn('gamma', sens)
        self.assertIn('beta', sens)
        self.assertEqual(len(sens['gamma']), 1)
        self.assertEqual(len(sens['beta']), 1)


class TestEdgeCases(unittest.TestCase):
    """Edge cases en randgevallen."""

    def test_single_edge_optimal(self):
        """Enkel edge: optimale ratio moet ~0.75 zijn (QAOA-1 op 1 edge)."""
        n, edges = 2, [(0, 1, 1.0)]
        result = compile_and_optimize(n, edges, p=1, n_restarts=5)
        # QAOA-1 op single edge: sin(4β)sin(2γ)/2 ≤ 0.5, dus ratio ≤ 0.75
        self.assertGreater(result['ratio'], 0.7)

    def test_zero_gamma_zero_beta(self):
        """Bij gamma=0, beta=0: cost = sum(w/2)."""
        n, edges = triangle_graph()
        expansion = compile_qaoa1_graph(n, edges)
        val = expansion.evaluate(np.array([0.0]), np.array([0.0]))
        expected = sum(w / 2.0 for _, _, w in edges)
        self.assertAlmostEqual(val, expected, places=10)

    def test_compile_time_recorded(self):
        n, edges = grid_4x4()
        expansion = compile_qaoa1_graph(n, edges)
        self.assertGreaterEqual(expansion.compile_time_s, 0)

    def test_expansion_properties(self):
        n, edges = path_graph(5)
        expansion = compile_qaoa1_graph(n, edges)
        self.assertEqual(expansion.p, 1)
        self.assertEqual(expansion.n_qubits, 5)
        self.assertEqual(expansion.n_edges, 4)
        self.assertGreater(expansion.n_terms, 0)


class TestSpeedup(unittest.TestCase):
    """Test dat Fourier compiler sneller is dan state-vector."""

    @unittest.skip("SV benchmark te traag voor CI")
    def test_fourier_faster_than_sv(self):
        """Fourier evaluatie moet sneller zijn dan SV op een 8-node graph."""
        n, edges = path_graph(8)  # kleiner dan grid_4x4 om timeout te voorkomen
        expansion = compile_qaoa1_graph(n, edges)

        n_evals = 50
        gammas = np.linspace(0.1, 2.5, n_evals)
        betas = np.linspace(0.1, 1.2, n_evals)

        # Fourier timing
        t0 = time.time()
        for g, b in zip(gammas, betas):
            expansion.evaluate(np.array([g]), np.array([b]))
        fourier_time = time.time() - t0

        # SV timing
        t0 = time.time()
        for g, b in zip(gammas, betas):
            state_vector_qaoa1_cost(n, edges, g, b)
        sv_time = time.time() - t0

        # Fourier moet sneller zijn
        speedup = sv_time / fourier_time if fourier_time > 0 else float('inf')
        self.assertGreater(speedup, 2.0,
            f"Speedup {speedup:.1f}x te laag (Fourier={fourier_time:.4f}s, "
            f"SV={sv_time:.4f}s)")


if __name__ == '__main__':
    unittest.main()
