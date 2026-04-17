#!/usr/bin/env python3
"""
Tests voor B80: MPQS (Message-Passing Quantum Solver).

Test suites:
  1. TestLightconeBuild        : BFS-lightcone correctheid
  2. TestQAOAStatevector       : pure-numpy QAOA basis-eigenschappen
  3. TestExpectationZZ         : ⟨Z_u Z_v⟩ correctheid
  4. TestGreedy1Flip           : lokale zoekmethode-eigenschappen
  5. TestBPOnTrees             : BP is exact op bomen
  6. TestBPOnSmall             : BP + refine geeft OPT op kleine grafen
  7. TestLightconeCorrectness  : lightcone-QAOA reproduces cut op kleine grafen
  8. TestEdgeCases             : lege graaf, enkele edge, disconnected
  9. TestDeterminism           : zelfde seed → zelfde output
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from b60_gw_bound import SimpleGraph, brute_force_maxcut, random_3regular
from b156_sos2_sdp import (
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    complete_bipartite,
)
from b80_mpqs import (
    mpqs_classical_bp,
    mpqs_lightcone,
    _build_lightcone,
    _qaoa_statevector,
    _expectation_zz,
    _greedy_1flip,
)


TOL = 1e-6


# ============================================================
# 1. Lightcone-constructie
# ============================================================

class TestLightconeBuild(unittest.TestCase):
    def test_radius_0_is_center_only(self):
        g = cycle_graph(5)
        verts, edges, mapping = _build_lightcone(g, center=2, radius=0)
        self.assertEqual(verts, [2])
        self.assertEqual(edges, [])
        self.assertEqual(mapping, {2: 0})

    def test_radius_1_on_cycle(self):
        g = cycle_graph(5)
        verts, edges, mapping = _build_lightcone(g, center=0, radius=1)
        # C_5: buren van 0 zijn 1 en 4
        self.assertEqual(set(verts), {0, 1, 4})
        # Edges binnen {0,1,4}: (0,1), (0,4)
        self.assertEqual(len(edges), 2)

    def test_radius_large_covers_connected_component(self):
        g = path_graph(6)
        verts, _edges, _ = _build_lightcone(g, center=0, radius=10)
        self.assertEqual(set(verts), {0, 1, 2, 3, 4, 5})

    def test_path_radius_2(self):
        g = path_graph(6)
        # vertex 2, radius 2 → {0,1,2,3,4}
        verts, edges, _ = _build_lightcone(g, center=2, radius=2)
        self.assertEqual(set(verts), {0, 1, 2, 3, 4})
        self.assertEqual(len(edges), 4)

    def test_disconnected_stays_disconnected(self):
        g = SimpleGraph(6)
        g.add_edge(0, 1); g.add_edge(1, 2)
        g.add_edge(3, 4); g.add_edge(4, 5)
        verts, _, _ = _build_lightcone(g, center=0, radius=10)
        self.assertEqual(set(verts), {0, 1, 2})

    def test_mapping_is_bijective(self):
        g = petersen_graph()
        verts, edges, mapping = _build_lightcone(g, center=0, radius=2)
        self.assertEqual(len(mapping), len(verts))
        # alle lokale id's in range [0, |L|)
        for g_id, l_id in mapping.items():
            self.assertIn(g_id, verts)
            self.assertGreaterEqual(l_id, 0)
            self.assertLess(l_id, len(verts))
        # alle edges binnen bereik
        for u, v, _ in edges:
            self.assertLess(u, len(verts))
            self.assertLess(v, len(verts))


# ============================================================
# 2. Pure-numpy QAOA statevector
# ============================================================

class TestQAOAStatevector(unittest.TestCase):
    def test_p0_is_uniform_plus_state(self):
        """Met p=0: ψ = |+⟩^⊗n, |ψ_i|² = 1/2^n."""
        psi = _qaoa_statevector(3, [(0, 1, 1.0), (1, 2, 1.0)], [], [])
        self.assertEqual(len(psi), 8)
        probs = np.abs(psi) ** 2
        np.testing.assert_allclose(probs, 1/8, atol=TOL)

    def test_unit_norm(self):
        np.random.seed(0)
        psi = _qaoa_statevector(4, [(0, 1, 1.0), (2, 3, 1.0), (0, 2, 1.0)],
                                 [0.4, 0.7], [0.3, 0.5])
        norm = float(np.sum(np.abs(psi) ** 2))
        self.assertAlmostEqual(norm, 1.0, delta=TOL)

    def test_zero_gamma_beta_stays_plus(self):
        """γ=β=0: blijft uniform, probs alle 1/2^n."""
        psi = _qaoa_statevector(3, [(0, 1, 1.0), (1, 2, 2.0)], [0.0], [0.0])
        probs = np.abs(psi) ** 2
        np.testing.assert_allclose(probs, 1/8, atol=TOL)

    def test_x_mixer_only_beta_pi_over_2_flips(self):
        """Zonder cost (γ=0) maar β=π/2: Rx(π) = −iX, dus |+⟩ blijft |+⟩.
        Dit test dat het X-mixer pad onze |+⟩ invariant laat."""
        psi = _qaoa_statevector(2, [], [0.0], [np.pi / 2])
        probs = np.abs(psi) ** 2
        np.testing.assert_allclose(probs, 1/4, atol=TOL)

    def test_single_edge_qaoa_nonzero(self):
        """Eén edge, γ=π/4, β=π/8: geef niet-triviale statevector."""
        psi = _qaoa_statevector(2, [(0, 1, 1.0)], [np.pi / 4], [np.pi / 8])
        probs = np.abs(psi) ** 2
        # Nog steeds genormaliseerd
        self.assertAlmostEqual(np.sum(probs), 1.0, delta=TOL)
        # en niet meer uniform (verschil op z'n minst meetbaar)
        self.assertFalse(np.allclose(probs, 1/4, atol=1e-3))


# ============================================================
# 3. ⟨Z_u Z_v⟩ verwachting
# ============================================================

class TestExpectationZZ(unittest.TestCase):
    def test_zz_plus_state_is_zero(self):
        """In |+⟩^⊗n: ⟨Z_u Z_v⟩ = 0 voor alle u≠v."""
        psi = _qaoa_statevector(3, [], [], [])
        for u in range(3):
            for v in range(u + 1, 3):
                self.assertAlmostEqual(_expectation_zz(psi, 3, u, v), 0.0, delta=TOL)

    def test_zz_zz_state_is_plus_one(self):
        """In |00⟩: ⟨Z_0 Z_1⟩ = (+1)(+1) = +1."""
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0  # |00⟩
        self.assertAlmostEqual(_expectation_zz(psi, 2, 0, 1), 1.0, delta=TOL)

    def test_zz_bell_minus_one(self):
        """In |01⟩: Z_0|01⟩ = +|01⟩, Z_1|01⟩ = −|01⟩, dus ⟨Z_0 Z_1⟩ = −1."""
        psi = np.zeros(4, dtype=complex)
        psi[1] = 1.0  # bit0=1, bit1=0 → state |01⟩ (convention bit0 is LSB)
        zz = _expectation_zz(psi, 2, 0, 1)
        self.assertAlmostEqual(zz, -1.0, delta=TOL)


# ============================================================
# 4. Greedy 1-flip
# ============================================================

class TestGreedy1Flip(unittest.TestCase):
    def test_k3_from_any_start(self):
        g = complete_graph(3)
        for start in ["000", "111", "001", "010", "011", "100", "101", "110"]:
            bits, val = _greedy_1flip(g, start)
            self.assertAlmostEqual(val, 2.0, delta=TOL)

    def test_already_optimal_unchanged(self):
        g = cycle_graph(4)
        bits, val = _greedy_1flip(g, "0101")
        self.assertAlmostEqual(val, 4.0, delta=TOL)

    def test_improves_suboptimal(self):
        g = complete_graph(4)
        # "0011" cut = 4 (bipartitie 2-2, alle edges tussen)
        # "0000" cut = 0
        _, val0 = _greedy_1flip(g, "0000")
        self.assertGreater(val0, 0.0)


# ============================================================
# 5. BP op bomen (exact)
# ============================================================

class TestBPOnTrees(unittest.TestCase):
    def test_path_p6(self):
        g = path_graph(6)
        res = mpqs_classical_bp(g)
        self.assertAlmostEqual(res["cut_value"], 5.0, delta=TOL)

    def test_path_p10(self):
        g = path_graph(10)
        res = mpqs_classical_bp(g)
        self.assertAlmostEqual(res["cut_value"], 9.0, delta=TOL)

    def test_star_graph(self):
        """Ster met 5 bladeren: OPT = 5 (snijd centrum vs alle bladeren)."""
        g = SimpleGraph(6)
        for i in range(1, 6):
            g.add_edge(0, i)
        res = mpqs_classical_bp(g)
        self.assertAlmostEqual(res["cut_value"], 5.0, delta=TOL)

    def test_weighted_tree(self):
        g = SimpleGraph(5)
        g.add_edge(0, 1, 2.0)
        g.add_edge(1, 2, 3.0)
        g.add_edge(2, 3, 1.5)
        g.add_edge(2, 4, 0.5)
        res = mpqs_classical_bp(g)
        total = sum(w for _, _, w in g.edges)
        self.assertAlmostEqual(res["cut_value"], total, delta=TOL)


# ============================================================
# 6. BP + refine op kleine grafen
# ============================================================

class TestBPOnSmall(unittest.TestCase):
    def test_k3(self):
        res = mpqs_classical_bp(complete_graph(3))
        self.assertAlmostEqual(res["cut_value"], 2.0, delta=TOL)

    def test_k4(self):
        res = mpqs_classical_bp(complete_graph(4))
        self.assertAlmostEqual(res["cut_value"], 4.0, delta=TOL)

    def test_c5(self):
        res = mpqs_classical_bp(cycle_graph(5))
        self.assertAlmostEqual(res["cut_value"], 4.0, delta=TOL)

    def test_c8_even(self):
        res = mpqs_classical_bp(cycle_graph(8))
        self.assertAlmostEqual(res["cut_value"], 8.0, delta=TOL)

    def test_petersen(self):
        res = mpqs_classical_bp(petersen_graph())
        self.assertAlmostEqual(res["cut_value"], 12.0, delta=TOL)

    def test_k33_bipartite(self):
        res = mpqs_classical_bp(complete_bipartite(3, 3))
        self.assertAlmostEqual(res["cut_value"], 9.0, delta=TOL)


# ============================================================
# 7. Lightcone correctheid
# ============================================================

class TestLightconeCorrectness(unittest.TestCase):
    def test_k3(self):
        res = mpqs_lightcone(complete_graph(3), radius=2)
        self.assertAlmostEqual(res["cut_value"], 2.0, delta=TOL)

    def test_k4(self):
        res = mpqs_lightcone(complete_graph(4), radius=2)
        self.assertAlmostEqual(res["cut_value"], 4.0, delta=TOL)

    def test_c5(self):
        res = mpqs_lightcone(cycle_graph(5), radius=2)
        self.assertAlmostEqual(res["cut_value"], 4.0, delta=TOL)

    def test_petersen(self):
        res = mpqs_lightcone(petersen_graph(), radius=2)
        self.assertAlmostEqual(res["cut_value"], 12.0, delta=TOL)

    def test_path_p6(self):
        res = mpqs_lightcone(path_graph(6), radius=2)
        self.assertAlmostEqual(res["cut_value"], 5.0, delta=TOL)

    def test_cut_bits_reproduce_cut_value(self):
        g = petersen_graph()
        res = mpqs_lightcone(g, radius=2)
        recomputed = g.cut_value(res["cut_bits"])
        self.assertAlmostEqual(recomputed, res["cut_value"], delta=TOL)

    def test_cut_below_or_equal_brute_force(self):
        """MPQS-cut ≤ brute-force-OPT (altijd)."""
        g = random_3regular(8, seed=13)
        opt = brute_force_maxcut(g)
        res = mpqs_lightcone(g, radius=2)
        self.assertLessEqual(res["cut_value"], opt + TOL)

    def test_lightcone_sizes_bounded_by_n(self):
        g = random_3regular(10, seed=42)
        res = mpqs_lightcone(g, radius=2)
        for L in res["lightcone_sizes"]:
            self.assertLessEqual(L, g.n)
            self.assertGreaterEqual(L, 1)


# ============================================================
# 8. Edge cases
# ============================================================

class TestEdgeCases(unittest.TestCase):
    def test_single_edge_bp(self):
        g = SimpleGraph(2)
        g.add_edge(0, 1)
        res = mpqs_classical_bp(g)
        self.assertAlmostEqual(res["cut_value"], 1.0, delta=TOL)

    def test_single_edge_lightcone(self):
        g = SimpleGraph(2)
        g.add_edge(0, 1)
        res = mpqs_lightcone(g, radius=1)
        self.assertAlmostEqual(res["cut_value"], 1.0, delta=TOL)

    def test_no_edges_bp(self):
        g = SimpleGraph(5)
        res = mpqs_classical_bp(g)
        self.assertAlmostEqual(res["cut_value"], 0.0, delta=TOL)

    def test_no_edges_lightcone(self):
        g = SimpleGraph(5)
        res = mpqs_lightcone(g, radius=2)
        self.assertAlmostEqual(res["cut_value"], 0.0, delta=TOL)

    def test_disconnected_two_triangles(self):
        g = SimpleGraph(6)
        g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(0, 2)
        g.add_edge(3, 4); g.add_edge(4, 5); g.add_edge(3, 5)
        res_bp = mpqs_classical_bp(g)
        res_lc = mpqs_lightcone(g, radius=2)
        self.assertAlmostEqual(res_bp["cut_value"], 4.0, delta=TOL)
        self.assertAlmostEqual(res_lc["cut_value"], 4.0, delta=TOL)


# ============================================================
# 9. Determinisme (reproduceerbaarheid)
# ============================================================

class TestDeterminism(unittest.TestCase):
    def test_bp_same_seed_same_result(self):
        g = random_3regular(10, seed=42)
        r1 = mpqs_classical_bp(g, seed=17)
        r2 = mpqs_classical_bp(g, seed=17)
        self.assertEqual(r1["cut_bits"], r2["cut_bits"])
        self.assertAlmostEqual(r1["cut_value"], r2["cut_value"], delta=TOL)

    def test_lightcone_deterministic(self):
        """Lightcone heeft vaste γ/β parameters en geen RNG, dus
        deterministisch voor gegeven input."""
        g = random_3regular(10, seed=42)
        r1 = mpqs_lightcone(g, radius=2)
        r2 = mpqs_lightcone(g, radius=2)
        self.assertEqual(r1["cut_bits"], r2["cut_bits"])
        self.assertAlmostEqual(r1["cut_value"], r2["cut_value"], delta=TOL)


# ============================================================
# 10. Lightcone-QAOA parameter-sensitivity
# ============================================================

class TestQAOAParameters(unittest.TestCase):
    def test_different_p_values_work(self):
        g = cycle_graph(6)
        for p in [1, 2, 3]:
            gammas = [0.3] * p
            betas = [0.2] * p
            res = mpqs_lightcone(g, radius=2, gammas=gammas, betas=betas)
            # C_6 heeft OPT = 6 (bipartiet); refine zou dat moeten bereiken
            self.assertAlmostEqual(res["cut_value"], 6.0, delta=TOL)

    def test_no_refine_still_gives_valid_cut(self):
        g = cycle_graph(6)
        res = mpqs_lightcone(g, radius=2, refine=False)
        # Cut is geldig (re-compute == opgeslagen)
        recomputed = g.cut_value(res["cut_bits"])
        self.assertAlmostEqual(recomputed, res["cut_value"], delta=TOL)
        # Cut ≤ OPT
        self.assertLessEqual(res["cut_value"], 6.0 + TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
