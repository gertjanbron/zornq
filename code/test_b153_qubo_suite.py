#!/usr/bin/env python3
"""Tests voor B153 Beyond-MaxCut QUBO Suite."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b153_qubo_suite import (
    QUBO,
    QUBOInstance,
    encode_markowitz,
    encode_max_k_cut,
    encode_mis,
    encode_weighted_maxcut,
    qubo_brute_force,
    qubo_local_search,
    qubo_random_restart,
    qubo_simulated_annealing,
    random_markowitz_instance,
    cycle_edges,
    complete_edges,
    petersen_edges,
    random_erdos_renyi_edges,
    random_weighted_edges,
    solve_problem,
)


# ============================================================
# Helpers
# ============================================================

def _all_bitstrings(n: int):
    for bits in range(2 ** n):
        yield np.array([(bits >> i) & 1 for i in range(n)], dtype=float)


# ============================================================
# QUBO datatype
# ============================================================

class TestQUBOClass(unittest.TestCase):
    def test_n_property(self):
        Q = np.eye(5)
        q = QUBO(Q)
        self.assertEqual(q.n, 5)

    def test_evaluate_diagonal(self):
        Q = np.diag([1.0, 2.0, -3.0])
        q = QUBO(Q)
        # x = (1,1,1) → 1+2-3 = 0
        self.assertAlmostEqual(q.evaluate(np.array([1, 1, 1])), 0.0)
        # x = (1,0,0) → 1
        self.assertAlmostEqual(q.evaluate(np.array([1, 0, 0])), 1.0)
        # x = (0,0,1) → -3
        self.assertAlmostEqual(q.evaluate(np.array([0, 0, 1])), -3.0)

    def test_evaluate_offdiag(self):
        # Q = [[0, 1], [1, 0]]: x^T Q x = 2 x_0 x_1
        q = QUBO(np.array([[0.0, 1.0], [1.0, 0.0]]))
        self.assertAlmostEqual(q.evaluate(np.array([1, 1])), 2.0)
        self.assertAlmostEqual(q.evaluate(np.array([1, 0])), 0.0)

    def test_offset_added(self):
        q = QUBO(np.zeros((2, 2)), offset=5.0)
        self.assertAlmostEqual(q.evaluate(np.array([1, 1])), 5.0)

    def test_symmetrize_on_construction(self):
        Q = np.array([[0.0, 2.0], [0.0, 0.0]])  # asymmetrisch
        q = QUBO(Q)
        # 2*x_0*x_1 / 2 + 2*x_0*x_1 / 2 = effectief 1 op (0,1) en (1,0)
        self.assertAlmostEqual(q.Q[0, 1], 1.0)
        self.assertAlmostEqual(q.Q[1, 0], 1.0)
        self.assertAlmostEqual(q.evaluate(np.array([1, 1])), 2.0)

    def test_evaluate_int_matches_array(self):
        rng = np.random.RandomState(0)
        Q = rng.normal(0, 1, (4, 4))
        q = QUBO(Q)
        for bits in range(16):
            x = np.array([(bits >> i) & 1 for i in range(4)], dtype=float)
            self.assertAlmostEqual(q.evaluate(x), q.evaluate_int(bits))

    def test_delta_flip_matches_explicit(self):
        rng = np.random.RandomState(1)
        Q = rng.normal(0, 1, (5, 5))
        q = QUBO(Q)
        x = rng.randint(0, 2, 5).astype(float)
        E0 = q.evaluate(x)
        for i in range(5):
            xp = x.copy()
            xp[i] = 1.0 - xp[i]
            E1 = q.evaluate(xp)
            self.assertAlmostEqual(q.delta_flip(x, i), E1 - E0, places=10)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            QUBO(np.zeros((3, 4)))


# ============================================================
# Weighted MaxCut
# ============================================================

class TestWeightedMaxCut(unittest.TestCase):
    def test_triangle_unweighted(self):
        # K_3: OPT = 2 (alle 1-vs-2 splits)
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        inst = encode_weighted_maxcut(3, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertAlmostEqual(d["value"], 2.0)
        self.assertTrue(d["feasible"])

    def test_path_p4(self):
        # Pad 0-1-2-3: bipartite, OPT = 3
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        inst = encode_weighted_maxcut(4, edges)
        res = qubo_brute_force(inst.qubo)
        self.assertAlmostEqual(inst.decode(res["x"])["value"], 3.0)

    def test_weighted_triangle(self):
        # K_3 met weights 1,2,3. OPT = bipartiete cut die 2+3=5 of 1+3=4 of 1+2=3 levert.
        # Maximale 2 edges in cut: 2+3 = 5.
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]
        inst = encode_weighted_maxcut(3, edges)
        res = qubo_brute_force(inst.qubo)
        self.assertAlmostEqual(inst.decode(res["x"])["value"], 5.0)

    def test_negative_weights(self):
        # Edge met negatieve weight: liever NIET cutten
        edges = [(0, 1, 1.0), (0, 1, -2.0)]  # netto -1 op die edge
        inst = encode_weighted_maxcut(2, edges)
        res = qubo_brute_force(inst.qubo)
        # Beste: niet cutten; value = 0 (als x=00 of x=11)
        # of cutten geeft 1 + (-2) = -1
        d = inst.decode(res["x"])
        self.assertAlmostEqual(d["value"], 0.0)

    def test_complete_graph_k4(self):
        # K_4: bipartiete cut OPT = 4 (2-2 split)
        edges = [(i, j, 1.0) for i in range(4) for j in range(i + 1, 4)]
        inst = encode_weighted_maxcut(4, edges)
        res = qubo_brute_force(inst.qubo)
        self.assertAlmostEqual(inst.decode(res["x"])["value"], 4.0)

    def test_petersen_graph(self):
        # Petersen MaxCut OPT = 12 (bekende waarde)
        edges = [(u, v, 1.0) for (u, v) in petersen_edges()]
        inst = encode_weighted_maxcut(10, edges)
        res = qubo_brute_force(inst.qubo)
        self.assertAlmostEqual(inst.decode(res["x"])["value"], 12.0)


# ============================================================
# Max-k-Cut
# ============================================================

class TestMaxKCut(unittest.TestCase):
    def test_k_eq_2_matches_maxcut(self):
        # Max-2-Cut moet gelijk zijn aan MaxCut op K_3
        edges_w = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        inst2 = encode_max_k_cut(3, edges_w, k=2)
        res2 = qubo_brute_force(inst2.qubo)
        d2 = inst2.decode(res2["x"])
        self.assertTrue(d2["feasible"])
        self.assertAlmostEqual(d2["value"], 2.0)

    def test_k_eq_3_on_k4_is_full(self):
        # K_4 met 3 kleuren: alle 3 edges kunnen gecut worden door verschillende kleuren
        # maar er zijn 4 knopen en 3 kleuren ⇒ 2 knopen krijgen dezelfde kleur ⇒ minstens 1 edge intern.
        # OPT = 6 - 1 = 5 (bekende max-3-cut waarde voor K_4)
        edges_w = [(i, j, 1.0) for i in range(4) for j in range(i + 1, 4)]
        inst = encode_max_k_cut(4, edges_w, k=3)
        res = qubo_brute_force(inst.qubo, max_n=14)
        d = inst.decode(res["x"])
        self.assertTrue(d["feasible"])
        self.assertAlmostEqual(d["value"], 5.0)

    def test_k_eq_3_on_triangle(self):
        # K_3 met 3 kleuren: alle 3 knopen kunnen unieke kleur krijgen ⇒ alle 3 edges in cut
        edges_w = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        inst = encode_max_k_cut(3, edges_w, k=3)
        res = qubo_brute_force(inst.qubo, max_n=12)
        d = inst.decode(res["x"])
        self.assertTrue(d["feasible"])
        self.assertAlmostEqual(d["value"], 3.0)

    def test_one_hot_penalty_enforces_feasibility(self):
        # Met grote penalty moeten optimale oplossingen feasible zijn
        edges_w = [(0, 1, 1.0)]
        inst = encode_max_k_cut(2, edges_w, k=2)
        res = qubo_brute_force(inst.qubo, max_n=8)
        d = inst.decode(res["x"])
        self.assertTrue(d["feasible"])

    def test_k_lt_2_raises(self):
        with self.assertRaises(ValueError):
            encode_max_k_cut(3, [], k=1)


# ============================================================
# Maximum Independent Set
# ============================================================

class TestMIS(unittest.TestCase):
    def test_no_edges(self):
        # Geen edges: alle knopen onafhankelijk
        inst = encode_mis(5, [])
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 5)
        self.assertTrue(d["feasible"])

    def test_complete_graph(self):
        # K_n: maximaal onafhankelijke set = 1
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        inst = encode_mis(5, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 1)
        self.assertTrue(d["feasible"])

    def test_cycle_c5(self):
        # C_5: max independent set = 2 (alternerend, niet 3 want 5 oneven)
        edges = cycle_edges(5)
        inst = encode_mis(5, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 2)
        self.assertTrue(d["feasible"])

    def test_cycle_c6(self):
        # C_6: max independent set = 3 (alternerend)
        edges = cycle_edges(6)
        inst = encode_mis(6, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 3)
        self.assertTrue(d["feasible"])

    def test_petersen(self):
        # Petersen: alpha(G) = 4 (bekend)
        edges = petersen_edges()
        inst = encode_mis(10, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 4)
        self.assertTrue(d["feasible"])

    def test_decode_violations(self):
        # Forceer een infeasibele x en check violations-counter
        edges = [(0, 1)]
        inst = encode_mis(2, edges)
        d = inst.decode(np.array([1, 1]))
        self.assertEqual(d["violations"], 1)
        self.assertFalse(d["feasible"])

    def test_path_p4(self):
        # P_4 (0-1-2-3): max indep set = 2 (knopen 0,2 of 1,3 of 0,3)
        edges = [(0, 1), (1, 2), (2, 3)]
        inst = encode_mis(4, edges)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertEqual(d["size"], 2)
        self.assertTrue(d["feasible"])


# ============================================================
# Markowitz portfolio
# ============================================================

class TestMarkowitz(unittest.TestCase):
    def test_obvious_winner(self):
        # 2 assets, K=1: kies degene met hoogste return / risk
        returns = np.array([1.0, 0.1])
        cov = np.eye(2)
        inst = encode_markowitz(returns, cov, budget=1, risk_aversion=1.0)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertTrue(d["feasible"])
        self.assertEqual(d["selected"], [0])

    def test_budget_constraint_enforced(self):
        # Sterke penalty + redelijk budget: optimaal moet feasible zijn
        rng = np.random.RandomState(7)
        returns = rng.normal(0, 0.5, 6)
        cov = np.eye(6)
        inst = encode_markowitz(returns, cov, budget=2, risk_aversion=0.1, penalty=10.0)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        self.assertTrue(d["feasible"])
        self.assertEqual(d["size"], 2)

    def test_risk_aversion_avoids_correlated(self):
        # 2 assets met identieke returns; één paar sterk gecorreleerd, K=2 forceert beide
        # ⇒ test dat utility lager is bij hogere correlatie (sanity check)
        returns = np.array([1.0, 1.0, 1.0])
        cov_low = np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
        cov_hi = np.eye(3) + 0.5 * (np.ones((3, 3)) - np.eye(3))
        inst_low = encode_markowitz(returns, cov_low, budget=2, risk_aversion=1.0)
        inst_hi = encode_markowitz(returns, cov_hi, budget=2, risk_aversion=1.0)
        r_low = qubo_brute_force(inst_low.qubo)
        r_hi = qubo_brute_force(inst_hi.qubo)
        u_low = inst_low.decode(r_low["x"])["utility"]
        u_hi = inst_hi.decode(r_hi["x"])["utility"]
        self.assertGreater(u_low, u_hi - 1e-9)

    def test_random_instance_factory(self):
        inst = random_markowitz_instance(5, seed=0)
        self.assertEqual(inst.qubo.n, 5)
        self.assertEqual(inst.metadata["n"], 5)
        self.assertGreaterEqual(inst.metadata["budget"], 1)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            encode_markowitz(np.ones(3), np.eye(3), budget=4)

    def test_qubo_minimum_matches_handpicked_max(self):
        # Verify min(QUBO) energy-as-utility matches the manual maximum.
        returns = np.array([1.0, 0.5])
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        inst = encode_markowitz(returns, cov, budget=1, risk_aversion=1.0, penalty=2.0)
        res = qubo_brute_force(inst.qubo)
        d = inst.decode(res["x"])
        # Hand-checked best: kies asset 0, utility = 1 - 1*1 = 0
        self.assertTrue(d["feasible"])
        self.assertAlmostEqual(d["utility"], 0.0)


# ============================================================
# Solvers
# ============================================================

class TestSolvers(unittest.TestCase):
    def setUp(self):
        # Triviaal probleem met bekende ground state: x = (1, 0, 1, 0, 1)
        # Q diagonaal = [-1, +1, -1, +1, -1] ⇒ alleen "1"-bits met negatieve diag.
        self.Q = np.diag([-1.0, 1.0, -1.0, 1.0, -1.0])
        self.qubo = QUBO(self.Q)

    def test_brute_force_finds_optimum(self):
        res = qubo_brute_force(self.qubo)
        self.assertAlmostEqual(res["energy"], -3.0)
        self.assertTrue(res["certified"])
        np.testing.assert_array_equal(res["x"], np.array([1, 0, 1, 0, 1]))

    def test_brute_force_max_n_guard(self):
        big = QUBO(np.eye(30))
        with self.assertRaises(ValueError):
            qubo_brute_force(big, max_n=22)

    def test_local_search_finds_optimum(self):
        res = qubo_local_search(self.qubo, seed=0)
        self.assertAlmostEqual(res["energy"], -3.0)
        self.assertFalse(res["certified"])

    def test_simulated_annealing_finds_optimum(self):
        res = qubo_simulated_annealing(self.qubo, n_sweeps=200, seed=0)
        self.assertAlmostEqual(res["energy"], -3.0)

    def test_random_restart_local_search(self):
        res = qubo_random_restart(self.qubo, n_starts=5, seed=0)
        self.assertAlmostEqual(res["energy"], -3.0)

    def test_random_restart_with_sa(self):
        res = qubo_random_restart(self.qubo, n_starts=3, seed=0,
                                  inner="simulated_annealing",
                                  inner_kwargs={"n_sweeps": 50})
        self.assertAlmostEqual(res["energy"], -3.0)

    def test_solve_problem_returns_decoded(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        inst = encode_weighted_maxcut(3, edges)
        out = solve_problem(inst, solver="brute_force")
        self.assertIn("decoded", out)
        self.assertIn("value", out["decoded"])
        self.assertEqual(out["problem"], "WeightedMaxCut")

    def test_solve_problem_unknown_solver_raises(self):
        inst = encode_weighted_maxcut(2, [(0, 1, 1.0)])
        with self.assertRaises(ValueError):
            solve_problem(inst, solver="quantum_magic")


# ============================================================
# Integratie: brute_force == local_search/sa op kleine instanties
# ============================================================

class TestSolverConsistency(unittest.TestCase):
    def test_maxcut_random_consistency(self):
        for seed in [0, 1, 2, 3]:
            edges = random_weighted_edges(7, p=0.5, seed=seed)
            inst = encode_weighted_maxcut(7, edges)
            bf = qubo_brute_force(inst.qubo)
            rr = qubo_random_restart(inst.qubo, n_starts=10,
                                     seed=seed, inner="local_search")
            self.assertLessEqual(bf["energy"], rr["energy"] + 1e-9)
            # LS hoeft optimum niet altijd te vinden, maar met 10 starts wel typisch
            # Toetsen we lossere claim: random_restart ≤ single LS:
            single_ls = qubo_local_search(inst.qubo, seed=seed)
            self.assertLessEqual(rr["energy"], single_ls["energy"] + 1e-9)

    def test_mis_random_consistency(self):
        edges = random_erdos_renyi_edges(8, 0.4, seed=42)
        inst = encode_mis(8, edges)
        bf = qubo_brute_force(inst.qubo)
        sa = qubo_simulated_annealing(inst.qubo, n_sweeps=200, seed=0)
        # SA moet ≥ optimum bereiken (lager is beter omdat min)
        self.assertLessEqual(sa["energy"], inst.qubo.evaluate(np.zeros(8)) + 1e-9)
        # Brute force is lower-bound op SA
        self.assertLessEqual(bf["energy"], sa["energy"] + 1e-9)

    def test_kcut_random_consistency_k3(self):
        edges_w = random_weighted_edges(4, p=0.6, seed=11)
        inst = encode_max_k_cut(4, edges_w, k=3)
        bf = qubo_brute_force(inst.qubo, max_n=14)
        sa = qubo_simulated_annealing(inst.qubo, n_sweeps=500, seed=2)
        self.assertLessEqual(bf["energy"], sa["energy"] + 1e-9)


# ============================================================
# Graaf-helpers
# ============================================================

class TestGraphHelpers(unittest.TestCase):
    def test_cycle_edges_count(self):
        self.assertEqual(len(cycle_edges(5)), 5)

    def test_complete_edges_count(self):
        self.assertEqual(len(complete_edges(6)), 15)

    def test_petersen_count(self):
        self.assertEqual(len(petersen_edges()), 15)

    def test_petersen_3regular(self):
        # Iedere knoop heeft graad 3
        deg = [0] * 10
        for u, v in petersen_edges():
            deg[u] += 1
            deg[v] += 1
        self.assertTrue(all(d == 3 for d in deg))

    def test_random_erdos_renyi_reproducible(self):
        e1 = random_erdos_renyi_edges(8, 0.5, seed=42)
        e2 = random_erdos_renyi_edges(8, 0.5, seed=42)
        self.assertEqual(e1, e2)

    def test_random_weighted_edges_have_weights(self):
        edges = random_weighted_edges(5, 1.0, seed=0, w_low=0.5, w_high=0.5)
        # p=1.0, w_low=w_high=0.5 ⇒ alle edges, alle weights 0.5
        self.assertEqual(len(edges), 10)
        for u, v, w in edges:
            self.assertAlmostEqual(w, 0.5)


# ============================================================
# CLI smoke
# ============================================================

class TestCLISmoke(unittest.TestCase):
    def test_cli_maxcut(self):
        from b153_qubo_suite import main
        rc = main(["maxcut", "--n", "4", "--p", "1.0", "--seed", "0",
                   "--solver", "brute_force"])
        self.assertEqual(rc, 0)

    def test_cli_kcut(self):
        from b153_qubo_suite import main
        rc = main(["kcut", "--n", "3", "--k", "3", "--p", "1.0", "--seed", "0",
                   "--solver", "brute_force"])
        self.assertEqual(rc, 0)

    def test_cli_mis(self):
        from b153_qubo_suite import main
        rc = main(["mis", "--n", "5", "--p", "0.4", "--seed", "0",
                   "--solver", "brute_force"])
        self.assertEqual(rc, 0)

    def test_cli_markowitz(self):
        from b153_qubo_suite import main
        rc = main(["markowitz", "--n", "5", "--budget", "2", "--seed", "0",
                   "--solver", "brute_force"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
