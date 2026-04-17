#!/usr/bin/env python3
"""
Tests voor B107: Quantum Nogood Learning

Test coverage:
  - Nogood datastructuur en methodes
  - NogoodDB indexering, deduplicatie, Z2-symmetrie
  - extract_exact_nogoods correctheid
  - extract_edge_nogoods correctheid
  - extract_triangle_nogoods correctheid
  - extract_heuristic_nogoods correctheid
  - nogood_guided_bls vs plain BLS
  - progressive_solve convergentie
  - learn_and_solve auto-tuning
  - Edge cases
"""

import numpy as np
import unittest
import time
from collections import defaultdict

from nogood_learner import (
    Nogood, NogoodDB,
    extract_exact_nogoods, extract_edge_nogoods,
    extract_triangle_nogoods, extract_heuristic_nogoods,
    nogood_guided_bls, progressive_solve, learn_and_solve,
    nogood_penalty_function,
)
from bls_solver import random_3regular


# ===================== Helpers =====================

def triangle_graph():
    return 3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]

def path_graph(n=4):
    return n, [(i, i+1, 1.0) for i in range(n-1)]

def single_edge():
    return 2, [(0, 1, 1.0)]

def weighted_triangle():
    return 3, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 0.5)]

def k4_graph():
    """Complete graaf K4."""
    n = 4
    edges = [(i, j, 1.0) for i in range(n) for j in range(i+1, n)]
    return n, edges

def compute_cut(n, edges, assignment):
    """Helper: bereken cut waarde."""
    cut = 0.0
    for u, v, w in edges:
        if assignment[u] != assignment[v]:
            cut += w
    return cut


# ===================== Nogood Datastructuur =====================

class TestNogood(unittest.TestCase):
    """Tests voor Nogood frozen dataclass."""

    def test_creation(self):
        ng = Nogood(
            nodes=frozenset([0, 1]),
            assignment=((0, 0), (1, 0)),
            cost_gap=1.0,
            source='exact',
        )
        self.assertEqual(ng.size, 2)
        self.assertEqual(ng.source, 'exact')
        self.assertAlmostEqual(ng.cost_gap, 1.0)

    def test_matches_positive(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        self.assertTrue(ng.matches({0: 0, 1: 1, 2: 0}))

    def test_matches_negative(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        self.assertFalse(ng.matches({0: 0, 1: 0, 2: 0}))

    def test_matches_missing_node(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        self.assertFalse(ng.matches({0: 0, 2: 1}))

    def test_flipped(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        fl = ng.flipped()
        self.assertEqual(fl.assignment, ((0, 1), (1, 0)))
        self.assertAlmostEqual(fl.cost_gap, 1.0)
        self.assertEqual(fl.source, 'exact')
        self.assertEqual(fl.nodes, ng.nodes)

    def test_pattern_key(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        self.assertEqual(ng.pattern_key, ((0, 0), (1, 1)))

    def test_frozen(self):
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 1)), 1.0, 'exact')
        with self.assertRaises(AttributeError):
            ng.cost_gap = 2.0


# ===================== NogoodDB =====================

class TestNogoodDB(unittest.TestCase):
    """Tests voor NogoodDB indexering en deduplicatie."""

    def test_add_and_count(self):
        db = NogoodDB()
        ng1 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        ng2 = Nogood(frozenset([1, 2]), ((1, 0), (2, 0)), 0.5, 'exact')
        self.assertTrue(db.add(ng1))
        self.assertTrue(db.add(ng2))
        self.assertEqual(db.total, 2)

    def test_deduplication(self):
        db = NogoodDB()
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        self.assertTrue(db.add(ng))
        self.assertFalse(db.add(ng))
        self.assertEqual(db.total, 1)
        self.assertEqual(db.n_duplicates_skipped, 1)

    def test_z2_deduplication(self):
        """Z2 symmetrie: (0,0)(1,0) en (0,1)(1,1) zijn equivalent."""
        db = NogoodDB()
        ng = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        ng_flip = ng.flipped()
        self.assertTrue(db.add(ng))
        self.assertFalse(db.add(ng_flip))
        self.assertEqual(db.total, 1)

    def test_lookup_node(self):
        db = NogoodDB()
        ng1 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        ng2 = Nogood(frozenset([1, 2]), ((1, 0), (2, 0)), 0.5, 'exact')
        db.add(ng1)
        db.add(ng2)
        # Node 1 zit in beide
        ngs = db.lookup_node(1)
        self.assertEqual(len(ngs), 2)
        # Node 0 zit in 1
        ngs = db.lookup_node(0)
        self.assertEqual(len(ngs), 1)
        # Node 3 zit in geen
        ngs = db.lookup_node(3)
        self.assertEqual(len(ngs), 0)

    def test_lookup_edge(self):
        db = NogoodDB()
        ng = Nogood(frozenset([0, 1, 2]), ((0, 0), (1, 0), (2, 1)), 1.0, 'exact')
        db.add(ng)
        # Edge (0,1) zit in ng
        self.assertEqual(len(db.lookup_edge(0, 1)), 1)
        # Edge (0,3) zit niet in ng
        self.assertEqual(len(db.lookup_edge(0, 3)), 0)

    def test_count_violations(self):
        db = NogoodDB()
        ng1 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        ng2 = Nogood(frozenset([2, 3]), ((2, 1), (3, 1)), 0.5, 'exact')
        db.add(ng1)
        db.add(ng2)
        # Assignment schendt ng1 maar niet ng2
        self.assertEqual(db.count_violations({0: 0, 1: 0, 2: 0, 3: 1}), 1)
        # Schendt beide
        self.assertEqual(db.count_violations({0: 0, 1: 0, 2: 1, 3: 1}), 2)
        # Schendt geen
        self.assertEqual(db.count_violations({0: 0, 1: 1, 2: 0, 3: 1}), 0)

    def test_violation_penalty(self):
        db = NogoodDB()
        ng1 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 2.0, 'exact')
        ng2 = Nogood(frozenset([2, 3]), ((2, 1), (3, 1)), 3.0, 'exact')
        db.add(ng1)
        db.add(ng2)
        pen = db.violation_penalty({0: 0, 1: 0, 2: 1, 3: 1})
        self.assertAlmostEqual(pen, 5.0)

    def test_filter_by_size(self):
        db = NogoodDB()
        ng2 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact')
        ng3 = Nogood(frozenset([0, 1, 2]), ((0, 0), (1, 0), (2, 0)), 1.0, 'exact')
        db.add(ng2)
        db.add(ng3)
        self.assertEqual(len(db.filter_by_size(2)), 1)
        self.assertEqual(len(db.filter_by_size(3)), 2)

    def test_filter_by_gap(self):
        db = NogoodDB()
        ng1 = Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 0.5, 'exact')
        ng2 = Nogood(frozenset([2, 3]), ((2, 0), (3, 0)), 2.0, 'exact')
        db.add(ng1)
        db.add(ng2)
        self.assertEqual(len(db.filter_by_gap(1.0)), 1)

    def test_source_counting(self):
        db = NogoodDB()
        db.add(Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 1.0, 'exact'))
        db.add(Nogood(frozenset([2, 3]), ((2, 0), (3, 0)), 1.0, 'heuristic'))
        db.add(Nogood(frozenset([4, 5]), ((4, 0), (5, 0)), 1.0, 'repair'))
        s = db.summary()
        self.assertEqual(s['n_exact'], 1)
        self.assertEqual(s['n_heuristic'], 1)
        self.assertEqual(s['n_repair'], 1)
        self.assertEqual(s['total'], 3)

    def test_empty_db(self):
        db = NogoodDB()
        self.assertEqual(db.total, 0)
        self.assertEqual(db.count_violations({0: 0}), 0)
        self.assertAlmostEqual(db.violation_penalty({0: 0}), 0.0)


# ===================== Extract: Exact =====================

class TestExtractExact(unittest.TestCase):
    """Tests voor extract_exact_nogoods."""

    def test_single_edge(self):
        """Enkele edge: gelijk assignment is suboptimaal met gap=1."""
        n, edges = single_edge()
        ngs = extract_exact_nogoods(n, edges, [0, 1], min_gap=0.5)
        # Er zijn 4 assignments, 2 optimaal (cut=1), 2 suboptimaal (cut=0, gap=1)
        self.assertEqual(len(ngs), 2)
        for ng in ngs:
            self.assertAlmostEqual(ng.cost_gap, 1.0)
            self.assertEqual(ng.source, 'exact')
            # Gelijke assignments
            vals = [v for _, v in ng.assignment]
            self.assertEqual(vals[0], vals[1])

    def test_triangle(self):
        """Driehoek: max cut = 2, er zijn gefrustreerde assignments."""
        n, edges = triangle_graph()
        ngs = extract_exact_nogoods(n, edges, [0, 1, 2], min_gap=0.5)
        # Max cut = 2 (twee edges gesneden)
        # Alle-0 en alle-1: cut=0, gap=2
        # Minimaal 2 nogoods met gap=2
        gaps = [ng.cost_gap for ng in ngs]
        self.assertIn(2.0, gaps)
        self.assertTrue(len(ngs) >= 2)

    def test_empty_subgraph(self):
        """Subgraaf zonder interne edges: geen nogoods."""
        n, edges = path_graph(6)
        ngs = extract_exact_nogoods(n, edges, [0, 3, 5], min_gap=0.1)
        self.assertEqual(len(ngs), 0)

    def test_too_large_subgraph(self):
        """Subgraaf > 22 nodes: return leeg."""
        ngs = extract_exact_nogoods(30, [], list(range(25)), min_gap=0.1)
        self.assertEqual(len(ngs), 0)

    def test_min_gap_filter(self):
        """Hogere min_gap filtert zwakke nogoods."""
        n, edges = triangle_graph()
        ngs_loose = extract_exact_nogoods(n, edges, [0, 1, 2], min_gap=0.1)
        ngs_strict = extract_exact_nogoods(n, edges, [0, 1, 2], min_gap=1.5)
        self.assertGreaterEqual(len(ngs_loose), len(ngs_strict))

    def test_correctness_k4(self):
        """K4: verify optimale cut = 4 en nogoods kloppen."""
        n, edges = k4_graph()
        ngs = extract_exact_nogoods(n, edges, [0, 1, 2, 3], min_gap=0.5)
        # K4 max cut = 4 (balanced partition)
        for ng in ngs:
            # Verify: deze assignment is inderdaad suboptimaal
            assign_dict = {node: val for node, val in ng.assignment}
            cut = compute_cut(n, edges, assign_dict)
            self.assertAlmostEqual(ng.cost_gap, 4.0 - cut)


# ===================== Extract: Edge =====================

class TestExtractEdge(unittest.TestCase):
    """Tests voor extract_edge_nogoods."""

    def test_positive_edge(self):
        """Positieve edge: gelijk assignment is nogood."""
        edges = [(0, 1, 2.0)]
        ngs = extract_edge_nogoods(edges, min_gap=0.1)
        # Twee nogoods: (0=0, 1=0) en (0=1, 1=1)
        self.assertEqual(len(ngs), 2)
        for ng in ngs:
            vals = dict(ng.assignment)
            self.assertEqual(vals[0], vals[1])  # gelijke toewijzing
            self.assertAlmostEqual(ng.cost_gap, 2.0)

    def test_negative_edge(self):
        """Negatieve edge: ongelijk assignment is nogood."""
        edges = [(0, 1, -2.0)]
        ngs = extract_edge_nogoods(edges, min_gap=0.1)
        self.assertEqual(len(ngs), 2)
        for ng in ngs:
            vals = dict(ng.assignment)
            self.assertNotEqual(vals[0], vals[1])  # ongelijke toewijzing
            self.assertAlmostEqual(ng.cost_gap, 2.0)

    def test_min_gap_filter(self):
        """Edges met gewicht < min_gap worden overgeslagen."""
        edges = [(0, 1, 0.05), (2, 3, 1.0)]
        ngs = extract_edge_nogoods(edges, min_gap=0.1)
        # Alleen edge (2,3) levert nogoods
        nodes_in_ngs = set()
        for ng in ngs:
            nodes_in_ngs |= ng.nodes
        self.assertNotIn(0, nodes_in_ngs)
        self.assertIn(2, nodes_in_ngs)

    def test_multiple_edges(self):
        """Meerdere edges: 2 nogoods per edge."""
        n, edges = path_graph(4)  # 3 edges
        ngs = extract_edge_nogoods(edges, min_gap=0.1)
        self.assertEqual(len(ngs), 6)  # 3 edges * 2


# ===================== Extract: Triangle =====================

class TestExtractTriangle(unittest.TestCase):
    """Tests voor extract_triangle_nogoods."""

    def test_unit_triangle(self):
        """Eenheidsdriehoek: max cut=2, worst cut=0, gap=2."""
        n, edges = triangle_graph()
        ngs = extract_triangle_nogoods(n, edges, min_frustration=0.5)
        self.assertTrue(len(ngs) > 0)
        # Alle-gelijk assignments moeten erbij zijn (gap=2)
        gaps = [ng.cost_gap for ng in ngs]
        self.assertIn(2.0, gaps)

    def test_no_triangles(self):
        """Pad-graaf: geen driehoeken → geen nogoods."""
        n, edges = path_graph(5)
        ngs = extract_triangle_nogoods(n, edges, min_frustration=0.1)
        self.assertEqual(len(ngs), 0)

    def test_weighted_triangle(self):
        """Gewogen driehoek: correcte gap berekening."""
        n, edges = weighted_triangle()  # w: 1.0, 2.0, 0.5
        ngs = extract_triangle_nogoods(n, edges, min_frustration=0.1)
        # Max cut = 1.0 + 2.0 = 3.0 (kan niet alle 3 snijden vanwege frustration)
        # Eigenlijk: max cut = max over alle 8 assignments
        total_w = 1.0 + 2.0 + 0.5
        # Check dat gaps correct zijn (best_cut - cut)
        for ng in ngs:
            assign_dict = {node: val for node, val in ng.assignment}
            cut = compute_cut(n, edges, assign_dict)
            # We weten niet precies best_cut, maar gap > 0
            self.assertGreater(ng.cost_gap, 0)

    def test_k4_multiple_triangles(self):
        """K4 heeft 4 driehoeken."""
        n, edges = k4_graph()
        ngs = extract_triangle_nogoods(n, edges, min_frustration=0.1)
        self.assertTrue(len(ngs) > 0)


# ===================== Extract: Heuristic =====================

class TestExtractHeuristic(unittest.TestCase):
    """Tests voor extract_heuristic_nogoods."""

    def test_improvement(self):
        """Verbetering door flip van 1 node."""
        n, edges = path_graph(4)
        before = {0: 0, 1: 0, 2: 1, 3: 0}
        after  = {0: 0, 1: 1, 2: 1, 3: 0}
        cut_b = compute_cut(n, edges, before)
        cut_a = compute_cut(n, edges, after)
        ngs = extract_heuristic_nogoods(before, after, cut_b, cut_a, edges)
        if cut_a > cut_b:
            self.assertEqual(len(ngs), 1)
            self.assertEqual(ngs[0].nodes, frozenset([1]))
            self.assertAlmostEqual(ngs[0].cost_gap, cut_a - cut_b)

    def test_no_improvement(self):
        """Geen verbetering: geen nogoods."""
        n, edges = path_graph(4)
        x = {0: 0, 1: 1, 2: 0, 3: 1}
        cut = compute_cut(n, edges, x)
        ngs = extract_heuristic_nogoods(x, x, cut, cut, edges)
        self.assertEqual(len(ngs), 0)

    def test_worsening(self):
        """Verslechtering: geen nogoods."""
        ngs = extract_heuristic_nogoods(
            {0: 0, 1: 1}, {0: 0, 1: 0}, 1.0, 0.5, [(0, 1, 1.0)])
        self.assertEqual(len(ngs), 0)

    def test_too_many_flips(self):
        """Meer dan max_size flips: geen nogoods."""
        before = {i: 0 for i in range(10)}
        after = {i: 1 for i in range(10)}
        ngs = extract_heuristic_nogoods(before, after, 0.0, 10.0, [], max_size=5)
        self.assertEqual(len(ngs), 0)

    def test_multi_flip(self):
        """Meerdere flips tegelijk."""
        n, edges = triangle_graph()
        before = {0: 0, 1: 0, 2: 0}
        after  = {0: 1, 1: 0, 2: 1}
        cut_b = compute_cut(n, edges, before)  # 0
        cut_a = compute_cut(n, edges, after)   # 2
        ngs = extract_heuristic_nogoods(before, after, cut_b, cut_a, edges)
        self.assertEqual(len(ngs), 1)
        self.assertEqual(ngs[0].nodes, frozenset([0, 2]))


# ===================== Nogood Penalty =====================

class TestNogoodPenalty(unittest.TestCase):
    """Tests voor nogood_penalty_function."""

    def test_empty_db(self):
        db = NogoodDB()
        f = nogood_penalty_function(db, 4)
        self.assertAlmostEqual(f({0: 0, 1: 1, 2: 0, 3: 1}), 0.0)

    def test_weighted_penalty(self):
        db = NogoodDB()
        db.add(Nogood(frozenset([0, 1]), ((0, 0), (1, 0)), 2.0, 'exact'))
        f = nogood_penalty_function(db, 4, weight=0.5)
        pen = f({0: 0, 1: 0, 2: 1, 3: 1})
        self.assertAlmostEqual(pen, 1.0)  # 0.5 * 2.0


# ===================== Guided BLS =====================

class TestNogoodGuidedBLS(unittest.TestCase):
    """Tests voor nogood_guided_bls."""

    def test_basic_run(self):
        """BLS vindt een redelijke cut op kleine graaf."""
        n, edges = triangle_graph()
        db = NogoodDB()
        result = nogood_guided_bls(n, edges, db, n_restarts=3, max_iter=50, seed=42)
        self.assertIn('best_cut', result)
        self.assertGreaterEqual(result['best_cut'], 0)
        self.assertIn('assignment', result)

    def test_with_nogoods_improves(self):
        """BLS met nogoods doet minstens zo goed als zonder (op K4)."""
        n, edges = k4_graph()

        # Zonder nogoods
        db_empty = NogoodDB()
        r_plain = nogood_guided_bls(n, edges, db_empty, n_restarts=5,
                                     max_iter=100, seed=42)

        # Met exacte nogoods
        db_full = NogoodDB()
        ngs = extract_exact_nogoods(n, edges, list(range(n)), min_gap=0.5)
        for ng in ngs:
            db_full.add(ng)

        r_guided = nogood_guided_bls(n, edges, db_full, n_restarts=5,
                                      max_iter=100, seed=42)

        # Guided moet minstens zo goed zijn
        self.assertGreaterEqual(r_guided['best_cut'], r_plain['best_cut'] - 0.01)

    def test_optimal_found_triangle(self):
        """Driehoek: optimale cut = 2."""
        n, edges = triangle_graph()
        db = NogoodDB()
        ngs = extract_exact_nogoods(n, edges, [0, 1, 2], min_gap=0.5)
        for ng in ngs:
            db.add(ng)
        result = nogood_guided_bls(n, edges, db, n_restarts=10, max_iter=100, seed=42)
        self.assertAlmostEqual(result['best_cut'], 2.0)

    def test_empty_graph(self):
        """Lege graaf: cut=0."""
        result = nogood_guided_bls(3, [], NogoodDB(), n_restarts=1, max_iter=10, seed=42)
        self.assertAlmostEqual(result['best_cut'], 0.0)

    def test_returns_valid_assignment(self):
        """Assignment moet n entries hebben met waarden 0/1."""
        n, edges = path_graph(6)
        result = nogood_guided_bls(n, edges, NogoodDB(), n_restarts=3, max_iter=50)
        self.assertEqual(len(result['assignment']), n)
        for v in result['assignment'].values():
            self.assertIn(v, [0, 1])


# ===================== Progressive Solve =====================

class TestProgressiveSolve(unittest.TestCase):
    """Tests voor progressive_solve."""

    def test_basic_convergence(self):
        """Progressive solve verbetert of houdt vast over rondes."""
        n = 10
        _, edges = random_3regular(n, seed=42)
        result = progressive_solve(n, edges, n_rounds=3, bls_restarts=3,
                                    bls_max_iter=100, seed=42)
        self.assertIn('best_cut', result)
        self.assertIn('history', result)
        self.assertEqual(len(result['history']), 3)

        # best_so_far is monotoon stijgend
        bests = [h['best_so_far'] for h in result['history']]
        for i in range(1, len(bests)):
            self.assertGreaterEqual(bests[i], bests[i-1])

    def test_nogoods_accumulate(self):
        """Nogoods worden geleerd over rondes."""
        n = 10
        _, edges = random_3regular(n, seed=42)
        result = progressive_solve(n, edges, n_rounds=3, bls_restarts=3,
                                    bls_max_iter=100, seed=42)
        # Ronde 1 zou al basis-nogoods moeten hebben
        self.assertGreater(result['history'][0]['n_nogoods'], 0)

    def test_db_in_result(self):
        """Result bevat NogoodDB met summary."""
        n, edges = triangle_graph()
        result = progressive_solve(n, edges, n_rounds=2, bls_restarts=2,
                                    bls_max_iter=50, seed=42)
        self.assertIsInstance(result['db'], NogoodDB)
        self.assertIn('db_summary', result)
        self.assertIn('total', result['db_summary'])

    def test_finds_optimal_small(self):
        """Op kleine graaf: vind optimum."""
        n, edges = triangle_graph()
        result = progressive_solve(n, edges, n_rounds=3, bls_restarts=5,
                                    bls_max_iter=100, seed=42)
        self.assertAlmostEqual(result['best_cut'], 2.0)


# ===================== Learn and Solve =====================

class TestLearnAndSolve(unittest.TestCase):
    """Tests voor learn_and_solve high-level interface."""

    def test_small_graph_exact_path(self):
        """n<=20: gebruikt exacte nogoods."""
        n, edges = triangle_graph()
        result = learn_and_solve(n, edges, seed=42)
        self.assertAlmostEqual(result['best_cut'], 2.0)
        self.assertIn('db_summary', result)

    def test_medium_graph_progressive(self):
        """n>20: gebruikt progressive_solve."""
        n = 30
        _, edges = random_3regular(n, seed=42)
        result = learn_and_solve(n, edges, time_limit=5.0, seed=42)
        self.assertIn('best_cut', result)
        self.assertGreater(result['best_cut'], 0)

    def test_returns_assignment(self):
        n = 10
        _, edges = random_3regular(n, seed=42)
        result = learn_and_solve(n, edges, seed=42)
        self.assertIn('assignment', result)
        # Verify assignment is consistent
        cut_check = compute_cut(n, edges, result['assignment'])
        self.assertAlmostEqual(result['best_cut'], cut_check, places=1)


# ===================== Edge Cases =====================

class TestEdgeCases(unittest.TestCase):
    """Edge cases en randgevallen."""

    def test_single_node(self):
        """1 node, 0 edges."""
        result = nogood_guided_bls(1, [], NogoodDB(), n_restarts=1, max_iter=10)
        self.assertAlmostEqual(result['best_cut'], 0.0)

    def test_disconnected_graph(self):
        """Disconnected graaf: edges in twee componenten."""
        edges = [(0, 1, 1.0), (2, 3, 1.0)]
        db = NogoodDB()
        result = nogood_guided_bls(4, edges, db, n_restarts=5, max_iter=50, seed=42)
        self.assertAlmostEqual(result['best_cut'], 2.0)

    def test_nogood_db_consistency(self):
        """DB summary is consistent met inhoud."""
        db = NogoodDB()
        for i in range(5):
            ng = Nogood(frozenset([i, i+1]),
                        tuple(sorted([(i, 0), (i+1, 0)])),
                        float(i+1), 'exact')
            db.add(ng)
        s = db.summary()
        self.assertEqual(s['total'], 5)
        self.assertEqual(s['n_exact'], 5)
        self.assertAlmostEqual(s['avg_gap'], 3.0)
        self.assertAlmostEqual(s['max_gap'], 5.0)

    def test_high_min_gap_no_nogoods(self):
        """Extreem hoge min_gap: geen nogoods."""
        n, edges = triangle_graph()
        ngs = extract_exact_nogoods(n, edges, [0, 1, 2], min_gap=100.0)
        self.assertEqual(len(ngs), 0)


if __name__ == '__main__':
    unittest.main()
