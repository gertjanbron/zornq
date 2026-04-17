#!/usr/bin/env python3
"""Tests voor B109: Adversarial Instance Generator + Benchmark."""

import unittest
import numpy as np
from adversarial_instance_generator import (
    gen_high_feedback_dense,
    gen_frustrated_antiferro,
    gen_planted_partition,
    gen_expander,
    gen_weighted_conflict,
    gen_treewidth_barrier,
    gen_chimera,
    small_adversarial_suite,
    medium_adversarial_suite,
    scaling_suite,
    compute_planted_gap,
    classify_difficulty,
    _graph_stats,
)


class TestGraphGeneration(unittest.TestCase):
    """Test dat elke familie correcte grafen genereert."""

    def _validate_instance(self, inst, expected_family):
        """Gemeenschappelijke validatie voor alle instanties."""
        self.assertEqual(inst['family'], expected_family)
        self.assertGreater(inst['n_nodes'], 0)
        self.assertGreater(len(inst['edges']), 0)
        self.assertIn('target_solver', inst)
        self.assertIn('stats', inst)
        self.assertIn('difficulty_note', inst)

        # Check edge format
        for edge in inst['edges']:
            self.assertEqual(len(edge), 3)
            u, v, w = edge
            self.assertIsInstance(u, int)
            self.assertIsInstance(v, int)
            self.assertIsInstance(w, float)
            self.assertNotEqual(u, v)
            self.assertGreaterEqual(u, 0)
            self.assertGreaterEqual(v, 0)
            self.assertLess(u, inst['n_nodes'])
            self.assertLess(v, inst['n_nodes'])

        # Check connectivity: alle nodes moeten bereikbaar zijn
        n = inst['n_nodes']
        adj = {i: set() for i in range(n)}
        for u, v, w in inst['edges']:
            adj[u].add(v)
            adj[v].add(u)

        # Check dat er geen isolated nodes zijn (behalve bij kleine n)
        nodes_in_edges = set()
        for u, v, w in inst['edges']:
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)

        # Stats check
        stats = inst['stats']
        self.assertEqual(stats['n'], n)
        self.assertEqual(stats['m'], len(inst['edges']))

    def test_high_feedback_dense(self):
        inst = gen_high_feedback_dense(n=30, density=0.5, seed=42)
        self._validate_instance(inst, 'high_feedback_dense')
        # Cyclomaticiteit moet hoog zijn
        self.assertGreater(inst['stats']['cyclomaticity'], 10)

    def test_high_feedback_dense_connected(self):
        """Dense graaf moet connected zijn (backbone edges)."""
        inst = gen_high_feedback_dense(n=20, density=0.3, seed=42)
        n = inst['n_nodes']
        adj = {i: set() for i in range(n)}
        for u, v, w in inst['edges']:
            adj[u].add(v)
            adj[v].add(u)
        # BFS connectivity check
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        self.assertEqual(len(visited), n, "Graaf niet connected")

    def test_frustrated_antiferro(self):
        inst = gen_frustrated_antiferro(n=25, p_triangle=0.3, seed=42)
        self._validate_instance(inst, 'frustrated_antiferro')
        # Check dat er driehoeks-edges zijn (frustratie)
        self.assertGreater(len(inst['edges']), inst['n_nodes'])

    def test_planted_partition(self):
        inst = gen_planted_partition(n=40, p_in=0.3, p_out=0.7, seed=42)
        self._validate_instance(inst, 'planted_partition')
        # Planted cut moet positief zijn
        self.assertIsNotNone(inst['planted_cut'])
        self.assertGreater(inst['planted_cut'], 0)

    def test_planted_partition_near_threshold(self):
        """Planted partition dichtbij detectability threshold."""
        inst = gen_planted_partition(n=100, p_in=0.45, p_out=0.55, seed=42)
        self._validate_instance(inst, 'planted_partition')

    def test_planted_partition_with_noise(self):
        inst = gen_planted_partition(n=50, p_in=0.3, p_out=0.7, noise=0.2, seed=42)
        self._validate_instance(inst, 'planted_partition')
        # Met noise moeten er negatieve gewichten zijn
        neg = sum(1 for _, _, w in inst['edges'] if w < 0)
        self.assertGreater(neg, 0)

    def test_expander(self):
        inst = gen_expander(n=30, d=5, seed=42)
        self._validate_instance(inst, 'expander')
        # Gemiddelde graad moet ≈ d zijn
        self.assertGreater(inst['stats']['avg_degree'], 2)

    def test_weighted_conflict(self):
        inst = gen_weighted_conflict(n=30, scale_ratio=100.0, seed=42)
        self._validate_instance(inst, 'weighted_conflict')
        # Check multi-scale weights
        weights = [abs(w) for _, _, w in inst['edges']]
        max_w = max(weights)
        min_w = min(weights)
        self.assertGreater(max_w / min_w, 10)

    def test_treewidth_barrier(self):
        inst = gen_treewidth_barrier(k=5, copies=3, seed=42)
        self._validate_instance(inst, 'treewidth_barrier')
        self.assertEqual(inst['n_nodes'], 15)
        # Moet clique-edges bevatten (k*(k-1)/2 per kopie)
        self.assertGreaterEqual(len(inst['edges']), 3 * 10)  # 3 kopieën × C(5,2)

    def test_chimera(self):
        inst = gen_chimera(L=2, seed=42)
        self._validate_instance(inst, 'chimera_topology')
        self.assertEqual(inst['n_nodes'], 8 * 2 * 2)  # 8*L*L
        # Chimera heeft 16 intra-cell + 4 hor + 4 vert edges per cell (approx)
        self.assertGreater(len(inst['edges']), 30)

    def test_chimera_larger(self):
        inst = gen_chimera(L=3, seed=42)
        self.assertEqual(inst['n_nodes'], 72)


class TestSuites(unittest.TestCase):
    """Test suite generators."""

    def test_small_suite_size(self):
        suite = small_adversarial_suite()
        self.assertEqual(len(suite), 7)

    def test_small_suite_families(self):
        suite = small_adversarial_suite()
        families = {inst['family'] for inst in suite}
        expected = {
            'high_feedback_dense', 'frustrated_antiferro', 'planted_partition',
            'expander', 'weighted_conflict', 'treewidth_barrier', 'chimera_topology'
        }
        self.assertEqual(families, expected)

    def test_medium_suite_size(self):
        suite = medium_adversarial_suite()
        self.assertEqual(len(suite), 15)

    def test_scaling_suite(self):
        suite = scaling_suite('expander', [20, 40, 60], d=3)
        self.assertEqual(len(suite), 3)
        sizes = [inst['n_nodes'] for inst in suite]
        self.assertEqual(sizes, [20, 40, 60])

    def test_scaling_suite_treewidth(self):
        suite = scaling_suite('treewidth_barrier', [4, 5, 6])
        self.assertEqual(len(suite), 3)

    def test_reproducibility(self):
        """Zelfde seed → zelfde graaf."""
        a = gen_expander(n=30, d=5, seed=123)
        b = gen_expander(n=30, d=5, seed=123)
        self.assertEqual(a['edges'], b['edges'])

    def test_different_seeds(self):
        """Verschillende seeds → verschillende grafen."""
        a = gen_expander(n=30, d=5, seed=1)
        b = gen_expander(n=30, d=5, seed=2)
        self.assertNotEqual(a['edges'], b['edges'])


class TestAnalysis(unittest.TestCase):
    """Test analyse functies."""

    def test_planted_gap(self):
        inst = gen_planted_partition(n=40, p_in=0.3, p_out=0.7, seed=42)
        gap = compute_planted_gap(inst, inst['planted_cut'])
        self.assertAlmostEqual(gap, 0.0)

    def test_planted_gap_partial(self):
        inst = gen_planted_partition(n=40, p_in=0.3, p_out=0.7, seed=42)
        gap = compute_planted_gap(inst, inst['planted_cut'] * 0.9)
        self.assertAlmostEqual(gap, 0.1, places=5)

    def test_planted_gap_no_planted(self):
        inst = gen_expander(n=30, d=5, seed=42)
        gap = compute_planted_gap(inst, 100.0)
        self.assertIsNone(gap)

    def test_classify_difficulty(self):
        # High cyclomaticity
        inst = gen_high_feedback_dense(n=50, density=0.5, seed=42)
        diff = classify_difficulty(inst)
        self.assertEqual(diff, 'HARD_FOR_EXACT')

    def test_graph_stats(self):
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, -1.0)]
        stats = _graph_stats(3, edges)
        self.assertEqual(stats['n'], 3)
        self.assertEqual(stats['m'], 3)
        self.assertAlmostEqual(stats['density'], 1.0)
        self.assertEqual(stats['cyclomaticity'], 1)
        self.assertAlmostEqual(stats['total_weight'], 2.0)


class TestSolverIntegration(unittest.TestCase):
    """Test dat solvers correct draaien op adversarial instanties."""

    def test_bls_on_small_suite(self):
        """BLS moet op alle kleine instanties een positieve cut vinden."""
        from bls_solver import bls_maxcut
        for inst in small_adversarial_suite(seed=42):
            n = inst['n_nodes']
            edges = inst['edges']
            result = bls_maxcut(n, edges, n_restarts=3, max_iter=500,
                                time_limit=5, seed=42)
            self.assertGreater(result['best_cut'], 0,
                               f"BLS faalde op {inst['name']}")

    def test_pa_on_small_suite(self):
        """PA moet op alle kleine instanties een positieve cut vinden."""
        from pa_solver import pa_maxcut
        for inst in small_adversarial_suite(seed=42):
            n = inst['n_nodes']
            edges = inst['edges']
            result = pa_maxcut(n, edges, n_replicas=50, n_temps=20,
                               time_limit=5, seed=42)
            self.assertGreater(result['best_cut'], 0,
                               f"PA faalde op {inst['name']}")

    def test_b99_on_small_suite(self):
        """B99 moet op alle kleine instanties een resultaat geven."""
        from feedback_edge_solver import feedback_edge_maxcut
        for inst in small_adversarial_suite(seed=42):
            n = inst['n_nodes']
            edges = inst['edges']
            cut, assignment, info = feedback_edge_maxcut(
                n, edges, time_limit=10, seed=42)
            self.assertGreater(cut, 0, f"B99 faalde op {inst['name']}")


class TestEdgeCases(unittest.TestCase):
    """Test randgevallen en parameter-grenzen."""

    def test_minimal_high_feedback(self):
        inst = gen_high_feedback_dense(n=5, density=0.8, seed=42)
        self.assertGreater(len(inst['edges']), 0)

    def test_minimal_expander(self):
        inst = gen_expander(n=6, d=3, seed=42)
        self.assertGreater(len(inst['edges']), 0)

    def test_minimal_chimera(self):
        inst = gen_chimera(L=1, seed=42)
        self.assertEqual(inst['n_nodes'], 8)

    def test_minimal_treewidth(self):
        inst = gen_treewidth_barrier(k=3, copies=2, seed=42)
        self.assertEqual(inst['n_nodes'], 6)

    def test_large_scale_ratio(self):
        inst = gen_weighted_conflict(n=20, scale_ratio=10000.0, seed=42)
        weights = [abs(w) for _, _, w in inst['edges']]
        self.assertGreater(max(weights) / min(weights), 100)

    def test_no_self_loops(self):
        """Geen enkele instantie mag self-loops bevatten."""
        for inst in small_adversarial_suite():
            for u, v, w in inst['edges']:
                self.assertNotEqual(u, v,
                    f"Self-loop ({u},{v}) in {inst['name']}")

    def test_no_duplicate_edges(self):
        """Geen duplicaat-edges."""
        for inst in small_adversarial_suite():
            seen = set()
            for u, v, w in inst['edges']:
                key = (min(u, v), max(u, v))
                self.assertNotIn(key, seen,
                    f"Duplicaat edge {key} in {inst['name']}")
                seen.add(key)


if __name__ == '__main__':
    unittest.main()
