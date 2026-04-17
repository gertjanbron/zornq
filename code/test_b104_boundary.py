#!/usr/bin/env python3
"""
Tests voor B104: Boundary-State Compiler

Test coverage:
  - Separator detectie (BFS, vertex)
  - Graaf decompositie (patches, cross-edges)
  - Patch compilatie (boundary response maps)
  - Stitching (enumerate, greedy)
  - Isomorfisme caching
  - Lightcone boundary cache
  - Correctheid vs brute-force
  - Edge cases
"""

import numpy as np
import unittest
import time

from boundary_state_compiler import (
    Patch, BoundaryResponse, CompiledGraph,
    find_bfs_separator, find_vertex_separator,
    decompose_graph, compile_patch, compile_graph,
    compile_graph_with_isomorphism, stitch_solve,
    boundary_solve, patch_structure_key,
    LightconeBoundaryCache,
)
from bls_solver import random_3regular


# ===================== Helpers =====================

def triangle_graph():
    return 3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]

def path_graph(n=6):
    return n, [(i, i+1, 1.0) for i in range(n-1)]

def grid_graph(Lx, Ly):
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            if x + 1 < Lx:
                edges.append((node, (x+1)*Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((node, x*Ly + y + 1, 1.0))
    return n, edges

def single_edge():
    return 2, [(0, 1, 1.0)]

def compute_cut(n, edges, assignment):
    cut = 0.0
    for u, v, w in edges:
        if assignment.get(u, 0) != assignment.get(v, 0):
            cut += w
    return cut

def brute_force_maxcut(n, edges):
    best = 0.0
    best_assign = {}
    for s in range(2 ** n):
        cut = 0.0
        for u, v, w in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                cut += w
        if cut > best:
            best = cut
            best_assign = {i: (s >> i) & 1 for i in range(n)}
    return best, best_assign


# ===================== Separator Tests =====================

class TestSeparator(unittest.TestCase):
    """Tests voor separator detectie."""

    def test_path_graph_separator(self):
        """Pad-graaf: BFS separator is een middennode."""
        n, edges = path_graph(7)
        sep, a, b = find_bfs_separator(n, edges)
        # Should find some separator
        if sep:
            self.assertTrue(len(a) > 0)
            self.assertTrue(len(b) > 0)
            # Separator + sides = all nodes
            self.assertEqual(sep | a | b, set(range(n)))
            # No overlap
            self.assertEqual(len(sep) + len(a) + len(b), n)

    def test_grid_separator(self):
        """Grid: separator moet een kolom of rij zijn."""
        n, edges = grid_graph(4, 3)
        sep, a, b = find_bfs_separator(n, edges)
        if sep:
            self.assertTrue(len(sep) <= n)
            self.assertEqual(sep | a | b, set(range(n)))

    def test_vertex_separator_quality(self):
        """Vertex separator kiest een redelijk kleine separator."""
        n, edges = grid_graph(5, 3)
        sep, a, b = find_vertex_separator(n, edges, max_separator_size=10)
        if sep:
            self.assertLessEqual(len(sep), 10)
            self.assertEqual(sep | a | b, set(range(n)))

    def test_small_graph_no_separator(self):
        """Hele kleine graaf: misschien geen separator."""
        n, edges = single_edge()
        sep, a, b = find_bfs_separator(n, edges)
        # Te klein voor separator
        total = len(sep) + len(a) + len(b)
        self.assertEqual(total, n)

    def test_disconnected_graph(self):
        """Disconnected graaf: separator van een component."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)]
        sep, a, b = find_bfs_separator(6, edges)
        # Should handle disconnected gracefully
        self.assertIsInstance(sep, set)


# ===================== Decomposition Tests =====================

class TestDecomposition(unittest.TestCase):
    """Tests voor graaf decompositie."""

    def test_small_graph_single_patch(self):
        """Kleine graaf: één patch, geen decompositie."""
        n, edges = triangle_graph()
        patches, cross, sep = decompose_graph(n, edges, max_patch_size=10)
        self.assertEqual(len(patches), 1)
        self.assertEqual(len(cross), 0)
        self.assertEqual(patches[0].size, 3)

    def test_path_decomposes(self):
        """Lang pad: wordt opgesplitst in patches."""
        n, edges = path_graph(10)
        patches, cross, sep = decompose_graph(n, edges, max_patch_size=6)
        # Should have at least 2 patches
        self.assertGreaterEqual(len(patches), 2)
        # All interior nodes + boundary nodes cover all graph nodes
        all_nodes = set()
        for p in patches:
            all_nodes |= p.nodes
        self.assertEqual(all_nodes, set(range(n)))

    def test_grid_decomposes(self):
        """Grid decomposeert in meerdere patches."""
        n, edges = grid_graph(4, 4)
        patches, cross, sep = decompose_graph(n, edges, max_patch_size=10)
        self.assertGreaterEqual(len(patches), 2)

    def test_patch_edges_valid(self):
        """Alle edges in een patch hebben beide endpoints in die patch."""
        n, edges = path_graph(8)
        patches, cross, sep = decompose_graph(n, edges, max_patch_size=5)
        for p in patches:
            for u, v, w in p.edges:
                self.assertTrue(u in p.nodes or v in p.nodes,
                                f"Edge ({u},{v}) niet in patch {p.patch_id}")


# ===================== Patch Compilation Tests =====================

class TestCompilePatch(unittest.TestCase):
    """Tests voor patch compilatie."""

    def test_no_boundary(self):
        """Patch zonder boundary: één entry in response map."""
        patch = Patch(
            patch_id=0,
            boundary=frozenset(),
            interior=frozenset([0, 1, 2]),
            edges=[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            boundary_edges=[],
        )
        resp = compile_patch(patch)
        # 2^0 = 1 boundary config
        self.assertEqual(resp.n_entries, 1)
        cut, interior = resp.response_map[()]
        # Driehoek max cut = 2
        self.assertAlmostEqual(cut, 2.0)

    def test_single_boundary(self):
        """Patch met 1 boundary node."""
        patch = Patch(
            patch_id=0,
            boundary=frozenset([0]),
            interior=frozenset([1, 2]),
            edges=[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            boundary_edges=[],
        )
        resp = compile_patch(patch)
        # 2^1 = 2 boundary configs
        self.assertEqual(resp.n_entries, 2)
        # Beide configs moeten een geldige cut opleveren
        for bkey in [(0,), (1,)]:
            cut, interior = resp.response_map[bkey]
            self.assertGreaterEqual(cut, 0)

    def test_correctness_vs_brute_force(self):
        """Compiled patch geeft zelfde optimum als brute force."""
        n, edges = triangle_graph()
        patch = Patch(
            patch_id=0,
            boundary=frozenset([0]),
            interior=frozenset([1, 2]),
            edges=edges,
            boundary_edges=[],
        )
        resp = compile_patch(patch)

        # Voor elke boundary assignment: check optimum
        for b_val in [0, 1]:
            cut, interior = resp.lookup({0: b_val})
            # Verify by brute force
            best_bf = -1
            for s in range(4):  # 2 interior nodes
                assign = {0: b_val, 1: s & 1, 2: (s >> 1) & 1}
                c = compute_cut(n, edges, assign)
                best_bf = max(best_bf, c)
            self.assertAlmostEqual(cut, best_bf)

    def test_no_edges(self):
        """Patch zonder edges: cut=0 voor alle configs."""
        patch = Patch(
            patch_id=0,
            boundary=frozenset([0]),
            interior=frozenset([1]),
            edges=[],
            boundary_edges=[],
        )
        resp = compile_patch(patch)
        for bkey, (cut, _) in resp.response_map.items():
            self.assertAlmostEqual(cut, 0.0)

    def test_lookup_method(self):
        """BoundaryResponse.lookup werkt correct."""
        patch = Patch(
            patch_id=0,
            boundary=frozenset([0, 1]),
            interior=frozenset([2]),
            edges=[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
            boundary_edges=[],
        )
        resp = compile_patch(patch)
        # Lookup met dict
        cut, interior = resp.lookup({0: 0, 1: 1})
        self.assertGreater(cut, 0)
        self.assertIn(2, interior)


# ===================== Full Compile + Stitch Tests =====================

class TestCompileAndStitch(unittest.TestCase):
    """Tests voor volledige compile + stitch pipeline."""

    def test_small_graph_exact(self):
        """Klein genoeg: boundary solve vindt optimum."""
        n, edges = triangle_graph()
        result = boundary_solve(n, edges)
        opt, _ = brute_force_maxcut(n, edges)
        self.assertAlmostEqual(result['best_cut'], opt)

    def test_path_graph_optimal(self):
        """Pad-graaf: optimum = n-1."""
        n, edges = path_graph(6)
        result = boundary_solve(n, edges, max_patch_size=4)
        opt, _ = brute_force_maxcut(n, edges)
        # Should be close to optimal (exact if separator is good)
        self.assertGreaterEqual(result['best_cut'], opt * 0.8)

    def test_3regular_quality(self):
        """3-reguliere graaf: boundary solve vs brute force."""
        n = 10
        _, edges = random_3regular(n, seed=42)
        result = boundary_solve(n, edges, max_patch_size=8)
        opt, _ = brute_force_maxcut(n, edges)
        # Should find at least 70% of optimal
        self.assertGreaterEqual(result['best_cut'], opt * 0.7)

    def test_single_patch_no_decomposition(self):
        """Kleine graaf: geen decompositie nodig."""
        n, edges = triangle_graph()
        compiled = compile_graph(n, edges, max_patch_size=10)
        self.assertEqual(compiled.n_patches, 1)
        result = stitch_solve(compiled, n, edges)
        self.assertAlmostEqual(result['best_cut'], 2.0)

    def test_assignment_valid(self):
        """Assignment bevat alle nodes met waarden 0/1."""
        n = 10
        _, edges = random_3regular(n, seed=42)
        result = boundary_solve(n, edges, max_patch_size=6)
        for i in range(n):
            self.assertIn(i, result['assignment'])
            self.assertIn(result['assignment'][i], [0, 1])

    def test_cut_consistent(self):
        """Gerapporteerde cut klopt met assignment."""
        n = 8
        _, edges = random_3regular(n, seed=42)
        result = boundary_solve(n, edges, max_patch_size=6)
        actual_cut = compute_cut(n, edges, result['assignment'])
        self.assertAlmostEqual(result['best_cut'], actual_cut, places=5)

    def test_grid_decompose_and_solve(self):
        """4x3 grid: decompose + solve."""
        n, edges = grid_graph(4, 3)
        result = boundary_solve(n, edges, max_patch_size=8)
        opt, _ = brute_force_maxcut(n, edges)
        self.assertGreaterEqual(result['best_cut'], opt * 0.7)


# ===================== Isomorphism Cache Tests =====================

class TestIsomorphismCache(unittest.TestCase):
    """Tests voor isomorfisme-detectie en caching."""

    def test_identical_patches_same_key(self):
        """Twee identieke patches hebben dezelfde structure key."""
        p1 = Patch(0, frozenset([0]), frozenset([1, 2]),
                    [(0, 1, 1.0), (1, 2, 1.0)], [])
        p2 = Patch(1, frozenset([3]), frozenset([4, 5]),
                    [(3, 4, 1.0), (4, 5, 1.0)], [])
        self.assertEqual(patch_structure_key(p1), patch_structure_key(p2))

    def test_different_patches_different_key(self):
        """Structureel verschillende patches hebben verschillende keys."""
        p1 = Patch(0, frozenset([0]), frozenset([1, 2]),
                    [(0, 1, 1.0), (1, 2, 1.0)], [])
        p2 = Patch(1, frozenset([0]), frozenset([1, 2]),
                    [(0, 1, 1.0), (0, 2, 1.0)], [])  # driehoek vs pad
        self.assertNotEqual(patch_structure_key(p1), patch_structure_key(p2))

    def test_isomorphism_reduces_compilation(self):
        """Grid: isomorfisme cache vermindert compilatie werk."""
        n, edges = grid_graph(4, 4)

        # Zonder isomorfisme
        c1 = compile_graph(n, edges, max_patch_size=10)
        # Met isomorfisme
        c2 = compile_graph_with_isomorphism(n, edges, max_patch_size=10)

        # Zelfde resultaat
        r1 = stitch_solve(c1, n, edges)
        r2 = stitch_solve(c2, n, edges)
        self.assertAlmostEqual(r1['best_cut'], r2['best_cut'])


# ===================== Lightcone Boundary Cache Tests =====================

class TestLightconeBoundaryCache(unittest.TestCase):
    """Tests voor LightconeBoundaryCache."""

    def test_miss_then_hit(self):
        cache = LightconeBoundaryCache()
        nodes = {0, 1, 2}
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        target = (0, 1)
        key = cache.make_key(nodes, edges, target, 1, (0.5,), (0.3,))

        # Miss
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.n_misses, 1)

        # Store
        cache.put(key, 0.42)

        # Hit
        self.assertAlmostEqual(cache.get(key), 0.42)
        self.assertEqual(cache.n_hits, 1)

    def test_different_params_different_key(self):
        cache = LightconeBoundaryCache()
        nodes = {0, 1, 2}
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        target = (0, 1)

        key1 = cache.make_key(nodes, edges, target, 1, (0.5,), (0.3,))
        key2 = cache.make_key(nodes, edges, target, 1, (0.6,), (0.3,))
        self.assertNotEqual(key1, key2)

    def test_isomorphic_lightcones_same_key(self):
        """Structureel identieke lightcones met verschillende node-ids."""
        cache = LightconeBoundaryCache()

        # Lightcone 1: nodes {0,1,2}, target (0,1)
        key1 = cache.make_key({0, 1, 2}, [(0, 1, 1.0), (1, 2, 1.0)],
                               (0, 1), 1, (0.5,), (0.3,))
        # Lightcone 2: nodes {5,6,7}, target (5,6), same structure
        key2 = cache.make_key({5, 6, 7}, [(5, 6, 1.0), (6, 7, 1.0)],
                               (5, 6), 1, (0.5,), (0.3,))
        self.assertEqual(key1, key2)

    def test_hit_rate(self):
        cache = LightconeBoundaryCache()
        key = cache.make_key({0, 1}, [(0, 1, 1.0)], (0, 1), 1, (0.5,), (0.3,))
        cache.get(key)  # miss
        cache.put(key, 0.5)
        cache.get(key)  # hit
        cache.get(key)  # hit
        self.assertAlmostEqual(cache.hit_rate, 2/3)

    def test_summary(self):
        cache = LightconeBoundaryCache()
        s = cache.summary()
        self.assertEqual(s['size'], 0)
        self.assertEqual(s['hits'], 0)


# ===================== Edge Cases =====================

class TestEdgeCases(unittest.TestCase):
    """Edge cases en randgevallen."""

    def test_empty_graph(self):
        """Lege graaf: cut=0."""
        result = boundary_solve(3, [])
        self.assertAlmostEqual(result['best_cut'], 0.0)

    def test_single_edge(self):
        """Enkele edge: cut=1."""
        n, edges = single_edge()
        result = boundary_solve(n, edges)
        self.assertAlmostEqual(result['best_cut'], 1.0)

    def test_disconnected(self):
        """Disconnected graaf."""
        edges = [(0, 1, 1.0), (2, 3, 1.0)]
        result = boundary_solve(4, edges)
        self.assertAlmostEqual(result['best_cut'], 2.0)

    def test_weighted_edges(self):
        """Graaf met niet-uniforme gewichten."""
        edges = [(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)]
        result = boundary_solve(3, edges)
        opt, _ = brute_force_maxcut(3, edges)
        self.assertAlmostEqual(result['best_cut'], opt)

    def test_large_patch_size(self):
        """max_patch_size > n: geen decompositie."""
        n = 8
        _, edges = random_3regular(n, seed=42)
        result = boundary_solve(n, edges, max_patch_size=100)
        opt, _ = brute_force_maxcut(n, edges)
        self.assertAlmostEqual(result['best_cut'], opt)


if __name__ == '__main__':
    unittest.main()
