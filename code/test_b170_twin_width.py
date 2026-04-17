#!/usr/bin/env python3
"""Tests voor B170 twin-width primitives + cograph MaxCut DP."""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b170_twin_width import (
    BLACK,
    RED,
    Trigraph,
    brute_force_maxcut,
    build_cotree,
    cograph_maxcut_exact,
    complete_bipartite_edges,
    complete_edges,
    cycle_edges,
    empty_edges,
    is_cograph,
    path_edges,
    petersen_edges,
    tree_edges_balanced_binary,
    twin_width_exact,
    twin_width_heuristic,
)


# ============================================================
# 1. Trigraph basics
# ============================================================

class TestTrigraph(unittest.TestCase):

    def test_empty(self) -> None:
        g = Trigraph.from_graph(3, [])
        self.assertEqual(g.vertices, {0, 1, 2})
        self.assertEqual(g.max_red_degree(), 0)
        self.assertEqual(g.red_degree(0), 0)

    def test_from_graph_colors_black(self) -> None:
        g = Trigraph.from_graph(3, [(0, 1), (1, 2)])
        self.assertEqual(g.adj[0][1], BLACK)
        self.assertEqual(g.adj[1][0], BLACK)
        self.assertEqual(g.adj[1][2], BLACK)
        self.assertNotIn(2, g.adj[0])

    def test_contract_same_neighbors(self) -> None:
        g = Trigraph.from_graph(3, [(0, 1), (0, 2), (1, 2)])
        g.contract(0, 1)
        self.assertEqual(g.vertices, {1, 2})
        self.assertEqual(g.adj[1][2], BLACK)
        self.assertEqual(g.max_red_degree(), 0)

    def test_contract_different_neighbors_makes_red(self) -> None:
        g = Trigraph.from_graph(3, [(0, 1), (1, 2)])
        g.contract(0, 1)
        self.assertEqual(g.vertices, {1, 2})
        self.assertEqual(g.adj[1][2], RED)
        self.assertEqual(g.red_degree(1), 1)

    def test_contract_removes_vertex(self) -> None:
        g = Trigraph.from_graph(4, [(0, 1), (2, 3)])
        g.contract(0, 1)
        self.assertNotIn(0, g.vertices)
        self.assertNotIn(0, g.adj)

    def test_copy_independent(self) -> None:
        g = Trigraph.from_graph(3, [(0, 1)])
        h = g.copy()
        h.contract(0, 1)
        self.assertIn(0, g.vertices)
        self.assertIn(1, g.vertices)
        self.assertEqual(g.adj[0][1], BLACK)

    def test_contract_rejects_invalid(self) -> None:
        g = Trigraph.from_graph(3, [(0, 1)])
        with self.assertRaises(ValueError):
            g.contract(0, 0)
        with self.assertRaises(ValueError):
            g.contract(0, 7)


# ============================================================
# 2. Twin-width bekende waarden
# ============================================================

class TestTwinWidthKnown(unittest.TestCase):

    def test_empty_graph(self) -> None:
        g = Trigraph.from_graph(5, [])
        d, seq = twin_width_heuristic(g)
        self.assertEqual(d, 0)

    def test_complete_graph(self) -> None:
        for n in (3, 4, 5, 6):
            g = Trigraph.from_graph(n, complete_edges(n))
            d, _ = twin_width_heuristic(g)
            self.assertEqual(d, 0, f"K_{n}: verwacht tww=0, kreeg {d}")

    def test_complete_bipartite(self) -> None:
        for a, b in [(2, 3), (3, 3), (3, 4), (2, 5)]:
            g = Trigraph.from_graph(a + b, complete_bipartite_edges(a, b))
            d, _ = twin_width_heuristic(g)
            self.assertEqual(d, 0, f"K_{{{a},{b}}}: verwacht tww=0, kreeg {d}")

    def test_path_tww_one(self) -> None:
        # P_3 = K_{1,2} (ster, cograph) heeft tww=0.
        # P_n voor n ≥ 4 heeft tww = 1.
        g = Trigraph.from_graph(3, path_edges(3))
        d, _ = twin_width_heuristic(g)
        self.assertEqual(d, 0, f"P_3 (= ster, cograph): verwacht tww=0, kreeg {d}")

        for n in (4, 5, 6, 7, 8):
            g = Trigraph.from_graph(n, path_edges(n))
            d, _ = twin_width_heuristic(g)
            self.assertLessEqual(d, 1, f"P_{n}: verwacht tww<=1, kreeg {d}")

    def test_cycle_tww_at_most_two(self) -> None:
        for n in (4, 5, 6, 7, 8):
            g = Trigraph.from_graph(n, cycle_edges(n))
            d, _ = twin_width_heuristic(g)
            self.assertLessEqual(d, 2, f"C_{n}: verwacht tww<=2, kreeg {d}")

    def test_tree_tww(self) -> None:
        # Elke boom heeft tww = 1. Greedy is niet optimaal; we checken
        # dat de heuristic <=2 geeft voor een kleine boom, en dat de
        # exacte variant de 1 vindt.
        n, edges = tree_edges_balanced_binary(2)  # 7 vertices
        g = Trigraph.from_graph(n, edges)
        d, _ = twin_width_heuristic(g)
        self.assertLessEqual(d, 2, f"tree(7): verwacht tww<=2, kreeg {d}")

        g_exact = Trigraph.from_graph(n, edges)
        d_e, _ = twin_width_exact(g_exact, max_n=8)
        self.assertEqual(d_e, 1, f"tree(7) exact: verwacht tww=1, kreeg {d_e}")

    def test_twin_width_exact_small(self) -> None:
        # C_4 = K_{2,2} -> tww = 0
        g = Trigraph.from_graph(4, cycle_edges(4))
        d, _ = twin_width_exact(g, max_n=8)
        self.assertEqual(d, 0)

        # K_4 -> tww = 0
        g = Trigraph.from_graph(4, complete_edges(4))
        d, _ = twin_width_exact(g, max_n=8)
        self.assertEqual(d, 0)

        # P_4 -> tww = 1
        g = Trigraph.from_graph(4, path_edges(4))
        d, _ = twin_width_exact(g, max_n=8)
        self.assertEqual(d, 1)


# ============================================================
# 3. Cograph herkenning
# ============================================================

class TestIsCograph(unittest.TestCase):

    def test_empty_is_cograph(self) -> None:
        self.assertTrue(is_cograph(5, []))

    def test_complete_is_cograph(self) -> None:
        for n in (2, 3, 4, 6):
            self.assertTrue(is_cograph(n, complete_edges(n)), f"K_{n}")

    def test_bipartite_is_cograph(self) -> None:
        for a, b in [(2, 3), (3, 3), (4, 2)]:
            self.assertTrue(is_cograph(a + b, complete_bipartite_edges(a, b)))

    def test_p4_not_cograph(self) -> None:
        self.assertFalse(is_cograph(4, path_edges(4)))

    def test_c5_not_cograph(self) -> None:
        self.assertFalse(is_cograph(5, cycle_edges(5)))

    def test_c4_is_cograph(self) -> None:
        self.assertTrue(is_cograph(4, cycle_edges(4)))

    def test_petersen_not_cograph(self) -> None:
        self.assertFalse(is_cograph(10, petersen_edges()))


# ============================================================
# 4. Cotree constructie
# ============================================================

class TestCotree(unittest.TestCase):

    def test_leaf(self) -> None:
        root = build_cotree(1, [])
        self.assertEqual(root.kind, "leaf")
        self.assertEqual(root.vertex, 0)

    def test_two_disconnected(self) -> None:
        root = build_cotree(2, [])
        self.assertEqual(root.kind, "parallel")
        self.assertEqual(len(root.children), 2)

    def test_edge(self) -> None:
        root = build_cotree(2, [(0, 1)])
        self.assertEqual(root.kind, "series")
        self.assertEqual(len(root.children), 2)

    def test_k3_is_series(self) -> None:
        root = build_cotree(3, complete_edges(3))
        self.assertEqual(root.kind, "series")
        self.assertEqual(root.size, 3)

    def test_empty_3_is_parallel(self) -> None:
        root = build_cotree(3, [])
        self.assertEqual(root.kind, "parallel")
        self.assertEqual(root.size, 3)

    def test_raises_on_p4(self) -> None:
        with self.assertRaises(ValueError):
            build_cotree(4, path_edges(4))


# ============================================================
# 5. Cograph MaxCut DP vs brute force
# ============================================================

class TestCographMaxCut(unittest.TestCase):

    def test_edge(self) -> None:
        res = cograph_maxcut_exact(2, [(0, 1)])
        self.assertEqual(res["value"], 1.0)

    def test_k3(self) -> None:
        res = cograph_maxcut_exact(3, complete_edges(3))
        self.assertEqual(res["value"], 2.0)

    def test_k4(self) -> None:
        res = cograph_maxcut_exact(4, complete_edges(4))
        self.assertEqual(res["value"], 4.0)

    def test_k5(self) -> None:
        res = cograph_maxcut_exact(5, complete_edges(5))
        self.assertEqual(res["value"], 6.0)

    def test_complete_bipartite(self) -> None:
        res = cograph_maxcut_exact(6, complete_bipartite_edges(3, 3))
        self.assertEqual(res["value"], 9.0)

    def test_c4(self) -> None:
        res = cograph_maxcut_exact(4, cycle_edges(4))
        self.assertEqual(res["value"], 4.0)

    def test_raises_on_non_cograph(self) -> None:
        with self.assertRaises(ValueError):
            cograph_maxcut_exact(5, cycle_edges(5))

    def test_vs_brute_force_complete(self) -> None:
        for n in (3, 4, 5, 6, 7):
            edges = complete_edges(n)
            dp = cograph_maxcut_exact(n, edges)
            bf = brute_force_maxcut(n, edges)
            self.assertEqual(dp["value"], bf["value"], f"K_{n}")

    def test_vs_brute_force_bipartite(self) -> None:
        for a, b in [(2, 3), (3, 3), (3, 4), (4, 4)]:
            edges = complete_bipartite_edges(a, b)
            dp = cograph_maxcut_exact(a + b, edges)
            bf = brute_force_maxcut(a + b, edges)
            self.assertEqual(dp["value"], bf["value"])

    def test_vs_brute_force_random_cographs(self) -> None:
        """Construeer random cographs via random parallel/series-composities."""
        rng = random.Random(2026)
        for trial in range(6):
            n = rng.choice([6, 7, 8])
            pieces = [[v] for v in range(n)]
            edges: list = []
            while len(pieces) > 1:
                i, j = rng.sample(range(len(pieces)), 2)
                a, b = pieces[i], pieces[j]
                if rng.random() < 0.5:
                    # series
                    for x in a:
                        for y in b:
                            edges.append((x, y))
                # parallel: niks toevoegen
                merged = a + b
                new_pieces = [p for k, p in enumerate(pieces) if k not in (i, j)]
                new_pieces.append(merged)
                pieces = new_pieces

            self.assertTrue(is_cograph(n, edges), "expected cograph")
            dp = cograph_maxcut_exact(n, edges)
            bf = brute_force_maxcut(n, edges)
            self.assertEqual(dp["value"], bf["value"],
                             f"trial {trial}: dp={dp['value']} vs bf={bf['value']}")

    def test_partition_is_valid(self) -> None:
        n = 6
        edges = complete_edges(n)
        res = cograph_maxcut_exact(n, edges)
        cut = sum(1 for (u, v) in edges
                  if res["partition"][u] != res["partition"][v])
        self.assertEqual(cut, res["value"])


# ============================================================
# 6. Twin-width sequentie-correctheid
# ============================================================

class TestTwinWidthSequence(unittest.TestCase):

    def test_sequence_length(self) -> None:
        n = 6
        g = Trigraph.from_graph(n, cycle_edges(n))
        d, seq = twin_width_heuristic(g)
        self.assertEqual(len(seq), n - 1)

    def test_sequence_reduces_to_one_vertex(self) -> None:
        n = 7
        g0 = Trigraph.from_graph(n, path_edges(n))
        _, seq = twin_width_heuristic(g0)
        g = g0.copy()
        for (keep, remove) in seq:
            g.contract(remove, keep)
        self.assertEqual(len(g.vertices), 1)

    def test_heuristic_upper_bounds_exact(self) -> None:
        n = 6
        g = Trigraph.from_graph(n, cycle_edges(n))
        d_h, _ = twin_width_heuristic(g)
        d_e, _ = twin_width_exact(g, max_n=8)
        self.assertLessEqual(d_e, d_h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
