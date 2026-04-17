#!/usr/bin/env python3
"""
test_fractal_solver.py - Tests voor B79: Fractal Quantum Solver.

Tests:
  1. UnionFind: componenten correct
  2. Batch-merge: 4-node keten → 1 super-node
  3. Reconstructie: spin-propagatie via ZZ-teken
  4. Small grid solve (4x3): ratio > 0.5 en O(log N) rondes
  5. solve_grid met LightconeQAOA (4x3)
  6. Vergelijking FQS vs RQAOA op 4x3
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractal_solver import UnionFind, FractalSolver
from rqaoa import WeightedGraph, brute_force_maxcut


def test_union_find():
    """Test 1: UnionFind componenten."""
    uf = UnionFind([0, 1, 2, 3, 4])
    uf.union(0, 1)
    uf.union(2, 3)
    comps = uf.components()

    assert len(comps) == 3, "Verwacht 3 componenten, got %d" % len(comps)
    # 0+1, 2+3, 4 elk een component
    sizes = sorted([len(c) for c in comps.values()])
    assert sizes == [1, 2, 2], "Verwacht [1,2,2], got %s" % sizes

    # Chain: union 1+2 → {0,1,2,3}
    uf.union(1, 2)
    comps = uf.components()
    assert len(comps) == 2, "Na chain: verwacht 2 componenten, got %d" % len(comps)
    sizes = sorted([len(c) for c in comps.values()])
    assert sizes == [1, 4], "Verwacht [1,4], got %s" % sizes

    print("  [PASS] test_union_find")


def test_batch_merge_chain():
    """Test 2: Batch-merge van een 4-node keten."""
    # Keten: 0 -- 1 -- 2 -- 3
    g = WeightedGraph()
    for i in range(4):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)

    solver = FractalSolver(g, merge_threshold=0.5, verbose=False)

    # Fake ZZ: alle sterk positief gecorreleerd
    zz = {(0, 1): 0.9, (1, 2): 0.8, (2, 3): 0.85}
    new_graph, merge_info = solver._batch_merge(g, zz, threshold=0.5)

    assert new_graph is not None, "Verwacht merge, got None"
    assert new_graph.n_nodes == 1, "Hele keten zou 1 super-node moeten worden, got %d" % new_graph.n_nodes
    assert new_graph.n_edges == 0, "Geen edges in 1-node graaf"
    assert len(merge_info['components']) == 1, "Verwacht 1 component"

    print("  [PASS] test_batch_merge_chain")


def test_batch_merge_two_clusters():
    """Test 2b: Twee clusters met zwakke verbinding."""
    # 0--1 sterk, 2--3 sterk, 1--2 zwak
    g = WeightedGraph()
    for i in range(4):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)

    solver = FractalSolver(g, verbose=False)
    zz = {(0, 1): 0.9, (1, 2): 0.3, (2, 3): 0.85}
    new_graph, merge_info = solver._batch_merge(g, zz, threshold=0.7)

    assert new_graph is not None, "Verwacht merge"
    assert new_graph.n_nodes == 2, "Verwacht 2 super-nodes, got %d" % new_graph.n_nodes
    assert new_graph.n_edges == 1, "Verwacht 1 verbinding, got %d" % new_graph.n_edges

    print("  [PASS] test_batch_merge_two_clusters")


def test_reconstruct_spins():
    """Test 3: Reconstructie: ZZ-teken bepaalt spin."""
    # Component {0, 1, 2}: root=0, zz(0,1)=+0.9 → zelfde spin,
    #                                zz(1,2)=-0.8 → tegengestelde spin
    merge_info = {
        'components': {0: [0, 1, 2]},
        'zz_to_root': {0: 1, 1: 1, 2: -1},  # 2 tegengesteld aan root
        'internal_offset': 0,
    }

    super_assign = {0: 1}
    full = FractalSolver.reconstruct([merge_info], super_assign)

    assert full[0] == 1, "Root moet +1 zijn"
    assert full[1] == 1, "Node 1 positief gecorreleerd met root"
    assert full[2] == -1, "Node 2 negatief gecorreleerd met root"

    # Test met root = -1
    super_assign2 = {0: -1}
    full2 = FractalSolver.reconstruct([merge_info], super_assign2)
    assert full2[0] == -1
    assert full2[1] == -1
    assert full2[2] == 1

    print("  [PASS] test_reconstruct_spins")


def test_small_graph_solve():
    """Test 4: FQS op 4x3 grid, vergelijk met brute force."""
    Lx, Ly = 4, 3
    g = WeightedGraph.grid(Lx, Ly)
    n_edges = g.n_edges

    # Brute force optimum
    bf_cut, bf_assign = brute_force_maxcut(g)
    bf_ratio = bf_cut / n_edges
    print("    Brute force: cut=%.0f ratio=%.6f" % (bf_cut, bf_ratio))

    # FQS
    solver = FractalSolver(g, p=1, merge_threshold=0.6,
                           brute_threshold=12, verbose=False)
    cut, assign, ratio, info = solver.solve(local_search=True)

    print("    FQS: cut=%.0f ratio=%.6f rondes=%d (%d->%d nodes)" %
          (cut, ratio, info['rounds'], info['n_start'], info['n_final']))

    assert ratio > 0.5, "FQS ratio moet >0.5 zijn, got %.4f" % ratio
    assert ratio <= bf_ratio + 0.001, \
        "FQS ratio (%.4f) kan niet hoger zijn dan brute force (%.4f)" % (ratio, bf_ratio)
    # O(log N) rondes: 12 nodes, max 3-4 rondes verwacht
    assert info['rounds'] <= 6, \
        "Verwacht ≤6 rondes voor 12 nodes, got %d" % info['rounds']

    print("  [PASS] test_small_graph_solve")


def test_solve_grid_lightcone():
    """Test 5: FQS solve_grid met LightconeQAOA op 4x3."""
    Lx, Ly = 4, 3
    g = WeightedGraph.grid(Lx, Ly)
    n_edges = g.n_edges

    solver = FractalSolver(g, p=1, merge_threshold=0.5,
                           brute_threshold=12, verbose=False)
    cut, assign, ratio, info = solver.solve_grid(
        Lx, Ly, local_search=True)

    print("    LightconeQAOA FQS: cut=%.0f ratio=%.6f rondes=%d" %
          (cut, ratio, info['rounds']))

    assert ratio > 0.5, "Lightcone FQS ratio moet >0.5, got %.4f" % ratio
    # Alle nodes moeten een assignment hebben
    assert len(assign) >= Lx * Ly, \
        "Verwacht %d assignments, got %d" % (Lx * Ly, len(assign))
    # Alle spins moeten +1 of -1 zijn
    for node, spin in assign.items():
        assert spin in (1, -1), "Node %d heeft spin %s" % (node, spin)

    print("  [PASS] test_solve_grid_lightcone")


def test_fqs_vs_rqaoa():
    """Test 6: FQS vs RQAOA op 4x3 — FQS zou vergelijkbaar moeten zijn."""
    from rqaoa import RQAOA

    Lx, Ly = 4, 3
    g = WeightedGraph.grid(Lx, Ly)
    n_edges = g.n_edges

    # FQS
    solver = FractalSolver(g, p=1, merge_threshold=0.5,
                           brute_threshold=12, verbose=False)
    fqs_cut, _, fqs_ratio, fqs_info = solver.solve_grid(
        Lx, Ly, local_search=True)

    # RQAOA (hergebruik zelfde params)
    rqaoa = RQAOA(g, p=1, verbose=False)
    r_cut, _, r_ratio, r_info = rqaoa.solve_grid_hybrid(
        Lx, Ly, gammas=fqs_info.get('gammas'), betas=fqs_info.get('betas'))

    print("    FQS:   ratio=%.6f  cut=%.0f  rondes=%d" %
          (fqs_ratio, fqs_cut, fqs_info['rounds']))
    print("    RQAOA: ratio=%.6f  cut=%.0f" % (r_ratio, r_cut))
    print("    Delta: %+.6f" % (fqs_ratio - r_ratio))

    # FQS mag niet veel slechter zijn dan RQAOA (na local search)
    assert fqs_ratio > r_ratio - 0.05, \
        "FQS (%.4f) significant slechter dan RQAOA (%.4f)" % (fqs_ratio, r_ratio)

    print("  [PASS] test_fqs_vs_rqaoa")


if __name__ == '__main__':
    print("=== B79 FQS Tests ===\n")

    test_union_find()
    test_batch_merge_chain()
    test_batch_merge_two_clusters()
    test_reconstruct_spins()
    test_small_graph_solve()
    test_solve_grid_lightcone()
    test_fqs_vs_rqaoa()

    print("\n=== Alle 7 tests PASSED ===")
