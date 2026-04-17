#\!/usr/bin/env python3
"""test_kuramoto_solver.py - Tests voor B92 Anti-Kuramoto MaxCut Solver."""

import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from kuramoto_solver import (
    kuramoto_maxcut, kuramoto_maxcut_grid,
    _compute_dtheta, _eval_cut_array, _local_search,
    _fiedler_init, _multi_angle_round
)
from rqaoa import WeightedGraph, brute_force_maxcut


def test_compute_dtheta():
    """Test dat anti-Kuramoto ODE correct berekent."""
    ei = np.array([0], dtype=np.int32)
    ej = np.array([1], dtype=np.int32)
    ew = np.array([1.0])
    theta = np.array([0.0, np.pi])
    dt = _compute_dtheta(theta, ei, ej, ew, 2)
    assert abs(dt[0]) < 1e-10 and abs(dt[1]) < 1e-10, "Anti-fase moet stabiel zijn"
    theta2 = np.array([0.0, 0.0])
    dt2 = _compute_dtheta(theta2, ei, ej, ew, 2)
    assert abs(dt2[0]) < 1e-10, "Gelijke fase moet stabiel zijn"
    theta3 = np.array([0.0, np.pi/2])
    dt3 = _compute_dtheta(theta3, ei, ej, ew, 2)
    assert dt3[0] > 0, "Node 0 gaat omhoog (richting pi)"
    assert dt3[1] < 0, "Node 1 gaat omlaag (weg van node 0)"
    print("PASS: test_compute_dtheta")


def test_eval_cut():
    """Test cut-evaluatie."""
    assign = np.array([0, 1, 0, 1], dtype=np.int32)
    ei = np.array([0, 1, 2, 0], dtype=np.int32)
    ej = np.array([1, 2, 3, 3], dtype=np.int32)
    ew = np.array([1.0, 1.0, 1.0, 1.0])
    cut = _eval_cut_array(assign, ei, ej, ew)
    assert cut == 4.0, "Verwacht 4.0, kreeg %s" % cut
    print("PASS: test_eval_cut")


def test_local_search_improves():
    """Test dat local search nooit verslechtert."""
    g = WeightedGraph.grid(4, 3)
    edges = [(i, j, w) for i, j, w in g.edges()]
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges])
    assign = np.random.randint(0, 2, g.n_nodes).astype(np.int32)
    cut_before = _eval_cut_array(assign, ei, ej, ew)
    assign2, cut_after, n_flips = _local_search(assign.copy(), ei, ej, ew, g.n_nodes)
    assert cut_after >= cut_before, "LS verslechterd: %s -> %s" % (cut_before, cut_after)
    print("PASS: test_local_search_improves (%d -> %d, %d flips)" % (cut_before, cut_after, n_flips))


def test_bipartite_grid_optimal():
    """Test dat Kuramoto optimum vindt op bipartite grids."""
    for Lx, Ly in [(4, 3), (6, 3), (8, 3)]:
        g = WeightedGraph.grid(Lx, Ly)
        edges = [(i, j, w) for i, j, w in g.edges()]
        n = g.n_nodes
        ne = g.n_edges
        if n <= 24:
            exact, _ = brute_force_maxcut(g)
        else:
            exact = ne
        r = kuramoto_maxcut(n, edges, n_restarts=5, max_iter=300, verbose=False)
        assert int(r['best_cut']) == exact, "%dx%d: %s \!= %s" % (Lx, Ly, r['best_cut'], exact)
        print("PASS: test_bipartite_grid_optimal %dx%d: %d/%d" % (Lx, Ly, int(r['best_cut']), ne))


def test_triangulated_grid():
    """Test op niet-bipartiet graaf."""
    g = WeightedGraph()
    edges = []
    Lx, Ly = 4, 3
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            g.add_node(node)
            if x + 1 < Lx:
                g.add_edge(node, (x + 1) * Ly + y, 1.0)
                edges.append((node, (x + 1) * Ly + y, 1.0))
            if y + 1 < Ly:
                g.add_edge(node, x * Ly + y + 1, 1.0)
                edges.append((node, x * Ly + y + 1, 1.0))
            if x + 1 < Lx and y + 1 < Ly:
                g.add_edge(node, (x + 1) * Ly + y + 1, 1.0)
                edges.append((node, (x + 1) * Ly + y + 1, 1.0))
    exact, _ = brute_force_maxcut(g)
    r = kuramoto_maxcut(g.n_nodes, edges, n_restarts=10, verbose=False)
    gap = exact - int(r['best_cut'])
    assert gap <= 2, "4x3 tri: gap=%d te groot" % gap
    print("PASS: test_triangulated_grid 4x3: cut=%d exact=%d gap=%d" % (int(r['best_cut']), exact, gap))


def test_grid_convenience():
    """Test kuramoto_maxcut_grid wrapper."""
    r = kuramoto_maxcut_grid(4, 3, n_restarts=3, verbose=False)
    assert r['best_cut'] == 17.0, "Verwacht 17, kreeg %s" % r['best_cut']
    assert r['ratio'] == 1.0, "Verwacht 1.0, kreeg %s" % r['ratio']
    print("PASS: test_grid_convenience")


if __name__ == '__main__':
    test_compute_dtheta()
    test_eval_cut()
    test_local_search_improves()
    test_bipartite_grid_optimal()
    test_triangulated_grid()
    test_grid_convenience()
    print()
    print("Alle 6 tests PASSED")
