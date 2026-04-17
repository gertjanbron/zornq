#!/usr/bin/env python3
"""Benchmark B99 v2 (multi-tree ensemble + greedy) on Gset."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from gset_loader import load_gset
from feedback_edge_solver import (
    max_spanning_tree, solve_tree_maxcut,
    _greedy_refine, _multi_tree_ensemble, _eval_full_cut
)
import numpy as np

targets = [
    ('G60', 14176), ('G62', 4868), ('G63', 27045),
    ('G65', 5558), ('G67', 6940), ('G70', 9541),
]

print("B99 v2: Multi-Tree Ensemble + Greedy Refinement")
print("=" * 75)
hdr = "%5s %6s %6s %5s %8s %8s %8s %6s"
print(hdr % ("Name", "n", "m", "k", "TreeOnly", "Greedy", "B99v2", "time"))
print("-" * 75)

for name, bks in targets:
    g, bks_val, info = load_gset(name)
    n = g.n_nodes
    edges = [(i, j, w) for i, j, w in g.edges()]

    t0 = time.time()

    # Single tree baseline
    tree, fb, adj = max_spanning_tree(n, edges)
    k = len(fb)
    fb_zeros = np.zeros(k, dtype=np.int8)
    ta, tc = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)
    tree_cut = _eval_full_cut(n, edges, ta)
    gap_tree = 100 * (bks_val - tree_cut) / bks_val

    # Single tree + greedy
    greedy_assign, greedy_cut, nf = _greedy_refine(n, edges, ta, max_passes=20)
    gap_greedy = 100 * (bks_val - greedy_cut) / bks_val

    # Multi-tree ensemble (no BLS, just trees + greedy)
    best_cut, best_assign = _multi_tree_ensemble(
        n, edges, time_limit=4, seed=42, n_trees=10, verbose=False)
    gap_v2 = 100 * (bks_val - best_cut) / bks_val
    elapsed = time.time() - t0

    row = "%5s %6d %6d %5d %7.1f%% %7.1f%% %7.1f%% %5.1fs"
    print(row % (name, n, len(edges), k,
                 gap_tree, gap_greedy, gap_v2, elapsed), flush=True)

print("=" * 75)
