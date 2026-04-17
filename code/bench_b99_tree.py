#!/usr/bin/env python3
"""Tree-only benchmark for B99 (no BLS polish)."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from gset_loader import load_gset, GSET_BKS
from feedback_edge_solver import max_spanning_tree, solve_tree_maxcut
import numpy as np

targets = ['G60', 'G62', 'G63', 'G65', 'G67', 'G70', 'G77', 'G81']

print("B99 Tree-Only Benchmark (no BLS polish)")
print("-" * 70)

for name in targets:
    g, bks, info = load_gset(name)
    n = g.n_nodes
    edges = [(i, j, w) for i, j, w in g.edges()]

    t0 = time.time()
    tree, fb, adj = max_spanning_tree(n, edges)
    k = len(fb)

    fb_zeros = np.zeros(k, dtype=np.int8)
    ta, tc = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)

    full_cut = 0.0
    for u, v, w in edges:
        if ta.get(int(u), 0) != ta.get(int(v), 0):
            full_cut += w
    elapsed = time.time() - t0
    gap = 100 * (bks - full_cut) / bks

    neg = sum(1 for _, _, w in edges if w < 0)
    sign = "+-1" if neg > 0 else "+1"

    print("%s | n=%5d | m=%5d | k=%5d | sign=%s | tree=%.0f | gap=%.1f%% | %.3fs" % (
        name, n, len(edges), k, sign, full_cut, gap, elapsed), flush=True)

print("-" * 70)
