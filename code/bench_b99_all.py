#!/usr/bin/env python3
"""Benchmark B99 on one Gset instance (pass name as argv[1])."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from gset_loader import load_gset
from bls_solver import bls_maxcut
from feedback_edge_solver import max_spanning_tree, solve_tree_maxcut
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'G60'

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
t_tree = time.time() - t0

warm = np.zeros(n, dtype=np.int32)
for nd, label in ta.items():
    if nd < n:
        warm[nd] = label

t1 = time.time()
tlimit = 5 if n <= 7000 else 2
r = bls_maxcut(n, edges, time_limit=tlimit, seed=42, x_init=warm)
bls_cut = r['best_cut']
t_bls = time.time() - t1

best = max(full_cut, bls_cut)
gap = 100 * (bks - best) / bks
gap_tree = 100 * (bks - full_cut) / bks
total = time.time() - t0

msg = "%s | n=%d | k=%d | tree=%.0f(%.1f%%) | B99=%.0f(%.1f%%) | %.1fs"
print(msg % (name, n, k, full_cut, gap_tree, best, gap, total), flush=True)
