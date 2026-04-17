#!/usr/bin/env python3
"""Quick G70 benchmark."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

print('loading...', flush=True)
from gset_loader import load_gset
from bls_solver import bls_maxcut
from feedback_edge_solver import max_spanning_tree, solve_tree_maxcut
import numpy as np

g, bks, info = load_gset('G70')
n = g.n_nodes
edges = [(i,j,w) for i,j,w in g.edges()]
print('G70: n=%d, m=%d, BKS=%d' % (n, len(edges), bks), flush=True)

t0 = time.time()
tree, fb, adj = max_spanning_tree(n, edges)
print('tree=%d, k=%d (%.3fs)' % (len(tree), len(fb), time.time()-t0), flush=True)

t1 = time.time()
fb_zeros = np.zeros(len(fb), dtype=np.int8)
ta, tc = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)
print('tree solve: %.3fs, tree_cut=%.1f' % (time.time()-t1, tc), flush=True)

full_cut = 0.0
for u, v, w in edges:
    au = ta.get(int(u), 0)
    av = ta.get(int(v), 0)
    if au != av:
        full_cut += w
gap_tree = 100 * (bks - full_cut) / bks
print('full graph with tree assign: %.1f, gap=%.1f%%' % (full_cut, gap_tree), flush=True)

warm = np.zeros(n, dtype=np.int32)
for node, label in ta.items():
    if node < n:
        warm[node] = label

print('starting BLS (5s)...', flush=True)
t2 = time.time()
r = bls_maxcut(n, edges, time_limit=5, seed=42, x_init=warm)
bls_cut = r['best_cut']
gap_bls = 100 * (bks - bls_cut) / bks
print('BLS: cut=%.1f, gap=%.1f%%, time=%.1fs' % (bls_cut, gap_bls, time.time()-t2), flush=True)
print('TOTAL: %.1fs' % (time.time()-t0), flush=True)
