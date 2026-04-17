#!/usr/bin/env python3
"""B99v2 on large instances (tree+greedy only, no BLS)."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from gset_loader import load_gset
from feedback_edge_solver import (
    _random_spanning_tree, solve_tree_maxcut,
    _greedy_refine, _eval_full_cut, UnionFind
)
from collections import defaultdict
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'G77'
bks_map = {'G77': 9926, 'G81': 14030}
bks = bks_map.get(name, 0)

g, bks_val, info = load_gset(name)
n = g.n_nodes
edges = [(i, j, w) for i, j, w in g.edges()]
print('%s: n=%d, m=%d, BKS=%d' % (name, n, len(edges), bks_val), flush=True)

rng = np.random.default_rng(42)
best_cut = -np.inf
best_assign = None

t0 = time.time()
for t_idx in range(15):
    if time.time() - t0 > 20:
        break
    tree, fb, adj = _random_spanning_tree(n, edges, rng)
    fb_zeros = np.zeros(len(fb), dtype=np.int8)
    ta, tc = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)
    ra, rc, nf = _greedy_refine(n, edges, ta, max_passes=15)
    if rc > best_cut:
        best_cut = rc
        best_assign = ra
        gap = 100 * (bks_val - best_cut) / bks_val
        print('  Tree %d: refined=%.0f, gap=%.1f%%, %d flips (%.1fs) *BEST*' % (
            t_idx+1, rc, gap, nf, time.time()-t0), flush=True)
    else:
        print('  Tree %d: refined=%.0f (%.1fs)' % (
            t_idx+1, rc, time.time()-t0), flush=True)

gap = 100 * (bks_val - best_cut) / bks_val
print('\n%s FINAL: cut=%.0f, gap=%.1f%%, time=%.1fs' % (
    name, best_cut, gap, time.time()-t0), flush=True)
