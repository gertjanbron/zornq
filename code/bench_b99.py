#!/usr/bin/env python3
"""Quick benchmark for B99 feedback-edge solver on Gset."""
import sys, time
sys.path.insert(0, '.')
from gset_loader import load_gset
from feedback_edge_solver import max_spanning_tree, solve_tree_maxcut
import numpy as np

def run_g70_tree_only():
    g, bks, info = load_gset('G70')
    n = g.n_nodes
    edges = [(i,j,w) for i,j,w in g.edges()]
    print('G70: n=%d, m=%d, BKS=%d' % (n, len(edges), bks))

    t0 = time.time()
    tree, fb, adj = max_spanning_tree(n, edges)
    print('Tree decomp: %.3fs, tree=%d, k=%d' % (time.time()-t0, len(tree), len(fb)))

    t1 = time.time()
    fb_zeros = np.zeros(len(fb), dtype=np.int8)
    tree_assign, tree_cut = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)
    print('Tree solve: %.3fs, tree_cut=%.1f' % (time.time()-t1, tree_cut))

    full_cut = 0.0
    for u,v,w in edges:
        a = tree_assign.get(int(u), 0)
        b = tree_assign.get(int(v), 0)
        if a != b:
            full_cut += w
    print('Full graph cut: %.1f, gap=%.1f%%' % (full_cut, 100*(bks-full_cut)/bks))
    return full_cut

def run_tree_plus_bls(name, bks_val, tlimit=10):
    """Tree warm-start + BLS, step by step."""
    from bls_solver import bls_maxcut
    g, bks, info = load_gset(name)
    n = g.n_nodes
    edges = [(i,j,w) for i,j,w in g.edges()]
    print('%s: n=%d, m=%d, BKS=%d' % (name, n, len(edges), bks))

    # Step 1: Tree decomposition + solve
    t0 = time.time()
    tree, fb, adj = max_spanning_tree(n, edges)
    fb_zeros = np.zeros(len(fb), dtype=np.int8)
    tree_assign, tree_cut = solve_tree_maxcut(n, adj, tree, fb, fb_zeros)

    # Eval on full graph
    full_cut = 0.0
    for u,v,w in edges:
        if tree_assign.get(int(u), 0) != tree_assign.get(int(v), 0):
            full_cut += w
    t_tree = time.time() - t0
    gap_tree = 100*(bks-full_cut)/bks
    print('  Tree: cut=%.0f, gap=%.1f%%, k=%d, time=%.2fs' % (full_cut, gap_tree, len(fb), t_tree))

    # Step 2: BLS polish with warm start
    warm = np.zeros(n, dtype=np.int32)
    for node, label in tree_assign.items():
        if node < n:
            warm[node] = label

    t1 = time.time()
    bls_result = bls_maxcut(n, edges, time_limit=tlimit, seed=42, x_init=warm)
    t_bls = time.time() - t1
    bls_cut = bls_result['best_cut']
    gap_bls = 100*(bks-bls_cut)/bks
    print('  BLS:  cut=%.0f, gap=%.1f%%, time=%.2fs' % (bls_cut, gap_bls, t_bls))

    best = max(full_cut, bls_cut)
    gap = 100*(bks-best)/bks
    print('  => Best: %.0f, gap=%.1f%%, total=%.1fs' % (best, gap, time.time()-t0))
    return best

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'tree'
    if mode == 'tree':
        run_g70_tree_only()
    elif mode == 'full':
        name = sys.argv[2] if len(sys.argv) > 2 else 'G70'
        tlimit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        bks_map = {'G60': 14176, 'G62': 4868, 'G63': 27045, 'G70': 9541,
                   'G65': 5558, 'G67': 6940, 'G77': 9926, 'G81': 14030}
        run_tree_plus_bls(name, bks_map.get(name, 0), tlimit=tlimit)
