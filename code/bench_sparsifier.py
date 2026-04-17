#!/usr/bin/env python3
"""
Benchmark: B118 Cut-Preserving Sparsifier

Resultaten (15 april 2026):
  - Dense pm1 grafen: sparsifier + B99v2 wint op n=200 (+36-38 voordeel)
  - Sparse Gset instances: geen reductie (al sparse, degree ~4-5)
  - Hybrid QAOA: lightcone te groot op dense grafen, sparsifier helpt niet genoeg

Usage: python bench_sparsifier.py
"""
import sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from cut_sparsifier import sparsify
from feedback_edge_solver import _multi_tree_ensemble, _greedy_refine


def make_dense_pm1(n, density=0.3, seed=42):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                edges.append((i, j, rng.choice([-1.0, 1.0])))
    return n, edges


if __name__ == '__main__':
    print("B118 Sparsifier: dense pm1 grafen")
    print("=" * 75)
    fmt_h = "%12s %5s %6s %6s %8s %8s %8s %6s"
    print(fmt_h % ("Graph", "n", "m_orig", "m_spar", "Direct", "Sparse", "Diff", "time"))
    print("-" * 75)

    for n_val, dens in [(100, 0.3), (100, 0.5), (200, 0.3), (200, 0.5)]:
        n, edges = make_dense_pm1(n_val, density=dens, seed=42)
        label = "n=%d d=%.1f" % (n_val, dens)

        t0 = time.time()
        direct_cut, direct_assign = _multi_tree_ensemble(
            n, edges, time_limit=3, seed=42, n_trees=8)

        t1 = time.time()
        sparse_edges, sp_info = sparsify(n, edges, epsilon=0.5, seed=42)
        sp_cut, sp_assign = _multi_tree_ensemble(
            n, sparse_edges, time_limit=3, seed=42, n_trees=8)
        _, sp_orig_cut, _ = _greedy_refine(n, edges, sp_assign, max_passes=15)
        sparse_time = time.time() - t1

        diff = sp_orig_cut - direct_cut
        fmt_r = "%12s %5d %6d %6d %8.1f %8.1f %+8.1f %5.1fs"
        print(fmt_r % (label, n, len(edges), len(sparse_edges),
                       direct_cut, sp_orig_cut, diff, sparse_time), flush=True)

    print("=" * 75)
