#!/usr/bin/env python3
"""test_multiscale_maxcut.py - Tests for B149 multiscale MaxCut bridge."""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiscale_maxcut import (
    build_multiscale_clusters,
    contract_graph,
    locality_preserving_order,
    multiscale_pa_maxcut,
)


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')


print('=== B149 Multiscale MaxCut Tests ===\n')

cycle_edges = [(i, (i + 1) % 12, 1.0) for i in range(12)]
order, info = locality_preserving_order(12, cycle_edges)
check('Order covers all nodes', sorted(order) == list(range(12)))
check('Ordering metadata present', info['ordering_method'] == 'component_diameter_sweep')

clusters, cluster_of, order_info = build_multiscale_clusters(12, cycle_edges, n_clusters=3)
check('Three clusters built', len(clusters) == 3)
check('Cluster map covers all nodes', np.all(cluster_of >= 0))
check('Cluster sizes sum to n', sum(len(c) for c in clusters) == 12)
check('Cluster builder keeps ordering info',
      order_info['ordering_method'] == 'component_diameter_sweep')

coarse_edges, meta = contract_graph(12, cycle_edges, clusters, cluster_of)
check('Coarse graph has edges', len(coarse_edges) > 0)
check('Boundary metadata length matches clusters', len(meta['boundary_weight']) == 3)

k5_edges = [(i, j, 1.0) for i in range(5) for j in range(i + 1, 5)]
r = multiscale_pa_maxcut(5, k5_edges, seed=42, time_limit=2.0)
check('Small-graph fallback solves K5 well', abs(r['best_cut'] - 6.0) < 0.5)
check('Small-graph fallback note set', 'multiscale-fallback-small' in r['solver_note'])

grid_edges = []
Lx, Ly = 4, 4
for x in range(Lx):
    for y in range(Ly):
        v = x * Ly + y
        if x + 1 < Lx:
            grid_edges.append((v, (x + 1) * Ly + y, 1.0))
        if y + 1 < Ly:
            grid_edges.append((v, x * Ly + y + 1, 1.0))
r = multiscale_pa_maxcut(16, grid_edges, seed=42, time_limit=2.0)
check('Grid instance returns positive cut', r['best_cut'] > 0)

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)
