#\!/usr/bin/env python3
"""test_bls_solver.py - Tests for B134 Breakout Local Search"""

import numpy as np
import sys, os, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bls_solver import (
    bls_maxcut, bls_maxcut_grid, _build_adj_arrays, _compute_cut,
    _compute_deltas, _update_deltas_after_flip,
    random_3regular, random_erdos_renyi, load_gset,
)
from pfaffian_oracle import pfaffian_maxcut

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

print('=== Delta computation ===')
# K3: triangle
k3 = [(0,1,1.0), (1,2,1.0), (0,2,1.0)]
adj, wt = _build_adj_arrays(3, k3)
x = np.array([0, 0, 0], dtype=np.int32)
delta = _compute_deltas(x, adj, wt, 3)
check('K3 all-zero: delta = [2,2,2]', np.allclose(delta, [2,2,2]))

x2 = np.array([1, 0, 0], dtype=np.int32)
delta2 = _compute_deltas(x2, adj, wt, 3)
check('K3 [1,0,0]: delta[0]=-2 (all edges cut)', delta2[0] == -2.0)

print('\n=== Delta update after flip ===')
x3 = np.array([0, 0, 0, 0], dtype=np.int32)
k4 = [(i,j,1.0) for i in range(4) for j in range(i+1,4)]
adj4, wt4 = _build_adj_arrays(4, k4)
d3 = _compute_deltas(x3, adj4, wt4, 4)
# Flip node 0
x3[0] = 1
_update_deltas_after_flip(0, x3, adj4, wt4, d3)
# Recompute from scratch and compare
d3_exact = _compute_deltas(x3, adj4, wt4, 4)
check('Delta update matches recompute', np.allclose(d3, d3_exact))

print('\n=== BLS vs exact (small instances) ===')

k5 = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
r_bls = bls_maxcut(5, k5, n_restarts=5, seed=42)
r_exact = pfaffian_maxcut(5, k5)
check(f'K5: BLS={r_bls["best_cut"]:.0f} == exact={r_exact["best_cut"]:.0f}',
      abs(r_bls['best_cut'] - r_exact['best_cut']) < 0.5)

k8 = [(i,j,1.0) for i in range(8) for j in range(i+1,8)]
r_bls = bls_maxcut(8, k8, n_restarts=10, seed=42)
r_exact = pfaffian_maxcut(8, k8)
check(f'K8: BLS={r_bls["best_cut"]:.0f} == exact={r_exact["best_cut"]:.0f}',
      abs(r_bls['best_cut'] - r_exact['best_cut']) < 0.5)

pet = [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)]
r_bls = bls_maxcut(10, pet, n_restarts=10, seed=42)
r_exact = pfaffian_maxcut(10, pet)
check(f'Petersen: BLS={r_bls["best_cut"]:.0f} == exact={r_exact["best_cut"]:.0f}',
      abs(r_bls['best_cut'] - r_exact['best_cut']) < 0.5)

tri_edges = []
for x in range(4):
    for y in range(3):
        node = x * 3 + y
        if x + 1 < 4: tri_edges.append((node, (x+1)*3+y, 1.0))
        if y + 1 < 3: tri_edges.append((node, x*3+y+1, 1.0))
        if x + 1 < 4 and y + 1 < 3: tri_edges.append((node, (x+1)*3+y+1, 1.0))
r_bls = bls_maxcut(12, tri_edges, n_restarts=20, seed=42)
r_exact = pfaffian_maxcut(12, tri_edges)
check(f'Tri 4x3: BLS={r_bls["best_cut"]:.0f} == exact={r_exact["best_cut"]:.0f}',
      abs(r_bls['best_cut'] - r_exact['best_cut']) < 0.5)

print('\n=== Custom initial solution ===')
x_init = np.array([1,0,1,0,1,0,1,0], dtype=np.int32)
r = bls_maxcut(8, k8, n_restarts=3, x_init=x_init, seed=42)
check('Custom init accepted', r['best_cut'] >= 14)

print('\n=== Time limit ===')
nn, edges = random_erdos_renyi(100, p=0.5, seed=99)
t0 = time.time()
r = bls_maxcut(nn, edges, n_restarts=1000, time_limit=1.0, seed=42)
dt = time.time() - t0
check(f'Time limit respected (dt={dt:.2f}s <= 1.5s)', dt < 1.5)
check(f'Found reasonable cut (>0)', r['best_cut'] > 0)

print('\n=== Scaling ===')
nn, edges = random_3regular(500, seed=500)
t0 = time.time()
r = bls_maxcut(nn, edges, n_restarts=5, max_iter=200, seed=42)
dt = time.time() - t0
check(f'3-reg n=500 in reasonable time ({dt:.1f}s < 10s)', dt < 10)
cut_ratio = r['best_cut'] / (len(edges))
check(f'3-reg n=500 ratio > 0.8 (got {cut_ratio:.3f})', cut_ratio > 0.8)

print('\n=== Graph generators ===')
nn, edges = random_3regular(20, seed=42)
check('3-regular: n=20', nn == 20)
check('3-regular: 30 edges', len(edges) == 30)

nn, edges = random_erdos_renyi(50, p=0.5, seed=42)
check('ER: n=50', nn == 50)
check('ER: ~600 edges', 400 < len(edges) < 800)

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)

