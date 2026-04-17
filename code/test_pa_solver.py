#\!/usr/bin/env python3
"""test_pa_solver.py - Tests for B135 Population Annealing"""

import numpy as np
import sys, os, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pa_solver import (
    pa_maxcut, pa_maxcut_grid, _build_adj_arrays,
    _compute_cut_batch, _compute_cut_single,
    _geometric_schedule, _linear_schedule,
)
from pfaffian_oracle import pfaffian_maxcut
from bls_solver import random_3regular

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

print('=== Cut computation ===')
k3 = [(0,1,1.0), (1,2,1.0), (0,2,1.0)]
ei = np.array([0,1,0], dtype=np.int32)
ej = np.array([1,2,2], dtype=np.int32)
ew = np.array([1.0,1.0,1.0])
pop = np.array([[1,0,0], [0,0,0], [1,0,1]], dtype=np.int8)
cuts = _compute_cut_batch(pop, ei, ej, ew)
check('K3 [1,0,0] cut=2', abs(cuts[0] - 2.0) < 1e-9)
check('K3 [0,0,0] cut=0', abs(cuts[1] - 0.0) < 1e-9)
check('K3 [1,0,1] cut=2', abs(cuts[2] - 2.0) < 1e-9)

print('\n=== Temperature schedule ===')
betas = _geometric_schedule(0.1, 5.0, 10)
check('Geometric: 10 steps', len(betas) == 10)
check('Geometric: starts at 0.1', abs(betas[0] - 0.1) < 1e-6)
check('Geometric: ends at 5.0', abs(betas[-1] - 5.0) < 1e-6)
check('Geometric: monotone increasing', all(betas[i] < betas[i+1] for i in range(len(betas)-1)))

betas_lin = _linear_schedule(0.0, 3.0, 5)
check('Linear: 5 steps', len(betas_lin) == 5)

print('\n=== PA vs Exact ===')

k5 = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
r = pa_maxcut(5, k5, n_replicas=50, n_temps=30, seed=42)
exact = pfaffian_maxcut(5, k5)
check(f'K5: PA={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

k8 = [(i,j,1.0) for i in range(8) for j in range(i+1,8)]
r = pa_maxcut(8, k8, n_replicas=100, n_temps=40, seed=42)
exact = pfaffian_maxcut(8, k8)
check(f'K8: PA={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

pet = [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)]
r = pa_maxcut(10, pet, n_replicas=100, n_temps=40, seed=42)
exact = pfaffian_maxcut(10, pet)
check(f'Petersen: PA={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

r = pa_maxcut_grid(4, 2, n_replicas=50, n_temps=30, seed=42)
check(f'Grid 4x2: PA={r["best_cut"]:.0f} == 10', abs(r['best_cut'] - 10.0) < 0.5)

r = pa_maxcut_grid(4, 3, triangular=True, n_replicas=100, n_temps=40, seed=42)
exact = pfaffian_maxcut(12, [(e[0],e[1],e[2]) for e in r.get("_edges", [])] if "_edges" in r else [])
check(f'Tri 4x3: PA={r["best_cut"]:.0f} >= 16', r['best_cut'] >= 16)

print('\n=== API structure ===')
r = pa_maxcut(5, k5, n_replicas=20, n_temps=10, seed=42)
check('Has best_cut', 'best_cut' in r)
check('Has assignment', 'assignment' in r)
check('Has history', 'history' in r)
check('Has time_s', 'time_s' in r)
check('History length == n_temps', len(r['history']) == 10)
check('Assignment has all nodes', len(r['assignment']) == 5)

print('\n=== Time limit ===')
nn, edges = random_3regular(200, seed=200)
t0 = time.time()
r = pa_maxcut(nn, edges, n_replicas=200, n_temps=200, time_limit=1.0, seed=42)
dt = time.time() - t0
check(f'Time limit respected ({dt:.2f}s < 1.5s)', dt < 1.5)
check('Found reasonable cut', r['best_cut'] > 200)

print('\n=== Scaling ===')
nn, edges = random_3regular(500, seed=500)
t0 = time.time()
r = pa_maxcut(nn, edges, n_replicas=100, n_temps=40, seed=42)
dt = time.time() - t0
check(f'3-reg n=500 in <10s ({dt:.1f}s)', dt < 10)
ratio = r['best_cut'] / len(edges)
check(f'3-reg n=500 ratio > 0.85 ({ratio:.3f})', ratio > 0.85)

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)

