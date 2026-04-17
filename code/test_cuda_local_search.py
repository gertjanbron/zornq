#!/usr/bin/env python3
"""test_cuda_local_search.py - Tests for B136 CUDA Local Search Kernel

Tests correctness of GPU kernels by comparing against CPU implementations.
Falls back to CPU-only validation if no GPU is present (sandbox mode).
"""

import numpy as np
import sys, os, time, types
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuda_local_search import (
    _CUDA_RUNTIME_DIRS, _build_csr, _run_cpu_pa_light, _should_use_cpu_pa,
    _should_use_dense2k_pa_only_route,
    _should_use_dense_dsbm_direct_route,
    _high_degree_sparse_dsbm_config,
    _should_use_medium_sparse_probe, _should_use_sparse_dsbm_route,
    _strong_sparse_dsbm_schedule,
    gpu_available, maxcut_bls, maxcut_pa, maxcut_pa_sparse_hybrid,
)
from bls_solver import bls_maxcut, random_3regular, random_erdos_renyi
from pa_solver import pa_maxcut
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

print(f'=== B136 CUDA Local Search Tests ===')
print(f'GPU available: {gpu_available()}')
print()

# ============================================================
# Runtime dir setup
# ============================================================
print('=== CUDA runtime dirs ===')
check('CUDA runtime dirs configured', _CUDA_RUNTIME_DIRS is not None)
if _CUDA_RUNTIME_DIRS is not None:
    check('CUDA temp dir exists', os.path.isdir(_CUDA_RUNTIME_DIRS['temp_root']))
    check('CuPy cache dir exists', os.path.isdir(_CUDA_RUNTIME_DIRS['cache_root']))
check('PA dispatch prefers CPU on large short-budget workload',
      _should_use_cpu_pa(800, [(0, 1, 1.0)] * 1600,
                         n_replicas=100, n_temps=60,
                         n_sweeps=5, time_limit=10.0))
check('Sparse dSBM route detected on very large sparse workload',
      _should_use_sparse_dsbm_route(10000, [(0, 1, 1.0)] * 20000,
                                    time_limit=10.0))
check('Medium sparse probe detected on 3k-node low-degree workload',
      _should_use_medium_sparse_probe(3000, [(0, 1, 1.0)] * 6000,
                                      time_limit=10.0))
check('Dense dSBM direct route detected on G48-like workload',
      _should_use_dense_dsbm_direct_route(3000, [(0, 1, 1.0)] * 6000,
                                          time_limit=10.0))
check('Dense2000 PA-only route detected on 2k-node workload',
      _should_use_dense2k_pa_only_route(2000, [(0, 1, 1.0)] * 20000,
                                        time_limit=10.0))
check('High-degree sparse config retunes schedule at 10s',
      _high_degree_sparse_dsbm_config(5000, [(0, 1, 1.0)] * 29570, 10.0)
      == (64, 3000, 2.0))

original_pa_solver = sys.modules.get('pa_solver')
captured = []
stub_module = types.ModuleType('pa_solver')

def _fake_pa_maxcut(n_nodes, edges, **kwargs):
    captured.append((n_nodes, kwargs))
    return {
        'best_cut': 1.0,
        'assignment': {0: 0, 1: 1},
        'time_s': 0.0,
        'history': [],
        'device': 'cpu',
    }

stub_module.pa_maxcut = _fake_pa_maxcut
sys.modules['pa_solver'] = stub_module
try:
    r_dense2k = _run_cpu_pa_light(2000, [(0, 1, 1.0)], seed=7, time_limit=1.0)
    r_large = _run_cpu_pa_light(3000, [(0, 1, 1.0)], seed=7, time_limit=1.0)
finally:
    if original_pa_solver is not None:
        sys.modules['pa_solver'] = original_pa_solver
    else:
        del sys.modules['pa_solver']

check('CPU PA light uses 150 replicas on dense2000 band',
      captured[0][1].get('n_replicas') == 150)
check('CPU PA light uses 100 replicas above dense2000 band',
      captured[1][1].get('n_replicas') == 100)
check('CPU PA light note records replica count',
      r_dense2k['solver_note'] == 'cpu-pa-light-probe(replicas=150)'
      and r_large['solver_note'] == 'cpu-pa-light-probe(replicas=100)')
check('Strong sparse dSBM schedule uses 96 restarts at 10s',
      _strong_sparse_dsbm_schedule(10.0) == (96, 3000))

# ============================================================
# CSR construction tests
# ============================================================
print('=== CSR construction ===')

k4 = [(0,1,1.0), (0,2,1.0), (0,3,1.0), (1,2,1.0), (1,3,1.0), (2,3,1.0)]
rp, ci, w = _build_csr(4, k4)
check('K4 row_ptr length', len(rp) == 5)
check('K4 total entries = 12', rp[4] == 12)
check('K4 degree all 3', all(rp[v+1] - rp[v] == 3 for v in range(4)))
check('K4 weights all 1.0', np.allclose(w, 1.0))

# Weighted graph
wg = [(0,1,2.5), (1,2,3.0)]
rp, ci, w = _build_csr(3, wg)
check('Weighted CSR row_ptr', list(rp) == [0, 1, 3, 4])
check('Weighted CSR has correct weights', abs(w[0] - 2.5) < 1e-5)

# Empty graph
rp, ci, w = _build_csr(3, [])
check('Empty graph CSR', list(rp) == [0, 0, 0, 0])

# ============================================================
# Unified API: BLS correctness
# ============================================================
print('\n=== BLS via unified API ===')

# K5: exact = 6 (bipartite: impossible, but max cut of K5 = 6)
k5 = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
r = maxcut_bls(5, k5, n_restarts=3, seed=42)
check(f'K5 BLS cut={r["best_cut"]:.0f} == 6', abs(r['best_cut'] - 6.0) < 0.5)

# K8
k8 = [(i,j,1.0) for i in range(8) for j in range(i+1,8)]
exact = pfaffian_maxcut(8, k8)
r = maxcut_bls(8, k8, n_restarts=5, seed=42)
check(f'K8 BLS={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

# Petersen graph
pet = [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)]
exact = pfaffian_maxcut(10, pet)
r = maxcut_bls(10, pet, n_restarts=5, seed=42)
check(f'Petersen BLS={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

# 3-regular n=50
nn, edges = random_3regular(50, seed=350)
r_ref = bls_maxcut(nn, edges, n_restarts=10, seed=42)
r_uni = maxcut_bls(nn, edges, n_restarts=10, seed=42)
ratio = r_uni['best_cut'] / max(r_ref['best_cut'], 1)
check(f'3-reg n=50 unified vs CPU ratio={ratio:.3f} >= 0.95', ratio >= 0.95)

# ============================================================
# Unified API: PA correctness
# ============================================================
print('\n=== PA via unified API ===')

r = maxcut_pa(5, k5, n_replicas=50, n_temps=30, seed=42)
check(f'K5 PA cut={r["best_cut"]:.0f} == 6', abs(r['best_cut'] - 6.0) < 0.5)

r = maxcut_pa(8, k8, n_replicas=100, n_temps=40, seed=42)
exact = pfaffian_maxcut(8, k8)
check(f'K8 PA={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

r = maxcut_pa(10, pet, n_replicas=100, n_temps=40, seed=42)
exact = pfaffian_maxcut(10, pet)
check(f'Petersen PA={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

r = maxcut_pa_sparse_hybrid(10, pet, n_replicas=100, n_temps=40, seed=42)
check(f'Petersen PA sparse hybrid={r["best_cut"]:.0f} == exact={exact["best_cut"]:.0f}',
      abs(r['best_cut'] - exact['best_cut']) < 0.5)

# API structure
check('PA has history', 'history' in r)
check('PA has time_s', 'time_s' in r)
check('PA has assignment', 'assignment' in r)

# ============================================================
# Scaling tests
# ============================================================
print('\n=== Scaling ===')

nn, edges = random_3regular(200, seed=1400)
t0 = time.time()
r = maxcut_bls(nn, edges, n_restarts=5, max_iter=500, seed=42)
dt = time.time() - t0
check(f'BLS n=200 in <5s ({dt:.2f}s)', dt < 5.0)
check(f'BLS n=200 cut > 250 ({r["best_cut"]:.0f})', r['best_cut'] > 250)

nn, edges = random_3regular(500, seed=3500)
t0 = time.time()
r = maxcut_bls(nn, edges, n_restarts=5, max_iter=500, seed=42)
dt = time.time() - t0
check(f'BLS n=500 in <10s ({dt:.2f}s)', dt < 10.0)
ratio = r['best_cut'] / len(edges)
check(f'BLS n=500 ratio > 0.84 ({ratio:.3f})', ratio > 0.84)

nn, edges = random_3regular(500, seed=3500)
t0 = time.time()
r = maxcut_pa(nn, edges, n_replicas=100, n_temps=30, seed=42)
dt = time.time() - t0
check(f'PA n=500 in <15s ({dt:.2f}s)', dt < 15.0)
ratio = r['best_cut'] / len(edges)
check(f'PA n=500 ratio > 0.85 ({ratio:.3f})', ratio > 0.85)

# ============================================================
# Time limit
# ============================================================
print('\n=== Time limit ===')

nn, edges = random_3regular(200, seed=1400)
t0 = time.time()
r = maxcut_bls(nn, edges, n_restarts=100, max_iter=10000, time_limit=1.0, seed=42)
dt = time.time() - t0
check(f'BLS time_limit respected ({dt:.2f}s < 1.5s)', dt < 1.5)

t0 = time.time()
r = maxcut_pa(nn, edges, n_replicas=200, n_temps=200, time_limit=1.0, seed=42)
dt = time.time() - t0
check(f'PA time_limit respected ({dt:.2f}s < 1.5s)', dt < 1.5)

# ============================================================
# Dense graphs
# ============================================================
print('\n=== Dense ER ===')

nn, edges = random_erdos_renyi(50, p=0.5, seed=250)
r_bls = maxcut_bls(nn, edges, n_restarts=5, seed=42)
r_pa = maxcut_pa(nn, edges, n_replicas=100, n_temps=30, seed=42)
check(f'ER n=50 BLS reasonable ({r_bls["best_cut"]:.0f})', r_bls['best_cut'] > 300)
check(f'ER n=50 PA reasonable ({r_pa["best_cut"]:.0f})', r_pa['best_cut'] > 300)

# ============================================================
# GPU-specific tests (skip if no GPU)
# ============================================================
if gpu_available():
    print('\n=== GPU-specific tests ===')
    import cupy as cp
    from cuda_local_search import (
        _gpu_compute_deltas, _gpu_compute_cut, _gpu_find_best
    )

    # Delta computation on GPU vs CPU
    nn = 20
    edges_20 = [(i,j,1.0) for i in range(nn) for j in range(i+1, nn)]
    rp, ci, w = _build_csr(nn, edges_20)
    d_rp = cp.asarray(rp)
    d_ci = cp.asarray(ci)
    d_w = cp.asarray(w)

    x = np.random.default_rng(42).integers(0, 2, size=nn).astype(np.int32)
    d_x = cp.asarray(x)
    d_delta = cp.zeros(nn, dtype=cp.float32)
    _gpu_compute_deltas(d_x, d_rp, d_ci, d_w, d_delta, nn)
    gpu_delta = d_delta.get()

    # CPU reference
    from bls_solver import _build_adj_arrays, _compute_deltas
    adj, wt = _build_adj_arrays(nn, edges_20)
    cpu_delta = _compute_deltas(x, adj, wt, nn).astype(np.float32)

    check('GPU delta matches CPU', np.allclose(gpu_delta, cpu_delta, atol=1e-4))

    # Cut computation
    ei = np.array([e[0] for e in edges_20], dtype=np.int32)
    ej = np.array([e[1] for e in edges_20], dtype=np.int32)
    ew = np.ones(len(edges_20), dtype=np.float32)
    gpu_cut = _gpu_compute_cut(d_x, cp.asarray(ei), cp.asarray(ej), cp.asarray(ew), len(edges_20))
    cpu_cut = sum(1 for i,j,_ in edges_20 if x[i] != x[j])
    check(f'GPU cut={gpu_cut:.0f} matches CPU cut={cpu_cut}', abs(gpu_cut - cpu_cut) < 0.5)

    print(f'\n  Device: {cp.cuda.Device(0).id}')
    mem = cp.cuda.Device(0).mem_info
    print(f'  VRAM: {mem[0]/1e9:.1f}/{mem[1]/1e9:.1f} GB free/total')
else:
    print('\n=== GPU not available, CPU fallback tested ===')

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)
