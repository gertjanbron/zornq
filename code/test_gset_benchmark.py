#!/usr/bin/env python3
"""test_gset_benchmark.py - Tests for B137 Gset Benchmark"""

import numpy as np
import sys, os, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gset_benchmark import (
    SOLVERS, run_single_benchmark, run_benchmark, save_json_report,
    load_builtin_instances, load_synthetic_instances,
    generate_synthetic_gset, infer_route_metadata, summarize_routebook,
    run_benchmark_checkpointed,
)
from evidence_capsule import verify_benchmark_capsule
from gset_loader import GSET_BKS
from quantum_inspired_baselines import run_dsbm_maxcut

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

print('=== B137 Gset Benchmark Tests ===\n')

# ============================================================
# Solver registry
# ============================================================
print('=== Solver registry ===')
check('bls registered', 'bls' in SOLVERS)
check('bls_heavy registered', 'bls_heavy' in SOLVERS)
check('pa registered', 'pa' in SOLVERS)
check('pa_heavy registered', 'pa_heavy' in SOLVERS)
check('cuda_bls registered', 'cuda_bls' in SOLVERS)
check('cuda_pa registered', 'cuda_pa' in SOLVERS)
check('pa_sparse_hybrid registered', 'pa_sparse_hybrid' in SOLVERS)
check('multiscale_pa registered', 'multiscale_pa' in SOLVERS)
check('combined registered', 'combined' in SOLVERS)
check('simcim registered', 'simcim' in SOLVERS)
check('dsbm registered', 'dsbm' in SOLVERS)
check('At least 11 solvers', len(SOLVERS) >= 11)

# ============================================================
# BKS database
# ============================================================
print('\n=== BKS database ===')
check('G1 in BKS', 'G1' in GSET_BKS)
check('G67 in BKS', 'G67' in GSET_BKS)
check('G14 BKS=3064', GSET_BKS['G14'][2] == 3064)
check('G22 BKS=13359', GSET_BKS['G22'][2] == 13359)
check('BKS has 40+ entries', len(GSET_BKS) >= 40)

# ============================================================
# Synthetic generation
# ============================================================
print('\n=== Synthetic generation ===')
n, edges, bks = generate_synthetic_gset('G14', seed=42)
check('G14 synthetic n=800', n == 800)
check('G14 synthetic has edges', len(edges) > 0)
check('G14 synthetic BKS=3064', bks == 3064)

n, edges, bks = generate_synthetic_gset('G43', seed=42)
check('G43 synthetic n=1000', n == 1000)

n, edges, bks = generate_synthetic_gset('NONEXISTENT', seed=42)
check('Nonexistent returns None', n is None)

# ============================================================
# Instance loading
# ============================================================
print('\n=== Instance loading ===')
builtin = load_builtin_instances()
check('Builtin has instances', len(builtin) > 0)
check('Builtin has K5', any(i['name'] == 'K5' for i in builtin))
check('Builtin has grids', any('grid' in i['name'] for i in builtin))
check('Builtin has 3-regular', any('3reg' in i['name'] for i in builtin))

# Check instance structure
inst = builtin[0]
check('Instance has name', 'name' in inst)
check('Instance has n_nodes', 'n_nodes' in inst)
check('Instance has edges', 'edges' in inst)
check('Instance has bks', 'bks' in inst)

print('\n=== Route metadata ===')
route = infer_route_metadata(18, [
    (0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0),
    (0, 3, 1.0), (1, 4, 1.0), (2, 5, 1.0),
    (3, 6, 1.0), (4, 7, 1.0), (5, 8, 1.0),
    (6, 7, 1.0), (7, 8, 1.0),
])
check('Route metadata has regime', bool(route['regime']))
check('Route metadata has first tool', bool(route['first_tool']))
check('Route metadata width proxy nonnegative', route['width_proxy'] >= 0)

# ============================================================
# Single benchmark run
# ============================================================
print('\n=== Single benchmark ===')
k5_edges = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
r = run_single_benchmark(5, k5_edges, 'bls', seed=42, time_limit=5.0)
check('BLS K5 cut=6', abs(r['best_cut'] - 6.0) < 0.5)
check('BLS K5 no error', r['error'] is None)
check('BLS K5 has time', r['time_s'] > 0)

r = run_single_benchmark(5, k5_edges, 'pa', seed=42, time_limit=5.0)
check('PA K5 cut=6', abs(r['best_cut'] - 6.0) < 0.5)

r = run_single_benchmark(5, k5_edges, 'combined', seed=42, time_limit=5.0)
check('Combined K5 cut=6', abs(r['best_cut'] - 6.0) < 0.5)

r = run_single_benchmark(5, k5_edges, 'simcim', seed=42, time_limit=2.0)
check('SimCIM K5 cut=6', abs(r['best_cut'] - 6.0) < 0.5)
check('SimCIM K5 no error', r['error'] is None)

r = run_single_benchmark(5, k5_edges, 'dsbm', seed=42, time_limit=2.0)
check('dSBM K5 cut=6', abs(r['best_cut'] - 6.0) < 0.5)
check('dSBM K5 no error', r['error'] is None)

# Sparse backend sanity: voorbij de oude dense limiet
big_cycle_n = 4001
big_cycle_edges = [(i, (i + 1) % big_cycle_n, 1.0) for i in range(big_cycle_n)]
r = run_single_benchmark(big_cycle_n, big_cycle_edges, 'dsbm', seed=42, time_limit=1.0)
check('dSBM sparse backend no error', r['error'] is None)
check('dSBM sparse backend positive cut', r['best_cut'] > 0)

strong = run_dsbm_maxcut(big_cycle_n, big_cycle_edges, seed=42, time_limit=1.0,
                         num_restarts=16, steps=400)
check('dSBM sparse override positive cut', strong['best_cut'] > 0)
check('dSBM sparse override note reflects schedule',
      'restarts=16' in strong['solver_note'] and 'steps=400' in strong['solver_note'])
scaled = run_dsbm_maxcut(big_cycle_n, big_cycle_edges, seed=42, time_limit=1.0,
                         num_restarts=16, steps=400, c0_scale=2.5)
check('dSBM sparse override note reflects c0 scale',
      'c0_scale=2.5' in scaled['solver_note'])

r = run_single_benchmark(big_cycle_n, big_cycle_edges, 'pa_sparse_hybrid', seed=42, time_limit=1.0)
check('PA sparse hybrid no error', r['error'] is None)
check('PA sparse hybrid positive cut', r['best_cut'] > 0)

r = run_single_benchmark(big_cycle_n, big_cycle_edges, 'multiscale_pa', seed=42, time_limit=1.0)
check('Multiscale PA no error', r['error'] is None)
check('Multiscale PA positive cut', r['best_cut'] > 0)

# ============================================================
# Multi-instance benchmark
# ============================================================
print('\n=== Multi-instance benchmark ===')
test_instances = [
    {'name': 'K5', 'n_nodes': 5, 'edges': k5_edges, 'bks': 6},
    {'name': 'K8', 'n_nodes': 8,
     'edges': [(i,j,1.0) for i in range(8) for j in range(i+1,8)],
     'bks': 16},
]
results = run_benchmark(test_instances, ['bls', 'pa', 'simcim', 'dsbm'],
                        seed=42, time_limit=5.0, verbose=False)
check('Results has 8 entries (2 instances x 4 solvers)', len(results) == 8)
check('All K5 cuts=6', all(r['cut'] == 6 for r in results if r['instance'] == 'K5'))
check('All K8 cuts=16', all(r['cut'] == 16 for r in results if r['instance'] == 'K8'))
check('Gap is 0 for exact', all(
    r['gap'] == 0 or r['gap'] is None
    for r in results if r['bks'] and r['n_nodes'] <= 10
))
check('Ratio is 1.0 for exact', all(
    abs(r['ratio'] - 1.0) < 0.01
    for r in results if r['ratio'] is not None and r['n_nodes'] <= 10
))
check('Benchmark rows have graph ids', all(r.get('graph_id') for r in results))
check('Benchmark rows have route regime', all(r.get('route_regime') for r in results))
routebook = summarize_routebook(results)
check('Routebook has regimes', len(routebook['regimes']) > 0)
check('Routebook has tool counts', len(routebook['first_tool_counts']) > 0)

# ============================================================
# Scaling benchmark (quick)
# ============================================================
print('\n=== Scaling ===')
from bls_solver import random_3regular
nn, edges = random_3regular(500, seed=3500)
t0 = time.time()
r = run_single_benchmark(nn, edges, 'bls', seed=42, time_limit=5.0)
dt = time.time() - t0
check(f'BLS n=500 in <5s ({dt:.2f}s)', dt < 5.0)
ratio = r['best_cut'] / len(edges)
check(f'BLS n=500 ratio > 0.85 ({ratio:.3f})', ratio > 0.85)

t0 = time.time()
r = run_single_benchmark(nn, edges, 'pa', seed=42, time_limit=5.0)
dt = time.time() - t0
check(f'PA n=500 in <5s ({dt:.2f}s)', dt < 5.0)
ratio = r['best_cut'] / len(edges)
check(f'PA n=500 ratio > 0.85 ({ratio:.3f})', ratio > 0.85)

# ============================================================
# Full builtin benchmark (quick mode)
# ============================================================
print('\n=== Full builtin benchmark (quick) ===')
t0 = time.time()
results = run_benchmark(builtin[:6], ['bls', 'pa'],
                        seed=42, time_limit=10.0, verbose=False)
dt = time.time() - t0
check(f'Builtin benchmark completes ({dt:.1f}s)', dt < 120)
check('Results generated', len(results) > 0)
# All small exact should match BKS
exact_results = [r for r in results if r['bks'] and r['n_nodes'] <= 10]
exact_match = sum(1 for r in exact_results if r['gap'] == 0)
check(f'Exact instances match BKS ({exact_match}/{len(exact_results)})',
      exact_match == len(exact_results))

print('\n=== Evidence capsule sidecars ===')
tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'b150_test_benchmark')
os.makedirs(tmpdir, exist_ok=True)
report_path = os.path.join(tmpdir, 'benchmark.json')
mini_results = run_benchmark(test_instances[:1], ['bls'], seed=42,
                             time_limit=2.0, verbose=False)
save_json_report(mini_results, report_path, metadata={
    'mode': 'test',
    'solvers': ['bls'],
    'seed': 42,
    'time_limit': 2.0,
    'n_instances': 1,
})
capsule_path = os.path.join(tmpdir, 'benchmark.capsule.json')
receipt_path = os.path.join(tmpdir, 'benchmark.receipt.json')
check('Capsule sidecar exists', os.path.exists(capsule_path))
check('Receipt sidecar exists', os.path.exists(receipt_path))
verify = verify_benchmark_capsule(report_path, capsule_path=capsule_path)
check('Evidence capsule verifies', verify['verified'])

print('\n=== Checkpoint resume ===')
ckpt_path = os.path.join(tmpdir, 'benchmark_checkpoint.json')
partial = run_benchmark_checkpointed(
    test_instances[:1],
    ['bls', 'pa'],
    seed=42,
    time_limit=2.0,
    verbose=False,
    checkpoint_path=ckpt_path,
    resume=False,
    reset=True,
    max_tasks=1,
)
check('Checkpoint partial run has one row', len(partial) == 1)
resumed = run_benchmark_checkpointed(
    test_instances[:1],
    ['bls', 'pa'],
    seed=42,
    time_limit=2.0,
    verbose=False,
    checkpoint_path=ckpt_path,
    resume=True,
)
check('Checkpoint resume completes all rows', len(resumed) == 2)
with open(ckpt_path, 'r', encoding='utf-8') as f:
    payload = json.load(f)
check('Checkpoint marks run completed', payload['checkpoint']['run_state'] == 'completed')
check('Checkpoint has routebook', len(payload.get('routebook', {})) > 0)
check('Checkpoint has adversarial slices', isinstance(payload.get('adversarial_slices'), list))

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)
