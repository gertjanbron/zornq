#!/usr/bin/env python3
"""
gset_benchmark.py - B137 Gset Benchmark + Vergelijking

Systematic benchmarking of ZornQ solvers on Gset (G1-G67) and
synthetic graphs at Gset scale. Compares BLS, PA, and CUDA variants
against Best-Known Solutions (BKS).

Supports three modes:
  1. Full Gset: Load actual Gset files from gset/ directory
  2. Synthetic: Generate random graphs matching Gset dimensions
  3. Builtin: Test on small builtin graphs for quick validation

Outputs:
  - Console summary table
  - JSON report (machine-readable)
  - CSV report (spreadsheet-friendly)

Usage:
    python gset_benchmark.py                       # auto-detect mode
    python gset_benchmark.py --mode synthetic      # synthetic Gset-scale
    python gset_benchmark.py --mode builtin        # quick validation
    python gset_benchmark.py --mode gset           # real Gset files
    python gset_benchmark.py --graphs G14,G22,G43  # specific instances
    python gset_benchmark.py --max-nodes 2000      # limit graph size
    python gset_benchmark.py --time-limit 30       # per-instance limit
    python gset_benchmark.py --solvers bls,pa      # specific solvers
    python gset_benchmark.py --output report.json  # save report

References:
  - Gset: https://web.stanford.edu/~yyye/yyye/Gset/
  - BKS sources: Benlic & Hao (2013), Dunning et al. (2018)
"""

import numpy as np
import sys
import os
import time
import json
import argparse
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_dispatcher import classify_graph
from gset_loader import GSET_BKS, parse_gset_file, load_graph
from bls_solver import bls_maxcut, random_3regular, random_erdos_renyi
from pa_solver import pa_maxcut
from cuda_local_search import (
    maxcut_bls as cuda_bls,
    maxcut_pa as cuda_pa,
    maxcut_pa_sparse_hybrid,
    gpu_available,
)
from evidence_capsule import graph_fingerprint, save_benchmark_capsule
from multiscale_maxcut import multiscale_pa_maxcut
from quantum_inspired_baselines import run_dsbm_maxcut, run_simcim_maxcut


# ============================================================
# Solver registry
# ============================================================

SOLVERS = {}


def register_solver(name, func, description=""):
    """Register a solver for benchmarking."""
    SOLVERS[name] = {'func': func, 'desc': description}


def _run_bls_light(n, edges, seed=42, time_limit=None):
    """BLS with light settings for quick benchmark."""
    return bls_maxcut(n, edges, n_restarts=5, max_iter=500,
                      max_no_improve=50, time_limit=time_limit, seed=seed)


def _run_bls_heavy(n, edges, seed=42, time_limit=None):
    """BLS with heavy settings for quality benchmark."""
    restarts = max(10, min(50, 5000 // max(n, 1)))
    return bls_maxcut(n, edges, n_restarts=restarts, max_iter=2000,
                      max_no_improve=200, time_limit=time_limit, seed=seed)


def _run_pa_light(n, edges, seed=42, time_limit=None):
    """PA with light settings."""
    return pa_maxcut(n, edges, n_replicas=100, n_temps=30,
                     n_sweeps=3, time_limit=time_limit, seed=seed)


def _run_pa_heavy(n, edges, seed=42, time_limit=None):
    """PA with heavy settings."""
    replicas = min(500, max(100, 50000 // max(n, 1)))
    return pa_maxcut(n, edges, n_replicas=replicas, n_temps=60,
                     n_sweeps=5, time_limit=time_limit, seed=seed)


def _run_cuda_bls(n, edges, seed=42, time_limit=None):
    """CUDA BLS (GPU or CPU fallback)."""
    restarts = max(10, min(50, 5000 // max(n, 1)))
    return cuda_bls(n, edges, n_restarts=restarts, max_iter=2000,
                    max_no_improve=200, time_limit=time_limit, seed=seed)


def _run_cuda_pa(n, edges, seed=42, time_limit=None):
    """CUDA PA (GPU or CPU fallback)."""
    replicas = min(500, max(100, 50000 // max(n, 1)))
    return cuda_pa(n, edges, n_replicas=replicas, n_temps=60,
                   n_sweeps=5, time_limit=time_limit, seed=seed)


def _run_pa_sparse_hybrid(n, edges, seed=42, time_limit=None):
    """Sparse-specialized PA route with dSBM dispatch/probe."""
    replicas = min(500, max(100, 50000 // max(n, 1)))
    return maxcut_pa_sparse_hybrid(
        n, edges,
        n_replicas=replicas, n_temps=60, n_sweeps=5,
        time_limit=time_limit, seed=seed,
    )


def _run_multiscale_pa(n, edges, seed=42, time_limit=None):
    """B149 multiscale ordering + coarse-to-fine PA warm start."""
    return multiscale_pa_maxcut(n, edges, seed=seed, time_limit=time_limit)


def _run_combined(n, edges, seed=42, time_limit=None):
    """Best of BLS + PA (each gets half the time budget)."""
    t_half = time_limit / 2.0 if time_limit else None
    r_bls = _run_bls_heavy(n, edges, seed=seed, time_limit=t_half)
    r_pa = _run_pa_heavy(n, edges, seed=seed + 1000 if seed else None,
                         time_limit=t_half)
    if r_pa['best_cut'] > r_bls['best_cut']:
        r_pa['time_s'] = r_bls['time_s'] + r_pa['time_s']
        r_pa['solver_note'] = f'PA won (BLS={r_bls["best_cut"]:.0f})'
        return r_pa
    else:
        r_bls['time_s'] = r_bls['time_s'] + r_pa['time_s']
        r_bls['solver_note'] = f'BLS won (PA={r_pa["best_cut"]:.0f})'
        return r_bls


def _run_simcim(n, edges, seed=42, time_limit=None):
    """SimCIM baseline uit P101."""
    return run_simcim_maxcut(n, edges, seed=seed, time_limit=time_limit)


def _run_dsbm(n, edges, seed=42, time_limit=None):
    """Discrete Simulated Bifurcation baseline uit P101."""
    return run_dsbm_maxcut(n, edges, seed=seed, time_limit=time_limit)


# Register all solvers
register_solver('bls', _run_bls_light, 'BLS (light: 5 restarts)')
register_solver('bls_heavy', _run_bls_heavy, 'BLS (heavy: adaptive restarts)')
register_solver('pa', _run_pa_light, 'PA (light: 100 replicas)')
register_solver('pa_heavy', _run_pa_heavy, 'PA (heavy: adaptive replicas)')
register_solver('cuda_bls', _run_cuda_bls, 'CUDA BLS (GPU or fallback)')
register_solver('cuda_pa', _run_cuda_pa, 'CUDA PA (GPU or fallback)')
register_solver('pa_sparse_hybrid', _run_pa_sparse_hybrid,
                'PA + dSBM sparse-specialized hybrid')
register_solver('multiscale_pa', _run_multiscale_pa,
                'B149 multiscale coarse-to-fine PA warm start')
register_solver('combined', _run_combined, 'Best of BLS + PA')
register_solver('simcim', _run_simcim, 'SimCIM baseline (P101)')
register_solver('dsbm', _run_dsbm, 'dSBM baseline (P101)')


# ============================================================
# Synthetic Gset-scale graph generation
# ============================================================

def generate_synthetic_gset(name, seed=42):
    """
    Generate a synthetic graph matching Gset dimensions.
    Returns (n_nodes, edges_list, bks_or_None).
    """
    if name not in GSET_BKS:
        return None, None, None

    n, m, bks = GSET_BKS[name]
    rng = np.random.default_rng(seed)

    # Determine graph type from density
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0

    edges_set = set()
    if density < 0.01:
        # Sparse: approximate as random regular
        deg = max(2, round(2.0 * m / n))
        # Generate random edges targeting this degree
        target_m = m
        attempts = 0
        while len(edges_set) < target_m and attempts < target_m * 10:
            u = rng.integers(0, n)
            v = rng.integers(0, n)
            if u != v:
                edge = (min(u, v), max(u, v))
                edges_set.add(edge)
            attempts += 1
    else:
        # Dense: Erdos-Renyi targeting m edges
        p = density
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges_set.add((i, j))
                if len(edges_set) >= m:
                    break
            if len(edges_set) >= m:
                break

    # Assign random weights {-1, +1} to match typical Gset structure
    edges = []
    for u, v in edges_set:
        # Most Gset instances use unit weights, some use {-1,+1}
        w = 1.0
        edges.append((u, v, w))

    return n, edges, bks


# ============================================================
# Gset file discovery
# ============================================================

def find_gset_files():
    """Find available Gset files in standard locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(script_dir, '..', 'gset'),
        os.path.join(script_dir, 'gset'),
        os.path.join(script_dir, '..', 'data', 'gset'),
    ]

    found = {}
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            name = fname.upper().replace('.TXT', '')
            if name.startswith('G') and name[1:].isdigit():
                found[name] = os.path.join(d, fname)

    return found


def download_gset_file(name, target_dir):
    """
    Attempt to download a Gset file from Stanford.
    Returns filepath if successful, None otherwise.
    """
    import urllib.request

    url = f"https://web.stanford.edu/~yyye/yyye/Gset/{name}"
    target = os.path.join(target_dir, name)

    try:
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        # Validate it's a proper Gset file
        with open(target) as f:
            line = f.readline().strip().split()
            if len(line) >= 2 and line[0].isdigit():
                return target
        os.remove(target)
    except Exception:
        pass

    return None


# ============================================================
# Benchmark runner
# ============================================================

def run_single_benchmark(n_nodes, edges, solver_name, seed=42, time_limit=None):
    """Run a single solver on a single instance."""
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}")

    func = SOLVERS[solver_name]['func']
    try:
        result = func(n_nodes, edges, seed=seed, time_limit=time_limit)
        return {
            'best_cut': float(result['best_cut']),
            'time_s': float(result['time_s']),
            'device': result.get('device', 'cpu'),
            'solver_note': result.get('solver_note', ''),
            'error': None,
        }
    except Exception as e:
        return {
            'best_cut': 0.0,
            'time_s': 0.0,
            'device': 'error',
            'solver_note': '',
            'error': str(e),
        }


def infer_route_metadata(n_nodes, edges):
    """Compact regimekaart inspired by P43 bounded-width vs irregular structure."""
    info = classify_graph(n_nodes, edges)
    if info.get('is_grid') and info.get('grid_dims') is not None:
        width_proxy = min(info['grid_dims'])
    else:
        width_proxy = info.get('max_degree', 0)

    if n_nodes <= 25:
        regime = 'exact_small'
        first_tool = 'exact_reference'
        reason = 'tiny instance where exact reference is still cheap'
    elif info.get('is_grid') and info.get('grid_dims') is not None:
        if width_proxy <= 4:
            regime = 'bounded_width_small'
            first_tool = 'pfaffian_or_boundary_mps'
            reason = 'narrow grid with explicit bounded-width structure'
        elif width_proxy <= 8:
            regime = 'bounded_width_gray'
            first_tool = 'compare_bp_vs_boundary_mps'
            reason = 'medium width grid where strip methods need honesty checks'
        else:
            regime = 'true_2d'
            first_tool = 'multiscale_or_classical_portfolio'
            reason = 'grid width is large enough that 1D strip logic becomes awkward'
    elif info.get('is_sparse') and info.get('max_degree', 0) <= 6:
        regime = 'sparse_quasi_1d'
        first_tool = 'lightcone_or_pa'
        reason = 'low-degree sparse family where local structure is still favorable'
    elif info.get('is_sparse'):
        regime = 'irregular_sparse'
        first_tool = 'bp_style_baseline_first'
        reason = 'sparse graph without a clean strip order; cheap structural baseline first'
    elif info.get('is_dense'):
        regime = 'dense_portfolio'
        first_tool = 'combined_or_pa'
        reason = 'dense interaction pattern favors classical portfolio search'
    else:
        regime = 'classical_mid'
        first_tool = 'combined'
        reason = 'mid-density family where no exact or bounded-width route is obvious'

    return {
        'regime': regime,
        'first_tool': first_tool,
        'reason': reason,
        'width_proxy': int(width_proxy),
        'density': float(info.get('density', 0.0)),
        'avg_degree': float(info.get('avg_degree', 0.0)),
        'max_degree': int(info.get('max_degree', 0)),
        'cycle_rank': int(info.get('cycle_rank', 0)),
        'is_grid': bool(info.get('is_grid', False)),
        'grid_dims': info.get('grid_dims'),
        'is_sparse': bool(info.get('is_sparse', False)),
        'is_dense': bool(info.get('is_dense', False)),
    }


def summarize_routebook(results):
    """Aggregate compact routing metadata for quick post-run decisioning."""
    by_regime = {}
    by_first_tool = {}
    for row in results:
        regime = row.get('route_regime', 'unknown')
        first_tool = row.get('route_first_tool', 'unknown')
        slot = by_regime.setdefault(regime, {
            'rows': 0,
            'instances': set(),
            'avg_gap_vals': [],
            'worst_gap_pct': None,
            'worst_instance': None,
        })
        slot['rows'] += 1
        slot['instances'].add(row.get('instance'))
        gap_pct = row.get('gap_pct')
        if gap_pct is not None:
            slot['avg_gap_vals'].append(float(gap_pct))
            if slot['worst_gap_pct'] is None or float(gap_pct) > slot['worst_gap_pct']:
                slot['worst_gap_pct'] = float(gap_pct)
                slot['worst_instance'] = row.get('instance')
        by_first_tool[first_tool] = by_first_tool.get(first_tool, 0) + 1

    return {
        'regimes': {
            regime: {
                'rows': slot['rows'],
                'instances': len(slot['instances']),
                'avg_gap_pct': (
                    sum(slot['avg_gap_vals']) / len(slot['avg_gap_vals'])
                    if slot['avg_gap_vals'] else None
                ),
                'worst_gap_pct': slot['worst_gap_pct'],
                'worst_instance': slot['worst_instance'],
            }
            for regime, slot in by_regime.items()
        },
        'first_tool_counts': by_first_tool,
    }


def summarize_adversarial_slices(results, top_k=6):
    """Return the hardest rows across distinct instances as compact stress slices."""
    scored = [r for r in results if r.get('gap_pct') is not None]
    scored.sort(key=lambda r: (float(r['gap_pct']), float(r.get('time_s', 0.0))), reverse=True)
    slices = []
    seen = set()
    for row in scored:
        inst = row.get('instance')
        if inst in seen:
            continue
        seen.add(inst)
        slices.append({
            'instance': inst,
            'solver': row.get('solver'),
            'gap_pct': float(row['gap_pct']),
            'route_regime': row.get('route_regime'),
            'route_first_tool': row.get('route_first_tool'),
            'n_nodes': int(row.get('n_nodes', 0)),
            'n_edges': int(row.get('n_edges', 0)),
        })
        if len(slices) >= top_k:
            break
    return slices


def _atomic_write_json(payload, filepath):
    """Atomically write checkpoint JSON to disk."""
    target_dir = os.path.dirname(filepath) or '.'
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=target_dir,
                                     encoding='utf-8', suffix='.tmp') as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        temp_path = tmp.name
    os.replace(temp_path, filepath)


def _build_benchmark_row(inst, solver, result, seed, time_limit, route_meta):
    """Construct a single benchmark result row with route metadata."""
    name = inst['name']
    n = inst['n_nodes']
    edges = inst['edges']
    bks = inst.get('bks')
    m = len(edges)
    graph_id = graph_fingerprint(n, edges)

    gap = None
    gap_pct = None
    if bks and bks > 0 and result['best_cut'] > 0:
        gap = bks - result['best_cut']
        gap_pct = 100.0 * gap / bks

    return {
        'instance': name,
        'graph_id': graph_id,
        'n_nodes': n,
        'n_edges': m,
        'bks': bks,
        'solver': solver,
        'seed': seed,
        'time_limit_s': time_limit,
        'cut': result['best_cut'],
        'time_s': result['time_s'],
        'gap': gap,
        'gap_pct': gap_pct,
        'ratio': result['best_cut'] / bks if bks and bks > 0 else None,
        'device': result['device'],
        'error': result['error'],
        'route_regime': route_meta['regime'],
        'route_first_tool': route_meta['first_tool'],
        'route_reason': route_meta['reason'],
        'route_width_proxy': route_meta['width_proxy'],
        'route_density': route_meta['density'],
        'route_avg_degree': route_meta['avg_degree'],
        'route_max_degree': route_meta['max_degree'],
        'route_cycle_rank': route_meta['cycle_rank'],
        'route_is_grid': route_meta['is_grid'],
    }


def run_benchmark(instances, solver_names, seed=42, time_limit=None, verbose=True):
    """
    Run benchmark on multiple instances with multiple solvers.

    Args:
        instances: list of dicts with keys: name, n_nodes, edges, bks
        solver_names: list of solver names to benchmark
        seed: random seed
        time_limit: per-instance per-solver time limit
        verbose: print progress

    Returns:
        list of result dicts
    """
    results = []

    for inst in instances:
        name = inst['name']
        n = inst['n_nodes']
        edges = inst['edges']
        bks = inst.get('bks')
        m = len(edges)
        route_meta = infer_route_metadata(n, edges)

        if verbose:
            bks_str = f", BKS={bks}" if bks else ""
            print(f"\n{'='*60}")
            print(f"  {name}: n={n}, m={m}{bks_str}")
            print(f"  route: {route_meta['regime']} -> {route_meta['first_tool']}")
            print(f"{'='*60}")

        for solver in solver_names:
            if verbose:
                print(f"  Running {solver}...", end='', flush=True)

            r = run_single_benchmark(n, edges, solver, seed=seed,
                                     time_limit=time_limit)

            row = _build_benchmark_row(inst, solver, r, seed, time_limit, route_meta)
            results.append(row)

            if verbose:
                cut_str = f"{r['best_cut']:.0f}"
                time_str = f"{r['time_s']:.2f}s"
                if row['gap_pct'] is not None:
                    gap_str = f"gap={row['gap_pct']:.2f}%"
                    ratio_str = f"ratio={row['ratio']:.4f}"
                    print(f" cut={cut_str}, {gap_str}, {ratio_str}, t={time_str}")
                else:
                    print(f" cut={cut_str}, t={time_str}")

                if r['error']:
                    print(f"    ERROR: {r['error']}")

    return results


def run_benchmark_checkpointed(instances, solver_names, seed=42, time_limit=None,
                               verbose=True, checkpoint_path=None, resume=False,
                               reset=False, max_tasks=None):
    """
    Run benchmark with resumable atomic checkpoints.

    Checkpoint unit is one (instance, solver) row.
    """
    if not checkpoint_path:
        return run_benchmark(instances, solver_names, seed=seed,
                             time_limit=time_limit, verbose=verbose)

    plan = []
    route_cache = {}
    instance_lookup = {}
    for inst in instances:
        instance_lookup[inst['name']] = inst
        route_cache[inst['name']] = infer_route_metadata(inst['n_nodes'], inst['edges'])
        for solver in solver_names:
            plan.append({'instance': inst['name'], 'solver': solver})

    payload = None
    if resume and not reset and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

    if payload is None:
        payload = {
            'metadata': {
                'seed': seed,
                'time_limit': time_limit,
                'n_instances': len(instances),
                'solver_names': list(solver_names),
            },
            'checkpoint': {
                'next_index': 0,
                'completed_rows': 0,
                'plan_count': len(plan),
                'last_instance': None,
                'last_solver': None,
                'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'run_state': 'running',
            },
            'plan': plan,
            'results': [],
            'routebook': {},
            'adversarial_slices': [],
        }
        _atomic_write_json(payload, checkpoint_path)

    results = list(payload.get('results', []))
    start_index = int(payload.get('checkpoint', {}).get('next_index', 0))
    tasks_done = 0

    for idx in range(start_index, len(plan)):
        if max_tasks is not None and tasks_done >= max_tasks:
            break
        task = plan[idx]
        inst = instance_lookup[task['instance']]
        solver = task['solver']
        route_meta = route_cache[inst['name']]

        if verbose:
            print(f"[checkpoint {idx + 1}/{len(plan)}] {inst['name']} -> {solver}")

        run_result = run_single_benchmark(
            inst['n_nodes'],
            inst['edges'],
            solver,
            seed=seed,
            time_limit=time_limit,
        )
        row = _build_benchmark_row(inst, solver, run_result, seed, time_limit, route_meta)
        results.append(row)
        tasks_done += 1

        payload['results'] = results
        payload['checkpoint']['next_index'] = idx + 1
        payload['checkpoint']['completed_rows'] = len(results)
        payload['checkpoint']['last_instance'] = inst['name']
        payload['checkpoint']['last_solver'] = solver
        payload['routebook'] = summarize_routebook(results)
        payload['adversarial_slices'] = summarize_adversarial_slices(results)
        if payload['checkpoint']['next_index'] >= len(plan):
            payload['checkpoint']['run_state'] = 'completed'
            payload['checkpoint']['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            payload['checkpoint']['run_state'] = 'running'
        _atomic_write_json(payload, checkpoint_path)

    return results


# ============================================================
# Instance loading
# ============================================================

def load_gset_instances(names=None, max_nodes=None):
    """Load Gset instances from files."""
    gset_files = find_gset_files()
    instances = []

    if names is None:
        names = sorted(gset_files.keys(),
                       key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)

    for name in names:
        name_upper = name.upper()
        if name_upper not in gset_files:
            print(f"  Warning: {name_upper} not found, skipping")
            continue

        if max_nodes:
            n_expected = GSET_BKS.get(name_upper, (99999,))[0]
            if n_expected > max_nodes:
                continue

        try:
            g, n_nodes, n_edges = parse_gset_file(gset_files[name_upper])
            edges = [(i, j, w) for i, j, w in g.edges()]
            bks = GSET_BKS.get(name_upper, (None, None, None))[2]
            instances.append({
                'name': name_upper,
                'n_nodes': g.n_nodes,
                'edges': edges,
                'bks': bks,
            })
        except Exception as e:
            print(f"  Error loading {name_upper}: {e}")

    return instances


def load_synthetic_instances(names=None, max_nodes=None, seed=42):
    """Generate synthetic instances matching Gset dimensions."""
    instances = []

    if names is None:
        # Default: representative subset
        names = ['G1', 'G11', 'G14', 'G22', 'G32', 'G43', 'G48', 'G51']

    for name in names:
        name_upper = name.upper()
        if name_upper not in GSET_BKS:
            continue

        n_expected = GSET_BKS[name_upper][0]
        if max_nodes and n_expected > max_nodes:
            continue

        n, edges, bks = generate_synthetic_gset(name_upper, seed=seed)
        if n is not None:
            instances.append({
                'name': f'syn_{name_upper}',
                'n_nodes': n,
                'edges': edges,
                'bks': None,  # BKS not valid for synthetic
            })

    return instances


def load_builtin_instances():
    """Load small builtin graphs for quick validation."""
    instances = []

    # Small graphs with known BKS
    small = [
        ('K5', 5, [(i,j,1.0) for i in range(5) for j in range(i+1,5)], 6),
        ('K8', 8, [(i,j,1.0) for i in range(8) for j in range(i+1,8)], 16),
        ('K10', 10, [(i,j,1.0) for i in range(10) for j in range(i+1,10)], 25),
        ('Petersen', 10,
         [(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0),(4,0,1.0),
          (5,7,1.0),(7,9,1.0),(9,6,1.0),(6,8,1.0),(8,5,1.0),
          (0,5,1.0),(1,6,1.0),(2,7,1.0),(3,8,1.0),(4,9,1.0)], 12),
    ]

    for name, n, edges, bks in small:
        instances.append({'name': name, 'n_nodes': n, 'edges': edges, 'bks': bks})

    # Grids (bipartite, BKS = n_edges)
    for Lx, Ly in [(10, 4), (20, 5), (50, 4), (100, 3)]:
        n = Lx * Ly
        edges = []
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                if x + 1 < Lx:
                    edges.append((node, (x+1)*Ly+y, 1.0))
                if y + 1 < Ly:
                    edges.append((node, x*Ly+y+1, 1.0))
        bks = len(edges)  # bipartite
        instances.append({
            'name': f'grid_{Lx}x{Ly}',
            'n_nodes': n,
            'edges': edges,
            'bks': bks,
        })

    # Random 3-regular at various scales
    for n in [100, 500, 1000]:
        try:
            nn, edges = random_3regular(n, seed=n * 7)
            instances.append({
                'name': f'3reg_{n}',
                'n_nodes': nn,
                'edges': edges,
                'bks': None,
            })
        except Exception:
            pass

    # Dense ER
    for n in [50, 100]:
        nn, edges = random_erdos_renyi(n, p=0.5, seed=n * 3)
        instances.append({
            'name': f'ER_{n}',
            'n_nodes': nn,
            'edges': edges,
            'bks': None,
        })

    return instances


# ============================================================
# Report generation
# ============================================================

def print_summary_table(results):
    """Print a formatted summary table."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*90}")

    # Group by instance
    instances = {}
    for r in results:
        key = r['instance']
        if key not in instances:
            instances[key] = []
        instances[key].append(r)

    # Header
    solvers = sorted(set(r['solver'] for r in results))
    hdr = f"{'Instance':<15} {'n':>6} {'m':>7} {'BKS':>7}"
    for s in solvers:
        hdr += f" | {s:>12} {'gap%':>6} {'time':>6}"
    print(hdr)
    print("-" * len(hdr))

    total_gap = {s: [] for s in solvers}

    for inst_name, rows in instances.items():
        row0 = rows[0]
        bks_str = f"{row0['bks']:.0f}" if row0['bks'] else "?"
        line = f"{inst_name:<15} {row0['n_nodes']:>6} {row0['n_edges']:>7} {bks_str:>7}"

        for s in solvers:
            r = next((r for r in rows if r['solver'] == s), None)
            if r:
                cut_str = f"{r['cut']:.0f}"
                time_str = f"{r['time_s']:.1f}s"
                if r['gap_pct'] is not None:
                    gap_str = f"{r['gap_pct']:.2f}%"
                    total_gap[s].append(r['gap_pct'])
                else:
                    gap_str = "?"
                line += f" | {cut_str:>12} {gap_str:>6} {time_str:>6}"
            else:
                line += f" | {'N/A':>12} {'':>6} {'':>6}"

        print(line)

    # Averages
    print("-" * len(hdr))
    avg_line = f"{'AVERAGE':<15} {'':>6} {'':>7} {'':>7}"
    for s in solvers:
        gaps = total_gap[s]
        if gaps:
            avg = np.mean(gaps)
            avg_line += f" | {'':>12} {avg:.2f}% {'':>6}"
        else:
            avg_line += f" | {'':>12} {'N/A':>6} {'':>6}"
    print(avg_line)
    print(f"{'='*90}")


def save_json_report(results, filepath, metadata=None):
    """Save results as JSON."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault('routebook', summarize_routebook(results))
    merged_metadata.setdefault('adversarial_slices', summarize_adversarial_slices(results))
    report = {
        'metadata': merged_metadata,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'gpu_available': gpu_available(),
        'results': results,
    }
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  JSON report saved to: {filepath}")
    capsule_path, receipt_path = save_benchmark_capsule(
        results,
        merged_metadata,
        filepath,
    )
    print(f"  Evidence capsule saved to: {capsule_path}")
    print(f"  Receipt saved to: {receipt_path}")


def save_csv_report(results, filepath):
    """Save results as CSV."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fields = ['instance', 'n_nodes', 'n_edges', 'bks', 'solver',
              'cut', 'gap', 'gap_pct', 'ratio', 'time_s', 'device', 'error',
              'route_regime', 'route_first_tool', 'route_width_proxy']
    with open(filepath, 'w') as f:
        f.write(','.join(fields) + '\n')
        for r in results:
            vals = [str(r.get(k, '')) for k in fields]
            f.write(','.join(vals) + '\n')
    print(f"  CSV report saved to: {filepath}")


# ============================================================
# Main benchmark runner
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='B137: Gset Benchmark')
    parser.add_argument('--mode', choices=['gset', 'synthetic', 'builtin', 'auto'],
                        default='auto', help='Benchmark mode')
    parser.add_argument('--graphs', type=str, default=None,
                        help='Comma-separated graph names (e.g., G14,G22,G43)')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum graph size')
    parser.add_argument('--time-limit', type=float, default=60.0,
                        help='Per-instance per-solver time limit (seconds)')
    parser.add_argument('--solvers', type=str, default=None,
                        help='Comma-separated solver names')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (JSON or CSV based on extension)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: light solvers, small instances')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resumable checkpoint JSON path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint')
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Ignore old checkpoint contents and start over')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  B137: ZornQ Gset Benchmark")
    print(f"  GPU: {'YES' if gpu_available() else 'NO (CPU fallback)'}")
    print(f"  Time limit: {args.time_limit}s per solver per instance")
    print(f"{'='*60}")

    # Determine solver set
    if args.solvers:
        solver_names = args.solvers.split(',')
    elif args.quick:
        solver_names = ['bls', 'pa', 'simcim', 'dsbm']
    else:
        solver_names = ['bls_heavy', 'pa_heavy', 'combined']
        if gpu_available():
            solver_names.extend(['cuda_bls', 'cuda_pa'])

    # Invalid solver check
    for s in solver_names:
        if s not in SOLVERS:
            print(f"  ERROR: Unknown solver '{s}'")
            print(f"  Available: {', '.join(SOLVERS.keys())}")
            sys.exit(1)

    print(f"  Solvers: {', '.join(solver_names)}")

    # Determine instances
    graph_names = args.graphs.split(',') if args.graphs else None

    if args.mode == 'auto':
        # Auto-detect: use Gset files if available, else builtin
        gset_files = find_gset_files()
        if gset_files:
            args.mode = 'gset'
            print(f"  Mode: gset (found {len(gset_files)} files)")
        else:
            args.mode = 'builtin'
            print(f"  Mode: builtin (no Gset files found)")

    if args.mode == 'gset':
        instances = load_gset_instances(graph_names, args.max_nodes)
    elif args.mode == 'synthetic':
        instances = load_synthetic_instances(graph_names, args.max_nodes, args.seed)
    else:
        instances = load_builtin_instances()

    if not instances:
        print("  No instances loaded!")
        sys.exit(1)

    print(f"  Instances: {len(instances)}")
    print()

    # Run benchmark
    results = run_benchmark_checkpointed(
        instances,
        solver_names,
        seed=args.seed,
        time_limit=args.time_limit,
        verbose=True,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        reset=args.reset_checkpoint,
    )

    # Summary
    print_summary_table(results)

    # Save report
    if args.output:
        metadata = {
            'mode': args.mode,
            'solvers': solver_names,
            'seed': args.seed,
            'time_limit': args.time_limit,
            'max_nodes': args.max_nodes,
            'n_instances': len(instances),
            'checkpoint_path': args.checkpoint,
            'resume': bool(args.resume),
        }
        if args.output.endswith('.csv'):
            save_csv_report(results, args.output)
        else:
            save_json_report(results, args.output, metadata)

    # Exit code: 0 if all exact instances matched BKS
    exact_fails = sum(1 for r in results
                      if r['bks'] and r['gap'] and r['gap'] > 0.5
                      and r['n_nodes'] <= 20)
    if exact_fails > 0:
        print(f"\n  WARNING: {exact_fails} exact instances did not match BKS")

    return results


if __name__ == '__main__':
    main()
