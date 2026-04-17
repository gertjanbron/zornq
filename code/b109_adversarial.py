#!/usr/bin/env python3
"""
B109: Adversarial Benchmark — run alle ZornQ solvers op adversarial instanties.

Vergelijkt per instantie:
  1. B99  — Feedback-Edge Skeleton Solver
  2. BLS  — Breakout Local Search
  3. PA   — Population Annealing
  4. GW   — Goemans-Williamson SDP bound (bovengrens, niet solver)

Meet: best_cut, gap_to_bound, runtime, of solver het planted optimum vindt.

Usage:
  python b109_adversarial.py                 # snelle suite (n ≤ 50)
  python b109_adversarial.py --medium        # medium suite (n ~ 50-200)
  python b109_adversarial.py --scaling       # schaaltests per familie
"""

import numpy as np
import time
import sys
import argparse
from typing import Dict, List, Tuple, Optional

from adversarial_instance_generator import (
    small_adversarial_suite,
    medium_adversarial_suite,
    scaling_suite,
    compute_planted_gap,
    classify_difficulty,
    print_instance_summary,
)
from feedback_edge_solver import feedback_edge_maxcut
from bls_solver import bls_maxcut
from pa_solver import pa_maxcut


# =====================================================================
# SOLVER WRAPPERS (uniforme interface)
# =====================================================================

def run_b99(n: int, edges: List, time_limit: float = 30,
            seed: int = 42) -> Dict:
    """Run B99 Feedback-Edge Skeleton Solver."""
    t0 = time.time()
    try:
        cut, assignment, info = feedback_edge_maxcut(
            n, edges, time_limit=time_limit, seed=seed, verbose=False)
        dt = time.time() - t0
        return {
            'solver': 'B99',
            'cut': float(cut),
            'time_s': dt,
            'success': True,
            'info': info,
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            'solver': 'B99',
            'cut': 0.0,
            'time_s': dt,
            'success': False,
            'error': str(e),
        }


def run_bls(n: int, edges: List, time_limit: float = 30,
            seed: int = 42) -> Dict:
    """Run BLS (Breakout Local Search)."""
    t0 = time.time()
    try:
        result = bls_maxcut(
            n, edges, n_restarts=10, max_iter=5000,
            time_limit=time_limit, seed=seed, verbose=False)
        dt = time.time() - t0
        return {
            'solver': 'BLS',
            'cut': float(result['best_cut']),
            'time_s': dt,
            'success': True,
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            'solver': 'BLS',
            'cut': 0.0,
            'time_s': dt,
            'success': False,
            'error': str(e),
        }


def run_pa(n: int, edges: List, time_limit: float = 30,
           seed: int = 42) -> Dict:
    """Run PA (Population Annealing)."""
    t0 = time.time()
    try:
        result = pa_maxcut(
            n, edges, n_replicas=100, n_temps=40,
            time_limit=time_limit, seed=seed, verbose=False)
        dt = time.time() - t0
        return {
            'solver': 'PA',
            'cut': float(result['best_cut']),
            'time_s': dt,
            'success': True,
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            'solver': 'PA',
            'cut': 0.0,
            'time_s': dt,
            'success': False,
            'error': str(e),
        }


def run_gw_bound(n: int, edges: List) -> Optional[float]:
    """Bereken GW-SDP bovengrens (optioneel, vereist cvxpy)."""
    try:
        from b60_gw_bound import SimpleGraph, gw_sdp_bound
        g = SimpleGraph(n)
        for u, v, w in edges:
            g.add_edge(u, v, w)
        result = gw_sdp_bound(g, verbose=False)
        return float(result['sdp_bound'])
    except Exception:
        return None


# =====================================================================
# BENCHMARK RUNNER
# =====================================================================

def benchmark_instance(instance: Dict, time_limit: float = 30,
                       seed: int = 42, compute_bound: bool = True,
                       verbose: bool = True) -> Dict:
    """Run alle solvers op één instantie."""
    n = instance['n_nodes']
    edges = instance['edges']
    name = instance['name']

    if verbose:
        print(f"\n{'='*70}")
        print(f"Instance: {name}  (n={n}, m={len(edges)})")
        print(f"Target: {instance['target_solver']}")
        print(f"Note: {instance['difficulty_note']}")

    results = {}

    # Run solvers
    for run_fn, label in [(run_b99, 'B99'), (run_bls, 'BLS'), (run_pa, 'PA')]:
        r = run_fn(n, edges, time_limit=time_limit, seed=seed)
        results[label] = r
        if verbose:
            status = f"cut={r['cut']:.1f}" if r['success'] else f"FAIL: {r.get('error', '?')}"
            print(f"  {label:4s}: {status:30s}  ({r['time_s']:.2f}s)")

    # GW bound
    bound = None
    if compute_bound and n <= 500:
        bound = run_gw_bound(n, edges)
        if verbose and bound is not None:
            print(f"  GW bound: {bound:.1f}")

    # Analyse
    best_cut = max(r['cut'] for r in results.values() if r['success'])
    winner = max((r for r in results.values() if r['success']),
                 key=lambda r: r['cut'], default=None)

    analysis = {
        'instance': name,
        'family': instance['family'],
        'n': n,
        'm': len(edges),
        'results': results,
        'best_cut': best_cut,
        'winner': winner['solver'] if winner else 'NONE',
        'gw_bound': bound,
    }

    # Planted gap
    planted_cut = instance.get('planted_cut')
    if planted_cut is not None and planted_cut > 0:
        analysis['planted_cut'] = planted_cut
        analysis['planted_found'] = best_cut >= planted_cut * 0.999
        if verbose:
            found = "YES" if analysis['planted_found'] else "NO"
            print(f"  Planted optimum: {planted_cut:.1f}  Found: {found}  "
                  f"(best={best_cut:.1f})")

    # Gap to bound
    if bound is not None and bound > 0:
        gap = (bound - best_cut) / bound * 100
        analysis['gap_pct'] = gap
        if verbose:
            print(f"  Gap to GW-SDP: {gap:.2f}%")

    # Per-solver gaps
    if bound is not None and bound > 0:
        for label, r in results.items():
            if r['success']:
                r['gap_pct'] = (bound - r['cut']) / bound * 100

    if verbose:
        print(f"  Winner: {analysis['winner']}")

    return analysis


def benchmark_suite(suite: List[Dict], time_limit: float = 30,
                    seed: int = 42, compute_bound: bool = True,
                    verbose: bool = True) -> List[Dict]:
    """Run benchmark op een suite van instanties."""
    results = []
    for inst in suite:
        r = benchmark_instance(inst, time_limit=time_limit, seed=seed,
                               compute_bound=compute_bound, verbose=verbose)
        results.append(r)
    return results


# =====================================================================
# RAPPORTAGE
# =====================================================================

def print_summary_table(results: List[Dict]):
    """Print samenvattende tabel van alle resultaten."""
    print(f"\n{'='*90}")
    print("SAMENVATTING")
    print(f"{'='*90}")
    print(f"{'Instance':40s} {'n':>5s} {'m':>6s} {'B99':>8s} {'BLS':>8s} "
          f"{'PA':>8s} {'Bound':>8s} {'Gap%':>6s} {'Winner':>6s}")
    print("-" * 90)

    wins = {'B99': 0, 'BLS': 0, 'PA': 0, 'TIE': 0}
    total_gap = {'B99': [], 'BLS': [], 'PA': []}

    for r in results:
        n = r['n']
        m = r['m']
        name = r['instance'][:38]
        bound_str = f"{r['gw_bound']:.1f}" if r.get('gw_bound') else "—"
        gap_str = f"{r['gap_pct']:.1f}" if r.get('gap_pct') is not None else "—"

        cuts = {}
        for solver in ['B99', 'BLS', 'PA']:
            sr = r['results'].get(solver, {})
            if sr.get('success'):
                cuts[solver] = sr['cut']
            else:
                cuts[solver] = None

        cut_strs = {}
        for solver in ['B99', 'BLS', 'PA']:
            if cuts[solver] is not None:
                cut_strs[solver] = f"{cuts[solver]:.1f}"
            else:
                cut_strs[solver] = "FAIL"

        winner = r['winner']
        if winner != 'NONE':
            # Check for ties
            best = max(c for c in cuts.values() if c is not None)
            n_best = sum(1 for c in cuts.values() if c is not None and abs(c - best) < 0.5)
            if n_best > 1:
                winner = 'TIE'
                wins['TIE'] += 1
            else:
                wins[winner] = wins.get(winner, 0) + 1

        # Track per-solver gaps
        for solver in ['B99', 'BLS', 'PA']:
            sr = r['results'].get(solver, {})
            if sr.get('gap_pct') is not None:
                total_gap[solver].append(sr['gap_pct'])

        print(f"{name:40s} {n:5d} {m:6d} {cut_strs['B99']:>8s} "
              f"{cut_strs['BLS']:>8s} {cut_strs['PA']:>8s} "
              f"{bound_str:>8s} {gap_str:>6s} {winner:>6s}")

    print("-" * 90)
    print(f"\nWins: B99={wins['B99']}  BLS={wins['BLS']}  PA={wins['PA']}  TIE={wins['TIE']}")

    for solver in ['B99', 'BLS', 'PA']:
        gaps = total_gap[solver]
        if gaps:
            print(f"  {solver} avg gap: {np.mean(gaps):.2f}%  "
                  f"max gap: {np.max(gaps):.2f}%  "
                  f"min gap: {np.min(gaps):.2f}%")


def print_per_family_analysis(results: List[Dict]):
    """Analyseer resultaten per familie."""
    from collections import defaultdict
    families = defaultdict(list)
    for r in results:
        families[r['family']].append(r)

    print(f"\n{'='*70}")
    print("PER-FAMILIE ANALYSE")
    print(f"{'='*70}")

    for family, fam_results in sorted(families.items()):
        print(f"\n--- {family} ({len(fam_results)} instanties) ---")

        for solver in ['B99', 'BLS', 'PA']:
            cuts = []
            gaps = []
            times = []
            for r in fam_results:
                sr = r['results'].get(solver, {})
                if sr.get('success'):
                    cuts.append(sr['cut'])
                    times.append(sr['time_s'])
                    if sr.get('gap_pct') is not None:
                        gaps.append(sr['gap_pct'])

            if cuts:
                gap_str = f"gap={np.mean(gaps):.1f}%" if gaps else "—"
                print(f"  {solver}: avg_time={np.mean(times):.2f}s  {gap_str}")

        # Welke solver wint het vaakst in deze familie?
        winner_counts = {'B99': 0, 'BLS': 0, 'PA': 0}
        for r in fam_results:
            w = r.get('winner', 'NONE')
            if w in winner_counts:
                winner_counts[w] += 1
        best_solver = max(winner_counts, key=winner_counts.get)
        print(f"  Beste solver: {best_solver} ({winner_counts[best_solver]}/{len(fam_results)})")


# =====================================================================
# MAIN
# =====================================================================

def run_b109_report(suite_type: str = 'small', time_limit: float = 30,
                    seed: int = 42, verbose: bool = True):
    """Volledige B109 benchmark run."""
    print("=" * 70)
    print("B109: ADVERSARIAL INSTANCE BENCHMARK")
    print(f"Suite: {suite_type}  Time limit: {time_limit}s  Seed: {seed}")
    print("=" * 70)

    if suite_type == 'small':
        suite = small_adversarial_suite(seed=seed)
    elif suite_type == 'medium':
        suite = medium_adversarial_suite(seed=seed)
    else:
        suite = small_adversarial_suite(seed=seed)

    results = benchmark_suite(suite, time_limit=time_limit, seed=seed,
                              compute_bound=True, verbose=verbose)

    print_summary_table(results)
    print_per_family_analysis(results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B109 Adversarial Benchmark')
    parser.add_argument('--medium', action='store_true')
    parser.add_argument('--time-limit', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    suite_type = 'medium' if args.medium else 'small'
    run_b109_report(suite_type=suite_type, time_limit=args.time_limit,
                    seed=args.seed)
