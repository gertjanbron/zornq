#!/usr/bin/env python3
"""
B73: Quantum-Guided Branch-and-Bound — Experiment Runner

Vergelijkt QBB (quantum, degree, hybrid branching) met BLS, PA, B99
op random 3-regular grafen en adversarial instanties.

Focuspunten:
  1. Exactheid: vindt QBB bewezen optimum?
  2. Nodes: reduceert quantum branching de B&B-boom?
  3. Snelheid: overhead van QAOA correlaties vs winst in pruning?
"""

import numpy as np
import time
import sys

from quantum_branch_bound import (
    quantum_branch_bound, qbb_maxcut, compare_branching_strategies,
    eval_cut, BnBResult,
)
from bls_solver import bls_maxcut, random_3regular
from pa_solver import pa_maxcut
from feedback_edge_solver import feedback_edge_maxcut


def random_weighted_graph(n, p=0.3, seed=42):
    """Random Erdos-Renyi met +-1 gewichten."""
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                w = rng.choice([-1.0, 1.0])
                edges.append((i, j, w))
    return n, edges


def brute_force_maxcut(n, edges):
    """Brute force voor n <= 20."""
    best = 0
    best_assign = {}
    for mask in range(2**n):
        assign = {i: (mask >> i) & 1 for i in range(n)}
        cut = eval_cut(n, edges, assign)
        if cut > best:
            best = cut
            best_assign = assign
    return best, best_assign


def benchmark_exactness(sizes=None, seed=42, time_limit=10):
    """Vergelijk QBB exactheid met brute force."""
    if sizes is None:
        sizes = [10, 12, 14, 16, 18]

    print("=" * 70)
    print("EXACTHEIDSTEST: QBB vs Brute Force")
    print("=" * 70)

    for n in sizes:
        _, edges = random_3regular(n, seed=seed)

        # Brute force (alleen als n <= 20)
        if n <= 20:
            t0 = time.time()
            bf_cut, _ = brute_force_maxcut(n, edges)
            bf_time = time.time() - t0
        else:
            bf_cut = None
            bf_time = 0

        # QBB degree
        t0 = time.time()
        qbb_deg = quantum_branch_bound(n, edges, branching='degree',
                                        time_limit=time_limit, verbose=False)
        deg_time = time.time() - t0

        # QBB hybrid (met mock correlaties voor snelheid)
        zz = {}
        for u, v, w in edges:
            zz[(min(u,v), max(u,v))] = np.random.uniform(-1, 1)
        qbb_hyb = quantum_branch_bound(n, edges, zz_dict=zz,
                                        branching='hybrid',
                                        time_limit=time_limit, verbose=False)

        bf_str = f"{bf_cut:.0f}" if bf_cut is not None else "—"
        exact_deg = "OK" if qbb_deg.is_exact else "TIMEOUT"
        exact_hyb = "OK" if qbb_hyb.is_exact else "TIMEOUT"
        match = "MATCH" if bf_cut is not None and qbb_deg.best_cut == bf_cut else ""

        print(f"  n={n:3d}  BF={bf_str:>5s}  QBB-deg={qbb_deg.best_cut:.0f} "
              f"({exact_deg}, {qbb_deg.nodes_explored} nodes, {deg_time:.2f}s)  "
              f"QBB-hyb={qbb_hyb.best_cut:.0f} ({exact_hyb}, {qbb_hyb.nodes_explored} nodes)  "
              f"{match}")


def benchmark_branching_comparison(n=16, n_instances=5, seed=42, time_limit=10):
    """Vergelijk branching strategieën op meerdere random instanties."""
    print("\n" + "=" * 70)
    print(f"BRANCHING VERGELIJKING: n={n}, {n_instances} instanties")
    print("=" * 70)

    stats = {'degree': [], 'hybrid': []}

    for i in range(n_instances):
        _, edges = random_3regular(n, seed=seed + i)

        # Brute force referentie
        bf_cut, _ = brute_force_maxcut(n, edges) if n <= 20 else (None, None)

        # Mock quantum correlaties
        zz = {}
        for u, v, w in edges:
            zz[(min(u,v), max(u,v))] = np.random.uniform(-1, 1)

        for strategy in ['degree', 'hybrid']:
            r = quantum_branch_bound(n, edges,
                                      zz_dict=zz if strategy == 'hybrid' else None,
                                      branching=strategy,
                                      time_limit=time_limit,
                                      verbose=False)
            stats[strategy].append({
                'nodes': r.nodes_explored,
                'pruned': r.nodes_pruned,
                'cut': r.best_cut,
                'exact': r.is_exact,
                'time': r.time_s,
            })

    print(f"\n{'Strategy':>10s} {'Avg Nodes':>12s} {'Avg Pruned':>12s} "
          f"{'Avg Cut':>10s} {'Exact%':>8s} {'Avg Time':>10s}")
    print("-" * 62)
    for s in ['degree', 'hybrid']:
        nodes = np.mean([x['nodes'] for x in stats[s]])
        pruned = np.mean([x['pruned'] for x in stats[s]])
        cut = np.mean([x['cut'] for x in stats[s]])
        exact_pct = 100 * np.mean([x['exact'] for x in stats[s]])
        avg_time = np.mean([x['time'] for x in stats[s]])
        print(f"{s:>10s} {nodes:12.0f} {pruned:12.0f} {cut:10.1f} "
              f"{exact_pct:7.0f}% {avg_time:10.3f}s")


def benchmark_vs_solvers(sizes=None, seed=42, time_limit=10):
    """Vergelijk QBB met BLS, PA, B99 op exact MaxCut."""
    if sizes is None:
        sizes = [12, 14, 16, 18, 20]

    print("\n" + "=" * 70)
    print("QBB vs SOLVERS: random 3-regular")
    print("=" * 70)
    print(f"{'n':>4s} {'BF':>6s} {'QBB':>6s} {'QBB-ex':>7s} "
          f"{'QBB-nd':>8s} {'BLS':>6s} {'PA':>6s} {'B99':>6s} "
          f"{'QBB-t':>7s} {'BLS-t':>7s} {'PA-t':>7s}")
    print("-" * 80)

    for n in sizes:
        _, edges = random_3regular(n, seed=seed)

        # Brute force
        bf_cut = None
        if n <= 20:
            bf_cut, _ = brute_force_maxcut(n, edges)

        # QBB
        qbb = quantum_branch_bound(n, edges, branching='degree',
                                    time_limit=time_limit, verbose=False)

        # BLS
        bls = bls_maxcut(n, edges, n_restarts=5, max_iter=2000,
                         time_limit=time_limit, seed=seed)

        # PA
        pa = pa_maxcut(n, edges, n_replicas=100, n_temps=40,
                       time_limit=time_limit, seed=seed)

        # B99
        b99_cut, _, _ = feedback_edge_maxcut(n, edges, time_limit=time_limit,
                                              seed=seed)

        bf_str = f"{bf_cut:.0f}" if bf_cut else "—"
        exact_str = "EXACT" if qbb.is_exact else "HEUR"
        print(f"{n:4d} {bf_str:>6s} {qbb.best_cut:6.0f} {exact_str:>7s} "
              f"{qbb.nodes_explored:8d} {bls['best_cut']:6.0f} "
              f"{pa['best_cut']:6.0f} {b99_cut:6.0f} "
              f"{qbb.time_s:7.3f} {bls['time_s']:7.3f} {pa['time_s']:7.3f}")


def benchmark_adversarial(seed=42, time_limit=10):
    """QBB op adversarial instanties uit B109."""
    try:
        from adversarial_instance_generator import small_adversarial_suite
    except ImportError:
        print("\nAdversarial suite niet beschikbaar, overslaan")
        return

    print("\n" + "=" * 70)
    print("QBB op ADVERSARIAL INSTANTIES (B109)")
    print("=" * 70)

    suite = small_adversarial_suite(seed=seed)
    print(f"{'Instance':>35s} {'n':>4s} {'QBB':>7s} {'exact':>6s} "
          f"{'nodes':>8s} {'BLS':>7s} {'PA':>7s} {'B99':>7s}")
    print("-" * 85)

    for inst in suite:
        n = inst['n_nodes']
        edges = inst['edges']
        name = inst['name'][:33]

        qbb = quantum_branch_bound(n, edges, branching='degree',
                                    time_limit=time_limit, verbose=False)
        bls = bls_maxcut(n, edges, n_restarts=5, time_limit=time_limit, seed=seed)
        pa = pa_maxcut(n, edges, n_replicas=100, time_limit=time_limit, seed=seed)
        b99_cut, _, _ = feedback_edge_maxcut(n, edges, time_limit=time_limit, seed=seed)

        exact_str = "EXACT" if qbb.is_exact else "HEUR"
        print(f"{name:>35s} {n:4d} {qbb.best_cut:7.1f} {exact_str:>6s} "
              f"{qbb.nodes_explored:8d} {bls['best_cut']:7.1f} "
              f"{pa['best_cut']:7.1f} {b99_cut:7.1f}")


def run_b73_report(time_limit=10, seed=42, verbose=True):
    """Volledige B73 benchmark."""
    print("=" * 70)
    print("B73: QUANTUM-GUIDED BRANCH-AND-BOUND")
    print(f"Time limit: {time_limit}s  Seed: {seed}")
    print("=" * 70)

    benchmark_exactness(sizes=[10, 12, 14, 16, 18], seed=seed,
                        time_limit=time_limit)
    benchmark_branching_comparison(n=16, n_instances=5, seed=seed,
                                   time_limit=time_limit)
    benchmark_vs_solvers(sizes=[12, 14, 16, 18, 20], seed=seed,
                          time_limit=time_limit)
    benchmark_adversarial(seed=seed, time_limit=time_limit)


if __name__ == '__main__':
    run_b73_report(time_limit=10, seed=42)
