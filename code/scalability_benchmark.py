#!/usr/bin/env python3
"""
scalability_benchmark.py - B133 Scalability Benchmark Suite

Systematische meting: waar wint QC-op-laptop vs klassiek, en waar niet?

Metingen:
  1. QAOA MaxCut: n vs wall-time (SV vs MPS vs BLS), chi-scaling
  2. VQE Heisenberg: n vs wall-time, chi vs fidelity
  3. Trotter evolutie: n vs wall-time, order vs fidelity
  4. Chi convergentie: fixed n, chi sweep, accuracy vs cost
  5. Gate-count scaling: circuit complexiteit vs probleemgrootte

Output: gestructureerd dict + console rapport.

Gebruik:
    from scalability_benchmark import run_full_suite
    results = run_full_suite(verbose=True)

    # Of individuele benchmarks
    from scalability_benchmark import bench_qaoa_scaling, bench_chi_convergence
    r = bench_qaoa_scaling(n_range=[8, 16, 32, 64], verbose=True)

Author: ZornQ project
Date: 16 april 2026
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any

from hamiltonian_compiler import Hamiltonian
from circuit_interface import Circuit, Gates, Observable, run_circuit
from quality_certificate import certify_energy, certify_circuit_result


# =====================================================================
# HULPFUNCTIES
# =====================================================================

def _make_random_graph(n, edge_prob=0.3, seed=42):
    """Genereer random graaf voor MaxCut benchmarking."""
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = rng.choice([-1, 1])
                edges.append((i, j, w))
    if not edges:
        edges.append((0, 1, 1))
    return edges


def _make_chain_graph(n):
    """Lineaire keten (voor 1D structuur)."""
    return [(i, i + 1, 1.0) for i in range(n - 1)]


def _timed(fn, *args, **kwargs):
    """Voer fn uit en geef (result, wall_time) tuple terug."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - t0


def _safe_run_circuit(circuit, observables=None, backend='auto',
                      max_chi=64, timeout=60.0):
    """run_circuit met timeout-bewaking (via wall-time check)."""
    t0 = time.time()
    try:
        result = run_circuit(circuit, observables=observables,
                             backend=backend, max_chi=max_chi)
        elapsed = time.time() - t0
        if elapsed > timeout:
            return None, elapsed
        return result, elapsed
    except Exception as e:
        return None, time.time() - t0


# =====================================================================
# BENCHMARK 1: QAOA MaxCut Scaling
# =====================================================================

def bench_qaoa_scaling(n_range=None, p=1, chi_values=None,
                       edge_prob=0.3, verbose=True):
    """Meet QAOA MaxCut wall-time vs n voor SV en MPS backends.

    Args:
        n_range: lijst van qubit-aantallen
        p: QAOA depth
        chi_values: MPS bond dimensies om te testen
        edge_prob: random graaf edge-kans
        verbose: print resultaten

    Returns:
        dict met meetresultaten per (n, backend, chi)
    """
    if n_range is None:
        n_range = [4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 50, 100]
    if chi_values is None:
        chi_values = [4, 8, 16, 32]

    if verbose:
        print("=" * 70)
        print("  BENCH 1: QAOA MaxCut Scaling (p=%d)" % p)
        print("  n: %s" % n_range)
        print("  chi: %s" % chi_values)
        print("=" * 70)

    gammas = [0.74 * np.pi] * p
    betas = [0.12 * np.pi] * p

    results = []

    for n in n_range:
        edges = _make_random_graph(n, edge_prob=edge_prob)
        n_edges = len(edges)

        # QAOA circuit
        qc = Circuit.qaoa_maxcut(n, edges, p=p, gammas=gammas, betas=betas)
        n_gates = len(qc)
        depth = qc.depth()

        # State vector (alleen voor kleine n)
        sv_time = None
        sv_energy = None
        if n <= 22:
            obs = {'C': Observable.maxcut_cost(edges)}
            result, elapsed = _timed(
                run_circuit, qc, observables=obs, backend='statevector')
            sv_time = elapsed
            sv_energy = result['observables']['C']
            if verbose:
                print("  n=%3d | SV:  %.4fs | E=%.3f | %d gates, depth %d" % (
                    n, sv_time, sv_energy, n_gates, depth))

        # MPS met verschillende chi
        for chi in chi_values:
            obs = {'C': Observable.maxcut_cost(edges)}
            result, elapsed = _safe_run_circuit(
                qc, observables=obs, backend='mps', max_chi=chi,
                timeout=120.0)
            if result is not None:
                mps_energy = result['observables']['C']
                max_chi_reached = result.get('max_chi_reached', chi)
                discarded = result.get('total_discarded', 0)
            else:
                mps_energy = None
                max_chi_reached = chi
                discarded = None

            entry = {
                'n': n, 'p': p, 'n_edges': n_edges,
                'n_gates': n_gates, 'depth': depth,
                'chi': chi, 'backend': 'mps',
                'time': elapsed,
                'energy': mps_energy,
                'sv_time': sv_time, 'sv_energy': sv_energy,
                'max_chi_reached': max_chi_reached,
                'total_discarded': discarded,
            }
            results.append(entry)

            if verbose and result is not None:
                err_str = ""
                if sv_energy is not None and mps_energy is not None:
                    err = abs(mps_energy - sv_energy)
                    err_str = " | err=%.2e" % err
                print("  n=%3d | MPS chi=%2d: %.4fs | E=%s | chi_max=%d%s" % (
                    n, chi, elapsed,
                    "%.3f" % mps_energy if mps_energy is not None else "TIMEOUT",
                    max_chi_reached, err_str))
            elif verbose:
                print("  n=%3d | MPS chi=%2d: %.4fs | TIMEOUT" % (
                    n, chi, elapsed))

    return {'bench': 'qaoa_scaling', 'p': p, 'results': results}


# =====================================================================
# BENCHMARK 2: VQE Heisenberg Scaling
# =====================================================================

def bench_vqe_scaling(n_range=None, depth=3, chi_values=None,
                      verbose=True):
    """Meet VQE circuit evaluatie-tijd vs n.

    NB: meet NIET optimizer convergentie (te langzaam voor benchmark),
    maar enkel de cost van 1 circuit evaluatie.

    Args:
        n_range: qubit-aantallen
        depth: VQE ansatz depth
        chi_values: MPS chi waarden
        verbose: print resultaten

    Returns:
        dict met meetresultaten
    """
    if n_range is None:
        n_range = [4, 6, 8, 10, 14, 20, 30, 50, 100]
    if chi_values is None:
        chi_values = [4, 8, 16, 32]

    if verbose:
        print("=" * 70)
        print("  BENCH 2: VQE Heisenberg Evaluatie Scaling (depth=%d)" % depth)
        print("  n: %s" % n_range)
        print("=" * 70)

    results = []
    rng = np.random.RandomState(42)

    for n in n_range:
        H = Hamiltonian.heisenberg_xxx(n, J=1.0)
        obs = H.to_observable()
        n_params = n * 2 * (depth + 1) + (n - 1) * depth
        params = rng.uniform(0, 2 * np.pi, n_params)
        qc = Circuit.hardware_efficient(n, depth=depth, params=params)
        n_gates = len(qc)

        # State vector
        sv_time = None
        sv_energy = None
        if n <= 22:
            result, elapsed = _timed(
                run_circuit, qc, observables={'E': obs}, backend='statevector')
            sv_time = elapsed
            sv_energy = result['observables']['E']
            if verbose:
                print("  n=%3d | SV:  %.4fs | E=%.3f | %d gates" % (
                    n, sv_time, sv_energy, n_gates))

        # MPS
        for chi in chi_values:
            result, elapsed = _safe_run_circuit(
                qc, observables={'E': obs}, backend='mps', max_chi=chi,
                timeout=120.0)
            if result is not None:
                mps_energy = result['observables']['E']
            else:
                mps_energy = None

            entry = {
                'n': n, 'depth': depth, 'n_gates': n_gates,
                'chi': chi, 'time': elapsed,
                'energy': mps_energy,
                'sv_time': sv_time, 'sv_energy': sv_energy,
            }
            results.append(entry)

            if verbose and result is not None:
                err_str = ""
                if sv_energy is not None and mps_energy is not None:
                    err_str = " | err=%.2e" % abs(mps_energy - sv_energy)
                print("  n=%3d | MPS chi=%2d: %.4fs | E=%s%s" % (
                    n, chi, elapsed,
                    "%.3f" % mps_energy if mps_energy is not None else "FAIL",
                    err_str))
            elif verbose:
                print("  n=%3d | MPS chi=%2d: %.4fs | TIMEOUT" % (
                    n, chi, elapsed))

    return {'bench': 'vqe_scaling', 'depth': depth, 'results': results}


# =====================================================================
# BENCHMARK 3: Trotter Evolution Scaling
# =====================================================================

def bench_trotter_scaling(n_range=None, t_evolve=1.0, steps=10,
                          orders=None, chi_values=None, verbose=True):
    """Meet Trotter circuit wall-time en fidelity vs n.

    Args:
        n_range: qubit-aantallen
        t_evolve: evolutietijd
        steps: Trotter stappen
        orders: Trotter ordes [1, 2, 4]
        chi_values: MPS chi waarden
        verbose: print resultaten

    Returns:
        dict met meetresultaten
    """
    if n_range is None:
        n_range = [4, 6, 8, 10, 14, 20, 30, 50]
    if orders is None:
        orders = [1, 2]
    if chi_values is None:
        chi_values = [4, 8, 16, 32]

    if verbose:
        print("=" * 70)
        print("  BENCH 3: Trotter Evolutie Scaling (t=%.1f, steps=%d)" % (
            t_evolve, steps))
        print("  n: %s, orders: %s" % (n_range, orders))
        print("=" * 70)

    results = []

    for order in orders:
        for n in n_range:
            H = Hamiltonian.ising_transverse(n, J=1.0, h=0.5)
            qc = H.trotter(t_evolve, steps=steps, order=order)
            n_gates = len(qc)
            depth = qc.depth()

            # State vector (referentie, kleine n)
            sv_time = None
            if n <= 20:
                result, elapsed = _timed(
                    run_circuit, qc, backend='statevector')
                sv_time = elapsed
                if verbose:
                    print("  T%d n=%3d | SV:  %.4fs | %d gates, depth %d" % (
                        order, n, sv_time, n_gates, depth))

            # MPS
            for chi in chi_values:
                result, elapsed = _safe_run_circuit(
                    qc, backend='mps', max_chi=chi, timeout=120.0)

                entry = {
                    'n': n, 'order': order, 'steps': steps,
                    't': t_evolve, 'n_gates': n_gates, 'depth': depth,
                    'chi': chi, 'time': elapsed,
                    'sv_time': sv_time,
                    'success': result is not None,
                }
                results.append(entry)

                if verbose:
                    status = "%.4fs" % elapsed if result is not None else "TIMEOUT"
                    print("  T%d n=%3d | MPS chi=%2d: %s" % (
                        order, n, chi, status))

    return {'bench': 'trotter_scaling', 't': t_evolve, 'steps': steps,
            'results': results}


# =====================================================================
# BENCHMARK 4: Chi Convergentie
# =====================================================================

def bench_chi_convergence(n=20, chi_range=None, verbose=True):
    """Meet MPS nauwkeurigheid vs chi voor een vast probleem.

    Vergelijkt met state vector referentie (exact).

    Args:
        n: aantal qubits (moet <= 22 voor SV referentie)
        chi_range: chi waarden om te testen
        verbose: print resultaten

    Returns:
        dict met chi, energy, error, time per chi
    """
    if chi_range is None:
        chi_range = [2, 4, 8, 12, 16, 24, 32, 48, 64]

    if verbose:
        print("=" * 70)
        print("  BENCH 4: Chi Convergentie (n=%d)" % n)
        print("  chi: %s" % chi_range)
        print("=" * 70)

    # QAOA circuit op random graaf
    p = 2
    edges = _make_random_graph(n, edge_prob=0.3, seed=42)
    gammas = [0.74 * np.pi] * p
    betas = [0.12 * np.pi] * p
    qc = Circuit.qaoa_maxcut(n, edges, p=p, gammas=gammas, betas=betas)
    obs = {'C': Observable.maxcut_cost(edges)}

    # Exact referentie
    if verbose:
        print("  State vector referentie...")
    ref_result, sv_time = _timed(
        run_circuit, qc, observables=obs, backend='statevector')
    exact_energy = ref_result['observables']['C']
    if verbose:
        print("  SV: E=%.6f (%.4fs)" % (exact_energy, sv_time))

    results = []
    for chi in chi_range:
        result, elapsed = _safe_run_circuit(
            qc, observables=obs, backend='mps', max_chi=chi, timeout=120.0)

        if result is not None:
            mps_energy = result['observables']['C']
            error = abs(mps_energy - exact_energy)
            rel_error = error / abs(exact_energy) if exact_energy != 0 else error
            max_chi = result.get('max_chi_reached', chi)
            discarded = result.get('total_discarded', 0)
        else:
            mps_energy = error = rel_error = None
            max_chi = chi
            discarded = None

        entry = {
            'chi': chi, 'time': elapsed,
            'energy': mps_energy, 'exact_energy': exact_energy,
            'error': error, 'rel_error': rel_error,
            'max_chi_reached': max_chi,
            'total_discarded': discarded,
        }
        results.append(entry)

        if verbose and mps_energy is not None:
            print("  chi=%3d: E=%.6f | err=%.2e | rel=%.2e | %.4fs | chi_max=%d" % (
                chi, mps_energy, error, rel_error, elapsed, max_chi))

    # Heisenberg model ook testen
    if verbose:
        print("\n  --- Heisenberg XXX (n=%d) ---" % n)
    H = Hamiltonian.heisenberg_xxx(n, J=1.0)
    h_obs = H.to_observable()
    rng = np.random.RandomState(42)
    n_params = n * 2 * 4 + (n - 1) * 3  # depth=3
    params = rng.uniform(0, 2 * np.pi, n_params)
    qc_h = Circuit.hardware_efficient(n, depth=3, params=params)

    ref_h, _ = _timed(run_circuit, qc_h, observables={'E': h_obs},
                       backend='statevector')
    exact_h = ref_h['observables']['E']

    heisenberg_results = []
    for chi in chi_range:
        result, elapsed = _safe_run_circuit(
            qc_h, observables={'E': h_obs}, backend='mps', max_chi=chi,
            timeout=120.0)
        if result is not None:
            mps_e = result['observables']['E']
            err = abs(mps_e - exact_h)
        else:
            mps_e = err = None

        heisenberg_results.append({
            'chi': chi, 'time': elapsed,
            'energy': mps_e, 'exact_energy': exact_h,
            'error': err,
        })

        if verbose and mps_e is not None:
            print("  chi=%3d: E=%.6f | err=%.2e | %.4fs" % (
                chi, mps_e, err, elapsed))

    return {
        'bench': 'chi_convergence', 'n': n,
        'qaoa_results': results,
        'heisenberg_results': heisenberg_results,
        'sv_time': sv_time,
    }


# =====================================================================
# BENCHMARK 5: Gate-Count en Circuit Complexiteit
# =====================================================================

def bench_circuit_complexity(n_range=None, verbose=True):
    """Meet circuit gate-count en depth vs n voor verschillende modellen.

    Args:
        n_range: qubit-aantallen
        verbose: print resultaten

    Returns:
        dict met gate-count en depth per (n, model)
    """
    if n_range is None:
        n_range = [4, 6, 8, 10, 14, 20, 30, 50, 100, 200]

    if verbose:
        print("=" * 70)
        print("  BENCH 5: Circuit Complexiteit vs n")
        print("  n: %s" % n_range)
        print("=" * 70)

    results = []
    for n in n_range:
        # QAOA MaxCut p=1
        edges = _make_chain_graph(n)
        qc = Circuit.qaoa_maxcut(n, edges, p=1,
                                  gammas=[0.5], betas=[0.3])
        entry_qaoa = {
            'model': 'QAOA-MaxCut-p1', 'n': n,
            'n_gates': len(qc), 'depth': qc.depth(),
            'n_1q': sum(1 for op in qc.ops if len(op.qubits) == 1),
            'n_2q': sum(1 for op in qc.ops if len(op.qubits) == 2),
        }
        results.append(entry_qaoa)

        # QAOA p=2
        qc2 = Circuit.qaoa_maxcut(n, edges, p=2,
                                   gammas=[0.5, 0.3], betas=[0.3, 0.2])
        entry_qaoa2 = {
            'model': 'QAOA-MaxCut-p2', 'n': n,
            'n_gates': len(qc2), 'depth': qc2.depth(),
            'n_1q': sum(1 for op in qc2.ops if len(op.qubits) == 1),
            'n_2q': sum(1 for op in qc2.ops if len(op.qubits) == 2),
        }
        results.append(entry_qaoa2)

        # VQE HEA depth=3
        qc_v = Circuit.hardware_efficient(n, depth=3)
        entry_vqe = {
            'model': 'VQE-HEA-d3', 'n': n,
            'n_gates': len(qc_v), 'depth': qc_v.depth(),
            'n_1q': sum(1 for op in qc_v.ops if len(op.qubits) == 1),
            'n_2q': sum(1 for op in qc_v.ops if len(op.qubits) == 2),
        }
        results.append(entry_vqe)

        # Trotter-2 (Ising)
        H = Hamiltonian.ising_transverse(n, J=1.0, h=0.5)
        qc_t = H.trotter(t=1.0, steps=5, order=2)
        entry_trotter = {
            'model': 'Trotter2-Ising-s5', 'n': n,
            'n_gates': len(qc_t), 'depth': qc_t.depth(),
            'n_1q': sum(1 for op in qc_t.ops if len(op.qubits) == 1),
            'n_2q': sum(1 for op in qc_t.ops if len(op.qubits) == 2),
        }
        results.append(entry_trotter)

        if verbose:
            print("  n=%3d | QAOA-p1: %4d gates (d=%3d) | VQE-d3: %4d (d=%3d) | T2: %5d (d=%4d)" % (
                n, entry_qaoa['n_gates'], entry_qaoa['depth'],
                entry_vqe['n_gates'], entry_vqe['depth'],
                entry_trotter['n_gates'], entry_trotter['depth']))

    return {'bench': 'circuit_complexity', 'results': results}


# =====================================================================
# BENCHMARK 6: Break-Even Analyse
# =====================================================================

def bench_break_even(verbose=True):
    """Vind het break-even punt: bij welke n is MPS sneller dan SV?

    Meet wall-time voor SV en MPS bij oplopende n.
    SV schaalt als O(2^n), MPS als O(n * chi^3).

    Returns:
        dict met break-even analyse
    """
    if verbose:
        print("=" * 70)
        print("  BENCH 6: Break-Even Analyse (SV vs MPS)")
        print("=" * 70)

    n_range = list(range(4, 21, 2))  # 4 t/m 20
    chi_values = [8, 16, 32]
    p = 1

    results = []
    for n in n_range:
        edges = _make_random_graph(n, edge_prob=min(0.5, 6.0 / n))
        gammas = [0.5]
        betas = [0.3]
        qc = Circuit.qaoa_maxcut(n, edges, p=p, gammas=gammas, betas=betas)
        obs = {'C': Observable.maxcut_cost(edges)}

        # SV timing
        _, sv_time = _timed(run_circuit, qc, observables=obs,
                            backend='statevector')

        mps_times = {}
        for chi in chi_values:
            _, elapsed = _safe_run_circuit(
                qc, observables=obs, backend='mps', max_chi=chi,
                timeout=60.0)
            mps_times[chi] = elapsed

        entry = {
            'n': n, 'sv_time': sv_time,
            'mps_times': mps_times,
            'n_gates': len(qc),
        }
        results.append(entry)

        if verbose:
            mps_str = " | ".join(
                "chi=%d: %.4fs" % (c, t) for c, t in sorted(mps_times.items()))
            faster = [c for c, t in mps_times.items() if t < sv_time]
            marker = " <-- MPS wins (chi=%s)" % faster if faster else ""
            print("  n=%2d | SV: %.4fs | %s%s" % (n, sv_time, mps_str, marker))

    # Analyseer break-even
    break_even = {}
    for chi in chi_values:
        for entry in results:
            if entry['mps_times'].get(chi, float('inf')) < entry['sv_time']:
                if chi not in break_even:
                    break_even[chi] = entry['n']

    if verbose:
        print("\n  Break-even punten:")
        for chi, n_be in sorted(break_even.items()):
            print("    chi=%2d: MPS wint vanaf n=%d" % (chi, n_be))
        if not break_even:
            print("    SV wint voor alle geteste n (tot n=22)")

    return {
        'bench': 'break_even', 'p': p,
        'results': results,
        'break_even': break_even,
    }


# =====================================================================
# BENCHMARK 7: Large-Scale MPS Demo
# =====================================================================

def bench_large_scale(n_values=None, chi=16, verbose=True):
    """Demonstreer MPS scaling naar grote n (100-10000 qubits).

    Args:
        n_values: lijst van grote qubit-aantallen
        chi: vaste bond dimensie
        verbose: print resultaten

    Returns:
        dict met timing per n
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000, 5000]

    if verbose:
        print("=" * 70)
        print("  BENCH 7: Large-Scale MPS Demo (chi=%d)" % chi)
        print("  n: %s" % n_values)
        print("=" * 70)

    results = []
    for n in n_values:
        # 1D keten MaxCut QAOA p=1
        edges = _make_chain_graph(n)
        qc = Circuit.qaoa_maxcut(n, edges, p=1,
                                  gammas=[0.5], betas=[0.3])
        obs = {'C': Observable.maxcut_cost(edges)}

        result, elapsed = _safe_run_circuit(
            qc, observables=obs, backend='mps', max_chi=chi,
            timeout=300.0)

        if result is not None:
            energy = result['observables']['C']
            max_chi_reached = result.get('max_chi_reached', chi)
            # MaxCut ratio: energy / total_weight
            total_weight = sum(w for _, _, w in edges)
            ratio = energy / total_weight if total_weight > 0 else 0

            entry = {
                'n': n, 'chi': chi, 'time': elapsed,
                'energy': energy, 'ratio': ratio,
                'n_gates': len(qc), 'max_chi': max_chi_reached,
                'gates_per_sec': len(qc) / elapsed if elapsed > 0 else 0,
            }
        else:
            entry = {
                'n': n, 'chi': chi, 'time': elapsed,
                'energy': None, 'timeout': True,
            }

        results.append(entry)

        if verbose:
            if result is not None:
                print("  n=%5d | %.3fs | E=%.1f | ratio=%.3f | %d gates | %.0f gates/s" % (
                    n, elapsed, energy, ratio, len(qc),
                    entry.get('gates_per_sec', 0)))
            else:
                print("  n=%5d | TIMEOUT (%.1fs)" % (n, elapsed))

    return {'bench': 'large_scale', 'chi': chi, 'results': results}


# =====================================================================
# VOLLEDIGE SUITE
# =====================================================================

def run_full_suite(verbose=True):
    """Draai alle benchmarks en geef samenvatting.

    Returns:
        dict met alle benchmark-resultaten
    """
    t0 = time.time()
    all_results = {}

    # 1. QAOA Scaling
    all_results['qaoa_scaling'] = bench_qaoa_scaling(
        n_range=[4, 8, 12, 16, 20, 30, 50, 100],
        chi_values=[4, 8, 16, 32],
        verbose=verbose)

    if verbose:
        print()

    # 2. VQE Scaling
    all_results['vqe_scaling'] = bench_vqe_scaling(
        n_range=[4, 8, 12, 16, 20, 30, 50, 100],
        chi_values=[4, 8, 16, 32],
        verbose=verbose)

    if verbose:
        print()

    # 3. Trotter Scaling
    all_results['trotter_scaling'] = bench_trotter_scaling(
        n_range=[4, 8, 12, 16, 20, 30, 50],
        orders=[1, 2],
        chi_values=[4, 8, 16, 32],
        verbose=verbose)

    if verbose:
        print()

    # 4. Chi Convergentie
    all_results['chi_convergence'] = bench_chi_convergence(
        n=16, chi_range=[2, 4, 8, 16, 32, 64],
        verbose=verbose)

    if verbose:
        print()

    # 5. Circuit Complexiteit
    all_results['circuit_complexity'] = bench_circuit_complexity(
        n_range=[4, 8, 16, 32, 64, 128, 256],
        verbose=verbose)

    if verbose:
        print()

    # 6. Break-Even
    all_results['break_even'] = bench_break_even(verbose=verbose)

    if verbose:
        print()

    # 7. Large-Scale
    all_results['large_scale'] = bench_large_scale(
        n_values=[100, 200, 500, 1000, 2000],
        chi=16, verbose=verbose)

    total_time = time.time() - t0

    if verbose:
        print()
        print("=" * 70)
        print("  SAMENVATTING - B133 Scalability Benchmark Suite")
        print("=" * 70)

        # QAOA conclusie
        qaoa = all_results['qaoa_scaling']['results']
        max_n = max(r['n'] for r in qaoa if r.get('energy') is not None)
        print("  QAOA MaxCut: tot n=%d (MPS chi=32)" % max_n)

        # Break-even
        be = all_results['break_even']['break_even']
        if be:
            print("  Break-even SV/MPS: %s" % ", ".join(
                "chi=%d: n>=%d" % (c, n) for c, n in sorted(be.items())))
        else:
            print("  Break-even: SV wint voor n<=22")

        # Large-scale
        ls = all_results['large_scale']['results']
        largest = max((r['n'] for r in ls if r.get('energy') is not None),
                      default=0)
        if largest > 0:
            lr = [r for r in ls if r['n'] == largest][0]
            print("  Large-scale: n=%d in %.2fs (%.0f gates/s)" % (
                largest, lr['time'], lr.get('gates_per_sec', 0)))

        # Chi convergentie
        cc = all_results['chi_convergence']['qaoa_results']
        converged = [r for r in cc if r.get('rel_error') is not None
                     and r['rel_error'] < 1e-6]
        if converged:
            min_chi = min(r['chi'] for r in converged)
            print("  Chi convergentie (n=16 QAOA): exact bij chi>=%d" % min_chi)

        print("  Totale benchmark-tijd: %.1fs" % total_time)

    all_results['total_time'] = total_time
    return all_results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    run_full_suite(verbose=True)
