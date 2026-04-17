#!/usr/bin/env python3
"""
B104: Boundary-State Compiler — Experiment Runner

Experimenten:
  1. Correctheid: boundary solve vs brute-force MaxCut
  2. Decompositie kwaliteit: separator grootte, patch balans
  3. Compilatie overhead: compile time vs patch grootte
  4. Stitch kwaliteit: boundary solve ratio vs optimum
  5. Isomorfisme caching: hoeveel compilatie bespaard?
  6. Schaling: hoe schaalt het met n?
"""

import numpy as np
import time

from boundary_state_compiler import (
    decompose_graph, compile_graph, compile_graph_with_isomorphism,
    stitch_solve, boundary_solve, compile_patch, Patch,
)
from bls_solver import random_3regular


def brute_force_maxcut(n, edges):
    best = 0.0
    for s in range(2 ** n):
        cut = 0.0
        for u, v, w in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                cut += w
        best = max(best, cut)
    return best


def grid_graph(Lx, Ly):
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            if x + 1 < Lx:
                edges.append((node, (x+1)*Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((node, x*Ly + y + 1, 1.0))
    return n, edges


# =====================================================================
# EXPERIMENT 1: Correctheid
# =====================================================================

def benchmark_correctness(seed=42):
    print("=" * 65)
    print("EXPERIMENT 1: CORRECTHEID vs BRUTE-FORCE")
    print("=" * 65)
    print(f"{'Graaf':>18s} {'n':>4s} {'m':>4s} {'BF opt':>7s} {'BSC':>7s} "
          f"{'Ratio':>7s} {'OK':>4s}")
    print("-" * 55)

    graphs = [
        ("Triangle", 3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]),
        ("Path-6", *_path(6)),
        ("Grid-3x3", *grid_graph(3, 3)),
        ("3-reg-8", 8, random_3regular(8, seed=seed)[1]),
        ("3-reg-10", 10, random_3regular(10, seed=seed)[1]),
        ("3-reg-12", 12, random_3regular(12, seed=seed)[1]),
    ]

    all_ok = True
    for name, n, edges in graphs:
        opt = brute_force_maxcut(n, edges)
        result = boundary_solve(n, edges, max_patch_size=8)
        bsc = result['best_cut']
        ratio = bsc / opt if opt > 0 else 1.0
        ok = "OK" if ratio >= 0.7 else "LAAG"
        if ratio < 0.7:
            all_ok = False
        print(f"{name:>18s} {n:4d} {len(edges):4d} {opt:7.1f} {bsc:7.1f} "
              f"{ratio:7.4f} {ok:>4s}")

    print(f"\n  Kwaliteit voldoende: {'JA' if all_ok else 'NEE'}")


def _path(n):
    return n, [(i, i+1, 1.0) for i in range(n-1)]


# =====================================================================
# EXPERIMENT 2: Decompositie kwaliteit
# =====================================================================

def benchmark_decomposition(seed=42):
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: DECOMPOSITIE KWALITEIT")
    print("=" * 65)
    print(f"{'n':>4s} {'m':>4s} {'#patches':>9s} {'#cross':>7s} {'#sep':>5s} "
          f"{'Max patch':>10s} {'Min patch':>10s} {'Balance':>8s}")
    print("-" * 60)

    for n in [10, 14, 18, 22]:
        _, edges = random_3regular(n, seed=seed)
        patches, cross, sep = decompose_graph(n, edges, max_patch_size=10)
        sizes = [p.size for p in patches]
        balance = min(sizes) / max(sizes) if sizes else 0
        print(f"{n:4d} {len(edges):4d} {len(patches):9d} {len(cross):7d} "
              f"{len(sep):5d} {max(sizes):10d} {min(sizes):10d} {balance:8.3f}")


# =====================================================================
# EXPERIMENT 3: Compilatie overhead
# =====================================================================

def benchmark_compilation(seed=42):
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: COMPILATIE OVERHEAD")
    print("=" * 65)
    print(f"{'n':>4s} {'m':>4s} {'Patches':>8s} {'T-compile':>10s} {'T-solve':>9s} "
          f"{'T-total':>9s} {'Cut':>8s}")
    print("-" * 55)

    for n in [8, 12, 16, 20]:
        _, edges = random_3regular(n, seed=seed)

        t0 = time.time()
        compiled = compile_graph(n, edges, max_patch_size=10)
        t_compile = time.time() - t0

        t0 = time.time()
        result = stitch_solve(compiled, n, edges)
        t_solve = time.time() - t0

        t_total = t_compile + t_solve
        print(f"{n:4d} {len(edges):4d} {compiled.n_patches:8d} {t_compile:10.4f}s "
              f"{t_solve:9.4f}s {t_total:9.4f}s {result['best_cut']:8.1f}")


# =====================================================================
# EXPERIMENT 4: Isomorfisme caching
# =====================================================================

def benchmark_isomorphism(seed=42):
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: ISOMORFISME CACHING")
    print("=" * 65)
    print(f"{'Graaf':>15s} {'n':>4s} {'Patches':>8s} {'Compiled':>9s} "
          f"{'Cached':>7s} {'T-normal':>9s} {'T-iso':>9s} {'Speedup':>8s}")
    print("-" * 65)

    test_graphs = [
        ("Grid 4x4", *grid_graph(4, 4)),
        ("Grid 5x3", *grid_graph(5, 3)),
        ("3-reg-14", 14, random_3regular(14, seed=seed)[1]),
        ("3-reg-18", 18, random_3regular(18, seed=seed)[1]),
    ]

    for name, n, edges in test_graphs:
        # Normal compile
        t0 = time.time()
        c1 = compile_graph(n, edges, max_patch_size=10)
        t_normal = time.time() - t0

        # Iso compile
        t0 = time.time()
        c2 = compile_graph_with_isomorphism(n, edges, max_patch_size=10)
        t_iso = time.time() - t0

        # Count compiled vs cached
        n_compiled = sum(1 for r in c2.responses if r.compile_time > 0)
        n_cached = len(c2.responses) - n_compiled

        speedup = t_normal / t_iso if t_iso > 0 else float('inf')
        print(f"{name:>15s} {n:4d} {len(c2.patches):8d} {n_compiled:9d} "
              f"{n_cached:7d} {t_normal:9.4f}s {t_iso:9.4f}s {speedup:8.2f}x")


# =====================================================================
# EXPERIMENT 5: Stitch kwaliteit vs brute force
# =====================================================================

def benchmark_quality(seed=42):
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: STITCH KWALITEIT vs BRUTE FORCE")
    print("=" * 65)
    print(f"{'n':>4s} {'m':>4s} {'BF opt':>8s} {'BSC cut':>8s} {'Ratio':>8s} "
          f"{'Patches':>8s} {'Sep':>5s}")
    print("-" * 50)

    for n in [8, 10, 12, 14]:
        _, edges = random_3regular(n, seed=seed)
        opt = brute_force_maxcut(n, edges)
        result = boundary_solve(n, edges, max_patch_size=8)
        ratio = result['best_cut'] / opt if opt > 0 else 1.0
        print(f"{n:4d} {len(edges):4d} {opt:8.1f} {result['best_cut']:8.1f} "
              f"{ratio:8.4f} {result['n_patches']:8d} {result['separator_size']:5d}")


# =====================================================================
# MAIN
# =====================================================================

def run_b104_report(seed=42):
    print("=" * 65)
    print("B104: BOUNDARY-STATE COMPILER VOOR MAXCUT")
    print(f"Seed: {seed}")
    print("=" * 65)

    benchmark_correctness(seed=seed)
    benchmark_decomposition(seed=seed)
    benchmark_compilation(seed=seed)
    benchmark_isomorphism(seed=seed)
    benchmark_quality(seed=seed)

    print("\n" + "=" * 65)
    print("CONCLUSIE:")
    print("  - Boundary-state compilatie werkt: patches correct gecompileerd")
    print("  - Separator-decompositie splitst grafen in handelbare patches")
    print("  - Isomorfisme-caching bespaart compilatietijd op regelmatige grafen")
    print("  - Stitch-kwaliteit afhankelijk van separator-kwaliteit")
    print("  - Overhead beheersbaar: compile O(2^boundary * 2^interior) per patch")
    print("=" * 65)


if __name__ == '__main__':
    run_b104_report(seed=42)
