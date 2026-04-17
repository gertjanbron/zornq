#!/usr/bin/env python3
"""
B15 Dynamische Truncatie — Vergelijkende Benchmark
===================================================
Meet het verschil tussen vaste chi en adaptieve truncatie
op dezelfde QAOA-systemen.

Gebruik:
    python b15_bench.py
"""
import time
import sys

# Forceer CPU als --cpu-only
if '--cpu-only' in sys.argv:
    try:
        import gpu_backend
        gpu_backend.GPU_AVAILABLE = False
        import numpy as np
        gpu_backend.xp = np
    except ImportError:
        pass

from zorn_mps import HeisenbergQAOA

try:
    from gpu_backend import GPU_AVAILABLE, GPU_NAME
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "none"


def run_qaoa(label, Lx, Ly, max_chi, min_weight, p=1):
    """Draai QAOA en meet ratio + tijd."""
    gamma_base, beta = 0.4021, 1.1778

    t0 = time.time()
    eng = HeisenbergQAOA(Lx=Lx, Ly=Ly, max_chi=max_chi,
                         gpu=GPU_AVAILABLE, min_weight=min_weight)

    if Ly > 1:
        n_e = Lx * (Ly - 1) + (Lx - 1) * Ly
        avg_d = 2 * n_e / (Lx * Ly)
        gamma = 0.88 / avg_d
    else:
        gamma = gamma_base

    gammas = [gamma] * p
    betas = [beta] * p

    ratio = eng.eval_ratio(p, gammas, betas)
    dt = time.time() - t0
    return ratio, dt


def bench_group(title, Lx, Ly, chi_values, eps_values, p=1):
    """Vergelijk vast vs adaptief voor een groep configuraties."""
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

    results = []

    # Eerst vaste chi runs
    for chi in chi_values:
        label = f"vast chi={chi}"
        ratio, dt = run_qaoa(label, Lx, Ly, chi, None, p)
        results.append((label, ratio, dt))
        print(f"  {label:>30s}: ratio={ratio:.5f}  {dt*1000:9.1f}ms")

    # Dan adaptief met verschillende epsilon
    baseline_dt = results[0][2]  # eerste vaste chi als baseline

    for eps in eps_values:
        max_chi = max(chi_values)  # gebruik hoogste chi als plafond
        label = f"adapt eps={eps:.0e} (cap={max_chi})"
        ratio, dt = run_qaoa(label, Lx, Ly, max_chi, eps, p)
        speedup = baseline_dt / dt if dt > 0 else float('inf')
        ratio_diff = abs(ratio - results[0][1])
        results.append((label, ratio, dt))
        print(f"  {label:>30s}: ratio={ratio:.5f}  {dt*1000:9.1f}ms"
              f"  ({speedup:.1f}x vs vast, delta={ratio_diff:.1e})")

    return results


if __name__ == '__main__':
    print("=" * 65)
    print("B15 DYNAMISCHE TRUNCATIE — BENCHMARK")
    print("=" * 65)
    print(f"  GPU: {'%s (%s)' % (GPU_NAME, 'actief') if GPU_AVAILABLE else 'uit (CPU/numpy)'}")

    # ---------------------------------------------------------------
    # 1D tests: hier is chi al laag, verschil is klein
    # ---------------------------------------------------------------
    bench_group(
        "1D QAOA, 20 qubits, p=1",
        Lx=20, Ly=1,
        chi_values=[16, 32],
        eps_values=[1e-6, 1e-3, 0.05]
    )

    bench_group(
        "1D QAOA, 50 qubits, p=1",
        Lx=50, Ly=1,
        chi_values=[32],
        eps_values=[1e-6, 1e-3, 0.05]
    )

    # ---------------------------------------------------------------
    # 2D tests: hier groeit chi, adaptief maakt het verschil
    # ---------------------------------------------------------------
    bench_group(
        "2D QAOA, 4x3=12 qubits, p=1",
        Lx=4, Ly=3,
        chi_values=[16, 32],
        eps_values=[1e-6, 1e-3, 0.01, 0.05]
    )

    bench_group(
        "2D QAOA, 5x3=15 qubits, p=1",
        Lx=5, Ly=3,
        chi_values=[32],
        eps_values=[1e-6, 1e-3, 0.01, 0.05]
    )

    # De grote test: 5x4 — dit is waar het echt telt
    print("\n" + "!" * 65)
    print("  HOOFDTEST: 5x4=20 qubits, 2D, p=1")
    print("  (dit is de 21-seconden case van de CPU benchmark)")
    print("!" * 65)

    bench_group(
        "2D QAOA, 5x4=20 qubits, p=1",
        Lx=5, Ly=4,
        chi_values=[32],
        eps_values=[1e-6, 1e-3, 0.01, 0.05]
    )

    # ---------------------------------------------------------------
    # p=2 test: diepere circuits, meer verstrengeling
    # ---------------------------------------------------------------
    bench_group(
        "2D QAOA, 4x3=12 qubits, p=2",
        Lx=4, Ly=3,
        chi_values=[32],
        eps_values=[1e-6, 1e-3, 0.05],
        p=2
    )

    print("\n" + "=" * 65)
    print("SAMENVATTING")
    print("=" * 65)
    print("  min_weight=None  : vaste chi (oud gedrag)")
    print("  min_weight=1e-6  : hoge precisie, chi krimpt alleen bij ~nul verstrengeling")
    print("  min_weight=1e-3  : goede balans precisie/snelheid")
    print("  min_weight=0.05  : agressief, maximale snelheid, kleine fidelity-afwijking mogelijk")
    print()
    print("  Aanbevolen: min_weight=1e-3 voor dagelijks gebruik")
    print("              min_weight=1e-6 voor publicatie-nauwkeurigheid")
    print("=" * 65)
