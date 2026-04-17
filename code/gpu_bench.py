#!/usr/bin/env python3
"""
ZornQ GPU Benchmark
===================
Draai dit op je machine met GPU om de speedup te meten.

Gebruik:
    python gpu_bench.py              # auto-detect GPU/CPU
    python gpu_bench.py --cpu-only   # forceer CPU voor vergelijking

Vereisten:
    pip install cupy-cuda12x   (of cupy-cuda11x voor oudere CUDA)
"""
import sys
import time
import numpy as np

# Forceer CPU als --cpu-only flag
force_cpu = '--cpu-only' in sys.argv

if force_cpu:
    import gpu_backend
    gpu_backend.GPU_AVAILABLE = False
    gpu_backend.xp = np
    print("MODUS: CPU (numpy) — geforceerd via --cpu-only\n")

from gpu_backend import (xp, xp_svd, xp_einsum, xp_rsvd,
                         to_device, to_numpy, sync,
                         GPU_AVAILABLE, GPU_NAME, gpu_memory_info)
from zorn_mps import HeisenbergQAOA


def banner():
    print("=" * 65)
    print("ZornQ GPU BENCHMARK")
    print("=" * 65)
    if GPU_AVAILABLE:
        print(f"  GPU:    {GPU_NAME}")
        mem = gpu_memory_info()
        if mem:
            print(f"  VRAM:   {mem[0]:.0f} / {mem[1]:.0f} MB")
        print(f"  Backend: cupy (GPU)")
    else:
        print(f"  Backend: numpy (CPU)")
    print()


def bench_svd():
    """SVD benchmark voor verschillende matrixgroottes."""
    print("--- SVD Benchmark (complex128) ---")
    sizes = [64, 128, 256, 512, 1024]
    for n in sizes:
        A = xp.random.randn(n, n).astype(complex) + 1j * xp.random.randn(n, n)

        # Warmup
        xp_svd(A, full_matrices=False)
        sync()

        # Timed (5 runs)
        t0 = time.time()
        for _ in range(5):
            xp_svd(A, full_matrices=False)
        sync()
        dt = (time.time() - t0) / 5
        print(f"  {n:>5}x{n}: {dt*1000:8.2f} ms")
    print()


def bench_einsum():
    """Einsum benchmark (2-site tensor contraction)."""
    print("--- Einsum Benchmark (2-site merge) ---")
    configs = [(16, 8), (32, 8), (64, 8), (128, 8), (64, 16)]
    for chi, d in configs:
        A = xp.random.randn(chi, d, chi).astype(complex)
        B = xp.random.randn(chi, d, chi).astype(complex)

        # Warmup
        xp_einsum('aib,bjc->aijc', A, B)
        sync()

        # Timed (20 runs)
        t0 = time.time()
        for _ in range(20):
            xp_einsum('aib,bjc->aijc', A, B)
        sync()
        dt = (time.time() - t0) / 20
        print(f"  chi={chi:>4}, d={d:>2}: {dt*1000:8.2f} ms")
    print()


def bench_rsvd():
    """Randomized SVD benchmark."""
    print("--- rSVD Benchmark (rank-k approx) ---")
    configs = [(512, 64), (1024, 64), (1024, 128)]
    for n, k in configs:
        M = xp.random.randn(n, n).astype(complex)

        # Warmup
        xp_rsvd(M, k)
        sync()

        # Timed
        t0 = time.time()
        for _ in range(5):
            xp_rsvd(M, k)
        sync()
        dt = (time.time() - t0) / 5
        print(f"  {n}x{n} -> rank {k}: {dt*1000:8.2f} ms")
    print()


def bench_qaoa():
    """End-to-end QAOA benchmark."""
    print("--- QAOA End-to-End Benchmark ---")
    configs = [
        (10, 1, 8,  "10 qubits, 1D, chi=8"),
        (20, 1, 16, "20 qubits, 1D, chi=16"),
        (50, 1, 32, "50 qubits, 1D, chi=32"),
        (4, 3, 16,  "4x3=12 qubits, 2D, chi=16"),
        (5, 4, 32,  "5x4=20 qubits, 2D, chi=32"),
    ]
    gamma, beta = 0.4021, 1.1778

    for Lx, Ly, chi, label in configs:
        try:
            t0 = time.time()
            eng = HeisenbergQAOA(Lx=Lx, Ly=Ly, max_chi=chi,
                                 gpu=GPU_AVAILABLE)
            if Ly > 1:
                n_e = Lx*(Ly-1) + (Lx-1)*Ly
                avg_d = 2*n_e/(Lx*Ly)
                g = 0.88/avg_d
            else:
                g = gamma
            ratio = eng.eval_ratio(1, [g], [beta])
            dt = time.time() - t0
            print(f"  {label:>35}: ratio={ratio:.4f}  {dt*1000:8.1f} ms")
        except Exception as e:
            print(f"  {label:>35}: FOUT — {e}")
    print()


if __name__ == '__main__':
    banner()
    bench_svd()
    bench_einsum()
    bench_rsvd()
    bench_qaoa()

    print("=" * 65)
    if GPU_AVAILABLE:
        print("TIP: Draai ook met --cpu-only om de speedup te vergelijken:")
        print("     python gpu_bench.py --cpu-only")
    else:
        print("TIP: Installeer cupy voor GPU-acceleratie:")
        print("     pip install cupy-cuda12x")
    print("=" * 65)
