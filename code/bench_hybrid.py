#!/usr/bin/env python3
"""
Benchmark: Hybrid QAOA+Classical vs Pure Classical MaxCut Solver

Tests the quantum advantage of QAOA-informed spanning trees on pm1 Ising instances.

Results (April 2026):
  Grid 10x4 pm1  (n=40):   advantage = +0
  Grid 15x6 pm1  (n=90):   advantage = +0
  Grid 20x8 pm1  (n=160):  advantage = +2
  Grid 25x10 pm1 (n=250):  advantage = +6
  Multi-seed 20x8 pm1:     avg=+2.5, wins=3/5

Usage: python bench_hybrid.py [--large]
"""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '.')

from hybrid_qaoa_solver import hybrid_qaoa_maxcut
import numpy as np


def make_grid(Lx, Ly, signed=False, seed=42):
    rng = np.random.default_rng(seed)
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                w = rng.choice([-1.0, 1.0]) if signed else 1.0
                edges.append((i, (x+1)*Ly + y, w))
            if y + 1 < Ly:
                w = rng.choice([-1.0, 1.0]) if signed else 1.0
                edges.append((i, x*Ly + y+1, w))
    return n, edges


if __name__ == '__main__':
    large = '--large' in sys.argv

    print("Hybrid QAOA+Classical vs Pure Classical MaxCut")
    print("=" * 75)
    fmt_h = "%20s %5s %5s %10s %10s %10s %6s"
    print(fmt_h % ("Graph", "n", "m", "Classical", "Hybrid", "Advntge", "time"))
    print("-" * 75)

    configs = [
        ("Grid 10x4", 10, 4, False),
        ("Grid 10x4 pm1", 10, 4, True),
        ("Grid 15x6", 15, 6, False),
        ("Grid 15x6 pm1", 15, 6, True),
        ("Grid 20x8", 20, 8, False),
        ("Grid 20x8 pm1", 20, 8, True),
    ]
    if large:
        configs.append(("Grid 25x10 pm1", 25, 10, True))

    for gname, lx, ly, signed in configs:
        n, edges = make_grid(lx, ly, signed=signed)
        t0 = time.time()
        cut, assign, info = hybrid_qaoa_maxcut(
            n, edges, p=1, time_limit=20, seed=42, n_trees=10, verbose=False)
        elapsed = time.time() - t0
        cl = info['classical_cut']
        hy = info['hybrid_cut']
        adv = info['quantum_advantage']
        fmt_r = "%20s %5d %5d %10.1f %10.1f %+10.1f %5.1fs"
        print(fmt_r % (gname, n, len(edges), cl, hy, adv, elapsed), flush=True)

    print("=" * 75)
    print("Advantage = hybrid - classical  (positive = QAOA helps)")
    print("Quantum advantage shows on pm1 (+-1 Ising) instances.")
