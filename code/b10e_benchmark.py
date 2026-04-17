#!/usr/bin/env python3
"""B10e benchmark — schaal PEPS QAOA op 2D grids.

Tabulate voor een reeks (Lx, Ly, p):
  - exact state-vector energie (n <= 12 qubits)
  - PEPS energie bij chi_max, chi_b
  - wall-time voor beide
  - verschil (diff)
  - maximale PEPS bond-dim na QAOA

Doel: laten zien dat PEPS schaalbaar matcht op kleine grids, en waar de
column-grouped MPS breekt (Ly >= 4, d >= 16) heeft PEPS nog bandbreedte
door de 2D topologie direct te representeren.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b10e_pepo import (
    peps_qaoa_maxcut,
    exact_qaoa_maxcut,
    grid_edges_flat,
    PEPS2D,
)


def run(cases, chi_max=4, chi_b=16):
    print("=" * 98)
    print(" B10e PEPS-QAOA Benchmark — 2D MaxCut op vierkante roosters")
    print(f"   chi_max = {chi_max},  chi_b = {chi_b}")
    print("=" * 98)
    hdr = (
        f"{'Lx':>3}{'Ly':>4}{'n':>4}{'p':>3}"
        f"{'E_exact':>12}{'t_exact(s)':>12}"
        f"{'E_peps':>12}{'t_peps(s)':>12}"
        f"{'diff':>12}{'chi_out':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (Lx, Ly, p, gammas, betas) in cases:
        n = Lx * Ly

        # Exact
        t0 = time.time()
        try:
            if n <= 14:
                eflat = grid_edges_flat(Lx, Ly)
                _, E_exact = exact_qaoa_maxcut(Lx, Ly, eflat, gammas, betas)
                t_exact = time.time() - t0
                exact_str = f"{E_exact:.6f}"
                t_exact_str = f"{t_exact:.3f}"
            else:
                E_exact = float("nan")
                t_exact = float("nan")
                exact_str = "—"
                t_exact_str = "—"
        except MemoryError:
            E_exact = float("nan")
            t_exact = float("nan")
            exact_str = "OOM"
            t_exact_str = "—"

        # PEPS
        t0 = time.time()
        peps, E_peps = peps_qaoa_maxcut(
            Lx, Ly, gammas, betas, chi_max=chi_max, chi_b=chi_b,
        )
        t_peps = time.time() - t0
        chi_out = peps.max_bond_dim()
        peps_str = f"{E_peps:.6f}"
        t_peps_str = f"{t_peps:.3f}"

        if not (exact_str == "—" or exact_str == "OOM"):
            diff = abs(E_peps - E_exact)
            diff_str = f"{diff:.2e}"
        else:
            diff_str = "—"

        print(f"{Lx:>3}{Ly:>4}{n:>4}{p:>3}"
              f"{exact_str:>12}{t_exact_str:>12}"
              f"{peps_str:>12}{t_peps_str:>12}"
              f"{diff_str:>12}{chi_out:>8}")

    print("-" * len(hdr))


def main():
    # Oplopende moeilijkheid: kleine grids → exact vergelijkbaar; middelgroot → alleen PEPS
    cases = [
        # (Lx, Ly, p, gammas, betas)
        (2, 2, 1, [0.4],          [0.3]),
        (2, 2, 2, [0.4, 0.2],     [0.3, 0.1]),
        (3, 2, 1, [0.4],          [0.3]),
        (3, 2, 2, [0.4, 0.2],     [0.3, 0.1]),
        (3, 3, 1, [0.4],          [0.3]),
        (3, 3, 2, [0.4, 0.2],     [0.3, 0.1]),
        (2, 4, 1, [0.4],          [0.3]),
        (3, 4, 1, [0.4],          [0.3]),
    ]
    run(cases, chi_max=4, chi_b=16)


if __name__ == "__main__":
    main()
