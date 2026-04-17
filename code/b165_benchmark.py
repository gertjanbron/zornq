#!/usr/bin/env python3
"""B165 Benchmark: Aer vs Noisy-Aer vs Hardware (skipped) over diverse grafen.

Vergelijkt voor elke graaf:
  - OPT (brute force)
  - Aer (geen ruis): E[H_C] en best-cut
  - Noisy-Aer (p1=1e-3, p2=1e-2): E[H_C] en best-cut
  - Hardware: SKIPPED_NO_TOKEN (geen IBM-token in sandbox)

Met identieke QAOA p=1 parameters (γ=0.7, β=0.4 — heuristisch goed voor MaxCut).
"""

from __future__ import annotations

import os
import time

from b60_gw_bound import SimpleGraph, brute_force_maxcut, random_3regular
from b156_sos2_sdp import (
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    complete_bipartite,
)
from b165_qiskit_runtime import qaoa_maxcut_run


def run() -> None:
    instances: list[tuple[str, SimpleGraph]] = [
        ("K_3 (triangle)",   complete_graph(3)),
        ("K_4",              complete_graph(4)),
        ("K_5",              complete_graph(5)),
        ("K_3,3 bipart",     complete_bipartite(3, 3)),
        ("C_4 (even)",       cycle_graph(4)),
        ("C_5 (odd)",        cycle_graph(5)),
        ("P_5 (path)",       path_graph(5)),
        ("Petersen",         petersen_graph()),
        ("3-reg n=8",        random_3regular(8, seed=42)),
    ]

    # Geen IBM-token — hardware-call moet SKIPPEN
    os.environ.pop("QISKIT_IBM_TOKEN", None)

    print("=" * 110)
    print(" B165 Benchmark: Aer vs Noisy-Aer vs Hardware (QAOA p=1, γ=0.7, β=0.4)")
    print("=" * 110)
    header = (
        f"{'Instance':<18}{'n':>3}{'m':>4}{'OPT':>5}"
        f"{'Aer_E':>10}{'Aer_best':>10}"
        f"{'Noisy_E':>10}{'Noisy_best':>12}"
        f"{'HW':>22}{'A_t':>7}{'N_t':>7}"
    )
    print(header)
    print("-" * len(header))

    n_aer_best = 0
    n_noisy_best = 0
    n_hw_skipped = 0
    n_total = 0

    for name, g in instances:
        opt = brute_force_maxcut(g) if g.n <= 18 else None

        t0 = time.time()
        aer = qaoa_maxcut_run(g, p=1, gammas=[0.7], betas=[0.4],
                               backend="aer", shots=4096, seed=42, verbose=False)
        a_t = time.time() - t0

        t0 = time.time()
        noi = qaoa_maxcut_run(g, p=1, gammas=[0.7], betas=[0.4],
                               backend="noisy", shots=4096, seed=42,
                               p1err=1e-3, p2err=1e-2, verbose=False)
        n_t = time.time() - t0

        hw = qaoa_maxcut_run(g, p=1, gammas=[0.7], betas=[0.4],
                              backend="hardware", shots=512,
                              ibm_backend="ibm_brisbane", verbose=False)
        hw_status = hw.get("status", "OK")

        opt_str = f"{opt:.0f}" if opt is not None else "—"
        a_e = f"{aer['qaoa_expectation']:.3f}"
        a_b = f"{aer['best_cut_seen']}/{opt:.0f}" if opt else f"{aer['best_cut_seen']}"
        n_e = f"{noi['qaoa_expectation']:.3f}"
        n_b = f"{noi['best_cut_seen']}/{opt:.0f}" if opt else f"{noi['best_cut_seen']}"

        print(f"{name:<18}{g.n:>3}{g.n_edges:>4}{opt_str:>5}"
              f"{a_e:>10}{a_b:>10}"
              f"{n_e:>10}{n_b:>12}"
              f"{hw_status:>22}{a_t:>7.2f}{n_t:>7.2f}")

        if opt and aer["best_cut_seen"] == opt:
            n_aer_best += 1
        if opt and noi["best_cut_seen"] == opt:
            n_noisy_best += 1
        if hw_status == "SKIPPED_NO_TOKEN":
            n_hw_skipped += 1
        if opt:
            n_total += 1

    print("-" * len(header))
    print(f"\n  Aer    (geen ruis) bereikt OPT (best-cut) op {n_aer_best}/{n_total} grafen")
    print(f"  Noisy  (p1=1e-3, p2=1e-2) bereikt OPT op       {n_noisy_best}/{n_total} grafen")
    print(f"  Hardware-pad: {n_hw_skipped}/{len(instances)} correct geSKIPPED zonder token")
    print()
    print("  → Set $QISKIT_IBM_TOKEN en kies een echte backend voor hardware-runs.")


if __name__ == "__main__":
    run()
