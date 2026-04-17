#!/usr/bin/env python3
"""B80 Benchmark: MPQS (BP + Lightcone) vs ILP-oracle vs GW SDP.

Laat voor elke graaf zien:
  - n, m
  - ILP-OPT (HiGHS certified, fallback brute force)
  - GW SDP upper bound
  - MPQS-BP cut value + ratio t.o.v. ILP
  - MPQS-Lightcone cut + ratio t.o.v. ILP
  - Wall-times

Doel: demonstreren dat beide MPQS-paden consistente cuts produceren, met de
lightcone variant als kwantum-geïnspireerde benadering en BP als klassiek
benchmark.
"""

from __future__ import annotations

import time

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    gw_sdp_bound,
    random_3regular,
)
from b156_sos2_sdp import (
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    complete_bipartite,
)
from b159_ilp_oracle import maxcut_ilp_highs
from b80_mpqs import mpqs_classical_bp, mpqs_lightcone


def _star_graph(n_leaves: int) -> SimpleGraph:
    g = SimpleGraph(n_leaves + 1)
    for i in range(1, n_leaves + 1):
        g.add_edge(0, i)
    return g


def run() -> None:
    instances: list[tuple[str, SimpleGraph]] = [
        ("K_3",           complete_graph(3)),
        ("K_4",           complete_graph(4)),
        ("K_5",           complete_graph(5)),
        ("K_3,3",         complete_bipartite(3, 3)),
        ("C_5",           cycle_graph(5)),
        ("C_7",           cycle_graph(7)),
        ("C_8",           cycle_graph(8)),
        ("P_8",           path_graph(8)),
        ("Star_5",        _star_graph(5)),
        ("Petersen",      petersen_graph()),
        ("3-reg n=10",    random_3regular(10, seed=42)),
        ("3-reg n=12",    random_3regular(12, seed=7)),
        ("3-reg n=14",    random_3regular(14, seed=3)),
    ]

    print("=" * 120)
    print(" B80 Benchmark: MPQS (Classical BP + Lightcone-QAOA) vs ILP-oracle")
    print("=" * 120)
    header = (
        f"{'Instance':<14}{'n':>3}{'m':>4}"
        f"{'OPT':>6}{'GW':>9}"
        f"{'BP':>6}{'r_BP':>7}{'t_BP':>7}"
        f"{'LC':>6}{'r_LC':>7}{'t_LC':>7}"
    )
    print(header)
    print("-" * len(header))

    n_bp_opt = 0
    n_lc_opt = 0
    total_tree = 0
    n_bp_tree_opt = 0
    n_instances = 0

    for name, g in instances:
        n_instances += 1
        # ILP-OPT
        ilp = maxcut_ilp_highs(g, time_limit=10.0)
        opt = ilp["opt_value"]

        # GW
        try:
            gw = gw_sdp_bound(g, verbose=False).get("sdp_bound")
        except Exception:
            gw = None

        # MPQS-BP
        t0 = time.time()
        bp_res = mpqs_classical_bp(g, verbose=False)
        t_bp = time.time() - t0
        cut_bp = bp_res["cut_value"]

        # MPQS-Lightcone
        t0 = time.time()
        lc_res = mpqs_lightcone(g, radius=2, verbose=False)
        t_lc = time.time() - t0
        cut_lc = lc_res["cut_value"]

        r_bp = cut_bp / opt if opt and opt > 0 else float("nan")
        r_lc = cut_lc / opt if opt and opt > 0 else float("nan")

        gw_s = f"{gw:.2f}" if gw is not None else "—"
        opt_s = f"{opt:.0f}" if opt is not None else "—"

        print(f"{name:<14}{g.n:>3}{g.n_edges:>4}"
              f"{opt_s:>6}{gw_s:>9}"
              f"{cut_bp:>6.1f}{r_bp:>7.3f}{t_bp:>7.3f}"
              f"{cut_lc:>6.1f}{r_lc:>7.3f}{t_lc:>7.3f}")

        if opt is not None:
            if abs(cut_bp - opt) < 1e-6:
                n_bp_opt += 1
            if abs(cut_lc - opt) < 1e-6:
                n_lc_opt += 1

        # Detecteer boom (tree): n - 1 edges én verbonden → MPQS-BP moet exact zijn
        if g.n_edges == g.n - 1:
            total_tree += 1
            if opt is not None and abs(cut_bp - opt) < 1e-6:
                n_bp_tree_opt += 1

    print("-" * len(header))
    print()
    print(f"  MPQS-BP       exact (= ILP-OPT):        {n_bp_opt}/{n_instances}")
    print(f"  MPQS-Lightcone exact (= ILP-OPT):        {n_lc_opt}/{n_instances}")
    if total_tree > 0:
        print(f"  MPQS-BP exact op bomen:                  {n_bp_tree_opt}/{total_tree}  (theoretisch {total_tree}/{total_tree})")
    print()
    print("  r_*: cut-value / ILP-OPT (1.000 = exact, <1 = onder-optimaal)")
    print("  t_*: wall-time in seconden")
    print("  BP op bomen = tree-exact;  op loopy: heuristiek + greedy-1-flip")
    print("  Lightcone = lokale p=1 QAOA met γ=0.3, β=0.2, radius=2")


if __name__ == "__main__":
    run()
