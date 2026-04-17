#!/usr/bin/env python3
"""B158 Benchmark: GW vs LP_triangle vs LP+OddCycle vs SoS-2 (B156).

Tabel-vergelijking over diverse grafen.
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
    sos2_sdp_bound,
)
from b158_cutting_planes import (
    lp_triangle_bound,
    lp_triangle_oddcycle_bound,
)


def run() -> None:
    instances: list[tuple[str, SimpleGraph]] = [
        ("K_3 (triangle)",   complete_graph(3)),
        ("K_4",              complete_graph(4)),
        ("K_5",              complete_graph(5)),
        ("K_6",              complete_graph(6)),
        ("K_3,3 bipart",     complete_bipartite(3, 3)),
        ("K_2,5 bipart",     complete_bipartite(2, 5)),
        ("C_5 (odd)",        cycle_graph(5)),
        ("C_7 (odd)",        cycle_graph(7)),
        ("C_8 (even)",       cycle_graph(8)),
        ("P_8 (path)",       path_graph(8)),
        ("Petersen",         petersen_graph()),
        ("3-reg n=10",       random_3regular(10, seed=42)),
        ("3-reg n=12",       random_3regular(12, seed=7)),
        ("3-reg n=14",       random_3regular(14, seed=3)),
    ]

    print("=" * 110)
    print(" B158 Benchmark: GW (SDP-1) vs LP_triangle vs LP+OddCycle vs SoS-2 (Lasserre-2)")
    print("=" * 110)
    header = (
        f"{'Instance':<18}{'n':>3}{'m':>4}{'OPT':>5}{'GW':>9}"
        f"{'LP_tri':>9}{'LP+OC':>9}{'SoS-2':>9}"
        f"{'cuts':>5}{'GW_t':>7}{'LPt':>7}{'OC_t':>7}{'S2_t':>7}"
    )
    print(header)
    print("-" * len(header))

    n_lpoc_exact = 0
    n_total_with_opt = 0

    for name, g in instances:
        opt = brute_force_maxcut(g) if g.n <= 18 else None

        t0 = time.time()
        gw = gw_sdp_bound(g, verbose=False).get("sdp_bound")
        gw_t = time.time() - t0

        r1 = lp_triangle_bound(g, extend=True, verbose=False)
        lp1 = r1.get("lp_bound")
        lp1_t = r1.get("solve_time")

        r2 = lp_triangle_oddcycle_bound(g, extend=True, verbose=False)
        lp2 = r2.get("lp_bound")
        lp2_t = r2.get("solve_time")
        cuts = r2.get("n_cuts_added", 0)

        r3 = sos2_sdp_bound(g, verbose=False)
        s2 = r3.get("sos2_bound")
        s2_t = r3.get("solve_time")

        opt_str = f"{opt:.0f}" if opt is not None else "—"
        gw_s = f"{gw:.4f}" if gw is not None else "FAIL"
        lp1_s = f"{lp1:.4f}" if lp1 is not None else "FAIL"
        lp2_s = f"{lp2:.4f}" if lp2 is not None else "FAIL"
        s2_s = f"{s2:.4f}" if s2 is not None else "FAIL"

        print(f"{name:<18}{g.n:>3}{g.n_edges:>4}{opt_str:>5}{gw_s:>9}"
              f"{lp1_s:>9}{lp2_s:>9}{s2_s:>9}"
              f"{cuts:>5}{gw_t:>7.2f}{lp1_t:>7.2f}{lp2_t:>7.2f}{s2_t:>7.2f}")

        if opt is not None and lp2 is not None and abs(lp2 - opt) < 5e-3:
            n_lpoc_exact += 1
        if opt is not None:
            n_total_with_opt += 1

    print("-" * len(header))
    print(f"\n  LP+OddCycle exact: {n_lpoc_exact}/{n_total_with_opt}")


if __name__ == "__main__":
    run()
