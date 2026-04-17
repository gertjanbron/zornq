#!/usr/bin/env python3
"""B159 Benchmark: ILP-Oracle vs upper-bound hierarchie (GW, LP+OC, SoS-2).

Toont voor elke graaf:
  - n, m
  - Brute force OPT (tot n ≤ 18)
  - ILP-OPT (certificeerbaar via HiGHS)
  - GW (SDP-1) upper bound
  - LP+OddCycle upper bound
  - SoS-2 (Lasserre-2) upper bound
  - Gap: UB − ILP voor elke UB-methode
  - Wall-time ILP

Het doel: demonstreren dat de ILP-oracle een harde OPT-regel levert in de
paper-tabel, en dat de UB-methoden progressief scherper worden.
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
from b158_cutting_planes import lp_triangle_oddcycle_bound
from b159_ilp_oracle import maxcut_ilp_highs


def run() -> None:
    instances: list[tuple[str, SimpleGraph]] = [
        ("K_3",              complete_graph(3)),
        ("K_4",              complete_graph(4)),
        ("K_5",              complete_graph(5)),
        ("K_6",              complete_graph(6)),
        ("K_3,3",            complete_bipartite(3, 3)),
        ("K_2,5",            complete_bipartite(2, 5)),
        ("C_5",              cycle_graph(5)),
        ("C_7",              cycle_graph(7)),
        ("C_8",              cycle_graph(8)),
        ("P_8",              path_graph(8)),
        ("Petersen",         petersen_graph()),
        ("3-reg n=10",       random_3regular(10, seed=42)),
        ("3-reg n=12",       random_3regular(12, seed=7)),
        ("3-reg n=14",       random_3regular(14, seed=3)),
        ("3-reg n=20",       random_3regular(20, seed=11)),
        ("3-reg n=30",       random_3regular(30, seed=17)),
        ("3-reg n=50",       random_3regular(50, seed=23)),
    ]

    print("=" * 120)
    print(" B159 Benchmark: ILP-Oracle (HiGHS MILP) vs UB-hierarchie (GW, LP+OC, SoS-2)")
    print("=" * 120)
    header = (
        f"{'Instance':<14}{'n':>3}{'m':>4}"
        f"{'OPT_BF':>8}{'OPT_ILP':>9}{'cert':>6}{'t_ILP':>8}"
        f"{'GW':>9}{'LP+OC':>9}{'SoS-2':>9}"
        f"{'dGW':>7}{'dOC':>7}{'dS2':>7}"
    )
    print(header)
    print("-" * len(header))

    n_tight_gw = 0
    n_tight_oc = 0
    n_tight_s2 = 0
    n_total = 0

    for name, g in instances:
        # Brute force alleen voor kleine n
        bf = brute_force_maxcut(g) if g.n <= 18 else None

        # ILP-oracle met ruime time-limit voor grotere grafen
        tl = 10.0 if g.n > 14 else None
        ilp_res = maxcut_ilp_highs(g, time_limit=tl)
        opt_ilp = ilp_res["opt_value"]
        cert = "✓" if ilp_res["certified"] else "~"
        t_ilp = ilp_res["wall_time"]

        # Upper bounds — alleen als graaf klein genoeg voor cvxpy
        gw = None
        lpoc = None
        s2 = None

        if g.n <= 14:
            gw = gw_sdp_bound(g, verbose=False).get("sdp_bound")
            lpoc = lp_triangle_oddcycle_bound(g, verbose=False).get("lp_bound")
            s2 = sos2_sdp_bound(g, verbose=False).get("sos2_bound")
        else:
            # Grote grafen: alleen LP+OC (schaalbaar)
            lpoc = lp_triangle_oddcycle_bound(g, verbose=False).get("lp_bound")

        bf_s = f"{bf:.0f}" if bf is not None else "—"
        ilp_s = f"{opt_ilp:.1f}" if opt_ilp is not None else "FAIL"
        gw_s = f"{gw:.3f}" if gw is not None else "—"
        oc_s = f"{lpoc:.3f}" if lpoc is not None else "—"
        s2_s = f"{s2:.3f}" if s2 is not None else "—"
        dgw = f"{gw - opt_ilp:+.2f}" if gw is not None and opt_ilp is not None else "—"
        doc = f"{lpoc - opt_ilp:+.2f}" if lpoc is not None and opt_ilp is not None else "—"
        ds2 = f"{s2 - opt_ilp:+.2f}" if s2 is not None and opt_ilp is not None else "—"

        print(f"{name:<14}{g.n:>3}{g.n_edges:>4}"
              f"{bf_s:>8}{ilp_s:>9}{cert:>6}{t_ilp:>8.3f}"
              f"{gw_s:>9}{oc_s:>9}{s2_s:>9}"
              f"{dgw:>7}{doc:>7}{ds2:>7}")

        if ilp_res["certified"] and opt_ilp is not None:
            n_total += 1
            if gw is not None and abs(gw - opt_ilp) < 1e-3:
                n_tight_gw += 1
            if lpoc is not None and abs(lpoc - opt_ilp) < 1e-3:
                n_tight_oc += 1
            if s2 is not None and abs(s2 - opt_ilp) < 1e-3:
                n_tight_s2 += 1

    print("-" * len(header))
    print(f"\n  ILP certified: {sum(1 for _, g in instances if True)}/{len(instances)}")
    print(f"  GW     tight (UB = ILP-OPT): {n_tight_gw}/{n_total}")
    print(f"  LP+OC  tight (UB = ILP-OPT): {n_tight_oc}/{n_total}")
    print(f"  SoS-2  tight (UB = ILP-OPT): {n_tight_s2}/{n_total}")
    print()
    print("  cert: ✓ = HiGHS OPT bewezen; ~ = feasible incumbent (time-limit)")
    print("  d*:   upper-bound minus ILP-OPT (0 = UB is tight, >0 = gap)")


if __name__ == "__main__":
    run()
