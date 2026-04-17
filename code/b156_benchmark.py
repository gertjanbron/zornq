#!/usr/bin/env python3
"""B156 Benchmark: SoS-2 (Lasserre level-2) vs GW (level-1) bovengrens.

Genereert een tabel die laat zien:
  - GW SDP bovengrens (=Lasserre level-1)
  - SoS-2 (Lasserre level-2) bovengrens
  - OPT (brute-force, indien n ≤ 18)
  - Tightening percentage en gap-tot-OPT
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
    complete_bipartite,
    complete_graph,
    cycle_graph,
    path_graph,
    petersen_graph,
    sos2_sdp_bound,
)


def run_benchmark() -> None:
    instances: list[tuple[str, SimpleGraph]] = [
        ("K_3 (triangle)",       complete_graph(3)),
        ("K_4",                  complete_graph(4)),
        ("K_5",                  complete_graph(5)),
        ("K_6",                  complete_graph(6)),
        ("K_3,3 (bipartite)",    complete_bipartite(3, 3)),
        ("K_2,5 (bipartite)",    complete_bipartite(2, 5)),
        ("C_5 (odd cycle)",      cycle_graph(5)),
        ("C_7 (odd cycle)",      cycle_graph(7)),
        ("C_8 (even cycle)",     cycle_graph(8)),
        ("P_8 (path)",           path_graph(8)),
        ("Petersen",             petersen_graph()),
        ("3-reg n=10 seed42",    random_3regular(10, seed=42)),
        ("3-reg n=12 seed7",     random_3regular(12, seed=7)),
        ("3-reg n=14 seed3",     random_3regular(14, seed=3)),
    ]

    print("=" * 100)
    print(" B156 Benchmark: SoS-2 (Lasserre level-2) vs GW (level-1) MaxCut bovengrens")
    print("=" * 100)
    header = (
        f"{'Instance':<24}{'n':>4}{'m':>5}{'OPT':>7}{'GW':>10}"
        f"{'SoS-2':>10}{'tight%':>9}{'gw_gap%':>10}{'sos2_gap%':>11}"
        f"{'GW_t':>7}{'SoS2_t':>9}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for name, g in instances:
        opt = brute_force_maxcut(g) if g.n <= 18 else None

        t0 = time.time()
        gw = gw_sdp_bound(g, verbose=False).get("sdp_bound")
        gw_t = time.time() - t0

        sos2_res = sos2_sdp_bound(g, verbose=False)
        sos2 = sos2_res.get("sos2_bound")
        sos2_t = sos2_res.get("solve_time", 0.0)

        tightening = (gw - sos2) / gw * 100.0 if (gw and sos2) else 0.0
        gw_gap = (gw - opt) / opt * 100.0 if (gw and opt) else None
        sos2_gap = (sos2 - opt) / opt * 100.0 if (sos2 and opt) else None

        opt_str = f"{opt:.0f}" if opt is not None else "—"
        gw_str = f"{gw:.4f}" if gw is not None else "FAIL"
        sos2_str = f"{sos2:.4f}" if sos2 is not None else "FAIL"
        gwgap_str = f"{gw_gap:+.2f}" if gw_gap is not None else "—"
        sgap_str = f"{sos2_gap:+.2f}" if sos2_gap is not None else "—"

        print(
            f"{name:<24}{g.n:>4}{g.n_edges:>5}{opt_str:>7}{gw_str:>10}"
            f"{sos2_str:>10}{tightening:>9.2f}{gwgap_str:>10}{sgap_str:>11}"
            f"{gw_t:>7.2f}{sos2_t:>9.2f}"
        )

        rows.append({
            "name": name, "n": g.n, "m": g.n_edges,
            "opt": opt, "gw": gw, "sos2": sos2,
            "tight_pct": tightening, "gw_gap_pct": gw_gap,
            "sos2_gap_pct": sos2_gap,
            "gw_time": gw_t, "sos2_time": sos2_t,
        })

    print("-" * len(header))

    # Aggregaten
    n_exact = sum(1 for r in rows if r["sos2"] is not None and r["opt"] is not None
                  and abs(r["sos2"] - r["opt"]) < 5e-3)
    n_total = sum(1 for r in rows if r["opt"] is not None)
    avg_tight = sum(r["tight_pct"] for r in rows) / len(rows)
    print(f"\n  SoS-2 exact: {n_exact}/{n_total}")
    print(f"  Gemiddelde tightening: {avg_tight:.2f}%")
    print(f"  Totaal: {len(instances)} instanties")


if __name__ == "__main__":
    run_benchmark()
