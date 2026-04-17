#!/usr/bin/env python3
"""B170 benchmark — twin-width heuristic + cograph MaxCut speed/correctness.

Drie sectie:

  1. Twin-width heuristic op klassieke grafen-families.
     Laat zien: K_n / K_{m,n} -> 0, paden -> 1, cycli -> 2, Petersen / C_n
     voor grotere n -> 2-4 afhankelijk van de heuristic.

  2. Cograph MaxCut DP vs brute force (correctheid + speed).
     Grote cographs waar brute force 2^n prohibitive is, maar DP O(n^3) is.

  3. Heuristic tww als difficulty-metric voor B130-dispatcher:
     toon tww per instance-klasse + voorgesteld solver-pad
     (cograph -> cograph_dp; kleine tww -> tree-DP/B42; groot -> QUBO/B153).
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b170_twin_width import (
    Trigraph,
    brute_force_maxcut,
    cograph_maxcut_exact,
    complete_bipartite_edges,
    complete_edges,
    cycle_edges,
    empty_edges,
    is_cograph,
    path_edges,
    petersen_edges,
    tree_edges_balanced_binary,
    twin_width_heuristic,
)


def _hdr(title: str) -> None:
    bar = "=" * 100
    print(bar)
    print(f"  {title}")
    print(bar)


def _row(cols, widths) -> None:
    print("  ".join(str(c).rjust(w) for c, w in zip(cols, widths)))


# ============================================================
# 1. Twin-width op klassieke families
# ============================================================

def bench_families() -> None:
    _hdr("1. Twin-width heuristic op klassieke graaf-families")
    cases = [
        ("K_4",              4,  complete_edges(4)),
        ("K_8",              8,  complete_edges(8)),
        ("K_12",             12, complete_edges(12)),
        ("K_{3,3}",          6,  complete_bipartite_edges(3, 3)),
        ("K_{4,4}",          8,  complete_bipartite_edges(4, 4)),
        ("K_{5,7}",          12, complete_bipartite_edges(5, 7)),
        ("P_6",              6,  path_edges(6)),
        ("P_10",             10, path_edges(10)),
        ("C_5",              5,  cycle_edges(5)),
        ("C_8",              8,  cycle_edges(8)),
        ("C_12",             12, cycle_edges(12)),
        ("Petersen",         10, petersen_edges()),
        ("tree(depth=3)",    15, tree_edges_balanced_binary(3)[1]),
        ("tree(depth=4)",    31, tree_edges_balanced_binary(4)[1]),
    ]
    widths = [20, 4, 6, 8, 8, 10, 12]
    _row(["family", "n", "edges", "tww_h", "cograph", "time_ms",
          "verdict"], widths)
    print("-" * 100)
    for label, n, edges in cases:
        g = Trigraph.from_graph(n, edges)
        t0 = time.time()
        d, _ = twin_width_heuristic(g)
        ms = (time.time() - t0) * 1000
        cog = is_cograph(n, edges)
        if cog:
            verdict = "cograph_dp"
        elif d <= 2:
            verdict = "tree_dp"
        elif d <= 5:
            verdict = "bounded_tww"
        else:
            verdict = "qubo"
        _row([label, n, len(edges), d, "Y" if cog else "N",
              f"{ms:.1f}", verdict], widths)
    print()


# ============================================================
# 2. Cograph MaxCut DP vs brute force
# ============================================================

def bench_cograph_maxcut() -> None:
    _hdr("2. Cograph MaxCut — DP (O(n^3)) vs brute force (O(2^n))")

    cases = []
    # a) complete graphs
    for n in (6, 10, 14, 18, 24):
        cases.append((f"K_{n}", n, complete_edges(n)))
    # b) complete bipartite
    for (a, b) in [(4, 4), (5, 5), (6, 8), (10, 10)]:
        cases.append((f"K_{{{a},{b}}}", a + b, complete_bipartite_edges(a, b)))
    # c) random cographs (series-parallel composities)
    rng = random.Random(42)
    for trial, n in enumerate([10, 14, 18, 24, 32]):
        pieces = [[v] for v in range(n)]
        edges = []
        while len(pieces) > 1:
            i, j = rng.sample(range(len(pieces)), 2)
            a, b = pieces[i], pieces[j]
            if rng.random() < 0.5:
                for x in a:
                    for y in b:
                        edges.append((x, y))
            merged = a + b
            pieces = [p for k, p in enumerate(pieces) if k not in (i, j)]
            pieces.append(merged)
        cases.append((f"random-cograph#{trial+1}", n, edges))

    widths = [20, 4, 6, 12, 12, 12, 12, 8]
    _row(["instance", "n", "edges", "DP cut", "BF cut", "DP ms", "BF ms",
          "match"], widths)
    print("-" * 100)
    for label, n, edges in cases:
        t0 = time.time()
        dp = cograph_maxcut_exact(n, edges)
        dp_ms = (time.time() - t0) * 1000
        if n <= 18:
            t0 = time.time()
            bf = brute_force_maxcut(n, edges)
            bf_ms = (time.time() - t0) * 1000
            match = "Y" if abs(dp["value"] - bf["value"]) < 1e-9 else "N"
            bf_cut_str = f"{bf['value']:.0f}"
            bf_ms_str = f"{bf_ms:.1f}"
        else:
            bf_cut_str = "—"
            bf_ms_str = "—"
            match = "skip"
        _row([label, n, len(edges),
              f"{dp['value']:.0f}", bf_cut_str,
              f"{dp_ms:.2f}", bf_ms_str, match], widths)
    print()


# ============================================================
# 3. Dispatcher-difficulty metric
# ============================================================

def bench_dispatcher_routing() -> None:
    _hdr("3. Dispatcher-voorstel op basis van tww (feature voor B130)")
    print("""
    Regel:
      tww = 0      -> cograph_maxcut_exact (O(n^3), dit module)
      tww <= 2     -> tree-DP / courcelle (B42)
      tww <= 5     -> bounded-tww DP (niet geïmplementeerd; toekomstig)
      anders        -> QUBO-suite (B153: LS + SA + RR) of GW (B60)
    """)

    cases = [
        ("K_10",          10, complete_edges(10)),
        ("K_{5,5}",       10, complete_bipartite_edges(5, 5)),
        ("P_12",          12, path_edges(12)),
        ("C_12",          12, cycle_edges(12)),
        ("Petersen",      10, petersen_edges()),
        ("tree(depth=3)", 15, tree_edges_balanced_binary(3)[1]),
    ]
    widths = [20, 4, 6, 8, 10, 20, 10]
    _row(["instance", "n", "edges", "tww_h", "cograph",
          "route", "time_ms"], widths)
    print("-" * 100)
    for label, n, edges in cases:
        g = Trigraph.from_graph(n, edges)
        t0 = time.time()
        d, _ = twin_width_heuristic(g)
        tww_ms = (time.time() - t0) * 1000
        cog = is_cograph(n, edges)
        if cog:
            route = "cograph_dp"
        elif d <= 2:
            route = "tree_dp (B42)"
        elif d <= 5:
            route = "bounded_tww_dp*"
        else:
            route = "qubo (B153)"
        _row([label, n, len(edges), d, "Y" if cog else "N",
              route, f"{tww_ms:.1f}"], widths)
    print("  * bounded_tww_dp: toekomstige opvolger, momenteel fall-back naar QUBO")
    print()


# ============================================================
# Hoofd
# ============================================================

def main() -> int:
    t0 = time.time()
    bench_families()
    bench_cograph_maxcut()
    bench_dispatcher_routing()
    print(f"\nTotaal walltime: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
