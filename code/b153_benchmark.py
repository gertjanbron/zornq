#!/usr/bin/env python3
"""B153 benchmark — toon dat de QUBO-suite op 4 probleemklassen werkt.

Per probleem-klasse:
  - Genereer/laad een handvol instanties van oplopende grootte.
  - Los op met brute_force (exact, n ≤ 18) + local_search + simulated_annealing
    + random_restart_LS.
  - Tabuleer (energie, gap, walltime).

Doel: dispatcher-claim "domein-agnostisch" hardmaken — één API + één set
generieke solvers werkt op MaxCut (gewogen), Max-k-Cut, MIS én Markowitz.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b153_qubo_suite import (
    encode_markowitz,
    encode_max_k_cut,
    encode_mis,
    encode_weighted_maxcut,
    qubo_brute_force,
    qubo_local_search,
    qubo_random_restart,
    qubo_simulated_annealing,
    random_markowitz_instance,
    cycle_edges,
    petersen_edges,
    random_erdos_renyi_edges,
    random_weighted_edges,
)


def _print_hdr(title: str) -> None:
    bar = "=" * 100
    print(bar)
    print(f"  {title}")
    print(bar)


def _print_row(cols: list[str], widths: list[int]) -> None:
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))


def _solve_all(qubo, want_brute_force: bool, sa_sweeps: int = 500,
               rr_starts: int = 10, seed: int = 42):
    out = {}
    if want_brute_force:
        out["bf"] = qubo_brute_force(qubo)
    out["ls"] = qubo_local_search(qubo, seed=seed)
    out["sa"] = qubo_simulated_annealing(qubo, n_sweeps=sa_sweeps, seed=seed)
    out["rr"] = qubo_random_restart(qubo, n_starts=rr_starts, seed=seed,
                                    inner="local_search")
    return out


def _gap(opt: float | None, e: float) -> str:
    """Gap als percentage tov bekend optimum (lower = better, want we minimaliseren).
    BF-min is OPT; gap = (e_h - opt) / |opt + 1e-12| * 100."""
    if opt is None:
        return "—"
    return f"{(e - opt) / max(abs(opt), 1.0) * 100:+.2f}%"


# ============================================================
# 1. Weighted MaxCut
# ============================================================

def bench_weighted_maxcut() -> None:
    _print_hdr("1. Weighted MaxCut — generieke QUBO-engine op gewogen MaxCut")
    cases = [
        ("K_3 unweighted",  3,  [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]),
        ("K_3 w=(1,2,3)",   3,  [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]),
        ("Petersen w=1",    10, [(u, v, 1.0) for (u, v) in petersen_edges()]),
        ("ER n=8 p=.5",     8,  random_weighted_edges(8, 0.5, seed=1)),
        ("ER n=12 p=.4",   12,  random_weighted_edges(12, 0.4, seed=2)),
        ("ER n=16 p=.3",   16,  random_weighted_edges(16, 0.3, seed=3)),
    ]
    widths = [22, 4, 6, 12, 12, 12, 12, 8, 8, 8]
    _print_row(["instance", "n", "edges",
                "BF cut", "LS cut", "SA cut", "RR cut",
                "LS gap", "SA gap", "RR gap"], widths)
    print("-" * 130)
    for label, n, edges in cases:
        inst = encode_weighted_maxcut(n, edges)
        out = _solve_all(inst.qubo, want_brute_force=(n <= 18))
        bf_cut = inst.decode(out["bf"]["x"])["value"] if "bf" in out else None
        rows = []
        for tag in ("bf", "ls", "sa", "rr"):
            if tag not in out:
                rows.append("—")
            else:
                v = inst.decode(out[tag]["x"])["value"]
                rows.append(f"{v:.4f}")
        # Gaps in cut-percentages (cut is max, dus heuristic cut moet ≤ BF cut)
        gaps = []
        for tag in ("ls", "sa", "rr"):
            if "bf" not in out:
                gaps.append("—")
            else:
                v = inst.decode(out[tag]["x"])["value"]
                if bf_cut == 0:
                    gaps.append("0.00%")
                else:
                    gaps.append(f"{(v - bf_cut) / bf_cut * 100:+.2f}%")
        _print_row([label, str(n), str(len(edges)),
                    rows[0], rows[1], rows[2], rows[3],
                    gaps[0], gaps[1], gaps[2]], widths)
    print()


# ============================================================
# 2. Max-k-Cut
# ============================================================

def bench_max_k_cut() -> None:
    _print_hdr("2. Max-k-Cut — k-partitionering via one-hot encoding")
    cases = [
        # (label, n, edges (unweighted), k)
        ("K_4 k=2",  4, [(i, j) for i in range(4) for j in range(i + 1, 4)], 2),
        ("K_4 k=3",  4, [(i, j) for i in range(4) for j in range(i + 1, 4)], 3),
        ("K_4 k=4",  4, [(i, j) for i in range(4) for j in range(i + 1, 4)], 4),
        ("C_5 k=3",  5, cycle_edges(5), 3),
        ("Petersen k=3", 10, petersen_edges(), 3),  # n*k = 30 → te groot voor BF
        ("ER n=6 p=.6 k=3", 6, random_erdos_renyi_edges(6, 0.6, seed=2), 3),
    ]
    widths = [22, 4, 4, 8, 12, 12, 12, 12]
    _print_row(["instance", "n", "k", "qubo_n",
                "BF cut", "LS cut", "SA cut", "RR cut"], widths)
    print("-" * 110)
    for label, n, edges, k in cases:
        edges_w = [(u, v, 1.0) for (u, v) in edges]
        inst = encode_max_k_cut(n, edges_w, k=k)
        N = n * k
        want_bf = N <= 16
        out = _solve_all(inst.qubo, want_brute_force=want_bf, sa_sweeps=1000)
        rows = []
        for tag in ("bf", "ls", "sa", "rr"):
            if tag not in out:
                rows.append("—")
            else:
                d = inst.decode(out[tag]["x"])
                marker = "" if d["feasible"] else "*"
                rows.append(f"{d['value']:.2f}{marker}")
        _print_row([label, str(n), str(k), str(N),
                    rows[0], rows[1], rows[2], rows[3]], widths)
    print("  *  = infeasible (one-hot constraint geschonden)")
    print()


# ============================================================
# 3. Maximum Independent Set
# ============================================================

def bench_mis() -> None:
    _print_hdr("3. Maximum Independent Set (MIS)")
    cases = [
        ("C_5",      5,  cycle_edges(5)),
        ("C_6",      6,  cycle_edges(6)),
        ("C_7",      7,  cycle_edges(7)),
        ("Petersen", 10, petersen_edges()),
        ("ER n=12 p=.3", 12, random_erdos_renyi_edges(12, 0.3, seed=2)),
        ("ER n=16 p=.25", 16, random_erdos_renyi_edges(16, 0.25, seed=3)),
        ("ER n=20 p=.2",  20, random_erdos_renyi_edges(20, 0.2, seed=4)),
    ]
    widths = [22, 4, 6, 8, 8, 8, 8, 10, 10, 10]
    _print_row(["instance", "n", "edges",
                "BF α", "LS α", "SA α", "RR α",
                "LS time", "SA time", "RR time"], widths)
    print("-" * 130)
    for label, n, edges in cases:
        inst = encode_mis(n, edges)
        want_bf = n <= 18
        out = _solve_all(inst.qubo, want_brute_force=want_bf, sa_sweeps=500)
        sizes = []
        for tag in ("bf", "ls", "sa", "rr"):
            if tag not in out:
                sizes.append("—")
            else:
                d = inst.decode(out[tag]["x"])
                marker = "" if d["feasible"] else "!"
                sizes.append(f"{d['size']}{marker}")
        ts = [f"{out[tag]['wall_time']*1000:.1f}ms" for tag in ("ls", "sa", "rr")]
        _print_row([label, str(n), str(len(edges)),
                    sizes[0], sizes[1], sizes[2], sizes[3],
                    ts[0], ts[1], ts[2]], widths)
    print("  ! = infeasible (penalty te laag of solver vast)")
    print()


# ============================================================
# 4. Markowitz portfolio
# ============================================================

def bench_markowitz() -> None:
    _print_hdr("4. Markowitz portfolio — financiële QUBO")
    cases = [
        # (n, K, λ, seed)
        (6,  2, 1.0, 0),
        (8,  3, 1.0, 1),
        (10, 4, 0.5, 2),
        (12, 5, 1.0, 3),
        (14, 5, 2.0, 4),
        (16, 6, 1.0, 5),  # 2^16 = 65536 → BF haalbaar maar traag
    ]
    widths = [3, 3, 6, 12, 12, 12, 12, 8, 8]
    _print_row(["n", "K", "λ",
                "BF utility", "LS utility", "SA utility", "RR utility",
                "LS feas", "RR feas"], widths)
    print("-" * 110)
    for n, K, lam, seed in cases:
        inst = random_markowitz_instance(n, seed=seed, budget=K, risk_aversion=lam)
        want_bf = n <= 18
        out = _solve_all(inst.qubo, want_brute_force=want_bf,
                         sa_sweeps=1000, rr_starts=20, seed=seed)
        utilities = []
        feasibles = {}
        for tag in ("bf", "ls", "sa", "rr"):
            if tag not in out:
                utilities.append("—")
            else:
                d = inst.decode(out[tag]["x"])
                u = d["utility"] if d["feasible"] else d["value"]
                marker = "" if d["feasible"] else "*"
                utilities.append(f"{u:+.4f}{marker}")
                feasibles[tag] = d["feasible"]
        _print_row([str(n), str(K), f"{lam:.1f}",
                    utilities[0], utilities[1], utilities[2], utilities[3],
                    "Y" if feasibles.get("ls") else "N",
                    "Y" if feasibles.get("rr") else "N"], widths)
    print("  * = infeasible (budget-constraint geschonden); utility = penalty-corrected")
    print()


# ============================================================
# Hoofd
# ============================================================

def main() -> int:
    t0 = time.time()
    bench_weighted_maxcut()
    bench_max_k_cut()
    bench_mis()
    bench_markowitz()
    print(f"\nTotaal walltime: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
