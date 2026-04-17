#!/usr/bin/env python3
"""B176 benchmark -- schaalbaarheid Frank-Wolfe SDP vs cvxpy interior-point.

Vier secties:
  1. Correctness-sanity: FW-sandwich rond cvxpy op kleine grafen.
  2. Schaalbaarheid: wall-time FW (iteratie-gecapped) vs cvxpy over n = 30..600.
  3. Convergentie-curve: f(X_k) en gap(X_k) over iteraties op n = 200.
  4. GW-rounding uit FW: hoe goed rondt de FW-embedding af (best-of-k).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import (
    brute_force_maxcut,
    cylinder_graph,
    random_3regular,
    random_erdos_renyi,
)
from b176_frank_wolfe_sdp import (
    cvxpy_reference_sdp,
    frank_wolfe_maxcut_sdp,
    gw_round_from_Y,
)


def _hdr(title: str) -> None:
    bar = "=" * 100
    print(bar); print("  " + title); print(bar)


# ============================================================
# 1. Correctness-sanity
# ============================================================


def bench_correctness() -> None:
    _hdr("1. Correctness: FW-sandwich rond cvxpy (kleine grafen)")
    print("  %-22s  %8s  %8s  %8s  %8s  %-5s" % (
        "graph", "LB_FW", "cvxpy", "UB_FW", "exact", "sw"))
    print("  " + "-" * 80)

    cases = [
        ("triangle K_3",           _triangle()),
        ("K_4",                    _k4()),
        ("cylinder 3x3",           cylinder_graph(3, 3)),
        ("cylinder 4x3",           cylinder_graph(4, 3)),
        ("3-reg n=20 seed=1",      random_3regular(20, seed=1)),
        ("3-reg n=30 seed=7",      random_3regular(30, seed=7)),
        ("ER n=14 p=0.4 seed=3",   random_erdos_renyi(14, p=0.4, seed=3)),
    ]
    for name, g in cases:
        res = frank_wolfe_maxcut_sdp(g, max_iter=600, tol=1e-5, rank_cap=16)
        ref = cvxpy_reference_sdp(g, verbose=False)
        cvx = ref["sdp_bound"]
        exact = brute_force_maxcut(g)
        exact_s = ("%d" % exact) if exact is not None else "  -"
        sw_ok = res.feasible_cut_lb - 1e-6 <= cvx <= res.sdp_upper_bound + 1e-6
        print("  %-22s  %8.3f  %8.3f  %8.3f  %8s  %-5s" % (
            name, res.feasible_cut_lb, cvx, res.sdp_upper_bound, exact_s,
            "OK" if sw_ok else "FAIL"))
    print()


def _triangle():
    from b60_gw_bound import SimpleGraph
    g = SimpleGraph(3)
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(0, 2)
    return g


def _k4():
    from b60_gw_bound import SimpleGraph
    g = SimpleGraph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    return g


# ============================================================
# 2. Schaalbaarheid: wall-time
# ============================================================


def bench_scalability() -> None:
    _hdr("2. Schaalbaarheid -- wall-time Frank-Wolfe vs cvxpy")
    print("  3-reguliere grafen, edges = 1.5 * n; cvxpy-SCS interior-point.")
    print()
    print("  %5s  %7s  %10s  %10s  %8s  %10s  %10s  %8s" % (
        "n", "edges", "FW_UB", "FW_LB", "FW_time", "cvx_SDP", "cvx_time", "speedup"))
    print("  " + "-" * 90)

    # Size sweep: laat cvxpy overslaan bij n > 200 om totale benchmark-tijd te sparen.
    sweep = [30, 60, 100, 200, 300, 500]
    for n in sweep:
        g = random_3regular(n, seed=n)
        # Frank-Wolfe -- gecapt aantal iteraties voor doorzichtige benchmark
        max_iter_cap = min(1200, max(300, 3 * n))
        t0 = time.time()
        res = frank_wolfe_maxcut_sdp(
            g,
            max_iter=max_iter_cap,
            tol=1e-5,
            rank_cap=min(48, int(np.sqrt(n)) + 12),
        )
        t_fw = time.time() - t0

        if n <= 200:
            t0 = time.time()
            try:
                ref = cvxpy_reference_sdp(g, verbose=False)
                t_cv = time.time() - t0
                cvx_val = ref["sdp_bound"]
            except Exception:
                t_cv = float("nan")
                cvx_val = float("nan")
            speedup = t_cv / t_fw if (t_fw > 0 and not np.isnan(t_cv)) else float("nan")
            print("  %5d  %7d  %10.2f  %10.2f  %7.2fs  %10.2f  %9.2fs  %7.2fx" % (
                n, g.n_edges, res.sdp_upper_bound, res.feasible_cut_lb, t_fw,
                cvx_val, t_cv, speedup))
        else:
            print("  %5d  %7d  %10.2f  %10.2f  %7.2fs  %10s  %10s  %8s" % (
                n, g.n_edges, res.sdp_upper_bound, res.feasible_cut_lb, t_fw,
                "(skip)", "(skip)", "-"))
    print()


# ============================================================
# 3. Convergentie-curve
# ============================================================


def bench_convergence() -> None:
    _hdr("3. Convergentie -- f(X_k), gap(X_k) over iteraties (n=100)")
    g = random_3regular(100, seed=42)
    res = frank_wolfe_maxcut_sdp(g, max_iter=1200, tol=0, rank_cap=32, verbose=False)
    hist = res.history
    print("  n=%d edges=%d  final UB=%.2f  LB=%.2f  solve=%ds" % (
        g.n, g.n_edges, res.sdp_upper_bound, res.feasible_cut_lb, int(res.solve_time)))
    print()
    print("  Decimatie (elke ~100 iters):")
    print("  %6s  %12s  %12s  %12s  %6s" % (
        "iter", "f(X_k)", "gap", "diag_err", "rank"))
    print("  " + "-" * 64)
    step = max(1, len(hist) // 12)
    for h in hist[::step]:
        print("  %6d  %+12.4f  %+12.4e  %12.4e  %6d" % (
            h["iter"], h["f"], h["gap"], h["diag_err_max"], h["rank"]))
    print("  %6d  %+12.4f  %+12.4e  %12.4e  %6d  <- laatste" % (
        hist[-1]["iter"], hist[-1]["f"], hist[-1]["gap"],
        hist[-1]["diag_err_max"], hist[-1]["rank"]))
    print()


# ============================================================
# 4. GW-rounding
# ============================================================


def bench_gw_rounding() -> None:
    _hdr("4. GW-rounding uit FW-embedding (best-of-k hyperplanes)")
    print("  %-22s  %6s  %6s  %6s  %7s  %7s" % (
        "graph", "n", "edges", "UB", "cut", "cut/UB"))
    print("  " + "-" * 70)

    cases = [
        ("cylinder 4x3",           cylinder_graph(4, 3), 17),
        ("cylinder 5x3",           cylinder_graph(5, 3), None),
        ("3-reg n=30 seed=1",      random_3regular(30, seed=1), None),
        ("3-reg n=60 seed=5",      random_3regular(60, seed=5), None),
        ("ER n=20 p=0.35 seed=8",  random_erdos_renyi(20, p=0.35, seed=8), None),
    ]
    for name, g, known in cases:
        res = frank_wolfe_maxcut_sdp(g, max_iter=min(800, max(300, 4 * g.n)),
                                     tol=1e-5, rank_cap=min(40, g.n))
        bs, cut = gw_round_from_Y(res.Y, g, n_trials=400, seed=1)
        ratio_ub = cut / res.sdp_upper_bound if res.sdp_upper_bound > 0 else 0.0
        exact_tag = "" if known is None else ("  [exact=%d]" % known)
        print("  %-22s  %6d  %6d  %6.2f  %7.1f  %6.3f%s" % (
            name, g.n, g.n_edges, res.sdp_upper_bound, cut, ratio_ub, exact_tag))
    print()


# ============================================================
# 5. GW-fractie tegen 0.87856 bovengrens
# ============================================================


def bench_gw_fraction() -> None:
    _hdr("5. 0.87856-garantie check: cut / UB versus theoretische GW-fractie")
    print("  Voor feasibele rounding geldt E[cut] >= 0.87856 * SDP_opt.")
    print("  We vergelijken de beste sample tegen de FW-UB.")
    print()

    vals = []
    for seed in (0, 1, 2, 3, 4):
        g = random_3regular(50, seed=seed)
        res = frank_wolfe_maxcut_sdp(g, max_iter=500, tol=1e-5, rank_cap=20)
        _, cut = gw_round_from_Y(res.Y, g, n_trials=400, seed=seed)
        vals.append(cut / res.sdp_upper_bound)
        print("  seed=%d  UB=%.2f  cut=%.1f  cut/UB=%.4f" % (
            seed, res.sdp_upper_bound, cut, cut / res.sdp_upper_bound))
    avg = float(np.mean(vals))
    mn = float(np.min(vals))
    print()
    print("  gemiddeld cut/UB: %.4f   minimum: %.4f   GW-theorie: >= 0.87856" % (avg, mn))
    print("  (cut/UB kan > 1 zijn als UB heel los is; dan alsnog geldig rounding-resultaat.)")
    print()


# ============================================================
# Main
# ============================================================


def main() -> int:
    t0 = time.time()
    bench_correctness()
    bench_scalability()
    bench_convergence()
    bench_gw_rounding()
    bench_gw_fraction()
    print("")
    print("Totaal walltime: %.2fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
