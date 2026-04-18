#!/usr/bin/env python3
"""B176b benchmark -- CGAL-SDP schaalbaarheid (paper-2 target).

Vier secties:
  1. Correctness-sandwich vs cvxpy + brute-force op kleine grafen.
  2. Head-to-head CGAL vs B176-FW over n = 100..2000.
  3. Paper-2 scale-panel: 3-reguliere grafen n = 500, 1000, 2000, 5000, 10000.
     Emit JSON/CSV + LaTeX booktabs-tabel onder ../docs/paper/tables/.
  4. Convergentie-curve CGAL op n=1000: UB, LB, diag_err over iteraties.

Usage
-----
    python b176b_benchmark.py                    # alle secties, full-scale
    python b176b_benchmark.py --quick            # alleen kleine n
    python b176b_benchmark.py --only scale       # alleen sectie 3

Output-files (sectie 3):
    ../docs/paper/data/b176b_cgal_results.json
    ../docs/paper/data/b176b_cgal_results.csv
    ../docs/paper/tables/b176b_scale_table.md
    ../docs/paper/tables/b176b_scale_table.tex
"""
from __future__ import annotations

import argparse
import csv
import json
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
from b176b_cgal_sdp import (
    cgal_maxcut_sdp,
    dual_upper_bound,
    head_to_head,
)


# ============================================================
# Output paths
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "paper", "data"))
TABLES_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "paper", "tables"))


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def _hdr(title: str) -> None:
    bar = "=" * 100
    print(bar); print("  " + title); print(bar)


# ============================================================
# 1. Correctness-sanity
# ============================================================


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


def bench_correctness() -> None:
    _hdr("1. Correctness: CGAL-sandwich vs cvxpy + brute-force")
    print("  %-24s  %8s  %8s  %8s  %8s  %-5s" % (
        "graph", "LB_CGAL", "cvxpy", "UB_CGAL", "exact", "sw"))
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
        res = cgal_maxcut_sdp(g, max_iter=600, tol=1e-5, rank_cap=16)
        try:
            ref = cvxpy_reference_sdp(g, verbose=False)
            cvx = ref["sdp_bound"]
        except Exception:
            cvx = float("nan")
        exact = brute_force_maxcut(g)
        if exact is None:
            exact_s = "  -"
        else:
            exact_s = "%d" % (exact[1] if isinstance(exact, tuple) else exact)
        lb = res.feasible_cut_lb
        ub = res.sdp_upper_bound
        sw_ok = (lb - 1e-4) <= (cvx if np.isfinite(cvx) else ub) <= (ub + 1e-4)
        print("  %-24s  %8.3f  %8.3f  %8.3f  %8s  %-5s" % (
            name, lb, cvx, ub, exact_s, "OK" if sw_ok else "FAIL"))
    print()


# ============================================================
# 2. Head-to-head CGAL vs B176-FW
# ============================================================


def bench_head_to_head() -> None:
    _hdr("2. Head-to-head: CGAL vs B176-FW (3-reg panel)")
    print("  %5s  %6s  %10s  %10s  %8s  %10s  %10s  %8s  %8s" % (
        "n", "edges", "FW_UB", "CGAL_UB", "delta%", "FW_time", "CGAL_time",
        "FW_diag", "CGAL_dg"))
    print("  " + "-" * 100)
    for n in [60, 100, 200, 400, 800, 1500]:
        g = random_3regular(n, seed=n)
        t0 = time.time()
        fw = frank_wolfe_maxcut_sdp(g, max_iter=400, tol=1e-5,
                                    rank_cap=min(48, int(np.sqrt(n)) + 12))
        t_fw = time.time() - t0
        t0 = time.time()
        cg = cgal_maxcut_sdp(g, max_iter=400, tol=1e-5,
                             rank_cap=min(48, int(np.sqrt(n)) + 12))
        t_cg = time.time() - t0
        delta = 100.0 * (cg.sdp_upper_bound - fw.sdp_upper_bound) / max(
            abs(fw.sdp_upper_bound), 1e-9)
        print("  %5d  %6d  %10.2f  %10.2f  %+7.2f%%  %10.3f  %10.3f  %8.1e  %8.2f" % (
            n, g.n_edges, fw.sdp_upper_bound, cg.sdp_upper_bound,
            delta, t_fw, t_cg, fw.diag_err_max, cg.dual_gap))
    print()


# ============================================================
# 3. Paper-2 scale-panel
# ============================================================


def _row_for_graph(g, max_iter: int, rank_cap: int, seed: int) -> dict:
    """Run CGAL + FW en retourneer een dict met alle relevante metrics."""
    # FW baseline
    t0 = time.time()
    fw = frank_wolfe_maxcut_sdp(g, max_iter=max_iter, tol=1e-4,
                                rank_cap=rank_cap, seed=seed)
    t_fw = time.time() - t0

    # CGAL
    t0 = time.time()
    cg = cgal_maxcut_sdp(g, max_iter=max_iter, tol=1e-4,
                         rank_cap=rank_cap, seed=seed,
                         dual_every=max(10, max_iter // 20))
    t_cg = time.time() - t0

    # GW rounding vanaf CGAL-Y
    t0 = time.time()
    _bs, gw_lb_cg = gw_round_from_Y(cg.Y, g, n_trials=200, seed=seed)
    t_gw = time.time() - t0

    return {
        "n": int(g.n),
        "m": int(g.n_edges),
        "seed": int(seed),
        # FW
        "fw_ub": float(fw.sdp_upper_bound),
        "fw_lb": float(fw.feasible_cut_lb),
        "fw_iter": int(fw.iterations),
        "fw_time_s": float(t_fw),
        "fw_diag_err": float(fw.diag_err_max),
        # CGAL
        "cgal_ub": float(cg.sdp_upper_bound),
        "cgal_lb": float(cg.feasible_cut_lb),
        "cgal_gw_lb": float(gw_lb_cg),
        "cgal_iter": int(cg.iterations),
        "cgal_time_s": float(t_cg),
        "cgal_gw_time_s": float(t_gw),
        "cgal_diag_err": float(cg.diag_err_max),
        "cgal_dual_gap": float(cg.dual_gap),
        "cgal_beta_final": float(cg.beta_final),
        "cgal_y_norm": float(np.linalg.norm(cg.y_final)),
        "delta_ub_pct": 100.0 * (cg.sdp_upper_bound - fw.sdp_upper_bound)
                        / max(abs(fw.sdp_upper_bound), 1e-9),
    }


def bench_scale_panel(include_10k: bool = True, seeds: tuple[int, ...] = (0,)) -> list[dict]:
    _hdr("3. Paper-2 scale-panel: 3-regulier random, n = 500..10000")
    _ensure_dirs()

    sizes = [500, 1000, 2000, 5000]
    if include_10k:
        sizes.append(10000)

    rows: list[dict] = []
    for n in sizes:
        rank_cap = min(128, int(np.sqrt(n)) + 24)
        max_iter = min(800, max(200, 2 * int(np.sqrt(n))))
        if n >= 10000:
            rank_cap = min(rank_cap, 64)
            max_iter = min(max_iter, 400)
        for seed in seeds:
            print(f"  -> n={n:5d}  rank_cap={rank_cap:3d}  max_iter={max_iter:4d}  seed={seed}")
            g = random_3regular(n, seed=seed)
            try:
                row = _row_for_graph(g, max_iter=max_iter, rank_cap=rank_cap, seed=seed)
                rows.append(row)
                print(f"     FW  UB={row['fw_ub']:10.2f}  t={row['fw_time_s']:6.2f}s  diag_err={row['fw_diag_err']:.2e}")
                print(f"     CG  UB={row['cgal_ub']:10.2f}  t={row['cgal_time_s']:6.2f}s  dual_gap={row['cgal_dual_gap']:8.2f}")
                print(f"     GW-LB (CGAL)={row['cgal_gw_lb']:10.2f}  delta_UB={row['delta_ub_pct']:+5.2f}%")
            except Exception as e:
                print(f"     *** SKIP n={n} seed={seed}: {e}")

    # Write JSON + CSV
    _write_artifacts(rows)
    return rows


def _write_artifacts(rows: list[dict]) -> None:
    json_path = os.path.join(DATA_DIR, "b176b_cgal_results.json")
    csv_path = os.path.join(DATA_DIR, "b176b_cgal_results.csv")
    md_path = os.path.join(TABLES_DIR, "b176b_scale_table.md")
    tex_path = os.path.join(TABLES_DIR, "b176b_scale_table.tex")

    with open(json_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Markdown-tabel (aggregate-per-n)
    agg = _aggregate(rows)
    with open(md_path, "w") as f:
        f.write("# B176b CGAL-SDP Scaling (3-regulier random)\n\n")
        f.write("| n | m | CGAL UB | CGAL GW-LB | CGAL diag_err | CGAL t (s) | FW UB | FW t (s) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in agg:
            f.write("| %d | %d | %.1f | %.1f | %.2e | %.2f | %.1f | %.2f |\n" % (
                r["n"], r["m"], r["cgal_ub"], r["cgal_gw_lb"],
                r["cgal_diag_err"], r["cgal_time_s"],
                r["fw_ub"], r["fw_time_s"]))

    # LaTeX booktabs
    with open(tex_path, "w") as f:
        f.write("% B176b CGAL-SDP Scaling -- generated by b176b_benchmark.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{CGAL-SDP schaalbaarheid op 3-reguliere random grafen. "
                "UB is de gecertificeerde duale bovengrens, GW-LB de beste cut uit "
                "200 Goemans-Williamson rounding-trials. $t$ is wall-time in seconden.}\n")
        f.write("\\label{tab:b176b-scale}\n")
        f.write("\\begin{tabular}{rrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("$n$ & $m$ & CGAL UB & GW-LB & diag err & $t_{\\mathrm{CGAL}}$ & FW UB & $t_{\\mathrm{FW}}$ \\\\\n")
        f.write("\\midrule\n")
        for r in agg:
            f.write("%d & %d & %.1f & %.1f & %.2e & %.2f & %.1f & %.2f \\\\\n" % (
                r["n"], r["m"], r["cgal_ub"], r["cgal_gw_lb"],
                r["cgal_diag_err"], r["cgal_time_s"],
                r["fw_ub"], r["fw_time_s"]))
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print()
    print("  Artifacts:")
    print(f"    {json_path}")
    print(f"    {csv_path}")
    print(f"    {md_path}")
    print(f"    {tex_path}")


def _aggregate(rows: list[dict]) -> list[dict]:
    """Aggregeer (mean) over seeds voor elke n."""
    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    agg = []
    for n in sorted(by_n):
        batch = by_n[n]
        agg.append({
            "n": n,
            "m": int(np.mean([r["m"] for r in batch])),
            "cgal_ub": float(np.mean([r["cgal_ub"] for r in batch])),
            "cgal_gw_lb": float(np.mean([r["cgal_gw_lb"] for r in batch])),
            "cgal_diag_err": float(np.mean([r["cgal_diag_err"] for r in batch])),
            "cgal_time_s": float(np.mean([r["cgal_time_s"] for r in batch])),
            "fw_ub": float(np.mean([r["fw_ub"] for r in batch])),
            "fw_time_s": float(np.mean([r["fw_time_s"] for r in batch])),
        })
    return agg


# ============================================================
# 4. Convergentie-curve
# ============================================================


def bench_convergence() -> None:
    _hdr("4. Convergentie-curve CGAL op n=1000 3-regulier")
    g = random_3regular(1000, seed=0)
    res = cgal_maxcut_sdp(g, max_iter=500, tol=0.0, rank_cap=48,
                          dual_every=10, verbose=False)
    # Log-snapshots
    snaps = [0, 10, 50, 100, 200, 300, 400, len(res.history) - 1]
    print("  %5s  %10s  %10s  %10s  %8s  %8s" % (
        "iter", "primal", "UB_best", "dual_gap", "diag_err", "elapsed"))
    print("  " + "-" * 70)
    for idx in snaps:
        if idx >= len(res.history):
            continue
        h = res.history[idx]
        print("  %5d  %10.2f  %10.2f  %10.2f  %8.2e  %8.2f" % (
            h["iter"], h["primal_cut"], h["ub_best"], h["dual_gap"],
            h["diag_err_max"], h["elapsed"]))
    print()


# ============================================================
# Main
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="B176b CGAL-SDP benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="sla n>=5000 over; 1 seed per n")
    parser.add_argument("--only", type=str, default=None,
                        choices=("correct", "h2h", "scale", "conv"),
                        help="run slechts een sectie")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="seeds voor scale-panel (default: [0])")
    args = parser.parse_args()

    sel = args.only
    if sel in (None, "correct"):
        bench_correctness()
    if sel in (None, "h2h") and not args.quick:
        bench_head_to_head()
    if sel in (None, "scale"):
        bench_scale_panel(include_10k=not args.quick, seeds=tuple(args.seeds))
    if sel in (None, "conv") and not args.quick:
        bench_convergence()

    return 0


if __name__ == "__main__":
    sys.exit(main())
