#!/usr/bin/env python3
"""B186: Solver-Selector Benchmark op het unified 14-instance panel.

Publiceert de keuze van de auto-dispatcher (B130) + het kwaliteitscertificaat
(B131) voor elke instantie in het B154/B177-panel. Zelfs zonder QC-voordeel
is een sterke auto-dispatcher mét duale certificaten een publiceerbaar
datapunt — dit script produceert direct de paper-tabel.

Inputs
------
Het 14-instance panel uit `b154_combined_leaderboard.build_panel()`:
  - 4 × Gset (petersen, cube, grid_4x3, cycle_8)
  - 5 × BiqMac (spinglass2d_L4_s0, spinglass2d_L5_s0, torus2d_L4_s1,
               pm1s_n20_s2, g05_n12_s3)
  - 5 × DIMACS (petersen, myciel3, k4, c6, queen5_5)

Solvers
-------
  - ILP        (B159 HiGHS MILP, certified EXACT-level certificaat)
  - FW-SDP     (B176 Frank-Wolfe-sandwich, BOUNDED/NEAR_EXACT certificaat)
  - Cograph-DP (B170 O(n^3) als graaf cograph is; anders skip)
  - Dispatcher (B130 ZornDispatcher.auto; gekozen strategy + certificate)

Output
------
  - docs/paper/data/b186_selector_results.json   (raw per-instance data)
  - docs/paper/data/b186_selector_results.csv    (pgfplots-ready)
  - docs/paper/tables/b186_selector_table.tex    (booktabs LaTeX tabel)
  - docs/paper/tables/b186_selector_table.md     (markdown spiegel)

Usage
-----
    python3 b186_solver_selector_benchmark.py
    python3 b186_solver_selector_benchmark.py --quick          # kleine subset
    python3 b186_solver_selector_benchmark.py --out DIR        # overrides
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import SimpleGraph
from b154_combined_leaderboard import build_panel, to_simple
from b159_ilp_oracle import maxcut_ilp_highs
from b176_frank_wolfe_sdp import frank_wolfe_maxcut_sdp
from b170_twin_width import is_cograph, cograph_maxcut_exact
from auto_dispatcher import solve_maxcut as dispatcher_solve
from quality_certificate import (
    CertificateLevel,
    certify_maxcut_from_fw,
    certify_maxcut_from_ilp,
)


# ============================================================
# Solver result container
# ============================================================

@dataclass
class SolverResult:
    solver: str
    cut_value: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    gap_pct: Optional[float] = None
    level: Optional[str] = None
    wall_time: float = 0.0
    extra: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


# ============================================================
# Solver runners
# ============================================================

def _graph_edges(g: SimpleGraph) -> list[tuple[int, int, float]]:
    """Extract (u, v, w) edges from a SimpleGraph."""
    return [(int(u), int(v), float(w)) for (u, v, w) in g.edges]


def run_ilp(g: SimpleGraph, time_limit: float = 15.0) -> SolverResult:
    """B159 ILP-oracle (HiGHS) → LEVEL 1 EXACT indien certified."""
    t0 = time.time()
    ilp = maxcut_ilp_highs(g, time_limit=time_limit)
    wall = time.time() - t0

    edges = _graph_edges(g)
    cert = certify_maxcut_from_ilp(ilp, n=g.n, edges=edges,
                                   cut_value=ilp.get("opt_value"))
    return SolverResult(
        solver="ilp",
        cut_value=ilp.get("opt_value"),
        upper_bound=cert.upper_bound,
        lower_bound=cert.lower_bound,
        gap_pct=cert.gap,
        level=cert.level.name if cert.level else None,
        wall_time=wall,
        extra={
            "certified": bool(ilp.get("certified", False)),
            "status": ilp.get("status", ""),
            "cut_bits": ilp.get("cut_bits"),
        },
    )


def run_fw_sdp(g: SimpleGraph, max_iter: int = 400,
               seed: int = 0) -> SolverResult:
    """B176 Frank-Wolfe-sandwich → LEVEL 2 BOUNDED / NEAR_EXACT."""
    t0 = time.time()
    fw = frank_wolfe_maxcut_sdp(g, max_iter=max_iter, seed=seed,
                                verbose=False)
    wall = time.time() - t0

    edges = _graph_edges(g)
    cert = certify_maxcut_from_fw(fw, n=g.n, edges=edges,
                                  cut_value=fw.feasible_cut_lb)
    return SolverResult(
        solver="fw_sdp",
        cut_value=fw.feasible_cut_lb,
        upper_bound=cert.upper_bound,
        lower_bound=cert.lower_bound,
        gap_pct=cert.gap,
        level=cert.level.name if cert.level else None,
        wall_time=wall,
        extra={
            "iterations": int(fw.iterations),
            "converged": bool(fw.converged),
            "final_gap": float(fw.final_gap),
            "gw_guaranteed": float(fw.gw_guaranteed),
            "penalty": float(fw.penalty),
        },
    )


def run_cograph_dp(g: SimpleGraph) -> SolverResult:
    """B170 cograph-DP — alleen als graaf P_4-vrij is (O(n^3))."""
    edges = [(int(u), int(v)) for (u, v, _w) in g.edges]

    if not is_cograph(g.n, edges):
        return SolverResult(
            solver="cograph_dp", skipped=True,
            skip_reason="not a cograph (contains P_4)",
        )

    t0 = time.time()
    res = cograph_maxcut_exact(g.n, edges)
    wall = time.time() - t0

    cut_val = float(res.get("value", 0.0))
    return SolverResult(
        solver="cograph_dp",
        cut_value=cut_val,
        upper_bound=cut_val,   # exact => LB = UB
        lower_bound=cut_val,
        gap_pct=0.0,
        level=CertificateLevel.EXACT.name,
        wall_time=wall,
        extra={"method": "cotree_dp_O(n^3)"},
    )


def run_dispatcher(g: SimpleGraph, time_budget: float = 5.0,
                   seed: int = 42) -> SolverResult:
    """B130 auto-dispatcher → logt gekozen strategy + certificate."""
    edges = [(int(u), int(v), float(w)) for (u, v, w) in g.edges]

    t0 = time.time()
    try:
        dres = dispatcher_solve(g.n, edges, time_budget=time_budget,
                                seed=seed, verbose=False)
    except Exception as exc:  # pragma: no cover — defensief
        return SolverResult(
            solver="dispatcher_auto", skipped=True,
            skip_reason=f"dispatcher raised: {exc!r}",
        )
    wall = time.time() - t0

    return SolverResult(
        solver="dispatcher_auto",
        cut_value=float(dres.best_cut),
        upper_bound=None,
        lower_bound=float(dres.best_cut),
        gap_pct=None,
        level=str(dres.certificate) if dres.certificate else None,
        wall_time=wall,
        extra={
            "strategy": dres.strategy,
            "tier": dres.tier,
            "solvers_used": list(dres.solvers_used),
            "ratio_to_m": dres.ratio,
            "is_exact": bool(dres.is_exact),
        },
    )


# ============================================================
# Panel runner
# ============================================================

def run_panel(panel: list[tuple[str, str, SimpleGraph, Optional[float]]],
              ilp_time: float = 15.0,
              fw_iters: int = 400,
              dispatcher_budget: float = 5.0,
              seed: int = 0) -> list[dict]:
    """Run alle solvers op alle instanties en retourneer per-instance dicts."""
    out: list[dict] = []
    for (dataset, name, g, known_opt) in panel:
        print(f"  > [{dataset}] {name}  (n={g.n}, m={g.n_edges})")

        res_ilp = run_ilp(g, time_limit=ilp_time)
        res_fw = run_fw_sdp(g, max_iter=fw_iters, seed=seed)
        res_cog = run_cograph_dp(g)
        res_dis = run_dispatcher(g, time_budget=dispatcher_budget, seed=42)

        row = {
            "dataset": dataset,
            "name": name,
            "n": g.n,
            "m": g.n_edges,
            "known_opt": known_opt,
            "solvers": {
                "ilp": asdict(res_ilp),
                "fw_sdp": asdict(res_fw),
                "cograph_dp": asdict(res_cog),
                "dispatcher_auto": asdict(res_dis),
            },
        }
        out.append(row)
    return out


# ============================================================
# LaTeX + Markdown emitters
# ============================================================

def _fmt(val, sig=".3f", dash="--"):
    if val is None:
        return dash
    try:
        return f"{float(val):{sig}}"
    except (TypeError, ValueError):
        return dash


def emit_latex_table(rows: list[dict]) -> str:
    """Booktabs-tabel: (dataset, name, n, m, OPT, cert, FW-UB, FW-LB, dispatch-strategy, cert-level, t_disp)."""
    lines: list[str] = []
    lines.append(r"% Auto-generated by b186_solver_selector_benchmark.py")
    lines.append(r"\begin{tabular}{llrrrrrrlr}")
    lines.append(r"\toprule")
    lines.append(r" Dataset & Instance & $n$ & $m$ "
                 r"& ILP-OPT & cert & FW-UB & FW-LB "
                 r"& Auto-strategy & $t_{\mathrm{auto}}$ (s) \\")
    lines.append(r"\midrule")

    for r in rows:
        s = r["solvers"]
        ilp = s["ilp"]
        fw = s["fw_sdp"]
        dis = s["dispatcher_auto"]

        cert = "EXACT" if ilp["extra"].get("certified") else "inc"
        row = " & ".join([
            r["dataset"],
            r["name"].replace("_", r"\_"),
            str(r["n"]),
            str(r["m"]),
            _fmt(ilp["cut_value"], ".1f"),
            cert,
            _fmt(fw["upper_bound"], ".2f"),
            _fmt(fw["lower_bound"], ".2f"),
            (dis["extra"].get("strategy", "--") or "--").replace("_", r"\_"),
            _fmt(dis["wall_time"], ".3f"),
        ]) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def emit_markdown_table(rows: list[dict]) -> str:
    """Markdown-spiegel van de LaTeX-tabel voor quick-view in PRs."""
    out: list[str] = []
    out.append("| Dataset | Instance | n | m | ILP-OPT | cert | FW-UB | FW-LB "
               "| Auto-strategy | t_auto (s) | Auto-cert |")
    out.append("|---|---|--:|--:|--:|---|--:|--:|---|--:|---|")
    for r in rows:
        s = r["solvers"]
        ilp = s["ilp"]
        fw = s["fw_sdp"]
        dis = s["dispatcher_auto"]
        cert = "EXACT" if ilp["extra"].get("certified") else "inc"
        out.append(
            f"| {r['dataset']} | {r['name']} | {r['n']} | {r['m']} "
            f"| {_fmt(ilp['cut_value'], '.1f')} | {cert} "
            f"| {_fmt(fw['upper_bound'], '.2f')} "
            f"| {_fmt(fw['lower_bound'], '.2f')} "
            f"| {dis['extra'].get('strategy', '-')} "
            f"| {_fmt(dis['wall_time'], '.3f')} "
            f"| {dis.get('level') or '-'} |"
        )
    return "\n".join(out) + "\n"


def emit_csv(rows: list[dict]) -> str:
    """PGFPlots-vriendelijke CSV: idx dataset name n m ilp_opt fw_ub fw_lb disp_strategy disp_level t_ilp t_fw t_disp."""
    lines = ["idx dataset name n m ilp_opt fw_ub fw_lb disp_strategy "
             "disp_level t_ilp t_fw t_disp"]
    for i, r in enumerate(rows):
        s = r["solvers"]
        ilp, fw, dis = s["ilp"], s["fw_sdp"], s["dispatcher_auto"]
        lines.append(" ".join([
            str(i),
            r["dataset"],
            r["name"],
            str(r["n"]),
            str(r["m"]),
            f"{ilp['cut_value']:.4f}" if ilp["cut_value"] is not None else "nan",
            f"{fw['upper_bound']:.4f}" if fw["upper_bound"] is not None else "nan",
            f"{fw['lower_bound']:.4f}" if fw["lower_bound"] is not None else "nan",
            (dis["extra"].get("strategy") or "--").replace(" ", "_"),
            (dis.get("level") or "--").replace(" ", "_"),
            f"{ilp['wall_time']:.4f}",
            f"{fw['wall_time']:.4f}",
            f"{dis['wall_time']:.4f}",
        ]))
    return "\n".join(lines) + "\n"


# ============================================================
# Console leaderboard
# ============================================================

def print_console(rows: list[dict]) -> None:
    """Print formatted leaderboard naar stdout."""
    print()
    print("=" * 120)
    print(" B186 Solver-Selector Benchmark — 14-instance panel")
    print("=" * 120)
    header = (f"{'Dataset':<8}{'Instance':<22}"
              f"{'n':>4}{'m':>5}"
              f"{'ILP':>8}{'cert':>6}"
              f"{'FW-UB':>9}{'FW-LB':>9}"
              f"{'Cog':>7}"
              f"{'Auto':>18}{'t_auto':>9}{'lvl':>8}")
    print(header)
    print("-" * len(header))

    for r in rows:
        s = r["solvers"]
        ilp, fw, cog, dis = s["ilp"], s["fw_sdp"], s["cograph_dp"], s["dispatcher_auto"]
        cert = "EXACT" if ilp["extra"].get("certified") else "inc"
        cog_s = _fmt(cog["cut_value"], ".1f") if not cog["skipped"] else "—"
        strat = (dis["extra"].get("strategy") or "—")[:17]
        lvl = (dis.get("level") or "—")[:7]

        print(f"{r['dataset']:<8}{r['name']:<22}"
              f"{r['n']:>4}{r['m']:>5}"
              f"{_fmt(ilp['cut_value'], '.1f'):>8}{cert:>6}"
              f"{_fmt(fw['upper_bound'], '.2f'):>9}"
              f"{_fmt(fw['lower_bound'], '.2f'):>9}"
              f"{cog_s:>7}"
              f"{strat:>18}{_fmt(dis['wall_time'], '.3f'):>9}{lvl:>8}")

    print("-" * len(header))
    n_tot = len(rows)
    n_cert = sum(1 for r in rows if r["solvers"]["ilp"]["extra"].get("certified"))
    n_cog = sum(1 for r in rows if not r["solvers"]["cograph_dp"]["skipped"])
    n_match = 0
    for r in rows:
        opt = r["solvers"]["ilp"]["cut_value"]
        dis_cut = r["solvers"]["dispatcher_auto"]["cut_value"]
        if opt is not None and dis_cut is not None and abs(opt - dis_cut) < 1e-6:
            n_match += 1

    print(f"  Totaal: {n_tot} instanties")
    print(f"  ILP-certified:     {n_cert}/{n_tot}")
    print(f"  Cograph-eligible:  {n_cog}/{n_tot}")
    print(f"  Auto == ILP-OPT:   {n_match}/{n_tot}")
    print()


# ============================================================
# Artifact writer
# ============================================================

def save_artifacts(rows: list[dict], out_root: str) -> dict[str, str]:
    """Schrijf JSON/CSV/TEX/MD-artifacts en geef paden terug."""
    data_dir = os.path.join(out_root, "data")
    tbl_dir = os.path.join(out_root, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "b186_selector_results.json")
    csv_path = os.path.join(data_dir, "b186_selector_results.csv")
    tex_path = os.path.join(tbl_dir, "b186_selector_table.tex")
    md_path = os.path.join(tbl_dir, "b186_selector_table.md")

    meta = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_instances": len(rows),
            "generator": "b186_solver_selector_benchmark.py",
        },
        "rows": rows,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(emit_csv(rows))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(emit_latex_table(rows))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(emit_markdown_table(rows))

    return {
        "json": json_path, "csv": csv_path,
        "tex": tex_path, "md": md_path,
    }


# ============================================================
# CLI entry
# ============================================================

def _default_out_root() -> str:
    """docs/paper/ in dezelfde project-root als deze module."""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(code_dir)
    return os.path.join(proj, "docs", "paper")


def run(quick: bool = False, out_root: Optional[str] = None,
        ilp_time: float = 15.0, fw_iters: int = 400) -> dict:
    """Publieke entry-point: run panel + save artifacts. Return result-dict."""
    panel = build_panel()
    if quick:
        # Voor CI: neem alleen de 4 kleinste instanties (één per dataset).
        panel = [row for row in panel if row[2].n <= 10][:4]
    print(f"B186: running {len(panel)} instanties...")
    t0 = time.time()
    rows = run_panel(panel, ilp_time=ilp_time, fw_iters=fw_iters)
    total = time.time() - t0

    print_console(rows)
    root = out_root or _default_out_root()
    paths = save_artifacts(rows, root)

    print(f"  Total wall-time: {total:.1f}s")
    print("  Artifacts:")
    for k, p in paths.items():
        print(f"    {k:<5} {p}")

    return {"rows": rows, "paths": paths, "wall_time": total}


def main() -> int:
    ap = argparse.ArgumentParser(description="B186 solver-selector benchmark")
    ap.add_argument("--quick", action="store_true",
                    help="run alleen kleinste subset (voor CI / smoke)")
    ap.add_argument("--out", type=str, default=None,
                    help="override output root (default: docs/paper/)")
    ap.add_argument("--ilp-time", type=float, default=15.0,
                    help="ILP time-limit per instance (default 15s)")
    ap.add_argument("--fw-iters", type=int, default=400,
                    help="Frank-Wolfe max_iter (default 400)")
    args = ap.parse_args()
    run(quick=args.quick, out_root=args.out,
        ilp_time=args.ilp_time, fw_iters=args.fw_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
