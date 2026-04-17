#!/usr/bin/env python3
"""B154c: Combined leaderboard — Gset + BiqMac + DIMACS.

Runt ILP-oracle (B159) + GW-bound (B60) + MPQS-BP/lightcone (B80) op een
gemeenschappelijk benchmark-panel uit drie datasets:

  - Gset: handmatig geselecteerde ingebouwde grafen (via gset_loader)
  - BiqMac: synthetische instanties uit `b154_biqmac_loader`
  - DIMACS: ingebouwde fixtures uit `b154_dimacs_loader`

Doel: één unified tabel met cut-value vs ILP-OPT voor alle drie datasets,
zodat de ZornQ-paper een multi-dataset benchmark kan tonen en schaalbreuk
type-specifiek te diagnosticeren is.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph
from b60_gw_bound import SimpleGraph, gw_sdp_bound, brute_force_maxcut
from b159_ilp_oracle import maxcut_ilp_highs
from b80_mpqs import mpqs_classical_bp, mpqs_lightcone

from b154_biqmac_loader import (
    biqmac_spinglass_2d,
    biqmac_torus_2d,
    biqmac_pm1s,
    biqmac_g05,
)
from b154_dimacs_loader import load_fixture
from gset_loader import make_petersen, make_cube, make_grid, make_cycle


# ============================================================
# WeightedGraph ↔ SimpleGraph bridge
# ============================================================

def to_simple(g_weighted: WeightedGraph) -> SimpleGraph:
    """Converteer rqaoa.WeightedGraph → b60.SimpleGraph."""
    n = g_weighted.n_nodes
    g = SimpleGraph(n)
    for i, j, w in g_weighted.edges():
        g.add_edge(i, j, w)
    return g


def from_builtin(name: str, fn, *args, **kwargs):
    """Gset-stijl ingebouwde wrapper: fn() → (graph, bks) of (graph, ...)."""
    res = fn(*args, **kwargs)
    g_w = res[0]
    bks = res[1] if len(res) > 1 else None
    return name, to_simple(g_w), bks


# ============================================================
# Instance panel
# ============================================================

def build_panel() -> list[tuple[str, str, SimpleGraph, float | None]]:
    """Bouw unified benchmark-panel.

    Return: lijst van (dataset, name, SimpleGraph, known_opt_or_None).
    """
    panel: list[tuple[str, str, SimpleGraph, float | None]] = []

    # --- Gset (ingebouwde, small-scale) ---
    name, g, bks = from_builtin("petersen", make_petersen)
    panel.append(("Gset", name, g, float(bks) if bks is not None else None))
    name, g, bks = from_builtin("cube", make_cube)
    panel.append(("Gset", name, g, float(bks) if bks is not None else None))
    name, g, bks = from_builtin("grid_4x3", make_grid, 4, 3)
    panel.append(("Gset", name, g, float(bks) if bks is not None else None))
    name, g, bks = from_builtin("cycle_8", make_cycle, 8)
    panel.append(("Gset", name, g, float(bks) if bks is not None else None))

    # --- BiqMac (synthetic) ---
    panel.append(("BiqMac", "spinglass2d_L4_s0", to_simple(biqmac_spinglass_2d(4, seed=0)), None))
    panel.append(("BiqMac", "spinglass2d_L5_s0", to_simple(biqmac_spinglass_2d(5, seed=0)), None))
    panel.append(("BiqMac", "torus2d_L4_s1",    to_simple(biqmac_torus_2d(4, seed=1)),     None))
    panel.append(("BiqMac", "pm1s_n20_s2",       to_simple(biqmac_pm1s(20, p=0.3, seed=2)), None))
    panel.append(("BiqMac", "g05_n12_s3",        to_simple(biqmac_g05(12, seed=3)),          None))

    # --- DIMACS (ingebouwde fixtures) ---
    for dname in ("petersen", "myciel3", "k4", "c6", "queen5_5"):
        g_w, _, _ = load_fixture(dname)
        panel.append(("DIMACS", dname, to_simple(g_w), None))

    return panel


# ============================================================
# Solver runner
# ============================================================

def run_one(name: str, g: SimpleGraph, known_opt: float | None,
            ilp_time: float = 10.0) -> dict:
    """Run alle solvers op één graaf en geef resultaten terug."""
    res: dict = {"name": name, "n": g.n, "m": g.n_edges}

    # ILP-oracle (certificeerbaar OPT of feasible incumbent)
    ilp = maxcut_ilp_highs(g, time_limit=ilp_time)
    res["opt"] = ilp["opt_value"]
    res["certified"] = ilp["certified"]
    res["t_ilp"] = ilp["wall_time"]

    # GW SDP-bound (alleen kleine grafen)
    try:
        if g.n <= 25:
            gw = gw_sdp_bound(g, verbose=False)
            res["gw"] = gw.get("sdp_bound")
        else:
            res["gw"] = None
    except Exception:
        res["gw"] = None

    # MPQS classical BP + refine
    t0 = time.time()
    bp = mpqs_classical_bp(g, verbose=False)
    res["bp"] = bp["cut_value"]
    res["t_bp"] = time.time() - t0

    # MPQS lightcone (alleen kleine grafen; lightcones kunnen anders blow-up'en)
    if g.n <= 20:
        t0 = time.time()
        lc = mpqs_lightcone(g, radius=2, verbose=False)
        res["lc"] = lc["cut_value"]
        res["t_lc"] = time.time() - t0
    else:
        res["lc"] = None
        res["t_lc"] = None

    # Known optimum (Gset BKS)
    res["known_opt"] = known_opt

    return res


# ============================================================
# Print leaderboard
# ============================================================

def print_leaderboard(panel_rows: list[tuple[str, dict]]) -> None:
    print("=" * 130)
    print(" B154 Combined Leaderboard: Gset + BiqMac + DIMACS")
    print("=" * 130)
    header = (
        f"{'Dataset':<8}{'Instance':<22}"
        f"{'n':>4}{'m':>5}"
        f"{'OPT':>8}{'cert':>5}{'t_ILP':>7}"
        f"{'GW':>9}"
        f"{'BP':>7}{'r_BP':>7}{'t_BP':>7}"
        f"{'LC':>7}{'r_LC':>7}{'t_LC':>7}"
    )
    print(header)
    print("-" * len(header))

    n_total = 0
    n_bp_opt = 0
    n_lc_opt = 0
    dataset_stats: dict[str, dict] = {}

    for dataset, r in panel_rows:
        opt = r.get("opt")
        cert = "✓" if r.get("certified") else "~"
        gw = r.get("gw")
        bp = r.get("bp")
        lc = r.get("lc")

        opt_s = f"{opt:.1f}" if opt is not None else "—"
        gw_s = f"{gw:.2f}" if gw is not None else "—"
        bp_s = f"{bp:.1f}" if bp is not None else "—"
        lc_s = f"{lc:.1f}" if lc is not None else "—"
        r_bp = (bp / opt) if (opt and bp and opt > 0) else float("nan")
        r_lc = (lc / opt) if (opt and lc and opt > 0) else float("nan")
        r_bp_s = f"{r_bp:.3f}" if r_bp == r_bp else "—"
        r_lc_s = f"{r_lc:.3f}" if r_lc == r_lc else "—"
        t_lc_s = f"{r['t_lc']:.3f}" if r.get("t_lc") is not None else "—"

        print(f"{dataset:<8}{r['name']:<22}"
              f"{r['n']:>4}{r['m']:>5}"
              f"{opt_s:>8}{cert:>5}{r['t_ilp']:>7.3f}"
              f"{gw_s:>9}"
              f"{bp_s:>7}{r_bp_s:>7}{r['t_bp']:>7.3f}"
              f"{lc_s:>7}{r_lc_s:>7}{t_lc_s:>7}")

        n_total += 1
        if opt is not None:
            if bp is not None and abs(bp - opt) < 1e-6:
                n_bp_opt += 1
            if lc is not None and abs(lc - opt) < 1e-6:
                n_lc_opt += 1
            stats = dataset_stats.setdefault(dataset, {"n": 0, "bp_opt": 0, "lc_opt": 0, "lc_runs": 0})
            stats["n"] += 1
            if bp is not None and abs(bp - opt) < 1e-6:
                stats["bp_opt"] += 1
            if lc is not None:
                stats["lc_runs"] += 1
                if abs(lc - opt) < 1e-6:
                    stats["lc_opt"] += 1

    print("-" * len(header))
    print()
    print(f"  Totaal: {n_total} instanties, {sum(1 for _, r in panel_rows if r.get('certified'))} ILP-certified")
    print()
    print(f"  Global:   MPQS-BP = OPT:       {n_bp_opt}/{n_total}")
    print(f"  Global:   MPQS-Lightcone = OPT: {n_lc_opt}/{n_total}")
    print()
    for ds, stats in dataset_stats.items():
        bp_frac = f"{stats['bp_opt']}/{stats['n']}"
        lc_frac = f"{stats['lc_opt']}/{stats['lc_runs']}" if stats['lc_runs'] else "—"
        print(f"  {ds:<8} BP-OPT = {bp_frac:<8} | LC-OPT = {lc_frac}")
    print()
    print("  cert: ✓ = HiGHS proof;  ~ = feasible incumbent (time-limit)")
    print("  r_*: cut / ILP-OPT.  t_*: wall-time (s).  GW: SDP upper bound (‒ voor n>25)")


def run() -> None:
    panel = build_panel()
    print(f"Building panel: {len(panel)} instances...")
    results: list[tuple[str, dict]] = []
    for dataset, name, g, known_opt in panel:
        r = run_one(name, g, known_opt)
        results.append((dataset, r))

    print_leaderboard(results)


if __name__ == "__main__":
    run()
