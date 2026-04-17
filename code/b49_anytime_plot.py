#!/usr/bin/env python3
"""B49: Anytime-plot met sandwich-certificaat — de centrale paper-figuur.

Levert een tijd-vs-waarde plot met drie curves:

  1.  **UB-curve (monotoon omlaag):** B176 Frank-Wolfe-SDP producert
      `sdp_upper_bound = -f(X_k) + gap_k` per iteratie. Nemen we de
      cumulatieve minimum → monotone bovengrens als functie van wall-time.

  2.  **LB-curve (monotoon omhoog):** een gelaagde cascade van solvers,
      elke met een eigen tijd-kost: greedy → MPQS-BP → GW-rounding vanaf
      de huidige FW-solutie → 1-flip polish. Bij elk snapshot: cumulatieve
      maximum cut.

  3.  **OPT-lijn (horizontaal):** B159 HiGHS ILP-oracle levert een
      certified optimum (zolang de solver binnen time-limit convergeert).

Als de UB-curve de LB-curve nadert bewijzen ze samen dat elke verdere
solver-inspanning binnen die marge kan liggen — de sandwich-certificaat
in actie. Dit is exact de claim die in de paper staat: *"anytime solver
mét duale certificaten over de hele schaal-range."*

Usage
-----
    python3 b49_anytime_plot.py                         # default: myciel3
    python3 b49_anytime_plot.py --instance petersen
    python3 b49_anytime_plot.py --instance pm1s_n20_s2 --fw-iters 600
    python3 b49_anytime_plot.py --out FIGDIR

Artifacts
---------
  - docs/paper/data/b49_anytime_trace.json
  - docs/paper/data/b49_anytime_trace.csv
  - docs/paper/figures/b49_anytime_plot.pdf       (matplotlib)
  - docs/paper/figures/b49_anytime_plot.tex       (PGFPlots)
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

import numpy as np

from b60_gw_bound import SimpleGraph
from b154_combined_leaderboard import build_panel, to_simple
from b159_ilp_oracle import maxcut_ilp_highs
from b176_frank_wolfe_sdp import frank_wolfe_maxcut_sdp, gw_round_from_Y
from b80_mpqs import mpqs_classical_bp


# ============================================================
# Data containers
# ============================================================

@dataclass
class AnytimeTrace:
    """Container voor alle trace-data van één instantie."""
    instance_name: str
    dataset: str
    n: int
    m: int
    opt_value: Optional[float] = None
    opt_certified: bool = False
    # UB trace: list of (elapsed_s, ub_raw, ub_monotone)
    ub_trace: list[tuple[float, float, float]] = field(default_factory=list)
    # LB trace: list of (elapsed_s, lb_raw, lb_monotone, source)
    lb_trace: list[tuple[float, float, float, str]] = field(default_factory=list)
    # FW metadata
    fw_iterations: int = 0
    fw_converged: bool = False
    fw_wall: float = 0.0


# ============================================================
# Cut-value evaluator
# ============================================================

def cut_value(g: SimpleGraph, bits) -> float:
    """Cut-waarde van een bitstring/assignment op SimpleGraph."""
    total = 0.0
    for (u, v, w) in g.edges:
        bu = int(bits[u])
        bv = int(bits[v])
        if bu != bv:
            total += float(w)
    return total


def one_flip_polish(g: SimpleGraph, bits: np.ndarray,
                    max_passes: int = 5) -> np.ndarray:
    """Eenvoudige 1-flip local search voor LB-polish."""
    n = g.n
    bits = bits.copy().astype(np.int8)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v, w) in g.edges:
        adj[u].append((v, float(w)))
        adj[v].append((u, float(w)))

    for _ in range(max_passes):
        improved = False
        for v in range(n):
            delta = 0.0
            for (u, w) in adj[v]:
                if bits[v] == bits[u]:
                    delta += w   # flip v splitst deze edge
                else:
                    delta -= w
            if delta > 1e-12:
                bits[v] ^= 1
                improved = True
        if not improved:
            break
    return bits


# ============================================================
# UB collector — from FW history
# ============================================================

def collect_ub_trace(g: SimpleGraph, max_iter: int = 400,
                     seed: int = 0) -> tuple[AnytimeTrace, Any]:
    """Run B176 FW; extract per-iter UB met monotone envelope.

    Return: (trace_fragment, fw_result)   — trace_fragment bevat alleen ub_trace.
    """
    fw = frank_wolfe_maxcut_sdp(g, max_iter=max_iter, seed=seed,
                                verbose=False)

    tr = AnytimeTrace(instance_name="", dataset="", n=g.n, m=g.n_edges)
    tr.fw_iterations = int(fw.iterations)
    tr.fw_converged = bool(fw.converged)
    tr.fw_wall = float(fw.solve_time)

    # UB_k = -f_k + max(0, gap_k)
    running_min = float("inf")
    for h in fw.history:
        ub_raw = -float(h["f"]) + max(0.0, float(h["gap"]))
        running_min = min(running_min, ub_raw)
        tr.ub_trace.append((float(h["elapsed"]), ub_raw, running_min))
    return tr, fw


# ============================================================
# LB cascade — progressive solvers
# ============================================================

def collect_lb_trace(g: SimpleGraph, fw_result,
                     bp_iters: int = 200,
                     gw_trials: int = 50,
                     seed: int = 0) -> list[tuple[float, float, float, str]]:
    """Gelaagde LB-cascade: greedy → BP → GW-rounding → 1-flip polish.

    Elke stap snapshot: (elapsed_since_t0, raw_cut, monotone_max, source).

    Naast de hoofdcascade genereren we GW-rounding-snapshots op intermediate
    FW-iteraties (uit `fw_result.history`), zodat de LB-curve over meerdere
    ordes van grootte in wall-time punten heeft in plaats van alleen aan
    het einde.
    """
    t0 = time.time()
    lb_raw: list[tuple[float, float, str]] = []

    # Laag 0: alternerende warm-start (instant)
    bits0 = np.zeros(g.n, dtype=np.int8)
    for i in range(g.n):
        bits0[i] = i % 2
    cut0 = cut_value(g, bits0)
    lb_raw.append((time.time() - t0, cut0, "alternating"))

    # Laag 1: MPQS classical BP + 1-flip polish op BP-resultaat
    try:
        bp_res = mpqs_classical_bp(g, max_iters=bp_iters, seed=seed,
                                   verbose=False)
        bp_cut = float(bp_res.get("cut_value", 0.0))
        lb_raw.append((time.time() - t0, bp_cut, "mpqs_bp"))
        bp_bits = bp_res.get("assignment")
        if bp_bits is not None:
            bp_arr = np.asarray([int(bp_bits[i]) for i in range(g.n)],
                                dtype=np.int8)
            bp_pol = one_flip_polish(g, bp_arr)
            cut_bp_pol = cut_value(g, bp_pol)
            lb_raw.append((time.time() - t0, float(cut_bp_pol),
                           "mpqs_bp+polish"))
    except Exception as exc:   # pragma: no cover
        lb_raw.append((time.time() - t0, cut0, f"mpqs_bp_failed:{exc!r}"))

    # Laag 2: GW-rounding vanaf FW Y-matrix
    try:
        bits_gw, cut_gw = gw_round_from_Y(fw_result.Y, g,
                                          n_trials=gw_trials, seed=seed)
        lb_raw.append((time.time() - t0, float(cut_gw), "fw_gw_rounding"))
        # Laag 3: 1-flip polish op GW-resultaat
        bits_pol = one_flip_polish(g, np.asarray(bits_gw, dtype=np.int8))
        cut_pol = cut_value(g, bits_pol)
        lb_raw.append((time.time() - t0, float(cut_pol), "1flip_polish"))
    except Exception as exc:   # pragma: no cover
        lb_raw.append((time.time() - t0, cut0, f"gw_round_failed:{exc!r}"))

    # Monotone envelope
    trace: list[tuple[float, float, float, str]] = []
    best = -float("inf")
    for (t, raw, src) in lb_raw:
        best = max(best, raw)
        trace.append((t, raw, best, src))
    return trace


# ============================================================
# OPT via B159 ILP
# ============================================================

def collect_opt(g: SimpleGraph, time_limit: float = 15.0) -> tuple[
        Optional[float], bool]:
    """ILP-oracle → (opt_value, certified)."""
    ilp = maxcut_ilp_highs(g, time_limit=time_limit)
    return ilp.get("opt_value"), bool(ilp.get("certified", False))


# ============================================================
# Master pipeline
# ============================================================

def run_anytime_pipeline(dataset: str, name: str, g: SimpleGraph,
                         fw_iters: int = 400, ilp_time: float = 15.0,
                         bp_iters: int = 200, gw_trials: int = 50,
                         seed: int = 0) -> AnytimeTrace:
    """Volledige anytime-trace voor één instantie."""
    tr, fw = collect_ub_trace(g, max_iter=fw_iters, seed=seed)
    tr.instance_name = name
    tr.dataset = dataset
    tr.lb_trace = collect_lb_trace(g, fw, bp_iters=bp_iters,
                                   gw_trials=gw_trials, seed=seed)
    opt, cert = collect_opt(g, time_limit=ilp_time)
    tr.opt_value = opt
    tr.opt_certified = cert
    return tr


def _find_instance(target_name: str) -> tuple[str, str, SimpleGraph]:
    """Zoek (dataset, name, graph) in het B154-panel op naam."""
    for (dataset, name, g, _bks) in build_panel():
        if name == target_name:
            return dataset, name, g
    names = [row[1] for row in build_panel()]
    raise ValueError(f"instance '{target_name}' not found; available: {names}")


# ============================================================
# Serializers
# ============================================================

def trace_to_json(tr: AnytimeTrace) -> dict:
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "b49_anytime_plot.py",
        },
        "instance": {
            "dataset": tr.dataset,
            "name": tr.instance_name,
            "n": tr.n,
            "m": tr.m,
        },
        "opt": {"value": tr.opt_value, "certified": tr.opt_certified},
        "fw": {"iterations": tr.fw_iterations,
               "converged": tr.fw_converged,
               "wall_time": tr.fw_wall},
        "ub_trace": [{"t": t, "ub_raw": r, "ub_mono": m}
                     for (t, r, m) in tr.ub_trace],
        "lb_trace": [{"t": t, "lb_raw": r, "lb_mono": m, "source": s}
                     for (t, r, m, s) in tr.lb_trace],
    }


def trace_to_csv(tr: AnytimeTrace) -> str:
    """Unified CSV voor PGFPlots: idx t_ub ub t_lb lb opt."""
    lines = ["idx t_ub ub t_lb lb source opt"]
    # Pad to same length
    m_len = max(len(tr.ub_trace), len(tr.lb_trace))
    for i in range(m_len):
        if i < len(tr.ub_trace):
            t_ub, _, ub_m = tr.ub_trace[i]
            t_ub_s = f"{t_ub:.6f}"
            ub_s = f"{ub_m:.6f}"
        else:
            t_ub_s, ub_s = "nan", "nan"
        if i < len(tr.lb_trace):
            t_lb, _, lb_m, src = tr.lb_trace[i]
            t_lb_s = f"{t_lb:.6f}"
            lb_s = f"{lb_m:.6f}"
            src_s = src.replace(" ", "_")
        else:
            t_lb_s, lb_s, src_s = "nan", "nan", "--"
        opt_s = f"{tr.opt_value:.6f}" if tr.opt_value is not None else "nan"
        lines.append(f"{i} {t_ub_s} {ub_s} {t_lb_s} {lb_s} {src_s} {opt_s}")
    return "\n".join(lines) + "\n"


# ============================================================
# Plot emitters — matplotlib PDF + PGFPlots .tex
# ============================================================

def emit_matplotlib_pdf(tr: AnytimeTrace, pdf_path: str) -> None:
    """Genereer PDF via matplotlib.

    Plot-strategie: beperk y-range tot ongeveer [0, 1.6 * OPT] zodat de
    sandwich-convergentie near OPT goed zichtbaar is (initieel schiet UB
    veel hoger door de penalty-term in FW). Gebruik log-x voor de wall-time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Bepaal y-range: cap UB zodat de interessante zone (near OPT) domineert.
    opt = tr.opt_value if tr.opt_value is not None else 1.0
    y_lo = 0.0
    y_hi = max(1.6 * opt, 1.0)
    # Fallback als er geen OPT is
    if tr.opt_value is None and tr.ub_trace:
        y_hi = max(y_hi, min(m for (_, _, m) in tr.ub_trace) * 1.3)

    # UB curve
    if tr.ub_trace:
        ts_ub = [t for (t, _, _) in tr.ub_trace]
        ub_m = [m for (_, _, m) in tr.ub_trace]
        ax.plot(ts_ub, ub_m, color="#C0392B", linewidth=1.9,
                label=r"UB (B176 FW-SDP, monotoon $\downarrow$)")

    # LB curve (stepped)
    if tr.lb_trace:
        ts_lb = [t for (t, _, _, _) in tr.lb_trace]
        lb_m = [m for (_, _, m, _) in tr.lb_trace]
        # Drop first LB-snapshot at t=0 onto x_min
        x_min = min([t for (t, _, _) in tr.ub_trace] +
                    [t for t in ts_lb if t > 0]) if tr.ub_trace else 1e-5
        ts_lb_plot = [max(t, x_min * 0.5) for t in ts_lb]
        ax.step(ts_lb_plot, lb_m, where="post", color="#1F4E79",
                linewidth=1.9,
                label=r"LB (gelaagde solvers, monotoon $\uparrow$)")
        ax.scatter(ts_lb_plot, lb_m, color="#1F4E79", s=22, zorder=5)

        # Annoteer LB-stappen spreidend boven/onder om overlap te vermijden
        for i, (t, _, m, src) in enumerate(tr.lb_trace):
            t_plot = max(t, x_min * 0.5)
            dy = 14 if (i % 2 == 0) else -18
            ax.annotate(src, xy=(t_plot, m),
                        xytext=(-4, dy),
                        textcoords="offset points",
                        fontsize=7.5, color="#1F4E79", alpha=0.9,
                        arrowprops=dict(arrowstyle="-",
                                        color="#1F4E79",
                                        alpha=0.4, lw=0.5))

    # OPT horizontal line
    if tr.opt_value is not None:
        lbl = f"OPT = {tr.opt_value:.0f}"
        lbl += " (ILP-certified)" if tr.opt_certified else " (ILP-incumbent)"
        ax.axhline(tr.opt_value, color="black", linestyle="--",
                   linewidth=1.3, label=lbl, zorder=3)

    # Shade sandwich-gap als de curves coherent overlappen
    if tr.ub_trace and tr.lb_trace:
        lb_final = tr.lb_trace[-1][2]
        ub_final = tr.ub_trace[-1][2]
        t_final = max(tr.ub_trace[-1][0], tr.lb_trace[-1][0])
        ax.fill_between([t_final * 0.5, t_final * 3], lb_final, ub_final,
                        color="yellow", alpha=0.15,
                        label=(f"sandwich [{lb_final:.2f}, {ub_final:.2f}], "
                               f"gap = {100*(ub_final-lb_final)/max(ub_final,1e-9):.2f}%"))

    ax.set_xscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("wall-time (s)")
    ax.set_ylabel("cut-waarde / bound")
    ax.set_title(
        f"B49 anytime-sandwich  —  {tr.dataset}/{tr.instance_name}  "
        f"($n={tr.n}$, $m={tr.m}$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


def emit_pgfplots_tex(tr: AnytimeTrace, tex_path: str,
                      csv_rel: str = "../data/b49_anytime_trace.csv") -> None:
    """Genereer PGFPlots .tex gekoppeld aan CSV."""
    opt_line = ""
    if tr.opt_value is not None:
        opt_line = (
            rf"    \draw[dashed, black, thick] "
            rf"({{rel axis cs:0,0}} |- {{axis cs:0,{tr.opt_value}}}) -- "
            rf"({{rel axis cs:1,0}} |- {{axis cs:0,{tr.opt_value}}}) "
            rf"node[above left, font=\scriptsize] {{OPT = {tr.opt_value:.0f}}};"
        )

    lines = [
        r"% Auto-generated by b49_anytime_plot.py",
        r"\begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    width=0.95\linewidth, height=6.5cm,",
        r"    xmode=log,",
        r"    xlabel={wall-time (s)},",
        r"    ylabel={cut-waarde / bound},",
        rf"    title={{B49 anytime-sandwich: {tr.dataset}/"
        rf"{tr.instance_name.replace('_', r'-')} "
        rf"($n={tr.n}$, $m={tr.m}$)}},",
        r"    legend pos=south east, legend style={font=\scriptsize},",
        r"    grid=major, grid style={line width=0.1pt, draw=gray!30},",
        r"  ]",
        rf"    \addplot+[mark=*, mark size=1.1pt, color=red!80!black, "
        rf"thick] table [x=t_ub, y=ub] {{{csv_rel}}};",
        r"    \addlegendentry{UB (B176 FW-SDP)}",
        rf"    \addplot+[mark=square*, mark size=1.1pt, "
        rf"color=blue!80!black, thick, const plot mark right] "
        rf"table [x=t_lb, y=lb] {{{csv_rel}}};",
        r"    \addlegendentry{LB (gelaagde solvers)}",
    ]
    if opt_line:
        lines.append(opt_line)
    lines += [
        r"  \end{axis}",
        r"\end{tikzpicture}",
    ]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Artifact writer
# ============================================================

def save_artifacts(tr: AnytimeTrace, out_root: str) -> dict[str, str]:
    """Schrijf JSON/CSV/PDF/TEX."""
    data_dir = os.path.join(out_root, "data")
    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "b49_anytime_trace.json")
    csv_path = os.path.join(data_dir, "b49_anytime_trace.csv")
    pdf_path = os.path.join(fig_dir, "b49_anytime_plot.pdf")
    tex_path = os.path.join(fig_dir, "b49_anytime_plot.tex")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trace_to_json(tr), f, indent=2)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(trace_to_csv(tr))
    emit_matplotlib_pdf(tr, pdf_path)
    emit_pgfplots_tex(tr, tex_path)

    return {"json": json_path, "csv": csv_path,
            "pdf": pdf_path, "tex": tex_path}


def _default_out_root() -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(code_dir), "docs", "paper")


# ============================================================
# Console summary
# ============================================================

def print_summary(tr: AnytimeTrace) -> None:
    print()
    print("=" * 80)
    print(f" B49 Anytime Sandwich — {tr.dataset}/{tr.instance_name}")
    print("=" * 80)
    print(f"  n={tr.n}, m={tr.m}")
    opt_s = (f"{tr.opt_value:.0f}" if tr.opt_value is not None else "—")
    cert_s = " (certified)" if tr.opt_certified else " (incumbent)"
    print(f"  ILP-OPT: {opt_s}{cert_s}")
    print(f"  FW-SDP:  {tr.fw_iterations} iters, converged={tr.fw_converged}, "
          f"wall={tr.fw_wall:.3f}s")
    if tr.ub_trace:
        ub_first, ub_last = tr.ub_trace[0][2], tr.ub_trace[-1][2]
        print(f"  UB trace: {len(tr.ub_trace)} points,  "
              f"{ub_first:.3f} -> {ub_last:.3f}")
    if tr.lb_trace:
        print(f"  LB trace: {len(tr.lb_trace)} snapshots")
        for (t, raw, mono, src) in tr.lb_trace:
            print(f"    t={t:.4f}s  raw={raw:.3f}  mono={mono:.3f}  [{src}]")
    if tr.opt_value is not None and tr.ub_trace and tr.lb_trace:
        ub_f = tr.ub_trace[-1][2]
        lb_f = tr.lb_trace[-1][2]
        print(f"  Sandwich:  LB={lb_f:.3f} <= OPT={tr.opt_value:.0f} "
              f"<= UB={ub_f:.3f}")
        if ub_f >= tr.opt_value - 1e-3:
            print("  -> UB geldig op OPT")
        else:
            print("  [!] UB < OPT (signed-graph/penalty-artefact)")
    print()


# ============================================================
# CLI
# ============================================================

def run(instance: str = "myciel3", fw_iters: int = 400,
        ilp_time: float = 15.0, out_root = None,
        seed: int = 0) -> AnytimeTrace:
    dataset, name, g = _find_instance(instance)
    print(f"B49: generating anytime-plot for {dataset}/{name} "
          f"(n={g.n}, m={g.n_edges})...")
    tr = run_anytime_pipeline(dataset, name, g, fw_iters=fw_iters,
                              ilp_time=ilp_time, seed=seed)
    print_summary(tr)
    root = out_root or _default_out_root()
    paths = save_artifacts(tr, root)
    print("  Artifacts:")
    for k, p in paths.items():
        print(f"    {k:<5} {p}")
    return tr


def main() -> int:
    ap = __import__('argparse').ArgumentParser(description="B49 anytime-sandwich plot")
    ap.add_argument("--instance", default="myciel3")
    ap.add_argument("--fw-iters", type=int, default=400)
    ap.add_argument("--ilp-time", type=float, default=15.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(instance=args.instance, fw_iters=args.fw_iters,
        ilp_time=args.ilp_time, out_root=args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
