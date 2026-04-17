#!/usr/bin/env python3
"""B177: Paper figure pipeline — JSON benchmark → matplotlib + PGFPlots.

This module provides a reproducible pipeline that converts benchmark JSON
artifacts (from B154 combined leaderboard or similar) into:

  1. Matplotlib PDF figures (for quick-view + arxiv submission).
  2. PGFPlots `.tex` fragments (for native LaTeX typesetting via pgfplots).

The pipeline is CLI-driven:

  python3 b177_figure_pipeline.py \
      --out-fig docs/paper/figures \
      --out-data docs/paper/data

Key functions:

  - `collect_leaderboard_data(...)`: runs the B154 panel and emits JSON.
  - `plot_leaderboard_ratio(data, path)`: bar chart of r_BP vs r_LC.
  - `plot_ilp_scaling(data, path)`: wall-time vs n for ILP-oracle.
  - `emit_pgfplots_leaderboard(data, path)`: PGFPlots-native `.tex`.

Design notes:
  * Matplotlib is used in *no-latex* mode for portability (no external
    LaTeX needed to regenerate PDFs).
  * All numerical data is dumped to JSON for full reproducibility.
  * PGFPlots fragments reference `data/*.csv` via `\\addplot table {...}`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from /code/ directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# JSON I/O
# ============================================================

def dump_json(obj: dict, path: str | Path) -> None:
    """Dump a dict to pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def load_json(path: str | Path) -> dict:
    """Load JSON back into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Benchmark data collection
# ============================================================

def collect_leaderboard_data(ilp_time: float = 10.0) -> dict:
    """Run the B154 combined leaderboard and capture results as JSON.

    Returns a dict with keys:
      - 'meta': {'timestamp', 'ilp_time_limit', 'n_instances'}
      - 'rows': list of dicts with per-instance benchmark data
    """
    from b154_combined_leaderboard import build_panel, run_one

    panel = build_panel()
    rows: list[dict] = []
    for dataset, name, g, known_opt in panel:
        r = run_one(name, g, known_opt, ilp_time=ilp_time)
        r["dataset"] = dataset
        # Make everything JSON-serializable
        for k, v in list(r.items()):
            if v is None:
                continue
            if isinstance(v, (bool, int, float, str)):
                continue
            r[k] = str(v)
        rows.append(r)

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ilp_time_limit": ilp_time,
            "n_instances": len(rows),
        },
        "rows": rows,
    }


def collect_ilp_scaling_data(sizes: list[int] | None = None,
                             seeds: list[int] | None = None,
                             ilp_time: float = 10.0) -> dict:
    """Run ILP-oracle on random 3-regular graphs, capture wall-time vs n.

    Returns dict with 'meta' and 'points' (list of {n, seed, t_ilp, opt}).
    """
    import random
    from rqaoa import WeightedGraph
    from b159_ilp_oracle import maxcut_ilp_highs
    from b154_combined_leaderboard import to_simple

    if sizes is None:
        sizes = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    if seeds is None:
        seeds = [0, 1, 2]

    def random_3reg(n: int, seed: int) -> WeightedGraph:
        """Simple uniform random 3-regular-ish graph (may not be exactly 3-regular for small n)."""
        rng = random.Random(seed)
        g = WeightedGraph()
        for v in range(n):
            g.add_node(v)
        edges: set = set()
        target_edges = max(1, (3 * n) // 2)
        attempts = 0
        while len(edges) < target_edges and attempts < 50 * target_edges:
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) not in edges:
                edges.add((i, j))
                g.add_edge(i, j, 1.0)
            attempts += 1
        return g

    points: list[dict] = []
    for n in sizes:
        for seed in seeds:
            g_w = random_3reg(n, seed)
            g = to_simple(g_w)
            ilp = maxcut_ilp_highs(g, time_limit=ilp_time)
            points.append({
                "n": n,
                "m": g.n_edges,
                "seed": seed,
                "t_ilp": float(ilp["wall_time"]),
                "opt": float(ilp["opt_value"]) if ilp["opt_value"] is not None else None,
                "certified": bool(ilp["certified"]),
            })

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ilp_time_limit": ilp_time,
            "graph_type": "random_uniform_3reg_target",
            "sizes": sizes,
            "seeds": seeds,
        },
        "points": points,
    }


# ============================================================
# Matplotlib figures
# ============================================================

def plot_leaderboard_ratio(data: dict, path: str | Path) -> None:
    """Bar chart: per-instance ratio r_BP vs r_LC, grouped by dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["rows"]
    names = [r["name"] for r in rows]
    ratios_bp = []
    ratios_lc = []
    for r in rows:
        opt = r.get("opt")
        bp = r.get("bp")
        lc = r.get("lc")
        ratios_bp.append((bp / opt) if (opt and bp) else 0.0)
        ratios_lc.append((lc / opt) if (opt and lc) else 0.0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(names))
    width = 0.38
    xs_bp = [i - width / 2 for i in x]
    xs_lc = [i + width / 2 for i in x]

    ax.bar(xs_bp, ratios_bp, width=width, label="MPQS-BP + 1-flip",
           color="#2E86AB", edgecolor="black", linewidth=0.4)
    ax.bar(xs_lc, ratios_lc, width=width, label="MPQS-Lightcone",
           color="#E85D75", edgecolor="black", linewidth=0.4)

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--",
               label=r"ILP-certified $\mathrm{OPT}$")
    ax.axhline(0.87856, color="gray", linewidth=0.5, linestyle=":",
               label=r"GW bound $\alpha_{\mathrm{GW}}$")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(r"ratio $r = \mathrm{cut}/\mathrm{OPT}$")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("ZornQ combined leaderboard — Gset + BiqMac + DIMACS")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Dataset-separator hints
    last = rows[0]["dataset"]
    for i, r in enumerate(rows):
        if r["dataset"] != last:
            ax.axvline(i - 0.5, color="black", linewidth=0.5, alpha=0.4)
            last = r["dataset"]

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ilp_scaling(data: dict, path: str | Path) -> None:
    """Scatter+line: ILP-oracle wall-time vs graph size n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    points = data["points"]
    bucket: dict[int, list[float]] = defaultdict(list)
    for p in points:
        bucket[p["n"]].append(p["t_ilp"])

    ns = sorted(bucket.keys())
    means = [sum(bucket[n]) / len(bucket[n]) for n in ns]
    mins = [min(bucket[n]) for n in ns]
    maxs = [max(bucket[n]) for n in ns]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ns, means, "o-", color="#2E86AB", linewidth=1.4,
            markersize=5, label="mean")
    ax.fill_between(ns, mins, maxs, color="#2E86AB", alpha=0.18, label="min–max")

    # All scatter points
    xs = [p["n"] for p in points]
    ys = [p["t_ilp"] for p in points]
    ax.scatter(xs, ys, color="#E85D75", alpha=0.55, s=16, label="per-seed")

    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":", label="1 s")
    ax.set_xlabel(r"graph size $n$")
    ax.set_ylabel(r"ILP-oracle wall-time $t_{\mathrm{ILP}}$ (s)")
    ax.set_yscale("log")
    ax.set_title("HiGHS ILP-oracle: wall-time vs $n$ (random near-3-regular)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(which="both", alpha=0.25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PGFPlots / CSV emission
# ============================================================

def emit_leaderboard_csv(data: dict, path: str | Path) -> None:
    """Emit a PGFPlots-friendly CSV (space-separated, first col = idx)."""
    rows = data["rows"]
    lines = ["idx dataset name n m opt bp lc r_bp r_lc"]
    for i, r in enumerate(rows):
        opt = r.get("opt") or 0
        bp = r.get("bp") or 0
        lc = r.get("lc") or 0
        r_bp = (bp / opt) if opt else 0.0
        r_lc = (lc / opt) if opt else 0.0
        lines.append(f"{i} {r['dataset']} {r['name']} {r['n']} {r['m']} "
                     f"{opt} {bp} {lc} {r_bp:.4f} {r_lc:.4f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def emit_ilp_scaling_csv(data: dict, path: str | Path) -> None:
    """Emit ILP scaling data as PGFPlots CSV."""
    points = data["points"]
    lines = ["n m seed t_ilp opt certified"]
    for p in points:
        cert = 1 if p["certified"] else 0
        opt = p["opt"] if p["opt"] is not None else "nan"
        lines.append(f"{p['n']} {p['m']} {p['seed']} {p['t_ilp']:.6f} {opt} {cert}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def emit_pgfplots_leaderboard(data: dict, path: str | Path,
                              csv_rel: str = "../data/b154_leaderboard.csv") -> None:
    """Emit a standalone PGFPlots `.tex` fragment for the leaderboard."""
    tex = r"""% Auto-generated by b177_figure_pipeline.py
\begin{tikzpicture}
  \begin{axis}[
    width=0.95\linewidth, height=6cm,
    ybar=1pt, bar width=5pt,
    ymin=0, ymax=1.1,
    ylabel={$r = \mathrm{cut}/\mathrm{OPT}$},
    xtick=data, xticklabel style={rotate=40, anchor=east, font=\scriptsize},
    xticklabels from table={""" + csv_rel + r"""}{name},
    legend pos=south east, legend style={font=\scriptsize},
    grid=major, grid style={line width=0.1pt, draw=gray!30},
  ]
    \addplot+[fill=blue!60, draw=black!70] table [x=idx, y=r_bp] {""" + csv_rel + r"""};
    \addlegendentry{MPQS-BP}
    \addplot+[fill=red!60, draw=black!70] table [x=idx, y=r_lc] {""" + csv_rel + r"""};
    \addlegendentry{MPQS-Lightcone}
    \draw[dashed, black] ({rel axis cs:0,0} |- {axis cs:0,1}) --
                         ({rel axis cs:1,0} |- {axis cs:0,1});
  \end{axis}
\end{tikzpicture}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def emit_pgfplots_ilp_scaling(data: dict, path: str | Path,
                              csv_rel: str = "../data/ilp_scaling.csv") -> None:
    """Emit a standalone PGFPlots `.tex` fragment for ILP-scaling."""
    tex = r"""% Auto-generated by b177_figure_pipeline.py
\begin{tikzpicture}
  \begin{axis}[
    width=0.8\linewidth, height=6cm,
    xlabel={graph size $n$},
    ylabel={$t_{\mathrm{ILP}}$ (s)},
    ymode=log,
    grid=major, grid style={line width=0.1pt, draw=gray!30},
    legend pos=north west, legend style={font=\scriptsize},
  ]
    \addplot+[only marks, mark=o, mark size=1.5pt, color=red!70]
      table [x=n, y=t_ilp] {""" + csv_rel + r"""};
    \addlegendentry{per-seed ILP wall-time}
    \draw[dashed, gray]
      (axis cs:5,1) -- (axis cs:55,1);
  \end{axis}
\end{tikzpicture}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


# ============================================================
# Top-level pipeline
# ============================================================

def run_pipeline(out_fig: str | Path,
                 out_data: str | Path,
                 do_figures: bool = True,
                 do_scaling: bool = True,
                 scaling_sizes: list[int] | None = None,
                 verbose: bool = True) -> dict:
    """Run the end-to-end pipeline.

    1. Collect benchmark data (B154 leaderboard + optional ILP-scaling).
    2. Dump JSON artifacts to `out_data`.
    3. Emit PGFPlots CSV + `.tex` fragments to `out_data`.
    4. Render matplotlib PDFs to `out_fig`.
    """
    out_fig = Path(out_fig)
    out_data = Path(out_data)
    out_fig.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    # Leaderboard
    if verbose:
        print("[b177] Collecting B154 combined leaderboard…")
    lb = collect_leaderboard_data()
    dump_json(lb, out_data / "b154_leaderboard.json")
    emit_leaderboard_csv(lb, out_data / "b154_leaderboard.csv")
    emit_pgfplots_leaderboard(lb, out_data / "b154_leaderboard.tex")
    summary["leaderboard"] = {
        "n_rows": len(lb["rows"]),
        "json": str(out_data / "b154_leaderboard.json"),
        "csv": str(out_data / "b154_leaderboard.csv"),
        "tex": str(out_data / "b154_leaderboard.tex"),
    }

    if do_figures:
        if verbose:
            print("[b177] Rendering leaderboard PDF…")
        plot_leaderboard_ratio(lb, out_fig / "b154_leaderboard_ratio.pdf")
        summary["leaderboard"]["pdf"] = str(out_fig / "b154_leaderboard_ratio.pdf")

    # ILP-scaling
    if do_scaling:
        if verbose:
            print("[b177] Collecting ILP-scaling data…")
        sizes = scaling_sizes or [10, 16, 22, 28, 34, 40, 46]
        sc = collect_ilp_scaling_data(sizes=sizes, seeds=[0, 1, 2])
        dump_json(sc, out_data / "ilp_scaling.json")
        emit_ilp_scaling_csv(sc, out_data / "ilp_scaling.csv")
        emit_pgfplots_ilp_scaling(sc, out_data / "ilp_scaling.tex")
        summary["scaling"] = {
            "n_points": len(sc["points"]),
            "json": str(out_data / "ilp_scaling.json"),
        }

        if do_figures:
            if verbose:
                print("[b177] Rendering ILP-scaling PDF…")
            plot_ilp_scaling(sc, out_fig / "ilp_scaling.pdf")
            summary["scaling"]["pdf"] = str(out_fig / "ilp_scaling.pdf")

    if verbose:
        print("[b177] Pipeline complete.")
        print(json.dumps(summary, indent=2))

    return summary


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZornQ B177 paper figure pipeline")
    p.add_argument("--out-fig", type=str, default="docs/paper/figures",
                   help="Output directory for rendered PDFs")
    p.add_argument("--out-data", type=str, default="docs/paper/data",
                   help="Output directory for JSON/CSV/PGFPlots data")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip matplotlib rendering (data-only run)")
    p.add_argument("--no-scaling", action="store_true",
                   help="Skip ILP-scaling experiment")
    p.add_argument("--fast", action="store_true",
                   help="Smaller scaling sweep for quick smoke-tests")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress status prints")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sizes = [10, 14, 18] if args.fast else None
    run_pipeline(
        out_fig=args.out_fig,
        out_data=args.out_data,
        do_figures=not args.no_figures,
        do_scaling=not args.no_scaling,
        scaling_sizes=sizes,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
