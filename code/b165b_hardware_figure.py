#!/usr/bin/env python3
"""B165b: Hardware-figuur voor de paper.

Leest `docs/paper/data/b165b_hardware_rows.json` (geproduceerd door
`b165b_parse_results.py`) en genereert een grouped bar-chart met 4 bars per
instance:

    noiseless | depolariserend | calibration-mirror | hardware

met per instance een horizontale dashed OPT-lijn en een AR-annotatie boven
elke hardware-bar. Dual output: matplotlib-PDF + PGFPlots TikZ `.tex`.

Usage
-----
  python3 b165b_hardware_figure.py
      [--rows docs/paper/data/b165b_hardware_rows.json]
      [--out-fig docs/paper/figures/b165b_hardware_comparison.pdf]
      [--out-tex docs/paper/figures/b165b_hardware_comparison.tex]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_rows(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Matplotlib figure
# ============================================================

def plot_hardware_comparison(rows: list[dict], out_path: str | Path) -> None:
    """Grouped bar-chart: 4 baselines × N instances, with OPT dashed + AR labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Noiseless", "Depolar.", "Cal.mirror", "Hardware"]
    colors = ["#2E86AB", "#87BBA2", "#F5B841", "#E85D75"]
    keys = ["exp_noiseless", "exp_depolarising", "exp_cal_mirror", "exp_hardware"]

    n_inst = len(rows)
    n_bar = len(labels)
    group_width = 0.78
    bar_w = group_width / n_bar

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_centers = list(range(n_inst))

    # Bars per baseline
    for b_idx, (label, color, key) in enumerate(zip(labels, colors, keys)):
        xs = [x + (b_idx - (n_bar - 1) / 2) * bar_w for x in x_centers]
        ys = [float(r.get(key, 0.0) or 0.0) for r in rows]
        ax.bar(xs, ys, width=bar_w * 0.9, label=label,
               color=color, edgecolor="black", linewidth=0.45)

        # AR-annotation above hardware bar only
        if key == "exp_hardware":
            for xi, r in zip(xs, rows):
                ar = float(r.get("approx_ratio", 0.0) or 0.0)
                hw = float(r.get(key, 0.0) or 0.0)
                ax.text(xi, hw + 0.35, f"AR={ar:.3f}",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                        color="#8B1E3F")

    # OPT-lines per instance (dashed, short horizontal segments)
    for xi, r in zip(x_centers, rows):
        opt = float(r.get("opt", 0.0) or 0.0)
        if opt <= 0:
            continue
        x0 = xi - group_width / 2
        x1 = xi + group_width / 2
        ax.hlines(opt, x0, x1, colors="black", linestyles="--",
                  linewidth=1.2, zorder=5,
                  label=(r"OPT (ILP-certified)" if xi == 0 else None))
        # Label OPT value at right edge of group
        ax.text(x1 + 0.02, opt, f"OPT={int(opt) if opt.is_integer() else opt:g}",
                va="center", ha="left", fontsize=8,
                color="black", fontweight="bold")

    # X-axis
    ax.set_xticks(x_centers)
    xticks = []
    for r in rows:
        n = r.get("n", "?")
        m = r.get("m", "?")
        xticks.append(f"{r['instance']}\n(n={n}, m={m})")
    ax.set_xticklabels(xticks, fontsize=9)

    # Y-axis
    max_val = max(
        max((float(r.get(k, 0.0) or 0.0) for k in keys), default=0.0) for r in rows
    )
    max_opt = max((float(r.get("opt", 0.0) or 0.0) for r in rows), default=0.0)
    ax.set_ylim(0.0, max(max_val, max_opt) * 1.18 + 0.5)
    ax.set_ylabel(r"QAOA-expectation $\mathbb{E}[H_C]$")

    # Backend in title
    backends = sorted({r.get("backend_name", "—") for r in rows if r.get("backend_name")})
    backend_str = " / ".join(backends) if backends else "hardware"
    ax.set_title(
        f"B165b — QAOA p=1 hardware-run op {backend_str}: "
        f"noiseless/depolar./cal-mirror vs echte hardware"
    )

    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PGFPlots TikZ emitter
# ============================================================

def emit_pgfplots_hardware(rows: list[dict], out_path: str | Path) -> None:
    """Emit standalone PGFPlots `.tex` fragment equivalent to the matplotlib figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        ("exp_noiseless",    "Noiseless",   "blue!70"),
        ("exp_depolarising", "Depolar.",    "green!50!black"),
        ("exp_cal_mirror",   "Cal.mirror",  "orange!80"),
        ("exp_hardware",     "Hardware",    "red!70"),
    ]

    # Symbolic x-coordinates
    symbolic_x = ",".join(r["instance"] for r in rows)

    lines: list[str] = []
    lines.append(r"% Auto-generated by b165b_hardware_figure.py")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"  \begin{axis}[")
    lines.append(r"    width=0.92\linewidth, height=6.5cm,")
    lines.append(r"    ybar=1pt, bar width=9pt,")
    lines.append(r"    ymin=0,")
    lines.append(r"    ylabel={$\mathbb{E}[H_C]$},")
    lines.append(r"    symbolic x coords={" + symbolic_x + r"},")
    lines.append(r"    xtick=data,")
    lines.append(r"    xticklabel style={font=\small},")
    lines.append(r"    legend pos=north west, legend style={font=\scriptsize},")
    lines.append(r"    grid=major, grid style={line width=0.1pt, draw=gray!30},")
    lines.append(r"    enlarge x limits=0.35,")
    lines.append(r"    nodes near coords style={font=\tiny},")
    lines.append(r"  ]")

    for key, label, color in keys:
        coords = " ".join(
            f"({r['instance']},{float(r.get(key, 0.0) or 0.0):.4f})"
            for r in rows
        )
        lines.append(r"    \addplot+[ybar, fill=" + color + r", draw=black!80] coordinates {" + coords + r"};")
        lines.append(r"    \addlegendentry{" + label + r"}")

    # OPT-lines (horizontal markers per instance) as separate scatter plot
    opt_coords = " ".join(
        f"({r['instance']},{float(r.get('opt', 0.0) or 0.0):.4f})"
        for r in rows if float(r.get("opt", 0.0) or 0.0) > 0
    )
    if opt_coords:
        lines.append(r"    \addplot+[only marks, mark=-, mark size=10pt, thick, black, draw=black] coordinates {" + opt_coords + r"};")
        lines.append(r"    \addlegendentry{OPT (ILP)}")

    # AR-annotations
    for r in rows:
        ar = float(r.get("approx_ratio", 0.0) or 0.0)
        hw = float(r.get("exp_hardware", 0.0) or 0.0)
        inst = r["instance"]
        lines.append(
            rf"    \node[font=\tiny, color=red!60!black, anchor=south] "
            rf"at (axis cs:{inst},{hw + 0.35:.4f}) {{AR={ar:.3f}}};"
        )

    lines.append(r"  \end{axis}")
    lines.append(r"\end{tikzpicture}")
    tex = "\n".join(lines) + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default=str(project_root / "docs" / "paper" / "data" / "b165b_hardware_rows.json"),
        help="Path to JSON with rows from b165b_parse_results.py",
    )
    parser.add_argument(
        "--out-fig",
        default=str(project_root / "docs" / "paper" / "figures" / "b165b_hardware_comparison.pdf"),
        help="Output PDF figure",
    )
    parser.add_argument(
        "--out-tex",
        default=str(project_root / "docs" / "paper" / "figures" / "b165b_hardware_comparison.tex"),
        help="Output PGFPlots TikZ fragment",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip matplotlib rendering (only emit .tex)",
    )
    args = parser.parse_args(argv)

    rows = _load_rows(args.rows)
    if not rows:
        print(f"ERROR: geen rows in {args.rows}", file=sys.stderr)
        return 1

    if not args.no_pdf:
        plot_hardware_comparison(rows, args.out_fig)
        print(f"[OK] PDF geschreven: {args.out_fig}")

    emit_pgfplots_hardware(rows, args.out_tex)
    print(f"[OK] PGFPlots TeX geschreven: {args.out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
