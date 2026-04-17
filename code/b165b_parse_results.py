#!/usr/bin/env python3
"""
B165b: Resultaten-parser & paper-tabel emitter.

Leest alle baseline-JSONs + (optioneel) hardware-job-bundles, voegt een ILP-OPT
kolom toe via B159, en produceert:
  - docs/paper/tables/b165b_hardware_table.tex   (LaTeX booktabs)
  - docs/paper/tables/b165b_hardware_table.md    (Markdown)

Kolommen
--------
  Instance | n | OPT (ILP) | Noiseless | Depolar. | Cal.mirror | Hardware | AR

Waarbij AR = hardware_E[H_C] / OPT als AR-kolom; andere kolommen tonen E[H_C].

Als hardware-bundle ontbreekt schrijven we "—" in die kolom; de tabel is dan
alvast "paper-ready" voor 3 van de 4 kolommen.

CLI
---
  python b165b_parse_results.py
  python b165b_parse_results.py --jobs-dir <pad> --baselines-dir <pad>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b165b_hardware_submit import (
    INSTANCE_SPECS,
    build_instance,
    load_bundle,
    SubmissionBundle,
)


# ============================================================
# Row container
# ============================================================

@dataclass
class TableRow:
    instance: str
    n: int
    m: int
    opt: float
    exp_noiseless: float
    exp_depolarising: float
    exp_cal_mirror: float
    exp_hardware: float | None
    best_hardware: int | None
    backend_name: str
    approx_ratio: float | None
    gammas: list[float]
    betas: list[float]


# ============================================================
# Assembly
# ============================================================

def _read_baseline(dir_: Path, inst: str, name: str) -> dict[str, Any] | None:
    p = dir_ / f"{inst}_{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def assemble_rows(jobs_dir: Path, baselines_dir: Path,
                  instances: list[str] | None = None,
                  ilp_time: float = 10.0) -> list[TableRow]:
    """Combineer alle artifacts tot tabel-rijen."""
    from b159_ilp_oracle import maxcut_ilp_highs
    from b165_qiskit_runtime import maxcut_value_from_counts, best_cut_from_counts

    instances = instances or list(INSTANCE_SPECS.keys())
    rows: list[TableRow] = []
    for inst in instances:
        g = build_instance(inst)
        # OPT via ILP
        try:
            ilp = maxcut_ilp_highs(g, time_limit=ilp_time)
            ov = ilp.get("opt_value") if isinstance(ilp, dict) else \
                getattr(ilp, "opt_value", None)
            opt = float(ov) if ov is not None else float("nan")
        except Exception:
            opt = float("nan")

        nl = _read_baseline(baselines_dir, inst, "noiseless") or {}
        dp = _read_baseline(baselines_dir, inst, "depolarising") or {}
        cal = _read_baseline(baselines_dir, inst, "calibration_mirror") or {}

        # Hardware-bundle zoeken: job_id-bestand (status=COMPLETED) of
        # prepared_<inst>.json (status=PREPARED/no hardware yet).
        hw_exp: float | None = None
        hw_best: int | None = None
        backend_name = ""
        gammas: list[float] = []
        betas: list[float] = []
        if jobs_dir.exists():
            for p in jobs_dir.glob("*.json"):
                try:
                    b = load_bundle(p)
                except Exception:
                    continue
                if b.instance != inst:
                    continue
                if b.gammas and not gammas:
                    gammas = b.gammas
                    betas = b.betas
                if b.status == "COMPLETED" and b.counts:
                    hw_exp = float(maxcut_value_from_counts(b.counts, g))
                    hw_best, _ = best_cut_from_counts(b.counts, g)
                    hw_best = int(hw_best)
                    backend_name = b.backend_name
                    break

        ar: float | None = None
        if hw_exp is not None and opt and opt > 0:
            ar = hw_exp / opt

        rows.append(TableRow(
            instance=inst, n=g.n, m=g.n_edges, opt=opt,
            exp_noiseless=float(nl.get("expectation", float("nan"))),
            exp_depolarising=float(dp.get("expectation", float("nan"))),
            exp_cal_mirror=float(cal.get("expectation", float("nan"))),
            exp_hardware=hw_exp, best_hardware=hw_best,
            backend_name=backend_name, approx_ratio=ar,
            gammas=gammas, betas=betas,
        ))
    return rows


# ============================================================
# Emitters
# ============================================================

def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{nd}f}"


def emit_markdown(rows: list[TableRow]) -> str:
    out: list[str] = []
    out.append("# B165b Hardware-run vs. Aer-baselines\n")
    out.append("| Instance | n | m | OPT (ILP) | Noiseless | Depolar. | "
               "Cal.mirror | Hardware | Best(HW) | Backend | AR |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        out.append(
            f"| {r.instance} | {r.n} | {r.m} | "
            f"{_fmt(r.opt, 0)} | {_fmt(r.exp_noiseless)} | "
            f"{_fmt(r.exp_depolarising)} | {_fmt(r.exp_cal_mirror)} | "
            f"{_fmt(r.exp_hardware)} | "
            f"{r.best_hardware if r.best_hardware is not None else '—'} | "
            f"{r.backend_name or '—'} | {_fmt(r.approx_ratio)} |"
        )
    out.append("")
    out.append("* E[H_C] = QAOA-expectation van de MaxCut-Hamiltoniaan; "
               "OPT via B159 HiGHS ILP; AR = E_hardware / OPT.*")
    return "\n".join(out) + "\n"


def emit_latex(rows: list[TableRow]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{B165b --- QAOA $p=1$ op IBM Quantum hardware "
                 r"vergeleken met drie Aer-baselines. "
                 r"$E[H_C]$ is de MaxCut-expectation uit de sample-distributie; "
                 r"OPT is ILP-certified via B159.}")
    lines.append(r"\label{tab:b165b-hardware}")
    lines.append(r"\begin{tabular}{l r r r r r r r r l r}")
    lines.append(r"\toprule")
    lines.append(r"Instance & $n$ & $m$ & OPT & Noiseless & Depolar.\ & "
                 r"Cal.\ mirror & Hardware & Best(HW) & Backend & AR \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r.instance} & {r.n} & {r.m} & {_fmt(r.opt, 0)} & "
            f"{_fmt(r.exp_noiseless)} & {_fmt(r.exp_depolarising)} & "
            f"{_fmt(r.exp_cal_mirror)} & {_fmt(r.exp_hardware)} & "
            f"{r.best_hardware if r.best_hardware is not None else '---'} & "
            f"{r.backend_name or '---'} & {_fmt(r.approx_ratio)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def emit_json(rows: list[TableRow]) -> str:
    return json.dumps([r.__dict__ for r in rows], indent=2, ensure_ascii=False)


# ============================================================
# Disk-writer
# ============================================================

def save_table(rows: list[TableRow], out_root: Path) -> dict[str, Path]:
    tables = out_root / "tables"
    data = out_root / "data"
    tables.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    md_path = tables / "b165b_hardware_table.md"
    tex_path = tables / "b165b_hardware_table.tex"
    json_path = data / "b165b_hardware_rows.json"

    md_path.write_text(emit_markdown(rows), encoding="utf-8")
    tex_path.write_text(emit_latex(rows), encoding="utf-8")
    json_path.write_text(emit_json(rows), encoding="utf-8")
    return {"md": md_path, "tex": tex_path, "json": json_path}


# ============================================================
# CLI
# ============================================================

def _default_jobs_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent / "docs" / "paper" / "hardware" / "jobs"


def _default_baselines_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent / "docs" / "paper" / "hardware" / "baselines"


def _default_paper_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent / "docs" / "paper"


def main() -> int:
    ap = argparse.ArgumentParser(description="B165b parse results → paper-tabel")
    ap.add_argument("--jobs-dir", default=None)
    ap.add_argument("--baselines-dir", default=None)
    ap.add_argument("--out-root", default=None,
                    help="Paper-root (default docs/paper). Tabellen naar "
                         "<out-root>/tables/, data naar <out-root>/data/.")
    ap.add_argument("--ilp-time", type=float, default=10.0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    jobs = Path(args.jobs_dir) if args.jobs_dir else _default_jobs_dir()
    base = Path(args.baselines_dir) if args.baselines_dir else _default_baselines_dir()
    out = Path(args.out_root) if args.out_root else _default_paper_root()
    insts = [args.only] if args.only else None

    rows = assemble_rows(jobs, base, instances=insts, ilp_time=args.ilp_time)
    paths = save_table(rows, out)
    print("=== B165b paper-tabel ===")
    print(emit_markdown(rows))
    print("Artifacts:")
    for k, v in paths.items():
        print(f"  {k:<4s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
