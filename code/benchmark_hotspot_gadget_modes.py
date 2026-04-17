#!/usr/bin/env python3
"""Compare hotspot_repair gadget modes: off, free, boundary."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from hotspot_repair import HotspotRepair


DEFAULT_CASES = [
    {'Lx': 4, 'Ly': 3, 'p_global': 1, 'p_local': 2, 'threshold': 0.4, 'warm': False},
    {'Lx': 6, 'Ly': 3, 'p_global': 1, 'p_local': 2, 'threshold': 0.4, 'warm': False},
    {'Lx': 8, 'Ly': 3, 'p_global': 1, 'p_local': 2, 'threshold': 0.4, 'warm': False},
    {'Lx': 10, 'Ly': 3, 'p_global': 1, 'p_local': 2, 'threshold': 0.4, 'warm': False},
]


def _case_label(case):
    warm = "warm" if case.get('warm') else "cold"
    return f"{case['Lx']}x{case['Ly']}_{warm}"


def run_case(case, mode):
    solver = HotspotRepair(
        case['Lx'],
        case['Ly'],
        p_global=case['p_global'],
        p_local=case['p_local'],
        frustration_threshold=case['threshold'],
        warm=case.get('warm', False),
        verbose=False,
        exact_gadget_mode=mode,
    )
    result = solver.solve()
    return {
        'case': _case_label(case),
        'mode': mode,
        'Lx': case['Lx'],
        'Ly': case['Ly'],
        'warm': bool(case.get('warm', False)),
        'ratio': float(result['ratio']),
        'ratio_tier1': float(result['ratio_tier1']),
        'delta': float(result['delta']),
        'n_hotspots': int(result['n_hotspots']),
        'n_exact_gadget_repairs': int(result.get('n_exact_gadget_repairs', 0)),
        'n_lightcone_repairs': int(result.get('n_lightcone_repairs', 0)),
        'elapsed': float(result['elapsed']),
    }


def summarize(rows):
    by_mode = {}
    by_case = {}

    for row in rows:
        mode = row['mode']
        case = row['case']
        bucket = by_mode.setdefault(mode, {
            'n_cases': 0,
            'avg_ratio': 0.0,
            'avg_delta': 0.0,
            'avg_exact_repairs': 0.0,
            'avg_lightcone_repairs': 0.0,
            'avg_elapsed': 0.0,
        })
        bucket['n_cases'] += 1
        bucket['avg_ratio'] += row['ratio']
        bucket['avg_delta'] += row['delta']
        bucket['avg_exact_repairs'] += row['n_exact_gadget_repairs']
        bucket['avg_lightcone_repairs'] += row['n_lightcone_repairs']
        bucket['avg_elapsed'] += row['elapsed']

        by_case.setdefault(case, {})[mode] = row

    for bucket in by_mode.values():
        n = max(1, bucket['n_cases'])
        bucket['avg_ratio'] /= n
        bucket['avg_delta'] /= n
        bucket['avg_exact_repairs'] /= n
        bucket['avg_lightcone_repairs'] /= n
        bucket['avg_elapsed'] /= n

    wins = {mode: 0 for mode in by_mode}
    for case_rows in by_case.values():
        best_mode = max(case_rows.values(), key=lambda row: row['ratio'])['mode']
        wins[best_mode] += 1

    return {
        'by_mode': by_mode,
        'case_wins': wins,
    }


def write_summary_md(path, rows, summary):
    lines = [
        "# Hotspot Gadget Mode Benchmark",
        "",
        "## Mode Summary",
        "",
        "| mode | avg ratio | avg delta | avg exact | avg lightcone | avg elapsed | case wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ('off', 'free', 'boundary'):
        bucket = summary['by_mode'][mode]
        lines.append(
            f"| {mode} | {bucket['avg_ratio']:.6f} | {bucket['avg_delta']:.6f} | "
            f"{bucket['avg_exact_repairs']:.2f} | {bucket['avg_lightcone_repairs']:.2f} | "
            f"{bucket['avg_elapsed']:.2f}s | {summary['case_wins'].get(mode, 0)} |"
        )

    lines.extend([
        "",
        "## Per Case",
        "",
        "| case | off ratio | free ratio | boundary ratio | best |",
        "| --- | ---: | ---: | ---: | --- |",
    ])

    case_names = sorted({row['case'] for row in rows})
    for case in case_names:
        case_rows = {row['mode']: row for row in rows if row['case'] == case}
        best = max(case_rows.values(), key=lambda row: row['ratio'])['mode']
        lines.append(
            f"| {case} | {case_rows['off']['ratio']:.6f} | {case_rows['free']['ratio']:.6f} | "
            f"{case_rows['boundary']['ratio']:.6f} | {best} |"
        )

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare hotspot gadget modes")
    parser.add_argument('--output-prefix', type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_prefix = args.output_prefix or f"hotspot_gadget_mode_compare_{timestamp}"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for case in DEFAULT_CASES:
        for mode in ('off', 'free', 'boundary'):
            row = run_case(case, mode)
            rows.append(row)
            print(
                f"{row['case']:>10}  mode={mode:<8} ratio={row['ratio']:.6f} "
                f"delta={row['delta']:+.6f} exact={row['n_exact_gadget_repairs']} "
                f"lightcone={row['n_lightcone_repairs']}"
            )

    summary = summarize(rows)
    json_path = os.path.join(results_dir, output_prefix + '.json')
    md_path = os.path.join(results_dir, output_prefix + '.md')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'rows': rows,
            'summary': summary,
        }, f, indent=2)
    write_summary_md(md_path, rows, summary)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
