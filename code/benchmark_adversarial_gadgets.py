#!/usr/bin/env python3
"""B144 benchmark harness for small adversarial gadget families."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from adversarial_gadget_generator import certify_gadget, extended_adversarial_suite
from bls_solver import bls_maxcut
from pa_solver import pa_maxcut


def _try_gw_bound(n_nodes, edges):
    try:
        from b60_gw_bound import SimpleGraph, gw_sdp_bound
    except Exception:
        return None

    try:
        g = SimpleGraph(n_nodes)
        for u, v, w in edges:
            g.add_edge(int(u), int(v), float(w))
        result = gw_sdp_bound(g, verbose=False)
    except Exception:
        return None

    if result.get('sdp_bound') is None:
        return None
    return {
        'sdp_bound': float(result['sdp_bound']),
        'sdp_ratio': float(result['sdp_ratio']),
        'gw_guaranteed': float(result['gw_guaranteed']),
        'gw_ratio': float(result['gw_ratio']),
        'status': result['status'],
    }


def run_one(record):
    n_nodes = record['n_nodes']
    edges = record['edges']
    exact = certify_gadget(record)
    optimum = float(exact['exact_optimal_weight'])

    bls = bls_maxcut(
        n_nodes, edges,
        n_restarts=8,
        max_iter=300,
        max_no_improve=40,
        time_limit=0.5,
        seed=42,
        verbose=False,
    )
    pa = pa_maxcut(
        n_nodes, edges,
        n_replicas=64,
        n_temps=32,
        beta_min=0.1,
        beta_max=4.0,
        n_sweeps=2,
        time_limit=0.5,
        seed=42,
        verbose=False,
    )
    gw = _try_gw_bound(n_nodes, edges)

    return {
        'name': record['name'],
        'family': record['family'],
        'n_nodes': n_nodes,
        'n_edges': len(edges),
        'literature_note': record['literature_note'],
        'exact_optimal_weight': optimum,
        'certificate': exact['certificate'],
        'bls_cut': float(bls['best_cut']),
        'bls_gap_pct': 100.0 * max(0.0, (optimum - float(bls['best_cut'])) / max(optimum, 1.0)),
        'bls_time_s': float(bls['time_s']),
        'pa_cut': float(pa['best_cut']),
        'pa_gap_pct': 100.0 * max(0.0, (optimum - float(pa['best_cut'])) / max(optimum, 1.0)),
        'pa_time_s': float(pa['time_s']),
        'gw': gw,
    }


def write_summary_md(path, rows):
    lines = [
        '# B144 Adversarial Gadget Benchmark',
        '',
        '| gadget | family | n | m | exact | BLS | PA | GW ratio |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        gw_ratio = '-' if row['gw'] is None else f"{row['gw']['gw_ratio']:.4f}"
        lines.append(
            f"| {row['name']} | {row['family']} | {row['n_nodes']} | {row['n_edges']} | "
            f"{row['exact_optimal_weight']:.1f} | {row['bls_cut']:.1f} | {row['pa_cut']:.1f} | {gw_ratio} |"
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark adversarial gadgets')
    parser.add_argument('--output-prefix', type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_prefix = args.output_prefix or f"adversarial_gadgets_{timestamp}"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for record in extended_adversarial_suite():
        row = run_one(record)
        rows.append(row)
        print(
            f"{row['name']:>24} exact={row['exact_optimal_weight']:.1f} "
            f"BLS={row['bls_cut']:.1f} PA={row['pa_cut']:.1f}"
        )

    json_path = os.path.join(results_dir, output_prefix + '.json')
    md_path = os.path.join(results_dir, output_prefix + '.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows}, f, indent=2)
    write_summary_md(md_path, rows)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
