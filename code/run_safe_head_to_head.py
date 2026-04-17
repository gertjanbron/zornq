#!/usr/bin/env python3
"""Run brede cuda_pa vs dSBM benchmark in veilige, resumable batches."""

import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def _numeric_gset_sort_key(name):
    stem = name.upper().replace('.TXT', '')
    if stem.startswith('G') and stem[1:].isdigit():
        return int(stem[1:])
    return 10**9


def _discover_gset_names(gset_dir):
    names = []
    for fname in os.listdir(gset_dir):
        stem = fname.upper().replace('.TXT', '')
        if stem.startswith('G') and stem[1:].isdigit():
            names.append(stem)
    return sorted(set(names), key=_numeric_gset_sort_key)


def _chunked(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _summarize(results):
    by_solver = {}
    for row in results:
        solver = row['solver']
        slot = by_solver.setdefault(solver, {'gaps': [], 'solved': 0})
        if row.get('gap_pct') is not None:
            slot['gaps'].append(float(row['gap_pct']))
        if row.get('error') is None:
            slot['solved'] += 1
    summary = {}
    for solver, slot in by_solver.items():
        gaps = slot['gaps']
        summary[solver] = {
            'instances_ok': slot['solved'],
            'avg_gap_pct': (sum(gaps) / len(gaps)) if gaps else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Safe resumable head-to-head runner')
    parser.add_argument('--time-limit', type=float, default=15.0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--solvers', type=str, default='cuda_pa,dsbm')
    parser.add_argument('--graphs', type=str, default=None,
                        help='Optional comma-separated graph names')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(code_dir, '..'))
    gset_dir = os.path.join(project_root, 'gset')

    if args.output_dir:
        output_dir = args.output_dir
    else:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, 'results', f'head_to_head_safe_{stamp}')
    os.makedirs(output_dir, exist_ok=True)

    if args.graphs:
        graph_names = [g.strip().upper() for g in args.graphs.split(',') if g.strip()]
    else:
        graph_names = _discover_gset_names(gset_dir)

    if not graph_names:
        raise SystemExit('No Gset graphs discovered')

    benchmark_py = os.path.join(code_dir, 'gset_benchmark.py')
    combined_path = os.path.join(output_dir, 'combined_results.json')
    manifest_path = os.path.join(output_dir, 'manifest.json')

    all_results = []
    completed_batches = []

    if args.resume and os.path.exists(combined_path):
        payload = _load_results(combined_path)
        all_results = payload.get('results', [])
        completed_batches = payload.get('completed_batches', [])

    batches = list(_chunked(graph_names, args.batch_size))
    total = len(batches)

    for batch_idx, batch in enumerate(batches, start=1):
        batch_name = f'batch_{batch_idx:02d}'
        batch_tag = ','.join(batch)
        batch_json = os.path.join(output_dir, f'{batch_name}.json')
        stdout_log = os.path.join(output_dir, f'{batch_name}.stdout.log')
        stderr_log = os.path.join(output_dir, f'{batch_name}.stderr.log')

        if args.resume and batch_name in completed_batches and os.path.exists(batch_json):
            print(f'[{batch_idx}/{total}] skip {batch_name} (already completed)')
            continue

        print(f'[{batch_idx}/{total}] run {batch_name}: {batch_tag}')
        cmd = [
            sys.executable,
            benchmark_py,
            '--mode', 'gset',
            '--graphs', batch_tag,
            '--solvers', args.solvers,
            '--time-limit', str(args.time_limit),
            '--output', batch_json,
        ]
        proc = subprocess.run(
            cmd,
            cwd=code_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )

        with open(stdout_log, 'w', encoding='utf-8') as f:
            f.write(proc.stdout)
        with open(stderr_log, 'w', encoding='utf-8') as f:
            f.write(proc.stderr)

        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f'{batch_name} failed with exit code {proc.returncode}')

        payload = _load_results(batch_json)
        batch_results = payload.get('results', [])
        all_results = [r for r in all_results if r.get('instance') not in set(batch)]
        all_results.extend(batch_results)
        completed_batches = [b for b in completed_batches if b != batch_name] + [batch_name]

        combined_payload = {
            'metadata': {
                'time_limit': args.time_limit,
                'batch_size': args.batch_size,
                'solvers': args.solvers.split(','),
                'output_dir': output_dir,
            },
            'completed_batches': completed_batches,
            'results': all_results,
            'summary': _summarize(all_results),
        }
        _save_json(combined_path, combined_payload)
        _save_json(manifest_path, {
            'batches': [
                {
                    'batch_name': f'batch_{i:02d}',
                    'graphs': batch,
                }
                for i, batch in enumerate(batches, start=1)
            ],
            'completed_batches': completed_batches,
        })

        print(f'[{batch_idx}/{total}] done {batch_name}')

    summary = _summarize(all_results)
    print('\n=== Combined summary ===')
    for solver, info in sorted(summary.items()):
        avg_gap = info['avg_gap_pct']
        avg_gap_str = f'{avg_gap:.2f}%' if avg_gap is not None else 'N/A'
        print(f'{solver}: ok={info["instances_ok"]}, avg_gap={avg_gap_str}')
    print(f'Combined report: {combined_path}')


if __name__ == '__main__':
    main()
