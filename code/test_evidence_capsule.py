#!/usr/bin/env python3
"""test_evidence_capsule.py - Tests for B150 evidence capsules."""

import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence_capsule import (
    build_benchmark_capsule,
    graph_fingerprint,
    save_benchmark_capsule,
    verify_benchmark_capsule,
)


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')


print('=== B150 Evidence Capsule Tests ===\n')

edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
gid_a = graph_fingerprint(3, edges)
gid_b = graph_fingerprint(3, [(1, 0, 1.0), (2, 1, 1.0), (2, 0, 1.0)])
check('Graph fingerprint order invariant', gid_a == gid_b)

results = [{
    'instance': 'K3',
    'graph_id': gid_a,
    'n_nodes': 3,
    'n_edges': 3,
    'bks': 2,
    'solver': 'demo',
    'seed': 42,
    'time_limit_s': 1.0,
    'cut': 2.0,
    'time_s': 0.1,
    'gap': 0.0,
    'gap_pct': 0.0,
    'ratio': 1.0,
    'device': 'cpu',
    'error': None,
}]
metadata = {
    'mode': 'test',
    'solvers': ['demo'],
    'seed': 42,
    'time_limit': 1.0,
    'n_instances': 1,
}

capsule = build_benchmark_capsule(results, metadata=metadata)
check('Capsule has evidence level', capsule['evidence_level'] in (
    'development', 'reproducible', 'benchmark_certificate'))
check('Capsule has trusted claims', capsule['claims']['per_solver']['demo']['avg_gap_pct'] == 0.0)

tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'b150_test_capsule')
os.makedirs(tmpdir, exist_ok=True)
report_path = os.path.join(tmpdir, 'report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': metadata,
        'timestamp': '2026-04-15T12:00:00',
        'gpu_available': False,
        'results': results,
    }, f, indent=2)

capsule_path, receipt_path = save_benchmark_capsule(results, metadata, report_path)
check('Capsule file written', os.path.exists(capsule_path))
check('Receipt file written', os.path.exists(receipt_path))

verify = verify_benchmark_capsule(report_path, capsule_path=capsule_path)
check('Verification succeeds on untampered artefacts', verify['verified'])

with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)
report['results'][0]['cut'] = 1.0
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
verify_bad = verify_benchmark_capsule(report_path, capsule_path=capsule_path)
check('Verification fails after tamper', not verify_bad['verified'])

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)
