#!/usr/bin/env python3
"""
evidence_capsule.py - B150 Evidence Capsules & Receipts

Small reproducibility layer for benchmark artefacts:
  - graph fingerprints
  - environment snapshot
  - claims summary over benchmark rows
  - sidecar capsule + receipt hashes
  - verification that source report and derived claims still match
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from audit_trail import get_environment

__version__ = "1.0.0"


def _stable_json_dumps(obj) -> str:
    """Canonical JSON string for hashing and equality checks."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        default=str,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _round_float(value, digits=12):
    """Stable float rounding for reproducible hashing."""
    return round(float(value), digits)


def graph_fingerprint(n_nodes: int, edges) -> str:
    """Deterministic graph fingerprint from normalized weighted edges."""
    h = hashlib.sha256()
    h.update(f"n={int(n_nodes)}\n".encode('utf-8'))
    norm_edges = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u == v:
            continue
        w = float(e[2]) if len(e) > 2 else 1.0
        if u > v:
            u, v = v, u
        norm_edges.append((u, v, _round_float(w)))
    norm_edges.sort()
    h.update(f"m={len(norm_edges)}\n".encode('utf-8'))
    for u, v, w in norm_edges:
        h.update(f"{u}:{v}:{w:.12g}\n".encode('utf-8'))
    return h.hexdigest()[:16]


def classify_evidence_level(results, environment, metadata):
    """
    Distinguish development evidence from stronger benchmark artefacts.

    Levels:
      - development: dirty tree or errors present
      - reproducible: clean rows, source metadata present
      - benchmark_certificate: reproducible + every row carries BKS context
    """
    git_dirty = bool(((environment or {}).get('git') or {}).get('dirty'))
    has_errors = any(r.get('error') for r in results)
    has_seed = metadata.get('seed') is not None
    has_time = metadata.get('time_limit') is not None
    has_bks = results and all(r.get('bks') is not None for r in results)

    if git_dirty or has_errors or not has_seed or not has_time:
        return 'development'
    if has_bks:
        return 'benchmark_certificate'
    return 'reproducible'


def summarize_benchmark_results(results):
    """Create a trusted claims summary over benchmark rows."""
    summary = {
        'n_rows': len(results),
        'n_errors': sum(1 for r in results if r.get('error')),
        'per_solver': {},
        'per_instance': {},
    }

    solver_groups = {}
    instance_groups = {}
    for row in results:
        solver_groups.setdefault(row.get('solver', 'unknown'), []).append(row)
        instance_groups.setdefault(row.get('instance', 'unknown'), []).append(row)

    for solver, rows in sorted(solver_groups.items()):
        valid_gap = [float(r['gap_pct']) for r in rows if r.get('gap_pct') is not None]
        valid_ratio = [float(r['ratio']) for r in rows if r.get('ratio') is not None]
        summary['per_solver'][solver] = {
            'n_rows': len(rows),
            'n_errors': sum(1 for r in rows if r.get('error')),
            'avg_gap_pct': sum(valid_gap) / len(valid_gap) if valid_gap else None,
            'avg_ratio': sum(valid_ratio) / len(valid_ratio) if valid_ratio else None,
            'within_1pct': sum(1 for r in rows
                               if r.get('gap_pct') is not None and float(r['gap_pct']) <= 1.0),
            'best_cut': max(float(r.get('cut', 0.0)) for r in rows) if rows else None,
        }

    for instance, rows in sorted(instance_groups.items()):
        best_row = max(rows, key=lambda r: float(r.get('cut', 0.0)))
        summary['per_instance'][instance] = {
            'n_rows': len(rows),
            'best_solver': best_row.get('solver'),
            'best_cut': float(best_row.get('cut', 0.0)),
            'bks': best_row.get('bks'),
            'graph_id': best_row.get('graph_id'),
        }

    return summary


def build_benchmark_capsule(results, metadata=None, report_path=None):
    """Build a benchmark capsule dict from rows and metadata."""
    metadata = dict(metadata or {})
    environment = get_environment()
    claims = summarize_benchmark_results(results)
    evidence_level = classify_evidence_level(results, environment, metadata)

    source_report = {}
    if report_path:
        source_report = {
            'path': os.path.abspath(report_path),
            'sha256': _sha256_file(report_path),
        }

    capsule = {
        'kind': 'zornq_benchmark_capsule',
        'capsule_version': __version__,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'benchmark': metadata,
        'environment': environment,
        'evidence_level': evidence_level,
        'claims': claims,
        'trusted_checks': {
            'source_report_present': bool(report_path),
            'results_have_graph_ids': all(bool(r.get('graph_id')) for r in results),
            'results_have_seed': metadata.get('seed') is not None,
            'results_have_time_limit': metadata.get('time_limit') is not None,
        },
        'source_report': source_report,
    }
    payload_hash = _sha256_text(_stable_json_dumps(capsule))
    capsule['receipt'] = {
        'payload_sha256': payload_hash,
        'claims_sha256': _sha256_text(_stable_json_dumps(claims)),
        'source_report_sha256': source_report.get('sha256'),
    }
    return capsule


def build_receipt(capsule, capsule_path=None):
    """Build a minimal receipt document from a full capsule."""
    return {
        'kind': 'zornq_benchmark_receipt',
        'capsule_version': capsule.get('capsule_version', __version__),
        'created_at': capsule.get('created_at'),
        'capsule_path': os.path.abspath(capsule_path) if capsule_path else None,
        'evidence_level': capsule.get('evidence_level'),
        'payload_sha256': ((capsule.get('receipt') or {}).get('payload_sha256')),
        'claims_sha256': ((capsule.get('receipt') or {}).get('claims_sha256')),
        'source_report_sha256': ((capsule.get('receipt') or {}).get('source_report_sha256')),
    }


def save_benchmark_capsule(results, metadata, report_path, capsule_path=None, receipt_path=None):
    """Write sidecar capsule + receipt next to a JSON benchmark report."""
    report_path = os.path.abspath(report_path)
    root, ext = os.path.splitext(report_path)
    if capsule_path is None:
        capsule_path = root + '.capsule.json'
    if receipt_path is None:
        receipt_path = root + '.receipt.json'

    capsule = build_benchmark_capsule(results, metadata=metadata, report_path=report_path)
    os.makedirs(os.path.dirname(capsule_path) or '.', exist_ok=True)
    with open(capsule_path, 'w', encoding='utf-8') as f:
        json.dump(capsule, f, indent=2, ensure_ascii=False, default=str)

    receipt = build_receipt(capsule, capsule_path=capsule_path)
    with open(receipt_path, 'w', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False, default=str)

    return capsule_path, receipt_path


def _compare_claims(left, right):
    return _stable_json_dumps(left) == _stable_json_dumps(right)


def verify_benchmark_capsule(report_path, capsule_path=None):
    """Recompute hashes and derived claims; return a verification report."""
    report_path = os.path.abspath(report_path)
    root, _ext = os.path.splitext(report_path)
    if capsule_path is None:
        capsule_path = root + '.capsule.json'
    capsule_path = os.path.abspath(capsule_path)

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    with open(capsule_path, 'r', encoding='utf-8') as f:
        capsule = json.load(f)

    metadata = report.get('metadata', {})
    results = report.get('results', [])
    recomputed_claims = summarize_benchmark_results(results)
    recomputed_report_hash = _sha256_file(report_path)

    capsule_no_receipt = dict(capsule)
    stored_receipt = capsule_no_receipt.pop('receipt', {})
    recomputed_payload_hash = _sha256_text(_stable_json_dumps(capsule_no_receipt))

    checks = {
        'source_report_hash_ok': (
            (capsule.get('source_report') or {}).get('sha256') == recomputed_report_hash
        ),
        'claims_ok': _compare_claims(capsule.get('claims', {}), recomputed_claims),
        'payload_hash_ok': stored_receipt.get('payload_sha256') == recomputed_payload_hash,
        'claims_hash_ok': (
            stored_receipt.get('claims_sha256') ==
            _sha256_text(_stable_json_dumps(recomputed_claims))
        ),
        'seed_present': metadata.get('seed') is not None,
        'time_limit_present': metadata.get('time_limit') is not None,
        'graph_ids_present': all(bool(r.get('graph_id')) for r in results),
    }

    return {
        'report_path': report_path,
        'capsule_path': capsule_path,
        'verified': all(checks.values()),
        'checks': checks,
        'evidence_level': capsule.get('evidence_level'),
    }


def _print_verification(report):
    status = 'OK' if report['verified'] else 'FAILED'
    print(f'=== B150 Evidence Capsule Verify: {status} ===')
    print(f"report:  {report['report_path']}")
    print(f"capsule: {report['capsule_path']}")
    for name, ok in report['checks'].items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def main():
    parser = argparse.ArgumentParser(description='B150 evidence capsule tools')
    parser.add_argument('--verify', metavar='REPORT',
                        help='Verify sidecar capsule for a benchmark JSON report')
    parser.add_argument('--capsule', metavar='FILE',
                        help='Explicit capsule path for verify mode')
    args = parser.parse_args()

    if args.verify:
        report = verify_benchmark_capsule(args.verify, capsule_path=args.capsule)
        _print_verification(report)
        raise SystemExit(0 if report['verified'] else 1)

    parser.print_help()


if __name__ == '__main__':
    main()
