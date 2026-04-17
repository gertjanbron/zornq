#!/usr/bin/env python3
"""
zornq_bench.py - B53: Experiment- en Regressieharnas.

Draait een vaste set benchmark-instanties en schrijft gestructureerde
resultaten weg als JSON + CSV. Maakt regressie-detectie en prestatie-
tracking automatisch.

Gebruik:
    python zornq_bench.py                       # small suite (~30s)
    python zornq_bench.py --suite medium         # medium (~5 min)
    python zornq_bench.py --suite full           # full (~30 min, GPU)
    python zornq_bench.py --compare vorige.json  # regressie-check
    python zornq_bench.py --list                 # toon suite-inhoud

Suites:
    small:  Snelle sanity check, 4 instanties, ~30s CPU
    medium: Productie-check, small + grotere grids, ~5 min GPU
    full:   Publicatie-dataset, medium + 100x4 + RQAOA, ~30 min GPU

Output:
    results/bench_{date}_{suite}.json   - volledige resultaten
    results/bench_{date}_{suite}.csv    - tabel voor analyse
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Suite-definities
# =====================================================================

def get_suites():
    """Definieer benchmark-suites.

    Elke instantie is een dict met:
        name:    menselijk label
        Lx, Ly:  grid dimensies
        p:       QAOA diepte
        method:  'progressive' | 'single'
        warmstart: 'fourier' | 'interp' | None
        gpu:     True/False
        fp32:    True/False
        ngamma:  grid-zoekpunten gamma
        nbeta:   grid-zoekpunten beta
        graph:   (optioneel) naam uit gset_loader voor niet-grid grafen
        bks:     (optioneel) bekende MaxCut waarde
    """
    # --- Small: snelle sanity check (~30s CPU) ---
    small = [
        {
            'name': '4x3_p1',
            'Lx': 4, 'Ly': 3, 'p': 1,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': '6x2_p2',
            'Lx': 6, 'Ly': 2, 'p': 2,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': '8x3_p1',
            'Lx': 8, 'Ly': 3, 'p': 1,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': '6x3_p2_fourier',
            'Lx': 6, 'Ly': 3, 'p': 2,
            'method': 'progressive', 'warmstart': 'fourier',
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
    ]

    # --- Medium: productie-check (~5 min GPU) ---
    medium_extra = [
        {
            'name': '12x3_p2_interp',
            'Lx': 12, 'Ly': 3, 'p': 2,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': '12x3_p2_fourier',
            'Lx': 12, 'Ly': 3, 'p': 2,
            'method': 'progressive', 'warmstart': 'fourier',
            'gpu': True, 'fp32': True,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': '20x3_p1',
            'Lx': 20, 'Ly': 3, 'p': 1,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 12, 'nbeta': 12,
        },
        {
            'name': '8x4_p1',
            'Lx': 8, 'Ly': 4, 'p': 1,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 10, 'nbeta': 10,
        },
    ]

    # --- Full: publicatie-dataset (~30 min GPU) ---
    full_extra = [
        {
            'name': '20x3_p2_interp',
            'Lx': 20, 'Ly': 3, 'p': 2,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 12, 'nbeta': 12,
        },
        {
            'name': '20x3_p2_fourier',
            'Lx': 20, 'Ly': 3, 'p': 2,
            'method': 'progressive', 'warmstart': 'fourier',
            'gpu': True, 'fp32': True,
            'ngamma': 12, 'nbeta': 12,
        },
        {
            'name': '20x3_p3_interp',
            'Lx': 20, 'Ly': 3, 'p': 3,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 12, 'nbeta': 12,
        },
        {
            'name': '100x3_p1',
            'Lx': 100, 'Ly': 3, 'p': 1,
            'method': 'progressive', 'warmstart': 'interp',
            'gpu': True, 'fp32': True,
            'ngamma': 12, 'nbeta': 12,
        },
    ]

    # --- Benchmark-grafen uit gset_loader (B61) ---
    # Kleine klassieke grafen met bekende BKS voor exacte verificatie
    bench_graphs_small = [
        {
            'name': 'petersen_p1',
            'graph': 'petersen', 'bks': 12,
            'Lx': 10, 'Ly': 1, 'p': 1,  # Lx=n_nodes placeholder
            'method': 'rqaoa', 'warmstart': None,
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': 'reg3_14_p1',
            'graph': 'reg3_14', 'seed': 42,
            'Lx': 14, 'Ly': 1, 'p': 1,
            'method': 'rqaoa', 'warmstart': None,
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
    ]

    bench_graphs_medium = [
        {
            'name': 'reg3_20_p1',
            'graph': 'reg3_20', 'seed': 42,
            'Lx': 20, 'Ly': 1, 'p': 1,
            'method': 'rqaoa', 'warmstart': None,
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
        {
            'name': 'er_16_p1',
            'graph': 'er_16', 'seed': 42,
            'Lx': 16, 'Ly': 1, 'p': 1,
            'method': 'rqaoa', 'warmstart': None,
            'gpu': False, 'fp32': False,
            'ngamma': 10, 'nbeta': 10,
        },
    ]

    return {
        'small': small + bench_graphs_small,
        'medium': small + medium_extra + bench_graphs_small + bench_graphs_medium,
        'full': small + medium_extra + full_extra + bench_graphs_small + bench_graphs_medium,
    }


# =====================================================================
# Runner: draait benchmark-instanties
# =====================================================================

def run_rqaoa_instance(inst, script_path):
    """Draai een RQAOA benchmark op een gset_loader graaf."""
    name = inst['name']
    graph_name = inst['graph']
    seed = inst.get('seed', None)
    bks = inst.get('bks', None)

    result = {
        'name': name,
        'graph': graph_name,
        'Lx': inst['Lx'], 'Ly': inst['Ly'],
        'p': inst['p'],
        'n_qubits': inst['Lx'],  # n_nodes
        'method': 'rqaoa',
        'warmstart': None,
        'gpu': False, 'fp32': False,
        'status': 'error',
        'ratio': None,
        'bks': bks,
        'runtime_sec': None,
        'n_evals': None,
        'ratios_per_p': {},
    }

    t0 = time.time()
    try:
        from gset_loader import load_graph
        from rqaoa import RQAOA, brute_force_maxcut

        g, loaded_bks, info = load_graph(graph_name, seed=seed)
        if bks is None and loaded_bks is not None:
            bks = loaded_bks
            result['bks'] = bks

        result['n_qubits'] = g.n_nodes

        # Brute force als klein genoeg
        exact_cut = None
        if g.n_nodes <= 22:
            exact_cut, _ = brute_force_maxcut(g)

        # RQAOA: solve_full voor kleine grafen (auto-optimaliseert p=1)
        rqaoa = RQAOA(g, p=inst['p'], verbose=False)
        cut_val, partition, rqaoa_info = rqaoa.solve_full()
        elapsed = time.time() - t0
        result['runtime_sec'] = round(elapsed, 2)

        total_weight = g.total_weight()
        ratio = cut_val / total_weight if total_weight > 0 else 0
        result['ratio'] = round(ratio, 6)
        result['cut'] = cut_val
        result['status'] = 'ok'

        if exact_cut is not None:
            result['exact_cut'] = exact_cut
            result['exact_ratio'] = round(exact_cut / total_weight, 6)
            result['gap_to_exact'] = round((exact_cut - cut_val) / total_weight, 6)

        if bks is not None:
            result['gap_to_bks'] = round((bks - cut_val) / total_weight, 6)

    except Exception as e:
        elapsed = time.time() - t0
        result['runtime_sec'] = round(elapsed, 2)
        result['status'] = 'exception'
        result['error'] = str(e)

    return result


def run_instance(inst, script_path):
    """Draai een enkele benchmark-instantie. Returns result dict."""
    name = inst['name']

    # RQAOA op benchmark-grafen (niet-grid) — apart pad
    if inst.get('method') == 'rqaoa' and inst.get('graph'):
        return run_rqaoa_instance(inst, script_path)

    result = {
        'name': name,
        'Lx': inst['Lx'],
        'Ly': inst['Ly'],
        'p': inst['p'],
        'n_qubits': inst['Lx'] * inst['Ly'],
        'method': inst['method'],
        'warmstart': inst.get('warmstart', 'interp'),
        'gpu': inst.get('gpu', False),
        'fp32': inst.get('fp32', False),
        'status': 'error',
        'ratio': None,
        'runtime_sec': None,
        'n_evals': None,
        'ratios_per_p': {},
    }

    # Bouw CLI commando
    cmd = [
        sys.executable, script_path,
        '--Lx', str(inst['Lx']),
        '--Ly', str(inst['Ly']),
        '--p', str(inst['p']),
        '--ngamma', str(inst.get('ngamma', 10)),
        '--nbeta', str(inst.get('nbeta', 10)),
        '--progressive',
        '--warmstart-method', inst.get('warmstart', 'interp'),
        '--quiet',
    ]
    if inst.get('gpu'):
        cmd.append('--gpu')
    if inst.get('fp32'):
        cmd.append('--fp32')

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=3600,  # 1 uur max per instantie
            cwd=os.path.dirname(script_path)
        )
        elapsed = time.time() - t0
        result['runtime_sec'] = round(elapsed, 2)

        # Parse output
        output = proc.stdout + proc.stderr
        result['raw_output'] = output

        # Zoek "Beste: p=X ratio=Y"
        for line in output.split('\n'):
            line = line.strip()
            if 'Beste: p=' in line and 'ratio=' in line:
                parts = line.split('ratio=')
                if len(parts) == 2:
                    result['ratio'] = float(parts[1].strip())
            # Zoek per-p resultaten uit de tabel
            # Format: "  1    0.678361  g=[...] b=[...]"
            if line and line[0].isdigit() and '0.' in line:
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        pi = int(tokens[0])
                        ri = float(tokens[1])
                        result['ratios_per_p'][str(pi)] = ri
                    except (ValueError, IndexError):
                        pass
            # Zoek "Totaal: Xs, Y evaluaties"
            if 'Totaal:' in line and 'evaluaties' in line:
                try:
                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'evaluaties' in part:
                            result['n_evals'] = int(part.split()[0])
                except (ValueError, IndexError):
                    pass

        if proc.returncode == 0 and result['ratio'] is not None:
            result['status'] = 'ok'
        elif proc.returncode != 0:
            result['status'] = 'error (exit %d)' % proc.returncode
            result['error'] = output[-500:] if len(output) > 500 else output

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result['runtime_sec'] = round(elapsed, 2)
        result['status'] = 'timeout'
    except Exception as e:
        elapsed = time.time() - t0
        result['runtime_sec'] = round(elapsed, 2)
        result['status'] = 'exception'
        result['error'] = str(e)

    return result


# =====================================================================
# Regressie-check: vergelijk twee resultaten-bestanden
# =====================================================================

def check_regression(current, previous, ratio_threshold=0.001, time_threshold=0.20):
    """Vergelijk twee resultaten en detecteer regressies.

    Regressie als:
      - ratio daalt > ratio_threshold (default 0.1%)
      - runtime stijgt > time_threshold (default 20%)

    Returns: list van regressie-beschrijvingen (leeg = alles ok)
    """
    regressions = []
    prev_by_name = {r['name']: r for r in previous}

    for cur in current:
        name = cur['name']
        if name not in prev_by_name:
            continue
        prev = prev_by_name[name]

        # Ratio regressie
        if cur.get('ratio') is not None and prev.get('ratio') is not None:
            delta = prev['ratio'] - cur['ratio']
            if delta > ratio_threshold:
                regressions.append(
                    "RATIO REGRESSIE: %s  %.6f -> %.6f (-%0.4f)" % (
                        name, prev['ratio'], cur['ratio'], delta))

        # Runtime regressie
        if cur.get('runtime_sec') and prev.get('runtime_sec'):
            speedup = cur['runtime_sec'] / max(prev['runtime_sec'], 0.01)
            if speedup > (1 + time_threshold):
                regressions.append(
                    "RUNTIME REGRESSIE: %s  %.1fs -> %.1fs (+%.0f%%)" % (
                        name, prev['runtime_sec'], cur['runtime_sec'],
                        (speedup - 1) * 100))

    return regressions


# =====================================================================
# Git info helper
# =====================================================================

def get_git_info():
    """Haal huidige git commit hash en status."""
    info = {'commit': 'unknown', 'dirty': False, 'branch': 'unknown'}
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except Exception:
        pass
    return info


# =====================================================================
# Output schrijvers
# =====================================================================

def write_json(results, metadata, filepath):
    """Schrijf resultaten als JSON."""
    # Verwijder raw_output (te groot voor JSON)
    clean_results = []
    for r in results:
        rc = {k: v for k, v in r.items() if k != 'raw_output'}
        clean_results.append(rc)

    data = {
        'metadata': metadata,
        'results': clean_results,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def write_csv(results, filepath):
    """Schrijf resultaten als CSV voor makkelijke analyse."""
    fieldnames = [
        'name', 'Lx', 'Ly', 'p', 'n_qubits',
        'method', 'warmstart', 'gpu', 'fp32',
        'status', 'ratio', 'runtime_sec', 'n_evals',
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def print_results_table(results):
    """Print resultaten als een mooie tabel."""
    hdr = "  %-22s %-6s %-3s %-10s %-10s %-8s %-8s"
    print(hdr % ('Name', 'Qubits', 'p', 'Ratio', 'Time(s)', 'Evals', 'Status'))
    print("  " + "-" * 72)
    for r in results:
        ratio_str = '%.6f' % r['ratio'] if r['ratio'] is not None else '-'
        time_str = '%.1f' % r['runtime_sec'] if r['runtime_sec'] is not None else '-'
        evals_str = str(r['n_evals']) if r['n_evals'] is not None else '-'
        row = "  %-22s %-6d %-3d %-10s %-10s %-8s %-8s"
        print(row % (
            r['name'], r['n_qubits'], r['p'],
            ratio_str, time_str, evals_str, r['status']))


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B53: ZornQ Experiment- en Regressieharnas')
    parser.add_argument('--suite', choices=['small', 'medium', 'full'],
                        default='small',
                        help='Benchmark suite (default: small)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Vergelijk met vorig resultaat-JSON voor regressie-check')
    parser.add_argument('--list', action='store_true',
                        help='Toon suite-inhoud zonder te draaien')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: results/)')
    parser.add_argument('--audit', action='store_true',
                        help='Genereer audit trail per instantie (B56)')
    args = parser.parse_args()

    suites = get_suites()

    # --list: toon suites
    if args.list:
        for suite_name, instances in suites.items():
            print("\n  Suite '%s' (%d instanties):" % (suite_name, len(instances)))
            for inst in instances:
                ws = inst.get('warmstart', 'interp')
                gpu = 'GPU' if inst.get('gpu') else 'CPU'
                fp = 'fp32' if inst.get('fp32') else 'fp64'
                print("    %-25s %dx%d p=%d  %s  %s %s  grid=%dx%d" % (
                    inst['name'], inst['Lx'], inst['Ly'], inst['p'],
                    ws, gpu, fp,
                    inst.get('ngamma', 10), inst.get('nbeta', 10)))
        return

    suite = suites[args.suite]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lc_script = os.path.join(script_dir, 'lightcone_qaoa.py')

    # Output directory
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(script_dir, '..', 'results')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    git_info = get_git_info()

    sep = "=" * 60
    print(sep)
    print("  B53: ZornQ Benchmark Harnas")
    print(sep)
    print("  Suite:   %s (%d instanties)" % (args.suite, len(suite)))
    print("  Git:     %s%s" % (
        git_info['commit'], ' (dirty)' if git_info['dirty'] else ''))
    print("  Start:   %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(sep)

    # Audit trail setup (B56)
    audit_dir = None
    if args.audit:
        try:
            from audit_trail import AuditTrail
            audit_dir = os.path.join(outdir, 'audit_%s_%s' % (timestamp, args.suite))
            os.makedirs(audit_dir, exist_ok=True)
            print("  Audit:   %s" % audit_dir)
        except ImportError:
            print("  WAARSCHUWING: audit_trail.py niet gevonden, --audit uitgeschakeld")
            audit_dir = None

    # Draai alle instanties
    results = []
    total_t0 = time.time()

    for i, inst in enumerate(suite):
        label = "[%d/%d] %s" % (i + 1, len(suite), inst['name'])
        print("\n  %s ..." % label, end='', flush=True)
        result = run_instance(inst, lc_script)
        results.append(result)

        if result['status'] == 'ok':
            print(" ratio=%.6f  %.1fs" % (
                result['ratio'], result['runtime_sec']))
        else:
            print(" %s  %.1fs" % (result['status'], result['runtime_sec'] or 0))

        # Audit trail per instantie (B56)
        if audit_dir and result['status'] == 'ok':
            try:
                graph_desc = "grid_%dx%d" % (inst['Lx'], inst['Ly'])
                audit = AuditTrail(graph_desc, p=inst['p'], method='lightcone',
                                   extra_config=inst)
                audit.log_phase("benchmark_run",
                    ratio=result.get('ratio'),
                    time_s=result.get('runtime_sec'),
                    n_evals=result.get('n_evals'))
                audit.set_result(
                    ratio=result['ratio'],
                    n_edges=result.get('n_edges', inst['Lx'] * (inst['Ly'] - 1) +
                                      (inst['Lx'] - 1) * inst['Ly']))
                audit_path = os.path.join(audit_dir, '%s.json' % inst['name'])
                audit.save(audit_path)
            except Exception as e:
                print("    (audit fout: %s)" % e)

    total_elapsed = time.time() - total_t0

    # Resultaten tabel
    print("\n" + sep)
    print("  RESULTATEN")
    print(sep)
    print_results_table(results)
    print("  " + "-" * 72)
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    print("  %d/%d geslaagd, totale tijd: %.1fs" % (
        n_ok, len(results), total_elapsed))

    # Metadata
    metadata = {
        'suite': args.suite,
        'timestamp': timestamp,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'git': git_info,
        'total_runtime_sec': round(total_elapsed, 2),
        'n_instances': len(suite),
        'n_ok': n_ok,
        'python': sys.version.split()[0],
        'platform': sys.platform,
    }

    # Schrijf output
    base = 'bench_%s_%s' % (timestamp, args.suite)
    json_path = os.path.join(outdir, base + '.json')
    csv_path = os.path.join(outdir, base + '.csv')

    write_json(results, metadata, json_path)
    write_csv(results, csv_path)

    # Audit trail HTML summary voor hele suite (B56)
    audit_html_path = None
    if audit_dir:
        try:
            from audit_trail import AuditTrail
            suite_audit = AuditTrail("suite_%s" % args.suite, p=0,
                                      method='benchmark',
                                      extra_config={'suite': args.suite,
                                                    'n_instances': len(suite)})
            for r in results:
                if r['status'] == 'ok':
                    suite_audit.log_phase(r['name'],
                        ratio=r.get('ratio'),
                        time_s=r.get('runtime_sec'),
                        n_evals=r.get('n_evals'))
            # Gemiddelde ratio als "resultaat"
            ok_ratios = [r['ratio'] for r in results if r.get('ratio')]
            if ok_ratios:
                avg_ratio = sum(ok_ratios) / len(ok_ratios)
                suite_audit.set_result(ratio=avg_ratio, n_ok=n_ok,
                                       n_total=len(results))
            suite_audit.set_diagnostics(total_time_s=total_elapsed)
            suite_json = os.path.join(audit_dir, '_suite_summary.json')
            audit_html_path = os.path.join(audit_dir, '_suite_summary.html')
            suite_audit.save(suite_json)
            suite_audit.save_html(audit_html_path)
        except Exception as e:
            print("  (suite audit fout: %s)" % e)

    print("\n  Output:")
    print("    JSON: %s" % json_path)
    print("    CSV:  %s" % csv_path)
    if audit_dir:
        print("    Audit: %s/ (%d artefacten)" % (
            audit_dir, len([f for f in os.listdir(audit_dir) if f.endswith('.json')])))

    # Regressie-check
    if args.compare:
        print("\n" + sep)
        print("  REGRESSIE-CHECK")
        print(sep)
        try:
            with open(args.compare) as f:
                prev_data = json.load(f)
            prev_results = prev_data.get('results', [])
            prev_meta = prev_data.get('metadata', {})
            print("  Vergelijking met: %s" % args.compare)
            print("  Vorige run: %s (git %s)" % (
                prev_meta.get('datetime', '?'),
                prev_meta.get('git', {}).get('commit', '?')))

            regressions = check_regression(results, prev_results)
            if not regressions:
                print("  Geen regressies gevonden\\!")
            else:
                print("  %d REGRESSIES GEVONDEN:" % len(regressions))
                for reg in regressions:
                    print("    %s" % reg)

            # Verbeteringen
            prev_by_name = {r['name']: r for r in prev_results}
            improvements = []
            for cur in results:
                name = cur['name']
                if name in prev_by_name:
                    prev = prev_by_name[name]
                    if cur.get('ratio') and prev.get('ratio'):
                        delta = cur['ratio'] - prev['ratio']
                        if delta > 0.001:
                            improvements.append(
                                "  VERBETERING: %s  %.6f -> %.6f (+%.4f)" % (
                                    name, prev['ratio'], cur['ratio'], delta))
                    if cur.get('runtime_sec') and prev.get('runtime_sec'):
                        speedup = prev['runtime_sec'] / max(cur['runtime_sec'], 0.01)
                        if speedup > 1.2:
                            improvements.append(
                                "  SPEEDUP: %s  %.1fs -> %.1fs (%.1fx sneller)" % (
                                    name, prev['runtime_sec'], cur['runtime_sec'], speedup))
            if improvements:
                print("\n  Verbeteringen:")
                for imp in improvements:
                    print("    %s" % imp)

        except Exception as e:
            print("  Fout bij laden vergelijkingsbestand: %s" % e)

    print("\n" + sep)
    if n_ok == len(results):
        print("  ALLE BENCHMARKS GESLAAGD")
    else:
        print("  %d/%d GEFAALD" % (len(results) - n_ok, len(results)))
    print(sep)

    # Exit code: 0 als alles ok
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
