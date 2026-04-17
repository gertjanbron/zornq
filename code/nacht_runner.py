#!/usr/bin/env python3
"""
Nacht-runner: draait meerdere benchmarks achter elkaar met logging.

Gebruik:
    python nacht_runner.py                   # standaard 20x3 p=3 suite
    python nacht_runner.py --Lx 20 --Ly 3   # custom grid
    python nacht_runner.py --skip-interp     # alleen fourier run

Output:
    results/nacht_YYYYMMDD_HHMMSS/
        run_fourier.log          - volledige stdout/stderr
        run_interp.log           - vergelijkingsrun
        checkpoint_fourier.json  - checkpoint (hervatbaar)
        checkpoint_interp.json   - checkpoint
        gw_bound.log             - GW-bound rapport
        samenvatting.txt         - overzicht resultaten
"""

import argparse
import os
import subprocess
import sys
import time
import json
from datetime import datetime


def run_command(cmd, logfile, label, timeout_hours=4):
    """Draai een commando, schrijf output naar logfile, print progress."""
    print("\n  [%s] %s" % (time.strftime('%H:%M:%S'), label))
    print("  Commando: %s" % ' '.join(cmd))
    print("  Log: %s" % logfile)

    t0 = time.time()
    with open(logfile, 'w') as f:
        f.write("# %s\n" % label)
        f.write("# Commando: %s\n" % ' '.join(cmd))
        f.write("# Start: %s\n\n" % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.flush()

        try:
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            proc.wait(timeout=timeout_hours * 3600)
            elapsed = time.time() - t0
            f.write("\n# Klaar in %.1fs (exit code %d)\n" % (elapsed, proc.returncode))
            print("  Klaar in %.1fs (exit code %d)" % (elapsed, proc.returncode))
            return proc.returncode == 0, elapsed
        except subprocess.TimeoutExpired:
            proc.kill()
            f.write("\n# TIMEOUT na %.1f uur\n" % timeout_hours)
            print("  TIMEOUT na %.1f uur" % timeout_hours)
            return False, time.time() - t0
        except Exception as e:
            f.write("\n# FOUT: %s\n" % str(e))
            print("  FOUT: %s" % str(e))
            return False, time.time() - t0


def parse_ratio_from_log(logfile):
    """Zoek de beste ratio in een logbestand."""
    best = None
    try:
        with open(logfile) as f:
            for line in f:
                if 'Beste: p=' in line:
                    # "  Beste: p=3 ratio=0.766123"
                    parts = line.strip().split('ratio=')
                    if len(parts) == 2:
                        best = float(parts[1])
    except Exception:
        pass
    return best


def main():
    parser = argparse.ArgumentParser(description='ZornQ Nacht-Runner')
    parser.add_argument('--Lx', type=int, default=20)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--ngamma', type=int, default=12)
    parser.add_argument('--nbeta', type=int, default=12)
    parser.add_argument('--skip-interp', action='store_true',
                        help='Sla de interp-vergelijkingsrun over')
    parser.add_argument('--skip-gw', action='store_true',
                        help='Sla GW-bound berekening over')
    parser.add_argument('--timeout', type=float, default=4,
                        help='Timeout per run in uren (default: 4)')
    args = parser.parse_args()

    # Maak output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'results', 'nacht_%s' % timestamp)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    lc_script = os.path.join(script_dir, 'lightcone_qaoa.py')
    gw_script = os.path.join(script_dir, 'b60_gw_bound.py')

    sep = "=" * 60
    print(sep)
    print("  ZornQ Nacht-Runner")
    print(sep)
    print("  Grid: %dx%d, p=%d" % (args.Lx, args.Ly, args.p))
    print("  Output: %s" % outdir)
    print("  Start: %s" % time.strftime('%Y-%m-%d %H:%M:%S'))
    print(sep)

    results = {}
    total_t0 = time.time()

    # --- Run 1: Fourier warm-start (hoofdrun) ---
    ckpt_fourier = os.path.join(outdir, 'checkpoint_fourier.json')
    ok, elapsed = run_command(
        [sys.executable, lc_script,
         '--Lx', str(args.Lx), '--Ly', str(args.Ly),
         '--p', str(args.p), '--progressive',
         '--warmstart-method', 'fourier',
         '--ngamma', str(args.ngamma), '--nbeta', str(args.nbeta),
         '--gpu', '--fp32',
         '--checkpoint', ckpt_fourier],
        os.path.join(outdir, 'run_fourier.log'),
        'Fourier warm-start (p=1->%d)' % args.p,
        timeout_hours=args.timeout
    )
    ratio_f = parse_ratio_from_log(os.path.join(outdir, 'run_fourier.log'))
    results['fourier'] = {'ok': ok, 'time': elapsed, 'ratio': ratio_f}

    # --- Run 2: Interp warm-start (vergelijking) ---
    if not args.skip_interp:
        ckpt_interp = os.path.join(outdir, 'checkpoint_interp.json')
        ok, elapsed = run_command(
            [sys.executable, lc_script,
             '--Lx', str(args.Lx), '--Ly', str(args.Ly),
             '--p', str(args.p), '--progressive',
             '--warmstart-method', 'interp',
             '--ngamma', str(args.ngamma), '--nbeta', str(args.nbeta),
             '--gpu', '--fp32',
             '--checkpoint', ckpt_interp],
            os.path.join(outdir, 'run_interp.log'),
            'Interp warm-start (p=1->%d)' % args.p,
            timeout_hours=args.timeout
        )
        ratio_i = parse_ratio_from_log(os.path.join(outdir, 'run_interp.log'))
        results['interp'] = {'ok': ok, 'time': elapsed, 'ratio': ratio_i}

    # --- Run 3: GW-bound ---
    if not args.skip_gw and os.path.exists(gw_script):
        gw_args = [sys.executable, gw_script,
                    '--Lx', str(args.Lx), '--Ly', str(args.Ly)]
        ok, elapsed = run_command(
            gw_args,
            os.path.join(outdir, 'gw_bound.log'),
            'GW-Bound rapport (%dx%d)' % (args.Lx, args.Ly),
            timeout_hours=0.5
        )
        results['gw_bound'] = {'ok': ok, 'time': elapsed}

    # --- Samenvatting ---
    total_elapsed = time.time() - total_t0
    summary_path = os.path.join(outdir, 'samenvatting.txt')
    with open(summary_path, 'w') as f:
        f.write("ZornQ Nacht-Runner Samenvatting\n")
        f.write("=" * 50 + "\n")
        f.write("Grid: %dx%d, p=%d\n" % (args.Lx, args.Ly, args.p))
        f.write("Datum: %s\n" % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write("Totale tijd: %.1f minuten\n\n" % (total_elapsed / 60))

        for name, r in results.items():
            status = "OK" if r['ok'] else "FOUT"
            line = "%s: %s (%.1f min)" % (name, status, r['time'] / 60)
            if r.get('ratio') is not None:
                line += " ratio=%.6f" % r['ratio']
            f.write(line + "\n")

        if 'fourier' in results and 'interp' in results:
            rf = results['fourier'].get('ratio')
            ri = results['interp'].get('ratio')
            if rf is not None and ri is not None:
                f.write("\nFourier vs Interp: %.6f vs %.6f " % (rf, ri))
                if abs(rf - ri) < 1e-6:
                    f.write("(gelijk)\n")
                elif rf > ri:
                    f.write("(fourier +%.6f)\n" % (rf - ri))
                else:
                    f.write("(interp +%.6f)\n" % (ri - rf))

    print("\n" + sep)
    print("  NACHT-RUNNER KLAAR")
    print(sep)
    print("  Totale tijd: %.1f minuten" % (total_elapsed / 60))
    for name, r in results.items():
        status = "OK" if r['ok'] else "FOUT"
        line = "  %s: %s (%.1f min)" % (name, status, r['time'] / 60)
        if r.get('ratio') is not None:
            line += " ratio=%.6f" % r['ratio']
        print(line)
    print("  Samenvatting: %s" % summary_path)
    print(sep)


if __name__ == '__main__':
    main()
