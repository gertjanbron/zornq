#\!/usr/bin/env python3
"""
nachtrun.py — ZornQ Nachtrun: B36 + B40 + B47
===============================================
Draait sequentieel:
  1. B36: Random Graph Testing (n=20, 22) met 3-regular en ER grafen
  2. B40: iTEBD-QAOA Transfer Matrix (Ly=4, p=2/3, chi=32/64)
  3. B47: RQAOA op grids (p=2, Lx=8..100, Ly=4)

Gebruik:
  python nachtrun.py              # volledige run
  python nachtrun.py --dry-run    # toon plan zonder uitvoering
  python nachtrun.py --skip b36   # sla B36 over
  python nachtrun.py --skip b40   # sla B40 over
  python nachtrun.py --skip b47   # sla B47 over

Resultaten: logs/<timestamp>_nachtrun/
"""

import subprocess
import sys
import os
import time
import datetime
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================
# Configuratie
# ============================================================

B36_RUNS = [
    # (n, samples, graph_type, edge_p, label)
    (20, 20, "3reg", 0.5, "3reg_n20"),
    (22, 15, "3reg", 0.5, "3reg_n22"),
    (20, 20, "er", 0.1, "er_n20_p01"),
    (20, 20, "er", 0.2, "er_n20_p02"),
    (22, 15, "er", 0.15, "er_n22_p015"),
]

B40_RUNS = [
    # (Ly, p_max, chi, pbc_y, progressive, label)
    (4, 2, 32, True, False, "Ly4_p2_chi32_cyl"),
    (4, 2, 64, True, False, "Ly4_p2_chi64_cyl"),
    (4, 3, 32, True, True, "Ly4_p3_chi32_cyl_prog"),
    (4, 3, 64, True, True, "Ly4_p3_chi64_cyl_prog"),
    (4, 2, 64, False, False, "Ly4_p2_chi64_strip"),
    (4, 3, 64, False, True, "Ly4_p3_chi64_strip_prog"),
]

B47_RUNS = [
    # (Lx, Ly, p, chi, label)
    (8, 4, 2, None, "8x4_p2"),
    (16, 4, 2, None, "16x4_p2"),
    (32, 4, 2, None, "32x4_p2"),
    (50, 4, 2, None, "50x4_p2"),
    (100, 4, 2, 32, "100x4_p2_chi32"),
    # Vergelijking p=1 vs p=2 op middelgrote grids
    (16, 4, 1, None, "16x4_p1_ref"),
    (32, 4, 1, None, "32x4_p1_ref"),
]


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logdir():
    ts = timestamp()
    logdir = os.path.join(SCRIPT_DIR, "logs", f"{ts}_nachtrun")
    os.makedirs(logdir, exist_ok=True)
    return logdir


def log(msg, logfile=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def run_cmd(cmd, logfile, label, dry_run=False):
    """Run a command, log stdout/stderr to file."""
    log(f"START: {label}", logfile)
    log(f"  CMD: {' '.join(cmd)}", logfile)
    if dry_run:
        log(f"  [DRY RUN] Skipped", logfile)
        return {"label": label, "status": "dry_run", "time_s": 0}
    
    t0 = time.time()
    outpath = logfile.replace(".log", f"_{label}.out")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 uur max per run
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0
        with open(outpath, "w") as f:
            f.write(f"=== {label} ===\n")
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"EXIT: {result.returncode}\n")
            f.write(f"TIME: {elapsed:.1f}s\n")
            f.write(f"\n=== STDOUT ===\n{result.stdout}\n")
            if result.stderr:
                f.write(f"\n=== STDERR ===\n{result.stderr}\n")
        
        status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
        log(f"DONE: {label} [{status}] in {elapsed:.1f}s", logfile)
        
        # Print last 5 lines of stdout as summary
        last_lines = result.stdout.strip().split("\n")[-5:]
        for line in last_lines:
            log(f"  > {line}", logfile)
        
        return {"label": label, "status": status, "time_s": elapsed, "output": outpath}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"TIMEOUT: {label} after {elapsed:.1f}s", logfile)
        return {"label": label, "status": "TIMEOUT", "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        log(f"ERROR: {label}: {e}", logfile)
        return {"label": label, "status": f"ERROR: {e}", "time_s": elapsed}


# ============================================================
# B36: Random Graph Testing
# ============================================================

def run_b36(logdir, logfile, dry_run=False):
    log("=" * 60, logfile)
    log("B36: Random Graph Testing (n=20-22, 3-reg + ER)", logfile)
    log("=" * 60, logfile)
    results = []
    for n, samples, gtype, edge_p, label in B36_RUNS:
        cmd = [sys.executable, "random_graph_test.py",
               "--n", str(n), "--samples", str(samples), "--type", gtype]
        if gtype == "er":
            cmd += ["--edge-p", str(edge_p)]
        r = run_cmd(cmd, logfile, f"b36_{label}", dry_run)
        results.append(r)
    return results


# ============================================================
# B40: iTEBD-QAOA Transfer Matrix
# ============================================================

def run_b40(logdir, logfile, dry_run=False):
    log("=" * 60, logfile)
    log("B40: iTEBD-QAOA Transfer Matrix (Ly=4, p=2/3, chi=32/64)", logfile)
    log("=" * 60, logfile)
    results = []
    for Ly, p_max, chi, pbc_y, progressive, label in B40_RUNS:
        cmd = [sys.executable, "transfer_matrix_qaoa.py",
               "--Ly", str(Ly), "--p", str(p_max), "--chi", str(chi)]
        if not pbc_y:
            cmd.append("--obc-y")
        if progressive:
            cmd.append("--progressive")
        r = run_cmd(cmd, logfile, f"b40_{label}", dry_run)
        results.append(r)
    return results


# ============================================================
# B47: RQAOA op grids
# ============================================================

def run_b47(logdir, logfile, dry_run=False):
    log("=" * 60, logfile)
    log("B47: RQAOA op grids (p=1/2, Lx=8..100, Ly=4)", logfile)
    log("=" * 60, logfile)
    results = []
    for Lx, Ly, p, chi, label in B47_RUNS:
        cmd = [sys.executable, "rqaoa.py",
               "--Lx", str(Lx), "--Ly", str(Ly), "--p", str(p)]
        if chi is not None:
            cmd += ["--chi", str(chi)]
        r = run_cmd(cmd, logfile, f"b47_{label}", dry_run)
        results.append(r)
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ZornQ Nachtrun: B36 + B40 + B47")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip items: b36 b40 b47")
    args = parser.parse_args()
    
    skip = set(s.lower() for s in args.skip)
    
    logdir = setup_logdir()
    logfile = os.path.join(logdir, "nachtrun.log")
    
    log("=" * 60, logfile)
    log("ZornQ Nachtrun", logfile)
    log(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f"Logdir: {logdir}", logfile)
    log(f"Dry run: {args.dry_run}", logfile)
    log(f"Skip: {skip or 'none'}", logfile)
    log("=" * 60, logfile)
    
    all_results = {}
    t_total = time.time()
    
    if "b36" not in skip:
        all_results["b36"] = run_b36(logdir, logfile, args.dry_run)
    else:
        log("SKIP: B36", logfile)
    
    if "b40" not in skip:
        all_results["b40"] = run_b40(logdir, logfile, args.dry_run)
    else:
        log("SKIP: B40", logfile)
    
    if "b47" not in skip:
        all_results["b47"] = run_b47(logdir, logfile, args.dry_run)
    else:
        log("SKIP: B47", logfile)
    
    total_time = time.time() - t_total
    
    # Summary
    log("\n" + "=" * 60, logfile)
    log("SAMENVATTING", logfile)
    log("=" * 60, logfile)
    
    total_ok = 0
    total_fail = 0
    for task, results in all_results.items():
        for r in results:
            status = r["status"]
            t = r["time_s"]
            if "OK" in status or status == "dry_run":
                total_ok += 1
            else:
                total_fail += 1
            log(f"  {r['label']:<30} {status:<15} {t:>8.1f}s", logfile)
    
    log(f"\nTotaal: {total_ok} OK, {total_fail} FAIL in {total_time:.0f}s ({total_time/60:.1f} min)", logfile)
    
    # Save summary JSON
    summary_path = os.path.join(logdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp(),
            "total_time_s": total_time,
            "results": all_results,
        }, f, indent=2, default=str)
    log(f"Summary: {summary_path}", logfile)
    log(f"Logs: {logdir}", logfile)


if __name__ == "__main__":
    main()
