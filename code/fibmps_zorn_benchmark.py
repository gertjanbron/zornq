#!/usr/bin/env python3
"""
FibMPS v129 vs v129+Zorn — Lokale benchmark op grotere instances
=================================================================

Draait v129 coordinate descent met en zonder Zorn-associator 
vertakking op MWIS instances waar bond_dim < exact, zodat v129
niet meer gegarandeerd optimaal is.

Gebruik:
    python fibmps_zorn_benchmark.py
    python fibmps_zorn_benchmark.py --sizes 5 6 7 8 --bond-dim 3
    python fibmps_zorn_benchmark.py --sizes 6 --bond-dim 2 --zorn-branches 12

Vereist: fibmps_v120_peps_mwis.py, fibmps_v123_frontier_pressure_peps.py,
         fibmps_v129_confidence_governed_frontier_peps.py
         in dezelfde directory.

Author: Gertjan & Claude — April 2026
"""

from __future__ import annotations

import argparse
import json
import random as pyrandom
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from fibmps_v120_peps_mwis import (
    solve_mwis_exact_bruteforce,
    is_valid_independent_set,
    score_assignment,
)
from fibmps_v129_confidence_governed_frontier_peps import (
    solve_confidence_governed_distilled_frontier_v129,
)


# ============================================================
# ZORN ALGEBRA
# ============================================================

class Zorn:
    __slots__ = ['a', 'b', 'al', 'be']
    
    def __init__(self, a, b, al, be):
        self.a = float(a)
        self.b = float(b)
        self.al = np.asarray(al, dtype=float)
        self.be = np.asarray(be, dtype=float)
    
    def __mul__(self, o):
        return Zorn(
            self.a * o.a + self.al @ o.be,
            self.be @ o.al + self.b * o.b,
            self.a * o.al + o.b * self.al + np.cross(self.be, o.be),
            o.a * self.be + self.b * o.be - np.cross(self.al, o.al)
        )
    
    def enorm(self):
        return np.sqrt(self.a**2 + self.b**2 + self.al @ self.al + self.be @ self.be)
    
    def norm(self):
        n = self.enorm()
        return Zorn(self.a/n, self.b/n, self.al/n, self.be/n) if n > 1e-15 else Zorn(1,1,np.zeros(3),np.zeros(3))
    
    @staticmethod
    def rand(rng):
        return Zorn(
            rng.standard_normal(), rng.standard_normal(),
            rng.standard_normal(3), rng.standard_normal(3)
        ).norm()


def zorn_seeds(base_seed: int, n_branches: int, rng: np.random.Generator) -> List[int]:
    """
    Genereer diverse seeds via Zorn-associator vertakking.
    
    Elke vertakking: drie random Zorn-elementen → associator →
    de niet-associatieve afwijking bepaalt de seed-offset.
    
    Dit geeft STRUCTUREEL diverse startpunten: de associator
    vertakt in 8 dimensies, waardoor de seeds niet-uniform
    maar algebraïsch gespreid zijn.
    """
    seeds = [base_seed]
    seen = {base_seed}
    
    for _ in range(n_branches - 1):
        A, B, C = Zorn.rand(rng), Zorn.rand(rng), Zorn.rand(rng)
        
        # Twee paden door de algebra
        L = (A * B) * C
        R = A * (B * C)
        
        # Associator componenten als seed-bron
        diff_a = abs(L.a - R.a)
        diff_b = abs(L.b - R.b)
        diff_al = np.sum(np.abs(L.al - R.al))
        diff_be = np.sum(np.abs(L.be - R.be))
        
        # Combineer tot unieke seed
        raw = int((diff_a * 1e8 + diff_b * 1e6 + diff_al * 1e4 + diff_be * 1e2)) % 100000
        seed = base_seed + raw + 1
        
        # Vermijd duplicaten
        while seed in seen:
            seed += 1
        seen.add(seed)
        seeds.append(seed)
    
    return seeds


# ============================================================
# INSTANCE GENERATIE
# ============================================================

def make_grid(h: int, w: int, seed: int, mode: str = "random"):
    """Genereer MWIS instance."""
    rng = pyrandom.Random(seed)
    rows = []
    for _ in range(h):
        row = []
        for _ in range(w):
            if mode == "random":
                row.append(float(rng.randint(1, 9) + rng.random()))
            elif mode == "near_tied":
                row.append(5.0 + rng.uniform(-0.4, 0.4))
            elif mode == "hard":
                # Gewichten dicht bij elkaar → moeilijker te onderscheiden
                row.append(3.0 + rng.uniform(-0.1, 0.1))
        rows.append(tuple(row))
    return tuple(rows)


def build_instances(sizes: List[int], seeds_per_size: int = 3,
                    seed_base: int = 90200) -> List[Tuple[str, Any]]:
    """Bouw een set instances van verschillende groottes."""
    instances = []
    for size in sizes:
        for s in range(seeds_per_size):
            seed = seed_base + size * 100 + s
            
            # Vierkant
            label = f"{size}x{size}_s{s}"
            instances.append((label, make_grid(size, size, seed)))
            
            # Rechthoekig (als size > 3)
            if size >= 4:
                label = f"{size-1}x{size+1}_s{s}"
                instances.append((label, make_grid(size-1, size+1, seed + 50)))
            
            # Near-tied variant (moeilijker)
            label = f"{size}x{size}_tied_s{s}"
            instances.append((label, make_grid(size, size, seed + 70, "near_tied")))
    
    return instances


# ============================================================
# BENCHMARK
# ============================================================

def run_comparison(
    instances: List[Tuple[str, Any]],
    bond_dim: int = 2,
    v129_iterations: int = 6,
    zorn_branches: int = 7,
    base_seed: int = 42,
    compute_exact: bool = True,
    exact_max_size: int = 20,  # max h*w voor brute-force exact
) -> Dict[str, Any]:
    """
    Vergelijk v129 standaard vs v129+Zorn vertakking.
    
    v129 standaard: bond_dim, v129_iterations iteraties, seed=base_seed
    v129+Zorn: bond_dim, 1 iteratie × zorn_branches seeds (zelfde totaal budget)
    """
    results = []
    zorn_rng = np.random.default_rng(base_seed)
    
    print(f"  {'Instance':>16s}  {'Shape':>6s}  {'Exact':>8s}  {'v129':>8s}  "
          f"{'v129+Z':>8s}  {'v129 r':>7s}  {'+Z r':>7s}  {'Δ':>6s}  {'t_v129':>7s}  {'t_+Z':>7s}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    for label, grid in instances:
        h = len(grid)
        w = len(grid[0])
        shape = f"{h}x{w}"
        
        # Exact oplossing (als niet te groot)
        exact_score = None
        if compute_exact and h * w <= exact_max_size:
            exact = solve_mwis_exact_bruteforce(grid)
            exact_score = exact.score
        
        # v129 standaard
        t1 = time.perf_counter()
        r129 = solve_confidence_governed_distilled_frontier_v129(
            grid, bond_dim=bond_dim,
            max_coordinate_iterations=v129_iterations,
            seed=base_seed
        )
        t_v129 = (time.perf_counter() - t1) * 1000
        
        v129_score = score_assignment(grid, r129.assignment) if is_valid_independent_set(grid, r129.assignment) else 0
        
        # v129 + Zorn vertakking
        seeds = zorn_seeds(base_seed, zorn_branches, zorn_rng)
        iters_per_branch = max(1, v129_iterations // zorn_branches)
        
        best_zorn_score = 0
        best_zorn_assign = None
        
        t2 = time.perf_counter()
        for seed in seeds:
            r = solve_confidence_governed_distilled_frontier_v129(
                grid, bond_dim=bond_dim,
                max_coordinate_iterations=iters_per_branch,
                seed=seed
            )
            s = score_assignment(grid, r.assignment) if is_valid_independent_set(grid, r.assignment) else 0
            if s > best_zorn_score:
                best_zorn_score = s
                best_zorn_assign = r.assignment
        t_zorn = (time.perf_counter() - t2) * 1000
        
        # Ratios
        if exact_score and exact_score > 0:
            v129_ratio = v129_score / exact_score
            zorn_ratio = best_zorn_score / exact_score
        else:
            # Gebruik v129 als referentie
            ref = max(v129_score, best_zorn_score)
            v129_ratio = v129_score / ref if ref > 0 else 0
            zorn_ratio = best_zorn_score / ref if ref > 0 else 0
        
        delta = best_zorn_score - v129_score
        delta_str = f"+{delta:.2f}" if delta > 0.01 else ("=" if abs(delta) < 0.01 else f"{delta:.2f}")
        
        result = {
            'label': label, 'shape': shape, 'h': h, 'w': w,
            'exact': exact_score,
            'v129_score': v129_score, 'v129_ratio': v129_ratio,
            'zorn_score': best_zorn_score, 'zorn_ratio': zorn_ratio,
            'delta': delta,
            't_v129_ms': t_v129, 't_zorn_ms': t_zorn,
        }
        results.append(result)
        
        exact_str = f"{exact_score:8.2f}" if exact_score else "      --"
        print(f"  {label:>16s}  {shape:>6s}  {exact_str}  {v129_score:8.2f}  "
              f"{best_zorn_score:8.2f}  {v129_ratio:7.4f}  {zorn_ratio:7.4f}  "
              f"{delta_str:>6s}  {t_v129:7.0f}  {t_zorn:7.0f}")
    
    return {
        'config': {
            'bond_dim': bond_dim,
            'v129_iterations': v129_iterations,
            'zorn_branches': zorn_branches,
            'base_seed': base_seed,
        },
        'results': results,
    }


def summarize(data: Dict) -> str:
    """Print samenvatting."""
    results = data['results']
    n = len(results)
    
    v129_ratios = [r['v129_ratio'] for r in results]
    zorn_ratios = [r['zorn_ratio'] for r in results]
    
    zorn_wins = sum(1 for r in results if r['delta'] > 0.01)
    v129_wins = sum(1 for r in results if r['delta'] < -0.01)
    ties = n - zorn_wins - v129_wins
    
    exact_results = [r for r in results if r['exact'] is not None]
    if exact_results:
        v129_exact_hits = sum(1 for r in exact_results if r['v129_ratio'] > 0.9999)
        zorn_exact_hits = sum(1 for r in exact_results if r['zorn_ratio'] > 0.9999)
    else:
        v129_exact_hits = zorn_exact_hits = 0
    
    lines = [
        f"",
        f"  SAMENVATTING",
        f"  {'─'*50}",
        f"  Instances: {n}",
        f"  Bond dim:  {data['config']['bond_dim']}",
        f"  v129 iteraties: {data['config']['v129_iterations']}",
        f"  Zorn branches:  {data['config']['zorn_branches']}",
        f"",
        f"  v129 mean ratio:      {np.mean(v129_ratios):.4f} ± {np.std(v129_ratios):.4f}",
        f"  v129+Zorn mean ratio: {np.mean(zorn_ratios):.4f} ± {np.std(zorn_ratios):.4f}",
        f"",
        f"  Zorn wint:  {zorn_wins}/{n}",
        f"  v129 wint:  {v129_wins}/{n}",
        f"  Gelijk:     {ties}/{n}",
    ]
    
    if exact_results:
        lines.extend([
            f"",
            f"  Exact hits (van {len(exact_results)} met exact ref):",
            f"    v129:      {v129_exact_hits}/{len(exact_results)}",
            f"    v129+Zorn: {zorn_exact_hits}/{len(exact_results)}",
        ])
    
    # Per-size breakdown
    sizes = sorted(set((r['h'], r['w']) for r in results))
    if len(sizes) > 1:
        lines.append(f"\n  Per grootte:")
        for h, w in sizes:
            subset = [r for r in results if r['h'] == h and r['w'] == w]
            v_mean = np.mean([r['v129_ratio'] for r in subset])
            z_mean = np.mean([r['zorn_ratio'] for r in subset])
            wins = sum(1 for r in subset if r['delta'] > 0.01)
            lines.append(f"    {h}×{w}: v129={v_mean:.4f} +Zorn={z_mean:.4f} "
                        f"(Zorn wint {wins}/{len(subset)})")
    
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FibMPS v129 vs v129+Zorn benchmark"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6],
                       help="Grid sizes to test (default: 4 5 6)")
    parser.add_argument("--bond-dim", type=int, default=2,
                       help="MPS bond dimension (default: 2)")
    parser.add_argument("--v129-iters", type=int, default=6,
                       help="v129 coordinate descent iterations (default: 6)")
    parser.add_argument("--zorn-branches", type=int, default=7,
                       help="Number of Zorn-associator branches (default: 7)")
    parser.add_argument("--seeds-per-size", type=int, default=3,
                       help="Random instances per grid size (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON path (default: fibmps_zorn_results.json)")
    args = parser.parse_args()
    
    print("═" * 70)
    print("  FibMPS v129 vs v129+Zorn BENCHMARK")
    print(f"  Sizes: {args.sizes}, bond_dim={args.bond_dim}, "
          f"branches={args.zorn_branches}")
    print("═" * 70)
    
    instances = build_instances(
        args.sizes, 
        seeds_per_size=args.seeds_per_size,
        seed_base=90200
    )
    
    print(f"\n  {len(instances)} instances gegenereerd\n")
    
    t0 = time.time()
    
    data = run_comparison(
        instances,
        bond_dim=args.bond_dim,
        v129_iterations=args.v129_iters,
        zorn_branches=args.zorn_branches,
        base_seed=args.seed,
    )
    
    elapsed = time.time() - t0
    
    print(f"\n{'─'*70}")
    print(summarize(data))
    print(f"\n  Totale tijd: {elapsed:.1f}s")
    
    # Save results
    output = args.output or Path("fibmps_zorn_results.json")
    data['elapsed_s'] = elapsed
    data['timestamp'] = time.time()
    with output.open('w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Resultaten opgeslagen: {output}")


if __name__ == '__main__':
    main()
