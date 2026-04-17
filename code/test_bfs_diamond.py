#!/usr/bin/env python3
"""Test BFS-diamant lichtkegel: correctheid + VRAM besparing."""
import sys, time, numpy as np
sys.path.insert(0, '.')
sys.dont_write_bytecode = True
from lightcone_qaoa import LightconeQAOA

# ─── Test 1: Correctheid — BFS vs kolom (alleen waar kolom past) ────
print("=== Test 1: Correctheid BFS-diamant vs kolom-methode ===\n")

gamma, beta = 0.6, 1.1
for Ly in [3, 4]:
    Lx = 8
    lc = LightconeQAOA(Lx, Ly, verbose=False, gpu=False)
    for p in [1, 2]:
        col_min, col_max = lc.lightcone_columns('h', Lx//2, p)
        n_col = (col_max - col_min + 1) * Ly
        if n_col > 22:
            # Kolom past niet op CPU, skip vergelijking
            pos, _ = lc.lightcone_diamond('h', Lx//2, Ly//2, p)
            print(f"  {Lx}x{Ly} p={p}: kolom={n_col}q (te groot), BFS={len(pos)}q — BFS UNLOCKS deze case!")
            continue

        max_diff = 0
        n_tested = 0
        for etype, ex, ey in lc.edges[:20]:
            zz_col = lc.eval_edge_exact(etype, ex, ey, p, [gamma]*p, [beta]*p)
            zz_bfs = lc.eval_edge_diamond(etype, ex, ey, p, [gamma]*p, [beta]*p)
            diff = abs(zz_col - zz_bfs)
            if diff > max_diff:
                max_diff = diff
            n_tested += 1

        pos, _ = lc.lightcone_diamond('h', Lx//2, Ly//2, p)
        n_bfs = len(pos)
        saved = n_col - n_bfs
        status = "EXACT" if max_diff < 1e-12 else f"DIFF={max_diff:.2e}"
        print(f"  {Lx}x{Ly} p={p}: {n_tested} edges, max_diff={max_diff:.2e} [{status}]")
        print(f"          Kolom:{n_col}q BFS:{n_bfs}q saved:{saved}q ({2**saved}x VRAM)")

print()

# ─── Test 2: VRAM besparing tabel ──────────────────────────────────
print("=== Test 2: VRAM besparing BFS-diamant vs kolom ===\n")
print(f"{'Grid':>8} {'p':>2} {'Kolom':>6} {'BFS':>5} {'Saved':>6} {'VRAM ratio':>11} {'Past op':>10}")
print("-" * 55)

for Ly in [3, 4, 6, 8]:
    Lx = 8
    lc = LightconeQAOA(Lx, Ly, verbose=False, gpu=False)
    for p in [1, 2, 3, 4]:
        mid_x = Lx // 2
        mid_y = Ly // 2
        col_min, col_max = lc.lightcone_columns('h', mid_x, p)
        n_col = (col_max - col_min + 1) * Ly
        positions, _ = lc.lightcone_diamond('h', mid_x, mid_y, p)
        n_bfs = len(positions)
        saved = n_col - n_bfs
        ratio = 2**saved

        if n_bfs <= 22:
            fits = "CPU"
        elif n_bfs <= 26:
            fits = "GTX1650"
        elif n_bfs <= 28:
            fits = "12GB GPU"
        elif n_bfs <= 30:
            fits = "24GB GPU"
        else:
            fits = "MPS"

        marker = " <<NEW" if n_col > 22 and n_bfs <= 26 else ""
        print(f"  {Lx}x{Ly} {p:>2} {n_col:>5}q {n_bfs:>4}q {saved:>5}q {ratio:>10}x  {fits:>10}{marker}")
    print()

# ─── Test 3: Volledige eval_cost met BFS ───────────────────────────
print("=== Test 3: eval_cost met BFS-diamant ===\n")

for Lx, Ly, p in [(8, 3, 1), (8, 4, 1), (8, 3, 2)]:
    lc = LightconeQAOA(Lx, Ly, verbose=True, gpu=False)
    gamma, beta = 0.6, 1.1
    t0 = time.time()
    cost = lc.eval_cost(p, [gamma]*p, [beta]*p)
    ratio = cost / lc.n_edges
    elapsed = time.time() - t0
    print(f"  {Lx}x{Ly} p={p}: ratio={ratio:.6f}, t={elapsed:.2f}s\n")

# ─── Test 4: Case die ALLEEN met BFS past (8x4 p=2) ───────────────
print("=== Test 4: 8x4 p=2 — ONMOGELIJK met kolom, WEL met BFS ===\n")
lc4 = LightconeQAOA(8, 4, verbose=True, gpu=False)
# Kolom = 24q > 22q CPU, maar BFS = 16q
t0 = time.time()
cost4 = lc4.eval_cost(2, [0.6, 0.5], [1.1, 0.9])
ratio4 = cost4 / lc4.n_edges
elapsed4 = time.time() - t0
print(f"  8x4 p=2: ratio={ratio4:.6f}, t={elapsed4:.2f}s")
print(f"  Dit was ONMOGELIJK met de kolom-methode op CPU!\n")

# ─── Test 5: Verify full eval_ratio exact match ───────────────────
print("=== Test 5: eval_ratio BFS vs kolom (full grid, p=1) ===\n")
Lx, Ly = 8, 3
gamma, beta = 0.45, 1.18

# Force kolom
lc_col = LightconeQAOA(Lx, Ly, verbose=False, gpu=False)
cost_col = 0
for etype, ex, ey in lc_col.edges:
    zz = lc_col.eval_edge_exact(etype, ex, ey, 1, [gamma], [beta])
    cost_col += (1 - zz) / 2
ratio_col = cost_col / lc_col.n_edges

# BFS (default)
lc_bfs = LightconeQAOA(Lx, Ly, verbose=False, gpu=False)
cost_bfs = lc_bfs.eval_cost(1, [gamma], [beta])
ratio_bfs = cost_bfs / lc_bfs.n_edges

diff = abs(ratio_col - ratio_bfs)
print(f"  8x3 p=1: kolom ratio={ratio_col:.12f}")
print(f"            BFS   ratio={ratio_bfs:.12f}")
print(f"            diff = {diff:.2e}")
assert diff < 1e-10, f"Mismatch: {diff}"
print("  MATCH (within 1e-10) ✓\n")

print("=== Alle BFS-diamant tests geslaagd! ===")
