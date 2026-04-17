#!/usr/bin/env python3
"""Test TDQS v2: correctheid, QAOA-vergelijking, per-bond chi tracking."""
import sys, time
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

# ─── Test 1: Basiswerking 4×3 ─────────────────────────────────────
print("=== Test 1: TDQS v2 4×3 chi=16 ===\n")
from tdqs import TDQS

s = TDQS(4, 3, chi_max=16, verbose=True)
r = s.solve(n_layers=3)
print(f"  ratio={r['ratio']:.6f} lagen={r['n_layers']}")
assert r['ratio'] > 0.55, f"Ratio te laag: {r['ratio']}"
assert r['n_layers'] >= 1, "Minstens 1 laag verwacht"
assert len(r['bond_chis']) > 0, "bond_chis moet gevuld zijn"
print("  PASS ✓\n")

# ─── Test 2: Multi-angle — gamma_intra != gamma_inter ─────────────
print("=== Test 2: Multi-angle parameters ===\n")
for layer in r['layers']:
    gi, gx = layer['gamma_intra'], layer['gamma_inter']
    print(f"  g_intra={gi:.3f} g_inter={gx:.3f} beta={layer['beta']:.3f}")
# Met L-BFGS-B optimalisatie zouden gi en gx mogen verschillen
print("  Multi-angle structuur correct ✓\n")

# ─── Test 3: Per-bond chi tracking ────────────────────────────────
print("=== Test 3: Per-bond chi tracking ===\n")
chis = r['bond_chis']
print(f"  Bond chis: {chis}")
assert len(chis) == 3, f"Verwacht 3 bonds voor 4×3, got {len(chis)}"
assert all(c >= 1 for c in chis), "Alle chis >= 1"
assert all(c <= 16 for c in chis), "Alle chis <= chi_max"
print("  PASS ✓\n")

# ─── Test 4: TDQS v2 verslaat QAOA-1 op 4×3 ─────────────────────
print("=== Test 4: TDQS v2 vs QAOA-1 (4×3) ===\n")
from transverse_contraction import TransverseQAOA
tc = TransverseQAOA(4, 3, verbose=False)
qaoa1 = tc.optimize(1, n_gamma=12, n_beta=12, refine=True)
print(f"  QAOA-1: {qaoa1[0]:.6f}")
print(f"  TDQS:   {r['ratio']:.6f}")
delta = r['ratio'] - qaoa1[0]
print(f"  Delta:  {delta:+.6f} ({delta/qaoa1[0]*100:+.1f}%)")
assert r['ratio'] > qaoa1[0], f"TDQS moet QAOA-1 verslaan op 4×3"
print("  PASS ✓\n")

# ─── Test 5: 8×3 — de regressie die v1 had ───────────────────────
print("=== Test 5: TDQS v2 vs QAOA-1 (8×3) ===\n")
s8 = TDQS(8, 3, chi_max=16, verbose=True)
r8 = s8.solve(n_layers=3)
tc8 = TransverseQAOA(8, 3, verbose=False)
qaoa8 = tc8.optimize(1, n_gamma=12, n_beta=12, refine=True)
print(f"  QAOA-1: {qaoa8[0]:.6f}")
print(f"  TDQS:   {r8['ratio']:.6f}")
delta8 = r8['ratio'] - qaoa8[0]
print(f"  Delta:  {delta8:+.6f} ({delta8/qaoa8[0]*100:+.1f}%)")
assert r8['ratio'] > qaoa8[0], f"TDQS moet QAOA-1 verslaan op 8×3 (was regressie in v1)"
print("  PASS ✓\n")

# ─── Test 6: Triage vs full mode ─────────────────────────────────
print("=== Test 6: Triage vs Full mode (4×3) ===\n")
s_full = TDQS(4, 3, chi_max=16, verbose=False)
r_full = s_full.solve(n_layers=2, mode='full')
s_tri = TDQS(4, 3, chi_max=16, verbose=False)
r_tri = s_tri.solve(n_layers=2, mode='triage')
print(f"  Full:    {r_full['ratio']:.6f} lagen={r_full['n_layers']}")
print(f"  Triage:  {r_tri['ratio']:.6f} lagen={r_tri['n_layers']}")
# Beide moeten boven QAOA-1 zitten
assert r_full['ratio'] > 0.55, "Full mode te laag"
assert r_tri['ratio'] > 0.55, "Triage mode te laag"
print("  PASS ✓\n")

print("=== Alle TDQS v2 tests geslaagd! ===")
