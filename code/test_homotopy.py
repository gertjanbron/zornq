#!/usr/bin/env python3
"""Test B71: Homotopy Optimizer — Parameter Continuation."""
import sys
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

import numpy as np

# --- Test 1: lambda=1 matcht TransverseQAOA ---
print("=== Test 1: eval_ratio_lambda(lam=1) == TransverseQAOA ===\n")
from homotopy_optimizer import HomotopyQAOA
from transverse_contraction import TransverseQAOA

h = HomotopyQAOA(4, 3, verbose=False)
t = TransverseQAOA(4, 3, verbose=False)

for g, b in [(0.5, 0.4), (1.0, 0.8), (2.5, 0.3)]:
    r_h = h.eval_ratio_lambda(1, [g], [b], lam=1.0)
    r_t = t.eval_ratio(1, [g], [b])
    assert abs(r_h - r_t) < 1e-10, \
        f"Mismatch: {r_h} vs {r_t} at g={g}, b={b}"
print("  Alle 3 parameter-sets matchen exact")
print("  PASS\n")

# --- Test 2: lambda=0 geeft geldige ratio ---
print("=== Test 2: eval_ratio_lambda(lam=0) is geldig ===\n")

r_0 = h.eval_ratio_lambda(1, [0.5], [0.4], lam=0.0)
assert 0 <= r_0 <= 1.0, f"Ongeldige ratio bij lam=0: {r_0}"
print(f"  lam=0: ratio={r_0:.6f} (geldig)")

# Lambda=0 met optimale params moet ratio=1.0 geven
# (3-qubit chain, p=1 QAOA kan optimal cut vinden)
r_0_opt, g_0, b_0 = h.optimize_at_lambda(1, 0.0, n_gamma=10, n_beta=10)
print(f"  lam=0 optimaal: ratio={r_0_opt:.6f}")
# Op Ly=3: 2 verticale edges per kolom, max cut=2, ratio=1.0 bereikbaar
# Maar QAOA-1 op een 3-qubit path met optimale params kan ratio=1.0 niet altijd halen
assert r_0_opt > 0.7, f"Optimale lam=0 ratio te laag: {r_0_opt}"
print("  PASS\n")

# --- Test 3: lambda monotoon (ratio daalt typisch als koppeling toeneemt) ---
print("=== Test 3: Lambda-pad smoothness ===\n")

# Optimale p=1 params bij lam=1
r_full, g_full, b_full = h.optimize_at_lambda(1, 1.0, n_gamma=10, n_beta=10)

# Meet ratio met die params bij verschillende lam
ratios = []
for lam in np.linspace(0, 1, 11):
    r = h.eval_ratio_lambda(1, g_full, b_full, lam=lam)
    ratios.append((float(lam), r))
    print(f"  lam={lam:.1f}: ratio={r:.6f}")

# De ratio moet smooth veranderen (geen sprongen > 0.2)
for i in range(1, len(ratios)):
    diff = abs(ratios[i][1] - ratios[i-1][1])
    assert diff < 0.2, f"Sprong bij lam={ratios[i][0]}: {diff:.4f}"
print("  Smooth pad bevestigd")
print("  PASS\n")

# --- Test 4: lambda-continuation vindt zelfde/betere ratio als direct ---
print("=== Test 4: Lambda-continuation 4x3 p=1 ===\n")

result_lam = h.solve_lambda(1, n_lambda=5, n_gamma=10, n_beta=10)

r_direct, _, _ = h.optimize_at_lambda(1, 1.0, n_gamma=10, n_beta=10)

assert result_lam['ratio'] >= r_direct - 0.001, \
    f"Homotopy ({result_lam['ratio']}) veel slechter dan direct ({r_direct})"
print(f"  Lambda-continuation: {result_lam['ratio']:.6f}")
print(f"  Direct:              {r_direct:.6f}")
print(f"  Delta:               {result_lam['ratio'] - r_direct:+.6f}")
print("  PASS\n")

# --- Test 5: p-continuation 4x3 p=1->2 ---
print("=== Test 5: p-Continuation 4x3 p=1->2 ===\n")

result_p = h.solve_p_continuation(2, n_gamma=10, n_beta=10)

assert len(result_p['layers']) == 2
assert result_p['layers'][0]['p'] == 1
assert result_p['layers'][1]['p'] == 2
assert result_p['layers'][1]['ratio'] > result_p['layers'][0]['ratio'], \
    "p=2 moet beter zijn dan p=1"

# p=2 moet beter zijn dan p=1
p1_ratio = result_p['layers'][0]['ratio']
p2_ratio = result_p['layers'][1]['ratio']
print(f"  p=1: {p1_ratio:.6f}")
print(f"  p=2: {p2_ratio:.6f} (+{p2_ratio - p1_ratio:.6f})")
assert p2_ratio > 0.75, f"p=2 ratio te laag: {p2_ratio}"
print("  PASS\n")

# --- Test 6: Auto mode 4x3 p=2 ---
print("=== Test 6: Auto mode 4x3 p=2 ===\n")

result_auto = h.solve(2, n_lambda=3, n_gamma=10, n_beta=10, mode='auto')

assert result_auto['ratio'] > 0.75, \
    f"Auto mode ratio te laag: {result_auto['ratio']}"
assert 'ratio_direct' in result_auto
assert 'delta' in result_auto

print(f"  Homotopy: {result_auto['ratio']:.6f}")
print(f"  Direct:   {result_auto['ratio_direct']:.6f}")
print(f"  Delta:    {result_auto['delta']:+.6f}")
print(f"  Tijd:     {result_auto['elapsed']:.1f}s")
print("  PASS\n")

# --- Summary ---
print("=" * 60)
print("=== Alle B71 Homotopy Optimizer tests geslaagd! ===")
print("=" * 60)
print(f"\nSamenvatting:")
print(f"  Lambda-cont 4x3 p=1: {result_lam['ratio']:.4f}")
print(f"  p-Continuation p=2:  {result_p['ratio']:.4f}")
print(f"  Auto mode p=2:       {result_auto['ratio']:.4f}")
