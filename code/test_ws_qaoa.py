#!/usr/bin/env python3
"""Test B69: WS-QAOA — SDP Warm-Started QAOA."""
import sys
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

import numpy as np

# --- Test 1: SDP solve + GW rounding ---
print("=== Test 1: SDP solve op 4x3 cylinder ===\n")
from ws_qaoa import gw_sdp_solve, _cylinder_edges

edges = _cylinder_edges(4, 3)
sdp = gw_sdp_solve(12, edges, verbose=True)
assert sdp['sdp_bound'] is not None, "SDP failed"
assert sdp['best_cut'] > 0, "GW rounding failed"
# 4x3 cylinder is bipartite: MaxCut = all 17 edges
assert abs(sdp['sdp_bound'] - 17.0) < 0.1, f"SDP bound wrong: {sdp['sdp_bound']}"
assert sdp['best_cut'] == 17, f"GW should find perfect cut: {sdp['best_cut']}"
print("  PASS\n")

# --- Test 2: Warm-start angles ---
print("=== Test 2: Warm-start angles ===\n")
from ws_qaoa import sdp_warm_start

angles = sdp_warm_start(4, 3, epsilon=0.25, mode='binary', verbose=False)
assert angles.shape == (4, 3), f"Shape wrong: {angles.shape}"

# Check angles are in valid range
assert np.all(angles >= 0) and np.all(angles <= np.pi), "Angles out of range"

# Binary mode: angles should be either 2*eps or pi-2*eps
eps = 0.25
unique_angles = set(np.round(angles.flatten(), 6))
expected = {round(2 * eps, 6), round(np.pi - 2 * eps, 6)}
assert unique_angles == expected, f"Unexpected angles: {unique_angles} vs {expected}"
print(f"  Binary angles: {unique_angles}")

# Continuous mode
angles_c = sdp_warm_start(4, 3, epsilon=0.25, mode='continuous', verbose=False)
assert angles_c.shape == (4, 3)
assert np.all(angles_c >= 2 * eps - 0.01)
assert np.all(angles_c <= np.pi - 2 * eps + 0.01)
print(f"  Continuous angles: mean={np.mean(angles_c):.3f} std={np.std(angles_c):.3f}")
print("  PASS\n")

# --- Test 3: MPS normalisatie ---
print("=== Test 3: Warm MPS normalisatie ===\n")
from ws_qaoa import warm_start_mps, cold_start_mps

ws_mps = warm_start_mps(4, 3, angles)
cold_mps = cold_start_mps(4, 3)

for x in range(4):
    norm_w = np.sum(np.abs(ws_mps[x]) ** 2)
    norm_c = np.sum(np.abs(cold_mps[x]) ** 2)
    assert abs(norm_w - 1.0) < 1e-10, f"Warm MPS site {x} not normalized: {norm_w}"
    assert abs(norm_c - 1.0) < 1e-10, f"Cold MPS site {x} not normalized: {norm_c}"
print("  All sites normalized to 1.0")

# Cold start = uniform
d = 8
for x in range(4):
    expected_amp = 1.0 / np.sqrt(d)
    assert np.allclose(cold_mps[x], expected_amp), "Cold MPS not uniform"
print("  Cold MPS = uniform |+> confirmed")
print("  PASS\n")

# --- Test 4: TransverseQAOA warm vs cold ---
print("=== Test 4: TransverseQAOA warm vs cold (4x3 p=1) ===\n")
from transverse_contraction import TransverseQAOA

tc = TransverseQAOA(4, 3, verbose=False)
cold_ratio = tc.optimize(1, n_gamma=12, n_beta=12, refine=True)[0]

angles_w = sdp_warm_start(4, 3, epsilon=0.2, mode='binary', verbose=False)
tc_w = TransverseQAOA(4, 3, verbose=False)
warm_ratio = tc_w.optimize(1, n_gamma=12, n_beta=12, refine=True,
                           warm_angles=angles_w)[0]

print(f"  Cold QAOA-1: {cold_ratio:.6f}")
print(f"  Warm QAOA-1: {warm_ratio:.6f}")
print(f"  Delta: {warm_ratio - cold_ratio:+.6f}")
assert warm_ratio > cold_ratio, "Warm should beat cold on bipartite grid"
assert warm_ratio > 0.95, f"Warm should be >0.95 on bipartite grid: {warm_ratio}"
print("  PASS\n")

# --- Test 5: TDQS warm vs cold ---
print("=== Test 5: TDQS warm vs cold (4x3) ===\n")
from tdqs import TDQS

s_cold = TDQS(4, 3, chi_max=16, verbose=False)
r_cold = s_cold.solve(n_layers=2)

angles_t = sdp_warm_start(4, 3, epsilon=0.2, mode='binary', verbose=False)
s_warm = TDQS(4, 3, chi_max=16, verbose=False, warm_angles=angles_t)
r_warm = s_warm.solve(n_layers=2)

print(f"  Cold TDQS: {r_cold['ratio']:.6f}")
print(f"  Warm TDQS: {r_warm['ratio']:.6f}")
print(f"  Delta: {r_warm['ratio'] - r_cold['ratio']:+.6f}")
assert r_warm['ratio'] > r_cold['ratio'], "Warm TDQS should beat cold"
assert r_warm['ratio'] > 0.95, f"Warm TDQS should be >0.95: {r_warm['ratio']}"
print("  PASS\n")

# --- Test 6: epsilon=pi/4 = cold start ---
print("=== Test 6: epsilon=pi/4 recovers cold start ===\n")
angles_cold = sdp_warm_start(4, 3, epsilon=np.pi / 4, mode='binary', verbose=False)
# All angles should be pi/2 (= |+>)
assert np.allclose(angles_cold, np.pi / 2, atol=1e-10), \
    f"eps=pi/4 should give pi/2 everywhere: {angles_cold}"
print("  eps=pi/4 -> all angles = pi/2 (cold start)")
print("  PASS\n")

print("=== Alle B69 WS-QAOA tests geslaagd! ===")
