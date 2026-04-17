#\!/usr/bin/env python3
"""Tests for B10f Classical Shadows / Monte Carlo MaxCut Solver."""

import numpy as np
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow_solver import (
    _precompute_cuts, _apply_qaoa_state_fast, _random_pauli_measurement,
    _shadow_expectation_zz, _shadow_expectation_median_of_means,
    _sample_bitstrings, _eval_cut_single, _local_search,
    shadow_energy, shadow_maxcut, shadow_maxcut_grid, shadow_convergence
)

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")


# ============================================================
# Test 1: Precompute cuts
# ============================================================
print("Test 1: _precompute_cuts")
# Triangle K3: 3 nodes, 3 edges
edges_k3 = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
cuts = _precompute_cuts(3, edges_k3)
# |000>=0, |001>=2, |010>=2, |011>=2, |100>=2, |101>=2, |110>=2, |111>=0
check("K3 all-zero cut = 0", cuts[0] == 0)
check("K3 all-one cut = 0", cuts[7] == 0)
check("K3 single-flip cut = 2", cuts[1] == 2)
check("K3 max cut = 2", np.max(cuts) == 2)


# ============================================================
# Test 2: QAOA state is normalized
# ============================================================
print("\nTest 2: QAOA state normalization")
n = 4
edges_4 = [(0,1,1.0), (1,2,1.0), (2,3,1.0)]  # path graph
psi, cuts = _apply_qaoa_state_fast(n, edges_4, [0.5], [0.8])
norm = np.sum(np.abs(psi)**2)
check(f"norm = {norm:.10f} approx 1", abs(norm - 1.0) < 1e-10)
check("state has 2^4=16 amplitudes", len(psi) == 16)


# ============================================================
# Test 3: Pauli measurement returns valid data
# ============================================================
print("\nTest 3: Random Pauli measurement")
rng = np.random.default_rng(42)
bases, outcome = _random_pauli_measurement(psi, n, rng)
check("bases has n elements", len(bases) == n)
check("outcome has n elements", len(outcome) == n)
check("bases in {0,1,2}", all(b in [0,1,2] for b in bases))
check("outcome in {0,1}", all(o in [0,1] for o in outcome))


# ============================================================
# Test 4: Shadow estimator unbiased on 2-node graph
# ============================================================
print("\nTest 4: Shadow estimator unbiased (2 nodes)")
n2 = 2
edges_2 = [(0, 1, 1.0)]
gammas = [0.5]
betas = [0.8]
psi2, cuts2 = _apply_qaoa_state_fast(n2, edges_2, gammas, betas)
probs2 = np.abs(psi2)**2
exact_E2 = np.dot(probs2, cuts2)

# Collect many shadows
rng2 = np.random.default_rng(123)
K = 20000
bases_list = []
outcomes_list = []
for _ in range(K):
    b, o = _random_pauli_measurement(psi2, n2, rng2)
    bases_list.append(b)
    outcomes_list.append(o)

est_E2 = _shadow_expectation_zz(bases_list, outcomes_list, n2, edges_2)
err2 = abs(est_E2 - exact_E2)
check(f"2-node shadow error = {err2:.4f} < 0.1", err2 < 0.1)


# ============================================================
# Test 5: Shadow estimator converges on 4x2 grid
# ============================================================
print("\nTest 5: Shadow convergence (4x2 grid)")
Lx, Ly = 4, 2
n8 = Lx * Ly
edges_8 = []
for x in range(Lx):
    for y in range(Ly):
        node = x * Ly + y
        if x + 1 < Lx: edges_8.append((node, (x+1)*Ly+y, 1.0))
        if y + 1 < Ly: edges_8.append((node, x*Ly+y+1, 1.0))

results = shadow_convergence(n8, edges_8, [0.5], [0.8],
    shadow_counts=[1000, 10000, 50000], seed=42)
err_1k = results[0]["abs_error"]
err_50k = results[2]["abs_error"]
check(f"50K error ({err_50k:.4f}) < 1K error ({err_1k:.4f}) or both small",
      err_50k < err_1k or err_50k < 0.2)
check(f"50K error ({err_50k:.4f}) < 0.5", err_50k < 0.5)


# ============================================================
# Test 6: shadow_energy returns correct structure
# ============================================================
print("\nTest 6: shadow_energy API")
res = shadow_energy(n2, edges_2, gammas, betas, n_shadows=2000, seed=42)
check("has energy_est", "energy_est" in res)
check("has energy_exact", "energy_exact" in res)
check("has abs_error", "abs_error" in res)
check("has stderr", "stderr" in res)
ee = res["energy_exact"]; check(f"energy_exact = {ee:.4f} matches direct calc",
      abs(res["energy_exact"] - exact_E2) < 1e-6)
ae = res["abs_error"]; check(f"abs_error ({ae:.4f}) < 0.3", ae < 0.3)


# ============================================================
# Test 7: Bitstring sampling
# ============================================================
print("\nTest 7: Bitstring sampling")
rng7 = np.random.default_rng(42)
indices = _sample_bitstrings(psi2, 10000, rng7)
check("sampled 10000 bitstrings", len(indices) == 10000)
check("all indices valid", np.all(indices >= 0) and np.all(indices < 4))
# Check distribution roughly matches |psi|^2
counts = np.bincount(indices, minlength=4) / 10000
max_diff = np.max(np.abs(counts - probs2))
check(f"sampling distribution close to |psi|^2 (max_diff={max_diff:.4f})", max_diff < 0.02)


# ============================================================
# Test 8: Local search finds optimum
# ============================================================
print("\nTest 8: Local search")
# Path graph 0-1-2-3: optimal cut = 3 (e.g., 0101)
best_bits, best_cut = _local_search(0, 4, edges_4)
check(f"local search from |0000> finds cut={best_cut:.0f} >= 3", best_cut >= 3)


# ============================================================
# Test 9: shadow_maxcut on small graphs
# ============================================================
print("\nTest 9: shadow_maxcut on triangle K3")
res9 = shadow_maxcut(3, edges_k3, p=1, n_restarts=10, n_samples=100,
                     n_shadows=200, local_search=True, seed=42)
bc9 = res9["best_cut"]; check(f"K3 best_cut={bc9:.0f} == 2 (exact opt)", bc9 == 2)
r9 = res9["ratio"]; check(f"K3 ratio={r9:.4f} == 1.0", abs(r9 - 1.0) < 1e-6)


# ============================================================
# Test 10: shadow_maxcut_grid
# ============================================================
print("\nTest 10: shadow_maxcut_grid 3x2")
res10 = shadow_maxcut_grid(3, 2, p=1, n_restarts=10, n_samples=100,
                           n_shadows=200, seed=42)
r10 = res10["ratio"]; check(f"3x2 ratio={r10:.4f} >= 0.9", r10 >= 0.9)
check("has time_s", "time_s" in res10)


# ============================================================
# Test 11: shadow_maxcut with p=2
# ============================================================
print("\nTest 11: shadow_maxcut p=2 on 4x2")
res11 = shadow_maxcut(n8, edges_8, p=2, n_restarts=15, n_samples=200,
                      n_shadows=300, seed=42)
r11 = res11["ratio"]; check(f"4x2 p=2 ratio={r11:.4f} >= 0.8", r11 >= 0.8)
bc11 = res11["best_cut"]; check(f"4x2 p=2 best_cut={bc11:.0f}", bc11 > 0)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed\!")
