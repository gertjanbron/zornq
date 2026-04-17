#!/usr/bin/env python3
"""Test B70: Hotspot Repair — Frustration-Patch Solver."""
import sys
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

import numpy as np

# --- Test 1: Per-edge ZZ correlaties ---
print("=== Test 1: Per-edge ZZ via Tier 1 (4x3 cold) ===\n")
from hotspot_repair import HotspotRepair

hr = HotspotRepair(4, 3, p_global=1, p_local=2,
                   frustration_threshold=0.4, verbose=False)
ratio_t1, zz, gammas, betas, _ = hr._tier1_global()

# Ratio should match known QAOA-1 value
assert abs(ratio_t1 - 0.6944) < 0.01, f"Tier 1 ratio wrong: {ratio_t1}"

# Should have all 17 edges
assert len(zz) == 17, f"Expected 17 edges, got {len(zz)}"

# All ZZ should be negative (cut edges in QAOA)
for key, val in zz.items():
    assert val < 0, f"Edge {key} has positive ZZ={val}"

# Sum of (1-ZZ)/2 should give ratio * n_edges
cost = sum((1 - v) / 2 for v in zz.values())
ratio_check = cost / 17
assert abs(ratio_check - ratio_t1) < 1e-6, \
    f"Ratio mismatch: {ratio_check} vs {ratio_t1}"
print(f"  Tier 1: ratio={ratio_t1:.6f}, {len(zz)} edges, all ZZ<0")
print("  PASS\n")

# --- Test 2: Hotspot identification ---
print("=== Test 2: Hotspot identification ===\n")

# threshold=0.4 should find ~9 edges (the ones with |ZZ|<0.4)
hotspots, cold_cost = hr._identify_hotspots(zz)
n_hot = len(hotspots)
assert n_hot > 0, "Should find some hotspots at threshold=0.4"
assert n_hot < 17, "Should not mark ALL edges as hotspots"

# Cold cost + hotspot cost should equal total cost
hot_cost = sum((1 - zz[k]) / 2 for k in hotspots)
total_cost = cold_cost + hot_cost
assert abs(total_cost - cost) < 1e-10, \
    f"Cost split wrong: {cold_cost} + {hot_cost} != {cost}"
print(f"  Hotspots: {n_hot}/17, cold_cost={cold_cost:.4f}")
print("  PASS\n")

# --- Test 3: Exact gadget patch path on a tiny hotspot ---
print("=== Test 3: Exact gadget repair path ===\n")

first_hotspot = sorted(hotspots)[0]
reference_assignment = hr._build_reference_assignment(zz)
gadget_repair = hr._exact_gadget_repair(first_hotspot, zz[first_hotspot],
                                        reference_assignment)
assert gadget_repair is not None, "Exact gadget repair should produce metadata"
assert gadget_repair['patch_nodes'] <= hr.exact_gadget_nodes_max
assert gadget_repair['patch_edges'] > 0
assert gadget_repair['boundary_pins'] > 0
assert gadget_repair['certificate'] == 'EXACT_GADGET'
print(f"  Gadget patch for {first_hotspot}: "
      f"nodes={gadget_repair['patch_nodes']}, "
      f"edges={gadget_repair['patch_edges']}, "
      f"pins={gadget_repair['boundary_pins']}, "
      f"used={gadget_repair['used']}")
print("  PASS\n")

# --- Test 4: Gadget mode variants ---
print("=== Test 4: Gadget mode variants ===\n")

hr_free = HotspotRepair(4, 3, p_global=1, p_local=2,
                        frustration_threshold=0.4, verbose=False,
                        exact_gadget_mode='free')
free_repair = hr_free._exact_gadget_repair(first_hotspot, zz[first_hotspot], None)
assert free_repair is not None, "Free gadget repair should produce metadata"
assert free_repair['mode'] == 'free'
assert free_repair['boundary_pins'] == 0

hr_off = HotspotRepair(4, 3, p_global=1, p_local=2,
                       frustration_threshold=0.4, verbose=False,
                       exact_gadget_mode='off')
off_result = hr_off.solve()
assert off_result['exact_gadget_mode'] == 'off'
assert off_result['n_exact_gadget_repairs'] == 0
print(f"  Free mode pins={free_repair['boundary_pins']}, "
      f"off mode exact repairs={off_result['n_exact_gadget_repairs']}")
print("  PASS\n")

# --- Test 5: Full solve cold (4x3) ---
print("=== Test 5: Full solve cold 4x3, threshold=0.4 ===\n")

hr3 = HotspotRepair(4, 3, p_global=1, p_local=2,
                    frustration_threshold=0.4, verbose=True)
result = hr3.solve()

assert result['ratio'] >= result['ratio_tier1'], \
    "Repaired ratio should be >= Tier 1"
assert result['delta'] >= 0, f"Delta should be >= 0: {result['delta']}"
assert result['n_hotspots'] > 0, "Should have hotspots"
assert result['n_exact_gadget_repairs'] > 0, "Should use exact gadget repairs on 4x3"
assert result['n_exact_gadget_repairs'] + result['n_lightcone_repairs'] == result['n_hotspots']
assert any(src == 'exact_gadget' for src in result['repair_sources'].values())
assert all(meta['boundary_pins'] > 0 for meta in result['gadget_meta'].values())
assert result['exact_gadget_mode'] == 'boundary'
assert 'elapsed' in result
print(f"\n  Repaired: {result['ratio']:.6f} (delta={result['delta']:+.6f})")
print("  PASS\n")

# --- Test 6: No hotspots case ---
print("=== Test 6: No hotspots (very low threshold) ===\n")

hr4 = HotspotRepair(4, 3, p_global=1, p_local=2,
                    frustration_threshold=0.1, verbose=False)
result4 = hr4.solve()

assert result4['n_hotspots'] == 0, "Should have no hotspots at threshold=0.1"
assert result4['delta'] == 0.0, "Delta should be 0 with no hotspots"
assert abs(result4['ratio'] - result4['ratio_tier1']) < 1e-10
print(f"  No hotspots: ratio={result4['ratio']:.6f}")
print("  PASS\n")

# --- Test 7: Full solve cold (8x3) ---
print("=== Test 7: Full solve cold 8x3, threshold=0.4 ===\n")

hr5 = HotspotRepair(8, 3, p_global=1, p_local=2,
                    frustration_threshold=0.4, verbose=True)
result5 = hr5.solve()

assert result5['ratio'] >= result5['ratio_tier1'], \
    "Repaired should be >= Tier 1"
assert result5['ratio'] > 0.72, \
    f"8x3 repaired should be > 0.72: {result5['ratio']}"
assert result5['delta'] > 0.03, \
    f"8x3 delta should be > 3%: {result5['delta']}"
print(f"\n  8x3 repaired: {result5['ratio']:.6f} (delta={result5['delta']:+.6f})")
print("  PASS\n")

# --- Test 8: Warm-start (8x3) ---
print("=== Test 8: Warm-start 8x3 ===\n")

hr6 = HotspotRepair(8, 3, p_global=1, p_local=2,
                    frustration_threshold=0.4, warm=True, verbose=True)
result6 = hr6.solve()

assert result6['ratio'] > 0.95, \
    f"Warm 8x3 should be > 0.95: {result6['ratio']}"
# With warm-start, likely no hotspots
print(f"\n  Warm 8x3: ratio={result6['ratio']:.6f}, "
      f"hotspots={result6['n_hotspots']}")
print("  PASS\n")

# --- Summary ---
print("=" * 60)
print("=== Alle B70 Hotspot Repair tests geslaagd! ===")
print("=" * 60)
print(f"\nSamenvatting:")
print(f"  4x3 cold:  {result['ratio_tier1']:.4f} -> {result['ratio']:.4f} "
      f"(+{result['delta']:.4f})")
print(f"  8x3 cold:  {result5['ratio_tier1']:.4f} -> {result5['ratio']:.4f} "
      f"(+{result5['delta']:.4f})")
print(f"  8x3 warm:  {result6['ratio']:.4f} (hotspots={result6['n_hotspots']})")
