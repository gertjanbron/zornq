#!/usr/bin/env python3
"""Test B76: p-ZNE — Richardson Extrapolatie op Circuitdiepte."""
import sys
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

import numpy as np

# --- Test 1: Data collectie p=1,2 op 4x3 ---
print("=== Test 1: Data collectie 4x3 p=1,2 ===\n")
from p_zne import collect_p_data, extrapolate, p_zne

data = collect_p_data(4, 3, p_max=2, n_gamma=10, n_beta=10, verbose=True)

assert len(data) == 2
assert data[0]['p'] == 1
assert data[1]['p'] == 2
assert data[1]['ratio'] > data[0]['ratio'], "p=2 moet beter zijn dan p=1"
assert 0.6 < data[0]['ratio'] < 0.8, f"p=1 ratio uit range: {data[0]['ratio']}"
assert 0.7 < data[1]['ratio'] < 0.9, f"p=2 ratio uit range: {data[1]['ratio']}"
print("  PASS\n")

# --- Test 2: Extrapolatie met 2 punten ---
print("=== Test 2: Extrapolatie 2 punten ===\n")

results = extrapolate(data, gw_ratio=1.0, verbose=True)

assert 'linear' in results, "Lineair moet beschikbaar zijn"
assert 'richardson' in results, "Richardson moet beschikbaar zijn"

# Extrapolatie moet hoger zijn dan p=2
for name, res in results.items():
    assert res['ratio_inf'] >= data[1]['ratio'] - 0.01, \
        f"{name} extrapolatie ({res['ratio_inf']}) lager dan p=2 ({data[1]['ratio']})"

# Op bipartite 4x3: limiet is 1.0, extrapolatie moet < 1.1
for name, res in results.items():
    assert res['ratio_inf'] < 1.1, \
        f"{name} extrapolatie onrealistisch hoog: {res['ratio_inf']}"

print("\n  PASS\n")

# --- Test 3: Volledige pipeline 4x3 ---
print("=== Test 3: Volledige p-ZNE pipeline 4x3 ===\n")

result = p_zne(4, 3, p_max=2, n_gamma=10, n_beta=10, verbose=True)

assert 'data' in result
assert 'extrapolations' in result
assert 'gw_ratio' in result
assert result['gw_ratio'] == 1.0, "4x3 is bipartite, GW=1.0"
assert 'elapsed' in result
print("\n  PASS\n")

# --- Test 4: 20x1 met p=1..3 (kwadratische fit) ---
print("=== Test 4: 20x1 p=1..3 (kwadratische fit) ===\n")

data_1d = collect_p_data(20, 1, p_max=3, verbose=True)

assert len(data_1d) == 3
# Ratios moeten monotoon stijgen
for i in range(1, len(data_1d)):
    assert data_1d[i]['ratio'] > data_1d[i-1]['ratio'], \
        f"Niet monotoon: p={data_1d[i]['p']}"

results_1d = extrapolate(data_1d, gw_ratio=1.0, verbose=True)

assert 'quadratic' in results_1d, "3 punten moet kwadratische fit geven"

# Kwadratische fit moet < 5% fout zijn op 1D chain
quad_err = abs(results_1d['quadratic']['ratio_inf'] - 1.0)
assert quad_err < 0.05, \
    f"Kwadratische fit fout {quad_err:.4f} > 5%"
print(f"\n  Kwadratische fit fout: {quad_err:.4f} ({quad_err*100:.1f}%)")
print("  PASS\n")

# --- Test 5: Monotonie van extrapolaties ---
print("=== Test 5: Extrapolaties zijn >= p=max ratio ===\n")

for name, res in results_1d.items():
    max_ratio = max(d['ratio'] for d in data_1d)
    # Meeste extrapolaties moeten >= max(ratios) zijn
    # (exponentieel kan iets lager uitvallen)
    if name != 'exponential':
        assert res['ratio_inf'] >= max_ratio - 0.01, \
            f"{name}: {res['ratio_inf']} < max_ratio {max_ratio}"
    print(f"  {name:14s}: r(inf) = {res['ratio_inf']:.6f}")

print("  PASS\n")

# --- Summary ---
print("=" * 60)
print("=== Alle B76 p-ZNE tests geslaagd! ===")
print("=" * 60)
print(f"\nSamenvatting:")
print(f"  4x3 p=1,2:  lineair r(inf)={result['extrapolations']['linear']['ratio_inf']:.4f}")
print(f"  20x1 p=1-3: kwadratisch r(inf)={results_1d['quadratic']['ratio_inf']:.4f} "
      f"(fout {quad_err*100:.1f}%)")
