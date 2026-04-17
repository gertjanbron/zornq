#\!/usr/bin/env python3
"""
test_pfaffian_oracle.py - Tests for B100 Pfaffian Oracle
"""

import numpy as np
import sys
import os
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pfaffian_oracle import (
    _pfaffian, _is_bipartite, _is_planar_simple, _detect_grid,
    _build_adjacency, _brute_force_maxcut, _kasteleyn_orientation,
    _eval_cut, _local_search_dict,
    pfaffian_maxcut, pfaffian_maxcut_grid, verify_pfaffian,
)


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')


# ========== Pfaffian tests ==========
print('=== Pfaffian computation ===')

# 2x2 skew-symmetric
A2 = np.array([[0.0, 3.0], [-3.0, 0.0]])
check('Pf 2x2 = A[0,1]', abs(_pfaffian(A2) - 3.0) < 1e-12)

check('Pf 0x0 = 1', _pfaffian(np.zeros((0, 0))) == 1.0)

check('Pf odd dim = 0', _pfaffian(np.zeros((3, 3))) == 0.0)

Z4 = np.zeros((4, 4))
check('Pf zero 4x4 = 0', abs(_pfaffian(Z4)) < 1e-12)

for n in [4, 6, 8, 10, 12, 20]:
    rng = np.random.default_rng(n * 7)
    A = rng.standard_normal((n, n))
    A = A - A.T
    pf = _pfaffian(A)
    det_A = np.linalg.det(A)
    rel_err = abs(pf**2 - det_A) / max(abs(det_A), 1e-15)
    check(f'Pf^2==det n={n} (rel_err={rel_err:.2e})', rel_err < 1e-8)

print('\n=== Pfaffian sign ===')
# Swapping two rows/cols should negate Pfaffian
rng = np.random.default_rng(99)
A = rng.standard_normal((6, 6))
A = A - A.T
pf1 = _pfaffian(A)
B = A.copy()
B[[0, 2]] = B[[2, 0]]
B[:, [0, 2]] = B[:, [2, 0]]
pf2 = _pfaffian(B)
check('Pf sign flip on row/col swap', abs(pf1 + pf2) < 1e-8)

print('\n=== Graph utilities ===')

adj_path = {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1}}
bip, parts = _is_bipartite(3, adj_path)
check('Path P3 is bipartite', bip)

adj_tri = {0: {1: 1, 2: 1}, 1: {0: 1, 2: 1}, 2: {0: 1, 1: 1}}
bip2, _ = _is_bipartite(3, adj_tri)
check('Triangle is not bipartite', not bip2)

edges_4x3 = []
for x in range(4):
    for y in range(3):
        node = x * 3 + y
        if x + 1 < 4:
            edges_4x3.append((node, (x+1)*3+y, 1.0))
        if y + 1 < 3:
            edges_4x3.append((node, x*3+y+1, 1.0))
adj_4x3 = _build_adjacency(12, edges_4x3)
grid = _detect_grid(12, adj_4x3)
check('Detect 4x3 grid', grid == (4, 3))

k5_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
check('K5 detected as non-planar', not _is_planar_simple(5, k5_edges))

tree_edges = [(0,1), (1,2), (2,3), (3,4), (4,5)]
check('Tree is planar', _is_planar_simple(6, tree_edges))

print('\n=== Brute force MaxCut ===')

k3 = [(0,1,1.0), (1,2,1.0), (0,2,1.0)]
cut3, _ = _brute_force_maxcut(3, k3)
check('K3 MaxCut = 2', abs(cut3 - 2.0) < 1e-9)

k4 = [(i,j,1.0) for i in range(4) for j in range(i+1,4)]
cut4, _ = _brute_force_maxcut(4, k4)
check('K4 MaxCut = 4', abs(cut4 - 4.0) < 1e-9)

g32 = []
for x in range(3):
    for y in range(2):
        n = x*2+y
        if x+1 < 3: g32.append((n, (x+1)*2+y, 1.0))
        if y+1 < 2: g32.append((n, x*2+y+1, 1.0))
cut32, _ = _brute_force_maxcut(6, g32)
check('3x2 grid MaxCut = 7', abs(cut32 - 7.0) < 1e-9)

cut_w, _ = _brute_force_maxcut(2, [(0, 1, 5.0)])
check('Weighted edge MaxCut = 5', abs(cut_w - 5.0) < 1e-9)

print('\n=== Local search ===')
init = {0: 0, 1: 0, 2: 0}
best, best_cut = _local_search_dict(init, 3, k3)
check('LS on K3 finds MaxCut=2', abs(best_cut - 2.0) < 1e-9)

print('\n=== Kasteleyn orientation ===')
# Kasteleyn matrix should be skew-symmetric
K = _kasteleyn_orientation(12, edges_4x3, adj_4x3)
check('Kasteleyn matrix is skew-symmetric', np.allclose(K, -K.T))

print('\n=== pfaffian_maxcut API ===')

r = pfaffian_maxcut_grid(4, 2)
check('4x2 grid exact=True', r['exact'])
check('4x2 grid MaxCut=10', abs(r['best_cut'] - 10.0) < 1e-9)
check('4x2 grid method=bipartite', r['method'] == 'bipartite')

r = pfaffian_maxcut_grid(10, 4)
check('10x4 grid exact=True', r['exact'])
check('10x4 grid MaxCut=66', abs(r['best_cut'] - 66.0) < 1e-9)

r = pfaffian_maxcut_grid(3, 2, triangular=True)
check('tri 3x2 exact=True', r['exact'])
check('tri 3x2 MaxCut=7', abs(r['best_cut'] - 7.0) < 1e-9)

r = pfaffian_maxcut(5, k5_edges)
check('K5 exact=True (brute force)', r['exact'])
check('K5 MaxCut=6', abs(r['best_cut'] - 6.0) < 1e-9)

a = r['assignment']
check('K5 assignment has all nodes', len(a) == 5)
cut_val = _eval_cut(a, k5_edges)
check('K5 assignment achieves MaxCut', abs(cut_val - r['best_cut']) < 1e-9)

t0 = time.time()
r = pfaffian_maxcut_grid(100, 50)
dt = time.time() - t0
check('100x50 grid fast (<0.1s)', dt < 0.1)
check('100x50 grid exact', r['exact'])
check('100x50 grid MaxCut=9850', abs(r['best_cut'] - 9850.0) < 1e-9)

print('\n=== Petersen graph (non-planar) ===')
pet_edges = [
    (0,1), (1,2), (2,3), (3,4), (4,0),  # outer
    (5,7), (7,9), (9,6), (6,8), (8,5),  # inner (pentagram)
    (0,5), (1,6), (2,7), (3,8), (4,9),  # spokes
]
r = pfaffian_maxcut(10, pet_edges)
check('Petersen exact=True (brute force n=10)', r['exact'])
check('Petersen MaxCut=12', abs(r['best_cut'] - 12.0) < 1e-9)

print('\n=== verify_pfaffian ===')
check('verify_pfaffian(4)', verify_pfaffian(4))
check('verify_pfaffian(12)', verify_pfaffian(12))

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(1 if failed else 0)

