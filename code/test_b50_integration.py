#!/usr/bin/env python3
"""B50 integratie test."""
import sys, time, random, numpy as np
sys.path.insert(0, '.')
sys.dont_write_bytecode = True
from auto_planner import ZornSolver

def count_cut(edges, bits):
    return sum(1 for u,v in edges if bits[u] != bits[v])

# Test 1: Boom (volledig reduceerbaar door B50)
edges_tree = [(0,1), (0,2), (1,3), (1,4), (2,5)]
print("=== Test 1: Boom n=6 (volledig reduceerbaar) ===")
solver = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r = solver.solve(n_nodes=6, edges=edges_tree, p=1)
actual = count_cut(edges_tree, r.best_bitstring)
print(f"Result: cut={r.cut_value}, actual={actual}, method={r.method}")
assert actual == 5, f"Expected 5, got {actual}"
print("OK\n")

# Test 2: Vierkant met pendanten
edges_sq = [(0,1),(1,2),(2,3),(3,0),(0,4),(2,5),(2,6)]
print("=== Test 2: Vierkant + 3 pendanten ===")
solver2 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r2 = solver2.solve(n_nodes=7, edges=edges_sq, p=1)
actual2 = count_cut(edges_sq, r2.best_bitstring)
print(f"Result: cut={r2.cut_value}, actual={actual2}, notes={r2.notes}")
assert actual2 == 7, f"Expected 7, got {actual2}"
print("OK\n")

# Test 3: Petersen (3-regular, geen pruning)
from gset_loader import load_graph
wg, bks, _ = load_graph('petersen')
edges_p = list(set((min(u,v),max(u,v)) for u,v,w in wg.edges()))
print("=== Test 3: Petersen (geen pruning verwacht) ===")
solver3 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r3 = solver3.solve(n_nodes=wg.n_nodes, edges=edges_p, p=1)
print(f"Result: cut={r3.cut_value}, bks={bks}\nOK\n")

# Test 4: Grid 4x3 (pruning skipped for grids)
Lx, Ly = 4, 3
edges_g = []
for x in range(Lx):
    for y in range(Ly):
        node = x*Ly+y
        if x+1<Lx: edges_g.append((node,(x+1)*Ly+y))
        if y+1<Ly: edges_g.append((node,x*Ly+y+1))
print("=== Test 4: Grid 4x3 (pruning skipped) ===")
solver4 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r4 = solver4.solve(n_nodes=12, edges=edges_g, p=1, method='heisenberg')
print(f"Result: cut={r4.cut_value}, ratio={r4.ratio:.4f}\nOK\n")

# Test 5: Sparse random graph met pendanten
random.seed(123)
n = 20
erdos_edges = set()
for i in range(n):
    for j in range(i+1, n):
        if random.random() < 0.12:
            erdos_edges.add((i, j))
edges_er = list(erdos_edges)
from collections import Counter
deg = Counter()
for u,v in edges_er:
    deg[u] += 1; deg[v] += 1
n_leaves = sum(1 for d in deg.values() if d == 1)
n_isolates = n - len(deg)
print(f"=== Test 5: ER n=20 p=0.12 ({len(edges_er)} edges, {n_leaves} leaves, {n_isolates} isolates) ===")
solver5 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r5 = solver5.solve(n_nodes=n, edges=edges_er, p=1)
if r5.best_bitstring is not None:
    actual5 = count_cut(edges_er, r5.best_bitstring)
    print(f"Result: cut={r5.cut_value}, actual={actual5}")
print("OK\n")

print("=== Alle B50-integratietests geslaagd! ===")
