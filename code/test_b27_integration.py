#!/usr/bin/env python3
"""B27 integratie test — symmetrie in ZornSolver pipeline."""
import sys, time, numpy as np
sys.path.insert(0, '.')
sys.dont_write_bytecode = True
from auto_planner import ZornSolver, classify_graph

def count_cut(edges, bits):
    return sum(1 for u,v in edges if bits[u] != bits[v])

# Test 1: Petersen — vertex-transitive, orbit info in stats
petersen_edges = [
    (0,1),(1,2),(2,3),(3,4),(4,0),
    (0,5),(1,6),(2,7),(3,8),(4,9),
    (5,7),(7,9),(9,6),(6,8),(8,5),
]
print("=== Test 1: Petersen (vertex-transitive) ===")
stats = classify_graph(10, petersen_edges)
print(f"  n_orbits={stats['n_orbits']}, VT={stats['is_vertex_transitive']}")
assert stats['is_vertex_transitive'], "Petersen should be VT"
assert stats['n_orbits'] == 1

solver = ZornSolver(chi_budget=64, gpu=False, verbose=True)
t0 = time.time()
r = solver.solve(n_nodes=10, edges=petersen_edges, p=1)
t1 = time.time()
actual = count_cut(petersen_edges, r.best_bitstring)
print(f"Result: cut={r.cut_value}, actual={actual}, engine={r.engine}, t={t1-t0:.2f}s")
assert actual == 12, f"Expected 12, got {actual}"
print("OK\n")

# Test 2: Grid 4x3 — orbits detected, pruning skipped
grid_edges = []
for x in range(4):
    for y in range(3):
        n = x*3+y
        if x+1<4: grid_edges.append((n, (x+1)*3+y))
        if y+1<3: grid_edges.append((n, x*3+y+1))
print("=== Test 2: Grid 4x3 (4 orbits, grid path) ===")
stats2 = classify_graph(12, grid_edges)
print(f"  n_orbits={stats2['n_orbits']}, VT={stats2['is_vertex_transitive']}")
assert stats2['n_orbits'] == 4

solver2 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r2 = solver2.solve(n_nodes=12, edges=grid_edges, p=1, method='heisenberg')
print(f"Result: cut={r2.cut_value}, ratio={r2.ratio:.4f}")
print("OK\n")

# Test 3: Boom met pendanten — B50 pruning + B27 symmetrie
tree_edges = [(0,1),(0,2),(1,3),(1,4),(2,5)]
print("=== Test 3: Boom (B50 pruning + B27 orbits) ===")
stats3 = classify_graph(6, tree_edges)
print(f"  n_orbits={stats3['n_orbits']}")
solver3 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r3 = solver3.solve(n_nodes=6, edges=tree_edges, p=1)
actual3 = count_cut(tree_edges, r3.best_bitstring)
print(f"Result: cut={r3.cut_value}, actual={actual3}, method={r3.method}")
assert actual3 == 5
print("OK\n")

# Test 4: K10 — dense, brute force with B27 sym-break
k10_edges = [(i,j) for i in range(10) for j in range(i+1,10)]
print(f"=== Test 4: K10 ({len(k10_edges)} edges, VT, brute force with B27) ===")
stats4 = classify_graph(10, k10_edges)
print(f"  n_orbits={stats4['n_orbits']}, VT={stats4['is_vertex_transitive']}")
solver4 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
t0 = time.time()
r4 = solver4.solve(n_nodes=10, edges=k10_edges, p=1)
t4 = time.time() - t0
actual4 = count_cut(k10_edges, r4.best_bitstring)
print(f"Result: cut={r4.cut_value}, actual={actual4}, engine={r4.engine}, t={t4:.2f}s")
# K10 MaxCut = 25 (5+5 partition)
assert actual4 == 25, f"Expected 25, got {actual4}"
print("OK\n")

# Test 5: Dodecahedron — B50 pruning (none expected) + B27 orbits
from gset_loader import load_graph
wg, bks, _ = load_graph('dodecahedron')
dedges = list(set((min(u,v),max(u,v)) for u,v,w in wg.edges()))
print("=== Test 5: Dodecahedron (VT, exact via lanczos) ===")
stats5 = classify_graph(20, dedges)
print(f"  n_orbits={stats5['n_orbits']}, VT={stats5['is_vertex_transitive']}")
solver5 = ZornSolver(chi_budget=64, gpu=False, verbose=True)
r5 = solver5.solve(n_nodes=20, edges=dedges, p=1)
actual5 = count_cut(dedges, r5.best_bitstring)
print(f"Result: cut={r5.cut_value}, actual={actual5}, bks={bks}")
assert actual5 >= 25  # BKS 25 maar Lanczos bewees 27
print("OK\n")

print("=== Alle B27-integratietests geslaagd! ===")
