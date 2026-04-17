#\!/usr/bin/env python3
"""Batch part 2 - medium graphs via Heisenberg/Lightcone engines."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gset_loader import load_graph, make_random_regular, make_erdos_renyi
from auto_planner import ZornSolver

def wg_to_edges(wg):
    edges, seen = [], set()
    for u, v, w in wg.edges():
        key = (min(u,v), max(u,v))
        if key not in seen:
            edges.append(key)
            seen.add(key)
    return wg.n_nodes, edges

# Medium: n=20-60, uses heisenberg/lightcone
suite = []

# Dodecahedron (n=20, 3-reg, non-bipartite)
g, bks, info = load_graph('dodecahedron', seed=42)
n, edges = wg_to_edges(g)
suite.append(('dodecahedron', n, edges, bks))

# Random 3-regular
for nn in [14, 16, 20]:
    g, bks, info = load_graph(f'reg3_{nn}', seed=42)
    n, edges = wg_to_edges(g)
    suite.append((f'reg3_{nn}', n, edges, bks))

# Grid 8x3 (heisenberg)
g, bks, info = load_graph('grid_8x3', seed=42)
n, edges = wg_to_edges(g)
suite.append(('grid_8x3', n, edges, bks))

# Torus 6x4 (lightcone)
g, bks, info = load_graph('torus_6x4', seed=42)
n, edges = wg_to_edges(g)
suite.append(('torus_6x4', n, edges, bks))

# ER 12
g, bks = make_erdos_renyi(12, p=0.3, seed=42)
n, edges = wg_to_edges(g)
suite.append(('ER_12', n, edges, bks))

solver = ZornSolver(chi_budget=8, gpu=False, verbose=False)

print(f"{'Graph':>14} {'n':>3} {'m':>3} {'BKS':>5} | {'Cut':>5} {'%BKS':>6} {'Method':>20} {'t':>5}")
print('-' * 75)

total_t = 0
for name, n, edges, bks in suite:
    m = len(edges)
    try:
        res = solver.solve(n, edges, p=1)
        total_t += res.wall_time
        bs = str(bks) if bks else '?'
        ap = f'{res.cut_value/bks:.1%}' if bks and bks > 0 else '-'
        ls = '+LS' if 'local_search' in ','.join(res.notes) else ''
        print(f'{name:>14} {n:>3} {m:>3} {bs:>5} | {res.cut_value:>5.0f} {ap:>6} {res.method:>20} {res.wall_time:>4.1f}s {ls}')
        sys.stdout.flush()
        gc.collect()
    except Exception as e:
        print(f'{name:>14} {n:>3} {m:>3} ERROR: {str(e)[:40]}')
        sys.stdout.flush()

print(f'\nTotal: {total_t:.1f}s')
