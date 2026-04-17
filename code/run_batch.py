#\!/usr/bin/env python3
"""Quick batch run - small graphs only."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gset_loader import load_graph
from auto_planner import ZornSolver

def wg_to_edges(wg):
    edges, seen = [], set()
    for u, v, w in wg.edges():
        key = (min(u,v), max(u,v))
        if key not in seen:
            edges.append(key)
            seen.add(key)
    return wg.n_nodes, edges

names = [
    'petersen', 'cube', 'K5', 'K8', 'K10',
    'cycle_8', 'cycle_11', 'cycle_20',
    'grid_4x3', 'grid_6x3',
    'tri_4x3', 'tri_6x3',
    'torus_4x4',
]

solver = ZornSolver(chi_budget=16, gpu=False, verbose=False)

print(f"{'Graph':>14} {'n':>3} {'m':>3} {'BKS':>5} | {'Cut':>5} {'%BKS':>6} {'Method':>16} {'t':>5}")
print('-' * 70)

total_t = 0
for name in names:
    try:
        g, bks, info = load_graph(name, seed=42)
        n, edges = wg_to_edges(g)
        res = solver.solve(n, edges, p=1)
        total_t += res.wall_time
        bs = str(bks) if bks else '?'
        ap = f'{res.cut_value/bks:.1%}' if bks and bks > 0 else '-'
        ls = '+LS' if 'local_search' in ','.join(res.notes) else ''
        print(f'{name:>14} {n:>3} {len(edges):>3} {bs:>5} | {res.cut_value:>5.0f} {ap:>6} {res.method:>16} {res.wall_time:>4.1f}s {ls}')
        sys.stdout.flush()
        gc.collect()
    except Exception as e:
        print(f'{name:>14} ERROR: {e}')
        sys.stdout.flush()

print(f'\nTotal: {total_t:.1f}s')
