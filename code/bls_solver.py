#\!/usr/bin/env python3
"""
bls_solver.py - B134 Breakout Local Search for MaxCut

Implementation of Benlic & Hao (2013) Breakout Local Search.
Combines steepest-ascent local search with adaptive perturbation
(random flips + tabu) to escape local optima.

Key ideas:
  1. Steepest-ascent hill climbing with O(1) delta-evaluation
  2. Perturbation strength adapts: weak -> strong as stagnation grows
  3. Tabu list prevents immediate reversal of flipped nodes
  4. Multi-restart with best-across-restarts tracking

References:
  - Benlic & Hao (2013): Breakout Local Search for Max-Clique/Cut
  - Festa et al. (2002): GRASP + path-relinking for MaxCut
"""

import numpy as np
import sys
import os
import time
from collections import defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Core data structures
# ============================================================

def _build_adj_arrays(n_nodes, edges):
    """Build adjacency arrays for O(1) delta evaluation."""
    adj = [[] for _ in range(n_nodes)]
    wt = [[] for _ in range(n_nodes)]
    for e in edges:
        i, j = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        adj[i].append(j)
        wt[i].append(w)
        adj[j].append(i)
        wt[j].append(w)
    # Convert to numpy for speed
    adj_np = [np.array(a, dtype=np.int32) for a in adj]
    wt_np = [np.array(w, dtype=np.float64) for w in wt]
    return adj_np, wt_np


def _compute_cut(x, adj, wt, n_nodes):
    """Compute cut value from assignment array x."""
    cut = 0.0
    for v in range(n_nodes):
        for k in range(len(adj[v])):
            u = adj[v][k]
            if v < u and (x[v] ^ x[u]):
                cut += wt[v][k]
    return cut


def _compute_deltas(x, adj, wt, n_nodes):
    """
    Compute delta[v] = gain from flipping node v.
    delta[v] = sum_u w(v,u) * (same_side - diff_side)
    Positive delta means flipping v improves the cut.
    """
    delta = np.zeros(n_nodes, dtype=np.float64)
    for v in range(n_nodes):
        d = 0.0
        for k in range(len(adj[v])):
            u = adj[v][k]
            w = wt[v][k]
            if x[v] == x[u]:
                d += w  # same side: flipping v cuts this edge
            else:
                d -= w  # diff side: flipping v uncuts this edge
        delta[v] = d
    return delta


def _update_deltas_after_flip(v, x, adj, wt, delta):
    """
    After flipping node v, update deltas for v and all neighbors.
    O(degree(v)) operation.
    """
    # v is already flipped in x
    delta[v] = -delta[v]
    for k in range(len(adj[v])):
        u = adj[v][k]
        w = wt[v][k]
        if x[v] == x[u]:
            # Now same side: flipping u would cut this edge
            delta[u] += 2.0 * w
        else:
            # Now diff side: flipping u would uncut this edge
            delta[u] -= 2.0 * w


# ============================================================
# Local search: steepest ascent
# ============================================================

def _steepest_ascent(x, adj, wt, n_nodes, delta, tabu, iteration):
    """
    One pass of steepest-ascent local search.
    Flips the node with highest positive delta (not tabu).
    Returns number of improving moves made.
    """
    moves = 0
    while True:
        best_v = -1
        best_d = 0.0
        for v in range(n_nodes):
            if delta[v] > best_d and tabu[v] < iteration:
                best_d = delta[v]
                best_v = v
        if best_v < 0:
            break
        # Flip best_v
        x[best_v] = 1 - x[best_v]
        _update_deltas_after_flip(best_v, x, adj, wt, delta)
        moves += 1
    return moves


# ============================================================
# Perturbation strategies
# ============================================================

def _perturb_weak(x, adj, wt, n_nodes, delta, rng, strength):
    """Weak perturbation: flip `strength` random nodes."""
    nodes = rng.choice(n_nodes, size=min(strength, n_nodes), replace=False)
    for v in nodes:
        x[v] = 1 - x[v]
        _update_deltas_after_flip(v, x, adj, wt, delta)
    return nodes


def _perturb_strong(x, adj, wt, n_nodes, delta, rng, strength):
    """
    Strong perturbation: flip `strength` nodes, biased toward
    nodes with least negative delta (most promising flips).
    """
    # Sort by delta descending, take from top half randomly
    order = np.argsort(-delta)
    pool_size = max(strength * 2, n_nodes // 4)
    pool = order[:pool_size]
    chosen = rng.choice(pool, size=min(strength, len(pool)), replace=False)
    for v in chosen:
        x[v] = 1 - x[v]
        _update_deltas_after_flip(v, x, adj, wt, delta)
    return chosen


# ============================================================
# Main BLS algorithm
# ============================================================

def _bls_single(n_nodes, adj, wt, max_iter, max_no_improve, tabu_tenure,
                perturb_min, perturb_max, rng, x_init=None, verbose=False):
    """
    Single run of Breakout Local Search.
    
    Args:
        n_nodes: number of nodes
        adj, wt: adjacency arrays from _build_adj_arrays
        max_iter: max BLS iterations (perturbation cycles)
        max_no_improve: stop after this many iterations without improvement
        tabu_tenure: base tabu tenure (nodes stay tabu for this many iters)
        perturb_min: minimum perturbation strength (# nodes to flip)
        perturb_max: maximum perturbation strength
        rng: numpy random generator
        x_init: initial solution (None for random)
    
    Returns:
        best_x, best_cut, iterations
    """
    # Initialize
    if x_init is not None:
        x = x_init.copy()
    else:
        x = rng.integers(0, 2, size=n_nodes).astype(np.int32)
    
    delta = _compute_deltas(x, adj, wt, n_nodes)
    tabu = np.zeros(n_nodes, dtype=np.int64)
    
    # Initial local search
    _steepest_ascent(x, adj, wt, n_nodes, delta, tabu, 0)
    current_cut = _compute_cut(x, adj, wt, n_nodes)
    
    best_x = x.copy()
    best_cut = current_cut
    
    no_improve = 0
    perturb_strength = perturb_min
    
    for iteration in range(1, max_iter + 1):
        # Adaptive perturbation: grow strength with stagnation
        if no_improve < max_no_improve // 3:
            strength = perturb_min
            flipped = _perturb_weak(x, adj, wt, n_nodes, delta, rng, strength)
        elif no_improve < 2 * max_no_improve // 3:
            strength = (perturb_min + perturb_max) // 2
            flipped = _perturb_weak(x, adj, wt, n_nodes, delta, rng, strength)
        else:
            strength = perturb_max
            flipped = _perturb_strong(x, adj, wt, n_nodes, delta, rng, strength)
        
        # Set tabu on perturbed nodes
        for v in flipped:
            tabu[v] = iteration + tabu_tenure + rng.integers(0, 3)
        
        # Local search from perturbed solution
        _steepest_ascent(x, adj, wt, n_nodes, delta, tabu, iteration)
        current_cut = _compute_cut(x, adj, wt, n_nodes)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_x = x.copy()
            no_improve = 0
            perturb_strength = perturb_min
            if verbose:
                print(f'  BLS iter {iteration}: new best = {best_cut:.1f}')
        else:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            break
    
    return best_x, best_cut, iteration


# ============================================================
# Public API
# ============================================================

def bls_maxcut(n_nodes, edges, n_restarts=10, max_iter=1000,
               max_no_improve=100, tabu_tenure=None,
               perturb_min=None, perturb_max=None,
               time_limit=None, x_init=None,
               seed=None, verbose=False):
    """
    Breakout Local Search for MaxCut.
    
    Args:
        n_nodes: number of nodes
        edges: list of (i, j) or (i, j, w)
        n_restarts: number of independent restarts
        max_iter: max BLS iterations per restart
        max_no_improve: stop restart after this many stagnant iterations
        tabu_tenure: base tabu tenure (default: sqrt(n_nodes))
        perturb_min: min perturbation strength (default: max(1, n//50))
        perturb_max: max perturbation strength (default: max(5, n//10))
        time_limit: wall-clock time limit in seconds (None for no limit)
        x_init: initial solution for first restart (numpy array of 0/1)
        seed: random seed
        verbose: print progress
    
    Returns:
        dict with best_cut, assignment, n_restarts_done, total_iterations, time_s
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    adj, wt = _build_adj_arrays(n_nodes, edges)
    
    # Defaults based on problem size
    if tabu_tenure is None:
        tabu_tenure = max(5, int(np.sqrt(n_nodes)))
    if perturb_min is None:
        perturb_min = max(1, n_nodes // 50)
    if perturb_max is None:
        perturb_max = max(5, n_nodes // 10)
    
    global_best_x = None
    global_best_cut = -1.0
    total_iters = 0
    restarts_done = 0
    
    for r in range(n_restarts):
        if time_limit is not None and (time.time() - t0) >= time_limit:
            break
        
        # Use provided init for first restart only
        init = x_init if (r == 0 and x_init is not None) else None
        
        bx, bc, iters = _bls_single(
            n_nodes, adj, wt, max_iter, max_no_improve,
            tabu_tenure, perturb_min, perturb_max, rng,
            x_init=init, verbose=verbose
        )
        total_iters += iters
        restarts_done += 1
        
        if bc > global_best_cut:
            global_best_cut = bc
            global_best_x = bx.copy()
            if verbose:
                t = time.time() - t0
                print(f'BLS restart {r+1}: new global best = {bc:.1f} (t={t:.2f}s)')
    
    elapsed = time.time() - t0
    assignment = {i: int(global_best_x[i]) for i in range(n_nodes)}
    
    return {
        'best_cut': global_best_cut,
        'assignment': assignment,
        'n_restarts_done': restarts_done,
        'total_iterations': total_iters,
        'time_s': elapsed,
    }


def bls_maxcut_grid(Lx, Ly, triangular=False, **kwargs):
    """Convenience wrapper for grid graphs."""
    n_nodes = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            if x + 1 < Lx:
                edges.append((node, (x + 1) * Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((node, x * Ly + y + 1, 1.0))
            if triangular and x + 1 < Lx and y + 1 < Ly:
                edges.append((node, (x + 1) * Ly + y + 1, 1.0))
    return bls_maxcut(n_nodes, edges, **kwargs)


# ============================================================
# Random graph generators
# ============================================================

def random_3regular(n, seed=None):
    """Generate random 3-regular graph."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        stubs = []
        for v in range(n):
            stubs.extend([v, v, v])
        stubs = np.array(stubs)
        rng.shuffle(stubs)
        edges_set = set()
        ok = True
        for k in range(0, len(stubs), 2):
            u, v = int(stubs[k]), int(stubs[k+1])
            if u == v or (min(u,v), max(u,v)) in edges_set:
                ok = False
                break
            edges_set.add((min(u,v), max(u,v)))
        if ok:
            return n, [(u, v, 1.0) for u, v in edges_set]
    raise RuntimeError("Failed to generate 3-regular graph")


def random_erdos_renyi(n, p=0.5, seed=None):
    """Generate Erdos-Renyi random graph G(n, p)."""
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j, 1.0))
    return n, edges


# ============================================================
# Gset loader
# ============================================================

def load_gset(filepath):
    """Load a Gset-format graph file."""
    edges = []
    n_nodes = 0
    with open(filepath) as f:
        header = f.readline().split()
        n_nodes = int(header[0])
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                i = int(parts[0]) - 1  # 1-indexed to 0-indexed
                j = int(parts[1]) - 1
                w = float(parts[2]) if len(parts) > 2 else 1.0
                edges.append((i, j, w))
    return n_nodes, edges


if __name__ == '__main__':
    print('=== B134 Breakout Local Search Demo ===\n')

    # Small tests
    print('--- Grid graphs ---')
    for Lx, Ly in [(4, 2), (5, 3), (10, 4), (20, 5)]:
        r = bls_maxcut_grid(Lx, Ly, n_restarts=5, seed=42, verbose=False)
        n = Lx * Ly
        t = r['time_s']
        c = r['best_cut']
        print(f'  {Lx}x{Ly} (n={n}): cut={c:.0f}, time={t:.4f}s')

    print('\n--- Triangular grids (non-bipartite) ---')
    for Lx, Ly in [(3, 2), (4, 3), (5, 4), (10, 5)]:
        r = bls_maxcut_grid(Lx, Ly, triangular=True, n_restarts=10, seed=42)
        n = Lx * Ly
        t = r['time_s']
        c = r['best_cut']
        print(f'  tri {Lx}x{Ly} (n={n}): cut={c:.0f}, time={t:.4f}s')

    print('\n--- Random 3-regular graphs ---')
    for n in [50, 100, 200, 500]:
        nn, edges = random_3regular(n, seed=n)
        r = bls_maxcut(nn, edges, n_restarts=10, seed=42)
        t = r['time_s']
        c = r['best_cut']
        ri = r['n_restarts_done']
        it = r['total_iterations']
        print(f'  3-reg n={n}: cut={c:.0f}, restarts={ri}, iters={it}, time={t:.2f}s')

    print('\n--- Random Erdos-Renyi G(n, 0.5) ---')
    for n in [50, 100, 200]:
        nn, edges = random_erdos_renyi(n, p=0.5, seed=n)
        r = bls_maxcut(nn, edges, n_restarts=10, max_iter=500, seed=42)
        t = r['time_s']
        c = r['best_cut']
        m = len(edges)
        print(f'  ER n={n} ({m} edges): cut={c:.0f}, time={t:.2f}s')

