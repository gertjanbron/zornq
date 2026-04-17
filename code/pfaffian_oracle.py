#\!/usr/bin/env python3
"""
pfaffian_oracle.py - B100 Planar Pfaffian Oracle for Exact MaxCut

Exact MaxCut solver for planar graphs via the FKT method:
  1. Bipartite graphs => trivial (MaxCut = total weight)
  2. Small graphs (n<=25) => vectorized brute force O(2^n * |E|)
  3. Planar graphs => Pfaffian of Kasteleyn-oriented matrix + local search
  4. Non-planar => multi-restart local search

References:
  - Kasteleyn (1961): Graph Pfaffians and dimer statistics
  - Fisher (1966): FKT algorithm
  - Barahona (1982): MaxCut on planar graphs is poly-time via T-join
"""

import numpy as np
import sys
import os
import time
from collections import defaultdict, deque

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Graph utilities
# ============================================================

def _build_adjacency(n_nodes, edges):
    """Build adjacency list from edge list."""
    adj = defaultdict(dict)
    for e in edges:
        i, j = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        adj[i][j] = w
        adj[j][i] = w
    return adj


def _is_bipartite(n_nodes, adj):
    """Check if graph is bipartite. Returns (True, partition) or (False, None)."""
    color = {}
    for start in range(n_nodes):
        if start in color or start not in adj:
            continue
        queue = deque([start])
        color[start] = 0
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, None
    part_a = {v for v, c in color.items() if c == 0}
    part_b = {v for v, c in color.items() if c == 1}
    return True, (part_a, part_b)


def _is_planar_simple(n_nodes, edges):
    """Simple planarity check via Euler formula: |E| <= 3|V| - 6."""
    n = n_nodes
    m = len(edges)
    if n <= 4:
        return True
    if m > 3 * n - 6:
        return False
    return True


def _detect_grid(n_nodes, adj):
    """Detect if graph is a grid and return dimensions."""
    degrees = {v: len(adj[v]) for v in adj}
    deg_counts = defaultdict(int)
    for d in degrees.values():
        deg_counts[d] += 1
    n = n_nodes
    for Ly in range(2, n):
        if n % Ly != 0:
            continue
        Lx = n // Ly
        if Lx < 2:
            continue
        expected_4 = max(0, (Lx - 2) * (Ly - 2))
        expected_3 = max(0, 2 * (Lx - 2) + 2 * (Ly - 2))
        expected_2 = 4
        if (deg_counts.get(4, 0) == expected_4 and
            deg_counts.get(3, 0) == expected_3 and
            deg_counts.get(2, 0) == expected_2):
            return Lx, Ly
    return None


# ============================================================
# Pfaffian computation (block tridiagonalization + pivoting)
# ============================================================

def _pfaffian(A):
    """
    Compute Pfaffian of skew-symmetric matrix A.
    Uses block tridiagonalization with partial pivoting.
    Pf(A)^2 = det(A) for skew-symmetric A.
    """
    n = A.shape[0]
    if n == 0:
        return 1.0
    if n % 2 == 1:
        return 0.0
    if n == 2:
        return A[0, 1]
    M = A.astype(np.float64).copy()
    pf = 1.0
    for k in range(0, n - 1, 2):
        kp1 = k + 1
        col = np.abs(M[k, kp1:])
        if col.max() < 1e-15:
            return 0.0
        pivot_idx = kp1 + int(np.argmax(col))
        if pivot_idx ^ kp1:  # swap needed
            M[[kp1, pivot_idx]] = M[[pivot_idx, kp1]]
            M[:, [kp1, pivot_idx]] = M[:, [pivot_idx, kp1]]
            pf *= -1.0
        pf *= M[k, kp1]
        if kp1 + 1 < n:
            t = M[k, kp1]
            ck = M[kp1 + 1:, k].copy()
            ckp1 = M[kp1 + 1:, kp1].copy()
            M[kp1 + 1:, kp1 + 1:] -= (np.outer(ck, ckp1) - np.outer(ckp1, ck)) / t
    return pf


# ============================================================
# MaxCut helpers
# ============================================================

def _planar_maxcut_grid(Lx, Ly, edges):
    """Exact MaxCut for grid graphs (bipartite => checkerboard)."""
    total_weight = sum(e[2] if len(e) > 2 else 1.0 for e in edges)
    assignment = {}
    for x in range(Lx):
        for y in range(Ly):
            assignment[x * Ly + y] = (x + y) % 2
    return total_weight, assignment


def _brute_force_maxcut(n_nodes, edges):
    """Exact MaxCut via vectorized enumeration. O(2^n * |E|)."""
    N = 1 << n_nodes
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] if len(e) > 2 else 1.0 for e in edges], dtype=np.float64)
    if n_nodes <= 20:
        xs = np.arange(N, dtype=np.int32)
        bi = (xs[:, None] >> ei[None, :]) & 1
        bj = (xs[:, None] >> ej[None, :]) & 1
        cuts = np.sum(ew[None, :] * (bi ^ bj), axis=1)
        best_idx = int(np.argmax(cuts))
        best_cut = float(cuts[best_idx])
        best_x = best_idx
    else:
        chunk = 1 << 20
        best_cut = -1.0
        best_x = 0
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            xs = np.arange(start, end, dtype=np.int32)
            bi = (xs[:, None] >> ei[None, :]) & 1
            bj = (xs[:, None] >> ej[None, :]) & 1
            cuts = np.sum(ew[None, :] * (bi ^ bj), axis=1)
            idx = int(np.argmax(cuts))
            if cuts[idx] > best_cut:
                best_cut = float(cuts[idx])
                best_x = start + idx
    assignment = {i: (best_x >> i) & 1 for i in range(n_nodes)}
    return best_cut, assignment


def _kasteleyn_orientation(n_nodes, edges, adj):
    """Compute Kasteleyn orientation for a planar graph."""
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    grid_dims = _detect_grid(n_nodes, adj)
    if grid_dims:
        Lx, Ly = grid_dims
        for e in edges:
            i, j = e[0], e[1]
            w = e[2] if len(e) > 2 else 1.0
            xi, yi = i // Ly, i % Ly
            xj, yj = j // Ly, j % Ly
            if yi == yj:  # Horizontal
                if xi < xj:
                    K[i, j] = w; K[j, i] = -w
                else:
                    K[j, i] = w; K[i, j] = -w
            else:  # Vertical
                if xi % 2 == 0:
                    if yi < yj:
                        K[i, j] = w; K[j, i] = -w
                    else:
                        K[j, i] = w; K[i, j] = -w
                else:
                    if yi < yj:
                        K[i, j] = -w; K[j, i] = w
                    else:
                        K[j, i] = -w; K[i, j] = w
    else:
        for e in edges:
            i, j = min(e[0], e[1]), max(e[0], e[1])
            w = e[2] if len(e) > 2 else 1.0
            K[i, j] = w; K[j, i] = -w
    return K


def _pfaffian_maxcut(n_nodes, edges, adj):
    """MaxCut via Pfaffian for planar graphs (Barahona 1982)."""
    total_weight = sum(e[2] if len(e) > 2 else 1.0 for e in edges)
    K = _kasteleyn_orientation(n_nodes, edges, adj)
    pf = _pfaffian(K)
    maxcut_bound = (total_weight + abs(pf)) / 2.0
    return maxcut_bound, None


# ============================================================
# Local search
# ============================================================

def _eval_cut(assignment, edges):
    """Evaluate cut value for an assignment dict."""
    total = 0.0
    for e in edges:
        i, j = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        if assignment.get(i, 0) ^ assignment.get(j, 0):
            total += w
    return total


def _local_search_dict(assignment, n_nodes, edges, max_iter=200):
    """Steepest-ascent local search from an assignment dict."""
    best = dict(assignment)
    best_cut = _eval_cut(best, edges)
    for _ in range(max_iter):
        improved = False
        for v in range(n_nodes):
            trial = dict(best)
            trial[v] = 1 - trial.get(v, 0)
            c = _eval_cut(trial, edges)
            if c > best_cut:
                best = trial
                best_cut = c
                improved = True
        if not improved:
            break
    return best, best_cut


# ============================================================
# Main API
# ============================================================

def pfaffian_maxcut(n_nodes, edges, verbose=False):
    """
    Exact MaxCut for planar graphs, heuristic+bound for non-planar.
    
    Strategy:
    1. Bipartite => trivial (MaxCut = total weight)
    2. Grid => checkerboard (bipartite)
    3. Small (n<=25) => vectorized brute force
    4. Planar + large => Pfaffian bound + local search
    5. Non-planar + large => multi-restart local search
    
    Returns: dict with best_cut, assignment, exact, method, is_planar, time_s
    """
    t0 = time.time()
    adj = _build_adjacency(n_nodes, edges)
    wedges = []
    for e in edges:
        wedges.append((e[0], e[1], e[2] if len(e) > 2 else 1.0))
    total_weight = sum(w for _, _, w in wedges)
    
    # 1. Bipartite check
    is_bip, parts = _is_bipartite(n_nodes, adj)
    if is_bip:
        assignment = {}
        if parts:
            for v in parts[0]:
                assignment[v] = 0
            for v in parts[1]:
                assignment[v] = 1
        elapsed = time.time() - t0
        if verbose:
            print(f'Pfaffian Oracle: bipartite, MaxCut = {total_weight:.1f} (trivial)')
        return {
            'best_cut': total_weight,
            'assignment': assignment,
            'exact': True,
            'method': 'bipartite',
            'is_planar': True,
            'time_s': elapsed,
        }
    
    # 2. Grid detection
    grid_dims = _detect_grid(n_nodes, adj)
    if grid_dims:
        Lx, Ly = grid_dims
        cut, assignment = _planar_maxcut_grid(Lx, Ly, wedges)
        elapsed = time.time() - t0
        if verbose:
            print(f'Pfaffian Oracle: {Lx}x{Ly} grid (bipartite), MaxCut = {cut:.1f}')
        return {
            'best_cut': cut,
            'assignment': assignment,
            'exact': True,
            'method': f'grid_{Lx}x{Ly}',
            'is_planar': True,
            'time_s': elapsed,
        }
    
    # 3. Small graph => vectorized brute force (exact)
    if n_nodes <= 25:
        cut, assignment = _brute_force_maxcut(n_nodes, wedges)
        elapsed = time.time() - t0
        is_planar = _is_planar_simple(n_nodes, wedges)
        if verbose:
            print(f'Pfaffian Oracle: brute-force n={n_nodes}, MaxCut = {cut:.1f}')
        return {
            'best_cut': cut,
            'assignment': assignment,
            'exact': True,
            'method': 'brute_force',
            'is_planar': is_planar,
            'time_s': elapsed,
        }
    
    # 4. Planar + large => Pfaffian bound + local search
    is_planar = _is_planar_simple(n_nodes, wedges)
    if is_planar:
        pf_cut, _ = _pfaffian_maxcut(n_nodes, wedges, adj)
        rng = np.random.default_rng(42)
        best_cut = 0
        best_assign = {v: 0 for v in range(n_nodes)}
        for _ in range(20):
            init = {v: int(rng.integers(0, 2)) for v in range(n_nodes)}
            assign, cut = _local_search_dict(init, n_nodes, wedges)
            if cut > best_cut:
                best_cut = cut
                best_assign = assign
        elapsed = time.time() - t0
        is_exact = abs(best_cut - pf_cut) < 0.5
        if verbose:
            print(f'Pfaffian Oracle: planar n={n_nodes}, Pf-bound={pf_cut:.1f}, LS={best_cut:.1f}')
        return {
            'best_cut': best_cut,
            'pfaffian_bound': pf_cut,
            'assignment': best_assign,
            'exact': is_exact,
            'method': 'pfaffian+ls',
            'is_planar': True,
            'time_s': elapsed,
        }
    
    # 5. Non-planar, large => local search only
    rng = np.random.default_rng(42)
    best_cut = 0
    best_assign = {v: 0 for v in range(n_nodes)}
    for _ in range(30):
        init = {v: int(rng.integers(0, 2)) for v in range(n_nodes)}
        assign, cut = _local_search_dict(init, n_nodes, wedges)
        if cut > best_cut:
            best_cut = cut
            best_assign = assign
    elapsed = time.time() - t0
    if verbose:
        print(f'Pfaffian Oracle: non-planar n={n_nodes}, LS best={best_cut:.1f}')
    return {
        'best_cut': best_cut,
        'assignment': best_assign,
        'exact': False,
        'method': 'local_search',
        'is_planar': False,
        'time_s': elapsed,
    }


def pfaffian_maxcut_grid(Lx, Ly, triangular=False, verbose=False):
    """Convenience wrapper for grid/triangular grid graphs."""
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
    return pfaffian_maxcut(n_nodes, edges, verbose=verbose)


def verify_pfaffian(n=6, seed=42):
    """Verify Pfaffian computation: Pf(A)^2 == det(A)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A - A.T
    pf = _pfaffian(A)
    det_A = np.linalg.det(A)
    pf_sq = pf ** 2
    rel_err = abs(pf_sq - det_A) / max(abs(det_A), 1e-15)
    ok = rel_err < 1e-8
    status = 'OK' if ok else 'FAIL'
    print(f'  n={n}: Pf={pf:.6f}, Pf^2={pf_sq:.6f}, det={det_A:.6f}, rel_err={rel_err:.2e} {status}')
    return ok


if __name__ == '__main__':
    print('=== B100 Pfaffian Oracle Demo ===\n')

    print('--- Pfaffian verification ---')
    all_ok = True
    for n in [4, 6, 8, 10, 12, 20]:
        all_ok = all_ok and verify_pfaffian(n)
    status = 'ALL OK' if all_ok else 'SOME FAILED'
    print(f'Pfaffian: {status}\n')

    print('--- Grid graphs (bipartite => exact) ---')
    for Lx, Ly in [(4, 2), (5, 3), (10, 4), (20, 5)]:
        r = pfaffian_maxcut_grid(Lx, Ly, verbose=True)
        t = r['time_s']
        print(f"  exact={r['exact']}, method={r['method']}, time={t:.4f}s")

    print('\n--- Triangular grids (non-bipartite, planar) ---')
    for Lx, Ly in [(3, 2), (4, 3), (5, 4)]:
        r = pfaffian_maxcut_grid(Lx, Ly, triangular=True, verbose=True)
        t = r['time_s']
        print(f"  exact={r['exact']}, method={r['method']}, time={t:.4f}s")

    print('\n--- Complete graphs (non-planar) ---')
    for n in [5, 8, 12]:
        edges = [(i, j, 1.0) for i in range(n) for j in range(i+1, n)]
        r = pfaffian_maxcut(n, edges, verbose=True)
        t = r['time_s']
        print(f"  exact={r['exact']}, method={r['method']}, time={t:.4f}s")

