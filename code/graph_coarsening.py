#!/usr/bin/env python3
"""
graph_coarsening.py - B72 Multiscale Graph Coarsening for MaxCut

Coarsen-Solve-Uncoarsen pipeline for large sparse graphs (n>2000).
Designed to close the gap on Gset G60-G81 where current solvers
hit 9-24% gap due to landscape complexity at scale.

Algorithm:
  1. COARSEN: Heavy-edge matching (HEM) contracts edge-pairs into
     super-nodes in rounds until graph is small enough (~500 nodes).
  2. SOLVE: Run combined solver (PA + BLS) on the coarsened graph.
  3. UNCOARSEN: Project assignment back through each level,
     refining with BLS at each step.

Heavy-Edge Matching (Karypis & Kumar 1998):
  - Visit nodes in random order
  - Match each unmatched node with its heaviest unmatched neighbor
  - Merge matched pairs into super-nodes
  - Coarsening ratio ~2x per round => log2(n/target) rounds

References:
  - Karypis & Kumar (1998): Multilevel k-way partitioning
  - Dhillon, Guan, Kulis (2007): Weighted kernel k-means for graph cuts
  - Loukas (2019): Graph reduction with spectral and cut guarantees

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
import time
import sys
import os

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Graph representation for coarsening
# ============================================================

def _build_adjacency(n_nodes, edges):
    """Build adjacency dict {node: {neighbor: weight}}."""
    adj = {i: {} for i in range(n_nodes)}
    for e in edges:
        if len(e) == 3:
            u, v, w = e
        else:
            u, v = e
            w = 1.0
        u, v = int(u), int(v)
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0) + w
        adj[v][u] = adj[v].get(u, 0) + w
    return adj


def _adj_to_edges(adj):
    """Convert adjacency dict back to edge list."""
    edges = []
    seen = set()
    for u in adj:
        for v, w in adj[u].items():
            if (min(u, v), max(u, v)) not in seen:
                seen.add((min(u, v), max(u, v)))
                edges.append((u, v, w))
    return edges


# ============================================================
# Heavy-Edge Matching (HEM) coarsening
# ============================================================

def heavy_edge_matching(n_nodes, adj, rng=None, positive_only=False):
    """
    One round of heavy-edge matching.

    Args:
        n_nodes: number of nodes
        adj: adjacency dict {node: {neighbor: weight}}
        rng: numpy random generator
        positive_only: if True, only match along positive-weight edges.
            Essential for signed-weight graphs (Ising instances) where
            merging nodes connected by negative edges destroys information.

    Returns:
        mapping: dict {fine_node -> coarse_node}
        n_coarse: number of coarse nodes
        groups: dict {coarse_node -> [fine_nodes]}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    matched = set()
    mapping = {}
    groups = {}
    coarse_id = 0

    # Visit nodes in random order
    order = rng.permutation(n_nodes)

    for u in order:
        if u in matched:
            continue

        # Find heaviest unmatched neighbor
        best_v = None
        best_w = -1
        min_w = 0.0 if positive_only else -np.inf
        for v, w in adj.get(u, {}).items():
            if v not in matched and w > best_w and w > min_w:
                best_w = w
                best_v = v

        if best_v is not None:
            # Match u with best_v
            matched.add(u)
            matched.add(best_v)
            mapping[u] = coarse_id
            mapping[best_v] = coarse_id
            groups[coarse_id] = [u, best_v]
        else:
            # Singleton (no unmatched neighbors)
            matched.add(u)
            mapping[u] = coarse_id
            groups[coarse_id] = [u]

        coarse_id += 1

    return mapping, coarse_id, groups


def coarsen_graph(n_nodes, adj, mapping, n_coarse):
    """
    Build coarsened graph from fine graph + mapping.

    Edge weights are summed: w_coarse(A,B) = sum of w(u,v)
    where u in A, v in B.
    """
    coarse_adj = {i: {} for i in range(n_coarse)}

    for u in adj:
        cu = mapping[u]
        for v, w in adj[u].items():
            cv = mapping[v]
            if cu != cv:  # Skip internal edges (they don't affect cut)
                coarse_adj[cu][cv] = coarse_adj[cu].get(cv, 0) + w

    # The fine adjacency is symmetric: adj[u][v] and adj[v][u] both exist.
    # So coarse_adj[A][B] accumulates from all fine (u->v) with u in A, v in B,
    # and coarse_adj[B][A] accumulates from all fine (v->u) with v in B, u in A.
    # Both directions hold the same total weight — already correct and symmetric.
    return coarse_adj


# ============================================================
# Multi-level coarsening
# ============================================================

def multilevel_coarsen(n_nodes, edges, target_size=500, seed=42):
    """
    Coarsen graph in rounds until size <= target_size.

    Returns:
        levels: list of (n_nodes, adj, mapping, groups) per level
        coarsest_n: final number of nodes
        coarsest_edges: edge list for the coarsest graph
    """
    rng = np.random.default_rng(seed)
    adj = _build_adjacency(n_nodes, edges)

    # Detect signed-weight graphs: only match positive edges
    has_negative = any(w < 0 for e in edges for w in [e[2] if len(e) > 2 else 1.0])

    levels = []
    current_n = n_nodes
    current_adj = adj

    max_rounds = 20  # Safety limit

    for round_i in range(max_rounds):
        if current_n <= target_size:
            break

        mapping, n_coarse, groups = heavy_edge_matching(
            current_n, current_adj, rng, positive_only=has_negative)

        # Check if coarsening made progress
        if n_coarse >= current_n * 0.9:
            # Less than 10% reduction — stop
            break

        coarse_adj = coarsen_graph(current_n, current_adj, mapping, n_coarse)

        levels.append({
            'n_fine': current_n,
            'n_coarse': n_coarse,
            'mapping': mapping,
            'groups': groups,
            'fine_adj': current_adj,
        })

        current_n = n_coarse
        current_adj = coarse_adj

    coarsest_edges = _adj_to_edges(current_adj)

    return levels, current_n, coarsest_edges


# ============================================================
# Uncoarsening (projection + refinement)
# ============================================================

def project_assignment(assignment, groups):
    """
    Project coarse assignment back to fine level.
    All fine nodes in a group get the same label as their coarse node.
    """
    fine_assignment = {}
    for coarse_node, fine_nodes in groups.items():
        label = assignment.get(coarse_node, 0)
        for fn in fine_nodes:
            fine_assignment[fn] = label
    return fine_assignment


def refine_assignment(n_nodes, adj, assignment, max_passes=5):
    """
    BLS-style refinement: flip nodes that improve the cut.
    Fast greedy local search on the fine level.
    """
    # Convert assignment to array
    spins = np.zeros(n_nodes, dtype=np.int8)
    for node, label in assignment.items():
        if node < n_nodes:
            spins[node] = 1 if label else -1

    # Fill any unassigned nodes
    spins[spins == 0] = 1

    # Compute initial deltas (gain from flipping each node)
    # delta[i] = cut_after_flip(i) - cut_before = sum of w(i,j) * s_i * s_j
    # (positive delta = flip improves cut)
    deltas = np.zeros(n_nodes)
    for u in adj:
        for v, w in adj[u].items():
            # If same spin: flipping u would cut this edge (+w)
            # If diff spin: flipping u would uncut this edge (-w)
            deltas[u] += w * spins[u] * spins[v]

    improved = True
    passes = 0
    total_flips = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # Steepest-descent pass
        order = np.argsort(-deltas)  # Best improvement first

        for u in order:
            if deltas[u] > 1e-10:
                # Flip u
                spins[u] *= -1
                total_flips += 1
                improved = True

                # Update deltas for u and all neighbors
                deltas[u] = -deltas[u]
                for v, w in adj[u].items():
                    # v's delta changes by 2*w*s_u_new*s_v
                    # s_u flipped, so change = -2 * w * s_u_old * s_v
                    # = +2 * w * s_u_new * s_v
                    deltas[v] += 2 * w * spins[u] * spins[v]

    # Compute final cut
    cut = 0.0
    for u in adj:
        for v, w in adj[u].items():
            if u < v and spins[u] != spins[v]:
                cut += w

    # Convert back to dict
    result = {i: (1 if spins[i] > 0 else 0) for i in range(n_nodes)}

    return result, cut, total_flips


def eval_cut(n_nodes, edges, assignment):
    """Evaluate cut value for an assignment."""
    cut = 0.0
    for e in edges:
        if len(e) == 3:
            u, v, w = e
        else:
            u, v = e
            w = 1.0
        u, v = int(u), int(v)
        a_u = assignment.get(u, 0)
        a_v = assignment.get(v, 0)
        if a_u != a_v:
            cut += w
    return cut


# ============================================================
# Main pipeline: Coarsen-Solve-Uncoarsen
# ============================================================

def coarsen_maxcut(n_nodes, edges, target_size=500, time_limit=60,
                   seed=42, verbose=False, solver='combined'):
    """
    B72 Multiscale Graph Coarsening for MaxCut.

    Pipeline:
      1. Coarsen graph to ~target_size via heavy-edge matching
      2. Solve coarsened graph with PA/BLS/combined
      3. Uncoarsen: project + refine at each level

    Args:
        n_nodes: number of nodes
        edges: list of (u, v) or (u, v, w)
        target_size: coarsen until this many nodes (default 500)
        time_limit: total time budget in seconds
        seed: random seed
        verbose: print progress
        solver: 'pa', 'bls', 'combined' (default)

    Returns:
        best_cut: best cut value found
        best_assignment: dict {node: 0/1}
        info: dict with timing and statistics
    """
    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("B72: Multiscale Graph Coarsening for MaxCut")
        print("  n=%d, edges=%d, target=%d, solver=%s" %
              (n_nodes, len(edges), target_size, solver))
        print("=" * 60)

    # --- Phase 1: Coarsen ---
    t_coarsen = time.time()
    levels, coarse_n, coarse_edges = multilevel_coarsen(
        n_nodes, edges, target_size=target_size, seed=seed)
    coarsen_time = time.time() - t_coarsen

    if verbose:
        print("\n  Coarsening: %d -> %d nodes in %d rounds (%.2fs)" %
              (n_nodes, coarse_n, len(levels), coarsen_time))
        for i, lv in enumerate(levels):
            print("    Round %d: %d -> %d nodes (%.1fx)" %
                  (i + 1, lv['n_fine'], lv['n_coarse'],
                   lv['n_fine'] / lv['n_coarse']))

    # --- Phase 2: Solve coarsened graph ---
    remaining_time = max(time_limit - (time.time() - t0), 5.0)
    coarse_time_budget = remaining_time * 0.7  # 70% for solving

    t_solve = time.time()

    if solver == 'pa':
        from pa_solver import pa_maxcut
        result = pa_maxcut(coarse_n, coarse_edges,
                           time_limit=coarse_time_budget, seed=seed)
        coarse_cut = result['best_cut']
        coarse_assign_dict = result['assignment']
    elif solver == 'bls':
        from bls_solver import bls_maxcut
        result = bls_maxcut(coarse_n, coarse_edges,
                            time_limit=coarse_time_budget, seed=seed)
        coarse_cut = result['best_cut']
        coarse_assign_dict = result['assignment']
    else:  # combined
        from pa_solver import pa_maxcut
        from bls_solver import bls_maxcut

        half_t = coarse_time_budget / 2
        pa_result = pa_maxcut(coarse_n, coarse_edges,
                              time_limit=half_t, seed=seed)
        # Warm-start BLS with PA result
        pa_assign_arr = np.zeros(coarse_n, dtype=np.int32)
        for node, label in pa_result['assignment'].items():
            pa_assign_arr[node] = label
        bls_result = bls_maxcut(coarse_n, coarse_edges,
                                time_limit=half_t, seed=seed,
                                x_init=pa_assign_arr)

        if bls_result['best_cut'] >= pa_result['best_cut']:
            coarse_cut = bls_result['best_cut']
            coarse_assign_dict = bls_result['assignment']
        else:
            coarse_cut = pa_result['best_cut']
            coarse_assign_dict = pa_result['assignment']

    solve_time = time.time() - t_solve

    # Normalize to dict {int: int}
    coarse_assignment = {}
    for i in range(coarse_n):
        coarse_assignment[i] = int(coarse_assign_dict.get(i, 0))

    if verbose:
        print("\n  Solve coarsened (%d nodes): cut=%.1f (%.2fs)" %
              (coarse_n, coarse_cut, solve_time))

    # --- Phase 3: Uncoarsen + refine ---
    t_uncoarsen = time.time()
    current_assignment = coarse_assignment

    for i in range(len(levels) - 1, -1, -1):
        lv = levels[i]

        # Project to fine level
        fine_assignment = project_assignment(current_assignment, lv['groups'])

        # Refine
        refined_assignment, refined_cut, n_flips = refine_assignment(
            lv['n_fine'], lv['fine_adj'], fine_assignment, max_passes=3)

        if verbose:
            proj_cut = eval_cut(lv['n_fine'],
                                _adj_to_edges(lv['fine_adj']),
                                fine_assignment)
            print("    Level %d: %d nodes, projected=%.1f, refined=%.1f (%+.1f, %d flips)" %
                  (i, lv['n_fine'], proj_cut, refined_cut,
                   refined_cut - proj_cut, n_flips))

        current_assignment = refined_assignment

    uncoarsen_time = time.time() - t_uncoarsen

    # Final cut on original graph
    best_cut = eval_cut(n_nodes, edges, current_assignment)

    # --- Optional: final BLS polish with remaining time ---
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0:
        from bls_solver import bls_maxcut

        # Convert dict to array for warm start
        warm = np.zeros(n_nodes, dtype=np.int32)
        for node, label in current_assignment.items():
            if node < n_nodes:
                warm[node] = label

        polish_result = bls_maxcut(
            n_nodes, edges, time_limit=remaining - 0.5,
            seed=seed, x_init=warm)

        if polish_result['best_cut'] > best_cut:
            if verbose:
                print("\n  BLS polish: %.1f -> %.1f (%+.1f)" %
                      (best_cut, polish_result['best_cut'],
                       polish_result['best_cut'] - best_cut))
            best_cut = polish_result['best_cut']
            current_assignment = polish_result['assignment']

    total_time = time.time() - t0

    if verbose:
        print("\n  Final: cut=%.1f, time=%.2fs" % (best_cut, total_time))
        print("    Coarsen: %.2fs, Solve: %.2fs, Uncoarsen: %.2fs" %
              (coarsen_time, solve_time, uncoarsen_time))
        print("=" * 60)

    info = {
        'coarsen_time': coarsen_time,
        'solve_time': solve_time,
        'uncoarsen_time': uncoarsen_time,
        'total_time': total_time,
        'n_levels': len(levels),
        'coarsest_n': coarse_n,
        'coarse_cut': coarse_cut,
        'solver': solver,
        'target_size': target_size,
    }

    return best_cut, current_assignment, info


# ============================================================
# Convenience: grid wrapper
# ============================================================

def coarsen_maxcut_grid(Lx, Ly, triangular=False, **kwargs):
    """Convenience wrapper for grid graphs."""
    n_nodes = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x + 1) * Ly + y, 1))
            if y + 1 < Ly:
                edges.append((i, x * Ly + y + 1, 1))
            if triangular and x + 1 < Lx and y + 1 < Ly:
                edges.append((i, (x + 1) * Ly + y + 1, 1))
    return coarsen_maxcut(n_nodes, edges, **kwargs)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='B72: Multiscale Graph Coarsening MaxCut')
    parser.add_argument('--Lx', type=int, default=100)
    parser.add_argument('--Ly', type=int, default=4)
    parser.add_argument('--target', type=int, default=500)
    parser.add_argument('--time-limit', type=float, default=30)
    parser.add_argument('--solver', default='combined',
                        choices=['pa', 'bls', 'combined'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cut, assign, info = coarsen_maxcut_grid(
        args.Lx, args.Ly,
        target_size=args.target,
        time_limit=args.time_limit,
        solver=args.solver,
        seed=args.seed,
        verbose=True)

    print("\nCut: %.1f / %d edges" % (cut, args.Lx * args.Ly * 2 - args.Lx - args.Ly))
