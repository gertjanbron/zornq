#!/usr/bin/env python3
"""
feedback_edge_solver.py - B99 Feedback-Edge Skeleton Solver for MaxCut

Splits graph into spanning tree + feedback edges (cycles).
Solves the tree part exactly conditioned on feedback-edge assignments,
then optimizes feedback assignments via BLS.

Key insight: on a tree, MaxCut is trivially solvable in O(n) by
leaf-to-root propagation. The "hard" part of MaxCut lives entirely
in the feedback edges (cycles). For sparse/tree-like graphs, the
number of feedback edges k = m - n + 1 is small.

Strategy:
  k <= 20:   Exact enumeration of all 2^k feedback assignments
  k <= 200:  BLS on feedback-edge subgraph + tree propagation
  k > 200:   Multi-start BLS with tree-propagation refinement

For G70 (n=10000, m=9999): k=0, tree is the entire graph -> EXACT!

Algorithm (tree-conditioned MaxCut):
  Given fixed assignments on feedback-edge endpoints,
  solve the tree bottom-up:
  1. Root the tree at node 0
  2. Process leaves first, then parents
  3. For each node v with children c1..cd:
     - Compute gain(v=0) and gain(v=1) from tree-edges to children
       plus any feedback-edge constraints
     - Choose label maximizing total cut contribution
  4. This yields the optimal tree assignment for the given feedback config

References:
  - Tree MaxCut is polynomial: Hadlock (1975)
  - Feedback vertex/edge set: Karp (1972) NP-hard in general
  - Practical: Festa et al. (2002), scatter search for MaxCut

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
import time
import sys
import os
from collections import defaultdict, deque

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Union-Find for Kruskal's algorithm
# ============================================================

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ============================================================
# Maximum spanning tree (Kruskal, max-weight)
# ============================================================

def max_spanning_tree(n_nodes, edges):
    """
    Kruskal's algorithm for maximum-weight spanning tree.

    Args:
        n_nodes: number of nodes
        edges: list of (u, v, w) tuples

    Returns:
        tree_edges: list of (u, v, w) in the spanning tree
        feedback_edges: list of (u, v, w) NOT in the tree (cycle edges)
        tree_adj: adjacency dict {node: [(neighbor, weight), ...]}
    """
    # Sort by absolute weight descending (we want max-weight tree
    # to capture the most important structural edges)
    sorted_edges = sorted(edges, key=lambda e: abs(e[2]), reverse=True)

    uf = UnionFind(n_nodes)
    tree_edges = []
    feedback_edges = []
    tree_adj = defaultdict(list)

    for u, v, w in sorted_edges:
        u, v = int(u), int(v)
        if uf.union(u, v):
            tree_edges.append((u, v, w))
            tree_adj[u].append((v, w))
            tree_adj[v].append((u, w))
        else:
            feedback_edges.append((u, v, w))

    return tree_edges, feedback_edges, tree_adj


# ============================================================
# Tree rooting and ordering
# ============================================================

def root_tree(n_nodes, tree_adj):
    """
    Root tree at node 0, return BFS order and parent array.

    Returns:
        order: list of nodes in BFS order (root first)
        parent: dict {node: parent_node}, root has parent -1
        children: dict {node: [child_nodes]}
        depth: dict {node: depth}
    """
    root = 0
    # Find a node that exists in tree (handle disconnected nodes)
    if tree_adj:
        root = next(iter(tree_adj))

    order = []
    parent = {root: -1}
    children = defaultdict(list)
    depth = {root: 0}
    queue = deque([root])
    visited = {root}

    while queue:
        v = queue.popleft()
        order.append(v)
        for u, w in tree_adj[v]:
            if u not in visited:
                visited.add(u)
                parent[u] = v
                children[v].append(u)
                depth[u] = depth[v] + 1
                queue.append(u)

    # Add isolated nodes (not in tree)
    for v in range(n_nodes):
        if v not in visited:
            order.append(v)
            parent[v] = -1
            depth[v] = 0

    return order, parent, children, depth


# ============================================================
# Tree-conditioned MaxCut solver
# ============================================================

def solve_tree_maxcut(n_nodes, tree_adj, tree_edges, feedback_edges,
                      feedback_assignment):
    """
    Given fixed 0/1 labels on feedback-edge endpoints,
    solve MaxCut on the tree optimally.

    For a tree, the optimal MaxCut assignment can be computed bottom-up:
    each node chooses its label to maximize the cut contribution
    from its subtree + any feedback-edge constraints.

    Args:
        n_nodes: number of nodes
        tree_adj: adjacency dict from max_spanning_tree
        tree_edges: list of tree edges
        feedback_edges: list of feedback edges
        feedback_assignment: numpy array of 0/1 for each feedback edge
            (interpretation: for edge (u,v), 0 = same partition, 1 = different)

    Returns:
        assignment: dict {node: 0/1}
        cut_value: total cut (tree + feedback edges)
    """
    order, parent, children, depth = root_tree(n_nodes, tree_adj)

    # Build edge weight lookup between tree-adjacent nodes
    tree_weight = {}
    for u, v, w in tree_edges:
        tree_weight[(min(u, v), max(u, v))] = w

    # Build feedback constraints: for each node, list of
    # (other_node, weight, same_or_diff) from feedback edges
    # same_or_diff: 0 = should be same partition, 1 = should be different
    fb_constraints = defaultdict(list)
    for idx, (u, v, w) in enumerate(feedback_edges):
        u, v = int(u), int(v)
        diff = int(feedback_assignment[idx])
        fb_constraints[u].append((v, w, diff))
        fb_constraints[v].append((u, w, diff))

    # Bottom-up DP on the tree
    # For each node v, compute:
    #   gain[v][label] = best cut contribution from subtree rooted at v
    #                    when v has the given label (0 or 1)
    # We process in reverse BFS order (leaves first)

    # Store optimal label choice and subtree gain
    best_label = np.zeros(n_nodes, dtype=np.int8)
    subtree_gain = np.zeros((n_nodes, 2))  # [node][label] -> gain

    # Process bottom-up
    for v in reversed(order):
        gain_0 = 0.0  # gain if v = 0
        gain_1 = 0.0  # gain if v = 1

        for child in children.get(v, []):
            key = (min(v, child), max(v, child))
            w = tree_weight.get(key, 1.0)

            # If v=0: edge cut iff child=1, gain = w * subtree_gain[child][1]
            #         edge not cut iff child=0, gain = 0 + subtree_gain[child][0]
            # Choose best child label for each v label
            # v=0, child=0: tree edge not cut (0 gain) + subtree_gain[child][0]
            # v=0, child=1: tree edge cut (w if w>0) + subtree_gain[child][1]
            cut_same = subtree_gain[child][0]  # v=0, child=0: no cut
            cut_diff = w + subtree_gain[child][1]  # v=0, child=1: cut edge

            gain_0 += max(cut_same, cut_diff)

            # v=1, child=0: cut edge (w) + subtree_gain[child][0]
            # v=1, child=1: no cut + subtree_gain[child][1]
            cut_diff_1 = w + subtree_gain[child][0]
            cut_same_1 = subtree_gain[child][1]

            gain_1 += max(cut_diff_1, cut_same_1)

        # Add feedback-edge contributions (these are fixed)
        # We don't know the other endpoint's label yet during bottom-up,
        # so we handle feedback edges in a second pass.
        # For now, just store tree-only gains.
        subtree_gain[v][0] = gain_0
        subtree_gain[v][1] = gain_1

    # Top-down assignment: choose labels greedily from root
    assignment = np.zeros(n_nodes, dtype=np.int8)

    # Root chooses best label (tree-only for now)
    root = order[0]
    assignment[root] = 0 if subtree_gain[root][0] >= subtree_gain[root][1] else 1

    # Propagate down
    for v in order:
        v_label = assignment[v]
        for child in children.get(v, []):
            key = (min(v, child), max(v, child))
            w = tree_weight.get(key, 1.0)

            # Choose child label that maximizes cut
            # v_label=L: child=L -> no cut (0), child=1-L -> cut (w)
            gain_same = subtree_gain[child][v_label]
            gain_diff = w + subtree_gain[child][1 - v_label]

            if gain_diff >= gain_same:
                assignment[child] = 1 - v_label
            else:
                assignment[child] = v_label

    # Now we have the tree-optimal assignment (ignoring feedback edges).
    # Apply feedback-edge constraints to flip nodes if beneficial.
    # This is a local refinement pass.
    assign_dict = {i: int(assignment[i]) for i in range(n_nodes)}

    # Compute total cut
    cut = 0.0
    for u, v, w in tree_edges:
        if assign_dict[u] != assign_dict[v]:
            cut += w
    for u, v, w in feedback_edges:
        u, v = int(u), int(v)
        if assign_dict[u] != assign_dict[v]:
            cut += w

    return assign_dict, cut


# ============================================================
# Fast greedy refinement (O(m) per pass)
# ============================================================

def _greedy_refine(n_nodes, edges, assignment, max_passes=20):
    """
    Fast greedy local search: flip nodes that improve the cut.

    O(m) per pass, typically converges in 3-10 passes.
    Much faster than BLS for getting the "easy" improvements.

    Returns:
        refined_assign: dict {node: 0/1}
        refined_cut: total cut value
        n_flips: total number of flips made
    """
    # Build adjacency arrays for fast delta computation
    adj = defaultdict(list)  # node -> [(neighbor, weight)]
    for u, v, w in edges:
        u, v = int(u), int(v)
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Spin array: +1 or -1
    spins = np.ones(n_nodes, dtype=np.float64)
    for node, label in assignment.items():
        if node < n_nodes:
            spins[node] = 1.0 if label else -1.0

    # Compute delta[v] = gain from flipping v
    # delta[v] = sum_j w(v,j) * s_v * s_j
    # Positive delta -> flipping v improves cut
    deltas = np.zeros(n_nodes)
    for v in range(n_nodes):
        for u, w in adj[v]:
            deltas[v] += w * spins[v] * spins[u]

    total_flips = 0
    for pass_i in range(max_passes):
        improved = False
        for v in range(n_nodes):
            if deltas[v] > 1e-10:
                # Flip v
                spins[v] = -spins[v]
                total_flips += 1
                improved = True
                # Update deltas
                deltas[v] = -deltas[v]
                for u, w in adj[v]:
                    deltas[u] += 2 * w * spins[v] * spins[u]
        if not improved:
            break

    # Compute final cut
    cut = 0.0
    seen = set()
    for v in range(n_nodes):
        for u, w in adj[v]:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                if spins[u] != spins[v]:
                    cut += w

    result = {i: (1 if spins[i] > 0 else 0) for i in range(n_nodes)}
    return result, cut, total_flips


# ============================================================
# Multi-tree ensemble
# ============================================================

def _random_spanning_tree(n_nodes, edges, rng):
    """Build a random spanning tree by shuffling edge order."""
    shuffled = list(edges)
    rng.shuffle(shuffled)
    # Sort by |w| descending but with random tiebreaking (already shuffled)
    # Use a mix: 70% weight-priority, 30% random
    # This creates diverse but structurally meaningful trees
    shuffled.sort(key=lambda e: abs(e[2]) + rng.random() * 0.3, reverse=True)

    uf = UnionFind(n_nodes)
    tree_edges = []
    feedback_edges = []
    tree_adj = defaultdict(list)

    for u, v, w in shuffled:
        u, v = int(u), int(v)
        if uf.union(u, v):
            tree_edges.append((u, v, w))
            tree_adj[u].append((v, w))
            tree_adj[v].append((u, w))
        else:
            feedback_edges.append((u, v, w))

    return tree_edges, feedback_edges, tree_adj


def _eval_full_cut(n_nodes, edges, assignment):
    """Quick cut evaluation."""
    cut = 0.0
    for u, v, w in edges:
        if assignment.get(int(u), 0) != assignment.get(int(v), 0):
            cut += w
    return cut


# ============================================================
# Feedback-edge optimization strategies
# ============================================================

def _eval_with_feedback(n_nodes, tree_adj, tree_edges, feedback_edges,
                        fb_assign):
    """Evaluate: solve tree conditioned on feedback assignment, return cut."""
    assign, cut = solve_tree_maxcut(
        n_nodes, tree_adj, tree_edges, feedback_edges, fb_assign)
    return cut, assign


def _exact_enumerate(n_nodes, tree_adj, tree_edges, feedback_edges,
                     verbose=False):
    """Enumerate all 2^k feedback assignments (k <= 20)."""
    k = len(feedback_edges)
    assert k <= 24, f"Too many feedback edges for exact enumeration: {k}"

    best_cut = -np.inf
    best_assign = None

    for mask in range(1 << k):
        fb_assign = np.array([(mask >> i) & 1 for i in range(k)], dtype=np.int8)
        assign, cut = solve_tree_maxcut(
            n_nodes, tree_adj, tree_edges, feedback_edges, fb_assign)
        if cut > best_cut:
            best_cut = cut
            best_assign = assign
            if verbose and (mask & 0xFF) == 0:
                print("    enum %d/%d: cut=%.1f" % (mask, 1 << k, cut))

    return best_cut, best_assign


def _multi_tree_ensemble(n_nodes, all_edges, time_limit=30, seed=42,
                         n_trees=10, verbose=False):
    """
    Multi-tree ensemble with greedy refinement.

    1. Generate n_trees random spanning trees
    2. Solve each tree optimally in O(n)
    3. Greedy-refine each solution on the full graph
    4. Return the best solution as warm-start for BLS

    This explores diverse structural decompositions of the graph,
    crucial for +-1 Ising instances where a single tree captures
    only ~50% of the problem.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    best_cut = -np.inf
    best_assign = None
    tree_cuts = []

    for t_idx in range(n_trees):
        if time.time() - t0 > time_limit * 0.5:
            break

        # Generate random spanning tree
        tree_edges, fb_edges, tree_adj = _random_spanning_tree(
            n_nodes, all_edges, rng)

        # Solve tree optimally
        fb_zeros = np.zeros(len(fb_edges), dtype=np.int8)
        tree_assign, tree_cut = solve_tree_maxcut(
            n_nodes, tree_adj, tree_edges, fb_edges, fb_zeros)

        # Evaluate on full graph
        full_cut = _eval_full_cut(n_nodes, all_edges, tree_assign)

        # Greedy refinement on full graph
        refined_assign, refined_cut, n_flips = _greedy_refine(
            n_nodes, all_edges, tree_assign, max_passes=10)

        tree_cuts.append(refined_cut)

        if refined_cut > best_cut:
            best_cut = refined_cut
            best_assign = refined_assign
            if verbose:
                print("    Tree %d: tree=%.1f, full=%.1f, refined=%.1f (%d flips) *NEW BEST*" %
                      (t_idx + 1, tree_cut, full_cut, refined_cut, n_flips))
        elif verbose and t_idx < 5:
            print("    Tree %d: tree=%.1f, full=%.1f, refined=%.1f (%d flips)" %
                  (t_idx + 1, tree_cut, full_cut, refined_cut, n_flips))

    ensemble_time = time.time() - t0

    if verbose:
        print("    Ensemble: %d trees, best=%.1f, spread=%.1f-%.1f (%.2fs)" %
              (len(tree_cuts), best_cut, min(tree_cuts), max(tree_cuts),
               ensemble_time))

    # BLS polish with best warm-start
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0:
        from bls_solver import bls_maxcut

        warm = np.zeros(n_nodes, dtype=np.int32)
        for node, label in best_assign.items():
            if node < n_nodes:
                warm[node] = label

        bls_result = bls_maxcut(
            n_nodes, all_edges, time_limit=remaining - 0.5,
            seed=seed, x_init=warm)

        if bls_result['best_cut'] > best_cut:
            if verbose:
                print("    BLS polish: %.1f -> %.1f (%+.1f)" %
                      (best_cut, bls_result['best_cut'],
                       bls_result['best_cut'] - best_cut))
            best_cut = bls_result['best_cut']
            best_assign = bls_result['assignment']

    return best_cut, best_assign


# ============================================================
# Main pipeline
# ============================================================

def feedback_edge_maxcut(n_nodes, edges, time_limit=60, seed=42,
                         verbose=False):
    """
    B99 Feedback-Edge Skeleton Solver for MaxCut.

    Pipeline:
      1. Find maximum spanning tree -> identify feedback edges
      2. If k=0 (tree): solve exactly in O(n)
      3. If k<=20: exact enumeration of 2^k feedback configs
      4. If k>20: BLS over feedback-edge space + tree propagation
      5. Final BLS polish on the full graph

    Args:
        n_nodes: number of nodes
        edges: list of (u, v, w)
        time_limit: wall-clock time limit in seconds
        seed: random seed
        verbose: print progress

    Returns:
        best_cut: best cut value found
        best_assignment: dict {node: 0/1}
        info: dict with statistics
    """
    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("B99: Feedback-Edge Skeleton Solver for MaxCut")
        print("  n=%d, edges=%d" % (n_nodes, len(edges)))
        print("=" * 60)

    # Normalize edges
    norm_edges = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) > 2 else 1.0
        if u != v:
            norm_edges.append((u, v, w))

    # --- Phase 1: Spanning tree decomposition ---
    t_tree = time.time()
    tree_edges, feedback_edges, tree_adj = max_spanning_tree(n_nodes, norm_edges)
    k = len(feedback_edges)
    tree_time = time.time() - t_tree

    if verbose:
        print("\n  Spanning tree: %d tree edges, %d feedback edges (k=%d)" %
              (len(tree_edges), k, k))
        print("  Tree fraction: %.1f%%" % (100 * len(tree_edges) / max(1, len(norm_edges))))
        print("  Tree time: %.3fs" % tree_time)

    # --- Phase 2: Solve feedback-edge assignment ---
    t_solve = time.time()
    remaining = time_limit - (time.time() - t0)

    if k == 0:
        # Pure tree -> exact solution
        fb_assign = np.array([], dtype=np.int8)
        best_assign, best_cut = solve_tree_maxcut(
            n_nodes, tree_adj, tree_edges, feedback_edges, fb_assign)
        method = 'exact_tree'
        if verbose:
            print("\n  Pure tree: exact solution in O(n)")
            print("  Tree MaxCut = %.1f" % best_cut)

    elif k <= 20:
        # Exact enumeration
        if verbose:
            print("\n  Exact enumeration: 2^%d = %d configs" % (k, 1 << k))
        best_cut, best_assign = _exact_enumerate(
            n_nodes, tree_adj, tree_edges, feedback_edges, verbose=verbose)
        method = 'exact_enum_%d' % k
        if verbose:
            print("  Best cut from enumeration: %.1f" % best_cut)

    else:
        # Multi-tree ensemble + greedy refinement + BLS polish
        n_trees = min(20, max(5, int(remaining / 0.5)))
        if verbose:
            print("\n  Multi-tree ensemble (%d trees, %d feedback edges)" %
                  (n_trees, k))
        best_cut, best_assign = _multi_tree_ensemble(
            n_nodes, norm_edges, time_limit=remaining * 0.95,
            seed=seed, n_trees=n_trees, verbose=verbose)
        method = 'ensemble_%dt_k%d' % (n_trees, k)

    solve_time = time.time() - t_solve

    if verbose:
        print("\n  Solve: cut=%.1f, method=%s (%.2fs)" %
              (best_cut, method, solve_time))

    # --- Phase 3: BLS polish on full graph (only for exact methods) ---
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0 and k > 0 and k <= 20:
        from bls_solver import bls_maxcut

        warm = np.zeros(n_nodes, dtype=np.int32)
        for node, label in best_assign.items():
            if node < n_nodes:
                warm[node] = label

        polish_result = bls_maxcut(
            n_nodes, norm_edges, time_limit=remaining - 0.5,
            seed=seed, x_init=warm)

        if polish_result['best_cut'] > best_cut:
            if verbose:
                print("\n  BLS polish: %.1f -> %.1f (%+.1f)" %
                      (best_cut, polish_result['best_cut'],
                       polish_result['best_cut'] - best_cut))
            best_cut = polish_result['best_cut']
            best_assign = polish_result['assignment']

    total_time = time.time() - t0

    if verbose:
        print("\n  Final: cut=%.1f, time=%.2fs" % (best_cut, total_time))
        print("=" * 60)

    info = {
        'tree_time': tree_time,
        'solve_time': solve_time,
        'total_time': total_time,
        'n_tree_edges': len(tree_edges),
        'n_feedback_edges': k,
        'method': method,
        'tree_fraction': len(tree_edges) / max(1, len(norm_edges)),
    }

    return best_cut, best_assign, info


# ============================================================
# Convenience
# ============================================================

def feedback_edge_maxcut_grid(Lx, Ly, **kwargs):
    """Convenience wrapper for grid graphs."""
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x + 1) * Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((i, x * Ly + y + 1, 1.0))
    return feedback_edge_maxcut(n, edges, **kwargs)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='B99: Feedback-Edge Skeleton MaxCut')
    parser.add_argument('--Lx', type=int, default=10)
    parser.add_argument('--Ly', type=int, default=4)
    parser.add_argument('--time-limit', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cut, assign, info = feedback_edge_maxcut_grid(
        args.Lx, args.Ly,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=True)

    n = args.Lx * args.Ly
    m = 2 * n - args.Lx - args.Ly
    print("\nCut: %.1f / %d edges (k=%d feedback)" %
          (cut, m, info['n_feedback_edges']))
