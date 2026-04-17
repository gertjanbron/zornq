#!/usr/bin/env python3
"""
multiscale_maxcut.py - B149 Multiscale Ordering & Cluster Routing Bridge

Coarse-to-fine MaxCut route inspired by P50:
  1. Build a locality-preserving graph order per component
  2. Split that order into contiguous clusters
  3. Solve the contracted cluster graph
  4. Lift the coarse assignment back to nodes
  5. Warm-start Population Annealing on the full graph

This is intentionally conservative: it adds a structural route without
changing the default production policy until benchmarks show a clear win.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pa_solver import _build_adj_arrays, _greedy_local_search, pa_maxcut


def _normalize_edges(edges):
    """Normalize edges to (u, v, w) with integer endpoints."""
    norm = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u == v:
            continue
        w = float(e[2]) if len(e) > 2 else 1.0
        norm.append((u, v, w))
    return norm


def _build_graph_views(n_nodes, edges):
    """Return adjacency plus degree-like strengths used by ordering."""
    adj = [[] for _ in range(n_nodes)]
    strength = np.zeros(n_nodes, dtype=np.float64)
    abs_strength = np.zeros(n_nodes, dtype=np.float64)
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)
        strength[u] += w
        strength[v] += w
        abs_strength[u] += abs(w)
        abs_strength[v] += abs(w)
    for v in range(n_nodes):
        if len(adj[v]) > 1:
            adj[v].sort(key=lambda nb: (-abs_strength[nb], nb))
    return adj, strength, abs_strength


def _bfs_distances(start, adj, allowed=None):
    """Unweighted BFS distances inside one component."""
    dist = {int(start): 0}
    queue = deque([int(start)])
    allowed_set = allowed if allowed is None else set(allowed)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if allowed_set is not None and v not in allowed_set:
                continue
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            queue.append(v)
    return dist


def _farthest_node(source, adj, allowed, abs_strength):
    """Pick a farthest node, preferring stronger nodes on ties."""
    dist = _bfs_distances(source, adj, allowed=allowed)
    return max(
        dist.items(),
        key=lambda kv: (kv[1], abs_strength[kv[0]], -kv[0]),
    )[0]


def _component_order(seed, adj, abs_strength, seen):
    """Locality-preserving order for one connected component."""
    queue = deque([seed])
    seen.add(seed)
    comp_nodes = []
    while queue:
        u = queue.popleft()
        comp_nodes.append(u)
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            queue.append(v)

    if len(comp_nodes) <= 2:
        return sorted(comp_nodes)

    comp_set = set(comp_nodes)
    anchor_a = max(comp_nodes, key=lambda v: (abs_strength[v], -v))
    anchor_b = _farthest_node(anchor_a, adj, comp_set, abs_strength)
    anchor_c = _farthest_node(anchor_b, adj, comp_set, abs_strength)

    dist_b = _bfs_distances(anchor_b, adj, allowed=comp_set)
    dist_c = _bfs_distances(anchor_c, adj, allowed=comp_set)
    return sorted(
        comp_nodes,
        key=lambda v: (
            dist_b.get(v, 10**9) - dist_c.get(v, 10**9),
            -abs_strength[v],
            v,
        ),
    )


def locality_preserving_order(n_nodes, edges):
    """
    Build a coarse linear order from graph geometry alone.

    This is the graph analogue of P50's locality-preserving address/prefix idea:
    keep nearby nodes nearby in a 1D order, then cluster on that axis.
    """
    edges = _normalize_edges(edges)
    adj, _strength, abs_strength = _build_graph_views(n_nodes, edges)

    seen = set()
    order = []
    component_roots = []
    while len(seen) < n_nodes:
        candidates = [v for v in range(n_nodes) if v not in seen]
        root = max(candidates, key=lambda v: (abs_strength[v], -v))
        component_roots.append(root)
        order.extend(_component_order(root, adj, abs_strength, seen))

    return order, {
        'ordering_method': 'component_diameter_sweep',
        'component_roots': component_roots,
    }


def _target_cluster_count(n_nodes, time_limit=None):
    """Choose a conservative cluster count for the coarse graph."""
    base = int(round(math.sqrt(max(n_nodes, 1))))
    if n_nodes >= 5000:
        base = max(base, 64)
    elif n_nodes >= 2000:
        base = max(base, 32)
    else:
        base = max(base, 16)
    if time_limit is not None and time_limit <= 5.0:
        base = min(base, 64)
    return int(max(4, min(n_nodes, min(96, base))))


def build_multiscale_clusters(n_nodes, edges, n_clusters=None, time_limit=None):
    """Return contiguous clusters on top of a locality-preserving order."""
    order, order_info = locality_preserving_order(n_nodes, edges)
    if n_clusters is None:
        n_clusters = _target_cluster_count(n_nodes, time_limit=time_limit)
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(order, dtype=np.int32),
                                                      n_clusters) if len(chunk) > 0]
    cluster_of = np.full(n_nodes, -1, dtype=np.int32)
    clusters = []
    for cid, chunk in enumerate(chunks):
        nodes = [int(v) for v in chunk]
        clusters.append(nodes)
        for v in nodes:
            cluster_of[v] = cid
    return clusters, cluster_of, order_info


def contract_graph(n_nodes, edges, clusters, cluster_of):
    """Aggregate the original graph into a coarse cluster graph."""
    del n_nodes  # kept for symmetric API
    inter = defaultdict(float)
    internal_weight = np.zeros(len(clusters), dtype=np.float64)
    cut_boundary = np.zeros(len(clusters), dtype=np.float64)

    for u, v, w in _normalize_edges(edges):
        cu = int(cluster_of[u])
        cv = int(cluster_of[v])
        if cu == cv:
            internal_weight[cu] += w
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        inter[key] += w
        cut_boundary[cu] += abs(w)
        cut_boundary[cv] += abs(w)

    coarse_edges = [
        (cu, cv, float(w))
        for (cu, cv), w in inter.items()
        if abs(w) > 1e-12
    ]
    return coarse_edges, {
        'internal_weight': internal_weight.tolist(),
        'boundary_weight': cut_boundary.tolist(),
    }


def _lift_assignment(n_nodes, cluster_of, coarse_assignment):
    """Lift cluster bits back to node bits."""
    x = np.zeros(n_nodes, dtype=np.int32)
    for node in range(n_nodes):
        cid = int(cluster_of[node])
        x[node] = 1 if int(coarse_assignment.get(cid, 0)) > 0 else 0
    return x


def _coarse_time_budget(total_budget):
    """Reserve a small slice for the coarse solve."""
    if total_budget is None or total_budget <= 0.0:
        return None
    return min(1.5, max(0.25, 0.15 * total_budget))


def _solve_coarse_graph(n_clusters, coarse_edges, seed=42, time_limit=None):
    """Solve the contracted graph with a heavier small-graph PA run."""
    if n_clusters <= 1 or not coarse_edges:
        return {
            'best_cut': 0.0,
            'assignment': {i: 0 for i in range(n_clusters)},
            'time_s': 0.0,
            'solver_note': 'coarse-trivial',
            'device': 'cpu',
        }

    replicas = min(256, max(96, 4096 // max(n_clusters, 1)))
    n_temps = 70 if n_clusters <= 48 else 60
    n_sweeps = 4 if n_clusters <= 64 else 3
    result = pa_maxcut(
        n_clusters,
        coarse_edges,
        n_replicas=replicas,
        n_temps=n_temps,
        n_sweeps=n_sweeps,
        time_limit=time_limit,
        seed=seed,
    )
    result['device'] = result.get('device', 'cpu')
    result['solver_note'] = (
        f"coarse-pa(clusters={n_clusters},replicas={replicas},temps={n_temps})"
    )
    return result


def _assignment_from_bits(bits):
    """Convert a bit vector to the benchmark assignment dict format."""
    return {i: int(bits[i]) for i in range(len(bits))}


def _full_pa_kwargs(n_nodes):
    """Warm-started PA schedule for the full graph."""
    if n_nodes <= 2500:
        return {'n_replicas': 150, 'n_temps': 50, 'n_sweeps': 3}
    if n_nodes <= 5000:
        return {'n_replicas': 120, 'n_temps': 50, 'n_sweeps': 3}
    return {'n_replicas': 96, 'n_temps': 45, 'n_sweeps': 3}


def multiscale_pa_maxcut(n_nodes, edges, seed=42, time_limit=None,
                         n_clusters=None, verbose=False):
    """
    B149 multiscale coarse-to-fine MaxCut route.

    Returns the same shape as the other benchmark solvers.
    """
    t0 = time.time()
    edges = _normalize_edges(edges)
    if n_nodes <= 64 or len(edges) <= 64:
        result = pa_maxcut(n_nodes, edges, seed=seed, time_limit=time_limit)
        result['device'] = result.get('device', 'cpu')
        result['solver_note'] = 'multiscale-fallback-small'
        return result

    clusters, cluster_of, order_info = build_multiscale_clusters(
        n_nodes, edges, n_clusters=n_clusters, time_limit=time_limit)
    coarse_edges, contract_info = contract_graph(n_nodes, edges, clusters, cluster_of)
    coarse_budget = _coarse_time_budget(time_limit)

    coarse = _solve_coarse_graph(
        len(clusters),
        coarse_edges,
        seed=seed,
        time_limit=coarse_budget,
    )
    x_init = _lift_assignment(n_nodes, cluster_of, coarse.get('assignment', {}))

    adj_arr, wt_arr, deg = _build_adj_arrays(n_nodes, edges)
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float64)

    lifted_bits, lifted_cut = _greedy_local_search(
        x_init.astype(np.int32, copy=True),
        adj_arr, wt_arr, deg, ei, ej, ew,
        max_iter=80,
    )

    elapsed = time.time() - t0
    remaining = None
    if time_limit is not None and time_limit > 0.0:
        remaining = max(0.0, time_limit - elapsed)

    best = {
        'best_cut': float(lifted_cut),
        'assignment': _assignment_from_bits(lifted_bits),
        'time_s': elapsed,
        'device': 'cpu',
        'solver_note': 'multiscale-lifted-greedy',
    }

    if remaining is not None and remaining < 0.5:
        best['solver_note'] = (
            f"multiscale-pa-short(clusters={len(clusters)},order={order_info['ordering_method']})"
        )
        return best

    pa_kwargs = _full_pa_kwargs(n_nodes)
    warm = pa_maxcut(
        n_nodes,
        edges,
        time_limit=remaining,
        seed=seed,
        x_init=lifted_bits,
        **pa_kwargs,
    )
    warm['device'] = warm.get('device', 'cpu')
    if warm.get('best_cut', -1.0) >= best['best_cut']:
        best = warm
    else:
        best['solver_note'] = 'multiscale-lifted-greedy-won'

    best['time_s'] = time.time() - t0
    avg_cluster = float(np.mean([len(c) for c in clusters])) if clusters else 0.0
    note = (
        f"multiscale-pa(clusters={len(clusters)},avg_cluster={avg_cluster:.1f},"
        f"coarse_cut={coarse.get('best_cut', 0.0):.0f},"
        f"order={order_info['ordering_method']})"
    )
    if contract_info['boundary_weight']:
        boundary_mean = float(np.mean(contract_info['boundary_weight']))
        note += f"[boundary_mean={boundary_mean:.1f}]"
    best['solver_note'] = note
    if verbose:
        print(f"[B149] {note}, cut={best['best_cut']:.0f}, t={best['time_s']:.2f}s")
    return best


if __name__ == '__main__':
    import argparse
    from gset_loader import load_gset

    parser = argparse.ArgumentParser(description='B149 multiscale MaxCut demo')
    parser.add_argument('--graph', type=str, default='G59')
    parser.add_argument('--time-limit', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    g, bks, _info = load_gset(args.graph)
    result = multiscale_pa_maxcut(
        g.n_nodes, list(g.edges()), seed=args.seed,
        time_limit=args.time_limit, verbose=True,
    )
    gap = 100.0 * (bks - result['best_cut']) / bks if bks else None
    print(result['solver_note'])
    if gap is not None:
        print(f"cut={result['best_cut']:.0f}, gap={gap:.3f}%")
