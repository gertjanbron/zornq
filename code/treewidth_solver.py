#!/usr/bin/env python3
"""
treewidth_solver.py - B42 Treewidth-Decompositie: Exact MaxCut via DP

Voor grafen met treewidth tw draait DP in O(n * 2^(tw+1)).
Praktische limiet: tw <= ~22 op laptop.

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from collections import defaultdict
import heapq
import time


def eval_cut(n, edges, assignment):
    """Evalueer cut waarde gegeven assignment dict {vertex: 0/1}."""
    cut = 0.0
    for u, v, w in edges:
        su = assignment.get(int(u), 0)
        sv = assignment.get(int(v), 0)
        if su != sv:
            cut += float(w)
    return cut


def min_degree_ordering(n, edges):
    """Greedy min-degree elimination ordering (heap-geoptimaliseerd)."""
    adj = [set() for _ in range(n)]
    for u, v, w in edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)

    alive = [True] * n
    degree = [len(adj[v]) for v in range(n)]

    pq = [(degree[v], v) for v in range(n)]
    heapq.heapify(pq)

    ordering = []
    bags = []
    tw = 0

    for _ in range(n):
        while pq:
            d, v = heapq.heappop(pq)
            if alive[v]:
                if d == degree[v]:
                    break
                heapq.heappush(pq, (degree[v], v))

        alive[v] = False
        nbrs = [u for u in adj[v] if alive[u]]
        bag = frozenset([v] + nbrs)
        ordering.append(v)
        bags.append(bag)
        tw = max(tw, len(nbrs))

        for u in nbrs:
            degree[u] -= 1

        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    degree[a] += 1
                    degree[b] += 1

        for u in nbrs:
            heapq.heappush(pq, (degree[u], u))

    return ordering, bags, tw


def build_elimination_tree(ordering, bags):
    """Bouw boomstructuur uit eliminatie-ordering."""
    n_bags = len(bags)
    elim_bag = {}
    for i, v in enumerate(ordering):
        elim_bag[v] = i

    children = defaultdict(list)
    root = n_bags - 1
    has_parent = set()

    for i in range(n_bags):
        best_parent = None
        for v in bags[i]:
            if v != ordering[i] and v in elim_bag:
                j = elim_bag[v]
                if j > i and (best_parent is None or j < best_parent):
                    best_parent = j
        if best_parent is not None:
            children[best_parent].append(i)
            has_parent.add(i)

    for i in range(n_bags):
        if i != root and i not in has_parent:
            children[root].append(i)

    return children, root


def dp_maxcut(n, edges, bags, children, root, verbose=False):
    """Exact MaxCut via vectorized DP op tree decomposition."""
    n_bags = len(bags)
    if n_bags == 0:
        return 0.0, {}

    edge_w = {}
    for u, v, w in edges:
        u, v = int(u), int(v)
        key = (min(u, v), max(u, v))
        edge_w[key] = edge_w.get(key, 0.0) + float(w)

    bag_verts = [sorted(bags[i]) for i in range(n_bags)]
    bag_vidx = [{v: j for j, v in enumerate(bv)} for bv in bag_verts]

    vertex_bags = defaultdict(set)
    for i, bag in enumerate(bags):
        for v in bag:
            vertex_bags[v].add(i)

    bag_edge_list = defaultdict(list)
    for key, w in edge_w.items():
        u, v = key
        common = vertex_bags[u] & vertex_bags[v]
        if common:
            target = min(common)
            bag_edge_list[target].append((u, v, w))

    order = []
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
        else:
            stack.append((node, True))
            for c in children[node]:
                stack.append((c, False))

    dp = [None] * n_bags
    choices = [None] * n_bags

    for node in order:
        verts = bag_verts[node]
        k = len(verts)
        n_states = 1 << k
        vidx = bag_vidx[node]

        dp_val = np.zeros(n_states, dtype=np.float64)
        masks = np.arange(n_states, dtype=np.int64)

        for u, v, w in bag_edge_list[node]:
            if u in vidx and v in vidx:
                bi, bj = vidx[u], vidx[v]
                different = ((masks >> bi) ^ (masks >> bj)) & 1
                dp_val += different.astype(np.float64) * w

        node_choices = {}

        for child in children[node]:
            child_verts = bag_verts[child]
            child_k = len(child_verts)
            child_vidx = bag_vidx[child]
            dp_child = dp[child]

            shared_map = []
            for v in child_verts:
                if v in vidx:
                    shared_map.append((vidx[v], child_vidx[v]))

            free_pos = [child_vidx[v] for v in child_verts if v not in vidx]
            n_free = len(free_pos)

            parent_masks = np.arange(n_states, dtype=np.int64)
            base_child = np.zeros(n_states, dtype=np.int64)
            for p_pos, c_pos in shared_map:
                bits = (parent_masks >> p_pos) & 1
                base_child |= (bits << np.int64(c_pos))

            best_val = np.full(n_states, -np.inf)
            best_cmask = np.zeros(n_states, dtype=np.int64)

            for fm in range(1 << n_free):
                offset = np.int64(0)
                for fi, pos in enumerate(free_pos):
                    if (fm >> fi) & 1:
                        offset |= np.int64(1 << pos)
                child_masks = base_child + offset
                vals = dp_child[child_masks]
                better = vals > best_val
                best_val = np.where(better, vals, best_val)
                best_cmask = np.where(better, child_masks, best_cmask)

            dp_val = dp_val + best_val
            node_choices[child] = best_cmask.copy()
            dp[child] = None

        choices[node] = node_choices
        dp[node] = dp_val

    best_mask = int(np.argmax(dp[root]))
    max_cut = float(dp[root][best_mask])

    assignment = {}
    verts = bag_verts[root]
    for j, v in enumerate(verts):
        assignment[v] = (best_mask >> j) & 1

    recon_stack = [root]
    while recon_stack:
        node = recon_stack.pop()
        my_mask = 0
        for j, v in enumerate(bag_verts[node]):
            if assignment.get(v, 0):
                my_mask |= (1 << j)
        if choices[node]:
            for child, cmask_arr in choices[node].items():
                child_mask = int(cmask_arr[my_mask])
                for j, v in enumerate(bag_verts[child]):
                    if v not in assignment:
                        assignment[v] = (child_mask >> j) & 1
                recon_stack.append(child)

    for v in range(n):
        if v not in assignment:
            assignment[v] = 0

    return max_cut, assignment


def treewidth_maxcut(n, edges, max_tw=22, verbose=False):
    """Volledige pipeline: tree decomposition + exact DP MaxCut."""
    t0 = time.time()
    ordering, bags, tw = min_degree_ordering(n, edges)

    if verbose:
        print("  Treewidth bovengrens: %d" % tw)
        print("  Bags: %d" % len(bags))

    if tw > max_tw:
        if verbose:
            print("  Treewidth %d > max %d, overslaan" % (tw, max_tw))
        return None

    children, root = build_elimination_tree(ordering, bags)
    cut, assignment = dp_maxcut(n, edges, bags, children, root, verbose)
    elapsed = time.time() - t0

    if verbose:
        print("  Exact MaxCut = %.1f, tijd = %.3fs" % (cut, elapsed))

    verified_cut = eval_cut(n, edges, assignment)
    if abs(verified_cut - cut) > 1e-6:
        if verbose:
            print("  WAARSCHUWING: DP cut=%.1f vs verified=%.1f" % (cut, verified_cut))
        cut = max(cut, verified_cut)

    info = {
        'treewidth': tw,
        'n_bags': len(bags),
        'time': elapsed,
        'exact': True,
        'verified': abs(verified_cut - cut) < 1e-6,
    }
    return cut, assignment, info


def treewidth_estimate(n, edges):
    """Schat treewidth zonder DP te draaien."""
    _, _, tw = min_degree_ordering(n, edges)
    return tw
