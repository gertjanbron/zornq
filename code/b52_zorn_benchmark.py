#!/usr/bin/env python3
"""
B52: Zorn-Heuristic Solver Benchmark.

Vergelijkt de Zorn-solver (octonion-gebaseerde heuristic) met
klassieke local search op dezelfde random grafen als B36.

Kernvraag: voegt de Zorn-algebra iets toe boven standaard
random init + greedy repair?

Gebruik:
  python b52_zorn_benchmark.py
  python b52_zorn_benchmark.py --n 20 --samples 30
  python b52_zorn_benchmark.py --type er --edge-p 0.3
"""

import numpy as np
import time
import math
import argparse
from dataclasses import dataclass
from typing import Any

# ============================================================
# Zorn algebra (uit zorn_solver/zorn_core.py)
# ============================================================

class Vec3:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z
    def dot(self, rhs):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    def cross(self, rhs):
        return Vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x)
    def scale(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)
    def add(self, rhs):
        return Vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    def sub(self, rhs):
        return Vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0)


class ZornState:
    __slots__ = ('a', 'b', 'u', 'v')
    def __init__(self, a, b, u, v):
        self.a = a; self.b = b; self.u = u; self.v = v
    def norm(self):
        return self.a * self.b - self.u.dot(self.v)
    def multiply(self, rhs):
        out_a = self.a * rhs.a + self.u.dot(rhs.v)
        out_b = self.b * rhs.b + self.v.dot(rhs.u)
        out_u = rhs.u.scale(self.a).add(self.u.scale(rhs.b)).sub(self.v.cross(rhs.v))
        out_v = self.v.scale(rhs.a).add(rhs.v.scale(self.b)).add(self.u.cross(rhs.u))
        return ZornState(a=out_a, b=out_b, u=out_u, v=out_v)
    @classmethod
    def identity(cls):
        return cls(1.0, 1.0, Vec3.zero(), Vec3.zero())
    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, Vec3.zero(), Vec3.zero())


# ============================================================
# Graph + solvers
# ============================================================

class SimpleGraph:
    """Gewogen graaf via adjacency dict."""
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.adj = {i: {} for i in range(n_nodes)}  # adj[u][v] = weight
        self.edges = []  # list of (u, v, w)

    def add_edge(self, u, v, w=1.0):
        self.adj[u][v] = w
        self.adj[v][u] = w
        self.edges.append((u, v, w))

    def cut_value(self, partition):
        """Bereken cut waarde voor een partitie (list van 0/1)."""
        total = 0.0
        for u, v, w in self.edges:
            if partition[u] != partition[v]:
                total += w
        return total

    def local_flip_gain(self, partition, node):
        """Hoeveel cut verbetert als we node flippen."""
        current = partition[node]
        gain = 0.0
        for nb, w in self.adj[node].items():
            if partition[nb] == current:
                gain += w  # was same side, flip makes it a cut
            else:
                gain -= w  # was cut, flip removes it
        return gain

    def weighted_degree(self, node):
        return sum(self.adj[node].values())


# ============================================================
# Graph generators (zelfde als B36)
# ============================================================

def random_3regular(n, seed=42):
    """Random 3-regular graph via pairing model."""
    rng = np.random.RandomState(seed)
    if n % 2 == 1:
        n += 1
    for attempt in range(100):
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or (min(u, v), max(u, v)) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        if valid:
            g = SimpleGraph(n)
            for u, v in edges:
                g.add_edge(u, v)
            return g
    raise RuntimeError("Failed to generate 3-regular graph")


def random_erdos_renyi(n, p=0.3, seed=42):
    rng = np.random.RandomState(seed)
    g = SimpleGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j)
    return g


# ============================================================
# Brute force (exact MaxCut, klein)
# ============================================================

def brute_force_maxcut(graph):
    best_cut = 0
    best_partition = None
    for mask in range(1 << graph.n):
        partition = [(mask >> i) & 1 for i in range(graph.n)]
        c = graph.cut_value(partition)
        if c > best_cut:
            best_cut = c
            best_partition = partition
    return best_cut, best_partition


# ============================================================
# Solver 1: Classical Local Search (baseline)
# ============================================================

def solve_classical_ls(graph, n_restarts=20, seed=42):
    rng = np.random.RandomState(seed)
    nodes = list(range(graph.n))
    best_cut = 0
    best_partition = None

    for _ in range(n_restarts):
        # Random start
        partition = [rng.randint(0, 2) for _ in range(graph.n)]

        # Steepest descent
        for _ in range(200):
            best_node, best_gain = None, 1e-12
            for node in nodes:
                gain = graph.local_flip_gain(partition, node)
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
            if best_node is None:
                break
            partition[best_node] = 1 - partition[best_node]

        c = graph.cut_value(partition)
        if c > best_cut:
            best_cut = c
            best_partition = list(partition)

    return best_cut, best_partition


# ============================================================
# Solver 2: Zorn Heuristic (uit zorn_solver/controller.py)
# ============================================================

def solve_zorn_heuristic(graph, max_iterations=32, seed=42,
                         triage_eps=1e-8, recovery_scale=0.25,
                         invalid_penalty=10.0, load_penalty=0.1,
                         norm_penalty=0.01):
    """Zorn-heuristic solver: octonion multiply + tropical scoring + greedy repair."""
    rng = np.random.default_rng(seed)

    # --- Triage policy ---
    def triage(state):
        n = state.norm()
        vals = [state.a, state.b, state.u.x, state.u.y, state.u.z,
                state.v.x, state.v.y, state.v.z]
        ok = all(math.isfinite(v) for v in vals) and abs(n) > triage_eps
        return ok, n, state if ok else ZornState.zero()

    # --- Tropical scoring ---
    def tropical_score(base_cost, lambda_valid, load, norm_abs):
        penalty = 0.0 if lambda_valid else invalid_penalty
        return base_cost + penalty + load_penalty * load + norm_penalty * norm_abs

    # --- Init: random ZornState per node ---
    def random_state(node):
        deg = graph.weighted_degree(node)
        vec = rng.normal(0.0, recovery_scale, size=6)
        return ZornState(
            a=1.0 + deg, b=1.0,
            u=Vec3(float(vec[0]), float(vec[1]), float(vec[2])),
            v=Vec3(float(vec[3]), float(vec[4]), float(vec[5])))

    states = [random_state(i) for i in range(graph.n)]
    loads = [0.0] * graph.n

    # --- Iteration loop ---
    for iteration in range(max_iterations):
        # Build proposals (bidirectional per edge)
        proposals = {}  # dst_id -> best (src_id, score, new_state_or_none)
        for u, v, w in graph.edges:
            for src, dst in [(u, v), (v, u)]:
                product = states[src].multiply(states[dst])
                valid, norm_val, tri_state = triage(product)
                score = tropical_score(w, valid, loads[dst], abs(norm_val))

                current = proposals.get(dst)
                if current is None or score < current[1]:
                    proposals[dst] = (src, score, valid, tri_state)

        # Commit winners
        touched = set()
        for dst, (src, score, valid, tri_state) in proposals.items():
            touched.add(dst)
            if valid:
                states[dst] = tri_state
                loads[dst] = max(0.0, loads[dst] * 0.9 - 0.1)
            else:
                states[dst] = random_state(dst)
                loads[dst] += 1.0

        for i in range(graph.n):
            if i not in touched:
                loads[i] = max(0.0, loads[i] * 0.95)

    # --- Decode partition ---
    partition = [1 if states[i].a >= states[i].b else 0
                 for i in range(graph.n)]

    # --- Greedy repair ---
    nodes = list(range(graph.n))
    for _ in range(200):
        best_node, best_gain = None, 1e-12
        for node in nodes:
            gain = graph.local_flip_gain(partition, node)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is None:
            break
        partition[best_node] = 1 - partition[best_node]

    cut = graph.cut_value(partition)
    return cut, partition


# ============================================================
# Solver 3: Zorn Heuristic ZONDER greedy repair
# (meet de pure bijdrage van de Zorn-algebra)
# ============================================================

def solve_zorn_no_repair(graph, max_iterations=32, seed=42, **kwargs):
    """Zorn-heuristic zonder greedy repair — puur Zorn-decodering."""
    rng = np.random.default_rng(seed)
    recovery_scale = kwargs.get('recovery_scale', 0.25)
    triage_eps = kwargs.get('triage_eps', 1e-8)
    invalid_penalty = kwargs.get('invalid_penalty', 10.0)
    load_penalty = kwargs.get('load_penalty', 0.1)
    norm_penalty = kwargs.get('norm_penalty', 0.01)

    def triage(state):
        n = state.norm()
        vals = [state.a, state.b, state.u.x, state.u.y, state.u.z,
                state.v.x, state.v.y, state.v.z]
        ok = all(math.isfinite(v) for v in vals) and abs(n) > triage_eps
        return ok, n, state if ok else ZornState.zero()

    def tropical_score(base_cost, lambda_valid, load, norm_abs):
        penalty = 0.0 if lambda_valid else invalid_penalty
        return base_cost + penalty + load_penalty * load + norm_penalty * norm_abs

    def random_state(node):
        deg = graph.weighted_degree(node)
        vec = rng.normal(0.0, recovery_scale, size=6)
        return ZornState(
            a=1.0 + deg, b=1.0,
            u=Vec3(float(vec[0]), float(vec[1]), float(vec[2])),
            v=Vec3(float(vec[3]), float(vec[4]), float(vec[5])))

    states = [random_state(i) for i in range(graph.n)]
    loads = [0.0] * graph.n

    for iteration in range(max_iterations):
        proposals = {}
        for u, v, w in graph.edges:
            for src, dst in [(u, v), (v, u)]:
                product = states[src].multiply(states[dst])
                valid, norm_val, tri_state = triage(product)
                score = tropical_score(w, valid, loads[dst], abs(norm_val))
                current = proposals.get(dst)
                if current is None or score < current[1]:
                    proposals[dst] = (src, score, valid, tri_state)

        touched = set()
        for dst, (src, score, valid, tri_state) in proposals.items():
            touched.add(dst)
            if valid:
                states[dst] = tri_state
                loads[dst] = max(0.0, loads[dst] * 0.9 - 0.1)
            else:
                states[dst] = random_state(dst)
                loads[dst] += 1.0

        for i in range(graph.n):
            if i not in touched:
                loads[i] = max(0.0, loads[i] * 0.95)

    # Decode WITHOUT repair
    partition = [1 if states[i].a >= states[i].b else 0
                 for i in range(graph.n)]
    cut = graph.cut_value(partition)
    return cut, partition


# ============================================================
# Benchmark runner
# ============================================================

def run_benchmark(n, graph_type, n_samples, edge_p=0.3):
    print("=" * 70)
    print("  B52: Zorn-Heuristic Benchmark")
    print("  %s grafen, n=%d, %d samples" % (graph_type, n, n_samples))
    print("=" * 70)

    results = {
        'classical_ls': [],
        'zorn_full': [],      # Zorn + greedy repair
        'zorn_raw': [],       # Zorn zonder repair (pure Zorn-bijdrage)
        'random_assign': [],  # Random baseline
    }
    times = {k: [] for k in results}
    exact_cuts = []

    for s in range(n_samples):
        seed = 1000 + s

        # Generate graph
        if graph_type == '3reg':
            g = random_3regular(n, seed=seed)
        elif graph_type == 'er':
            g = random_erdos_renyi(n, p=edge_p, seed=seed)
        else:
            raise ValueError("Unknown type: %s" % graph_type)

        if not g.edges:
            continue

        # Exact (brute force)
        if n <= 22:
            exact_cut, _ = brute_force_maxcut(g)
        else:
            exact_cut = None
        exact_cuts.append(exact_cut)

        # Random baseline
        rng = np.random.RandomState(seed)
        rand_part = [rng.randint(0, 2) for _ in range(g.n)]
        rand_cut = g.cut_value(rand_part)
        results['random_assign'].append(rand_cut)

        # Classical LS
        t0 = time.time()
        cls_cut, _ = solve_classical_ls(g, n_restarts=20, seed=seed)
        times['classical_ls'].append(time.time() - t0)
        results['classical_ls'].append(cls_cut)

        # Zorn + repair
        t0 = time.time()
        zorn_cut, _ = solve_zorn_heuristic(g, max_iterations=64, seed=seed)
        times['zorn_full'].append(time.time() - t0)
        results['zorn_full'].append(zorn_cut)

        # Zorn raw (no repair)
        t0 = time.time()
        zorn_raw_cut, _ = solve_zorn_no_repair(g, max_iterations=64, seed=seed)
        times['zorn_raw'].append(time.time() - t0)
        results['zorn_raw'].append(zorn_raw_cut)

    # --- Report ---
    print()
    print("  %-20s %8s %8s %8s %8s %8s" % (
        "Methode", "Gem.cut", "Ratio", "Perfect", "Gem.ms", "Best"))
    print("  " + "-" * 72)

    for method in ['random_assign', 'zorn_raw', 'zorn_full', 'classical_ls']:
        cuts = results[method]
        avg_cut = np.mean(cuts)

        if exact_cuts[0] is not None:
            ratios = [c / e if e > 0 else 0 for c, e in zip(cuts, exact_cuts)]
            avg_ratio = np.mean(ratios)
            n_perfect = sum(1 for c, e in zip(cuts, exact_cuts) if abs(c - e) < 0.01)
        else:
            avg_ratio = 0
            n_perfect = 0

        avg_ms = np.mean(times.get(method, [0])) * 1000
        best_cut = max(cuts)

        label = {
            'random_assign': 'Random',
            'zorn_raw': 'Zorn (raw)',
            'zorn_full': 'Zorn + repair',
            'classical_ls': 'Classical LS',
        }[method]

        print("  %-20s %8.1f %8.4f %6d/%d %7.0fms %8.1f" % (
            label, avg_cut, avg_ratio, n_perfect, n_samples, avg_ms, best_cut))

    print()

    # --- Per-instantie vergelijking ---
    if n_samples <= 30:
        zorn_wins = 0
        ls_wins = 0
        ties = 0
        for i in range(len(results['zorn_full'])):
            zf = results['zorn_full'][i]
            cl = results['classical_ls'][i]
            if zf > cl + 0.01:
                zorn_wins += 1
            elif cl > zf + 0.01:
                ls_wins += 1
            else:
                ties += 1
        print("  Head-to-head (Zorn+repair vs Classical LS):")
        print("    Zorn wint: %d | LS wint: %d | Gelijk: %d" % (
            zorn_wins, ls_wins, ties))

    # --- Zorn raw vs random: meet de pure Zorn-bijdrage ---
    raw_cuts = results['zorn_raw']
    rand_cuts = results['random_assign']
    zorn_beter = sum(1 for z, r in zip(raw_cuts, rand_cuts) if z > r + 0.01)
    print()
    print("  Zorn raw (zonder repair) vs Random assignment:")
    print("    Zorn beter: %d/%d" % (zorn_beter, n_samples))
    if exact_cuts[0] is not None:
        raw_ratios = [c / e if e > 0 else 0 for c, e in zip(raw_cuts, exact_cuts)]
        rand_ratios = [c / e if e > 0 else 0 for c, e in zip(rand_cuts, exact_cuts)]
        print("    Zorn raw gem. ratio:  %.4f" % np.mean(raw_ratios))
        print("    Random gem. ratio:    %.4f" % np.mean(rand_ratios))

    print()
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B52: Zorn Heuristic Benchmark')
    parser.add_argument('--n', type=int, default=16, help='Graph size')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--type', choices=['3reg', 'er'], default='3reg')
    parser.add_argument('--edge-p', type=float, default=0.3)
    args = parser.parse_args()

    run_benchmark(args.n, args.type, args.samples, args.edge_p)
