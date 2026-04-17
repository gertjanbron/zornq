#!/usr/bin/env python3
"""
B60: Goemans-Williamson SDP Bound Reporter.

Berekent de SDP-relaxatie bovengrens voor MaxCut op een graaf.
Elke solver-run krijgt daarmee een gap-to-bound:
  niet alleen "ratio = 0.766"
  maar ook   "ratio = 0.766, GW-bound = 0.872, gap = 12.2%"

De GW-bound is de tightste bekende polynomiale bovengrens voor MaxCut.
De SDP-waarde is een bovengrens op OPT; de GW-rounding geeft een
0.878-approximatie ondergrens.

Gebruik:
  python b60_gw_bound.py                         # 4x3 cylinder
  python b60_gw_bound.py --Lx 8 --Ly 3           # 8x3 cylinder
  python b60_gw_bound.py --Lx 6 --Ly 3 --triangular  # gefrustreerd
  python b60_gw_bound.py --random-3reg 16         # random 3-regulier

Vereist: pip install cvxpy
"""

import numpy as np
import cvxpy as cp
import time
import argparse
import math


# ============================================================
# Graaf
# ============================================================

class SimpleGraph:
    """Ongewogen graaf via adjacency + edge list."""
    def __init__(self, n):
        self.n = n
        self.edges = []
        self.adj = {i: [] for i in range(n)}

    def add_edge(self, u, v, w=1.0):
        self.edges.append((u, v, w))
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    @property
    def n_edges(self):
        return len(self.edges)

    def cut_value(self, bitstring):
        total = 0.0
        for u, v, w in self.edges:
            if bitstring[u] != bitstring[v]:
                total += w
        return total

    def total_weight(self):
        return sum(w for _, _, w in self.edges)


# ============================================================
# Graaf generators
# ============================================================

def cylinder_graph(Lx, Ly, triangular=False):
    """Cylinder-rooster, optioneel triangulair."""
    g = SimpleGraph(Lx * Ly)
    for x in range(Lx - 1):
        for y in range(Ly):
            g.add_edge(x * Ly + y, (x + 1) * Ly + y)
    for x in range(Lx):
        for y in range(Ly - 1):
            g.add_edge(x * Ly + y, x * Ly + y + 1)
    if triangular:
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                g.add_edge(x * Ly + y, (x + 1) * Ly + y + 1)
    return g


def random_3regular(n, seed=42):
    """Random 3-reguliere graaf via pairing model."""
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
# Brute force (voor verificatie)
# ============================================================

def brute_force_maxcut(graph):
    if graph.n > 22:
        return None
    best = 0
    for mask in range(1 << graph.n):
        bs = [(mask >> i) & 1 for i in range(graph.n)]
        c = graph.cut_value(bs)
        if c > best:
            best = c
    return best


# ============================================================
# GW SDP Bound
# ============================================================

def gw_sdp_bound(graph, verbose=True):
    """Bereken Goemans-Williamson SDP bovengrens voor MaxCut.

    Maximaliseert: sum_{(i,j) in E} w_ij * (1 - X_ij) / 2
    Subject to:    X is PSD, X_ii = 1

    Returns: dict met sdp_bound, gw_ratio, solve_time, status
    """
    n = graph.n
    t0 = time.time()

    # SDP variabele
    X = cp.Variable((n, n), symmetric=True)

    # Constraint: X is PSD met diag = 1
    constraints = [X >> 0]  # PSD
    constraints += [X[i, i] == 1 for i in range(n)]

    # Objective: maximize sum w_ij (1 - X_ij) / 2
    obj = 0
    for u, v, w in graph.edges:
        obj = obj + w * (1 - X[u, v]) / 2

    prob = cp.Problem(cp.Maximize(obj), constraints)

    # Solve met SCS (open-source, geen licentie nodig)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000,
                   eps=1e-6)
    except cp.SolverError:
        # Fallback naar ECOS als SCS faalt
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            return {'sdp_bound': None, 'status': 'FAILED',
                    'solve_time': time.time() - t0}

    solve_time = time.time() - t0
    status = prob.status

    if prob.value is None:
        return {'sdp_bound': None, 'status': status,
                'solve_time': solve_time}

    sdp_bound = float(prob.value)
    # GW garantie: rounding geeft >= 0.878 * SDP
    gw_guaranteed = 0.87856 * sdp_bound

    result = {
        'sdp_bound': sdp_bound,
        'sdp_ratio': sdp_bound / graph.n_edges if graph.n_edges > 0 else 0,
        'gw_guaranteed': gw_guaranteed,
        'gw_ratio': gw_guaranteed / graph.n_edges if graph.n_edges > 0 else 0,
        'solve_time': solve_time,
        'status': status,
        'n': n,
        'n_edges': graph.n_edges,
    }

    if verbose:
        print("  GW-SDP Bound:")
        print("    SDP bovengrens:    %.2f / %d edges (ratio %.6f)" % (
            sdp_bound, graph.n_edges, result['sdp_ratio']))
        print("    GW ondergrens:     %.2f (0.878 × SDP)" % gw_guaranteed)
        print("    Oplostijd:         %.3fs" % solve_time)
        print("    Status:            %s" % status)

    return result


def report_gap(solver_cut, sdp_result, label="Solver"):
    """Rapporteer gap tussen solver-resultaat en SDP-bound."""
    if sdp_result['sdp_bound'] is None:
        print("  %s: gap niet beschikbaar (SDP failed)" % label)
        return

    sdp = sdp_result['sdp_bound']
    n_edges = sdp_result['n_edges']
    ratio = solver_cut / n_edges if n_edges > 0 else 0
    sdp_ratio = sdp_result['sdp_ratio']

    # Gap to SDP upper bound
    if sdp > 0:
        gap_pct = 100 * (1 - solver_cut / sdp)
    else:
        gap_pct = 0

    print("  %s:" % label)
    print("    Cut:           %.1f / %d (ratio %.6f)" % (solver_cut, n_edges, ratio))
    print("    SDP bound:     %.1f (ratio %.6f)" % (sdp, sdp_ratio))
    print("    Gap to bound:  %.2f%%" % gap_pct)

    # Vergelijk met GW-garantie
    gw = sdp_result['gw_guaranteed']
    if solver_cut >= gw:
        print("    vs GW 0.878:   BOVEN GW-grens (%.1f >= %.1f)" % (solver_cut, gw))
    else:
        print("    vs GW 0.878:   onder GW-grens (%.1f < %.1f)" % (solver_cut, gw))


# ============================================================
# Demo / CLI
# ============================================================

def run_demo(graph, label=""):
    n = graph.n
    print("=" * 60)
    print("  B60: GW-Bound Report%s" % (" — " + label if label else ""))
    print("  Graaf: n=%d, %d edges" % (n, graph.n_edges))
    print("=" * 60)

    # SDP bound
    sdp = gw_sdp_bound(graph, verbose=True)

    # Exact (als klein genoeg)
    exact = brute_force_maxcut(graph)
    if exact is not None:
        print("\n  Exact MaxCut: %d / %d (ratio %.6f)" % (
            exact, graph.n_edges, exact / graph.n_edges))
        report_gap(exact, sdp, "Exact")

        # Sanity check: SDP >= OPT
        if sdp['sdp_bound'] is not None and sdp['sdp_bound'] < exact - 0.01:
            print("  WARNING: SDP bound < exact! (numerieke fout)")
    else:
        print("\n  Exact: overgeslagen (n=%d > 22)" % n)

    # Random + repair baseline
    print("\n  Random+repair baseline (20 restarts):")
    rng = np.random.RandomState(42)
    best_cut = 0
    for _ in range(20):
        bs = rng.randint(0, 2, size=n)
        # Steepest descent
        for _ in range(200):
            best_node, best_gain = -1, 0
            for node in range(n):
                gain = 0
                for nb, w in graph.adj[node]:
                    if bs[nb] == bs[node]:
                        gain += w
                    else:
                        gain -= w
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
            if best_node < 0:
                break
            bs[best_node] = 1 - bs[best_node]
        c = graph.cut_value(bs)
        if c > best_cut:
            best_cut = c
    report_gap(best_cut, sdp, "Random+repair (best of 20)")

    print("\n" + "=" * 60)
    return sdp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B60: GW-SDP Bound')
    parser.add_argument('--Lx', type=int, default=4)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--triangular', action='store_true')
    parser.add_argument('--random-3reg', type=int, default=0,
                        help='Random 3-reguliere graaf met n nodes')
    parser.add_argument('--random-er', type=int, default=0,
                        help='Erdos-Renyi graaf met n nodes')
    parser.add_argument('--edge-p', type=float, default=0.3)
    args = parser.parse_args()

    if args.random_3reg > 0:
        g = random_3regular(args.random_3reg)
        run_demo(g, "3-regulier n=%d" % args.random_3reg)
    elif args.random_er > 0:
        g = random_erdos_renyi(args.random_er, p=args.edge_p)
        run_demo(g, "ER n=%d p=%.2f" % (args.random_er, args.edge_p))
    else:
        topo = "triangulair" if args.triangular else "vierkant"
        g = cylinder_graph(args.Lx, args.Ly, args.triangular)
        run_demo(g, "%dx%d %s" % (args.Lx, args.Ly, topo))
