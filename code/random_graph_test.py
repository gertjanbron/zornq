#!/usr/bin/env python3
"""
random_graph_test.py - B36: Random Graph Testing.

Test RQAOA op niet-triviale grafen (3-regular, Erdos-Renyi) waar
MaxCut NIET triviaal is. Vergelijkt:

  1. RQAOA: QAOA p=1 -> ZZ correlaties -> greedy -> local search
  2. Puur klassiek: random assignment -> local search (zelfde LS, geen quantum)
  3. Brute force: exact MaxCut (referentie, <= 25 nodes)

Kernvraag: helpen de quantum-correlaties daadwerkelijk, of doet de
local search al het werk?

Gebruik:
  python random_graph_test.py                      # standaard benchmark
  python random_graph_test.py --n 16 --samples 50  # 16 nodes, 50 grafen
  python random_graph_test.py --type er --edge-p 0.5  # Erdos-Renyi
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph, GeneralQAOA, RQAOA, brute_force_maxcut


# =====================================================================
# Random graaf generatoren
# =====================================================================

def random_3regular(n, seed=None):
    """Genereer een random 3-reguliere graaf op n nodes.

    Gebruikt het pairing model: maak 3 kopien van elke node,
    shuffle, en koppel in paren. Verwijder self-loops en multi-edges.
    Herhaal tot een geldige graaf ontstaat.

    Vereist: n even (3n/2 edges).
    """
    if n % 2 != 0:
        raise ValueError("3-regular graaf vereist even n, kreeg %d" % n)

    rng = np.random.RandomState(seed)
    max_attempts = 100

    for attempt in range(max_attempts):
        # Maak 3 kopien van elke node
        points = []
        for i in range(n):
            points.extend([i, i, i])

        rng.shuffle(points)

        # Koppel in paren
        graph = WeightedGraph()
        for i in range(n):
            graph.add_node(i)

        valid = True
        seen_edges = set()
        for k in range(0, len(points), 2):
            u, v = points[k], points[k + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in seen_edges:
                valid = False
                break
            seen_edges.add(edge)
            graph.add_edge(u, v, 1.0)

        if valid and graph.n_edges == 3 * n // 2:
            return graph

    # Fallback: bouw een deterministische 3-reguliere graaf
    # (ring + matching)
    graph = WeightedGraph()
    for i in range(n):
        graph.add_node(i)
        graph.add_edge(i, (i + 1) % n, 1.0)
    for i in range(0, n, 2):
        j = (i + n // 2) % n
        if j != i and j != (i + 1) % n and j != (i - 1) % n:
            graph.add_edge(i, j, 1.0)
    return graph


def random_erdos_renyi(n, p=0.5, seed=None):
    """Genereer een Erdos-Renyi G(n, p) random graaf."""
    rng = np.random.RandomState(seed)
    graph = WeightedGraph()
    for i in range(n):
        graph.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                graph.add_edge(i, j, 1.0)
    return graph


def random_weighted_graph(n, p=0.5, seed=None):
    """Erdos-Renyi met random gewichten in [0.5, 2.0]."""
    rng = np.random.RandomState(seed)
    graph = WeightedGraph()
    for i in range(n):
        graph.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = 0.5 + 1.5 * rng.random()
                graph.add_edge(i, j, w)
    return graph


# =====================================================================
# Solvers
# =====================================================================

def solve_rqaoa(graph, p=1, n_gamma=10, n_beta=10, reorder='auto'):
    """End-to-end RQAOA benchmark wrapper.

    Returns: (cut, ratio, info)
    """
    t0 = time.time()
    ne = graph.n_edges
    if ne == 0:
        return 0.0, 0.0, {'method': 'rqaoa', 'time': 0}

    rqaoa = RQAOA(graph, p=p, verbose=False)
    result = rqaoa.solve(mode='auto', n_gamma=n_gamma, n_beta=n_beta,
                         local_search=True, reorder=reorder)

    info = dict(result.info)
    info.update({
        'method': 'rqaoa',
        'gammas': info.get('gammas'),
        'betas': info.get('betas'),
        'time': time.time() - t0,
    })
    return result.cut_value, result.ratio, info


def solve_classical_ls(graph, n_restarts=20, seed=42):
    """Puur klassiek: random assignment -> local search.

    Geen quantum-input. Zelfde local search als RQAOA.

    Returns: (cut, ratio, info)
    """
    t0 = time.time()
    ne = graph.n_edges
    if ne == 0:
        return 0.0, 0.0, {'method': 'classical_ls', 'time': 0}

    nodes = graph.nodes
    n = len(nodes)
    rng = np.random.RandomState(seed)

    # Dummy RQAOA object voor _local_search
    rqaoa = RQAOA(graph, p=1, verbose=False)

    best_cut = -1
    best_assignment = None

    for restart in range(n_restarts):
        # Random assignment
        assignment = {}
        for node in nodes:
            assignment[node] = 1 if rng.random() < 0.5 else -1

        # Local search (steepest descent + perturbaties)
        cut, assignment, n_flips = rqaoa._local_search(
            graph, assignment, max_rounds=100, n_restarts=10)

        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment

    ratio = best_cut / ne if ne > 0 else 0
    info = {
        'method': 'classical_ls',
        'n_restarts': n_restarts,
        'time': time.time() - t0,
    }
    return best_cut, ratio, info


def solve_brute_force(graph):
    """Exact MaxCut via brute force. Referentie.

    Returns: (cut, ratio, info)
    """
    t0 = time.time()
    ne = graph.n_edges
    if ne == 0:
        return 0.0, 0.0, {'method': 'brute_force', 'time': 0}

    cut, assignment = brute_force_maxcut(graph)
    ratio = cut / ne if ne > 0 else 0
    info = {
        'method': 'brute_force',
        'time': time.time() - t0,
    }
    return cut, ratio, info


# =====================================================================
# Benchmark suite
# =====================================================================

def run_benchmark(n, graph_type='3reg', n_samples=20, edge_p=0.5, p=1,
                  reorder='auto', verbose=True):
    """Draai volledige benchmark op random grafen.

    Args:
        n: aantal nodes per graaf
        graph_type: '3reg', 'er' (Erdos-Renyi), 'weighted'
        n_samples: aantal random grafen
        edge_p: edge probability voor ER grafen
        verbose: print per-graaf resultaten

    Returns: dict met statistieken
    """
    results = {
        'rqaoa_ratios': [],
        'classical_ratios': [],
        'exact_ratios': [],
        'rqaoa_approx': [],      # rqaoa_cut / exact_cut
        'classical_approx': [],   # classical_cut / exact_cut
        'rqaoa_times': [],
        'classical_times': [],
        'qaoa_raw_ratios': [],   # QAOA zonder local search
        'n_edges': [],
    }

    can_brute = (n <= 25)

    if verbose:
        print("\n" + "=" * 70)
        print("B36 Random Graph Benchmark: %s, n=%d, %d samples, reorder=%s" %
              (graph_type, n, n_samples, reorder))
        print("=" * 70)
        if can_brute:
            print("%-6s  %5s  %8s  %8s  %8s  %8s  %8s" %
                  ("Graph", "Edges", "Exact", "RQAOA", "Classic",
                   "R/Exact", "C/Exact"))
        else:
            print("%-6s  %5s  %8s  %8s" %
                  ("Graph", "Edges", "RQAOA", "Classic"))
        print("-" * 70)

    for sample in range(n_samples):
        seed = 1000 + sample

        # Genereer graaf
        if graph_type == '3reg':
            graph = random_3regular(n, seed=seed)
        elif graph_type == 'er':
            graph = random_erdos_renyi(n, p=edge_p, seed=seed)
        elif graph_type == 'weighted':
            graph = random_weighted_graph(n, p=edge_p, seed=seed)
        else:
            raise ValueError("Onbekend type: %s" % graph_type)

        ne = graph.n_edges
        if ne == 0:
            continue

        results['n_edges'].append(ne)

        # RQAOA
        rqaoa_cut, rqaoa_ratio, rqaoa_info = solve_rqaoa(
            graph, p=p, reorder=reorder)
        results['rqaoa_ratios'].append(rqaoa_ratio)
        results['rqaoa_times'].append(rqaoa_info['time'])
        results['qaoa_raw_ratios'].append(
            rqaoa_info.get('qaoa_ratio', 0))

        # Klassiek
        cls_cut, cls_ratio, cls_info = solve_classical_ls(
            graph, n_restarts=20, seed=seed + 5000)
        results['classical_ratios'].append(cls_ratio)
        results['classical_times'].append(cls_info['time'])

        # Brute force (referentie)
        if can_brute:
            bf_cut, bf_ratio, bf_info = solve_brute_force(graph)
            results['exact_ratios'].append(bf_ratio)
            r_approx = rqaoa_cut / bf_cut if bf_cut > 0 else 0
            c_approx = cls_cut / bf_cut if bf_cut > 0 else 0
            results['rqaoa_approx'].append(r_approx)
            results['classical_approx'].append(c_approx)

            if verbose:
                print("  #%-3d  %5d  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f" %
                      (sample, ne, bf_ratio, rqaoa_ratio, cls_ratio,
                       r_approx, c_approx))
        else:
            if verbose:
                print("  #%-3d  %5d  %8.4f  %8.4f" %
                      (sample, ne, rqaoa_ratio, cls_ratio))

    # Statistieken
    if verbose:
        print("-" * 70)
        print("\nStatistieken over %d grafen:" % len(results['rqaoa_ratios']))

        rqaoa_arr = np.array(results['rqaoa_ratios'])
        cls_arr = np.array(results['classical_ratios'])
        qaoa_arr = np.array(results['qaoa_raw_ratios'])

        print("  QAOA p=%d (raw):  mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
              (p, qaoa_arr.mean(), qaoa_arr.std(), qaoa_arr.min(), qaoa_arr.max()))
        print("  RQAOA + LS:       mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
              (rqaoa_arr.mean(), rqaoa_arr.std(), rqaoa_arr.min(),
               rqaoa_arr.max()))
        print("  Klassiek LS:      mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
              (cls_arr.mean(), cls_arr.std(), cls_arr.min(), cls_arr.max()))

        if can_brute and results['rqaoa_approx']:
            ra = np.array(results['rqaoa_approx'])
            ca = np.array(results['classical_approx'])
            print("\n  Approximatieratio (cut/exact):")
            print("    RQAOA:     mean=%.4f  std=%.4f  min=%.4f" %
                  (ra.mean(), ra.std(), ra.min()))
            print("    Klassiek:  mean=%.4f  std=%.4f  min=%.4f" %
                  (ca.mean(), ca.std(), ca.min()))

            # Hoe vaak wint RQAOA?
            n_rqaoa_wins = np.sum(rqaoa_arr > cls_arr + 1e-9)
            n_classical_wins = np.sum(cls_arr > rqaoa_arr + 1e-9)
            n_ties = len(rqaoa_arr) - n_rqaoa_wins - n_classical_wins
            print("\n  Head-to-head: RQAOA wint %d, Klassiek wint %d, "
                  "gelijk %d (van %d)" %
                  (n_rqaoa_wins, n_classical_wins, n_ties,
                   len(rqaoa_arr)))

            # Perfecte oplossingen
            n_rqaoa_perfect = np.sum(ra > 0.999)
            n_cls_perfect = np.sum(ca > 0.999)
            print("  Perfecte oplossingen: RQAOA %d/%d, Klassiek %d/%d" %
                  (n_rqaoa_perfect, len(ra), n_cls_perfect, len(ca)))

        print("\n  Gemiddelde tijd per graaf:")
        print("    RQAOA:    %.3fs" % np.mean(results['rqaoa_times']))
        print("    Klassiek: %.3fs" % np.mean(results['classical_times']))

    return results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B36: Random Graph Testing — RQAOA vs Klassiek')
    parser.add_argument('--n', type=int, default=14,
                        help='Aantal nodes (default: 14)')
    parser.add_argument('--samples', type=int, default=20,
                        help='Aantal random grafen (default: 20)')
    parser.add_argument('--type', type=str, default='3reg',
                        choices=['3reg', 'er', 'weighted'],
                        help='Graaftype (default: 3reg)')
    parser.add_argument('--edge-p', type=float, default=0.5,
                        help='Edge probability voor ER (default: 0.5)')
    parser.add_argument('--p', type=int, default=1,
                        help='QAOA diepte voor RQAOA (default: 1)')
    parser.add_argument('--reorder', type=str, default='auto',
                        choices=['auto', 'none', 'fiedler'],
                        help='Ordering preprocessing voor RQAOA (default: auto)')
    args = parser.parse_args()

    # Kleine test eerst
    print("Quick sanity check...")
    g = random_3regular(8, seed=0)
    print("  3-regular(8): %d nodes, %d edges" % (g.n_nodes, g.n_edges))
    bf_cut, _ = brute_force_maxcut(g)
    print("  Exact MaxCut: %.0f / %d = %.4f" %
          (bf_cut, g.n_edges, bf_cut / g.n_edges))

    # Hoofdbenchmark
    run_benchmark(args.n, graph_type=args.type, n_samples=args.samples,
                  edge_p=args.edge_p, p=args.p,
                  reorder=args.reorder, verbose=True)


if __name__ == '__main__':
    main()
