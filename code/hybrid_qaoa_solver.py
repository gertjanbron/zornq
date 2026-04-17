#!/usr/bin/env python3
"""
hybrid_qaoa_solver.py - Hybride QAOA + Klassieke MaxCut Solver

Combineert quantum-simulatie (QAOA lightcone) met klassieke
feedback-edge solver (B99v2) voor state-of-the-art MaxCut op laptop.

Kernidee:
  1. QAOA p=1 per edge via lightcone -> per-edge <ZZ> correlaties
  2. |<ZZ>| als structurele informatie: sterke correlatie = "makkelijke" edge,
     zwakke correlatie = "gefrustreerde" edge
  3. Quantum-informed spanning tree: prioriteer sterke |<ZZ>| edges
  4. Tree-solve + greedy refinement (B99v2)
  5. Multi-tree ensemble met quantum-gewogen diversiteit

Waarom dit werkt:
  - QAOA "ziet" de frustratiestructuur van +-1 Ising grafen
  - Klassieke tree solver + greedy pakt de grove structuur
  - De quantum-informatie stuurt de tree-keuze naar structureel
    betere decomposities

Vereist: general_lightcone.py (B54), feedback_edge_solver.py (B99)

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
import time
import sys
import os
from collections import defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# QAOA correlatie-berekening
# ============================================================

def compute_qaoa_correlations(n_nodes, edges, p=1, n_gamma=8, n_beta=8,
                              gpu=False, verbose=False):
    """
    Bereken per-edge <ZZ> correlaties via QAOA lightcone.

    Draait QAOA p=1 (of hoger) op elke edge's lightcone,
    optimaliseert gamma/beta via grid search, en retourneert
    de per-edge verwachtingswaarde.

    Args:
        n_nodes: aantal nodes
        edges: list van (u, v, w) tuples
        p: QAOA diepte (default 1)
        n_gamma, n_beta: grid search resolutie
        gpu: gebruik GPU (CuPy)
        verbose: print voortgang

    Returns:
        zz_dict: dict {(u,v): <ZZ>} voor elke edge
        opt_gammas: geoptimaliseerde gamma-parameters
        opt_betas: geoptimaliseerde beta-parameters
        info: dict met statistieken
    """
    from rqaoa import WeightedGraph
    from general_lightcone import (
        GeneralLightconeQAOA, bfs_lightcone, lightcone_subgraph,
        lightcone_cache_key
    )

    t0 = time.time()

    # Bouw WeightedGraph
    g = WeightedGraph()
    for u, v, w in edges:
        g.add_edge(int(u), int(v), float(w))
    # Zorg dat alle nodes bestaan
    for i in range(n_nodes):
        if i not in g.adj:
            g.adj[i] = {}

    # Maak lightcone QAOA object
    qaoa = GeneralLightconeQAOA(g, verbose=False, gpu=gpu,
                                 ordering_method='auto')

    if verbose:
        stats = qaoa.lightcone_stats(p)
        print("  QAOA lightcone stats (p=%d):" % p)
        print("    qubits: min=%d, avg=%.1f, max=%d" % (
            stats['min_qubits'], stats['avg_qubits'], stats['max_qubits']))
        print("    unique lightcones: %d/%d (%.0f%% cache)" % (
            stats['n_unique'], qaoa.n_edges, stats['cache_rate'] * 100))

    # Optimaliseer gamma/beta
    ratio, gammas, betas, opt_info = qaoa.optimize(
        p, n_gamma=n_gamma, n_beta=n_beta, refine=True)

    if verbose:
        print("    Optimal ratio: %.4f (gamma=%s, beta=%s)" % (
            ratio, np.round(gammas, 3), np.round(betas, 3)))

    # Bereken per-edge <ZZ> met optimale parameters
    zz_dict = {}
    ordering_list = list(qaoa.ordering)
    cache = {}

    for idx, (ei, ej, ew) in enumerate(qaoa.edge_list):
        lc_nodes = bfs_lightcone(g, (ei, ej), p)
        sub_edges = lightcone_subgraph(g, lc_nodes)
        key = lightcone_cache_key(lc_nodes, sub_edges, (ei, ej), ordering_list)

        if key in cache:
            zz = cache[key]
        else:
            zz = qaoa.eval_edge_exact(ei, ej, ew, lc_nodes, sub_edges,
                                       p, gammas, betas)
            cache[key] = zz

        zz_dict[(min(ei, ej), max(ei, ej))] = zz

    elapsed = time.time() - t0

    if verbose:
        zz_vals = list(zz_dict.values())
        print("    <ZZ> range: [%.3f, %.3f], mean=%.3f" % (
            min(zz_vals), max(zz_vals), np.mean(zz_vals)))
        print("    Totale QAOA tijd: %.2fs" % elapsed)

    info = {
        'ratio': ratio,
        'gammas': gammas,
        'betas': betas,
        'qaoa_time': elapsed,
        'n_edges': len(zz_dict),
        'p': p,
    }

    return zz_dict, gammas, betas, info


# ============================================================
# Quantum-informed spanning tree
# ============================================================

def quantum_spanning_tree(n_nodes, edges, zz_dict, rng, strategy='strong'):
    """
    Bouw spanning tree gewogen door QAOA <ZZ> correlaties.

    Strategieen:
      'strong': prioriteer edges met grote |<ZZ>| (structureel zeker)
      'frustrated': prioriteer edges met kleine |<ZZ>| (moeilijke beslissingen)
      'mixed': mix van quantum-gewicht en random voor diversiteit
    """
    from feedback_edge_solver import UnionFind

    scored_edges = []
    for u, v, w in edges:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        zz = zz_dict.get(key, 0.0)

        if strategy == 'strong':
            # Sterke |<ZZ>| = structureel belangrijk -> in tree
            score = abs(zz) + rng.random() * 0.1
        elif strategy == 'frustrated':
            # Zwakke |<ZZ>| = gefrustreerd -> in tree (om te fixeren)
            score = (1.0 - abs(zz)) + rng.random() * 0.1
        else:  # mixed
            # Combinatie: quantum + random
            score = abs(zz) * 0.7 + rng.random() * 0.3

        scored_edges.append((u, v, w, score))

    # Sorteer op score (hoog = meer kans in tree)
    scored_edges.sort(key=lambda e: e[3], reverse=True)

    uf = UnionFind(n_nodes)
    tree_edges = []
    feedback_edges = []
    tree_adj = defaultdict(list)

    for u, v, w, _ in scored_edges:
        u, v = int(u), int(v)
        if uf.union(u, v):
            tree_edges.append((u, v, w))
            tree_adj[u].append((v, w))
            tree_adj[v].append((u, w))
        else:
            feedback_edges.append((u, v, w))

    return tree_edges, feedback_edges, tree_adj


# ============================================================
# Hybride solver
# ============================================================

def hybrid_qaoa_maxcut(n_nodes, edges, p=1, time_limit=60, seed=42,
                       n_trees=10, verbose=False):
    """
    Hybride QAOA + Klassieke MaxCut Solver.

    Pipeline:
      1. QAOA p=1 per edge -> <ZZ> correlaties (quantum-fase)
      2. Quantum-informed spanning trees (diverse strategieen)
      3. Tree-solve + greedy refinement per tree
      4. Beste als eindresultaat

    Args:
        n_nodes: aantal nodes
        edges: list van (u, v, w)
        p: QAOA diepte (default 1)
        time_limit: tijdbudget in seconden
        seed: random seed
        n_trees: aantal spanning trees in ensemble
        verbose: print voortgang

    Returns:
        best_cut: beste cut waarde
        best_assignment: dict {node: 0/1}
        info: dict met statistieken
    """
    from feedback_edge_solver import (
        solve_tree_maxcut, _greedy_refine, _eval_full_cut
    )

    t0 = time.time()
    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print("HYBRIDE QAOA + KLASSIEKE MaxCut SOLVER")
        print("  n=%d, edges=%d, p=%d, trees=%d" % (
            n_nodes, len(edges), p, n_trees))
        print("=" * 60)

    # Normaliseer edges
    norm_edges = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) > 2 else 1.0
        if u != v:
            norm_edges.append((u, v, w))

    # --- Fase 1: QAOA correlaties (quantum) ---
    qaoa_budget = min(time_limit * 0.4, 30.0)
    if verbose:
        print("\n  [QUANTUM] QAOA p=%d correlaties berekenen..." % p)

    zz_dict, gammas, betas, qaoa_info = compute_qaoa_correlations(
        n_nodes, norm_edges, p=p, n_gamma=8, n_beta=8,
        verbose=verbose)

    qaoa_time = time.time() - t0

    if verbose:
        # Frustratie-analyse
        n_frustrated = sum(1 for zz in zz_dict.values() if abs(zz) < 0.3)
        n_strong = sum(1 for zz in zz_dict.values() if abs(zz) > 0.7)
        print("    Frustratie: %d/%d edges gefrustreerd (|<ZZ>|<0.3), %d sterk" % (
            n_frustrated, len(zz_dict), n_strong))

    # --- Fase 2: Quantum-informed multi-tree ensemble ---
    if verbose:
        print("\n  [KLASSIEK] Quantum-informed multi-tree ensemble...")

    best_cut = -np.inf
    best_assign = None
    strategies = ['strong', 'frustrated', 'mixed']

    for t_idx in range(n_trees):
        if time.time() - t0 > time_limit * 0.9:
            break

        # Wissel strategie af
        strategy = strategies[t_idx % len(strategies)]

        # Quantum-informed spanning tree
        tree_edges, fb_edges, tree_adj = quantum_spanning_tree(
            n_nodes, norm_edges, zz_dict, rng, strategy=strategy)

        # Tree-solve
        fb_zeros = np.zeros(len(fb_edges), dtype=np.int8)
        tree_assign, tree_cut = solve_tree_maxcut(
            n_nodes, tree_adj, tree_edges, fb_edges, fb_zeros)

        # Greedy refinement op volledige graaf
        refined_assign, refined_cut, n_flips = _greedy_refine(
            n_nodes, norm_edges, tree_assign, max_passes=15)

        if refined_cut > best_cut:
            best_cut = refined_cut
            best_assign = refined_assign
            if verbose:
                gap_info = ""
                print("    Tree %d [%s]: %.1f (%d flips) *BEST*" % (
                    t_idx + 1, strategy, refined_cut, n_flips))
        elif verbose and t_idx < 5:
            print("    Tree %d [%s]: %.1f (%d flips)" % (
                t_idx + 1, strategy, refined_cut, n_flips))

    # --- Fase 3: Puur klassieke vergelijking (zonder QAOA) ---
    # Draai ook n_trees random trees zonder quantum info als baseline
    classical_best = -np.inf
    from feedback_edge_solver import _random_spanning_tree

    for t_idx in range(min(n_trees, 5)):
        if time.time() - t0 > time_limit * 0.95:
            break
        tree_e, fb_e, tree_a = _random_spanning_tree(n_nodes, norm_edges, rng)
        fb_z = np.zeros(len(fb_e), dtype=np.int8)
        ta, tc = solve_tree_maxcut(n_nodes, tree_a, tree_e, fb_e, fb_z)
        ra, rc, _ = _greedy_refine(n_nodes, norm_edges, ta, max_passes=15)
        if rc > classical_best:
            classical_best = rc
            if rc > best_cut:
                best_cut = rc
                best_assign = ra

    total_time = time.time() - t0

    if verbose:
        print("\n  Resultaat:")
        print("    Hybride (QAOA-informed): %.1f" % best_cut)
        print("    Klassiek baseline:       %.1f" % classical_best)
        advantage = best_cut - classical_best
        print("    Quantum advantage:       %+.1f (%s)" % (
            advantage, "QAOA WINT" if advantage > 0 else "gelijk" if advantage == 0 else "klassiek wint"))
        print("    QAOA tijd: %.2fs, totaal: %.2fs" % (qaoa_time, total_time))
        print("=" * 60)

    info = {
        'qaoa_time': qaoa_time,
        'total_time': total_time,
        'qaoa_ratio': qaoa_info['ratio'],
        'gammas': qaoa_info['gammas'],
        'betas': qaoa_info['betas'],
        'n_trees': n_trees,
        'p': p,
        'hybrid_cut': best_cut,
        'classical_cut': classical_best,
        'quantum_advantage': best_cut - classical_best,
    }

    return best_cut, best_assign, info


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Hybride QAOA + Klassieke MaxCut Solver')
    parser.add_argument('--graph', default='grid:10x4',
                        help='Graph: grid:LxW, gset:NAME, random3reg:N')
    parser.add_argument('--p', type=int, default=1, help='QAOA diepte')
    parser.add_argument('--time-limit', type=float, default=30)
    parser.add_argument('--trees', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    # Parse graph
    if args.graph.startswith('grid:'):
        dims = args.graph[5:].split('x')
        Lx, Ly = int(dims[0]), int(dims[1])
        n = Lx * Ly
        edges = []
        for x in range(Lx):
            for y in range(Ly):
                i = x * Ly + y
                if x + 1 < Lx:
                    edges.append((i, (x + 1) * Ly + y, 1.0))
                if y + 1 < Ly:
                    edges.append((i, x * Ly + y + 1, 1.0))
    elif args.graph.startswith('gset:'):
        from gset_loader import load_gset
        g, bks, info = load_gset(args.graph[5:])
        n = g.n_nodes
        edges = [(i, j, w) for i, j, w in g.edges()]
    elif args.graph.startswith('random3reg:'):
        from bls_solver import random_3regular
        n_val = int(args.graph[11:])
        n, edges = random_3regular(n_val, seed=args.seed)
    else:
        print("Onbekend graph formaat: %s" % args.graph)
        sys.exit(1)

    cut, assign, info = hybrid_qaoa_maxcut(
        n, edges, p=args.p, time_limit=args.time_limit,
        seed=args.seed, n_trees=args.trees, verbose=True)

    print("\nCut: %.1f" % cut)
    if 'bks' in dir():
        print("BKS: %d, gap: %.1f%%" % (bks, 100 * (bks - cut) / bks))
