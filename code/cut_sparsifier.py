#!/usr/bin/env python3
"""
cut_sparsifier.py - Cut-Preserving Graph Sparsifier (B118)

Reduceert het aantal edges terwijl alle cuts tot (1 +/- epsilon) behouden
blijven. Versnelt alle downstream solvers (B99v2, B128 hybrid, QAOA lightcone).

Drie methoden:
  1. Effectieve-weerstand sampling (Spielman-Srivastava): theoretisch optimaal,
     O(m log n / eps^2) edges, maar vereist Laplaciaan-solve.
  2. Degree-gewogen sampling: goedkope proxy, sample edges proportioneel aan
     1/min(deg(u), deg(v)). Geen Laplaciaan nodig, werkt goed in de praktijk.
  3. Weight-thresholding: verwijder edges met |w| < threshold. Simpelste methode,
     niet cut-preserving maar vaak voldoende als preprocessing.

Voor MaxCut op +-1 Ising:
  - Edges met |w|=1 hebben gelijke kans om positief of negatief te zijn
  - Degree-gewogen sampling behoudt de frustratiestructuur
  - Herschaling: w_new = w_old / p_e (onvertekend)

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from collections import defaultdict
import time


def effective_resistance_sparsify(n_nodes, edges, epsilon=0.3, seed=42):
    """
    Spielman-Srivastava spectrale sparsifier.

    Berekent effectieve weerstanden via Laplaciaan pseudo-inverse,
    dan sample edges proportioneel aan w_e * R_e.

    Args:
        n_nodes: aantal nodes
        edges: list van (u, v, w) tuples
        epsilon: approximatie-parameter (kleiner = nauwkeuriger, meer edges)
        seed: random seed

    Returns:
        sparse_edges: list van (u, v, w_new) tuples
        info: dict met statistieken
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    m = len(edges)
    if m == 0:
        return [], {'method': 'effective_resistance', 'original_m': 0, 'sparse_m': 0}

    # Bouw Laplaciaan
    L = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for u, v, w in edges:
        u, v = int(u), int(v)
        aw = abs(w)
        L[u, u] += aw
        L[v, v] += aw
        L[u, v] -= aw
        L[v, u] -= aw

    # Pseudo-inverse via eigendecompositie
    # L is singulier (nullspace = constant vector), dus gebruik pinv
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Zet kleine eigenwaarden op 0 (nullspace)
        tol = 1e-10 * max(abs(eigenvalues))
        L_pinv = np.zeros_like(L)
        for i in range(n_nodes):
            if abs(eigenvalues[i]) > tol:
                L_pinv += (1.0 / eigenvalues[i]) * np.outer(eigenvectors[:, i], eigenvectors[:, i])
    except np.linalg.LinAlgError:
        # Fallback naar degree-gewogen als eigh faalt
        return degree_weighted_sparsify(n_nodes, edges, epsilon=epsilon, seed=seed)

    # Bereken effectieve weerstanden
    R_eff = np.zeros(m)
    for idx, (u, v, w) in enumerate(edges):
        u, v = int(u), int(v)
        # R_eff(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)
        diff = L_pinv[u, :] - L_pinv[v, :]
        R_eff[idx] = diff[u] - diff[v]
        R_eff[idx] = max(R_eff[idx], 0)  # numerieke stabiliteit

    # Sampling probabiliteiten: p_e = min(1, C * w_e * R_e * log(n) / eps^2)
    C = 4.0  # constante uit Spielman-Srivastava bewijs
    q = m  # target aantal samples
    target_m = int(C * n_nodes * np.log(max(n_nodes, 2)) / (epsilon ** 2))
    target_m = min(target_m, m)  # nooit meer dan origineel

    # Normaliseer sampling kansen
    scores = np.array([abs(w) * R_eff[idx] for idx, (u, v, w) in enumerate(edges)])
    total_score = scores.sum()
    if total_score < 1e-15:
        # Alle weerstanden 0 → graaf is disconnected of trivial
        return degree_weighted_sparsify(n_nodes, edges, epsilon=epsilon, seed=seed)

    probs = scores / total_score
    probs = np.clip(probs * target_m, 0, 1)  # cap bij 1

    # Sample
    sparse_edges = []
    edge_counts = defaultdict(float)
    for idx, (u, v, w) in enumerate(edges):
        if rng.random() < probs[idx]:
            # Herschaal gewicht: w_new = w / p_e
            p_e = max(probs[idx], 1e-15)
            w_new = float(w) / p_e
            key = (min(int(u), int(v)), max(int(u), int(v)))
            if key in edge_counts:
                # Combineer dubbele edges
                edge_counts[key] += w_new
            else:
                edge_counts[key] = w_new

    sparse_edges = [(u, v, w) for (u, v), w in edge_counts.items()]

    elapsed = time.time() - t0
    info = {
        'method': 'effective_resistance',
        'original_m': m,
        'sparse_m': len(sparse_edges),
        'reduction': 1.0 - len(sparse_edges) / max(m, 1),
        'epsilon': epsilon,
        'target_m': target_m,
        'time': elapsed,
        'mean_R_eff': float(np.mean(R_eff)),
        'max_R_eff': float(np.max(R_eff)),
    }

    return sparse_edges, info


def degree_weighted_sparsify(n_nodes, edges, epsilon=0.3, seed=42):
    """
    Degree-gewogen sparsifier (goedkope proxy voor effectieve weerstand).

    Sample edges met kans p_e = min(1, C / (eps^2 * min(deg(u), deg(v)))).
    Edges naar lage-graad nodes worden altijd behouden (structureel belangrijk).
    Edges tussen hoge-graad nodes worden gesampled.

    Args:
        n_nodes: aantal nodes
        edges: list van (u, v, w) tuples
        epsilon: approximatie-parameter
        seed: random seed

    Returns:
        sparse_edges: list van (u, v, w_new) tuples
        info: dict met statistieken
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    m = len(edges)
    if m == 0:
        return [], {'method': 'degree_weighted', 'original_m': 0, 'sparse_m': 0}

    # Bereken graden
    degree = defaultdict(int)
    for u, v, w in edges:
        degree[int(u)] += 1
        degree[int(v)] += 1

    # Sampling kansen
    C = 9.0 * np.log(max(n_nodes, 2))  # constante met log(n) factor
    sparse_edges = []

    for u, v, w in edges:
        u_int, v_int = int(u), int(v)
        min_deg = min(degree[u_int], degree[v_int])
        # Kans proportioneel aan 1/min_deg
        p_e = min(1.0, C / (epsilon ** 2 * max(min_deg, 1)))

        if rng.random() < p_e:
            # Herschaal: w_new = w / p_e (onvertekend)
            w_new = float(w) / p_e
            sparse_edges.append((u_int, v_int, w_new))

    elapsed = time.time() - t0
    info = {
        'method': 'degree_weighted',
        'original_m': m,
        'sparse_m': len(sparse_edges),
        'reduction': 1.0 - len(sparse_edges) / max(m, 1),
        'epsilon': epsilon,
        'time': elapsed,
        'avg_degree': 2 * m / max(n_nodes, 1),
    }

    return sparse_edges, info


def weight_threshold_sparsify(n_nodes, edges, threshold=0.01):
    """
    Simpele gewicht-drempel sparsifier.

    Verwijder edges met |w| < threshold. Geen herschaling (exacte methode
    als threshold klein genoeg is).

    Nuttig als preprocessing voor gewogen grafen met veel zwakke edges.
    """
    t0 = time.time()
    m = len(edges)
    sparse_edges = [(int(u), int(v), float(w)) for u, v, w in edges
                    if abs(float(w)) >= threshold]

    info = {
        'method': 'weight_threshold',
        'original_m': m,
        'sparse_m': len(sparse_edges),
        'reduction': 1.0 - len(sparse_edges) / max(m, 1),
        'threshold': threshold,
        'time': time.time() - t0,
    }
    return sparse_edges, info


def sparsify(n_nodes, edges, epsilon=0.3, method='auto', seed=42):
    """
    Hoofd-interface: kies automatisch de beste sparsificatie-methode.

    method='auto': kies op basis van grafgrootte
      - n <= 2000: effective_resistance (exact, O(n^2) geheugen)
      - n > 2000: degree_weighted (goedkoop, O(m))

    method='er': effective_resistance
    method='dw': degree_weighted
    method='wt': weight_threshold (epsilon wordt als threshold gebruikt)

    Returns:
        sparse_edges: list van (u, v, w_new)
        info: dict met statistieken
    """
    if method == 'auto':
        if n_nodes <= 2000:
            method = 'er'
        else:
            method = 'dw'

    if method == 'er':
        return effective_resistance_sparsify(n_nodes, edges, epsilon=epsilon, seed=seed)
    elif method == 'dw':
        return degree_weighted_sparsify(n_nodes, edges, epsilon=epsilon, seed=seed)
    elif method == 'wt':
        return weight_threshold_sparsify(n_nodes, edges, threshold=epsilon)
    else:
        raise ValueError("Onbekende methode: %s" % method)


def sparsified_maxcut(n_nodes, edges, solver_fn, epsilon=0.3,
                      method='auto', seed=42, verbose=False):
    """
    Wrapper: sparsify graaf, los MaxCut op sparsified versie op,
    evalueer op originele graaf.

    Args:
        n_nodes: aantal nodes
        edges: originele edges (u, v, w)
        solver_fn: functie(n_nodes, edges) -> (cut, assignment)
        epsilon: sparsificatie-parameter
        method: 'auto', 'er', 'dw', 'wt'
        seed: random seed
        verbose: print voortgang

    Returns:
        cut_original: cut-waarde op originele graaf
        assignment: dict {node: 0/1}
        info: dict met statistieken
    """
    t0 = time.time()

    # Sparsify
    sparse_edges, sparse_info = sparsify(n_nodes, edges, epsilon=epsilon,
                                          method=method, seed=seed)

    if verbose:
        print("  Sparsifier [%s]: %d -> %d edges (%.0f%% reductie, %.3fs)" % (
            sparse_info['method'], sparse_info['original_m'],
            sparse_info['sparse_m'], sparse_info['reduction'] * 100,
            sparse_info['time']))

    # Los op op sparsified graaf
    t_solve = time.time()
    cut_sparse, assignment = solver_fn(n_nodes, sparse_edges)
    solve_time = time.time() - t_solve

    # Evalueer op originele graaf
    cut_original = 0.0
    for u, v, w in edges:
        u, v = int(u), int(v)
        if assignment.get(u, 0) != assignment.get(v, 0):
            cut_original += float(w)

    total_time = time.time() - t0
    info = {
        'sparse_info': sparse_info,
        'cut_sparse': cut_sparse,
        'cut_original': cut_original,
        'solve_time': solve_time,
        'total_time': total_time,
    }

    if verbose:
        print("  Solver: %.1f op sparse, %.1f op origineel (%.3fs)" % (
            cut_sparse, cut_original, solve_time))

    return cut_original, assignment, info


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='B118: Cut-Preserving Sparsifier')
    parser.add_argument('--n', type=int, default=100,
                        help='Aantal nodes voor test grid')
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--method', default='auto',
                        choices=['auto', 'er', 'dw', 'wt'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Maak test grid
    Lx = int(np.sqrt(args.n))
    Ly = max(args.n // Lx, 2)
    n = Lx * Ly
    edges = []
    rng = np.random.default_rng(args.seed)
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x + 1) * Ly + y, rng.choice([-1.0, 1.0])))
            if y + 1 < Ly:
                edges.append((i, x * Ly + y + 1, rng.choice([-1.0, 1.0])))

    print("Grid %dx%d: n=%d, m=%d" % (Lx, Ly, n, len(edges)))

    sparse_edges, info = sparsify(n, edges, epsilon=args.epsilon,
                                   method=args.method, seed=args.seed)

    print("Sparsified: %d edges (%.0f%% reductie)" % (
        info['sparse_m'], info['reduction'] * 100))
    print("Methode: %s, tijd: %.3fs" % (info['method'], info['time']))
