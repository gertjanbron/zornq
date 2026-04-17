#!/usr/bin/env python3
"""
B62: QAOA + Local Search Refinement.

Kernvraag: levert QAOA-sampling betere startpunten op voor local search
dan random initialisatie? Als ja, dan bevat de QAOA-distributie
nuttige informatie over de probleemstructuur.

Aanpak:
  1. Bouw cylinder-grid graaf (Lx x Ly)
  2. Vind optimale QAOA-parameters via grid search op volledige state vector
  3. Sample K bitstrings uit de QAOA-distributie |psi(g*,b*)|^2
  4. Repareer elke sample met steepest-descent local search
  5. Vergelijk met K random starts + dezelfde repair

Ablation-matrix:
  A. Random start + repair (baseline)
  B. QAOA-sampling + repair (de kerntest)
  C. Greedy degree-heuristic + repair (simpele deterministische baseline)
  D. Alleen QAOA verwachtingswaarde (geen sampling, geen repair)

Gebruik:
  python b62_qaoa_vs_ls.py
  python b62_qaoa_vs_ls.py --Lx 6 --Ly 3 --p 2 --samples 200
  python b62_qaoa_vs_ls.py --Lx 8 --Ly 3 --p 1 --samples 500
"""

import numpy as np
import math
import time
import argparse


# ============================================================
# Graaf
# ============================================================

class CylinderGraph:
    """Ongewogen cylinder-rooster Lx x Ly, optioneel met diagonalen (frustratie)."""

    def __init__(self, Lx, Ly, triangular=False):
        self.Lx = Lx
        self.Ly = Ly
        self.n = Lx * Ly
        self.triangular = triangular
        self.edges = []
        self.adj = {i: [] for i in range(self.n)}

        def add(u, v):
            self.edges.append((u, v))
            self.adj[u].append(v)
            self.adj[v].append(u)

        # Horizontale edges
        for x in range(Lx - 1):
            for y in range(Ly):
                add(x * Ly + y, (x + 1) * Ly + y)
        # Verticale edges
        for x in range(Lx):
            for y in range(Ly - 1):
                add(x * Ly + y, x * Ly + y + 1)
        # Diagonale edges (maakt driehoeken -> frustratie)
        if triangular:
            for x in range(Lx - 1):
                for y in range(Ly - 1):
                    # NE diagonaal: (x,y) -> (x+1,y+1)
                    add(x * Ly + y, (x + 1) * Ly + y + 1)

        self.n_edges = len(self.edges)

    def cut_value(self, bitstring):
        """Bereken cut waarde voor een bitstring (array van 0/1)."""
        total = 0
        for u, v in self.edges:
            if bitstring[u] != bitstring[v]:
                total += 1
        return total

    def local_flip_gain(self, bitstring, node):
        """Hoeveel cut verbetert als we node flippen."""
        current = bitstring[node]
        gain = 0
        for nb in self.adj[node]:
            if bitstring[nb] == current:
                gain += 1   # was same side, flip maakt cut
            else:
                gain -= 1   # was cut, flip verwijdert
        return gain


# ============================================================
# Brute force MaxCut
# ============================================================

def brute_force_maxcut(graph):
    """Exacte MaxCut via brute force (n <= 22)."""
    best_cut = 0
    best_bs = None
    for mask in range(1 << graph.n):
        bs = np.array([(mask >> i) & 1 for i in range(graph.n)], dtype=np.int8)
        c = graph.cut_value(bs)
        if c > best_cut:
            best_cut = c
            best_bs = bs.copy()
    return best_cut, best_bs


# ============================================================
# Steepest descent repair
# ============================================================

def steepest_descent(graph, bitstring, max_iters=500):
    """Steepest descent 1-flip local search. Retourneert verbeterde bitstring."""
    bs = bitstring.copy()
    for _ in range(max_iters):
        best_node, best_gain = -1, 0
        for node in range(graph.n):
            gain = graph.local_flip_gain(bs, node)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node < 0:
            break
        bs[best_node] = 1 - bs[best_node]
    return bs, graph.cut_value(bs)


# ============================================================
# Full-circuit QAOA state vector
# ============================================================

def qaoa_state_vector(graph, p, gammas, betas):
    """Bereken de volledige QAOA state vector |psi(gamma, beta)>.

    Returns: probability array (2^n,) met |psi|^2 per bitstring.
    """
    n = graph.n
    dim = 1 << n

    # Bouw cost Hamiltonian diagonaal: H_C |z> = C(z) |z>
    # Maar we gebruiken de ZZ-formulation: (1 - Z_i Z_j)/2 per edge
    # Fase: exp(-i gamma H_C) = exp(-i gamma sum_edges (1-ZZ)/2)
    # = globale fase * exp(i gamma/2 sum_edges Z_i Z_j)

    # Precompute ZZ diagonaal per edge
    bitstrings = np.arange(dim)
    zz_diag = np.zeros(dim, dtype=np.float32)
    for u, v in graph.edges:
        z_u = 1 - 2 * ((bitstrings >> u) & 1)  # +1 of -1
        z_v = 1 - 2 * ((bitstrings >> v) & 1)
        zz_diag += z_u.astype(np.float32) * z_v.astype(np.float32)

    # Start: |+>^n
    state = np.ones(dim, dtype=np.complex64) / np.sqrt(np.float32(dim))

    # QAOA lagen
    for layer in range(p):
        # Fase operator: exp(-i gamma (sum (1-ZZ)/2))
        # = exp(-i gamma n_edges/2) * exp(i gamma/2 * zz_diag)
        # De globale fase maakt niet uit voor sampling
        phase = np.exp(np.complex64(1j) * np.float32(gammas[layer] / 2) * zz_diag)
        state *= phase

        # Mixer: Rx(beta) op alle qubits
        cb = np.float32(math.cos(betas[layer]))
        msb = np.complex64(-1j * math.sin(betas[layer]))
        for q in range(n):
            s = state.reshape(1 << (n - q - 1), 2, 1 << q)
            tmp = cb * s[:, 0, :] + msb * s[:, 1, :]
            s[:, 1, :] = msb * s[:, 0, :] + cb * s[:, 1, :]
            s[:, 0, :] = tmp
            state = s.reshape(-1)

    probs = np.abs(state) ** 2
    return probs


def qaoa_expected_cut(graph, probs):
    """Bereken verwachte cut uit QAOA-verdeling."""
    n = graph.n
    dim = len(probs)
    bitstrings = np.arange(dim)
    total = 0.0
    for u, v in graph.edges:
        z_u = 1 - 2 * ((bitstrings >> u) & 1)
        z_v = 1 - 2 * ((bitstrings >> v) & 1)
        zz = (z_u * z_v).astype(np.float32)
        total += float(np.dot(probs, (1 - zz) / 2))
    return total


# ============================================================
# Grid search optimizer
# ============================================================

def optimize_qaoa(graph, p, n_gamma=20, n_beta=20):
    """Vind optimale QAOA-parameters via grid search (p=1,2).

    Returns: best_ratio, best_gammas, best_betas
    """
    if p == 1:
        best_cut = 0
        best_g, best_b = 0, 0
        for ig in range(n_gamma):
            gamma = (ig + 0.5) / n_gamma * np.pi / 2
            for ib in range(n_beta):
                beta = (ib + 0.5) / n_beta * np.pi / 2
                probs = qaoa_state_vector(graph, 1, [gamma], [beta])
                cut = qaoa_expected_cut(graph, probs)
                if cut > best_cut:
                    best_cut = cut
                    best_g, best_b = gamma, beta
        return best_cut / graph.n_edges, [best_g], [best_b]

    elif p == 2:
        # Warm-start: dupliceer p=1 optimum
        _, g1_list, b1_list = optimize_qaoa(graph, 1, n_gamma, n_beta)
        g1, b1 = g1_list[0], b1_list[0]

        # Mini grid search rond warm-start
        best_cut = 0
        best_gs, best_bs = [g1, g1], [b1, b1]
        offsets = np.linspace(-0.15, 0.15, 5)
        for dg1 in offsets:
            for dg2 in offsets:
                for db1 in offsets:
                    for db2 in offsets:
                        gs = [g1 + dg1, g1 + dg2]
                        bs = [b1 + db1, b1 + db2]
                        probs = qaoa_state_vector(graph, 2, gs, bs)
                        cut = qaoa_expected_cut(graph, probs)
                        if cut > best_cut:
                            best_cut = cut
                            best_gs, best_bs = list(gs), list(bs)
        return best_cut / graph.n_edges, best_gs, best_bs

    else:
        raise ValueError("Grid search only for p=1,2. Use progressive for p>2.")


# ============================================================
# Sampling strategies
# ============================================================

def sample_qaoa(probs, K, rng):
    """Sample K bitstrings uit QAOA-verdeling."""
    dim = len(probs)
    n = int(np.log2(dim))
    indices = rng.choice(dim, size=K, p=probs)
    samples = np.array([[(idx >> q) & 1 for q in range(n)] for idx in indices],
                       dtype=np.int8)
    return samples


def sample_random(n, K, rng):
    """K random bitstrings."""
    return rng.integers(0, 2, size=(K, n)).astype(np.int8)


def sample_greedy_degree(graph, K, rng):
    """Greedy degree-heuristic: hoogste-graad nodes eerst toewijzen."""
    n = graph.n
    degrees = [len(graph.adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    samples = []
    for _ in range(K):
        bs = np.zeros(n, dtype=np.int8)
        # Greedy: wijs toe aan de kant die de meeste cuts maakt
        for node in order:
            gain_0, gain_1 = 0, 0
            for nb in graph.adj[node]:
                if bs[nb] == 0:
                    gain_1 += 1  # node=1, nb=0 -> cut
                else:
                    gain_0 += 1  # node=0, nb=1 -> cut
            bs[node] = 1 if gain_1 >= gain_0 else 0
        # Kleine random perturbatie voor diversiteit
        n_flip = rng.integers(0, max(1, n // 5))
        flip_nodes = rng.choice(n, size=n_flip, replace=False)
        for f in flip_nodes:
            bs[f] = 1 - bs[f]
        samples.append(bs)
    return np.array(samples, dtype=np.int8)


# ============================================================
# Benchmark runner
# ============================================================

def run_benchmark(Lx, Ly, p, K, n_gamma, n_beta, seed=42, triangular=False):
    rng = np.random.default_rng(seed)
    graph = CylinderGraph(Lx, Ly, triangular=triangular)
    n = graph.n
    print("=" * 70)
    print("  B62: QAOA + Local Search Refinement")
    topo = "triangulair" if triangular else "vierkant"
    print("  Grid: %dx%d %s (%d qubits, %d edges)" % (Lx, Ly, topo, n, graph.n_edges))
    print("  QAOA diepte: p=%d, samples: K=%d" % (p, K))
    print("=" * 70)

    # --- Exacte oplossing ---
    exact_cut = None
    if n <= 22:
        print("\n  [1/5] Brute force exact MaxCut...")
        t0 = time.time()
        exact_cut, exact_bs = brute_force_maxcut(graph)
        print("    Exact optimum: %d/%d edges (ratio %.4f) in %.1fs" % (
            exact_cut, graph.n_edges, exact_cut / graph.n_edges, time.time() - t0))
    else:
        print("\n  [1/5] Brute force overgeslagen (n=%d > 22)" % n)

    # --- QAOA optimalisatie ---
    print("\n  [2/5] QAOA parameter-optimalisatie (p=%d)..." % p)
    t0 = time.time()
    ratio, gammas, betas = optimize_qaoa(graph, p, n_gamma, n_beta)
    opt_time = time.time() - t0
    print("    Ratio: %.6f" % ratio)
    print("    Gammas: %s" % gammas)
    print("    Betas:  %s" % betas)
    print("    Tijd: %.1fs" % opt_time)

    expected_cut = ratio * graph.n_edges
    if exact_cut:
        print("    Gap to exact: %.2f%%" % (100 * (1 - expected_cut / exact_cut)))

    # --- QAOA state vector + sampling ---
    print("\n  [3/5] QAOA state vector + sampling (%d samples)..." % K)
    t0 = time.time()
    probs = qaoa_state_vector(graph, p, gammas, betas)
    sv_time = time.time() - t0
    qaoa_samples = sample_qaoa(probs, K, rng)
    print("    State vector: %.3fs" % sv_time)

    # Diagnostiek: entropie en top-bitstrings
    nonzero = probs[probs > 1e-12]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    top_idx = np.argsort(probs)[-5:][::-1]
    print("    Shannon entropie: %.2f bits (max %d)" % (entropy, n))
    print("    Top-5 bitstrings:")
    for idx in top_idx:
        bs = np.array([(idx >> q) & 1 for q in range(n)], dtype=np.int8)
        cut = graph.cut_value(bs)
        print("      p=%.6f  cut=%d/%d (%.4f)" % (
            probs[idx], cut, graph.n_edges, cut / graph.n_edges))

    # --- Alle strategies draaien ---
    print("\n  [4/5] Ablation: sampling + repair...")

    random_samples = sample_random(n, K, rng)
    greedy_samples = sample_greedy_degree(graph, K, rng)

    strategies = {
        'A. Random + repair': random_samples,
        'B. QAOA + repair': qaoa_samples,
        'C. Greedy-deg + repair': greedy_samples,
    }

    results = {}
    for name, samples in strategies.items():
        t0 = time.time()
        cuts_before = []
        cuts_after = []
        for i in range(K):
            c_before = graph.cut_value(samples[i])
            cuts_before.append(c_before)
            _, c_after = steepest_descent(graph, samples[i])
            cuts_after.append(c_after)
        elapsed = time.time() - t0

        results[name] = {
            'cuts_before': cuts_before,
            'cuts_after': cuts_after,
            'time': elapsed,
        }
        avg_before = np.mean(cuts_before)
        avg_after = np.mean(cuts_after)
        best_after = max(cuts_after)
        n_optimal = sum(1 for c in cuts_after if exact_cut and c >= exact_cut)
        repair_gain = avg_after - avg_before

        ratio_before = avg_before / graph.n_edges
        ratio_after = avg_after / graph.n_edges

        print("    %-25s  voor=%.3f  na=%.3f  best=%d  repair=+%.1f  opt=%d/%d  (%.1fs)" % (
            name, ratio_before, ratio_after, best_after, repair_gain,
            n_optimal, K, elapsed))

    # --- Samenvattende tabel ---
    print("\n  [5/5] Samenvatting")
    print("  " + "-" * 68)
    print("  %-25s %8s %8s %8s %8s" % ("Methode", "Gem.cut", "Ratio", "Best", "% Opt"))
    print("  " + "-" * 68)

    # D. Alleen QAOA verwachting (geen sampling, geen repair)
    print("  %-25s %8.1f %8.4f %8s %8s" % (
        "D. QAOA verwachting", expected_cut, ratio, "-", "-"))

    for name in ['A. Random + repair', 'B. QAOA + repair', 'C. Greedy-deg + repair']:
        r = results[name]
        avg = np.mean(r['cuts_after'])
        best = max(r['cuts_after'])
        rat = avg / graph.n_edges
        if exact_cut:
            pct_opt = 100 * sum(1 for c in r['cuts_after'] if c >= exact_cut) / K
            print("  %-25s %8.1f %8.4f %8d %7.1f%%" % (name, avg, rat, best, pct_opt))
        else:
            print("  %-25s %8.1f %8.4f %8d %8s" % (name, avg, rat, best, "-"))

    if exact_cut:
        print("  %-25s %8d %8.4f %8s %8s" % (
            "Exact optimum", exact_cut, exact_cut / graph.n_edges, "-", "-"))
    print("  " + "-" * 68)

    # --- Head-to-head ---
    print("\n  Head-to-head (per sample, na repair):")
    qaoa_cuts = results['B. QAOA + repair']['cuts_after']
    random_cuts = results['A. Random + repair']['cuts_after']
    qaoa_wins = sum(1 for q, r in zip(qaoa_cuts, random_cuts) if q > r)
    random_wins = sum(1 for q, r in zip(qaoa_cuts, random_cuts) if r > q)
    ties = K - qaoa_wins - random_wins
    print("    QAOA wint: %d | Random wint: %d | Gelijk: %d" % (
        qaoa_wins, random_wins, ties))

    avg_qaoa = np.mean(qaoa_cuts)
    avg_rand = np.mean(random_cuts)
    if avg_rand > 0:
        print("    QAOA gem: %.2f | Random gem: %.2f | Verschil: %+.2f" % (
            avg_qaoa, avg_rand, avg_qaoa - avg_rand))

    print("\n" + "=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B62: QAOA + Local Search Refinement')
    parser.add_argument('--Lx', type=int, default=4,
                        help='Grid breedte (default 4)')
    parser.add_argument('--Ly', type=int, default=3,
                        help='Grid hoogte (default 3)')
    parser.add_argument('--p', type=int, default=1,
                        help='QAOA diepte (default 1)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Aantal samples K (default 200)')
    parser.add_argument('--ngamma', type=int, default=20,
                        help='Grid search resolutie gamma')
    parser.add_argument('--nbeta', type=int, default=20,
                        help='Grid search resolutie beta')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--triangular', action='store_true',
                        help='Voeg diagonale edges toe (gefrustreerd rooster)')
    args = parser.parse_args()

    run_benchmark(args.Lx, args.Ly, args.p, args.samples,
                  args.ngamma, args.nbeta, args.seed, args.triangular)
