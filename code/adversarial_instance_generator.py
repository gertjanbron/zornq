#!/usr/bin/env python3
"""
B109: Adversarial Instance Generator voor ZornQ Solvers

Genereert doelbewust grafen waar specifieke ZornQ-solvers falen of moeite
hebben. Elke familie richt zich op een bekende structurele zwakte:

Familie 1 — high_feedback_dense:
  Raakt B99 (Feedback-Edge Skeleton Solver). B99 decomposeert G in spanning
  tree + k feedback edges, en enumereert 2^k configuraties. Bij dense grafen
  (hoge cyclomatische complexiteit) is k ~ m-n+1, exponentieel duur.

Familie 2 — frustrated_antiferro:
  Raakt BLS/PA (lokale search). Oneven cycli met ±1 gewichten creëren
  gefrustreerde landschappen vol lokale optima. BLS raakt gevangen in
  metastabiele toestanden.

Familie 3 — planted_partition:
  Raakt alle heuristische solvers. Planted bisection met ruis: het optimum
  is bekend (de planting), maar de ruis maakt het informatietheorisch moeilijk
  te vinden. Bij p_in ≈ p_out zit je rond de detectability threshold.

Familie 4 — expander_ramanujan:
  Raakt MPS-QAOA en chi-gebaseerde methoden. Expanders hebben hoge
  entanglement/connectivity, waardoor chi snel groeit. Ook moeilijk voor
  GW (gap ≈ 3-5%).

Familie 5 — weighted_conflict:
  Raakt alle solvers die gewichten niet goed hanteren. Combinatie van grote
  en kleine gewichten creëert schaal-conflicten in de objectieffunctie.

Familie 6 — treewidth_barrier:
  Raakt B42 (treewidth DP). Grafen met gecontroleerd hoge treewidth maar
  klein n, zodat exact methoden net niet haalbaar zijn.

Familie 7 — chimera_topology:
  Raakt ordering-gebaseerde methoden (QAOA cilinder). Chimera-achtige
  topologie (D-Wave hardware-graaf) met niet-planaire kruisingen.

Referenties:
  [1] Benlic & Hao (2013) — BLS origineel
  [2] Dunning et al. (2018) — planted partition MaxCut
  [3] Lubotzky, Phillips, Sarnak (1988) — Ramanujan grafen
  [4] Boettcher (2023) — Ising spin glass fase-overgangen
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math

Edge = Tuple[int, int, float]


# =====================================================================
# HELPER FUNCTIES
# =====================================================================

def _edge_list_to_adj(n: int, edges: List[Edge]) -> Dict[int, List[Tuple[int, float]]]:
    """Bouw adjacency list van edge list."""
    adj = {i: [] for i in range(n)}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _graph_stats(n: int, edges: List[Edge]) -> Dict:
    """Bereken basisstatistieken van een graaf."""
    m = len(edges)
    degs = np.zeros(n)
    total_weight = 0.0
    for u, v, w in edges:
        degs[u] += 1
        degs[v] += 1
        total_weight += w
    return {
        'n': n,
        'm': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0,
        'avg_degree': degs.mean(),
        'max_degree': int(degs.max()),
        'min_degree': int(degs.min()),
        'total_weight': total_weight,
        'cyclomaticity': m - n + 1,
    }


def _make_instance(name: str, family: str, n: int, edges: List[Edge],
                   target_solver: str, difficulty_note: str,
                   planted_cut: Optional[float] = None,
                   **params) -> Dict:
    """Maak gestandaardiseerd instance-record."""
    stats = _graph_stats(n, edges)
    return {
        'name': name,
        'family': family,
        'n_nodes': n,
        'edges': [(int(u), int(v), float(w)) for u, v, w in edges],
        'target_solver': target_solver,
        'difficulty_note': difficulty_note,
        'planted_cut': planted_cut,
        'stats': stats,
        'params': params,
    }


# =====================================================================
# FAMILIE 1: HIGH FEEDBACK DENSE (target: B99)
# =====================================================================

def gen_high_feedback_dense(n: int = 50, density: float = 0.5,
                            seed: int = 42) -> Dict:
    """
    Dichte random graaf met hoog cyclomatisch getal.

    B99 moet 2^k feedback-edge configuraties enumereren met k = m - n + 1.
    Bij density=0.5 en n=50: k ≈ 575, ver boven de 2^20 grens.
    Fallback naar BLS-polish, maar verliest exactheid.
    """
    rng = np.random.RandomState(seed)
    edges = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.choice([-1.0, 1.0])
                edges.append((i, j, w))
                seen.add((i, j))

    # Zorg dat graaf connected is
    for i in range(n - 1):
        if (i, i + 1) not in seen:
            edges.append((i, i + 1, 1.0))
            seen.add((i, i + 1))

    return _make_instance(
        name=f'high_feedback_n{n}_d{density:.1f}_s{seed}',
        family='high_feedback_dense',
        n=n,
        edges=edges,
        target_solver='B99 (feedback-edge)',
        difficulty_note=f'Cyclomaticity ~{len(edges) - n + 1}, forces BLS fallback',
        density=density, seed=seed,
    )


# =====================================================================
# FAMILIE 2: FRUSTRATED ANTIFERROMAGNET (target: BLS/PA)
# =====================================================================

def gen_frustrated_antiferro(n: int = 60, p_triangle: float = 0.3,
                              seed: int = 42) -> Dict:
    """
    Triangulaire frustratie-rooster met random antiferromagnetische koppelingen.

    Elk driehoekig facet creëert frustratie (oneven cyclus met negatieve
    product). BLS en PA raken gevangen in lokale optima doordat elke
    flip een driehoek verstoort.
    """
    rng = np.random.RandomState(seed)
    # Start met een driehoeksrooster (grid + diagonalen)
    side = int(math.ceil(math.sqrt(n)))
    actual_n = side * side
    if actual_n < n:
        actual_n = (side + 1) * (side + 1)
        side += 1
    actual_n = min(actual_n, n)

    edges = []
    seen = set()

    def add_edge(u, v, w):
        if u < actual_n and v < actual_n and u != v:
            key = (min(u, v), max(u, v))
            if key not in seen:
                edges.append((key[0], key[1], w))
                seen.add(key)

    # Grid edges: positief (antiferro = +1 wil tegengestelde spins)
    for r in range(side):
        for c in range(side):
            node = r * side + c
            if node >= actual_n:
                continue
            # Rechts
            if c + 1 < side:
                right = r * side + c + 1
                if right < actual_n:
                    add_edge(node, right, 1.0)
            # Onder
            if r + 1 < side:
                below = (r + 1) * side + c
                if below < actual_n:
                    add_edge(node, below, 1.0)
            # Diagonaal (met kans p_triangle) — creëert driehoeken
            # Driehoek frustratie: 3 edges met w=+1 op 3 nodes kan niet
            # alle 3 gesneden worden (oneven cyclus).
            if r + 1 < side and c + 1 < side and rng.random() < p_triangle:
                diag = (r + 1) * side + c + 1
                if diag < actual_n:
                    add_edge(node, diag, 1.0)

    # Extra random frustratie-edges (mix +1/-1 voor schaalconflict)
    n_extra = int(actual_n * 0.3)
    for _ in range(n_extra):
        u = rng.randint(actual_n)
        v = rng.randint(actual_n)
        if u != v:
            w = rng.choice([-1.0, 1.0]) if rng.random() < 0.3 else 1.0
            add_edge(u, v, w)

    return _make_instance(
        name=f'frustrated_af_n{actual_n}_pt{p_triangle:.1f}_s{seed}',
        family='frustrated_antiferro',
        n=actual_n,
        edges=edges,
        target_solver='BLS/PA (local search)',
        difficulty_note='Dense frustrated landscape with many local optima',
        p_triangle=p_triangle, seed=seed,
    )


# =====================================================================
# FAMILIE 3: PLANTED PARTITION (target: alle heuristieken)
# =====================================================================

def gen_planted_partition(n: int = 100, p_in: float = 0.3, p_out: float = 0.7,
                          noise: float = 0.0, seed: int = 42) -> Dict:
    """
    Planted bisection MaxCut instantie.

    Verdeel nodes in twee gelijke groepen S₀, S₁. Edges binnen groep met
    kans p_in, edges tussen groepen met kans p_out. Het optimum is (bij
    benadering) de geplante partitie. Met noise>0 worden edges random
    hergewogen (±1).

    Moeilijkheid: als p_in/(p_in+p_out) → 0.5 wordt detectie
    informatietheorisch onmogelijk (Decelle et al. 2011).
    """
    rng = np.random.RandomState(seed)
    half = n // 2
    partition = np.zeros(n, dtype=int)
    partition[half:] = 1

    edges = []
    seen = set()
    planted_cut = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            same = partition[i] == partition[j]
            p = p_in if same else p_out
            if rng.random() < p:
                if noise > 0 and rng.random() < noise:
                    w = rng.choice([-1.0, 1.0])
                else:
                    w = 1.0
                edges.append((i, j, w))
                seen.add((i, j))
                if partition[i] != partition[j]:
                    planted_cut += w

    return _make_instance(
        name=f'planted_n{n}_in{p_in:.2f}_out{p_out:.2f}_s{seed}',
        family='planted_partition',
        n=n,
        edges=edges,
        target_solver='alle heuristieken',
        difficulty_note=f'Planted bisection, ratio p_in/p_out={p_in/p_out:.2f}',
        planted_cut=planted_cut,
        p_in=p_in, p_out=p_out, noise=noise, seed=seed,
    )


# =====================================================================
# FAMILIE 4: EXPANDER / PSEUDO-RAMANUJAN (target: MPS-QAOA, chi-methoden)
# =====================================================================

def gen_expander(n: int = 80, d: int = 5, seed: int = 42) -> Dict:
    """
    Random d-reguliere expander graaf.

    Expanders hebben spectral gap λ₁ - λ₂ > 0 (Ramanujan-grens: 2√(d-1)).
    Hoge connectiviteit → hoge entanglement → chi-muur voor MPS-QAOA.
    GW-gap is typisch 3-5% op d-reguliere grafen.

    Constructie via random perfect matchings (Bollobás, 1988).
    """
    rng = np.random.RandomState(seed)
    assert d >= 3, "d moet >= 3 zijn"
    assert n % 2 == 0, "n moet even zijn voor d-regulier"
    assert d < n, "d moet < n zijn"

    edges = []
    seen = set()

    # Bouw d random perfect matchings
    for matching_idx in range(d):
        perm = rng.permutation(n)
        for i in range(0, n, 2):
            u, v = int(perm[i]), int(perm[i + 1])
            key = (min(u, v), max(u, v))
            if key not in seen:
                edges.append((key[0], key[1], 1.0))
                seen.add(key)

    # Als n niet voldoende edges heeft, vul aan met random edges
    target_edges = n * d // 2
    attempts = 0
    while len(edges) < target_edges * 0.8 and attempts < target_edges * 10:
        u = rng.randint(n)
        v = rng.randint(n)
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in seen:
                edges.append((key[0], key[1], 1.0))
                seen.add(key)
        attempts += 1

    return _make_instance(
        name=f'expander_n{n}_d{d}_s{seed}',
        family='expander',
        n=n,
        edges=edges,
        target_solver='MPS-QAOA / chi-methoden',
        difficulty_note=f'Pseudo-{d}-regular expander, high connectivity → chi-wall',
        d=d, seed=seed,
    )


# =====================================================================
# FAMILIE 5: WEIGHTED CONFLICT (target: gewichtsgevoelige solvers)
# =====================================================================

def gen_weighted_conflict(n: int = 40, scale_ratio: float = 100.0,
                          seed: int = 42) -> Dict:
    """
    Graaf met multi-schaal gewichten: sommige edges wegen 1, andere
    wegen scale_ratio. Dit creëert een conflicterend landschap waar
    solvers die normaliseren of uniform samplen systematisch falen.

    De "belangrijke" (zware) edges vormen een pad, de "lichte" edges
    vormen een dichte wolk. Optimaal: snij alle zware edges + maximaal
    lichte. Maar heuristieken die op totaal gewicht kijken kunnen
    door de lichte wolk misleid worden.
    """
    rng = np.random.RandomState(seed)
    edges = []
    seen = set()

    # Zware pad (backbone) — moet volledig gesneden worden
    heavy_path = list(range(n // 4))
    for i in range(len(heavy_path) - 1):
        edges.append((heavy_path[i], heavy_path[i + 1], scale_ratio))
        seen.add((heavy_path[i], heavy_path[i + 1]))

    # Lichte dichte wolk op overige nodes
    cloud_start = n // 4
    cloud_size = n - cloud_start
    for i in range(cloud_start, n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                w = rng.uniform(0.1, 1.0)
                edges.append((i, j, w))
                seen.add((i, j))

    # Bridges: verbind pad met wolk (conflicterend)
    for i in range(min(5, len(heavy_path))):
        targets = rng.choice(range(cloud_start, n), size=2, replace=False)
        for t in targets:
            key = (min(heavy_path[i], t), max(heavy_path[i], t))
            if key not in seen:
                w = rng.uniform(0.5, 2.0)
                edges.append((key[0], key[1], w))
                seen.add(key)

    return _make_instance(
        name=f'weighted_conflict_n{n}_r{scale_ratio:.0f}_s{seed}',
        family='weighted_conflict',
        n=n,
        edges=edges,
        target_solver='alle solvers (gewichtsgevoelig)',
        difficulty_note=f'Multi-schaal: max/min gewicht ≈ {scale_ratio:.0f}×',
        scale_ratio=scale_ratio, seed=seed,
    )


# =====================================================================
# FAMILIE 6: TREEWIDTH BARRIER (target: B42 DP)
# =====================================================================

def gen_treewidth_barrier(k: int = 6, copies: int = 3,
                          seed: int = 42) -> Dict:
    """
    Graaf met gecontroleerde treewidth ≈ k via Kₖ-minor embedding.

    Bouw 'copies' kopieën van K_k (complete graaf op k nodes) en verbind
    ze met dunne bruggen. Dit geeft treewidth ≈ k-1, wat bij k≥6 de
    2^tw DP-grens van B42 bereikt.

    B42 runtime: O(n × 2^tw), dus tw=15 → ~32768× trager dan tw=5.
    """
    rng = np.random.RandomState(seed)
    n = k * copies
    edges = []
    seen = set()

    # Bouw Kₖ-kopieën
    for c in range(copies):
        offset = c * k
        for i in range(k):
            for j in range(i + 1, k):
                u, v = offset + i, offset + j
                w = rng.choice([-1.0, 1.0])
                edges.append((u, v, w))
                seen.add((u, v))

    # Verbind kopieën met dunne bruggen (behoudt treewidth)
    for c in range(copies - 1):
        u = c * k + rng.randint(k)
        v = (c + 1) * k + rng.randint(k)
        edges.append((u, v, 1.0))
        seen.add((min(u, v), max(u, v)))

    return _make_instance(
        name=f'tw_barrier_k{k}_c{copies}_s{seed}',
        family='treewidth_barrier',
        n=n,
        edges=edges,
        target_solver='B42 (treewidth DP)',
        difficulty_note=f'Treewidth ≈ {k-1}, {copies} Kₖ-cliques',
        k=k, copies=copies, seed=seed,
    )


# =====================================================================
# FAMILIE 7: CHIMERA TOPOLOGY (target: ordering-methoden)
# =====================================================================

def gen_chimera(L: int = 4, seed: int = 42) -> Dict:
    """
    Chimera-achtige graaf (D-Wave hardware topologie).

    L×L unit cells, elk met K_{4,4} bipartiet subgraaf. Horizontale en
    verticale koppelingen tussen cellen. Niet-planair, moeilijk te ordenen
    voor MPS.

    n = 8·L², edges = 4·L²·(4 + 1_hor + 1_vert) ≈ 24·L²
    """
    rng = np.random.RandomState(seed)
    n = 8 * L * L
    edges = []
    seen = set()

    def node(r, c, q):
        """Node index in unit cell (r,c), qubit q ∈ {0..7}."""
        return (r * L + c) * 8 + q

    for r in range(L):
        for c in range(L):
            # Intra-cell: K_{4,4} tussen qubits 0-3 en 4-7
            for i in range(4):
                for j in range(4, 8):
                    u, v = node(r, c, i), node(r, c, j)
                    w = rng.choice([-1.0, 1.0])
                    edges.append((u, v, w))
                    seen.add((u, v))

            # Horizontale koppelingen (qubits 0-3 verbinden cellen)
            if c + 1 < L:
                for q in range(4):
                    u = node(r, c, q)
                    v = node(r, c + 1, q)
                    w = rng.choice([-1.0, 1.0])
                    edges.append((u, v, w))
                    seen.add((u, v))

            # Verticale koppelingen (qubits 4-7 verbinden cellen)
            if r + 1 < L:
                for q in range(4, 8):
                    u = node(r, c, q)
                    v = node(r + 1, c, q)
                    w = rng.choice([-1.0, 1.0])
                    edges.append((u, v, w))
                    seen.add((u, v))

    return _make_instance(
        name=f'chimera_L{L}_s{seed}',
        family='chimera_topology',
        n=n,
        edges=edges,
        target_solver='MPS-QAOA / ordering-methoden',
        difficulty_note=f'Chimera L={L}: {n} nodes, non-planar K44 cells',
        L=L, seed=seed,
    )


# =====================================================================
# SUITE GENERATORS
# =====================================================================

def small_adversarial_suite(seed: int = 42) -> List[Dict]:
    """Kleine suite voor snelle tests (n ≤ 50)."""
    return [
        gen_high_feedback_dense(n=30, density=0.5, seed=seed),
        gen_frustrated_antiferro(n=25, p_triangle=0.4, seed=seed),
        gen_planted_partition(n=40, p_in=0.3, p_out=0.7, seed=seed),
        gen_expander(n=30, d=5, seed=seed),
        gen_weighted_conflict(n=30, scale_ratio=100.0, seed=seed),
        gen_treewidth_barrier(k=5, copies=3, seed=seed),
        gen_chimera(L=2, seed=seed),
    ]


def medium_adversarial_suite(seed: int = 42) -> List[Dict]:
    """Medium suite (n ~ 50-200)."""
    return [
        gen_high_feedback_dense(n=80, density=0.4, seed=seed),
        gen_high_feedback_dense(n=100, density=0.6, seed=seed + 1),
        gen_frustrated_antiferro(n=64, p_triangle=0.3, seed=seed),
        gen_frustrated_antiferro(n=100, p_triangle=0.5, seed=seed + 1),
        gen_planted_partition(n=100, p_in=0.3, p_out=0.7, seed=seed),
        gen_planted_partition(n=100, p_in=0.4, p_out=0.6, noise=0.1, seed=seed + 1),
        gen_planted_partition(n=200, p_in=0.35, p_out=0.65, seed=seed + 2),
        gen_expander(n=80, d=5, seed=seed),
        gen_expander(n=100, d=7, seed=seed + 1),
        gen_weighted_conflict(n=60, scale_ratio=100.0, seed=seed),
        gen_weighted_conflict(n=80, scale_ratio=1000.0, seed=seed + 1),
        gen_treewidth_barrier(k=6, copies=4, seed=seed),
        gen_treewidth_barrier(k=8, copies=3, seed=seed + 1),
        gen_chimera(L=3, seed=seed),
        gen_chimera(L=4, seed=seed + 1),
    ]


def scaling_suite(family: str, sizes: List[int], seed: int = 42,
                  **kwargs) -> List[Dict]:
    """Genereer een schaalreeks voor één familie."""
    generators = {
        'high_feedback_dense': gen_high_feedback_dense,
        'frustrated_antiferro': gen_frustrated_antiferro,
        'planted_partition': gen_planted_partition,
        'expander': gen_expander,
        'weighted_conflict': gen_weighted_conflict,
        'treewidth_barrier': gen_treewidth_barrier,
        'chimera_topology': gen_chimera,
    }
    gen = generators[family]
    results = []
    for i, n in enumerate(sizes):
        if family == 'treewidth_barrier':
            results.append(gen(k=n, copies=kwargs.get('copies', 3),
                              seed=seed + i))
        elif family == 'chimera_topology':
            results.append(gen(L=n, seed=seed + i))
        else:
            results.append(gen(n=n, seed=seed + i, **kwargs))
    return results


# =====================================================================
# ANALYSE FUNCTIES
# =====================================================================

def compute_planted_gap(instance: Dict, solver_cut: float) -> Optional[float]:
    """Bereken gap t.o.v. geplant optimum (indien beschikbaar)."""
    if instance.get('planted_cut') is not None:
        planted = instance['planted_cut']
        if planted > 0:
            return (planted - solver_cut) / planted
    return None


def classify_difficulty(instance: Dict) -> str:
    """Classificeer verwachte moeilijkheid op basis van structuur."""
    stats = instance['stats']
    n = stats['n']
    m = stats['m']
    density = stats['density']
    cyclo = stats['cyclomaticity']

    if cyclo > 20 and n > 25:
        return 'HARD_FOR_EXACT'
    if density > 0.3 and n > 50:
        return 'HARD_FOR_LOCAL_SEARCH'
    if stats['avg_degree'] >= 5 and n > 30:
        return 'HARD_FOR_MPS'
    return 'MODERATE'


def print_instance_summary(instance: Dict):
    """Print beknopte samenvatting van een instance."""
    stats = instance['stats']
    print(f"  {instance['name']:40s}  n={stats['n']:4d}  m={stats['m']:5d}  "
          f"density={stats['density']:.3f}  deg_avg={stats['avg_degree']:.1f}  "
          f"cyclo={stats['cyclomaticity']:5d}  target={instance['target_solver']}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    print("=== B109 Adversarial Instance Generator ===\n")

    print("--- Small Suite ---")
    for inst in small_adversarial_suite():
        print_instance_summary(inst)

    print("\n--- Medium Suite ---")
    for inst in medium_adversarial_suite():
        print_instance_summary(inst)

    print("\n--- Scaling: Expander n=30..200 ---")
    for inst in scaling_suite('expander', [30, 50, 80, 120, 200], d=5):
        print_instance_summary(inst)
