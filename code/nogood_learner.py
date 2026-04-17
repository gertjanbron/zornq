#!/usr/bin/env python3
"""
B107: Quantum Nogood Learning voor MaxCut

Een progressief lerende solver die "nogoods" — verboden lokale toewijzingspatronen —
ontdekt uit exacte patches, B&B runs, en repair-iteraties, en deze gebruikt om
toekomstige solver-runs te versnellen.

Kernidee (SAT/constraint solving):
  Een "nogood" is een partieel assignment dat bewezen suboptimaal is.
  Bijvoorbeeld: als we WETEN dat nodes {2,5,7} in partitie (0,1,0) minstens
  3 cut-waarde kost t.o.v. het optimum, dan kunnen we die configuratie vermijden.

Drie bronnen van nogoods:
  1. EXACT: Treewidth-DP (B42) op kleine subgrafen levert optimale oplossingen.
     Elk niet-optimaal partial assignment is een nogood.
  2. BRANCH-AND-BOUND: B73 QBB prune nodes zijn nogoods.
  3. HEURISTIC: BLS/PA lokale optima die verbeteren na perturbatie onthullen
     welke lokale patronen "vast" zaten.

Gebruik:
  - Nogoods filteren de zoekruimte van BLS (tabu op nogood-patronen)
  - Nogoods sturen PA resampling (penalty op nogood-configuraties)
  - Nogoods versterken B99 tree-selectie (vermijd feedback-edges met nogoods)

Referenties:
  [1] Marques-Silva & Sakallah (1999) — GRASP/CDCL nogood learning
  [2] Gomes et al. (2008) — Constraint-based combinatorial optimization
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import time
import hashlib


# =====================================================================
# DATA STRUCTUREN
# =====================================================================

Edge = Tuple[int, int, float]


@dataclass(frozen=True)
class Nogood:
    """Een verboden partieel assignment.

    nodes: frozenset van betrokken node-indices
    assignment: tuple van (node, value) paren, gesorteerd op node
    cost_gap: hoeveel slechter dit pattern is dan het beste bekende alternatief
    source: waar deze nogood vandaan komt ('exact', 'bnb', 'heuristic', 'repair')
    subgraph_hash: hash van de subgraaf (voor snelle lookup)
    """
    nodes: FrozenSet[int]
    assignment: Tuple[Tuple[int, int], ...]  # ((node, 0|1), ...)
    cost_gap: float  # >= 0, hoe suboptimaal
    source: str  # 'exact', 'bnb', 'heuristic', 'repair'

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def pattern_key(self) -> Tuple[Tuple[int, int], ...]:
        """Unieke sleutel voor dit assignment-pattern."""
        return self.assignment

    def matches(self, full_assignment: Dict[int, int]) -> bool:
        """Check of een volledig assignment dit nogood-pattern bevat."""
        for node, val in self.assignment:
            if full_assignment.get(node) != val:
                return False
        return True

    def flipped(self) -> 'Nogood':
        """Complement: flip alle bits (equivalent onder Z2-symmetrie)."""
        flipped_assign = tuple((n, 1 - v) for n, v in self.assignment)
        return Nogood(
            nodes=self.nodes,
            assignment=flipped_assign,
            cost_gap=self.cost_gap,
            source=self.source,
        )


@dataclass
class NogoodDB:
    """Database van geleerde nogoods met efficiënte lookup."""
    nogoods: List[Nogood] = field(default_factory=list)
    # Index: node -> lijst van nogoods die deze node bevatten
    _node_index: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    # Set van pattern_keys voor deduplicatie
    _seen: Set[Tuple[Tuple[int, int], ...]] = field(default_factory=set)
    # Statistieken
    n_exact: int = 0
    n_bnb: int = 0
    n_heuristic: int = 0
    n_repair: int = 0
    n_duplicates_skipped: int = 0

    def add(self, nogood: Nogood) -> bool:
        """Voeg een nogood toe. Return True als nieuw, False als duplicate."""
        key = nogood.pattern_key
        # Check ook het complement (Z2 symmetrie)
        flipped_key = nogood.flipped().pattern_key
        if key in self._seen or flipped_key in self._seen:
            self.n_duplicates_skipped += 1
            return False

        idx = len(self.nogoods)
        self.nogoods.append(nogood)
        self._seen.add(key)
        for node in nogood.nodes:
            self._node_index[node].append(idx)

        if nogood.source == 'exact':
            self.n_exact += 1
        elif nogood.source == 'bnb':
            self.n_bnb += 1
        elif nogood.source == 'heuristic':
            self.n_heuristic += 1
        elif nogood.source == 'repair':
            self.n_repair += 1
        return True

    def lookup_node(self, node: int) -> List[Nogood]:
        """Alle nogoods die een specifieke node bevatten."""
        return [self.nogoods[i] for i in self._node_index.get(node, [])]

    def lookup_edge(self, u: int, v: int) -> List[Nogood]:
        """Alle nogoods die BEIDE nodes van een edge bevatten."""
        u_set = set(self._node_index.get(u, []))
        v_set = set(self._node_index.get(v, []))
        both = u_set & v_set
        return [self.nogoods[i] for i in both]

    def count_violations(self, assignment: Dict[int, int]) -> int:
        """Tel hoeveel nogoods geschonden worden door een assignment."""
        count = 0
        for ng in self.nogoods:
            if ng.matches(assignment):
                count += 1
        return count

    def violation_penalty(self, assignment: Dict[int, int]) -> float:
        """Som van cost_gaps van geschonden nogoods."""
        penalty = 0.0
        for ng in self.nogoods:
            if ng.matches(assignment):
                penalty += ng.cost_gap
        return penalty

    def filter_by_size(self, max_size: int) -> List[Nogood]:
        """Alleen nogoods met ≤ max_size nodes."""
        return [ng for ng in self.nogoods if ng.size <= max_size]

    def filter_by_gap(self, min_gap: float) -> List[Nogood]:
        """Alleen nogoods met cost_gap >= min_gap."""
        return [ng for ng in self.nogoods if ng.cost_gap >= min_gap]

    @property
    def total(self) -> int:
        return len(self.nogoods)

    def summary(self) -> Dict:
        """Statistieken over de database."""
        sizes = [ng.size for ng in self.nogoods] if self.nogoods else [0]
        gaps = [ng.cost_gap for ng in self.nogoods] if self.nogoods else [0]
        return {
            'total': self.total,
            'n_exact': self.n_exact,
            'n_bnb': self.n_bnb,
            'n_heuristic': self.n_heuristic,
            'n_repair': self.n_repair,
            'n_duplicates_skipped': self.n_duplicates_skipped,
            'avg_size': np.mean(sizes),
            'max_size': max(sizes),
            'avg_gap': np.mean(gaps),
            'max_gap': max(gaps),
            'n_indexed_nodes': len(self._node_index),
        }


# =====================================================================
# NOGOOD EXTRACTIE: EXACT (via brute-force op kleine subgrafen)
# =====================================================================

def extract_exact_nogoods(n: int, edges: List[Edge],
                           subgraph_nodes: List[int],
                           min_gap: float = 0.5) -> List[Nogood]:
    """
    Extract nogoods uit een exacte oplossing van een subgraaf.

    Brute-force alle 2^k assignments op de subgraaf, vind het optimum,
    en markeer alle assignments die >= min_gap slechter zijn als nogoods.

    Args:
        n: totaal aantal nodes in de volledige graaf
        edges: alle edges
        subgraph_nodes: nodes in de subgraaf (max ~20)
        min_gap: minimum cost-verschil om als nogood te registreren

    Returns:
        Lijst van Nogood objecten
    """
    k = len(subgraph_nodes)
    if k > 22:
        return []  # te groot voor brute force

    node_set = set(subgraph_nodes)
    node_list = sorted(subgraph_nodes)
    node_idx = {v: i for i, v in enumerate(node_list)}

    # Filter edges die volledig binnen de subgraaf liggen
    sub_edges = [(u, v, w) for u, v, w in edges
                 if u in node_set and v in node_set]

    if not sub_edges:
        return []

    # Brute force alle assignments
    best_cut = -np.inf
    cuts = np.zeros(2 ** k)

    for s in range(2 ** k):
        cut = 0.0
        for u, v, w in sub_edges:
            bu = (s >> node_idx[u]) & 1
            bv = (s >> node_idx[v]) & 1
            if bu != bv:
                cut += w
        cuts[s] = cut
        if cut > best_cut:
            best_cut = cut

    # Nogoods: alle assignments met gap >= min_gap
    nogoods = []
    for s in range(2 ** k):
        gap = best_cut - cuts[s]
        if gap >= min_gap:
            assign = tuple((node_list[i], (s >> i) & 1) for i in range(k))
            ng = Nogood(
                nodes=frozenset(node_list),
                assignment=assign,
                cost_gap=gap,
                source='exact',
            )
            nogoods.append(ng)

    return nogoods


def extract_edge_nogoods(edges: List[Edge], min_gap: float = 0.1) -> List[Nogood]:
    """
    Extract triviale 2-node nogoods: voor elke edge met w>0 is (u=same, v=same)
    een nogood met gap=w. Voor w<0 is (u=diff, v=diff) een nogood.

    Dit zijn de eenvoudigste nogoods maar ze zijn altijd correct.
    """
    nogoods = []
    for u, v, w in edges:
        if w > min_gap:
            # Positief gewicht: gelijke toewijzing is suboptimaal
            for val in [0, 1]:
                ng = Nogood(
                    nodes=frozenset([u, v]),
                    assignment=tuple(sorted([(u, val), (v, val)])),
                    cost_gap=w,
                    source='exact',
                )
                nogoods.append(ng)
        elif w < -min_gap:
            # Negatief gewicht: verschillende toewijzing is suboptimaal
            for val in [0, 1]:
                ng = Nogood(
                    nodes=frozenset([u, v]),
                    assignment=tuple(sorted([(u, val), (v, 1 - val)])),
                    cost_gap=-w,
                    source='exact',
                )
                nogoods.append(ng)
    return nogoods


# =====================================================================
# NOGOOD EXTRACTIE: HEURISTIC (uit BLS/PA runs)
# =====================================================================

def extract_heuristic_nogoods(assignment_before: Dict[int, int],
                               assignment_after: Dict[int, int],
                               cut_before: float,
                               cut_after: float,
                               edges: List[Edge],
                               max_size: int = 5) -> List[Nogood]:
    """
    Leer nogoods uit een verbetering: welke lokale patronen waren "fout"?

    Vergelijk een assignment vóór en na een verbetering (BLS perturbatie,
    PA resample, etc.). De geflipte nodes bevatten het "probleem".

    Args:
        assignment_before: assignment vóór verbetering
        assignment_after: assignment ná verbetering
        cut_before: cut waarde vóór
        cut_after: cut waarde ná (moet beter zijn)
        edges: alle edges
        max_size: maximale nogood grootte

    Returns:
        Lijst van Nogood objecten
    """
    if cut_after <= cut_before:
        return []  # geen verbetering

    gap = cut_after - cut_before

    # Vind geflipte nodes
    flipped = set()
    for node in assignment_before:
        if assignment_before[node] != assignment_after.get(node, -1):
            flipped.add(node)

    if not flipped or len(flipped) > max_size:
        return []

    # Het "before" pattern op de geflipte nodes is een nogood
    assign = tuple(sorted((n, assignment_before[n]) for n in flipped))
    ng = Nogood(
        nodes=frozenset(flipped),
        assignment=assign,
        cost_gap=gap,
        source='heuristic',
    )
    return [ng]


# =====================================================================
# NOGOOD EXTRACTIE: TRIANGLE / FRUSTRATED CYCLE
# =====================================================================

def extract_triangle_nogoods(n: int, edges: List[Edge],
                              min_frustration: float = 0.5) -> List[Nogood]:
    """
    Identificeer gefrustreerde driehoeken en extraheer nogoods.

    Een driehoek (u,v,w) met gewichten w1, w2, w3 is gefrustreerd als
    niet alle edges tegelijk "tevreden" (gesneden) kunnen worden.
    Het slechtste assignment per driehoek is een nogood.
    """
    # Bouw adjacency
    adj = defaultdict(dict)
    for u, v, w in edges:
        adj[u][v] = w
        adj[v][u] = w

    nogoods = []
    seen_triangles = set()

    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w_node in adj[v]:
                if w_node <= v or w_node not in adj[u]:
                    continue
                tri = tuple(sorted([u, v, w_node]))
                if tri in seen_triangles:
                    continue
                seen_triangles.add(tri)

                # Gewichten
                w_uv = adj[u][v]
                w_vw = adj[v][w_node]
                w_uw = adj[u][w_node]

                # Alle 8 assignments, bereken cut per driehoek
                tri_nodes = list(tri)
                best_cut = -np.inf
                all_cuts = []
                for s in range(8):
                    bits = [(s >> i) & 1 for i in range(3)]
                    cut = 0.0
                    if bits[0] != bits[1]:
                        cut += w_uv
                    if bits[1] != bits[2]:
                        cut += w_vw
                    if bits[0] != bits[2]:
                        cut += w_uw
                    all_cuts.append((s, cut))
                    best_cut = max(best_cut, cut)

                # Nogoods: assignments met gap >= min_frustration
                for s, cut in all_cuts:
                    gap = best_cut - cut
                    if gap >= min_frustration:
                        assign = tuple((tri_nodes[i], (s >> i) & 1) for i in range(3))
                        ng = Nogood(
                            nodes=frozenset(tri_nodes),
                            assignment=tuple(sorted(assign)),
                            cost_gap=gap,
                            source='exact',
                        )
                        nogoods.append(ng)

    return nogoods


# =====================================================================
# NOGOOD-GUIDED SOLVER
# =====================================================================

def nogood_penalty_function(db: NogoodDB, n: int,
                             weight: float = 1.0) -> callable:
    """
    Maak een penalty-functie die een assignment scoort op basis van nogoods.

    Returns:
        Functie f(assignment_dict) -> float penalty (hoger = meer nogoods geschonden)
    """
    if db.total == 0:
        return lambda x: 0.0

    def penalty(assignment: Dict[int, int]) -> float:
        return weight * db.violation_penalty(assignment)

    return penalty


def nogood_guided_bls(n: int, edges: List[Edge], db: NogoodDB,
                       n_restarts: int = 10, max_iter: int = 1000,
                       nogood_weight: float = 0.5,
                       seed: int = 42) -> Dict:
    """
    BLS met nogood-penalty: vermijd bekende slechte patronen.

    Nogood-integratie:
      - Bij elke flip-evaluatie: tel penalty van nogoods die geactiveerd worden
      - Effective delta = cut_delta - nogood_weight * penalty_delta
      - Dit stuurt de search weg van bekende slechte configuraties

    Args:
        n: aantal nodes
        edges: edge list
        db: NogoodDB met geleerde nogoods
        n_restarts: aantal BLS restarts
        max_iter: max iteraties per restart
        nogood_weight: gewicht van nogood-penalty (0 = puur BLS)
        seed: random seed

    Returns:
        dict met best_cut, assignment, nogoods_avoided, etc.
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    # Bouw adjacency
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Pre-compute: voor elke node, welke nogoods zijn relevant?
    node_nogoods = defaultdict(list)
    for ng in db.nogoods:
        for node in ng.nodes:
            node_nogoods[node].append(ng)

    def compute_cut(x):
        cut = 0.0
        for u, v, w in edges:
            if x[u] != x[v]:
                cut += w
        return cut

    def compute_delta(x, node):
        """Delta in cut als we node flippen."""
        delta = 0.0
        for nb, w in adj[node]:
            if x[node] == x[nb]:
                delta += w  # snijdt nu
            else:
                delta -= w  # snijdt niet meer
        return delta

    def compute_nogood_delta(x, node):
        """Delta in nogood-penalty als we node flippen."""
        delta = 0.0
        x_new = x.copy()
        x_new[node] = 1 - x_new[node]

        for ng in node_nogoods.get(node, []):
            was_violated = ng.matches(x)
            is_violated = ng.matches(x_new)
            if was_violated and not is_violated:
                delta -= ng.cost_gap  # penalty vermindert
            elif not was_violated and is_violated:
                delta += ng.cost_gap  # penalty neemt toe
        return delta

    global_best_cut = -np.inf
    global_best_x = None
    total_nogoods_avoided = 0

    for restart in range(n_restarts):
        x = {i: rng.randint(0, 2) for i in range(n)}
        current_cut = compute_cut(x)

        if current_cut > global_best_cut:
            global_best_cut = current_cut
            global_best_x = x.copy()

        for iteration in range(max_iter):
            # Steepest ascent met nogood-penalty
            best_node = -1
            best_effective_delta = -np.inf

            for node in range(n):
                cut_delta = compute_delta(x, node)
                if nogood_weight > 0 and node_nogoods.get(node):
                    ng_delta = compute_nogood_delta(x, node)
                    effective = cut_delta - nogood_weight * ng_delta
                else:
                    effective = cut_delta

                if effective > best_effective_delta:
                    best_effective_delta = effective
                    best_node = node

            if best_effective_delta <= 0:
                # Lokaal optimum — random perturbatie
                n_flip = max(1, n // 10)
                for _ in range(n_flip):
                    node = rng.randint(0, n)
                    x[node] = 1 - x[node]
                current_cut = compute_cut(x)
                continue

            # Flip de beste node
            ng_delta = compute_nogood_delta(x, best_node)
            if ng_delta < 0:
                total_nogoods_avoided += 1

            x[best_node] = 1 - x[best_node]
            current_cut += compute_delta(x, best_node)
            # Herbereken voor correctheid
            current_cut = compute_cut(x)

            if current_cut > global_best_cut:
                global_best_cut = current_cut
                global_best_x = x.copy()

    elapsed = time.time() - t0

    return {
        'best_cut': global_best_cut,
        'assignment': global_best_x if global_best_x else {i: 0 for i in range(n)},
        'nogoods_avoided': total_nogoods_avoided,
        'n_nogoods_used': db.total,
        'time_s': elapsed,
    }


# =====================================================================
# PROGRESSIVE LEARNING PIPELINE
# =====================================================================

def progressive_solve(n: int, edges: List[Edge],
                       n_rounds: int = 3,
                       bls_restarts: int = 5,
                       bls_max_iter: int = 500,
                       max_subgraph: int = 15,
                       nogood_weight: float = 0.5,
                       seed: int = 42,
                       verbose: bool = False) -> Dict:
    """
    Progressieve solver: afwisselend nogoods leren en solver runnen.

    Ronde 1: Basis-nogoods (edges + triangles) + eerste BLS run
    Ronde 2+: Leer uit verbeteringen + re-run BLS met nogood-guidance

    Args:
        n: aantal nodes
        edges: edge list
        n_rounds: aantal learn-solve rondes
        bls_restarts: BLS restarts per ronde
        bls_max_iter: BLS iteraties per restart
        max_subgraph: max grootte voor exacte subgraaf-extractie
        nogood_weight: penalty gewicht
        seed: random seed
        verbose: print voortgang

    Returns:
        dict met best_cut, assignment, db (NogoodDB), history
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)
    db = NogoodDB()
    history = []

    # ---- Ronde 0: Basis-nogoods ----
    if verbose:
        print("Ronde 0: Basis-nogoods (edges + driehoeken)...")

    edge_ngs = extract_edge_nogoods(edges, min_gap=0.1)
    for ng in edge_ngs:
        db.add(ng)

    tri_ngs = extract_triangle_nogoods(n, edges, min_frustration=0.5)
    for ng in tri_ngs:
        db.add(ng)

    if verbose:
        print(f"  {db.total} nogoods geleerd ({db.n_exact} exact)")

    # ---- Ronde 1+: Learn-Solve loop ----
    best_cut = -np.inf
    best_assignment = None

    for rnd in range(n_rounds):
        if verbose:
            print(f"\nRonde {rnd + 1}/{n_rounds}: BLS met {db.total} nogoods...")

        result = nogood_guided_bls(
            n, edges, db,
            n_restarts=bls_restarts,
            max_iter=bls_max_iter,
            nogood_weight=nogood_weight,
            seed=seed + rnd * 1000,
        )

        round_cut = result['best_cut']
        round_assign = result['assignment']

        # Leer uit verbetering (als we verbeterd zijn)
        n_new = 0
        if best_assignment is not None and round_cut > best_cut:
            new_ngs = extract_heuristic_nogoods(
                best_assignment, round_assign,
                best_cut, round_cut,
                edges, max_size=5,
            )
            for ng in new_ngs:
                if db.add(ng):
                    n_new += 1

        if round_cut > best_cut:
            best_cut = round_cut
            best_assignment = round_assign.copy()

        # Probeer ook kleine subgraaf-extractie rond gefrustreerde edges
        if rnd > 0:
            adj = defaultdict(dict)
            for u, v, w in edges:
                adj[u][v] = w
                adj[v][u] = w

            # Vind meest "gefrustreerde" nodes: hoge penalty
            node_penalty = {}
            for node in range(n):
                pen = 0.0
                for ng in db.lookup_node(node):
                    if ng.matches(best_assignment):
                        pen += ng.cost_gap
                node_penalty[node] = pen

            # Top-k gefrustreerde nodes → kleine subgraaf
            frustrated = sorted(node_penalty.keys(),
                                key=lambda x: node_penalty[x], reverse=True)
            if frustrated:
                # Neem top node + buren als subgraaf
                center = frustrated[0]
                subgraph = {center}
                for nb in adj[center]:
                    subgraph.add(nb)
                    if len(subgraph) >= max_subgraph:
                        break

                if len(subgraph) <= max_subgraph:
                    sub_ngs = extract_exact_nogoods(n, edges, list(subgraph),
                                                    min_gap=0.5)
                    for ng in sub_ngs:
                        if db.add(ng):
                            n_new += 1

        history.append({
            'round': rnd + 1,
            'cut': round_cut,
            'best_so_far': best_cut,
            'n_nogoods': db.total,
            'n_new_nogoods': n_new,
            'nogoods_avoided': result['nogoods_avoided'],
            'time_s': result['time_s'],
        })

        if verbose:
            print(f"  Cut: {round_cut:.1f}, best: {best_cut:.1f}, "
                  f"nogoods: {db.total} (+{n_new} new), "
                  f"avoided: {result['nogoods_avoided']}")

    elapsed = time.time() - t0

    return {
        'best_cut': best_cut,
        'assignment': best_assignment,
        'db': db,
        'history': history,
        'total_time': elapsed,
        'db_summary': db.summary(),
    }


# =====================================================================
# CONVENIENCE: NOGOOD-ENHANCED ANYTIME SOLVER
# =====================================================================

def learn_and_solve(n: int, edges: List[Edge],
                     time_limit: float = 10.0,
                     seed: int = 42,
                     verbose: bool = False) -> Dict:
    """
    High-level interface: learn nogoods + solve in één call.

    Automatisch getuned op basis van graafgrootte:
      n <= 20: exacte exhaustive nogoods + BLS
      n <= 100: triangle nogoods + progressive BLS
      n > 100: edge nogoods + progressive BLS

    Returns:
        dict met best_cut, assignment, db_summary, etc.
    """
    if n <= 20:
        # Klein genoeg voor globale exacte nogoods
        db = NogoodDB()
        ngs = extract_exact_nogoods(n, edges, list(range(n)), min_gap=0.5)
        for ng in ngs:
            db.add(ng)
        # BLS met volledige nogood database
        result = nogood_guided_bls(n, edges, db, n_restarts=10,
                                    max_iter=500, seed=seed)
        result['db_summary'] = db.summary()
        return result
    else:
        return progressive_solve(
            n, edges,
            n_rounds=min(5, max(2, int(time_limit / 2))),
            bls_restarts=5,
            bls_max_iter=500,
            max_subgraph=min(15, n // 5),
            seed=seed,
            verbose=verbose,
        )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    from bls_solver import random_3regular

    print("=== B107: Quantum Nogood Learning ===\n")

    _, edges = random_3regular(20, seed=42)
    n = 20

    result = progressive_solve(n, edges, n_rounds=3, verbose=True, seed=42)

    print(f"\nResultaat: best cut = {result['best_cut']:.1f}")
    print(f"DB: {result['db_summary']}")
