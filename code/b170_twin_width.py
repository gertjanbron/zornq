#!/usr/bin/env python3
"""B170 — Twin-width primitives + cograph-special-case MaxCut.

Twin-width (Bonnet-Kim-Thomasse-Watrigant 2020, JACM) is een grafenparameter
die meet hoe "dense / structuurarm" een graaf is. Veel klassieke NP-harde
problemen (MaxCut, k-kleuring, ...) zijn FPT geparametriseerd op tw(G):
*als* je een contractie-sequentie met max red-degree d krijgt, los je MaxCut
op in tijd f(d) · poly(n).

Dit module levert de **basis-laag**:

  - Trigraph: graaf met zwarte (= hard) en rode (= error / ambiguous) edges.
  - contract(u, v): merge u in v. Voor elke derde knoop w:
      * u-w en v-w beide zwart  -> blijft zwart
      * u-w en v-w beide niet    -> blijft niet
      * anders (inclusief een van beide rood) -> wordt rood.
    Eventuele edge u-v verdwijnt.
  - twin_width_heuristic(g): greedy min-red-degree contractie-sequentie,
    O(n^4) maar robuust.
  - Cograph-herkenning via P_4-free-check (O(n^4)).
  - Cotree-decompositie (parallel = unie, series = join).
  - Cograph MaxCut via DP op de cotree: O(n^3).

De laatste twee geven een *exacte* poly-tijd MaxCut voor alle cographs
(tw = 0), wat de gebruikelijke bottleneck van B42 (DP op boombreedte) omzeilt
voor dense bipartiete en cluster-achtige input.

Koppeling met B130-dispatcher: als graph-detectie cograph zegt, roep
`cograph_maxcut_exact` aan; anders bereken `twin_width_heuristic` en laat
de dispatcher beslissen (klein d -> DP; groot d -> QUBO via B153).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# 1. Trigraph + contractie
# ============================================================

BLACK = "B"
RED = "R"


@dataclass
class Trigraph:
    """Graaf met zwarte en rode edges.

    adj[u][v] in {'B', 'R'} voor adjacency. Afwezigheid = no edge.
    Self-loops zijn verboden.
    """

    vertices: Set[int] = field(default_factory=set)
    adj: Dict[int, Dict[int, str]] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, n: int, edges: List[Tuple[int, int]]) -> "Trigraph":
        """Start-trigraaf uit een gewone graaf: alle edges zwart."""
        g = cls()
        for v in range(n):
            g.vertices.add(v)
            g.adj[v] = {}
        for (u, v) in edges:
            if u == v:
                continue
            g.adj[u][v] = BLACK
            g.adj[v][u] = BLACK
        return g

    def copy(self) -> "Trigraph":
        h = Trigraph()
        h.vertices = set(self.vertices)
        h.adj = {v: dict(self.adj[v]) for v in self.vertices}
        return h

    def red_degree(self, v: int) -> int:
        return sum(1 for c in self.adj[v].values() if c == RED)

    def max_red_degree(self) -> int:
        if not self.vertices:
            return 0
        return max(self.red_degree(v) for v in self.vertices)

    def contract(self, u: int, v: int) -> None:
        """Merge u in v (u wordt verwijderd, v erft aangepaste edges).

        Voor elke w ∉ {u, v}:
          * u-w zwart en v-w zwart  -> v-w zwart
          * u-w afwezig en v-w afwezig -> v-w afwezig
          * anders -> v-w rood
        Bestaande u-v edge (zwart of rood) wordt genegeerd / verwijderd.
        """
        if u not in self.vertices or v not in self.vertices:
            raise ValueError(f"vertices {u}, {v} must both exist")
        if u == v:
            raise ValueError("cannot contract vertex with itself")

        for w in list(self.vertices):
            if w == u or w == v:
                continue
            uw = self.adj[u].get(w)
            vw = self.adj[v].get(w)
            if uw == BLACK and vw == BLACK:
                new = BLACK
            elif uw is None and vw is None:
                new = None
            else:
                new = RED
            # Werk adj[v][w] / adj[w][v] bij
            if new is None:
                self.adj[v].pop(w, None)
                self.adj[w].pop(v, None)
            else:
                self.adj[v][w] = new
                self.adj[w][v] = new
            # Schoon adj[u] op (niet strikt nodig, u wordt verwijderd)
            self.adj[u].pop(w, None)
            self.adj[w].pop(u, None)

        # Verwijder de u-v edge zelf
        self.adj[v].pop(u, None)
        self.adj[u].pop(v, None)
        # Verwijder u
        self.vertices.remove(u)
        del self.adj[u]


# ============================================================
# 2. Greedy twin-width heuristic
# ============================================================

def twin_width_heuristic(g0: Trigraph) -> Tuple[int, List[Tuple[int, int]]]:
    """Greedy contractie-sequentie: minimaliseer max-red-degree lokaal.

    Retourneert (d_max, sequence) waarbij:
      - d_max = max over alle tussentrigrafen van max_red_degree.
      - sequence = lijst (u_keep, v_remove) van contracties in volgorde.

    Dit is een bovengrens op de echte twin-width; *alleen* voor n ≤ 20
    is exact pairs-enumeratie praktisch.
    """
    g = g0.copy()
    n = len(g.vertices)
    sequence: List[Tuple[int, int]] = []
    d_max = g.max_red_degree()
    if n <= 1:
        return 0, sequence

    while len(g.vertices) > 1:
        best_d = None
        best_pair = None
        verts = sorted(g.vertices)
        for u, v in itertools.combinations(verts, 2):
            trial = g.copy()
            trial.contract(u, v)
            d = trial.max_red_degree()
            if best_d is None or d < best_d:
                best_d = d
                best_pair = (u, v)
                if d == 0:
                    break  # niks beters mogelijk
        assert best_pair is not None
        u, v = best_pair
        g.contract(u, v)
        sequence.append((v, u))  # conventie: (keep, remove) = (v, u) want u werd verwijderd
        if best_d > d_max:
            d_max = best_d

    return d_max, sequence


def twin_width_exact(g0: Trigraph, max_n: int = 8) -> Tuple[int, List[Tuple[int, int]]]:
    """Exacte tww via enumeratie van alle contractie-volgordes (n!/2 routes).

    ONLY voor kleine grafen (n ≤ 8). Gebruikt branch-and-bound: zodra het
    lopende max-red-degree >= huidige beste d*, snijd af.
    """
    n = len(g0.vertices)
    if n > max_n:
        raise ValueError(f"twin_width_exact: n={n} > max_n={max_n}; te duur")
    if n <= 1:
        return 0, []

    best_d = [n]  # upper bound
    best_seq: List[Tuple[int, int]] = []

    def recurse(g: Trigraph, cur_max: int, seq: List[Tuple[int, int]]) -> None:
        if cur_max >= best_d[0]:
            return
        if len(g.vertices) == 1:
            best_d[0] = cur_max
            best_seq.clear()
            best_seq.extend(seq)
            return
        verts = sorted(g.vertices)
        for u, v in itertools.combinations(verts, 2):
            trial = g.copy()
            trial.contract(u, v)
            new_max = max(cur_max, trial.max_red_degree())
            if new_max >= best_d[0]:
                continue
            seq.append((v, u))
            recurse(trial, new_max, seq)
            seq.pop()

    recurse(g0.copy(), g0.max_red_degree(), [])
    return best_d[0], best_seq


# ============================================================
# 3. Cograph herkenning
# ============================================================

def _induced_subgraph_has_p4(n: int, neigh: List[Set[int]], subset: List[int]) -> bool:
    """Is er een geïnduceerde P_4 (a-b-c-d, verder geen edges) in subset?"""
    s = set(subset)
    for (a, b, c, d) in itertools.permutations(subset, 4):
        if a > d:  # P_4 is symmetrisch
            continue
        # edges a-b, b-c, c-d moeten er zijn
        if b not in neigh[a] or c not in neigh[b] or d not in neigh[c]:
            continue
        # a-c, a-d, b-d moeten er niet zijn
        if c in neigh[a] or d in neigh[a] or d in neigh[b]:
            continue
        return True
    return False


def is_cograph(n: int, edges: List[Tuple[int, int]]) -> bool:
    """Een graaf is cograph iff geen geïnduceerde P_4 heeft. O(n^4)."""
    neigh: List[Set[int]] = [set() for _ in range(n)]
    for (u, v) in edges:
        if u == v:
            continue
        neigh[u].add(v)
        neigh[v].add(u)
    return not _induced_subgraph_has_p4(n, neigh, list(range(n)))


# ============================================================
# 4. Cotree constructie
# ============================================================

@dataclass
class CotreeNode:
    kind: str  # 'leaf', 'parallel', 'series'
    vertex: Optional[int] = None  # alleen bij leaf
    children: List["CotreeNode"] = field(default_factory=list)

    @property
    def size(self) -> int:
        if self.kind == "leaf":
            return 1
        return sum(c.size for c in self.children)


def _connected_components(verts: List[int], neigh: List[Set[int]]) -> List[List[int]]:
    """BFS connected components op de geïnduceerde subgraaf."""
    vs = set(verts)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for start in verts:
        if start in seen:
            continue
        stack = [start]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for w in neigh[u]:
                if w in vs and w not in seen:
                    stack.append(w)
        comps.append(comp)
    return comps


def _complement_components(verts: List[int], neigh: List[Set[int]]) -> List[List[int]]:
    """Connected components van het complement van de geïnduceerde subgraaf."""
    vs = set(verts)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for start in verts:
        if start in seen:
            continue
        # BFS in complement: w is 'buur' als w in vs \ {u} en w ∉ neigh[u]
        stack = [start]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for w in vs:
                if w == u or w in seen:
                    continue
                if w not in neigh[u]:
                    stack.append(w)
        comps.append(comp)
    return comps


def build_cotree(n: int, edges: List[Tuple[int, int]]) -> CotreeNode:
    """Bouw cotree voor een cograph. Roept ValueError bij niet-cograph.

    Recursief:
      - |V|=1: leaf.
      - G disconnected: parallel-node met cotree per component.
      - complement(G) disconnected: series-node met cotree per co-component.
      - anders: geen cograph (zou niet mogen, want we namen aan is_cograph).
    """
    neigh: List[Set[int]] = [set() for _ in range(n)]
    for (u, v) in edges:
        if u == v:
            continue
        neigh[u].add(v)
        neigh[v].add(u)

    def build(verts: List[int]) -> CotreeNode:
        if len(verts) == 1:
            return CotreeNode(kind="leaf", vertex=verts[0])
        comps = _connected_components(verts, neigh)
        if len(comps) > 1:
            return CotreeNode(
                kind="parallel",
                children=[build(c) for c in comps],
            )
        co_comps = _complement_components(verts, neigh)
        if len(co_comps) > 1:
            return CotreeNode(
                kind="series",
                children=[build(c) for c in co_comps],
            )
        raise ValueError(
            f"subset {verts} is niet decomposeerbaar — input is geen cograph"
        )

    return build(list(range(n)))


# ============================================================
# 5. Cograph MaxCut via cotree-DP
# ============================================================

def cograph_maxcut_exact(n: int, edges: List[Tuple[int, int]]) -> Dict:
    """Exacte MaxCut op een cograph via cotree-DP in O(n^3).

    DP-state: voor elke cotree-node N een lijst `dp` met
        dp[k] = (best_cut, assignment_dict)
    waar k = aantal vertices van N dat aan kant 0 staat (0 ≤ k ≤ N.size),
    en `best_cut` = aantal / gewicht cut-edges *binnen* de induced subgraaf
    van N (edges buiten N komen hoger in de boom).

    Parallel-node (onafhankelijke unie, geen extra cross-edges):
        combineer kinderen via convolutie: dp_new[k1+k2] = dp1[k1] + dp2[k2].
    Series-node (complete bipartiete tussen elk paar kinderen):
        tussen kind i en j zijn alle vertices onderling verbonden. De
        cross-edge bijdrage voor partitie (k_i, size_i - k_i) en (k_j,
        size_j - k_j) = k_i*(size_j - k_j) + (size_i - k_i)*k_j.

    Retourneert {'value': ..., 'partition': {v: 0/1}}.
    """
    if n == 0:
        return {"value": 0.0, "partition": {}}
    if not is_cograph(n, edges):
        raise ValueError("graph is not a cograph; use twin_width DP instead")

    root = build_cotree(n, edges)

    # dp(node) -> list[(cut_value, assign_dict)] indexed op k = aantal links
    def dp(node: CotreeNode) -> List[Tuple[float, Dict[int, int]]]:
        if node.kind == "leaf":
            v = node.vertex
            # k=0: rechts; k=1: links
            return [(0.0, {v: 1}), (0.0, {v: 0})]

        if node.kind == "parallel":
            # combineer via convolutie
            tables = [dp(c) for c in node.children]
            total_size = node.size
            cur: List[Tuple[float, Dict[int, int]]] = [(0.0, {})]
            cur_size = 0
            for tbl in tables:
                m = len(tbl) - 1
                new: List[Tuple[float, Dict[int, int]]] = [
                    (float("-inf"), {}) for _ in range(cur_size + m + 1)
                ]
                for k1 in range(cur_size + 1):
                    v1, a1 = cur[k1]
                    if v1 == float("-inf"):
                        continue
                    for k2 in range(m + 1):
                        v2, a2 = tbl[k2]
                        if v2 == float("-inf"):
                            continue
                        total = v1 + v2
                        if total > new[k1 + k2][0]:
                            merged = dict(a1)
                            merged.update(a2)
                            new[k1 + k2] = (total, merged)
                cur = new
                cur_size += m
            assert cur_size == total_size
            return cur

        if node.kind == "series":
            # complete bipartiete tussen elk paar kinderen
            child_tables = [dp(c) for c in node.children]
            child_sizes = [c.size for c in node.children]
            total_size = node.size

            # Combineer stapsgewijs: hou (cum_size, dp_cum) bij, voeg elk kind toe.
            cum_size = child_sizes[0]
            cum_dp = list(child_tables[0])
            for idx in range(1, len(child_tables)):
                sz = child_sizes[idx]
                tbl = child_tables[idx]
                new_size = cum_size + sz
                new: List[Tuple[float, Dict[int, int]]] = [
                    (float("-inf"), {}) for _ in range(new_size + 1)
                ]
                for k1 in range(cum_size + 1):
                    v1, a1 = cum_dp[k1]
                    if v1 == float("-inf"):
                        continue
                    for k2 in range(sz + 1):
                        v2, a2 = tbl[k2]
                        if v2 == float("-inf"):
                            continue
                        cross = k1 * (sz - k2) + (cum_size - k1) * k2
                        total = v1 + v2 + cross
                        if total > new[k1 + k2][0]:
                            merged = dict(a1)
                            merged.update(a2)
                            new[k1 + k2] = (total, merged)
                cum_dp = new
                cum_size = new_size
            return cum_dp

        raise RuntimeError(f"unknown cotree node kind: {node.kind}")

    table = dp(root)
    best = max(table, key=lambda t: t[0])
    return {"value": float(best[0]), "partition": best[1]}


# ============================================================
# 6. Family generators — voor tests & benchmark
# ============================================================

def empty_edges(n: int) -> List[Tuple[int, int]]:
    return []


def complete_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def complete_bipartite_edges(a: int, b: int) -> List[Tuple[int, int]]:
    return [(i, a + j) for i in range(a) for j in range(b)]


def path_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n: int) -> List[Tuple[int, int]]:
    if n < 3:
        return path_edges(n)
    return [(i, (i + 1) % n) for i in range(n)]


def tree_edges_balanced_binary(depth: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Gebalanceerde binaire boom: parent i -> kinderen 2i+1, 2i+2."""
    n = 2 ** (depth + 1) - 1
    edges = [(i, 2 * i + 1) for i in range((n - 1) // 2)] + \
            [(i, 2 * i + 2) for i in range((n - 1) // 2) if 2 * i + 2 < n]
    return n, edges


def petersen_edges() -> List[Tuple[int, int]]:
    # outer 5-cycle, inner pentagram, spokes
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    return outer + inner + spokes


# ============================================================
# 7. Brute-force MaxCut (verificatie voor tests)
# ============================================================

def brute_force_maxcut(n: int, edges: List[Tuple[int, int]],
                      weights: Optional[List[float]] = None) -> Dict:
    """Exacte MaxCut via enumeratie (2^n). Alleen voor kleine n."""
    if weights is None:
        weights = [1.0] * len(edges)
    best_cut = -1.0
    best_bits = 0
    for bits in range(2 ** n):
        cut = 0.0
        for (u, v), w in zip(edges, weights):
            if ((bits >> u) & 1) != ((bits >> v) & 1):
                cut += w
        if cut > best_cut:
            best_cut = cut
            best_bits = bits
    partition = {v: (best_bits >> v) & 1 for v in range(n)}
    return {"value": best_cut, "partition": partition}


# ============================================================
# 8. CLI
# ============================================================

def _cli_demo() -> int:
    print("=" * 70)
    print("  B170 Twin-width Parameter — demo")
    print("=" * 70)

    demos: List[Tuple[str, int, List[Tuple[int, int]]]] = [
        ("K_4", 4, complete_edges(4)),
        ("C_5", 5, cycle_edges(5)),
        ("K_{3,3}", 6, complete_bipartite_edges(3, 3)),
        ("Petersen", 10, petersen_edges()),
    ]
    for label, n, edges in demos:
        g = Trigraph.from_graph(n, edges)
        t0 = time.time()
        d, seq = twin_width_heuristic(g)
        t1 = time.time()
        cog = is_cograph(n, edges)
        cog_cut = "—"
        if cog:
            res = cograph_maxcut_exact(n, edges)
            cog_cut = f"{res['value']:.0f}"
        print(f"  {label:<10} n={n:<2}  edges={len(edges):<3}  tww≤{d}  "
              f"cograph={str(cog):<5}  cograph-MaxCut={cog_cut}  "
              f"({(t1-t0)*1000:.1f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_demo())
