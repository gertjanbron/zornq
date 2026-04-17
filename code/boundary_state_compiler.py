#!/usr/bin/env python3
"""
B104: Boundary-State Compiler voor MaxCut

Precompute een map van boundary state → optimale interne respons voor kleine
separators en patches. Daarmee worden veel subproblemen lookup- of DP-achtig
in plaats van telkens volledig herrekend.

Kernidee:
  1. Decompositie: splits graaf in patches via separator-detectie
  2. Compilatie: voor elke patch, enumerate alle mogelijke boundary-assignments
     en bereken optimale interne cut + assignment via brute-force of DP
  3. Cache: sla de boundary→response maps op
  4. Stitching: combineer patches via boundary-condities tot globale oplossing

Twee use-cases:
  A. CLASSICAL: Versneld MaxCut zoeken door patch-decompositie
  B. QUANTUM: Precompiled boundary-state environments voor lightcone QAOA

Referenties:
  [1] Lipton & Tarjan (1979) — separator theorems
  [2] Bodlaender (2005) — dynamic programming on tree decompositions
  [3] B21 Lightcone Graph-Stitching: translatie-invariante patches
  [4] B119 Schur-Complement: separator detectie

Synergieën: B21 (Lightcone), B42 (Treewidth DP), B100 (Pfaffian Oracle),
            B119 (Schur-Complement), FQS (B79)
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import time
import hashlib

Edge = Tuple[int, int, float]


# =====================================================================
# DATA STRUCTUREN
# =====================================================================

@dataclass
class Patch:
    """Een patch (deelgraaf) met boundary en interior nodes.

    boundary: nodes die grenzen aan andere patches
    interior: nodes die volledig binnen de patch liggen
    edges: alle edges (u, v, w) binnen de patch (incl. boundary-boundary)
    boundary_edges: edges die de boundary verbinden met andere patches
    """
    patch_id: int
    boundary: FrozenSet[int]
    interior: FrozenSet[int]
    edges: List[Edge]  # interne edges (inclusief boundary-boundary)
    boundary_edges: List[Edge]  # cross-patch edges (boundary naar andere patch)

    @property
    def nodes(self) -> FrozenSet[int]:
        return self.boundary | self.interior

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def n_boundary(self) -> int:
        return len(self.boundary)


@dataclass
class BoundaryResponse:
    """Gecompileerde response map voor een patch.

    Voor elke mogelijke boundary assignment: optimale interne cut + assignment.

    boundary_nodes: gesorteerde lijst van boundary nodes
    interior_nodes: gesorteerde lijst van interior nodes
    response_map: dict[boundary_assignment_tuple] -> (optimal_cut, interior_assignment_dict)
    """
    patch_id: int
    boundary_nodes: List[int]
    interior_nodes: List[int]
    response_map: Dict[Tuple[int, ...], Tuple[float, Dict[int, int]]]
    compile_time: float = 0.0

    @property
    def n_boundary(self) -> int:
        return len(self.boundary_nodes)

    @property
    def n_interior(self) -> int:
        return len(self.interior_nodes)

    @property
    def n_entries(self) -> int:
        return len(self.response_map)

    def lookup(self, boundary_assignment: Dict[int, int]) -> Tuple[float, Dict[int, int]]:
        """Snel opzoeken van optimale respons voor gegeven boundary state."""
        key = tuple(boundary_assignment[n] for n in self.boundary_nodes)
        return self.response_map[key]


@dataclass
class CompiledGraph:
    """Volledig gecompileerde graafdecompositie.

    patches: de patches
    responses: BoundaryResponse per patch
    cross_edges: edges tussen patches (via boundary nodes)
    """
    patches: List[Patch]
    responses: List[BoundaryResponse]
    cross_edges: List[Edge]
    separator: Set[int]
    compile_time: float = 0.0

    @property
    def n_patches(self) -> int:
        return len(self.patches)


# =====================================================================
# SEPARATOR DETECTIE
# =====================================================================

def find_bfs_separator(n: int, edges: List[Edge],
                        source: Optional[int] = None) -> Tuple[Set[int], Set[int], Set[int]]:
    """BFS-layer separator: kies smalste BFS-layer als separator.

    Returns:
        separator, side_a, side_b (sets van nodes)
    """
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)

    if source is None:
        degrees = {v: len(adj[v]) for v in range(n)}
        source = min(degrees, key=degrees.get) if degrees else 0

    visited = {source}
    layers = [[source]]
    while len(visited) < n:
        next_layer = []
        for v in layers[-1]:
            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    next_layer.append(u)
        if not next_layer:
            break
        layers.append(next_layer)

    if len(layers) < 3:
        return set(), set(range(n)), set()

    min_width = len(layers[1])
    min_idx = 1
    for i in range(1, len(layers) - 1):
        if len(layers[i]) < min_width:
            min_width = len(layers[i])
            min_idx = i

    separator = set(layers[min_idx])
    side_a = set()
    for i in range(min_idx):
        side_a.update(layers[i])
    side_b = set()
    for i in range(min_idx + 1, len(layers)):
        side_b.update(layers[i])

    return separator, side_a, side_b


def find_vertex_separator(n: int, edges: List[Edge],
                           max_separator_size: int = 10) -> Tuple[Set[int], Set[int], Set[int]]:
    """Vind een goede vertex separator via meerdere BFS-startpunten.

    Probeert BFS vanuit meerdere bronnen en kiest de beste separator
    (kleinste die de graaf in twee min-of-meer gelijke delen splitst).

    Returns:
        separator, side_a, side_b
    """
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)

    best_sep = set(range(n))
    best_a = set()
    best_b = set()
    best_score = float('inf')

    # Probeer vanuit periferieknopen (verste BFS-punten)
    sources = [0]
    # Voeg node met hoogste graad toe
    degrees = {v: len(adj[v]) for v in range(n)}
    if degrees:
        sources.append(max(degrees, key=degrees.get))
        sources.append(min(degrees, key=degrees.get))

    for source in sources:
        sep, a, b = find_bfs_separator(n, edges, source=source)
        if not sep:
            continue
        if len(sep) > max_separator_size:
            continue

        # Score: kleinere separator met gebalanceerde split is beter
        balance = abs(len(a) - len(b)) / max(len(a) + len(b), 1)
        score = len(sep) * (1 + balance)

        if score < best_score:
            best_score = score
            best_sep = sep
            best_a = a
            best_b = b

    return best_sep, best_a, best_b


# =====================================================================
# GRAAF DECOMPOSITIE
# =====================================================================

def decompose_graph(n: int, edges: List[Edge],
                     max_patch_size: int = 18,
                     max_separator_size: int = 8) -> Tuple[List[Patch], List[Edge], Set[int]]:
    """Decomposeer een graaf in patches via separator-detectie.

    Recursief: als een patch te groot is, split opnieuw.

    Args:
        n: aantal nodes
        edges: edge list
        max_patch_size: max nodes per patch (incl. boundary)
        max_separator_size: max separator grootte

    Returns:
        patches: lijst van Patch objecten
        cross_edges: edges tussen patches
        separator_nodes: alle separator nodes
    """
    adj = defaultdict(dict)
    for u, v, w in edges:
        adj[u][v] = w
        adj[v][u] = w

    all_nodes = set(range(n))

    # Als graaf klein genoeg: één patch
    if n <= max_patch_size:
        patch = Patch(
            patch_id=0,
            boundary=frozenset(),
            interior=frozenset(all_nodes),
            edges=list(edges),
            boundary_edges=[],
        )
        return [patch], [], set()

    # Zoek separator
    sep, side_a, side_b = find_vertex_separator(n, edges, max_separator_size)

    if not sep or not side_a or not side_b:
        # Kan niet splitsen: hele graaf is één patch
        patch = Patch(
            patch_id=0,
            boundary=frozenset(),
            interior=frozenset(all_nodes),
            edges=list(edges),
            boundary_edges=[],
        )
        return [patch], [], set()

    # Separator nodes zijn boundary voor beide kanten
    # Wijs separator toe aan beide patches als boundary
    patch_a_nodes = side_a | sep
    patch_b_nodes = side_b | sep

    # Edges classificeren
    edges_a = []
    edges_b = []
    cross = []

    for u, v, w in edges:
        u_in_a = u in patch_a_nodes
        v_in_a = v in patch_a_nodes
        u_in_b = u in patch_b_nodes
        v_in_b = v in patch_b_nodes

        if u_in_a and v_in_a:
            edges_a.append((u, v, w))
        if u_in_b and v_in_b:
            edges_b.append((u, v, w))
        # Cross edges: alleen als een node in side_a en andere in side_b
        # (niet via separator)
        if (u in side_a and v in side_b) or (u in side_b and v in side_a):
            cross.append((u, v, w))

    # Boundary edges: edges van separator naar de andere kant
    boundary_edges_a = [(u, v, w) for u, v, w in edges
                         if (u in sep and v in side_b) or (v in sep and u in side_b)]
    boundary_edges_b = [(u, v, w) for u, v, w in edges
                         if (u in sep and v in side_a) or (v in sep and u in side_a)]

    patches = []
    all_cross = list(cross)
    all_sep = set(sep)

    # Patch A
    if len(patch_a_nodes) <= max_patch_size:
        patches.append(Patch(
            patch_id=len(patches),
            boundary=frozenset(sep),
            interior=frozenset(side_a),
            edges=edges_a,
            boundary_edges=boundary_edges_a,
        ))
    else:
        # Recursief opsplitsen — herbouw als sub-problem
        node_map_a = {old: new for new, old in enumerate(sorted(patch_a_nodes))}
        inv_map_a = {new: old for old, new in node_map_a.items()}
        sub_edges_a = [(node_map_a[u], node_map_a[v], w) for u, v, w in edges_a]
        sub_patches, sub_cross, sub_sep = decompose_graph(
            len(patch_a_nodes), sub_edges_a, max_patch_size, max_separator_size
        )
        # Map terug naar originele node-ids
        for sp in sub_patches:
            orig_boundary = frozenset(inv_map_a[n] for n in sp.boundary) | frozenset(
                s for s in sep if s in patch_a_nodes and s not in side_a
            )
            orig_interior = frozenset(inv_map_a[n] for n in sp.interior)
            # Alleen echte interior (niet separator)
            real_interior = orig_interior - sep
            real_boundary = (orig_boundary | (orig_interior & sep))
            orig_edges = [(inv_map_a[u], inv_map_a[v], w) for u, v, w in sp.edges]
            sp_new = Patch(
                patch_id=len(patches),
                boundary=frozenset(real_boundary),
                interior=frozenset(real_interior),
                edges=orig_edges,
                boundary_edges=[],
            )
            patches.append(sp_new)
        for u, v, w in sub_cross:
            all_cross.append((inv_map_a[u], inv_map_a[v], w))
        all_sep |= {inv_map_a[s] for s in sub_sep}

    # Patch B
    if len(patch_b_nodes) <= max_patch_size:
        patches.append(Patch(
            patch_id=len(patches),
            boundary=frozenset(sep),
            interior=frozenset(side_b),
            edges=edges_b,
            boundary_edges=boundary_edges_b,
        ))
    else:
        node_map_b = {old: new for new, old in enumerate(sorted(patch_b_nodes))}
        inv_map_b = {new: old for old, new in node_map_b.items()}
        sub_edges_b = [(node_map_b[u], node_map_b[v], w) for u, v, w in edges_b]
        sub_patches, sub_cross, sub_sep = decompose_graph(
            len(patch_b_nodes), sub_edges_b, max_patch_size, max_separator_size
        )
        for sp in sub_patches:
            orig_boundary = frozenset(inv_map_b[n] for n in sp.boundary) | frozenset(
                s for s in sep if s in patch_b_nodes and s not in side_b
            )
            orig_interior = frozenset(inv_map_b[n] for n in sp.interior)
            real_interior = orig_interior - sep
            real_boundary = (orig_boundary | (orig_interior & sep))
            orig_edges = [(inv_map_b[u], inv_map_b[v], w) for u, v, w in sp.edges]
            sp_new = Patch(
                patch_id=len(patches),
                boundary=frozenset(real_boundary),
                interior=frozenset(real_interior),
                edges=orig_edges,
                boundary_edges=[],
            )
            patches.append(sp_new)
        for u, v, w in sub_cross:
            all_cross.append((inv_map_b[u], inv_map_b[v], w))
        all_sep |= {inv_map_b[s] for s in sub_sep}

    return patches, all_cross, all_sep


# =====================================================================
# BOUNDARY-STATE COMPILATIE
# =====================================================================

def compile_patch(patch: Patch) -> BoundaryResponse:
    """Compileer een patch: voor elke boundary assignment, vind optimale interior.

    Brute-force enumerate alle interior assignments. Voor elke boundary
    assignment selecteer de interior assignment die de interne cut maximaliseert.

    Complexiteit: O(2^boundary * 2^interior) per patch.
    """
    t0 = time.time()

    boundary_list = sorted(patch.boundary)
    interior_list = sorted(patch.interior)
    n_boundary = len(boundary_list)
    n_interior = len(interior_list)

    # Node → lokale index mapping
    node_set = set(boundary_list) | set(interior_list)

    # Filter edges: alleen edges met beide endpoints in deze patch
    patch_edges = [(u, v, w) for u, v, w in patch.edges
                    if u in node_set and v in node_set]

    if not patch_edges:
        # Geen edges: elke boundary state → cut=0, willekeurige interior
        response_map = {}
        default_interior = {n: 0 for n in interior_list}
        for bs in range(2 ** n_boundary):
            bkey = tuple((bs >> i) & 1 for i in range(n_boundary))
            response_map[bkey] = (0.0, dict(default_interior))
        return BoundaryResponse(
            patch_id=patch.patch_id,
            boundary_nodes=boundary_list,
            interior_nodes=interior_list,
            response_map=response_map,
            compile_time=time.time() - t0,
        )

    # Precompute: classify edges
    # boundary-boundary, boundary-interior, interior-interior
    response_map = {}

    for bs in range(2 ** n_boundary):
        b_assign = {}
        for i, node in enumerate(boundary_list):
            b_assign[node] = (bs >> i) & 1

        best_cut = -np.inf
        best_interior = None

        for Is in range(2 ** n_interior):
            i_assign = {}
            for i, node in enumerate(interior_list):
                i_assign[node] = (Is >> i) & 1

            # Bereken cut over patch edges
            full_assign = {**b_assign, **i_assign}
            cut = 0.0
            for u, v, w in patch_edges:
                if full_assign.get(u, 0) != full_assign.get(v, 0):
                    cut += w

            if cut > best_cut:
                best_cut = cut
                best_interior = dict(i_assign)

        bkey = tuple((bs >> i) & 1 for i in range(n_boundary))
        response_map[bkey] = (best_cut, best_interior if best_interior else {})

    return BoundaryResponse(
        patch_id=patch.patch_id,
        boundary_nodes=boundary_list,
        interior_nodes=interior_list,
        response_map=response_map,
        compile_time=time.time() - t0,
    )


def compile_graph(n: int, edges: List[Edge],
                   max_patch_size: int = 18,
                   max_separator_size: int = 8,
                   verbose: bool = False) -> CompiledGraph:
    """Compileer een volledige graaf: decompositie + patch compilatie.

    Args:
        n: aantal nodes
        edges: edge list
        max_patch_size: max nodes per patch
        max_separator_size: max separator grootte
        verbose: print voortgang

    Returns:
        CompiledGraph met gecompileerde patches
    """
    t0 = time.time()

    if verbose:
        print(f"Decomposing graph (n={n}, m={len(edges)})...")

    patches, cross_edges, separator = decompose_graph(
        n, edges, max_patch_size, max_separator_size
    )

    if verbose:
        print(f"  {len(patches)} patches, {len(cross_edges)} cross edges, "
              f"{len(separator)} separator nodes")

    responses = []
    for p in patches:
        if verbose:
            print(f"  Compiling patch {p.patch_id}: {p.size} nodes "
                  f"({p.n_boundary} boundary, {len(p.interior)} interior)...")

        resp = compile_patch(p)
        responses.append(resp)

        if verbose:
            print(f"    {resp.n_entries} entries, {resp.compile_time:.3f}s")

    elapsed = time.time() - t0

    return CompiledGraph(
        patches=patches,
        responses=responses,
        cross_edges=cross_edges,
        separator=separator,
        compile_time=elapsed,
    )


# =====================================================================
# STITCHING: Gecombineerde optimalisatie via boundary matching
# =====================================================================

def stitch_solve(compiled: CompiledGraph, n: int, edges: List[Edge],
                  method: str = 'enumerate') -> Dict:
    """Combineer gecompileerde patches tot een globale oplossing.

    Methode 'enumerate':
      Enumerate alle boundary (separator) assignments.
      Voor elk boundary assignment: lookup optimale interior per patch.
      Kies het boundary assignment dat de totale cut maximaliseert.

    Args:
        compiled: CompiledGraph
        n: totaal aantal nodes
        edges: alle edges
        method: 'enumerate' (brute force over boundary)

    Returns:
        dict met best_cut, assignment, etc.
    """
    t0 = time.time()

    if not compiled.patches:
        return {'best_cut': 0.0, 'assignment': {i: 0 for i in range(n)},
                'time_s': 0.0}

    # Als er maar 1 patch is zonder boundary → directe lookup
    if len(compiled.patches) == 1 and compiled.patches[0].n_boundary == 0:
        resp = compiled.responses[0]
        if resp.n_entries > 0:
            # Enige entry: boundary = ()
            cut, interior = resp.response_map.get((), (0.0, {}))
            return {'best_cut': cut, 'assignment': interior,
                    'time_s': time.time() - t0}

    # Verzamel alle boundary nodes
    all_boundary = set()
    for p in compiled.patches:
        all_boundary |= set(p.boundary)

    boundary_list = sorted(all_boundary)
    n_boundary = len(boundary_list)

    if n_boundary > 25:
        # Te veel boundary nodes voor brute force
        # Fallback: greedy per patch
        return _greedy_stitch(compiled, n, edges)

    best_total_cut = -np.inf
    best_assignment = None

    # Precompute adjacency voor cross-edge evaluatie
    cross_edge_list = compiled.cross_edges

    for bs in range(2 ** n_boundary):
        b_assign = {}
        for i, node in enumerate(boundary_list):
            b_assign[node] = (bs >> i) & 1

        total_cut = 0.0
        full_assign = dict(b_assign)
        valid = True

        for resp in compiled.responses:
            # Maak boundary key voor deze patch
            try:
                bkey = tuple(b_assign[n] for n in resp.boundary_nodes)
            except KeyError:
                # Boundary node niet in globale assignment → skip
                valid = False
                break

            if bkey not in resp.response_map:
                valid = False
                break

            patch_cut, interior = resp.response_map[bkey]
            total_cut += patch_cut
            full_assign.update(interior)

        if not valid:
            continue

        # Herbereken werkelijke cut over alle originele edges
        # (patch_cut sommen kunnen dubbeltellen door gedeelde separator edges)
        total_cut = 0.0
        for u, v, w in edges:
            if full_assign.get(u, 0) != full_assign.get(v, 0):
                total_cut += w

        if total_cut > best_total_cut:
            best_total_cut = total_cut
            best_assignment = dict(full_assign)

    elapsed = time.time() - t0

    if best_assignment is None:
        best_assignment = {i: 0 for i in range(n)}
        best_total_cut = 0.0
    else:
        # Herbereken werkelijke cut (patch sommen kunnen dubbeltellen via separator)
        best_total_cut = 0.0
        for u, v, w in edges:
            if best_assignment.get(u, 0) != best_assignment.get(v, 0):
                best_total_cut += w

    return {
        'best_cut': best_total_cut,
        'assignment': best_assignment,
        'n_boundary': n_boundary,
        'n_boundary_configs': 2 ** n_boundary,
        'time_s': elapsed,
    }


def _greedy_stitch(compiled: CompiledGraph, n: int, edges: List[Edge]) -> Dict:
    """Greedy stitching: per patch, kies boundary die lokaal optimaal is."""
    t0 = time.time()

    assignment = {}

    for resp in compiled.responses:
        # Kies de boundary assignment met hoogste lokale cut
        best_cut = -np.inf
        best_bkey = None
        for bkey, (cut, interior) in resp.response_map.items():
            if cut > best_cut:
                # Check consistentie met reeds gekozen assignments
                conflict = False
                for i, node in enumerate(resp.boundary_nodes):
                    if node in assignment and assignment[node] != bkey[i]:
                        conflict = True
                        break
                if not conflict:
                    best_cut = cut
                    best_bkey = bkey

        if best_bkey is not None:
            for i, node in enumerate(resp.boundary_nodes):
                assignment[node] = best_bkey[i]
            _, interior = resp.response_map[best_bkey]
            assignment.update(interior)

    # Vul ontbrekende nodes
    for i in range(n):
        if i not in assignment:
            assignment[i] = 0

    # Bereken werkelijke cut
    total_cut = 0.0
    for u, v, w in edges:
        if assignment.get(u, 0) != assignment.get(v, 0):
            total_cut += w

    return {
        'best_cut': total_cut,
        'assignment': assignment,
        'method': 'greedy',
        'time_s': time.time() - t0,
    }


# =====================================================================
# LIGHTCONE BOUNDARY-STATE CACHE
# =====================================================================

@dataclass
class LightconeBoundaryCache:
    """Cache voor boundary-state environments in lightcone QAOA.

    Slaat precomputed <ZZ> waarden op voor lightcone subgrafen,
    geïndexeerd op structureel-kanonieke keys + QAOA parameters.
    """
    _cache: Dict[Tuple, float] = field(default_factory=dict)
    n_hits: int = 0
    n_misses: int = 0

    def make_key(self, lc_nodes: Set[int], sub_edges: List[Edge],
                  target_edge: Tuple[int, int],
                  p: int, gammas: Tuple[float, ...],
                  betas: Tuple[float, ...]) -> Tuple:
        """Maak een cache key die structuur + parameters combineert.

        Structureel-kanonieke relabeling + parameter hash.
        """
        # Structurele key (isomorfisme)
        i_t, j_t = target_edge
        remaining = sorted([n for n in lc_nodes if n != i_t and n != j_t])
        relabel = {i_t: 0, j_t: 1}
        for idx, node in enumerate(remaining):
            relabel[node] = idx + 2

        relabeled_edges = tuple(sorted(
            (relabel[i], relabel[j], w) for i, j, w in sub_edges
            if i in relabel and j in relabel
        ))

        n_nodes = len(lc_nodes)

        # Parameter key (afgerond voor floating point)
        param_key = (p,
                     tuple(round(g, 10) for g in gammas),
                     tuple(round(b, 10) for b in betas))

        return (n_nodes, relabeled_edges, param_key)

    def get(self, key: Tuple) -> Optional[float]:
        """Lookup. Return None bij miss."""
        if key in self._cache:
            self.n_hits += 1
            return self._cache[key]
        self.n_misses += 1
        return None

    def put(self, key: Tuple, value: float):
        """Store waarde."""
        self._cache[key] = value

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.n_hits + self.n_misses
        return self.n_hits / total if total > 0 else 0.0

    def summary(self) -> Dict:
        return {
            'size': self.size,
            'hits': self.n_hits,
            'misses': self.n_misses,
            'hit_rate': self.hit_rate,
        }


# =====================================================================
# PATCH ISOMORFISME CACHE
# =====================================================================

def patch_structure_key(patch: Patch) -> Tuple:
    """Genereer een structurele key voor patch isomorfisme-detectie.

    Twee patches met dezelfde lokale structuur (degree sequence, edge pattern)
    hebben dezelfde boundary response map (modulo relabeling).
    """
    # Kanonieke relabeling: boundary eerst (gesorteerd), dan interior
    boundary_sorted = sorted(patch.boundary)
    interior_sorted = sorted(patch.interior)
    all_nodes = boundary_sorted + interior_sorted
    relabel = {old: new for new, old in enumerate(all_nodes)}

    relabeled_edges = tuple(sorted(
        (relabel.get(u, u), relabel.get(v, v), w)
        for u, v, w in patch.edges
        if u in relabel and v in relabel
    ))

    return (len(boundary_sorted), len(interior_sorted), relabeled_edges)


def compile_graph_with_isomorphism(n: int, edges: List[Edge],
                                     max_patch_size: int = 18,
                                     max_separator_size: int = 8,
                                     verbose: bool = False) -> CompiledGraph:
    """Compileer met isomorfisme-caching: structureel identieke patches
    worden maar één keer gecompileerd.

    Dit is de versnelde versie van compile_graph.
    """
    t0 = time.time()

    patches, cross_edges, separator = decompose_graph(
        n, edges, max_patch_size, max_separator_size
    )

    if verbose:
        print(f"Decomposing: {len(patches)} patches, {len(cross_edges)} cross, "
              f"{len(separator)} sep nodes")

    iso_cache = {}  # structure_key -> BoundaryResponse
    responses = []
    n_compiled = 0
    n_cached = 0

    for p in patches:
        skey = patch_structure_key(p)

        if skey in iso_cache:
            # Hergebruik: remap de response
            cached_resp = iso_cache[skey]
            # Remap boundary en interior nodes
            boundary_sorted_cached = cached_resp.boundary_nodes
            interior_sorted_cached = cached_resp.interior_nodes
            boundary_sorted_new = sorted(p.boundary)
            interior_sorted_new = sorted(p.interior)

            # Bouw node mapping: cached → new
            node_map = {}
            for old, new in zip(boundary_sorted_cached, boundary_sorted_new):
                node_map[old] = new
            for old, new in zip(interior_sorted_cached, interior_sorted_new):
                node_map[old] = new

            # Remap response
            new_response_map = {}
            for bkey, (cut, interior_assign) in cached_resp.response_map.items():
                new_interior = {node_map.get(k, k): v for k, v in interior_assign.items()}
                new_response_map[bkey] = (cut, new_interior)

            resp = BoundaryResponse(
                patch_id=p.patch_id,
                boundary_nodes=boundary_sorted_new,
                interior_nodes=interior_sorted_new,
                response_map=new_response_map,
                compile_time=0.0,
            )
            n_cached += 1
        else:
            resp = compile_patch(p)
            iso_cache[skey] = resp
            n_compiled += 1

        responses.append(resp)

    elapsed = time.time() - t0

    if verbose:
        print(f"  {n_compiled} compiled, {n_cached} cached via isomorphism")

    return CompiledGraph(
        patches=patches,
        responses=responses,
        cross_edges=cross_edges,
        separator=separator,
        compile_time=elapsed,
    )


# =====================================================================
# CONVENIENCE: COMPILE + SOLVE
# =====================================================================

def boundary_solve(n: int, edges: List[Edge],
                    max_patch_size: int = 18,
                    use_isomorphism: bool = True,
                    verbose: bool = False) -> Dict:
    """High-level interface: compileer graaf + stitch-solve.

    Returns:
        dict met best_cut, assignment, compile_time, solve_time, etc.
    """
    t0 = time.time()

    if use_isomorphism:
        compiled = compile_graph_with_isomorphism(
            n, edges, max_patch_size=max_patch_size, verbose=verbose
        )
    else:
        compiled = compile_graph(
            n, edges, max_patch_size=max_patch_size, verbose=verbose
        )

    result = stitch_solve(compiled, n, edges)
    result['compile_time'] = compiled.compile_time
    result['n_patches'] = compiled.n_patches
    result['separator_size'] = len(compiled.separator)
    result['total_time'] = time.time() - t0

    return result


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    from bls_solver import random_3regular

    print("=== B104: Boundary-State Compiler ===\n")

    for n in [12, 16, 20]:
        _, edges = random_3regular(n, seed=42)
        print(f"n={n}, m={len(edges)}")

        result = boundary_solve(n, edges, max_patch_size=14, verbose=True)
        print(f"  Best cut: {result['best_cut']:.1f}")
        print(f"  Patches: {result['n_patches']}, separator: {result['separator_size']}")
        print(f"  Compile: {result['compile_time']:.3f}s, solve: {result['time_s']:.3f}s")
        print()
