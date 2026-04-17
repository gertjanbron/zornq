#!/usr/bin/env python3
"""
schur_complement.py - Schur-Complement Separator Elimination (B119)

Elimineer "makkelijke" nodes (lage graad, binnengebieden) uit een MaxCut
probleem en vouw hun effect samen tot effectieve randkoppelingen.

Kernidee voor MaxCut:
  Een node v met buren {u1, u2, ..., uk} kan geëlimineerd worden als we
  het effect van v's optimale spin-keuze absorberen in de overgebleven
  koppelingen. Voor MaxCut op Ising (+/-1 spins):

  Degree-1 eliminatie (leaf):
    v -- u met gewicht w. v kiest altijd spin tegengesteld aan u.
    Elimineer v, voeg |w| toe aan een constante offset. Triviaal.

  Degree-2 eliminatie (chain):
    u -- v -- w met gewichten J_uv en J_vw.
    v kiest spin om max(J_uv*s_u*s_v + J_vw*s_v*s_w) te maximaliseren.
    Dit geeft een effectieve koppeling J_eff(u,w) en constante offset.
    J_eff = |J_uv + J_vw|/2 - |J_uv - J_vw|/2 (als J tekens gelijk)
    Exact: J_eff = (|J_uv + J_vw| - |J_uv - J_vw|) / 2

  Degree-3+ eliminatie (star):
    Exacte eliminatie produceert hogere-orde interacties (3-body etc).
    Benadering: elimineer via mean-field of greedy toewijzing.

Pipeline:
  1. Identificeer separators (bijv. via BFS layers of Metis-style bisectie)
  2. Elimineer leaves iteratief (gratis, exact)
  3. Elimineer degree-2 chains (exact, produceert effectieve edges)
  4. Optioneel: elimineer lage-graad interior nodes via mean-field
  5. Los gereduceerd probleem op
  6. Reconstruct volledige toewijzing door terugsubstitutie

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from collections import defaultdict
import time


class ReducedGraph:
    """Graaf met ondersteuning voor node-eliminatie en reconstructie."""

    def __init__(self, n_nodes, edges):
        self.original_n = n_nodes
        self.adj = defaultdict(dict)  # adj[u][v] = w
        self.alive = set()
        self.offset = 0.0  # constante bijdrage van geëlimineerde nodes
        self.elimination_order = []  # [(node, neighbors_at_elimination, rule)]

        for u, v, w in edges:
            u, v = int(u), int(v)
            if u == v:
                continue
            w = float(w)
            # Symmetrisch, combineer parallelle edges
            if v in self.adj[u]:
                self.adj[u][v] += w
                self.adj[v][u] += w
            else:
                self.adj[u][v] = w
                self.adj[v][u] = w
            self.alive.add(u)
            self.alive.add(v)

        # Voeg geïsoleerde nodes toe
        for i in range(n_nodes):
            self.alive.add(i)

    @property
    def n_alive(self):
        return len(self.alive)

    @property
    def n_edges(self):
        count = 0
        for u in self.alive:
            for v in self.adj[u]:
                if v in self.alive and v > u:
                    count += 1
        return count

    def degree(self, v):
        """Graad van node v (alleen levende buren)."""
        return sum(1 for u in self.adj[v] if u in self.alive)

    def neighbors(self, v):
        """Levende buren van v."""
        return [(u, self.adj[v][u]) for u in self.adj[v] if u in self.alive]

    def eliminate_leaf(self, v):
        """Elimineer degree-1 node. Exact."""
        nbrs = self.neighbors(v)
        if len(nbrs) != 1:
            return False

        u, w = nbrs[0]
        # v kiest spin tegengesteld aan u als w>0, gelijk als w<0
        # Bijdrage aan cut: |w|
        self.offset += abs(w)
        self.elimination_order.append((v, [(u, w)], 'leaf'))
        self.alive.discard(v)
        # Verwijder edge
        if v in self.adj[u]:
            del self.adj[u][v]
        return True

    def eliminate_degree2(self, v):
        """Elimineer degree-2 node (chain contraction). Exact."""
        nbrs = self.neighbors(v)
        if len(nbrs) != 2:
            return False

        u1, w1 = nbrs[0]
        u2, w2 = nbrs[1]

        # v zit in keten u1 -- v -- u2
        # v kiest spin s_v om J1*s_u1*s_v + J2*s_v*s_u2 te maximaliseren
        # = s_v * (J1*s_u1 + J2*s_u2)
        # Optimaal: s_v = sign(J1*s_u1 + J2*s_u2)
        #
        # Twee gevallen voor (s_u1, s_u2):
        # Gelijk (s_u1 = s_u2 = s):  optimaal s_v geeft |J1 + J2|
        # Ongelijk (s_u1 = -s_u2):   optimaal s_v geeft |J1 - J2|
        #
        # MaxCut bijdrage van v:
        #   als u1,u2 in zelfde partitie: |J1+J2|
        #   als u1,u2 in versch. partitie: |J1-J2|
        #
        # Dit is equivalent aan een effectieve koppeling:
        #   J_eff(u1,u2) = (|J1+J2| - |J1-J2|) / 2
        #   offset += (|J1+J2| + |J1-J2|) / 2

        sum_J = abs(w1 + w2)
        diff_J = abs(w1 - w2)
        J_eff = (sum_J - diff_J) / 2.0
        self.offset += (sum_J + diff_J) / 2.0

        self.elimination_order.append((v, [(u1, w1), (u2, w2)], 'chain'))
        self.alive.discard(v)

        # Verwijder edges naar v
        if v in self.adj[u1]:
            del self.adj[u1][v]
        if v in self.adj[u2]:
            del self.adj[u2][v]

        # Voeg effectieve edge toe (of combineer met bestaande)
        if abs(J_eff) > 1e-15:
            if u2 in self.adj[u1]:
                self.adj[u1][u2] += J_eff
                self.adj[u2][u1] += J_eff
            else:
                self.adj[u1][u2] = J_eff
                self.adj[u2][u1] = J_eff

        return True

    def eliminate_isolated(self, v):
        """Elimineer geïsoleerde node (degree 0)."""
        if self.degree(v) != 0:
            return False
        self.elimination_order.append((v, [], 'isolated'))
        self.alive.discard(v)
        return True

    def reduce_iterative(self, max_degree=2, max_rounds=100, verbose=False):
        """
        Iteratief elimineer nodes met degree <= max_degree.
        Herhaalt tot geen verdere reductie mogelijk.

        Returns: info dict
        """
        t0 = time.time()
        n_start = self.n_alive
        m_start = self.n_edges
        total_elim = 0
        n_leaves = 0
        n_chains = 0
        n_isolated = 0

        for round_i in range(max_rounds):
            eliminated_this_round = 0

            # Sorteer nodes op graad (laag eerst)
            nodes = sorted(self.alive, key=lambda v: self.degree(v))

            for v in nodes:
                if v not in self.alive:
                    continue
                d = self.degree(v)

                if d == 0:
                    if self.eliminate_isolated(v):
                        eliminated_this_round += 1
                        n_isolated += 1
                elif d == 1:
                    if self.eliminate_leaf(v):
                        eliminated_this_round += 1
                        n_leaves += 1
                elif d == 2 and max_degree >= 2:
                    if self.eliminate_degree2(v):
                        eliminated_this_round += 1
                        n_chains += 1

            total_elim += eliminated_this_round
            if eliminated_this_round == 0:
                break

        elapsed = time.time() - t0
        if verbose:
            print("  Schur reduction: %d -> %d nodes, %d -> %d edges (%.3fs)" % (
                n_start, self.n_alive, m_start, self.n_edges, elapsed))
            print("    Eliminated: %d leaves, %d chains, %d isolated" % (
                n_leaves, n_chains, n_isolated))
            print("    Offset: %.1f, rounds: %d" % (self.offset, round_i + 1))

        return {
            'n_start': n_start,
            'n_end': self.n_alive,
            'm_start': m_start,
            'm_end': self.n_edges,
            'n_eliminated': total_elim,
            'n_leaves': n_leaves,
            'n_chains': n_chains,
            'n_isolated': n_isolated,
            'offset': self.offset,
            'rounds': round_i + 1,
            'time': elapsed,
        }

    def get_reduced_edges(self):
        """Retourneer edges van de gereduceerde graaf."""
        edges = []
        seen = set()
        for u in self.alive:
            for v, w in self.adj[u].items():
                if v in self.alive and (min(u, v), max(u, v)) not in seen:
                    if abs(w) > 1e-15:
                        edges.append((u, v, w))
                        seen.add((min(u, v), max(u, v)))
        return edges

    def get_node_mapping(self):
        """Maak compacte node-mapping voor gereduceerde graaf."""
        alive_sorted = sorted(self.alive)
        old_to_new = {old: new for new, old in enumerate(alive_sorted)}
        new_to_old = {new: old for old, new in old_to_new.items()}
        return old_to_new, new_to_old

    def reconstruct_assignment(self, reduced_assignment):
        """
        Reconstrueer volledige toewijzing uit gereduceerde toewijzing
        door geëlimineerde nodes terug te substitueren.

        Args:
            reduced_assignment: dict {node: 0/1} voor levende nodes

        Returns:
            full_assignment: dict {node: 0/1} voor alle nodes
        """
        assignment = dict(reduced_assignment)

        # Loop eliminatie-volgorde achterstevoren
        for v, nbrs, rule in reversed(self.elimination_order):
            if rule == 'isolated':
                assignment[v] = 0  # willekeurig
            elif rule == 'leaf':
                u, w = nbrs[0]
                s_u = 1.0 if assignment.get(u, 0) == 1 else -1.0
                # v kiest spin om w * s_u * s_v te maximaliseren
                # Maximaal als s_v = -sign(w) * s_u (snijd de edge als w>0)
                if w > 0:
                    assignment[v] = 0 if assignment.get(u, 0) == 1 else 1
                else:
                    assignment[v] = assignment.get(u, 0)
            elif rule == 'chain':
                u1, w1 = nbrs[0]
                u2, w2 = nbrs[1]
                s_u1 = 1.0 if assignment.get(u1, 0) == 1 else -1.0
                s_u2 = 1.0 if assignment.get(u2, 0) == 1 else -1.0
                # v kiest spin om w1*s_u1*s_v + w2*s_v*s_u2 te maximaliseren
                # = s_v * (w1*s_u1 + w2*s_u2)
                field = w1 * s_u1 + w2 * s_u2
                # Maximaal als s_v = sign(field) -> snij zoveel mogelijk
                if field > 0:
                    assignment[v] = 1
                elif field < 0:
                    assignment[v] = 0
                else:
                    assignment[v] = 0  # willekeurig bij gelijkspel

        return assignment


def schur_maxcut(n_nodes, edges, solver_fn, max_degree=2,
                 verbose=False):
    """
    Schur-complement preprocessing + solve + reconstructie.

    Pipeline:
      1. Reduceer graaf (elimineer leaves en chains)
      2. Los gereduceerd probleem op met solver_fn
      3. Reconstrueer volledige toewijzing
      4. Greedy refinement op origineel

    Args:
        n_nodes: oorspronkelijk aantal nodes
        edges: originele edges (u, v, w)
        solver_fn: functie(n_nodes, edges) -> (cut, assignment_dict)
        max_degree: max graad voor eliminatie (1=alleen leaves, 2=+chains)
        verbose: print voortgang

    Returns:
        cut: cut op originele graaf
        assignment: dict {node: 0/1}
        info: dict met statistieken
    """
    from feedback_edge_solver import _greedy_refine

    t0 = time.time()

    # Stap 1: Reduceer
    rg = ReducedGraph(n_nodes, edges)
    reduce_info = rg.reduce_iterative(max_degree=max_degree, verbose=verbose)

    # Stap 2: Los gereduceerd probleem op
    reduced_edges = rg.get_reduced_edges()
    old_to_new, new_to_old = rg.get_node_mapping()
    n_reduced = len(old_to_new)

    # Remap edges naar compacte indices
    remapped_edges = []
    for u, v, w in reduced_edges:
        remapped_edges.append((old_to_new[u], old_to_new[v], w))

    if verbose:
        print("  Solving reduced graph: n=%d, m=%d" % (n_reduced, len(remapped_edges)))

    t_solve = time.time()
    if n_reduced > 0 and len(remapped_edges) > 0:
        reduced_cut, reduced_assign = solver_fn(n_reduced, remapped_edges)
    else:
        reduced_cut = 0
        reduced_assign = {i: 0 for i in range(n_reduced)}
    solve_time = time.time() - t_solve

    # Remap terug naar originele indices
    orig_reduced_assign = {}
    for new_idx, label in reduced_assign.items():
        if new_idx in new_to_old:
            orig_reduced_assign[new_to_old[new_idx]] = label

    # Stap 3: Reconstrueer volledige toewijzing
    full_assign = rg.reconstruct_assignment(orig_reduced_assign)

    # Stap 4: Greedy refinement op originele graaf
    refined_assign, refined_cut, n_flips = _greedy_refine(
        n_nodes, edges, full_assign, max_passes=15)

    total_time = time.time() - t0

    if verbose:
        # Vergelijk: reconstructed vs refined
        recon_cut = sum(float(w) for u, v, w in edges
                        if full_assign.get(int(u), 0) != full_assign.get(int(v), 0))
        print("  Reconstructed cut: %.1f, after greedy: %.1f (+%.1f, %d flips)" % (
            recon_cut, refined_cut, refined_cut - recon_cut, n_flips))
        print("  Offset from eliminated nodes: %.1f" % rg.offset)
        print("  Total time: %.3fs (reduce %.3fs, solve %.3fs)" % (
            total_time, reduce_info['time'], solve_time))

    info = {
        'reduce_info': reduce_info,
        'n_reduced': n_reduced,
        'm_reduced': len(remapped_edges),
        'reduced_cut': reduced_cut,
        'offset': rg.offset,
        'solve_time': solve_time,
        'total_time': total_time,
        'n_flips': n_flips,
        'reduction_ratio': 1.0 - n_reduced / max(n_nodes, 1),
    }

    return refined_cut, refined_assign, info


# ============================================================
# Separator finding (voor toekomstig gebruik)
# ============================================================

def find_bfs_separator(n_nodes, edges, source=None):
    """
    Vind een separator via BFS-layers.
    Kies de layer met de kleinste breedte als separator.

    Returns:
        separator: set van nodes
        interior: set van nodes aan een kant
        exterior: set van nodes aan andere kant
    """
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    if source is None:
        # Kies node met laagste graad
        degrees = {v: len(adj[v]) for v in range(n_nodes)}
        source = min(degrees, key=degrees.get)

    # BFS layers
    visited = {source}
    layers = [[source]]
    while len(visited) < n_nodes:
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
        return set(), set(range(n_nodes)), set()

    # Kies smalste layer als separator
    min_width = len(layers[1])
    min_idx = 1
    for i in range(1, len(layers) - 1):
        if len(layers[i]) < min_width:
            min_width = len(layers[i])
            min_idx = i

    separator = set(layers[min_idx])
    interior = set()
    for i in range(min_idx):
        interior.update(layers[i])
    exterior = set()
    for i in range(min_idx + 1, len(layers)):
        exterior.update(layers[i])

    return separator, interior, exterior


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='B119: Schur-Complement Separator Elimination')
    parser.add_argument('--graph', default='grid:10x4',
                        help='Graph: grid:LxW')
    parser.add_argument('--signed', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.graph.startswith('grid:'):
        dims = args.graph[5:].split('x')
        Lx, Ly = int(dims[0]), int(dims[1])
        n = Lx * Ly
        rng = np.random.default_rng(args.seed)
        edges = []
        for x in range(Lx):
            for y in range(Ly):
                i = x * Ly + y
                if x + 1 < Lx:
                    w = rng.choice([-1.0, 1.0]) if args.signed else 1.0
                    edges.append((i, (x + 1) * Ly + y, w))
                if y + 1 < Ly:
                    w = rng.choice([-1.0, 1.0]) if args.signed else 1.0
                    edges.append((i, x * Ly + y + 1, w))
    else:
        print("Onbekend formaat: %s" % args.graph)
        import sys
        sys.exit(1)

    print("Graph: %s, n=%d, m=%d" % (args.graph, n, len(edges)))

    # Toon reductie
    rg = ReducedGraph(n, edges)
    info = rg.reduce_iterative(verbose=True)
    print("\nGereduceerd: %d nodes, %d edges" % (rg.n_alive, rg.n_edges))
