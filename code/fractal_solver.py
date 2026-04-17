#!/usr/bin/env python3
"""
fractal_solver.py - B79: Fractal Quantum Solver (FQS) voor MaxCut.

Fractal coarsening: na p=1 QAOA, batch-merge alle sterk gecorreleerde
node-paren (|⟨ZZ⟩| > threshold), herbouw gewogen graaf. Herhaal tot
de graaf klein genoeg is voor brute force. O(log N) schaling.

Verschil met RQAOA (B47):
  - RQAOA: merge ÉÉN edge per iteratie → O(N) iteraties
  - FQS:   batch-merge ALLE sterke edges per ronde → O(log N) rondes

Gebruik:
  python fractal_solver.py --Lx 8 --Ly 3
  python fractal_solver.py --Lx 20 --Ly 4 --gpu
  python fractal_solver.py --Lx 8 --Ly 4 --merge-threshold 0.5
"""

import numpy as np
import time
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph, GeneralQAOA, brute_force_maxcut

try:
    from bm_solver import bm_sdp_solve_fast, bm_sdp_solve
    BM_AVAILABLE = True
except ImportError:
    BM_AVAILABLE = False


# =====================================================================
# Union-Find voor batch-merge
# =====================================================================

class UnionFind:
    """Disjoint-set voor het clusteren van sterk gecorreleerde nodes."""

    def __init__(self, nodes):
        """nodes: lijst van node-labels."""
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def components(self):
        """Return dict {root: [nodes...]}."""
        comp = defaultdict(list)
        for n in self.parent:
            comp[self.find(n)].append(n)
        return dict(comp)


# =====================================================================
# FQS Engine
# =====================================================================

class FractalSolver:
    """Fractal Quantum Solver: batch-coarsen via ⟨ZZ⟩ correlaties.

    Algoritme:
      1. Evalueer alle ⟨ZZ⟩ (lightcone of state vector)
      2. Batch-merge nodes met |⟨ZZ⟩| > threshold via union-find
      3. Herbouw gewogen graaf met super-nodes
      4. Herhaal tot n_nodes <= brute_threshold
      5. Brute force + reconstructie + local search
    """

    def __init__(self, graph, p=1, merge_threshold=0.7,
                 brute_threshold=22, verbose=True):
        """
        Args:
            graph: WeightedGraph
            p: QAOA-diepte per ronde (default 1)
            merge_threshold: |⟨ZZ⟩| drempel voor merge (default 0.7)
            brute_threshold: brute force bij n_nodes <= dit (default 22)
            verbose: print voortgang
        """
        self.original_graph = graph.copy()
        self.p = p
        self.merge_threshold = merge_threshold
        self.brute_threshold = brute_threshold
        self.verbose = verbose

    # -----------------------------------------------------------------
    # ZZ evaluatie op willekeurige gewogen graaf (state vector, <=22 nodes)
    # -----------------------------------------------------------------

    def _eval_zz_general(self, graph):
        """Evalueer ⟨ZZ⟩ voor alle edges via GeneralQAOA state vector.

        Returns: dict {(i,j): zz} met i < j.
        """
        n = graph.n_nodes
        if n == 0:
            return {}

        qaoa = GeneralQAOA(graph, verbose=False)
        _, opt_g, opt_b = qaoa.optimize_p1(n_gamma=10, n_beta=10)
        zz = qaoa.eval_all_zz(self.p, opt_g, opt_b)
        return zz

    # -----------------------------------------------------------------
    # ZZ evaluatie via Burer-Monteiro (voor grafen > 22 nodes)
    # -----------------------------------------------------------------

    def _eval_zz_bm(self, graph):
        """Evalueer ZZ-proxy voor alle edges via BM inner products.

        Gebruikt v_i . v_j als proxy voor ⟨ZZ⟩:
          v_i . v_j > 0 → positief gecorreleerd (zelfde partitie)
          v_i . v_j < 0 → anti-gecorreleerd (verschillende partities)

        Schaalbaar tot duizenden nodes.
        Returns: dict {(i,j): zz_proxy}, bm_result
        """
        n = graph.n_nodes
        nodes = graph.nodes
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Bouw edge-lijst in BM-formaat
        bm_edges = []
        for i, j, w in graph.edges():
            bm_edges.append((node_to_idx[i], node_to_idx[j], w))

        # BM solve
        if n <= 2000:
            bm = bm_sdp_solve_fast(n, bm_edges, n_restarts=3,
                                    max_iter=300, verbose=False)
        else:
            bm = bm_sdp_solve(n, bm_edges, n_restarts=3,
                               max_iter=300, verbose=False)

        V = bm['vectors']

        # Inner products als ZZ-proxy
        zz = {}
        for i, j, w in graph.edges():
            ii, jj = node_to_idx[i], node_to_idx[j]
            corr = float(np.dot(V[ii], V[jj]))
            zz[(i, j)] = corr

        if self.verbose:
            zz_arr = np.array(list(zz.values()))
            print("    BM <ZZ> proxy: range [%.4f, %.4f], "
                  "%d anti-gecorreleerd" %
                  (zz_arr.min(), zz_arr.max(), np.sum(zz_arr < 0)))

        return zz, bm

    # -----------------------------------------------------------------
    # BM-rounding als solver (voor grafen > brute_threshold)
    # -----------------------------------------------------------------

    def _solve_bm(self, graph):
        """Los MaxCut op via BM-rounding. Schaalbaar tot duizenden nodes.

        Returns: (cut, assignment)
        """
        nodes = graph.nodes
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        bm_edges = []
        for i, j, w in graph.edges():
            bm_edges.append((node_to_idx[i], node_to_idx[j], w))

        n = graph.n_nodes
        if n <= 2000:
            bm = bm_sdp_solve_fast(n, bm_edges, n_restarts=5, verbose=False)
        else:
            bm = bm_sdp_solve(n, bm_edges, n_restarts=5, verbose=False)

        # Vertaal assignment terug naar originele node-labels
        assignment = {}
        for node in nodes:
            idx = node_to_idx[node]
            assignment[node] = 1 - 2 * bm['best_assignment'][idx]  # 0/1 → +1/-1

        cut = self.eval_cut(graph, assignment)
        return cut, assignment

    # -----------------------------------------------------------------
    # Batch-merge via union-find
    # -----------------------------------------------------------------

    def _batch_merge(self, graph, zz_values, threshold):
        """Merge alle node-paren met |⟨ZZ⟩| > threshold.

        1. Identificeer sterke edges
        2. Union-find → componenten (super-nodes)
        3. Bouw gereduceerde gewogen graaf
        4. Sla merge-info op voor reconstructie

        Args:
            graph: huidige WeightedGraph
            zz_values: dict {(i,j): zz}
            threshold: |⟨ZZ⟩| drempel

        Returns:
            new_graph: gereduceerde WeightedGraph (of None als geen merges)
            merge_info: dict met reconstructie-info
        """
        # Vind sterke edges
        strong = []
        for (i, j), zz in zz_values.items():
            if abs(zz) > threshold:
                strong.append((i, j, zz))

        if not strong:
            return None, None

        # Union-find op alle nodes van de graaf
        uf = UnionFind(graph.nodes)
        for i, j, zz in strong:
            uf.union(i, j)

        comps = uf.components()
        n_before = graph.n_nodes
        n_after = len(comps)

        if n_after == n_before:
            return None, None  # geen merges

        if self.verbose:
            sizes = sorted([len(c) for c in comps.values()], reverse=True)
            print("    %d sterke edges, %d -> %d super-nodes (grootste: %s)" %
                  (len(strong), n_before, n_after,
                   sizes[:5] if len(sizes) > 5 else sizes))

        # Bouw gereduceerde graaf
        new_graph = WeightedGraph()
        for root in comps:
            new_graph.add_node(root)

        # Aggregeer edges: som van gewichten tussen componenten
        agg = defaultdict(float)
        for i, j, w in graph.edges():
            ri, rj = uf.find(i), uf.find(j)
            if ri != rj:
                key = (min(ri, rj), max(ri, rj))
                agg[key] += w

        for (ri, rj), w in agg.items():
            if abs(w) > 1e-12:
                new_graph.add_edge(ri, rj, w)

        # Sla zz-waarden op per component voor reconstructie
        # Per component: wie zit erin, en wat is de correlatie met de root
        merge_info = {
            'components': comps,         # {root: [nodes]}
            'zz_to_root': {},            # {node: zz met root (>0 = zelfde spin)}
            'internal_offset': 0.0,      # cut-bijdrage van gemergte edges
        }

        # Bereken interne offset (edges binnen componenten)
        internal = 0.0
        for i, j, zz in strong:
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                # Interne edge: bijdrage aan cut = (1 - zz)/2 * weight
                w = 0
                for a, b, ww in graph.edges():
                    if (a == i and b == j) or (a == j and b == i):
                        w = ww
                        break
                internal += w * (1 - (1 if zz > 0 else -1)) / 2

        merge_info['internal_offset'] = internal

        # ZZ-correlatie van elke node t.o.v. root van zijn component
        # Propageer via BFS over sterke edges
        strong_adj = defaultdict(list)
        strong_zz = {}
        for i, j, zz in strong:
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:  # alleen interne edges
                strong_adj[i].append(j)
                strong_adj[j].append(i)
                strong_zz[(min(i,j), max(i,j))] = zz

        for root, members in comps.items():
            merge_info['zz_to_root'][root] = 1  # root correleert positief met zichzelf
            if len(members) <= 1:
                continue

            # BFS vanuit root
            visited = {root}
            queue = [root]
            spin_rel = {root: 1}  # +1 = zelfde spin als root, -1 = opposite

            while queue:
                node = queue.pop(0)
                for nb in strong_adj.get(node, []):
                    if nb in visited:
                        continue
                    if uf.find(nb) != root:
                        continue
                    # Correlatie node↔nb
                    key = (min(node, nb), max(node, nb))
                    zz = strong_zz.get(key, 0)
                    # zz > 0 → zelfde spin als node → spin_rel[nb] = spin_rel[node]
                    # zz < 0 → tegengestelde spin → spin_rel[nb] = -spin_rel[node]
                    spin_rel[nb] = spin_rel[node] * (1 if zz > 0 else -1)
                    visited.add(nb)
                    queue.append(nb)

            for member in members:
                merge_info['zz_to_root'][member] = spin_rel.get(member, 1)

        return new_graph, merge_info

    # -----------------------------------------------------------------
    # Reconstructie
    # -----------------------------------------------------------------

    @staticmethod
    def reconstruct(merge_history, super_assignment):
        """Reconstrueer volledige assignment uit super-node assignment.

        Args:
            merge_history: lijst van merge_info dicts (oudste eerst)
            super_assignment: dict {super_node: +1/-1}

        Returns:
            dict {node: +1/-1} voor alle originele nodes
        """
        assignment = dict(super_assignment)

        for info in reversed(merge_history):
            new_assignment = {}
            for root, members in info['components'].items():
                root_spin = assignment.get(root, 1)
                for member in members:
                    rel = info['zz_to_root'].get(member, 1)
                    new_assignment[member] = root_spin * rel
            # Behoud nodes die niet in een component zaten
            for node, spin in assignment.items():
                if node not in new_assignment:
                    new_assignment[node] = spin
            assignment = new_assignment

        return assignment

    # -----------------------------------------------------------------
    # Local search (steepest descent + restarts)
    # -----------------------------------------------------------------

    def _local_search(self, graph, assignment, max_rounds=100, n_restarts=None):
        """Multi-strategy local search op de originele graaf.

        Returns: (cut, assignment, n_flips)
        """
        assignment = dict(assignment)
        nodes = list(graph.adj.keys())
        n_nodes = len(nodes)
        n_flips_total = 0

        if n_restarts is None:
            n_restarts = min(50, max(10, n_nodes // 4))

        def _delta(ass, node):
            d = 0.0
            z = ass.get(node, 1)
            for nb, w in graph.adj[node].items():
                d += w * z * ass.get(nb, 1)
            return d

        def _steepest(ass):
            flips = 0
            for _ in range(max_rounds):
                best_node, best_d = None, 1e-12
                for node in nodes:
                    d = _delta(ass, node)
                    if d > best_d:
                        best_d = d
                        best_node = node
                if best_node is None:
                    break
                ass[best_node] = -ass[best_node]
                flips += 1
            return flips

        def _eval_cut(ass):
            c = 0.0
            for i, j, w in graph.edges():
                c += w * (1 - ass.get(i, 1) * ass.get(j, 1)) / 2
            return c

        # Fase 1: steepest descent
        n_flips_total += _steepest(assignment)
        best_cut = _eval_cut(assignment)
        best_ass = dict(assignment)

        # Fase 2: random restarts
        rng = np.random.RandomState(42)
        for restart in range(n_restarts):
            trial = dict(best_ass)
            frac = 0.02 + 0.23 * (restart / max(1, n_restarts - 1))
            k = max(2, int(n_nodes * frac))
            flip_nodes = rng.choice(nodes, size=min(k, n_nodes), replace=False)
            for node in flip_nodes:
                trial[node] = -trial.get(node, 1)
            flips = _steepest(trial)
            n_flips_total += flips + k
            trial_cut = _eval_cut(trial)
            if trial_cut > best_cut:
                best_cut = trial_cut
                best_ass = trial

        return best_cut, best_ass, n_flips_total

    # -----------------------------------------------------------------
    # Eval cut op originele graaf
    # -----------------------------------------------------------------

    @staticmethod
    def eval_cut(graph, assignment):
        """Bereken cut-waarde."""
        c = 0.0
        for i, j, w in graph.edges():
            c += w * (1 - assignment.get(i, 1) * assignment.get(j, 1)) / 2
        return c

    # -----------------------------------------------------------------
    # solve(): willekeurige graaf (≤ 22 nodes per ronde, of greedy)
    # -----------------------------------------------------------------

    def solve(self, max_rounds=10, local_search=True):
        """FQS op willekeurige graaf.

        Returns: (cut, assignment, ratio, info)
        """
        t0 = time.time()
        graph = self.original_graph.copy()
        merge_history = []
        n_start = graph.n_nodes
        n_edges_orig = self.original_graph.n_edges

        if self.verbose:
            print("\n=== B79 FQS: %d nodes, %d edges ===" %
                  (n_start, n_edges_orig))

        # BM direct-solve shortcut voor middelgrote grafen
        if BM_AVAILABLE and n_start <= 1500 and n_start > self.brute_threshold:
            if self.verbose:
                print("\n  BM direct-solve (n=%d ≤ 1500)..." % n_start)
            bm_cut, bm_assign = self._solve_bm(graph)
            if self.verbose:
                print("    BM-rounding: cut=%.1f ratio=%.6f" %
                      (bm_cut, bm_cut / n_edges_orig if n_edges_orig > 0 else 0))
            if local_search:
                ls_cut, bm_assign, n_flips = self._local_search(
                    self.original_graph, bm_assign)
                if self.verbose:
                    print("    Local search: %d flips, cut %.1f -> %.1f" %
                          (n_flips, bm_cut, ls_cut))
                final_cut = ls_cut
            else:
                final_cut = bm_cut
            ratio = final_cut / n_edges_orig if n_edges_orig > 0 else 0
            elapsed = time.time() - t0
            if self.verbose:
                print("  Resultaat: cut=%.1f ratio=%.6f [%.2fs]" %
                      (final_cut, ratio, elapsed))
            return final_cut, bm_assign, ratio, {
                'rounds': 0, 'n_start': n_start, 'n_final': n_start,
                'raw_cut': bm_cut, 'final_cut': final_cut,
                'ratio': ratio, 'elapsed': elapsed, 'method': 'bm_direct'}

        threshold = self.merge_threshold
        round_num = 0

        while graph.n_nodes > self.brute_threshold and round_num < max_rounds:
            round_num += 1
            if self.verbose:
                print("\n  Ronde %d: %d nodes, %d edges (threshold=%.2f)" %
                      (round_num, graph.n_nodes, graph.n_edges, threshold))

            # Evalueer ZZ
            if graph.n_nodes <= 22:
                zz = self._eval_zz_general(graph)
                if self.verbose:
                    zz_arr = np.array(list(zz.values()))
                    print("    <ZZ> range: [%.4f, %.4f]" %
                          (zz_arr.min(), zz_arr.max()) if len(zz_arr) else "    Geen edges")
            elif BM_AVAILABLE:
                # BM inner products als ZZ-proxy (B91)
                if self.verbose:
                    print("    BM-correlaties voor %d nodes..." % graph.n_nodes)
                zz, _ = self._eval_zz_bm(graph)
            else:
                # Fallback: gewicht-heuristiek (geen BM beschikbaar)
                if self.verbose:
                    print("    Gewicht-heuristiek voor %d nodes..." % graph.n_nodes)
                zz = {}
                max_w = max(abs(w) for _, _, w in graph.edges()) if graph.n_edges > 0 else 1.0
                for i, j, w in graph.edges():
                    zz[(i, j)] = -w / max_w

            # Batch merge
            new_graph, merge_info = self._batch_merge(graph, zz, threshold)

            if new_graph is None:
                # Verlaag threshold
                threshold = max(0.1, threshold - 0.15)
                if self.verbose:
                    print("    Geen merges. Verlaag threshold naar %.2f" % threshold)
                new_graph, merge_info = self._batch_merge(graph, zz, threshold)
                if new_graph is None:
                    if self.verbose:
                        print("    Nog steeds geen merges. Stop coarsening.")
                    break

            merge_history.append(merge_info)
            graph = new_graph

        # Brute force op gereduceerde graaf
        if self.verbose:
            print("\n  Brute force op %d nodes, %d edges" %
                  (graph.n_nodes, graph.n_edges))

        if graph.n_nodes > 0 and graph.n_edges > 0:
            if graph.n_nodes <= 25:
                bf_cut, bf_assign = brute_force_maxcut(graph)
                if self.verbose:
                    print("    Brute force cut: %.1f" % bf_cut)
            elif BM_AVAILABLE:
                # BM-rounding als solver (B91)
                bf_cut, bf_assign = self._solve_bm(graph)
                if self.verbose:
                    print("    BM-rounding cut: %.1f (%d nodes)" %
                          (bf_cut, graph.n_nodes))
            else:
                bf_assign = {n: 1 for n in graph.nodes}
                bf_cut = 0
                if self.verbose:
                    print("    Geen solver beschikbaar, triviale assignment")
        else:
            bf_assign = {n: 1 for n in graph.nodes}
            bf_cut = 0
            if self.verbose:
                print("    Lege graaf, cut=0")

        # Reconstructie
        full_assign = self.reconstruct(merge_history, bf_assign)
        raw_cut = self.eval_cut(self.original_graph, full_assign)

        if self.verbose:
            print("\n  Gereconstrueerde cut: %.1f / %d edges = %.6f" %
                  (raw_cut, n_edges_orig,
                   raw_cut / n_edges_orig if n_edges_orig > 0 else 0))

        # Local search
        if local_search:
            ls_cut, full_assign, n_flips = self._local_search(
                self.original_graph, full_assign)
            if self.verbose:
                delta = ls_cut - raw_cut
                print("  Local search: %d flips, cut %.1f -> %.1f (%+.1f)" %
                      (n_flips, raw_cut, ls_cut, delta))
            final_cut = ls_cut
        else:
            final_cut = raw_cut

        ratio = final_cut / n_edges_orig if n_edges_orig > 0 else 0
        elapsed = time.time() - t0

        if self.verbose:
            print("\n" + "=" * 60)
            print("  FQS Resultaat:")
            print("    Cut: %.1f / %d edges" % (final_cut, n_edges_orig))
            print("    Ratio: %.6f" % ratio)
            print("    Rondes: %d (%d -> %d nodes)" %
                  (round_num, n_start, graph.n_nodes))
            print("    Tijd: %.2fs" % elapsed)
            print("=" * 60)

        info = {
            'rounds': round_num,
            'n_start': n_start,
            'n_final': graph.n_nodes,
            'raw_cut': raw_cut,
            'final_cut': final_cut,
            'ratio': ratio,
            'elapsed': elapsed,
        }
        return final_cut, full_assign, ratio, info

    # -----------------------------------------------------------------
    # solve_grid(): grid-specifiek met LightconeQAOA voor eerste ronde
    # -----------------------------------------------------------------

    def solve_grid(self, Lx, Ly, gammas=None, betas=None,
                   gpu=False, chi=None, max_rounds=10, local_search=True):
        """FQS voor grid-grafen met LightconeQAOA.

        Eerste ronde: LightconeQAOA → exacte ⟨ZZ⟩ (willekeurige grootte).
        Volgende rondes: GeneralQAOA state vector (als n ≤ 22) of greedy.

        Returns: (cut, assignment, ratio, info)
        """
        from lightcone_qaoa import LightconeQAOA

        t0 = time.time()
        n_nodes = Lx * Ly
        graph = WeightedGraph.grid(Lx, Ly)
        self.original_graph = graph.copy()
        n_edges = graph.n_edges

        if self.verbose:
            print("\n=== B79 FQS Grid: %dx%d (%d nodes, %d edges) ===" %
                  (Lx, Ly, n_nodes, n_edges))

        # --- BM direct-solve shortcut (B91 integratie) ---
        # Voor grafen ≤ 1500 nodes: BM+LS is sneller en beter dan
        # coarsening (dat faalt op uniform-gewicht grafen door gebrek
        # aan ZZ-differentiatie).
        if BM_AVAILABLE and n_nodes <= 1500:
            if self.verbose:
                print("\n  BM direct-solve (n=%d ≤ 1500)..." % n_nodes)
            t_bm = time.time()
            bm_cut, bm_assign = self._solve_bm(graph)
            if self.verbose:
                print("    BM-rounding: cut=%.1f ratio=%.6f [%.2fs]" %
                      (bm_cut, bm_cut / n_edges if n_edges > 0 else 0,
                       time.time() - t_bm))

            # Local search
            if local_search:
                ls_cut, bm_assign, n_flips = self._local_search(
                    graph, bm_assign)
                if self.verbose:
                    print("    Local search: %d flips, cut %.1f -> %.1f (%+.1f)" %
                          (n_flips, bm_cut, ls_cut, ls_cut - bm_cut))
                final_cut = ls_cut
            else:
                final_cut = bm_cut

            ratio = final_cut / n_edges if n_edges > 0 else 0
            elapsed = time.time() - t0

            if self.verbose:
                print("")
                print("=" * 60)
                print("  B79 FQS Grid Resultaat (BM direct):")
                print("    Cut: %.1f / %d edges" % (final_cut, n_edges))
                print("    Ratio: %.6f" % ratio)
                print("    Methode: BM direct-solve + local search")
                print("    Tijd: %.2fs" % elapsed)
                print("=" * 60)

            info = {
                'rounds': 0,
                'n_start': n_nodes,
                'n_final': n_nodes,
                'raw_cut': bm_cut,
                'final_cut': final_cut,
                'ratio': ratio,
                'elapsed': elapsed,
                'method': 'bm_direct',
            }
            return final_cut, bm_assign, ratio, info

        # --- Ronde 1: LightconeQAOA → alle ⟨ZZ⟩ ---
        if self.verbose:
            print("\n  Ronde 1: LightconeQAOA p=%d..." % self.p)
        t1 = time.time()

        lc = LightconeQAOA(Lx, Ly, verbose=False, chi=chi, gpu=gpu)

        if gammas is None or betas is None:
            _, gammas, betas, opt_info = lc.optimize(
                self.p, n_gamma=12, n_beta=12, refine=True)
            if self.verbose:
                print("    Geoptimaliseerd: gamma=%s beta=%s ratio=%.6f [%.1fs]" %
                      (gammas, betas, opt_info.get('grid_ratio', 0),
                       opt_info.get('total_time', 0)))

        # Evalueer per-edge ZZ
        zz_values = {}
        for etype, ex, ey in lc.edges:
            zz = lc.eval_edge(etype, ex, ey, self.p, gammas, betas)
            if etype == 'h':
                i = ex * Ly + ey
                j = (ex + 1) * Ly + ey
            else:  # 'v'
                i = ex * Ly + ey
                j = ex * Ly + ey + 1
            zz_values[(min(i, j), max(i, j))] = zz

        if self.verbose:
            zz_arr = np.array(list(zz_values.values()))
            n_cut = np.sum(zz_arr < 0)
            print("    %d edges, <ZZ> range [%.4f, %.4f], %d anti-gecorreleerd [%.1fs]" %
                  (len(zz_arr), zz_arr.min(), zz_arr.max(), n_cut,
                   time.time() - t1))

        # Batch merge
        threshold = self.merge_threshold
        new_graph, merge_info = self._batch_merge(graph, zz_values, threshold)
        merge_history = []

        if new_graph is not None:
            merge_history.append(merge_info)
            graph = new_graph
        else:
            # Verlaag threshold
            threshold = max(0.1, threshold - 0.15)
            if self.verbose:
                print("    Geen merges bij %.2f. Probeer %.2f..." %
                      (self.merge_threshold, threshold))
            new_graph, merge_info = self._batch_merge(
                graph, zz_values, threshold)
            if new_graph is not None:
                merge_history.append(merge_info)
                graph = new_graph

        # --- Rondes 2+: GeneralQAOA of greedy ---
        round_num = 1
        while graph.n_nodes > self.brute_threshold and round_num < max_rounds:
            round_num += 1
            if self.verbose:
                print("\n  Ronde %d: %d nodes, %d edges" %
                      (round_num, graph.n_nodes, graph.n_edges))

            if graph.n_nodes <= 22:
                zz = self._eval_zz_general(graph)
            elif BM_AVAILABLE:
                if self.verbose:
                    print("    BM-correlaties voor %d nodes..." % graph.n_nodes)
                zz, _ = self._eval_zz_bm(graph)
            else:
                zz = {}
                max_w = max(abs(w) for _, _, w in graph.edges()) if graph.n_edges > 0 else 1.0
                for i, j, w in graph.edges():
                    zz[(i, j)] = -w / max_w

            new_graph, merge_info = self._batch_merge(graph, zz, threshold)
            if new_graph is None:
                threshold = max(0.1, threshold - 0.15)
                new_graph, merge_info = self._batch_merge(graph, zz, threshold)
                if new_graph is None:
                    break

            merge_history.append(merge_info)
            graph = new_graph

        # --- Brute force ---
        if self.verbose:
            print("\n  Brute force op %d nodes, %d edges" %
                  (graph.n_nodes, graph.n_edges))

        if graph.n_nodes > 0 and graph.n_edges > 0:
            if graph.n_nodes <= 25:
                bf_cut, bf_assign = brute_force_maxcut(graph)
                if self.verbose:
                    print("    Brute force cut: %.1f" % bf_cut)
            elif BM_AVAILABLE:
                bf_cut, bf_assign = self._solve_bm(graph)
                if self.verbose:
                    print("    BM-rounding cut: %.1f (%d nodes)" %
                          (bf_cut, graph.n_nodes))
            else:
                bf_assign = {n: 1 for n in graph.nodes}
                bf_cut = 0
        else:
            bf_assign = {n: 1 for n in graph.nodes}
            bf_cut = 0

        # --- Reconstructie ---
        full_assign = self.reconstruct(merge_history, bf_assign)
        raw_cut = self.eval_cut(self.original_graph, full_assign)

        if self.verbose:
            print("  Gereconstrueerd: cut=%.1f ratio=%.6f" %
                  (raw_cut, raw_cut / n_edges if n_edges > 0 else 0))

        # --- Local search ---
        if local_search:
            ls_cut, full_assign, n_flips = self._local_search(
                self.original_graph, full_assign)
            if self.verbose:
                print("  Local search: %d flips, cut %.1f -> %.1f (%+.1f)" %
                      (n_flips, raw_cut, ls_cut, ls_cut - raw_cut))
            final_cut = ls_cut
        else:
            final_cut = raw_cut

        ratio = final_cut / n_edges if n_edges > 0 else 0
        elapsed = time.time() - t0

        if self.verbose:
            print("")
            print("=" * 60)
            print("  B79 FQS Grid Resultaat:")
            print("    Cut: %.1f / %d edges" % (final_cut, n_edges))
            print("    Ratio: %.6f" % ratio)
            print("    Rondes: %d (%d -> %d nodes)" %
                  (round_num, n_nodes, graph.n_nodes))
            print("    Tijd: %.2fs" % elapsed)
            print("=" * 60)

        info = {
            'rounds': round_num,
            'n_start': n_nodes,
            'n_final': graph.n_nodes,
            'raw_cut': raw_cut,
            'final_cut': final_cut,
            'ratio': ratio,
            'elapsed': elapsed,
            'gammas': list(gammas),
            'betas': list(betas),
        }
        return final_cut, full_assign, ratio, info


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B79: FQS - Fractal Quantum Solver voor MaxCut')
    parser.add_argument('--Lx', type=int, default=8)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--merge-threshold', type=float, default=0.7)
    parser.add_argument('--brute-threshold', type=int, default=22)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--chi', type=int, default=None)
    parser.add_argument('--no-ls', action='store_true',
                        help='Skip local search')
    args = parser.parse_args()

    graph = WeightedGraph.grid(args.Lx, args.Ly)
    solver = FractalSolver(
        graph, p=args.p,
        merge_threshold=args.merge_threshold,
        brute_threshold=args.brute_threshold,
        verbose=True)

    cut, assignment, ratio, info = solver.solve_grid(
        args.Lx, args.Ly,
        gpu=args.gpu, chi=args.chi,
        local_search=not args.no_ls)

    # Vergelijking met standaard RQAOA
    print("")
    print("--- Vergelijking met RQAOA ---")
    from rqaoa import RQAOA
    rqaoa = RQAOA(graph, p=args.p, verbose=False)
    r_cut, _, r_ratio, r_info = rqaoa.solve_grid_hybrid(
        args.Lx, args.Ly,
        gammas=info.get('gammas'), betas=info.get('betas'),
        gpu=args.gpu, chi=args.chi)
    print("  RQAOA:  ratio=%.6f  cut=%.1f" % (r_ratio, r_cut))
    print("  FQS:    ratio=%.6f  cut=%.1f" % (ratio, cut))
    print("  Delta:  %+.6f (%+.1f edges)" % (ratio - r_ratio, cut - r_cut))


if __name__ == '__main__':
    main()
