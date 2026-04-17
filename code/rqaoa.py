#!/usr/bin/env python3
"""
rqaoa.py - B47: Recursive QAOA voor MaxCut.

Bravyi, Kliesch, Koenig, Tang (2020): draai herhaaldelijk p=1 QAOA
op een krimpende graaf. Bevries de sterkst gecorreleerde edge per
iteratie, reduceer de graaf, herhaal tot brute-force.

Twee modi:
  'fast'  — Evalueer ⟨ZZ⟩ eenmaal (lightcone), reduceer greedy.
             O(1) QAOA-evaluatie. Geschikt voor grote grafen.
  'full'  — Her-evalueer QAOA na elke reductie (state vector).
             Nauwkeuriger, maar beperkt tot ≤22 qubits.

Hybride modus voor grids:
  1. Lightcone p=1 op het grid → alle ⟨ZZ⟩ (snel, exact)
  2. Greedy reductie tot ≤ 22 qubits
  3. Iteratieve RQAOA met state vector → brute force

Gebruik:
  python rqaoa.py --Lx 8 --Ly 4                   # 8x4 grid, fast RQAOA
  python rqaoa.py --Lx 20 --Ly 4                   # 20x4, 80 qubits
  python rqaoa.py --Lx 8 --Ly 4 --mode full        # iteratief (klein genoeg)
  python rqaoa.py --Lx 20 --Ly 4 --p 2 --gpu       # met p=2 + GPU
"""

import numpy as np
import time
import argparse
import copy
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Resultaatcontainer
# =====================================================================

@dataclass
class RQAOAResult:
    """Compact resultaatobject voor integratie in planners en benchmarks."""

    cut_value: float
    assignment: Dict[int, int]
    bitstring: np.ndarray
    ratio: float
    info: Dict[str, Any]


# =====================================================================
# Gewogen graaf representatie
# =====================================================================

class WeightedGraph:
    """Simpele gewogen graaf als adjacency dict."""

    def __init__(self):
        self.adj = {}  # {node: {neighbor: weight}}

    @property
    def nodes(self):
        return sorted(self.adj.keys())

    @property
    def n_nodes(self):
        return len(self.adj)

    @property
    def n_edges(self):
        return sum(len(nb) for nb in self.adj.values()) // 2

    def add_node(self, i):
        if i not in self.adj:
            self.adj[i] = {}

    def add_edge(self, i, j, w=1.0):
        self.add_node(i)
        self.add_node(j)
        self.adj[i][j] = self.adj[i].get(j, 0) + w
        self.adj[j][i] = self.adj[j].get(i, 0) + w

    def remove_node(self, i):
        for j in list(self.adj.get(i, {}).keys()):
            if i in self.adj[j]:
                del self.adj[j][i]
        if i in self.adj:
            del self.adj[i]

    def edges(self):
        """Geeft lijst van (i, j, w) met i < j."""
        seen = set()
        for i in self.adj:
            for j, w in self.adj[i].items():
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    yield key[0], key[1], w

    def total_weight(self):
        return sum(w for _, _, w in self.edges())

    def copy(self):
        g = WeightedGraph()
        for i in self.adj:
            g.adj[i] = dict(self.adj[i])
        return g

    @staticmethod
    def grid(Lx, Ly):
        """Maak Lx × Ly grid graaf."""
        g = WeightedGraph()
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                g.add_node(node)
                if x + 1 < Lx:
                    g.add_edge(node, (x + 1) * Ly + y)
                if y + 1 < Ly:
                    g.add_edge(node, x * Ly + y + 1)
        return g


# =====================================================================
# QAOA State-Vector Evaluator (willekeurige gewogen graaf)
# =====================================================================

class GeneralQAOA:
    """QAOA MaxCut op willekeurige gewogen graaf via state vector.

    Limiet: ~22 qubits op CPU (4M amplitudes).
    """

    MAX_QUBITS = 22

    def __init__(self, graph, verbose=False):
        self.graph = graph
        self.verbose = verbose

    def eval_all_zz(self, p, gammas, betas):
        """Evalueer ZZ voor ALLE edges tegelijk (gevectoriseerd).

        Returns: dict {(i,j): float} met i < j.
        """
        nodes = self.graph.nodes
        n = len(nodes)
        if n > self.MAX_QUBITS:
            raise ValueError("GeneralQAOA: %d qubits > max %d" %
                             (n, self.MAX_QUBITS))
        if n == 0:
            return {}

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        dim = 2 ** n
        bits = np.arange(dim, dtype=np.int64)

        # Precache alle Z-diagonalen
        z_diag = np.empty((n, dim), dtype=np.float64)
        for q in range(n):
            z_diag[q] = 1.0 - 2.0 * ((bits >> q) & 1)

        # Precache ZZ-product voor fase-separator (gamma-onafhankelijk)
        edge_list = [(node_to_idx[i], node_to_idx[j], w)
                     for i, j, w in self.graph.edges()]
        zz_diag = np.zeros(dim, dtype=np.float64)
        for ii, jj, w in edge_list:
            zz_diag += w * z_diag[ii] * z_diag[jj]

        # |+> state
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)

        # QAOA circuit
        for layer in range(p):
            # Fase-separator: exp(-ig * sum w_ij Z_i Z_j)
            state *= np.exp(-1j * gammas[layer] * zz_diag)

            # Mixer: prod_i Rx(2b)
            cb = np.cos(betas[layer])
            sb = np.sin(betas[layer])
            for q in range(n):
                s = state.reshape(2 ** (n - q - 1), 2, 2 ** q)
                s0 = s[:, 0, :].copy()
                s1 = s[:, 1, :].copy()
                s[:, 0, :] = cb * s0 - 1j * sb * s1
                s[:, 1, :] = -1j * sb * s0 + cb * s1
                state = s.reshape(-1)

        # Meet ZZ voor alle edges
        probs = np.abs(state) ** 2
        zz_values = {}
        for i, j, w in self.graph.edges():
            ii, jj = node_to_idx[i], node_to_idx[j]
            zz = float(np.dot(probs, z_diag[ii] * z_diag[jj]))
            zz_values[(i, j)] = zz

        return zz_values

    def eval_ratio(self, p, gammas, betas):
        """Volledige MaxCut-ratio voor de gegeven QAOA-hoeken."""
        total_weight = self.graph.total_weight()
        if total_weight <= 0:
            return 0.0

        zz = self.eval_all_zz(p, gammas, betas)
        cost = sum(w * (1 - zz[(min(i, j), max(i, j))]) / 2
                   for i, j, w in self.graph.edges())
        return cost / total_weight


    def optimize_p1(self, n_gamma=12, n_beta=12):
        """Optimaliseer gamma, beta voor p=1 via grid search + scipy."""
        best_ratio = -1
        best_g, best_b = 0.5, 0.3

        for g in np.linspace(0.1, np.pi - 0.1, n_gamma):
            for b in np.linspace(0.1, np.pi / 2 - 0.1, n_beta):
                zz = self.eval_all_zz(1, [g], [b])
                cost = sum(w * (1 - zz[(min(i,j), max(i,j))]) / 2
                           for i, j, w in self.graph.edges())
                ratio = cost / self.graph.total_weight() if self.graph.total_weight() > 0 else 0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_g, best_b = g, b

        # Scipy verfijning
        try:
            from scipy.optimize import minimize

            def neg_ratio(params):
                zz = self.eval_all_zz(1, [params[0]], [params[1]])
                cost = sum(w * (1 - zz[(min(i,j), max(i,j))]) / 2
                           for i, j, w in self.graph.edges())
                return -cost / self.graph.total_weight() if self.graph.total_weight() > 0 else 0

            result = minimize(neg_ratio, [best_g, best_b], method='L-BFGS-B',
                              bounds=[(0.01, np.pi), (0.01, np.pi / 2)],
                              options={'maxiter': 30})
            if -result.fun > best_ratio:
                best_ratio = -result.fun
                best_g, best_b = result.x
        except ImportError:
            pass

        return best_ratio, [best_g], [best_b]

    def _refine_angles(self, p, init_gammas, init_betas, best_ratio):
        """SciPy-verfijning voor willekeurige p."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            return best_ratio, list(init_gammas), list(init_betas), 0

        x0 = np.array(list(init_gammas) + list(init_betas), dtype=float)

        def neg_ratio(params):
            gs = np.clip(params[:p], 0.01, np.pi)
            bs = np.clip(params[p:], 0.01, np.pi / 2)
            return -self.eval_ratio(p, list(gs), list(bs))

        result = minimize(
            neg_ratio,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max(120, 80 * p), 'xatol': 1e-4,
                     'fatol': 1e-5, 'adaptive': True},
        )

        new_ratio = -float(result.fun)
        if new_ratio > best_ratio:
            gs = np.clip(result.x[:p], 0.01, np.pi)
            bs = np.clip(result.x[p:], 0.01, np.pi / 2)
            return new_ratio, list(gs), list(bs), int(result.nfev)

        return best_ratio, list(init_gammas), list(init_betas), int(result.nfev)

    def optimize(self, p, n_gamma=12, n_beta=12, refine=True):
        """Optimaliseer QAOA-hoeken voor algemene p.

        Voor p>1 gebruiken we de beste p=1-hoeken als seed en laten SciPy
        vervolgens de volledige 2p-dimensionale hoekvector verfijnen.
        """
        if p <= 1:
            ratio, gammas, betas = self.optimize_p1(n_gamma=n_gamma,
                                                    n_beta=n_beta)
            return ratio, gammas, betas, {
                'n_evals': n_gamma * n_beta,
                'seed_strategy': 'p1_grid',
            }

        base_ratio, base_g, base_b = self.optimize_p1(n_gamma=n_gamma,
                                                      n_beta=n_beta)
        best_gammas = [base_g[0]] * p
        best_betas = [base_b[0]] * p
        best_ratio = self.eval_ratio(p, best_gammas, best_betas)
        n_evals = n_gamma * n_beta + 1

        if refine:
            best_ratio, best_gammas, best_betas, n_refine = self._refine_angles(
                p, best_gammas, best_betas, best_ratio)
            n_evals += n_refine

        info = {
            'n_evals': n_evals,
            'seed_ratio_p1': base_ratio,
            'seed_strategy': 'repeat_best_p1',
        }
        return best_ratio, best_gammas, best_betas, info


# =====================================================================
# Brute-Force MaxCut
# =====================================================================

def brute_force_maxcut(graph):
    """Exact MaxCut via brute force. Retourneert (cut_value, assignment).

    assignment: dict {node: +1 of -1}
    """
    nodes = graph.nodes
    n = len(nodes)
    if n > 25:
        raise ValueError("Brute force te groot: %d nodes" % n)
    if n == 0:
        return 0.0, {}

    node_idx = {node: idx for idx, node in enumerate(nodes)}
    edge_list = [(i, j, w) for i, j, w in graph.edges()]

    best_cut = -1e9
    best_x = 0

    for x in range(2 ** n):
        cut = 0.0
        for i, j, w in edge_list:
            zi = 1 - 2 * ((x >> node_idx[i]) & 1)
            zj = 1 - 2 * ((x >> node_idx[j]) & 1)
            cut += w * (1 - zi * zj) / 2
        if cut > best_cut:
            best_cut = cut
            best_x = x

    assignment = {}
    for node in nodes:
        assignment[node] = 1 - 2 * ((best_x >> node_idx[node]) & 1)

    return best_cut, assignment


# =====================================================================
# RQAOA Engine
# =====================================================================

class RQAOA:
    """Recursive QAOA voor MaxCut.

    Bravyi et al. (2020): bevries sterkste correlatie per iteratie,
    reduceer de graaf, herhaal tot brute-force.
    """

    def __init__(self, graph, p=1, verbose=True):
        self.original_graph = graph.copy()
        self.p = p
        self.verbose = verbose

    @staticmethod
    def _detect_grid_dims(graph):
        """Detecteer simpele Lx x Ly-roosters met natuurlijke labeling."""
        nodes = graph.nodes
        n = len(nodes)
        if nodes != list(range(n)):
            return None

        n_edges = graph.n_edges
        degree_counts = {}
        for node in nodes:
            d = len(graph.adj.get(node, {}))
            degree_counts[d] = degree_counts.get(d, 0) + 1

        for Ly in range(1, int(np.sqrt(max(n, 1))) + 2):
            if Ly == 0 or n % Ly != 0:
                continue
            Lx = n // Ly
            if Lx < Ly:
                break

            expected_edges = Lx * (Ly - 1) + (Lx - 1) * Ly
            if expected_edges != n_edges:
                continue

            if Lx == 1 or Ly == 1:
                longer = max(Lx, Ly)
                if degree_counts.get(1, 0) == 2 and degree_counts.get(2, 0) == max(0, longer - 2):
                    return Lx, Ly
                continue

            corners = 4
            borders = 2 * max(0, Lx - 2) + 2 * max(0, Ly - 2)
            interior = max(0, (Lx - 2) * (Ly - 2))
            if (degree_counts.get(2, 0) == corners and
                    degree_counts.get(3, 0) == borders and
                    degree_counts.get(4, 0) == interior):
                return Lx, Ly

        return None

    @staticmethod
    def _assignment_to_bitstring(graph, assignment):
        """Converteer +/-1 assignment naar 0/1 bitstring in node-volgorde."""
        return np.array(
            [0 if assignment.get(node, 1) == 1 else 1 for node in graph.nodes],
            dtype=np.int8,
        )

    @staticmethod
    def _spectral_seed_assignment(graph):
        """Klassieke fallback-seed via kleinste eigenvector van de adjacency."""
        nodes = graph.nodes
        n = len(nodes)
        if n == 0:
            return {}
        if n == 1:
            return {nodes[0]: 1}

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=float)
        for i, j, w in graph.edges():
            ii = node_to_idx[i]
            jj = node_to_idx[j]
            adj[ii, jj] = w
            adj[jj, ii] = w

        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        vec = eigenvectors[:, 0]
        if np.all(vec >= 0) or np.all(vec <= 0):
            vec = np.array([1.0 if (idx % 2 == 0) else -1.0
                            for idx in range(n)], dtype=float)

        assignment = {}
        for node, idx in node_to_idx.items():
            assignment[node] = 1 if vec[idx] >= 0 else -1
        return assignment

    @staticmethod
    def _resolve_ordering_method(ordering_method, reorder):
        """Vertaal backlog-API `reorder` naar concrete B54-ordering."""
        if reorder is None:
            return ordering_method
        if reorder == 'none':
            return 'natural'
        if reorder == 'fiedler':
            return 'fiedler'
        if reorder == 'auto':
            return 'auto'
        raise ValueError("Onbekende reorder-modus: %s" % reorder)

    def _eval_all_zz_general_lightcone(self, p, gammas=None, betas=None,
                                       gpu=False, fp32=False,
                                       ordering_method='auto',
                                       n_gamma=10, n_beta=10):
        """Bereken alle <ZZ> voor een algemene graaf via B54-lightcones."""
        from general_lightcone import (
            GeneralLightconeQAOA,
            bfs_lightcone,
            lightcone_subgraph,
            lightcone_cache_key,
        )

        qaoa = GeneralLightconeQAOA(
            self.original_graph,
            verbose=self.verbose,
            gpu=gpu,
            fp32=fp32,
            ordering_method=ordering_method,
        )

        opt_ratio = None
        opt_info = {}
        if gammas is None or betas is None:
            opt_ratio, gammas, betas, opt_info = qaoa.optimize(
                p, n_gamma=n_gamma, n_beta=n_beta, refine=True)
        else:
            gammas = list(gammas)
            betas = list(betas)

        zz_values = {}
        cache = {}
        n_computed = 0
        n_cached = 0
        ordering_list = list(qaoa.ordering)
        t0 = time.time()

        for idx, (ei, ej, ew) in enumerate(qaoa.edge_list):
            lc_nodes = bfs_lightcone(qaoa.graph, (ei, ej), p)
            sub_edges = lightcone_subgraph(qaoa.graph, lc_nodes)
            key = lightcone_cache_key(lc_nodes, sub_edges, (ei, ej), ordering_list)

            if key in cache:
                zz = cache[key]
                n_cached += 1
            else:
                zz = qaoa.eval_edge_exact(
                    ei, ej, ew, lc_nodes, sub_edges, p, gammas, betas)
                cache[key] = zz
                n_computed += 1

            zz_values[(min(ei, ej), max(ei, ej))] = zz

            if self.verbose and (idx + 1) % 25 == 0:
                print("    B47/B54: %d/%d edges (%.1fs, %d uniek, %d cached)" % (
                    idx + 1, qaoa.n_edges, time.time() - t0,
                    n_computed, n_cached))

        info = {
            'qaoa_ratio': opt_ratio,
            'gammas': gammas,
            'betas': betas,
            'ordering': qaoa.ordering_name,
            'bandwidth': qaoa.bandwidth,
            'n_unique_lightcones': n_computed,
            'n_cached_lightcones': n_cached,
            'lightcone_time': time.time() - t0,
        }
        info.update(opt_info)
        return zz_values, gammas, betas, info

    # -----------------------------------------------------------------
    # Graaf-reductie
    # -----------------------------------------------------------------

    @staticmethod
    def reduce_one(graph, zz_values):
        """Eén RQAOA-reductiestap.

        1. Vind edge met max |⟨ZZ⟩|
        2. Fix correlatie (z_j = c · z_i)
        3. Merge j in i, update gewichten

        Returns: (i_kept, j_removed, c, offset_delta)
        """
        # Vind sterkste correlatie
        best_edge = None
        best_abs = -1.0
        for (i, j), zz in zz_values.items():
            if i in graph.adj and j in graph.adj:
                if abs(zz) > best_abs:
                    best_abs = abs(zz)
                    best_edge = (i, j)
                    best_c = 1 if zz > 0 else -1

        if best_edge is None:
            return None, None, 0, 0.0

        i, j = best_edge
        c = best_c

        # Offset van edge (i,j) zelf
        w_ij = graph.adj[i].get(j, 0)
        offset = w_ij * (1 - c) / 2

        # Merge j in i: update buren van j
        for k, w_jk in list(graph.adj[j].items()):
            if k == i:
                continue

            offset += w_jk * (1 - c) / 2

            # Nieuw gewicht voor edge (i, k)
            w_ik = graph.adj[i].get(k, 0)
            new_w = w_ik + c * w_jk

            # Verwijder oude edge (j, k)
            if j in graph.adj[k]:
                del graph.adj[k][j]

            # Zet nieuwe edge (i, k)
            if abs(new_w) > 1e-12:
                graph.adj[i][k] = new_w
                graph.adj[k][i] = new_w
            else:
                if k in graph.adj[i]:
                    del graph.adj[i][k]
                if i in graph.adj[k]:
                    del graph.adj[k][i]

        # Verwijder edge (i, j) en node j
        if j in graph.adj[i]:
            del graph.adj[i][j]
        del graph.adj[j]

        return i, j, c, offset

    # -----------------------------------------------------------------
    # Reconstructie: van gereduceerde assignment naar volledige assignment
    # -----------------------------------------------------------------

    @staticmethod
    def reconstruct(reductions, reduced_assignment):
        """Reconstrueer volledige assignment vanuit gereduceerde.

        reductions: lijst van (i_kept, j_removed, c) tuples
        reduced_assignment: dict {node: +1/-1} voor de gereduceerde graaf
        """
        assignment = dict(reduced_assignment)

        for i, j, c in reversed(reductions):
            if i in assignment:
                assignment[j] = c * assignment[i]
            else:
                assignment[j] = c  # default: +1 als i niet bekend

        return assignment

    # -----------------------------------------------------------------
    # Evalueer een assignment op de originele graaf
    # -----------------------------------------------------------------

    @staticmethod
    def eval_cut(graph, assignment):
        """Bereken cut-waarde van een assignment op een graaf."""
        cut = 0.0
        for i, j, w in graph.edges():
            zi = assignment.get(i, 1)
            zj = assignment.get(j, 1)
            cut += w * (1 - zi * zj) / 2
        return cut

    # -----------------------------------------------------------------
    # Fast RQAOA: single-shot evaluatie + greedy reductie
    # -----------------------------------------------------------------

    def solve_fast(self, zz_values, brute_threshold=16, local_search=True):
        """Fast RQAOA via greedy spin-assignment.

        Sorteert edges op |⟨ZZ⟩| (sterkste correlatie eerst).
        Wijst spins toe via propagatie: z_j = c · z_i.
        O(E log E) — geen QAOA her-evaluatie, geen brute force nodig.

        Args:
            zz_values: dict {(i,j): ⟨ZZ⟩} van de originele graaf
            brute_threshold: (ongebruikt in deze modus, voor API-compatibiliteit)

        Returns: (cut_value, assignment, info)
        """
        t0 = time.time()

        # Sorteer edges op |ZZ| dalend (sterkste correlatie eerst)
        sorted_edges = sorted(zz_values.items(),
                              key=lambda kv: abs(kv[1]), reverse=True)

        # Greedy spin-assignment
        assignment = {}
        n_propagated = 0
        n_conflicts = 0
        n_seeds = 0

        for (i, j), zz in sorted_edges:
            c = 1 if zz > 0 else -1  # z_j = c · z_i

            i_has = i in assignment
            j_has = j in assignment

            if not i_has and not j_has:
                # Nieuwe component: kies z_i = +1, z_j = c
                assignment[i] = 1
                assignment[j] = c
                n_seeds += 1
                n_propagated += 1
            elif i_has and not j_has:
                assignment[j] = c * assignment[i]
                n_propagated += 1
            elif j_has and not i_has:
                assignment[i] = c * assignment[j]
                n_propagated += 1
            else:
                # Beide al toegewezen: check consistentie
                expected_j = c * assignment[i]
                if assignment[j] != expected_j:
                    n_conflicts += 1

        # Resterende nodes (geen edges): z = +1
        for node in self.original_graph.adj:
            if node not in assignment:
                assignment[node] = 1

        cut_greedy = self.eval_cut(self.original_graph, assignment)
        n_edges = self.original_graph.n_edges
        ratio_greedy = cut_greedy / n_edges if n_edges > 0 else 0

        if self.verbose:
            print("  RQAOA greedy: %d edges, %d propagaties, "
                  "%d seeds, %d conflicten" %
                  (len(sorted_edges), n_propagated, n_seeds, n_conflicts))
            print("  Greedy cut: %.0f / %d  ratio=%.6f" %
                  (cut_greedy, n_edges, ratio_greedy))

        # Local search: flip-if-better
        n_flips = 0
        if local_search:
            cut_ls, assignment, n_flips = self._local_search(
                self.original_graph, assignment)
            ratio_ls = cut_ls / n_edges if n_edges > 0 else 0
            if self.verbose:
                print("  Local search: %d flips, cut %.0f -> %.0f (%+.0f), "
                      "ratio %.6f -> %.6f" %
                      (n_flips, cut_greedy, cut_ls, cut_ls - cut_greedy,
                       ratio_greedy, ratio_ls))
            cut = cut_ls
        else:
            cut = cut_greedy

        ratio = cut / n_edges if n_edges > 0 else 0

        info = {
            'n_propagated': n_propagated,
            'n_conflicts': n_conflicts,
            'n_seeds': n_seeds,
            'n_flips': n_flips,
            'cut_greedy': cut_greedy,
            'ratio_greedy': ratio_greedy,
            'time': time.time() - t0,
        }
        return cut, assignment, info

    # -----------------------------------------------------------------
    # Local search post-processing
    # -----------------------------------------------------------------

    def _local_search(self, graph, assignment, max_rounds=100,
                      n_restarts=None):
        """Multi-strategy local search: steepest descent + random restarts.

        Fase 1: Steepest descent (flip node met hoogste delta per ronde).
        Fase 2: Random perturbatie + steepest descent (escape local optima).
                 Varieert perturbatie-grootte per restart.

        Returns: (improved_cut, improved_assignment, n_flips)
        """
        assignment = dict(assignment)
        nodes = list(graph.adj.keys())
        n_nodes = len(nodes)
        n_flips_total = 0

        # Restarts schalen met graafgrootte
        if n_restarts is None:
            n_restarts = min(50, max(10, n_nodes // 4))

        def _compute_delta(ass, node):
            """Bereken cut-verbetering als we node flippen."""
            delta = 0.0
            z_i = ass.get(node, 1)
            for nb, w in graph.adj[node].items():
                delta += w * z_i * ass.get(nb, 1)
            return delta

        def _steepest_descent(ass):
            """Flip steeds de node met hoogste delta tot geen verbetering."""
            flips = 0
            for _ in range(max_rounds):
                best_node = None
                best_delta = 1e-12
                for node in nodes:
                    d = _compute_delta(ass, node)
                    if d > best_delta:
                        best_delta = d
                        best_node = node
                if best_node is None:
                    break
                ass[best_node] = -ass[best_node]
                flips += 1
            return flips

        # Fase 1: steepest descent op huidige assignment
        n_flips_total += _steepest_descent(assignment)
        best_cut = self.eval_cut(graph, assignment)
        best_assignment = dict(assignment)

        # Fase 2: random restarts met variabele perturbatie
        rng = np.random.RandomState(42)
        for restart in range(n_restarts):
            trial = dict(best_assignment)

            # Varieer perturbatie: 2% tot 25% van nodes
            frac = 0.02 + 0.23 * (restart / max(1, n_restarts - 1))
            k = max(2, int(n_nodes * frac))
            flip_nodes = rng.choice(nodes, size=k, replace=False)
            for node in flip_nodes:
                trial[node] = -trial.get(node, 1)

            flips = _steepest_descent(trial)
            n_flips_total += flips + k

            trial_cut = self.eval_cut(graph, trial)
            if trial_cut > best_cut:
                best_cut = trial_cut
                best_assignment = trial
                if self.verbose:
                    print("    LS restart %d (k=%d): cut %.0f (verbeterd!)" %
                          (restart, k, trial_cut))

        return best_cut, best_assignment, n_flips_total

    # -----------------------------------------------------------------
    # Full RQAOA: her-evalueer QAOA na elke reductie (klein)
    # -----------------------------------------------------------------

    def solve_full(self, gammas=None, betas=None,
                   brute_threshold=8, reeval_threshold=22):
        """Full RQAOA: her-evalueer QAOA bij elke reductie.

        Fase 1: als N > reeval_threshold, gebruik fast-modus (single-shot)
        Fase 2: als N ≤ reeval_threshold, her-evalueer met state vector

        Args:
            gammas, betas: QAOA parameters (default: auto-optimaliseer)
            brute_threshold: brute force wanneer N ≤ dit
            reeval_threshold: max N voor state vector her-evaluatie

        Returns: (cut_value, assignment, info)
        """
        t0 = time.time()
        graph = self.original_graph.copy()
        reductions = []
        total_offset = 0.0
        n_start = graph.n_nodes
        p = self.p
        q_ratio = None
        q_info = {}
        opt_g = list(gammas) if gammas is not None else None
        opt_b = list(betas) if betas is not None else None
        initial_q_ratio = None
        initial_opt_g = None
        initial_opt_b = None
        initial_q_info = {}

        if self.verbose:
            print("  RQAOA full: %d nodes, %d edges, p=%d" %
                  (graph.n_nodes, graph.n_edges, p))

        if graph.n_nodes > reeval_threshold:
            raise ValueError(
                "solve_full ondersteunt maximaal %d nodes; gebruik solve(mode='auto') "
                "voor grotere grafen" % reeval_threshold)

        step = 0

        # Fase 2: iteratieve RQAOA met state vector
        while graph.n_nodes > brute_threshold and graph.n_nodes <= reeval_threshold:
            # Evalueer QAOA op huidige graaf
            if gammas is None or betas is None:
                # Auto-optimaliseer
                qaoa = GeneralQAOA(graph, verbose=False)
                q_ratio, opt_g, opt_b, q_info = qaoa.optimize(
                    p, n_gamma=10, n_beta=10, refine=True)
            else:
                qaoa = GeneralQAOA(graph, verbose=False)
                opt_g, opt_b = gammas, betas
                q_ratio = qaoa.eval_ratio(p, opt_g, opt_b)
                q_info = {}

            if initial_q_ratio is None:
                initial_q_ratio = q_ratio
                initial_opt_g = list(opt_g)
                initial_opt_b = list(opt_b)
                initial_q_info = dict(q_info)

            qaoa = GeneralQAOA(graph, verbose=False)
            zz = qaoa.eval_all_zz(p, opt_g, opt_b)

            # Reduceer
            i, j, c, offset = self.reduce_one(graph, zz)
            if i is None:
                break
            reductions.append((i, j, c))
            total_offset += offset
            step += 1

            if self.verbose and step % 5 == 0:
                print("    Full stap %d: %d nodes, offset=%.1f" %
                      (step, graph.n_nodes, total_offset))

        # Brute force op restgraaf
        if graph.n_nodes > 0 and graph.n_edges > 0:
            bf_cut, bf_assign = brute_force_maxcut(graph)
        else:
            bf_cut = 0.0
            bf_assign = {n: 1 for n in graph.nodes}

        # Reconstructie
        full_assignment = self.reconstruct(reductions, bf_assign)
        actual_cut = self.eval_cut(self.original_graph, full_assignment)

        if self.verbose:
            print("    Gereduceerd: %d -> %d nodes in %d stappen" %
                  (n_start, graph.n_nodes, step))
            print("    Verificatie op origineel: cut=%.1f" % actual_cut)

        info = {
            'n_reductions': step,
            'n_remaining': graph.n_nodes,
            'offset': total_offset,
            'bf_cut': bf_cut,
            'qaoa_ratio': initial_q_ratio if initial_q_ratio is not None else q_ratio,
            'gammas': initial_opt_g if initial_opt_g is not None else opt_g,
            'betas': initial_opt_b if initial_opt_b is not None else opt_b,
            'n_evals': initial_q_info.get('n_evals') if initial_q_info else q_info.get('n_evals'),
            'time': time.time() - t0,
        }
        return actual_cut, full_assignment, info

    # -----------------------------------------------------------------
    # Hybride solver voor grids: lightcone → fast → full → brute force
    # -----------------------------------------------------------------

    def solve_grid_hybrid(self, Lx, Ly, gammas=None, betas=None,
                          brute_threshold=12, gpu=False, chi=None):
        """Hybride RQAOA voor grid-grafen.

        1. Lightcone QAOA op het grid → alle ⟨ZZ⟩ (snel)
        2. Greedy reductie tot ≤ 22 qubits
        3. Iteratieve RQAOA met state vector
        4. Brute force op restgraaf

        Args:
            Lx, Ly: grid dimensies
            gammas, betas: QAOA params (default: auto-optimaliseer via lightcone)
            brute_threshold: brute force drempel
            gpu: gebruik GPU voor lightcone
            chi: MPS chi voor hybride lightcone

        Returns: (cut_value, assignment, ratio, info)
        """
        from lightcone_qaoa import LightconeQAOA

        t0 = time.time()
        p = self.p
        graph = self.original_graph.copy()
        n_start = graph.n_nodes
        n_edges = graph.n_edges

        if self.verbose:
            print("\n=== B47 RQAOA Hybride: %dx%d grid (%d qubits, %d edges) ===" %
                  (Lx, Ly, n_start, n_edges))
            print("  p=%d, brute_threshold=%d" % (p, brute_threshold))

        # --- Stap 1: Lightcone QAOA → optimaliseer + evalueer alle ⟨ZZ⟩ ---
        lc = LightconeQAOA(Lx, Ly, verbose=self.verbose, chi=chi, gpu=gpu)
        opt_info = {}

        if gammas is None or betas is None:
            if self.verbose:
                print("\n  Fase 0: Optimaliseer gamma/beta via lightcone...")
            _, opt_gammas, opt_betas, opt_info = lc.optimize(
                p=p, n_gamma=12, n_beta=12, refine=True)
            gammas, betas = opt_gammas, opt_betas
            if self.verbose:
                print("    Optimaal: gamma=%s beta=%s ratio=%.6f [%.1fs]" %
                      (gammas, betas, opt_info.get('grid_ratio', 0),
                       opt_info.get('total_time', 0)))
        else:
            if self.verbose:
                print("  Fase 0: Gebruik opgegeven gamma=%s beta=%s" %
                      (gammas, betas))

        if self.verbose:
            print("\n  Fase 1: Evalueer alle <ZZ> via lightcone...")
        t1 = time.time()

        # Bouw edge mapping: lightcone edge → graph node pair
        zz_values = {}
        for etype, ex, ey in lc.edges:
            zz = lc.eval_edge(etype, ex, ey, p, gammas, betas)
            if etype == 'h':
                i = ex * Ly + ey
                j = (ex + 1) * Ly + ey
            else:
                i = ex * Ly + ey
                j = ex * Ly + ey + 1
            zz_values[(min(i, j), max(i, j))] = zz

        if self.verbose:
            eval_time = time.time() - t1
            print("    %d edges geëvalueerd in %.2fs" %
                  (len(zz_values), eval_time))

            # Statistieken
            zz_arr = np.array(list(zz_values.values()))
            print("    <ZZ> range: [%.4f, %.4f], mean=%.4f" %
                  (zz_arr.min(), zz_arr.max(), zz_arr.mean()))
            n_cut = np.sum(zz_arr < 0)
            print("    %d/%d edges met <ZZ><0 (anti-gecorreleerd = gesneden)" %
                  (n_cut, len(zz_arr)))

        # --- Stap 2: Greedy spin-assignment via solve_fast ---
        if self.verbose:
            print("\n  Fase 2: Greedy spin-assignment...")

        cut, full_assignment, fast_info = self.solve_fast(zz_values)
        ratio = cut / n_edges if n_edges > 0 else 0

        total_time = time.time() - t0

        if self.verbose:
            print("\n" + "=" * 60)
            print("  RQAOA Resultaat:")
            print("    Cut-waarde: %.1f / %d edges" % (cut, n_edges))
            print("    Ratio: %.6f" % ratio)
            print("    Propagaties: %d, seeds: %d, conflicten: %d" %
                  (fast_info['n_propagated'], fast_info['n_seeds'],
                   fast_info['n_conflicts']))
            print("    Totale tijd: %.1fs" % total_time)
            print("=" * 60)

        info = {
            'actual_cut': cut,
            'ratio': ratio,
            'qaoa_ratio': opt_info.get('grid_ratio'),
            'n_propagated': fast_info['n_propagated'],
            'n_conflicts': fast_info['n_conflicts'],
            'n_seeds': fast_info['n_seeds'],
            'time': total_time,
            'gammas': gammas,
            'betas': betas,
        }
        return cut, full_assignment, ratio, info

    def solve(self, mode='auto', gammas=None, betas=None,
              brute_threshold=8, reeval_threshold=22, local_search=True,
              gpu=False, fp32=False, ordering_method='auto',
              reorder=None,
              n_gamma=10, n_beta=10):
        """End-to-end RQAOA entrypoint voor planners en benchmarks."""
        total_weight = self.original_graph.total_weight()
        n_nodes = self.original_graph.n_nodes
        n_edges = self.original_graph.n_edges
        density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))
                   if n_nodes > 1 else 0.0)
        max_degree = max((len(nb) for nb in self.original_graph.adj.values()),
                         default=0)
        est_lightcone_qubits = min(n_nodes, 2 + 2 * self.p * max_degree)
        if total_weight <= 0:
            empty_assignment = {node: 1 for node in self.original_graph.nodes}
            return RQAOAResult(
                cut_value=0.0,
                assignment=empty_assignment,
                bitstring=self._assignment_to_bitstring(self.original_graph,
                                                        empty_assignment),
                ratio=0.0,
                info={'mode': 'empty'},
            )

        resolved_ordering = self._resolve_ordering_method(
            ordering_method, reorder)
        grid_dims = self._detect_grid_dims(self.original_graph)
        chosen_mode = mode
        if chosen_mode == 'auto':
            if grid_dims is not None:
                chosen_mode = 'grid'
            elif self.original_graph.n_nodes <= reeval_threshold:
                chosen_mode = 'full'
            else:
                chosen_mode = 'fast'

        if chosen_mode == 'grid':
            if grid_dims is None:
                raise ValueError("grid-modus gevraagd, maar graaf lijkt geen rooster")
            Lx, Ly = grid_dims
            cut, assignment, ratio, info = self.solve_grid_hybrid(
                Lx, Ly, gammas=gammas, betas=betas,
                brute_threshold=brute_threshold, gpu=gpu)
            info['mode'] = 'grid_hybrid'
        elif chosen_mode == 'full':
            cut, assignment, info = self.solve_full(
                gammas=gammas, betas=betas,
                brute_threshold=brute_threshold,
                reeval_threshold=reeval_threshold)
            ratio = cut / total_weight
            info['mode'] = 'full_statevector'
        elif chosen_mode == 'fast':
            try:
                if density > 0.15 or est_lightcone_qubits > 22:
                    raise RuntimeError(
                        "lightcone guard actief (density=%.3f, est_lc=%d)" % (
                            density, est_lightcone_qubits))
                zz_values, gammas, betas, zz_info = self._eval_all_zz_general_lightcone(
                    self.p, gammas=gammas, betas=betas, gpu=gpu, fp32=fp32,
                    ordering_method=resolved_ordering, n_gamma=n_gamma,
                    n_beta=n_beta)
                cut, assignment, fast_info = self.solve_fast(
                    zz_values, brute_threshold=brute_threshold,
                    local_search=local_search)
                ratio = cut / total_weight
                info = dict(zz_info)
                info.update(fast_info)
                info['mode'] = 'fast_lightcone'
            except Exception as exc:
                assignment = self._spectral_seed_assignment(self.original_graph)
                seed_cut = self.eval_cut(self.original_graph, assignment)
                n_flips = 0
                if local_search:
                    cut, assignment, n_flips = self._local_search(
                        self.original_graph, assignment)
                else:
                    cut = seed_cut
                ratio = cut / total_weight
                info = {
                    'mode': 'spectral_fallback',
                    'fallback_reason': str(exc),
                    'seed_cut': seed_cut,
                    'n_flips': n_flips,
                    'density': density,
                    'est_lightcone_qubits': est_lightcone_qubits,
                    'qaoa_ratio': None,
                }
        else:
            raise ValueError("Onbekende solve-modus: %s" % chosen_mode)

        info['requested_reorder'] = reorder if reorder is not None else 'inherit'
        info['resolved_ordering'] = info.get('ordering', resolved_ordering)

        bitstring = self._assignment_to_bitstring(self.original_graph, assignment)
        return RQAOAResult(
            cut_value=cut,
            assignment=assignment,
            bitstring=bitstring,
            ratio=ratio,
            info=info,
        )


# =====================================================================
# QAOA ratio vergelijking (baseline)
# =====================================================================

def qaoa_ratio_baseline(Lx, Ly, p, gammas, betas, gpu=False, chi=None):
    """Bereken standaard QAOA-ratio (zonder RQAOA) als baseline."""
    from lightcone_qaoa import LightconeQAOA
    lc = LightconeQAOA(Lx, Ly, verbose=False, chi=chi, gpu=gpu)
    cost = lc.eval_cost(p, gammas, betas)
    return cost / lc.n_edges


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B47: RQAOA — Recursive QAOA voor MaxCut')
    parser.add_argument('--Lx', type=int, default=8,
                        help='Grid breedte (default: 8)')
    parser.add_argument('--Ly', type=int, default=4,
                        help='Grid hoogte (default: 4)')
    parser.add_argument('--p', type=int, default=1,
                        help='QAOA diepte (default: 1)')
    parser.add_argument('--brute-threshold', type=int, default=12,
                        help='Brute force drempel (default: 12)')
    parser.add_argument('--gpu', action='store_true',
                        help='Gebruik GPU voor lightcone')
    parser.add_argument('--chi', type=int, default=None,
                        help='MPS chi voor hybride lightcone')
    parser.add_argument('--gamma', type=float, nargs='+', default=None,
                        help='Vaste gamma waarden')
    parser.add_argument('--beta', type=float, nargs='+', default=None,
                        help='Vaste beta waarden')
    args = parser.parse_args()

    Lx, Ly = args.Lx, args.Ly
    n_qubits = Lx * Ly
    graph = WeightedGraph.grid(Lx, Ly)

    print("=" * 60)
    print("B47: RQAOA op %dx%d grid (%d qubits, %d edges)" %
          (Lx, Ly, n_qubits, graph.n_edges))
    print("=" * 60)

    # RQAOA
    rqaoa = RQAOA(graph, p=args.p, verbose=True)
    cut, assignment, ratio, info = rqaoa.solve_grid_hybrid(
        Lx, Ly,
        gammas=args.gamma, betas=args.beta,
        brute_threshold=args.brute_threshold,
        gpu=args.gpu, chi=args.chi)

    # Vergelijk met standaard QAOA
    print("\n--- Vergelijking ---")
    g, b = info['gammas'], info['betas']
    qaoa_r = qaoa_ratio_baseline(Lx, Ly, args.p, g, b,
                                 gpu=args.gpu, chi=args.chi)
    print("  Standaard QAOA p=%d ratio: %.6f" % (args.p, qaoa_r))
    print("  RQAOA cut-ratio:          %.6f" % ratio)
    improvement = ratio - qaoa_r
    print("  Verbetering:              %+.6f (%+.1f%%)" %
          (improvement, 100 * improvement / qaoa_r if qaoa_r > 0 else 0))

    # MaxCut referentie
    n_edges = graph.n_edges
    max_cut_bipartite = n_edges  # grid is bipartiet → max cut = alle edges
    print("\n  Max cut (bipartiet): %d (ratio=1.000)" % max_cut_bipartite)
    print("  RQAOA cut:           %.0f (ratio=%.4f)" % (cut, ratio))


if __name__ == '__main__':
    main()
