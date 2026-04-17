#!/usr/bin/env python3
"""
general_lightcone.py - B54: Lightcone QAOA op willekeurige grafen.

Generaliseert lightcone_qaoa.py (B21) van grids naar willekeurige grafen.
Elke edge krijgt een BFS-lichtkegel: alle nodes binnen afstand p.
Slimme qubit-ordering (Cuthill-McKee / Fiedler) minimaliseert lightcone-grootte.

Kernidee:
  1. Herorden nodes zodat buren dicht bij elkaar zitten (kleinere bandbreedte)
  2. Per edge: BFS tot diepte p → lightcone subgraaf
  3. State vector simulatie op subgraaf (exact, geen truncatie)
  4. Isomorfisme-caching: structureel identieke lightcones hergebruiken

Gebruik:
  python general_lightcone.py --graph petersen --p 1
  python general_lightcone.py --graph gset:G14 --p 1
  python general_lightcone.py --graph triangular:6x3 --p 1 --gpu
  python general_lightcone.py --graph random_geometric:50:0.3 --p 2
  python general_lightcone.py --graph grid:20x3 --p 3  # vergelijk met B21

Bouwt voort op: lightcone_qaoa.py (B21), rqaoa.py (WeightedGraph),
                gset_loader.py (B61), ma_qaoa.py (B67)
"""

import numpy as np
import math
import time
import argparse
import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

SV_THRESHOLD_GPU = 26
SV_THRESHOLD_CPU = 22


# =====================================================================
# Qubit-ordering algoritmen
# =====================================================================

def ordering_cuthill_mckee(graph, reverse=True):
    """Cuthill-McKee (of Reverse CM) qubit-ordering.

    Minimaliseert bandbreedte van de adjacency matrix.
    Reverse CM geeft typisch iets betere resultaten.

    Args:
        graph: WeightedGraph
        reverse: gebruik Reverse Cuthill-McKee (default: True)

    Returns:
        list van node-ids in nieuwe volgorde
    """
    nodes = graph.nodes
    n = len(nodes)
    if n == 0:
        return []

    adj = graph.adj
    visited = set()
    order = []

    # Start bij node met laagste graad (CM heuristiek)
    start = min(nodes, key=lambda v: len(adj.get(v, {})))

    queue = deque([start])
    visited.add(start)

    while len(order) < n:
        if not queue:
            # Disconnected component: pak eerste onbezochte node
            remaining = [v for v in nodes if v not in visited]
            if not remaining:
                break
            start = min(remaining, key=lambda v: len(adj.get(v, {})))
            queue.append(start)
            visited.add(start)

        node = queue.popleft()
        order.append(node)

        # Buren sorteren op graad (CM: laagste graad eerst)
        neighbors = sorted(
            [nb for nb in adj.get(node, {}) if nb not in visited],
            key=lambda v: len(adj.get(v, {}))
        )
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    if reverse:
        order = list(reversed(order))

    return order


def ordering_fiedler(graph):
    """Spectrale ordering via Fiedler-vector (2e eigenwaarde Laplaciaan).

    De Fiedler-vector geeft een natuurlijke 1D-embedding van de graaf.
    Nodes worden gesorteerd op hun Fiedler-vector component.

    Args:
        graph: WeightedGraph

    Returns:
        list van node-ids in spectrale volgorde
    """
    nodes = graph.nodes
    n = len(nodes)
    if n <= 2:
        return nodes

    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Bouw Laplaciaan
    L = np.zeros((n, n))
    for i, j, w in graph.edges():
        ii, jj = node_to_idx[i], node_to_idx[j]
        L[ii, jj] = -w
        L[jj, ii] = -w
        L[ii, ii] += w
        L[jj, jj] += w

    # Fiedler vector = eigenvector bij 2e kleinste eigenwaarde
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]  # 2e kolom (0-indexed)

    # Sorteer nodes op Fiedler-vector component
    sorted_indices = np.argsort(fiedler)
    return [nodes[i] for i in sorted_indices]


def ordering_natural(graph):
    """Natuurlijke (numerieke) volgorde."""
    return graph.nodes


def compute_bandwidth(graph, ordering):
    """Bereken bandbreedte van adjacency matrix gegeven een ordering.

    Bandbreedte = max |pos(i) - pos(j)| over alle edges (i,j).
    Kleinere bandbreedte → kleinere lightcones.
    """
    pos = {node: idx for idx, node in enumerate(ordering)}
    max_bw = 0
    for i, j, w in graph.edges():
        bw = abs(pos[i] - pos[j])
        if bw > max_bw:
            max_bw = bw
    return max_bw


def best_ordering(graph):
    """Probeer alle orderings en kies de beste (kleinste bandbreedte).

    Returns:
        (ordering, method_name, bandwidth)
    """
    candidates = [
        ('natural', ordering_natural(graph)),
        ('rcm', ordering_cuthill_mckee(graph, reverse=True)),
        ('fiedler', ordering_fiedler(graph)),
    ]

    best = None
    for name, order in candidates:
        bw = compute_bandwidth(graph, order)
        if best is None or bw < best[2]:
            best = (order, name, bw)

    return best


# =====================================================================
# BFS Lightcone
# =====================================================================

def bfs_lightcone(graph, edge_nodes, p):
    """BFS-lichtkegel: alle nodes binnen afstand p van de edge-nodes.

    Args:
        graph: WeightedGraph
        edge_nodes: tuple (i, j) - de target edge
        p: circuit diepte (BFS radius)

    Returns:
        set van nodes in de lightcone
    """
    adj = graph.adj
    visited = set(edge_nodes)
    frontier = set(edge_nodes)

    for _ in range(p):
        next_frontier = set()
        for node in frontier:
            for nb in adj.get(node, {}):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier

    return visited


def lightcone_subgraph(graph, lc_nodes):
    """Extraheer de subgraaf geïnduceerd door lightcone-nodes.

    Args:
        graph: WeightedGraph
        lc_nodes: set van nodes in de lightcone

    Returns:
        list van (i, j, w) edges in de subgraaf (met originele node-ids)
    """
    edges = []
    seen = set()
    for node in lc_nodes:
        for nb, w in graph.adj.get(node, {}).items():
            if nb in lc_nodes:
                key = (min(node, nb), max(node, nb))
                if key not in seen:
                    seen.add(key)
                    edges.append((key[0], key[1], w))
    return edges


# =====================================================================
# Lightcone cache key (isomorfisme-detectie)
# =====================================================================

def lightcone_cache_key(lc_nodes, sub_edges, target_edge, ordering):
    """Genereer cache key voor structurele equivalentie van lightcones.

    Twee lightcones zijn equivalent als ze dezelfde lokale structuur
    hebben (zelfde graad-sequentie, zelfde edge-patroon relatief tot
    de target edge).

    Gebruikt een kanonieke relabeling: target edge nodes worden 0,1,
    rest in BFS-volgorde.

    Args:
        lc_nodes: set van nodes
        sub_edges: list van (i,j,w) in subgraaf
        target_edge: (i, j) target edge
        ordering: globale node-ordering (voor consistente labeling)

    Returns:
        hashable tuple als cache key
    """
    # Kanonieke relabeling: target edge = (0, 1), rest in volgorde
    # van hun positie in de globale ordering
    i_t, j_t = target_edge
    remaining = sorted(
        [n for n in lc_nodes if n != i_t and n != j_t],
        key=lambda n: ordering.index(n) if n in ordering else n
    )
    relabel = {i_t: 0, j_t: 1}
    for idx, node in enumerate(remaining):
        relabel[node] = idx + 2

    # Relabel edges en sorteer
    relabeled_edges = tuple(sorted(
        (relabel[i], relabel[j], w)
        for i, j, w in sub_edges
    ))

    n_nodes = len(lc_nodes)
    return (n_nodes, relabeled_edges)


# =====================================================================
# GeneralLightconeQAOA
# =====================================================================

class GeneralLightconeQAOA:
    """QAOA MaxCut via lightcone decomposition op willekeurige grafen.

    Generaliseert LightconeQAOA (B21) van grids naar elke graaf.
    """

    def __init__(self, graph, verbose=True, gpu=False, fp32=False,
                 ordering_method='auto'):
        """
        Args:
            graph: WeightedGraph instantie
            verbose: print voortgang
            gpu: gebruik CuPy/GPU
            fp32: single precision
            ordering_method: 'auto', 'natural', 'rcm', 'fiedler'
        """
        self.graph = graph
        self.verbose = verbose
        self.gpu = gpu
        self.fp32 = fp32

        # Qubit ordering
        if ordering_method == 'auto':
            self.ordering, self.ordering_name, self.bandwidth = best_ordering(graph)
        elif ordering_method == 'rcm':
            self.ordering = ordering_cuthill_mckee(graph, reverse=True)
            self.ordering_name = 'rcm'
            self.bandwidth = compute_bandwidth(graph, self.ordering)
        elif ordering_method == 'fiedler':
            self.ordering = ordering_fiedler(graph)
            self.ordering_name = 'fiedler'
            self.bandwidth = compute_bandwidth(graph, self.ordering)
        else:
            self.ordering = ordering_natural(graph)
            self.ordering_name = 'natural'
            self.bandwidth = compute_bandwidth(graph, self.ordering)

        # Node-positie lookup (voor snelle index)
        self.node_pos = {node: idx for idx, node in enumerate(self.ordering)}

        # Edge lijst
        self.edge_list = list(graph.edges())  # [(i, j, w), ...]
        self.n_edges = len(self.edge_list)
        self.total_weight = sum(abs(w) for _, _, w in self.edge_list)

        # Pre-allocated buffers (lazy init)
        self._buf_max_dim = 0
        self._buf_state = None
        self._buf_hphase = None

        if verbose:
            print("  [B54] Graaf: %d nodes, %d edges" % (graph.n_nodes, self.n_edges))
            print("  [B54] Ordering: %s (bandbreedte=%d)" % (
                self.ordering_name, self.bandwidth))

    def lightcone_stats(self, p):
        """Bereken lightcone-statistieken voor alle edges.

        Returns:
            dict met min/max/avg lightcone grootte, cache hit rate
        """
        sizes = []
        cache = set()
        n_unique = 0

        ordering_list = list(self.ordering)

        for i, j, w in self.edge_list:
            lc_nodes = bfs_lightcone(self.graph, (i, j), p)
            sizes.append(len(lc_nodes))

            sub_edges = lightcone_subgraph(self.graph, lc_nodes)
            key = lightcone_cache_key(lc_nodes, sub_edges, (i, j), ordering_list)
            if key not in cache:
                cache.add(key)
                n_unique += 1

        return {
            'min_qubits': min(sizes),
            'max_qubits': max(sizes),
            'avg_qubits': sum(sizes) / len(sizes),
            'n_unique': n_unique,
            'n_cached': self.n_edges - n_unique,
            'cache_rate': (self.n_edges - n_unique) / max(self.n_edges, 1),
        }

    def eval_edge_exact(self, target_i, target_j, target_w, lc_nodes,
                        sub_edges, p, gammas, betas):
        """Bereken <ZZ> (of <w*ZZ>) voor een edge exact via state vector.

        Args:
            target_i, target_j: target edge endpoints
            target_w: edge weight
            lc_nodes: set van nodes in de lightcone
            sub_edges: edges in de subgraaf
            p: circuit diepte
            gammas, betas: QAOA parameters (lijsten van lengte p)

        Returns:
            float: <ZZ> verwachtingswaarde
        """
        use_gpu = self.gpu and GPU_AVAILABLE
        xp = cp if use_gpu else np

        if self.fp32:
            fdtype = xp.float32
            cdtype = xp.complex64
        else:
            fdtype = xp.float64
            cdtype = xp.complex128

        # Map lightcone nodes naar lokale qubit-indices
        lc_sorted = sorted(lc_nodes, key=lambda n: self.node_pos.get(n, n))
        n_qubits = len(lc_sorted)
        dim = 2 ** n_qubits

        sv_limit = SV_THRESHOLD_GPU if use_gpu else SV_THRESHOLD_CPU
        if n_qubits > sv_limit:
            raise ValueError(
                "Lichtkegel %d qubits > %d limiet (edge %d-%d)" % (
                    n_qubits, sv_limit, target_i, target_j))

        local_idx = {node: qi for qi, node in enumerate(lc_sorted)}

        # Buffer management
        if dim > self._buf_max_dim:
            self._buf_max_dim = dim
            self._buf_state = xp.empty(dim, dtype=cdtype)
            self._buf_hphase = xp.empty(dim, dtype=fdtype)

        state = self._buf_state[:dim]
        H_phase = self._buf_hphase[:dim]

        bitstrings = xp.arange(dim)

        # Z-diagonalen cachen
        z_cache = {}
        for node in lc_sorted:
            qi = local_idx[node]
            z_cache[node] = (1 - 2 * ((bitstrings >> qi) & 1)).astype(fdtype)

        del bitstrings

        # Fase-Hamiltoniaan: H_phase = sum_edges w_ij * Z_i * Z_j
        H_phase[:] = 0
        for ei, ej, ew in sub_edges:
            H_phase += fdtype(ew) * z_cache[ei] * z_cache[ej]

        # Bewaar target-edge Z's, verwijder rest
        z_obs_a = z_cache[target_i]
        z_obs_b = z_cache[target_j]
        del z_cache

        # Mixer: Rx(beta) op alle qubits
        def apply_rx_all(state, beta):
            cb = fdtype(math.cos(float(beta)))
            msb = cdtype(-1j * math.sin(float(beta)))

            if use_gpu and hasattr(xp, 'fuse'):
                @xp.fuse()
                def _rx_fused(s0, s1):
                    return cb * s0 + msb * s1, msb * s0 + cb * s1

                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    new0, new1 = _rx_fused(s[:, 0, :], s[:, 1, :])
                    s[:, 0, :] = new0
                    s[:, 1, :] = new1
                    state = s.reshape(-1)
            else:
                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    tmp = cb * s[:, 0, :] + msb * s[:, 1, :]
                    s[:, 1, :] = msb * s[:, 0, :] + cb * s[:, 1, :]
                    s[:, 0, :] = tmp
                    state = s.reshape(-1)

            return state

        # |+>^n
        state[:] = cdtype(1.0 / math.sqrt(dim))

        # QAOA circuit
        for layer in range(p):
            gamma_scaled = fdtype(gammas[layer])
            state *= xp.exp(cdtype(-1j) * gamma_scaled * H_phase)
            state = apply_rx_all(state, betas[layer])

        # Meet <ZZ>
        zz_obs = z_obs_a * z_obs_b
        probs = xp.abs(state) ** 2
        result = float(xp.dot(probs, zz_obs))

        del probs, zz_obs, z_obs_a, z_obs_b

        return result

    def eval_cost(self, p, gammas, betas):
        """Bereken volledige MaxCut cost via lightcone per edge.

        Returns: ratio = cost / total_weight
        """
        total = 0.0
        t0 = time.time()

        if self.gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        cache = {}
        n_computed = 0
        n_cached = 0
        ordering_list = list(self.ordering)

        for idx, (ei, ej, ew) in enumerate(self.edge_list):
            # BFS lightcone
            lc_nodes = bfs_lightcone(self.graph, (ei, ej), p)
            sub_edges = lightcone_subgraph(self.graph, lc_nodes)

            # Cache key
            key = lightcone_cache_key(lc_nodes, sub_edges, (ei, ej), ordering_list)

            if key in cache:
                zz = cache[key]
                n_cached += 1
            else:
                zz = self.eval_edge_exact(ei, ej, ew, lc_nodes, sub_edges,
                                          p, gammas, betas)
                cache[key] = zz
                n_computed += 1

            total += ew * (1 - zz) / 2

            if self.verbose and (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print("    %d/%d edges (%.1fs, %d berekend, %d cached)" % (
                    idx + 1, self.n_edges, elapsed, n_computed, n_cached))

        elapsed = time.time() - t0
        ratio = total / self.total_weight

        if self.verbose:
            print("    Klaar: %d edges in %.3fs (%d uniek, %d cached)" % (
                self.n_edges, elapsed, n_computed, n_cached))

        return ratio

    def eval_ratio(self, p, gammas, betas):
        """Wrapper voor optimizer compatibiliteit."""
        return self.eval_cost(p, gammas, betas)

    def optimize(self, p, n_gamma=10, n_beta=10, refine=True):
        """Grid search + scipy verfijning.

        Returns: (ratio, gammas, betas, info)
        """
        t0 = time.time()
        old_verbose = self.verbose

        # Grid search
        gamma_range = np.linspace(0.05, np.pi, n_gamma)
        beta_range = np.linspace(0.05, np.pi / 2, n_beta)

        best_ratio = -1
        best_g = gamma_range[0]
        best_b = beta_range[0]
        n_evals = 0

        self.verbose = False
        for gi, g in enumerate(gamma_range):
            for b in beta_range:
                r = self.eval_ratio(p, [g] * p, [b] * p)
                n_evals += 1
                if r > best_ratio:
                    best_ratio = r
                    best_g = g
                    best_b = b
            if old_verbose and (gi + 1) % max(1, n_gamma // 5) == 0:
                self.verbose = old_verbose
                print("    Grid: %d/%d gamma, best=%.6f (%.1fs)" % (
                    gi + 1, n_gamma, best_ratio, time.time() - t0))
                self.verbose = False

        self.verbose = old_verbose
        best_gammas = [best_g] * p
        best_betas = [best_b] * p
        grid_time = time.time() - t0

        if self.verbose:
            print("    Grid klaar: ratio=%.6f (%.1fs, %d evals)" % (
                best_ratio, grid_time, n_evals))

        # Scipy verfijning
        if refine:
            try:
                from scipy.optimize import minimize as scipy_minimize

                def neg_ratio(params):
                    gs = list(params[:p])
                    bs = list(params[p:])
                    return -self.eval_ratio(p, gs, bs)

                x0 = best_gammas + best_betas
                result = scipy_minimize(neg_ratio, x0, method='Nelder-Mead',
                                        options={'maxiter': 200, 'xatol': 1e-5,
                                                 'fatol': 1e-6, 'adaptive': True})
                n_evals += result.nfev
                if -result.fun > best_ratio:
                    old_best = best_ratio
                    best_ratio = -result.fun
                    best_gammas = list(result.x[:p])
                    best_betas = list(result.x[p:])
                    if self.verbose:
                        print("    Scipy klaar: ratio=%.6f (+%.6f) (%d extra evals)" % (
                            best_ratio, best_ratio - old_best, result.nfev))

            except ImportError:
                if self.verbose:
                    print("    (scipy niet beschikbaar)")

        total_time = time.time() - t0
        info = {
            'grid_gamma': best_g,
            'grid_beta': best_b,
            'grid_time': grid_time,
            'total_time': total_time,
            'n_evals': n_evals,
        }

        return best_ratio, best_gammas, best_betas, info

    def optimize_progressive(self, p_max, n_gamma=10, n_beta=10,
                             refine=True, method='interp'):
        """Progressieve optimalisatie: p=1 -> p=2 -> ... -> p_max.

        Returns: dict met resultaten per p-niveau
        """
        old_verbose = self.verbose
        results = {}

        if old_verbose:
            print("\n  === Progressive optimizer: p=1 -> p=%d ===" % p_max)
            print("  Warm-start methode: %s\n" % method)

        # p=1: volledige grid search
        self.verbose = old_verbose
        ratio, gammas, betas, info = self.optimize(
            p=1, n_gamma=n_gamma, n_beta=n_beta, refine=refine)
        results[1] = {
            'ratio': ratio, 'gammas': list(gammas),
            'betas': list(betas), 'info': info
        }
        if old_verbose:
            print("  p=1: ratio=%.6f (%.1fs)\n" % (ratio, info['total_time']))

        # p=2 t/m p_max: warm-start
        for p in range(2, p_max + 1):
            t0 = time.time()
            if old_verbose:
                print("  p=%d: warm-start vanuit p=%d..." % (p, p - 1))

            prev_g = results[p - 1]['gammas']
            prev_b = results[p - 1]['betas']

            # Interp warm-start
            if method == 'interp':
                init_g = _interp_warmstart(prev_g, p)
                init_b = _interp_warmstart(prev_b, p)
            else:  # fourier / append
                init_g = prev_g + [prev_g[-1] * 0.5]
                init_b = prev_b + [prev_b[-1] * 0.5]

            # Evalueer startpunt
            self.verbose = False
            init_ratio = self.eval_ratio(p, init_g, init_b)
            n_evals = 1
            best_ratio = init_ratio
            best_gammas = list(init_g)
            best_betas = list(init_b)

            if old_verbose:
                print("    Init ratio: %.6f" % init_ratio)

            # Mini grid + scipy
            if refine:
                try:
                    from scipy.optimize import minimize as scipy_minimize

                    def neg_ratio(params):
                        gs = list(params[:p])
                        bs = list(params[p:])
                        return -self.eval_ratio(p, gs, bs)

                    # Multi-restart
                    for restart in range(3):
                        if restart == 0:
                            x0 = init_g + init_b
                        else:
                            x0 = [g + np.random.normal(0, 0.1) for g in best_gammas] + \
                                 [b + np.random.normal(0, 0.1) for b in best_betas]

                        result = scipy_minimize(neg_ratio, x0,
                                                method='Nelder-Mead',
                                                options={'maxiter': 300,
                                                         'xatol': 1e-5,
                                                         'fatol': 1e-6,
                                                         'adaptive': True})
                        n_evals += result.nfev
                        if -result.fun > best_ratio:
                            best_ratio = -result.fun
                            best_gammas = list(result.x[:p])
                            best_betas = list(result.x[p:])

                except ImportError:
                    pass

            self.verbose = old_verbose
            elapsed = time.time() - t0
            results[p] = {
                'ratio': best_ratio, 'gammas': best_gammas,
                'betas': best_betas, 'info': {
                    'total_time': elapsed, 'n_evals': n_evals,
                    'init_ratio': init_ratio, 'warmstart_from': p - 1
                }
            }
            if old_verbose:
                print("  p=%d: ratio=%.6f (%.1fs, %d evals)\n" % (
                    p, best_ratio, elapsed, n_evals))

        self.verbose = old_verbose
        return results


def _interp_warmstart(params, new_p):
    """Interpolatie warm-start: p-1 params -> p params."""
    old_p = len(params)
    if old_p == 0:
        return [0.3] * new_p
    x_old = np.linspace(0, 1, old_p)
    x_new = np.linspace(0, 1, new_p)
    return list(np.interp(x_new, x_old, params))


# =====================================================================
# Graaf-generators voor benchmark-topologieën
# =====================================================================

def make_triangular_grid(Lx, Ly):
    """Triangulair rooster: grid + diagonale edges.

    Elke cel (x,y)-(x+1,y)-(x,y+1)-(x+1,y+1) krijgt een diagonaal
    (x,y)-(x+1,y+1). Gefrustreerd — MaxCut is niet triviaal.
    """
    from rqaoa import WeightedGraph
    g = WeightedGraph.grid(Lx, Ly)
    for x in range(Lx - 1):
        for y in range(Ly - 1):
            # Diagonaal: (x,y) - (x+1, y+1)
            node_a = x * Ly + y
            node_b = (x + 1) * Ly + (y + 1)
            g.add_edge(node_a, node_b)
    return g


def make_random_geometric(n, radius, seed=42):
    """Random geometric graph: nodes uniform in [0,1]², edge als d < radius."""
    from rqaoa import WeightedGraph
    rng = np.random.RandomState(seed)
    positions = rng.rand(n, 2)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            if d < radius:
                g.add_edge(i, j)
    return g


def make_watts_strogatz(n, k, p_rewire, seed=42):
    """Watts-Strogatz small-world graph.

    Start met ring-lattice (k nearest neighbors), rewire met kans p.
    p=0: lokaal (lightcone klein), p=1: random (lightcone groot).
    """
    from rqaoa import WeightedGraph
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)

    # Ring lattice
    edges = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % n
            edges.add((min(i, nb), max(i, nb)))

    # Rewire
    rewired = set()
    for i, nb_orig in list(edges):
        if rng.rand() < p_rewire:
            # Rewire: vervang (i, nb) door (i, random)
            for _ in range(100):
                new_nb = rng.randint(0, n)
                if new_nb != i and (min(i, new_nb), max(i, new_nb)) not in edges:
                    rewired.add((i, nb_orig))
                    edges.discard((i, nb_orig))
                    edges.add((min(i, new_nb), max(i, new_nb)))
                    break

    for i, j in edges:
        g.add_edge(i, j)

    return g


def make_heavy_hex(n_rows):
    """Heavy-hex graph (IBM hardware topologie).

    Hexagonaal rooster met extra 'bridge' nodes op de horizontale edges.
    """
    from rqaoa import WeightedGraph
    g = WeightedGraph()
    node_id = 0
    grid = {}  # (row, col) -> node_id

    # Maak rijen van data qubits
    cols = n_rows * 2
    for r in range(n_rows):
        for c in range(cols):
            grid[(r, c)] = node_id
            g.add_node(node_id)
            node_id += 1
            # Horizontaal
            if c > 0:
                g.add_edge(grid[(r, c - 1)], grid[(r, c)])

    # Verticale connecties (alleen op even/oneven posities afhankelijk van rij)
    for r in range(n_rows - 1):
        for c in range(cols):
            if (r + c) % 2 == 0:
                # Voeg bridge node toe
                bridge = node_id
                g.add_node(bridge)
                node_id += 1
                g.add_edge(grid[(r, c)], bridge)
                g.add_edge(bridge, grid[(r + 1, c)])

    return g


# =====================================================================
# CLI
# =====================================================================

def load_cli_graph(graph_str):
    """Parse CLI --graph argument naar WeightedGraph.

    Formaten:
      petersen, dodecahedron          - built-in (gset_loader)
      grid:LxxLy                      - grid
      triangular:LxxLy               - triangulair grid
      random_geometric:N:R[:seed]     - random geometric
      watts_strogatz:N:K:P[:seed]     - small-world
      heavy_hex:N                     - IBM topologie
      gset:G14                        - Gset benchmark
      file:pad/naar/bestand.txt       - Gset-format bestand
    """
    from rqaoa import WeightedGraph

    if graph_str.startswith('grid:'):
        parts = graph_str[5:].lower().split('x')
        Lx, Ly = int(parts[0]), int(parts[1])
        return WeightedGraph.grid(Lx, Ly), "grid_%dx%d" % (Lx, Ly)

    if graph_str.startswith('triangular:'):
        parts = graph_str[11:].lower().split('x')
        Lx, Ly = int(parts[0]), int(parts[1])
        return make_triangular_grid(Lx, Ly), "triangular_%dx%d" % (Lx, Ly)

    if graph_str.startswith('random_geometric:'):
        parts = graph_str[17:].split(':')
        n, r = int(parts[0]), float(parts[1])
        seed = int(parts[2]) if len(parts) > 2 else 42
        return make_random_geometric(n, r, seed), "rgg_%d_%.2f_s%d" % (n, r, seed)

    if graph_str.startswith('watts_strogatz:'):
        parts = graph_str[15:].split(':')
        n, k, p = int(parts[0]), int(parts[1]), float(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 42
        return make_watts_strogatz(n, k, p, seed), "ws_%d_%d_%.2f_s%d" % (n, k, p, seed)

    if graph_str.startswith('heavy_hex:'):
        n_rows = int(graph_str[10:])
        return make_heavy_hex(n_rows), "heavy_hex_%d" % n_rows

    # gset_loader built-ins
    try:
        from gset_loader import load_graph
        graph, bks, info = load_graph(graph_str)
        return graph, graph_str
    except Exception:
        pass

    raise ValueError("Onbekende graaf: %s" % graph_str)


def main():
    parser = argparse.ArgumentParser(
        description='B54: Lightcone QAOA op willekeurige grafen')
    parser.add_argument('--graph', type=str, default='petersen',
                        help='Graaf specificatie (petersen, grid:8x4, etc.)')
    parser.add_argument('--p', type=int, default=1, help='Circuit diepte')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--ordering', choices=['auto', 'natural', 'rcm', 'fiedler'],
                        default='auto')
    parser.add_argument('--ngamma', type=int, default=10)
    parser.add_argument('--nbeta', type=int, default=10)
    parser.add_argument('--stats-only', action='store_true',
                        help='Toon alleen lightcone-statistieken')
    parser.add_argument('--compare-orderings', action='store_true',
                        help='Vergelijk alle ordering-methoden')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  B54: General Lightcone QAOA")
    print(sep)

    graph, graph_name = load_cli_graph(args.graph)
    print("  Graaf: %s (%d nodes, %d edges)" % (
        graph_name, graph.n_nodes, graph.n_edges))

    # Vergelijk orderings
    if args.compare_orderings:
        print("\n  Ordering vergelijking:")
        for name, order_fn in [('natural', ordering_natural),
                                ('rcm', lambda g: ordering_cuthill_mckee(g, True)),
                                ('fiedler', ordering_fiedler)]:
            order = order_fn(graph)
            bw = compute_bandwidth(graph, order)
            print("    %-10s: bandbreedte=%d" % (name, bw))

        # Lightcone impact per ordering
        print("\n  Lightcone impact (p=%d):" % args.p)
        for name, order_fn in [('natural', ordering_natural),
                                ('rcm', lambda g: ordering_cuthill_mckee(g, True)),
                                ('fiedler', ordering_fiedler)]:
            engine = GeneralLightconeQAOA(graph, verbose=False,
                                          ordering_method=name)
            stats = engine.lightcone_stats(args.p)
            print("    %-10s: avg=%.1f qubits, max=%d, uniek=%d/%d (%.0f%% cache)" % (
                name, stats['avg_qubits'], stats['max_qubits'],
                stats['n_unique'], engine.n_edges, stats['cache_rate'] * 100))
        print(sep)
        return

    engine = GeneralLightconeQAOA(graph, verbose=True, gpu=args.gpu,
                                   fp32=args.fp32,
                                   ordering_method=args.ordering)

    # Lightcone stats
    stats = engine.lightcone_stats(args.p)
    print("  Lightcone p=%d: avg=%.1f, max=%d, uniek=%d/%d" % (
        args.p, stats['avg_qubits'], stats['max_qubits'],
        stats['n_unique'], engine.n_edges))

    if stats['max_qubits'] > SV_THRESHOLD_CPU:
        print("  WAARSCHUWING: max lightcone %d qubits > %d limiet!" % (
            stats['max_qubits'], SV_THRESHOLD_CPU))

    if args.stats_only:
        print(sep)
        return

    # Optimalisatie
    print("\n  Optimalisatie p=1->%d:" % args.p)
    results = engine.optimize_progressive(args.p, n_gamma=args.ngamma,
                                           n_beta=args.nbeta)

    # Eindresultaat
    best = results[args.p]
    print(sep)
    print("  RESULTAAT: %s p=%d" % (graph_name, args.p))
    print("  Ratio:     %.6f" % best['ratio'])
    print("  Gammas:    %s" % ", ".join("%.4f" % g for g in best['gammas']))
    print("  Betas:     %s" % ", ".join("%.4f" % b for b in best['betas']))
    print("  Tijd:      %.1fs" % best['info']['total_time'])
    print("  Evaluaties:%d" % best['info']['n_evals'])
    print(sep)

    # Vergelijk met brute force op kleine grafen
    if graph.n_nodes <= 20:
        try:
            from rqaoa import brute_force_maxcut
            bf_cut = brute_force_maxcut(graph)
            exact_ratio = bf_cut / engine.total_weight
            print("  Brute force MaxCut: %d (ratio=%.6f)" % (bf_cut, exact_ratio))
            print("  QAOA / exact: %.2f%%" % (best['ratio'] / exact_ratio * 100))
        except Exception as e:
            print("  (brute force niet beschikbaar: %s)" % e)
    print(sep)


if __name__ == '__main__':
    main()
    main()
