"""
ZornSolver: Auto-Hybride Planner (B48)
=======================================
Uniforme solver-interface die per instantie automatisch de beste
simulatiemethode kiest. De gebruiker geeft alleen het probleem.

Gebruik:
    from zorn_solver import ZornSolver
    solver = ZornSolver(gpu=False, verbose=True)
    result = solver.solve(graph, p=1)
    print(result)

Beschikbare engines (automatisch geselecteerd):
    B21  LightconeQAOA       — sparse grafen, exact state vector per edge
    B54  GeneralLightconeQAOA — willekeurige grafen, lightcone decompositie
    B9   HeisenbergQAOA       — 2D grids, Heisenberg-MPO
    B29  TTCrossQAOA          — brede grids (Ly>=5), RSVD
    B47  RQAOA                — grote willekeurige grafen, recursief
    B50  GraphPruning         — preprocessing: verwijder leaves/isolates exact

Architectuur:
    1. Graph classifier → easy/hard triage
    2. Method dispatcher → kies engine op basis van structuur + budget
    3. Parameter optimizer → grid search + refinement
    4. Local search polisher → steepest descent op bitstring
"""
# SYNC_MARKER_B27
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

# =====================================================================
# RESULT DATACLASS
# =====================================================================

@dataclass
class SolverResult:
    """Resultaat van ZornSolver.solve()."""
    cut_value: float               # Geschatte MaxCut waarde
    ratio: float                   # cut / n_edges
    best_bitstring: Optional[np.ndarray] = None  # Beste gevonden bitstring
    method: str = ""               # Gebruikte methode
    engine: str = ""               # Specifieke engine class
    p: int = 1                     # QAOA diepte
    gammas: Optional[list] = None
    betas: Optional[list] = None
    wall_time: float = 0.0        # Totale tijd (seconden)
    triage: str = ""              # Triage classificatie
    graph_stats: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __repr__(self):
        return (f"SolverResult(cut={self.cut_value:.2f}, ratio={self.ratio:.6f}, "
                f"method={self.method}, engine={self.engine}, "
                f"p={self.p}, time={self.wall_time:.2f}s, triage={self.triage})")


# =====================================================================
# GRAPH CLASSIFIER (triage)
# =====================================================================

def classify_graph(n_nodes: int, edges: list, weights: Optional[dict] = None) -> Dict:
    """Statische analyse van een graaf voor triage.
    
    Returns dict met:
        n_nodes, n_edges, avg_degree, max_degree, density,
        is_grid, grid_dims, is_bipartite, is_sparse,
        treewidth_upper, classification ('easy'|'medium'|'hard')
    """
    stats = {
        'n_nodes': n_nodes,
        'n_edges': len(edges),
        'avg_degree': 2 * len(edges) / n_nodes if n_nodes > 0 else 0,
    }
    
    # Degree analysis
    degree = np.zeros(n_nodes, dtype=int)
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
    
    stats['max_degree'] = int(degree.max()) if n_nodes > 0 else 0
    stats['min_degree'] = int(degree.min()) if n_nodes > 0 else 0
    stats['density'] = 2 * len(edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    # Grid detection
    grid_dims = _detect_grid(n_nodes, edges, degree, adj)
    stats['is_grid'] = grid_dims is not None
    stats['grid_dims'] = grid_dims
    
    # Bipartiteness check (BFS 2-coloring)
    stats['is_bipartite'] = _is_bipartite(n_nodes, adj)
    
    # Sparsity
    stats['is_sparse'] = stats['density'] < 0.1
    
    # Treewidth upper bound (greedy minimum degree)
    stats['treewidth_upper'] = _treewidth_greedy(n_nodes, adj)
    
    # Symmetrie-analyse (B27)
    try:
        from graph_automorphism import symmetry_info as _sym_info
        sym = _sym_info(n_nodes, edges, weights)
        stats['n_orbits'] = sym.n_orbits
        stats['orbit_sizes'] = sym.orbit_sizes
        stats['is_vertex_transitive'] = sym.is_vertex_transitive
        stats['symmetry_factor'] = sym.symmetry_factor
    except ImportError:
        stats['n_orbits'] = n_nodes
        stats['orbit_sizes'] = [1] * n_nodes
        stats['is_vertex_transitive'] = False
        stats['symmetry_factor'] = 1.0

    # Classification
    if n_nodes <= 20 and len(edges) <= 30:
        stats['classification'] = 'easy'
    elif stats['is_grid'] and grid_dims is not None:
        Lx, Ly = grid_dims
        if Ly <= 4:
            stats['classification'] = 'medium'  # MPS handles well
        else:
            stats['classification'] = 'hard'    # d-wall territory
    elif stats['treewidth_upper'] <= 10:
        stats['classification'] = 'medium'
    elif stats['is_sparse'] and n_nodes <= 100:
        stats['classification'] = 'medium'
    else:
        stats['classification'] = 'hard'

    return stats


def _detect_grid(n_nodes, edges, degree, adj):
    """Detecteer of de graaf een Lx×Ly grid is. Return (Lx,Ly) of None."""
    n_edges = len(edges)
    # Grid n=Lx*Ly has edges = Lx*(Ly-1) + (Lx-1)*Ly
    # Try all factor pairs
    for Ly in range(1, int(np.sqrt(n_nodes)) + 2):
        if n_nodes % Ly != 0:
            continue
        Lx = n_nodes // Ly
        if Lx < Ly:
            break
        expected_edges = Lx * (Ly - 1) + (Lx - 1) * Ly
        if n_edges != expected_edges:
            continue
        # Check degree distribution: corners=2, edges=3, interior=4
        d_counts = np.bincount(degree, minlength=5)
        corners = 4 if Lx > 1 and Ly > 1 else (2 if Lx > 1 or Ly > 1 else 1)
        if Lx == 1 or Ly == 1:
            # 1D chain
            if Lx > 1 and d_counts[1] == 2 and d_counts[2] == Lx - 2:
                return (Lx, 1)
            continue
        expected_corners = 4
        expected_border = 2 * (Lx - 2) + 2 * (Ly - 2)
        expected_interior = (Lx - 2) * (Ly - 2)
        if (d_counts[2] == expected_corners and 
            d_counts[3] == expected_border and
            d_counts[4] == expected_interior):
            return (Lx, Ly)
    return None


def _is_bipartite(n_nodes, adj):
    """BFS 2-coloring check."""
    color = [-1] * n_nodes
    for start in range(n_nodes):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def _treewidth_greedy(n_nodes, adj):
    """Greedy minimum-degree treewidth bovengrens."""
    if n_nodes <= 1:
        return 0
    # Work with sets for efficient neighbor operations
    neighbors = [set(a) for a in adj]
    alive = set(range(n_nodes))
    tw = 0
    
    for _ in range(n_nodes):
        if not alive:
            break
        # Find node with minimum degree among alive nodes
        min_deg = n_nodes + 1
        min_node = -1
        for u in alive:
            d = len(neighbors[u] & alive)
            if d < min_deg:
                min_deg = d
                min_node = u
        
        tw = max(tw, min_deg)
        
        # Eliminate: connect all neighbors, remove node
        nbrs = list(neighbors[min_node] & alive)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                neighbors[nbrs[i]].add(nbrs[j])
                neighbors[nbrs[j]].add(nbrs[i])
        alive.discard(min_node)
    
    return tw


# =====================================================================
# LOCAL SEARCH POLISHER
# =====================================================================

def local_search_maxcut(n_nodes: int, edges: list, 
                         weights: Optional[dict] = None,
                         init_bitstring: Optional[np.ndarray] = None,
                         n_restarts: int = 10,
                         rng: Optional[np.random.Generator] = None) -> Tuple[float, np.ndarray]:
    """Steepest-descent local search met random restarts.
    
    Returns (best_cut_value, best_bitstring).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Edge weights
    w = {}
    for u, v in edges:
        key = (min(u,v), max(u,v))
        w[key] = weights.get(key, 1.0) if weights else 1.0
    
    def cut_value(bits):
        return sum(w[min(u,v), max(u,v)] for u, v in edges if bits[u] != bits[v])
    
    def steepest_descent(bits):
        bits = bits.copy()
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_flip = -1
            for i in range(n_nodes):
                # Compute gain of flipping bit i
                gain = 0
                for j in (adj[i] if i < len(adj) else []):
                    key = (min(i,j), max(i,j))
                    if bits[i] == bits[j]:
                        gain += w.get(key, 1.0)  # cutting a new edge
                    else:
                        gain -= w.get(key, 1.0)  # uncuts an edge
                if gain > best_gain:
                    best_gain = gain
                    best_flip = i
            if best_flip >= 0:
                bits[best_flip] = 1 - bits[best_flip]
                improved = True
        return bits, cut_value(bits)
    
    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_cut = 0
    best_bits = None
    
    for r in range(n_restarts):
        if r == 0 and init_bitstring is not None:
            bits = init_bitstring.copy()
        else:
            bits = rng.integers(0, 2, size=n_nodes)
        
        bits, cv = steepest_descent(bits)
        if cv > best_cut:
            best_cut = cv
            best_bits = bits.copy()
    
    return best_cut, best_bits


# =====================================================================
# METHOD DISPATCHER
# =====================================================================

def _pick_method(stats: Dict, p: int, chi_budget: int, gpu: bool,
                 time_budget: float) -> Tuple[str, Dict]:
    """Kies de beste methode op basis van graph stats en budget.
    
    Returns (method_name, config_dict).
    """
    n = stats['n_nodes']
    m = stats['n_edges']
    is_grid = stats['is_grid']
    grid_dims = stats['grid_dims']
    tw = stats['treewidth_upper']
    
    # 1. Kleine grafen → exact methoden
    if n <= 20:
        return 'lanczos_exact', {'n_qubits': n}
    if n <= 22:
        return 'brute_force', {'n_qubits': n}
    
    # 2. Grids → specialized grid engines
    if is_grid and grid_dims is not None:
        Lx, Ly = grid_dims
        if Ly == 1:
            # 1D chain → Heisenberg exact
            return 'heisenberg', {'Lx': Lx, 'Ly': 1, 'chi': 4}
        elif Ly <= 4:
            # MPS handles well, use Heisenberg-MPO
            return 'heisenberg', {'Lx': Lx, 'Ly': Ly, 'chi': chi_budget}
        elif Ly <= 8:
            # d-wall territory → RSVD engine
            return 'rsvd', {'Lx': Lx, 'Ly': Ly, 'chi': chi_budget}
        else:
            # Very wide → lightcone + MPS hybrid
            return 'lightcone', {'Lx': Lx, 'Ly': Ly, 'chi': chi_budget}
    
    # 3. Sparse non-grid → lightcone decomposition
    if stats['is_sparse'] and n <= 500:
        # Check if lightcone sizes are manageable
        max_lc = min(n, 2 * p * stats['max_degree'] + 2)
        if max_lc <= 26 and gpu:
            return 'general_lightcone', {'max_lc_qubits': max_lc, 'gpu': True}
        elif max_lc <= 22:
            return 'general_lightcone', {'max_lc_qubits': max_lc, 'gpu': False}
        else:
            return 'general_lightcone_mps', {'max_lc_qubits': max_lc, 'chi': chi_budget}
    
    # 4. Dense or large → RQAOA
    if n > 100 or stats['density'] > 0.3:
        return 'rqaoa', {'mode': 'fast'}
    
    # 5. Medium graphs → general lightcone with MPS fallback
    return 'general_lightcone_mps', {'chi': chi_budget}


def _choose_reorder(stats: Dict, method_name: str, n_nodes: int,
                    edges: list, weights: Optional[dict] = None,
                    requested: str = 'auto') -> Tuple[str, Dict]:
    """Kies een conservatieve reorder-strategie voor tensor-routes."""
    info = {'requested_reorder': requested}

    if requested in ('none', 'fiedler'):
        info['chosen_reorder'] = requested
        return requested, info

    if requested != 'auto':
        raise ValueError(f"Onbekende reorder-modus: {requested}")

    if method_name not in ('general_lightcone', 'general_lightcone_mps', 'rqaoa'):
        info['chosen_reorder'] = 'none'
        info['reorder_reason'] = 'method_not_order_sensitive'
        return 'none', info

    if stats.get('is_grid'):
        info['chosen_reorder'] = 'none'
        info['reorder_reason'] = 'grid_preserves_native_layout'
        return 'none', info

    if not stats.get('is_sparse', False):
        info['chosen_reorder'] = 'none'
        info['reorder_reason'] = 'dense_graph'
        return 'none', info

    if n_nodes > 512:
        info['chosen_reorder'] = 'none'
        info['reorder_reason'] = 'graph_too_large_for_auto_fiedler'
        return 'none', info

    try:
        from general_lightcone import (
            compute_bandwidth,
            ordering_fiedler,
            ordering_natural,
        )
        from rqaoa import WeightedGraph

        graph = WeightedGraph()
        for i in range(n_nodes):
            graph.add_node(i)
        for u, v in edges:
            w = weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0
            graph.add_edge(u, v, w)

        bw_natural = compute_bandwidth(graph, ordering_natural(graph))
        bw_fiedler = compute_bandwidth(graph, ordering_fiedler(graph))
        info['natural_bandwidth'] = bw_natural
        info['fiedler_bandwidth'] = bw_fiedler

        if bw_fiedler + 2 < 0.85 * max(bw_natural, 1):
            info['chosen_reorder'] = 'fiedler'
            info['reorder_reason'] = 'fiedler_bandwidth_win'
            return 'fiedler', info
    except Exception as exc:
        info['reorder_reason'] = f'heuristic_failed:{exc}'

    info['chosen_reorder'] = 'none'
    info.setdefault('reorder_reason', 'no_clear_bandwidth_gain')
    return 'none', info


# =====================================================================
# ZORN SOLVER
# =====================================================================

class ZornSolver:
    """Uniforme MaxCut solver — kiest automatisch de beste methode.
    
    Parameters
    ----------
    chi_budget : int
        Maximale bond dimensie voor MPS-methoden.
    gpu : bool
        Gebruik GPU als beschikbaar.
    time_budget : float
        Maximale wandkloktijd in seconden (0 = onbeperkt).
    mixed_precision : bool
        B19: fp32 tensors, fp64 SVD.
    verbose : bool
        Print voortgang.
    """
    
    def __init__(self, chi_budget: int = 32, gpu: bool = False,
                 time_budget: float = 0, mixed_precision: bool = False,
                 reorder: str = 'auto', verbose: bool = True):
        self.chi_budget = chi_budget
        self.gpu = gpu
        self.time_budget = time_budget
        self.mixed_precision = mixed_precision
        self.reorder = reorder
        self.verbose = verbose
    
    def solve(self, n_nodes: int, edges: list,
              weights: Optional[dict] = None,
              p: int = 1,
              method: Optional[str] = None,
              reorder: Optional[str] = None) -> SolverResult:
        """Los MaxCut op.
        
        Parameters
        ----------
        n_nodes : int
            Aantal nodes in de graaf.
        edges : list of (int, int)
            Lijst van edges.
        weights : dict, optional
            Edge weights {(u,v): w}. Default: unit weights.
        p : int
            QAOA circuit diepte.
        method : str, optional
            Forceer een methode. None = automatisch.
        
        Returns
        -------
        SolverResult
        """
        t0 = time.time()
        n_edges_orig = len(edges)

        # 1. Classify (originele graaf)
        stats = classify_graph(n_nodes, edges, weights)
        if self.verbose:
            grid_str = f" ({stats['grid_dims'][0]}×{stats['grid_dims'][1]})" if stats['is_grid'] else ""
            vt_str = " VT" if stats.get('is_vertex_transitive') else ""
            orb_str = f", {stats.get('n_orbits', '?')} orbits{vt_str}"
            print(f"[ZornSolver] n={n_nodes}, m={n_edges_orig}{grid_str}, "
                  f"tw≤{stats['treewidth_upper']}{orb_str}, class={stats['classification']}")

        # 1b. Prune (B50) — verwijder leaves, isolates exact
        prune_result = None
        if not stats['is_grid']:  # Grids hebben geen leaves, skip pruning
            from graph_pruning import prune_graph
            prune_result = prune_graph(n_nodes, edges, weights, verbose=self.verbose)

            if prune_result.nodes_removed > 0:
                if self.verbose:
                    print(f"[ZornSolver] B50 pruning: {n_nodes}->{prune_result.n_nodes} nodes, "
                          f"guaranteed_cut={prune_result.guaranteed_cut:.1f}")

                if prune_result.n_nodes == 0:
                    # Hele graaf was reduceerbaar!
                    full_bits = prune_result.reconstruct(np.array([], dtype=int))
                    result = SolverResult(
                        cut_value=prune_result.guaranteed_cut,
                        ratio=prune_result.guaranteed_cut / n_edges_orig if n_edges_orig else 0,
                        best_bitstring=full_bits,
                        method='pruning_exact', engine='B50',
                        p=0, wall_time=time.time() - t0,
                        graph_stats=stats, triage=stats['classification'])
                    result.notes.append(f"fully_pruned:{prune_result.summary()}")
                    if self.verbose:
                        print(f"[ZornSolver] Done (fully pruned): ratio={result.ratio:.6f}, "
                              f"cut={result.cut_value:.1f}/{n_edges_orig}, "
                              f"time={result.wall_time:.2f}s")
                    return result

                # Werk met gereduceerde graaf
                n_nodes = prune_result.n_nodes
                edges = prune_result.edges
                weights = prune_result.weights
                # Herclassificeer gereduceerde graaf
                stats = classify_graph(n_nodes, edges, weights)
                if self.verbose:
                    grid_str2 = f" ({stats['grid_dims'][0]}×{stats['grid_dims'][1]})" if stats['is_grid'] else ""
                    print(f"[ZornSolver] Reduced graph: n={n_nodes}, m={len(edges)}{grid_str2}, "
                          f"class={stats['classification']}")

        # 2. Pick method
        if method is not None:
            method_name = method
            # Build config from stats for forced methods
            config = {}
            if stats['is_grid'] and stats['grid_dims']:
                config['Lx'], config['Ly'] = stats['grid_dims']
                config['chi'] = self.chi_budget
        else:
            method_name, config = _pick_method(
                stats, p, self.chi_budget, self.gpu, self.time_budget)

        reorder_request = self.reorder if reorder is None else reorder
        chosen_reorder, reorder_info = _choose_reorder(
            stats, method_name, n_nodes, edges, weights,
            requested=reorder_request)
        if method_name in ('general_lightcone', 'general_lightcone_mps', 'rqaoa'):
            config = dict(config)
            config['reorder'] = chosen_reorder
            config['reorder_info'] = reorder_info

        if self.verbose:
            print(f"[ZornSolver] Method: {method_name} {config}")

        # 3. Run engine
        result = self._run_engine(method_name, config, n_nodes, edges, weights, p, stats)

        # 4. Local search polish
        if result.best_bitstring is None and n_nodes <= 10000:
            # Generate bitstring from QAOA angles if available
            pass  # TODO: QAOA sampling

        if n_nodes <= 5000:
            ls_cut, ls_bits = local_search_maxcut(
                n_nodes, edges, weights,
                init_bitstring=result.best_bitstring,
                n_restarts=min(50, max(5, 1000 // n_nodes)))
            if ls_cut > result.cut_value:
                if self.verbose:
                    print(f"[ZornSolver] Local search improved: {result.cut_value:.1f} -> {ls_cut:.1f}")
                result.cut_value = ls_cut
                result.best_bitstring = ls_bits
                result.ratio = ls_cut / len(edges) if edges else 0
                result.notes.append(f"local_search_improved")

        # 5. Reconstruct full bitstring als gepruned (B50)
        if prune_result is not None and prune_result.nodes_removed > 0:
            if result.best_bitstring is not None:
                full_bits = prune_result.reconstruct(result.best_bitstring)
                result.best_bitstring = full_bits
            result.cut_value += prune_result.guaranteed_cut
            result.ratio = result.cut_value / n_edges_orig if n_edges_orig else 0
            result.notes.append(f"B50_pruned:{prune_result.nodes_removed}_nodes")

        result.wall_time = time.time() - t0
        result.graph_stats = stats
        result.triage = stats['classification']
        if method_name in ('general_lightcone', 'general_lightcone_mps', 'rqaoa'):
            reorder_note = f"reorder:{chosen_reorder}"
            if reorder_note not in result.notes:
                result.notes.append(reorder_note)
            reason = reorder_info.get('reorder_reason')
            if reason:
                reason_note = f"reorder_reason:{reason}"
                if reason_note not in result.notes:
                    result.notes.append(reason_note)
            if 'natural_bandwidth' in reorder_info:
                bw_note = (
                    f"bandwidth:natural={reorder_info['natural_bandwidth']},"
                    f"fiedler={reorder_info['fiedler_bandwidth']}")
                if bw_note not in result.notes:
                    result.notes.append(bw_note)

        if self.verbose:
            print(f"[ZornSolver] Done: ratio={result.ratio:.6f}, "
                  f"cut={result.cut_value:.1f}/{n_edges_orig}, "
                  f"time={result.wall_time:.2f}s")

        return result
    
    def _run_engine(self, method_name, config, n_nodes, edges, weights, p, stats):
        """Draai de gekozen engine."""
        n_edges = len(edges)
        
        if method_name == 'lanczos_exact':
            return self._run_lanczos(n_nodes, edges, weights, n_edges)

        elif method_name == 'brute_force':
            return self._brute_force(n_nodes, edges, weights, n_edges)
        
        elif method_name == 'heisenberg':
            return self._run_heisenberg(config, n_edges, p)
        
        elif method_name == 'rsvd':
            return self._run_rsvd(config, n_edges, p)
        
        elif method_name in ('lightcone', 'general_lightcone', 'general_lightcone_mps'):
            return self._run_lightcone(method_name, config, n_nodes, edges, n_edges, p)
        
        elif method_name == 'rqaoa':
            return self._run_rqaoa(config, n_nodes, edges, weights, n_edges)
        
        else:
            raise ValueError(f"Onbekende methode: {method_name}")
    
    def _run_lanczos(self, n_nodes, edges, weights, n_edges):
        """Exact MaxCut via Lanczos (B37)."""
        from lanczos_bench import lanczos_maxcut
        mc = lanczos_maxcut(edges, n_nodes, weights)
        return SolverResult(
            cut_value=mc.max_cut, ratio=mc.ratio,
            best_bitstring=mc.bitstring, method='lanczos_exact',
            engine='lanczos_bench', p=0)

    def _brute_force(self, n_nodes, edges, weights, n_edges):
        """Exact brute-force met B27 symmetrie-breking (2x speedup)."""
        from graph_automorphism import orbit_brute_force
        best_cut, best_bits = orbit_brute_force(n_nodes, edges, weights,
                                                 max_nodes=24)
        return SolverResult(
            cut_value=best_cut, ratio=best_cut / n_edges if n_edges else 0,
            best_bitstring=best_bits, method='brute_force', engine='B27_sym',
            p=0)
    
    def _run_heisenberg(self, config, n_edges, p):
        """HeisenbergQAOA voor grids — met B57 warm-start."""
        from zorn_mps import HeisenbergQAOA
        from param_library import warm_grid_search, ParamLibrary
        Lx, Ly = config['Lx'], config['Ly']
        chi = config.get('chi', self.chi_budget)

        qaoa = HeisenbergQAOA(Lx, Ly, max_chi=chi, gpu=self.gpu,
                               mixed_precision=self.mixed_precision)

        def eval_fn(gammas, betas):
            return qaoa.eval_ratio(p, gammas, betas)

        # Build stats for param lookup
        stats = {'is_grid': True, 'grid_dims': (Lx, Ly),
                 'avg_degree': 2 * n_edges / (Lx * Ly),
                 'min_degree': 2, 'max_degree': 4,
                 'is_bipartite': True, 'density': 0}

        best_r, best_gammas, best_betas, source = warm_grid_search(
            eval_fn, stats, p=p, n_points=8, spread=0.3)

        if self.verbose:
            print(f"[ZornSolver] Warm-start source: {source}")

        # Refine with scipy
        try:
            from scipy.optimize import minimize
            def neg_ratio(params):
                return -qaoa.eval_ratio(p, list(params[:p]), list(params[p:]))
            x0 = list(best_gammas) + list(best_betas)
            res = minimize(neg_ratio, x0, method='Nelder-Mead',
                          options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
            if -res.fun > best_r:
                best_r = -res.fun
                best_gammas = list(res.x[:p])
                best_betas = list(res.x[p:])
        except ImportError:
            pass

        cut = best_r * n_edges
        return SolverResult(
            cut_value=cut, ratio=best_r,
            method='heisenberg_mpo', engine='HeisenbergQAOA',
            p=p, gammas=best_gammas, betas=best_betas)
    
    def _run_rsvd(self, config, n_edges, p):
        """TTCrossQAOA met RSVD voor brede grids."""
        from tt_cross_qaoa import TTCrossQAOA
        Lx, Ly = config['Lx'], config['Ly']
        chi = config.get('chi', self.chi_budget)

        qaoa = TTCrossQAOA(Lx, Ly, chi_max=chi)
        best_r, best_g, best_b = qaoa.optimize(p, n_gamma=12, n_beta=12, refine=True)

        return SolverResult(
            cut_value=best_r * n_edges, ratio=best_r,
            method='rsvd_ttcross', engine='TTCrossQAOA', p=p,
            gammas=best_g, betas=best_b)

    def _run_lightcone(self, method_name, config, n_nodes, edges, n_edges, p):
        """Lightcone QAOA -- grid (B21) of general (B54).

        Nu met BFS-diamant: exacte causale kooi i.p.v. kolom-blokken.
        """
        from param_library import warm_grid_search, ParamLibrary

        if method_name == 'lightcone':
            # Grid lightcone (B21) -- nu met BFS-diamant automatisch
            from lightcone_qaoa import LightconeQAOA
            Lx = config.get('Lx', 8)
            Ly = config.get('Ly', 4)
            chi = config.get('chi', None)
            qaoa = LightconeQAOA(Lx, Ly, verbose=self.verbose, chi=chi,
                                  gpu=self.gpu, fp32=self.mixed_precision)

            def eval_fn(gammas, betas):
                return qaoa.eval_ratio(p, gammas, betas)

            # B57 warm-start
            stats = {'is_grid': True, 'grid_dims': (Lx, Ly),
                     'avg_degree': 2 * n_edges / (Lx * Ly),
                     'min_degree': 2, 'max_degree': 4,
                     'is_bipartite': True, 'density': 0}
            best_r, best_gammas, best_betas, source = warm_grid_search(
                eval_fn, stats, p=p, n_points=8, spread=0.3)

            if self.verbose:
                print(f"[ZornSolver] Warm-start source: {source}")

        else:
            # General lightcone (B54) -- willekeurige grafen
            from general_lightcone import GeneralLightconeQAOA
            from rqaoa import WeightedGraph

            wg = WeightedGraph()
            for i in range(n_nodes):
                wg.add_node(i)
            for u, v in edges:
                wg.add_edge(u, v)

            reorder = config.get('reorder', 'none')
            ordering_method = (
                'fiedler' if reorder == 'fiedler'
                else 'natural' if reorder == 'none'
                else 'auto'
            )
            qaoa = GeneralLightconeQAOA(
                wg, verbose=self.verbose, gpu=self.gpu,
                ordering_method=ordering_method)

            stats = classify_graph(n_nodes, edges)

            def eval_fn(gammas, betas):
                return qaoa.eval_ratio(p, gammas, betas)

            best_r, best_gammas, best_betas, source = warm_grid_search(
                eval_fn, stats, p=p, n_points=8, spread=0.3)

            if self.verbose:
                print(f"[ZornSolver] Warm-start source: {source}")

        result = SolverResult(
            cut_value=best_r * n_edges, ratio=best_r,
            method=method_name, engine='LightconeQAOA_BFS', p=p,
            gammas=best_gammas, betas=best_betas)
        if method_name != 'lightcone':
            result.notes.append(f"reorder:{config.get('reorder', 'none')}")
            result.notes.append(f"ordering:{qaoa.ordering_name}")
            result.notes.append(f"bandwidth:{qaoa.bandwidth}")
        return result

    def _run_rqaoa(self, config, n_nodes, edges, weights, n_edges):
        """RQAOA voor grote willekeurige grafen."""
        from rqaoa import WeightedGraph, RQAOA

        wg = WeightedGraph()
        for i in range(n_nodes):
            wg.add_node(i)
        for u, v in edges:
            w = weights.get((min(u,v), max(u,v)), 1.0) if weights else 1.0
            wg.add_edge(u, v, w)

        rqaoa = RQAOA(wg, p=1, verbose=self.verbose)
        reorder = config.get('reorder', 'auto')
        result = rqaoa.solve(mode=config.get('mode', 'auto'),
                             reorder=reorder)

        notes = []
        notes.append(f"reorder:{result.info.get('requested_reorder', reorder)}")
        if result.info.get('resolved_ordering'):
            notes.append(f"ordering:{result.info['resolved_ordering']}")
        if result.info.get('bandwidth') is not None:
            notes.append(f"bandwidth:{result.info['bandwidth']}")
        if result.info.get('mode') == 'spectral_fallback':
            notes.append('RQAOA spectral fallback gebruikt')

        return SolverResult(
            cut_value=result.cut_value,
            ratio=result.ratio if n_edges else 0,
            best_bitstring=result.bitstring,
            method='rqaoa', engine='RQAOA', p=1, notes=notes)
