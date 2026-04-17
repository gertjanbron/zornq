#!/usr/bin/env python3
"""
auto_dispatcher.py - B130 Auto-Dispatcher / Strategy Selector

De unified compute engine van ZornQ. Kiest automatisch de beste
strategie voor elk probleem op basis van grafstructuur, grootte,
beschikbare hardware en tijdbudget.

Integreert ALLE solvers:
  Tier 1 (Exact):     Brute-force, Lanczos, Pfaffian Oracle
  Tier 2 (Quantum):   MPS-QAOA (Heisenberg, lightcone, RSVD, RQAOA)
  Tier 3 (Classical): BLS, PA, Combined BLS+PA, CUDA varianten

Strategie-matrix:
  n <= 25, bipartite planar  -> Pfaffian Oracle (exact, O(n^3))
  n <= 20                    -> Lanczos exact
  n <= 22                    -> Brute-force + symmetrie
  Grid Ly <= 4               -> MPS-QAOA + BLS polish
  Grid Ly 5-8                -> RSVD-QAOA + PA polish
  Sparse, n <= 500           -> Lightcone QAOA + BLS polish
  n <= 2000                  -> PA (best quality) of BLS (fast)
  n > 2000, GPU              -> CUDA PA/BLS
  n > 2000, no GPU           -> PA + BLS combined
  Fallback                   -> BLS (always works, fast)

Architectuur:
  1. classify()    -> graph analysis (structure, size, features)
  2. select()      -> pick strategy tier + solver
  3. execute()     -> run solver pipeline (possibly multi-stage)
  4. certify()     -> quality assessment (gap bound, confidence)

Usage:
    from auto_dispatcher import ZornDispatcher
    d = ZornDispatcher()
    result = d.solve(n_nodes, edges)
    print(result)

    # Of kort:
    from auto_dispatcher import solve_maxcut
    result = solve_maxcut(n_nodes, edges)
"""

import numpy as np
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bandit_planner import (
    DispatcherBandit,
    family_key_from_info,
    reward_from_dispatch_result,
)

# ---- B170 twin-width integratie (optioneel) ----
# Cograph-detectie + tww als difficulty-feature voor dispatcher-routing.
# Maakt het mogelijk niet-bipartiete cographs (b.v. K_n voor n>25) in
# O(n^3) exact op te lossen, waar brute-force 2^n vastloopt.
try:
    from b170_twin_width import (
        Trigraph, twin_width_heuristic, is_cograph, cograph_maxcut_exact,
    )
    _B170_AVAILABLE = True
except ImportError:  # pragma: no cover
    _B170_AVAILABLE = False


def _compute_tww_feature(n, edges):
    """Bereken twin-width / cograph-features indien graaf-grootte dit toestaat.

    Returns dict met:
      - 'is_cograph' : bool of None (None = niet berekend)
      - 'tww'        : int of None (upper bound via greedy heuristic)

    Budget:
      * cograph-check is O(n^4) via P_4-enumeratie; cap n <= 100.
      * tww-heuristiek is O(n^5); cap n <= 32.
    Boven deze drempels blijven features None zonder exception.
    """
    feat = {'is_cograph': None, 'tww': None}
    if not _B170_AVAILABLE or n <= 1:
        return feat
    uw_edges = [(int(e[0]), int(e[1])) for e in edges]
    if n <= 100:
        try:
            feat['is_cograph'] = bool(is_cograph(n, uw_edges))
        except Exception:  # pragma: no cover - defensief
            feat['is_cograph'] = None
    if n <= 32:
        try:
            g = Trigraph.from_graph(n, uw_edges)
            d_max, _seq = twin_width_heuristic(g)
            feat['tww'] = int(d_max)
        except Exception:  # pragma: no cover - defensief
            feat['tww'] = None
    return feat


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class DispatchResult:
    """Resultaat van ZornDispatcher.solve()."""
    best_cut: float
    assignment: Dict[int, int]
    ratio: Optional[float] = None       # cut / n_edges
    gap_bound: Optional[float] = None   # gap to known upper bound
    is_exact: bool = False              # is this provably optimal?
    strategy: str = ""                  # strategy name
    tier: str = ""                      # exact / quantum / classical
    solvers_used: List[str] = field(default_factory=list)
    time_s: float = 0.0
    graph_info: Dict = field(default_factory=dict)
    certificate: str = ""              # quality certificate
    notes: List[str] = field(default_factory=list)

    def summary(self):
        exact_str = " [EXACT]" if self.is_exact else ""
        cert = f", cert={self.certificate}" if self.certificate else ""
        return (f"cut={self.best_cut:.0f}, ratio={self.ratio:.4f}{exact_str}, "
                f"strategy={self.strategy}, tier={self.tier}, "
                f"solvers={'+'.join(self.solvers_used)}, "
                f"time={self.time_s:.2f}s{cert}")


# ============================================================
# Graph classifier
# ============================================================

def has_signed_edges(edges, tol=1e-12):
    """Return True als ten minste één edge een negatief gewicht heeft.

    Pfaffian-Oracle's bipartite- en grid-short-circuits aannemen dat
    `max-cut = sum(weights)`. Voor signed instanties (BiqMac-spinglass,
    random ±1) klopt die aanname niet: een signed instance kan negatieve
    totale gewichtsom hebben terwijl de optimale cut-waarde strikt
    positief is (cut alleen de positieve edges).

    Gebruikt in:
    - `select_strategy`: signed instanties *niet* naar pfaffian_exact/
      exact_small/exact_brute routen (die gebruiken allemaal
      pfaffian_maxcut onder de motorkap).
    - `certify_result`: defense-in-depth — als een solver per ongeluk
      `is_exact=True` retourneert op een signed instance, downgrade het
      certificaat naar APPROXIMATE i.p.v. EXACT.

    Edges zonder expliciet gewicht (len(e)==2) zijn impliciet +1 en dus
    nooit signed.
    """
    for e in edges:
        if len(e) > 2 and float(e[2]) < -tol:
            return True
    return False


def classify_graph(n_nodes, edges):
    """
    Classify graph for strategy selection.
    Returns dict with structural features.
    """
    n = n_nodes
    m = len(edges)

    info = {
        'n_nodes': n, 'n_edges': m,
        'density': 2.0 * m / (n * (n - 1)) if n > 1 else 0,
    }

    # Degree analysis
    degree = np.zeros(n, dtype=np.int32)
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)

    info['max_degree'] = int(degree.max()) if n > 0 else 0
    info['min_degree'] = int(degree.min()) if n > 0 else 0
    info['avg_degree'] = float(degree.mean()) if n > 0 else 0
    info['degree_std'] = float(degree.std()) if n > 0 else 0.0
    info['is_regular'] = info['max_degree'] == info['min_degree']
    info['leaf_fraction'] = float(np.mean(degree <= 1)) if n > 0 else 0.0
    hub_threshold = max(8, int(np.ceil(info['avg_degree'] + info['degree_std'])))
    info['hub_fraction'] = float(np.mean(degree >= hub_threshold)) if n > 0 else 0.0

    # Bipartiteness (BFS 2-coloring)
    color = [-1] * n
    is_bipartite = True
    components = 0
    for start in range(n):
        if color[start] != -1:
            continue
        components += 1
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
            if not is_bipartite:
                break
        if not is_bipartite:
            break
    info['is_bipartite'] = is_bipartite
    info['n_components'] = components
    info['cycle_rank'] = max(0, m - n + components)

    # Grid detection
    grid = _detect_grid(n, m, degree, adj)
    info['is_grid'] = grid is not None
    info['grid_dims'] = grid

    # Planarity heuristic (for Pfaffian): planar if m <= 3n - 6
    info['possibly_planar'] = m <= 3 * n - 6 if n >= 3 else True

    # Sparsity classes
    info['is_sparse'] = info['density'] < 0.05
    info['is_dense'] = info['density'] > 0.3

    # Unweighted-check: cograph-DP is alleen correct voor unit-weights.
    info['is_unweighted'] = all(
        len(e) == 2 or abs(float(e[2]) - 1.0) < 1e-9 for e in edges
    )

    # Signed-check (Dag 8 correctheids-vangnet): Pfaffian Oracle's bipartite-
    # en grid-branches gaan uit van *max* = sum-of-all-weights; voor signed
    # instanties (BiqMac-spinglass, random ±1) is dat aantoonbaar fout.
    # has_signed_edges() wordt door `select_strategy` gebruikt om pfaffian-
    # based exact routes uit te sluiten, en door `certify_result` als
    # defense-in-depth om een falselijk EXACT-certificaat te downgraden.
    info['has_signed_edges'] = has_signed_edges(edges)

    # B170 twin-width features (cograph-detectie + tww difficulty-metric).
    # None indien graaf te groot voor budget (zie _compute_tww_feature).
    tww_feat = _compute_tww_feature(n, edges)
    info['is_cograph'] = tww_feat['is_cograph']
    info['tww'] = tww_feat['tww']

    return info


def _detect_grid(n, m, degree, adj):
    """Detect if graph is a rectangular grid. Returns (Lx, Ly) or None."""
    for Ly in range(1, int(np.sqrt(n)) + 2):
        if n % Ly != 0:
            continue
        Lx = n // Ly
        if Lx < Ly:
            break
        expected_m = Lx * (Ly - 1) + (Lx - 1) * Ly
        if m != expected_m:
            continue
        if Lx == 1 or Ly == 1:
            if Lx > 1:
                d_counts = np.bincount(degree, minlength=3)
                if d_counts[1] == 2 and d_counts[2] == max(0, Lx - 2):
                    return (Lx, 1)
            continue
        d_counts = np.bincount(degree, minlength=5)
        expected_corners = 4
        expected_border = 2 * (Lx - 2) + 2 * (Ly - 2)
        expected_interior = (Lx - 2) * (Ly - 2)
        if (d_counts[2] == expected_corners and
            d_counts[3] == expected_border and
            d_counts[4] == expected_interior):
            return (Lx, Ly)
    return None


# ============================================================
# Strategy selection
# ============================================================

def select_strategy(info, time_budget=None, gpu=False, prefer_exact=True,
                    prefer_quantum=True, reorder='auto'):
    """
    Select best strategy based on graph classification.

    Returns (strategy_name, tier, solver_pipeline).
    Pipeline is a list of (solver_name, config) tuples to run in sequence.
    Later stages can use results from earlier stages as warm-start.
    """
    n = info['n_nodes']
    m = info['n_edges']

    # ---- TIER 1: EXACT ----

    # B131-Dag-8: signed instanties (negatieve edge-gewichten) worden NIET
    # naar pfaffian-based routes gerouteerd. pfaffian_maxcut's bipartite-
    # en grid-short-circuits nemen `cut = sum(weights)` aan, wat aantoonbaar
    # fout is op signed instanties (zie B186 selector failure-modes op
    # spinglass2d_L4/L5 en torus2d/pm1s). Signed instanties vallen door
    # naar FW-SDP/classical tier-2/3; het signed-brute-force pad
    # (`exact_small_signed`) is een expliciete safe-vector voor n<=20.
    is_signed = info.get('has_signed_edges', False)

    # Pfaffian: restrict to clear grid-like planar cases.
    # The coarse m <= 3n-6 heuristic is not strong enough to certify general
    # planarity, and otherwise large sparse Gset instances can be misrouted.
    if (prefer_exact and info['is_bipartite'] and info['is_grid']
            and info['possibly_planar'] and n <= 10000
            and not is_signed):
        return ('pfaffian_exact', 'exact',
                [('pfaffian', {'n': n})])

    # Cograph DP (B170): O(n^3) exact voor unweighted cographs.
    # Wint van brute-force zodra n > 20 en wint van QAOA/classical op elk
    # cograph omdat het provably-optimal is. Alleen voor unit-weights
    # (signed instanties al uitgesloten via is_unweighted=False).
    if (prefer_exact and info.get('is_cograph') is True
            and info.get('is_unweighted', False) and n <= 2000):
        return ('cograph_dp', 'exact',
                [('cograph_dp', {'n': n})])

    # Signed small graphs: use actual weighted brute-force enumeration
    # (sign-aware), not pfaffian_maxcut's bipartite/grid short-circuits.
    if is_signed and prefer_exact and n <= 20:
        return ('exact_small_signed', 'exact',
                [('signed_brute_force', {'n': n})])

    # Brute-force: small graphs (unsigned only — signed valt hierboven uit).
    if n <= 25 and prefer_exact and not is_signed:
        if n <= 20:
            return ('exact_small', 'exact',
                    [('brute_force', {'n': n})])
        else:
            return ('exact_brute', 'exact',
                    [('brute_force', {'n': n})])

    # ---- TIER 2: QUANTUM (MPS-QAOA) ----

    # B131-Dag-8: signed grids (n>20, dus geen exact_small_signed) MOGEN
    # door QAOA worden aangepakt, maar de huidige Heisenberg/RSVD-QAOA-stubs
    # zijn niet sign-aware getest op BiqMac-spinglass. Route signed grids
    # direct naar classical PA (tier 3), die aantoonbaar correct signed
    # MaxCut evalueert via bls_solver/pa_solver's XOR-cut berekening.
    if (prefer_quantum and info['is_grid'] and info['grid_dims'] is not None
            and not is_signed):
        Lx, Ly = info['grid_dims']
        if Ly <= 4:
            # MPS-QAOA is excellent on narrow grids
            return ('mps_qaoa_grid', 'quantum',
                    [('qaoa', {'type': 'heisenberg', 'Lx': Lx, 'Ly': Ly}),
                     ('pa_polish', {'n': n})])
        elif Ly <= 8:
            return ('mps_qaoa_wide', 'quantum',
                    [('qaoa', {'type': 'rsvd', 'Lx': Lx, 'Ly': Ly}),
                     ('pa_polish', {'n': n})])

    if (prefer_quantum and info['is_sparse'] and n <= 500
            and info['max_degree'] <= 10 and not is_signed):
        # Try QAOA with classical polish; QAOA stub falls back to PA if unavailable
        return ('lightcone_qaoa', 'quantum',
                [('qaoa', {'type': 'lightcone', 'reorder': reorder}),
                 ('bls_polish', {'n': n})])

    # ---- TIER 3: CLASSICAL ----

    # Small-medium: PA is best quality
    if n <= 500:
        return ('pa_primary', 'classical',
                [('pa', {'heavy': True}),
                 ('bls_polish', {'n': n})])

    # Medium: combined BLS+PA
    if n <= 2000:
        return ('combined_medium', 'classical',
                [('combined', {'heavy': True})])

    # Large sparse with GPU: sparse-specialized PA route
    if n > 2000 and gpu and info['is_sparse']:
        return ('cuda_sparse_large', 'classical',
                [('pa_sparse_hybrid', {}),
                 ('cuda_bls_polish', {})])

    # Large with GPU: CUDA solvers
    if n > 2000 and gpu:
        return ('cuda_large', 'classical',
                [('cuda_pa', {}),
                 ('cuda_bls_polish', {})])

    # Large without GPU: PA + BLS combined
    if n > 2000:
        return ('classical_large', 'classical',
                [('pa', {'heavy': True, 'time_share': 0.6}),
                 ('bls', {'heavy': True, 'time_share': 0.4})])

    # Fallback: BLS always works
    return ('bls_fallback', 'classical',
            [('bls', {'heavy': False})])


# ============================================================
# Solver wrappers
# ============================================================

def _warm_start_array(n_nodes, warm_start):
    """Convert cached assignment dict to a dense 0/1 numpy vector."""
    if not warm_start or not warm_start.get('assignment'):
        return None
    x_init = np.zeros(n_nodes, dtype=np.int32)
    for v, s in warm_start['assignment'].items():
        x_init[int(v)] = int(s)
    return x_init


def _run_pfaffian(n_nodes, edges, config, time_budget=None, seed=42,
                  warm_start=None):
    """Run Pfaffian Oracle (exact for bipartite planar, unsigned only)."""
    from pfaffian_oracle import pfaffian_maxcut
    t0 = time.time()
    # Zie _run_brute_force: pfaffian_maxcut is niet sign-aware op bipartite/
    # grid short-circuits. Signed instanties worden door select_strategy al
    # uitgesloten; deze check is defense-in-depth voor directe aanroepen.
    if has_signed_edges(edges):
        raise ValueError(
            "_run_pfaffian ontvangt signed edges; pfaffian_maxcut's "
            "bipartite/grid short-circuits retourneren sum(weights) en "
            "zijn niet correct voor negatieve gewichten. Gebruik "
            "_run_signed_brute_force voor n<=20 of FW-SDP anders."
        )
    r = pfaffian_maxcut(n_nodes, edges)
    return {
        'best_cut': r['best_cut'],
        'assignment': r.get('assignment', {}),
        'is_exact': True,
        'solver': 'pfaffian',
        'time_s': time.time() - t0,
    }


def _run_brute_force(n_nodes, edges, config, time_budget=None, seed=42,
                     warm_start=None):
    """Run brute-force / Pfaffian oracle for small graphs (unsigned only)."""
    from pfaffian_oracle import pfaffian_maxcut
    t0 = time.time()
    # Safety-net: pfaffian_maxcut's bipartite/grid branches zijn fout op
    # signed instanties. De dispatcher mag deze solver niet selecteren voor
    # signed edges (zie select_strategy), maar als iemand hem direct
    # aanroept, signaleren we dat hier expliciet.
    if has_signed_edges(edges):
        raise ValueError(
            "_run_brute_force ontvangt signed edges; route via "
            "_run_signed_brute_force. pfaffian_maxcut's bipartite/grid "
            "short-circuits zijn niet sign-aware."
        )
    r = pfaffian_maxcut(n_nodes, edges)
    return {
        'best_cut': r['best_cut'],
        'assignment': r.get('assignment', {}),
        'is_exact': True,
        'solver': 'brute_force',
        'time_s': time.time() - t0,
    }


def _run_signed_brute_force(n_nodes, edges, config, time_budget=None,
                            seed=42, warm_start=None):
    """Exacte enumeratie van MaxCut voor signed instanties (n<=20).

    Bijpassend complement voor `_run_brute_force`: loopt over alle 2^n
    bit-strings en berekent sum(w_e * (x_u XOR x_v)) per cut. Werkt
    correct onafhankelijk van edge-teken.

    Dispatcher routet hierheen wanneer `info['has_signed_edges']=True`
    en n<=20 (zie select_strategy → 'exact_small_signed'). De assignment
    wordt teruggerapporteerd als dict {node: 0/1}.
    """
    t0 = time.time()
    if n_nodes > 24:
        raise ValueError(
            f"_run_signed_brute_force: n={n_nodes} te groot voor exacte "
            "enumeratie (limiet 24)."
        )
    ei = np.array([int(e[0]) for e in edges], dtype=np.int32)
    ej = np.array([int(e[1]) for e in edges], dtype=np.int32)
    ew = np.array(
        [float(e[2]) if len(e) > 2 else 1.0 for e in edges],
        dtype=np.float64,
    )
    N = 1 << n_nodes
    xs = np.arange(N, dtype=np.int64)
    bi = (xs[:, None] >> ei[None, :]) & 1
    bj = (xs[:, None] >> ej[None, :]) & 1
    cuts = np.sum(ew[None, :] * (bi ^ bj), axis=1)
    best_idx = int(np.argmax(cuts))
    best_cut = float(cuts[best_idx])
    assignment = {i: int((best_idx >> i) & 1) for i in range(n_nodes)}
    return {
        'best_cut': best_cut,
        'assignment': assignment,
        'is_exact': True,
        'solver': 'signed_brute_force',
        'time_s': time.time() - t0,
    }


def _run_cograph_dp(n_nodes, edges, config, time_budget=None, seed=42,
                    warm_start=None):
    """Exact O(n^3) MaxCut op cographs via B170 cotree-DP.

    Alleen correct voor unit-weight grafen; dispatcher check dit vooraf
    via info['is_unweighted'] + info['is_cograph']. Als de routing-rule
    faalt (bv. niet-cograph in de praktijk), valt deze solver terug op
    een ValueError die de pipeline opvangt.
    """
    from b170_twin_width import cograph_maxcut_exact
    t0 = time.time()
    uw_edges = [(int(e[0]), int(e[1])) for e in edges]
    r = cograph_maxcut_exact(n_nodes, uw_edges)
    assignment = {int(v): int(s) for v, s in r['partition'].items()}
    # Verify weighted cut (voor defense-in-depth: als iemand toch gewichten
    # doorgeeft, klopt de cut-waarde nog met de werkelijke assignment).
    cut = 0.0
    for e in edges:
        u, v = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) > 2 else 1.0
        if assignment.get(u, 0) != assignment.get(v, 0):
            cut += w
    return {
        'best_cut': cut,
        'assignment': assignment,
        'is_exact': True,
        'solver': 'cograph_dp',
        'time_s': time.time() - t0,
    }


def _run_bls(n_nodes, edges, config, time_budget=None, seed=42,
             warm_start=None):
    """Run BLS solver."""
    from bls_solver import bls_maxcut
    heavy = config.get('heavy', False)
    if heavy:
        restarts = max(10, min(50, 5000 // max(n_nodes, 1)))
        max_iter = 2000
        max_no_improve = 200
    else:
        restarts = 5
        max_iter = 500
        max_no_improve = 50

    tl = time_budget
    if config.get('time_share') and time_budget:
        tl = time_budget * config['time_share']

    x_init = _warm_start_array(n_nodes, warm_start)
    r = bls_maxcut(n_nodes, edges, n_restarts=restarts, max_iter=max_iter,
                   max_no_improve=max_no_improve, time_limit=tl, seed=seed,
                   x_init=x_init)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'bls',
        'time_s': r['time_s'],
    }


def _run_pa(n_nodes, edges, config, time_budget=None, seed=42,
            warm_start=None):
    """Run Population Annealing solver."""
    from pa_solver import pa_maxcut
    heavy = config.get('heavy', False)
    if heavy:
        replicas = min(500, max(100, 50000 // max(n_nodes, 1)))
        n_temps = 60
        n_sweeps = 5
    else:
        replicas = 100
        n_temps = 30
        n_sweeps = 3

    tl = time_budget
    if config.get('time_share') and time_budget:
        tl = time_budget * config['time_share']

    x_init = _warm_start_array(n_nodes, warm_start)
    r = pa_maxcut(n_nodes, edges, n_replicas=replicas, n_temps=n_temps,
                  n_sweeps=n_sweeps, time_limit=tl, seed=seed, x_init=x_init)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'pa',
        'time_s': r['time_s'],
    }


def _run_combined(n_nodes, edges, config, time_budget=None, seed=42,
                  warm_start=None):
    """Run best-of BLS + PA."""
    t_half = time_budget / 2.0 if time_budget else None
    r_bls = _run_bls(n_nodes, edges, {'heavy': True}, t_half, seed,
                     warm_start=warm_start)
    r_pa = _run_pa(n_nodes, edges, {'heavy': True}, t_half,
                   seed + 1000 if seed else None, warm_start=warm_start)
    if r_pa['best_cut'] >= r_bls['best_cut']:
        r_pa['time_s'] += r_bls['time_s']
        r_pa['solver'] = 'combined(pa_won)'
        return r_pa
    else:
        r_bls['time_s'] += r_pa['time_s']
        r_bls['solver'] = 'combined(bls_won)'
        return r_bls


def _run_cuda_bls(n_nodes, edges, config, time_budget=None, seed=42,
                  warm_start=None):
    """Run CUDA BLS."""
    from cuda_local_search import maxcut_bls
    restarts = max(10, min(50, 5000 // max(n_nodes, 1)))
    r = maxcut_bls(n_nodes, edges, n_restarts=restarts, max_iter=2000,
                   max_no_improve=200, time_limit=time_budget, seed=seed)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'cuda_bls',
        'time_s': r['time_s'],
    }


def _run_cuda_pa(n_nodes, edges, config, time_budget=None, seed=42,
                 warm_start=None):
    """Run CUDA PA."""
    from cuda_local_search import maxcut_pa
    replicas = min(500, max(100, 50000 // max(n_nodes, 1)))
    r = maxcut_pa(n_nodes, edges, n_replicas=replicas, n_temps=60,
                  n_sweeps=5, time_limit=time_budget, seed=seed)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'cuda_pa',
        'time_s': r['time_s'],
    }


def _run_pa_sparse_hybrid(n_nodes, edges, config, time_budget=None, seed=42,
                          warm_start=None):
    """Run sparse-specialized PA hybrid."""
    from cuda_local_search import maxcut_pa_sparse_hybrid
    replicas = min(500, max(100, 50000 // max(n_nodes, 1)))
    r = maxcut_pa_sparse_hybrid(n_nodes, edges, n_replicas=replicas, n_temps=60,
                                n_sweeps=5, time_limit=time_budget, seed=seed)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'pa_sparse_hybrid',
        'time_s': r['time_s'],
    }


def _run_pa_polish(n_nodes, edges, config, time_budget=None, seed=42,
                   warm_start=None):
    """PA used as polish after QAOA — warm-starts from QAOA assignment."""
    from pa_solver import pa_maxcut
    r = pa_maxcut(n_nodes, edges, n_replicas=50, n_temps=20,
                  n_sweeps=3, time_limit=time_budget, seed=seed)
    # If warm-start assignment given, also try BLS from that
    if warm_start and warm_start.get('assignment'):
        from bls_solver import bls_maxcut
        x_init = np.zeros(n_nodes, dtype=np.int32)
        for v, s in warm_start['assignment'].items():
            x_init[int(v)] = int(s)
        edge_list = [(int(e[0]), int(e[1]),
                      float(e[2]) if len(e) > 2 else 1.0) for e in edges]
        r2 = bls_maxcut(n_nodes, edge_list, n_restarts=3, max_iter=500,
                        x_init=x_init, time_limit=time_budget, seed=seed)
        if r2['best_cut'] > r['best_cut']:
            return {
                'best_cut': r2['best_cut'],
                'assignment': r2['assignment'],
                'is_exact': False,
                'solver': 'pa_polish+bls_warm',
                'time_s': r['time_s'] + r2['time_s'],
            }
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'pa_polish',
        'time_s': r['time_s'],
    }


def _run_bls_polish(n_nodes, edges, config, time_budget=None, seed=42,
                    warm_start=None):
    """BLS polish, optionally warm-started."""
    from bls_solver import bls_maxcut
    x_init = None
    if warm_start and warm_start.get('assignment'):
        x_init = np.zeros(n_nodes, dtype=np.int32)
        for v, s in warm_start['assignment'].items():
            x_init[int(v)] = int(s)
    edge_list = [(int(e[0]), int(e[1]),
                  float(e[2]) if len(e) > 2 else 1.0) for e in edges]
    r = bls_maxcut(n_nodes, edge_list, n_restarts=5, max_iter=500,
                   x_init=x_init, time_limit=time_budget, seed=seed)
    return {
        'best_cut': r['best_cut'],
        'assignment': r['assignment'],
        'is_exact': False,
        'solver': 'bls_polish',
        'time_s': r['time_s'],
    }


def _run_qaoa_stub(n_nodes, edges, config, time_budget=None, seed=42,
                   warm_start=None):
    """
    QAOA stub — delegates to auto_planner.py ZornSolver for actual
    MPS-QAOA execution. Returns result or falls back to PA.
    """
    try:
        from auto_planner import ZornSolver, SolverResult
        reorder = config.get('reorder', 'auto')
        solver = ZornSolver(chi_budget=32, gpu=False,
                            reorder=reorder, verbose=False)
        edge_tuples = [(int(e[0]), int(e[1])) for e in edges]
        sr = solver.solve(n_nodes, edge_tuples, p=1, reorder=reorder)
        assignment = {}
        if sr.best_bitstring is not None:
            for i in range(n_nodes):
                assignment[i] = int(sr.best_bitstring[i])
        return {
            'best_cut': sr.cut_value,
            'assignment': assignment,
            'is_exact': False,
            'solver': f'qaoa_{sr.engine}',
            'time_s': sr.wall_time,
        }
    except Exception as e:
        # QAOA not available or failed — fall back to PA
        return _run_pa(n_nodes, edges, {'heavy': False}, time_budget, seed)


# Solver dispatch table
SOLVER_FUNCS = {
    'pfaffian': _run_pfaffian,
    'brute_force': _run_brute_force,
    'signed_brute_force': _run_signed_brute_force,
    'cograph_dp': _run_cograph_dp,
    'bls': _run_bls,
    'pa': _run_pa,
    'combined': _run_combined,
    'cuda_bls': _run_cuda_bls,
    'cuda_pa': _run_cuda_pa,
    'pa_sparse_hybrid': _run_pa_sparse_hybrid,
    'pa_polish': _run_pa_polish,
    'bls_polish': _run_bls_polish,
    'cuda_bls_polish': _run_bls_polish,  # alias
    'qaoa': _run_qaoa_stub,
}


# ============================================================
# Quality certification
# ============================================================

def certify_result(best_cut, n_nodes, edges, info, is_exact, strategy=None):
    """
    Assign quality certificate based on result and known bounds.

    Certificates:
        EXACT       — provably optimal
        NEAR_EXACT  — within 1% of known bound
        GOOD        — within 5% of known bound
        APPROXIMATE — best effort, no strong guarantee

    B131-Dag-8 defense-in-depth: als `is_exact=True` afkomt uit een
    pfaffian-based route (pfaffian_exact / exact_small / exact_brute)
    op een signed instance, downgrade naar APPROXIMATE. pfaffian_maxcut
    is niet sign-aware op bipartite/grid-branches; de dispatcher routet
    signed instanties tegenwoordig al om (naar `exact_small_signed` voor
    n<=20 of FW-SDP anders), maar dit is een tweede lijn tegen
    regressies.
    """
    unsafe_for_signed = {
        'pfaffian_exact', 'exact_small', 'exact_brute'
    }
    if (is_exact and strategy in unsafe_for_signed
            and info.get('has_signed_edges', False)):
        return 'APPROXIMATE'
    if is_exact:
        return 'EXACT'

    m = len(edges)
    # Upper bound: total weight (trivial), or GW 0.878 approx ratio
    total_w = sum(float(e[2]) if len(e) > 2 else 1.0 for e in edges)

    # For bipartite: exact upper bound is total weight
    if info.get('is_bipartite'):
        if abs(best_cut - total_w) < 0.5:
            return 'EXACT'
        gap = (total_w - best_cut) / total_w
        if gap < 0.01:
            return 'NEAR_EXACT'
        elif gap < 0.05:
            return 'GOOD'

    # For general: use ratio to total edges
    ratio = best_cut / total_w if total_w > 0 else 0

    if ratio >= 0.99:
        return 'NEAR_EXACT'
    elif ratio >= 0.90:
        return 'GOOD'
    else:
        return 'APPROXIMATE'


# ============================================================
# Main dispatcher
# ============================================================

class ZornDispatcher:
    """
    Unified MaxCut compute engine.

    Automatically selects the best strategy for each problem instance.
    Combines exact methods, quantum simulation (MPS-QAOA), and
    classical heuristics (BLS, PA) in an optimal pipeline.
    """

    def __init__(self, gpu=False, time_budget=None, prefer_exact=True,
                 prefer_quantum=True, seed=42, reorder='auto',
                 verbose=True, enable_bandit=False,
                 bandit_ucb_c=0.8, bandit_eps=0.05,
                 bandit_scope='all'):
        self.gpu = gpu
        self.time_budget = time_budget
        self.prefer_exact = prefer_exact
        self.prefer_quantum = prefer_quantum
        self.seed = seed
        self.reorder = reorder
        self.verbose = verbose
        self.enable_bandit = enable_bandit
        self.bandit_scope = bandit_scope
        self.bandit = None
        if enable_bandit:
            self.bandit = DispatcherBandit(
                seed=seed,
                ucb_c=bandit_ucb_c,
                eps=bandit_eps,
            )

    def _bandit_scope_allows(self, info):
        """Restrict bandit activation to requested structural pockets."""
        if self.bandit_scope in (None, '', 'all'):
            return True
        if self.bandit_scope == 'hub2000':
            return (
                1500 <= info['n_nodes'] <= 2500 and
                10000 <= info['n_edges'] <= 13000 and
                not info.get('is_grid', False)
            )
        if self.bandit_scope == 'medium':
            return 500 < info['n_nodes'] <= 2000 and not info.get('is_grid', False)
        return False

    def _bandit_candidates(self, info, strategy, tier, pipeline):
        """Return optional portfolio arms for ambiguous classical families."""
        if (not self.enable_bandit or self.bandit is None or tier != 'classical'
                or info.get('is_grid') or info['n_nodes'] <= 500
                or not self._bandit_scope_allows(info)):
            return None

        n = info['n_nodes']
        if n <= 2000:
            return {
                'combined': ('combined_medium', tier, [('combined', {
                    'heavy': True, 'use_warm_start': True,
                })]),
                'pa_then_bls': ('bandit_pa_then_bls', tier, [
                    ('pa', {'heavy': True, 'use_warm_start': True}),
                    ('bls_polish', {'n': n}),
                ]),
                'bls_then_pa': ('bandit_bls_then_pa', tier, [
                    ('bls', {'heavy': True, 'use_warm_start': True}),
                    ('pa_polish', {'n': n}),
                ]),
            }

        if self.gpu:
            return None

        return {
            'classical_large': ('classical_large', tier, [
                ('pa', {
                    'heavy': True,
                    'time_share': 0.6,
                    'use_warm_start': True,
                }),
                ('bls', {
                    'heavy': True,
                    'time_share': 0.4,
                    'use_previous_stage': True,
                }),
            ]),
            'combined_large': ('combined_large_bandit', tier, [
                ('combined', {'heavy': True, 'use_warm_start': True}),
            ]),
        }

    def solve(self, n_nodes, edges, time_budget=None):
        """
        Solve MaxCut for the given graph.

        Args:
            n_nodes: number of nodes
            edges: list of (i, j) or (i, j, w)
            time_budget: override instance time budget (seconds)

        Returns:
            DispatchResult with best solution and metadata
        """
        t0 = time.time()
        tb = time_budget or self.time_budget

        # 1. Classify
        info = classify_graph(n_nodes, edges)
        if self.verbose:
            bp = "bipartite" if info['is_bipartite'] else "non-bipartite"
            grid = f" grid {info['grid_dims']}" if info['is_grid'] else ""
            plan = f" planar?" if info['possibly_planar'] else ""
            print(f"[Dispatcher] n={n_nodes}, m={len(edges)}, "
                  f"d_avg={info['avg_degree']:.1f}, {bp}{grid}{plan}")

        # 2. Select strategy
        strategy, tier, pipeline = select_strategy(
            info, tb, self.gpu, self.prefer_exact,
            self.prefer_quantum, reorder=self.reorder)

        bandit_arm = None
        bandit_family = None
        bandit_warm_start = None
        notes = []
        bandit_arms = self._bandit_candidates(info, strategy, tier, pipeline)
        if bandit_arms:
            bandit_family = family_key_from_info(info)
            arm_names = sorted(bandit_arms.keys())
            bandit_arm = self.bandit.choose(bandit_family, arm_names)
            strategy, tier, pipeline = bandit_arms[bandit_arm]
            bandit_warm_start = self.bandit.get_warm_start(bandit_family)
            notes.extend([
                f'bandit_family={bandit_family}',
                f'bandit_arm={bandit_arm}',
            ])
            notes.append(
                'bandit_warm_cache=hit' if bandit_warm_start else
                'bandit_warm_cache=miss'
            )

        if self.verbose:
            solvers = '+'.join(s for s, _ in pipeline)
            print(f"[Dispatcher] Strategy: {strategy} (tier={tier}, pipeline={solvers})")

        # 3. Execute pipeline
        best_result = None
        solvers_used = []
        total_time = 0.0

        for solver_name, config in pipeline:
            func = SOLVER_FUNCS.get(solver_name)
            if func is None:
                if self.verbose:
                    print(f"[Dispatcher] Warning: unknown solver '{solver_name}', skipping")
                continue

            # Time budget for this stage
            stage_budget = None
            if tb:
                remaining = tb - total_time
                if remaining <= 0:
                    break
                stage_budget = remaining

            try:
                warm_start = None
                if best_result and ('polish' in solver_name or
                                    config.get('use_previous_stage')):
                    warm_start = best_result
                elif (bandit_warm_start is not None and
                      config.get('use_warm_start')):
                    warm_start = bandit_warm_start

                r = func(n_nodes, edges, config, stage_budget, self.seed,
                         warm_start=warm_start)

                solvers_used.append(r.get('solver', solver_name))
                total_time += r.get('time_s', 0)

                # Keep best result across pipeline stages
                if best_result is None or r['best_cut'] > best_result['best_cut']:
                    best_result = r
                elif r['is_exact']:
                    best_result = r  # exact always wins

                if self.verbose:
                    exact_str = " [EXACT]" if r['is_exact'] else ""
                    print(f"[Dispatcher] {r['solver']}: cut={r['best_cut']:.0f}{exact_str} "
                          f"({r['time_s']:.2f}s)")

            except Exception as e:
                if self.verbose:
                    print(f"[Dispatcher] {solver_name} failed: {e}")
                solvers_used.append(f'{solver_name}(FAILED)')

        if best_result is None:
            # Emergency fallback: basic BLS
            if self.verbose:
                print("[Dispatcher] All solvers failed, using BLS fallback")
            best_result = _run_bls(n_nodes, edges, {'heavy': False}, tb, self.seed)
            solvers_used.append('bls_fallback')
            total_time += best_result['time_s']

        # 4. Certify
        cert = certify_result(best_result['best_cut'], n_nodes, edges,
                              info, best_result.get('is_exact', False),
                              strategy=strategy)

        # 5. Build result
        m = len(edges)
        m = len(edges)
        total_w = sum(float(e[2]) if len(e) > 2 else 1.0 for e in edges)

        graph_info = dict(info)
        if bandit_family:
            graph_info['bandit_family'] = bandit_family
            graph_info['bandit_arm'] = bandit_arm
            graph_info['bandit_stats'] = self.bandit.get_stats(bandit_family)

        result = DispatchResult(
            best_cut=best_result['best_cut'],
            assignment=best_result.get('assignment', {}),
            ratio=best_result['best_cut'] / total_w if total_w > 0 else 0,
            is_exact=best_result.get('is_exact', False),
            strategy=strategy,
            tier=tier,
            solvers_used=solvers_used,
            time_s=time.time() - t0,
            graph_info=graph_info,
            certificate=cert,
            notes=notes,
        )

        if bandit_family:
            reward = reward_from_dispatch_result(result)
            self.bandit.update(bandit_family, bandit_arm, reward)
            remembered = self.bandit.remember_result(bandit_family, result)
            result.notes.append(f'bandit_reward={reward:.4f}')
            result.notes.append(
                'bandit_memory=updated' if remembered else 'bandit_memory=kept'
            )
            result.graph_info['bandit_stats'] = self.bandit.get_stats(bandit_family)

        if self.verbose:
            print(f"[Dispatcher] Result: {result.summary()}")

        return result


def solve_maxcut(n_nodes, edges, gpu=False, time_budget=None,
                 seed=42, reorder='auto', verbose=False,
                 enable_bandit=False, bandit_scope='all'):
    """One-line MaxCut solver. Auto-selects best strategy."""
    d = ZornDispatcher(gpu=gpu, time_budget=time_budget,
                       seed=seed, reorder=reorder, verbose=verbose,
                       enable_bandit=enable_bandit,
                       bandit_scope=bandit_scope)
    return d.solve(n_nodes, edges)


if __name__ == '__main__':
    print("=== B130 Auto-Dispatcher Demo ===")
    d = ZornDispatcher(verbose=True)
    k5 = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
    d.solve(5, k5)
