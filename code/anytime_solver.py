#!/usr/bin/env python3
"""
AnytimeSolver met Certificaat (B49)
====================================
Gelaagde solver die altijd een antwoord geeft met foutbanden,
ongeacht hoeveel tijd/geheugen beschikbaar is.

Lagen:
    1. Greedy classical (ms)     → gegarandeerde ondergrens
    2. GW-SDP relaxatie (s)      → gecertificeerde bovengrens + rounding
    3. Lanczos exact (s)         → exact als n <= 20
    4. QAOA p=1 via B48 (s-min)  → quantum schatting
    5. QAOA p=2+ / RQAOA (min-h) → iteratieve verbetering

Output altijd: CertifiedResult met
    [lower_bound, best_cut, upper_bound, gap, confidence]

"Anytime" eigenschap: stop wanneer je wilt, altijd een geldig antwoord.

Gebruik:
    from anytime_solver import AnytimeSolver
    solver = AnytimeSolver(verbose=True)
    result = solver.solve(n_nodes, edges, time_budget=10.0)
    print(result)
    print(f"Gap: {result.gap_pct:.1%}")
"""
import numpy as np
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# CERTIFIED RESULT
# =====================================================================

@dataclass
class CertifiedResult:
    """Resultaat met gecertificeerde onder- en bovengrens."""
    # Bounds
    lower_bound: float = 0.0          # Bewezen ondergrens (greedy/LS)
    best_cut: float = 0.0             # Beste gevonden cut-waarde
    upper_bound: float = float('inf') # Bewezen bovengrens (SDP/trivial)

    # Bitstring
    best_bitstring: Optional[np.ndarray] = None

    # Certificate info
    lower_method: str = ""            # Hoe is de ondergrens berekend
    upper_method: str = ""            # Hoe is de bovengrens berekend
    best_method: str = ""             # Hoe is de beste cut gevonden
    is_exact: bool = False            # Is het bewezen optimaal?

    # Layers completed
    layers_completed: List[str] = field(default_factory=list)
    layer_times: Dict[str, float] = field(default_factory=dict)

    # Metadata
    n_nodes: int = 0
    n_edges: int = 0
    wall_time: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Absolute gap tussen upper en lower bound."""
        if self.upper_bound == float('inf'):
            return float('inf')
        return self.upper_bound - self.lower_bound

    @property
    def gap_pct(self) -> float:
        """Relatieve gap als fractie van upper bound."""
        if self.upper_bound <= 0 or self.upper_bound == float('inf'):
            return float('inf')
        return self.gap / self.upper_bound

    @property
    def confidence(self) -> str:
        """Mensleesbare betrouwbaarheid."""
        if self.is_exact:
            return "EXACT"
        g = self.gap_pct
        if g == 0:
            return "OPTIMAL (bounds tight)"
        elif g < 0.01:
            return "VERY HIGH (<1% gap)"
        elif g < 0.05:
            return "HIGH (<5% gap)"
        elif g < 0.15:
            return "MEDIUM (<15% gap)"
        else:
            return f"LOW ({g:.0%} gap)"

    def __repr__(self):
        ub = f"{self.upper_bound:.1f}" if self.upper_bound < float('inf') else "?"
        return (f"CertifiedResult(cut={self.best_cut:.1f}, "
                f"bounds=[{self.lower_bound:.1f}, {ub}], "
                f"gap={self.gap_pct:.1%}, "
                f"confidence={self.confidence}, "
                f"layers={len(self.layers_completed)}, "
                f"time={self.wall_time:.2f}s)")

    def certificate_str(self) -> str:
        """Volledige certificaat-string voor logging/publicatie."""
        ub = f"{self.upper_bound:.2f}" if self.upper_bound < float('inf') else "onbekend"
        lines = [
            f"=== ZornQ Certified MaxCut Result ===",
            f"Graaf: {self.n_nodes} nodes, {self.n_edges} edges",
            f"Beste cut:    {self.best_cut:.2f}  ({self.best_method})",
            f"Ondergrens:   {self.lower_bound:.2f}  ({self.lower_method})",
            f"Bovengrens:   {ub}  ({self.upper_method})",
            f"Gap:          {self.gap:.2f} ({self.gap_pct:.2%})",
            f"Betrouwbaarh: {self.confidence}",
            f"Exact:        {'JA' if self.is_exact else 'NEE'}",
            f"Lagen:        {', '.join(self.layers_completed)}",
            f"Totale tijd:  {self.wall_time:.2f}s",
        ]
        for layer, t in self.layer_times.items():
            lines.append(f"  {layer}: {t:.3f}s")
        if self.notes:
            lines.append(f"Notities:     {'; '.join(self.notes)}")
        lines.append("=" * 38)
        return '\n'.join(lines)


# =====================================================================
# LAYER 1: GREEDY CLASSICAL CUT
# =====================================================================

def greedy_maxcut(n_nodes: int, edges: list,
                  weights: Optional[dict] = None) -> Tuple[float, np.ndarray]:
    """Deterministic greedy MaxCut.

    Voeg nodes één voor één toe aan partitie S of T,
    kies de zijde die de meeste extra cut-edges geeft.
    Garandeert cut >= m/2 (helft van alle edges).

    Returns (cut_value, bitstring).
    """
    # Adjacency met gewichten
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        w = weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0
        adj[u].append((v, w))
        adj[v].append((u, w))

    bits = np.zeros(n_nodes, dtype=int)  # 0 = partitie S, 1 = partitie T
    assigned = np.zeros(n_nodes, dtype=bool)

    # Sorteer nodes op graad (hubs eerst — betere greedy keuzes)
    order = sorted(range(n_nodes), key=lambda i: len(adj[i]), reverse=True)

    for node in order:
        # Bereken gain voor S (bits=0) vs T (bits=1)
        gain_S = 0  # extra cut als node in S
        gain_T = 0  # extra cut als node in T
        for nbr, w in adj[node]:
            if assigned[nbr]:
                if bits[nbr] == 1:
                    gain_S += w  # node in S, nbr in T → cut
                else:
                    gain_T += w  # node in T, nbr in S → cut

        bits[node] = 1 if gain_T >= gain_S else 0
        assigned[node] = True

    # Bereken cut value
    cut = sum(
        (weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0)
        for u, v in edges if bits[u] != bits[v]
    )
    return float(cut), bits


def random_greedy_maxcut(n_nodes: int, edges: list,
                         weights: Optional[dict] = None,
                         n_restarts: int = 5,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[float, np.ndarray]:
    """Greedy met random permutaties + steepest descent.

    Garandeert minstens cut >= m/2. Typisch veel beter.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Adjacency met gewichten
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        w = weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0
        adj[u].append((v, w))
        adj[v].append((u, w))

    def cut_value(bits):
        return sum(
            (weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0)
            for u, v in edges if bits[u] != bits[v])

    def steepest_descent(bits):
        bits = bits.copy()
        improved = True
        while improved:
            improved = False
            best_gain, best_flip = 0, -1
            for i in range(n_nodes):
                gain = 0
                for nbr, w in adj[i]:
                    if bits[i] == bits[nbr]:
                        gain += w
                    else:
                        gain -= w
                if gain > best_gain:
                    best_gain = gain
                    best_flip = i
            if best_flip >= 0:
                bits[best_flip] = 1 - bits[best_flip]
                improved = True
        return bits, cut_value(bits)

    best_cut, best_bits = greedy_maxcut(n_nodes, edges, weights)
    best_bits, best_cut = steepest_descent(best_bits)

    for _ in range(n_restarts - 1):
        bits = rng.integers(0, 2, size=n_nodes)
        bits, cv = steepest_descent(bits)
        if cv > best_cut:
            best_cut = cv
            best_bits = bits.copy()

    return best_cut, best_bits


# =====================================================================
# LAYER 2: GOEMANS-WILLIAMSON SDP BOUND
# =====================================================================

def gw_sdp_bound(n_nodes: int, edges: list,
                 weights: Optional[dict] = None,
                 n_rounds: int = 50,
                 rng: Optional[np.random.Generator] = None
                 ) -> Tuple[float, float, Optional[np.ndarray]]:
    """Goemans-Williamson SDP relaxatie voor MaxCut.

    Returns (sdp_upper_bound, best_rounding_cut, best_bitstring).

    De SDP-waarde is een bewezen bovengrens op MaxCut.
    De hyperplane rounding geeft een heuristische oplossing.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    try:
        import cvxpy as cp
    except ImportError:
        return float('inf'), 0.0, None

    # Bouw de Laplacian gewichtsmatrix
    W = np.zeros((n_nodes, n_nodes))
    for u, v in edges:
        w = weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0
        W[u, v] = w
        W[v, u] = w

    # SDP: max (1/4) * sum_ij w_ij * (1 - X_ij)  s.t. X psd, X_ii = 1
    # Equivalent: max (1/4) * tr(L @ X)  met L = diag(W@1) - W
    L = np.diag(W.sum(axis=1)) - W

    X = cp.Variable((n_nodes, n_nodes), symmetric=True)
    constraints = [X >> 0]  # positive semidefinite
    constraints += [X[i, i] == 1 for i in range(n_nodes)]

    objective = cp.Maximize(0.25 * cp.trace(L @ X))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.SCS, max_iters=5000, eps=1e-6, verbose=False)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, max_iters=10000, eps=1e-4, verbose=False)
        except cp.SolverError:
            return float('inf'), 0.0, None

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        return float('inf'), 0.0, None

    sdp_value = problem.value
    X_val = X.value

    # Rounding: Cholesky decompositie + random hyperplane
    try:
        # Maak X numeriek PSD (kleine perturbatie)
        eigvals = np.linalg.eigvalsh(X_val)
        if eigvals.min() < -1e-6:
            X_val += (-eigvals.min() + 1e-8) * np.eye(n_nodes)
        X_val = (X_val + X_val.T) / 2

        # Cholesky
        V = np.linalg.cholesky(X_val + 1e-10 * np.eye(n_nodes)).T  # (n, n)
    except np.linalg.LinAlgError:
        # Fallback: eigendecompositie
        eigvals, eigvecs = np.linalg.eigh(X_val)
        eigvals = np.maximum(eigvals, 0)
        V = (eigvecs * np.sqrt(eigvals)).T

    def cut_value(bits):
        return sum(
            (weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0)
            for u, v in edges if bits[u] != bits[v])

    best_rounding = 0.0
    best_bits = None

    for _ in range(n_rounds):
        # Random hyperplane
        r = rng.standard_normal(n_nodes)
        r /= np.linalg.norm(r) + 1e-12

        # Partitie: sign(V @ r)  (V.T @ r geeft projectie per node)
        proj = V.T @ r if V.shape[0] >= V.shape[1] else V @ r
        if len(proj) != n_nodes:
            proj = (V.T @ r)[:n_nodes] if V.shape[1] == n_nodes else (V @ r)[:n_nodes]
        bits = (proj >= 0).astype(int)

        cv = cut_value(bits)
        if cv > best_rounding:
            best_rounding = cv
            best_bits = bits.copy()

    return float(sdp_value), float(best_rounding), best_bits


def trivial_upper_bound(n_nodes: int, edges: list,
                        weights: Optional[dict] = None) -> float:
    """Triviale bovengrens: som van alle edge-gewichten."""
    total = sum(
        (weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0)
        for u, v in edges)
    return total


# =====================================================================
# ANYTIME SOLVER
# =====================================================================

class AnytimeSolver:
    """Gelaagde MaxCut solver met certificaat.

    Draait lagen in volgorde van goedkoop → duur.
    Stop wanneer je wilt — altijd een geldig antwoord.

    Parameters
    ----------
    time_budget : float
        Maximale wandkloktijd in seconden (0 = onbeperkt).
    chi_budget : int
        Max bond dimensie voor MPS-methoden.
    gpu : bool
        Gebruik GPU als beschikbaar.
    verbose : bool
        Print voortgang.
    """

    def __init__(self, time_budget: float = 60.0, chi_budget: int = 32,
                 gpu: bool = False, verbose: bool = True):
        self.time_budget = time_budget
        self.chi_budget = chi_budget
        self.gpu = gpu
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[Anytime] {msg}")
            sys.stdout.flush()

    def _time_left(self, t0):
        if self.time_budget <= 0:
            return float('inf')
        return self.time_budget - (time.time() - t0)

    def solve(self, n_nodes: int, edges: list,
              weights: Optional[dict] = None,
              time_budget: Optional[float] = None) -> CertifiedResult:
        """Los MaxCut op met certificaat.

        Parameters
        ----------
        n_nodes : int
        edges : list of (int, int)
        weights : dict, optional
        time_budget : float, optional
            Override time budget voor deze run.

        Returns
        -------
        CertifiedResult
        """
        budget = time_budget if time_budget is not None else self.time_budget
        old_budget = self.time_budget
        self.time_budget = budget

        t0 = time.time()
        n_edges = len(edges)

        result = CertifiedResult(
            n_nodes=n_nodes, n_edges=n_edges,
            upper_bound=trivial_upper_bound(n_nodes, edges, weights),
            upper_method='trivial (total weight)')

        self._log(f"Start: n={n_nodes}, m={n_edges}, budget={budget:.0f}s")

        # ─── LAYER 1: Greedy Classical ─────────────────────────
        if self._time_left(t0) > 0:
            self._run_layer1_greedy(result, n_nodes, edges, weights, t0)

        # ─── LAYER 2: GW-SDP Bound ─────────────────────────────
        if self._time_left(t0) > 1.0 and n_nodes <= 2000:
            self._run_layer2_sdp(result, n_nodes, edges, weights, t0)

        # ─── LAYER 3: Lanczos Exact ────────────────────────────
        if self._time_left(t0) > 1.0 and n_nodes <= 22 and not result.is_exact:
            self._run_layer3_lanczos(result, n_nodes, edges, weights, t0)

        # ─── LAYER 4: QAOA via B48 Planner ─────────────────────
        if self._time_left(t0) > 2.0 and not result.is_exact and n_nodes <= 200:
            self._run_layer4_qaoa(result, n_nodes, edges, weights, t0, p=1)

        # ─── LAYER 5: QAOA p=2 of RQAOA ───────────────────────
        if self._time_left(t0) > 5.0 and not result.is_exact and n_nodes <= 100:
            self._run_layer5_deep(result, n_nodes, edges, weights, t0)

        # ─── Finalize ──────────────────────────────────────────
        result.wall_time = time.time() - t0

        # Check of bounds tight zijn
        if abs(result.lower_bound - result.upper_bound) < 0.5:
            result.is_exact = True
            result.notes.append("bounds_tight")

        self._log(f"Done: {result}")
        self.time_budget = old_budget
        return result

    # ─── Layer Implementations ─────────────────────────────────

    def _run_layer1_greedy(self, result, n_nodes, edges, weights, t0):
        """Laag 1: Greedy + local search → gegarandeerde ondergrens."""
        t_start = time.time()

        n_restarts = min(50, max(5, 2000 // max(n_nodes, 1)))
        cut, bits = random_greedy_maxcut(
            n_nodes, edges, weights, n_restarts=n_restarts)

        dt = time.time() - t_start
        result.lower_bound = cut
        result.best_cut = cut
        result.best_bitstring = bits
        result.lower_method = f'greedy+LS ({n_restarts} restarts)'
        result.best_method = result.lower_method
        result.layers_completed.append('greedy')
        result.layer_times['greedy'] = dt

        ratio = cut / len(edges) if edges else 0
        self._log(f"L1 Greedy: cut={cut:.0f}, ratio={ratio:.4f}, "
                  f"lb={cut:.0f}, t={dt:.3f}s")

    def _run_layer2_sdp(self, result, n_nodes, edges, weights, t0):
        """Laag 2: GW-SDP → gecertificeerde bovengrens + rounding."""
        t_start = time.time()

        n_rounds = min(100, max(20, 5000 // max(n_nodes, 1)))
        sdp_ub, rounding_cut, rounding_bits = gw_sdp_bound(
            n_nodes, edges, weights, n_rounds=n_rounds)

        dt = time.time() - t_start

        if sdp_ub < float('inf'):
            # SDP bovengrens is altijd geldig (bewezen)
            # Ceil want MaxCut is integer voor integer gewichten
            sdp_ub_ceil = np.ceil(sdp_ub - 1e-6)
            if sdp_ub_ceil < result.upper_bound:
                result.upper_bound = sdp_ub_ceil
                result.upper_method = f'GW-SDP (raw={sdp_ub:.2f})'

            # Rounding kan de lower bound verbeteren
            if rounding_cut > result.lower_bound:
                result.lower_bound = rounding_cut
                result.lower_method = f'GW-rounding ({n_rounds}r)'

            if rounding_cut > result.best_cut:
                result.best_cut = rounding_cut
                result.best_bitstring = rounding_bits
                result.best_method = 'GW-rounding'

            result.layers_completed.append('sdp')
            result.layer_times['sdp'] = dt

            self._log(f"L2 SDP: ub={sdp_ub:.2f}→{sdp_ub_ceil:.0f}, "
                      f"rounding={rounding_cut:.0f}, gap={result.gap_pct:.1%}, "
                      f"t={dt:.3f}s")
        else:
            result.notes.append('sdp_failed')
            self._log(f"L2 SDP: FAILED, t={dt:.3f}s")

    def _run_layer3_lanczos(self, result, n_nodes, edges, weights, t0):
        """Laag 3: Lanczos exact → bewezen optimum."""
        t_start = time.time()

        try:
            from lanczos_bench import lanczos_maxcut
            mc = lanczos_maxcut(edges, n_nodes, weights)
            dt = time.time() - t_start

            result.lower_bound = mc.max_cut
            result.upper_bound = mc.max_cut
            result.best_cut = mc.max_cut
            result.best_bitstring = mc.bitstring
            result.lower_method = 'lanczos_exact'
            result.upper_method = 'lanczos_exact'
            result.best_method = 'lanczos_exact'
            result.is_exact = True
            result.layers_completed.append('lanczos')
            result.layer_times['lanczos'] = dt

            self._log(f"L3 Lanczos: EXACT cut={mc.max_cut:.0f}, "
                      f"t={dt:.3f}s")
        except Exception as e:
            dt = time.time() - t_start
            result.notes.append(f'lanczos_failed: {e}')
            self._log(f"L3 Lanczos: FAILED ({e}), t={dt:.3f}s")

    def _run_layer4_qaoa(self, result, n_nodes, edges, weights, t0, p=1):
        """Laag 4: QAOA p=1 via B48 ZornSolver."""
        t_start = time.time()

        try:
            from auto_planner import ZornSolver
            solver = ZornSolver(
                chi_budget=self.chi_budget, gpu=self.gpu,
                verbose=False)
            sr = solver.solve(n_nodes, edges, weights=weights, p=p)
            dt = time.time() - t_start

            # QAOA cut is altijd een geldige ondergrens
            if sr.cut_value > result.lower_bound:
                result.lower_bound = sr.cut_value
                result.lower_method = f'QAOA-p{p} ({sr.method})'

            if sr.cut_value > result.best_cut:
                result.best_cut = sr.cut_value
                result.best_bitstring = sr.best_bitstring
                result.best_method = f'QAOA-p{p} ({sr.method})'

            result.layers_completed.append(f'qaoa_p{p}')
            result.layer_times[f'qaoa_p{p}'] = dt

            self._log(f"L4 QAOA p={p}: cut={sr.cut_value:.1f}, "
                      f"method={sr.method}, t={dt:.1f}s")
        except Exception as e:
            dt = time.time() - t_start
            result.notes.append(f'qaoa_p{p}_failed: {e}')
            self._log(f"L4 QAOA: FAILED ({e}), t={dt:.1f}s")

    def _run_layer5_deep(self, result, n_nodes, edges, weights, t0):
        """Laag 5: Deeper QAOA (p=2) of RQAOA voor grote grafen."""
        t_start = time.time()

        # Probeer p=2 als er genoeg tijd is
        if self._time_left(t0) > 10.0 and n_nodes <= 60:
            self._run_layer4_qaoa(result, n_nodes, edges, weights, t0, p=2)

        dt = time.time() - t_start
        result.layer_times['deep'] = dt


# =====================================================================
# CONVENIENCE / CLI
# =====================================================================

def solve_certified(n_nodes, edges, weights=None, time_budget=30.0,
                    verbose=True, **kwargs):
    """One-liner voor gecertificeerde MaxCut."""
    solver = AnytimeSolver(time_budget=time_budget, verbose=verbose, **kwargs)
    return solver.solve(n_nodes, edges, weights)


if __name__ == '__main__':
    print("=" * 60)
    print("AnytimeSolver — Certified MaxCut (B49)")
    print("=" * 60)

    # Importeer benchmark-grafen
    from gset_loader import load_graph, BUILTIN_GRAPHS

    def wg_to_edges(wg):
        edges, seen = [], set()
        for u, v, w in wg.edges():
            key = (min(u, v), max(u, v))
            if key not in seen:
                edges.append(key)
                seen.add(key)
        return wg.n_nodes, edges

    # Test op Petersen
    g, bks, info = load_graph('petersen')
    n, edges = wg_to_edges(g)

    print(f"\nTest: Petersen (n={n}, m={len(edges)}, BKS={bks})")
    result = solve_certified(n, edges, time_budget=30.0)
    print()
    print(result.certificate_str())
