#!/usr/bin/env python3
"""
B153: Beyond-MaxCut QUBO Suite.

Maakt de ZornQ dispatcher-claim "domein-agnostisch" hard door 4 standaard
QUBO-problemen op één API te zetten:

1. Weighted MaxCut       — uitbreiding van bestaande SimpleGraph met edge-weights.
2. Max-k-Cut             — k-partitionering via one-hot encoding (k=2,3,4,...).
3. Maximum Independent Set (MIS) — penalty-gebaseerde QUBO-formulering.
4. Markowitz portfolio   — financiële QUBO (Lucas 2014 §6.3).

Architectuur
------------
    Probleem (graaf, returns, ...)  ──encode──▶  QUBO  ──solver──▶  x*
                                                  │                   │
                                                  └────decode─────────┘
                                                          │
                                                  feasible? value?

Elke encoder retourneert een `QUBOInstance` met:
    - `qubo`     : QUBO matrix Q (symmetrisch) + offset
    - `decode(x)`: vertaalt bitstring terug naar probleem-specifieke output
                   ({"value", "feasible", "details"})

Solvers werken puur op QUBO en zijn dus generiek (brute-force, gulzige LS,
gesimuleerde annealing, multi-start LS). Brute-force is EXACT-gecertificeerd
(B131); heuristieken zijn LOWER_BOUND-gecertificeerd.

Conventie: `min x^T Q x + offset` met x ∈ {0,1}^n. Maximalisatie wordt
geëncodeerd door de QUBO-coëfficiënten te negeren — `decode` keert dat om.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ============================================================
# QUBO datatype
# ============================================================

@dataclass
class QUBO:
    """Symmetrische QUBO: minimize x^T Q x + offset, x ∈ {0,1}^n."""
    Q: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        Q = np.asarray(self.Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q moet vierkant 2D zijn, kreeg shape {Q.shape}")
        # Symmetriseer (sommige encoders bouwen alleen bovendriehoek)
        self.Q = (Q + Q.T) / 2.0

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def evaluate_int(self, bits: int) -> float:
        """Evalueer voor x als integer-bitstring (LSB = x_0)."""
        x = np.array([(bits >> i) & 1 for i in range(self.n)], dtype=float)
        return self.evaluate(x)

    def delta_flip(self, x: np.ndarray, i: int) -> float:
        """Δ E bij flip van x[i]. O(n) i.p.v. O(n²)."""
        # f(x) = x^T Q x. Stel x' = x met i-de bit geflipt.
        # Δ = f(x') - f(x).
        # x'_i = 1 - x_i, dus dx_i = (1 - 2 x_i).
        # Δ = Q_ii × (x'_i² - x_i²) + 2 dx_i × Σ_{j≠i} Q_ij × x_j
        #   = Q_ii × (x'_i - x_i) + 2 dx_i × (Q_i: · x - Q_ii x_i)
        #   (omdat x²=x voor binair en symmetrisch Q)
        # Eenvoudiger: Δ = (1 - 2 x_i) × (Q_ii + 2 Σ_{j≠i} Q_ij x_j)
        dx = 1.0 - 2.0 * x[i]
        # 2 × Σ_{j≠i} Q_ij x_j = 2 × (Q_i: · x − Q_ii × x_i)
        s = 2.0 * (self.Q[i] @ x - self.Q[i, i] * x[i])
        return float(dx * (self.Q[i, i] + s))


# ============================================================
# Probleem-instanties (encoder + decoder samen)
# ============================================================

@dataclass
class QUBOInstance:
    """Probleem-specifieke wrapper rond een QUBO."""
    qubo: QUBO
    name: str
    decode: Callable[[np.ndarray], dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 1. Weighted MaxCut
# ============================================================

def encode_weighted_maxcut(
    n: int,
    edges: list[tuple[int, int, float]],
) -> QUBOInstance:
    """Weighted MaxCut → QUBO.

    Maximaliseer  Σ_{(i,j)} w_ij (x_i + x_j − 2 x_i x_j)  met x ∈ {0,1}^n.

    Min-formulering met Q_ii = -deg_w(i),  Q_ij = w_ij  voor (i,j) ∈ E.
    (factor 2 op off-diag wordt opgevangen door symmetrische evaluatie:
    x^T Q x = Σ Q_ii x_i + 2 Σ_{i<j} Q_ij x_i x_j)
    """
    Q = np.zeros((n, n))
    for (u, v, w) in edges:
        if u == v:
            continue
        Q[u, u] -= w
        Q[v, v] -= w
        Q[u, v] += w
        Q[v, u] += w
    qubo = QUBO(Q, offset=0.0)

    def decode(x: np.ndarray) -> dict[str, Any]:
        cut = 0.0
        for (u, v, w) in edges:
            if int(x[u]) != int(x[v]):
                cut += w
        return {
            "value": cut,
            "feasible": True,
            "partition": [int(b) for b in x],
        }

    return QUBOInstance(
        qubo=qubo,
        name="WeightedMaxCut",
        decode=decode,
        metadata={"n": n, "n_edges": len(edges),
                  "total_weight": sum(abs(w) for _, _, w in edges)},
    )


# ============================================================
# 2. Max-k-Cut
# ============================================================

def encode_max_k_cut(
    n: int,
    edges: list[tuple[int, int, float]],
    k: int,
    penalty: float | None = None,
) -> QUBOInstance:
    """Max-k-Cut via one-hot encoding.

    Variabelen: x[i,c] ∈ {0,1} voor knoop i, kleur c. Index = i*k + c.
    Constraint: Σ_c x[i,c] = 1 (one-hot) — penalty A (Σ_c x[i,c] − 1)².
    Doel:  max Σ_{(i,j)} w_ij (1 − Σ_c x[i,c] x[j,c]).
    """
    if k < 2:
        raise ValueError("k moet >= 2")
    if penalty is None:
        # Voldoende om one-hot af te dwingen: A > totaal edge-gewicht
        penalty = max(1.0, 2.0 * sum(abs(w) for _, _, w in edges))

    N = n * k
    Q = np.zeros((N, N))
    offset = 0.0

    def idx(i: int, c: int) -> int:
        return i * k + c

    # One-hot penalty per knoop: A (Σ_c x_{i,c} − 1)²
    #   = A × [Σ_c x_{i,c} + 2 Σ_{c<c'} x_{i,c} x_{i,c'} − 2 Σ_c x_{i,c} + 1]
    #   = A × [-Σ_c x_{i,c} + 2 Σ_{c<c'} x_{i,c} x_{i,c'} + 1]
    for i in range(n):
        for c in range(k):
            Q[idx(i, c), idx(i, c)] -= penalty
        for c in range(k):
            for cp in range(c + 1, k):
                Q[idx(i, c), idx(i, cp)] += penalty
                Q[idx(i, cp), idx(i, c)] += penalty
        offset += penalty

    # Cost-term: + Σ_{(i,j)∈E} w_ij Σ_c x_{i,c} x_{j,c}  (= − cut)
    for (u, v, w) in edges:
        if u == v:
            continue
        for c in range(k):
            Q[idx(u, c), idx(v, c)] += w / 2.0
            Q[idx(v, c), idx(u, c)] += w / 2.0

    # Voor maximalisatie gaat de offset ook in: oorspronkelijk obj_max = m − ...
    # We willen min(obj_min) = min(- obj_max + const). const is de m-term.
    total_w = sum(w for _, _, w in edges)
    offset -= total_w

    qubo = QUBO(Q, offset=offset)

    def decode(x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x).reshape(n, k)
        # Feasibility: precies één 1 per rij
        sums = x.sum(axis=1)
        feasible = bool(np.all(sums == 1))
        # Voor infeasible rows: pak argmax (best-effort)
        colors = x.argmax(axis=1).tolist()
        cut = 0.0
        for (u, v, w) in edges:
            if colors[u] != colors[v]:
                cut += w
        return {
            "value": cut,
            "feasible": feasible,
            "colors": colors,
            "row_sums": sums.tolist(),
            "k": k,
        }

    return QUBOInstance(
        qubo=qubo,
        name=f"Max-{k}-Cut",
        decode=decode,
        metadata={"n": n, "k": k, "n_edges": len(edges), "penalty": penalty},
    )


# ============================================================
# 3. Maximum Independent Set (MIS)
# ============================================================

def encode_mis(
    n: int,
    edges: list[tuple[int, int]],
    penalty: float | None = None,
) -> QUBOInstance:
    """Maximum Independent Set → QUBO.

    Maximaliseer Σ x_i  s.t.  x_i x_j = 0 voor (i,j) ∈ E.
    QUBO:  min  − Σ x_i  +  A Σ_{(i,j)∈E} x_i x_j.

    Penalty A moet > 1 zijn (typisch A = n + 1 garandeert dat geen
    infeasibele oplossing kan winnen van de feasibele met grootste size).
    """
    if penalty is None:
        penalty = float(n + 1)

    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] -= 1.0
    for (u, v) in edges:
        if u == v:
            continue
        Q[u, v] += penalty / 2.0
        Q[v, u] += penalty / 2.0

    qubo = QUBO(Q, offset=0.0)

    edge_set = {(min(u, v), max(u, v)) for (u, v) in edges}

    def decode(x: np.ndarray) -> dict[str, Any]:
        selected = [int(i) for i, b in enumerate(x) if b > 0.5]
        size = len(selected)
        # Feasibility check
        violations = 0
        for i, u in enumerate(selected):
            for v in selected[i + 1:]:
                if (min(u, v), max(u, v)) in edge_set:
                    violations += 1
        feasible = violations == 0
        return {
            "value": size if feasible else size - violations,
            "feasible": feasible,
            "selected": selected,
            "size": size,
            "violations": violations,
        }

    return QUBOInstance(
        qubo=qubo,
        name="MIS",
        decode=decode,
        metadata={"n": n, "n_edges": len(edges), "penalty": penalty},
    )


# ============================================================
# 4. Markowitz portfolio
# ============================================================

def encode_markowitz(
    returns: np.ndarray,        # (n,) verwachte returns
    cov: np.ndarray,            # (n, n) covariantie-matrix (Σ ⪰ 0)
    budget: int,                # K = aantal te selecteren assets
    risk_aversion: float = 1.0, # λ
    penalty: float | None = None,  # A voor budget-constraint
) -> QUBOInstance:
    """Markowitz portfolio-optimalisatie als QUBO.

    Maximaliseer:  μ^T x − λ x^T Σ x − A (1^T x − K)²
    waar x ∈ {0,1}^n selecteert welke assets in de portfolio zitten.

    Min-vorm na expansie:
      Q = λ Σ + A 11^T   (off-diag uit penalty + cov)
      lin (op diag) = − μ − 2AK + A    (de A voor x_i² uit penalty)
      offset = A K²
    """
    n = len(returns)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n, n):
        raise ValueError("cov-shape moet (n, n) zijn")
    if not (0 <= budget <= n):
        raise ValueError("budget moet in [0, n] liggen")
    if penalty is None:
        # Maak penalty voldoende groot: max diag(cov) + |returns| schaal
        penalty = max(1.0, float(np.max(np.abs(returns))) + risk_aversion * float(np.max(np.abs(cov))))

    K = budget
    A = penalty

    Q_quad = risk_aversion * cov + A * np.ones((n, n))
    # Linear coef: − μ − 2AK + A  (de A komt van A × x_i² in de penalty-expansie:
    #   (Σ x_i − K)² = (Σ x_i)² − 2K Σ x_i + K²
    #   (Σ x_i)² = Σ x_i² + 2 Σ_{i<j} x_i x_j = Σ x_i + 2 Σ_{i<j} x_i x_j
    # dus diag krijgt + A − 2AK − μ)
    lin = -returns - 2.0 * A * K + A
    Q = Q_quad.copy()
    for i in range(n):
        Q[i, i] += lin[i]
        # Trek de A op de off-diagonal van het 11^T deel niet dubbel;
        # 11^T heeft 1 op diag wat al meegenomen wordt via lin (de + A)
        Q[i, i] -= A  # corrigeer: 11^T diag al meegenomen via Q_quad

    offset = A * K * K
    qubo = QUBO(Q, offset=offset)

    def decode(x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=float)
        selected = [int(i) for i, b in enumerate(x) if b > 0.5]
        size = len(selected)
        feasible = size == K
        ret = float(returns @ x)
        risk = float(x @ cov @ x)
        utility = ret - risk_aversion * risk
        return {
            "value": utility if feasible else utility - A * (size - K) ** 2,
            "feasible": feasible,
            "selected": selected,
            "size": size,
            "expected_return": ret,
            "variance": risk,
            "utility": utility,
        }

    return QUBOInstance(
        qubo=qubo,
        name="Markowitz",
        decode=decode,
        metadata={"n": n, "budget": K, "risk_aversion": risk_aversion,
                  "penalty": A},
    )


def random_markowitz_instance(
    n: int,
    seed: int = 42,
    budget: int | None = None,
    risk_aversion: float = 1.0,
) -> QUBOInstance:
    """Genereer een Markowitz-instantie met realistische return/cov."""
    rng = np.random.RandomState(seed)
    if budget is None:
        budget = max(1, n // 3)
    # Returns: log-normaal-achtig, sommige positief sommige negatief
    returns = rng.normal(0.05, 0.10, n)
    # Covariantie: random PSD via X X^T + diag
    X = rng.normal(0, 1, (n, max(2, n // 2)))
    cov = (X @ X.T) / X.shape[1] + 0.01 * np.eye(n)
    return encode_markowitz(returns, cov, budget=budget, risk_aversion=risk_aversion)


# ============================================================
# Generieke QUBO-solvers
# ============================================================

def qubo_brute_force(qubo: QUBO, max_n: int = 22) -> dict[str, Any]:
    """Exact via 2^n enumeratie (n ≤ max_n)."""
    n = qubo.n
    if n > max_n:
        raise ValueError(f"n={n} > max_n={max_n}; brute force niet haalbaar")
    t0 = time.time()
    best_x = np.zeros(n, dtype=float)
    best_E = qubo.evaluate(best_x)
    for bits in range(2 ** n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        E = qubo.evaluate(x)
        if E < best_E:
            best_E = E
            best_x = x.copy()
    return {
        "x": best_x,
        "energy": best_E,
        "wall_time": time.time() - t0,
        "certified": True,
        "method": "brute_force",
    }


def qubo_local_search(
    qubo: QUBO,
    x0: np.ndarray | None = None,
    max_iter: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Gulzige 1-flip local search (greedy descent)."""
    n = qubo.n
    rng = np.random.RandomState(seed)
    if x0 is None:
        x = rng.randint(0, 2, n).astype(float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    t0 = time.time()
    E = qubo.evaluate(x)
    iters = 0
    for _ in range(max_iter):
        # Vind beste flip
        deltas = np.array([qubo.delta_flip(x, i) for i in range(n)])
        i_best = int(np.argmin(deltas))
        if deltas[i_best] >= -1e-12:
            break
        x[i_best] = 1.0 - x[i_best]
        E += deltas[i_best]
        iters += 1
    return {
        "x": x,
        "energy": E,
        "iterations": iters,
        "wall_time": time.time() - t0,
        "certified": False,
        "method": "local_search",
    }


def qubo_simulated_annealing(
    qubo: QUBO,
    n_sweeps: int = 1000,
    T_start: float = 5.0,
    T_end: float = 0.01,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Klassieke SA: 1-flip Metropolis met geometrische cooling."""
    n = qubo.n
    rng = np.random.RandomState(seed)
    if x0 is None:
        x = rng.randint(0, 2, n).astype(float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    E = qubo.evaluate(x)
    best_x, best_E = x.copy(), E
    t0 = time.time()
    if n_sweeps <= 0:
        return {"x": x, "energy": E, "best_energy": E,
                "wall_time": 0.0, "certified": False,
                "method": "simulated_annealing"}
    cooling = (T_end / T_start) ** (1.0 / max(1, n_sweeps - 1))
    T = T_start
    for sweep in range(n_sweeps):
        for _ in range(n):
            i = rng.randint(0, n)
            dE = qubo.delta_flip(x, i)
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-12)):
                x[i] = 1.0 - x[i]
                E += dE
                if E < best_E:
                    best_E = E
                    best_x = x.copy()
        T *= cooling
    return {
        "x": best_x,
        "energy": best_E,
        "wall_time": time.time() - t0,
        "certified": False,
        "method": "simulated_annealing",
        "n_sweeps": n_sweeps,
    }


def qubo_random_restart(
    qubo: QUBO,
    n_starts: int = 20,
    seed: int = 0,
    inner: str = "local_search",
    inner_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Multi-start met LS of SA als inner solver."""
    inner_kwargs = inner_kwargs or {}
    rng = np.random.RandomState(seed)
    t0 = time.time()
    best = None
    for s in range(n_starts):
        x0 = rng.randint(0, 2, qubo.n).astype(float)
        if inner == "local_search":
            res = qubo_local_search(qubo, x0=x0, seed=int(rng.randint(0, 2**31 - 1)),
                                    **inner_kwargs)
        elif inner == "simulated_annealing":
            res = qubo_simulated_annealing(qubo, x0=x0,
                                           seed=int(rng.randint(0, 2**31 - 1)),
                                           **inner_kwargs)
        else:
            raise ValueError(f"Onbekende inner='{inner}'")
        if best is None or res["energy"] < best["energy"]:
            best = res
            best["start_index"] = s
    best["wall_time"] = time.time() - t0
    best["method"] = f"random_restart_{inner}_{n_starts}"
    return best


# ============================================================
# Probleem-specifieke conveniences
# ============================================================

def solve_problem(
    instance: QUBOInstance,
    solver: str = "brute_force",
    **solver_kwargs: Any,
) -> dict[str, Any]:
    """Los probleem op + decode resultaat."""
    solvers = {
        "brute_force": qubo_brute_force,
        "local_search": qubo_local_search,
        "simulated_annealing": qubo_simulated_annealing,
        "random_restart": qubo_random_restart,
    }
    if solver not in solvers:
        raise ValueError(f"Onbekende solver='{solver}'; opties: {list(solvers)}")
    raw = solvers[solver](instance.qubo, **solver_kwargs)
    decoded = instance.decode(raw["x"])
    return {
        **raw,
        "decoded": decoded,
        "problem": instance.name,
    }


# ============================================================
# Graaf-helpers (kleine dependency-loze versie)
# ============================================================

def cycle_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def complete_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def petersen_edges() -> list[tuple[int, int]]:
    """Petersen-graaf: 10 knopen, 15 edges (3-reguliere geen-Hamiltoniaan-cycle)."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + ((i + 2) % 5)) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return outer + inner + spokes


def random_erdos_renyi_edges(n: int, p: float, seed: int = 42) -> list[tuple[int, int]]:
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def random_weighted_edges(
    n: int, p: float, seed: int = 42,
    w_low: float = 0.1, w_high: float = 1.0,
) -> list[tuple[int, int, float]]:
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.uniform(w_low, w_high)
                edges.append((i, j, w))
    return edges


# ============================================================
# CLI
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B153 Beyond-MaxCut QUBO suite")
    sub = p.add_subparsers(dest="problem", required=True)

    # weighted-maxcut
    pmc = sub.add_parser("maxcut", help="Weighted MaxCut")
    pmc.add_argument("--n", type=int, default=8)
    pmc.add_argument("--p", type=float, default=0.5)
    pmc.add_argument("--seed", type=int, default=42)
    pmc.add_argument("--solver", default="brute_force")

    # max-k-cut
    pkc = sub.add_parser("kcut", help="Max-k-Cut")
    pkc.add_argument("--n", type=int, default=6)
    pkc.add_argument("--k", type=int, default=3)
    pkc.add_argument("--p", type=float, default=0.5)
    pkc.add_argument("--seed", type=int, default=42)
    pkc.add_argument("--solver", default="local_search")

    # mis
    pmis = sub.add_parser("mis", help="Maximum Independent Set")
    pmis.add_argument("--n", type=int, default=10)
    pmis.add_argument("--p", type=float, default=0.4)
    pmis.add_argument("--seed", type=int, default=42)
    pmis.add_argument("--solver", default="brute_force")

    # markowitz
    pmark = sub.add_parser("markowitz", help="Markowitz portfolio")
    pmark.add_argument("--n", type=int, default=8)
    pmark.add_argument("--budget", type=int, default=3)
    pmark.add_argument("--risk", type=float, default=1.0)
    pmark.add_argument("--seed", type=int, default=42)
    pmark.add_argument("--solver", default="brute_force")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.problem == "maxcut":
        edges = random_weighted_edges(args.n, args.p, seed=args.seed)
        inst = encode_weighted_maxcut(args.n, edges)
    elif args.problem == "kcut":
        edges_w = random_weighted_edges(args.n, args.p, seed=args.seed)
        inst = encode_max_k_cut(args.n, edges_w, k=args.k)
    elif args.problem == "mis":
        edges = random_erdos_renyi_edges(args.n, args.p, seed=args.seed)
        inst = encode_mis(args.n, edges)
    elif args.problem == "markowitz":
        inst = random_markowitz_instance(args.n, seed=args.seed,
                                         budget=args.budget,
                                         risk_aversion=args.risk)
    else:
        raise ValueError(f"Onbekend probleem: {args.problem}")

    print(f"== {inst.name} ==  (n_qubo={inst.qubo.n})")
    for k, v in inst.metadata.items():
        print(f"  {k}: {v}")
    res = solve_problem(inst, solver=args.solver)
    print(f"  solver:   {res['method']}")
    print(f"  energy:   {res['energy']:.6f}")
    print(f"  walltime: {res['wall_time']:.4f}s")
    d = res["decoded"]
    print(f"  feasible: {d['feasible']}")
    print(f"  value:    {d.get('value')}")
    extra = {k: v for k, v in d.items() if k not in {"value", "feasible"}}
    for k, v in extra.items():
        if isinstance(v, list) and len(v) > 16:
            print(f"  {k}: <{len(v)} entries>")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
