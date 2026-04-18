#!/usr/bin/env python3
"""B176 -- Frank-Wolfe / Conditional Gradient solver voor MaxCut-SDP.

Doel: schaalbare GW-bound (B60) tot n ~ 10000 op een laptop, waar cvxpy
(SCS/MOSEK interior-point) in de praktijk vastloopt voorbij n ~ 500.

Algoritme
---------
MaxCut-SDP:
    max  (1/4)*tr(L*X)     s.t.  X >= 0 (PSD),  X_ii = 1                (P)

Equivalente relaxatie met diagonaal-penalty over het gescaled
spectraplex  Delta_n = {X >= 0, tr(X) = n}:

    min  f(X) = -(1/4)*tr(L*X) + (lam/2)*||diag(X) - 1||^2              (P')

Frank-Wolfe (Jaggi 2013, Hazan 2008) op (P'):

  1. Gradient:  G = grad f(X) = -(1/4)*L + lam*diag(diag(X) - 1)
  2. LMO:       V* = n * v*v^T       met v = bottom-eigenvector(G)
  3. Step-size: gesloten-vorm line-search (f is kwadratisch in gamma)
  4. Update:    X <- (1-gamma)*X + gamma*V*

Opslag:  X = Y*Y^T  met  Y in R^{n x r}.  Rank-cap r via SVD-truncatie.

Gradient-matvec is matrix-vrij: diag(Y*Y^T) = row-norms^2, L*v via sparse.

Duality gap:  g(X) = <G, X> - <G, V*> >= f(X) - f(X*) >= 0  (FW-certificaat).

Valide bovengrens op cut_SDP
----------------------------
f(X) = -cut(X) + (lam/2)*||d||^2  met  cut(X) = (1/4)*tr(L*X),  d = diag(X)-1.
Voor elke feasibele X0 met diag(X0) = 1 geldt f(X0) = -cut(X0), dus f* ligt
onder f(SDP-optimum) = -cut_SDP, d.w.z. -f* >= cut_SDP.

FW-gap: f(X_k) - f* <= gap_k, dus -f* <= -f(X_k) + gap_k.

Gevolg:  cut_SDP <= -f(X_k) + gap_k = (1/4)*tr(L*X_k) - (lam/2)*||d||^2 + gap_k.

Dit is de gerapporteerde `sdp_upper_bound`.  Naarmate lam groot wordt en FW
convergeert, trekt deze naar cut_SDP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Hergebruik graaf-types en helpers van B60 waar mogelijk.
try:  # graceful fallback als b60 niet geimporteerd kan worden
    from b60_gw_bound import SimpleGraph, cylinder_graph, random_3regular, random_erdos_renyi
except ImportError:  # pragma: no cover
    SimpleGraph = None  # type: ignore


# ============================================================
# Graaf -> Laplaciaan
# ============================================================


def graph_laplacian(graph) -> sp.csr_matrix:
    """Sparse graph Laplacian L = D - W (CSR)."""
    n = graph.n
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    deg = np.zeros(n, dtype=float)
    for u, v, w in graph.edges:
        w = float(w)
        rows.append(u); cols.append(v); vals.append(-w)
        rows.append(v); cols.append(u); vals.append(-w)
        deg[u] += w
        deg[v] += w
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(deg[i])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)


# ============================================================
# Linear Minimization Oracle (LMO) over het gescaled spectraplex
# ============================================================


def lmo_spectraplex(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    tol: float = 1e-8,
    maxiter: int = 1000,
    dense_fallback_below: int = 40,
    dense_G: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Find  v in argmin_{||v||=1} v^T * G * v   and its eigenvalue.

    Returns (v, lam) met G*v = lam*v en lam = lambda_min(G).
    LMO-waarde over Delta_n (tr = n) is dan  n*v*v^T  met inner product
    n*lam tegen G.
    """
    if n <= dense_fallback_below:
        if dense_G is None:
            G = np.empty((n, n), dtype=float)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                G[:, i] = matvec(e)
            G = 0.5 * (G + G.T)
        else:
            G = 0.5 * (dense_G + dense_G.T)
        w, V = np.linalg.eigh(G)
        return V[:, 0].copy(), float(w[0])

    # Scale maxiter with problem size for large graphs
    effective_maxiter = max(maxiter, min(5000, n * 2))
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    try:
        w, V = spla.eigsh(op, k=1, which="SA", tol=tol, maxiter=effective_maxiter)
        return V[:, 0].copy(), float(w[0])
    except spla.ArpackNoConvergence as exc:
        if exc.eigenvalues.size > 0:
            idx = int(np.argmin(exc.eigenvalues))
            return exc.eigenvectors[:, idx].copy(), float(exc.eigenvalues[idx])
        # Retry with looser tolerance
        try:
            w, V = spla.eigsh(op, k=1, which="SA", tol=max(tol, 1e-4),
                              maxiter=effective_maxiter * 2)
            return V[:, 0].copy(), float(w[0])
        except spla.ArpackNoConvergence as exc2:
            if exc2.eigenvalues.size > 0:
                idx = int(np.argmin(exc2.eigenvalues))
                return exc2.eigenvectors[:, idx].copy(), float(exc2.eigenvalues[idx])
            # Last resort: return a random unit vector with Rayleigh quotient
            rng = np.random.default_rng(42)
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v) + 1e-18
            lam = float(v @ matvec(v))
            return v, lam


# ============================================================
# Resultaat-container
# ============================================================


@dataclass
class FWResult:
    sdp_bound: float                # (1/4)*tr(L*X_k)  raw primal-objective
    sdp_upper_bound: float          # VALIDE bovengrens op cut_SDP (zie docstring)
    feasible_cut_lb: float          # (1/4)*tr(L*Xhat) voor row-genormaliseerd Xhat
    Y: np.ndarray                   # X = Y*Y^T
    X_diag: np.ndarray
    diag_err_max: float
    iterations: int
    final_gap: float
    converged: bool
    solve_time: float
    history: list[dict] = field(default_factory=list)
    n: int = 0
    n_edges: int = 0
    penalty: float = 0.0

    @property
    def gw_guaranteed(self) -> float:
        """0.87856 * valide bovengrens op cut_SDP."""
        return 0.87856 * max(self.sdp_upper_bound, 0.0)


# ============================================================
# Frank-Wolfe-solver
# ============================================================


def frank_wolfe_maxcut_sdp(
    graph,
    max_iter: int = 500,
    tol: float = 1e-4,
    penalty: Optional[float] = None,
    rank_cap: int = 64,
    step_rule: str = "linesearch",
    seed: int = 0,
    verbose: bool = False,
    lmo_tol: float = 1e-8,
    dense_fallback_below: int = 40,
) -> FWResult:
    """Frank-Wolfe op MaxCut-SDP-relaxatie via gescaled spectraplex + penalty.

    Parameters
    ----------
    graph : B60 SimpleGraph (of any object met n, edges, adj)
    max_iter : max aantal FW-stappen
    tol : FW-duality-gap tolerantie (stop als gap < tol * max(1, |f|))
    penalty : lam voor diagonaal-penalty; None -> auto ~ ||L||_1 / n
    rank_cap : max aantal Y-kolommen; overschrijding triggert SVD-truncatie
    step_rule : 'linesearch' (gesloten-vorm) of 'jaggi' (gamma = 2/(k+2))
    seed : RNG-seed voor initiele vector
    verbose : print per 10 iteraties
    lmo_tol : eigsh tolerance
    dense_fallback_below : dense eigh voor n onder deze drempel
    """
    n = graph.n
    L = graph_laplacian(graph)
    L_abs_sum = float(np.abs(L).sum())

    if penalty is None:
        penalty = max(1.0, L_abs_sum / max(n, 1))
    lam = float(penalty)

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    v0 /= np.linalg.norm(v0) + 1e-18
    Y = (np.sqrt(n) * v0)[:, None].copy()   # X_0 = n*v0*v0^T, tr = n

    history: list[dict] = []
    t0 = time.time()
    gap = float("inf")
    k = 0
    converged = False
    f_X = 0.0
    diag_X = np.ones(n)

    for k in range(max_iter):
        diag_X = np.einsum("ij,ij->i", Y, Y)
        LY = L.dot(Y)
        tr_LX = float(np.einsum("ij,ij->", LY, Y))
        d = diag_X - 1.0
        ddsq = float(d @ d)
        f_X = -0.25 * tr_LX + 0.5 * lam * ddsq

        def gmv(x: np.ndarray) -> np.ndarray:
            return -0.25 * L.dot(x) + lam * (d * x)

        if n <= dense_fallback_below:
            G_dense = -0.25 * L.toarray() + lam * np.diag(d)
        else:
            G_dense = None

        v, lam_min = lmo_spectraplex(
            gmv, n,
            tol=lmo_tol,
            dense_fallback_below=dense_fallback_below,
            dense_G=G_dense,
        )

        grad_inner_X = -0.25 * tr_LX + lam * float(d @ diag_X)
        lmo_val = n * lam_min
        gap = grad_inner_X - lmo_val

        history.append({
            "iter": k,
            "f": f_X,
            "gap": gap,
            "tr_LX": tr_LX,
            "diag_err_max": float(np.max(np.abs(d))),
            "lam_min": lam_min,
            "rank": int(Y.shape[1]),
            "elapsed": time.time() - t0,
        })

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print("  iter %4d  f=%+.6f  gap=%+.3e  diag_err=%.2e  rank=%d" % (
                k, f_X, gap, np.max(np.abs(d)), Y.shape[1]))

        if gap <= tol * max(1.0, abs(f_X)):
            converged = True
            break
        if gap < 0 and abs(gap) < 1e-12:
            converged = True
            break

        if step_rule == "linesearch":
            v_sq = v * v
            alpha = n * v_sq - diag_X
            vLv = float(v @ L.dot(v))
            aa = float(alpha @ alpha)
            da = float(d @ alpha)
            num = 0.25 * (n * vLv - tr_LX) - lam * da
            den = lam * aa
            if den > 1e-14:
                gamma = float(np.clip(num / den, 0.0, 1.0))
            else:
                gamma = 2.0 / (k + 2.0)
        elif step_rule == "jaggi":
            gamma = 2.0 / (k + 2.0)
        else:
            raise ValueError("onbekende step_rule: %r" % step_rule)

        if gamma <= 0.0:
            converged = True
            break

        one_minus = max(1.0 - gamma, 0.0)
        Y = np.sqrt(one_minus) * Y
        new_col = np.sqrt(gamma * n) * v
        Y = np.concatenate([Y, new_col[:, None]], axis=1)

        if Y.shape[1] > rank_cap:
            U, S, _ = np.linalg.svd(Y, full_matrices=False)
            r = min(rank_cap, int(np.sum(S > 1e-12)))
            r = max(r, 1)
            Y = U[:, :r] * S[:r]

    solve_time = time.time() - t0

    diag_X = np.einsum("ij,ij->i", Y, Y)
    LY = L.dot(Y)
    tr_LX = float(np.einsum("ij,ij->", LY, Y))
    d_final = diag_X - 1.0
    diag_err_max = float(np.max(np.abs(d_final)))

    sdp_bound_raw = 0.25 * tr_LX

    # Valide bovengrens cut_SDP <= -f(X_k) + gap
    dd_final = float(d_final @ d_final)
    f_final = -0.25 * tr_LX + 0.5 * lam * dd_final
    sdp_upper_bound = -f_final + max(0.0, gap)

    # Feasible lower bound via row-normalisatie: X_hat = diag(1/||Y_i||) Y Y^T ...
    row_norms = np.linalg.norm(Y, axis=1)
    safe = np.where(row_norms > 1e-12, row_norms, 1.0)
    Y_hat = Y / safe[:, None]
    LYh = L.dot(Y_hat)
    tr_LX_feas = float(np.einsum("ij,ij->", LYh, Y_hat))
    feasible_cut_lb = 0.25 * tr_LX_feas

    return FWResult(
        sdp_bound=sdp_bound_raw,
        sdp_upper_bound=sdp_upper_bound,
        feasible_cut_lb=feasible_cut_lb,
        Y=Y,
        X_diag=diag_X,
        diag_err_max=diag_err_max,
        iterations=k + 1,
        final_gap=float(gap),
        converged=converged,
        solve_time=solve_time,
        history=history,
        n=n,
        n_edges=graph.n_edges,
        penalty=lam,
    )


# ============================================================
# GW-rounding vanaf low-rank Y
# ============================================================


def gw_round_from_Y(
    Y: np.ndarray,
    graph,
    n_trials: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Goemans-Williamson hyperplane-rounding vanaf Y (X = Y*Y^T).

    Normaliseert rijen (X_ii kan != 1 zijn na penalty) voor een pseudo-
    unit-sfeer inbedding, trekt ``n_trials`` random-Gaussian-richtingen
    en houdt de beste cut.

    Returns (best_bitstring, best_cut_value).
    """
    rng = np.random.default_rng(seed)
    n, r = Y.shape
    row_norms = np.linalg.norm(Y, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    Yn = Y / row_norms

    best_cut = -np.inf
    best_bs = np.zeros(n, dtype=int)
    for _ in range(n_trials):
        g = rng.standard_normal(r)
        scores = Yn @ g
        bs = (scores >= 0).astype(int)
        c = graph.cut_value(bs)
        if c > best_cut:
            best_cut = float(c)
            best_bs = bs.copy()
    return best_bs, float(best_cut)


# ============================================================
# Convenience: vergelijkingsfunctie met cvxpy
# ============================================================


def cvxpy_reference_sdp(graph, verbose: bool = False):
    """Roept B60's gw_sdp_bound aan als referentie (vereist cvxpy)."""
    try:
        from b60_gw_bound import gw_sdp_bound
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("cvxpy/B60 niet beschikbaar: %s" % e)
    return gw_sdp_bound(graph, verbose=verbose)


# ============================================================
# CLI-demo
# ============================================================


def _demo() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="B176 Frank-Wolfe SDP demo")
    parser.add_argument("--n", type=int, default=60, help="aantal knooppunten (3-reg)")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--rank-cap", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="ook cvxpy-referentie draaien")
    args = parser.parse_args()

    if SimpleGraph is None:
        raise SystemExit("B60 niet importable -- demo niet beschikbaar")
    g = random_3regular(args.n)
    print("Graaf: 3-regulier n=%d edges=%d" % (g.n, g.n_edges))

    res = frank_wolfe_maxcut_sdp(
        g,
        max_iter=args.max_iter,
        tol=args.tol,
        rank_cap=args.rank_cap,
        verbose=args.verbose,
    )
    print("")
    print("FW SDP raw (0.25*tr(L*X_k)) : %.4f" % res.sdp_bound)
    print("FW upper bound              : %.4f  (>= cut_SDP, certificate)" % res.sdp_upper_bound)
    print("FW feasible lower bound     : %.4f  (<= cut_SDP, row-norm)" % res.feasible_cut_lb)
    print("  iterations   : %d" % res.iterations)
    print("  final gap    : %.3e" % res.final_gap)
    print("  diag_err_max : %.3e" % res.diag_err_max)
    print("  rank(Y)      : %d" % res.Y.shape[1])
    print("  solve_time   : %.3fs" % res.solve_time)

    bs, cut = gw_round_from_Y(res.Y, g, n_trials=200)
    print("")
    print("GW-rounding best cut (200 trials): %.1f / %d" % (cut, g.n_edges))

    if args.compare:
        ref = cvxpy_reference_sdp(g, verbose=False)
        print("")
        print("cvxpy-SDP bound: %.4f   (%.3fs)" % (ref["sdp_bound"], ref["solve_time"]))
        rel = abs(res.sdp_upper_bound - ref["sdp_bound"]) / max(abs(ref["sdp_bound"]), 1e-9)
        print("relative delta (UB vs cvxpy): %.3f%%" % (100 * rel))


if __name__ == "__main__":  # pragma: no cover
    _demo()
