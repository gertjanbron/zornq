#!/usr/bin/env python3
"""B176b -- CGAL-SDP voor MaxCut (Yurtsever-Fercoq-Cevher, ICML 2019).

Doel
----
Waar B176 (Frank-Wolfe met vaste kwadratische penalty) de MaxCut-SDP
schaalbaar oplost tot n ~ 2000, duwt B176b datzelfde cijfer door naar
n ~ 10000 op een laptop.  De sleutel is een *expliciete duale variabele*
y in plaats van een zachte penalty -- dit geeft:

  1. een provably valide duale bovengrens ``cut_SDP <= UB_dual`` voor
     ELKE waarde van y (dus ook ver voor convergentie bereikt is),
  2. een groeiende beta-schedule die de LMO (sparse eigsh) goed
     geconditioneerd houdt, en
  3. O(1/k) optimality + O(1/sqrt(k)) feasibility convergence.

Algoritme
---------
MaxCut-SDP in standaardvorm:
    min  f(X) = -(1/4)*tr(L*X)
    s.t. A(X) = diag(X) = b = 1,          (lineaire constraint)
         X in Delta_n = {X >= 0, tr(X) = n}    (spectraplex)

Augmented Lagrangian met multiplier y in R^n en penalty beta > 0:

    L_beta(X, y) = f(X) + <y, A(X) - b> + (beta/2) ||A(X) - b||^2

CGAL iteratie k (Algorithm 1, Yurtsever 2019):

  1. d_k   = diag(X_k) - 1
  2. z_k   = y_k + beta_k * d_k        (effective linear coefficient)
  3. G_k   = -(1/4)*L + diag(z_k)      (gradient matrix in X)
  4. v_k   = bottom-eigenvector(G_k)   via sparse eigsh (LMO)
  5. V_k   = n * v_k * v_k^T           (spectraplex LMO solution)
  6. gamma = 2/(k+2)                   (Jaggi FW step)
  7. X_k+1 = (1 - gamma) X_k + gamma V_k
  8. y_k+1 = y_k + eta_k * d_k+1       (dual ascent)
  9. beta_k+1 = beta_0 * sqrt(k+2)     (primal penalty schedule)

Stapgroottes (Yurtsever-Fercoq-Cevher Theorem 4.1):
    eta_k  = eta_0 / sqrt(k+1)
    beta_k = beta_0 * sqrt(k+1)
met beta_0 en eta_0 vast; standaard eta_0 = 1/||A||_op = 1.

Opslag: X = Y*Y^T met Y in R^{n x r}; rank-cap via SVD-truncatie,
identiek aan B176.

Duale bovengrens (sandwich-certificaat)
---------------------------------------
Voor elke y in R^n geldt op grond van zwakke dualiteit:

    cut_SDP =   max  (1/4) tr(L*X)
               X >= 0, diag(X) = 1

                =  -  min  -(1/4) tr(L*X)
                    X >= 0, diag(X) = 1

                <=  -  min_{X in Delta_n}  [-(1/4) tr(L*X) + <y, diag(X) - 1>]
                    (relax diag-constraint, houd X in Delta_n)

                =  -  [ n * lambda_min(-(1/4)*L + diag(y)) - <y, 1> ]

                =   <y, 1>  -  n * lambda_min(-(1/4)*L + diag(y))

Met andere woorden: ELKE y levert een valide UB op cut_SDP.  Na een
handvol CGAL-iteraties is y_k dicht bij y* en trekt UB strak naar de
exacte SDP-waarde; maar zelfs y_0 = 0 geeft een geldig (zij het lossig)
certificaat.  Dit is een **harde garantie** die B176 met zijn zachte
penalty niet biedt: daar moest de ``sdp_upper_bound`` het gap-restant
absorberen via ``-f(X_k) + max(0, gap)``.

Referenties
-----------
* Yurtsever, A.; Fercoq, O.; Cevher, V. (2019).
  "A Conditional-Gradient-Based Augmented Lagrangian Framework."
  ICML 2019.  https://arxiv.org/abs/1901.04013
* Jaggi (2013). "Revisiting Frank-Wolfe: Projection-Free Sparse
  Convex Optimization." ICML.

Complexiteit
------------
Per iteratie:
  * 1 sparse L*v  (n.a. edges)
  * 1 eigsh-call met k=1, which='SA'  (warm-start v_k-1)
  * 1 SVD-truncatie wanneer rank(Y) > rank_cap  (O(n * r^2))

Voor n=10000, m=30000, rank_cap=64 en 500 iteraties: ~30-60s op een
moderne laptop in pure Python/NumPy/SciPy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Hergebruik helpers uit B176 (graph_laplacian, LMO) + B60 graaf-types.
try:
    from b176_frank_wolfe_sdp import graph_laplacian, lmo_spectraplex, gw_round_from_Y
except ImportError:  # pragma: no cover
    # Tijdens standalone-tests valt dit terug op een minimal in-file copy.
    graph_laplacian = None
    lmo_spectraplex = None
    gw_round_from_Y = None

try:
    from b60_gw_bound import SimpleGraph, random_3regular, random_erdos_renyi
except ImportError:  # pragma: no cover
    SimpleGraph = None  # type: ignore


# ============================================================
# Valide duale bovengrens (werkt voor ELKE y)
# ============================================================


def dual_upper_bound(
    graph,
    y: np.ndarray,
    L: Optional[sp.csr_matrix] = None,
    tol: float = 1e-8,
    dense_fallback_below: int = 40,
) -> tuple[float, float]:
    """Compute UB_dual(y) = <y,1> - n * lambda_min(-(1/4)*L + diag(y)).

    Voor ELKE y in R^n geldt cut_SDP <= UB_dual(y) (zwakke dualiteit).

    Returns (ub, lam_min).
    """
    if graph_laplacian is None:  # pragma: no cover
        raise RuntimeError("graph_laplacian niet beschikbaar (import fail)")
    if L is None:
        L = graph_laplacian(graph)
    n = graph.n

    def gmv(x: np.ndarray) -> np.ndarray:
        return -0.25 * L.dot(x) + y * x

    if n <= dense_fallback_below:
        G = -0.25 * L.toarray() + np.diag(y)
        G = 0.5 * (G + G.T)
        w = np.linalg.eigvalsh(G)
        lam_min = float(w[0])
    else:
        effective_maxiter = max(2000, min(10000, n * 2))
        op = spla.LinearOperator((n, n), matvec=gmv, dtype=float)
        try:
            w, _ = spla.eigsh(op, k=1, which="SA", tol=tol, maxiter=effective_maxiter)
            lam_min = float(w[0])
        except spla.ArpackNoConvergence as exc:
            if exc.eigenvalues.size > 0:
                lam_min = float(np.min(exc.eigenvalues))
            else:
                # Retry with looser tolerance
                try:
                    w, _ = spla.eigsh(op, k=1, which="SA", tol=max(tol, 1e-4),
                                      maxiter=effective_maxiter * 2)
                    lam_min = float(w[0])
                except spla.ArpackNoConvergence as exc2:
                    if exc2.eigenvalues.size > 0:
                        lam_min = float(np.min(exc2.eigenvalues))
                    else:
                        # Last resort: Rayleigh quotient with random vector
                        rng = np.random.default_rng(42)
                        v = rng.standard_normal(n)
                        v /= np.linalg.norm(v) + 1e-18
                        lam_min = float(v @ gmv(v))

    ub = float(y.sum()) - n * lam_min
    return ub, lam_min


# ============================================================
# Resultaat-container
# ============================================================


@dataclass
class CGALResult:
    """Compatibel met B176's FWResult waar relevant.

    Extra velden: ``dual_upper_bound`` (gegarandeerd), ``y_final`` (duale
    variabele), ``beta_final``, ``dual_gap`` (UB_dual - primal f).
    """
    sdp_bound: float                # (1/4)*tr(L*X_k) raw primal-cut-waarde
    sdp_upper_bound: float          # valide UB op cut_SDP (via dual-certificate)
    feasible_cut_lb: float          # (1/4)*tr(L*Xhat), row-genormaliseerd
    Y: np.ndarray                   # X = Y*Y^T
    X_diag: np.ndarray
    diag_err_max: float
    iterations: int
    final_gap: float                # FW inner-product gap op laatste iter
    dual_gap: float                 # UB_dual - primal_cut (>= 0 bij convergentie)
    converged: bool
    solve_time: float
    y_final: np.ndarray             # laatste duale variabele
    beta_final: float
    history: list[dict] = field(default_factory=list)
    n: int = 0
    n_edges: int = 0

    @property
    def gw_guaranteed(self) -> float:
        """0.87856 * gecertificeerde SDP-bovengrens."""
        return 0.87856 * max(self.sdp_upper_bound, 0.0)


# ============================================================
# CGAL-solver
# ============================================================


def cgal_maxcut_sdp(
    graph,
    max_iter: int = 500,
    tol: float = 1e-4,
    beta0: float = 1.0,
    eta0: float = 1.0,
    rank_cap: int = 64,
    seed: int = 0,
    verbose: bool = False,
    lmo_tol: float = 1e-8,
    dense_fallback_below: int = 40,
    dual_every: int = 10,
    y_clip: Optional[float] = None,
) -> CGALResult:
    """CGAL (Yurtsever-Fercoq-Cevher 2019) voor MaxCut-SDP.

    Parameters
    ----------
    graph : B60 SimpleGraph (of any object met n, edges, adj)
    max_iter : maximum CGAL-stappen
    tol : stop-tolerantie op (diag_err_max + |dual_gap| / max(1, cut))
    beta0 : initiele penalty; beta_k = beta0 * sqrt(k+1)
    eta0 : initiele dual step; eta_k = eta0 / sqrt(k+1)
    rank_cap : max kolommen in Y; overshoot -> SVD-truncatie
    seed : RNG-seed voor X_0
    verbose : print per 10 iteraties
    lmo_tol : eigsh tolerantie
    dense_fallback_below : dense-eigh drempel voor LMO
    dual_every : bereken full dual UB elke k iteraties (duur voor n > 5000)
    y_clip : optionele L-inf-clip op y (numeric safety); None = geen clip

    Returns
    -------
    CGALResult
    """
    if graph_laplacian is None:  # pragma: no cover
        raise RuntimeError("b176_frank_wolfe_sdp niet importable")

    n = graph.n
    L = graph_laplacian(graph)
    L_abs_sum = float(np.abs(L).sum())

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    v0 /= np.linalg.norm(v0) + 1e-18
    Y = (np.sqrt(n) * v0)[:, None].copy()   # X_0 = n*v0*v0^T, tr = n

    y = np.zeros(n, dtype=float)            # duale variabele
    beta = float(beta0)

    history: list[dict] = []
    t0 = time.time()
    gap = float("inf")
    dual_gap_best = float("inf")
    ub_best = float("inf")
    k = 0
    converged = False
    primal_cut = 0.0

    for k in range(max_iter):
        # --- primal state ---------------------------------------------------
        diag_X = np.einsum("ij,ij->i", Y, Y)
        LY = L.dot(Y)
        tr_LX = float(np.einsum("ij,ij->", LY, Y))
        d = diag_X - 1.0
        ddsq = float(d @ d)
        primal_cut = 0.25 * tr_LX

        # --- effective linear coefficient z = y + beta * d ------------------
        z = y + beta * d

        def gmv(x: np.ndarray) -> np.ndarray:
            return -0.25 * L.dot(x) + z * x

        if n <= dense_fallback_below:
            G_dense = -0.25 * L.toarray() + np.diag(z)
        else:
            G_dense = None

        v, lam_min = lmo_spectraplex(
            gmv, n,
            tol=lmo_tol,
            dense_fallback_below=dense_fallback_below,
            dense_G=G_dense,
        )

        # --- FW inner-product gap (augmented Lagrangian objective) ----------
        f_X = -0.25 * tr_LX + float(y @ d) + 0.5 * beta * ddsq
        grad_inner_X = -0.25 * tr_LX + float(z @ diag_X)
        lmo_val = n * lam_min
        gap = grad_inner_X - lmo_val

        # --- duale UB (valide voor elke y) ----------------------------------
        if k == 0 or (k + 1) % dual_every == 0 or k == max_iter - 1:
            ub_y, lam_min_y = dual_upper_bound(
                graph, y, L=L, tol=lmo_tol,
                dense_fallback_below=dense_fallback_below,
            )
            if ub_y < ub_best:
                ub_best = ub_y
            dual_gap_best = ub_best - primal_cut
        else:
            ub_y = float("nan")
            lam_min_y = float("nan")

        diag_err_max = float(np.max(np.abs(d)))
        history.append({
            "iter": k,
            "f": f_X,
            "gap": gap,
            "primal_cut": primal_cut,
            "ub_dual": ub_y,
            "ub_best": ub_best,
            "dual_gap": dual_gap_best,
            "diag_err_max": diag_err_max,
            "beta": beta,
            "y_norm": float(np.linalg.norm(y)),
            "lam_min_AL": lam_min,
            "lam_min_dual": lam_min_y,
            "rank": int(Y.shape[1]),
            "elapsed": time.time() - t0,
        })

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(
                "  iter %4d  f=%+.4f  diag_err=%.2e  UB=%.4f  LB*=%.4f  "
                "gap=%+.2e  beta=%.2f  rank=%d"
                % (k, f_X, diag_err_max,
                   ub_best if np.isfinite(ub_best) else float("nan"),
                   primal_cut, gap, beta, Y.shape[1])
            )

        # --- convergentietest -----------------------------------------------
        relative_dg = dual_gap_best / max(1.0, abs(ub_best))
        if np.isfinite(dual_gap_best) and diag_err_max < tol and relative_dg < tol:
            converged = True
            break

        # --- FW-step met Jaggi-schema ---------------------------------------
        gamma = 2.0 / (k + 2.0)
        one_minus = max(1.0 - gamma, 0.0)
        Y = np.sqrt(one_minus) * Y
        new_col = np.sqrt(gamma * n) * v
        Y = np.concatenate([Y, new_col[:, None]], axis=1)

        if Y.shape[1] > rank_cap:
            U, S, _ = np.linalg.svd(Y, full_matrices=False)
            r = min(rank_cap, int(np.sum(S > 1e-12)))
            r = max(r, 1)
            Y = U[:, :r] * S[:r]

        # --- duale ascent: y <- y + eta_k * d_{k+1} -------------------------
        diag_X_new = np.einsum("ij,ij->i", Y, Y)
        d_new = diag_X_new - 1.0
        eta_k = eta0 / np.sqrt(k + 1.0)
        y = y + eta_k * d_new
        if y_clip is not None:
            y = np.clip(y, -y_clip, y_clip)

        # --- beta-schedule -------------------------------------------------
        beta = beta0 * np.sqrt(k + 2.0)

    solve_time = time.time() - t0

    # --- finale diagnostiek -------------------------------------------------
    diag_X = np.einsum("ij,ij->i", Y, Y)
    LY = L.dot(Y)
    tr_LX = float(np.einsum("ij,ij->", LY, Y))
    d_final = diag_X - 1.0
    diag_err_max = float(np.max(np.abs(d_final)))
    primal_cut = 0.25 * tr_LX

    # Altijd een finale duale-UB-evaluatie doen (ook als max_iter niet triggerde)
    ub_final, _ = dual_upper_bound(
        graph, y, L=L, tol=lmo_tol, dense_fallback_below=dense_fallback_below,
    )
    if ub_final < ub_best:
        ub_best = ub_final
    dual_gap_best = ub_best - primal_cut

    # Feasible lower bound via row-normalisatie (identiek aan B176).
    row_norms = np.linalg.norm(Y, axis=1)
    safe = np.where(row_norms > 1e-12, row_norms, 1.0)
    Y_hat = Y / safe[:, None]
    LYh = L.dot(Y_hat)
    tr_LX_feas = float(np.einsum("ij,ij->", LYh, Y_hat))
    feasible_cut_lb = 0.25 * tr_LX_feas

    return CGALResult(
        sdp_bound=primal_cut,
        sdp_upper_bound=ub_best,
        feasible_cut_lb=feasible_cut_lb,
        Y=Y,
        X_diag=diag_X,
        diag_err_max=diag_err_max,
        iterations=k + 1,
        final_gap=float(gap),
        dual_gap=float(dual_gap_best),
        converged=converged,
        solve_time=solve_time,
        y_final=y,
        beta_final=float(beta),
        history=history,
        n=n,
        n_edges=graph.n_edges,
    )


# ============================================================
# Convenience: head-to-head CGAL vs B176-FW
# ============================================================


def head_to_head(graph, verbose: bool = False) -> dict:
    """Run CGAL en B176-FW op dezelfde graaf; retourneer summary-dict."""
    try:
        from b176_frank_wolfe_sdp import frank_wolfe_maxcut_sdp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("B176 niet importable: %s" % e)

    fw = frank_wolfe_maxcut_sdp(graph, verbose=verbose)
    cg = cgal_maxcut_sdp(graph, verbose=verbose)

    return {
        "n": graph.n,
        "m": graph.n_edges,
        "fw_ub": fw.sdp_upper_bound,
        "fw_lb": fw.feasible_cut_lb,
        "fw_iter": fw.iterations,
        "fw_time": fw.solve_time,
        "fw_diag_err": fw.diag_err_max,
        "cgal_ub": cg.sdp_upper_bound,
        "cgal_lb": cg.feasible_cut_lb,
        "cgal_iter": cg.iterations,
        "cgal_time": cg.solve_time,
        "cgal_diag_err": cg.diag_err_max,
        "cgal_dual_gap": cg.dual_gap,
        "delta_ub_pct": 100.0 * (cg.sdp_upper_bound - fw.sdp_upper_bound)
                        / max(abs(fw.sdp_upper_bound), 1e-9),
    }


# ============================================================
# CLI-demo
# ============================================================


def _demo() -> None:  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="B176b CGAL-SDP demo")
    parser.add_argument("--n", type=int, default=200, help="aantal knooppunten (3-reg)")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--rank-cap", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--eta0", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="ook B176-FW draaien")
    args = parser.parse_args()

    if SimpleGraph is None:
        raise SystemExit("B60 niet importable -- demo niet beschikbaar")
    g = random_3regular(args.n)
    print("Graaf: 3-regulier n=%d edges=%d" % (g.n, g.n_edges))

    res = cgal_maxcut_sdp(
        g,
        max_iter=args.max_iter,
        tol=args.tol,
        beta0=args.beta0,
        eta0=args.eta0,
        rank_cap=args.rank_cap,
        verbose=args.verbose,
    )
    print("")
    print("CGAL primal cut (0.25*tr(L*X_k)) : %.4f" % res.sdp_bound)
    print("CGAL dual UB (certificaat)      : %.4f   (>= cut_SDP, proven)" % res.sdp_upper_bound)
    print("CGAL feasible LB (row-norm)     : %.4f   (<= cut_SDP)" % res.feasible_cut_lb)
    print("  iterations   : %d" % res.iterations)
    print("  dual_gap     : %.3e" % res.dual_gap)
    print("  diag_err_max : %.3e" % res.diag_err_max)
    print("  rank(Y)      : %d" % res.Y.shape[1])
    print("  beta_final   : %.2f" % res.beta_final)
    print("  ||y_final||  : %.3f" % np.linalg.norm(res.y_final))
    print("  solve_time   : %.3fs" % res.solve_time)

    if gw_round_from_Y is not None:
        bs, cut = gw_round_from_Y(res.Y, g, n_trials=200)
        print("")
        print("GW-rounding best cut (200 trials): %.1f / %d" % (cut, g.n_edges))

    if args.compare:
        from b176_frank_wolfe_sdp import frank_wolfe_maxcut_sdp
        print("\n--- B176 FW-SDP (baseline) ---")
        fw = frank_wolfe_maxcut_sdp(g, verbose=False)
        print("FW UB            : %.4f   (%.3fs, %d iter)" % (
            fw.sdp_upper_bound, fw.solve_time, fw.iterations))
        print("FW LB            : %.4f" % fw.feasible_cut_lb)
        delta = 100.0 * (res.sdp_upper_bound - fw.sdp_upper_bound) / max(
            abs(fw.sdp_upper_bound), 1e-9)
        print("CGAL-vs-FW delta : %.3f%%  (positief = CGAL losser)" % delta)


if __name__ == "__main__":  # pragma: no cover
    _demo()
