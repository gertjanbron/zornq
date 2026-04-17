#\!/usr/bin/env python3
"""
p_zne.py - B76: p-ZNE — Richardson Extrapolatie op Circuitdiepte.

Bereken geoptimaliseerde QAOA ratios bij p=1, p=2 (exact via TransverseQAOA),
en extrapoleer naar p->inf via:
  1. Lineaire fit: r(p) = a + b/p
  2. Kwadratische fit: r(p) = a + b/p + c/p^2
  3. Richardson extrapolatie: combinatie van twee p-waarden
  4. Pade [1/1] approximant: r(p) = (a + b/p) / (1 + c/p)
  5. Exponentieel: r(p) = r_inf - A * exp(-alpha * p)

Op bipartite grids is de exacte p->inf limiet = 1.0 (GW optimum),
wat een perfecte validatie-target geeft.

Bouwt voort op: transverse_contraction.py (B26), homotopy_optimizer.py (B71)
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def collect_p_data(Lx, Ly, p_max=2, n_gamma=12, n_beta=12, verbose=True):
    """Verzamel geoptimaliseerde ratio's voor p=1..p_max.

    Gebruikt p-continuation (B71) voor efficiente optimalisatie.
    """
    from homotopy_optimizer import HomotopyQAOA

    h = HomotopyQAOA(Lx, Ly, verbose=False)
    data = []

    if verbose:
        print(f"  [B76] Data verzamelen: {Lx}x{Ly} p=1..{p_max}")

    gammas = None
    betas = None

    for p in range(1, p_max + 1):
        t0 = time.time()

        if p == 1:
            ratio, gammas, betas = h.optimize_at_lambda(
                p, 1.0, n_gamma=n_gamma, n_beta=n_beta, maxiter=200)
        else:
            init_g = gammas + [gammas[-1]]
            init_b = betas + [betas[-1]]
            ratio, gammas, betas = h.optimize_at_lambda(
                p, 1.0, init_gammas=init_g, init_betas=init_b, maxiter=150)

        dt = time.time() - t0
        data.append({
            "p": p,
            "ratio": float(ratio),
            "gammas": [float(g) for g in gammas],
            "betas": [float(b) for b in betas],
            "time": dt,
        })

        if verbose:
            print(f"    p={p}: ratio={ratio:.6f} ({dt:.1f}s)")

    return data


def extrapolate(data, gw_ratio=None, verbose=True):
    """Extrapoleer ratio naar p->inf via meerdere methoden.

    Args:
        data: lijst van dicts met "p" en "ratio"
        gw_ratio: GW-bound/optimale ratio (voor validatie)
        verbose: print resultaten

    Returns:
        dict met extrapolaties per methode
    """
    ps = np.array([d["p"] for d in data], dtype=float)
    rs = np.array([d["ratio"] for d in data])

    if verbose:
        print(f"\n  [B76] Extrapolatie vanuit {len(data)} datapunten:")
        for d in data:
            p_val = d["p"]
            ratio_val = d["ratio"]
            print(f"    p={p_val}: ratio={ratio_val:.6f}")

    results = {}
    x = 1.0 / ps  # extrapoleren naar x=0 (p=inf)

    # === 1. Lineaire fit: r = a + b/p ===
    if len(data) >= 2:
        coeffs = np.polyfit(x, rs, 1)
        r_inf = coeffs[-1]
        results["linear"] = {
            "ratio_inf": float(r_inf),
            "coeffs": coeffs.tolist(),
            "formula": f"r(p) = {coeffs[1]:.6f} + {coeffs[0]:.6f}/p",
        }
        if verbose:
            print(f"    Lineair:      r(inf) = {r_inf:.6f}")

    # === 2. Kwadratische fit: r = a + b/p + c/p^2 ===
    if len(data) >= 3:
        coeffs = np.polyfit(x, rs, 2)
        r_inf = coeffs[-1]
        results["quadratic"] = {
            "ratio_inf": float(r_inf),
            "coeffs": coeffs.tolist(),
        }
        if verbose:
            print(f"    Kwadratisch:  r(inf) = {r_inf:.6f}")

    # === 3. Richardson extrapolatie ===
    # Aanname: fout ~ 1/p, dus r_inf = (p2*r2 - p1*r1) / (p2 - p1)
    if len(data) >= 2:
        # Gebruik de twee hoogste p-waarden
        sorted_data = sorted(data, key=lambda d: d["p"])
        d1, d2 = sorted_data[-2], sorted_data[-1]
        p1, p2 = d1["p"], d2["p"]
        r1, r2 = d1["ratio"], d2["ratio"]

        if p2 != p1:
            r_rich = (p2 * r2 - p1 * r1) / (p2 - p1)
            results["richardson"] = {
                "ratio_inf": float(r_rich),
                "p_pair": [p1, p2],
                "formula": f"({p2}*{r2:.6f} - {p1}*{r1:.6f}) / ({p2}-{p1})",
            }
            if verbose:
                print(f"    Richardson:   r(inf) = {r_rich:.6f} "
                      f"(p={p1},{p2})")

    # === 4. Pade [1/1]: r(p) = (a0 + a1/p) / (1 + b1/p) ===
    # Herschrijf: r * (1 + b1/p) = a0 + a1/p
    # r = a0 + (a1 - r*b1)/p
    # Met 2 datapunten: 3 unknowns, underdetermined
    # Gebruik constraint: r(0) is geldig (r bij p=inf = a0)
    if len(data) >= 2:
        # Gebruik 2-punt Pade: stel r(inf) = L, dan
        # r(p) = L + (r(p) - L) correctie
        # Simpelere aanpak: exponentieel fit
        pass

    # === 5. Exponentieel: r(p) = r_inf - A * exp(-alpha * p) ===
    if len(data) >= 2:
        # Twee-parameter fit: r_inf is vrij, A en alpha volgen
        # Met 2 punten: r1 = L - A*exp(-a*p1), r2 = L - A*exp(-a*p2)
        # Probeer r_inf in [max(ratios), 1.0] en minimaliseer residual
        from scipy.optimize import minimize_scalar, minimize

        def exp_residual(params):
            L, A, alpha = params
            pred = L - A * np.exp(-alpha * ps)
            return np.sum((pred - rs) ** 2)

        # Multi-start
        best_res = None
        for L_init in np.linspace(max(rs) + 0.01, 1.0, 5):
            for alpha_init in [0.5, 1.0, 2.0]:
                A_init = (L_init - rs[0]) / max(np.exp(-alpha_init * ps[0]), 1e-10)
                try:
                    res = minimize(exp_residual, [L_init, A_init, alpha_init],
                                   bounds=[(max(rs), 1.5), (0, 10), (0.01, 10)],
                                   method="L-BFGS-B",
                                   options={"maxiter": 100})
                    if best_res is None or res.fun < best_res.fun:
                        best_res = res
                except Exception:
                    pass

        if best_res is not None and best_res.success:
            L, A, alpha = best_res.x
            r_exp = float(L)
            results["exponential"] = {
                "ratio_inf": r_exp,
                "A": float(A),
                "alpha": float(alpha),
                "formula": f"r(p) = {L:.6f} - {A:.6f} * exp(-{alpha:.4f}*p)",
            }
            if verbose:
                print(f"    Exponentieel: r(inf) = {r_exp:.6f} "
                      f"(A={A:.4f}, alpha={alpha:.4f})")

    # === Samenvatting ===
    if verbose and results:
        ext_values = [v["ratio_inf"] for v in results.values()]
        mean_ext = np.mean(ext_values)
        spread = np.max(ext_values) - np.min(ext_values)
        print(f"\n    Gemiddelde:   r(inf) = {mean_ext:.6f} "
              f"+/- {spread/2:.6f}")
        if gw_ratio is not None:
            print(f"    GW optimum:   {gw_ratio:.6f}")
            for name, res in results.items():
                err = abs(res["ratio_inf"] - gw_ratio)
                pct = err / gw_ratio * 100
                print(f"    {name:14s}: fout = {err:.6f} "
                      f"({pct:.2f}%)")

    return results


def p_zne(Lx, Ly, p_max=2, n_gamma=12, n_beta=12, verbose=True):
    """Volledige p-ZNE pipeline: data verzamelen + extrapoleren.

    Returns:
        dict met data, extrapolaties, en GW-bound
    """
    t0 = time.time()

    if verbose:
        print(f"  [B76] p-ZNE: {Lx}x{Ly} p=1..{p_max}")
        print()

    # Stap 1: Verzamel E(p) data
    data = collect_p_data(Lx, Ly, p_max=p_max,
                          n_gamma=n_gamma, n_beta=n_beta, verbose=verbose)

    # Stap 2: GW bound (als referentie)
    gw_ratio = None
    try:
        from ws_qaoa import gw_sdp_solve
        n = Lx * Ly
        edges = []
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                if y < Ly - 1:
                    edges.append((node, node + 1, 1.0))
                if x < Lx - 1:
                    edges.append((node, node + Ly, 1.0))
        n_edges = len(edges)
        gw = gw_sdp_solve(n, edges, verbose=False)
        gw_ratio = gw["best_cut"] / n_edges
        if verbose:
            sdp_ratio = gw["sdp_bound"] / n_edges
            print(f"\n    GW: cut={gw['best_cut']}/{n_edges} "
                  f"({gw_ratio:.6f}), SDP={sdp_ratio:.6f}")
    except Exception as e:
        if verbose:
            print(f"    GW niet beschikbaar: {e}")

    # Stap 3: Extrapoleer
    results = extrapolate(data, gw_ratio=gw_ratio, verbose=verbose)

    total_time = time.time() - t0

    if verbose:
        print(f"\n    Totale tijd: {total_time:.1f}s")

    return {
        "data": data,
        "extrapolations": results,
        "gw_ratio": gw_ratio,
        "elapsed": total_time,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B76: p-ZNE — Richardson Extrapolatie op Circuitdiepte")
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=3)
    parser.add_argument("--p-max", type=int, default=2,
                        help="Maximale p voor data (default: 2)")
    parser.add_argument("--n-gamma", type=int, default=12)
    parser.add_argument("--n-beta", type=int, default=12)
    args = parser.parse_args()

    p_zne(args.Lx, args.Ly, p_max=args.p_max,
          n_gamma=args.n_gamma, n_beta=args.n_beta)


if __name__ == "__main__":
    main()
