#!/usr/bin/env python3
"""B176c benchmark -- GPU-eigsh voor CGAL-SDP, schaalpanel + backend-head-to-head.

Doel
----
Kwantificeer de winst van GPU-backed eigsh (LOBPCG / Lanczos) t.o.v.
scipy-ARPACK in twee dimensies:

  1. **Per-call wall-time** op een synthetische shifted-Laplacian-sequentie
     met realistische z-drift (B176b-iteraties zijn goed voorbeeldig).
  2. **End-to-end CGAL-SDP wall-time** op 3-reguliere grafen
     n in {2000, 5000, 10000}, 3 seeds.

Sectie-indeling
---------------
  1. Per-call micro-benchmark (geen CGAL-infra): meet eigsh-only voor de
     4 backends x (cold, warm-start) combinaties.
  2. CGAL-end-to-end: run B176b's cgal_maxcut_sdp met een backend-override
     en meet totale wall-time + SDP-bound.
  3. Emit CSV + eenvoudige matplotlib-plot (log-log wall-time vs n, per
     backend). PNG + CSV in docs/paper/data/.

Gebruik
-------
    python b176c_benchmark.py                       # alle secties, full-scale
    python b176c_benchmark.py --quick               # n <= 500
    python b176c_benchmark.py --only micro          # alleen sectie 1
    python b176c_benchmark.py --only scale          # alleen sectie 2
    python b176c_benchmark.py --backends scipy_arpack,scipy_lobpcg

Op een laptop ZONDER CuPy draaien alleen de twee scipy-backends; op een
laptop MET CuPy + GTX 1650 draait het volledige 4x3x3 panel (ca. 15-30
min totale wall-time volgens back-of-envelope).

Output-files:
    ../docs/paper/data/b176c_micro.csv
    ../docs/paper/data/b176c_scale.csv
    ../docs/paper/data/b176c_walltime.png     (indien matplotlib aanwezig)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import random_3regular
from b176_frank_wolfe_sdp import graph_laplacian
from b176c_gpu_eigsh import (
    available_backends,
    clear_gpu_cache,
    cupy_available,
    gpu_eigsh_smallest,
)


# ============================================================
# Paths
# ============================================================


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "paper", "data"))


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _hdr(title: str) -> None:
    bar = "=" * 90
    print(bar)
    print("  " + title)
    print(bar)


# ============================================================
# 1. Per-call micro-benchmark
# ============================================================


def _synth_z_sequence(n: int, n_steps: int, seed: int = 0) -> list[np.ndarray]:
    """Simulate de z-drift zoals in B176b-iteraties.

    z_k = z_{k-1} + step * random_direction(n), met decaying step.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    zs = []
    for k in range(n_steps):
        step = 0.1 / np.sqrt(k + 1.0)
        z = z + step * rng.standard_normal(n)
        zs.append(z.copy())
    return zs


def micro_benchmark(
    ns: list[int],
    backends: list[str],
    seeds: list[int],
    n_steps: int = 10,
    tol: float = 1e-6,
) -> list[dict]:
    """Per-call eigsh-micro-benchmark.

    Voor elke (n, backend, seed):
      * Bouw 3-reg Laplacian L
      * Maak z-sequentie van n_steps elementen
      * Run eigsh voor elk z:
          - cold: v0=None voor elk
          - warm: v0 = v_{k-1} vanaf stap 2
      * Meet totale wall-time + gemiddelde matvec-count.
    """
    rows = []
    for n in ns:
        for seed in seeds:
            g = random_3regular(n, seed=seed)
            L = graph_laplacian(g).tocsr()
            zs = _synth_z_sequence(n, n_steps, seed=seed)

            for backend in backends:
                for warm in (False, True):
                    clear_gpu_cache()
                    v_prev = None
                    walls = []
                    nmvs = []
                    lams = []
                    for z in zs:
                        t0 = time.perf_counter()
                        res = gpu_eigsh_smallest(
                            L, z, v0=(v_prev if warm else None),
                            tol=tol, backend=backend)
                        walls.append(time.perf_counter() - t0)
                        nmvs.append(res.info.get("n_matvec") or 0)
                        lams.append(res.lam)
                        v_prev = res.v

                    row = {
                        "n": n, "seed": seed, "backend": backend,
                        "warm_start": int(warm),
                        "n_steps": n_steps,
                        "tol": tol,
                        "wall_mean": float(np.mean(walls)),
                        "wall_total": float(np.sum(walls)),
                        "wall_first": float(walls[0]),
                        "wall_steady": float(np.mean(walls[1:]) if len(walls) > 1 else walls[0]),
                        "nmv_mean": float(np.mean(nmvs)),
                        "lam_first": float(lams[0]),
                        "lam_last": float(lams[-1]),
                    }
                    rows.append(row)
                    print(f"  n={n:6d} seed={seed} backend={backend:14s} "
                          f"warm={int(warm)}  wall/call={row['wall_steady']*1e3:7.1f}ms  "
                          f"nmv/call={row['nmv_mean']:6.1f}")
    return rows


# ============================================================
# 2. CGAL end-to-end schaalpanel (optioneel, vereist b176b)
# ============================================================


def _run_cgal_with_backend(graph, backend: str, **cgal_kwargs) -> dict:
    """Run CGAL-SDP met monkeypatched lmo_spectraplex + dual_upper_bound."""
    from b176_frank_wolfe_sdp import lmo_spectraplex as _orig_lmo  # noqa: F401
    import b176_frank_wolfe_sdp as _b176
    import b176b_cgal_sdp as _b176b

    if backend == "scipy_arpack":
        # Default B176b-gedrag -- geen monkeypatch nodig
        from b176b_cgal_sdp import cgal_maxcut_sdp
        t0 = time.perf_counter()
        res = cgal_maxcut_sdp(graph, **cgal_kwargs)
        return {
            "backend": backend, "n": graph.n, "m": graph.n_edges,
            "sdp_ub": res.sdp_upper_bound, "sdp_lb": res.feasible_cut_lb,
            "iter": res.iterations, "solve_time": res.solve_time,
            "wall_time": time.perf_counter() - t0,
            "converged": res.converged,
            "diag_err": res.diag_err_max,
        }

    # Voor de andere backends: monkeypatch lmo_spectraplex via
    # een wrapper die gpu_eigsh_smallest gebruikt.
    orig_lmo = _b176.lmo_spectraplex

    # We hebben in lmo_spectraplex geen directe toegang tot L en z, alleen
    # een matvec. Deze integratie vereist daarom een lichte refactor-patch
    # die in de B176c-integratie-commit wordt geleverd (B176c-integration
    # PR). Voor deze benchmark gebruiken we een simpele wrapper die alleen
    # de outer eigsh-call vervangt door te reconstrueren (kost 1 extra L-matvec
    # per dim om G dense te bouwen -- onbetaalbaar voor n > 1000).
    # Oplossing: deze benchmark meet alleen 'scipy_arpack' end-to-end via
    # deze pad; de andere backends meten we via micro_benchmark() die direct
    # (L, z) aangeeft. Voor een echte end-to-end-vergelijking zie de
    # B176c-integration PR die lmo_spectraplex herschrijft om (L, z) door
    # te geven.
    raise NotImplementedError(
        f"End-to-end CGAL met backend={backend!r} vereist de B176c-integration "
        f"PR (lmo_spectraplex refactor). Gebruik voorlopig --only micro."
    )


def scale_panel(
    ns: list[int], seeds: list[int], backends: list[str], max_iter: int = 200,
) -> list[dict]:
    rows = []
    for n in ns:
        for seed in seeds:
            g = random_3regular(n, seed=seed)
            for backend in backends:
                if backend != "scipy_arpack":
                    print(f"  [skip] n={n} seed={seed} backend={backend}: "
                          f"wacht op B176c-integration PR")
                    continue
                try:
                    row = _run_cgal_with_backend(
                        g, backend, max_iter=max_iter, tol=1e-4, verbose=False)
                    rows.append(row)
                    print(f"  n={n:6d} seed={seed} backend={backend:14s}  "
                          f"UB={row['sdp_ub']:8.2f}  iter={row['iter']:3d}  "
                          f"wall={row['wall_time']:7.1f}s")
                except Exception as e:
                    print(f"  [FAIL] n={n} seed={seed} backend={backend}: {e}")
    return rows


# ============================================================
# 3. Output-helpers
# ============================================================


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"  (geen rijen voor {path})")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  [csv] {path}  ({len(rows)} rows)")


def plot_wall_vs_n(rows: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [plot-skip] matplotlib niet beschikbaar: {e}")
        return
    # Group by (backend, warm_start) over n
    from collections import defaultdict
    series: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        series[(r["backend"], r["warm_start"])][r["n"]].append(r["wall_steady"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for (backend, warm), perN in sorted(series.items()):
        ns = sorted(perN.keys())
        means = [np.mean(perN[n]) for n in ns]
        label = f"{backend}" + (" (warm)" if warm else " (cold)")
        ls = "-" if warm else "--"
        ax.loglog(ns, means, marker="o", linestyle=ls, label=label)
    ax.set_xlabel("n (graph size)")
    ax.set_ylabel("wall-time per eigsh call [s]")
    ax.set_title("B176c: per-call eigsh wall-time vs n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  [png] {path}")


# ============================================================
# CLI
# ============================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="B176c GPU-eigsh benchmark")
    ap.add_argument("--quick", action="store_true",
                    help="Kleine n voor snelle sandbox-runs")
    ap.add_argument("--only", choices=["micro", "scale"], default=None,
                    help="Alleen een subset draaien")
    ap.add_argument("--backends", type=str, default=None,
                    help="Komma-gescheiden backend-lijst; default = alle beschikbare")
    ap.add_argument("--seeds", type=int, default=3, help="Aantal seeds per n")
    ap.add_argument("--n-steps", type=int, default=10,
                    help="Lengte van z-sequentie in micro-benchmark")
    args = ap.parse_args()

    _ensure_dirs()

    backends = (args.backends.split(",") if args.backends
                else available_backends())
    print(f"CuPy beschikbaar : {cupy_available()}")
    print(f"Backends         : {backends}")

    ns_micro = [200, 500, 1000] if args.quick else [500, 1000, 2000, 5000, 10000]
    ns_scale = [200, 500] if args.quick else [2000, 5000, 10000]
    seeds = list(range(args.seeds))

    all_rows_micro: list[dict] = []
    all_rows_scale: list[dict] = []

    if args.only in (None, "micro"):
        _hdr("1. Per-call micro-benchmark")
        all_rows_micro = micro_benchmark(
            ns=ns_micro, backends=backends, seeds=seeds,
            n_steps=args.n_steps)
        write_csv(all_rows_micro,
                  os.path.join(DATA_DIR, "b176c_micro.csv"))
        plot_wall_vs_n(
            all_rows_micro,
            os.path.join(DATA_DIR, "b176c_walltime.png"))

    if args.only in (None, "scale"):
        _hdr("2. CGAL end-to-end schaalpanel (scipy_arpack baseline)")
        all_rows_scale = scale_panel(
            ns=ns_scale, seeds=seeds, backends=backends)
        write_csv(all_rows_scale,
                  os.path.join(DATA_DIR, "b176c_scale.csv"))

    _hdr("Samenvatting")
    if all_rows_micro:
        print(f"  Micro-rijen     : {len(all_rows_micro)}")
        # Speedup-tabel: voor elke n, toon scipy_arpack(cold) / cupy_lobpcg(warm)
        from collections import defaultdict
        by_n: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        for r in all_rows_micro:
            key = f"{r['backend']}/{'warm' if r['warm_start'] else 'cold'}"
            by_n[r["n"]][key].append(r["wall_steady"])
        print(f"  {'n':>6s}  {'baseline (arpack cold)':>25s}  "
              f"{'best other':>25s}  {'speedup':>10s}")
        for n in sorted(by_n.keys()):
            means = {k: float(np.mean(v)) for k, v in by_n[n].items()}
            if "scipy_arpack/cold" not in means:
                continue
            base = means["scipy_arpack/cold"]
            others = {k: v for k, v in means.items() if k != "scipy_arpack/cold"}
            if not others:
                continue
            best_k = min(others, key=others.get)
            best_v = others[best_k]
            speedup = base / best_v if best_v > 0 else float("inf")
            print(f"  {n:>6d}  {base*1e3:>22.2f}ms  "
                  f"{best_k+' '+f'{best_v*1e3:.2f}ms':>25s}  {speedup:>9.2f}x")
    if all_rows_scale:
        print(f"  Scale-rijen     : {len(all_rows_scale)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
