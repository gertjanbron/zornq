#!/usr/bin/env python3
"""
B39 Experiment Runner: TRG / HOTRG — Tensor Renormalization Group voor 2D QAOA MaxCut

Experimenten:
  1. Ising benchmark: TRG en HOTRG vs exact op kleine grids
  2. TRG vs HOTRG nauwkeurigheid bij varierende chi_max
  3. QAOA 2D exact: correctheid en ratios op 2D grids
  4. Schaalbaarheid: TRG/HOTRG timing vs grid grootte
  5. QAOA parameter scan: optimale gamma/beta voor 2D grids
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trg_hotrg import (
    ising_partition_trg, ising_free_energy_exact,
    qaoa_2d_exact, qaoa_2d_ratio,
    trg_qaoa_cost,
    build_qaoa_tensor_grid, trg_contract, hotrg_contract,
)


def experiment_1_ising_benchmark():
    """Exp 1: Ising partitie functie — TRG en HOTRG vs exact.

    Benchmark op kleine grids met periodieke BC.
    Verwacht: exacte match voor 2x2 en 4x4 bij voldoende chi.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Ising partitie functie benchmark")
    print("=" * 70)

    results = []
    for Lx, Ly in [(2, 2), (3, 3), (4, 4)]:
        n = Lx * Ly
        for beta in [0.1, 0.44, 1.0]:
            exact = ising_free_energy_exact(Lx, Ly, beta)
            for method in ['trg', 'hotrg']:
                for chi in [4, 8, 16]:
                    t0 = time.time()
                    approx = ising_partition_trg(Lx, Ly, beta, chi_max=chi, method=method)
                    dt = time.time() - t0
                    err = abs(approx - exact)
                    rel_err = err / abs(exact) if abs(exact) > 0 else 0
                    results.append({
                        'grid': f'{Lx}x{Ly}', 'n': n, 'beta': beta,
                        'method': method, 'chi': chi,
                        'exact': exact, 'approx': approx,
                        'abs_err': err, 'rel_err': rel_err, 'time': dt
                    })

    # Print tabel
    print(f"\n{'Grid':>5s} {'beta':>5s} {'Method':>6s} {'chi':>4s} "
          f"{'Exact':>10s} {'Approx':>10s} {'AbsErr':>10s} {'RelErr':>10s} {'Time':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['grid']:>5s} {r['beta']:>5.2f} {r['method']:>6s} {r['chi']:>4d} "
              f"{r['exact']:>10.6f} {r['approx']:>10.6f} {r['abs_err']:>10.2e} "
              f"{r['rel_err']:>10.2e} {r['time']:>7.4f}s")

    # Samenvatting
    exact_cases = [r for r in results if r['abs_err'] < 1e-10]
    print(f"\nExact (err < 1e-10): {len(exact_cases)}/{len(results)} cases")
    hotrg_better = sum(1 for i in range(0, len(results), 2)
                       if i+1 < len(results) and
                       results[i]['method'] == 'trg' and results[i+1]['method'] == 'hotrg' and
                       results[i+1]['abs_err'] <= results[i]['abs_err'])
    print(f"HOTRG <= TRG error: {hotrg_better} vergelijkbare paren")

    return results


def experiment_2_chi_convergence():
    """Exp 2: Convergentie van TRG en HOTRG met toenemende chi_max.

    Meet hoe snel de fout afneemt als functie van bond dimensie.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Chi-convergentie TRG vs HOTRG")
    print("=" * 70)

    Lx, Ly = 4, 4
    betas = [0.3, 0.44, 0.8]
    chi_values = [4, 8, 12, 16]

    for beta in betas:
        exact = ising_free_energy_exact(Lx, Ly, beta)
        print(f"\n{Lx}x{Ly}, beta={beta:.2f}, exact={exact:.8f}")
        print(f"  {'chi':>4s}  {'TRG err':>12s}  {'HOTRG err':>12s}  {'TRG time':>10s}  {'HOTRG time':>10s}")

        for chi in chi_values:
            t0 = time.time()
            trg_val = ising_partition_trg(Lx, Ly, beta, chi_max=chi, method='trg')
            t_trg = time.time() - t0

            t0 = time.time()
            hotrg_val = ising_partition_trg(Lx, Ly, beta, chi_max=chi, method='hotrg')
            t_hotrg = time.time() - t0

            trg_err = abs(trg_val - exact)
            hotrg_err = abs(hotrg_val - exact)

            print(f"  {chi:>4d}  {trg_err:>12.4e}  {hotrg_err:>12.4e}  "
                  f"{t_trg:>9.4f}s  {t_hotrg:>9.4f}s")


def experiment_3_qaoa_2d():
    """Exp 3: QAOA 2D exact — correctheid en approximation ratios.

    Vergelijk QAOA cost op 2D grids met bekende waarden.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: QAOA 2D exact op kleine grids")
    print("=" * 70)

    # Optimale p=1 parameters voor 2D grids
    # beta* ≈ 1.1778 (universeel), gamma* ≈ 0.88 / avg_degree
    gamma_star = 0.88 / 4.0  # avg_degree ≈ 4 voor grote 2D grid (per. BC)
    beta_star = 1.1778

    print(f"\nOptimale parameters: gamma*={gamma_star:.4f}, beta*={beta_star:.4f}")
    print(f"\n{'Grid':>5s} {'n':>3s} {'m':>3s} {'Cost':>8s} {'MaxCut':>8s} {'Ratio':>7s} {'Time':>8s}")
    print("-" * 55)

    for Lx, Ly in [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]:
        n = Lx * Ly
        # Aantal edges met periodieke BC
        m = 2 * n  # elke site heeft 4 buren, elke edge dubbel geteld

        t0 = time.time()
        cost = qaoa_2d_exact(Lx, Ly, 1, [gamma_star], [beta_star])
        dt = time.time() - t0

        ratio = cost / m if m > 0 else 0

        print(f"{Lx}x{Ly:>3d} {n:>3d} {m:>3d} {cost:>8.4f} {m:>8d} {ratio:>7.4f} {dt:>7.3f}s")

    # Parameter scan voor 3x3
    print("\nParameter scan voor 3x3 grid (p=1):")
    Lx, Ly = 3, 3
    m = 2 * Lx * Ly
    best_cost = 0
    best_params = (0, 0)

    gammas_scan = np.linspace(0.05, 0.8, 16)
    betas_scan = np.linspace(0.3, 1.8, 16)

    for g in gammas_scan:
        for b in betas_scan:
            cost = qaoa_2d_exact(Lx, Ly, 1, [g], [b])
            if cost > best_cost:
                best_cost = cost
                best_params = (g, b)

    print(f"  Best cost: {best_cost:.4f} (ratio: {best_cost/m:.4f})")
    print(f"  Best params: gamma={best_params[0]:.4f}, beta={best_params[1]:.4f}")


def experiment_4_scaling():
    """Exp 4: Schaalbaarheid van TRG/HOTRG contractie.

    Meet wallclock tijd als functie van grid grootte en chi_max.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Schaalbaarheid TRG/HOTRG")
    print("=" * 70)

    beta = 0.44
    chi = 8

    print(f"\nFixed chi={chi}, beta={beta}")
    print(f"{'Grid':>6s} {'n':>4s} {'TRG time':>10s} {'HOTRG time':>10s}")
    print("-" * 40)

    for Lx, Ly in [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]:
        n = Lx * Ly

        t_trg_str = "N/A"
        t_hotrg_str = "N/A"

        try:
            t0 = time.time()
            ising_partition_trg(Lx, Ly, beta, chi_max=chi, method='trg')
            t_trg_str = f"{time.time() - t0:>8.4f}s"
        except Exception:
            t_trg_str = "   FAIL  "

        try:
            t0 = time.time()
            ising_partition_trg(Lx, Ly, beta, chi_max=chi, method='hotrg')
            t_hotrg_str = f"{time.time() - t0:>8.4f}s"
        except Exception:
            t_hotrg_str = "   FAIL  "

        print(f"{Lx}x{Ly:>3d} {n:>4d} {t_trg_str:>10s} {t_hotrg_str:>10s}")


def experiment_5_qaoa_trg_vs_exact():
    """Exp 5: TRG-based QAOA cost vs exact voor kleine grids.

    Vergelijk trg_qaoa_cost met qaoa_2d_exact.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: TRG QAOA cost vs exact")
    print("=" * 70)

    gamma_star = 0.88 / 4.0
    beta_star = 1.1778

    print(f"\n{'Grid':>5s} {'Exact':>10s} {'TRG(chi=8)':>12s} {'TRG(chi=16)':>12s} {'Err(8)':>10s} {'Err(16)':>10s}")
    print("-" * 65)

    for Lx, Ly in [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]:
        n = Lx * Ly
        exact = qaoa_2d_exact(Lx, Ly, 1, [gamma_star], [beta_star])

        trg8 = trg_qaoa_cost(Lx, Ly, 1, [gamma_star], [beta_star], chi_max=8)
        trg16 = trg_qaoa_cost(Lx, Ly, 1, [gamma_star], [beta_star], chi_max=16)

        err8 = abs(trg8 - exact)
        err16 = abs(trg16 - exact)

        print(f"{Lx}x{Ly:>3d} {exact:>10.4f} {trg8:>12.4f} {trg16:>12.4f} "
              f"{err8:>10.2e} {err16:>10.2e}")


if __name__ == '__main__':
    print("=" * 70)
    print("B39: TRG / HOTRG - Experiment Suite")
    print("=" * 70)

    experiment_1_ising_benchmark()
    experiment_2_chi_convergence()
    experiment_3_qaoa_2d()
    experiment_4_scaling()
    experiment_5_qaoa_trg_vs_exact()

    print("\n" + "=" * 70)
    print("ALLE EXPERIMENTEN VOLTOOID")
    print("=" * 70)
