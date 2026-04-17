#!/usr/bin/env python3
"""
B34 Experiment Runner: Mid-Circuit Measurement / Adaptieve Projectie

Experimenten:
  1. MPS QAOA correctheid: MPS vs state vector op kleine ketens
  2. Mid-circuit meting effect: fideliteit en cost na meting op verschil. posities
  3. Multi-branch sampling: convergentie van verwachtingswaarden
  4. Adaptieve meetpunt selectie: entropie-gebaseerde selectie vs random
  5. Bond-dimensie reductie: chi-verloop met/zonder metingen
  6. Schaalbaarheid: grotere ketens met mid-circuit metingen
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from midcircuit_measurement import (
    MPS, mps_compress,
    apply_phase_gate, apply_mixer_gate, apply_qaoa_layer,
    qaoa_mps, qaoa_statevector,
    measure_qubit, measure_qubits, split_mps_at_measured,
    select_measurement_sites, adaptive_measurement_schedule,
    multi_branch_expectation,
    expectation_z, expectation_zz,
    maxcut_cost_mps, maxcut_cost_statevector,
)


def experiment_1_mps_correctheid():
    """Exp 1: MPS QAOA vs state vector referentie.

    Vergelijk fideliteit en MaxCut cost op kleine 1D-ketens.
    """
    print("=" * 70)
    print("EXPERIMENT 1: MPS QAOA correctheid vs state vector")
    print("=" * 70)

    results = []
    for n in [4, 6, 8, 10]:
        edges = [(i, i + 1) for i in range(n - 1)]
        for p in [1, 2, 3]:
            rng = np.random.default_rng(42)
            gammas = [0.3 + 0.1 * rng.standard_normal() for _ in range(p)]
            betas = [0.7 + 0.1 * rng.standard_normal() for _ in range(p)]

            if n <= 10:
                # State vector referentie
                t0 = time.time()
                psi_sv = qaoa_statevector(n, edges, gammas, betas)
                t_sv = time.time() - t0
                cost_sv = maxcut_cost_statevector(psi_sv, n, edges)
            else:
                psi_sv = None
                cost_sv = None
                t_sv = 0

            # MPS exact
            t0 = time.time()
            mps = qaoa_mps(n, edges, gammas, betas)
            t_mps = time.time() - t0
            cost_mps = maxcut_cost_mps(mps, edges)

            fid = None
            if psi_sv is not None:
                psi_mps = mps.to_statevector()
                fid = abs(np.dot(psi_sv.conj(), psi_mps)) ** 2

            results.append({
                'n': n, 'p': p,
                'cost_sv': cost_sv, 'cost_mps': cost_mps,
                'fid': fid, 'max_chi': mps.max_bond_dim(),
                't_sv': t_sv, 't_mps': t_mps,
            })

    print(f"\n{'n':>3s} {'p':>2s} {'Cost(SV)':>9s} {'Cost(MPS)':>10s} "
          f"{'Fidelity':>10s} {'MaxChi':>7s} {'T(SV)':>8s} {'T(MPS)':>8s}")
    print("-" * 65)
    for r in results:
        fid_str = f"{r['fid']:.8f}" if r['fid'] is not None else "N/A"
        csv_str = f"{r['cost_sv']:.4f}" if r['cost_sv'] is not None else "N/A"
        print(f"{r['n']:>3d} {r['p']:>2d} {csv_str:>9s} {r['cost_mps']:>10.4f} "
              f"{fid_str:>10s} {r['max_chi']:>7d} {r['t_sv']:>7.4f}s {r['t_mps']:>7.4f}s")

    return results


def experiment_2_midcircuit_effect():
    """Exp 2: Effect van mid-circuit meting op fideliteit en cost.

    Meet 1 qubit na laag p/2, vergelijk met ongemeten referentie.
    Test op verschillende meetposities en uitkomsten.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Mid-circuit meting effect op cost en fideliteit")
    print("=" * 70)

    n = 8
    edges = [(i, i + 1) for i in range(n - 1)]
    p = 4
    rng_params = np.random.default_rng(42)
    gammas = [0.3 + 0.1 * rng_params.standard_normal() for _ in range(p)]
    betas = [0.7 + 0.1 * rng_params.standard_normal() for _ in range(p)]

    # Referentie: zonder meting
    mps_ref = qaoa_mps(n, edges, gammas, betas)
    cost_ref = maxcut_cost_mps(mps_ref, edges)
    print(f"\nReferentie (geen meting): cost = {cost_ref:.4f}, max_chi = {mps_ref.max_bond_dim()}")

    # Meet na laag 2
    measure_after = 2
    print(f"\nMeting na laag {measure_after}:")
    print(f"{'Site':>5s} {'Out':>4s} {'Cost':>8s} {'dCost':>8s} {'MaxChi':>7s} {'BornP':>7s}")
    print("-" * 45)

    for site in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        for outcome in [0, 1]:
            # Run QAOA tot laag measure_after
            mps = qaoa_mps(n, edges, gammas[:measure_after], betas[:measure_after])

            # Meet
            _, born_p = measure_qubit(mps, site, outcome=outcome)

            # Ga door met resterende lagen
            for layer in range(measure_after, p):
                apply_qaoa_layer(mps, edges, gammas[layer], betas[layer])
            mps.normalize()

            cost = maxcut_cost_mps(mps, edges)
            dcost = cost - cost_ref

            print(f"{site:>5d} {outcome:>4d} {cost:>8.4f} {dcost:>+8.4f} "
                  f"{mps.max_bond_dim():>7d} {born_p:>7.4f}")

    return cost_ref


def experiment_3_multi_branch():
    """Exp 3: Multi-branch sampling convergentie.

    Meet 1 of 2 qubits, middel over branches.
    Vergelijk met exacte referentie.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Multi-branch sampling convergentie")
    print("=" * 70)

    n = 6
    edges = [(i, i + 1) for i in range(n - 1)]
    gammas = [0.3]
    betas = [0.7]
    rng = np.random.default_rng(42)

    # Referentie
    psi_sv = qaoa_statevector(n, edges, gammas, betas)
    cost_exact = maxcut_cost_statevector(psi_sv, n, edges)
    print(f"\nExact cost: {cost_exact:.4f}")

    # Varieer aantal branches en meetposities
    branch_counts = [4, 16, 64, 256]
    measure_configs = [
        ([n // 2], "midden"),
        ([1, n - 2], "rand"),
        ([n // 4, 3 * n // 4], "kwart"),
    ]

    for sites, label in measure_configs:
        print(f"\nMeting sites {sites} ({label}):")
        print(f"  {'Branches':>9s} {'Mean':>8s} {'StdErr':>8s} {'|Err|':>8s}")
        print("  " + "-" * 40)

        for nb in branch_counts:
            mean, stderr, stats = multi_branch_expectation(
                n, edges, gammas, betas,
                measure_sites=sites,
                measure_after_layer=0,
                observable_fn=lambda m: maxcut_cost_mps(m, edges),
                n_branches=nb,
                rng=rng,
            )
            err = abs(mean - cost_exact)
            print(f"  {nb:>9d} {mean:>8.4f} {stderr:>8.4f} {err:>8.4f}")


def experiment_4_adaptive_selectie():
    """Exp 4: Adaptieve meetpunt selectie vs random.

    Vergelijk entropie-gebaseerde selectie met willekeurige selectie.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Adaptieve vs random meetpunt selectie")
    print("=" * 70)

    n = 10
    edges = [(i, i + 1) for i in range(n - 1)]
    p = 3
    rng_params = np.random.default_rng(42)
    gammas = [0.3 + 0.1 * rng_params.standard_normal() for _ in range(p)]
    betas = [0.7 + 0.1 * rng_params.standard_normal() for _ in range(p)]

    # Referentie
    mps_ref = qaoa_mps(n, edges, gammas, betas)
    cost_ref = maxcut_cost_mps(mps_ref, edges)
    print(f"\nReferentie (geen meting): cost = {cost_ref:.4f}")

    # Na laag 1: analyseer entropie
    mps_after_1 = qaoa_mps(n, edges, gammas[:1], betas[:1])
    entropies = mps_after_1.all_bond_entropies()
    print(f"\nBond entropieen na laag 1:")
    for i, e in enumerate(entropies):
        bar = "#" * int(e * 20)
        print(f"  Bond {i:>2d}|{i+1:<2d}: {e:.4f} {bar}")

    # Adaptieve selectie
    for threshold in [0.1, 0.3, 0.5, 0.8]:
        sites, ents = select_measurement_sites(
            mps_after_1.copy(), entropy_threshold=threshold
        )
        print(f"\n  Drempel={threshold:.1f}: {len(sites)} sites geselecteerd: {sites}")

        if sites:
            # Meet en ga door
            rng = np.random.default_rng(123)
            n_trials = 32
            costs = []
            for _ in range(n_trials):
                mps = qaoa_mps(n, edges, gammas[:1], betas[:1])
                measure_qubits(mps, sites, rng=rng)
                for layer in range(1, p):
                    apply_qaoa_layer(mps, edges, gammas[layer], betas[layer])
                mps.normalize()
                costs.append(maxcut_cost_mps(mps, edges))
            mean_cost = np.mean(costs)
            std_cost = np.std(costs) / np.sqrt(n_trials)
            print(f"    Gemiddelde cost: {mean_cost:.4f} +/- {std_cost:.4f} "
                  f"(ref: {cost_ref:.4f}, delta: {mean_cost - cost_ref:+.4f})")

    # Random selectie ter vergelijking
    print(f"\n  Random selectie (2 sites):")
    rng = np.random.default_rng(456)
    n_trials = 32
    for _ in range(3):
        random_sites = sorted(rng.choice(n, size=2, replace=False))
        costs = []
        rng2 = np.random.default_rng(789)
        for _ in range(n_trials):
            mps = qaoa_mps(n, edges, gammas[:1], betas[:1])
            measure_qubits(mps, random_sites, rng=rng2)
            for layer in range(1, p):
                apply_qaoa_layer(mps, edges, gammas[layer], betas[layer])
            mps.normalize()
            costs.append(maxcut_cost_mps(mps, edges))
        mean_cost = np.mean(costs)
        print(f"    Sites {random_sites}: cost = {mean_cost:.4f} "
              f"(delta: {mean_cost - cost_ref:+.4f})")


def experiment_5_chi_reductie():
    """Exp 5: Bond-dimensie reductie door mid-circuit metingen.

    Vergelijk chi-verloop met en zonder metingen tijdens QAOA.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Bond-dimensie reductie door metingen")
    print("=" * 70)

    n = 12
    edges = [(i, i + 1) for i in range(n - 1)]
    p = 6
    rng_params = np.random.default_rng(42)
    gammas = [0.3 + 0.05 * rng_params.standard_normal() for _ in range(p)]
    betas = [0.7 + 0.05 * rng_params.standard_normal() for _ in range(p)]

    # Zonder meting
    print(f"\nZonder metingen (n={n}, p={p}):")
    mps_no_meas = qaoa_mps(n, edges, gammas[:1], betas[:1])
    print(f"  Na laag 1: max_chi = {mps_no_meas.max_bond_dim()}, "
          f"bonds = {mps_no_meas.bond_dims()}")
    for layer in range(1, p):
        apply_qaoa_layer(mps_no_meas, edges, gammas[layer], betas[layer])
        print(f"  Na laag {layer+1}: max_chi = {mps_no_meas.max_bond_dim()}, "
              f"bonds = {mps_no_meas.bond_dims()}")
    cost_no_meas = maxcut_cost_mps(mps_no_meas, edges)
    print(f"  Finale cost: {cost_no_meas:.4f}")

    # Met meting elke 2 lagen
    print(f"\nMet adaptieve metingen (threshold=0.3, elke 2 lagen):")
    rng = np.random.default_rng(42)
    mps_meas = qaoa_mps(n, edges, gammas[:1], betas[:1])
    total_measured = 0
    for layer in range(1, p):
        apply_qaoa_layer(mps_meas, edges, gammas[layer], betas[layer])
        if layer % 2 == 0 and layer < p - 1:
            sites, ents = select_measurement_sites(
                mps_meas, entropy_threshold=0.3, max_sites=2
            )
            if sites:
                measure_qubits(mps_meas, sites, rng=rng)
                total_measured += len(sites)
                print(f"  Na laag {layer+1}: gemeten sites {sites}, "
                      f"max_chi = {mps_meas.max_bond_dim()}")
            else:
                print(f"  Na laag {layer+1}: geen sites onder drempel, "
                      f"max_chi = {mps_meas.max_bond_dim()}")
        else:
            print(f"  Na laag {layer+1}: max_chi = {mps_meas.max_bond_dim()}")

    mps_meas.normalize()
    cost_meas = maxcut_cost_mps(mps_meas, edges)
    print(f"  Totaal gemeten: {total_measured} qubits")
    print(f"  Finale cost: {cost_meas:.4f} (ref: {cost_no_meas:.4f}, "
          f"delta: {cost_meas - cost_no_meas:+.4f})")

    # Met compressie
    print(f"\nMet compressie (chi_max=8) + metingen:")
    rng = np.random.default_rng(42)
    mps_comp = qaoa_mps(n, edges, gammas[:1], betas[:1], chi_max=8)
    for layer in range(1, p):
        apply_qaoa_layer(mps_comp, edges, gammas[layer], betas[layer], chi_max=8)
        if layer % 2 == 0 and layer < p - 1:
            sites, _ = select_measurement_sites(mps_comp, entropy_threshold=0.3, max_sites=2)
            if sites:
                measure_qubits(mps_comp, sites, rng=rng)
        print(f"  Na laag {layer+1}: max_chi = {mps_comp.max_bond_dim()}")
    mps_comp.normalize()
    cost_comp = maxcut_cost_mps(mps_comp, edges)
    print(f"  Finale cost: {cost_comp:.4f}")


def experiment_6_schaalbaarheid():
    """Exp 6: Schaalbaarheid naar grotere ketens.

    Meet wallclock tijd en chi-groei.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Schaalbaarheid mid-circuit metingen")
    print("=" * 70)

    p = 2
    rng_params = np.random.default_rng(42)
    gammas = [0.3, 0.5]
    betas = [0.7, 0.4]

    print(f"\n{'n':>4s} {'p':>2s} {'Methode':>15s} {'MaxChi':>7s} {'Cost':>8s} {'Time':>8s}")
    print("-" * 55)

    for n in [8, 12, 16, 20, 30, 50]:
        edges = [(i, i + 1) for i in range(n - 1)]

        # Zonder meting
        t0 = time.time()
        mps = qaoa_mps(n, edges, gammas, betas)
        dt = time.time() - t0
        cost = maxcut_cost_mps(mps, edges)
        print(f"{n:>4d} {p:>2d} {'geen meting':>15s} {mps.max_bond_dim():>7d} "
              f"{cost:>8.4f} {dt:>7.3f}s")

        # Met compressie chi=8
        t0 = time.time()
        mps = qaoa_mps(n, edges, gammas, betas, chi_max=8)
        dt = time.time() - t0
        cost = maxcut_cost_mps(mps, edges)
        print(f"{n:>4d} {p:>2d} {'chi_max=8':>15s} {mps.max_bond_dim():>7d} "
              f"{cost:>8.4f} {dt:>7.3f}s")

        # Met adaptieve metingen + compressie
        t0 = time.time()
        rng = np.random.default_rng(42)
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)
        mps_out, stats = adaptive_measurement_schedule(
            mps, edges, gammas, betas,
            chi_max=8, entropy_threshold=0.3,
            measure_every=1, max_measure_fraction=0.2,
            rng=rng,
        )
        dt = time.time() - t0
        cost = maxcut_cost_mps(mps_out, edges)
        n_meas = sum(stats['measurements_per_layer'])
        print(f"{n:>4d} {p:>2d} {'adapt+chi=8':>15s} {mps_out.max_bond_dim():>7d} "
              f"{cost:>8.4f} {dt:>7.3f}s (meas:{n_meas})")


if __name__ == '__main__':
    print("=" * 70)
    print("B34: Mid-Circuit Measurement - Experiment Suite")
    print("=" * 70)

    experiment_1_mps_correctheid()
    experiment_2_midcircuit_effect()
    experiment_3_multi_branch()
    experiment_4_adaptive_selectie()
    experiment_5_chi_reductie()
    experiment_6_schaalbaarheid()

    print("\n" + "=" * 70)
    print("ALLE EXPERIMENTEN VOLTOOID")
    print("=" * 70)
