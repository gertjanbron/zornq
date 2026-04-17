#!/usr/bin/env python3
"""
b14_mera.py — B14 MERA vs MPS Vergelijkingsexperiment

Vergelijkt MERA met MPS op twee fronten:
  1. Grondtoestandsenergie (1D TFIM bij kritiek punt)
  2. QAOA-toestandscompressie (MaxCut bij p=1,2,3)

Kernvraag: Kan MERA bij lage chi (4-8) dezelfde nauwkeurigheid
bereiken als MPS bij hoge chi (16-64)?

Referenties:
  Vidal (2007) — MERA
  Evenbly & Vidal (2009) — MERA optimalisatie
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from zorn_mera import (
    ZornMERA, compress_to_mps, mps_energy,
    build_tfim_terms, build_heisenberg_terms, build_maxcut_terms,
    exact_ground_state, qaoa_statevector, _apply_H_standalone,
)


# =====================================================================
# EXPERIMENT 1: TFIM GRONDTOESTAND — MERA vs MPS
# =====================================================================

def tfim_benchmark(n=8, chis_mps=None, chis_mera=None,
                   n_sweeps=25, n_trials=3, verbose=True):
    """Vergelijk MERA vs MPS op 1D TFIM bij h/J=1 (kritiek punt).

    Args:
        n: Aantal sites (macht van 2)
        chis_mps: Bond-dimensies voor MPS-compressie
        chis_mera: Bond-dimensies voor MERA-optimalisatie
        n_sweeps: Optimalisatie-sweeps per MERA-run
        n_trials: Random restarts voor MERA
        verbose: Print details

    Returns:
        dict met resultaten
    """
    if chis_mps is None:
        chis_mps = [2, 4, 8, 16, 32]
    if chis_mera is None:
        chis_mera = [2, 4, 8]

    terms = build_tfim_terms(n, J=1.0, h=1.0)
    E_exact, psi_exact = exact_ground_state(terms, n)

    if verbose:
        print("\n" + "=" * 60)
        print("  TFIM n=%d, h/J=1.0 (kritiek punt)" % n)
        print("  E_exact = %.8f" % E_exact)
        print("=" * 60)

    results = {'n': n, 'E_exact': E_exact, 'mps': [], 'mera': []}

    # --- MPS Compressie ---
    if verbose:
        print("\n  MPS Compressie (optimale SVD-truncatie):")
    for chi in chis_mps:
        t0 = time.time()
        tensors, fid, max_chi = compress_to_mps(psi_exact, n, 2, chi)
        E_mps = mps_energy(tensors, terms, n, 2)
        dt = time.time() - t0

        n_params = sum(t.size for t in tensors)
        err = abs(E_mps - E_exact)
        rel_err = err / abs(E_exact)

        results['mps'].append({
            'chi': chi, 'E': E_mps, 'fidelity': fid,
            'error': err, 'rel_error': rel_err,
            'n_params': n_params, 'time': dt,
        })
        if verbose:
            print("    chi=%2d: E=%.6f  err=%.2e  fid=%.6f  params=%d  (%.3fs)" % (
                chi, E_mps, err, fid, n_params, dt))

    # --- MERA Optimalisatie ---
    if verbose:
        print("\n  MERA Variationeel (environment-linearisatie):")
    for chi in chis_mera:
        best_E = np.inf
        best_fid = 0.0
        total_time = 0.0

        for trial in range(n_trials):
            mera = ZornMERA(n, d=2, chi=chi)
            mera.init_random(seed=1000 + trial * 100 + chi)

            t0 = time.time()
            energies = mera.optimize(terms, n_sweeps=n_sweeps, verbose=False)
            dt = time.time() - t0
            total_time += dt

            E_final = energies[-1]
            fid = mera.fidelity(psi_exact)

            if E_final < best_E:
                best_E = E_final
                best_fid = fid

        err = abs(best_E - E_exact)
        rel_err = err / abs(E_exact)
        n_params = mera.total_params

        results['mera'].append({
            'chi': chi, 'E': best_E, 'fidelity': best_fid,
            'error': err, 'rel_error': rel_err,
            'n_params': n_params, 'time': total_time,
            'n_trials': n_trials, 'n_sweeps': n_sweeps,
        })
        if verbose:
            print("    chi=%2d: E=%.6f  err=%.2e  fid=%.6f  params=%d  (%.1fs, %d trials)" % (
                chi, best_E, err, best_fid, n_params, total_time, n_trials))

    return results


# =====================================================================
# EXPERIMENT 2: QAOA TOESTANDSCOMPRESSIE
# =====================================================================

def qaoa_compression_benchmark(n=8, p_values=None, chis=None, verbose=True):
    """Vergelijk MERA vs MPS voor QAOA-toestandscompressie.

    Idee: bereken |ψ_QAOA⟩ exact, comprimeer naar MPS en MERA,
    vergelijk chi-vereisten voor dezelfde nauwkeurigheid.
    """
    if p_values is None:
        p_values = [1, 2, 3]
    if chis is None:
        chis = [2, 4, 8, 16]

    # Lineaire keten + extra edges voor 2D-achtige connectiviteit
    edges = [(i, i + 1) for i in range(n - 1)]

    # Vaste QAOA-parameters (redelijke waarden)
    gamma_base = 0.3
    beta_base = 0.7

    results_all = {'n': n, 'edges': edges, 'tests': []}

    if verbose:
        print("\n" + "=" * 60)
        print("  QAOA Compressie: n=%d, %d edges" % (n, len(edges)))
        print("=" * 60)

    for p in p_values:
        gammas = [gamma_base * (l + 1) / p for l in range(p)]
        betas = [beta_base * (p - l) / p for l in range(p)]

        terms = build_maxcut_terms(n, edges)

        # Bereken exacte QAOA-toestand
        psi_qaoa = qaoa_statevector(n, edges, p, gammas, betas)

        # Bereken QAOA energie
        H_psi = _apply_H_standalone(psi_qaoa, terms, n, 2)
        E_qaoa = np.vdot(psi_qaoa, H_psi).real

        if verbose:
            print("\n  p=%d: E_QAOA = %.6f" % (p, E_qaoa))

        test_result = {'p': p, 'E_qaoa': E_qaoa, 'mps': [], 'mera': []}

        # --- MPS Compressie ---
        for chi in chis:
            tensors, fid, max_chi = compress_to_mps(psi_qaoa, n, 2, chi)
            E_mps = mps_energy(tensors, terms, n, 2)
            err = abs(E_mps - E_qaoa) / max(abs(E_qaoa), 1e-10)

            test_result['mps'].append({
                'chi': chi, 'fidelity': fid, 'E': E_mps, 'rel_error': err,
            })
            if verbose:
                print("    MPS  chi=%2d: fid=%.6f  E=%.6f  rel_err=%.2e" % (
                    chi, fid, E_mps, err))

        # --- MERA Compressie (optimaliseer naar QAOA-toestand) ---
        for chi in [c for c in chis if c <= 8]:
            best_fid = 0.0
            best_E = np.inf

            for trial in range(3):
                mera = ZornMERA(n, d=2, chi=chi)
                mera.init_random(seed=2000 + p * 100 + chi * 10 + trial)

                # Optimaliseer om fidelity met QAOA-toestand te maximaliseren
                # door ⟨H⟩ te minimaliseren (als H = -|ψ_qaoa⟩⟨ψ_qaoa|)
                # Praktischer: gebruik de MaxCut Hamiltoniaan
                energies = mera.optimize(terms, n_sweeps=20, verbose=False)
                E_final = energies[-1]
                fid = mera.fidelity(psi_qaoa)

                if fid > best_fid:
                    best_fid = fid
                    best_E = E_final

            err = abs(best_E - E_qaoa) / max(abs(E_qaoa), 1e-10)
            test_result['mera'].append({
                'chi': chi, 'fidelity': best_fid, 'E': best_E, 'rel_error': err,
            })
            if verbose:
                print("    MERA chi=%2d: fid=%.6f  E=%.6f  rel_err=%.2e" % (
                    chi, best_fid, best_E, err))

        results_all['tests'].append(test_result)

    return results_all


# =====================================================================
# EXPERIMENT 3: ENTANGLEMENT ENTROPIE
# =====================================================================

def entanglement_entropy(psi, n, d, cut):
    """Bereken entanglement entropie S(A) voor bipartitie A|B bij positie cut."""
    dim_A = d ** cut
    dim_B = d ** (n - cut)
    rho_AB = psi.reshape(dim_A, dim_B)
    _, S, _ = np.linalg.svd(rho_AB, full_matrices=False)
    S = S[S > 1e-15]
    S2 = S ** 2
    return -np.sum(S2 * np.log(S2 + 1e-30))


def entropy_scaling_test(n=8, verbose=True):
    """Test entanglement-entropie scaling voor TFIM en QAOA."""
    results = {}

    # TFIM bij kritiek punt
    terms = build_tfim_terms(n)
    _, psi = exact_ground_state(terms, n)
    S_vals = [entanglement_entropy(psi, n, 2, cut) for cut in range(1, n)]

    results['tfim'] = {'S': S_vals, 'max_S': max(S_vals)}

    # QAOA bij toenemend p
    edges = [(i, i + 1) for i in range(n - 1)]
    for p in [1, 2, 3]:
        gammas = [0.3 * (l + 1) / p for l in range(p)]
        betas = [0.7 * (p - l) / p for l in range(p)]
        psi_q = qaoa_statevector(n, edges, p, gammas, betas)
        S_vals = [entanglement_entropy(psi_q, n, 2, cut)
                  for cut in range(1, n)]
        results['qaoa_p%d' % p] = {'S': S_vals, 'max_S': max(S_vals)}

    if verbose:
        print("\n  Entanglement Entropie (half-chain, n=%d):" % n)
        mid = n // 2
        for key, val in results.items():
            print("    %s: S(n/2) = %.4f, max S = %.4f" % (
                key, val['S'][mid - 1], val['max_S']))

    return results


# =====================================================================
# VOLLEDIG RAPPORT
# =====================================================================

def run_b14_report(verbose=True):
    """Draai het volledige B14 experiment: MERA vs MPS."""
    report = {}

    print("\n" + "#" * 70)
    print("#  B14: MERA Tensor Network — Chi-Muur Vergelijking")
    print("#" * 70)

    # --- Deel 1: TFIM ---
    print("\n" + "=" * 70)
    print("  DEEL 1: TFIM Grondtoestand (kritiek punt)")
    print("=" * 70)

    report['tfim_n8'] = tfim_benchmark(
        n=8, chis_mps=[2, 4, 8, 16], chis_mera=[2, 4],
        n_sweeps=25, n_trials=3, verbose=verbose)

    # --- Deel 2: QAOA Compressie ---
    print("\n" + "=" * 70)
    print("  DEEL 2: QAOA Toestandscompressie")
    print("=" * 70)

    report['qaoa_n8'] = qaoa_compression_benchmark(
        n=8, p_values=[1, 2, 3], chis=[2, 4, 8, 16],
        verbose=verbose)

    # --- Deel 3: Entanglement ---
    print("\n" + "=" * 70)
    print("  DEEL 3: Entanglement Entropie Scaling")
    print("=" * 70)

    report['entropy'] = entropy_scaling_test(n=8, verbose=verbose)

    # --- Samenvatting ---
    print("\n" + "=" * 70)
    print("  SAMENVATTING")
    print("=" * 70)

    # TFIM resultaten
    tfim = report['tfim_n8']
    print("\n  TFIM n=8 (E_exact = %.6f):" % tfim['E_exact'])
    for r in tfim['mps']:
        print("    MPS  chi=%2d: err=%.2e  fid=%.6f  (%d params)" % (
            r['chi'], r['error'], r['fidelity'], r['n_params']))
    for r in tfim['mera']:
        print("    MERA chi=%2d: err=%.2e  fid=%.6f  (%d params)" % (
            r['chi'], r['error'], r['fidelity'], r['n_params']))

    # QAOA resultaten
    print("\n  QAOA Compressie n=8:")
    for test in report['qaoa_n8']['tests']:
        print("    p=%d (E_QAOA=%.4f):" % (test['p'], test['E_qaoa']))
        for r in test['mps'][:3]:  # top 3 chi
            print("      MPS  chi=%2d: fid=%.6f" % (r['chi'], r['fidelity']))
        for r in test['mera'][:2]:
            print("      MERA chi=%2d: fid=%.6f" % (r['chi'], r['fidelity']))

    # Entropie
    entropy = report['entropy']
    print("\n  Max Entanglement Entropie:")
    for key, val in entropy.items():
        print("    %s: %.4f" % (key, val['max_S']))

    # Verdict
    # Vergelijk MPS chi=4 fidelity met MERA chi=4 fidelity
    mps_fid_4 = [r['fidelity'] for r in tfim['mps'] if r['chi'] == 4]
    mera_fid_4 = [r['fidelity'] for r in tfim['mera'] if r['chi'] == 4]

    if mps_fid_4 and mera_fid_4:
        mps_f = mps_fid_4[0]
        mera_f = mera_fid_4[0]
        if mera_f > mps_f * 1.01:
            verdict = "MERA wint bij gelijke chi (fid %.4f vs %.4f)" % (mera_f, mps_f)
        elif mps_f > mera_f * 1.01:
            verdict = "MPS wint bij gelijke chi (fid %.4f vs %.4f)" % (mps_f, mera_f)
        else:
            verdict = "Vergelijkbaar bij chi=4 (fid MPS=%.4f, MERA=%.4f)" % (mps_f, mera_f)
    else:
        verdict = "Onvoldoende data voor verdict"

    # Check of MERA chi=4 ≈ MPS chi=16
    mps_fid_16 = [r['fidelity'] for r in tfim['mps'] if r['chi'] == 16]
    chi_advantage = "onbekend"
    if mps_fid_16 and mera_fid_4:
        mps_f16 = mps_fid_16[0]
        mera_f4 = mera_fid_4[0]
        if mera_f4 >= mps_f16 * 0.99:
            chi_advantage = "MERA chi=4 ≥ MPS chi=16 → 4× chi-reductie"
        else:
            chi_advantage = "MERA chi=4 < MPS chi=16 → geen chi-voordeel"

    print("\n  Verdict: %s" % verdict)
    print("  Chi-voordeel: %s" % chi_advantage)

    report['verdict'] = verdict
    report['chi_advantage'] = chi_advantage
    return report


if __name__ == '__main__':
    report = run_b14_report(verbose=True)
