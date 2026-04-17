#\!/usr/bin/env python3
"""
shadow_solver.py - B10f Classical Shadows / Monte Carlo MaxCut Solver

Chi-onafhankelijke energieschatting voor QAOA MaxCut via:
  1. Random Pauli-basis metingen (Classical Shadows, Huang & Preskill 2020)
  2. Stochastische trace estimation: Tr(H_C rho) via random samples
  3. Optional control variates met lage-chi baseline

Twee modi:
  - "shadow": Simuleer QAOA state-vector, meet in random Pauli-bases,
    reconstrueer <H_C> statistisch. Schaalt O(n log n) metingen.
  - "bitstring": Sample bitstrings uit QAOA-state, evalueer cut-waarde
    per bitstring, neem gemiddelde. Schaalt O(1/eps^2) samples.
  - "maxcut": Combineer sampling + local search voor beste cut.

State-vector sim nodig (n <= 22), maar de SCHATTING is chi-onafhankelijk.
"""

import numpy as np
import time
import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqaoa import WeightedGraph


# ============================================================
# QAOA State-Vector Engine (reused from qits_solver pattern)
# ============================================================

def _precompute_cuts(n_nodes, edges):
    """Vectorized cut values for all 2^n bitstrings."""
    N = 1 << n_nodes
    cuts = np.zeros(N, dtype=np.float64)
    z = np.arange(N, dtype=np.int64)
    for i, j, w in edges:
        bi = (z >> i) & 1
        bj = (z >> j) & 1
        cuts += w * (bi ^ bj).astype(np.float64)
    return cuts


def _apply_qaoa_state(n_nodes, edges, gammas, betas):
    """Build QAOA state |gamma, beta> via state-vector simulation."""
    N = 1 << n_nodes
    # Start from |+>^n
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
    cuts = _precompute_cuts(n_nodes, edges)
    for gamma, beta in zip(gammas, betas):
        # Cost unitary: e^{-i gamma H_C}
        psi *= np.exp(-1j * gamma * cuts)
        # Mixer unitary: e^{-i beta B} with B = sum X_i
        # Decomposes into product of single-qubit rotations
        for q in range(n_nodes):
            stride = 1 << q
            for start in range(0, N, 2 * stride):
                idx0 = np.arange(start, start + stride)
                idx1 = idx0 + stride
                a = psi[idx0]
                b = psi[idx1]
                cb = np.cos(beta)
                sb = np.sin(beta)
                psi[idx0] = cb * a - 1j * sb * b
                psi[idx1] = -1j * sb * a + cb * b
    return psi, cuts


def _apply_qaoa_state_fast(n_nodes, edges, gammas, betas):
    """Optimized QAOA state builder using reshape trick for mixer."""
    N = 1 << n_nodes
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
    cuts = _precompute_cuts(n_nodes, edges)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * cuts)
        cb = np.cos(beta)
        sb = np.sin(beta)
        for q in range(n_nodes):
            shape = [1] * (n_nodes + 1)
            shape[n_nodes - 1 - q] = 2
            shape[-1] = -1
            # Reshape: group amplitudes by qubit q
            psi2 = psi.reshape([2] * n_nodes)
            # Apply Rx(2*beta) on qubit q
            # Swap axes to put target qubit first
            psi2 = np.moveaxis(psi2, q, 0)
            old_shape = psi2.shape
            psi2 = psi2.reshape(2, -1)
            new0 = cb * psi2[0] - 1j * sb * psi2[1]
            new1 = -1j * sb * psi2[0] + cb * psi2[1]
            psi2[0] = new0
            psi2[1] = new1
            psi2 = psi2.reshape(old_shape)
            psi2 = np.moveaxis(psi2, 0, q)
            psi = psi2.reshape(N)
    return psi, cuts


# ============================================================
# Classical Shadows
# ============================================================

def _random_pauli_measurement(psi, n_nodes, rng):
    """
    Single random Pauli-basis measurement.
    Returns (bases, outcome) where bases[i] in {0,1,2} = {X,Y,Z}
    and outcome[i] in {0,1} is the measurement result.
    """
    N = 1 << n_nodes
    bases = rng.integers(0, 3, size=n_nodes)
    # Rotate state into measurement basis
    phi = psi.copy()
    for q in range(n_nodes):
        if bases[q] == 2:  # Z basis - no rotation needed
            continue
        phi2 = phi.reshape([2] * n_nodes)
        # axis mapping: reshape([2]*n) puts MSB at axis 0
        # qubit q corresponds to axis (n_nodes - 1 - q)
        ax = n_nodes - 1 - q
        phi2 = np.moveaxis(phi2, ax, 0)
        old_shape = phi2.shape
        phi2 = phi2.reshape(2, -1)
        if bases[q] == 0:  # X basis: H gate
            s2 = 1.0 / np.sqrt(2.0)
            new0 = s2 * (phi2[0] + phi2[1])
            new1 = s2 * (phi2[0] - phi2[1])
        else:  # Y basis: Sdg then H
            s2 = 1.0 / np.sqrt(2.0)
            new0 = s2 * (phi2[0] - 1j * phi2[1])
            new1 = s2 * (phi2[0] + 1j * phi2[1])
        phi2[0] = new0
        phi2[1] = new1
        phi2 = phi2.reshape(old_shape)
        phi2 = np.moveaxis(phi2, 0, ax)
        phi = phi2.reshape(N)
    # Sample from computational basis
    probs = np.abs(phi) ** 2
    probs /= probs.sum()  # Normalize for numerical safety
    idx = rng.choice(N, p=probs)
    outcome = np.array([(idx >> q) & 1 for q in range(n_nodes)])
    return bases, outcome


def _shadow_expectation_zz(bases_list, outcomes_list, n_nodes, edges):
    """
    Estimate <H_C> = sum_{ij} w_{ij} (1 - Z_i Z_j) / 2 from classical shadows.
    
    For each shadow (bases, outcome), the single-qubit shadow operator is:
      rho_hat_i = 3 * |b_i><b_i| - I  (for Pauli basis measurement)
    
    For Z_i Z_j estimation, only shadows where BOTH qubits i,j were measured
    in Z-basis contribute: <Z_i Z_j> = (3^2) * (-1)^(b_i+b_j) averaged.
    """
    n_shadows = len(bases_list)
    total_cut = 0.0
    for i, j, w in edges:
        zz_sum = 0.0
        zz_count = 0
        for s in range(n_shadows):
            # Both qubits must be measured in Z basis
            if bases_list[s][i] == 2 and bases_list[s][j] == 2:
                # Z eigenvalue: (-1)^outcome
                zi = 1 - 2 * outcomes_list[s][i]
                zj = 1 - 2 * outcomes_list[s][j]
                zz_sum += zi * zj
                zz_count += 1
        # Factor 3^2 = 9 from shadow channel inversion (n_bases=3)
        # Divide by n_shadows (total), not zz_count: non-Z shadows contribute 0
        zz_est = 9.0 * zz_sum / n_shadows
        total_cut += w * (1.0 - zz_est) / 2.0
    return total_cut


def _shadow_expectation_median_of_means(bases_list, outcomes_list, n_nodes, edges, n_groups=10):
    """
    Robust shadow estimator using median-of-means.
    Split shadows into groups, compute mean per group, take median.
    """
    n_shadows = len(bases_list)
    if n_shadows < n_groups:
        return _shadow_expectation_zz(bases_list, outcomes_list, n_nodes, edges)
    group_size = n_shadows // n_groups
    group_estimates = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        est = _shadow_expectation_zz(bases_list[start:end], outcomes_list[start:end], n_nodes, edges)
        group_estimates.append(est)
    return np.median(group_estimates)


# ============================================================
# Bitstring Sampling + Local Search
# ============================================================

def _sample_bitstrings(psi, n_samples, rng):
    """Sample bitstrings from QAOA state probability distribution."""
    N = len(psi)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    indices = rng.choice(N, size=n_samples, p=probs)
    return indices


def _eval_cut_single(x_int, n_nodes, edges):
    """Evaluate cut value for a single bitstring (as integer)."""
    total = 0.0
    for i, j, w in edges:
        bi = (x_int >> i) & 1
        bj = (x_int >> j) & 1
        if bi ^ bj:
            total += w
    return total


def _local_search(x_int, n_nodes, edges, max_iter=100):
    """Steepest-ascent local search from a bitstring."""
    best = x_int
    best_cut = _eval_cut_single(best, n_nodes, edges)
    for _ in range(max_iter):
        improved = False
        for q in range(n_nodes):
            candidate = best ^ (1 << q)
            c = _eval_cut_single(candidate, n_nodes, edges)
            if c > best_cut:
                best = candidate
                best_cut = c
                improved = True
        if not improved:
            break
    return best, best_cut


# ============================================================
# Shadow Energy Estimator
# ============================================================

def shadow_energy(n_nodes, edges, gammas, betas, n_shadows=1000,
                  method="median_of_means", n_groups=10, seed=None, verbose=False):
    """
    Estimate QAOA energy <H_C> using classical shadows.
    
    Parameters:
        n_nodes: number of graph nodes
        edges: list of (i, j, w) weighted edges
        gammas, betas: QAOA parameters (lists of length p)
        n_shadows: number of random Pauli measurements
        method: "mean" or "median_of_means"
        n_groups: groups for median-of-means
        seed: random seed
        verbose: print progress
    
    Returns:
        dict with energy estimate, stderr, exact energy, n_shadows
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    
    # Build QAOA state
    psi, cuts = _apply_qaoa_state_fast(n_nodes, edges, gammas, betas)
    
    # Exact energy for reference
    probs = np.abs(psi) ** 2
    exact_energy = np.dot(probs, cuts)
    
    # Collect shadows
    bases_list = []
    outcomes_list = []
    for s in range(n_shadows):
        bases, outcome = _random_pauli_measurement(psi, n_nodes, rng)
        bases_list.append(bases)
        outcomes_list.append(outcome)
    
    # Estimate energy
    if method == "median_of_means":
        est_energy = _shadow_expectation_median_of_means(
            bases_list, outcomes_list, n_nodes, edges, n_groups)
    else:
        est_energy = _shadow_expectation_zz(bases_list, outcomes_list, n_nodes, edges)
    
    # Bootstrap stderr
    n_boot = 50
    boot_energies = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_shadows, size=n_shadows)
        b_bases = [bases_list[i] for i in idx]
        b_outcomes = [outcomes_list[i] for i in idx]
        if method == "median_of_means":
            be = _shadow_expectation_median_of_means(b_bases, b_outcomes, n_nodes, edges, n_groups)
        else:
            be = _shadow_expectation_zz(b_bases, b_outcomes, n_nodes, edges)
        boot_energies.append(be)
    stderr = np.std(boot_energies)
    
    elapsed = time.time() - t0
    
    if verbose:
        print(f"Shadow energy: {est_energy:.4f} (exact: {exact_energy:.4f})")
        print(f"  Error: {abs(est_energy - exact_energy):.4f}, Stderr: {stderr:.4f}")
        print(f"  {n_shadows} shadows in {elapsed:.3f}s")
    
    return {
        "energy_est": est_energy,
        "energy_exact": exact_energy,
        "abs_error": abs(est_energy - exact_energy),
        "rel_error": abs(est_energy - exact_energy) / max(abs(exact_energy), 1e-10),
        "stderr": stderr,
        "n_shadows": n_shadows,
        "time_s": elapsed,
    }


# ============================================================
# Shadow MaxCut Solver
# ============================================================

def shadow_maxcut(n_nodes, edges, p=1, n_shadows=500, n_samples=200,
                  n_restarts=5, local_search=True, verbose=False, seed=None):
    """
    MaxCut solver combining QAOA sampling + classical shadows + local search.
    
    Strategy:
      1. For each restart, pick random gamma/beta parameters
      2. Build QAOA state, estimate energy via shadows
      3. Sample top bitstrings, apply local search
      4. Return best cut found across all restarts
    
    Parameters:
        n_nodes: number of graph nodes
        edges: list of (i, j, w) weighted edges
        p: QAOA depth
        n_shadows: shadows per restart for energy estimation
        n_samples: bitstring samples per restart
        n_restarts: number of random parameter restarts
        local_search: apply local search to sampled bitstrings
        verbose: print progress
        seed: random seed
    
    Returns:
        dict with best_cut, best_bitstring, shadow_energies, etc.
    """
    assert n_nodes <= 22, f"State-vector limit: n_nodes={n_nodes} > 22"
    rng = np.random.default_rng(seed)
    t0 = time.time()
    
    # Weighted edges
    wedges = [(i, j, w if len(e) > 2 else 1.0) for e in edges for i, j, w in [
        (e[0], e[1], e[2] if len(e) > 2 else 1.0)]]
    # Simpler:
    wedges = []
    for e in edges:
        if len(e) == 3:
            wedges.append((e[0], e[1], e[2]))
        else:
            wedges.append((e[0], e[1], 1.0))
    
    best_cut = -1
    best_bits = 0
    best_energy_est = 0
    all_energies = []
    
    for r in range(n_restarts):
        # Random QAOA parameters
        gammas = rng.uniform(0, np.pi, size=p)
        betas = rng.uniform(0, np.pi / 2, size=p)
        
        # Build state
        psi, cuts = _apply_qaoa_state_fast(n_nodes, wedges, gammas, betas)
        
        # Shadow energy estimate
        bases_list = []
        outcomes_list = []
        for _ in range(n_shadows):
            bases, outcome = _random_pauli_measurement(psi, n_nodes, rng)
            bases_list.append(bases)
            outcomes_list.append(outcome)
        energy_est = _shadow_expectation_median_of_means(
            bases_list, outcomes_list, n_nodes, wedges)
        all_energies.append(energy_est)
        
        # Sample bitstrings
        indices = _sample_bitstrings(psi, n_samples, rng)
        for idx in indices:
            if local_search:
                bits, cut = _local_search(int(idx), n_nodes, wedges)
            else:
                bits = int(idx)
                cut = _eval_cut_single(bits, n_nodes, wedges)
            if cut > best_cut:
                best_cut = cut
                best_bits = bits
                best_energy_est = energy_est
        
        if verbose:
            print(f"  restart {r+1}/{n_restarts}: shadow_E={energy_est:.2f}, best_cut={best_cut:.1f}")
    
    elapsed = time.time() - t0
    
    # Exact optimum for comparison
    all_cuts = _precompute_cuts(n_nodes, wedges)
    exact_opt = float(np.max(all_cuts))
    
    result = {
        "best_cut": best_cut,
        "best_bitstring": best_bits,
        "exact_optimum": exact_opt,
        "ratio": best_cut / exact_opt if exact_opt > 0 else 1.0,
        "shadow_energies": all_energies,
        "n_restarts": n_restarts,
        "n_shadows": n_shadows,
        "n_samples": n_samples,
        "time_s": elapsed,
    }
    
    if verbose:
        print(f"Shadow MaxCut: best={best_cut:.1f}/{exact_opt:.1f} (ratio={result['ratio']:.4f}")
        print(f"  {n_restarts} restarts x {n_shadows} shadows + {n_samples} samples in {elapsed:.3f}s")
    
    return result


def shadow_maxcut_grid(Lx, Ly, **kwargs):
    """Convenience wrapper for grid graphs."""
    n_nodes = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            if x + 1 < Lx:
                edges.append((node, (x + 1) * Ly + y, 1.0))
            if y + 1 < Ly:
                edges.append((node, x * Ly + y + 1, 1.0))
    return shadow_maxcut(n_nodes, edges, **kwargs)


# ============================================================
# Convergence Analysis
# ============================================================

def shadow_convergence(n_nodes, edges, gammas, betas,
                       shadow_counts=None, seed=None, verbose=False):
    """
    Analyze how shadow energy estimate converges with number of shadows.
    Returns list of (n_shadows, estimate, error, stderr) tuples.
    """
    if shadow_counts is None:
        shadow_counts = [50, 100, 200, 500, 1000, 2000, 5000]
    rng = np.random.default_rng(seed)
    
    # Build state once
    psi, cuts = _apply_qaoa_state_fast(n_nodes, edges, gammas, betas)
    probs = np.abs(psi) ** 2
    exact_energy = np.dot(probs, cuts)
    
    # Collect max shadows
    max_shadows = max(shadow_counts)
    all_bases = []
    all_outcomes = []
    for _ in range(max_shadows):
        bases, outcome = _random_pauli_measurement(psi, n_nodes, rng)
        all_bases.append(bases)
        all_outcomes.append(outcome)
    
    results = []
    for ns in shadow_counts:
        est = _shadow_expectation_median_of_means(
            all_bases[:ns], all_outcomes[:ns], n_nodes, edges)
        err = abs(est - exact_energy)
        # Bootstrap stderr
        boots = []
        for _ in range(30):
            idx = rng.integers(0, ns, size=ns)
            b = _shadow_expectation_median_of_means(
                [all_bases[i] for i in idx], [all_outcomes[i] for i in idx],
                n_nodes, edges)
            boots.append(b)
        se = np.std(boots)
        results.append({"n_shadows": ns, "estimate": est, "exact": exact_energy,
                         "abs_error": err, "stderr": se})
        if verbose:
            print(f"  K={ns:5d}: est={est:.4f}, exact={exact_energy:.4f}, err={err:.4f}, se={se:.4f}")
    return results


if __name__ == "__main__":
    print("=== B10f Classical Shadows Demo ===")
    # 4x2 grid
    Lx, Ly = 4, 2
    n = Lx * Ly
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            if x + 1 < Lx: edges.append((node, (x+1)*Ly+y, 1.0))
            if y + 1 < Ly: edges.append((node, x*Ly+y+1, 1.0))
    
    print("\n1. Shadow Energy Convergence:")
    gammas = [0.5]
    betas = [0.8]
    shadow_convergence(n, edges, gammas, betas, verbose=True, seed=42)
    
    print("\n2. Shadow MaxCut:")
    result = shadow_maxcut(n, edges, p=1, n_restarts=10, verbose=True, seed=42)
    ratio = result["ratio"]; print(f"   Final ratio: {ratio:.4f}")
