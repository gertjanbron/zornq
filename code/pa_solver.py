#\!/usr/bin/env python3
"""
pa_solver.py - B135 Population Annealing Monte Carlo for MaxCut

Population Annealing: a parallel Monte Carlo method that
simultaneously cools a population of replicas. At each
temperature step:
  1. Resample: clone good replicas, kill bad ones (Boltzmann weights)
  2. Sweep: each replica does Metropolis sweeps at current temperature
  3. Track: record best-ever solution across all replicas

Key advantages over simulated annealing:
  - Embarrassingly parallel (each replica independent during sweeps)
  - Resampling provides population diversity without random restarts
  - Adaptive temperature schedule via energy variance

References:
  - Hukushima & Iba (2003): Population Annealing
  - Wang, Machta et al. (2015): PA for spin glasses
  - GPU-PA for MaxCut: recent results (2025) on Gset G63
"""

import numpy as np
import sys
import os
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Graph representation for vectorized operations
# ============================================================

def _build_adj_arrays(n_nodes, edges):
    """Build adjacency arrays for fast cut evaluation."""
    adj = [[] for _ in range(n_nodes)]
    wt = [[] for _ in range(n_nodes)]
    for e in edges:
        i, j = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        adj[i].append(j)
        wt[i].append(w)
        adj[j].append(i)
        wt[j].append(w)
    # Convert to padded numpy arrays for vectorized access
    max_deg = max(len(a) for a in adj) if adj else 0
    adj_arr = np.full((n_nodes, max_deg), -1, dtype=np.int32)
    wt_arr = np.zeros((n_nodes, max_deg), dtype=np.float64)
    deg = np.zeros(n_nodes, dtype=np.int32)
    for v in range(n_nodes):
        d = len(adj[v])
        deg[v] = d
        adj_arr[v, :d] = adj[v]
        wt_arr[v, :d] = wt[v]
    return adj_arr, wt_arr, deg


def _compute_cut_batch(population, ei, ej, ew):
    """
    Compute cut values for entire population at once.
    population: (R, n_nodes) int8 array
    Returns: (R,) float64 array of cut values
    """
    # XOR bits for each edge across all replicas
    diff = population[:, ei] ^ population[:, ej]  # (R, m)
    cuts = np.sum(ew[None, :] * diff, axis=1)
    return cuts


def _compute_cut_single(x, ei, ej, ew):
    """Cut value for single assignment."""
    return float(np.sum(ew * (x[ei] ^ x[ej])))


# ============================================================
# Metropolis sweep (vectorized over population)
# ============================================================

def _metropolis_sweep(population, adj_arr, wt_arr, deg, beta, rng, n_sweeps=1):
    """
    Perform n_sweeps Metropolis sweeps on entire population.
    Each sweep visits all nodes in random order.
    
    population: (R, n_nodes) int8 array, modified in-place
    beta: inverse temperature
    """
    R, n_nodes = population.shape
    max_deg = adj_arr.shape[1]
    
    for _ in range(n_sweeps):
        order = rng.permutation(n_nodes)
        for v in order:
            d = int(deg[v])
            if d == 0:
                continue
            # Neighbors and weights for node v
            nbrs = adj_arr[v, :d]  # (d,)
            ws = wt_arr[v, :d]  # (d,)
            
            # Current spin values of neighbors across all replicas
            nbr_spins = population[:, nbrs]  # (R, d)
            v_spin = population[:, v:v+1]  # (R, 1)
            
            # Delta E = sum_u w(v,u) * (2*same - 2*diff)
            # same means both equal, flipping v would cut edge (gain)
            # For MaxCut: we want to MAXIMIZE cut
            # delta = sum w * (same - diff) = gain from flipping
            same = (nbr_spins == v_spin).astype(np.float64)  # (R, d)
            delta = np.sum(ws[None, :] * (2.0 * same - 1.0), axis=1)  # (R,)
            
            # Accept if delta > 0 (improves cut) or with prob exp(beta*delta)
            # We maximize cut, so accept improvements always
            # For worsening moves: accept with prob exp(-beta * |delta|)
            # Since we maximize: prob = exp(beta * delta) when delta < 0
            rand = rng.random(R)
            accept = (delta > 0) | (rand < np.exp(np.clip(beta * delta, -30, 0)))
            
            # Flip accepted nodes
            population[accept, v] = 1 - population[accept, v]


# ============================================================
# Resampling step
# ============================================================

def _resample(population, cuts, beta_old, beta_new, rng, target_R):
    """
    Resample population based on Boltzmann weight change.
    Replicas with higher cut get cloned, lower get killed.
    
    Weight of replica i: w_i = exp((beta_new - beta_old) * cut_i)
    Normalize to get expected copies.
    """
    R = population.shape[0]
    dbeta = beta_new - beta_old
    
    if abs(dbeta) < 1e-15:
        return population, cuts
    
    # Log-weights for numerical stability
    log_w = dbeta * cuts
    log_w -= np.max(log_w)  # shift for stability
    w = np.exp(log_w)
    w_sum = np.sum(w)
    
    if w_sum < 1e-15:
        return population, cuts
    
    # Expected copies per replica
    probs = w / w_sum
    
    # Multinomial resampling
    indices = rng.choice(R, size=target_R, p=probs)
    
    new_pop = population[indices].copy()
    new_cuts = cuts[indices].copy()
    
    return new_pop, new_cuts


# ============================================================
# Temperature schedule
# ============================================================

def _geometric_schedule(beta_min, beta_max, n_temps):
    """Geometric temperature schedule (inverse temps)."""
    if n_temps <= 1:
        return np.array([beta_max])
    return np.geomspace(max(beta_min, 1e-6), beta_max, n_temps)


def _linear_schedule(beta_min, beta_max, n_temps):
    """Linear temperature schedule."""
    return np.linspace(beta_min, beta_max, n_temps)


def _normalize_init_bits(x_init, n_nodes):
    """Normalize dict/list/array init state to an int8 bit vector."""
    if x_init is None:
        return None
    if isinstance(x_init, dict):
        x = np.zeros(n_nodes, dtype=np.int8)
        for i in range(n_nodes):
            val = int(x_init.get(i, 0))
            x[i] = 1 if val > 0 else 0
        return x
    x = np.asarray(x_init, dtype=np.int8).reshape(-1)
    if x.shape[0] != n_nodes:
        raise ValueError(f"x_init length {x.shape[0]} != n_nodes {n_nodes}")
    return np.where(x > 0, 1, 0).astype(np.int8, copy=False)


def _initialize_population(rng, n_replicas, n_nodes, x_init=None):
    """Initialize PA population, optionally with a small seeded replica block."""
    population = rng.integers(0, 2, size=(n_replicas, n_nodes)).astype(np.int8)
    x0 = _normalize_init_bits(x_init, n_nodes)
    if x0 is None or n_replicas <= 0:
        return population

    seed_count = min(n_replicas, max(1, n_replicas // 8))
    population[0] = x0
    if seed_count == 1:
        return population

    flip_base = max(1, n_nodes // 200)
    for ridx in range(1, seed_count):
        candidate = x0.copy()
        n_flips = min(n_nodes, flip_base * ridx)
        flip_nodes = rng.choice(n_nodes, size=n_flips, replace=False)
        candidate[flip_nodes] = 1 - candidate[flip_nodes]
        population[ridx] = candidate
    return population


# ============================================================
# Main PA algorithm
# ============================================================

def pa_maxcut(n_nodes, edges, n_replicas=100, n_temps=50,
              beta_min=0.1, beta_max=5.0, n_sweeps=3,
              schedule="geometric", time_limit=None,
              seed=None, verbose=False, x_init=None):
    """
    Population Annealing for MaxCut.
    
    Args:
        n_nodes: number of nodes
        edges: list of (i, j) or (i, j, w)
        n_replicas: population size
        n_temps: number of temperature steps
        beta_min: starting inverse temperature (high T, random)
        beta_max: final inverse temperature (low T, frozen)
        n_sweeps: Metropolis sweeps per temperature step
        schedule: "geometric" or "linear"
        time_limit: wall-clock time limit in seconds
        seed: random seed
        verbose: print progress
        x_init: optional warm start for a seeded replica block
    
    Returns:
        dict with best_cut, assignment, history, time_s
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    # Build graph structures
    adj_arr, wt_arr, deg = _build_adj_arrays(n_nodes, edges)
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] if len(e) > 2 else 1.0 for e in edges], dtype=np.float64)
    
    # Temperature schedule
    if schedule == "geometric":
        betas = _geometric_schedule(beta_min, beta_max, n_temps)
    else:
        betas = _linear_schedule(beta_min, beta_max, n_temps)
    
    # Initialize random population
    population = _initialize_population(rng, n_replicas, n_nodes, x_init=x_init)
    
    # Compute initial cuts
    cuts = _compute_cut_batch(population, ei, ej, ew)
    
    # Track best
    best_idx = int(np.argmax(cuts))
    best_cut = float(cuts[best_idx])
    best_x = population[best_idx].copy()
    
    history = []
    beta_prev = 0.0  # start from infinite temperature
    
    for step, beta in enumerate(betas):
        if time_limit is not None and (time.time() - t0) >= time_limit:
            break
        
        # 1. Resample based on energy change
        if step > 0:
            population, cuts = _resample(
                population, cuts, beta_prev, beta, rng, n_replicas
            )
        
        # 2. Metropolis sweeps at current temperature
        _metropolis_sweep(population, adj_arr, wt_arr, deg, beta, rng, n_sweeps)
        
        # 3. Recompute cuts after sweeps
        cuts = _compute_cut_batch(population, ei, ej, ew)
        
        # 4. Track best
        step_best_idx = int(np.argmax(cuts))
        step_best = float(cuts[step_best_idx])
        if step_best > best_cut:
            best_cut = step_best
            best_x = population[step_best_idx].copy()
        
        mean_cut = float(np.mean(cuts))
        history.append((float(beta), mean_cut, step_best))
        
        if verbose and (step % max(1, n_temps // 10) == 0 or step == len(betas) - 1):
            t = time.time() - t0
            print(f'  PA step {step+1}/{n_temps}: beta={beta:.3f}, mean={mean_cut:.1f}, best={best_cut:.1f}, t={t:.2f}s')
        
        beta_prev = beta
    
    # Final local search on best replica
    best_x, best_cut = _greedy_local_search(best_x.astype(np.int32), adj_arr, wt_arr, deg, ei, ej, ew)
    
    elapsed = time.time() - t0
    assignment = {i: int(best_x[i]) for i in range(n_nodes)}
    
    return {
        'best_cut': best_cut,
        'assignment': assignment,
        'history': history,
        'n_temps_done': len(history),
        'time_s': elapsed,
    }


def _greedy_local_search(x, adj_arr, wt_arr, deg, ei, ej, ew, max_iter=200):
    """Quick steepest-ascent local search to polish PA result."""
    n = len(x)
    best_cut = _compute_cut_single(x, ei, ej, ew)
    for _ in range(max_iter):
        improved = False
        for v in range(n):
            d = int(deg[v])
            if d == 0:
                continue
            nbrs = adj_arr[v, :d]
            ws = wt_arr[v, :d]
            delta = 0.0
            for k in range(d):
                u = nbrs[k]
                if x[v] == x[u]:
                    delta += ws[k]
                else:
                    delta -= ws[k]
            if delta > 1e-12:
                x[v] = 1 - x[v]
                best_cut += delta
                improved = True
        if not improved:
            break
    return x, best_cut


def pa_maxcut_grid(Lx, Ly, triangular=False, **kwargs):
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
            if triangular and x + 1 < Lx and y + 1 < Ly:
                edges.append((node, (x + 1) * Ly + y + 1, 1.0))
    return pa_maxcut(n_nodes, edges, **kwargs)


if __name__ == '__main__':
    print('=== B135 Population Annealing Demo ===\n')

    print('--- Grid graphs ---')
    for Lx, Ly in [(4, 2), (5, 3), (10, 4), (20, 5)]:
        r = pa_maxcut_grid(Lx, Ly, n_replicas=50, n_temps=30, seed=42)
        n = Lx * Ly
        print(f'  {Lx}x{Ly} (n={n}): cut={r["best_cut"]:.0f}, time={r["time_s"]:.3f}s')

    print('\n--- Triangular grids ---')
    for Lx, Ly in [(3, 2), (4, 3), (5, 4)]:
        r = pa_maxcut_grid(Lx, Ly, triangular=True, n_replicas=100, n_temps=40, seed=42)
        n = Lx * Ly
        print(f'  tri {Lx}x{Ly} (n={n}): cut={r["best_cut"]:.0f}, time={r["time_s"]:.3f}s')

    print('\n--- Random 3-regular ---')
    sys.path.insert(0, ".")
    from bls_solver import random_3regular, random_erdos_renyi
    for n in [50, 100, 200, 500]:
        nn, edges = random_3regular(n, seed=n)
        r = pa_maxcut(nn, edges, n_replicas=100, n_temps=50, seed=42)
        print(f'  3-reg n={n}: cut={r["best_cut"]:.0f}, time={r["time_s"]:.2f}s')

    print('\n--- ER G(n, 0.5) ---')
    for n in [50, 100]:
        nn, edges = random_erdos_renyi(n, p=0.5, seed=n)
        r = pa_maxcut(nn, edges, n_replicas=100, n_temps=40, seed=42)
        print(f'  ER n={n} ({len(edges)} edges): cut={r["best_cut"]:.0f}, time={r["time_s"]:.2f}s')

    print('\n--- Verbose run (3-reg n=100) ---')
    nn, edges = random_3regular(100, seed=100)
    r = pa_maxcut(nn, edges, n_replicas=200, n_temps=60, n_sweeps=5, seed=42, verbose=True)
    print(f'  Final: cut={r["best_cut"]:.0f}')
