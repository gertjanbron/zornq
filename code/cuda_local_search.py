#!/usr/bin/env python3
"""
cuda_local_search.py - B136 CUDA Local Search Kernel for MaxCut

GPU-accelerated inner loops for BLS and Population Annealing.
Moves the hot loops (delta computation, steepest ascent, Metropolis
sweep) to CUDA kernels via CuPy RawKernel.

Target: GTX 1650 (896 CUDA cores, 4GB VRAM)
Expected speedup: 50-100x on n>=1000 vs pure Python.

Architecture:
  - CSR graph format on GPU (compact, cache-friendly)
  - One thread per node for delta computation
  - Parallel reduction for argmax (steepest ascent)
  - One thread per (replica, node) for PA Metropolis

Provides drop-in replacements:
  - cuda_bls_maxcut()   -> replaces bls_maxcut()
  - cuda_pa_maxcut()    -> replaces pa_maxcut()
  - Falls back to CPU implementations if no GPU available.

References:
  - Benlic & Hao (2013): Breakout Local Search
  - Amey & Machta (2018): GPU Population Annealing
"""

import numpy as np
import sys
import os
import time
import tempfile
import shutil
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if os.name == 'nt':
    class _SafeTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None,
                     ignore_cleanup_errors=True):
            self._ignore_cleanup_errors = ignore_cleanup_errors
            self._base_dir = dir or tempfile.gettempdir()
            self._prefix = prefix or 'tmp'
            self._suffix = suffix or ''
            self.name = None

            for _ in range(100):
                candidate = os.path.join(
                    self._base_dir,
                    f"{self._prefix}{uuid.uuid4().hex}{self._suffix}",
                )
                try:
                    os.makedirs(candidate, exist_ok=False)
                    self.name = candidate
                    break
                except FileExistsError:
                    continue

            if self.name is None:
                raise FileExistsError(
                    f"Could not create temporary directory inside {self._base_dir}"
                )

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            self.cleanup()
            return False

        def cleanup(self):
            if not self.name:
                return
            try:
                shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
            except PermissionError:
                return

    tempfile.TemporaryDirectory = _SafeTemporaryDirectory

# CuPy/NVRTC schrijft tijdelijke bronbestanden weg tijdens kernel-compilatie.
# Op sommige Windows-omgevingen geeft cleanup van %TEMP% permission errors, dus
# we sturen die tijdelijke runtime-bestanden naar een project-lokale map.
def _configure_cuda_runtime_dirs():
    """Return project-local CUDA temp/cache dirs after creating them."""
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    )
    runtime_root = os.path.join(project_root, 'runtime_cuda')
    temp_root = os.path.join(runtime_root, 'temp')
    cache_root = os.path.join(runtime_root, 'cupy_cache')

    try:
        os.makedirs(temp_root, exist_ok=True)
        os.makedirs(cache_root, exist_ok=True)
    except Exception:
        return None

    if os.name == 'nt':
        os.environ['TMP'] = temp_root
        os.environ['TEMP'] = temp_root
    else:
        os.environ.setdefault('TMPDIR', temp_root)

    os.environ.setdefault('CUPY_CACHE_DIR', cache_root)
    return {
        'runtime_root': runtime_root,
        'temp_root': temp_root,
        'cache_root': cache_root,
    }


_CUDA_RUNTIME_DIRS = _configure_cuda_runtime_dirs()

# ============================================================
# GPU availability check
# ============================================================

_HAS_GPU = False
cp = None

try:
    import cupy as _cp
    _cp.cuda.Device(0).compute_capability
    cp = _cp
    _HAS_GPU = True
except Exception:
    pass


# ============================================================
# CSR graph format (shared by all kernels)
# ============================================================

def _build_csr(n_nodes, edges):
    """
    Build Compressed Sparse Row graph for GPU.
    Returns: row_ptr (n+1,), col_idx (2m,), weights (2m,)
    Each undirected edge stored twice (i->j and j->i).
    """
    adj = [[] for _ in range(n_nodes)]
    wt = [[] for _ in range(n_nodes)]
    for e in edges:
        i, j = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) > 2 else 1.0
        adj[i].append(j)
        wt[i].append(w)
        adj[j].append(i)
        wt[j].append(w)
    row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for v in range(n_nodes):
        row_ptr[v + 1] = row_ptr[v] + len(adj[v])
    nnz = int(row_ptr[n_nodes])
    col_idx = np.zeros(nnz, dtype=np.int32)
    weights = np.zeros(nnz, dtype=np.float32)
    for v in range(n_nodes):
        start = int(row_ptr[v])
        for k, u in enumerate(adj[v]):
            col_idx[start + k] = u
            weights[start + k] = wt[v][k]
    return row_ptr, col_idx, weights


# ============================================================
# CUDA Kernels (CuPy RawKernel)
# ============================================================

if _HAS_GPU:

    # --- Kernel 1: Compute delta for all nodes ---
    _delta_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_deltas(
        const int* __restrict__ x,
        const int* __restrict__ row_ptr,
        const int* __restrict__ col_idx,
        const float* __restrict__ weights,
        float* __restrict__ delta,
        int n_nodes
    ) {
        int v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= n_nodes) return;

        int xv = x[v];
        float d = 0.0f;
        int start = row_ptr[v];
        int end = row_ptr[v + 1];

        for (int k = start; k < end; k++) {
            int u = col_idx[k];
            float w = weights[k];
            if (xv == x[u])
                d += w;   // same side: flipping v cuts this edge
            else
                d -= w;   // diff side: flipping v uncuts this edge
        }
        delta[v] = d;
    }
    ''', 'compute_deltas')

    # --- Kernel 2: Update deltas after flipping node v ---
    _update_delta_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void update_deltas_after_flip(
        const int* __restrict__ x,
        const int* __restrict__ row_ptr,
        const int* __restrict__ col_idx,
        const float* __restrict__ weights,
        float* __restrict__ delta,
        int v
    ) {
        // Thread 0 handles v itself
        // Threads 1..deg(v) handle neighbors
        int start = row_ptr[v];
        int end = row_ptr[v + 1];
        int deg = end - start;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid == 0) {
            delta[v] = -delta[v];
        } else if (tid <= deg) {
            int k = start + tid - 1;
            int u = col_idx[k];
            float w = weights[k];
            if (x[v] == x[u])
                atomicAdd(&delta[u], 2.0f * w);
            else
                atomicAdd(&delta[u], -2.0f * w);
        }
    }
    ''', 'update_deltas_after_flip')

    # --- Kernel 3: Find argmax of delta (not tabu) ---
    # Uses shared memory reduction within a block, then CPU picks block winners
    _argmax_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void find_best_node(
        const float* __restrict__ delta,
        const long long* __restrict__ tabu,
        long long iteration,
        int n_nodes,
        float* __restrict__ block_best_val,
        int* __restrict__ block_best_idx
    ) {
        extern __shared__ char smem[];
        float* sval = (float*)smem;
        int* sidx = (int*)(smem + blockDim.x * sizeof(float));

        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        float my_val = -1e30f;
        int my_idx = -1;

        if (gid < n_nodes && delta[gid] > 0.0f && tabu[gid] < iteration) {
            my_val = delta[gid];
            my_idx = gid;
        }

        sval[tid] = my_val;
        sidx[tid] = my_idx;
        __syncthreads();

        // Reduction within block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (sval[tid + s] > sval[tid]) {
                    sval[tid] = sval[tid + s];
                    sidx[tid] = sidx[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            block_best_val[blockIdx.x] = sval[0];
            block_best_idx[blockIdx.x] = sidx[0];
        }
    }
    ''', 'find_best_node')

    # --- Kernel 4: Compute cut value (parallel edge sum) ---
    _cut_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_cut(
        const int* __restrict__ x,
        const int* __restrict__ ei,
        const int* __restrict__ ej,
        const float* __restrict__ ew,
        float* __restrict__ partial_sums,
        int m
    ) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        float val = 0.0f;
        if (gid < m) {
            if (x[ei[gid]] != x[ej[gid]])
                val = ew[gid];
        }
        sdata[tid] = val;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial_sums[blockIdx.x] = sdata[0];
    }
    ''', 'compute_cut')

    # --- Kernel 5: PA Metropolis sweep (one thread per replica) ---
    _metropolis_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void metropolis_node(
        int* __restrict__ population,   // (R, n_nodes) row-major
        const int* __restrict__ row_ptr,
        const int* __restrict__ col_idx,
        const float* __restrict__ weights,
        float beta,
        const float* __restrict__ rand_vals,  // (R,) uniform random
        int n_nodes,
        int v,      // which node to process
        int R       // population size
    ) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= R) return;

        int offset = r * n_nodes;
        int xv = population[offset + v];
        float d = 0.0f;

        int start = row_ptr[v];
        int end = row_ptr[v + 1];
        for (int k = start; k < end; k++) {
            int u = col_idx[k];
            float w = weights[k];
            if (xv == population[offset + u])
                d += w;
            else
                d -= w;
        }

        // Accept if improves (d > 0) or with Metropolis probability
        float accept_p = (d > 0.0f) ? 1.0f : expf(fminf(beta * d, 0.0f));
        if (rand_vals[r] < accept_p) {
            population[offset + v] = 1 - xv;
        }
    }
    ''', 'metropolis_node')

    # --- Kernel 6: Batch cut for PA population ---
    _batch_cut_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void batch_cut(
        const int* __restrict__ population,  // (R, n_nodes) row-major
        const int* __restrict__ ei,
        const int* __restrict__ ej,
        const float* __restrict__ ew,
        float* __restrict__ cuts,  // (R,) output
        int n_nodes,
        int m,  // number of edges
        int R
    ) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= R) return;

        int offset = r * n_nodes;
        float cut = 0.0f;
        for (int e = 0; e < m; e++) {
            if (population[offset + ei[e]] != population[offset + ej[e]])
                cut += ew[e];
        }
        cuts[r] = cut;
    }
    ''', 'batch_cut')


# ============================================================
# GPU helper functions
# ============================================================

def _gpu_compute_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, n_nodes):
    """Launch delta computation kernel."""
    block = 256
    grid = (n_nodes + block - 1) // block
    _delta_kernel((grid,), (block,),
                  (d_x, d_row_ptr, d_col_idx, d_weights, d_delta, np.int32(n_nodes)))


def _gpu_update_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, v, deg_v):
    """Launch delta update kernel after flipping node v."""
    n_threads = deg_v + 1
    block = min(256, n_threads)
    grid = (n_threads + block - 1) // block
    _update_delta_kernel((grid,), (block,),
                         (d_x, d_row_ptr, d_col_idx, d_weights, d_delta, np.int32(v)))


def _gpu_find_best(d_delta, d_tabu, iteration, n_nodes):
    """Find node with highest positive non-tabu delta. Returns (v, delta_v)."""
    block = 256
    grid = (n_nodes + block - 1) // block
    d_bval = cp.empty(grid, dtype=cp.float32)
    d_bidx = cp.empty(grid, dtype=cp.int32)
    smem = block * (4 + 4)  # float + int per thread
    _argmax_kernel((grid,), (block,),
                   (d_delta, d_tabu, np.int64(iteration), np.int32(n_nodes),
                    d_bval, d_bidx),
                   shared_mem=smem)
    # Reduce block winners on CPU (few blocks)
    bval = d_bval.get()
    bidx = d_bidx.get()
    best = int(np.argmax(bval))
    if bval[best] <= 0:
        return -1, 0.0
    return int(bidx[best]), float(bval[best])


def _gpu_compute_cut(d_x, d_ei, d_ej, d_ew, m):
    """Compute cut value on GPU."""
    block = 256
    grid = (m + block - 1) // block
    d_partial = cp.empty(grid, dtype=cp.float32)
    _cut_kernel((grid,), (block,),
                (d_x, d_ei, d_ej, d_ew, d_partial, np.int32(m)),
                shared_mem=block * 4)
    return float(cp.sum(d_partial))


# ============================================================
# GPU BLS
# ============================================================

def cuda_bls_maxcut(n_nodes, edges, n_restarts=10, max_iter=1000,
                    max_no_improve=100, tabu_tenure=None, perturb_min=None,
                    perturb_max=None, time_limit=None, seed=None, verbose=False):
    """
    GPU-accelerated Breakout Local Search for MaxCut.

    Drop-in replacement for bls_maxcut(). Falls back to CPU if no GPU.

    The hot path (delta computation + argmax + delta update) runs on GPU.
    Perturbation and restart logic stays on CPU (negligible cost).
    """
    if not _HAS_GPU:
        from bls_solver import bls_maxcut
        return bls_maxcut(n_nodes, edges, n_restarts=n_restarts,
                          max_iter=max_iter, max_no_improve=max_no_improve,
                          tabu_tenure=tabu_tenure, perturb_min=perturb_min,
                          perturb_max=perturb_max, time_limit=time_limit,
                          seed=seed, verbose=verbose)

    t0 = time.time()
    rng = np.random.default_rng(seed)

    # Defaults
    if tabu_tenure is None:
        tabu_tenure = max(7, n_nodes // 20)
    if perturb_min is None:
        perturb_min = max(1, n_nodes // 50)
    if perturb_max is None:
        perturb_max = max(perturb_min + 1, n_nodes // 10)

    # Build CSR graph on GPU
    row_ptr, col_idx, weights = _build_csr(n_nodes, edges)
    d_row_ptr = cp.asarray(row_ptr)
    d_col_idx = cp.asarray(col_idx)
    d_weights = cp.asarray(weights)

    # Edge arrays for cut computation
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] if len(e) > 2 else 1.0 for e in edges], dtype=np.float32)
    d_ei = cp.asarray(ei)
    d_ej = cp.asarray(ej)
    d_ew = cp.asarray(ew)
    m = len(edges)

    # Degree array for delta updates
    deg = np.diff(row_ptr)

    # Global best
    global_best_cut = -1.0
    global_best_x = None
    total_iters = 0

    for restart in range(n_restarts):
        if time_limit and (time.time() - t0) >= time_limit:
            break

        # Random initial solution
        x = rng.integers(0, 2, size=n_nodes).astype(np.int32)
        d_x = cp.asarray(x)
        d_delta = cp.zeros(n_nodes, dtype=cp.float32)
        d_tabu = cp.zeros(n_nodes, dtype=cp.int64)

        # Compute initial deltas on GPU
        _gpu_compute_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, n_nodes)

        # Initial steepest ascent
        while True:
            v, dv = _gpu_find_best(d_delta, d_tabu, 0, n_nodes)
            if v < 0:
                break
            d_x[v] = 1 - d_x[v]
            _gpu_update_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, v, int(deg[v]))

        current_cut = _gpu_compute_cut(d_x, d_ei, d_ej, d_ew, m)
        best_cut = current_cut
        best_x_gpu = d_x.copy()

        no_improve = 0

        for iteration in range(1, max_iter + 1):
            if time_limit and (time.time() - t0) >= time_limit:
                break

            total_iters += 1

            # Perturbation (on CPU, copy back) - fast enough
            x_cpu = d_x.get()

            if no_improve < max_no_improve // 3:
                strength = perturb_min
            elif no_improve < 2 * max_no_improve // 3:
                strength = (perturb_min + perturb_max) // 2
            else:
                strength = perturb_max

            nodes = rng.choice(n_nodes, size=min(strength, n_nodes), replace=False)
            for v in nodes:
                x_cpu[v] = 1 - x_cpu[v]

            d_x = cp.asarray(x_cpu)

            # Set tabu on perturbed nodes
            for v in nodes:
                d_tabu[int(v)] = iteration + tabu_tenure + rng.integers(0, 3)

            # Recompute deltas on GPU (cheaper than incremental after many flips)
            _gpu_compute_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, n_nodes)

            # Steepest ascent on GPU
            while True:
                v, dv = _gpu_find_best(d_delta, d_tabu, iteration, n_nodes)
                if v < 0:
                    break
                d_x[v] = 1 - d_x[v]
                _gpu_update_deltas(d_x, d_row_ptr, d_col_idx, d_weights, d_delta, v, int(deg[v]))

            current_cut = _gpu_compute_cut(d_x, d_ei, d_ej, d_ew, m)

            if current_cut > best_cut:
                best_cut = current_cut
                best_x_gpu = d_x.copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= max_no_improve:
                break

        if best_cut > global_best_cut:
            global_best_cut = best_cut
            global_best_x = best_x_gpu.get()

        if verbose:
            elapsed = time.time() - t0
            print(f'  BLS restart {restart+1}/{n_restarts}: cut={best_cut:.0f} (global={global_best_cut:.0f}), t={elapsed:.2f}s')

    elapsed = time.time() - t0
    assignment = {i: int(global_best_x[i]) for i in range(n_nodes)}

    return {
        'best_cut': global_best_cut,
        'assignment': assignment,
        'n_restarts_done': restart + 1 if n_restarts > 0 else 0,
        'total_iterations': total_iters,
        'time_s': elapsed,
        'device': 'cuda',
    }


# ============================================================
# GPU Population Annealing
# ============================================================

def cuda_pa_maxcut(n_nodes, edges, n_replicas=100, n_temps=50,
                   beta_min=0.1, beta_max=5.0, n_sweeps=3,
                   schedule="geometric", time_limit=None,
                   seed=None, verbose=False):
    """
    GPU-accelerated Population Annealing for MaxCut.

    Drop-in replacement for pa_maxcut(). Falls back to CPU if no GPU.

    Each Metropolis sweep node-visit launches R threads (one per replica).
    Cut computation also fully on GPU.
    """
    if not _HAS_GPU:
        from pa_solver import pa_maxcut
        return pa_maxcut(n_nodes, edges, n_replicas=n_replicas,
                         n_temps=n_temps, beta_min=beta_min, beta_max=beta_max,
                         n_sweeps=n_sweeps, schedule=schedule,
                         time_limit=time_limit, seed=seed, verbose=verbose)

    t0 = time.time()
    rng = np.random.default_rng(seed)

    # Build CSR graph on GPU
    row_ptr, col_idx, weights = _build_csr(n_nodes, edges)
    d_row_ptr = cp.asarray(row_ptr)
    d_col_idx = cp.asarray(col_idx)
    d_weights = cp.asarray(weights)

    # Edge arrays for cut computation
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] if len(e) > 2 else 1.0 for e in edges], dtype=np.float32)
    d_ei = cp.asarray(ei)
    d_ej = cp.asarray(ej)
    d_ew = cp.asarray(ew)
    m = len(edges)

    # Temperature schedule
    if schedule == "geometric":
        betas = np.geomspace(max(beta_min, 1e-6), beta_max, n_temps)
    else:
        betas = np.linspace(beta_min, beta_max, n_temps)

    # Initialize random population on GPU
    pop_np = rng.integers(0, 2, size=(n_replicas, n_nodes)).astype(np.int32)
    d_pop = cp.asarray(pop_np)  # (R, n_nodes) row-major int32

    # Compute initial cuts
    d_cuts = cp.zeros(n_replicas, dtype=cp.float32)
    block = 256
    grid_r = (n_replicas + block - 1) // block
    _batch_cut_kernel((grid_r,), (block,),
                      (d_pop, d_ei, d_ej, d_ew, d_cuts,
                       np.int32(n_nodes), np.int32(m), np.int32(n_replicas)))

    # Track best
    best_idx = int(cp.argmax(d_cuts))
    best_cut = float(d_cuts[best_idx])
    best_x = d_pop[best_idx].get().copy()

    history = []
    beta_prev = 0.0

    for step, beta in enumerate(betas):
        if time_limit is not None and (time.time() - t0) >= time_limit:
            break

        R = d_pop.shape[0]
        grid_r = (R + block - 1) // block

        # 1. Resample (on CPU, small overhead)
        if step > 0:
            dbeta = beta - beta_prev
            if abs(dbeta) > 1e-15:
                cuts_cpu = d_cuts.get()
                log_w = dbeta * cuts_cpu
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                w_sum = np.sum(w)
                if w_sum > 1e-15:
                    probs = w / w_sum
                    indices = rng.choice(R, size=n_replicas, p=probs)
                    d_pop = d_pop[indices].copy()
                    d_cuts = d_cuts[indices].copy()
                    R = n_replicas
                    grid_r = (R + block - 1) // block

        # 2. Metropolis sweeps on GPU
        order = rng.permutation(n_nodes)
        for _ in range(n_sweeps):
            for v in order:
                d_rand = cp.asarray(rng.random(R).astype(np.float32))
                _metropolis_kernel(
                    (grid_r,), (block,),
                    (d_pop, d_row_ptr, d_col_idx, d_weights,
                     np.float32(beta), d_rand,
                     np.int32(n_nodes), np.int32(int(v)), np.int32(R)))

        # 3. Recompute cuts on GPU
        _batch_cut_kernel((grid_r,), (block,),
                          (d_pop, d_ei, d_ej, d_ew, d_cuts,
                           np.int32(n_nodes), np.int32(m), np.int32(R)))

        # 4. Track best
        step_best_idx = int(cp.argmax(d_cuts))
        step_best = float(d_cuts[step_best_idx])
        if step_best > best_cut:
            best_cut = step_best
            best_x = d_pop[step_best_idx].get().copy()

        mean_cut = float(cp.mean(d_cuts))
        history.append((float(beta), mean_cut, step_best))

        if verbose and (step % max(1, n_temps // 10) == 0 or step == len(betas) - 1):
            t = time.time() - t0
            print(f'  GPU-PA step {step+1}/{n_temps}: beta={beta:.3f}, mean={mean_cut:.1f}, best={best_cut:.1f}, t={t:.2f}s')

        beta_prev = beta

    # Final greedy local search on CPU (polish best replica)
    from pa_solver import _greedy_local_search, _build_adj_arrays
    adj_arr, wt_arr, deg = _build_adj_arrays(n_nodes, edges)
    ei_np = np.array([e[0] for e in edges], dtype=np.int32)
    ej_np = np.array([e[1] for e in edges], dtype=np.int32)
    ew_np = np.array([e[2] if len(e) > 2 else 1.0 for e in edges], dtype=np.float64)
    best_x_i32 = best_x.astype(np.int32)
    best_x_i32, best_cut = _greedy_local_search(best_x_i32, adj_arr, wt_arr, deg, ei_np, ej_np, ew_np)

    elapsed = time.time() - t0
    assignment = {i: int(best_x_i32[i]) for i in range(n_nodes)}

    return {
        'best_cut': best_cut,
        'assignment': assignment,
        'history': history,
        'n_temps_done': len(history),
        'time_s': elapsed,
        'device': 'cuda',
    }


# ============================================================
# Unified API: auto-selects GPU or CPU
# ============================================================

def _should_use_cpu_pa(n_nodes, edges, n_replicas=100, n_temps=50,
                       n_sweeps=3, time_limit=None, **_ignored):
    """
    Heuristic dispatch for PA backend.

    Current GPU-PA launches one kernel per node visit, so on laptop-scale
    benchmarks with modest time budgets the CPU/vectorized implementation can
    finish far more temperature steps and win both on quality and wall time.
    """
    if not _HAS_GPU:
        return True

    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0
    launch_work = n_nodes * max(n_replicas, 1) * max(n_sweeps, 1)

    if time_limit is not None and time_limit <= 15.0 and launch_work >= 50000:
        return True

    if launch_work >= 250000 and avg_deg <= 64.0:
        return True

    return False


def _autotune_pa_kwargs(n_nodes, edges, kwargs):
    """Adjust PA schedule for known short-budget regimes."""
    tuned = dict(kwargs)
    notes = []

    time_limit = tuned.get('time_limit')
    n_temps = tuned.get('n_temps', 50)
    n_sweeps = tuned.get('n_sweeps', 3)
    n_replicas = tuned.get('n_replicas', 100)
    beta_max = tuned.get('beta_max', 5.0)

    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0

    if (time_limit is not None and time_limit <= 15.0 and
            n_nodes >= 5000 and avg_deg <= 4.5 and
            n_temps >= 40 and n_sweeps >= 5):
        tuned['n_temps'] = 20
        tuned['n_sweeps'] = 5
        tuned['n_replicas'] = n_replicas
        tuned['beta_max'] = beta_max
        notes.append('short-budget-sparse-schedule')

    return tuned, notes


def _should_use_sparse_dsbm_route(n_nodes, edges, time_limit=None, **_ignored):
    """Detect very large sparse graphs where dSBM is the stronger laptop route."""
    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0
    if time_limit is None:
        time_limit = 0.0
    if time_limit > 15.0:
        return False
    if n_nodes >= 5000 and avg_deg <= 12.0:
        return True
    return False


def _should_use_medium_sparse_probe(n_nodes, edges, time_limit=None, **_ignored):
    """
    Detect medium-large graphs where a cheap dSBM probe should compete
    against a CPU PA light run.

    This targets the awkward 2k-4k node laptop families where the robust
    CPU/GPU PA dispatch can underperform either dSBM or a plain CPU PA light
    schedule, while dSBM is cheap enough to probe first.
    """
    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0
    if time_limit is None or time_limit <= 0.0:
        return False
    if time_limit > 15.0:
        return False
    if 2000 <= n_nodes < 5000:
        return True
    return False


def _should_use_dense_dsbm_direct_route(n_nodes, edges, time_limit=None, **_ignored):
    """Detect the G48/G49-style pocket where direct dense dSBM dominates."""
    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0
    if time_limit is None or time_limit <= 0.0:
        return False
    if time_limit > 15.0:
        return False
    if 2800 <= n_nodes <= 3500 and avg_deg <= 4.5:
        return True
    return False


def _should_use_dense2k_pa_only_route(n_nodes, edges, time_limit=None, **_ignored):
    """Detect the 2k-node family where pure CPU PA-light beats probe overhead."""
    if time_limit is None or time_limit <= 0.0:
        return False
    if time_limit > 15.0:
        return False
    if 2000 <= n_nodes < 3000:
        return True
    return False


def _normalize_assignment_bits(assignment):
    """Convert mixed {-1,+1} / {0,1} assignments to {0,1}."""
    out = {}
    for key, value in assignment.items():
        iv = int(value)
        out[int(key)] = 1 if iv > 0 else 0
    return out


def _run_cpu_pa_light(n_nodes, edges, seed=42, time_limit=None):
    """Reference-quality CPU PA probe for medium-size graph families."""
    from pa_solver import pa_maxcut

    n_replicas = 150 if n_nodes <= 2500 else 100
    result = pa_maxcut(
        n_nodes,
        edges,
        n_replicas=n_replicas,
        n_temps=50,
        n_sweeps=3,
        time_limit=time_limit,
        seed=seed,
    )
    result['device'] = result.get('device', 'cpu')
    result['solver_note'] = f'cpu-pa-light-probe(replicas={n_replicas})'
    return result


def _strong_sparse_dsbm_schedule(time_limit):
    """Choose a stronger sparse-dSBM schedule that actually uses laptop budget."""
    if time_limit is None or time_limit <= 0.0:
        return 32, 1200
    if time_limit <= 2.0:
        return 16, 400
    if time_limit <= 5.0:
        return 32, 1200
    if time_limit <= 10.0:
        return 96, 3000
    return 96, 3000


def _high_degree_sparse_dsbm_config(n_nodes, edges, time_limit):
    """Seed-averaged winning config for the high-degree sparse pocket."""
    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0
    if n_nodes >= 5000 and avg_deg >= 10.0:
        if time_limit is None or time_limit <= 0.0:
            return 32, 1200, 2.0
        if time_limit <= 2.0:
            return 16, 400, 2.0
        if time_limit <= 5.0:
            return 32, 1200, 2.0
        if time_limit <= 10.0:
            return 64, 3000, 2.0
        return 96, 3000, 2.0
    return None


def _run_dense_dsbm_direct(n_nodes, edges, seed=42, time_limit=None):
    """Direct dense dSBM route for the 3k-node low-degree pocket."""
    from quantum_inspired_baselines import run_dsbm_maxcut

    result = run_dsbm_maxcut(
        n_nodes,
        edges,
        seed=seed,
        time_limit=time_limit,
        num_restarts=32,
        steps=1200,
    )
    result['assignment'] = _normalize_assignment_bits(result.get('assignment', {}))
    result['solver_note'] = 'dense-dsbm-direct(restarts=32,steps=1200)'
    return result


def _run_sparse_dsbm_strong(n_nodes, edges, seed=42, time_limit=None):
    """High-quality sparse dSBM route for large sparse Gset families."""
    from quantum_inspired_baselines import run_dsbm_maxcut

    tuned = _high_degree_sparse_dsbm_config(n_nodes, edges, time_limit)
    if tuned is None:
        restarts, steps = _strong_sparse_dsbm_schedule(time_limit)
        c0_scale = 1.0
    else:
        restarts, steps, c0_scale = tuned
    result = run_dsbm_maxcut(
        n_nodes,
        edges,
        seed=seed,
        time_limit=time_limit,
        num_restarts=restarts,
        steps=steps,
        c0_scale=c0_scale,
    )
    result['assignment'] = _normalize_assignment_bits(result.get('assignment', {}))
    note = f'sparse-dsbm-strong(restarts={restarts},steps={steps})'
    if c0_scale != 1.0:
        note += f'[c0x{c0_scale:g}]'
    result['solver_note'] = note
    return result


def maxcut_pa_sparse_hybrid(n_nodes, edges, **kwargs):
    """
    Sparse-specialized PA route.

    Strategy:
      - very large sparse + short budget: dispatch directly to dSBM
      - dense 2k family: skip probe overhead and run CPU PA-light directly
      - mid-large laptop families: quick dSBM probe plus CPU PA light, return
        the better result
      - otherwise: standard maxcut_pa path
    """
    pa_kwargs, tune_notes = _autotune_pa_kwargs(n_nodes, edges, kwargs)
    time_limit = pa_kwargs.get('time_limit')
    m = len(edges)
    avg_deg = (2.0 * m / max(n_nodes, 1)) if n_nodes > 0 else 0.0

    if _should_use_sparse_dsbm_route(n_nodes, edges, **pa_kwargs):
        r = _run_sparse_dsbm_strong(
            n_nodes, edges,
            seed=pa_kwargs.get('seed', 42),
            time_limit=time_limit,
        )
        note_parts = [r.get('solver_note', 'sparse-dsbm-strong'),
                      'sparse-dsbm-direct'] + tune_notes
        r['solver_note'] = '+'.join(note_parts)
        return r

    if _should_use_dense_dsbm_direct_route(n_nodes, edges, **pa_kwargs):
        r = _run_dense_dsbm_direct(
            n_nodes, edges,
            seed=pa_kwargs.get('seed', 42),
            time_limit=time_limit,
        )
        note_parts = [r.get('solver_note', 'dense-dsbm-direct')] + tune_notes
        r['solver_note'] = '+'.join(note_parts)
        return r

    if _should_use_dense2k_pa_only_route(n_nodes, edges, **pa_kwargs):
        r = _run_cpu_pa_light(
            n_nodes, edges,
            seed=pa_kwargs.get('seed', 42),
            time_limit=time_limit,
        )
        note_parts = [r.get('solver_note', 'cpu-pa-light-probe'),
                      'dense2k-pa-only'] + tune_notes
        r['solver_note'] = '+'.join(note_parts)
        return r

    medium_sparse_probe = _should_use_medium_sparse_probe(
        n_nodes, edges, **pa_kwargs)
    if medium_sparse_probe:
        from quantum_inspired_baselines import run_dsbm_maxcut

        dsbm_budget = max(0.5, min(2.0, 0.2 * time_limit))
        pa_budget = max(0.0, time_limit - dsbm_budget)

        r_dsbm = run_dsbm_maxcut(
            n_nodes, edges,
            seed=pa_kwargs.get('seed', 42),
            time_limit=dsbm_budget,
        )
        r_dsbm['assignment'] = _normalize_assignment_bits(r_dsbm.get('assignment', {}))
        r_dsbm['solver_note'] = '+'.join(['medium-sparse-dsbm-probe'] + tune_notes)

        if pa_budget >= 2.0:
            r_pa = _run_cpu_pa_light(
                n_nodes, edges,
                seed=pa_kwargs.get('seed', 42),
                time_limit=pa_budget,
            )
            if r_pa['best_cut'] >= r_dsbm['best_cut']:
                r_pa['time_s'] = r_pa.get('time_s', 0.0) + r_dsbm.get('time_s', 0.0)
                base_note = r_pa.get('solver_note', 'pa')
                r_pa['solver_note'] = (
                    f'{base_note}+medium-sparse-dsbm-probe({r_dsbm["best_cut"]:.0f})'
                )
                return r_pa

            r_dsbm['time_s'] = r_dsbm.get('time_s', 0.0) + r_pa.get('time_s', 0.0)
            r_dsbm['solver_note'] = (
                f'{r_dsbm["solver_note"]}+cpu-pa-light({r_pa["best_cut"]:.0f})'
            )
            return r_dsbm

        return r_dsbm

    return maxcut_pa(n_nodes, edges, **pa_kwargs)

def maxcut_bls(n_nodes, edges, **kwargs):
    """BLS MaxCut: GPU if available, else CPU."""
    if _HAS_GPU:
        return cuda_bls_maxcut(n_nodes, edges, **kwargs)
    from bls_solver import bls_maxcut
    return bls_maxcut(n_nodes, edges, **kwargs)


def maxcut_pa(n_nodes, edges, **kwargs):
    """PA MaxCut: GPU if available, else CPU."""
    pa_kwargs, tune_notes = _autotune_pa_kwargs(n_nodes, edges, kwargs)

    if not _should_use_cpu_pa(n_nodes, edges, **pa_kwargs):
        result = cuda_pa_maxcut(n_nodes, edges, **pa_kwargs)
        if tune_notes:
            result['solver_note'] = '+'.join(tune_notes)
        return result
    from pa_solver import pa_maxcut
    result = pa_maxcut(n_nodes, edges, **pa_kwargs)
    result['device'] = result.get('device', 'cpu')
    note_parts = ['cpu-pa-dispatch'] + tune_notes
    result['solver_note'] = '+'.join(note_parts)
    return result


def gpu_available():
    """Check if CUDA GPU is available."""
    return _HAS_GPU


# ============================================================
# Demo / self-test
# ============================================================

if __name__ == '__main__':
    print(f'=== B136 CUDA Local Search ===')
    print(f'GPU available: {_HAS_GPU}')
    if _HAS_GPU:
        dev = cp.cuda.Device(0)
        print(f'Device: {dev.id}, Compute: {dev.compute_capability}')
        mem = dev.mem_info
        print(f'VRAM: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free')
    print()

    from bls_solver import random_3regular, random_erdos_renyi, bls_maxcut
    from pa_solver import pa_maxcut

    # Compare GPU vs CPU BLS
    print('--- BLS: GPU vs CPU ---')
    for n in [100, 500, 1000, 2000]:
        nn, edges = random_3regular(n, seed=n * 7)

        t0 = time.time()
        r_cpu = bls_maxcut(nn, edges, n_restarts=5, max_iter=500, seed=42)
        t_cpu = time.time() - t0

        t0 = time.time()
        r_gpu = maxcut_bls(nn, edges, n_restarts=5, max_iter=500, seed=42)
        t_gpu = time.time() - t0

        speedup = t_cpu / max(t_gpu, 1e-6)
        dev = r_gpu.get('device', 'cpu')
        print(f'  n={n:5d}: CPU={r_cpu["best_cut"]:.0f} ({t_cpu:.3f}s)  '
              f'{dev.upper()}={r_gpu["best_cut"]:.0f} ({t_gpu:.3f}s)  '
              f'speedup={speedup:.1f}x')

    # Compare GPU vs CPU PA
    print('\n--- PA: GPU vs CPU ---')
    for n in [100, 500, 1000]:
        nn, edges = random_3regular(n, seed=n * 7)

        t0 = time.time()
        r_cpu = pa_maxcut(nn, edges, n_replicas=100, n_temps=30, seed=42)
        t_cpu = time.time() - t0

        t0 = time.time()
        r_gpu = maxcut_pa(nn, edges, n_replicas=100, n_temps=30, seed=42)
        t_gpu = time.time() - t0

        speedup = t_cpu / max(t_gpu, 1e-6)
        dev = r_gpu.get('device', 'cpu')
        print(f'  n={n:5d}: CPU={r_cpu["best_cut"]:.0f} ({t_cpu:.3f}s)  '
              f'{dev.upper()}={r_gpu["best_cut"]:.0f} ({t_gpu:.3f}s)  '
              f'speedup={speedup:.1f}x')

    print('\nDone.')
