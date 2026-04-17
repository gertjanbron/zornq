#!/usr/bin/env python3
"""Quantum-inspired MaxCut baselines: SimCIM and dSBM.

Gebaseerd op de P101 LATRE referentie-implementaties, maar hier verpakt
als ZornQ-vriendelijke benchmark-adapters met tijdslimiet en een dense/
sparse backend-keuze zodat ook grotere Gset-families haalbaar blijven.
"""

import math
import time

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None


MAX_DENSE_NODES = 3000


def _require_torch():
    if torch is None:
        raise ImportError("torch is vereist voor SimCIM/dSBM")


def _choose_backend(n_nodes):
    return 'dense' if n_nodes <= MAX_DENSE_NODES else 'sparse'


def _choose_device(n_nodes, backend, device=None):
    _require_torch()
    if device:
        return device
    if torch.cuda.is_available():
        if backend == 'sparse' or n_nodes <= 2500:
            return 'cuda'
    return 'cpu'


def np_clip(value, low, high):
    return max(low, min(high, value))


def edge_list_to_dense_j(n_nodes, edges, device='cpu', dtype=None):
    """Converteer edge-lijst naar symmetrische dense J-matrix."""
    _require_torch()
    if dtype is None:
        dtype = torch.float32
    J = torch.zeros((n_nodes, n_nodes), device=device, dtype=dtype)
    for edge in edges:
        u = int(edge[0])
        v = int(edge[1])
        w = float(edge[2]) if len(edge) >= 3 else 1.0
        if u == v:
            continue
        J[u, v] += w
        J[v, u] += w
    J.fill_diagonal_(0)
    return J


def edge_list_to_sparse_j(n_nodes, edges, device='cpu', dtype=None):
    """Converteer edge-lijst naar compacte sparse edge-arrays."""
    _require_torch()
    if dtype is None:
        dtype = torch.float32
    if not edges:
        empty_idx = torch.empty(0, dtype=torch.long, device=device)
        empty_w = torch.empty(0, dtype=dtype, device=device)
        return empty_idx, empty_idx, empty_w, torch.tensor(0.0, dtype=dtype, device=device)

    u_idx = torch.tensor([int(e[0]) for e in edges], dtype=torch.long, device=device)
    v_idx = torch.tensor([int(e[1]) for e in edges], dtype=torch.long, device=device)
    weights = torch.tensor(
        [float(e[2]) if len(e) >= 3 else 1.0 for e in edges],
        dtype=dtype,
        device=device,
    )
    total_weight = weights.sum()
    return u_idx, v_idx, weights, total_weight


def _cut_from_spins_dense(J, spins):
    spins = spins.float()
    sJs = (spins @ J * spins).sum(dim=1)
    return (J.sum() - sJs) / 4.0


def _cut_from_spins_sparse(total_weight, weights, spins, u_idx, v_idx):
    if weights.numel() == 0:
        return torch.zeros(spins.shape[0], device=spins.device, dtype=spins.dtype)
    edge_corr = spins[:, u_idx] * spins[:, v_idx]
    return (total_weight - (edge_corr * weights).sum(dim=1)) / 2.0


def _edge_matvec(x, u_idx, v_idx, weights, negate=False):
    """Bereken x @ J via edge-lijst zonder dense matrix."""
    if weights.numel() == 0:
        return torch.zeros_like(x)

    scaled_w = -weights if negate else weights
    source_u = x[:, v_idx] * scaled_w
    source_v = x[:, u_idx] * scaled_w

    jx = torch.zeros_like(x)
    jx.index_add_(1, u_idx, source_u)
    jx.index_add_(1, v_idx, source_v)
    return jx


def _spins_to_assignment(spins):
    spins = spins.detach().cpu()
    return {
        int(i): int(1 if spins[i].item() >= 0 else -1)
        for i in range(spins.numel())
    }


def _implicit_dense_std(n_nodes, weights):
    """Std van de impliciete dense symmetrische J-matrix met veel nullen."""
    if weights.numel() == 0 or n_nodes <= 0:
        return 0.0
    total_entries = float(n_nodes * n_nodes)
    mean = float((2.0 * weights.sum()).item() / total_entries)
    mean_sq = float((2.0 * (weights * weights).sum()).item() / total_entries)
    var = max(mean_sq - mean * mean, 0.0)
    return math.sqrt(var)


def _budget_params(n_nodes, time_limit, family):
    """Kies restart/step-budget op basis van grootte en wandklokbudget."""
    if time_limit is None or time_limit <= 0:
        time_limit = 30.0

    scale = max(1.0, n_nodes / 250.0)
    time_factor = min(2.5, max(0.35, time_limit / 30.0))

    if family == 'simcim':
        base_steps = int(np_clip(700 / scale, 200, 1600))
        base_restarts = int(np_clip(64 / math.sqrt(scale), 12, 64))
    else:
        base_steps = int(np_clip(900 / scale, 250, 1800))
        base_restarts = int(np_clip(64 / math.sqrt(scale), 12, 64))

    restarts_cap = max(8, 250000 // max(n_nodes, 1))
    steps = max(100, int(base_steps * time_factor))
    restarts = max(
        8,
        min(restarts_cap, int(round(base_restarts * math.sqrt(time_factor)))),
    )
    return restarts, steps


def simcim_solve_dense(J, num_restarts=64, steps=1000, p_min=-1.0, p_max=1.0,
                       zeta=0.1, sigma=0.03, dt=0.05, device='cpu',
                       seed=None, time_limit=None):
    """P101 SimCIM, dense backend."""
    _require_torch()
    if seed is not None:
        torch.manual_seed(seed)

    J = J.to(device=device, dtype=torch.float32)
    J = J - torch.diag(torch.diag(J))
    n_nodes = J.shape[0]
    x = 0.01 * torch.randn(num_restarts, n_nodes, device=device)
    pump_schedule = torch.linspace(p_min, p_max, steps, device=device)
    J_dyn = -J

    best_cut = float('-inf')
    best_spins = None
    started = time.time()

    for t in range(steps):
        p_t = pump_schedule[t]
        jx = x @ J_dyn
        drift = (p_t - 1.0 - x * x) * x + zeta * jx
        noise = sigma * torch.randn_like(x)
        x = x + dt * drift + noise * (dt ** 0.5)

        if t % 25 == 0 or t == steps - 1:
            spins = torch.sign(x)
            spins[spins == 0] = 1
            cuts = _cut_from_spins_dense(J, spins)
            local_best, idx = cuts.max(dim=0)
            local_best = float(local_best.item())
            if local_best > best_cut:
                best_cut = local_best
                best_spins = spins[idx].clone()
            if time_limit and (time.time() - started) >= time_limit:
                break

    return float(best_cut), best_spins


def simcim_solve_sparse(n_nodes, edges, num_restarts=64, steps=1000,
                        p_min=-1.0, p_max=1.0, zeta=0.1, sigma=0.03,
                        dt=0.05, device='cpu', seed=None, time_limit=None):
    """P101 SimCIM, sparse edge-list backend."""
    _require_torch()
    if seed is not None:
        torch.manual_seed(seed)

    u_idx, v_idx, weights, total_weight = edge_list_to_sparse_j(
        n_nodes, edges, device=device)
    x = 0.01 * torch.randn(num_restarts, n_nodes, device=device)
    pump_schedule = torch.linspace(p_min, p_max, steps, device=device)

    best_cut = float('-inf')
    best_spins = None
    started = time.time()

    for t in range(steps):
        p_t = pump_schedule[t]
        jx = _edge_matvec(x, u_idx, v_idx, weights, negate=True)
        drift = (p_t - 1.0 - x * x) * x + zeta * jx
        noise = sigma * torch.randn_like(x)
        x = x + dt * drift + noise * (dt ** 0.5)

        if t % 25 == 0 or t == steps - 1:
            spins = torch.sign(x)
            spins[spins == 0] = 1
            cuts = _cut_from_spins_sparse(total_weight, weights, spins, u_idx, v_idx)
            local_best, idx = cuts.max(dim=0)
            local_best = float(local_best.item())
            if local_best > best_cut:
                best_cut = local_best
                best_spins = spins[idx].clone()
            if time_limit and (time.time() - started) >= time_limit:
                break

    return float(best_cut), best_spins


def dsbm_solve_dense(J, num_restarts=64, steps=1000, a0=1.0, c0=None, dt=0.5,
                     device='cpu', seed=None, time_limit=None):
    """P101 discrete SBM, dense backend."""
    _require_torch()
    if seed is not None:
        torch.manual_seed(seed)

    J = J.to(device=device, dtype=torch.float32)
    J = J - torch.diag(torch.diag(J))
    n_nodes = J.shape[0]

    if c0 is None:
        c0 = 0.5 / (math.sqrt(max(n_nodes, 1)) * float(J.std().item()) + 1e-9)

    J_dyn = -J
    x = 0.1 * torch.randn(num_restarts, n_nodes, device=device)
    y = 0.1 * torch.randn(num_restarts, n_nodes, device=device)

    best_cut = float('-inf')
    best_spins = None
    started = time.time()

    for t in range(steps):
        a_t = (t + 1) / steps
        jx = x @ J_dyn
        y = y + dt * (-(a0 - a_t) * x - a0 * x ** 3 + c0 * jx)
        x = x + dt * a0 * y

        over = x.abs() > 1.0
        x = torch.where(over, torch.sign(x), x)
        y = torch.where(over, torch.zeros_like(y), y)

        if t % 25 == 0 or t == steps - 1:
            spins = torch.sign(x)
            spins[spins == 0] = 1
            cuts = _cut_from_spins_dense(J, spins)
            local_best, idx = cuts.max(dim=0)
            local_best = float(local_best.item())
            if local_best > best_cut:
                best_cut = local_best
                best_spins = spins[idx].clone()
            if time_limit and (time.time() - started) >= time_limit:
                break

    return float(best_cut), best_spins


def dsbm_solve_sparse(n_nodes, edges, num_restarts=64, steps=1000,
                      a0=1.0, c0=None, dt=0.5, device='cpu',
                      seed=None, time_limit=None):
    """P101 discrete SBM, sparse edge-list backend."""
    _require_torch()
    if seed is not None:
        torch.manual_seed(seed)

    u_idx, v_idx, weights, total_weight = edge_list_to_sparse_j(
        n_nodes, edges, device=device)
    if c0 is None:
        std = _implicit_dense_std(n_nodes, weights)
        c0 = 0.5 / (math.sqrt(max(n_nodes, 1)) * std + 1e-9)

    x = 0.1 * torch.randn(num_restarts, n_nodes, device=device)
    y = 0.1 * torch.randn(num_restarts, n_nodes, device=device)

    best_cut = float('-inf')
    best_spins = None
    started = time.time()

    for t in range(steps):
        a_t = (t + 1) / steps
        jx = _edge_matvec(x, u_idx, v_idx, weights, negate=True)
        y = y + dt * (-(a0 - a_t) * x - a0 * x ** 3 + c0 * jx)
        x = x + dt * a0 * y

        over = x.abs() > 1.0
        x = torch.where(over, torch.sign(x), x)
        y = torch.where(over, torch.zeros_like(y), y)

        if t % 25 == 0 or t == steps - 1:
            spins = torch.sign(x)
            spins[spins == 0] = 1
            cuts = _cut_from_spins_sparse(total_weight, weights, spins, u_idx, v_idx)
            local_best, idx = cuts.max(dim=0)
            local_best = float(local_best.item())
            if local_best > best_cut:
                best_cut = local_best
                best_spins = spins[idx].clone()
            if time_limit and (time.time() - started) >= time_limit:
                break

    return float(best_cut), best_spins


def run_simcim_maxcut(n_nodes, edges, seed=42, time_limit=None, device=None):
    """Benchmark-adapter voor SimCIM."""
    _require_torch()
    backend = _choose_backend(n_nodes)
    runtime_device = _choose_device(n_nodes, backend, device=device)
    restarts, steps = _budget_params(n_nodes, time_limit, family='simcim')

    t0 = time.time()
    if backend == 'dense':
        J = edge_list_to_dense_j(n_nodes, edges, device=runtime_device)
        best_cut, best_spins = simcim_solve_dense(
            J, num_restarts=restarts, steps=steps, device=runtime_device,
            seed=seed, time_limit=time_limit)
    else:
        best_cut, best_spins = simcim_solve_sparse(
            n_nodes, edges, num_restarts=restarts, steps=steps,
            device=runtime_device, seed=seed, time_limit=time_limit)
    elapsed = time.time() - t0

    return {
        'best_cut': float(best_cut),
        'assignment': _spins_to_assignment(best_spins) if best_spins is not None else {},
        'time_s': elapsed,
        'device': runtime_device,
        'solver_note': f'backend={backend},restarts={restarts},steps={steps}',
    }


def run_dsbm_maxcut(n_nodes, edges, seed=42, time_limit=None, device=None,
                    num_restarts=None, steps=None, c0_scale=1.0):
    """Benchmark-adapter voor discrete Simulated Bifurcation."""
    _require_torch()
    backend = _choose_backend(n_nodes)
    runtime_device = _choose_device(n_nodes, backend, device=device)
    auto_restarts, auto_steps = _budget_params(n_nodes, time_limit, family='dsbm')
    restarts = auto_restarts if num_restarts is None else int(num_restarts)
    steps = auto_steps if steps is None else int(steps)

    t0 = time.time()
    if backend == 'dense':
        J = edge_list_to_dense_j(n_nodes, edges, device=runtime_device)
        c0 = None
        if c0_scale != 1.0:
            c0 = (
                0.5 / (math.sqrt(max(n_nodes, 1)) * float(J.std().item()) + 1e-9)
            ) * c0_scale
        best_cut, best_spins = dsbm_solve_dense(
            J, num_restarts=restarts, steps=steps, c0=c0,
            device=runtime_device, seed=seed, time_limit=time_limit)
    else:
        c0 = None
        if c0_scale != 1.0:
            _, _, weights, _ = edge_list_to_sparse_j(
                n_nodes, edges, device=runtime_device)
            std = _implicit_dense_std(n_nodes, weights)
            c0 = (0.5 / (math.sqrt(max(n_nodes, 1)) * std + 1e-9)) * c0_scale
        best_cut, best_spins = dsbm_solve_sparse(
            n_nodes, edges, num_restarts=restarts, steps=steps, c0=c0,
            device=runtime_device, seed=seed, time_limit=time_limit)
    elapsed = time.time() - t0

    note = f'backend={backend},restarts={restarts},steps={steps}'
    if c0_scale != 1.0:
        note += f',c0_scale={c0_scale:g}'

    return {
        'best_cut': float(best_cut),
        'assignment': _spins_to_assignment(best_spins) if best_spins is not None else {},
        'time_s': elapsed,
        'device': runtime_device,
        'solver_note': note,
    }
