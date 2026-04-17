#!/usr/bin/env python3
"""Seed-averaged policy study for selected Gset families.

Compares the current pa_sparse_hybrid policy against targeted high-degree sparse
variants without changing the repo-wide default until evidence is strong enough.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from auto_dispatcher import ZornDispatcher
from cuda_local_search import _run_cpu_pa_light, maxcut_pa_sparse_hybrid
from gset_loader import load_gset
from multiscale_maxcut import multiscale_pa_maxcut
from pa_solver import pa_maxcut
from quantum_inspired_baselines import run_dsbm_maxcut


FAMILIES = {
    'highdeg-sparse': ['G58', 'G59', 'G63', 'G64'],
    'g59g64-pocket': ['G59', 'G64'],
    'hub2000': [f'G{k}' for k in range(35, 43)],
    'sparse-large': [f'G{k}' for k in range(56, 68)],
    'dense2000': [f'G{k}' for k in range(23, 43)],
    'signed2000': [f'G{k}' for k in range(27, 35)],
    'g29g32-pocket': ['G29', 'G30', 'G31', 'G32'],
}


def _numeric_gset_sort_key(name):
    stem = name.upper().replace('.TXT', '')
    if stem.startswith('G') and stem[1:].isdigit():
        return int(stem[1:])
    return 10**9


def _discover_gset_names():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    gset_dir = os.path.join(os.path.normpath(os.path.join(code_dir, '..')), 'gset')
    names = []
    for fname in os.listdir(gset_dir):
        stem = fname.upper().replace('.TXT', '')
        if stem.startswith('G') and stem[1:].isdigit():
            names.append(stem)
    return sorted(set(names), key=_numeric_gset_sort_key)


def _avg_degree(n_nodes, edges):
    return (2.0 * len(edges) / max(n_nodes, 1)) if n_nodes > 0 else 0.0


def _is_highdeg_sparse(n_nodes, edges):
    return n_nodes >= 5000 and _avg_degree(n_nodes, edges) >= 10.0


def _is_dense2000(n_nodes, edges):
    return 2000 <= n_nodes < 3000


def _is_signed_pm1(edges):
    saw_pos = False
    saw_neg = False
    for e in edges:
        w = float(e[2] if len(e) > 2 else 1.0)
        if abs(abs(w) - 1.0) > 1e-9:
            return False
        if w > 0:
            saw_pos = True
        elif w < 0:
            saw_neg = True
    return saw_pos and saw_neg


def _run_current(n_nodes, edges, seed, time_limit):
    return maxcut_pa_sparse_hybrid(n_nodes, edges, seed=seed, time_limit=time_limit)


def _run_highdeg_alt(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges):
        return run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
            num_restarts=64,
            steps=3000,
            c0_scale=2.0,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_best_of_runs(n_nodes, edges, runs):
    results = []
    for restarts, steps, c0_scale, seed, budget in runs:
        r = run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=budget,
            num_restarts=restarts,
            steps=steps,
            c0_scale=c0_scale,
        )
        results.append(r)
    best = max(results, key=lambda r: r['best_cut'])
    return {
        'best_cut': best['best_cut'],
        'time_s': sum(r.get('time_s', 0.0) for r in results),
        'device': best.get('device', 'cpu'),
        'solver_note': 'best-of[' + ';'.join(r.get('solver_note', '') for r in results) + ']',
    }


def _run_highdeg_dual_mix_a(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges) and time_limit >= 9.0:
        return _run_best_of_runs(
            n_nodes,
            edges,
            [
                (32, 1200, 2.0, seed, 2.5),
                (64, 3000, 2.5, seed + 1, max(0.0, time_limit - 2.5)),
            ],
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_highdeg_dual_mix_b(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges) and time_limit >= 9.0:
        return _run_best_of_runs(
            n_nodes,
            edges,
            [
                (32, 1200, 2.5, seed, 2.5),
                (64, 3000, 2.0, seed + 1, max(0.0, time_limit - 2.5)),
            ],
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_dense2k_pa_only(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges):
        return _run_cpu_pa_light(
            n_nodes, edges,
            seed=seed,
            time_limit=time_limit,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_dense2k_probe_1s(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and time_limit >= 2.0:
        probe_budget = min(1.0, max(0.5, 0.1 * time_limit))
        pa_budget = max(0.0, time_limit - probe_budget)
        r_dsbm = run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=probe_budget,
        )
        if pa_budget < 2.0:
            r_dsbm['solver_note'] = 'dense2k-dsbm-probe-only(1s)'
            return r_dsbm
        r_pa = _run_cpu_pa_light(
            n_nodes, edges,
            seed=seed,
            time_limit=pa_budget,
        )
        if r_pa['best_cut'] >= r_dsbm['best_cut']:
            r_pa['time_s'] = r_pa.get('time_s', 0.0) + r_dsbm.get('time_s', 0.0)
            r_pa['solver_note'] = (
                f"{r_pa.get('solver_note', 'cpu-pa-light')}+dense2k-dsbm-probe({r_dsbm['best_cut']:.0f})"
            )
            return r_pa
        r_dsbm['time_s'] = r_dsbm.get('time_s', 0.0) + r_pa.get('time_s', 0.0)
        r_dsbm['solver_note'] = (
            f"dense2k-dsbm-probe(1s)+cpu-pa-light({r_pa['best_cut']:.0f})"
        )
        return r_dsbm
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_dense2k_pa_heavy(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges):
        result = pa_maxcut(
            n_nodes,
            edges,
            n_replicas=200,
            n_temps=60,
            n_sweeps=3,
            time_limit=time_limit,
            seed=seed,
        )
        result['device'] = result.get('device', 'cpu')
        result['solver_note'] = 'dense2k-pa-heavy(replicas=200,temps=60)'
        return result
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_combo_highdeg_dense2k(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges):
        return _run_dense2k_pa_only(n_nodes, edges, seed, time_limit)
    if _is_highdeg_sparse(n_nodes, edges):
        return _run_highdeg_alt(n_nodes, edges, seed, time_limit)
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_signed2000_probe_1s(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and _is_signed_pm1(edges) and time_limit >= 2.0:
        probe_budget = min(1.0, max(0.5, 0.1 * time_limit))
        pa_budget = max(0.0, time_limit - probe_budget)
        r_dsbm = run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=probe_budget,
        )
        if pa_budget < 2.0:
            r_dsbm['solver_note'] = 'signed2000-dsbm-probe-only(1s)'
            return r_dsbm
        r_pa = _run_cpu_pa_light(
            n_nodes, edges,
            seed=seed,
            time_limit=pa_budget,
        )
        if r_pa['best_cut'] >= r_dsbm['best_cut']:
            r_pa['time_s'] = r_pa.get('time_s', 0.0) + r_dsbm.get('time_s', 0.0)
            r_pa['solver_note'] = (
                f"{r_pa.get('solver_note', 'cpu-pa-light')}+signed2000-dsbm-probe({r_dsbm['best_cut']:.0f})"
            )
            return r_pa
        r_dsbm['time_s'] = r_dsbm.get('time_s', 0.0) + r_pa.get('time_s', 0.0)
        r_dsbm['solver_note'] = (
            f"signed2000-dsbm-probe(1s)+cpu-pa-light({r_pa['best_cut']:.0f})"
        )
        return r_dsbm
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_signed2000_dsbm_auto(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and _is_signed_pm1(edges):
        r = run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
        )
        r['solver_note'] = 'signed2000-dsbm-auto'
        return r
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_signed2000_dsbm_tuned(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and _is_signed_pm1(edges):
        r = run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
            num_restarts=64,
            steps=3000,
        )
        r['solver_note'] = 'signed2000-dsbm-tuned(restarts=64,steps=3000)'
        return r
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_highdeg_c0_25(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges):
        return run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
            num_restarts=64,
            steps=3000,
            c0_scale=2.5,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_highdeg_96_c0_20(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges):
        return run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
            num_restarts=96,
            steps=3000,
            c0_scale=2.0,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_highdeg_96_c0_25(n_nodes, edges, seed, time_limit):
    if _is_highdeg_sparse(n_nodes, edges):
        return run_dsbm_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
            num_restarts=96,
            steps=3000,
            c0_scale=2.5,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_hub2000_pa_heavy(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and 10000 <= len(edges) <= 13000:
        result = pa_maxcut(
            n_nodes,
            edges,
            n_replicas=200,
            n_temps=60,
            n_sweeps=3,
            time_limit=time_limit,
            seed=seed,
        )
        result['device'] = result.get('device', 'cpu')
        result['solver_note'] = 'hub2000-pa-heavy(replicas=200,temps=60)'
        return result
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_hub2000_pa_wide(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and 10000 <= len(edges) <= 13000:
        result = pa_maxcut(
            n_nodes,
            edges,
            n_replicas=250,
            n_temps=50,
            n_sweeps=3,
            time_limit=time_limit,
            seed=seed,
        )
        result['device'] = result.get('device', 'cpu')
        result['solver_note'] = 'hub2000-pa-wide(replicas=250,temps=50)'
        return result
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_hub2000_pa_dual(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and 10000 <= len(edges) <= 13000:
        if time_limit is None or time_limit <= 0.0:
            return _run_current(n_nodes, edges, seed, time_limit)
        budget = max(0.5, time_limit / 2.0)
        r1 = _run_cpu_pa_light(n_nodes, edges, seed=seed, time_limit=budget)
        r2 = _run_cpu_pa_light(n_nodes, edges, seed=seed + 1, time_limit=budget)
        best = r1 if r1['best_cut'] >= r2['best_cut'] else r2
        return {
            'best_cut': best['best_cut'],
            'assignment': best.get('assignment', {}),
            'time_s': r1.get('time_s', 0.0) + r2.get('time_s', 0.0),
            'device': best.get('device', 'cpu'),
            'solver_note': (
                f"hub2000-pa-dual(best={best['best_cut']:.0f};"
                f"other={(r2 if best is r1 else r1)['best_cut']:.0f})"
            ),
            'history': best.get('history', []),
        }
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_dsbm_seeded_pa_light(n_nodes, edges, seed, time_limit, dsbm_budget):
    if time_limit is None or time_limit <= dsbm_budget:
        return _run_current(n_nodes, edges, seed, time_limit)
    r_d = run_dsbm_maxcut(
        n_nodes,
        edges,
        seed=seed,
        time_limit=dsbm_budget,
    )
    x0 = [0 if int(v) < 0 else 1 for _, v in sorted(r_d['assignment'].items())]
    n_replicas = 150 if n_nodes <= 2500 else 100
    r_p = pa_maxcut(
        n_nodes,
        edges,
        n_replicas=n_replicas,
        n_temps=50,
        n_sweeps=3,
        time_limit=max(0.5, time_limit - dsbm_budget),
        seed=seed,
        x_init=x0,
    )
    r_p['time_s'] = r_d.get('time_s', 0.0) + r_p.get('time_s', 0.0)
    r_p['solver_note'] = (
        f"dsbm-seeded-pa(dsbm={r_d['best_cut']:.0f},budget={dsbm_budget:g}s,"
        f"replicas={n_replicas})"
    )
    return r_p


def _run_hub2000_seeded_pa_1s(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and 10000 <= len(edges) <= 13000:
        return _run_dsbm_seeded_pa_light(n_nodes, edges, seed, time_limit, dsbm_budget=1.0)
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_hub2000_seeded_pa_15s(n_nodes, edges, seed, time_limit):
    if _is_dense2000(n_nodes, edges) and 10000 <= len(edges) <= 13000:
        return _run_dsbm_seeded_pa_light(n_nodes, edges, seed, time_limit, dsbm_budget=1.5)
    return _run_current(n_nodes, edges, seed, time_limit)


def _run_multiscale_bridge(n_nodes, edges, seed, time_limit):
    if n_nodes >= 1500:
        return multiscale_pa_maxcut(
            n_nodes,
            edges,
            seed=seed,
            time_limit=time_limit,
        )
    return _run_current(n_nodes, edges, seed, time_limit)


def _make_dispatcher_policy(enable_bandit=False, prefer_quantum=False,
                            gpu=False, bandit_scope='all'):
    """Build a persistent dispatcher session per seed for online portfolio tests."""
    dispatchers = {}

    def _run_dispatcher_policy(n_nodes, edges, seed, time_limit):
        dispatcher = dispatchers.get(seed)
        if dispatcher is None:
            dispatcher = ZornDispatcher(
                gpu=gpu,
                time_budget=time_limit,
                prefer_exact=True,
                prefer_quantum=prefer_quantum,
                seed=seed,
                verbose=False,
                enable_bandit=enable_bandit,
                bandit_scope=bandit_scope,
            )
            dispatchers[seed] = dispatcher
        result = dispatcher.solve(n_nodes, edges, time_budget=time_limit)
        return {
            'best_cut': result.best_cut,
            'assignment': result.assignment,
            'time_s': result.time_s,
            'device': 'cpu',
            'solver_note': (
                f"strategy={result.strategy};"
                f"solvers={'+'.join(result.solvers_used)};"
                f"notes={'|'.join(result.notes)}"
            ),
        }

    return _run_dispatcher_policy


POLICIES = {
    'current': _run_current,
    'multiscale_bridge': _run_multiscale_bridge,
    'highdeg_alt': _run_highdeg_alt,
    'highdeg_dual_mix_a': _run_highdeg_dual_mix_a,
    'highdeg_dual_mix_b': _run_highdeg_dual_mix_b,
    'dense2k_pa_only': _run_dense2k_pa_only,
    'dense2k_probe_1s': _run_dense2k_probe_1s,
    'dense2k_pa_heavy': _run_dense2k_pa_heavy,
    'combo_highdeg_dense2k': _run_combo_highdeg_dense2k,
    'signed2000_probe_1s': _run_signed2000_probe_1s,
    'signed2000_dsbm_auto': _run_signed2000_dsbm_auto,
    'signed2000_dsbm_tuned': _run_signed2000_dsbm_tuned,
    'highdeg_c0_25': _run_highdeg_c0_25,
    'highdeg_96_c0_20': _run_highdeg_96_c0_20,
    'highdeg_96_c0_25': _run_highdeg_96_c0_25,
    'hub2000_pa_heavy': _run_hub2000_pa_heavy,
    'hub2000_pa_wide': _run_hub2000_pa_wide,
    'hub2000_pa_dual': _run_hub2000_pa_dual,
    'hub2000_seeded_pa_1s': _run_hub2000_seeded_pa_1s,
    'hub2000_seeded_pa_15s': _run_hub2000_seeded_pa_15s,
}

STATEFUL_POLICY_FACTORIES = {
    'dispatcher_static': lambda: _make_dispatcher_policy(enable_bandit=False),
    'dispatcher_bandit': lambda: _make_dispatcher_policy(enable_bandit=True),
    'dispatcher_bandit_hubonly': (
        lambda: _make_dispatcher_policy(enable_bandit=True, bandit_scope='hub2000')
    ),
}


def _load_instances(family_names):
    instances = []
    for family in family_names:
        if family == 'full-gset':
            graph_names = _discover_gset_names()
        else:
            if family not in FAMILIES:
                raise ValueError(f'Unknown family: {family}')
            graph_names = FAMILIES[family]
        for name in graph_names:
            g, bks, _info = load_gset(name)
            instances.append((family, name, g.n_nodes, list(g.edges()), bks))
    return instances


def run_study(families, policies, seeds, time_limit, verbose=True):
    instances = _load_instances(families)
    policy_runners = {}
    for policy_name in policies:
        if policy_name in STATEFUL_POLICY_FACTORIES:
            policy_runners[policy_name] = STATEFUL_POLICY_FACTORIES[policy_name]()
        else:
            policy_runners[policy_name] = POLICIES[policy_name]
    rows = []
    total = len(instances) * len(policies) * len(seeds)
    done = 0

    for family, name, n_nodes, edges, bks in instances:
        for policy_name in policies:
            fn = policy_runners[policy_name]
            for seed in seeds:
                done += 1
                if verbose:
                    print(f'[{done}/{total}] {family} {name} {policy_name} seed={seed}')
                t0 = time.time()
                result = fn(n_nodes, edges, seed=seed, time_limit=time_limit)
                elapsed = time.time() - t0
                cut = float(result['best_cut'])
                gap_pct = 100.0 * (bks - cut) / bks
                rows.append({
                    'family': family,
                    'instance': name,
                    'policy': policy_name,
                    'seed': seed,
                    'n_nodes': n_nodes,
                    'n_edges': len(edges),
                    'bks': bks,
                    'cut': cut,
                    'gap_pct': gap_pct,
                    'time_s': float(result.get('time_s', elapsed)),
                    'device': result.get('device', 'cpu'),
                    'solver_note': result.get('solver_note', ''),
                })
    return rows


def summarize(rows):
    per_policy = {}
    per_family_policy = {}
    for row in rows:
        per_policy.setdefault(row['policy'], []).append(row['gap_pct'])
        per_family_policy.setdefault((row['family'], row['policy']), []).append(row['gap_pct'])

    summary = {
        'overall': {},
        'by_family': {},
    }
    for policy, vals in per_policy.items():
        summary['overall'][policy] = {
            'avg_gap_pct': sum(vals) / len(vals),
            'std_gap_pct': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            'num_runs': len(vals),
        }
    for (family, policy), vals in per_family_policy.items():
        fam = summary['by_family'].setdefault(family, {})
        fam[policy] = {
            'avg_gap_pct': sum(vals) / len(vals),
            'std_gap_pct': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            'num_runs': len(vals),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Seed-averaged family policy study')
    parser.add_argument('--families', type=str, default='highdeg-sparse,sparse-large')
    parser.add_argument('--policies', type=str,
                        default='current,highdeg_alt,highdeg_dual_mix_a,highdeg_dual_mix_b')
    parser.add_argument('--seeds', type=str, default='42,43,44')
    parser.add_argument('--time-limit', type=float, default=10.0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    families = [x.strip() for x in args.families.split(',') if x.strip()]
    policies = [x.strip() for x in args.policies.split(',') if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]

    for policy in policies:
        if policy not in POLICIES and policy not in STATEFUL_POLICY_FACTORIES:
            raise SystemExit(f'Unknown policy: {policy}')

    rows = run_study(families, policies, seeds, args.time_limit, verbose=True)
    summary = summarize(rows)

    print('\n=== Overall ===')
    for policy, info in sorted(summary['overall'].items(),
                               key=lambda kv: kv[1]['avg_gap_pct']):
        print(f"{policy}: avg_gap={info['avg_gap_pct']:.3f}% "
              f"std={info['std_gap_pct']:.3f} n={info['num_runs']}")

    print('\n=== By family ===')
    for family, info in summary['by_family'].items():
        print(f'[{family}]')
        for policy, slot in sorted(info.items(), key=lambda kv: kv[1]['avg_gap_pct']):
            print(f"  {policy}: avg_gap={slot['avg_gap_pct']:.3f}% "
                  f"std={slot['std_gap_pct']:.3f} n={slot['num_runs']}")

    if args.output:
        payload = {
            'metadata': {
                'families': families,
                'policies': policies,
                'seeds': seeds,
                'time_limit': args.time_limit,
            },
            'summary': summary,
            'rows': rows,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f'\nSaved report: {args.output}')


if __name__ == '__main__':
    main()
