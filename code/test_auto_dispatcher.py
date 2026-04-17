#!/usr/bin/env python3
"""test_auto_dispatcher.py - Tests for B130 Auto-Dispatcher"""

import numpy as np
import sys, os, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_dispatcher import (
    ZornDispatcher, solve_maxcut, classify_graph, select_strategy,
    certify_result, DispatchResult, _compute_tww_feature, _B170_AVAILABLE,
)
from bls_solver import random_3regular, random_erdos_renyi

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')

print('=== B130 Auto-Dispatcher Tests ===\n')

# ============================================================
# Graph classifier
# ============================================================
print('=== Graph classifier ===')

# K5
k5 = [(i,j,1.0) for i in range(5) for j in range(i+1,5)]
info = classify_graph(5, k5)
check('K5 n=5', info['n_nodes'] == 5)
check('K5 m=10', info['n_edges'] == 10)
check('K5 not bipartite', not info['is_bipartite'])
check('K5 not grid', not info['is_grid'])
check('K5 dense', info['density'] > 0.5)

# Grid 6x3
edges_grid = []
for x in range(6):
    for y in range(3):
        node = x*3+y
        if x+1<6: edges_grid.append((node,(x+1)*3+y,1.0))
        if y+1<3: edges_grid.append((node,x*3+y+1,1.0))
info = classify_graph(18, edges_grid)
check('Grid 6x3 is_grid', info['is_grid'])
check('Grid 6x3 dims', info['grid_dims'] == (6, 3))
check('Grid 6x3 bipartite', info['is_bipartite'])
check('Grid 6x3 possibly_planar', info['possibly_planar'])
check('Grid 6x3 has cycle_rank', info['cycle_rank'] >= 0)

# Petersen (non-bipartite, non-planar-ish)
pet = [(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0),(4,0,1.0),
       (5,7,1.0),(7,9,1.0),(9,6,1.0),(6,8,1.0),(8,5,1.0),
       (0,5,1.0),(1,6,1.0),(2,7,1.0),(3,8,1.0),(4,9,1.0)]
info = classify_graph(10, pet)
check('Petersen not bipartite', not info['is_bipartite'])
check('Petersen not grid', not info['is_grid'])
check('Petersen regular', info['is_regular'])
check('Petersen has degree_std', info['degree_std'] >= 0.0)

# ============================================================
# Strategy selection
# ============================================================
print('\n=== Strategy selection ===')

# Bipartite planar -> Pfaffian
info = classify_graph(18, edges_grid)
strat, tier, pipeline = select_strategy(info)
check('Grid -> pfaffian_exact', strat == 'pfaffian_exact')
check('Grid -> tier exact', tier == 'exact')

# Small graph -> exact
info = classify_graph(10, pet)
strat, tier, pipeline = select_strategy(info)
check('Small -> exact', tier == 'exact')

# Medium random -> quantum (lightcone) or classical
nn, edges = random_3regular(200, seed=1400)
info = classify_graph(nn, edges)
strat, tier, pipeline = select_strategy(info)
# Sparse 3-reg with low degree -> quantum lightcone (falls back to PA if unavailable)
check('3-reg n=200 -> quantum or classical', tier in ('quantum', 'classical'))
check('3-reg n=200 -> has pipeline', len(pipeline) > 0)

# Large -> classical combined
nn, edges = random_3regular(1000, seed=7000)
info = classify_graph(nn, edges)
strat, tier, pipeline = select_strategy(info)
check('3-reg n=1000 -> classical', tier == 'classical')
check('3-reg n=1000 -> combined', 'combined' in strat)

# ============================================================
# Certification
# ============================================================
print('\n=== Certification ===')

check('Exact cert', certify_result(6, 5, k5, {'is_bipartite': False}, True) == 'EXACT')
grid_info = classify_graph(18, edges_grid)
check('Bipartite exact cert', certify_result(len(edges_grid), 18, edges_grid, grid_info, False) == 'EXACT')
# gap = 0.2/27 = 0.74% < 1% -> NEAR_EXACT
# Non-bipartite: ratio 0.995 -> NEAR_EXACT
check('Near-exact cert', certify_result(9.95, 5, k5, {'is_bipartite': False}, False) == 'NEAR_EXACT')
check('Good cert', certify_result(len(edges_grid) * 0.93, 18, edges_grid, grid_info, False) == 'GOOD')
check('Approx cert', certify_result(len(edges_grid) * 0.5, 18, edges_grid, grid_info, False) == 'APPROXIMATE')

# ============================================================
# Dispatcher: exact instances
# ============================================================
print('\n=== Dispatcher: exact instances ===')

d = ZornDispatcher(verbose=False)

# K5
r = d.solve(5, k5)
check('K5 cut=6', abs(r.best_cut - 6) < 0.5)
check('K5 is_exact', r.is_exact)
check('K5 cert EXACT', r.certificate == 'EXACT')

# K8
k8 = [(i,j,1.0) for i in range(8) for j in range(i+1,8)]
r = d.solve(8, k8)
check('K8 cut=16', abs(r.best_cut - 16) < 0.5)

# Petersen
r = d.solve(10, pet)
check('Petersen cut=12', abs(r.best_cut - 12) < 0.5)
check('Petersen is_exact', r.is_exact)

# Grid 6x3 (bipartite -> Pfaffian -> exact)
r = d.solve(18, edges_grid)
check(f'Grid 6x3 cut={r.best_cut:.0f}=={len(edges_grid)}', abs(r.best_cut - len(edges_grid)) < 0.5)
check('Grid 6x3 is_exact', r.is_exact)
check('Grid 6x3 tier=exact', r.tier == 'exact')

# ============================================================
# Dispatcher: classical instances
# ============================================================
print('\n=== Dispatcher: classical instances ===')

# 3-regular n=100
nn, edges = random_3regular(100, seed=700)
r = d.solve(nn, edges, time_budget=5)
ratio = r.best_cut / len(edges)
check(f'3-reg n=100 ratio={ratio:.3f} > 0.85', ratio > 0.85)
check('3-reg n=100 tier', r.tier in ('quantum', 'classical'))
check('3-reg n=100 has assignment', len(r.assignment) == nn)

# 3-regular n=500
nn, edges = random_3regular(500, seed=3500)
r = d.solve(nn, edges, time_budget=5)
ratio = r.best_cut / len(edges)
check(f'3-reg n=500 ratio={ratio:.3f} > 0.85', ratio > 0.85)

# Dense ER
nn, edges = random_erdos_renyi(50, p=0.5, seed=150)
r = d.solve(nn, edges, time_budget=5)
check(f'ER n=50 cut={r.best_cut:.0f} > 300', r.best_cut > 300)

# ============================================================
# Dispatcher: large grid (bipartite exact via Pfaffian)
# ============================================================
print('\n=== Dispatcher: large bipartite ===')

edges_big = []
for x in range(30):
    for y in range(4):
        node = x*4+y
        if x+1<30: edges_big.append((node,(x+1)*4+y,1.0))
        if y+1<4: edges_big.append((node,x*4+y+1,1.0))
r = d.solve(120, edges_big, time_budget=10)
check(f'Grid 30x4 cut={r.best_cut:.0f}=={len(edges_big)}',
      abs(r.best_cut - len(edges_big)) < 0.5)
check('Grid 30x4 is_exact', r.is_exact)
check('Grid 30x4 tier=exact', r.tier == 'exact')

# ============================================================
# Convenience function
# ============================================================
print('\n=== solve_maxcut() convenience ===')

r = solve_maxcut(5, k5)
check('solve_maxcut K5 cut=6', abs(r.best_cut - 6) < 0.5)
check('solve_maxcut returns DispatchResult', isinstance(r, DispatchResult))

# ============================================================
# Result structure
# ============================================================
print('\n=== Result structure ===')

r = d.solve(10, pet)
check('Has best_cut', hasattr(r, 'best_cut'))
check('Has assignment', hasattr(r, 'assignment'))
check('Has ratio', r.ratio is not None)
check('Has strategy', len(r.strategy) > 0)
check('Has tier', r.tier in ('exact', 'quantum', 'classical'))
check('Has solvers_used', len(r.solvers_used) > 0)
check('Has time_s', r.time_s >= 0)
check('Has certificate', len(r.certificate) > 0)
check('Has graph_info', len(r.graph_info) > 0)
check('summary() works', len(r.summary()) > 0)

# ============================================================
# Time budget
# ============================================================
print('\n=== Time budget ===')

d_fast = ZornDispatcher(verbose=False, prefer_quantum=False)
nn, edges = random_3regular(200, seed=1400)
t_start = time.time()
r = d_fast.solve(nn, edges, time_budget=2.0)
elapsed = time.time() - t_start
check('Time budget respected (%.2fs < 3.0s)' % elapsed, elapsed < 3.0)

# ============================================================
# B151 Bandit planner
# ============================================================
print('\n=== B151 bandit planner ===')

d_bandit = ZornDispatcher(
    verbose=False,
    prefer_quantum=False,
    enable_bandit=True,
    seed=17,
)
nn, edges = random_3regular(600, seed=4200)
r1 = d_bandit.solve(nn, edges, time_budget=1.5)
r2 = d_bandit.solve(nn, edges, time_budget=1.5)
r3 = d_bandit.solve(nn, edges, time_budget=1.5)
check('Bandit solve returns assignment', len(r3.assignment) == nn)
check('Bandit family stored in graph_info', 'bandit_family' in r3.graph_info)
check('Bandit arm stored in graph_info', 'bandit_arm' in r3.graph_info)
check('Bandit notes present', any(n.startswith('bandit_arm=') for n in r3.notes))
check('Bandit memory available after repeated solves',
      d_bandit.bandit.get_warm_start(r3.graph_info['bandit_family']) is not None)
check('Bandit stats updated',
      len(d_bandit.bandit.get_stats(r3.graph_info['bandit_family'])) > 0)

d_bandit_hub = ZornDispatcher(
    verbose=False,
    prefer_quantum=False,
    enable_bandit=True,
    bandit_scope='hub2000',
    seed=17,
)
check('Hub-only scope blocks medium sparse family',
      not d_bandit_hub._bandit_scope_allows({'n_nodes': 600, 'n_edges': 900, 'is_grid': False}))
check('Hub-only scope allows hub2000-like family',
      d_bandit_hub._bandit_scope_allows({'n_nodes': 2000, 'n_edges': 11778, 'is_grid': False}))

# ========================================================

# ============================================================
# B170 twin-width / cograph integratie
# ============================================================
print('\n=== B170 twin-width / cograph dispatcher integratie ===')

# Classify heeft nu is_cograph, tww, is_unweighted velden
info_k5 = classify_graph(5, k5)
check('classify_graph returns is_unweighted', 'is_unweighted' in info_k5)
check('K5 is_unweighted', info_k5['is_unweighted'] is True)
check('classify_graph returns is_cograph', 'is_cograph' in info_k5)
check('classify_graph returns tww', 'tww' in info_k5)

if _B170_AVAILABLE:
    check('K5 detected as cograph', info_k5['is_cograph'] is True)
    check('K5 has tww computed (n<=32)', info_k5['tww'] is not None)
    check('K5 tww==0 (complete graph)', info_k5['tww'] == 0)

    info_pet = classify_graph(10, pet)
    check('Petersen not cograph', info_pet['is_cograph'] is False)
    check('Petersen has tww computed', info_pet['tww'] is not None)

    k30 = [(i, j, 1.0) for i in range(30) for j in range(i + 1, 30)]
    info_k30 = classify_graph(30, k30)
    check('K30 detected as cograph', info_k30['is_cograph'] is True)

    strat, tier, pipeline = select_strategy(info_k30)
    check('K30 -> cograph_dp strategy', strat == 'cograph_dp')
    check('K30 -> tier exact', tier == 'exact')
    check('K30 -> pipeline has cograph_dp solver',
          any(s == 'cograph_dp' for s, _ in pipeline))

    k5_weighted = [(i, j, 2.5) for i in range(5) for j in range(i + 1, 5)]
    info_k5w = classify_graph(5, k5_weighted)
    check('Weighted K5 not flagged as unweighted', info_k5w['is_unweighted'] is False)

    d_tww = ZornDispatcher(verbose=False)
    r_k30 = d_tww.solve(30, k30)
    check('K30 via dispatcher is_exact', r_k30.is_exact)
    check('K30 via dispatcher tier=exact', r_k30.tier == 'exact')
    check('K30 via dispatcher strategy=cograph_dp',
          'cograph_dp' in r_k30.strategy or 'cograph_dp' in r_k30.solvers_used)
    check(f'K30 via dispatcher cut={r_k30.best_cut:.0f}==225',
          abs(r_k30.best_cut - 225.0) < 0.5)

    feat = _compute_tww_feature(5, [(i, j) for i in range(5) for j in range(i + 1, 5)])
    check('_compute_tww_feature returns dict', isinstance(feat, dict))
    check('_compute_tww_feature has is_cograph', feat['is_cograph'] is True)
    check('_compute_tww_feature has tww', feat['tww'] == 0)

    bigger_cograph = [(i, j) for i in range(40) for j in range(i + 1, 40)]
    feat_big = _compute_tww_feature(40, bigger_cograph)
    check('tww None at n=40 (budget cap)', feat_big['tww'] is None)
    check('is_cograph computed at n=40', feat_big['is_cograph'] is True)

    feat_huge = _compute_tww_feature(200, [])
    check('both None at n=200', feat_huge['tww'] is None and feat_huge['is_cograph'] is None)
else:
    print('  SKIP: B170 not available')


# ============================================================
# Dag-8: signed-instance detection + downgrade
# ============================================================
# B131-Dag-8 vangt een bekende correctheids-bug op: pfaffian_maxcut's
# bipartite/grid short-circuits retourneren `sum(weights)` als cut-waarde,
# wat aantoonbaar fout is op signed instanties (negatieve edges kunnen
# zowel totale som als optimale cut omlaag trekken). De dispatcher
# implementeert vier defense-in-depth lagen:
#   1. has_signed_edges()   signaleert het geval;
#   2. select_strategy()    routet signed naar exact_small_signed of
#                           FW-SDP i.p.v. pfaffian_exact/exact_small;
#   3. _run_pfaffian/_run_brute_force raisen bij directe aanroep;
#   4. certify_result()     downgrade EXACT naar APPROXIMATE als een
#                           signed instance tóch via een pfaffian-route
#                           door het net glipt.
print('\n=== Dag-8 signed-instance detection + downgrade ===')

from auto_dispatcher import (
    has_signed_edges, _run_signed_brute_force, _run_pfaffian, _run_brute_force,
    SOLVER_FUNCS,
)

# has_signed_edges detectie
check('has_signed_edges: empty edges -> False',
      has_signed_edges([]) is False)
check('has_signed_edges: unit-weight (no w) -> False',
      has_signed_edges([(0, 1), (1, 2)]) is False)
check('has_signed_edges: positive weights -> False',
      has_signed_edges([(0, 1, 1.0), (1, 2, 2.5)]) is False)
check('has_signed_edges: zero weights -> False',
      has_signed_edges([(0, 1, 0.0)]) is False)
check('has_signed_edges: one negative -> True',
      has_signed_edges([(0, 1, 1.0), (1, 2, -1.0)]) is True)
check('has_signed_edges: all negative -> True',
      has_signed_edges([(0, 1, -1.0), (1, 2, -2.0)]) is True)

# classify_graph zet has_signed_edges op info
info_clean = classify_graph(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
info_signed = classify_graph(3, [(0, 1, 1.0), (1, 2, -1.0), (2, 0, 1.0)])
check('classify_graph: clean -> has_signed_edges=False',
      info_clean['has_signed_edges'] is False)
check('classify_graph: signed -> has_signed_edges=True',
      info_signed['has_signed_edges'] is True)

# select_strategy routet signed bipartite-grid NIET naar pfaffian_exact
grid_signed_edges = [
    (0, 1, 1.0), (1, 2, 1.0),
    (3, 4, 1.0), (4, 5, 1.0),
    (0, 3, 1.0), (1, 4, -1.0),
    (2, 5, 1.0),
]
info_gs = classify_graph(6, grid_signed_edges)
strategy_gs, tier_gs, _ = select_strategy(info_gs)
check(f'select_strategy: signed n=6 not pfaffian_exact (got {strategy_gs})',
      strategy_gs != 'pfaffian_exact')
check('select_strategy: signed small graph -> exact_small_signed',
      strategy_gs == 'exact_small_signed')

# Clean grid still routes to pfaffian_exact
grid_clean_edges = [(e[0], e[1], abs(e[2])) for e in grid_signed_edges]
info_gc = classify_graph(6, grid_clean_edges)
strategy_gc, _, _ = select_strategy(info_gc)
check(f'select_strategy: clean bipartite grid -> pfaffian_exact (got {strategy_gc})',
      strategy_gc == 'pfaffian_exact')

# signed_brute_force in SOLVER_FUNCS
check('signed_brute_force registered in SOLVER_FUNCS',
      'signed_brute_force' in SOLVER_FUNCS)

# _run_signed_brute_force correctheid
tri_signed = [(0, 1, 1.0), (1, 2, -1.0), (2, 0, 1.0)]
r_sbf = _run_signed_brute_force(3, tri_signed, {}, None, 42)
check('_run_signed_brute_force: triangle signed cut=2.0',
      abs(r_sbf['best_cut'] - 2.0) < 1e-9)
check('_run_signed_brute_force: is_exact=True', r_sbf['is_exact'] is True)
check('_run_signed_brute_force: solver name',
      r_sbf['solver'] == 'signed_brute_force')

# raise at n>24
try:
    _run_signed_brute_force(25, [], {}, None, 42)
    check('_run_signed_brute_force raises at n=25', False)
except ValueError:
    check('_run_signed_brute_force raises at n=25', True)

# _run_pfaffian raised op signed edges
try:
    _run_pfaffian(3, tri_signed, {}, None, 42)
    check('_run_pfaffian raises on signed edges', False)
except ValueError:
    check('_run_pfaffian raises on signed edges', True)

# _run_brute_force raised op signed edges
try:
    _run_brute_force(3, tri_signed, {}, None, 42)
    check('_run_brute_force raises on signed edges', False)
except ValueError:
    check('_run_brute_force raises on signed edges', True)

# certify_result downgrade op signed + pfaffian-route
cert_downgrade = certify_result(5.0, 10, [(0, 1, -1.0)],
                                {'has_signed_edges': True},
                                is_exact=True, strategy='pfaffian_exact')
check("certify_result: pfaffian_exact+signed -> APPROXIMATE",
      cert_downgrade == 'APPROXIMATE')
cert_downgrade2 = certify_result(5.0, 10, [(0, 1, -1.0)],
                                 {'has_signed_edges': True},
                                 is_exact=True, strategy='exact_small')
check("certify_result: exact_small+signed -> APPROXIMATE",
      cert_downgrade2 == 'APPROXIMATE')
cert_downgrade3 = certify_result(5.0, 10, [(0, 1, -1.0)],
                                 {'has_signed_edges': True},
                                 is_exact=True, strategy='exact_brute')
check("certify_result: exact_brute+signed -> APPROXIMATE",
      cert_downgrade3 == 'APPROXIMATE')

# Unsigned + pfaffian_exact stays EXACT (no downgrade)
cert_keep = certify_result(5.0, 10, [(0, 1, 1.0)],
                           {'has_signed_edges': False},
                           is_exact=True, strategy='pfaffian_exact')
check("certify_result: unsigned pfaffian_exact stays EXACT",
      cert_keep == 'EXACT')

# signed + safe route (exact_small_signed) -> EXACT
cert_safe = certify_result(5.0, 10, [(0, 1, -1.0)],
                           {'has_signed_edges': True},
                           is_exact=True, strategy='exact_small_signed')
check("certify_result: signed exact_small_signed keeps EXACT",
      cert_safe == 'EXACT')

# End-to-end: dispatcher on signed triangle
d_signed = ZornDispatcher(verbose=False)
r_e2e = d_signed.solve(3, tri_signed)
check(f'dispatcher signed e2e: cut=2.0 (got {r_e2e.best_cut})',
      abs(r_e2e.best_cut - 2.0) < 1e-9)
check(f'dispatcher signed e2e: strategy=exact_small_signed',
      r_e2e.strategy == 'exact_small_signed')
check(f'dispatcher signed e2e: cert=EXACT (got {r_e2e.certificate})',
      r_e2e.certificate == 'EXACT')


print('\n=== RESULTS: %d passed, %d failed ===' % (passed, failed))
sys.exit(1 if failed else 0)
