#!/usr/bin/env python3
"""test_bandit_planner.py - Tests for B151 bandit planner."""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bandit_planner import (
    DispatcherBandit,
    family_key_from_info,
    reward_from_dispatch_result,
)


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


class DummyResult:
    def __init__(self, best_cut, ratio, time_s, certificate='GOOD',
                 assignment=None, strategy='dummy'):
        self.best_cut = best_cut
        self.ratio = ratio
        self.time_s = time_s
        self.certificate = certificate
        self.assignment = assignment or {}
        self.strategy = strategy


print('=== B151 Bandit Planner Tests ===\n')

print('=== Family key ===')
info = {
    'n_nodes': 1200,
    'density': 0.03,
    'avg_degree': 5.0,
    'max_degree': 7,
    'cycle_rank': 80,
    'leaf_fraction': 0.02,
    'hub_fraction': 0.00,
    'is_regular': True,
    'is_bipartite': False,
    'is_grid': False,
}
family = family_key_from_info(info)
check('Family key has medium bucket', 'medium' in family)
check('Family key has sparse bucket', 'sparse' in family)
check('Family key has regular flag', 'regular' in family)

print('\n=== UCB choose/update ===')
bandit = DispatcherBandit(seed=7, eps=0.0, ucb_c=0.5)
arms = ['a', 'b']
first = bandit.choose('fam', arms)
bandit.update('fam', first, 0.2)
second = bandit.choose('fam', arms)
bandit.update('fam', second, 0.9)
third = bandit.choose('fam', arms)
check('Bandit explores both unseen arms', {first, second} == set(arms))
check('Bandit prefers higher reward arm after updates', third == second)

print('\n=== Warm-start memory ===')
result1 = DummyResult(
    best_cut=10.0, ratio=0.80, time_s=1.0,
    assignment={0: 1, 1: 0}, strategy='arm_a',
)
result2 = DummyResult(
    best_cut=9.0, ratio=0.75, time_s=0.8,
    assignment={0: 0, 1: 1}, strategy='arm_b',
)
stored1 = bandit.remember_result('fam', result1)
stored2 = bandit.remember_result('fam', result2)
warm = bandit.get_warm_start('fam')
check('First result stored in warm memory', stored1)
check('Worse result does not replace warm memory', not stored2)
check('Warm memory keeps best cut', abs(warm['best_cut'] - 10.0) < 1e-9)
check('Warm memory keeps assignment', warm['assignment'] == {0: 1, 1: 0})

print('\n=== Reward shaping ===')
r_good = DummyResult(best_cut=20.0, ratio=0.90, time_s=2.0, certificate='GOOD')
r_exact = DummyResult(best_cut=20.0, ratio=0.90, time_s=2.0, certificate='EXACT')
check('Exact gets higher reward than good',
      reward_from_dispatch_result(r_exact) > reward_from_dispatch_result(r_good))

print('\n=== RESULTS: %d passed, %d failed ===' % (passed, failed))
sys.exit(1 if failed else 0)
