#!/usr/bin/env python3
"""Tests for B144 adversarial gadget generator."""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_gadget_generator import (
    certify_gadget,
    default_adversarial_suite,
    extended_adversarial_suite,
    generate_double_noise_separator,
    generate_mobius_ladder,
    generate_noise_cycle_cloud,
    generate_twisted_ladder,
    generate_twisted_noise_separator,
)
from bls_solver import bls_maxcut
from pa_solver import pa_maxcut


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


print('=== B144 Adversarial Gadget Generator Tests ===\n')

twisted = generate_twisted_ladder(4)
mobius = generate_mobius_ladder(5)
noise = generate_noise_cycle_cloud(3)
suite = default_adversarial_suite()
stitched_twisted = generate_twisted_noise_separator()
stitched_noise = generate_double_noise_separator()
extended = extended_adversarial_suite()

check('Twisted ladder has 8 nodes', twisted['n_nodes'] == 8)
check('Twisted ladder has 13 edges', len(twisted['edges']) == 13)
check('Mobius ladder has 10 nodes', mobius['n_nodes'] == 10)
check('Mobius ladder has 15 edges', len(mobius['edges']) == 15)
check('Noise cycle cloud has 6 nodes', noise['n_nodes'] == 6)
check('Noise cycle cloud has 12 edges', len(noise['edges']) == 12)
check('Default suite has 3 gadgets', len(suite) == 3)
check('Default suite names are unique', len({g['name'] for g in suite}) == len(suite))
check('Twisted+noise separator has 16 nodes', stitched_twisted['n_nodes'] == 16)
check('Twisted+noise separator has 30 edges', len(stitched_twisted['edges']) == 30)
check('Double noise separator has 14 nodes', stitched_noise['n_nodes'] == 14)
check('Double noise separator has 29 edges', len(stitched_noise['edges']) == 29)
check('Extended suite has 5 gadgets', len(extended) == 5)
check('Extended suite names are unique', len({g['name'] for g in extended}) == len(extended))

for gadget in extended:
    print(f"\n=== {gadget['name']} ===")
    certified = certify_gadget(gadget)
    check(f"{gadget['name']} exact certificate", certified['certificate'] == 'EXACT_GADGET')
    check(f"{gadget['name']} exact optimum exists", certified['exact_optimal_weight'] is not None)

    bls = bls_maxcut(
        gadget['n_nodes'], gadget['edges'],
        n_restarts=4, max_iter=150, max_no_improve=20,
        time_limit=0.5, seed=42, verbose=False,
    )
    pa = pa_maxcut(
        gadget['n_nodes'], gadget['edges'],
        n_replicas=32, n_temps=16, n_sweeps=2,
        beta_min=0.1, beta_max=3.0,
        time_limit=0.5, seed=42, verbose=False,
    )
    exact = float(certified['exact_optimal_weight'])
    check(f"{gadget['name']} BLS <= exact", float(bls['best_cut']) <= exact + 1e-9)
    check(f"{gadget['name']} PA <= exact", float(pa['best_cut']) <= exact + 1e-9)

print('\n=== RESULTS: %d passed, %d failed ===' % (passed, failed))
sys.exit(1 if failed else 0)
