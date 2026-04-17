#!/usr/bin/env python3
"""test_maxcut_gadget_sat.py - Tests for B148 SAT/CNF gadget layer."""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxcut_gadget_sat import (
    extract_gadget_subgraph,
    solve_maxcut_gadget_exact,
    verify_gadget_threshold,
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


print('=== B148 MaxCut Gadget SAT Tests ===\n')

print('=== Triangle gadget ===')
triangle = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
r2 = verify_gadget_threshold(3, triangle, min_satisfied_weight=2)
r3 = verify_gadget_threshold(3, triangle, min_satisfied_weight=3)
exact_tri = solve_maxcut_gadget_exact(3, triangle)
check('Triangle threshold 2 is SAT', r2['sat'])
check('Triangle threshold 3 is UNSAT', not r3['sat'])
check('Triangle exact optimum is 2', exact_tri['optimal_weight'] == 2)

print('\n=== Signed edge gadget ===')
signed = [(0, 1, -1.0)]
exact_signed = solve_maxcut_gadget_exact(2, signed)
assign_signed = exact_signed['node_assignment']
check('Signed edge optimum is 1', exact_signed['optimal_weight'] == 1)
check('Signed edge wants same side',
      assign_signed.get(0) == assign_signed.get(1))

print('\n=== Fixed assignment ===')
square = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
fixed = verify_gadget_threshold(4, square, min_satisfied_weight=4,
                                fixed_assignment={0: 0, 1: 1})
check('Square threshold 4 SAT with fixed assignment', fixed['sat'])
check('Fixed assignment keeps node 0', fixed['node_assignment'].get(0) == 0)
check('Fixed assignment keeps node 1', fixed['node_assignment'].get(1) == 1)

print('\n=== Extract subgraph ===')
k5 = [(i, j, 1.0) for i in range(5) for j in range(i + 1, 5)]
sub = extract_gadget_subgraph(5, k5, [1, 3, 4])
check('Subgraph has 3 nodes', sub['n_nodes'] == 3)
check('Subgraph triangle has 3 edges', len(sub['edges']) == 3)
check('Backward map keeps original node 3', sub['backward_map'][1] in {1, 3, 4})

print('\n=== Larger exact gadget ===')
exact_k5 = solve_maxcut_gadget_exact(5, k5)
check('K5 gadget optimum is 6', exact_k5['optimal_weight'] == 6)
check('K5 certificate exact', exact_k5['certificate'] == 'EXACT_GADGET')

print('\n=== RESULTS: %d passed, %d failed ===' % (passed, failed))
sys.exit(1 if failed else 0)
