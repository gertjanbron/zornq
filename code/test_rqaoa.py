#!/usr/bin/env python3
"""Gerichte regressietests voor B47 RQAOA."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_planner import ZornSolver
from rqaoa import GeneralQAOA, RQAOA, WeightedGraph


def _count_cut(edges, bits):
    return sum(1 for u, v in edges if bits[u] != bits[v])


def _cycle_graph(n):
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
        g.add_edge(i, (i + 1) % n)
    return g


def test_general_qaoa_supports_p2():
    """p=2 optimalisatie moet geldige hoekvectoren teruggeven."""
    g = _cycle_graph(6)
    qaoa = GeneralQAOA(g, verbose=False)
    ratio, gammas, betas, info = qaoa.optimize(
        p=2, n_gamma=4, n_beta=4, refine=False)

    assert 0.0 <= ratio <= 1.0
    assert len(gammas) == 2
    assert len(betas) == 2
    assert info['n_evals'] >= 17


def test_rqaoa_solve_auto_cycle24():
    """Auto-modus moet voor sparse >22 nodes de lightcone-route pakken."""
    g = _cycle_graph(24)
    rqaoa = RQAOA(g, p=1, verbose=False)
    result = rqaoa.solve(mode='auto', n_gamma=4, n_beta=4)

    assert result.info['mode'] == 'fast_lightcone'
    assert len(result.assignment) == 24
    assert len(result.bitstring) == 24
    assert abs(result.cut_value - 24.0) < 1e-9
    assert abs(result.ratio - 1.0) < 1e-9


def test_rqaoa_reorder_metadata():
    """B141 reorder-modus moet doorlopen tot in de ordering-metadata."""
    g = _cycle_graph(24)
    rqaoa = RQAOA(g, p=1, verbose=False)

    natural = rqaoa.solve(mode='fast', reorder='none', n_gamma=4, n_beta=4)
    fiedler = rqaoa.solve(mode='fast', reorder='fiedler', n_gamma=4, n_beta=4)

    assert natural.info['requested_reorder'] == 'none'
    assert natural.info['resolved_ordering'] == 'natural'
    assert fiedler.info['requested_reorder'] == 'fiedler'
    assert fiedler.info['resolved_ordering'] == 'fiedler'
    assert abs(natural.cut_value - 24.0) < 1e-9
    assert abs(fiedler.cut_value - 24.0) < 1e-9


def test_zornsolver_forced_rqaoa_cycle24():
    """ZornSolver moet de nieuwe RQAOA-entrypoint kunnen gebruiken."""
    g = _cycle_graph(24)
    edges = [(i, (i + 1) % 24) for i in range(24)]

    solver = ZornSolver(chi_budget=64, gpu=False,
                        reorder='fiedler', verbose=False)
    result = solver.solve(n_nodes=24, edges=edges, p=1,
                          method='rqaoa', reorder='fiedler')

    actual_cut = _count_cut(edges, result.best_bitstring)
    assert result.method == 'rqaoa'
    assert result.engine == 'RQAOA'
    assert any(note == 'reorder:fiedler' for note in result.notes)
    assert actual_cut == 24
    assert abs(result.cut_value - 24.0) < 1e-9


if __name__ == '__main__':
    test_general_qaoa_supports_p2()
    test_rqaoa_solve_auto_cycle24()
    test_rqaoa_reorder_metadata()
    test_zornsolver_forced_rqaoa_cycle24()
    print("=== Alle B47-tests geslaagd ===")
