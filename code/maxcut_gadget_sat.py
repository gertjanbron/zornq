#!/usr/bin/env python3
"""
maxcut_gadget_sat.py - B148 small CNF/SAT layer for MaxCut gadgets.

Focuses on small local subgraphs / witness patches:
- encode signed MaxCut/Ising satisfaction as CNF
- verify threshold claims on tiny gadgets
- exact small-gadget solve via repeated SAT calls
- export DIMACS for external SAT solvers if desired
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Edge = Tuple[int, int, float]


@dataclass
class CNFEncoding:
    num_vars: int
    clauses: List[List[int]]
    node_vars: Dict[int, int]
    sat_vars: List[int]
    normalized_edges: List[Edge]
    threshold: int
    total_weight: int


class VarPool:
    def __init__(self) -> None:
        self._next_var = 1

    def new_var(self) -> int:
        out = self._next_var
        self._next_var += 1
        return out

    @property
    def num_vars(self) -> int:
        return self._next_var - 1


def normalize_gadget_edges(edges: Sequence[Sequence[float]]) -> List[Edge]:
    """Normalize to 0-indexed integer-friendly gadget edges."""
    normalized: List[Edge] = []
    for raw in edges:
        if len(raw) < 2:
            continue
        u = int(raw[0])
        v = int(raw[1])
        if u == v:
            continue
        w = float(raw[2]) if len(raw) > 2 else 1.0
        normalized.append((u, v, w))
    return normalized


def extract_gadget_subgraph(n_nodes: int, edges: Sequence[Sequence[float]],
                            nodes: Sequence[int]) -> Dict[str, object]:
    """Extract and relabel an induced gadget subgraph."""
    ordered = sorted({int(v) for v in nodes if 0 <= int(v) < n_nodes})
    forward = {old: new for new, old in enumerate(ordered)}
    backward = {new: old for old, new in forward.items()}
    sub_edges: List[Edge] = []
    for raw in edges:
        if len(raw) < 2:
            continue
        u = int(raw[0])
        v = int(raw[1])
        if u in forward and v in forward:
            w = float(raw[2]) if len(raw) > 2 else 1.0
            sub_edges.append((forward[u], forward[v], w))
    return {
        'n_nodes': len(ordered),
        'edges': sub_edges,
        'forward_map': forward,
        'backward_map': backward,
    }


def _edge_sat_xor_clauses(a: int, b: int, s: int) -> List[List[int]]:
    """Clauses for s <-> (a XOR b)."""
    return [
        [a, b, -s],
        [a, -b, s],
        [-a, b, s],
        [-a, -b, -s],
    ]


def _edge_sat_xnor_clauses(a: int, b: int, s: int) -> List[List[int]]:
    """Clauses for s <-> (a XNOR b)."""
    return [
        [a, b, s],
        [a, -b, -s],
        [-a, b, -s],
        [-a, -b, s],
    ]


def _add_at_least_k(clauses: List[List[int]], lits: Sequence[int], k: int) -> None:
    """
    Naive CNF encoding of AtLeast(k, lits).

    This is intentionally combinatorial but perfectly fine for small gadgets.
    """
    n = len(lits)
    if k <= 0:
        return
    if k > n:
        clauses.append([])
        return
    subset_size = n - k + 1
    for subset in combinations(lits, subset_size):
        clauses.append(list(subset))


def encode_maxcut_threshold_cnf(n_nodes: int, edges: Sequence[Sequence[float]],
                                min_satisfied_weight: int,
                                fixed_assignment: Optional[Dict[int, int]] = None
                                ) -> CNFEncoding:
    """
    Encode a small signed MaxCut/Ising gadget threshold as CNF.

    Positive weights want opposite sides, negative weights want same side.
    Absolute integer edge weights are treated as multiplicities in the threshold.
    """
    fixed_assignment = fixed_assignment or {}
    normalized = normalize_gadget_edges(edges)
    pool = VarPool()
    clauses: List[List[int]] = []
    node_vars = {i: pool.new_var() for i in range(int(n_nodes))}
    sat_vars: List[int] = []
    weighted_sat_lits: List[int] = []
    total_weight = 0

    for u, v, w in normalized:
        weight = abs(int(round(w)))
        if weight <= 0:
            continue
        sat_var = pool.new_var()
        sat_vars.append(sat_var)
        if w >= 0:
            clauses.extend(_edge_sat_xor_clauses(node_vars[u], node_vars[v], sat_var))
        else:
            clauses.extend(_edge_sat_xnor_clauses(node_vars[u], node_vars[v], sat_var))
        weighted_sat_lits.extend([sat_var] * weight)
        total_weight += weight

    for node, bit in fixed_assignment.items():
        node = int(node)
        bit = int(bit)
        if node not in node_vars:
            raise ValueError(f'fixed_assignment references unknown node {node}')
        clauses.append([node_vars[node] if bit else -node_vars[node]])

    _add_at_least_k(clauses, weighted_sat_lits, int(min_satisfied_weight))
    return CNFEncoding(
        num_vars=pool.num_vars,
        clauses=clauses,
        node_vars=node_vars,
        sat_vars=sat_vars,
        normalized_edges=normalized,
        threshold=int(min_satisfied_weight),
        total_weight=total_weight,
    )


def evaluate_signed_cut(n_nodes: int, edges: Sequence[Sequence[float]],
                        assignment: Dict[int, int]) -> int:
    """Return signed satisfaction weight for a 0/1 assignment."""
    total = 0
    for u, v, w in normalize_gadget_edges(edges):
        bit_u = int(assignment.get(int(u), 0))
        bit_v = int(assignment.get(int(v), 0))
        sat = (bit_u != bit_v) if w >= 0 else (bit_u == bit_v)
        if sat:
            total += abs(int(round(w)))
    return total


def _simplify_clauses(clauses: Sequence[Sequence[int]],
                      assignment: Dict[int, bool]) -> Optional[List[List[int]]]:
    simplified: List[List[int]] = []
    for clause in clauses:
        keep: List[int] = []
        satisfied = False
        for lit in clause:
            var = abs(int(lit))
            val = assignment.get(var)
            if val is None:
                keep.append(int(lit))
                continue
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if satisfied:
            continue
        if not keep:
            return None
        simplified.append(keep)
    return simplified


def _unit_propagate(clauses: Sequence[Sequence[int]],
                    assignment: Dict[int, bool]
                    ) -> Tuple[Optional[List[List[int]]], Dict[int, bool]]:
    current = [list(c) for c in clauses]
    current_assignment = dict(assignment)
    while True:
        unit = None
        for clause in current:
            if len(clause) == 1:
                unit = clause[0]
                break
        if unit is None:
            return current, current_assignment
        var = abs(unit)
        val = unit > 0
        prev = current_assignment.get(var)
        if prev is not None and prev != val:
            return None, current_assignment
        current_assignment[var] = val
        current = _simplify_clauses(current, current_assignment)
        if current is None:
            return None, current_assignment


def solve_cnf_dpll(num_vars: int, clauses: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Tiny DPLL solver for small gadget CNFs."""
    simplified, assignment = _unit_propagate(clauses, {})
    if simplified is None:
        return {'sat': False, 'assignment': {}}

    def choose_var(local_clauses: Sequence[Sequence[int]],
                   local_assignment: Dict[int, bool]) -> Optional[int]:
        for clause in local_clauses:
            for lit in clause:
                var = abs(int(lit))
                if var not in local_assignment:
                    return var
        return None

    def dfs(local_clauses: Sequence[Sequence[int]],
            local_assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        if not local_clauses:
            return dict(local_assignment)
        var = choose_var(local_clauses, local_assignment)
        if var is None:
            return dict(local_assignment)
        for val in (True, False):
            trial = dict(local_assignment)
            trial[var] = val
            next_clauses = _simplify_clauses(local_clauses, trial)
            if next_clauses is None:
                continue
            next_clauses, trial = _unit_propagate(next_clauses, trial)
            if next_clauses is None:
                continue
            out = dfs(next_clauses, trial)
            if out is not None:
                return out
        return None

    model = dfs(simplified, assignment)
    if model is None:
        return {'sat': False, 'assignment': {}}
    full_assignment = {var: bool(model.get(var, False)) for var in range(1, num_vars + 1)}
    return {'sat': True, 'assignment': full_assignment}


def decode_node_assignment(node_vars: Dict[int, int],
                           sat_assignment: Dict[int, bool]) -> Dict[int, int]:
    return {
        node: 1 if bool(sat_assignment.get(var, False)) else 0
        for node, var in node_vars.items()
    }


def verify_gadget_threshold(n_nodes: int, edges: Sequence[Sequence[float]],
                            min_satisfied_weight: int,
                            fixed_assignment: Optional[Dict[int, int]] = None
                            ) -> Dict[str, object]:
    """Solve one threshold claim for a tiny gadget."""
    encoding = encode_maxcut_threshold_cnf(
        n_nodes,
        edges,
        min_satisfied_weight=min_satisfied_weight,
        fixed_assignment=fixed_assignment,
    )
    sat_result = solve_cnf_dpll(encoding.num_vars, encoding.clauses)
    node_assignment = (
        decode_node_assignment(encoding.node_vars, sat_result['assignment'])
        if sat_result['sat'] else {}
    )
    achieved = (
        evaluate_signed_cut(n_nodes, encoding.normalized_edges, node_assignment)
        if sat_result['sat'] else None
    )
    return {
        'sat': bool(sat_result['sat']),
        'threshold': int(min_satisfied_weight),
        'total_weight': encoding.total_weight,
        'node_assignment': node_assignment,
        'achieved_weight': achieved,
        'cnf_variables': encoding.num_vars,
        'cnf_clauses': len(encoding.clauses),
    }


def solve_maxcut_gadget_exact(n_nodes: int, edges: Sequence[Sequence[float]],
                              fixed_assignment: Optional[Dict[int, int]] = None
                              ) -> Dict[str, object]:
    """Exact tiny-gadget solve via descending SAT thresholds."""
    normalized = normalize_gadget_edges(edges)
    total_weight = sum(abs(int(round(w))) for _, _, w in normalized)
    for threshold in range(total_weight, -1, -1):
        result = verify_gadget_threshold(
            n_nodes,
            normalized,
            min_satisfied_weight=threshold,
            fixed_assignment=fixed_assignment,
        )
        if result['sat']:
            result['optimal_weight'] = threshold
            result['certificate'] = 'EXACT_GADGET'
            return result
    return {
        'sat': False,
        'optimal_weight': None,
        'certificate': 'UNSAT_GADGET',
        'node_assignment': {},
        'achieved_weight': None,
    }


def write_dimacs(filepath: str, encoding: CNFEncoding) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"p cnf {encoding.num_vars} {len(encoding.clauses)}\n")
        for clause in encoding.clauses:
            f.write(' '.join(str(lit) for lit in clause) + ' 0\n')


if __name__ == '__main__':
    # Tiny self-demo
    tri = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    out = solve_maxcut_gadget_exact(3, tri)
    print(out)
