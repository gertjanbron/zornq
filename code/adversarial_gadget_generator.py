#!/usr/bin/env python3
"""B144: small adversarial MaxCut gadget generator."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from maxcut_gadget_sat import solve_maxcut_gadget_exact


Edge = Tuple[int, int, float]


def _make_record(name: str, family: str, n_nodes: int, edges: Sequence[Edge],
                 note: str, **params) -> Dict[str, object]:
    return {
        'name': name,
        'family': family,
        'n_nodes': int(n_nodes),
        'edges': [(int(u), int(v), float(w)) for u, v, w in edges],
        'literature_note': note,
        'params': dict(params),
    }


def _offset_edges(edges: Sequence[Edge], offset: int) -> List[Edge]:
    return [(u + offset, v + offset, w) for u, v, w in edges]


def stitch_with_separator(left: Dict[str, object], right: Dict[str, object],
                          left_ports: Sequence[int], right_ports: Sequence[int],
                          name: str, family: str, note: str,
                          separator_edge_weight: float = -1.0,
                          bridge_weight: float = 1.0) -> Dict[str, object]:
    """
    Stitch two gadgets through a tiny separator interface.

    The separator nodes mediate interactions between two local hard pockets while
    keeping the whole gadget small enough for exact certification.
    """
    if len(left_ports) != len(right_ports):
        raise ValueError('left_ports and right_ports must have the same length')
    if len(left_ports) == 0:
        raise ValueError('need at least one interface port')

    left_n = int(left['n_nodes'])
    right_n = int(right['n_nodes'])
    right_offset = left_n
    sep_offset = left_n + right_n

    edges: List[Edge] = []
    edges.extend(_offset_edges(left['edges'], 0))
    edges.extend(_offset_edges(right['edges'], right_offset))

    separator_nodes = []
    for idx, (lp, rp) in enumerate(zip(left_ports, right_ports)):
        sep = sep_offset + idx
        separator_nodes.append(sep)
        edges.append((int(lp), sep, float(bridge_weight)))
        edges.append((sep, int(rp) + right_offset, float(bridge_weight)))

    for i in range(len(separator_nodes) - 1):
        edges.append((separator_nodes[i], separator_nodes[i + 1],
                      float(separator_edge_weight)))

    return _make_record(
        name=name,
        family=family,
        n_nodes=left_n + right_n + len(separator_nodes),
        edges=edges,
        note=note,
        left_family=left['family'],
        right_family=right['family'],
        left_ports=list(left_ports),
        right_ports=list(right_ports),
        separator_size=len(separator_nodes),
    )


def generate_twisted_ladder(rungs: int = 4) -> Dict[str, object]:
    """
    Open ladder with alternating diagonals.

    This is a small synthetic frustrated family: lots of overlapping odd cycles,
    but still tiny enough for exact certification.
    """
    if rungs < 3:
        raise ValueError('rungs must be >= 3')
    n = 2 * rungs
    top = list(range(rungs))
    bottom = list(range(rungs, 2 * rungs))
    edges: List[Edge] = []

    for i in range(rungs - 1):
        edges.append((top[i], top[i + 1], 1.0))
        edges.append((bottom[i], bottom[i + 1], 1.0))
    for i in range(rungs):
        edges.append((top[i], bottom[i], 1.0))
    for i in range(rungs - 1):
        edges.append((top[i], bottom[i + 1], 1.0))

    return _make_record(
        name=f'twisted_ladder_r{rungs}',
        family='twisted_ladder',
        n_nodes=n,
        edges=edges,
        note='Synthetic local gadget with overlapping odd-cycle motifs.',
        rungs=rungs,
    )


def generate_mobius_ladder(rungs: int = 5) -> Dict[str, object]:
    """
    Mobius ladder graph.

    Symmetric and highly frustrated once used as an unweighted MaxCut gadget.
    """
    if rungs < 3:
        raise ValueError('rungs must be >= 3')
    n = 2 * rungs
    edges: List[Edge] = []

    for i in range(n):
        edges.append((i, (i + 1) % n, 1.0))
    for i in range(rungs):
        edges.append((i, i + rungs, 1.0))

    return _make_record(
        name=f'mobius_ladder_r{rungs}',
        family='mobius_ladder',
        n_nodes=n,
        edges=edges,
        note='Symmetric non-planar ladder, useful as a tiny adversarial benchmark.',
        rungs=rungs,
    )


def generate_noise_cycle_cloud(layers: int = 5) -> Dict[str, object]:
    """
    Odd cycle of two-node clouds with alternating cross couplings.

    This is a modest, literature-inspired stand-in for odd-cycle blowup / noise
    motifs rather than a formal KKMO reduction.
    """
    if layers < 3 or layers % 2 == 0:
        raise ValueError('layers must be odd and >= 3')

    n = 2 * layers
    edges: List[Edge] = []

    def a(i: int) -> int:
        return 2 * i

    def b(i: int) -> int:
        return 2 * i + 1

    for i in range(layers):
        j = (i + 1) % layers
        edges.append((a(i), b(i), 1.0))
        edges.append((a(i), a(j), 1.0))
        edges.append((b(i), b(j), 1.0))
        if i % 2 == 0:
            edges.append((a(i), b(j), 1.0))
        else:
            edges.append((b(i), a(j), 1.0))

    return _make_record(
        name=f'noise_cycle_cloud_l{layers}',
        family='noise_cycle_cloud',
        n_nodes=n,
        edges=edges,
        note=('Synthetic odd-cycle blowup analogue; inspired by layered '
              'noise/dictatorship motifs, not a formal KKMO gadget.'),
        layers=layers,
    )


def default_adversarial_suite() -> List[Dict[str, object]]:
    """Small certified-first B144 suite."""
    return [
        generate_twisted_ladder(4),
        generate_mobius_ladder(5),
        generate_noise_cycle_cloud(3),
    ]


def generate_twisted_noise_separator() -> Dict[str, object]:
    left = generate_twisted_ladder(4)
    right = generate_noise_cycle_cloud(3)
    return stitch_with_separator(
        left,
        right,
        left_ports=[1, 6],
        right_ports=[0, 5],
        name='twisted_noise_separator',
        family='separator_stitch',
        note=('Separator-stitched gadget joining an odd-cycle ladder motif to a '
              'small noise-cycle cloud.'),
        separator_edge_weight=-1.0,
        bridge_weight=1.0,
    )


def generate_double_noise_separator() -> Dict[str, object]:
    left = generate_noise_cycle_cloud(3)
    right = generate_noise_cycle_cloud(3)
    return stitch_with_separator(
        left,
        right,
        left_ports=[0, 5],
        right_ports=[1, 4],
        name='double_noise_separator',
        family='separator_stitch',
        note=('Two cloud gadgets stitched through a frustrated separator '
              'interface to create a larger exact benchmark.'),
        separator_edge_weight=-1.0,
        bridge_weight=1.0,
    )


def extended_adversarial_suite() -> List[Dict[str, object]]:
    """Default base suite plus larger separator-stitched gadgets."""
    return default_adversarial_suite() + [
        generate_twisted_noise_separator(),
        generate_double_noise_separator(),
    ]


def certify_gadget(record: Dict[str, object]) -> Dict[str, object]:
    """Attach an exact small-gadget certificate."""
    exact = solve_maxcut_gadget_exact(record['n_nodes'], record['edges'])
    out = dict(record)
    out['exact_optimal_weight'] = exact['optimal_weight']
    out['exact_assignment'] = exact['node_assignment']
    out['certificate'] = exact['certificate']
    return out


if __name__ == '__main__':
    for gadget in default_adversarial_suite():
        certified = certify_gadget(gadget)
        print(certified['name'], certified['exact_optimal_weight'])
