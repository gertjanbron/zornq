#!/usr/bin/env python3
"""
B32 Experiment Runner: Tropische Tensor Netwerken (MAP via min/+ Algebra)

Experimenten:
  1. Correctheid: tropische eliminatie vs brute-force op diverse grafen
  2. Sandwich bound: QAOA cost vs tropische MaxCut
  3. 2D grids: tropische contractie op Lx x Ly grids
  4. Treewidth en eliminatievolgorde: impact op efficiëntie
  5. Gewogen en Ising grafen
  6. Schaalbaarheid: grote 1D ketens en grids
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tropical_tensor import (
    TropicalTensor, tropical_contract, tropical_multiply,
    build_maxcut_tropical_network, solve_maxcut_tropical,
    solve_maxcut_tropical_elim, min_degree_order,
    tropical_transfer_matrix_1d,
    _tropical_brute_force,
    qaoa_tropical_map, qaoa_expected_cost,
    build_2d_grid_edges, solve_maxcut_2d_tropical,
    sandwich_bound,
    random_weighted_graph, ising_weighted_graph,
)


def experiment_1_correctheid():
    """Exp 1: Tropische eliminatie vs brute-force."""
    print("=" * 70)
    print("EXPERIMENT 1: Correctheid tropische eliminatie vs brute-force")
    print("=" * 70)

    test_cases = [
        ("Pad-4", 4, [(i, i+1) for i in range(3)]),
        ("Pad-8", 8, [(i, i+1) for i in range(7)]),
        ("Driehoek", 3, [(0,1),(1,2),(0,2)]),
        ("K4", 4, [(i,j) for i in range(4) for j in range(i+1,4)]),
        ("K5", 5, [(i,j) for i in range(5) for j in range(i+1,5)]),
        ("Cycle-6", 6, [(i,(i+1)%6) for i in range(6)]),
        ("Petersen-5", 5, [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(1,3),(2,4),(3,0),(4,1)]),
        ("Grid-2x3", 6, build_2d_grid_edges(2, 3)),
        ("Grid-3x3", 9, build_2d_grid_edges(3, 3)),
        ("Grid-4x4", 16, build_2d_grid_edges(4, 4)),
    ]

    print(f"\n{'Graaf':>15s} {'n':>3s} {'|E|':>4s} {'BF':>6s} {'Elim':>6s} "
          f"{'Match':>6s} {'tw':>3s} {'T(BF)':>8s} {'T(Elim)':>8s}")
    print("-" * 70)

    for name, n, edges in test_cases:
        # Brute force
        t0 = time.time()
        if n <= 20:
            cut_bf, _ = _tropical_brute_force(n, edges, {})
            t_bf = time.time() - t0
        else:
            cut_bf = None
            t_bf = 0

        # Eliminatie
        t0 = time.time()
        order, tw = min_degree_order(n, edges)
        cut_elim, config = solve_maxcut_tropical_elim(n, edges, elim_order=order)
        t_elim = time.time() - t0

        match = "OK" if cut_bf is not None and abs(cut_elim - cut_bf) < 1e-10 else "N/A"
        bf_str = f"{cut_bf:.0f}" if cut_bf is not None else "skip"

        print(f"{name:>15s} {n:>3d} {len(edges):>4d} {bf_str:>6s} {cut_elim:>6.0f} "
              f"{match:>6s} {tw:>3d} {t_bf:>7.4f}s {t_elim:>7.4f}s")


def experiment_2_sandwich():
    """Exp 2: Sandwich bound QAOA vs tropische MaxCut."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sandwich bound — QAOA cost vs C_max")
    print("=" * 70)

    configs = [
        ("Pad-6", 6, [(i, i+1) for i in range(5)]),
        ("Pad-8", 8, [(i, i+1) for i in range(7)]),
        ("Cycle-6", 6, [(i,(i+1)%6) for i in range(6)]),
        ("Grid-2x2", 4, build_2d_grid_edges(2, 2)),
        ("Grid-2x3", 6, build_2d_grid_edges(2, 3)),
        ("Grid-3x3", 9, build_2d_grid_edges(3, 3)),
        ("K4", 4, [(i,j) for i in range(4) for j in range(i+1,4)]),
    ]

    print(f"\n{'Graaf':>10s} {'p':>2s} {'QAOA':>8s} {'MAP':>8s} "
          f"{'C_max':>6s} {'QAOA/C':>7s} {'MAP/C':>7s}")
    print("-" * 55)

    for name, n, edges in configs:
        for p in [1, 2]:
            rng = np.random.default_rng(42)
            gammas = [0.3 + 0.1 * rng.standard_normal() for _ in range(p)]
            betas = [0.7 + 0.1 * rng.standard_normal() for _ in range(p)]

            result = sandwich_bound(n, edges, gammas, betas)

            print(f"{name:>10s} {p:>2d} {result['qaoa_cost']:>8.3f} "
                  f"{result['map_cost']:>8.1f} {result['tropical_max']:>6.0f} "
                  f"{result['qaoa_ratio']:>7.3f} {result['map_ratio']:>7.3f}")


def experiment_3_2d_grids():
    """Exp 3: Tropische contractie op 2D grids."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: 2D Grid MaxCut via tropische eliminatie")
    print("=" * 70)

    print(f"\n{'Grid':>8s} {'n':>4s} {'|E|':>4s} {'C_max':>6s} "
          f"{'tw':>3s} {'Time':>8s} {'Configuratie':>20s}")
    print("-" * 60)

    for Lx, Ly in [(2,2), (2,3), (3,3), (3,4), (4,4), (4,5), (5,5), (6,6)]:
        n = Lx * Ly
        t0 = time.time()
        try:
            cut, config, tw = solve_maxcut_2d_tropical(Lx, Ly)
            dt = time.time() - t0
            config_str = ''.join(str(c) for c in config[:20])
            if n > 20:
                config_str += "..."
            print(f"{Lx}x{Ly:>2d} {n:>4d} {len(build_2d_grid_edges(Lx,Ly)):>4d} "
                  f"{cut:>6.0f} {tw:>3d} {dt:>7.3f}s {config_str:>20s}")
        except Exception as e:
            dt = time.time() - t0
            print(f"{Lx}x{Ly:>2d} {n:>4d}  --- FOUT: {e} ({dt:.3f}s)")

    # Checkerboard verificatie
    print("\nCheckerboard verificatie (open BC):")
    for Lx, Ly in [(2,2), (3,3), (4,4), (5,5)]:
        n = Lx * Ly
        edges = build_2d_grid_edges(Lx, Ly)
        # Checkerboard MaxCut voor open grid = |E| als bipartiet
        # Grid is bipartiet, dus MaxCut = |E|
        cut, _, _ = solve_maxcut_2d_tropical(Lx, Ly)
        print(f"  {Lx}x{Ly}: C_max={cut:.0f}, |E|={len(edges)}, "
              f"ratio={cut/len(edges):.3f}")


def experiment_4_treewidth():
    """Exp 4: Treewidth en eliminatievolgorde."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Treewidth en eliminatievolgorde impact")
    print("=" * 70)

    grafen = [
        ("Pad-20", 20, [(i, i+1) for i in range(19)]),
        ("Cycle-20", 20, [(i,(i+1)%20) for i in range(20)]),
        ("Grid-4x4", 16, build_2d_grid_edges(4, 4)),
        ("Grid-5x5", 25, build_2d_grid_edges(5, 5)),
        ("Grid-4x6", 24, build_2d_grid_edges(4, 6)),
    ]

    print(f"\n{'Graaf':>12s} {'n':>4s} {'|E|':>4s} {'tw(MD)':>7s} "
          f"{'C_max':>6s} {'T(MD)':>8s} {'T(Nat)':>8s}")
    print("-" * 60)

    for name, n, edges in grafen:
        # Min-degree order
        t0 = time.time()
        order_md, tw_md = min_degree_order(n, edges)
        cut_md, _ = solve_maxcut_tropical_elim(n, edges, elim_order=order_md)
        t_md = time.time() - t0

        # Natuurlijke volgorde (0,1,...,n-1)
        t0 = time.time()
        cut_nat, _ = solve_maxcut_tropical_elim(n, edges, elim_order=list(range(n)))
        t_nat = time.time() - t0

        print(f"{name:>12s} {n:>4d} {len(edges):>4d} {tw_md:>7d} "
              f"{cut_md:>6.0f} {t_md:>7.3f}s {t_nat:>7.3f}s")


def experiment_5_gewogen():
    """Exp 5: Gewogen en Ising grafen."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Gewogen en Ising grafen")
    print("=" * 70)

    n = 10
    edges = [(i, i+1) for i in range(n-1)] + [(0, n-1)]  # Cycle

    rng = np.random.default_rng(42)

    # Uniform gewichten
    print(f"\nCycle n={n}, uniform gewichten w=1:")
    cut, config = solve_maxcut_tropical_elim(n, edges)
    print(f"  C_max = {cut:.0f} (verwacht: {n} als n even, {n-1} als n oneven)")

    # Random gewichten
    print(f"\nRandom gewichten [0.5, 2.0]:")
    for seed in range(3):
        rng_w = np.random.default_rng(seed)
        weights = random_weighted_graph(n, edges, rng_w, (0.5, 2.0))
        cut, config = solve_maxcut_tropical_elim(n, edges, weights)
        total_w = sum(weights.values())
        print(f"  Seed {seed}: C_max = {cut:.2f}, total_w = {total_w:.2f}, "
              f"ratio = {cut/total_w:.3f}")

    # Ising (+/-1)
    print(f"\nIsing gewichten +/-1:")
    for seed in range(5):
        rng_w = np.random.default_rng(seed)
        weights = ising_weighted_graph(n, edges, rng_w)
        cut, config = solve_maxcut_tropical_elim(n, edges, weights)
        n_pos = sum(1 for w in weights.values() if w > 0)
        n_neg = sum(1 for w in weights.values() if w < 0)
        print(f"  Seed {seed}: C_max = {cut:.1f}, +1 edges: {n_pos}, "
              f"-1 edges: {n_neg}")


def experiment_6_schaalbaarheid():
    """Exp 6: Schaalbaarheid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Schaalbaarheid tropische contractie")
    print("=" * 70)

    # 1D ketens
    print(f"\n1D Ketens (pad):")
    print(f"  {'n':>6s} {'C_max':>6s} {'tw':>3s} {'Time':>8s}")
    print("  " + "-" * 30)
    for n in [10, 50, 100, 500, 1000, 5000]:
        edges = [(i, i+1) for i in range(n-1)]
        t0 = time.time()
        cut, config = solve_maxcut_tropical_elim(n, edges)
        dt = time.time() - t0
        print(f"  {n:>6d} {cut:>6.0f} {1:>3d} {dt:>7.3f}s")

    # 2D grids
    print(f"\n2D Grids:")
    print(f"  {'Grid':>8s} {'n':>5s} {'C_max':>6s} {'tw':>4s} {'Time':>8s}")
    print("  " + "-" * 40)
    for Lx, Ly in [(3,3), (4,4), (5,5), (6,6), (8,8), (10,10)]:
        n = Lx * Ly
        edges = build_2d_grid_edges(Lx, Ly)
        t0 = time.time()
        try:
            order, tw = min_degree_order(n, edges)
            cut, config = solve_maxcut_tropical_elim(n, edges, elim_order=order)
            dt = time.time() - t0
            print(f"  {Lx}x{Ly:>2d} {n:>5d} {cut:>6.0f} {tw:>4d} {dt:>7.3f}s")
        except MemoryError:
            dt = time.time() - t0
            print(f"  {Lx}x{Ly:>2d} {n:>5d}   --- MEMORY ({dt:.3f}s)")
        except Exception as e:
            dt = time.time() - t0
            print(f"  {Lx}x{Ly:>2d} {n:>5d}   --- FOUT: {str(e)[:30]} ({dt:.3f}s)")

    # Transfer matrix vergelijking (1D)
    print(f"\n1D Transfer matrix vs eliminatie:")
    print(f"  {'n':>6s} {'T(TM)':>8s} {'T(Elim)':>8s} {'Match':>6s}")
    print("  " + "-" * 35)
    for n in [10, 100, 1000]:
        edges = [(i, i+1) for i in range(n-1)]
        t0 = time.time()
        cut_tm, _ = tropical_transfer_matrix_1d(n, edges)
        t_tm = time.time() - t0
        t0 = time.time()
        cut_el, _ = solve_maxcut_tropical_elim(n, edges)
        t_el = time.time() - t0
        match = "OK" if abs(cut_tm - cut_el) < 1e-10 else "FAIL"
        print(f"  {n:>6d} {t_tm:>7.4f}s {t_el:>7.4f}s {match:>6s}")


if __name__ == '__main__':
    print("=" * 70)
    print("B32: Tropische Tensor Netwerken — Experiment Suite")
    print("=" * 70)

    experiment_1_correctheid()
    experiment_2_sandwich()
    experiment_3_2d_grids()
    experiment_4_treewidth()
    experiment_5_gewogen()
    experiment_6_schaalbaarheid()

    print("\n" + "=" * 70)
    print("ALLE EXPERIMENTEN VOLTOOID")
    print("=" * 70)
