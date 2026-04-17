#!/usr/bin/env python3
"""
B158: Triangle + Odd-Cycle Cutting Planes voor MaxCut.

Klassieke LP-relaxatie van het cut-polytoop CUT(G):
  Variabelen: y_e ∈ [0,1] voor elke edge (cut-indicator)
  Triangle-facetten: voor elke driehoek {i,j,k}:
      y_ij + y_jk + y_ik ≤ 2          (niet alle drie cut)
      y_ij - y_jk - y_ik ≤ 0          (drie permutaties van het pariteits-facet)
  Odd-cycle-facetten: voor elke oneven cykel C en oneven F ⊆ C:
      Σ_{e ∈ F} y_e − Σ_{e ∈ C \\ F} y_e ≤ |F| − 1

Voor sparse grafen breiden we eventueel uit naar K_n (gewicht 0 op niet-edges)
zodat we alle driehoeken meenemen — dit geeft een sterk strakkere LP. Voor
zeer dense grafen is dat al automatisch.

Odd-cycle separatie gebeurt iteratief via de "signed-graph" truc:
  Bouw een hulpgraaf met vertex-paren (v, A) en (v, B). Voor elke edge (u,v)
  voeg toe:
    (u,A)-(v,B) met gewicht y_uv      ("edge in F", verwacht parity-flip)
    (u,A)-(v,A) met gewicht 1-y_uv    ("edge buiten F", geen flip)
  en de symmetrische kopie. Een korste pad (v,A) → (v,B) komt overeen met
  de meest geschonden odd-cycle inequality. Als zijn lengte < 1, dan is er
  een violation van 1 - lengte.

Vergelijking:
  - B60 GW (SDP-bound)
  - B156 SoS-2 (Lasserre level-2)
  - B158 LP_triangle, LP_triangle+odd-cycle

Voor dense grafen verwacht je:
  LP_triangle_K_n ≥ GW ≥ SoS-2 ≥ OPT
en LP+odd-cycle laat ruimte zien tot LP_triangle dicht. Voor sparse grafen
kan LP+cuts veel goedkoper de gap krijgen dan SoS-2 wat O(n^4) variabelen kost.

Gebruik:
  python b158_cutting_planes.py --n 6
  python b158_cutting_planes.py --petersen
  python b158_cutting_planes.py --compare --cycle 7
  python b158_cutting_planes.py --extend --random 12
"""

from __future__ import annotations

import argparse
import heapq
import time
from itertools import combinations
from typing import Any

import numpy as np
import scipy.optimize as opt

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    gw_sdp_bound,
    random_3regular,
    random_erdos_renyi,
)


# ============================================================
# Hulp-functies
# ============================================================

def _edge_index(graph: SimpleGraph) -> dict[tuple[int, int], int]:
    """Map (u, v) met u<v naar een variabele-index 0..m-1."""
    idx = {}
    for k, (u, v, _w) in enumerate(graph.edges):
        a, b = (u, v) if u < v else (v, u)
        idx[(a, b)] = k
    return idx


def _edge_weight(graph: SimpleGraph) -> np.ndarray:
    return np.array([w for _u, _v, w in graph.edges], dtype=np.float64)


def _extend_to_complete(graph: SimpleGraph) -> tuple[SimpleGraph, dict[tuple[int, int], int]]:
    """Maak een K_n-extensie waarin niet-edges gewicht 0 hebben.

    Hierdoor kunnen we triangle-inequalities over álle triples van V toepassen.
    """
    n = graph.n
    g_ext = SimpleGraph(n)
    # Originele edges
    for u, v, w in graph.edges:
        a, b = (u, v) if u < v else (v, u)
        g_ext.add_edge(a, b, w)
    # Voeg dummy-edges toe met gewicht 0 voor ontbrekende paren
    have = {tuple(sorted((u, v))) for u, v, _ in graph.edges}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in have:
                g_ext.add_edge(i, j, 0.0)
    return g_ext, _edge_index(g_ext)


# ============================================================
# Triangle-LP (alle driehoeken in graaf)
# ============================================================

def _triangle_constraints(
    graph: SimpleGraph,
    edge_idx: dict[tuple[int, int], int],
) -> list[tuple[np.ndarray, float]]:
    """Genereer triangle-inequalities (4 per driehoek).

    Een 'driehoek' vereist dat álle drie edges (i,j), (j,k), (i,k) bestaan
    in `graph`. Voor sparse grafen geeft dit weinig constraints; gebruik
    `_extend_to_complete` voor de volledige K_n-aanpak.
    """
    n = graph.n
    m = graph.n_edges
    have = {(u, v) if u < v else (v, u): k for k, (u, v, _) in enumerate(graph.edges)}
    constraints: list[tuple[np.ndarray, float]] = []

    for i, j, k in combinations(range(n), 3):
        ij = (i, j)
        jk = (j, k)
        ik = (i, k)
        if ij in have and jk in have and ik in have:
            a = have[ij]
            b = have[jk]
            c = have[ik]
            # 4 facetten per driehoek
            for signs, rhs in [
                ((+1, +1, +1), 2.0),
                ((+1, -1, -1), 0.0),
                ((-1, +1, -1), 0.0),
                ((-1, -1, +1), 0.0),
            ]:
                row = np.zeros(m)
                row[a] = signs[0]
                row[b] = signs[1]
                row[c] = signs[2]
                constraints.append((row, rhs))
    return constraints


def lp_triangle_bound(
    graph: SimpleGraph,
    extend: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """LP-relaxatie van het cut-polytoop met triangle-inequalities.

    Parameters
    ----------
    graph   : invoer-graaf
    extend  : indien True, breid uit naar K_n (gewicht 0 op niet-edges)
              en voeg ALLE C(n,3) triangles toe — sterk strakker maar O(n^3).
    verbose : print samenvatting
    """
    n = graph.n
    t0 = time.time()
    if extend:
        work_g, edge_idx = _extend_to_complete(graph)
    else:
        work_g, edge_idx = graph, _edge_index(graph)

    m_total = work_g.n_edges
    weights = _edge_weight(work_g)

    # max c·y  ≡  min -c·y
    c_obj = -weights

    constraints = _triangle_constraints(work_g, edge_idx)
    if not constraints:
        # Geen driehoeken: bound = Σ w_e (triviaal, alle edges in cut)
        bound = float(weights.sum())
        return {
            "lp_bound": bound, "n": n, "m": graph.n_edges,
            "n_triangles": 0, "n_oddcycles_added": 0,
            "solve_time": time.time() - t0,
            "status": "trivial",
        }

    A_ub = np.array([row for row, _ in constraints])
    b_ub = np.array([rhs for _, rhs in constraints])

    bounds = [(0.0, 1.0)] * m_total
    res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"presolve": True})

    solve_time = time.time() - t0
    if not res.success:
        return {
            "lp_bound": None, "status": res.message,
            "solve_time": solve_time, "n": n, "m": graph.n_edges,
        }

    bound = float(-res.fun)
    return {
        "lp_bound": bound,
        "lp_ratio": bound / graph.n_edges if graph.n_edges > 0 else 0.0,
        "n": n,
        "m": graph.n_edges,
        "m_extended": m_total,
        "n_triangles": len(constraints) // 4,
        "n_oddcycles_added": 0,
        "y_solution": res.x,
        "solve_time": solve_time,
        "status": "optimal",
    }


# ============================================================
# Odd-cycle separatie via signed-graph
# ============================================================

def _shortest_signed_path(
    n: int,
    edge_list: list[tuple[int, int, float]],  # (u, v, y_uv)
    start: int,
) -> tuple[float, list[tuple[int, int, str]]] | None:
    """Bereken kortste pad van (start, A) naar (start, B) in de signed-graph.

    Knopen zijn (vertex, side) met side ∈ {0,1}. Voor elke edge (u,v,y):
        (u,0) -- (v,0)  weight y      (geen flip, edge buiten F: violatie y)
        (u,1) -- (v,1)  weight y
        (u,0) -- (v,1)  weight 1-y    (flip, edge in F: violatie 1-y)
        (u,1) -- (v,0)  weight 1-y

    De parity van het aantal "flip"-edges in een gesloten wandeling moet
    oneven zijn om van side 0 naar side 1 te komen — wat overeenkomt met een
    oneven F ⊆ C in een cykel. Totale lengte = Σ_{e∈F}(1-y) + Σ_{e∉F}y; bij
    lengte < 1 hebben we een violatie van de odd-cycle inequality.

    Retourneert (lengte, edges-met-label) of None.
    """
    # Pre-bouw adjacency
    adj: list[list[tuple[int, int, float, str]]] = [[] for _ in range(2 * n)]
    for u, v, y in edge_list:
        a0, a1 = u, u + n
        b0, b1 = v, v + n
        # Geen flip (edge buiten F): violatie = y
        w_out = max(y, 0.0)
        adj[a0].append((b0, u, w_out, "out"))
        adj[b0].append((a0, u, w_out, "out"))
        adj[a1].append((b1, u, w_out, "out"))
        adj[b1].append((a1, u, w_out, "out"))
        # Flip (edge in F): violatie = 1-y
        w_in = max(1.0 - y, 0.0)
        adj[a0].append((b1, u, w_in, "in"))
        adj[b1].append((a0, u, w_in, "in"))
        adj[a1].append((b0, u, w_in, "in"))
        adj[b0].append((a1, u, w_in, "in"))

    src = start
    dst = start + n

    # Dijkstra
    dist = [float("inf")] * (2 * n)
    prev: list[tuple[int, int, str] | None] = [None] * (2 * n)
    dist[src] = 0.0
    pq: list[tuple[float, int]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        if u == dst:
            break
        for vnode, _eu, w_e, label in adj[u]:
            nd = d + w_e
            if nd < dist[vnode] - 1e-12:
                dist[vnode] = nd
                prev[vnode] = (u, _eu, label)
                heapq.heappush(pq, (nd, vnode))

    if dist[dst] == float("inf"):
        return None

    # Reconstrueer pad
    path: list[tuple[int, int, str]] = []
    cur = dst
    while prev[cur] is not None:
        u, _eu, label = prev[cur]  # type: ignore[misc]
        a, b = (cur % n, u % n)
        path.append((min(a, b), max(a, b), label))
        cur = u
    path.reverse()
    return dist[dst], path


def _separate_odd_cycle(
    graph: SimpleGraph,
    y: np.ndarray,
    edge_idx: dict[tuple[int, int], int],
    tol: float = 1e-4,
    max_cuts: int = 20,
) -> list[tuple[np.ndarray, float]]:
    """Vind tot `max_cuts` violated odd-cycle inequalities.

    Iteratie: voor elke vertex v, vind kortste signed pad v_A → v_B; lengte
    < 1 ⟹ odd cycle met violatie 1-lengte.
    """
    n = graph.n
    edge_list = []
    for (u, v), k in edge_idx.items():
        edge_list.append((u, v, float(y[k])))

    cuts: list[tuple[np.ndarray, float]] = []
    seen_cycles: set[frozenset] = set()

    for v_start in range(n):
        if len(cuts) >= max_cuts:
            break
        result = _shortest_signed_path(n, edge_list, v_start)
        if result is None:
            continue
        length, path = result
        if length >= 1.0 - tol:
            continue

        # Bouw odd-cycle constraint: verzamel uniek edge-set
        edges_in_F: list[int] = []
        edges_out_F: list[int] = []
        all_edge_ids = []
        for a, b, label in path:
            key = (min(a, b), max(a, b))
            if key not in edge_idx:
                continue
            ei = edge_idx[key]
            all_edge_ids.append(ei)
            if label == "in":
                edges_in_F.append(ei)
            else:
                edges_out_F.append(ei)

        if not edges_in_F:
            continue
        # |F| moet oneven zijn
        if len(edges_in_F) % 2 == 0:
            continue

        cycle_signature = frozenset(all_edge_ids)
        if cycle_signature in seen_cycles or len(cycle_signature) < 3:
            continue
        seen_cycles.add(cycle_signature)

        # Inequality: Σ_{e ∈ F} y_e − Σ_{e ∈ C\F} y_e ≤ |F| − 1
        m = len(y)
        row = np.zeros(m)
        for ei in edges_in_F:
            row[ei] += 1.0
        for ei in edges_out_F:
            row[ei] += -1.0
        rhs = float(len(edges_in_F) - 1)
        cuts.append((row, rhs))

    return cuts


def lp_triangle_oddcycle_bound(
    graph: SimpleGraph,
    extend: bool = True,
    max_iters: int = 30,
    max_cuts_per_iter: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """LP triangle-bound + iteratieve odd-cycle separatie."""
    n = graph.n
    t0 = time.time()

    if extend:
        work_g, edge_idx = _extend_to_complete(graph)
    else:
        work_g, edge_idx = graph, _edge_index(graph)
    m_total = work_g.n_edges
    weights = _edge_weight(work_g)

    c_obj = -weights
    constraints = _triangle_constraints(work_g, edge_idx)
    A_rows: list[np.ndarray] = [r for r, _ in constraints]
    b_vals: list[float] = [v for _, v in constraints]

    n_iter = 0
    n_cuts_added = 0
    bound_history: list[float] = []

    while n_iter < max_iters:
        n_iter += 1
        if A_rows:
            A_ub = np.array(A_rows)
            b_ub = np.array(b_vals)
        else:
            A_ub = None
            b_ub = None
        bounds = [(0.0, 1.0)] * m_total
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs", options={"presolve": True})
        if not res.success:
            return {
                "lp_bound": None, "status": res.message,
                "solve_time": time.time() - t0,
                "n": n, "m": graph.n_edges,
                "n_iters": n_iter, "n_cuts_added": n_cuts_added,
            }
        bound = float(-res.fun)
        bound_history.append(bound)

        # Probeer odd-cycles te separeren
        new_cuts = _separate_odd_cycle(work_g, res.x, edge_idx,
                                        max_cuts=max_cuts_per_iter)
        if not new_cuts:
            break
        for r, v in new_cuts:
            A_rows.append(r)
            b_vals.append(v)
        n_cuts_added += len(new_cuts)

    solve_time = time.time() - t0
    return {
        "lp_bound": bound,
        "lp_ratio": bound / graph.n_edges if graph.n_edges > 0 else 0.0,
        "n": n, "m": graph.n_edges,
        "m_extended": m_total,
        "n_triangles": len(constraints) // 4,
        "n_iters": n_iter,
        "n_cuts_added": n_cuts_added,
        "bound_history": bound_history,
        "y_solution": res.x,
        "solve_time": solve_time,
        "status": "optimal",
    }


# ============================================================
# Vergelijking GW vs SoS-2 vs LP_triangle vs LP+odd-cycle
# ============================================================

def compare_all_bounds(
    graph: SimpleGraph,
    name: str = "graph",
    verbose: bool = True,
    include_sos2: bool = True,
) -> dict[str, Any]:
    opt_val = brute_force_maxcut(graph) if graph.n <= 18 else None

    t0 = time.time()
    gw = gw_sdp_bound(graph, verbose=False).get("sdp_bound")
    gw_t = time.time() - t0

    res_lp = lp_triangle_bound(graph, extend=True, verbose=False)
    lp_b = res_lp.get("lp_bound")
    lp_t = res_lp.get("solve_time")

    res_lpoc = lp_triangle_oddcycle_bound(graph, extend=True, verbose=False)
    lpoc_b = res_lpoc.get("lp_bound")
    lpoc_t = res_lpoc.get("solve_time")
    lpoc_cuts = res_lpoc.get("n_cuts_added", 0)

    sos2_b = None
    sos2_t = None
    if include_sos2 and graph.n <= 14:
        try:
            from b156_sos2_sdp import sos2_sdp_bound
            res_sos2 = sos2_sdp_bound(graph, verbose=False)
            sos2_b = res_sos2.get("sos2_bound")
            sos2_t = res_sos2.get("solve_time")
        except Exception as e:
            sos2_b = None
            sos2_t = f"FAIL: {e}"

    result = {
        "name": name, "n": graph.n, "n_edges": graph.n_edges,
        "opt": opt_val,
        "gw": gw, "gw_time": gw_t,
        "lp_triangle": lp_b, "lp_triangle_time": lp_t,
        "lp_oddcycle": lpoc_b, "lp_oddcycle_time": lpoc_t,
        "lp_oddcycle_cuts_added": lpoc_cuts,
        "sos2": sos2_b, "sos2_time": sos2_t,
    }

    if verbose:
        print(f"\n=== {name} (n={graph.n}, m={graph.n_edges}) ===")
        if opt_val is not None:
            print(f"  OPT (brute):     {opt_val}")
        print(f"  GW (SDP):        {gw:.4f}  ({gw_t:.2f}s)")
        print(f"  LP triangle:     {lp_b:.4f}  ({lp_t:.2f}s)")
        print(f"  LP+odd-cycle:    {lpoc_b:.4f}  ({lpoc_t:.2f}s, "
              f"{lpoc_cuts} cuts)")
        if sos2_b is not None:
            print(f"  SoS-2:           {sos2_b:.4f}  ({sos2_t:.2f}s)")

    return result


# ============================================================
# CLI
# ============================================================

def _build_graph(args: argparse.Namespace) -> tuple[SimpleGraph, str]:
    if args.n is not None:
        from b156_sos2_sdp import complete_graph
        return complete_graph(args.n), f"K_{args.n}"
    if args.cycle is not None:
        from b156_sos2_sdp import cycle_graph
        return cycle_graph(args.cycle), f"C_{args.cycle}"
    if args.petersen:
        from b156_sos2_sdp import petersen_graph
        return petersen_graph(), "Petersen"
    if args.bipartite is not None:
        from b156_sos2_sdp import complete_bipartite
        a, b = args.bipartite
        return complete_bipartite(a, b), f"K_{a},{b}"
    if args.random is not None:
        return random_3regular(args.random, seed=args.seed), f"3reg_n{args.random}"
    if args.erdos is not None:
        return random_erdos_renyi(args.erdos, p=args.p, seed=args.seed), \
               f"ER_n{args.erdos}_p{args.p}"
    from b156_sos2_sdp import petersen_graph
    return petersen_graph(), "Petersen"


def main() -> None:
    parser = argparse.ArgumentParser(description="B158 Triangle+Odd-Cycle LP")
    parser.add_argument("--n", type=int, help="Complete graph K_n")
    parser.add_argument("--cycle", type=int, help="Cycle C_n")
    parser.add_argument("--petersen", action="store_true")
    parser.add_argument("--bipartite", type=int, nargs=2, metavar=("A", "B"))
    parser.add_argument("--random", type=int, help="random 3-regulier n")
    parser.add_argument("--erdos", type=int, help="Erdős-Rényi n")
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extend", action="store_true",
                        help="Vul aan tot K_n voor alle triangle-constraints")
    parser.add_argument("--compare", action="store_true",
                        help="Vergelijk GW, LP_triangle, LP+OC, SoS-2")
    args = parser.parse_args()
    graph, name = _build_graph(args)

    if args.compare:
        compare_all_bounds(graph, name=name)
    else:
        print(f"=== {name}, n={graph.n}, m={graph.n_edges} ===")
        r1 = lp_triangle_bound(graph, extend=args.extend)
        b1 = r1.get("lp_bound")
        n_tri = r1.get("n_triangles")
        t1 = r1.get("solve_time")
        print("  LP triangle bound:  {:.4f}  (triangles={}, tijd={:.2f}s)".format(
            b1, n_tri, t1))
        r2 = lp_triangle_oddcycle_bound(graph, extend=args.extend)
        b2 = r2.get("lp_bound")
        n_it = r2.get("n_iters")
        n_cu = r2.get("n_cuts_added")
        t2 = r2.get("solve_time")
        print("  LP+odd-cycle bound: {:.4f}  (iters={}, cuts={}, tijd={:.2f}s)".format(
            b2, n_it, n_cu, t2))


if __name__ == "__main__":
    main()
