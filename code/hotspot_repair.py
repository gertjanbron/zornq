#!/usr/bin/env python3
"""
hotspot_repair.py - B70: Frustration-Patch Solver

Strategy:
  1. Tier 1: fast global solve via TransverseQAOA
  2. Identify hotspot edges with small absolute ZZ correlation
  3. Tier 2: repair hotspots with either:
     - tiny exact gadget checks on local patches
     - lightcone fallback for the remaining edges
  4. Combine cold edges from Tier 1 with repaired hotspot estimates
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class HotspotRepair:
    """Two-tier solver: cheap global solve plus expensive local patch repair."""

    def __init__(self, Lx, Ly, p_global=1, p_local=2,
                 frustration_threshold=0.3, warm=False, verbose=True,
                 exact_gadget_nodes_max=8, exact_patch_radius=1,
                 exact_gadget_mode='boundary'):
        """
        Args:
            Lx, Ly: grid dimensions
            p_global: QAOA depth for the global solve
            p_local: QAOA depth for local lightcone repair
            frustration_threshold: edges with |ZZ| below this are hotspots
            warm: use WS-QAOA warm-start
            verbose: print details
            exact_gadget_nodes_max: max nodes for a tiny exact SAT gadget patch
            exact_patch_radius: BFS radius around a hotspot edge for gadget patches
            exact_gadget_mode: 'off', 'free', or 'boundary'
        """
        self.Lx = Lx
        self.Ly = Ly
        self.p_global = p_global
        self.p_local = p_local
        self.threshold = frustration_threshold
        self.warm = warm
        self.verbose = verbose
        self.exact_gadget_nodes_max = max(0, int(exact_gadget_nodes_max))
        self.exact_patch_radius = max(0, int(exact_patch_radius))
        self.exact_gadget_mode = str(exact_gadget_mode).strip().lower()
        if self.exact_gadget_mode not in {'off', 'free', 'boundary'}:
            raise ValueError(
                "exact_gadget_mode must be one of: 'off', 'free', 'boundary'"
            )

        # Edge list in the same ordering as the transverse solver.
        self.edges_v = []
        self.edges_h = []
        for x in range(Lx):
            for y in range(Ly - 1):
                self.edges_v.append((x, y))
        for x in range(Lx - 1):
            for y in range(Ly):
                self.edges_h.append((x, y))
        self.n_edges = len(self.edges_v) + len(self.edges_h)

        # Small primal graph representation for gadget patches.
        self.n_nodes = self.Lx * self.Ly
        self.graph_edges = []
        self.graph_adjacency = {node: set() for node in range(self.n_nodes)}
        for x in range(self.Lx):
            for y in range(self.Ly):
                u = self._node_id(x, y)
                if x + 1 < self.Lx:
                    v = self._node_id(x + 1, y)
                    self.graph_edges.append((u, v, 1.0))
                    self.graph_adjacency[u].add(v)
                    self.graph_adjacency[v].add(u)
                if y + 1 < self.Ly:
                    v = self._node_id(x, y + 1)
                    self.graph_edges.append((u, v, 1.0))
                    self.graph_adjacency[u].add(v)
                    self.graph_adjacency[v].add(u)

    def _node_id(self, x, y):
        return x * self.Ly + y

    def _edge_nodes(self, etype, ex, ey):
        if etype == 'v':
            return self._node_id(ex, ey), self._node_id(ex, ey + 1)
        if etype == 'h':
            return self._node_id(ex, ey), self._node_id(ex + 1, ey)
        raise ValueError(f"Unknown edge type: {etype}")

    def _collect_patch_nodes(self, etype, ex, ey, radius):
        """Collect a tiny BFS patch around the hotspot edge."""
        start_nodes = self._edge_nodes(etype, ex, ey)
        seen = {node: 0 for node in start_nodes}
        queue = deque((node, 0) for node in start_nodes)

        while queue:
            node, dist = queue.popleft()
            if dist >= radius:
                continue
            for nb in self.graph_adjacency[node]:
                if nb in seen:
                    continue
                seen[nb] = dist + 1
                queue.append((nb, dist + 1))

        return sorted(seen)

    def _tier1_global(self):
        """Tier 1: fast global solve via TransverseQAOA."""
        from transverse_contraction import TransverseQAOA

        warm_angles = None
        if self.warm:
            from ws_qaoa import sdp_warm_start
            warm_angles = sdp_warm_start(
                self.Lx, self.Ly, epsilon=0.2,
                mode='binary', verbose=self.verbose)

        tc = TransverseQAOA(self.Lx, self.Ly, verbose=False)
        ratio, gammas, betas, info = tc.optimize(
            self.p_global, n_gamma=12, n_beta=12, refine=True,
            warm_angles=warm_angles)

        if self.verbose:
            ws = " (warm)" if self.warm else ""
            print(f"  [B70] Tier 1: QAOA-{self.p_global}{ws} ratio={ratio:.6f} "
                  f"({info['total_time']:.2f}s)")

        zz_per_edge = self._compute_per_edge_zz(tc, gammas, betas, warm_angles)
        return ratio, zz_per_edge, gammas, betas, warm_angles

    def _compute_per_edge_zz(self, tc, gammas, betas, warm_angles=None):
        """Compute ZZ per edge via TransverseQAOA."""
        Lx, Ly, d = tc.Lx, tc.Ly, tc.d
        p = len(gammas)

        if warm_angles is not None:
            from ws_qaoa import warm_start_mps
            mps = warm_start_mps(Lx, Ly, warm_angles)
        else:
            mps = [np.ones((1, d, 1), dtype=complex) / np.sqrt(d)
                   for _ in range(Lx)]

        chi_exact = d ** p
        for layer in range(p):
            g, b = gammas[layer], betas[layer]
            intra = tc._zz_intra_diag(g)
            for x in range(Lx):
                mps[x] = mps[x] * intra[None, :, None]
            inter = tc._zz_inter_diag(g)
            for x in range(Lx - 1):
                mps = tc._apply_2site_exact(mps, x, inter, chi_exact)
            rx = tc._rx_col(b)
            for x in range(Lx):
                mps[x] = np.einsum('ij,ajb->aib', rx, mps[x])

        env_L, env_R = tc._build_envs(mps)

        zz = {}
        for x in range(Lx):
            for y in range(Ly - 1):
                zz_diag = tc._zz_1site_obs(y, y + 1)
                val = tc._expect_1site_diag(mps, x, zz_diag, env_L, env_R)
                zz[('v', x, y)] = val

        for x in range(Lx - 1):
            for y in range(Ly):
                zz_2d = tc._zz_2site_obs(y)
                val = tc._expect_2site_diag(mps, x, zz_2d, env_L, env_R)
                zz[('h', x, y)] = val

        return zz

    def _identify_hotspots(self, zz_per_edge):
        """Edges with |ZZ| < threshold are treated as hotspots."""
        hotspots = set()
        cold_cost = 0.0

        for key, zz_val in zz_per_edge.items():
            if abs(zz_val) < self.threshold:
                hotspots.add(key)
            else:
                cold_cost += (1 - zz_val) / 2

        if self.verbose:
            n_hot = len(hotspots)
            n_total = len(zz_per_edge)
            avg_zz_hot = (
                np.mean([abs(zz_per_edge[k]) for k in hotspots])
                if hotspots else 0
            )
            print(f"  [B70] Hotspots: {n_hot}/{n_total} edges "
                  f"(threshold={self.threshold}, avg |ZZ|={avg_zz_hot:.3f})")

        return hotspots, cold_cost

    def _build_reference_assignment(self, zz_per_edge):
        """
        Build a global reference bitstring from the most reliable Tier-1 edges.

        We propagate high-|ZZ| parity preferences first, then fill any remaining
        disconnected nodes with the grid checkerboard parity as a stable default.
        """
        assignment = {}
        ordered_edges = sorted(
            zz_per_edge.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )

        for key, zz_val in ordered_edges:
            etype, ex, ey = key
            u, v = self._edge_nodes(etype, ex, ey)
            want_same = bool(zz_val > 0)

            if not assignment:
                assignment[u] = 0

            if u in assignment and v not in assignment:
                assignment[v] = assignment[u] if want_same else 1 - assignment[u]
            elif v in assignment and u not in assignment:
                assignment[u] = assignment[v] if want_same else 1 - assignment[v]

        for x in range(self.Lx):
            for y in range(self.Ly):
                node = self._node_id(x, y)
                if node not in assignment:
                    assignment[node] = (x + y) % 2

        return assignment

    def _boundary_fixed_assignment(self, patch_nodes, key, reference_assignment):
        """
        Pin patch-boundary nodes to the global reference assignment.

        The hotspot edge endpoints stay free so the tiny exact gadget can still
        decide whether this edge should be same or opposite.
        """
        hotspot_nodes = set(self._edge_nodes(*key))
        patch_node_set = set(patch_nodes)
        fixed = {}
        for node in patch_nodes:
            if node in hotspot_nodes:
                continue
            touches_outside = any(
                nb not in patch_node_set for nb in self.graph_adjacency[node]
            )
            if touches_outside:
                fixed[node] = int(reference_assignment[node])
        return fixed

    def _exact_gadget_repair(self, key, zz_tier1, reference_assignment):
        """
        Try a tiny exact gadget repair on one hotspot edge.

        This path is conservative by design: we only keep the gadget result if
        it yields a strictly better local edge-cost than the Tier-1 estimate.
        """
        if self.exact_gadget_mode == 'off' or self.exact_gadget_nodes_max < 2:
            return None

        from maxcut_gadget_sat import extract_gadget_subgraph, solve_maxcut_gadget_exact

        etype, ex, ey = key
        patch_nodes = None
        chosen_radius = None
        for radius in range(self.exact_patch_radius, -1, -1):
            candidate_nodes = self._collect_patch_nodes(etype, ex, ey, radius)
            if len(candidate_nodes) <= self.exact_gadget_nodes_max:
                patch_nodes = candidate_nodes
                chosen_radius = radius
                break

        if patch_nodes is None or len(patch_nodes) < 2:
            return None

        patch = extract_gadget_subgraph(self.n_nodes, self.graph_edges, patch_nodes)
        if patch['n_nodes'] < 2 or not patch['edges']:
            return None

        u_old, v_old = self._edge_nodes(etype, ex, ey)
        forward = patch['forward_map']
        if u_old not in forward or v_old not in forward:
            return None

        u = forward[u_old]
        v = forward[v_old]
        if self.exact_gadget_mode == 'boundary':
            boundary_fixed = {
                forward[node]: bit
                for node, bit in self._boundary_fixed_assignment(
                    patch_nodes, key, reference_assignment).items()
                if node in forward
            }
        else:
            boundary_fixed = {}
        same = solve_maxcut_gadget_exact(
            patch['n_nodes'], patch['edges'],
            fixed_assignment={**boundary_fixed, u: 0, v: 0})
        opposite = solve_maxcut_gadget_exact(
            patch['n_nodes'], patch['edges'],
            fixed_assignment={**boundary_fixed, u: 0, v: 1})
        if not same.get('sat') or not opposite.get('sat'):
            return None

        same_weight = int(same['optimal_weight'])
        opposite_weight = int(opposite['optimal_weight'])
        if opposite_weight > same_weight:
            exact_cost = 1.0
        elif same_weight > opposite_weight:
            exact_cost = 0.0
        else:
            exact_cost = 0.5

        tier1_cost = (1 - zz_tier1) / 2
        return {
            'used': exact_cost > tier1_cost + 1e-12,
            'zz': 1 - 2 * exact_cost,
            'exact_cost': exact_cost,
            'tier1_cost': tier1_cost,
            'patch_nodes': patch['n_nodes'],
            'patch_edges': len(patch['edges']),
            'same_weight': same_weight,
            'opposite_weight': opposite_weight,
            'radius': chosen_radius,
            'boundary_pins': len(boundary_fixed),
            'mode': self.exact_gadget_mode,
            'certificate': opposite.get('certificate', 'EXACT_GADGET'),
        }

    def _repair_hotspots_lightcone(self, hotspot_edges, gammas_global, betas_global):
        """Lightcone fallback for hotspots not handled by exact gadgets."""
        from lightcone_qaoa import LightconeQAOA

        if not hotspot_edges:
            return {}

        lc = LightconeQAOA(self.Lx, self.Ly, verbose=False)
        p_local = self.p_local

        if p_local > self.p_global:
            g_avg = np.mean(gammas_global)
            b_avg = np.mean(betas_global)
            gammas_local = list(gammas_global) + [g_avg] * (p_local - self.p_global)
            betas_local = list(betas_global) + [b_avg] * (p_local - self.p_global)
        else:
            gammas_local = list(gammas_global[:p_local])
            betas_local = list(betas_global[:p_local])

        ref_edge = sorted(hotspot_edges)[0]
        best_ratio = -1
        best_g, best_b = gammas_local[0], betas_local[0]

        for g in np.linspace(0.1, np.pi / 2, 8):
            for b in np.linspace(0.1, np.pi / 2, 8):
                gs = [g] * p_local
                bs = [b] * p_local
                try:
                    zz = lc.eval_edge(ref_edge[0], ref_edge[1], ref_edge[2],
                                      p_local, gs, bs)
                    r = (1 - zz) / 2
                    if r > best_ratio:
                        best_ratio = r
                        best_g, best_b = g, b
                except Exception:
                    pass

        gammas_local = [best_g] * p_local
        betas_local = [best_b] * p_local

        try:
            from scipy.optimize import minimize as sp_min

            def neg_cost(params):
                gs = list(params[:p_local])
                bs = list(params[p_local:])
                total = 0
                for etype, ex, ey in sorted(hotspot_edges)[:5]:
                    try:
                        zz = lc.eval_edge(etype, ex, ey, p_local, gs, bs)
                        total += (1 - zz) / 2
                    except Exception:
                        pass
                return -total

            x0 = gammas_local + betas_local
            res = sp_min(neg_cost, x0, method='Nelder-Mead',
                         options={'maxiter': 50, 'xatol': 1e-4})
            gammas_local = list(res.x[:p_local])
            betas_local = list(res.x[p_local:])
        except Exception:
            pass

        zz_repaired = {}
        for etype, ex, ey in sorted(hotspot_edges):
            try:
                zz = lc.eval_edge(etype, ex, ey, p_local, gammas_local, betas_local)
                zz_repaired[(etype, ex, ey)] = zz
            except Exception as e:
                if self.verbose:
                    print(f"  [B70]   WARN: edge ({etype},{ex},{ey}) failed: {e}")
                try:
                    zz = lc.eval_edge(etype, ex, ey, 1,
                                      gammas_global[:1], betas_global[:1])
                    zz_repaired[(etype, ex, ey)] = zz
                except Exception:
                    pass

        return zz_repaired

    def _tier2_patches(self, hotspot_edges, zz_per_edge, gammas_global, betas_global):
        """
        Tier 2 repair:
        - first try tiny exact gadget patches
        - then repair the rest via lightcone
        """
        if not hotspot_edges:
            return {}, {'exact_gadget': 0, 'lightcone': 0, 'sources': {}, 'gadget_meta': {}}

        zz_repaired = {}
        repair_sources = {}
        gadget_meta = {}
        t0 = time.time()
        reference_assignment = (
            self._build_reference_assignment(zz_per_edge)
            if self.exact_gadget_mode == 'boundary' else None
        )

        for key in sorted(hotspot_edges):
            repair = self._exact_gadget_repair(
                key, zz_per_edge[key], reference_assignment)
            if repair is None:
                continue
            gadget_meta[key] = repair
            if repair['used']:
                zz_repaired[key] = repair['zz']
                repair_sources[key] = 'exact_gadget'

        remaining_edges = [key for key in sorted(hotspot_edges) if key not in zz_repaired]
        if remaining_edges:
            lightcone_repairs = self._repair_hotspots_lightcone(
                remaining_edges, gammas_global, betas_global)
            for key, val in lightcone_repairs.items():
                zz_repaired[key] = val
                repair_sources[key] = 'lightcone'

        elapsed = time.time() - t0
        if self.verbose:
            n_exact = sum(1 for src in repair_sources.values() if src == 'exact_gadget')
            n_lightcone = sum(1 for src in repair_sources.values() if src == 'lightcone')
            print(f"  [B70] Tier 2: {len(zz_repaired)} repaired "
                  f"(exact={n_exact}, lightcone={n_lightcone}, {elapsed:.2f}s)")

        return zz_repaired, {
            'exact_gadget': sum(1 for src in repair_sources.values() if src == 'exact_gadget'),
            'lightcone': sum(1 for src in repair_sources.values() if src == 'lightcone'),
            'sources': repair_sources,
            'gadget_meta': gadget_meta,
        }

    def solve(self):
        """Run the two-tier hotspot repair solver."""
        t0 = time.time()

        if self.verbose:
            ws = " + WS-QAOA" if self.warm else ""
            print(f"\n  [B70] Hotspot Repair: {self.Lx}x{self.Ly} "
                  f"p_global={self.p_global} p_local={self.p_local}{ws}")

        ratio_t1, zz_per_edge, gammas, betas, warm_angles = self._tier1_global()

        hotspots, cold_cost = self._identify_hotspots(zz_per_edge)
        if not hotspots:
            if self.verbose:
                print("  [B70] No hotspots found - Tier 1 is already good")
            return {
                'ratio': ratio_t1,
                'ratio_tier1': ratio_t1,
                'ratio_repaired': ratio_t1,
                'delta': 0.0,
                'n_hotspots': 0,
                'n_edges': self.n_edges,
                'n_exact_gadget_repairs': 0,
                'n_lightcone_repairs': 0,
                'exact_gadget_mode': self.exact_gadget_mode,
                'elapsed': time.time() - t0,
            }

        zz_repaired, repair_stats = self._tier2_patches(
            hotspots, zz_per_edge, gammas, betas)

        repaired_cost = cold_cost
        for key in hotspots:
            if key in zz_repaired:
                repaired_cost += (1 - zz_repaired[key]) / 2
            else:
                repaired_cost += (1 - zz_per_edge[key]) / 2

        ratio_repaired = repaired_cost / self.n_edges
        elapsed = time.time() - t0

        t1_hotspot_cost = sum((1 - zz_per_edge[k]) / 2 for k in hotspots)
        t2_hotspot_cost = sum((1 - zz_repaired.get(k, zz_per_edge[k])) / 2
                              for k in hotspots)

        if self.verbose:
            print("\n  [B70] Results:")
            print(f"    Tier 1 (global):   ratio={ratio_t1:.6f}")
            print(f"    Tier 2 (repaired): ratio={ratio_repaired:.6f}")
            print(f"    Delta:             {ratio_repaired - ratio_t1:+.6f}")
            print(f"    Hotspot cost T1:   {t1_hotspot_cost:.4f} "
                  f"({t1_hotspot_cost/self.n_edges:.4f}/edge)")
            print(f"    Hotspot cost T2:   {t2_hotspot_cost:.4f} "
                  f"({t2_hotspot_cost/self.n_edges:.4f}/edge)")
            print(f"    Exact gadget:      {repair_stats['exact_gadget']} edges")
            print(f"    Lightcone:         {repair_stats['lightcone']} edges")
            print(f"    Total time:        {elapsed:.2f}s")

        return {
            'ratio': ratio_repaired,
            'ratio_tier1': ratio_t1,
            'ratio_repaired': ratio_repaired,
            'delta': ratio_repaired - ratio_t1,
            'n_hotspots': len(hotspots),
            'n_edges': self.n_edges,
            'hotspot_edges': sorted(hotspots),
            'zz_tier1': zz_per_edge,
            'zz_repaired': zz_repaired,
            'repair_sources': repair_stats['sources'],
            'gadget_meta': repair_stats['gadget_meta'],
            'n_exact_gadget_repairs': repair_stats['exact_gadget'],
            'n_lightcone_repairs': repair_stats['lightcone'],
            'exact_gadget_mode': self.exact_gadget_mode,
            'gammas': gammas,
            'betas': betas,
            'warm_angles': warm_angles,
            'elapsed': elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description='B70: Hotspot Repair')
    parser.add_argument('--Lx', type=int, default=8)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--p-global', type=int, default=1)
    parser.add_argument('--p-local', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--exact-gadget-nodes-max', type=int, default=8)
    parser.add_argument('--exact-patch-radius', type=int, default=1)
    parser.add_argument('--exact-gadget-mode', type=str, default='boundary',
                        choices=['off', 'free', 'boundary'])
    args = parser.parse_args()

    solver = HotspotRepair(
        args.Lx, args.Ly,
        p_global=args.p_global,
        p_local=args.p_local,
        frustration_threshold=args.threshold,
        warm=args.warm,
        verbose=True,
        exact_gadget_nodes_max=args.exact_gadget_nodes_max,
        exact_patch_radius=args.exact_patch_radius,
        exact_gadget_mode=args.exact_gadget_mode,
    )

    result = solver.solve()
    print(f"\nFinal ratio: {result['ratio']:.6f} "
          f"(Tier 1: {result['ratio_tier1']:.6f}, "
          f"delta: {result['delta']:+.6f})")


if __name__ == '__main__':
    main()
