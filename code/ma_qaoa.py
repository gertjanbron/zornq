#!/usr/bin/env python3
"""
ma_qaoa.py - B67: Multi-Angle QAOA voor MaxCut.

Standaard QAOA geeft alle edges dezelfde gamma per laag. Multi-Angle QAOA
kent aparte gamma's toe per graadklasse van de edges. Op een Lx x Ly grid
zijn er typisch 4 klassen: (2,3), (3,3), (3,4), (4,4).

Dit verhoogt de expressiviteit zonder p te vergroten:
  p=3 standaard:  3 gamma + 3 beta = 6 parameters
  p=3 ma-QAOA:   12 gamma + 3 beta = 15 parameters (bij 4 klassen)

De optimizer gebruikt scipy.minimize met warm-start vanuit standaard-QAOA.

Gebruik:
  python ma_qaoa.py --Lx 6 --Ly 3 --p 2                  # vergelijk standaard vs ma
  python ma_qaoa.py --Lx 20 --Ly 3 --p 2 --gpu --fp32     # GPU productierun
  python ma_qaoa.py --Lx 8 --Ly 4 --p 1 --verbose         # details per edge-klasse

Bouwt voort op: lightcone_qaoa.py (LightconeQAOA als basis)
"""

import numpy as np
import math
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightcone_qaoa import LightconeQAOA, GPU_AVAILABLE

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:
    scipy_minimize = None


# =====================================================================
# Edge-klasse classificatie
# =====================================================================

def classify_grid_edges(Lx, Ly):
    """Classificeer alle edges op een Lx x Ly grid naar graadpaar-klasse.

    Returns:
        classes: dict {class_id: [(etype, ex, ey), ...]}
        edge_class: dict {(etype, ex, ey): class_id}
        class_names: dict {class_id: "(d1,d2)" label}
    """
    def degree(x, y):
        return (x > 0) + (x < Lx - 1) + (y > 0) + (y < Ly - 1)

    classes = {}
    edge_class = {}
    class_names = {}

    # Horizontale edges
    for x in range(Lx - 1):
        for y in range(Ly):
            d1 = degree(x, y)
            d2 = degree(x + 1, y)
            cid = (min(d1, d2), max(d1, d2))
            edge_class[('h', x, y)] = cid
            if cid not in classes:
                classes[cid] = []
                class_names[cid] = "(%d,%d)" % cid
            classes[cid].append(('h', x, y))

    # Verticale edges
    for x in range(Lx):
        for y in range(Ly - 1):
            d1 = degree(x, y)
            d2 = degree(x, y + 1)
            cid = (min(d1, d2), max(d1, d2))
            edge_class[('v', x, y)] = cid
            if cid not in classes:
                classes[cid] = []
                class_names[cid] = "(%d,%d)" % cid
            classes[cid].append(('v', x, y))

    return classes, edge_class, class_names


# =====================================================================
# Multi-Angle LightconeQAOA
# =====================================================================

class MultiAngleQAOA(LightconeQAOA):
    """Multi-Angle QAOA: aparte gamma per edge-graadklasse.

    Subclass van LightconeQAOA. Override eval_edge_exact om per-klasse
    gamma's te ondersteunen. eval_cost_ma berekent de totale cost
    met klasse-specifieke parameters.
    """

    def __init__(self, Lx, Ly=4, verbose=True, chi=None, gpu=False, fp32=False):
        super().__init__(Lx, Ly, verbose=verbose, chi=chi, gpu=gpu, fp32=fp32)

        # Classificeer edges
        self.classes, self.edge_class_map, self.class_names = \
            classify_grid_edges(Lx, Ly)
        self.class_ids = sorted(self.classes.keys())
        self.n_classes = len(self.class_ids)
        self.class_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}

        if verbose:
            print("  [B67] Edge-klassen: %d" % self.n_classes)
            for cid in self.class_ids:
                n = len(self.classes[cid])
                print("    %s: %d edges (%.1f%%)" % (
                    self.class_names[cid], n, 100 * n / self.n_edges))

    def _node_degree(self, x, y):
        """Graad van node (x,y) op het volledige grid."""
        return (x > 0) + (x < self.Lx - 1) + (y > 0) + (y < self.Ly - 1)

    def eval_edge_exact_ma(self, edge_type, edge_x, edge_y, p,
                           gamma_per_class, betas):
        """Bereken <ZZ> met per-klasse gamma's.

        Args:
            gamma_per_class: list van dicts, gamma_per_class[layer][class_id] = gamma
            betas: list van floats (1 per laag, gedeeld)
        """
        use_gpu = self.gpu and GPU_AVAILABLE
        xp = cp if use_gpu else np

        if self.fp32:
            fdtype = xp.float32
            cdtype = xp.complex64
        else:
            fdtype = xp.float64
            cdtype = xp.complex128

        Ly = self.Ly
        col_min, col_max = self.lightcone_columns(edge_type, edge_x, p)
        n_cols = col_max - col_min + 1
        n_qubits = n_cols * Ly
        dim = 2 ** n_qubits

        sv_limit = 26 if use_gpu else 22
        if n_qubits > sv_limit:
            raise ValueError("Lichtkegel %d qubits > %d" % (n_qubits, sv_limit))

        # B65: Buffer reuse
        if dim > self._buf_max_dim:
            self._buf_max_dim = dim
            self._buf_state = xp.empty(dim, dtype=cdtype)
            self._buf_hphase = xp.empty(dim, dtype=fdtype)
            self._buf_z = xp.empty(dim, dtype=fdtype)

        state = self._buf_state[:dim]
        z_scratch = self._buf_z[:dim]

        bitstrings = xp.arange(dim)

        def qi(col, row):
            return (col - col_min) * Ly + row

        # Z-diagonalen cache
        z_cache = {}
        for x in range(col_min, col_max + 1):
            for y in range(Ly):
                idx = qi(x, y)
                z_cache[(x, y)] = (1 - 2 * ((bitstrings >> idx) & 1)).astype(fdtype)
        del bitstrings

        # B67: Decomponeer H_phase per edge-klasse
        # Bereken ZZ-product per edge in het sub-circuit en groepeer per klasse
        h_phase_per_class = {}
        for cid in self.class_ids:
            h_phase_per_class[cid] = xp.zeros(dim, dtype=fdtype)

        # Verticale edges in sub-circuit
        for x in range(col_min, col_max + 1):
            for y in range(Ly - 1):
                cid = self.edge_class_map[('v', x, y)]
                h_phase_per_class[cid] += z_cache[(x, y)] * z_cache[(x, y + 1)]

        # Horizontale edges in sub-circuit
        for x in range(col_min, col_max):
            for y in range(Ly):
                cid = self.edge_class_map[('h', x, y)]
                h_phase_per_class[cid] += z_cache[(x, y)] * z_cache[(x + 1, y)]

        # Target edge observatie
        if edge_type == 'h':
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x + 1, edge_y)]
        else:
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x, edge_y + 1)]
        del z_cache

        # Mixer functie (zelfde als base class)
        def apply_rx_all(state, beta):
            cb = fdtype(math.cos(float(beta)))
            msb = cdtype(-1j * math.sin(float(beta)))
            if use_gpu and hasattr(xp, 'fuse'):
                @xp.fuse()
                def _rx_fused(s0, s1):
                    return cb * s0 + msb * s1, msb * s0 + cb * s1
                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    new0, new1 = _rx_fused(s[:, 0, :], s[:, 1, :])
                    s[:, 0, :] = new0
                    s[:, 1, :] = new1
                    state = s.reshape(-1)
            else:
                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    tmp = cb * s[:, 0, :] + msb * s[:, 1, :]
                    s[:, 1, :] = msb * s[:, 0, :] + cb * s[:, 1, :]
                    s[:, 0, :] = tmp
                    state = s.reshape(-1)
            return state

        # Start: |+>^n
        state[:] = cdtype(1.0 / math.sqrt(dim))

        # QAOA circuit met per-klasse gamma's
        for layer in range(p):
            # Combineer H_phase: sum(gamma_c * H_phase_c)
            combined_phase = xp.zeros(dim, dtype=fdtype)
            for cid in self.class_ids:
                gamma_c = fdtype(gamma_per_class[layer].get(cid, 0.0))
                if gamma_c != 0:
                    combined_phase += gamma_c * h_phase_per_class[cid]

            state *= xp.exp(cdtype(-1j) * combined_phase)
            state = apply_rx_all(state, betas[layer])

        # Meet <ZZ>
        zz_obs = z_obs_a * z_obs_b
        probs = xp.abs(state)**2
        result = float(xp.dot(probs, zz_obs))

        del probs, zz_obs, z_obs_a, z_obs_b, combined_phase
        for cid in list(h_phase_per_class.keys()):
            del h_phase_per_class[cid]

        return result

    def eval_cost_ma(self, p, gamma_per_class, betas):
        """Bereken totale MaxCut cost met per-klasse gamma's.

        gamma_per_class: list van dicts, gamma_per_class[layer][class_id] = gamma
        betas: list van floats
        """
        total = 0.0

        if self.gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        # B66: Symmetrie-caching (werkt ook voor ma-QAOA omdat
        # symmetrische edges dezelfde klasse hebben en dezelfde sub-circuit)
        cache = {}
        Ly = self.Ly

        for idx, (etype, ex, ey) in enumerate(self.edges):
            col_min, col_max = self.lightcone_columns(etype, ex, p)

            if etype == 'h':
                ey_sym = min(ey, Ly - 1 - ey)
            else:
                ey_sym = min(ey, Ly - 2 - ey)

            dist_left = ex - col_min
            dist_right = col_max - (ex + (1 if etype == 'h' else 0))
            n_cols = col_max - col_min + 1

            is_bulk = (col_min > 0 and col_max < self.Lx - 1)
            if is_bulk:
                cache_key = (etype, ey_sym, n_cols, 'bulk')
            else:
                cache_key = (etype, ey_sym, dist_left, dist_right)

            if cache_key in cache:
                zz = cache[cache_key]
            else:
                zz = self.eval_edge_exact_ma(etype, ex, ey, p,
                                              gamma_per_class, betas)
                cache[cache_key] = zz

            total += 0.5 * (1 - zz)

        ratio = total / self.n_edges
        return ratio

    def eval_ratio_ma(self, p, gamma_per_class, betas):
        """Wrapper: bereken ratio met ma-QAOA parameters."""
        return self.eval_cost_ma(p, gamma_per_class, betas)

    def _pack_params(self, gamma_per_class_list, betas):
        """Pack ma-QAOA params naar platte vector voor scipy."""
        vec = []
        for layer_gammas in gamma_per_class_list:
            for cid in self.class_ids:
                vec.append(layer_gammas.get(cid, 0.0))
        vec.extend(betas)
        return np.array(vec)

    def _unpack_params(self, vec, p):
        """Unpack platte vector naar gamma_per_class + betas."""
        n_c = self.n_classes
        gamma_per_class = []
        for layer in range(p):
            layer_gammas = {}
            for i, cid in enumerate(self.class_ids):
                layer_gammas[cid] = float(vec[layer * n_c + i])
            gamma_per_class.append(layer_gammas)
        betas = list(vec[p * n_c:])
        return gamma_per_class, betas

    def optimize_ma(self, p, gammas_init=None, betas_init=None,
                    method='Nelder-Mead', maxiter=500):
        """Optimaliseer ma-QAOA parameters via scipy.minimize.

        Warm-start: als gammas_init/betas_init gegeven, gebruik die als
        startpunt (uitgebreid naar per-klasse: elke klasse krijgt dezelfde
        initiële gamma).
        """
        if scipy_minimize is None:
            raise ImportError("scipy vereist voor ma-QAOA optimizer")

        n_c = self.n_classes
        n_params = p * n_c + p  # p*n_classes gammas + p betas

        # Warm-start vanuit standaard QAOA
        if gammas_init is not None and betas_init is not None:
            gamma_per_class_init = []
            for layer in range(p):
                layer_gammas = {cid: gammas_init[layer] for cid in self.class_ids}
                gamma_per_class_init.append(layer_gammas)
            x0 = self._pack_params(gamma_per_class_init, list(betas_init))
        else:
            # Random init
            x0 = np.random.uniform(0, np.pi, n_params)

        # Evaluatie-teller
        self._ma_n_evals = 0
        self._ma_best_ratio = 0.0
        self._ma_best_params = None
        t0 = time.time()

        def objective(x):
            gamma_pc, betas = self._unpack_params(x, p)
            ratio = self.eval_ratio_ma(p, gamma_pc, betas)
            self._ma_n_evals += 1
            if ratio > self._ma_best_ratio:
                self._ma_best_ratio = ratio
                self._ma_best_params = x.copy()
            return -ratio  # minimize negatieve ratio

        if self.verbose:
            print("  [B67] ma-QAOA optimalisatie: p=%d, %d klassen, %d params" % (
                p, n_c, n_params))
            if gammas_init is not None:
                init_ratio = self.eval_ratio_ma(
                    p,
                    [{cid: gammas_init[layer] for cid in self.class_ids}
                     for layer in range(p)],
                    list(betas_init))
                print("  [B67] Warm-start ratio (uniform): %.6f" % init_ratio)

        result = scipy_minimize(
            objective, x0,
            method=method,
            options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-6,
                     'adaptive': True}
        )

        elapsed = time.time() - t0

        # Unpack beste params
        best_gamma_pc, best_betas = self._unpack_params(
            self._ma_best_params if self._ma_best_params is not None else result.x,
            p)

        info = {
            'total_time': elapsed,
            'n_evals': self._ma_n_evals,
            'scipy_success': result.success,
            'scipy_message': result.message,
            'n_classes': n_c,
            'n_params': n_params,
        }

        if self.verbose:
            print("  [B67] Resultaat: ratio=%.6f (%d evals, %.1fs)" % (
                self._ma_best_ratio, self._ma_n_evals, elapsed))
            print("  [B67] Parameters per laag:")
            for layer in range(p):
                parts = []
                for cid in self.class_ids:
                    parts.append("%s=%.4f" % (
                        self.class_names[cid], best_gamma_pc[layer][cid]))
                print("    Laag %d: gamma[%s] beta=%.4f" % (
                    layer + 1, ", ".join(parts), best_betas[layer]))

        return {
            'ratio': self._ma_best_ratio,
            'gamma_per_class': best_gamma_pc,
            'betas': best_betas,
            'info': info,
        }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='B67: Multi-Angle QAOA')
    parser.add_argument('--Lx', type=int, default=6)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--ngamma', type=int, default=10,
                        help='Grid-zoekpunten voor standaard QAOA warm-start')
    parser.add_argument('--nbeta', type=int, default=10)
    parser.add_argument('--maxiter', type=int, default=500,
                        help='Max iteraties voor ma-QAOA optimizer')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-standard', action='store_true',
                        help='Sla standaard QAOA over (gebruik random init)')
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  B67: Multi-Angle QAOA")
    print(sep)

    # Stap 1: Standaard QAOA als warm-start
    std_gammas = None
    std_betas = None
    std_ratio = None

    if not args.skip_standard:
        print("\n  Stap 1: Standaard QAOA (warm-start)")
        print("  " + "-" * 50)
        lc = LightconeQAOA(args.Lx, args.Ly, verbose=args.verbose,
                           gpu=args.gpu, fp32=args.fp32)
        t0 = time.time()
        std_results = lc.optimize_progressive(
            p_max=args.p, n_gamma=args.ngamma, n_beta=args.nbeta)
        std_elapsed = time.time() - t0

        best_p = max(std_results.keys())
        std_ratio = std_results[best_p]['ratio']
        std_gammas = std_results[best_p]['gammas']
        std_betas = std_results[best_p]['betas']

        print("  Standaard QAOA: ratio=%.6f (%.1fs)" % (std_ratio, std_elapsed))

    # Stap 2: Multi-Angle QAOA
    print("\n  Stap 2: Multi-Angle QAOA")
    print("  " + "-" * 50)
    ma = MultiAngleQAOA(args.Lx, args.Ly, verbose=True,
                        gpu=args.gpu, fp32=args.fp32)

    ma_result = ma.optimize_ma(
        p=args.p,
        gammas_init=std_gammas,
        betas_init=std_betas,
        maxiter=args.maxiter)

    # Vergelijking
    print("\n" + sep)
    print("  VERGELIJKING")
    print(sep)
    print("  Grid: %dx%d, p=%d" % (args.Lx, args.Ly, args.p))
    print("  Edge-klassen: %d %s" % (
        ma.n_classes, [ma.class_names[c] for c in ma.class_ids]))
    if std_ratio is not None:
        print("  Standaard QAOA:  ratio=%.6f  (6 params)" % std_ratio)
    ma_ratio = ma_result['ratio']
    n_params = ma_result['info']['n_params']
    print("  Multi-Angle:     ratio=%.6f  (%d params)" % (ma_ratio, n_params))
    if std_ratio is not None:
        delta = ma_ratio - std_ratio
        print("  Verschil:        %+.6f (%+.2f%%)" % (
            delta, 100 * delta / max(std_ratio, 1e-10)))
    print("  Evaluaties:      %d" % ma_result['info']['n_evals'])
    print("  Tijd:            %.1fs" % ma_result['info']['total_time'])
    print(sep)



if __name__ == '__main__':
    main()
