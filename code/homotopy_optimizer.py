#\!/usr/bin/env python3
"""
homotopy_optimizer.py - B71: Homotopy Optimizer — Parameter Continuation.

Optimaliseer QAOA-parameters via homotopy continuation:
  1. λ-continuation: H(λ) = H_intra + λ·H_inter (p=1, goedkoop)
  2. p-continuation: layer-by-layer warm-start (p=1→p_max)
  3. Auto: combineer beide

Bouwt voort op: transverse_contraction.py (B26), ws_qaoa.py (B69)
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transverse_contraction import TransverseQAOA


class HomotopyQAOA(TransverseQAOA):
    """QAOA MaxCut optimizer via homotopy continuation.

    Twee continuation-strategieën:
    1. λ-continuation: interpoleer inter-column koppelingen van 0→1
    2. p-continuation: optimaliseer layer-by-layer (p=1→p_max)
    """

    def __init__(self, Lx, Ly=1, verbose=True):
        super().__init__(Lx, Ly, verbose=verbose)
        self.n_intra = (Ly - 1) * Lx
        self.n_inter = Ly * (Lx - 1)

    def _zz_inter_diag_lambda(self, gamma, lam):
        """exp(-i*gamma*lam * sum_y Z_y^L Z_y^R)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        phase = np.zeros((d, d))
        for y in range(Ly):
            z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
            z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
            phase += z_L * z_R
        return np.exp(-1j * gamma * lam * phase)

    def eval_ratio_lambda(self, p, gammas, betas, lam=1.0,
                          warm_angles=None):
        """Bereken MaxCut ratio bij interpolatieparameter lam.

        Gates: intra-ZZ altijd vol, inter-ZZ geschaald met lam.
        Kosten: n_intra + lam * n_inter edges.
        """
        Lx, Ly, d = self.Lx, self.Ly, self.d
        chi_exact = d ** p

        if warm_angles is not None:
            from ws_qaoa import warm_start_mps
            mps = warm_start_mps(Lx, Ly, warm_angles)
        else:
            mps = [np.ones((1, d, 1), dtype=complex) / np.sqrt(d)
                   for _ in range(Lx)]

        for layer in range(p):
            g = gammas[layer]
            b = betas[layer]

            intra = self._zz_intra_diag(g)
            for x in range(Lx):
                mps[x] = mps[x] * intra[None, :, None]

            if lam > 1e-12:
                inter = self._zz_inter_diag_lambda(g, lam)
                for x in range(Lx - 1):
                    mps = self._apply_2site_exact(mps, x, inter, chi_exact)

            rx = self._rx_col(b)
            for x in range(Lx):
                mps[x] = np.einsum('ij,ajb->aib', rx, mps[x])

        env_L, env_R = self._build_envs(mps)
        total_cost = 0.0

        for x in range(Lx):
            for y in range(Ly - 1):
                zz_diag = self._zz_1site_obs(y, y + 1)
                zz_val = self._expect_1site_diag(
                    mps, x, zz_diag, env_L, env_R)
                total_cost += (1 - zz_val) / 2

        if lam > 1e-12:
            for x in range(Lx - 1):
                for y in range(Ly):
                    obs = self._zz_2site_obs(y)
                    zz_val = self._expect_2site_diag(
                        mps, x, obs, env_L, env_R)
                    total_cost += lam * (1 - zz_val) / 2

        n_eff = self.n_intra + lam * self.n_inter
        if n_eff < 1e-10:
            return 0.0
        return total_cost / n_eff

    def optimize_at_lambda(self, p, lam, init_gammas=None, init_betas=None,
                           n_gamma=10, n_beta=10, refine=True,
                           warm_angles=None, maxiter=None):
        """Optimaliseer (gamma, beta) bij vaste lam."""
        from scipy.optimize import minimize as scipy_minimize

        is_warm = init_gammas is not None and init_betas is not None
        if maxiter is None:
            maxiter = 40 if is_warm else 150

        def neg_ratio(params):
            gs = list(params[:p])
            bs = list(params[p:])
            return -self.eval_ratio_lambda(p, gs, bs, lam=lam,
                                           warm_angles=warm_angles)

        if is_warm:
            best_gammas = list(init_gammas)
            best_betas = list(init_betas)
            best_ratio = self.eval_ratio_lambda(
                p, best_gammas, best_betas, lam=lam,
                warm_angles=warm_angles)
        else:
            gamma_range = np.linspace(0.05, np.pi, n_gamma)
            beta_range = np.linspace(0.05, np.pi / 2, n_beta)
            best_ratio = -1
            best_gammas = [gamma_range[0]] * p
            best_betas = [beta_range[0]] * p
            old_verbose = self.verbose
            self.verbose = False
            for g in gamma_range:
                for b in beta_range:
                    r = self.eval_ratio_lambda(
                        p, [g] * p, [b] * p, lam=lam,
                        warm_angles=warm_angles)
                    if r > best_ratio:
                        best_ratio = r
                        best_gammas = [g] * p
                        best_betas = [b] * p
            self.verbose = old_verbose

        if refine:
            x0 = best_gammas + best_betas
            bounds = [(0.01, np.pi)] * p + [(0.01, np.pi / 2)] * p
            result = scipy_minimize(
                neg_ratio, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-8})
            if -result.fun > best_ratio:
                best_ratio = -result.fun
                best_gammas = list(result.x[:p])
                best_betas = list(result.x[p:])

        return best_ratio, best_gammas, best_betas

    def solve_lambda(self, p, n_lambda=5, n_gamma=10, n_beta=10,
                     warm_angles=None):
        """Lambda-continuation: lam=0 -> lam=1."""
        t0 = time.time()
        lambdas = np.linspace(0, 1, n_lambda + 1)

        if self.verbose:
            print(f"  [B71] lam-Homotopy: {self.Lx}x{self.Ly} "
                  f"p={p} n_lam={n_lambda}")

        trajectory = []
        gammas = None
        betas = None

        for i, lam in enumerate(lambdas):
            t_step = time.time()
            if i == 0:
                ratio, gammas, betas = self.optimize_at_lambda(
                    p, lam, n_gamma=n_gamma, n_beta=n_beta,
                    warm_angles=warm_angles)
            else:
                ratio, gammas, betas = self.optimize_at_lambda(
                    p, lam, init_gammas=gammas, init_betas=betas,
                    warm_angles=warm_angles)

            dt = time.time() - t_step
            trajectory.append({
                'lambda': float(lam), 'ratio': float(ratio),
                'gammas': [float(g) for g in gammas],
                'betas': [float(b) for b in betas],
                'time': dt,
            })
            if self.verbose:
                g_str = ', '.join(f'{g:.4f}' for g in gammas)
                b_str = ', '.join(f'{b:.4f}' for b in betas)
                print(f"    lam={lam:.2f}: ratio={ratio:.6f} "
                      f"g=[{g_str}] b=[{b_str}] ({dt:.2f}s)")

        return {
            'ratio': trajectory[-1]['ratio'],
            'gammas': trajectory[-1]['gammas'],
            'betas': trajectory[-1]['betas'],
            'trajectory': trajectory,
            'elapsed': time.time() - t0,
        }

    def solve_p_continuation(self, p_max, n_gamma=10, n_beta=10,
                             warm_angles=None):
        """p-Continuation: layer-by-layer."""
        t0 = time.time()
        if self.verbose:
            print(f"  [B71] p-Continuation: {self.Lx}x{self.Ly} "
                  f"p=1->{p_max}")

        layers = []
        for p in range(1, p_max + 1):
            t_step = time.time()
            if p == 1:
                ratio, gammas, betas = self.optimize_at_lambda(
                    p, 1.0, n_gamma=n_gamma, n_beta=n_beta,
                    warm_angles=warm_angles, maxiter=200)
            else:
                prev_g = layers[-1]['gammas']
                prev_b = layers[-1]['betas']
                init_g = prev_g + [prev_g[-1]]
                init_b = prev_b + [prev_b[-1]]
                ratio, gammas, betas = self.optimize_at_lambda(
                    p, 1.0, init_gammas=init_g, init_betas=init_b,
                    warm_angles=warm_angles, maxiter=100)

            dt = time.time() - t_step
            layers.append({
                'p': p, 'ratio': float(ratio),
                'gammas': [float(g) for g in gammas],
                'betas': [float(b) for b in betas],
                'time': dt,
            })
            if self.verbose:
                g_str = ', '.join(f'{g:.4f}' for g in gammas)
                b_str = ', '.join(f'{b:.4f}' for b in betas)
                print(f"    p={p}: ratio={ratio:.6f} "
                      f"g=[{g_str}] b=[{b_str}] ({dt:.1f}s)")

        return {
            'ratio': layers[-1]['ratio'],
            'gammas': layers[-1]['gammas'],
            'betas': layers[-1]['betas'],
            'layers': layers,
            'elapsed': time.time() - t0,
        }

    def solve(self, p_max, n_lambda=5, n_gamma=10, n_beta=10,
              warm_angles=None, mode='auto'):
        """Volledige homotopy solver.

        Modes:
            'lambda': alleen lam-continuation bij vaste p
            'p': alleen p-continuation (layer-by-layer)
            'auto': lam-continuation bij p=1, dan p-continuation tot p_max
        """
        t0 = time.time()

        if mode == 'lambda':
            result = self.solve_lambda(
                p_max, n_lambda=n_lambda, n_gamma=n_gamma, n_beta=n_beta,
                warm_angles=warm_angles)
        elif mode == 'p':
            result = self.solve_p_continuation(
                p_max, n_gamma=n_gamma, n_beta=n_beta,
                warm_angles=warm_angles)
        else:
            if self.verbose:
                print(f"  [B71] Auto: lam-homotopy p=1, "
                      f"dan p-continuation tot p={p_max}")

            lam_result = self.solve_lambda(
                1, n_lambda=n_lambda, n_gamma=n_gamma, n_beta=n_beta,
                warm_angles=warm_angles)

            if p_max == 1:
                result = lam_result
            else:
                if self.verbose:
                    print(f"\n  [B71] p-Continuation vanuit lam-optimum...")

                layers = [{
                    'p': 1, 'ratio': lam_result['ratio'],
                    'gammas': lam_result['gammas'],
                    'betas': lam_result['betas'],
                    'time': lam_result['elapsed'],
                }]
                gammas = lam_result['gammas']
                betas = lam_result['betas']

                for p in range(2, p_max + 1):
                    t_step = time.time()
                    init_g = gammas + [gammas[-1]]
                    init_b = betas + [betas[-1]]
                    ratio, gammas, betas = self.optimize_at_lambda(
                        p, 1.0, init_gammas=init_g, init_betas=init_b,
                        warm_angles=warm_angles, maxiter=100)
                    dt = time.time() - t_step
                    layer = {
                        'p': p, 'ratio': float(ratio),
                        'gammas': [float(g) for g in gammas],
                        'betas': [float(b) for b in betas],
                        'time': dt,
                    }
                    layers.append(layer)
                    if self.verbose:
                        g_str = ', '.join(f'{g:.4f}' for g in gammas)
                        b_str = ', '.join(f'{b:.4f}' for b in betas)
                        print(f"    p={p}: ratio={ratio:.6f} "
                              f"g=[{g_str}] b=[{b_str}] ({dt:.1f}s)")

                result = {
                    'ratio': layers[-1]['ratio'],
                    'gammas': layers[-1]['gammas'],
                    'betas': layers[-1]['betas'],
                    'layers': layers,
                    'lambda_trajectory': lam_result.get('trajectory'),
                    'elapsed': time.time() - t0,
                }

        # Vergelijk met directe optimalisatie
        if self.verbose:
            print(f"\n  [B71] Direct optimalisatie p={p_max} (baseline)...")
        t_direct = time.time()
        ratio_direct, g_direct, b_direct = self.optimize_at_lambda(
            p_max, 1.0, n_gamma=n_gamma, n_beta=n_beta,
            warm_angles=warm_angles, maxiter=200)
        dt_direct = time.time() - t_direct

        delta = result['ratio'] - ratio_direct

        if self.verbose:
            print(f"    Direct: ratio={ratio_direct:.6f} ({dt_direct:.1f}s)")
            print(f"\n  [B71] Eindresultaten:")
            print(f"    Homotopy ratio: {result['ratio']:.6f}")
            print(f"    Direct ratio:   {ratio_direct:.6f}")
            print(f"    Delta:          {delta:+.6f}")
            print(f"    Totale tijd:    {time.time() - t0:.1f}s")

        result['ratio_direct'] = ratio_direct
        result['delta'] = delta
        result['gammas_direct'] = [float(g) for g in g_direct]
        result['betas_direct'] = [float(b) for b in b_direct]
        result['elapsed'] = time.time() - t0

        return result


def main():
    parser = argparse.ArgumentParser(
        description='B71: Homotopy Optimizer')
    parser.add_argument('--Lx', type=int, default=8)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--p', type=int, default=2,
                        help='Max QAOA-diepte (default: 2)')
    parser.add_argument('--n-lambda', type=int, default=5,
                        help='Lambda-stappen (default: 5)')
    parser.add_argument('--n-gamma', type=int, default=10)
    parser.add_argument('--n-beta', type=int, default=10)
    parser.add_argument('--mode', choices=['auto', 'lambda', 'p'],
                        default='auto')
    parser.add_argument('--warm', action='store_true',
                        help='WS-QAOA warm-start (B69)')
    args = parser.parse_args()

    warm_angles = None
    if args.warm:
        from ws_qaoa import sdp_warm_start
        Lx, Ly = args.Lx, args.Ly
        edges = []
        for x in range(Lx):
            for y in range(Ly):
                node = x * Ly + y
                if y < Ly - 1:
                    edges.append((node, node + 1))
                if x < Lx - 1:
                    edges.append((node, node + Ly))
        warm_angles = sdp_warm_start(Lx, Ly, edges, epsilon=0.1,
                                     mode='binary', verbose=True)

    solver = HomotopyQAOA(args.Lx, args.Ly, verbose=True)
    solver.solve(args.p, n_lambda=args.n_lambda,
                 n_gamma=args.n_gamma, n_beta=args.n_beta,
                 warm_angles=warm_angles, mode=args.mode)


if __name__ == '__main__':
    main()
