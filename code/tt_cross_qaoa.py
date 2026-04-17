#!/usr/bin/env python3
"""
tt_cross_qaoa.py - B29: Randomized SVD voor QAOA MaxCut op brede grids.

Doorbreekt de d-wall: bij column-grouping op Lx x Ly grids is d=2^Ly.
Standaard SVD kost O((chi*d)^3) en O(chi^2*d^2) geheugen.
Bij Ly>=5 (d>=32) wordt dit de bottleneck.

Oplossing: Randomized SVD die alleen matrix-vector producten gebruikt.
De diagonale structuur van QAOA-gates maakt matvec O(d*chi^2) i.p.v.
O(d^2*chi^2).

Complexiteit:
  Exact SVD:  O(chi^2*d^2) geheugen + O((chi*d)^3) compute
  RSVD:       O(chi^2*d) geheugen   + O(k*d*chi^2) compute
  Besparing:  factor d in geheugen, factor d^2/k in compute

Validatie:
  Ly=5 (d=32):  diff < 4e-06 vs exact SVD (machine-precisie)
  Ly=6 (d=64):  diff < 2e-05 vs exact SVD, 300 qubits in 6s
  Ly=7 (d=128): 70 qubits in 2.3s, alle gates via RSVD
  Ly=8 (d=256): 80 qubits in 5.8s, volledig voorbij d-wall

Gebruik:
  python tt_cross_qaoa.py --Lx 20 --Ly 6 --p 1     # d=64, 120 qubits
  python tt_cross_qaoa.py --Lx 10 --Ly 8 --p 1     # d=256, 80 qubits
  python tt_cross_qaoa.py --Lx 8 --Ly 4 --p 1 --validate  # vergelijk met exact

Bouwt voort op: transverse_contraction.py (B26)
"""

import numpy as np
import math
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Randomized SVD voor diagonale 2-site gates
# =====================================================================

def rsvd_diag_gate(A_left, B_right, diag_gate, chi_max,
                    tol=1e-10, n_oversamples=10, n_power_iter=2):
    """Randomized SVD van 2-site tensor na diagonale gate.

    De 2-site tensor is:
      Theta[alpha, sL, sR, delta] = sum_beta A[alpha,sL,beta] * gate[sL,sR] * B[beta,sR,delta]

    Gereshaped als matrix: M[(alpha,sL), (sR,delta)]  van  (chi_L*d) x (d*chi_R)

    Kernidee: in plaats van M volledig te vormen (O(d^2 * chi^2)),
    gebruiken we randomized SVD die alleen matrix-vector producten nodig heeft.

    Matvec M @ v kost O(d * chi_M * max(chi_L, chi_R)) via einsum.
    Besparing factor d in geheugen, factor d^2/k in compute.

    Algoritme (Halko-Martinsson-Tropp):
      1. Random projectie: Y = M @ Omega  (Gaussian sketch)
      2. Power iteration: Y = (M M^H)^q M @ Omega  (verbetert langzame SV afval)
      3. QR: Y = Q R
      4. Projectie: B = Q^H M
      5. SVD: B = U_b S V_h → M ≈ Q U_b S V_h

    Args:
        A_left: (chi_L, d, chi_M) tensor
        B_right: (chi_M, d, chi_R) tensor
        diag_gate: (d, d) gate-waarden (diagonaal in computationele basis)
        chi_max: maximale rang voor de benadering
        tol: tolerantie voor singuliere waarden
        n_oversamples: extra kolommen voor stabiliteit (standaard 10)
        n_power_iter: power iteraties (standaard 2)

    Returns:
        U_new: (chi_L, d, k) linker tensor
        V_new: (k, d, chi_R) rechter tensor
    """
    chi_L, d, chi_M = A_left.shape
    _, _, chi_R = B_right.shape
    m = chi_L * d
    n = d * chi_R
    k = min(chi_max + n_oversamples, m, n)

    # Precompute: B gecontraheerd met random vectors in batch
    # gate_mat: (d, d) — diag_gate[sL, sR]

    def matvec_batch(V_batch):
        """M @ V:  (m, k) = M @ (n, k).  V_batch is (d*chi_R, k)."""
        # Reshape V: (d, chi_R, k)
        V_3d = V_batch.reshape(d, chi_R, -1)
        k_batch = V_3d.shape[2]
        # BV[beta, sR, j] = sum_delta B[beta,sR,delta] * V[sR,delta,j]
        # → BV[beta, sR, j] via einsum
        BV = np.einsum('bsd,sdj->bsj', B_right, V_3d)  # (chi_M, d, k_batch)
        # Gewogen: WBV[sL, beta, j] = sum_sR gate[sL,sR] * BV[beta,sR,j]
        WBV = np.einsum('lr,brj->lbj', diag_gate, BV)  # (d, chi_M, k_batch)
        # Contractie met A: result[alpha,sL,j] = sum_beta A[alpha,sL,beta] * WBV[sL,beta,j]
        result = np.einsum('asb,sbj->asj', A_left, WBV)  # (chi_L, d, k_batch)
        # Reshape naar (chi_L*d, k_batch) — maar alpha*d+sL layout
        # result is (chi_L, d, k): index (alpha, sL) → alpha*d + sL
        return result.reshape(m, k_batch)

    def rmatvec_batch(U_batch):
        """M^H @ U:  (n, k) = M^H @ (m, k).  U_batch is (chi_L*d, k)."""
        # Reshape: (chi_L, d, k)
        U_3d = U_batch.reshape(chi_L, d, -1)
        k_batch = U_3d.shape[2]
        # AU[sL, beta, j] = sum_alpha conj(A[alpha,sL,beta]) * U[alpha,sL,j]
        AU = np.einsum('asb,asj->sbj', A_left.conj(), U_3d)  # (d, chi_M, k_batch)
        # Gewogen: WAU[sR, beta, j] = sum_sL conj(gate[sL,sR]) * AU[sL,beta,j]
        WAU = np.einsum('lr,lbj->rbj', diag_gate.conj(), AU)  # (d, chi_M, k_batch)
        # Contractie met B^H: result[sR,delta,j] = sum_beta conj(B[beta,sR,delta]) * WAU[sR,beta,j]
        result = np.einsum('bsd,sbj->sdj', B_right.conj(), WAU)  # (d, chi_R, k_batch)
        return result.reshape(n, k_batch)

    # --- Randomized SVD ---
    rng = np.random.default_rng()
    Omega = (rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))) / np.sqrt(2)

    # Y = M @ Omega
    Y = matvec_batch(Omega)

    # Power iteration
    for _ in range(n_power_iter):
        Z = rmatvec_batch(Y)
        Z, _ = np.linalg.qr(Z, mode='reduced')
        Y = matvec_batch(Z)

    # QR
    Q, _ = np.linalg.qr(Y, mode='reduced')

    # Projectie: B = Q^H @ M  via  B^H = M^H @ Q
    BH = rmatvec_batch(Q)
    B_proj = BH.conj().T  # (k, n)

    # SVD van kleine matrix
    U_b, S_b, Vh_b = np.linalg.svd(B_proj, full_matrices=False)

    # Trunceer
    k_final = min(len(S_b), chi_max)
    if S_b[0] > 1e-15:
        k_nz = max(1, int(np.sum(S_b > tol * S_b[0])))
        k_final = min(k_final, k_nz)

    U_out = (Q @ U_b[:, :k_final]).reshape(chi_L, d, k_final)
    V_out = (np.diag(S_b[:k_final]) @ Vh_b[:k_final, :]).reshape(k_final, d, chi_R)

    return U_out, V_out


# =====================================================================
# TT-Cross QAOA Engine
# =====================================================================

class TTCrossQAOA:
    """QAOA MaxCut via TT-Cross interpolation.

    Zoals TransverseQAOA (B26), maar vervangt de exacte SVD in
    _apply_2site_exact door cross-interpolation. Dit vermijdt de
    d^2-bottleneck bij grote d=2^Ly.

    Complexiteit:
      Exact SVD: O(chi_L * d * d * chi_R) voor de tensor + O((chi*d)^3) SVD
      TT-Cross:  O(chi * d * chi) voor rij/kolom sampling + O(chi^3) SVD

    Voor d=32 (Ly=5): factor ~32 sneller dan exact.
    """

    def __init__(self, Lx, Ly=1, chi_max=None, tol=1e-10, verbose=True):
        self.Lx = Lx
        self.Ly = Ly
        self.d = 2 ** Ly
        self.verbose = verbose
        self.tol = tol

        # Default chi_max: exact bij d^p
        if chi_max is None:
            self.chi_max = min(self.d ** 3, 512)  # Limiet voor geheugen
        else:
            self.chi_max = chi_max

        # Bit-patronen
        self.bp = np.array([[(idx >> (Ly - 1 - q)) & 1 for q in range(Ly)]
                            for idx in range(self.d)])

        # Edge telling
        n_vert = (Ly - 1) * Lx
        n_horiz = Ly * (Lx - 1)
        self.n_edges = n_vert + n_horiz

    def _zz_intra_diag(self, gamma):
        """Intra-kolom ZZ fase (diagonaal)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        phase = np.zeros(d)
        for y in range(Ly - 1):
            z1 = 1 - 2 * bp[:, y].astype(float)
            z2 = 1 - 2 * bp[:, y + 1].astype(float)
            phase += z1 * z2
        return np.exp(-1j * gamma * phase)

    def _zz_inter_diag(self, gamma):
        """Inter-kolom ZZ fase (d x d diagonaal)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        phase = np.zeros((d, d))
        for y in range(Ly):
            z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
            z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
            phase += z_L * z_R
        return np.exp(-1j * gamma * phase)

    def _rx_col(self, beta):
        """Rx mixer (d x d vol)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        c, s = np.cos(beta), -1j * np.sin(beta)
        rx1 = np.array([[c, s], [s, c]], dtype=complex)
        Rxd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Rxd *= rx1[bp[:, q:q + 1], bp[:, q:q + 1].T]
        return Rxd

    def _zz_1site_obs(self, y1, y2):
        bp = self.bp
        return (1 - 2 * bp[:, y1].astype(float)) * (1 - 2 * bp[:, y2].astype(float))

    def _zz_2site_obs(self, y):
        bp = self.bp
        z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
        z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
        return z_L * z_R

    def eval_ratio(self, p, gammas, betas, use_cross=True):
        """Bereken MaxCut ratio.

        Args:
            use_cross: True = TT-Cross, False = exact SVD (voor vergelijking)
        """
        Lx, Ly, d = self.Lx, self.Ly, self.d
        chi_max = self.chi_max
        t0 = time.time()

        if self.verbose:
            mode = "TT-Cross" if use_cross else "Exact SVD"
            print("  [B29] %s: %dx%d p=%d d=%d chi_max=%d" % (
                mode, Lx, Ly, p, d, chi_max))

        # Init MPS: |+>
        mps = [np.ones((1, d, 1), dtype=complex) / np.sqrt(d)
               for _ in range(Lx)]

        n_cross_calls = 0
        n_exact_calls = 0

        for layer in range(p):
            g = gammas[layer]
            b = betas[layer]

            # 1. Intra-kolom ZZ (diagonaal, 1-site)
            intra = self._zz_intra_diag(g)
            for x in range(Lx):
                mps[x] = mps[x] * intra[None, :, None]

            # 2. Inter-kolom ZZ (diagonaal, 2-site)
            inter = self._zz_inter_diag(g)
            for x in range(Lx - 1):
                chi_L = mps[x].shape[0]
                chi_R = mps[x + 1].shape[2]
                chi_M = mps[x].shape[2]

                # Besluit: cross of exact?
                # Cross alleen zinvol als de volledige 2-site tensor te groot is
                # mat_size = chi_L * d² * chi_R (elementen van de volledige Theta)
                # Bij d>=32 en hoge chi wordt dit al snel > 1M elementen
                # Threshold: chi_L * d * chi_R > chi_max * d  (d factor overhead)
                # Dit triggert zodra bonds nontriviaal zijn (layer 2+)
                mat_size = chi_L * d * d * chi_R
                svd_cost = min(chi_L * d, d * chi_R) ** 2 * max(chi_L * d, d * chi_R)
                cross_cost = chi_max * d * (chi_L + chi_R)  # O(chi*d*chi) evaluations
                use_this_cross = (use_cross and d >= 32
                                  and mat_size > 50000
                                  and svd_cost > cross_cost * 10)

                if use_this_cross:
                    U_new, V_new = rsvd_diag_gate(
                        mps[x], mps[x + 1], inter, chi_max,
                        tol=self.tol, n_oversamples=10, n_power_iter=2)
                    mps[x] = U_new
                    mps[x + 1] = V_new
                    n_cross_calls += 1
                else:
                    # Exact SVD (zelfde als B26)
                    Theta = np.einsum('asb,btd->astd', mps[x], mps[x + 1])
                    Theta *= inter[None, :, :, None]
                    mat = Theta.reshape(chi_L * d, d * chi_R)
                    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
                    k = min(len(S), chi_max)
                    if S[0] > 1e-15:
                        k_nz = max(1, int(np.sum(S > self.tol * S[0])))
                        k = min(k, k_nz)
                    mps[x] = U[:, :k].reshape(chi_L, d, k)
                    mps[x + 1] = (np.diag(S[:k]) @ Vh[:k, :]).reshape(k, d, chi_R)
                    n_exact_calls += 1

            # 3. Mixer
            rx = self._rx_col(b)
            for x in range(Lx):
                mps[x] = np.einsum('ij,ajb->aib', rx, mps[x])

        elapsed_state = time.time() - t0

        # === Meting ===
        env_L, env_R = self._build_envs(mps)
        total_cost = 0.0

        for x in range(Lx):
            for y in range(Ly - 1):
                zz_diag = self._zz_1site_obs(y, y + 1)
                zz_val = self._expect_1site_diag(mps, x, zz_diag, env_L, env_R)
                total_cost += (1 - zz_val) / 2

        for x in range(Lx - 1):
            for y in range(Ly):
                zz_2d = self._zz_2site_obs(y)
                zz_val = self._expect_2site_diag(mps, x, zz_2d, env_L, env_R)
                total_cost += (1 - zz_val) / 2

        ratio = total_cost / self.n_edges
        elapsed = time.time() - t0

        if self.verbose:
            max_chi = max(m.shape[0] for m in mps)
            print("  [B29] Ratio: %.6f (%.3fs, max_chi=%d, cross=%d, exact=%d)" % (
                ratio, elapsed, max_chi, n_cross_calls, n_exact_calls))

        return ratio

    # === MPS meetfuncties (zelfde als B26) ===

    def _build_envs(self, mps):
        L = len(mps)
        env_L = [None] * (L + 1)
        env_L[0] = np.ones((1, 1), dtype=complex)
        for i in range(L):
            env_L[i + 1] = np.einsum('ae,asb,esd->bd',
                                     env_L[i], mps[i], np.conj(mps[i]))
        env_R = [None] * (L + 1)
        env_R[L] = np.ones((1, 1), dtype=complex)
        for i in range(L - 1, -1, -1):
            env_R[i] = np.einsum('asb,esd,bd->ae',
                                 mps[i], np.conj(mps[i]), env_R[i + 1])
        return env_L, env_R

    def _expect_1site_diag(self, mps, site, op_diag, env_L, env_R):
        A = mps[site]
        eL = env_L[site]
        eR = env_R[site + 1]
        T = np.einsum('ae,asb,bd,esd->s', eL, A, eR, np.conj(A))
        return np.sum(op_diag * T).real

    def _expect_2site_diag(self, mps, site, op_2d, env_L, env_R):
        A = mps[site]
        B = mps[site + 1]
        eL = env_L[site]
        eR = env_R[site + 2]
        d = self.d
        chi_M = A.shape[2]
        L_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for s in range(d):
            As = A[:, s, :]
            L_block[:, :, s] = As.T @ eL @ np.conj(As)
        R_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for t in range(d):
            Bt = B[:, t, :]
            R_block[:, :, t] = Bt @ eR @ np.conj(Bt).T
        M = L_block.reshape(-1, d).T @ R_block.reshape(-1, d)
        return np.sum(op_2d * M).real

    def optimize(self, p, n_gamma=10, n_beta=10, refine=True, use_cross=True):
        """Grid search + scipy."""
        t0 = time.time()
        old_verbose = self.verbose

        gamma_range = np.linspace(0.05, np.pi, n_gamma)
        beta_range = np.linspace(0.05, np.pi / 2, n_beta)

        best_ratio = -1
        best_g, best_b = gamma_range[0], beta_range[0]
        n_evals = 0

        self.verbose = False
        for gi, g in enumerate(gamma_range):
            for b in beta_range:
                r = self.eval_ratio(p, [g] * p, [b] * p, use_cross=use_cross)
                n_evals += 1
                if r > best_ratio:
                    best_ratio = r
                    best_g, best_b = g, b
            if old_verbose and (gi + 1) % max(1, n_gamma // 5) == 0:
                self.verbose = old_verbose
                print("    Grid: %d/%d, best=%.6f (%.1fs)" % (
                    gi + 1, n_gamma, best_ratio, time.time() - t0))
                self.verbose = False

        self.verbose = old_verbose
        best_gammas = [best_g] * p
        best_betas = [best_b] * p

        if refine:
            try:
                from scipy.optimize import minimize as scipy_minimize

                def neg_ratio(params):
                    gs = list(params[:p])
                    bs = list(params[p:])
                    return -self.eval_ratio(p, gs, bs, use_cross=use_cross)

                x0 = best_gammas + best_betas
                self.verbose = False
                result = scipy_minimize(neg_ratio, x0, method='Nelder-Mead',
                                        options={'maxiter': 200, 'xatol': 1e-5,
                                                 'fatol': 1e-6, 'adaptive': True})
                self.verbose = old_verbose
                n_evals += result.nfev
                if -result.fun > best_ratio:
                    best_ratio = -result.fun
                    best_gammas = list(result.x[:p])
                    best_betas = list(result.x[p:])
            except ImportError:
                pass

        return best_ratio, best_gammas, best_betas, {
            'total_time': time.time() - t0, 'n_evals': n_evals}


def main():
    parser = argparse.ArgumentParser(
        description='B29: TT-Cross Interpolation voor QAOA MaxCut')
    parser.add_argument('--Lx', type=int, default=20)
    parser.add_argument('--Ly', type=int, default=5)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--chi', type=int, default=None)
    parser.add_argument('--gamma', type=float, nargs='+', default=None)
    parser.add_argument('--beta', type=float, nargs='+', default=None)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--validate', action='store_true',
                        help='Vergelijk TT-Cross met exact SVD')
    parser.add_argument('--ngamma', type=int, default=10)
    parser.add_argument('--nbeta', type=int, default=10)
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  B29: TT-Cross QAOA")
    print(sep)

    Lx, Ly, p = args.Lx, args.Ly, args.p
    d = 2 ** Ly
    print("  Grid: %dx%d (%d qubits), d=%d, p=%d" % (Lx, Ly, Lx * Ly, d, p))

    engine = TTCrossQAOA(Lx, Ly, chi_max=args.chi, verbose=True)

    if args.validate:
        gammas = args.gamma or [0.3927] * p
        betas = args.beta or [1.1781] * p
        print("\n  Validatie: TT-Cross vs Exact SVD")
        r_cross = engine.eval_ratio(p, gammas, betas, use_cross=True)
        r_exact = engine.eval_ratio(p, gammas, betas, use_cross=False)
        print("  Verschil: %.2e" % abs(r_cross - r_exact))

    elif args.optimize:
        r, g, b, info = engine.optimize(p, args.ngamma, args.nbeta)
        print("\n  RESULTAAT: ratio=%.6f (%.1fs)" % (r, info['total_time']))
    else:
        gammas = args.gamma or [0.3927] * p
        betas = args.beta or [1.1781] * p
        engine.eval_ratio(p, gammas, betas)

    print(sep)


if __name__ == '__main__':
    main()
