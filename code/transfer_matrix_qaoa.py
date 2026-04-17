#!/usr/bin/env python3
"""
transfer_matrix_qaoa.py - B40: QAOA MaxCut op oneindige cilinder/strip.

Bereken de QAOA MaxCut approximation ratio in de thermodynamische limiet
(N→∞) voor een oneindige cilinder met omtrek Ly.

Methode: MPS in het Schrödinger-beeld op een eindige strip van L kolommen,
lang genoeg zodat het midden het oneindige-systeem gedrag vertoont.
Column-grouping: elke site = Ly qubits, d = 2^Ly.

Voordelen t.o.v. lightcone (B21):
  - Geen randeffecten (bulk-meting in het midden)
  - Bonddimensie chi i.p.v. 2^n state vector
  - Ly=4 p=3: lightcone nodig 2^32 (onmogelijk), MPS: chi x 16 x chi (~MB)
  - Directe vergelijking met GW-bound voor oneindige systemen
  - Eén evaluatie → ratio voor N→∞ (geen stitching nodig)

Bouwt voort op: zorn_mps.py (gate-constructors) + lightcone_qaoa.py (optimizer)

Gebruik:
  python transfer_matrix_qaoa.py --Ly 4 --p 1
  python transfer_matrix_qaoa.py --Ly 4 --p 2 --chi 64
  python transfer_matrix_qaoa.py --Ly 4 --p 3 --chi 128 --progressive
  python transfer_matrix_qaoa.py --Ly 1 --p 1              # 1D: moet 0.75 geven
  python transfer_matrix_qaoa.py --Ly 4 --p 1 --obc-y      # strip (geen PBC in y)
"""

import numpy as np
import time
import argparse


class InfiniteCylinderQAOA:
    """QAOA MaxCut op oneindige cilinder via MPS (Schrödinger-beeld).

    Simuleert een eindige strip van L kolommen. Het midden geeft
    de verwachtingswaarden voor het oneindige systeem (tot op
    exponentieel kleine randcorrecties).

    Elke 'site' in de MPS vertegenwoordigt één kolom van Ly qubits.
    Lokale Hilbert-ruimte: d = 2^Ly.
    """

    def __init__(self, Ly, max_chi=64, pbc_y=True, verbose=True):
        self.Ly = Ly
        self.d = 2 ** Ly
        self.max_chi = max_chi
        self.pbc_y = pbc_y
        self.verbose = verbose

        # Bit-patronen: bp[s, q] = qubit q van configuratie s
        self.bp = np.array([[(idx >> (Ly - 1 - q)) & 1 for q in range(Ly)]
                            for idx in range(self.d)])

        # Aantal verticale edges per kolom
        if pbc_y and Ly >= 3:
            self.n_vert = Ly
        elif Ly >= 2:
            self.n_vert = Ly - 1
        else:
            self.n_vert = 0

        # Totaal edges per eenheidscel (1 kolom + 1 horizontale bond)
        self.n_edges_per_cell = self.n_vert + Ly

    # =================================================================
    # Gate constructors (zelfde fysica als HeisenbergQAOA in zorn_mps.py)
    # =================================================================

    def _build_hadamard(self):
        """H^{⊗Ly} matrix (d × d)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        Hd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Hd *= H1[bp[:, q:q + 1], bp[:, q:q + 1].T]
        return Hd

    def _zz_intra_diag(self, gamma):
        """Diagonale fase voor verticale edges (binnen kolom).

        Bevat PBC wrap-around (y=Ly-1 → y=0) als pbc_y=True en Ly>=3.
        Retourneert d-vector.
        """
        d, Ly, bp = self.d, self.Ly, self.bp
        diag = np.ones(d, dtype=complex)
        for y in range(Ly - 1):
            z1 = 1 - 2 * bp[:, y].astype(float)
            z2 = 1 - 2 * bp[:, y + 1].astype(float)
            diag *= np.exp(-1j * gamma * z1 * z2)
        if self.pbc_y and Ly >= 3:
            z1 = 1 - 2 * bp[:, Ly - 1].astype(float)
            z2 = 1 - 2 * bp[:, 0].astype(float)
            diag *= np.exp(-1j * gamma * z1 * z2)
        return diag

    def _zz_inter_diag(self, gamma):
        """Diagonale fase voor horizontale edges (tussen kolommen).

        Retourneert d²-vector: G[sL * d + sR].
        """
        d, Ly, bp = self.d, self.Ly, self.bp
        iL = np.arange(d * d) // d
        iR = np.arange(d * d) % d
        diag = np.ones(d * d, dtype=complex)
        for y in range(Ly):
            z1 = 1 - 2 * bp[iL, y].astype(float)
            z2 = 1 - 2 * bp[iR, y].astype(float)
            diag *= np.exp(-1j * gamma * z1 * z2)
        return diag

    def _rx_col(self, beta):
        """Rx(2β)^{⊗Ly} mixer matrix (d × d)."""
        d, Ly, bp = self.d, self.Ly, self.bp
        c, s = np.cos(beta), np.sin(beta)
        rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        Rxd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Rxd *= rx[bp[:, q:q + 1], bp[:, q:q + 1].T]
        return Rxd

    # =================================================================
    # MPS operaties (Schrödinger-beeld)
    # =================================================================

    def _init_state(self, L):
        """Initialiseer |+⟩^{⊗(L·Ly)} als product-state MPS.

        |+⟩ = H|0⟩, dus na Hadamard: uniforme superpositie.
        Elke site: (1/√d) · [1, 1, ..., 1], bond dimensie 1.
        """
        return [np.ones((1, self.d, 1), dtype=complex) / np.sqrt(self.d)
                for _ in range(L)]

    def _apply_1site(self, mps, site, U):
        """Pas d×d matrix U toe op site: A'[α,s',β] = Σ_s U[s',s] A[α,s,β]."""
        mps[site] = np.einsum('ij,ajb->aib', U, mps[site])

    def _apply_1site_diag(self, mps, site, diag):
        """Pas diagonale 1-site gate toe: A'[α,s,β] = diag[s] · A[α,s,β]."""
        mps[site] = mps[site] * diag[None, :, None]

    def _apply_2site_diag(self, mps, site, diag_dd):
        """Pas diagonale 2-site gate toe met SVD-truncatie.

        Gate: G[sL, sR] diagonaal in de computationele basis.
        SVD split + truncatie tot max_chi.
        Links→rechts sweep houdt links-canonieke vorm.
        """
        d = self.d
        A = mps[site]          # (chi_L, d, chi_M)
        B = mps[site + 1]      # (chi_M, d, chi_R)
        chi_L, chi_R = A.shape[0], B.shape[2]

        # Contractie: Θ[α, sL, sR, δ] = A[α,sL,β] B[β,sR,δ]
        Theta = np.einsum('asb,btd->astd', A, B)

        # Gate toepassen
        Theta *= diag_dd.reshape(1, d, d, 1)

        # SVD
        mat = Theta.reshape(chi_L * d, d * chi_R)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncatie
        k = min(len(S), self.max_chi)
        if S[0] > 1e-15:
            k_nz = max(1, int(np.sum(S > 1e-12 * S[0])))
            k = min(k, k_nz)

        # Split: links-canoniek (U isometrisch)
        mps[site] = U[:, :k].reshape(chi_L, d, k)
        mps[site + 1] = (np.diag(S[:k]) @ Vh[:k, :]).reshape(k, d, chi_R)

    # =================================================================
    # Omgevingen en verwachtingswaarden
    # =================================================================

    def _build_envs(self, mps):
        """Bouw linker- en rechter-omgevingen voor alle sites.

        env_L[i]: contractie van sites 0, ..., i-1. Shape (chi_i, chi_i).
        env_R[i]: contractie van sites i, ..., L-1. Shape (chi_i, chi_i).

        env_L[0] = env_R[L] = [[1]] (rand).
        """
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
        """⟨O⟩ voor diagonale 1-site operator op gegeven site.

        op_diag: d-vector met O[s] = eigenwaarde van operator voor basis |s⟩.
        """
        A = mps[site]       # (chi_L, d, chi_R)
        eL = env_L[site]    # (chi_L, chi_L)
        eR = env_R[site + 1]  # (chi_R, chi_R)

        # T[s] = Tr(eL @ A[:,s,:] @ eR @ conj(A[:,s,:])^T)
        # Gevectoriseerd: T[s] = einsum('ae,asb,bd,esd->', eL, A, eR, conj(A))
        T = np.einsum('ae,asb,bd,esd->s', eL, A, eR, np.conj(A))
        return np.sum(op_diag * T).real

    def _expect_2site_diag(self, mps, site, op_diag_2d, env_L, env_R):
        """⟨O⟩ voor diagonale 2-site operator op sites (site, site+1).

        op_diag_2d: (d, d) array met O[sL, sR].

        Efficiënte contractie in drie stappen:
        1. Links-blok L[b,f,s] via site-tensor A
        2. Rechts-blok R[b,f,t] via site-tensor B
        3. Combineer: result = Σ_{s,t} O[s,t] · Σ_{b,f} L[b,f,s] · R[b,f,t]
        """
        A = mps[site]           # (chi_L, d, chi_M)
        B = mps[site + 1]       # (chi_M, d, chi_R)
        eL = env_L[site]        # (chi_L, chi_L)
        eR = env_R[site + 2]    # (chi_R, chi_R)
        d = self.d
        chi_M = A.shape[2]

        # Stap 1: Links-blok L[b,f,s] = Σ_{a,e} eL[a,e] A[a,s,b] A*[e,s,f]
        # Per s: L_s = A[:,s,:]^T @ eL @ conj(A[:,s,:])   (chi_M × chi_M)
        L_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for s in range(d):
            As = A[:, s, :]   # (chi_L, chi_M)
            L_block[:, :, s] = As.T @ eL @ np.conj(As)

        # Stap 2: Rechts-blok R[b,f,t] = Σ_{g,h} B[b,t,g] eR[g,h] B*[f,t,h]
        # Per t: R_t = B[:,t,:] @ eR @ conj(B[:,t,:])^T   (chi_M × chi_M)
        R_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for t in range(d):
            Bt = B[:, t, :]   # (chi_M, chi_R)
            R_block[:, :, t] = Bt @ eR @ np.conj(Bt).T

        # Stap 3: M[s,t] = Σ_{b,f} L[b,f,s] · R[b,f,t]
        L_flat = L_block.reshape(-1, d)   # (chi_M², d)
        R_flat = R_block.reshape(-1, d)   # (chi_M², d)
        M = L_flat.T @ R_flat             # (d, d)

        return np.sum(op_diag_2d * M).real

    # =================================================================
    # QAOA evaluatie
    # =================================================================

    def _strip_length(self, p):
        """Minimale striplengte voor QAOA diepte p.

        Lichtkegel reikt 2p kolommen. L = 4p+4 geeft ruim marge.
        """
        return max(4 * p + 4, 12)

    def eval_state(self, p, gammas, betas, L=None):
        """Evolueer QAOA-toestand voor p lagen. Retourneert MPS.

        Begint met |+⟩ (uniforme superpositie = post-Hadamard toestand).
        Elke laag: intra-ZZ → inter-ZZ (met SVD-truncatie) → Rx-mixer.
        """
        if L is None:
            L = self._strip_length(p)

        mps = self._init_state(L)

        for layer in range(p):
            g, b = gammas[layer], betas[layer]

            # 1. Fase-separator: intra-kolom ZZ (1-site diagonaal, geen chi-groei)
            intra = self._zz_intra_diag(g)
            for x in range(L):
                self._apply_1site_diag(mps, x, intra)

            # 2. Fase-separator: inter-kolom ZZ (2-site diagonaal, chi groeit)
            inter = self._zz_inter_diag(g)
            for x in range(L - 1):
                self._apply_2site_diag(mps, x, inter)

            # 3. Mixer: Rx(2β)^⊗Ly (1-site vol, geen chi-groei)
            rx = self._rx_col(b)
            for x in range(L):
                self._apply_1site(mps, x, rx)

        return mps

    def eval_ratio(self, p, gammas, betas, L=None):
        """Bereken approximation ratio in de bulk (midden van strip).

        ratio = (cost_vert + cost_horiz) / n_edges_per_cell

        cost_vert  = Σ_y (1 - ⟨Z_y Z_{y+1}⟩) / 2  (verticale edges)
        cost_horiz = Σ_y (1 - ⟨Z_y^L Z_y^R⟩) / 2  (horizontale edges)
        """
        if L is None:
            L = self._strip_length(p)

        mps = self.eval_state(p, gammas, betas, L)
        env_L, env_R = self._build_envs(mps)
        norm = env_L[L][0, 0].real  # = ⟨ψ|ψ⟩

        mid = L // 2  # bulk kolom
        bp = self.bp

        # --- Verticale edges (intra-kolom) op site mid ---
        cost_vert = 0.0
        # OBC edges: (y, y+1) voor y = 0 .. Ly-2
        for y in range(self.Ly - 1):
            z1 = (1 - 2 * bp[:, y]).astype(float)
            z2 = (1 - 2 * bp[:, y + 1]).astype(float)
            zz = self._expect_1site_diag(mps, mid, z1 * z2, env_L, env_R) / norm
            cost_vert += (1 - zz) / 2

        # PBC wrap-around: (Ly-1, 0)
        if self.pbc_y and self.Ly >= 3:
            z1 = (1 - 2 * bp[:, self.Ly - 1]).astype(float)
            z2 = (1 - 2 * bp[:, 0]).astype(float)
            zz = self._expect_1site_diag(mps, mid, z1 * z2, env_L, env_R) / norm
            cost_vert += (1 - zz) / 2

        # --- Horizontale edges (inter-kolom) op bond (mid-1, mid) ---
        cost_horiz = 0.0
        for y in range(self.Ly):
            z_L = (1 - 2 * bp[:, y]).astype(float)
            z_R = (1 - 2 * bp[:, y]).astype(float)
            op_2d = np.outer(z_L, z_R)
            zz = self._expect_2site_diag(mps, mid - 1, op_2d,
                                         env_L, env_R) / norm
            cost_horiz += (1 - zz) / 2

        ratio = (cost_vert + cost_horiz) / self.n_edges_per_cell

        if self.verbose:
            max_chi = max(A.shape[0] for A in mps)
            print("  Vert cost: %.6f (%d edges)  Horiz cost: %.6f (%d edges)"
                  "  ratio=%.6f  chi_max=%d" %
                  (cost_vert, self.n_vert, cost_horiz, self.Ly,
                   ratio, max_chi))

        return ratio

    # =================================================================
    # Optimizer
    # =================================================================

    def optimize(self, p, n_gamma=10, n_beta=10, refine=True):
        """Grid search + scipy L-BFGS-B voor vaste p."""
        t0 = time.time()
        old_verbose = self.verbose

        gammas_grid = np.linspace(0.1, np.pi - 0.1, n_gamma)
        betas_grid = np.linspace(0.1, np.pi / 2 - 0.1, n_beta)

        best_ratio = -1
        best_gammas, best_betas = [np.pi / 4] * p, [np.pi / 8] * p
        n_evals = 0
        grid_ratio = -1

        if p == 1:
            if old_verbose:
                print("  Grid search: %d x %d = %d punten" %
                      (n_gamma, n_beta, n_gamma * n_beta))
            self.verbose = False
            for g in gammas_grid:
                for b in betas_grid:
                    r = self.eval_ratio(1, [g], [b])
                    n_evals += 1
                    if r > best_ratio:
                        best_ratio = r
                        best_gammas, best_betas = [g], [b]
            self.verbose = old_verbose
            grid_ratio = best_ratio
            grid_time = time.time() - t0
            if old_verbose:
                print("  Grid best: ratio=%.6f (gamma=%.4f, beta=%.4f) [%.1fs]" %
                      (best_ratio, best_gammas[0], best_betas[0], grid_time))
        else:
            grid_time = 0

        # Scipy verfijning
        if refine:
            self.verbose = False
            try:
                from scipy.optimize import minimize as sp_minimize

                def neg_ratio(params):
                    gs = list(params[:p])
                    bs = list(params[p:])
                    return -self.eval_ratio(p, gs, bs)

                x0 = np.array(best_gammas + best_betas)
                # Parameter symmetrie: beta in (0, pi/2)
                bounds = [(0.01, np.pi)] * p + [(0.01, np.pi / 2)] * p

                if old_verbose:
                    print("  Scipy L-BFGS-B start...")
                t1 = time.time()
                result = sp_minimize(neg_ratio, x0, method='L-BFGS-B',
                                     bounds=bounds,
                                     options={'maxiter': 100, 'ftol': 1e-8})
                n_evals += result.nfev

                if -result.fun > best_ratio:
                    best_ratio = -result.fun
                    best_gammas = list(result.x[:p])
                    best_betas = list(result.x[p:])

                if old_verbose:
                    print("  Scipy klaar: ratio=%.6f (+%.6f) [%.1fs, %d evals]" %
                          (best_ratio, best_ratio - max(grid_ratio, 0),
                           time.time() - t1, result.nfev))
            except ImportError:
                if old_verbose:
                    print("  [scipy niet beschikbaar, skip verfijning]")
            self.verbose = old_verbose

        total_time = time.time() - t0
        info = {
            'grid_ratio': grid_ratio,
            'grid_time': grid_time,
            'total_time': total_time,
            'n_evals': n_evals,
        }
        return best_ratio, best_gammas, best_betas, info

    # =================================================================
    # Warm-starting en progressieve optimizer
    # =================================================================

    @staticmethod
    def warmstart_params(gammas, betas, method='interp'):
        """Genereer startparameters voor p+1 vanuit optimale p-parameters.

        'interp': lineaire interpolatie (Zhou et al. 2020, adiabatisch pad)
        'append': voeg kleine perturbatie toe aan het eind
        """
        p = len(gammas)
        if method == 'interp' and p >= 2:
            old_x = np.linspace(0, 1, p)
            new_x = np.linspace(0, 1, p + 1)
            new_gammas = list(np.interp(new_x, old_x, gammas))
            new_betas = list(np.interp(new_x, old_x, betas))
        else:
            new_gammas = list(gammas) + [gammas[-1] * 0.5]
            new_betas = list(betas) + [betas[-1] * 0.5]
        return new_gammas, new_betas

    def optimize_progressive(self, p_max, n_gamma=10, n_beta=10,
                             refine=True, method='interp'):
        """Progressieve optimalisatie: p=1 → p=2 → ... → p_max.

        Elke stap warm-start vanuit de vorige optimale parameters.
        Grid search alleen bij p=1; hogere p starten direct bij scipy.
        """
        old_verbose = self.verbose
        results = {}

        if old_verbose:
            bc = "cilinder" if self.pbc_y else "strip"
            print("\n=== B40 Progressive: Ly=%d %s, chi=%d, %s ===" %
                  (self.Ly, bc, self.max_chi, method))

        # --- p=1: grid + scipy ---
        self.verbose = old_verbose
        ratio, gammas, betas, info = self.optimize(
            p=1, n_gamma=n_gamma, n_beta=n_beta, refine=refine)
        results[1] = {'ratio': ratio, 'gammas': gammas,
                       'betas': betas, 'info': info}
        if old_verbose:
            print("  => p=1 best: ratio=%.6f [%.1fs]\n" %
                  (ratio, info['total_time']))

        # --- p=2 .. p_max: warm-start + scipy ---
        for p in range(2, p_max + 1):
            t0 = time.time()
            if old_verbose:
                print("  p=%d: warm-start vanuit p=%d..." % (p, p - 1))

            prev_g = results[p - 1]['gammas']
            prev_b = results[p - 1]['betas']
            init_g, init_b = self.warmstart_params(prev_g, prev_b, method)

            if old_verbose:
                g_str = ", ".join("%.4f" % v for v in init_g)
                b_str = ", ".join("%.4f" % v for v in init_b)
                print("    Init: gammas=[%s]  betas=[%s]" % (g_str, b_str))

            self.verbose = False
            init_ratio = self.eval_ratio(p, init_g, init_b)
            n_evals = 1

            if old_verbose:
                print("    Init ratio: %.6f" % init_ratio)

            best_ratio = init_ratio
            best_gammas = list(init_g)
            best_betas = list(init_b)

            if refine:
                try:
                    from scipy.optimize import minimize as sp_minimize

                    def neg_ratio(params, _p=p):
                        gs = list(params[:_p])
                        bs = list(params[_p:])
                        return -self.eval_ratio(_p, gs, bs)

                    x0 = np.array(init_g + init_b)
                    bounds = [(0.01, np.pi)] * p + [(0.01, np.pi / 2)] * p

                    result = sp_minimize(neg_ratio, x0, method='L-BFGS-B',
                                         bounds=bounds,
                                         options={'maxiter': 100, 'ftol': 1e-8})
                    n_evals += result.nfev

                    if -result.fun > best_ratio:
                        best_ratio = -result.fun
                        best_gammas = list(result.x[:p])
                        best_betas = list(result.x[p:])
                except ImportError:
                    pass

            self.verbose = old_verbose
            total_time = time.time() - t0
            results[p] = {
                'ratio': best_ratio, 'gammas': best_gammas,
                'betas': best_betas,
                'info': {'total_time': total_time, 'n_evals': n_evals}
            }

            if old_verbose:
                print("  => p=%d best: ratio=%.6f [%.1fs, %d evals]\n" %
                      (p, best_ratio, total_time, n_evals))

        return results

    # =================================================================
    # Convergentie-check
    # =================================================================

    def check_convergence(self, p, gammas, betas, L_min=None, L_max=None):
        """Controleer of de striplengte voldoende is.

        Vergelijk ratio bij L en L+4. Als het verschil < 1e-6,
        is de strip lang genoeg.
        """
        if L_min is None:
            L_min = self._strip_length(p)
        if L_max is None:
            L_max = L_min + 8

        old_verbose = self.verbose
        self.verbose = False
        results = {}
        for L in range(L_min, L_max + 1, 4):
            r = self.eval_ratio(p, gammas, betas, L=L)
            results[L] = r

        self.verbose = old_verbose
        if old_verbose:
            print("  Convergentie-check:")
            for L, r in sorted(results.items()):
                print("    L=%d: ratio=%.8f" % (L, r))
            Ls = sorted(results.keys())
            if len(Ls) >= 2:
                diff = abs(results[Ls[-1]] - results[Ls[-2]])
                print("    Delta(L=%d->%d) = %.2e %s" %
                      (Ls[-2], Ls[-1], diff,
                       "(geconvergeerd)" if diff < 1e-6 else "(niet geconvergeerd)"))
        return results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B40: QAOA MaxCut op oneindige cilinder via MPS')
    parser.add_argument('--Ly', type=int, default=4,
                        help='Cilinder omtrek (default: 4)')
    parser.add_argument('--p', type=int, default=1,
                        help='QAOA diepte (default: 1)')
    parser.add_argument('--chi', type=int, default=64,
                        help='Max bonddimensie (default: 64)')
    parser.add_argument('--n-gamma', type=int, default=10,
                        help='Grid punten gamma (default: 10)')
    parser.add_argument('--n-beta', type=int, default=10,
                        help='Grid punten beta (default: 10)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip scipy verfijning')
    parser.add_argument('--obc-y', action='store_true',
                        help='Open BC in y (strip i.p.v. cilinder)')
    parser.add_argument('--progressive', action='store_true',
                        help='Progressieve optimizer: p=1->2->...->p')
    parser.add_argument('--warmstart-method', choices=['interp', 'append'],
                        default='interp', help='Warm-start methode')
    parser.add_argument('--check-conv', action='store_true',
                        help='Controleer convergentie t.o.v. striplengte')
    parser.add_argument('--L', type=int, default=None,
                        help='Override striplengte (default: auto)')
    args = parser.parse_args()

    pbc_y = not args.obc_y
    bc_str = "cilinder (PBC-y)" if pbc_y else "strip (OBC-y)"

    print("=" * 65)
    print("B40: QAOA MaxCut op oneindige %s" % bc_str)
    print("  Ly=%d  p=%d  max_chi=%d  d=%d" %
          (args.Ly, args.p, args.chi, 2 ** args.Ly))
    print("=" * 65)

    solver = InfiniteCylinderQAOA(
        Ly=args.Ly, max_chi=args.chi, pbc_y=pbc_y, verbose=True)

    t_start = time.time()

    if args.progressive:
        results = solver.optimize_progressive(
            p_max=args.p, n_gamma=args.n_gamma, n_beta=args.n_beta,
            refine=not args.no_refine, method=args.warmstart_method)

        # Samenvattingstabel
        print("\n" + "=" * 65)
        print("Samenvatting (Ly=%d %s chi=%d):" % (args.Ly, bc_str, args.chi))
        print("-" * 65)
        fmt = "  %d | %.6f | %-28s | %-28s | %.1fs"
        print("  p |  ratio   | gammas                       "
              "| betas                        | tijd")
        print("-" * 65)
        for p_level in sorted(results.keys()):
            r = results[p_level]
            g_str = "[" + ",".join("%.4f" % v for v in r['gammas']) + "]"
            b_str = "[" + ",".join("%.4f" % v for v in r['betas']) + "]"
            print(fmt % (p_level, r['ratio'], g_str, b_str,
                         r['info']['total_time']))
        print("=" * 65)

        # Convergentie-check op het beste resultaat
        if args.check_conv:
            best_p = max(results.keys())
            r = results[best_p]
            print("\nConvergentie-check voor p=%d:" % best_p)
            solver.check_convergence(best_p, r['gammas'], r['betas'])

    else:
        ratio, gammas, betas, info = solver.optimize(
            p=args.p, n_gamma=args.n_gamma, n_beta=args.n_beta,
            refine=not args.no_refine)

        print("\n" + "=" * 65)
        print("Resultaat (Ly=%d %s p=%d chi=%d):" %
              (args.Ly, bc_str, args.p, args.chi))
        print("  ratio (N->inf) = %.6f" % ratio)
        print("  gammas = %s" % gammas)
        print("  betas  = %s" % betas)
        print("  Tijd: %.1fs (%d evaluaties)" %
              (info['total_time'], info['n_evals']))
        print("=" * 65)

        if args.check_conv:
            print("\nConvergentie-check:")
            solver.check_convergence(args.p, gammas, betas)

    print("\nTotale tijd: %.1fs" % (time.time() - t_start))


if __name__ == '__main__':
    main()
