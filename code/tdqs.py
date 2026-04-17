#\!/usr/bin/env python3
"""
tdqs.py - B41: Triage-Driven Quantum Solver (TDQS) v2

Chi-Aware Gate Selection met:
  - L-BFGS-B optimizer (vervangt grid search)
  - Joint re-optimalisatie van alle lagen
  - Per-bond chi tracking
  - Multi-angle: aparte gamma_intra, gamma_inter per laag

Architectuur:
  MPS op cilindrisch grid Lx x Ly (PBC in y).
  Kolom = 2^Ly dimensionale site.
  Gates: intra-column ZZ (1-site diag, chi-neutraal),
         inter-column ZZ (2-site diag, chi-duur),
         Rx mixer (1-site full, chi-neutraal).

Vergelijking met QAOA:
  - Zelfde probleem: MaxCut op grids
  - Verschil: TDQS selecteert gates adaptief + optimiseert ALLE lagen joint
  - Verwachting: beter bij laag chi-budget

Gebruik:
  python tdqs.py --Lx 4 --Ly 3 --chi 16
  python tdqs.py --Lx 8 --Ly 3 --chi 16 --compare
  python tdqs.py --Lx 4 --Ly 3 --chi 16 --mode full
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TDQS:
    """Triage-Driven Quantum Solver v2.

    Verbeteringen t.o.v. v1:
      1. L-BFGS-B ipv grid search (10-100x betere convergentie)
      2. Joint re-optimalisatie van alle lagen na elke toevoeging
      3. Per-bond chi tracking voor slimmere triage
      4. Multi-angle: gamma_intra, gamma_inter, beta per laag
    """

    def __init__(self, Lx, Ly, chi_max=16, verbose=True, warm_angles=None):
        self.Lx = Lx
        self.Ly = Ly
        self.d = 2 ** Ly
        self.chi_max = chi_max
        self.verbose = verbose
        self.warm_angles = warm_angles  # (Lx, Ly) array of θ or None

        # Bit-patronen voor kolom-configuraties
        self.bp = np.array([[(idx >> (Ly - 1 - q)) & 1 for q in range(Ly)]
                            for idx in range(self.d)])

        # Edges opsommen
        self.edges_intra = []  # (x, y1, y2) -- verticaal binnen kolom
        self.edges_inter = []  # (x, y) -- horizontaal tussen kolom x en x+1
        for x in range(Lx):
            for y in range(Ly - 1):
                self.edges_intra.append((x, y, y + 1))
        for x in range(Lx - 1):
            for y in range(Ly):
                self.edges_inter.append((x, y))
        self.n_edges = len(self.edges_intra) + len(self.edges_inter)

    # =================================================================
    # Gate constructors
    # =================================================================

    def _zz_intra_diag(self, gamma, y1, y2):
        """exp(-i*gamma * Z_{y1} Z_{y2}) diagonaal d-vector."""
        bp = self.bp
        z1 = 1 - 2 * bp[:, y1].astype(float)
        z2 = 1 - 2 * bp[:, y2].astype(float)
        return np.exp(-1j * gamma * z1 * z2)

    def _zz_inter_diag(self, gamma, y):
        """exp(-i*gamma * Z_y^L Z_y^R) diagonaal d x d."""
        bp = self.bp
        z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
        z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
        return np.exp(-1j * gamma * z_L * z_R)

    def _zz_inter_batched(self, gamma, y_list):
        """Gebatchte inter-column ZZ gate voor meerdere qubits tegelijk."""
        d, bp = self.d, self.bp
        phase = np.zeros((d, d))
        for y in y_list:
            z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
            z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
            phase += z_L * z_R
        return np.exp(-1j * gamma * phase)

    def _rx_col_full(self, beta):
        """Rx(2*beta) op alle qubits tegelijk. d x d matrix."""
        d, Ly, bp = self.d, self.Ly, self.bp
        c, s = np.cos(beta), -1j * np.sin(beta)
        rx1 = np.array([[c, s], [s, c]], dtype=complex)
        M = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            M *= rx1[bp[:, q:q + 1], bp[:, q:q + 1].T]
        return M

    # =================================================================
    # MPS operaties
    # =================================================================

    def _init_mps(self):
        """Initialiseer MPS als |+>^N of warm-start (B69)."""
        if self.warm_angles is not None:
            from ws_qaoa import warm_start_mps
            return warm_start_mps(self.Lx, self.Ly, self.warm_angles)
        d = self.d
        return [np.ones((1, d, 1), dtype=complex) / np.sqrt(d)
                for _ in range(self.Lx)]

    def _apply_1site_diag(self, mps, x, gate_diag):
        mps[x] = mps[x] * gate_diag[None, :, None]

    def _apply_1site_full(self, mps, x, gate_mat):
        mps[x] = np.einsum('ij,ajb->aib', gate_mat, mps[x])

    def _apply_2site_diag(self, mps, x, gate_2d, chi_max):
        """2-site gate met SVD truncatie. Returns delta_chi."""
        d = self.d
        A, B = mps[x], mps[x + 1]
        chi_L, chi_R = A.shape[0], B.shape[2]
        chi_old = A.shape[2]

        Theta = np.einsum('asb,btd->astd', A, B)
        Theta *= gate_2d[None, :, :, None]

        mat = Theta.reshape(chi_L * d, d * chi_R)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        k = min(len(S), chi_max)
        if S[0] > 1e-15:
            k_nz = max(1, int(np.sum(S > 1e-14 * S[0])))
            k = min(k, k_nz)

        mps[x] = U[:, :k].reshape(chi_L, d, k)
        mps[x + 1] = (np.diag(S[:k]) @ Vh[:k, :]).reshape(k, d, chi_R)
        return max(0, k - chi_old)

    def _max_chi(self, mps):
        return max(m.shape[0] for m in mps)

    def _bond_chis(self, mps):
        """Per-bond chi vector. bond[i] = chi between site i and i+1."""
        return [mps[i].shape[2] for i in range(len(mps) - 1)]

    def _copy_mps(self, mps):
        return [m.copy() for m in mps]

    # =================================================================
    # Energie meting
    # =================================================================

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
        T = np.einsum('ae,asb,bd,esd->s', env_L[site], A,
                       env_R[site + 1], np.conj(A))
        return np.sum(op_diag * T).real

    def _expect_2site_diag(self, mps, site, op_2d, env_L, env_R):
        d = self.d
        A, B = mps[site], mps[site + 1]
        chi_M = A.shape[2]
        L_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for s in range(d):
            As = A[:, s, :]
            L_block[:, :, s] = As.T @ env_L[site] @ np.conj(As)
        R_block = np.zeros((chi_M, chi_M, d), dtype=complex)
        for t in range(d):
            Bt = B[:, t, :]
            R_block[:, :, t] = Bt @ env_R[site + 2] @ np.conj(Bt).T
        M = L_block.reshape(-1, d).T @ R_block.reshape(-1, d)
        return np.sum(op_2d * M).real

    def _measure_cost(self, mps):
        """Totale MaxCut cost = sum_edges (1 - <ZZ>)/2."""
        env_L, env_R = self._build_envs(mps)
        bp = self.bp
        total_cost = 0.0
        for (x, y1, y2) in self.edges_intra:
            z1 = 1 - 2 * bp[:, y1].astype(float)
            z2 = 1 - 2 * bp[:, y2].astype(float)
            zz_val = self._expect_1site_diag(mps, x, z1 * z2, env_L, env_R)
            total_cost += (1 - zz_val) / 2
        for (x, y) in self.edges_inter:
            z_L = (1 - 2 * bp[:, y].astype(float))[:, None]
            z_R = (1 - 2 * bp[:, y].astype(float))[None, :]
            zz_val = self._expect_2site_diag(mps, x, z_L * z_R, env_L, env_R)
            total_cost += (1 - zz_val) / 2
        return total_cost

    def _measure_ratio(self, mps):
        return self._measure_cost(mps) / self.n_edges

    # =================================================================
    # Layer application (multi-angle)
    # =================================================================

    def _apply_layer_multi(self, mps, gamma_intra, gamma_inter, beta, zz_set):
        """Pas een laag toe met aparte gamma voor intra en inter gates.

        Args:
            mps: MPS (in-place gewijzigd)
            gamma_intra: hoek voor intra-column ZZ
            gamma_inter: hoek voor inter-column ZZ
            beta: hoek voor Rx mixer
            zz_set: set van (type, info) tuples

        Returns:
            max delta_chi
        """
        max_delta_chi = 0

        # 1. Intra-column ZZ (gratis)
        for (gtype, ginfo) in zz_set:
            if gtype == 'intra':
                x, y1, y2 = ginfo
                gate = self._zz_intra_diag(gamma_intra, y1, y2)
                self._apply_1site_diag(mps, x, gate)

        # 2. Inter-column ZZ: batch per kolom-paar
        inter_by_col = {}
        for (gtype, ginfo) in zz_set:
            if gtype == 'inter':
                x, y = ginfo
                if x not in inter_by_col:
                    inter_by_col[x] = []
                inter_by_col[x].append(y)

        for x in sorted(inter_by_col.keys()):
            gate = self._zz_inter_batched(gamma_inter, inter_by_col[x])
            dc = self._apply_2site_diag(mps, x, gate, self.chi_max)
            max_delta_chi = max(max_delta_chi, dc)

        # 3. Mixer
        rx = self._rx_col_full(beta)
        for x in range(self.Lx):
            self._apply_1site_full(mps, x, rx)

        return max_delta_chi

    def _apply_layer(self, mps, gamma, beta, zz_set):
        """Backwards-compatible: single gamma for all ZZ."""
        return self._apply_layer_multi(mps, gamma, gamma, beta, zz_set)

    # =================================================================
    # Full circuit builder (for joint optimization)
    # =================================================================

    def _build_and_eval(self, params, layer_sets, return_mps=False):
        """Bouw circuit vanuit params en evalueer ratio.

        params: [g_intra_1, g_inter_1, beta_1, g_intra_2, g_inter_2, beta_2, ...]
        layer_sets: list of zz_set per laag

        Returns: ratio (float), optioneel mps
        """
        n_layers = len(layer_sets)
        assert len(params) == 3 * n_layers

        mps = self._init_mps()
        for i in range(n_layers):
            gi = params[3 * i]
            gx = params[3 * i + 1]
            b = params[3 * i + 2]
            self._apply_layer_multi(mps, gi, gx, b, layer_sets[i])

        ratio = self._measure_ratio(mps)
        if return_mps:
            return ratio, mps
        return ratio

    def _neg_ratio(self, params, layer_sets):
        """Negatieve ratio voor minimalisatie."""
        return -self._build_and_eval(params, layer_sets)

    # =================================================================
    # L-BFGS-B optimizer
    # =================================================================

    def _optimize_new_layer(self, mps_before, zz_set, init_gi, init_gx, init_b,
                           max_iter=80):
        """Optimaliseer parameters van alleen de nieuwste laag.

        Veel sneller dan joint: slechts 3 parameters, geen rebuild.
        """
        from scipy.optimize import minimize as sp_min

        bounds = [(0.01, np.pi), (0.01, np.pi), (0.01, np.pi)]

        def neg_r(params):
            gi, gx, b = params
            trial = self._copy_mps(mps_before)
            self._apply_layer_multi(trial, gi, gx, b, zz_set)
            return -self._measure_ratio(trial)

        new_ratio = -1
        best_params = np.array([init_gi, init_gx, init_b])

        # Multi-start: init + 1 perturbation
        starts = [best_params.copy()]
        rng = np.random.RandomState(42)
        perturbed = best_params + rng.uniform(-0.5, 0.5, 3)
        for j in range(3):
            perturbed[j] = max(bounds[j][0], min(bounds[j][1], perturbed[j]))
        starts.append(perturbed)

        for x0 in starts:
            try:
                res = sp_min(neg_r, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'ftol': 1e-10})
                r = -res.fun
                if r > new_ratio:
                    new_ratio = r
                    best_params = res.x.copy()
            except Exception:
                pass

        return best_params, new_ratio

    def _joint_polish(self, layer_sets, init_params, max_iter=40):
        """Korte joint optimalisatie van alle lagen samen.

        Gebruikt voor fijnafstelling nadat elke laag individueel is geoptimaliseerd.
        """
        from scipy.optimize import minimize as sp_min

        n_layers = len(layer_sets)
        bounds = [(0.01, np.pi), (0.01, np.pi), (0.01, np.pi)] * n_layers
        init_params = np.array(init_params, dtype=float)

        try:
            res = sp_min(self._neg_ratio, init_params, args=(layer_sets,),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-10})
            r = -res.fun
            if r > self._build_and_eval(init_params, layer_sets):
                return res.x.copy(), r
        except Exception:
            pass
        return init_params.copy(), self._build_and_eval(init_params, layer_sets)

    def _quick_grid_search(self, mps, zz_set, n_g=None, n_b=None):
        """Snel rooster voor startwaarden van een nieuwe laag.

        Returns: (ratio, gamma_intra, gamma_inter, beta)
        """
        chi = self._max_chi(mps)
        if n_g is None:
            n_g = 8 if chi > 1 else 15
        if n_b is None:
            n_b = 6 if chi > 1 else 12
        g_vals = np.linspace(0.05, np.pi, n_g)
        b_vals = np.linspace(0.05, 1.5, n_b)

        best = (-1, 0.5, 0.5, 0.3)
        for g in g_vals:
            for b in b_vals:
                trial = self._copy_mps(mps)
                self._apply_layer(trial, g, b, zz_set)
                r = self._measure_ratio(trial)
                if r > best[0]:
                    best = (r, g, g, b)
        return best

    # =================================================================
    # Triage: chi-aware edge selection
    # =================================================================

    def _triage_select(self, mps, all_intra, all_inter, gamma_inter, beta):
        """Selecteer inter-edges op basis van per-bond chi impact.

        Strategie: verwijder inter-edges die:
          a) op al-verzadigde bonds zitten (chi == chi_max)
          b) weinig energie-winst geven t.o.v. chi-kost

        Returns: active_set (set van (type, info) tuples)
        """
        bond_chis = self._bond_chis(mps)
        full_set = set(all_intra + all_inter)

        # Altijd alle intra meenemen (gratis)
        active_set = set(all_intra)

        # Sorteer inter-edges op bond-chi (laagste eerst = meeste ruimte)
        inter_with_chi = []
        for e in all_inter:
            x = e[1][0]  # kolom index
            bc = bond_chis[x] if x < len(bond_chis) else self.chi_max
            inter_with_chi.append((bc, e))
        inter_with_chi.sort(key=lambda t: t[0])

        # Fase 1: voeg alle inter-edges toe waar chi nog ruimte heeft
        saturated = []
        for bc, e in inter_with_chi:
            if bc < self.chi_max * 0.9:  # 90% threshold
                active_set.add(e)
            else:
                saturated.append(e)

        # Fase 2: voor verzadigde bonds, test of toevoegen netto helpt
        if saturated:
            base_trial = self._copy_mps(mps)
            self._apply_layer(base_trial, gamma_inter, beta, active_set)
            base_ratio = self._measure_ratio(base_trial)

            for e in saturated:
                test_set = active_set | {e}
                trial = self._copy_mps(mps)
                self._apply_layer(trial, gamma_inter, beta, test_set)
                r = self._measure_ratio(trial)
                # Alleen toevoegen als winst > 0.05% (waard de chi-kost)
                if r > base_ratio + 0.0005:
                    active_set.add(e)

        return active_set

    # =================================================================
    # TDQS Solve
    # =================================================================

    def solve(self, n_layers=None, mode='triage'):
        """Voer TDQS v2 uit.

        Modes:
          'full':    Alle edges elke laag
          'triage':  Chi-aware edge-selectie per laag

        Algoritme:
          1. Begin met |+>
          2. Per laag:
             a. Quick grid search voor startwaarden nieuwe laag
             b. Triage edge-selectie (als mode='triage')
             c. L-BFGS-B optimaliseer ALLE lagen joint
          3. Stop als geen verbetering of chi-budget op

        Returns: dict met ratio, layers, timing info
        """
        if n_layers is None:
            n_layers = 10

        t0 = time.time()

        all_intra = [('intra', (x, y1, y2)) for (x, y1, y2) in self.edges_intra]
        all_inter = [('inter', (x, y)) for (x, y) in self.edges_inter]
        full_set = set(all_intra + all_inter)

        # Per-layer state
        layer_sets = []          # zz_set per laag
        layer_params = []        # [g_intra, g_inter, beta] per laag
        ratios = []
        chi_history = []

        # Initial state
        mps_init = self._init_mps()
        ratios.append(self._measure_ratio(mps_init))
        chi_history.append(1)

        if self.verbose:
            print(f"  [B41v2] TDQS: {self.Lx}x{self.Ly} chi_max={self.chi_max}")
            print(f"  [B41v2] Edges: {len(all_intra)} intra, {len(all_inter)} inter")
            print(f"  [B41v2] Start ratio: {ratios[0]:.6f}")

        for layer_idx in range(n_layers):
            # Bouw huidige MPS tot aan deze laag
            if layer_sets:
                flat_params = []
                for p in layer_params:
                    flat_params.extend(p)
                _, current_mps = self._build_and_eval(flat_params, layer_sets,
                                                       return_mps=True)
            else:
                current_mps = self._init_mps()

            current_ratio = self._measure_ratio(current_mps)
            current_chi = self._max_chi(current_mps)

            if current_chi >= self.chi_max:
                if self.verbose:
                    print(f"  [B41v2] Chi-budget op (chi={current_chi})")
                break

            # --- Stap a: Quick grid voor startwaarden ---
            grid_ratio, gi0, gx0, b0 = self._quick_grid_search(
                current_mps, full_set, n_g=8, n_b=6)

            if grid_ratio <= current_ratio + 1e-8:
                if self.verbose:
                    print(f"  [B41v2] Laag {layer_idx+1}: grid vindt geen verbetering, stop")
                break

            # --- Stap b: Triage edge selectie ---
            if mode == 'triage':
                active_set = self._triage_select(
                    current_mps, all_intra, all_inter, gx0, b0)
            else:
                active_set = full_set

            # --- Stap c: Optimaliseer nieuwe laag (3 params, snel) ---
            mi = 40 if current_chi > 1 else 80
            new_params, new_ratio = self._optimize_new_layer(
                current_mps, active_set, gi0, gx0, b0, max_iter=mi)

            # Check verbetering
            if new_ratio <= current_ratio + 1e-6:
                if self.verbose:
                    print(f"  [B41v2] Laag {layer_idx+1}: geen verbetering "
                          f"({new_ratio:.6f} vs {current_ratio:.6f}), stop")
                break

            # Accepteer laag
            layer_sets = layer_sets + [active_set]
            layer_params = layer_params + [list(new_params)]

            # Bouw MPS met huidige params
            flat_params = []
            for p in layer_params:
                flat_params.extend(p)
            new_ratio, new_mps = self._build_and_eval(flat_params, layer_sets,
                                                       return_mps=True)

            # --- Stap d: Optionele joint polish (alleen als snel genoeg) ---
            do_polish = len(layer_sets) >= 2 and self._max_chi(new_mps) < self.chi_max * 0.7
            if do_polish:
                flat_params = []
                for p in layer_params:
                    flat_params.extend(p)
                polished, pol_ratio = self._joint_polish(
                    layer_sets, flat_params, max_iter=20)
                if pol_ratio > new_ratio + 1e-6:
                    new_ratio = pol_ratio
                    layer_params = []
                    for i in range(len(layer_sets)):
                        layer_params.append([polished[3*i], polished[3*i+1], polished[3*i+2]])
                    if self.verbose:
                        print(f"  [B41v2]   Joint polish: -> {pol_ratio:.6f}")

            # Bouw MPS voor chi tracking
            flat_params = []
            for p in layer_params:
                flat_params.extend(p)
            _, new_mps = self._build_and_eval(flat_params, layer_sets,
                                               return_mps=True)

            n_inter = sum(1 for g, _ in active_set if g == 'inter')
            ratios.append(new_ratio)
            chi_history.append(self._max_chi(new_mps))

            if self.verbose:
                gi, gx, b = layer_params[-1]
                bond_info = self._bond_chis(new_mps)
                chi_str = '/'.join(str(c) for c in bond_info)
                print(f"  [B41v2] Laag {layer_idx+1}: "
                      f"g_i={gi:.3f} g_x={gx:.3f} b={b:.3f} "
                      f"inter={n_inter}/{len(all_inter)} "
                      f"ratio={new_ratio:.6f} chi=[{chi_str}]")

        elapsed = time.time() - t0

        # Final MPS
        if layer_sets:
            flat_params = []
            for p in layer_params:
                flat_params.extend(p)
            final_ratio, final_mps = self._build_and_eval(
                flat_params, layer_sets, return_mps=True)
        else:
            final_ratio = ratios[0]
            final_mps = self._init_mps()

        result = {
            'ratio': final_ratio,
            'layers': [{'gamma_intra': p[0], 'gamma_inter': p[1], 'beta': p[2],
                         'n_inter': sum(1 for g, _ in s if g == 'inter'),
                         'n_inter_total': len(all_inter)}
                        for p, s in zip(layer_params, layer_sets)],
            'n_layers': len(layer_sets),
            'ratios': ratios,
            'chi_history': chi_history,
            'max_chi': self._max_chi(final_mps) if layer_sets else 1,
            'bond_chis': self._bond_chis(final_mps) if layer_sets else [],
            'elapsed': elapsed,
        }

        if self.verbose:
            print(f"  [B41v2] Klaar: ratio={final_ratio:.6f} lagen={len(layer_sets)} "
                  f"chi={result['max_chi']} ({elapsed:.2f}s)")

        return result

    # =================================================================
    # Vergelijking met QAOA
    # =================================================================

    def compare_with_qaoa(self, p_max=3, n_gamma=10, n_beta=10):
        """Vergelijk TDQS v2 met QAOA."""
        from transverse_contraction import TransverseQAOA

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  TDQS v2 vs QAOA: {self.Lx}x{self.Ly} chi_max={self.chi_max}")
            print(f"{'='*60}\n")

        # TDQS
        tdqs_result = self.solve()

        # QAOA
        qaoa_results = []
        for p in range(1, p_max + 1):
            chi_exact = self.d ** p
            if chi_exact <= self.chi_max:
                tc = TransverseQAOA(self.Lx, self.Ly, verbose=False)
                opt = tc.optimize(p, n_gamma=n_gamma, n_beta=n_beta, refine=True)
                qaoa_results.append({'p': p, 'ratio': opt[0], 'chi': chi_exact})
            else:
                try:
                    from tt_cross_qaoa import TTCrossQAOA
                    tt = TTCrossQAOA(self.Lx, self.Ly, chi_max=self.chi_max,
                                     verbose=False)
                    opt = tt.optimize(p, n_gamma=n_gamma, n_beta=n_beta, refine=True)
                    qaoa_results.append({'p': p, 'ratio': opt[0], 'chi': self.chi_max})
                except Exception:
                    break

        # Rapportage
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Resultaten:")
            print(f"{'='*60}")
            print(f"  TDQS v2: ratio={tdqs_result['ratio']:.6f}  "
                  f"lagen={tdqs_result['n_layers']}  chi={tdqs_result['max_chi']}  "
                  f"({tdqs_result['elapsed']:.2f}s)")
            for qr in qaoa_results:
                delta = tdqs_result['ratio'] - qr['ratio']
                sign = '+' if delta > 0 else ''
                print(f"  QAOA-{qr['p']}:  ratio={qr['ratio']:.6f}  chi={qr['chi']}"
                      f"  (TDQS {sign}{delta:.6f})")
            print()

        return tdqs_result, qaoa_results


# =================================================================
# CLI
# =================================================================

def main():
    parser = argparse.ArgumentParser(description='B41: TDQS v2')
    parser.add_argument('--Lx', type=int, default=4)
    parser.add_argument('--Ly', type=int, default=3)
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--mode', choices=['full', 'triage'], default='triage')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    solver = TDQS(args.Lx, args.Ly, chi_max=args.chi, verbose=True)

    if args.compare:
        solver.compare_with_qaoa()
    else:
        result = solver.solve(n_layers=args.layers, mode=args.mode)
        print(f"\nFinale ratio: {result['ratio']:.6f}")


if __name__ == '__main__':
    main()
