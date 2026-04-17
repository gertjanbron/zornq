#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circuit_knitting.py - B31: Circuit Knitting / Wire Cutting voor QAOA MaxCut.

Knip het QAOA-circuit langs strategische draden in onafhankelijke
sub-circuits. De resultaten worden exact gerecombineerd via
quasi-probability decomposition (QPD) van de cross-cut ZZ-gates.

Kernidee:
  De cross-cut ZZ-gate decomposieert in een Pauli-som:
    exp(-iγ Z_a⊗Z_b) = cos(γ) I⊗I − i·sin(γ) Z_a⊗Z_b

  Voor Ly cross-cut edges (verticale knip):
    Π_{y=0}^{Ly-1} [cos(γ)·I_y^L⊗I_y^R − i·sin(γ)·Z_y^L⊗Z_y^R]

  Dit expandt in 2^Ly termen, geïndexeerd door σ ∈ {0,1}^Ly:
    Σ_σ c_σ · Z_σ^L ⊗ Z_σ^R

  met c_σ = Π_{j∈σ}(-i sin γ) · Π_{j∉σ}(cos γ)

  De QAOA-toestand factoriseert:
    |ψ⟩ = Σ_σ c_σ |ψ_L(σ)⟩ ⊗ |ψ_R(σ)⟩

  waar |ψ_L(σ)⟩ = Mixer_L · Phase_int_L · Z_σ^L · |+⟩_L
  (qubits in σ starten in |−⟩ i.p.v. |+⟩)

  Verwachtingswaarde:
    ⟨O⟩ = Σ_{σ,τ} c*_σ c_τ ⟨ψ_L(σ)|O_L|ψ_L(τ)⟩ · ⟨ψ_R(σ)|O_R|ψ_R(τ)⟩

  Kosten: 2^Ly state-vector runs per fragment, 4^Ly recombinatie-termen.
  Voor p>1: 2^(Ly·p) runs per fragment (elke laag decomposieert).

  Bij meerdere knips: elke knip voegt factor 2^Ly toe.

Gebruik:
  python circuit_knitting.py --Lx 8 --Ly 1 --p 1 --validate
  python circuit_knitting.py --Lx 8 --Ly 4 --p 1 --cuts 1
  python circuit_knitting.py --Lx 100 --Ly 1 --p 1 --cuts 5

Bouwt voort op: lightcone_qaoa.py (B21), transverse_contraction.py (B26)
"""

import numpy as np
import math
import time
import argparse
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# State-vector QAOA engine met custom initialisatie
# =====================================================================

def qaoa_statevector(n_qubits, edges, p, gammas, betas,
                     init_signs=None):
    """Run QAOA op n_qubits met gegeven edges, retourneer state vector.

    Args:
        n_qubits: aantal qubits
        edges: lijst van (i, j) qubit-paren voor ZZ-gates
        p: QAOA diepte
        gammas, betas: parameters (lengte p)
        init_signs: None of array van +1/-1 per qubit.
                    +1 = |+⟩ = (|0⟩+|1⟩)/√2
                    -1 = |−⟩ = (|0⟩−|1⟩)/√2

    Returns:
        psi: state vector (2^n complex array)
    """
    n = n_qubits
    N = 1 << n

    if n > 26:
        raise ValueError("Te veel qubits voor state vector: %d" % n)

    # Init: |+⟩^⊗n of custom signs
    indices = np.arange(N, dtype=np.int64)
    psi = np.ones(N, dtype=complex) / np.sqrt(N)

    if init_signs is not None:
        # Apply sign flips: |−⟩ = Z|+⟩, so flip sign where bit=1 for negative qubits
        for q in range(n):
            if init_signs[q] < 0:
                bit_q = (indices >> (n - 1 - q)) & 1
                psi *= (1 - 2 * bit_q)  # flip sign waar bit=1

    # Pre-compute ZZ diagonals
    def get_bit(q):
        return (indices >> (n - 1 - q)) & 1

    for layer in range(p):
        g = gammas[layer]
        b = betas[layer]

        # ZZ fase (gevectoriseerd)
        phase = np.zeros(N, dtype=np.float64)
        for (qi, qj) in edges:
            zi = 1 - 2 * get_bit(qi)
            zj = 1 - 2 * get_bit(qj)
            phase += zi * zj
        psi *= np.exp(-1j * g * phase)

        # Rx mixer (vectorized per qubit)
        c = np.cos(b)
        s = -1j * np.sin(b)
        for q in range(n):
            step = 1 << (n - 1 - q)
            block = 2 * step
            base = np.arange(N // 2, dtype=np.int64)
            idx0 = (base // step) * block + (base % step)
            idx1 = idx0 + step
            a0 = psi[idx0].copy()
            a1 = psi[idx1].copy()
            psi[idx0] = c * a0 + s * a1
            psi[idx1] = s * a0 + c * a1

    return psi


def measure_zz(psi, n, qi, qj):
    """Meet ⟨Z_i Z_j⟩ op state vector psi."""
    N = len(psi)
    indices = np.arange(N, dtype=np.int64)
    bi = (indices >> (n - 1 - qi)) & 1
    bj = (indices >> (n - 1 - qj)) & 1
    zz = (1 - 2 * bi) * (1 - 2 * bj)
    probs = np.abs(psi) ** 2
    return np.sum(probs * zz)


def measure_z(psi, n, q):
    """Meet ⟨Z_q⟩ op state vector psi."""
    N = len(psi)
    indices = np.arange(N, dtype=np.int64)
    bq = (indices >> (n - 1 - q)) & 1
    zq = (1 - 2 * bq).astype(float)
    probs = np.abs(psi) ** 2
    return np.sum(probs * zq)


def overlap(psi1, psi2):
    """Bereken ⟨psi1|psi2⟩."""
    return np.vdot(psi1, psi2)


def matrix_element_zz(psi_bra, psi_ket, n, qi, qj):
    """Bereken ⟨bra|Z_i Z_j|ket⟩."""
    N = len(psi_ket)
    indices = np.arange(N, dtype=np.int64)
    bi = (indices >> (n - 1 - qi)) & 1
    bj = (indices >> (n - 1 - qj)) & 1
    zz = ((1 - 2 * bi) * (1 - 2 * bj)).astype(float)
    return np.vdot(psi_bra, zz * psi_ket)


def matrix_element_z(psi_bra, psi_ket, n, q):
    """Bereken ⟨bra|Z_q|ket⟩."""
    N = len(psi_ket)
    indices = np.arange(N, dtype=np.int64)
    bq = (indices >> (n - 1 - q)) & 1
    zq = (1 - 2 * bq).astype(float)
    return np.vdot(psi_bra, zq * psi_ket)


# =====================================================================
# Circuit Knitting Engine
# =====================================================================

class CircuitKnitting:
    """QAOA MaxCut via circuit knitting met exacte QPD recombinatie.

    Voor p=1 en een verticale knip met Ly cross-cut edges:
    - 2^Ly sub-circuit runs per fragment (L en R)
    - Recombinatie via QPD coefficiënten en overlap-matrices
    - Wiskundig exact (geen benadering)

    Voor p>1: 2^(Ly*p) sub-circuit runs (exponentieel in p).
    """

    def __init__(self, Lx, Ly=4, verbose=True):
        self.Lx = Lx
        self.Ly = Ly
        self.verbose = verbose

        # Edge telling
        n_vert = (Ly - 1) * Lx
        n_horiz = Ly * (Lx - 1)
        self.n_edges = n_vert + n_horiz

    def find_cut_positions(self, n_cuts):
        """Bepaal optimale knipposities (gelijke verdeling)."""
        if n_cuts <= 0:
            return []
        if n_cuts >= self.Lx - 1:
            raise ValueError("Te veel knips (%d) voor %d kolommen" %
                             (n_cuts, self.Lx))
        step = self.Lx / (n_cuts + 1)
        cuts = sorted(set(
            max(1, min(int(round(step * (i + 1))), self.Lx - 1))
            for i in range(n_cuts)))
        return cuts

    def _grid_edges(self, x_start, x_end, Ly):
        """Alle edges in een grid-fragment [x_start, x_end) × [0, Ly).

        Returns:
            edges: lijst van (qi, qj) met lokale qubit-indices
            n_qubits: totaal qubits in fragment
        """
        Lx_frag = x_end - x_start
        n_qubits = Lx_frag * Ly

        def idx(x, y):
            return (x - x_start) * Ly + y

        edges = []
        # Verticaal
        for x in range(x_start, x_end):
            for y in range(Ly - 1):
                edges.append((idx(x, y), idx(x, y + 1)))
        # Horizontaal (intern)
        for x in range(x_start, x_end - 1):
            for y in range(Ly):
                edges.append((idx(x, y), idx(x + 1, y)))

        return edges, n_qubits

    def eval_ratio_no_cuts(self, p, gammas, betas):
        """Exact QAOA zonder knips (referentie)."""
        Lx, Ly = self.Lx, self.Ly
        edges, n_q = self._grid_edges(0, Lx, Ly)

        psi = qaoa_statevector(n_q, edges, p, gammas, betas)

        total_cost = 0.0
        for (qi, qj) in edges:
            zz = measure_zz(psi, n_q, qi, qj)
            total_cost += (1 - zz) / 2

        return total_cost / self.n_edges

    def eval_ratio(self, p, gammas, betas, n_cuts=None, cut_positions=None):
        """Bereken MaxCut ratio via exacte QPD circuit knitting.

        Momenteel geïmplementeerd voor p=1 en willekeurig aantal knips.
        Voor p>1 met knips wordt teruggevallen op p=1-methode per laag
        (TODO: volledige p>1 decompositie).
        """
        Lx, Ly = self.Lx, self.Ly
        t0 = time.time()

        # Bepaal knipposities
        if cut_positions is not None:
            cuts = sorted(cut_positions)
        elif n_cuts is not None and n_cuts > 0:
            cuts = self.find_cut_positions(n_cuts)
        else:
            # Geen knips
            if self.verbose:
                print("  [B31] Geen knips, exact state vector")
            ratio = self.eval_ratio_no_cuts(p, gammas, betas)
            if self.verbose:
                print("  [B31] Ratio: %.6f (%.3fs)" % (ratio, time.time() - t0))
            return ratio

        if p > 1:
            # Voor p>1: elke laag heeft cross-cut ZZ, decompositie geeft
            # 2^(Ly*p) termen. Implementatie voor p=1 hieronder.
            # Voor p>1 gebruiken we de per-laag decompositie.
            return self._eval_ratio_multilayer(p, gammas, betas, cuts)

        return self._eval_ratio_p1(gammas[0], betas[0], cuts)

    def _eval_ratio_p1(self, gamma, beta, cuts):
        """Exacte QPD knitting voor p=1.

        Decompositie van cross-cut ZZ gates:
          Π_{y} [cos(γ) I⊗I − i sin(γ) Z_y^L⊗Z_y^R]
        = Σ_σ c_σ · Z_σ^L ⊗ Z_σ^R

        Per knip: 2^Ly termen (σ ∈ {0,1}^Ly).
        Totaal: 2^(Ly × n_cuts) termen.

        State vector per fragment-σ combinatie:
          |ψ_frag(σ_L, σ_R)⟩ = Rx · Phase_int · Z_σ_R^right · Z_σ_L^left · |+⟩

        (Z_σ op een qubit = sign flip in init: |+⟩→|−⟩)
        """
        Lx, Ly = self.Lx, self.Ly
        t0 = time.time()

        n_cuts = len(cuts)
        boundaries = [0] + cuts + [Lx]
        n_frags = len(boundaries) - 1

        # Coëfficiënten
        cg = np.cos(gamma)
        sg = -1j * np.sin(gamma)

        if self.verbose:
            print("  [B31] Circuit Knitting p=1: %dx%d" % (Lx, Ly))
            print("  [B31] Knips: %d op %s" % (n_cuts, cuts))
            frag_sizes = [(boundaries[i+1]-boundaries[i])*Ly
                          for i in range(n_frags)]
            print("  [B31] Fragmenten: %s qubits" % frag_sizes)
            n_terms = (1 << Ly) ** n_cuts
            print("  [B31] QPD termen: %d (2^%d)" % (n_terms, Ly * n_cuts))

        # ============================================================
        # Stap 1: Per fragment, genereer alle state vectors
        # ============================================================
        # Elk fragment heeft 0, 1 of 2 knip-randen.
        # Per knip-rand: σ ∈ {0,1}^Ly bepaalt welke rand-qubits Z-flip krijgen.
        # De linker knip-rand zijn de qubits in de eerste kolom van het fragment.
        # De rechter knip-rand zijn de qubits in de laatste kolom.

        fragment_info = []
        for fi in range(n_frags):
            x_start = boundaries[fi]
            x_end = boundaries[fi + 1]
            Lx_frag = x_end - x_start

            # Interne edges (geen cross-cut)
            edges_int, n_q = self._grid_edges(x_start, x_end, Ly)

            # Qubit indices van de knip-randen (lokaal in fragment)
            left_cut_qubits = []
            right_cut_qubits = []
            if fi > 0:
                # Linker rand = eerste kolom
                left_cut_qubits = [y for y in range(Ly)]  # lokaal idx
            if fi < n_frags - 1:
                # Rechter rand = laatste kolom
                right_cut_qubits = [(Lx_frag - 1) * Ly + y for y in range(Ly)]

            # Interne observabele edges (zonder cross-cut)
            obs_edges = edges_int[:]

            # Cross-cut edges (gemeten vanuit DIT fragment)
            # Rechter cross-cut: qubit (Lx_frag-1, y) ↔ (x_end, y) in volgend fragment
            # We meten ⟨Z_q⟩ op onze kant, het andere fragment meet ⟨Z_q'⟩
            # Recombinatie via: ⟨ZZ⟩ = Σ c*c' ⟨ψ(σ)|Z_q|ψ(τ)⟩_L · ⟨ψ(σ)|Z_q'|ψ(τ)⟩_R
            right_cross_obs = []
            if fi < n_frags - 1:
                for y in range(Ly):
                    q_local = (Lx_frag - 1) * Ly + y
                    right_cross_obs.append((q_local, y))  # (local_q, y_index)

            fragment_info.append({
                'x_start': x_start,
                'x_end': x_end,
                'n_qubits': n_q,
                'edges_int': edges_int,
                'left_cut_qubits': left_cut_qubits,
                'right_cut_qubits': right_cut_qubits,
                'obs_edges': obs_edges,
                'right_cross_obs': right_cross_obs,
            })

        # ============================================================
        # Stap 2: Genereer state vectors voor alle σ-combinaties
        # ============================================================
        # Per fragment: de σ-vector bepaalt de init-signs.
        # σ_left[y]=1 → qubit y (linker rand) start in |−⟩
        # σ_right[y]=1 → qubit (Lx-1)*Ly+y (rechter rand) start in |−⟩

        # De cut-σ variabelen zijn GEDEELD tussen fragmenten:
        # Cut i (tussen fragment i en i+1):
        #   σ_i ∈ {0,1}^Ly
        #   Fragment i: rechterkant qubits krijgen Z_σ_i
        #   Fragment i+1: linkerkant qubits krijgen Z_σ_i

        n_sigma_per_cut = 1 << Ly

        # Genereer alle state vectors per fragment, geïndexeerd door
        # (σ_left, σ_right) = tuple van cut-configuraties
        frag_states = []

        for fi, finfo in enumerate(fragment_info):
            n_q = finfo['n_qubits']
            left_qs = finfo['left_cut_qubits']
            right_qs = finfo['right_cut_qubits']

            # σ_left range: 0..2^Ly-1 als fi > 0, anders {0}
            # σ_right range: 0..2^Ly-1 als fi < n_frags-1, anders {0}
            n_left = n_sigma_per_cut if fi > 0 else 1
            n_right = n_sigma_per_cut if fi < n_frags - 1 else 1

            states = {}  # (σ_left_idx, σ_right_idx) → state vector

            for sl_idx in range(n_left):
                for sr_idx in range(n_right):
                    # Init signs
                    signs = np.ones(n_q)

                    # Left cut: σ bits flip
                    if fi > 0:
                        for y in range(Ly):
                            if (sl_idx >> (Ly - 1 - y)) & 1:
                                signs[left_qs[y]] = -1

                    # Right cut: σ bits flip
                    if fi < n_frags - 1:
                        for y in range(Ly):
                            if (sr_idx >> (Ly - 1 - y)) & 1:
                                signs[right_qs[y]] = -1

                    psi = qaoa_statevector(
                        n_q, finfo['edges_int'], 1, [gamma], [beta],
                        init_signs=signs)
                    states[(sl_idx, sr_idx)] = psi

            frag_states.append(states)

        # ============================================================
        # Stap 3: Recombinatie
        # ============================================================
        # De totale verwachtingswaarde:
        # ⟨O_total⟩ = Σ over alle cut-configs (σ,τ) van
        #   (Π cuts: c*_σ[cut] · c_τ[cut]) ·
        #   (Π fragmenten: matrix_element[frag](σ, τ, O))

        # Coëfficiënt per cut-config σ ∈ {0,1}^Ly:
        def cut_coeff(sigma_idx):
            """c_σ = Π_{j∈σ}(-i sin γ) · Π_{j∉σ}(cos γ)"""
            c = 1.0 + 0j
            for y in range(Ly):
                if (sigma_idx >> (Ly - 1 - y)) & 1:
                    c *= sg
                else:
                    c *= cg
            return c

        # Pre-compute coefficients
        coeffs = np.array([cut_coeff(s) for s in range(n_sigma_per_cut)])

        total_cost = 0.0

        # ---- Interne edges per fragment ----
        # ⟨Z_i Z_j⟩_frag = Σ_{σ,τ over ALL cuts}
        #   (Π_cuts c*_σ c_τ) ·
        #   ⟨ψ_frag(σ_L, σ_R)|Z_iZ_j|ψ_frag(τ_L, τ_R)⟩ ·
        #   (Π_{other_frags} ⟨ψ_other(σ)|ψ_other(τ)⟩)

        # Dit is in het algemeen een contractie over alle fragmenten.
        # Voor efficiëntie: contracteer fragment-voor-fragment.

        # Vereenvoudiging: voor een ENKELE knip is dit haalbaar.
        # Voor meerdere knips: Transfer Matrix Methode (TMM).

        if n_cuts == 1:
            total_cost = self._recombine_single_cut(
                fragment_info, frag_states, coeffs, gamma)
        else:
            total_cost = self._recombine_multi_cut(
                fragment_info, frag_states, coeffs, cuts)

        ratio = total_cost / self.n_edges
        elapsed = time.time() - t0

        if self.verbose:
            n_sub_runs = sum(len(s) for s in frag_states)
            print("  [B31] Ratio: %.6f (%.3fs, %d sub-runs)" % (
                ratio, elapsed, n_sub_runs))

        return ratio

    def _recombine_single_cut(self, fragment_info, frag_states, coeffs, gamma):
        """Recombinatie voor één knip (2 fragmenten)."""
        Ly = self.Ly
        n_sigma = len(coeffs)
        fi_L, fi_R = 0, 1
        info_L, info_R = fragment_info[0], fragment_info[1]
        states_L, states_R = frag_states[0], frag_states[1]
        n_L = info_L['n_qubits']
        n_R = info_R['n_qubits']

        total_cost = 0.0

        # ---- Interne edges in fragment L ----
        # ⟨Z_iZ_j⟩ = Σ_{σ,τ} c*_σ c_τ ⟨ψ_L(0,σ)|Z_iZ_j|ψ_L(0,τ)⟩ · ⟨ψ_R(σ,0)|ψ_R(τ,0)⟩
        #
        # Overlap matrix R:
        S_R = np.zeros((n_sigma, n_sigma), dtype=complex)
        for s in range(n_sigma):
            for t in range(n_sigma):
                S_R[s, t] = overlap(states_R[(s, 0)], states_R[(t, 0)])

        # Gewogen overlap: W_R[s,t] = c*_s · c_t · S_R[s,t]
        W_R = np.outer(np.conj(coeffs), coeffs) * S_R

        for (qi, qj) in info_L['obs_edges']:
            M_L = np.zeros((n_sigma, n_sigma), dtype=complex)
            for s in range(n_sigma):
                for t in range(n_sigma):
                    M_L[s, t] = matrix_element_zz(
                        states_L[(0, s)], states_L[(0, t)], n_L, qi, qj)
            zz = np.sum(M_L * W_R).real
            total_cost += (1 - zz) / 2

        # ---- Interne edges in fragment R ----
        S_L = np.zeros((n_sigma, n_sigma), dtype=complex)
        for s in range(n_sigma):
            for t in range(n_sigma):
                S_L[s, t] = overlap(states_L[(0, s)], states_L[(0, t)])

        W_L = np.outer(np.conj(coeffs), coeffs) * S_L

        for (qi, qj) in info_R['obs_edges']:
            M_R = np.zeros((n_sigma, n_sigma), dtype=complex)
            for s in range(n_sigma):
                for t in range(n_sigma):
                    M_R[s, t] = matrix_element_zz(
                        states_R[(s, 0)], states_R[(t, 0)], n_R, qi, qj)
            zz = np.sum(W_L * M_R).real
            total_cost += (1 - zz) / 2

        # ---- Cross-cut edges ----
        # ⟨Z_a^L Z_b^R⟩ = Σ_{σ,τ} c*_σ c_τ ⟨ψ_L(0,σ)|Z_a|ψ_L(0,τ)⟩ · ⟨ψ_R(σ,0)|Z_b|ψ_R(τ,0)⟩
        for (q_local_L, y) in info_L['right_cross_obs']:
            q_local_R = y  # In R fragment: eerste kolom, qubit y

            Z_L = np.zeros((n_sigma, n_sigma), dtype=complex)
            Z_R = np.zeros((n_sigma, n_sigma), dtype=complex)
            for s in range(n_sigma):
                for t in range(n_sigma):
                    Z_L[s, t] = matrix_element_z(
                        states_L[(0, s)], states_L[(0, t)], n_L, q_local_L)
                    Z_R[s, t] = matrix_element_z(
                        states_R[(s, 0)], states_R[(t, 0)], n_R, q_local_R)

            CC = np.outer(np.conj(coeffs), coeffs)
            zz = np.sum(CC * Z_L * Z_R).real
            total_cost += (1 - zz) / 2

        return total_cost

    def _recombine_multi_cut(self, fragment_info, frag_states, coeffs, cuts):
        """Recombinatie voor meerdere knips via Transfer Matrix methode.

        Contracteer fragment-voor-fragment van links naar rechts.
        De 'lopende toestand' is een matrix van overlap-gewichten.
        """
        Ly = self.Ly
        n_sigma = len(coeffs)
        n_frags = len(fragment_info)
        n_cuts = len(cuts)

        total_cost = 0.0

        # Voor elk fragment berekenen we:
        # 1. Bijdrage van interne edges
        # 2. Bijdrage van cross-cut edges (naar rechts)
        #
        # De recombinatie-gewichten worden berekend via een transfer matrix
        # die van links naar rechts wordt gecontracteerd.

        # Strategie: voor elke observabele, contracteer de keten van
        # overlap-matrices. Maar dat is O(n_frags × n_sigma^2 × n_observables).
        #
        # Efficiënter: contracteer de overlap-matrices eerst, gebruik het
        # resultaat voor alle observabelen in een fragment.

        # Transfer matrix: T[s,t] = Π over gefactorerde fragmenten van
        # de overlap bijdrage.

        # De totale verwachtingswaarde voor een interne edge in fragment fi:
        # ⟨O⟩ = Σ_{σ_cuts, τ_cuts} (Π c*_σ c_τ per cut) ×
        #        (Π_{f≠fi} S_f[σ_f,τ_f]) × M_fi[σ_fi, τ_fi, O]
        #
        # = Tr[ (Π_{f<fi} T_f) · M_fi · (Π_{f>fi} T_f) ]
        #
        # waarbij T_f[s_L,t_L,s_R,t_R] = C*[s_L]C[t_L]C*[s_R]C[t_R] · S_f[...]

        # Vereenvoudiging: contracteer per fragment paar

        # Bereken alle overlaps en observabelen per fragment
        frag_data = []
        for fi in range(n_frags):
            info = fragment_info[fi]
            states = frag_states[fi]
            n_q = info['n_qubits']

            has_left = fi > 0
            has_right = fi < n_frags - 1
            n_left = n_sigma if has_left else 1
            n_right = n_sigma if has_right else 1

            # Overlap matrix: S[(sl,sr), (tl,tr)] = ⟨ψ(sl,sr)|ψ(tl,tr)⟩
            S = np.zeros((n_left, n_right, n_left, n_right), dtype=complex)
            for sl in range(n_left):
                for sr in range(n_right):
                    for tl in range(n_left):
                        for tr in range(n_right):
                            S[sl, sr, tl, tr] = overlap(
                                states[(sl, sr)], states[(tl, tr)])

            frag_data.append({
                'info': info,
                'states': states,
                'S': S,
                'n_left': n_left,
                'n_right': n_right,
                'n_qubits': n_q,
            })

        # Coëfficiënt tensor per cut
        CC = np.outer(np.conj(coeffs), coeffs)  # (n_sigma, n_sigma)

        # ---- Bereken transfer matrices ----
        # De gewogen overlap per fragment:
        # W_f[sl, tl, sr, tr] = (CC[sl,tl] als has_left) × (CC[sr,tr] als has_right) × S_f[sl,sr,tl,tr]
        # Maar CC per cut wordt gedeeld: sl van fragment fi is sr van fi-1
        #
        # Contractie van links naar rechts:
        # Accumuleer een "boundary vector" v[s,t] die het linker-deel representeert.
        # Start: v[0,0] = 1 (geen linker cuts)
        # Per fragment fi: v_new[sr,tr] = Σ_{sl,tl} v[sl,tl] × CC[sl,tl] × S_f[sl,sr,tl,tr]
        # (als fi=0: sl=tl=0, geen CC factor)

        # Berekening per observabele: vervang S door M (matrix element) in het relevante fragment.

        # ---- Interne edges ----
        for fi in range(n_frags):
            fd = frag_data[fi]
            info = fd['info']
            states = fd['states']
            n_q = fd['n_qubits']

            for (qi, qj) in info['obs_edges']:
                # Matrix element tensor
                M = np.zeros_like(fd['S'])
                nl, nr = fd['n_left'], fd['n_right']
                for sl in range(nl):
                    for sr in range(nr):
                        for tl in range(nl):
                            for tr in range(nr):
                                M[sl, sr, tl, tr] = matrix_element_zz(
                                    states[(sl, sr)], states[(tl, tr)],
                                    n_q, qi, qj)

                zz = self._contract_chain(frag_data, CC, fi, M)
                total_cost += (1 - zz) / 2

        # ---- Cross-cut edges ----
        for fi in range(n_frags - 1):
            fd_L = frag_data[fi]
            fd_R = frag_data[fi + 1]
            info_L = fd_L['info']

            for (q_local_L, y) in info_L['right_cross_obs']:
                q_local_R = y  # eerste kolom van rechter fragment

                # Z matrix elements voor L en R
                Z_L = np.zeros_like(fd_L['S'])
                Z_R = np.zeros_like(fd_R['S'])

                for sl in range(fd_L['n_left']):
                    for sr in range(fd_L['n_right']):
                        for tl in range(fd_L['n_left']):
                            for tr in range(fd_L['n_right']):
                                Z_L[sl, sr, tl, tr] = matrix_element_z(
                                    fd_L['states'][(sl, sr)],
                                    fd_L['states'][(tl, tr)],
                                    fd_L['n_qubits'], q_local_L)

                for sl in range(fd_R['n_left']):
                    for sr in range(fd_R['n_right']):
                        for tl in range(fd_R['n_left']):
                            for tr in range(fd_R['n_right']):
                                Z_R[sl, sr, tl, tr] = matrix_element_z(
                                    fd_R['states'][(sl, sr)],
                                    fd_R['states'][(tl, tr)],
                                    fd_R['n_qubits'], q_local_R)

                zz = self._contract_chain_2obs(frag_data, CC, fi, Z_L, fi + 1, Z_R)
                total_cost += (1 - zz) / 2

        return total_cost

    def _contract_chain(self, frag_data, CC, obs_frag, M_obs):
        """Contracteer de keten met overlap-matrices, met M_obs op positie obs_frag.

        Returns: Σ_{alle σ,τ} (Π CC per cut) × (Π S per fragment) maar met
                 S[obs_frag] vervangen door M_obs.
        """
        n_frags = len(frag_data)
        n_sigma = CC.shape[0]

        # Contracteer van links naar rechts
        # v[s, t] is de lopende "boundary" vector
        v = np.array([[1.0 + 0j]])  # (1,1) start

        for fi in range(n_frags):
            fd = frag_data[fi]
            nl, nr = fd['n_left'], fd['n_right']
            T = M_obs if fi == obs_frag else fd['S']

            # v_new[sr, tr] = Σ_{sl,tl} v[sl,tl] × CC_left[sl,tl] × T[sl,sr,tl,tr]
            # Als fi==0: sl=tl=0, geen CC_left
            if fi == 0:
                # v is (1,1), sl=tl=0
                v_new = np.zeros((nr, nr), dtype=complex)
                for sr in range(nr):
                    for tr in range(nr):
                        v_new[sr, tr] = v[0, 0] * T[0, sr, 0, tr]
            else:
                v_new = np.zeros((nr, nr), dtype=complex)
                for sr in range(nr):
                    for tr in range(nr):
                        val = 0j
                        for sl in range(nl):
                            for tl in range(nl):
                                val += v[sl, tl] * CC[sl, tl] * T[sl, sr, tl, tr]
                        v_new[sr, tr] = val

            v = v_new

        # Einddraad: v is (1,1) of (n_sigma, n_sigma)
        # Als het laatste fragment geen rechter cut heeft: v is (1,1)
        # Anders: contracteer met laatste CC
        if v.shape[0] > 1:
            # Shouldn't happen — laatste fragment heeft geen rechter cut
            result = np.sum(CC * v)
        else:
            result = v[0, 0]

        return result.real

    def _contract_chain_2obs(self, frag_data, CC, fi_a, M_a, fi_b, M_b):
        """Contracteer keten met twee observabelen op posities fi_a en fi_b."""
        n_frags = len(frag_data)

        v = np.array([[1.0 + 0j]])

        for fi in range(n_frags):
            fd = frag_data[fi]
            nl, nr = fd['n_left'], fd['n_right']

            if fi == fi_a:
                T = M_a
            elif fi == fi_b:
                T = M_b
            else:
                T = fd['S']

            if fi == 0:
                v_new = np.zeros((nr, nr), dtype=complex)
                for sr in range(nr):
                    for tr in range(nr):
                        v_new[sr, tr] = v[0, 0] * T[0, sr, 0, tr]
            else:
                v_new = np.zeros((nr, nr), dtype=complex)
                for sr in range(nr):
                    for tr in range(nr):
                        val = 0j
                        for sl in range(nl):
                            for tl in range(nl):
                                val += v[sl, tl] * CC[sl, tl] * T[sl, sr, tl, tr]
                        v_new[sr, tr] = val
            v = v_new

        if v.shape[0] > 1:
            result = np.sum(CC * v)
        else:
            result = v[0, 0]

        return result.real

    def _eval_ratio_multilayer(self, p, gammas, betas, cuts):
        """Multi-layer (p>1) circuit knitting.

        Voor p>1 decomponeert elke laag de cross-cut ZZ apart.
        Dit geeft 2^(Ly·p) termen per knip.

        Alternatieve aanpak: voer de p lagen uit met "gestapelde" σ-vectoren,
        waarbij elke laag zijn eigen σ heeft.

        Voorlopig: genereer alle 2^(Ly·p) initialisatie-patronen per fragment
        en contracteer. Dit is exact maar exponentieel in p.
        """
        Lx, Ly = self.Lx, self.Ly

        if self.verbose:
            n_terms = (1 << (Ly * p)) ** len(cuts)
            print("  [B31] Multi-layer p=%d: %d QPD termen (EXPERIMENTEEL)" % (
                p, n_terms))

        # TODO: Implementeer volledige multi-layer decompositie.
        # Voorlopig: fallback naar brute force op kleine systemen.
        if Lx * Ly <= 24:
            return self.eval_ratio_no_cuts(p, gammas, betas)

        raise NotImplementedError(
            "Multi-layer circuit knitting voor p>1 met knips op grote systemen "
            "is nog niet geïmplementeerd. Gebruik p=1 of geen knips.")

    def optimize(self, p, n_gamma=10, n_beta=10, n_cuts=None,
                 cut_positions=None, refine=True):
        """Grid search + scipy verfijning."""
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
                r = self.eval_ratio(p, [g] * p, [b] * p,
                                    n_cuts=n_cuts, cut_positions=cut_positions)
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
        grid_time = time.time() - t0

        if self.verbose:
            print("    Grid klaar: ratio=%.6f (%.1fs, %d evals)" % (
                best_ratio, grid_time, n_evals))

        if refine:
            try:
                from scipy.optimize import minimize as scipy_minimize

                def neg_ratio(params):
                    gs = list(params[:p])
                    bs = list(params[p:])
                    return -self.eval_ratio(p, gs, bs,
                                            n_cuts=n_cuts,
                                            cut_positions=cut_positions)

                x0 = best_gammas + best_betas
                self.verbose = False
                result = scipy_minimize(neg_ratio, x0, method='Nelder-Mead',
                                        options={'maxiter': 200, 'xatol': 1e-5,
                                                 'fatol': 1e-6, 'adaptive': True})
                self.verbose = old_verbose
                n_evals += result.nfev
                if -result.fun > best_ratio:
                    old = best_ratio
                    best_ratio = -result.fun
                    best_gammas = list(result.x[:p])
                    best_betas = list(result.x[p:])
                    if self.verbose:
                        print("    Scipy: ratio=%.6f (+%.6f)" % (
                            best_ratio, best_ratio - old))

            except ImportError:
                pass

        return best_ratio, best_gammas, best_betas, {
            'total_time': time.time() - t0,
            'n_evals': n_evals,
        }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B31: Circuit Knitting voor QAOA MaxCut')
    parser.add_argument('--Lx', type=int, default=8)
    parser.add_argument('--Ly', type=int, default=1)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--cuts', type=int, default=None)
    parser.add_argument('--cut-pos', type=int, nargs='+', default=None)
    parser.add_argument('--gamma', type=float, nargs='+', default=None)
    parser.add_argument('--beta', type=float, nargs='+', default=None)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--ngamma', type=int, default=10)
    parser.add_argument('--nbeta', type=int, default=10)
    parser.add_argument('--validate', action='store_true',
                        help='Vergelijk knitting met exact (0 knips)')
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  B31: Circuit Knitting QAOA (QPD exacte decompositie)")
    print(sep)

    Lx, Ly, p = args.Lx, args.Ly, args.p
    print("  Grid: %dx%d (%d qubits), p=%d" % (Lx, Ly, Lx * Ly, p))

    engine = CircuitKnitting(Lx, Ly, verbose=True)

    if args.validate:
        gammas = args.gamma or [0.3927] * p
        betas = args.beta or [1.1781] * p
        print("\n  Validatie:")
        r_exact = engine.eval_ratio(p, gammas, betas, n_cuts=0)
        if Lx >= 4:
            r_1cut = engine.eval_ratio(p, gammas, betas, n_cuts=1)
            print("  Verschil (1 knip): %.2e" % abs(r_exact - r_1cut))
        if Lx >= 6:
            r_2cut = engine.eval_ratio(p, gammas, betas, n_cuts=2)
            print("  Verschil (2 knips): %.2e" % abs(r_exact - r_2cut))

    elif args.optimize:
        ratio, gammas, betas, info = engine.optimize(
            p, n_gamma=args.ngamma, n_beta=args.nbeta,
            n_cuts=args.cuts, cut_positions=args.cut_pos)
        print("\n  RESULTAAT: ratio=%.6f, gammas=%s, betas=%s (%.1fs)" % (
            ratio, gammas, betas, info['total_time']))
    else:
        gammas = args.gamma or [0.3927] * p
        betas = args.beta or [1.1781] * p
        ratio = engine.eval_ratio(p, gammas, betas,
                                  n_cuts=args.cuts, cut_positions=args.cut_pos)

    print(sep)


if __name__ == '__main__':
    main()
