#!/usr/bin/env python3
"""
zorn_mera.py — Binary MERA Tensor Network Engine (B14)

Binary MERA (Multi-scale Entanglement Renormalization Ansatz, Vidal 2007).
Vangt volume-law verstrengeling met polynomiale kosten door hiërarchische
disentangler-lagen die ver-uit-elkaar-liggende qubits direct verbinden.

Structuur per laag τ (bottom-up):
  1. Disentanglers u op bridge-paren: (1,2), (3,4), (5,6), ...
  2. Isometrieën w op merge-paren: (0,1), (2,3), (4,5), ...

Na log₂(n) lagen: 1 top-site met toestandsvector |t⟩.

Vergelijking met MPS:
  MPS:  O(n·χ³) contractie, area-law (S ~ const)
  MERA: O(n·log(n)·χ⁶) contractie, volume-law (S ~ log L)

Referenties:
  [1] Vidal, PRL 99, 220405 (2007) — originele MERA
  [2] Evenbly & Vidal, PRB 79, 144108 (2009) — optimalisatie
  [3] Swingle, PRD 86, 065007 (2012) — holografische connectie
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import eigsh, LinearOperator
import time
from typing import List, Tuple, Optional, Dict


# =====================================================================
# HELPER FUNCTIES
# =====================================================================

def _random_isometry(dim_out, dim_in1, dim_in2, rng):
    """Genereer random isometrie (dim_out, dim_in1, dim_in2) via QR."""
    dim_in = dim_in1 * dim_in2
    A = rng.standard_normal((max(dim_out, dim_in), max(dim_out, dim_in))) \
        + 1j * rng.standard_normal((max(dim_out, dim_in), max(dim_out, dim_in)))
    Q, _ = np.linalg.qr(A)
    return Q[:dim_out, :dim_in].reshape(dim_out, dim_in1, dim_in2).copy()


def _random_unitary(dim1, dim2, rng):
    """Genereer random unitaire (dim1, dim2, dim1, dim2) via QR."""
    dim = dim1 * dim2
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    return Q.reshape(dim1, dim2, dim1, dim2).copy()


def _apply_2site_to_tensor(psi_tensor, u, ax1, ax2):
    """Pas 2-site unitaire u[o1,o2,i1,i2] toe op assen ax1,ax2 van tensor."""
    psi = np.moveaxis(psi_tensor, [ax1, ax2], [-2, -1])
    shape = psi.shape
    d1, d2 = shape[-2], shape[-1]
    batch = shape[:-2]
    psi_flat = psi.reshape(-1, d1, d2)
    result = np.einsum('abij,nij->nab', u, psi_flat)
    result = result.reshape(batch + (d1, d2))
    return np.moveaxis(result, [-2, -1], [ax1, ax2])


# =====================================================================
# ZORN MERA KLASSE
# =====================================================================

class ZornMERA:
    """Binary MERA tensor netwerk.

    Args:
        n_sites: Aantal fysieke sites (moet macht van 2 zijn)
        d: Fysieke dimensie (2 voor qubits)
        chi: Bond-dimensie op alle interne niveaus
    """

    def __init__(self, n_sites: int, d: int = 2, chi: int = 4):
        assert n_sites > 0 and (n_sites & (n_sites - 1)) == 0, \
            "n_sites moet een macht van 2 zijn"
        assert n_sites >= 2, "n_sites >= 2 vereist"
        assert chi >= 1, "chi >= 1 vereist"

        self.n_sites = n_sites
        self.d = d
        self.chi = chi
        self.n_layers = int(np.log2(n_sites))

        # Bouw laagstructuur
        self.layers = []
        for tau in range(self.n_layers):
            n_tau = n_sites >> tau
            dim_in = d if tau == 0 else chi
            dim_out = chi

            # Disentanglers op bridge-paren: (1,2), (3,4), ...
            n_disent = max(0, n_tau // 2 - 1)
            disentanglers = []
            for _ in range(n_disent):
                u = np.eye(dim_in ** 2, dtype=complex).reshape(
                    dim_in, dim_in, dim_in, dim_in)
                disentanglers.append(u)

            # Isometrieën op merge-paren: (0,1), (2,3), ...
            n_isom = n_tau // 2
            isometries = []
            for _ in range(n_isom):
                w = np.zeros((dim_out, dim_in, dim_in), dtype=complex)
                for a in range(min(dim_out, dim_in ** 2)):
                    w[a, a // dim_in, a % dim_in] = 1.0
                isometries.append(w)

            self.layers.append({
                'disentanglers': disentanglers,
                'isometries': isometries,
                'dim_in': dim_in,
                'dim_out': dim_out,
                'n_sites': n_tau,
                'n_disent': n_disent,
                'n_isom': n_isom,
            })

        # Top tensor
        self.top = np.zeros(chi, dtype=complex)
        self.top[0] = 1.0

    # -----------------------------------------------------------------
    # INITIALISATIE
    # -----------------------------------------------------------------

    def init_product(self, state: int = 0):
        """Initialiseer als producttoestand |state⟩^⊗n.

        Alleen state=0 wordt exact ondersteund (|0...0⟩).
        """
        d = self.d
        chi = self.chi

        for tau in range(self.n_layers):
            layer = self.layers[tau]
            dim_in = layer['dim_in']

            for i in range(layer['n_disent']):
                layer['disentanglers'][i] = np.eye(
                    dim_in ** 2, dtype=complex
                ).reshape(dim_in, dim_in, dim_in, dim_in)

            for i in range(layer['n_isom']):
                w = np.zeros((chi, dim_in, dim_in), dtype=complex)
                for a in range(min(chi, dim_in ** 2)):
                    w[a, a // dim_in, a % dim_in] = 1.0
                layer['isometries'][i] = w

        self.top = np.zeros(chi, dtype=complex)
        self.top[0] = 1.0
        return self

    def init_random(self, seed: Optional[int] = None):
        """Initialiseer met random isometrieën/unitairen (Haar)."""
        rng = np.random.default_rng(seed)

        for tau in range(self.n_layers):
            layer = self.layers[tau]
            dim_in = layer['dim_in']
            dim_out = layer['dim_out']

            for i in range(layer['n_disent']):
                layer['disentanglers'][i] = _random_unitary(dim_in, dim_in, rng)

            for i in range(layer['n_isom']):
                layer['isometries'][i] = _random_isometry(
                    dim_out, dim_in, dim_in, rng)

        t = rng.standard_normal(self.chi) + 1j * rng.standard_normal(self.chi)
        self.top = t / np.linalg.norm(t)
        return self

    # -----------------------------------------------------------------
    # TOESTANDSVECTOR CONTRACTIE
    # -----------------------------------------------------------------

    def to_statevector(self) -> np.ndarray:
        """Contract MERA naar volledige toestandsvector (d^n).

        Kosten: O(χ^3 · n · log n). Geheugen: O(d^n).
        Alleen bruikbaar voor n ≤ ~20.
        """
        labels = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

        psi = self.top.copy()
        site_dims = [self.chi]

        for tau in range(self.n_layers - 1, -1, -1):
            layer = self.layers[tau]
            isoms = layer['isometries']
            disents = layer['disentanglers']
            dim_in = layer['dim_in']
            n_tau = layer['n_sites']
            n_above = n_tau // 2

            psi_tensor = psi.reshape(site_dims)

            # Stap 1: Expandeer via isometrieën
            psi_idx = labels[:n_above]
            out_idx = ''
            tensors = [psi_tensor]
            subscripts = [psi_idx]
            next_lbl = n_above

            for k in range(n_above):
                w = isoms[k]
                w_lbl = labels[k] + labels[next_lbl] + labels[next_lbl + 1]
                subscripts.append(w_lbl)
                tensors.append(w)
                out_idx += labels[next_lbl] + labels[next_lbl + 1]
                next_lbl += 2

            einsum_str = ','.join(subscripts) + '->' + out_idx
            psi_tensor = np.einsum(einsum_str, *tensors, optimize=True)

            # Stap 2: Pas disentanglers toe
            for di in range(len(disents)):
                u = disents[di]
                s1 = 2 * di + 1
                s2 = 2 * di + 2
                psi_tensor = _apply_2site_to_tensor(psi_tensor, u, s1, s2)

            site_dims = [dim_in] * n_tau
            psi = psi_tensor.reshape(-1)

        return psi

    # -----------------------------------------------------------------
    # VERWACHTINGSWAARDEN
    # -----------------------------------------------------------------

    def _apply_H(self, psi: np.ndarray, terms: List[Tuple]) -> np.ndarray:
        """Bereken H|ψ⟩ zonder volledige H-matrix."""
        n = self.n_sites
        d = self.d
        result = np.zeros_like(psi)
        psi_tensor = psi.reshape([d] * n)

        for coeff, op, sites in terms:
            if len(sites) == 1:
                s = sites[0]
                O_psi = np.tensordot(op, psi_tensor, axes=([1], [s]))
                O_psi = np.moveaxis(O_psi, 0, s)
                result += coeff * O_psi.reshape(-1)
            elif len(sites) == 2:
                s1, s2 = sites
                op4 = op if op.ndim == 4 else op.reshape(d, d, d, d)
                O_psi = np.tensordot(op4, psi_tensor, axes=([2, 3], [s1, s2]))
                O_psi = np.moveaxis(O_psi, [0, 1], [s1, s2])
                result += coeff * O_psi.reshape(-1)

        return result

    def expectation_local(self, obs: np.ndarray, site: int) -> complex:
        """Bereken ⟨ψ|O_site|ψ⟩."""
        psi = self.to_statevector()
        n = self.n_sites
        d = self.d
        psi_tensor = psi.reshape([d] * n)
        O_psi = np.tensordot(obs, psi_tensor, axes=([1], [site]))
        O_psi = np.moveaxis(O_psi, 0, site)
        return np.vdot(psi, O_psi.reshape(-1))

    def expectation_2point(self, op1: np.ndarray, s1: int,
                           op2: np.ndarray, s2: int) -> complex:
        """Bereken ⟨ψ|O1_{s1} O2_{s2}|ψ⟩."""
        psi = self.to_statevector()
        n = self.n_sites
        d = self.d
        psi_tensor = psi.reshape([d] * n)

        O_psi = np.tensordot(op2, psi_tensor, axes=([1], [s2]))
        O_psi = np.moveaxis(O_psi, 0, s2)
        O_psi = np.tensordot(op1, O_psi, axes=([1], [s1]))
        O_psi = np.moveaxis(O_psi, 0, s1)

        return np.vdot(psi, O_psi.reshape(-1))

    def energy(self, terms: List[Tuple]) -> float:
        """Bereken ⟨ψ|H|ψ⟩.

        terms: [(coeff, op, sites), ...]
          1-site: op (d,d), sites (s,)
          2-site: op (d²,d²) of (d,d,d,d), sites (s1,s2)
        """
        psi = self.to_statevector()
        H_psi = self._apply_H(psi, terms)
        return np.vdot(psi, H_psi).real

    # -----------------------------------------------------------------
    # VARIATIONELE OPTIMALISATIE
    # -----------------------------------------------------------------

    def optimize(self, terms: List[Tuple], n_sweeps: int = 20,
                 verbose: bool = False) -> List[float]:
        """Optimaliseer MERA-tensors om ⟨H⟩ te minimaliseren.

        Gebruikt environment-linearisatie + polaire decompositie.
        Werkt voor n ≤ 16 (gebruikt to_statevector intern).

        Args:
            terms: Hamiltoniaan [(coeff, op, sites), ...]
            n_sweeps: Aantal sweeps
            verbose: Print voortgang

        Returns:
            Lijst van energieën per sweep
        """
        energies = []

        for sweep in range(n_sweeps):
            # Top tensor
            self._update_top(terms)

            # Lagen van boven naar beneden
            for tau in range(self.n_layers - 1, -1, -1):
                layer = self.layers[tau]

                for k in range(layer['n_isom']):
                    self._update_isometry(tau, k, terms)

                for k in range(layer['n_disent']):
                    self._update_disentangler(tau, k, terms)

            E = self.energy(terms)
            energies.append(E)

            if verbose and (sweep % 5 == 0 or sweep == n_sweeps - 1):
                print("  Sweep %d: E = %.8f" % (sweep, E))

        return energies

    def _compute_phi_matrix(self, tau, tensor_type, k):
        """Bereken Φ-matrix via probing: |ψ⟩ = Σ T[idx] |φ_idx⟩."""
        layer = self.layers[tau]
        dim = self.d ** self.n_sites

        if tensor_type == 'isometry':
            T_old = layer['isometries'][k].copy()
            shape = T_old.shape
            n_basis = int(np.prod(shape))

            Phi = np.zeros((dim, n_basis), dtype=complex)
            for idx in range(n_basis):
                mi = np.unravel_index(idx, shape)
                T_probe = np.zeros(shape, dtype=complex)
                T_probe[mi] = 1.0
                layer['isometries'][k] = T_probe
                Phi[:, idx] = self.to_statevector()

            layer['isometries'][k] = T_old
            return Phi, shape

        elif tensor_type == 'disentangler':
            T_old = layer['disentanglers'][k].copy()
            shape = T_old.shape
            n_basis = int(np.prod(shape))

            Phi = np.zeros((dim, n_basis), dtype=complex)
            for idx in range(n_basis):
                mi = np.unravel_index(idx, shape)
                T_probe = np.zeros(shape, dtype=complex)
                T_probe[mi] = 1.0
                layer['disentanglers'][k] = T_probe
                Phi[:, idx] = self.to_statevector()

            layer['disentanglers'][k] = T_old
            return Phi, shape

    def _update_isometry(self, tau, k, terms):
        """Update isometrie via eigendecompositie van M = Φ†HΦ.

        Wiskundige basis: |ψ⟩ = Σ_ij W[i,j] |φ_j⟩ is lineair in W.
        Energie E = Tr(W M W†) met M = Φ†HΦ.
        Minimaliseer E onder WW† = I → W = V[:, :chi].conj().T
        waar V de eigenvectoren van M zijn (kleinste eigenwaarden).
        Dit is het GLOBALE minimum (Eckart-Young voor Hermitische vormen).
        """
        layer = self.layers[tau]
        w_old = layer['isometries'][k]
        shape = w_old.shape
        dim_out = shape[0]  # chi
        n_basis = int(np.prod(shape))

        Phi, _ = self._compute_phi_matrix(tau, 'isometry', k)

        # Bouw M = Φ†HΦ (effectieve Hamiltoniaan in tensor-ruimte)
        H_Phi = np.zeros_like(Phi)
        for b in range(n_basis):
            H_Phi[:, b] = self._apply_H(Phi[:, b], terms)
        M = Phi.conj().T @ H_Phi
        M = 0.5 * (M + M.conj().T)  # numeriek Hermitisch maken

        # Eigendecompositie: neem chi kleinste eigenvectoren
        eigvals, eigvecs = np.linalg.eigh(M)
        W = eigvecs[:, :dim_out].conj().T  # (chi, n_basis)
        layer['isometries'][k] = W.reshape(shape)

    def _update_disentangler(self, tau, k, terms):
        """Update disentangler via polaire decompositie met energiedaling-check.

        Voor unitaire (vierkante) tensoren is Tr(UMU†) invariant onder
        unitaire conjugatie, dus polaire decompositie geeft het juiste
        minimum. Maar de linearisatie kan falen bij sterke niet-lineariteit,
        daarom: revert als energie stijgt.
        """
        layer = self.layers[tau]
        u_old = layer['disentanglers'][k].copy()
        dim = u_old.shape[0] * u_old.shape[1]

        # Energie vóór update
        E_before = self.energy(terms)

        Phi, shape = self._compute_phi_matrix(tau, 'disentangler', k)
        psi = self.to_statevector()
        H_psi = self._apply_H(psi, terms)

        env = Phi.conj().T @ H_psi
        env_mat = env.reshape(dim, dim)

        U_e, S_e, Vh_e = np.linalg.svd(env_mat, full_matrices=True)
        u_new = -(U_e @ Vh_e).reshape(shape)
        layer['disentanglers'][k] = u_new

        # Energiedaling-check: revert als energie stijgt
        E_after = self.energy(terms)
        if E_after > E_before + 1e-14:
            layer['disentanglers'][k] = u_old

    def _update_top(self, terms):
        """Update top tensor via effectieve Hamiltoniaan."""
        chi = self.chi
        dim = self.d ** self.n_sites
        top_old = self.top.copy()

        # Bouw Φ en H_eff
        Phi = np.zeros((dim, chi), dtype=complex)
        for a in range(chi):
            t = np.zeros(chi, dtype=complex)
            t[a] = 1.0
            self.top = t
            Phi[:, a] = self.to_statevector()
        self.top = top_old

        # H_eff[a,b] = ⟨φ_a|H|φ_b⟩
        H_Phi = np.zeros((dim, chi), dtype=complex)
        for b in range(chi):
            H_Phi[:, b] = self._apply_H(Phi[:, b], terms)

        H_eff = Phi.conj().T @ H_Phi

        # Hermitisch maken (numerieke afronding)
        H_eff = 0.5 * (H_eff + H_eff.conj().T)

        eigvals, eigvecs = np.linalg.eigh(H_eff)
        self.top = eigvecs[:, 0]

    # -----------------------------------------------------------------
    # DIAGNOSTIEK
    # -----------------------------------------------------------------

    def fidelity(self, psi_exact: np.ndarray) -> float:
        """Bereken |⟨ψ_MERA|ψ_exact⟩|²."""
        psi = self.to_statevector()
        psi_n = psi / np.linalg.norm(psi)
        psi_e = psi_exact / np.linalg.norm(psi_exact)
        return abs(np.vdot(psi_n, psi_e)) ** 2

    @property
    def total_params(self) -> int:
        """Totaal aantal complexe parameters."""
        count = 0
        for layer in self.layers:
            for u in layer['disentanglers']:
                count += int(np.prod(u.shape))
            for w in layer['isometries']:
                count += int(np.prod(w.shape))
        count += self.chi
        return count

    @property
    def memory_bytes(self) -> int:
        """Geheugengebruik in bytes."""
        total = self.top.nbytes
        for layer in self.layers:
            for u in layer['disentanglers']:
                total += u.nbytes
            for w in layer['isometries']:
                total += w.nbytes
        return total

    def chi_profile(self) -> Dict[str, any]:
        """Overzicht van dimensies per laag."""
        profile = {}
        for tau in range(self.n_layers):
            layer = self.layers[tau]
            profile['layer_%d' % tau] = {
                'n_sites': layer['n_sites'],
                'dim_in': layer['dim_in'],
                'n_disent': layer['n_disent'],
                'n_isom': layer['n_isom'],
            }
        return profile


# =====================================================================
# MPS COMPRESSIE (voor vergelijking)
# =====================================================================

def compress_to_mps(psi, n, d, chi):
    """Comprimeer toestandsvector naar MPS via sequentiële SVD.

    Dit geeft de OPTIMALE MPS-benadering bij gegeven chi (Eckart-Young).

    Args:
        psi: Toestandsvector (d^n,)
        n: Aantal sites
        d: Lokale dimensie
        chi: Maximale bond-dimensie

    Returns:
        (tensors, fidelity, max_chi_used)
        tensors: lijst van (chi_L, d, chi_R) arrays
        fidelity: |⟨ψ_MPS|ψ_orig⟩|²
        max_chi_used: maximale bond-dimensie
    """
    psi = psi / np.linalg.norm(psi)
    remainder = psi.reshape(d, -1)
    tensors = []
    chi_left = 1
    discarded_sq = 0.0

    for s in range(n - 1):
        if s > 0:
            remainder = remainder.reshape(chi_left * d, -1)

        U, S, Vh = np.linalg.svd(remainder, full_matrices=False)
        k = min(chi, len(S))

        # Truncatieverlies
        if k < len(S):
            discarded_sq += np.sum(S[k:] ** 2)

        U_t = U[:, :k]
        S_t = S[:k]
        Vh_t = Vh[:k, :]

        tensors.append(U_t.reshape(chi_left, d, k))
        remainder = np.diag(S_t) @ Vh_t
        chi_left = k

    tensors.append(remainder.reshape(chi_left, d, 1))
    max_chi_used = max(t.shape[2] for t in tensors[:-1]) if n > 1 else 1
    fidelity = max(0.0, 1.0 - discarded_sq)

    return tensors, fidelity, max_chi_used


def mps_energy(tensors, terms, n, d):
    """Bereken ⟨ψ_MPS|H|ψ_MPS⟩ via toestandsvector-reconstructie."""
    # Reconstructie
    C = tensors[0]  # (1, d, chi)
    for A in tensors[1:]:
        # C: (..., d_prev, chi_L), A: (chi_L, d, chi_R)
        cl = C.shape[-1]
        C = np.einsum('...i,ijk->...jk', C, A)
    psi = C.reshape(-1)
    psi = psi / np.linalg.norm(psi)

    # Energie
    result = _apply_H_standalone(psi, terms, n, d)
    return np.vdot(psi, result).real


# =====================================================================
# HAMILTONIAAN BOUWERS
# =====================================================================

# Pauli-matrices
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_ZZ = np.kron(_Z, _Z)  # (4,4)


def build_tfim_terms(n, J=1.0, h=1.0):
    """1D Transverse-Field Ising Model: H = -J Σ ZZ - h Σ X.

    Kritiek punt bij h/J = 1 (c=1/2 CFT).
    """
    terms = []
    for i in range(n - 1):
        terms.append((-J, _ZZ, (i, i + 1)))
    for i in range(n):
        terms.append((-h, _X, (i,)))
    return terms


def build_heisenberg_terms(n, J=1.0):
    """1D Heisenberg: H = J Σ (XX + YY + ZZ)."""
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    terms = []
    for i in range(n - 1):
        terms.append((J, XX, (i, i + 1)))
        terms.append((J, YY, (i, i + 1)))
        terms.append((J, ZZ, (i, i + 1)))
    return terms


def build_maxcut_terms(n, edges):
    """MaxCut Hamiltoniaan: H = Σ_{(i,j)} (1 - Z_i Z_j)/2."""
    terms = []
    for i, j in edges:
        terms.append((-0.5, _ZZ, (i, j)))
        # Constante +0.5 per edge (offset, niet nodig voor optimalisatie)
    return terms


# =====================================================================
# EXACTE GRONDTOESTAND
# =====================================================================

def _apply_H_standalone(psi, terms, n, d):
    """Bereken H|ψ⟩ (standalone versie)."""
    result = np.zeros_like(psi)
    psi_tensor = psi.reshape([d] * n)

    for coeff, op, sites in terms:
        if len(sites) == 1:
            s = sites[0]
            O_psi = np.tensordot(op, psi_tensor, axes=([1], [s]))
            O_psi = np.moveaxis(O_psi, 0, s)
            result += coeff * O_psi.reshape(-1)
        elif len(sites) == 2:
            s1, s2 = sites
            op4 = op if op.ndim == 4 else op.reshape(d, d, d, d)
            O_psi = np.tensordot(op4, psi_tensor, axes=([2, 3], [s1, s2]))
            O_psi = np.moveaxis(O_psi, [0, 1], [s1, s2])
            result += coeff * O_psi.reshape(-1)

    return result


def exact_ground_state(terms, n, d=2):
    """Bereken exacte grondtoestand via diagonalisatie.

    Gebruikt volledige diag voor dim ≤ 4096, Lanczos anders.

    Returns:
        (E0, psi0) — grondtoestandsenergie en -vector
    """
    dim = d ** n

    def matvec(v):
        return _apply_H_standalone(v, terms, n, d)

    if dim <= 4096:
        H_mat = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            e_i = np.zeros(dim, dtype=complex)
            e_i[i] = 1.0
            H_mat[:, i] = matvec(e_i)
        H_mat = 0.5 * (H_mat + H_mat.conj().T)
        eigvals, eigvecs = np.linalg.eigh(H_mat)
        return eigvals[0], eigvecs[:, 0]
    else:
        H_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
        eigvals, eigvecs = eigsh(H_op, k=1, which='SA')
        return eigvals[0], eigvecs[:, 0]


# =====================================================================
# ENTANGLEMENT ENTROPIE
# =====================================================================

def entanglement_entropy(psi, n, d, cut):
    """Bereken entanglement entropie S(A) voor bipartitie A|B bij positie cut.

    Args:
        psi: Toestandsvector (d^n,)
        n: Aantal sites
        d: Lokale dimensie
        cut: Bipartitie-positie (A = sites 0..cut-1, B = rest)

    Returns:
        Von Neumann entropie S = -Tr(ρ_A log ρ_A)
    """
    dim_A = d ** cut
    dim_B = d ** (n - cut)
    rho_AB = psi.reshape(dim_A, dim_B)
    _, S, _ = np.linalg.svd(rho_AB, full_matrices=False)
    S = S[S > 1e-15]
    S2 = S ** 2
    return float(-np.sum(S2 * np.log(S2 + 1e-30)))


# =====================================================================
# QAOA TOESTANDSVECTOR (voor compressie-benchmark)
# =====================================================================

def qaoa_statevector(n, edges, p, gammas, betas):
    """Bereken QAOA |ψ(γ,β)⟩ exact (state vector).

    |ψ⟩ = Π_l [e^{-iβ_l X} e^{-iγ_l C}] |+⟩^n
    """
    d = 2
    dim = d ** n

    # |+⟩^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for l in range(p):
        # e^{-iγ C}: C = Σ (1-ZZ)/2
        # Op computationele basis: C|z⟩ = cut(z)|z⟩
        # e^{-iγC}|z⟩ = e^{-iγ·cut(z)}|z⟩
        phases = np.zeros(dim)
        for idx in range(dim):
            bits = [(idx >> k) & 1 for k in range(n)]
            cut = 0
            for i, j in edges:
                if bits[i] != bits[j]:
                    cut += 1
            phases[idx] = cut
        psi *= np.exp(-1j * gammas[l] * phases)

        # e^{-iβ X} = Π_i e^{-iβ X_i}
        # RX(2β) = [[cos β, -i sin β], [-i sin β, cos β]]
        c = np.cos(betas[l])
        s = np.sin(betas[l])
        rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

        psi_tensor = psi.reshape([2] * n)
        for i in range(n):
            psi_tensor = np.tensordot(rx, psi_tensor, axes=([1], [i]))
            psi_tensor = np.moveaxis(psi_tensor, 0, i)
        psi = psi_tensor.reshape(-1)

    return psi


# =====================================================================
# MAIN (selftest)
# =====================================================================

if __name__ == '__main__':
    print("=== ZornMERA selftest ===")

    # Test 1: product state
    mera = ZornMERA(8, d=2, chi=4)
    mera.init_product(0)
    psi = mera.to_statevector()
    expected = np.zeros(256, dtype=complex)
    expected[0] = 1.0
    err = np.linalg.norm(psi - expected)
    print("Product |0⟩^8: error = %.2e %s" % (err, "OK" if err < 1e-12 else "FAIL"))

    # Test 2: normalisatie random MERA
    mera.init_random(seed=42)
    psi = mera.to_statevector()
    norm = np.linalg.norm(psi)
    print("Random MERA norm: %.6f %s" % (norm, "OK" if abs(norm - 1) < 0.1 else "FAIL"))

    # Test 3: TFIM optimalisatie
    n = 8
    terms = build_tfim_terms(n, J=1.0, h=1.0)
    E_exact, psi_exact = exact_ground_state(terms, n)
    print("TFIM n=%d exact E0 = %.6f" % (n, E_exact))

    mera = ZornMERA(n, d=2, chi=4)
    mera.init_random(seed=123)
    energies = mera.optimize(terms, n_sweeps=15, verbose=True)
    E_mera = energies[-1]
    fid = mera.fidelity(psi_exact)
    print("MERA chi=4: E = %.6f (err=%.2e), fid = %.4f" % (
        E_mera, abs(E_mera - E_exact), fid))

    # Test 4: MPS compressie vergelijking
    for chi in [2, 4, 8]:
        _, fid_mps, _ = compress_to_mps(psi_exact, n, 2, chi)
        print("MPS chi=%d: fidelity = %.6f" % (chi, fid_mps))
