#!/usr/bin/env python3
"""B160 — QSVT / Block-Encoding Framework (pragmatic core).

Quantum Singular Value Transformation (Gilyén-Su-Low-Wiebe 2019) unifieert
Trotter-Suzuki, amplitude amplification, eigenvalue-filtering en
matrix-inversion onder één vlag. Voor ZornQ leveren we de **bruikbare kern**:

  1. Pauli-som-datatype + dichte matrix-representatie.
  2. LCU block-encoding (Childs-Wiebe 2012 stijl): gegeven H = Σ_k c_k P_k,
     bouw een unitaire U_H op (m_a + n) qubits waarvan het (⟨0|_a ⊗ I)-
     projectieblok gelijk is aan H/α met α = Σ |c_k|.
  3. Quantum Signal Processing (QSP): pas een polynoom p(x) toe op een
     1-qubit "signaal" unitair W(x) via een sequentie van Z-rotaties met
     fase-angles φ. Verificatie via vergelijking met Chebyshev-polynomen.
  4. QSVT op block-encoded Hermitische matrices A met ‖A‖ ≤ 1: bouw
     U_Φ = e^{iφ_0 Z_a} · [W · e^{iφ_k Z_a}]^d waarbij W een projector-
     gecontroleerde rotatie is die U_H gebruikt.
  5. **Hamiltonian-simulatie via Jacobi-Anger**: e^{-iHt} ≈ Σ_{k=0}^{K}
     c_k(αt) T_k(H/α) (Chebyshev-expansie). Dit is de QSVT-manier om
     Trotter te vervangen en heeft exponentiële convergentie in K.
     We leveren het direct als dichte-matrix operator; de volledige QSVT-
     circuitvertaling (fase-angle-oplossing) wordt geparkeerd voor B160b.
  6. Vergelijking met B129 Trotter (1e/2e orde) op Ising en Heisenberg
     Hamiltonians — laat zien waar Jacobi-Anger significant winst geeft
     ten opzichte van standaard Trotter bij gelijke kosten.

Pragmatische keuze: volledige "phase factor" compilatie (Haah 2019,
Dong-Meng-Whaley-Lin 2020 SymQSP-algoritme) is numeriek onstabiel en
onderzoeks-niveau. We leveren de volledige wiskundige structuur en alle
primitieven die nodig zijn om de stap naar circuit-vertaling later te
maken — met complete unit-tests die de correctheid op matrix-niveau
controleren.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.special import jv as _bessel_jv  # type: ignore
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ============================================================
# 1. Pauli-primitief + Pauli-som
# ============================================================

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MAP = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


def pauli_matrix(label: str) -> np.ndarray:
    """Dichte matrix voor een Pauli-string 'IXYZ...'. Qubit-volgorde: label[0]
    is qubit 0 in little-endian (tensor-product volgorde standaard van links)."""
    if not label:
        return np.array([[1.0 + 0j]])
    M = _PAULI_MAP[label[0]]
    for ch in label[1:]:
        M = np.kron(M, _PAULI_MAP[ch])
    return M


@dataclass
class PauliSum:
    """H = Σ_i coef_i · Pauli_i over n qubits. Labels zijn strings van lengte n."""

    n_qubits: int
    terms: List[Tuple[complex, str]]

    def to_matrix(self) -> np.ndarray:
        dim = 2 ** self.n_qubits
        M = np.zeros((dim, dim), dtype=complex)
        for c, lbl in self.terms:
            if len(lbl) != self.n_qubits:
                raise ValueError(
                    f"pauli label '{lbl}' heeft {len(lbl)} tekens, verwacht {self.n_qubits}"
                )
            M = M + c * pauli_matrix(lbl)
        return M

    def alpha(self) -> float:
        """LCU-normalisatie α = Σ_i |c_i|. ||H||_op ≤ α."""
        return float(sum(abs(c) for c, _ in self.terms))

    def __len__(self) -> int:
        return len(self.terms)

    @classmethod
    def from_hamiltonian(cls, H_obj) -> "PauliSum":
        """Converteer een hamiltonian_compiler.Hamiltonian naar PauliSum."""
        n = H_obj.n_qubits
        new_terms: List[Tuple[complex, str]] = []
        for coeff, pmap in H_obj.terms:
            if not pmap:
                continue  # constante term — globale fase, skip
            label = ["I"] * n
            for q, P in pmap.items():
                label[q] = P
            new_terms.append((complex(coeff), "".join(label)))
        return cls(n_qubits=n, terms=new_terms)


# ============================================================
# 2. LCU Block-encoding
# ============================================================

def _householder_extend(target: np.ndarray) -> np.ndarray:
    """Bouw een unitaire matrix waarvan kolom 0 gelijk is aan `target`
    (een genormaliseerde vector). Rest wordt via Gram-Schmidt aangevuld."""
    dim = target.shape[0]
    M = np.eye(dim, dtype=complex)
    M[:, 0] = target
    # Orthogonaliseer kolommen 1..dim-1 tegen de rest, niet tegen kolom 0
    for j in range(1, dim):
        v = M[:, j].copy()
        for k in range(j):
            v -= np.vdot(M[:, k], v) * M[:, k]
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            # val terug op een willekeurige standaardbasisvector
            for k in range(dim):
                e = np.zeros(dim, dtype=complex)
                e[k] = 1.0
                v = e.copy()
                for m in range(j):
                    v -= np.vdot(M[:, m], v) * M[:, m]
                if np.linalg.norm(v) > 1e-10:
                    break
            nrm = np.linalg.norm(v)
        M[:, j] = v / nrm
    return M


def prepare_unitary(coefs: Sequence[complex]) -> Tuple[np.ndarray, int]:
    """PREP op m_a ancilla-qubits: |0⟩ → Σ_i √(|c_i|/α) |i⟩ (reële amplituden).

    Fase van c_i wordt later door SELECT verwerkt.
    Retourneert (U_prep, m_a)."""
    L = len(coefs)
    m = max(1, int(np.ceil(np.log2(max(L, 2)))))
    dim = 2 ** m
    alpha = sum(abs(c) for c in coefs)
    if alpha == 0:
        raise ValueError("alpha=0: triviale Pauli-som")
    target = np.zeros(dim, dtype=complex)
    for i, c in enumerate(coefs):
        target[i] = np.sqrt(abs(c) / alpha)
    U = _householder_extend(target)
    return U, m


def select_unitary(
    coefs: Sequence[complex],
    pauli_labels: Sequence[str],
    n_qubits: int,
    m_ancilla: int,
) -> np.ndarray:
    """SELECT = Σ_i sign(c_i) · |i⟩⟨i|_a ⊗ P_i; ancilla i ≥ L → I (padding).

    Werkt in ancilla-eerst ordening: U[(i*dim_s + a), (j*dim_s + b)] voor
    ancilla-indices i,j en systeem-indices a,b."""
    dim_a = 2 ** m_ancilla
    dim_s = 2 ** n_qubits
    L = len(coefs)
    U = np.zeros((dim_a * dim_s, dim_a * dim_s), dtype=complex)
    for i in range(dim_a):
        proj = np.zeros((dim_a, dim_a), dtype=complex)
        proj[i, i] = 1.0
        if i < L:
            phase = coefs[i] / abs(coefs[i]) if abs(coefs[i]) > 0 else 1.0
            P = phase * pauli_matrix(pauli_labels[i])
        else:
            P = np.eye(dim_s, dtype=complex)  # padding
        U = U + np.kron(proj, P)
    return U


@dataclass
class BlockEncoding:
    """Block-encoding van H/α met m_a ancilla-qubits.

    Voor een unitaire U van grootte (2^m_a · 2^n, 2^m_a · 2^n) geldt:
        (⟨0|_a ⊗ I_s) U (|0⟩_a ⊗ I_s) = H / α.
    """

    U: np.ndarray
    m_ancilla: int
    n_qubits: int
    alpha: float

    def top_left_block(self) -> np.ndarray:
        """Extraheer (⟨0|_a ⊗ I) U (|0⟩_a ⊗ I) — verwacht = H/α."""
        dim_s = 2 ** self.n_qubits
        # ancilla |0⟩ ↔ index 0: blok U[0:dim_s, 0:dim_s] in ancilla-eerst ordening
        return self.U[:dim_s, :dim_s]


def block_encode_pauli_sum(ps: PauliSum) -> BlockEncoding:
    """Bouw LCU block-encoding van H = Σ c_i P_i; retourneert BlockEncoding."""
    if len(ps) == 0:
        raise ValueError("lege PauliSum kan niet block-encoded worden")
    coefs = [c for c, _ in ps.terms]
    labels = [lbl for _, lbl in ps.terms]
    PREP, m = prepare_unitary(coefs)
    SEL = select_unitary(coefs, labels, ps.n_qubits, m)
    dim_s = 2 ** ps.n_qubits
    # (PREP ⊗ I_s) in ancilla-eerst ordening
    PREP_full = np.kron(PREP, np.eye(dim_s, dtype=complex))
    U = PREP_full.conj().T @ SEL @ PREP_full
    return BlockEncoding(U=U, m_ancilla=m, n_qubits=ps.n_qubits, alpha=ps.alpha())


def verify_block_encoding(be: BlockEncoding, H: np.ndarray,
                          atol: float = 1e-10) -> bool:
    """Check of (⟨0|_a ⊗ I) U (|0⟩_a ⊗ I) = H/α en U unitair."""
    U = be.U
    # unitariteit
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=atol):
        return False
    block = be.top_left_block()
    expected = H / be.alpha
    return bool(np.allclose(block, expected, atol=atol))


# ============================================================
# 3. Quantum Signal Processing (QSP) — 1 qubit
# ============================================================

def qsp_signal(x: float) -> np.ndarray:
    """W(x) = [[x, i√(1-x²)], [i√(1-x²), x]]. Werkt als e^{iθ X} met cos θ = x."""
    if abs(x) > 1 + 1e-12:
        raise ValueError(f"QSP: |x|>1 (x={x})")
    x = max(-1.0, min(1.0, x))
    s = np.sqrt(max(0.0, 1 - x * x))
    return np.array([[x, 1j * s], [1j * s, x]], dtype=complex)


def rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(1j * phi), 0], [0, np.exp(-1j * phi)]], dtype=complex
    )


def qsp_unitary(x: float, phases: Sequence[float]) -> np.ndarray:
    """QSP-unitair U_Φ(x) = R_z(φ_0) · [ W(x) · R_z(φ_k) ]_{k=1..d}.

    Top-left element is p(x) voor een reëel/complex polynoom van graad ≤ d
    bepaald door de fase-sequentie Φ. Voorbeelden:
      - Φ = (0, 0):                  p(x) = x (identiteit-polynoom)
      - Φ = (-π/4, π/4, π/4, π/4):  p(x) = x² (Chebyshev T_2 scaled)
    """
    if len(phases) < 1:
        raise ValueError("phases must have at least 1 entry")
    W = qsp_signal(x)
    U = rz(phases[0])
    for phi in phases[1:]:
        U = U @ W @ rz(phi)
    return U


def qsp_polynomial_values(phases: Sequence[float],
                          xs: Sequence[float]) -> np.ndarray:
    """Evalueer p(x) = (U_Φ(x))[0,0] op een reeks punten x ∈ [-1, 1]."""
    vals = np.zeros(len(xs), dtype=complex)
    for i, x in enumerate(xs):
        vals[i] = qsp_unitary(x, phases)[0, 0]
    return vals


def chebyshev_T_phases(k: int) -> List[float]:
    """Fase-sequentie voor QSP-polynoom p(x) = T_k(x) (reëel, graad k).

    Met Φ = (0, 0, ..., 0) (k+1 nullen) geldt U_Φ(x) = W(x)^k en
      W(x)^k = cos(kθ)·I + i sin(kθ)·X   met x = cos θ,
    dus top-left = cos(kθ) = T_k(x).
    """
    return [0.0] * (k + 1)


# ============================================================
# 4. Chebyshev matrix-polynoom
# ============================================================

def chebyshev_T_matrix(k: int, A: np.ndarray) -> np.ndarray:
    """T_k(A) voor Hermitische A met ‖A‖ ≤ 1. Drie-term-recursie:
    T_0 = I,  T_1 = A,  T_{n+1} = 2 A T_n − T_{n-1}."""
    dim = A.shape[0]
    if k == 0:
        return np.eye(dim, dtype=complex)
    if k == 1:
        return A.astype(complex)
    T_prev = np.eye(dim, dtype=complex)
    T_curr = A.astype(complex)
    for _ in range(k - 1):
        T_next = 2 * A @ T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
    return T_curr


# ============================================================
# 5. Jacobi-Anger Hamiltonian-simulatie
# ============================================================

def bessel_j(k: int, x: float) -> float:
    """J_k(x) — gebruikt scipy.special.jv indien beschikbaar, anders
    een reeks-implementatie die stabiel is voor |x| ≲ 50."""
    if _HAS_SCIPY:
        return float(_bessel_jv(k, x))
    # Taylor-reeks: J_k(x) = Σ_{m=0}∞ (-1)^m / (m! (m+k)!) · (x/2)^{2m+k}
    if k < 0:
        return ((-1) ** k) * bessel_j(-k, x)
    s = 0.0
    term = (x / 2.0) ** k / float(np.math.factorial(k))
    for m in range(0, 60):
        if m == 0:
            s += term
        else:
            term *= -(x / 2.0) ** 2 / (m * (m + k))
            s += term
        if abs(term) < 1e-20:
            break
    return s


def jacobi_anger_truncation(tau: float, eps: float = 1e-12) -> int:
    """Smallest K zodat Σ_{k>K} 2|J_k(τ)| < ε.  Vuistregel: K ≈ e|τ|/2 + ln(1/ε)."""
    K = max(8, int(np.ceil(np.e * abs(tau) / 2.0 + np.log(1.0 / max(eps, 1e-30)))))
    # Empirische check: verhoog K tot rest-tail klein is.
    while abs(bessel_j(K, tau)) > eps and K < 400:
        K += 1
    return K


def hamiltonian_simulation_qsvt(H: np.ndarray, t: float, alpha: Optional[float] = None,
                                K: Optional[int] = None,
                                eps: float = 1e-12) -> np.ndarray:
    """Exponentieer e^{-iHt} via Jacobi-Anger Chebyshev-expansie.

    e^{-iτx} = J_0(τ) + 2 Σ_{k≥1} (-i)^k J_k(τ) T_k(x),   x = A, τ = αt.
    Truncatie bij K termen: spectrale fout ≤ 2 Σ_{k>K} |J_k(τ)|, exponentieel
    in K voor K > e|τ|/2.

    Args:
      H: Hermitische matrix
      t: tijd
      alpha: normalisatie (standaard: spectrale norm van H via eig). Als
             alpha < ||H||, faalt de recursie (buiten Chebyshev-domein).
      K: truncatie-graad (standaard via `jacobi_anger_truncation(α·t, eps)`).
    """
    if alpha is None:
        eigvals = np.linalg.eigvalsh((H + H.conj().T) / 2)
        alpha = float(max(abs(eigvals)))
    if alpha == 0:
        return np.eye(H.shape[0], dtype=complex)
    A = H / alpha
    tau = alpha * t
    if K is None:
        K = jacobi_anger_truncation(tau, eps)
    dim = A.shape[0]
    # Start-recursie
    U = bessel_j(0, tau) * np.eye(dim, dtype=complex)
    if K >= 1:
        T_prev = np.eye(dim, dtype=complex)  # T_0
        T_curr = A.astype(complex)           # T_1
        U = U + 2.0 * ((-1j) ** 1) * bessel_j(1, tau) * T_curr
        for k in range(2, K + 1):
            T_next = 2.0 * (A @ T_curr) - T_prev
            T_prev, T_curr = T_curr, T_next
            U = U + 2.0 * ((-1j) ** k) * bessel_j(k, tau) * T_curr
    return U


def trotter_reference(H_terms: List[Tuple[complex, np.ndarray]], t: float,
                      steps: int, order: int = 1) -> np.ndarray:
    """Directe dichte Trotter als baseline: voor elk fragment (c_k, P_k)
    bereken exp(-i · (t/steps) · c_k · P_k) en vermenigvuldig. Voor order=2
    gebruiken we de symmetrische Strang-splitting."""
    from scipy.linalg import expm
    dim = H_terms[0][1].shape[0]
    dt = t / steps
    U_step = np.eye(dim, dtype=complex)
    if order == 1:
        for c, P in H_terms:
            U_step = expm(-1j * dt * c * P) @ U_step
    elif order == 2:
        # S_2(dt) = Π_k e^{-i dt/2 H_k} · Π_k e^{-i dt/2 H_k}   (reversed)
        U1 = np.eye(dim, dtype=complex)
        for c, P in H_terms:
            U1 = expm(-1j * (dt / 2) * c * P) @ U1
        U2 = np.eye(dim, dtype=complex)
        for c, P in reversed(H_terms):
            U2 = expm(-1j * (dt / 2) * c * P) @ U2
        U_step = U2 @ U1
    else:
        raise ValueError(f"order {order} niet ondersteund (1 of 2)")
    U_tot = np.eye(dim, dtype=complex)
    for _ in range(steps):
        U_tot = U_step @ U_tot
    return U_tot


# ============================================================
# 6. CLI demo
# ============================================================

def _demo() -> int:
    print("=" * 72)
    print("  B160 QSVT / Block-Encoding Framework — demo")
    print("=" * 72)

    # Demo 1: block-encoding van een kleine Pauli-som
    ps = PauliSum(n_qubits=2, terms=[
        (1.0, "XX"),
        (0.5, "ZZ"),
        (-0.3, "IX"),
    ])
    H = ps.to_matrix()
    be = block_encode_pauli_sum(ps)
    ok = verify_block_encoding(be, H)
    print(f"\n[1] LCU block-encoding (2 qubits, 3 terms):")
    print(f"    alpha = {be.alpha:.4f}, m_ancilla = {be.m_ancilla}")
    print(f"    ||H||_op = {np.linalg.norm(H, 2):.4f}")
    print(f"    verify  = {'OK' if ok else 'FAIL'}")

    # Demo 2: QSP — polynoom T_2
    phases = chebyshev_T_phases(2)
    xs = np.linspace(-0.9, 0.9, 9)
    vals = qsp_polynomial_values(phases, xs)
    from numpy.polynomial.chebyshev import Chebyshev
    T2 = Chebyshev([0, 0, 1])
    print(f"\n[2] QSP T_2(x) (fases = {phases}):")
    print(f"    x        p(x) (QSP top-left)     T_2(x)·i^2   diff")
    for x, v in zip(xs, vals):
        t2 = T2(x) * (1j ** 2)
        print(f"    {x:+.2f}   {v.real:+.4f}{v.imag:+.4f}j     "
              f"{t2.real:+.4f}{t2.imag:+.4f}j     {abs(v - t2):.2e}")

    # Demo 3: Jacobi-Anger Ham-simulatie vs exact
    from scipy.linalg import expm
    H_mat = ps.to_matrix()
    alpha = ps.alpha()
    t = 0.5
    U_jaq = hamiltonian_simulation_qsvt(H_mat, t, alpha=alpha)
    U_exact = expm(-1j * H_mat * t)
    err = np.linalg.norm(U_jaq - U_exact, 2)
    print(f"\n[3] Jacobi-Anger Ham-sim @ t={t:.2f}, alpha={alpha:.3f}:")
    print(f"    ||U_JA - U_exact||_2 = {err:.2e}   (K=auto)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
