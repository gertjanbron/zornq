#!/usr/bin/env python3
"""B12 — Octonion-spinor correspondentie (split-octonionen ↔ fermionische Fock).

Onderzoeksvraag (zornq_backlog.md):
    "Bestaat er een constructieve relatie tussen de split-octonionische
     Zorn-representatie en fermionische Fock-ruimtes?"

Achtergrond: De kruisproduct-antisymmetrie α×β = −β×α, α×α = 0 heeft formeel
dezelfde structuur als fermionische anti-commutatie {c_i, c_j} = 0, c_i² = 0.
Split-octonionen zijn 8-dimensionaal; de fermionische Fock-ruimte op 3 modi
is ook 8-dimensionaal. De vraag is of dit meer is dan dimensie-toeval.

Dit bestand levert het numerieke antwoord:

  A. Split-octonion Zorn-algebra 𝕆_s met basis {e₀, j₁..j₆, e₇} waarvan:
     - e₀ = (1,0,0,0) en e₇ = (0,0,0,1) orthogonale primitieve idempotenten
       (Peirce-decompositie) met e₀·e₇ = e₇·e₀ = 0, e₀² = e₀, e₇² = e₇, e₀+e₇ = 1.
     - j₁, j₂, j₃ (α-basis) en j₄, j₅, j₆ (β-basis) zijn allen NILPOTENT: j_i² = 0.
     - Cross-product-relaties: j_i j_j = -j_k+3 voor (i,j,k) Levi-Civita-triplet.

  B. Fermionische Fock-ruimte F₃ = Λ(ℂ³) (8-dim). Creatie c_i† en annihilatie c_i
     voldoen aan {c_i, c_j†} = δ_ij en c_i² = 0.

  C. Constructieve linear bijection Φ: 𝕆_s → F₃ gedefinieerd door
         e₀ ↦ |0⟩,   j_i ↦ |i⟩ (i=1,2,3),
         j_{i+3} ↦ |jk⟩ (i,j,k cyclisch, met ε-teken),  e₇ ↦ -|123⟩.
     Φ is een iso als lineaire ruimten. Het is GEEN algebra-iso want octonionen
     zijn niet-associatief terwijl fermion-operatoren associatief zijn — dit
     wordt numeriek bevestigd via expliciete associator-norm.

  D. Cl(n) gamma-matrices uit LINKS-vermenigvuldiging L_a : 𝕆_s → 𝕆_s (8x8 reëel):
         γ_i = L(j_i + j_{i+3})    (i=1..3), γ_i² = +I
         γ_{i+3} = L(j_i - j_{i+3}) (i=1..3), γ_{i+3}² = -I
         γ_7 = L(e₇ - e₀),                    γ_7² = +I
     Deze 7 matrices genereren Cl(4,3) (signatuur (4,3)) op de 8-dim spinor-
     representatie 𝕆_s. Dit is de *correcte* Clifford-𝕆_s-koppeling (Baez 2002,
     Dray-Manogue), niet de incorrecte claim Cl(4,4) ≅ 𝕆_s (256-dim vs 8-dim).

  E. Triality-indicatie: L en R (rechts-vermenigvuldiging) geven twee niet-
     equivalente 8-dim representaties van Im(𝕆_s). Samen met de vector-rep V
     (adjoint-actie) is dit het drieluik V, S⁺, S⁻ van de triality-automorfisme
     van Spin(3,4) (split-octonion versie van Baez 2002, §3.3).

Licentie: Onderzoekscode voor ZornQ-project.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


# ============================================================
#  1. Zorn split-octonion algebra
# ============================================================

def zorn_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Split-octonion product in Zorn-vector-vorm (a, α, β, b) ∈ ℝ⁸.

    A = (a, α, β, b),  B = (c, γ, δ, d)  met a,b,c,d scalair en α,β,γ,δ ∈ ℝ³.
    Product gedefinieerd door de Zorn-matrixalgebra:
        (A·B) = (a·c + α·δ,   a·γ + d·α + β×δ,   c·β + b·δ - α×γ,   β·γ + b·d)
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    a, al, be, b = A[0], A[1:4], A[4:7], A[7]
    c, ga, de, d = B[0], B[1:4], B[4:7], B[7]
    out = np.empty(8, dtype=float)
    out[0] = a * c + al @ de
    out[1:4] = a * ga + d * al + np.cross(be, de)
    out[4:7] = c * be + b * de - np.cross(al, ga)
    out[7] = be @ ga + b * d
    return out


def zorn_conjugate(A: np.ndarray) -> np.ndarray:
    """Conjugatie: (a, α, β, b) → (b, -α, -β, a).

    Voldoet aan (AB)* = B*A* en N(A) = A·A* als scalar."""
    A = np.asarray(A, dtype=float)
    return np.array([A[7], -A[1], -A[2], -A[3], -A[4], -A[5], -A[6], A[0]])


def zorn_norm(A: np.ndarray) -> float:
    """N(a, α, β, b) = ab - α·β. Signatuur (4,4) op 𝕆_s.

    Gelijk aan het 'scalar-deel' van A·A* = A*·A."""
    A = np.asarray(A, dtype=float)
    return float(A[0] * A[7] - A[1:4] @ A[4:7])


def zorn_trace(A: np.ndarray) -> float:
    """Reduced trace: tr(a, α, β, b) = a + b."""
    A = np.asarray(A, dtype=float)
    return float(A[0] + A[7])


def associator(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[A, B, C] = (AB)C - A(BC).

    Nul iff A, B, C in een associatieve subalgebra. Non-triviaal op 𝕆_s bewijst
    niet-associativiteit."""
    return zorn_mul(zorn_mul(A, B), C) - zorn_mul(A, zorn_mul(B, C))


# Canonieke basis van 𝕆_s: 8 eenheidsvectoren langs de 8 Zorn-coördinaten.
#   e₀ = (1,0,0,0),  j₁ = (0,ê₁,0,0),  j₂ = (0,ê₂,0,0),  j₃ = (0,ê₃,0,0),
#   j₄ = (0,0,ê₁,0), j₅ = (0,0,ê₂,0), j₆ = (0,0,ê₃,0),   e₇ = (0,0,0,1).
def basis_vector(k: int) -> np.ndarray:
    """k=0 → e₀, k=1..3 → j_k (α-basis), k=4..6 → j_k (β-basis), k=7 → e₇."""
    v = np.zeros(8, dtype=float)
    v[k] = 1.0
    return v


# Identiteitselement 1 = e₀ + e₇.
def zorn_identity() -> np.ndarray:
    return np.array([1.0, 0, 0, 0, 0, 0, 0, 1.0])


# Alias-basis: BASIS_NAMES[k] = standaard naam (voor diagnostiek)
BASIS_NAMES = ["e0", "j1", "j2", "j3", "j4", "j5", "j6", "e7"]


# ============================================================
#  2. Links- en rechts-vermenigvuldigingsmatrices
# ============================================================

def L_matrix(A: np.ndarray) -> np.ndarray:
    """8×8 reële matrix L_A met (L_A x) = A · x voor alle x ∈ 𝕆_s."""
    M = np.empty((8, 8), dtype=float)
    for k in range(8):
        M[:, k] = zorn_mul(A, basis_vector(k))
    return M


def R_matrix(A: np.ndarray) -> np.ndarray:
    """8×8 reële matrix R_A met (R_A x) = x · A."""
    M = np.empty((8, 8), dtype=float)
    for k in range(8):
        M[:, k] = zorn_mul(basis_vector(k), A)
    return M


def anticommutator(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """{X, Y} = XY + YX voor 8×8 matrices."""
    return X @ Y + Y @ X


def commutator(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y - Y @ X


# ============================================================
#  3. Fermionische Fock-ruimte op 3 modi (Jordan-Wigner)
# ============================================================

# Basisordering in Fock-ruimte F₃ = Λ(ℂ³):
#   index 0: |∅⟩ = |000⟩  (vacuum)
#   index 1: |1⟩ = |100⟩  = c_1† |0⟩
#   index 2: |2⟩ = |010⟩  = c_2† |0⟩
#   index 3: |3⟩ = |001⟩  = c_3† |0⟩
#   index 4: |12⟩ = c_1† c_2† |0⟩   (JW-teken: +1)
#   index 5: |13⟩ = c_1† c_3† |0⟩
#   index 6: |23⟩ = c_2† c_3† |0⟩
#   index 7: |123⟩ = c_1† c_2† c_3† |0⟩
#
# We gebruiken bitmask s ⊆ {1,2,3} → index = popcnt-lexicografische rangorde
# onder de voorwaarde dat we deze exacte volgorde blijven gebruiken.

FOCK_BASIS = [
    frozenset(),              # 0: |∅⟩
    frozenset({1}),           # 1
    frozenset({2}),           # 2
    frozenset({3}),           # 3
    frozenset({1, 2}),        # 4
    frozenset({1, 3}),        # 5
    frozenset({2, 3}),        # 6
    frozenset({1, 2, 3}),     # 7
]
FOCK_INDEX = {s: i for i, s in enumerate(FOCK_BASIS)}


def fock_creation(i: int) -> np.ndarray:
    """c_i† als 8x8 reële matrix met Jordan-Wigner-tekens (JW-ordening 1<2<3).

    c_i† |σ⟩ = 0 als i∈σ, anders = (-1)^{#{k∈σ : k<i}} |σ ∪ {i}⟩.
    """
    assert 1 <= i <= 3
    M = np.zeros((8, 8), dtype=float)
    for col, sigma in enumerate(FOCK_BASIS):
        if i in sigma:
            continue
        # Tellen van hoeveel bezette modi k met k<i al aanwezig zijn
        sign = (-1) ** sum(1 for k in sigma if k < i)
        new_sigma = sigma | {i}
        row = FOCK_INDEX[new_sigma]
        M[row, col] = sign
    return M


def fock_annihilation(i: int) -> np.ndarray:
    """c_i = (c_i†)†. Omdat de basis reëel is, is dit gewoon de transpose."""
    return fock_creation(i).T


def fock_number(i: int) -> np.ndarray:
    """n_i = c_i† c_i."""
    return fock_creation(i) @ fock_annihilation(i)


# ============================================================
#  4. Constructieve bijection Φ : 𝕆_s → F₃
# ============================================================
#
# Observeer de volgende structuur:
#   j_i  (i=1,2,3)  ↔  c_i† |0⟩ = |i⟩
#   e₀              ↔  |0⟩
#   j_i j_j         ↔  c_i† c_j† |0⟩ = |ij⟩   (match in teken na ε-permutatie)
#   j_i j_j j_k     ↔  c_1† c_2† c_3† |0⟩ = |123⟩  (linker-bracket-afhankelijk)
#
# We bouwen Φ expliciet uit als 8×8 matrix.

def phi_bijection() -> np.ndarray:
    """8×8 matrix van Φ: 𝕆_s → F₃. Kolommen van Φ zijn beelden van de
    8 basisvectoren van 𝕆_s geschreven in de Fock-basis."""
    M = np.zeros((8, 8), dtype=float)
    #   Φ(e₀) = |∅⟩  → index 0
    M[FOCK_INDEX[frozenset()], 0] = 1.0
    #   Φ(j_i) = |i⟩  voor i=1,2,3
    for i in (1, 2, 3):
        M[FOCK_INDEX[frozenset({i})], i] = 1.0
    #   Φ(j_{i+3}) = σ_i · |j k⟩  waarbij (i,j,k) cyclisch.
    # We kiezen de tekens zodanig dat j_i · j_j = ±j_{k+3} onder Zorn-product
    # en c_i† c_j† |0⟩ = |ij⟩ onder fermion-algebra beide *dezelfde* Levi-Civita-
    # conventie weerspiegelen. Expliciete keuze:
    #   j₁·j₂ = -j₆ (in β-basis), dus Φ(j_6) = -|12⟩; idem
    #   j₂·j₃ = -j₄             →  Φ(j_4) = -|23⟩
    #   j₃·j₁ = -j_5            →  Φ(j_5) = -|13⟩   (cyclisch)
    M[FOCK_INDEX[frozenset({2, 3})], 4] = -1.0   # Φ(j_4) = -|23⟩
    M[FOCK_INDEX[frozenset({1, 3})], 5] = -1.0   # Φ(j_5) = -|13⟩
    M[FOCK_INDEX[frozenset({1, 2})], 6] = -1.0   # Φ(j_6) = -|12⟩
    #   Φ(e₇) = -|123⟩  (want (j_1 j_2) j_3 = -e_7 in links-bracket-conventie)
    M[FOCK_INDEX[frozenset({1, 2, 3})], 7] = -1.0
    return M


# ============================================================
#  5. Cl(n) gamma-matrices uit L-multiplication
# ============================================================

def clifford_generators_7d() -> list[np.ndarray]:
    """Construeer 7 Clifford-generatoren uit L-multiplications op 𝕆_s.

    Conventie:  (j_i + j_{i+3})² = 1     (split-reëel, γ_i² = +I)
                (j_i - j_{i+3})² = -1    (split-imaginair, γ_{i+3}² = -I)
                (e₇ - e₀)² = 1           (γ_7² = +I)

    Levert 7 matrices γ₁..γ₇ die op de 8-dim spinor-rep 𝕆_s een Clifford-
    algebra Cl(4,3) genereren: {γ_μ, γ_ν} = 2 η_{μν} I met η = diag(+,+,+,-,-,-,+).
    """
    gens = []
    for i in (1, 2, 3):
        a_plus = basis_vector(i) + basis_vector(i + 3)
        gens.append(L_matrix(a_plus))
    for i in (1, 2, 3):
        a_minus = basis_vector(i) - basis_vector(i + 3)
        gens.append(L_matrix(a_minus))
    a7 = basis_vector(7) - basis_vector(0)   # e_7 - e_0
    gens.append(L_matrix(a7))
    return gens


def clifford_metric(gens: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    """Return Gram-achtige metric η met η_{μν} = ½·{γ_μ, γ_ν}_{11}  (numeriek).

    Voor correcte Clifford-generatoren zijn alle {γ_μ, γ_ν} een scalair·I,
    zodat η_{μν} uniek bepaald is."""
    n = len(gens)
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            AC = anticommutator(gens[i], gens[j])
            # Test proportional to I
            avg = np.mean(np.diag(AC))
            off = AC - avg * np.eye(AC.shape[0])
            if np.max(np.abs(off)) > tol:
                raise ValueError(
                    f"{{γ_{i},γ_{j}}} niet proportional to I "
                    f"(off-diag residual {np.max(np.abs(off)):.3e})")
            eta[i, j] = avg / 2.0
    return eta


# ============================================================
#  6. Falsificatie-helpers
# ============================================================

def clifford_algebra_dim(p: int, q: int) -> int:
    """Dimensie van Cl(p, q) als reële algebra: 2^{p+q}."""
    return 2 ** (p + q)


# ============================================================
#  7. Moufang-test (verifieert alternatief-algebra-structuur)
# ============================================================

def moufang_identity_left(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                          tol: float = 1e-10) -> float:
    """Links-Moufang: A(B(AC)) = ((AB)A)C  (Moufang 1935).

    Returnt de norm van het residu. Voor 𝕆_s strikt nul (alternatief)."""
    lhs = zorn_mul(A, zorn_mul(B, zorn_mul(A, C)))
    rhs = zorn_mul(zorn_mul(zorn_mul(A, B), A), C)
    return float(np.linalg.norm(lhs - rhs))


def moufang_identity_right(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                           tol: float = 1e-10) -> float:
    """Rechts-Moufang: ((CA)B)A = C(A(BA))."""
    lhs = zorn_mul(zorn_mul(zorn_mul(C, A), B), A)
    rhs = zorn_mul(C, zorn_mul(A, zorn_mul(B, A)))
    return float(np.linalg.norm(lhs - rhs))


# ============================================================
#  8. Triality-indicator (L vs R equivalentie)
# ============================================================

def triality_discrepancy() -> dict:
    """Meet hoezeer de L- en R-representaties van Im(𝕆_s) verschillen.

    Voor een associatieve algebra is L(a) = R(a)^T (of tenminste conjugaat).
    Voor 𝕆_s is L ≠ R omdat links- en rechts-multiplication op een niet-
    associatieve algebra verschillende reps geven. De Spin(4,4)-triality
    verbindt L, R en de vector-rep (adjoint)."""
    out = {}
    total_diff = 0.0
    for k in range(1, 8):
        a = basis_vector(k)
        Lm = L_matrix(a)
        Rm = R_matrix(a)
        # L en R zouden in associatieve algebra (bijv. Clifford) of gelijk zijn
        # of elkaars transpose. Voor 𝕆_s meten we ||L - R||, ||L - R^T||.
        d1 = np.linalg.norm(Lm - Rm)
        d2 = np.linalg.norm(Lm - Rm.T)
        total_diff += d1
        out[BASIS_NAMES[k]] = (d1, d2)
    out["__total_L_minus_R__"] = total_diff
    return out


# ============================================================
#  9. Hoofd-CLI / demo
# ============================================================

def _demo() -> None:
    print("== B12 — Octonion-spinor correspondentie (demo) ==")
    # Peirce-idempotenten
    e0 = basis_vector(0)
    e7 = basis_vector(7)
    print(f"  e0 * e0 == e0 : {np.allclose(zorn_mul(e0, e0), e0)}")
    print(f"  e7 * e7 == e7 : {np.allclose(zorn_mul(e7, e7), e7)}")
    print(f"  e0 * e7 == 0  : {np.allclose(zorn_mul(e0, e7), np.zeros(8))}")
    print(f"  e7 * e0 == 0  : {np.allclose(zorn_mul(e7, e0), np.zeros(8))}")
    print(f"  e0 + e7 == 1  : "
          f"{np.allclose(e0 + e7, zorn_identity())}")
    # Nilpotente imaginairen
    for k in (1, 2, 3, 4, 5, 6):
        jk = basis_vector(k)
        sq = zorn_mul(jk, jk)
        print(f"  j{k} * j{k} == 0 : {np.allclose(sq, np.zeros(8))}")
    # Fermion-achtige anti-commutator
    j1 = basis_vector(1); j4 = basis_vector(4)
    print(f"  j1·j4 + j4·j1 == 1 (identity) : "
          f"{np.allclose(zorn_mul(j1, j4) + zorn_mul(j4, j1), zorn_identity())}")
    # Cl(4,3)-generatoren
    gens = clifford_generators_7d()
    eta = clifford_metric(gens)
    print(f"  Cl(n)-signatuur (diag van η): {np.diag(eta).astype(int).tolist()}")
    # Bijection
    Phi = phi_bijection()
    print(f"  Φ unitair/orthogonaal? |det Φ| = {abs(np.linalg.det(Phi)):.3f}")
    # Associator voorbeeld
    j1 = basis_vector(1); j2 = basis_vector(2); j3 = basis_vector(3)
    ass = associator(j1, j2, j3)
    print(f"  [j1,j2,j3] = {ass.astype(int).tolist()}  (non-zero → "
          f"niet-associatief)")


if __name__ == "__main__":
    _demo()
