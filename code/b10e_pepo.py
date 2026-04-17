#!/usr/bin/env python3
"""B10e: PEPO / PEPS — 2D tensor network voor vierkante roosters.

Achtergrond
-----------
MPS/MPO engines (B10, B10b, B10d) werken per "kolom-groep": bij Ly=2,3 worden
kolom-qubits samengevoegd tot lokale dimensie d=2^Ly. Voor Ly≥5 (d≥32) wordt
dat onhanteerbaar — zowel qua gate-tensor grootte (64^2) als qua SVD-kosten.

PEPS (Projected Entangled Pair States) lost dit op door elke site een tensor
te geven met 4 virtuele bonds (links/rechts/boven/onder) + 1 fysieke leg.
De 2D structuur komt terug in de tensor-topologie i.p.v. in de lokale
fysieke dimensie.

Deze module implementeert:

1. `PEPS2D`: data-structuur voor een 2D PEPS op een Lx×Ly rooster.
2. Single-site + two-site gate applicatie (horizontaal + verticaal) met
   "simple update" SVD-truncatie.
3. Boundary-MPO contractie voor exacte / benaderde expectation values
   ⟨ψ|O|ψ⟩ — met iteratieve kolom-sweep + MPS-achtige boundary compressie.
4. 2D MaxCut QAOA pipeline die PEPS gebruikt voor state-evolutie.
5. Exact state-vector referentie voor Lx·Ly ≤ 16 qubits (validatie).

Leg-conventies
--------------
`PEPS2D` slaat tensors op als `self.T[x][y]` met shape
`(D_L, D_R, D_U, D_D, d)`, waarbij randsites bond-dimensie 1 hebben op de
ontbrekende richting. Fysieke leg is d=2 (qubit).

Coordinaten: x∈[0,Lx), y∈[0,Ly). (x+1,y) is rechts, (x,y+1) is beneden.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd
from typing import Iterable

# ============================================================
# Pauli / gate primitives
# ============================================================

I2 = np.eye(2, dtype=complex)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ZZg(gamma: float) -> np.ndarray:
    """Reshape(2,2,2,2) diagonale exp(-i γ Z⊗Z)."""
    d = np.array([np.exp(-1j * gamma), np.exp(1j * gamma),
                  np.exp(1j * gamma), np.exp(-1j * gamma)], dtype=complex)
    return np.diag(d).reshape(2, 2, 2, 2)


# ============================================================
# PEPS2D data structure
# ============================================================

class PEPS2D:
    """2D PEPS op een Lx×Ly vierkant rooster.

    Tensor `T[x][y]` heeft shape `(D_L, D_R, D_U, D_D, d)`.
    Op de rand is de ontbrekende bond-dimensie 1.
    """

    def __init__(self, Lx: int, Ly: int, chi_max: int = 8, d: int = 2):
        self.Lx = Lx
        self.Ly = Ly
        self.chi_max = chi_max
        self.d = d
        self.T: list[list[np.ndarray]] = [
            [None for _ in range(Ly)] for _ in range(Lx)  # type: ignore
        ]

    # ------------------------------------------------------
    # Constructors
    # ------------------------------------------------------

    @classmethod
    def from_product_vec(cls, Lx: int, Ly: int, vec: np.ndarray,
                         chi_max: int = 8) -> "PEPS2D":
        """PEPS met op elke site hetzelfde product-state vector `vec` (d,)."""
        p = cls(Lx, Ly, chi_max=chi_max, d=vec.size)
        v = vec.astype(complex).reshape(1, 1, 1, 1, vec.size)
        for x in range(Lx):
            for y in range(Ly):
                p.T[x][y] = v.copy()
        return p

    @classmethod
    def plus_state(cls, Lx: int, Ly: int, chi_max: int = 8) -> "PEPS2D":
        """|+⟩^⊗n op een Lx×Ly rooster."""
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        return cls.from_product_vec(Lx, Ly, plus, chi_max=chi_max)

    @classmethod
    def zero_state(cls, Lx: int, Ly: int, chi_max: int = 8) -> "PEPS2D":
        """|0⟩^⊗n."""
        v = np.array([1.0, 0.0], dtype=complex)
        return cls.from_product_vec(Lx, Ly, v, chi_max=chi_max)

    # ------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------

    def copy(self) -> "PEPS2D":
        out = PEPS2D(self.Lx, self.Ly, chi_max=self.chi_max, d=self.d)
        for x in range(self.Lx):
            for y in range(self.Ly):
                out.T[x][y] = self.T[x][y].copy()
        return out

    def max_bond_dim(self) -> int:
        m = 1
        for x in range(self.Lx):
            for y in range(self.Ly):
                shp = self.T[x][y].shape
                m = max(m, shp[0], shp[1], shp[2], shp[3])
        return m

    def shapes(self) -> list[tuple]:
        """Flat list van tensor-shapes (debug-hulp)."""
        return [self.T[x][y].shape for x in range(self.Lx)
                for y in range(self.Ly)]

    # ------------------------------------------------------
    # Single-site gate
    # ------------------------------------------------------

    def apply_single(self, x: int, y: int, g: np.ndarray) -> None:
        """Pas een 1-site gate g (d×d) toe op site (x,y)."""
        T = self.T[x][y]  # (L,R,U,D,d)
        # contract physical leg
        new = np.einsum("ab,LRUDb->LRUDa", g, T)
        self.T[x][y] = new

    # ------------------------------------------------------
    # Two-site gates: horizontaal + verticaal
    # ------------------------------------------------------

    def apply_two_horizontal(self, x: int, y: int, g: np.ndarray,
                             chi_max: int | None = None) -> None:
        """2-site gate op bond (x,y)-(x+1,y).

        g heeft shape (d,d,d,d) = g[a,b,a',b'] = ⟨a,b|g|a',b'⟩ waarbij
        eerste index = site (x,y), tweede = site (x+1,y).

        Simple-update: merge twee sites, apply gate, SVD over de gedeelde
        bond, truncate tot chi_max.
        """
        if x + 1 >= self.Lx:
            raise ValueError("Geen rechter-buurman op x={}".format(x))
        chi_max = chi_max if chi_max is not None else self.chi_max

        TL = self.T[x][y]       # (Ll, Lr, Lu, Ld, d)  — Lr = gedeelde bond
        TR = self.T[x + 1][y]   # (Rl, Rr, Ru, Rd, d)  — Rl = gedeelde bond
        Ll, Lr, Lu, Ld, d = TL.shape
        Rl, Rr, Ru, Rd, _ = TR.shape
        assert Lr == Rl, "Gedeelde bond dim mismatch: {} vs {}".format(Lr, Rl)

        # Merge over shared bond via tensordot:
        #   TL[Ll,Lr,Lu,Ld,d_L]  ×  TR[Rl,Rr,Ru,Rd,d_R]  over Lr=Rl
        # shape na: (Ll, Lu, Ld, d_L, Rr, Ru, Rd, d_R)
        M = np.tensordot(TL, TR, axes=([1], [0]))

        # Apply gate: g heeft shape (d,d,d,d) = g[a,b,a',b']
        # Contract d_L (M-axis 3) met a', d_R (M-axis 7) met b'.
        # Simpelste route: permuteer M → (Ll,Lu,Ld,Rr,Ru,Rd, d_L, d_R),
        # flatten (d_L,d_R), matmul met g_flat, reshape terug.
        M = np.transpose(M, (0, 1, 2, 4, 5, 6, 3, 7))
        g_flat = g.reshape(d * d, d * d)   # rij-index = (a,b), kol-index = (a',b')
        M2 = M.reshape(Ll * Lu * Ld * Rr * Ru * Rd, d * d)
        # (..., a'b') @ g_flat.T  →  (..., ab)   (row-major flatten)
        M2 = M2 @ g_flat.T
        M = M2.reshape(Ll, Lu, Ld, Rr, Ru, Rd, d, d)
        # Zet fysieke legs terug naar hun originele posities:
        # (Ll,Lu,Ld, d_L, Rr, Ru, Rd, d_R)
        M = np.transpose(M, (0, 1, 2, 6, 3, 4, 5, 7))

        # SVD over gedeelde bond: A = (Ll,Lu,Ld,d_L) ; B = (Rr,Ru,Rd,d_R)
        A_dim = Ll * Lu * Ld * d
        B_dim = Rr * Ru * Rd * d
        mat = M.reshape(A_dim, B_dim)
        U, S, Vh = svd(mat, full_matrices=False)
        # Drop zero singular values (below relative tol) om ruis in de bond te voorkomen
        if S.size > 0:
            s_tol = max(S[0], 1.0) * 1e-12
            keep_nonzero = int((S > s_tol).sum())
        else:
            keep_nonzero = 0
        new_D = max(1, min(chi_max, keep_nonzero))
        U = U[:, :new_D]
        S = S[:new_D]
        Vh = Vh[:new_D, :]

        # Split S evenredig: half in U, half in Vh (absorption keuze)
        sqrtS = np.sqrt(S + 0.0j)
        U = U * sqrtS
        Vh = (sqrtS[:, None]) * Vh

        # Reshape terug
        TL_new = U.reshape(Ll, Lu, Ld, d, new_D)
        # Legs: (Ll, Lu, Ld, d, Lr_new) — zet Lr_new op plaats 1: (Ll,Lr_new,Lu,Ld,d)
        TL_new = np.transpose(TL_new, (0, 4, 1, 2, 3))

        TR_new = Vh.reshape(new_D, Rr, Ru, Rd, d)
        # Legs: (Rl_new, Rr, Ru, Rd, d)
        self.T[x][y] = TL_new
        self.T[x + 1][y] = TR_new

    def apply_two_vertical(self, x: int, y: int, g: np.ndarray,
                           chi_max: int | None = None) -> None:
        """2-site gate op bond (x,y)-(x,y+1).

        Analoog aan horizontaal maar voor up/down bonds.
        """
        if y + 1 >= self.Ly:
            raise ValueError("Geen onder-buurman op y={}".format(y))
        chi_max = chi_max if chi_max is not None else self.chi_max

        TU = self.T[x][y]       # (Ll, Lr, Lu, Ld, d) — Ld = gedeelde bond
        TD = self.T[x][y + 1]   # (Rl, Rr, Ru, Rd, d) — Ru = gedeelde bond
        Ul, Ur, Uu, Ud, d = TU.shape
        Dl, Dr, Du, Dd, _ = TD.shape
        assert Ud == Du, "Gedeelde bond dim mismatch: {} vs {}".format(Ud, Du)

        # Contract over gedeelde bond (TU axis 3, TD axis 2)
        M = np.tensordot(TU, TD, axes=([3], [2]))
        # shape: (Ul, Ur, Uu, d_U, Dl, Dr, Dd, d_D)

        # Permute → fysieke legs achteraan:
        # (Ul, Ur, Uu, Dl, Dr, Dd, d_U, d_D)
        M = np.transpose(M, (0, 1, 2, 4, 5, 6, 3, 7))

        g4 = g.reshape(d * d, d * d)
        M2 = M.reshape(Ul * Ur * Uu * Dl * Dr * Dd, d * d)
        M2 = M2 @ g4.T
        M = M2.reshape(Ul, Ur, Uu, Dl, Dr, Dd, d, d)
        # Terugzetten: (Ul, Ur, Uu, d_U, Dl, Dr, Dd, d_D)
        M = np.transpose(M, (0, 1, 2, 6, 3, 4, 5, 7))

        # SVD: A = (Ul, Ur, Uu, d_U) ; B = (Dl, Dr, Dd, d_D)
        A_dim = Ul * Ur * Uu * d
        B_dim = Dl * Dr * Dd * d
        mat = M.reshape(A_dim, B_dim)
        U, S, Vh = svd(mat, full_matrices=False)
        if S.size > 0:
            s_tol = max(S[0], 1.0) * 1e-12
            keep_nonzero = int((S > s_tol).sum())
        else:
            keep_nonzero = 0
        new_D = max(1, min(chi_max, keep_nonzero))
        U = U[:, :new_D]
        S = S[:new_D]
        Vh = Vh[:new_D, :]
        sqrtS = np.sqrt(S + 0.0j)
        U = U * sqrtS
        Vh = sqrtS[:, None] * Vh

        TU_new = U.reshape(Ul, Ur, Uu, d, new_D)
        # Zet new_D (= new Ud) op plaats 3: already is (Ul,Ur,Uu,d,D) → (Ul,Ur,Uu,D,d)
        TU_new = np.transpose(TU_new, (0, 1, 2, 4, 3))
        TD_new = Vh.reshape(new_D, Dl, Dr, Dd, d)
        # Zet new_D (= new Du) op plaats 2: (D,Dl,Dr,Dd,d) → (Dl,Dr,D,Dd,d)
        TD_new = np.transpose(TD_new, (1, 2, 0, 3, 4))

        self.T[x][y] = TU_new
        self.T[x][y + 1] = TD_new


# ============================================================
# Boundary-MPO contractie voor ⟨ψ|∏O_i|ψ⟩
# ============================================================

def _site_double(T: np.ndarray, op: np.ndarray | None = None) -> np.ndarray:
    """Bouw dubbele-laag tensor (bra⊗ket) op één site, optioneel met operator.

    T: (L,R,U,D,d)  — PEPS tensor
    op: (d,d) of None. Indien None, gebruik identiteit.

    Return: (L², R², U², D²)  — gecombineerde bra/ket bonds.
    """
    L, R, U, D, d = T.shape
    Tc = T.conj()
    if op is None:
        op = np.eye(d, dtype=complex)
    # O[a,b] × T[L,R,U,D,b] × T*[L',R',U',D',a]
    # contract: ket@op → T_op[L,R,U,D,a] = sum_b op[a,b] * T[...,b]
    T_op = np.einsum("ab,LRUDb->LRUDa", op, T)
    # bra (conj) × T_op over a:
    # E[L,L',R,R',U,U',D,D'] = sum_a T_op[L,R,U,D,a] * Tc[L',R',U',D',a]
    E = np.einsum("LRUDa,MNOPa->LMRNUODP", T_op, Tc)
    # combineer bra/ket bonds
    out = E.reshape(L * L, R * R, U * U, D * D)
    return out


def _contract_column_to_boundary(col_doubles: list[np.ndarray],
                                 left_boundary: list[np.ndarray],
                                 chi_b: int) -> list[np.ndarray]:
    """Absorbeer een kolom van doubles in de links-boundary MPS.

    col_doubles[y]: (L, R, U, D)   voor y=0..Ly-1
    left_boundary[y]: (bl, br, L_y)  (boundary MPS tensoren — L_y matcht col.L)

    Return: nieuwe boundary-MPS, gecomprimeerd tot chi_b.
    """
    Ly = len(col_doubles)
    assert len(left_boundary) == Ly

    # 1) Contract elk col_doubles[y] met boundary[y] over de linker-bond L.
    new_bd: list[np.ndarray] = []
    for y in range(Ly):
        B = left_boundary[y]       # (bl, br, L)
        C = col_doubles[y]         # (L, R, U, D)
        # M = einsum("bl,br,L; L,R,U,D -> bl,br,R,U,D", B, C)
        M = np.einsum("ijL,LRUD->ijRUD", B, C)
        new_bd.append(M)

    # 2) Contract verticaal: verbind U van y met D van y-1.
    # Volg standaard MPS-vorm: elke tensor krijgt shape (left_bond, right_bond, phys)
    # waarbij phys hier de RIGHT-bond R is van col_doubles.
    # Verticale contractie: U[y] * D[y-1] verbindt; maak nieuwe MPS met
    # gecombineerde bra/ket kolom-bonds.
    # new_bd[y] heeft shape (bl, br, R, U, D).
    # We moeten (U_y, D_{y-1}) contraheren.
    # Bouw nieuwe MPS: phys = R; left/right bond = (bl, br) × (U of D) gecombineerd.
    # Aanpak: "fuse" met bouw van MPS_form[y] met:
    #   shape (lbond, rbond, phys) waarbij lbond = bl * U, rbond = br * D, phys = R.
    # Verticale matching gebeurt via MPS-compressie (zie stap 3).
    mps_form: list[np.ndarray] = []
    for y in range(Ly):
        bl, br, R, U, D = new_bd[y].shape
        # Order: (bl, U, br, D, R)  — zodat left=(bl,U) en right=(br,D), phys=R
        T = np.transpose(new_bd[y], (0, 3, 1, 4, 2))
        T = T.reshape(bl * U, br * D, R)
        mps_form.append(T)

    # 3) Boundary contraction in verticale richting is impliciet: in de
    # standaard boundary-MPO contractie plaatsen we de kolom als
    # "column MPO" en contraheren met de boundary-MPS over R (rechter-bond).
    # In onze formulering is de "boundary MPS" *al* links van de kolom, en
    # we absorberen de kolom erin. De physische leg van de nieuwe boundary
    # is dus R (rechter-bond van de kolom).
    # Verticale MPO-contractie: voor y=1..Ly-1, contract de (D)-tak van
    # tensor y-1 met de (U)-tak van tensor y. In mps_form zit D in de
    # "right" bond van y-1 en U in de "left" bond van y. Dit betekent dat
    # left-right bonds automatisch matchen in de MPS-vorm wanneer we elkaar
    # opvolgend naar rechts schuiven... maar in 1-MPS-vorm is dat juist
    # de chain-structuur die we willen.
    # Om double-layer (ket+bra) vertical bonds te "sluiten", moeten we
    # ook intern D_y met U_{y+1} matchen. In onze huidige coding zit D op
    # "right" van y en U op "left" van y+1, dus in MPS-chain opeenvolging
    # is dat consistent.

    # 4) Compressie: canonicaliseer + SVD-truncatie tot chi_b.
    mps_form = _mps_canonicalize(mps_form, chi_max=chi_b)
    return mps_form


def _mps_canonicalize(mps: list[np.ndarray], chi_max: int) -> list[np.ndarray]:
    """Breng een open-boundary MPS in gemengde-canonical vorm en trunceer.

    Input: lijst van tensors shape (l, r, p).
    Output: zelfde lijst, maar met bond-dim ≤ chi_max via SVD-sweep.
    """
    n = len(mps)
    if n == 0:
        return mps
    # Left-canonical sweep
    for i in range(n - 1):
        T = mps[i]  # (l, r, p)
        l, r, p = T.shape
        M = T.transpose(0, 2, 1).reshape(l * p, r)
        U, S, Vh = svd(M, full_matrices=False)
        keep = min(chi_max, S.size)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        # New tensor
        mps[i] = U.reshape(l, p, keep).transpose(0, 2, 1)
        # Absorb SVh into next
        SVh = np.diag(S) @ Vh
        T_next = mps[i + 1]
        l2, r2, p2 = T_next.shape
        # Matmul over r2? No: SVh @ T_next[l2=?, ...]
        # SVh shape: (keep, r) — r = originele rechter bond van T.
        # T_next[l2, r2, p2] met l2 = r. We willen (keep, r2, p2).
        T_next_new = np.einsum("ab,bcd->acd", SVh, T_next)
        mps[i + 1] = T_next_new
    # Right-canonical sweep (nu alle tensoren ≤ chi_max)
    for i in range(n - 1, 0, -1):
        T = mps[i]  # (l, r, p)
        l, r, p = T.shape
        M = T.reshape(l, r * p)
        U, S, Vh = svd(M, full_matrices=False)
        keep = min(chi_max, S.size)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        mps[i] = Vh.reshape(keep, r, p)
        US = U @ np.diag(S)
        T_prev = mps[i - 1]
        mps[i - 1] = np.einsum("abc,bd->adc", T_prev, US)
    return mps


def expectation_value(peps: PEPS2D,
                      ops: dict[tuple[int, int], np.ndarray] | None = None,
                      chi_b: int = 32) -> complex:
    """Compute ⟨ψ|∏_{(x,y)∈ops} O_{xy}|ψ⟩ via boundary-MPO contractie.

    Args:
      peps: PEPS2D state.
      ops: dict mapping (x,y) → 2x2 operator. Ontbrekende sites = identiteit.
      chi_b: boundary-MPS truncatie.

    Returns:
      scalar (complex).
    """
    Lx, Ly = peps.Lx, peps.Ly
    ops = ops or {}

    # Bouw alle double-layer tensoren
    doubles: list[list[np.ndarray]] = [
        [None for _ in range(Ly)] for _ in range(Lx)  # type: ignore
    ]
    for x in range(Lx):
        for y in range(Ly):
            op = ops.get((x, y))
            doubles[x][y] = _site_double(peps.T[x][y], op)

    # Initiale boundary-MPS aan de linkerkant (vóór kolom 0)
    # Elke tensor is scalar "1" met bond dim 1 op beide zijden en phys=L²=1.
    # Omdat de eerste kolom links bond=1 heeft (rand), is dat consistent.
    boundary = []
    for y in range(Ly):
        # Phys dim = L² van site (0,y) links-bond. Rand: L=1 → L²=1.
        L_dim = peps.T[0][y].shape[0]
        assert L_dim == 1, (
            "Linker-rand moet bond dim 1 hebben (x=0,y={}); kreeg {}"
            .format(y, L_dim))
        # Shape (1, 1, 1)
        boundary.append(np.ones((1, 1, 1), dtype=complex))

    # Sweep kolommen van links naar rechts; absorbeer telkens een kolom
    for x in range(Lx):
        col_doubles = [doubles[x][y] for y in range(Ly)]
        boundary = _contract_column_to_boundary(col_doubles, boundary, chi_b)

    # Contracteer uiteindelijke boundary-MPS met de rechter-rand: alle
    # phys legs (R²=1 op rand) en alle left/right bonds (1) moeten sluiten.
    # De chain is nu één lijst van tensors shape (l,r,p=R²=1). Reduceer tot scalar.
    # Contract alle tensors langs de ketting + trace over l/r.
    result = np.array([[1.0]], dtype=complex)  # (1,1)
    for T in boundary:
        l, r, p = T.shape
        assert p == 1, (
            "Rechter-rand moet bond dim 1 hebben, kreeg phys={}".format(p))
        M = T[:, :, 0]  # (l, r)
        result = result @ M
    # result is (1,1); maar we hebben nog de "verticale" trace nodig.
    # In onze MPS-vorm zit de verticale structuur al impliciet in de l/r bonds.
    # Laatste sluiting: element result[0,0].
    return complex(result[0, 0])


# ============================================================
# Snelheids-hulp: exact state-vector voor kleine Lx·Ly ≤ 16
# ============================================================

def exact_plus_state(n: int) -> np.ndarray:
    """|+⟩^⊗n in 2^n-dim state vector."""
    v = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    state = v.copy()
    for _ in range(n - 1):
        state = np.kron(state, v)
    return state


def apply_single_sv(state: np.ndarray, n: int, q: int, g: np.ndarray) -> np.ndarray:
    """Pas 1-qubit gate toe op state-vector (qubit 0 = minst significant).

    Conventie: de state-vector is geïndexeerd met bit 0 = q0 (little-endian).
    """
    shape = [2] * n
    s = state.reshape(shape)
    # Tensor-axes: axis k correspondeert met qubit k (little-endian).
    # We moeten de gate toepassen op axis q.
    s = np.tensordot(g, s, axes=([1], [q]))  # contracts input axis
    # gate output-axis is nu op positie 0; verplaats naar positie q
    perm = list(range(1, n))
    perm.insert(q, 0)
    s = np.transpose(s, perm)
    return s.reshape(2 ** n)


def apply_zz_sv(state: np.ndarray, n: int, q1: int, q2: int,
                gamma: float) -> np.ndarray:
    """Exacte diagonale ZZ(γ) op qubits q1, q2."""
    if q1 == q2:
        raise ValueError("q1 en q2 moeten verschillen")
    # diagonaal: e^{-i γ z1 z2} met z = +1 voor bit 0, -1 voor bit 1
    # => bit q1 XOR bit q2 == 0: factor e^{-i γ}, anders e^{+i γ}
    idx = np.arange(2 ** n)
    b1 = (idx >> q1) & 1
    b2 = (idx >> q2) & 1
    parity = b1 ^ b2
    factor = np.where(parity == 0,
                      np.exp(-1j * gamma),
                      np.exp(1j * gamma))
    return state * factor


def exact_qaoa_maxcut(Lx: int, Ly: int,
                      edges: Iterable[tuple[int, int]],
                      gammas: Iterable[float],
                      betas: Iterable[float]) -> tuple[np.ndarray, float]:
    """Exacte state-vector QAOA MaxCut op Lx·Ly qubits.

    Return: (final state, <H_C>).
    Qubit-index: q = y * Lx + x (row-major).
    """
    n = Lx * Ly
    state = exact_plus_state(n)
    edges = list(edges)
    for gamma, beta in zip(gammas, betas):
        # ZZ-laag
        for (q1, q2) in edges:
            state = apply_zz_sv(state, n, q1, q2, gamma)
        # X-mixer (Rx op elke qubit)
        g = Rx(2.0 * beta)
        for q in range(n):
            state = apply_single_sv(state, n, q, g)
    # Energie: <H_C> = sum_edges 0.5 * (1 - <Z_i Z_j>)
    energy = 0.0
    for (q1, q2) in edges:
        idx = np.arange(2 ** n)
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        z1z2 = np.where(b1 == b2, 1.0, -1.0)
        zz = np.sum(z1z2 * np.abs(state) ** 2)
        energy += 0.5 * (1.0 - zz)
    return state, float(energy)


# ============================================================
# 2D MaxCut QAOA via PEPS
# ============================================================

def grid_edges(Lx: int, Ly: int) -> list[tuple[int, int, int, int, str]]:
    """Alle grid-edges in 2D rooster.

    Geeft (x1, y1, x2, y2, dir) met dir ∈ {'h', 'v'}.
    """
    edges = []
    for y in range(Ly):
        for x in range(Lx):
            if x + 1 < Lx:
                edges.append((x, y, x + 1, y, "h"))
            if y + 1 < Ly:
                edges.append((x, y, x, y + 1, "v"))
    return edges


def grid_edges_flat(Lx: int, Ly: int) -> list[tuple[int, int]]:
    """Zelfde, maar als flat qubit-indices q = y*Lx + x."""
    edges = []
    for y in range(Ly):
        for x in range(Lx):
            q = y * Lx + x
            if x + 1 < Lx:
                edges.append((q, q + 1))
            if y + 1 < Ly:
                edges.append((q, q + Lx))
    return edges


def peps_qaoa_maxcut(Lx: int, Ly: int,
                     gammas: Iterable[float],
                     betas: Iterable[float],
                     chi_max: int = 4,
                     chi_b: int = 16) -> tuple[PEPS2D, float]:
    """Run QAOA MaxCut op een 2D grid met PEPS.

    Returns:
      (peps_final, energy ⟨H_C⟩)
    """
    peps = PEPS2D.plus_state(Lx, Ly, chi_max=chi_max)
    edges_2d = grid_edges(Lx, Ly)
    gammas = list(gammas)
    betas = list(betas)
    for gamma, beta in zip(gammas, betas):
        # ZZ-laag via 2-site gates
        zz = ZZg(gamma)  # (2,2,2,2)
        for (x1, y1, x2, y2, direction) in edges_2d:
            if direction == "h":
                peps.apply_two_horizontal(x1, y1, zz, chi_max=chi_max)
            else:
                peps.apply_two_vertical(x1, y1, zz, chi_max=chi_max)
        # X-mixer: Rx op elke site
        rx = Rx(2.0 * beta)
        for x in range(Lx):
            for y in range(Ly):
                peps.apply_single(x, y, rx)

    # Energie = sum_edges 0.5 * (1 - <Z_i Z_j>)
    energy = 0.0
    norm2 = expectation_value(peps, ops=None, chi_b=chi_b).real
    for (x1, y1, x2, y2, direction) in edges_2d:
        ops = {(x1, y1): Z_MAT, (x2, y2): Z_MAT}
        zz_val = expectation_value(peps, ops=ops, chi_b=chi_b)
        zz_real = zz_val.real / max(norm2, 1e-12)
        energy += 0.5 * (1.0 - zz_real)
    return peps, float(energy)


# ============================================================
# CLI demo
# ============================================================

def _demo() -> None:
    Lx, Ly = 3, 3
    p = 1
    gammas = [0.4]
    betas = [0.3]
    edges_flat = grid_edges_flat(Lx, Ly)
    _, E_exact = exact_qaoa_maxcut(Lx, Ly, edges_flat, gammas, betas)
    _, E_peps = peps_qaoa_maxcut(Lx, Ly, gammas, betas,
                                 chi_max=4, chi_b=16)
    print(f"[b10e] {Lx}x{Ly}, p={p}: exact E={E_exact:.6f}, PEPS E={E_peps:.6f}, "
          f"diff={abs(E_exact - E_peps):.3e}")


if __name__ == "__main__":
    _demo()
