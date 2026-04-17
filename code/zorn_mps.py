"""
ZornMPS: Unified Tensor Network Engine
=======================================
Integreert alle bewezen resultaten uit ZornQ B1-B13:

  - Zorn-element als lokale node (d=8, 3 qubits per site)
  - Dual mode: Schrödinger (state MPS) en Heisenberg (operator MPO)
  - 7-operatie informatiecomplete reconstructie
  - SVD-truncatie met split-norm kwaliteitsmeter
  - Column-grouping voor 2D/3D roosters
  - rSVD voor grote lokale dimensies (d≥16)
  - Sz-symmetrische blok-diagonale SVD (B13, 2-3× speedup)
  - GPU-acceleratie via cupy (B11b, 10-100× speedup op SVD/einsum)

Gebruik:
  mps = ZornMPS(n_qubits=300, max_chi=32)
  mps.apply_gate(ZZ_gate, [i, j])
  energy = mps.expectation(hamiltonian)
  quality = mps.split_norm_quality()

GPU gebruik:
  from gpu_backend import gpu_info
  gpu_info()  # check beschikbaarheid
  mps = ZornMPS(n_qubits=300, max_chi=32, gpu=True)

Architectuur volgt de spec: lokale nodes, geen globale state,
SVD-truncatie als geheugenbeheersing.
"""
import numpy as np
from numpy.linalg import svd as np_svd
from typing import List, Tuple, Optional, Literal

# GPU backend: transparante cupy/numpy switch
try:
    from gpu_backend import (xp, xp_svd, xp_svd_mp, xp_einsum, xp_rsvd, xp_diag,
                             to_device, to_numpy, sync, GPU_AVAILABLE,
                             MP_COMPLEX, HP_COMPLEX)
except ImportError:
    # Fallback als gpu_backend.py niet gevonden wordt
    xp = np
    xp_svd = lambda m, **kw: np.linalg.svd(m, **kw)
    xp_svd_mp = lambda m, **kw: np.linalg.svd(m, **kw)
    xp_einsum = np.einsum
    xp_diag = np.diag
    to_device = lambda x: x
    to_numpy = lambda x: x
    sync = lambda: None
    GPU_AVAILABLE = False
    MP_COMPLEX = np.complex64
    HP_COMPLEX = np.complex128

    def xp_rsvd(M, k, p=5):
        m, n = M.shape
        r = min(k + p, min(m, n))
        Omega = np.random.randn(n, r).astype(M.dtype)
        Y = M @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.conj().T @ M
        Ub, S, V = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
        return U[:, :k], S[:k], V[:k, :]

# =====================================================================
# ZORN ALGEBRA
# =====================================================================

def zmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Zorn matrix product: (a, α, β, b) × (c, γ, δ, d)"""
    a, al, be, b = A[0], A[1:4], A[4:7], A[7]
    c, ga, de, d = B[0], B[1:4], B[4:7], B[7]
    return np.array([a*c + al@de,
                     *(a*ga + d*al + np.cross(be, de)),
                     *(c*be + b*de - np.cross(al, ga)),
                     be@ga + b*d])

def zconj(A: np.ndarray) -> np.ndarray:
    """Zorn conjugatie: (a, α, β, b) → (b, -β, -α, a)"""
    return np.array([A[7], *(-A[4:7]), *(-A[1:4]), A[0]])

def znorm(A: np.ndarray) -> complex:
    """Split-norm: N(z) = ab - α·β"""
    return A[0]*A[7] - A[1:4]@A[4:7]

def zinv(A: np.ndarray) -> np.ndarray:
    """Zorn inversie via conjugaat/norm"""
    c = zconj(A); n = znorm(A)
    return c/n if abs(n) > 1e-15 else c

def zhodge(A: np.ndarray) -> np.ndarray:
    """Hodge dualiteit: cyclische 3-vector permutatie"""
    return np.array([A[0], A[2], A[3], A[1], A[5], A[6], A[4], A[7]])

def zassoc(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Associator: [A,B,C] = (AB)C - A(BC)"""
    return zmul(zmul(A, B), C) - zmul(A, zmul(B, C))

def zjordan(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Jordan triple product: ABA"""
    return zmul(zmul(A, B), A)


# =====================================================================
# RANDOMIZED SVD
# =====================================================================

def rsvd(M, k: int, p: int = 5) -> Tuple:
    """Halko-Martinsson-Tropp randomized SVD. O(k·m·n).
    Werkt op zowel numpy als cupy arrays.
    """
    return xp_rsvd(M, k, p)


# =====================================================================
# Sz-SYMMETRIC BLOCK-DIAGONAL SVD (bewezen in B13)
# =====================================================================

def _sz_values(n_qubits: int) -> np.ndarray:
    """Sz eigenvalue per basis state voor n qubits.
    |0⟩ = spin up (+1/2), |1⟩ = spin down (-1/2).
    """
    d = 2**n_qubits
    sz = np.zeros(d)
    for i in range(d):
        for q in range(n_qubits):
            sz[i] += 0.5 - ((i >> (n_qubits - 1 - q)) & 1)
    return sz


class SzBlockSVD:
    """Blok-diagonale SVD die Sz-behoud exploiteert.

    Voor Sz-conserverende gates (Heisenberg, XXZ, etc.) is de
    twee-site tensor blok-diagonaal in totale Sz-lading.
    SVD per blok i.p.v. op de volle matrix geeft 2-5× speedup.

    Bewezen in B13b: Heisenberg gate is 100% Sz-conserverend.
    d=8 (3 qubits): 4 unieke Sz-waarden {-3/2,-1/2,+1/2,+3/2}
    met multipliciteiten [1,3,3,1].
    """

    def __init__(self, n_qubits_per_site: int):
        self.d = 2**n_qubits_per_site
        self.n_qubits = n_qubits_per_site
        self.sz_phys = _sz_values(n_qubits_per_site)

    def __call__(self, theta: np.ndarray, cl: int, d: int, cr: int,
                 q_left: np.ndarray, q_right: np.ndarray,
                 max_chi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Blok-diagonale SVD met Sz-symmetrie.

        Parameters
        ----------
        theta : (cl, d, d, cr) tensor na gate-toepassing
        cl, d, cr : dimensies
        q_left : (cl,) Sz-ladingen op linker bond
        q_right : (cr,) Sz-ladingen op rechter bond
        max_chi : maximale bond dimensie

        Returns
        -------
        U, S, Vh : getrunceerde SVD-factoren
        q_mid : (k,) Sz-ladingen op nieuwe midden-bond
        discarded : fractie weggegooid gewicht

        Conservatiewet: q_left[a] + sz[i] = q_mid[m] = q_right[b] - sz[j]
        """
        # Rij-lading: Q = q_left[a] + sz[i]
        qlc = np.add.outer(q_left, self.sz_phys).ravel()
        # Kolom-lading: Q = q_right[b] - sz[j]
        qrc = np.add.outer(-self.sz_phys, q_right).ravel()
        qli = np.round(2 * qlc).astype(int)
        qri = np.round(2 * qrc).astype(int)

        mat = theta.reshape(cl * d, d * cr)

        # SVD per Sz-blok
        blocks = []
        all_sv = []
        for q in np.unique(qli):
            li = np.where(qli == q)[0]
            ri = np.where(qri == q)[0]
            if len(li) == 0 or len(ri) == 0:
                continue
            blk = mat[np.ix_(li, ri)]
            if np.max(np.abs(blk)) < 1e-15:
                continue
            Ub, Sb, Vb = np_svd(blk, full_matrices=False)
            nz = max(1, int(np.sum(Sb > 1e-12 * Sb[0])))
            blocks.append((q, li, ri, Ub[:, :nz], Sb[:nz], Vb[:nz, :]))
            all_sv.extend(Sb[:nz].tolist())

        if not all_sv:
            U = np.zeros((cl * d, 1), dtype=complex); U[0, 0] = 1
            V = np.zeros((1, d * cr), dtype=complex); V[0, 0] = 1
            return U, np.array([0.0]), V, np.array([0.0]), 0.0

        # Globale truncatie: behoud top-max_chi singuliere waarden
        all_sv = np.sort(all_sv)[::-1]
        tot_sq = np.sum(np.array(all_sv)**2)
        k = min(len(all_sv), max_chi)
        thr = all_sv[k - 1] if k < len(all_sv) else 0.0
        disc = np.sum(np.array(all_sv[k:])**2) / tot_sq if k < len(all_sv) else 0.0

        # Assembleer getrunceerd resultaat
        U = np.zeros((cl * d, k), dtype=complex)
        S = np.zeros(k)
        V = np.zeros((k, d * cr), dtype=complex)
        qm = np.zeros(k)
        col = 0
        for q, li, ri, Ub, Sb, Vb in blocks:
            n_keep = int(np.sum(Sb >= thr - 1e-15))
            n_keep = min(n_keep, k - col)
            if n_keep <= 0:
                continue
            U[np.ix_(li, np.arange(col, col + n_keep))] = Ub[:, :n_keep]
            S[col:col + n_keep] = Sb[:n_keep]
            V[np.ix_(np.arange(col, col + n_keep), ri)] = Vb[:n_keep, :]
            qm[col:col + n_keep] = q / 2.0
            col += n_keep

        return U[:, :col], S[:col], V[:col, :], qm[:col], disc


# =====================================================================
# ZORN NODE
# =====================================================================

class ZornNode:
    """Eén knoop in de Zorn-MPS keten.

    Tensor shape: (chi_left, d, chi_right)
    d = 8 (Zorn-element) of 2^Ly (column-grouped)
    """

    def __init__(self, d: int, chi_left: int = 1, chi_right: int = 1,
                 dtype=np.complex128):
        self.d = d
        self.tensor = np.zeros((chi_left, d, chi_right), dtype=dtype)
    
    @property
    def chi_left(self) -> int:
        return self.tensor.shape[0]
    
    @property
    def chi_right(self) -> int:
        return self.tensor.shape[2]
    
    @property
    def shape(self) -> tuple:
        return self.tensor.shape


# =====================================================================
# ZORN MPS ENGINE
# =====================================================================

class ZornMPS:
    """Unified Zorn-MPS tensor network engine.
    
    Parameters
    ----------
    n_sites : int
        Aantal sites in de MPS keten.
    d : int
        Lokale dimensie per site (8 voor Zorn, 2^Ly voor column-grouped).
    max_chi : int
        Harde bovengrens voor bond dimensie.
    mode : 'schrodinger' of 'heisenberg'
        Schrödinger: evolueer de toestand.
        Heisenberg: evolueer de operator (reversed gates, chi compacter).
    use_rsvd : bool
        Gebruik randomized SVD voor d >= rsvd_threshold.
    rsvd_threshold : int
        Minimale d waarvoor rSVD wordt ingezet.
    use_sz : bool
        Gebruik Sz-symmetrische blok-diagonale SVD (B13).
        Vereist dat gates Sz conserveren (Heisenberg, XXZ, etc.).
        Geeft 2-5× speedup bij d=8, chi≥32.
    n_qubits_per_site : int
        Aantal qubits per site (voor Sz-berekening). Default: log2(d).
    gpu : bool
        Gebruik GPU-acceleratie via cupy (B11b).
        Automatische fallback naar CPU als geen GPU beschikbaar.
        Verwachte speedup: 10-100× op SVD, 20-50× op einsum.
    mixed_precision : bool
        B19: Sla tensoren op als complex64 (fp32) i.p.v. complex128 (fp64).
        Halveert VRAM-gebruik. SVD wordt intern in fp64 uitgevoerd voor
        numerieke stabiliteit; einsum draait natively op fp32.
        Typische fout: < 1e-6 op energiewaarden.
    """

    def __init__(self, n_sites: int, d: int = 8, max_chi: int = 64,
                 mode: Literal['schrodinger', 'heisenberg'] = 'schrodinger',
                 use_rsvd: bool = True, rsvd_threshold: int = 16,
                 use_sz: bool = False, n_qubits_per_site: Optional[int] = None,
                 gpu: bool = False,
                 min_weight: Optional[float] = None,
                 mixed_precision: bool = False):
        self.n_sites = n_sites
        self.d = d
        self.max_chi = max_chi
        self.min_weight = min_weight  # B15: adaptieve truncatie-drempel
        self.mode = mode
        self.use_rsvd = use_rsvd
        self.rsvd_threshold = rsvd_threshold
        self.use_sz = use_sz
        self.mixed_precision = mixed_precision  # B19

        # B19: kies tensor dtype
        self._cdtype = MP_COMPLEX if mixed_precision else np.complex128

        # GPU setup
        self.gpu = gpu and GPU_AVAILABLE
        if gpu and not GPU_AVAILABLE:
            import warnings
            warnings.warn("GPU gevraagd maar cupy niet beschikbaar. Fallback naar CPU.")
        # Kies array module: xp (cupy als GPU actief, anders numpy)
        self._xp = xp if self.gpu else np

        # Sz-symmetrie setup
        if use_sz:
            nq = n_qubits_per_site if n_qubits_per_site else int(np.log2(d))
            self._sz_svd = SzBlockSVD(nq)
            self._bond_charges: List[Optional[np.ndarray]] = [None] * (n_sites - 1)
        else:
            self._sz_svd = None
            self._bond_charges = []

        # Initialiseer nodes (B19: complex64 als mixed_precision)
        self.nodes: List[ZornNode] = []
        for i in range(n_sites):
            node = ZornNode(d, chi_left=1, chi_right=1, dtype=self._cdtype)
            self.nodes.append(node)

        # Truncatie-statistieken
        self._total_discarded = 0.0
        self._n_truncations = 0
        self._max_chi_reached = 1
    
    # -----------------------------------------------------------------
    # INITIALISATIE
    # -----------------------------------------------------------------
    
    def init_product_state(self, local_states: Optional[np.ndarray] = None,
                           basis_indices: Optional[List[int]] = None):
        """Initialiseer als productstate.

        local_states: (n_sites, d) array, of None voor |+...+⟩
        basis_indices: list van basis-state indices per site (voor Sz-tracking).
                       Bijv. [0, 7, 0, 7, ...] voor Néel-state bij d=8.
        """
        d = self.d
        _xp = self._xp
        if local_states is None:
            psi_local = _xp.ones(d, dtype=self._cdtype) / _xp.sqrt(_xp.array(d, dtype=float))

        for i, node in enumerate(self.nodes):
            if local_states is not None:
                psi_local = to_device(local_states[i]) if self.gpu else local_states[i]
            node.tensor = psi_local.reshape(1, d, 1).copy()

        # Initialiseer Sz bond-ladingen
        if self.use_sz and basis_indices is not None:
            sz_phys = self._sz_svd.sz_phys
            cum = 0.0
            for i in range(self.n_sites - 1):
                cum += sz_phys[basis_indices[i]]
                self._bond_charges[i] = np.array([cum])
    
    def init_identity_mpo(self):
        """Initialiseer als identiteits-MPO (voor Heisenberg mode).
        
        Tensor shape: (chi_left, d_bra, d_ket, chi_right)
        We slaan dit op als (chi_left, d², chi_right) met d²=d_bra*d_ket.
        """
        d = self.d
        for node in self.nodes:
            # I = Σ_i |i⟩⟨i| → reshaped als d² vector
            node.d = d * d
            ident = np.eye(d, dtype=self._cdtype).ravel()
            node.tensor = ident.reshape(1, d*d, 1).copy()
    
    # -----------------------------------------------------------------
    # SVD TRUNCATIE (de kern van het geheugenbeheersing)
    # -----------------------------------------------------------------
    
    def _svd_truncate(self, mat) -> Tuple:
        """SVD + truncatie. Retourneert U, S, Vh, discarded_weight.

        Truncatie-strategie (B15):
        - Als min_weight is gezet: adaptief — knip totdat discarded weight >= epsilon,
          met max_chi als veiligheidsplafond. Chi ademt mee met het netwerk.
        - Anders: vaste truncatie op max_chi (oorspronkelijk gedrag).

        B19 Mixed-Precision: SVD altijd in fp64, U/Vh terug naar _cdtype.
        Werkt op zowel numpy als cupy arrays (GPU-transparant).
        """
        m, n = mat.shape
        chi = self.max_chi
        _xp = self._xp

        if self.use_rsvd and min(m, n) > self.rsvd_threshold and chi < min(m, n) // 2:
            U, S, Vh = rsvd(mat, chi, p=10)
            discarded = 0.0
        else:
            # B19: SVD in fp64 voor stabiliteit, zelfs als mat is fp32
            if self.mixed_precision:
                U, S, Vh = xp_svd_mp(mat, full_matrices=False)
            else:
                U, S, Vh = xp_svd(mat, full_matrices=False)
            Sa = _xp.abs(S)  # S is altijd fp64

            # Bepaal k: aantal singuliere waarden om te behouden
            k_nonzero = max(1, int(_xp.sum(Sa > 1e-12 * Sa[0]))) if float(Sa[0]) > 1e-15 else 1

            if self.min_weight is not None:
                # B15: Adaptieve truncatie — chi ademt mee
                Sa_sq = Sa[:k_nonzero]**2
                total_sq = float(_xp.sum(Sa_sq))
                k = min(k_nonzero, chi)
                if total_sq > 0:
                    running_disc = total_sq
                    for j in range(min(k_nonzero, chi)):
                        running_disc -= float(Sa_sq[j])
                        if running_disc / total_sq < self.min_weight:
                            k = j + 1
                            break
            else:
                k = min(k_nonzero, chi)

            discarded = float(_xp.sum(Sa[k:]**2) / _xp.sum(Sa**2)) if len(Sa) > k else 0.0
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]

        self._total_discarded += discarded
        self._n_truncations += 1
        self._max_chi_reached = max(self._max_chi_reached, len(S))

        return U, S, Vh, discarded
    
    # -----------------------------------------------------------------
    # GATE TOEPASSING
    # -----------------------------------------------------------------
    
    def _cast_gate(self, gate):
        """B19: Cast gate naar tensor dtype (fp32 als mixed_precision)."""
        if self.mixed_precision and gate.dtype != self._cdtype:
            gate = gate.astype(self._cdtype)
        return gate

    def _store_svd_result(self, s, U, S, Vh, shape_L, shape_R):
        """Sla SVD-resultaat op in nodes. B19: cast naar _cdtype.

        S is altijd fp64 → diag(S)@Vh wordt fp64 → cast terug.
        shape_L: (chi_left, d_dims..., k)
        shape_R: (k, d_dims..., chi_right)
        """
        _xp = self._xp
        k = len(S)
        A = U.reshape(shape_L)
        B = (xp_diag(S) @ Vh).reshape(shape_R)
        if self.mixed_precision:
            A = A.astype(self._cdtype) if A.dtype != self._cdtype else A
            B = B.astype(self._cdtype) if B.dtype != self._cdtype else B
        self.nodes[s].tensor = A
        self.nodes[s+1].tensor = B
        return k

    def apply_1site_gate(self, gate, site: int):
        """Pas een 1-site gate toe (d×d matrix). GPU-transparant."""
        node = self.nodes[site]
        gate = self._cast_gate(gate)
        if self.gpu:
            gate = to_device(gate)

        if self.mode == 'schrodinger':
            node.tensor = xp_einsum('ji,aib->ajb', gate, node.tensor)
        else:
            d = self.d
            T = node.tensor.reshape(node.chi_left, d, d, node.chi_right)
            Gd = gate.conj().T
            T = xp_einsum('ij,ajkb,kl->ailb', Gd, T, gate)
            node.tensor = T.reshape(node.chi_left, d*d, node.chi_right)

    def apply_1site_diag(self, diag, site: int):
        """Pas een diagonale 1-site gate toe. Efficiënt: geen matmul."""
        node = self.nodes[site]
        diag = self._cast_gate(diag)
        if self.gpu:
            diag = to_device(diag)

        if self.mode == 'schrodinger':
            node.tensor = node.tensor * diag[None, :, None]
        else:
            _xp = self._xp
            d = self.d
            T = node.tensor.reshape(node.chi_left, d, d, node.chi_right)
            T = T * _xp.conj(diag)[None, :, None, None] * diag[None, None, :, None]
            node.tensor = T.reshape(node.chi_left, d*d, node.chi_right)
    
    def apply_2site_gate(self, gate, site: int):
        """Pas een 2-site gate toe op site en site+1. GPU-transparant.

        gate: (d², d²) matrix of (d, d, d, d) tensor.
        Voert SVD-truncatie uit op de bond.
        """
        s = site
        _xp = self._xp
        gate = self._cast_gate(gate)
        if self.gpu:
            gate = to_device(gate)
        cl = self.nodes[s].chi_left
        cr = self.nodes[s+1].chi_right

        # Merge twee sites
        Theta = xp_einsum('aib,bjc->aijc', self.nodes[s].tensor, self.nodes[s+1].tensor)

        if self.mode == 'schrodinger':
            d = self.d
            G = gate.reshape(d, d, d, d) if gate.ndim != 4 else gate
            Theta_new = xp_einsum('ijkl,akld->aijd', G, Theta)

            # SVD + truncatie (met optionele Sz-symmetrie)
            if self.use_sz and self._sz_svd is not None:
                # Sz-blok SVD draait op CPU (indexing-intensief)
                Theta_cpu = to_numpy(Theta_new) if self.gpu else Theta_new
                q_left = self._bond_charges[s - 1] if s > 0 else np.array([0.0])
                q_right = self._bond_charges[s + 1] if s + 1 < self.n_sites - 1 else np.array([0.0])
                U, S, Vh, q_mid, disc = self._sz_svd(
                    Theta_cpu, cl, d, cr, q_left, q_right, self.max_chi)
                self._bond_charges[s] = q_mid
                if self.gpu:
                    U, S, Vh = to_device(U), to_device(S), to_device(Vh)
            else:
                mat = Theta_new.reshape(cl * d, d * cr)
                U, S, Vh, disc = self._svd_truncate(mat)

            k = self._store_svd_result(s, U, S, Vh, (cl, d, -1), (-1, d, cr))
        else:
            d = self.d
            T = Theta.reshape(cl, d, d, d, d, cr)
            G = gate.reshape(d, d, d, d) if gate.ndim != 4 else gate
            Gd = G.conj().transpose(2, 3, 0, 1)
            T_new = xp_einsum('IJij,aijklb,klKL->aIJKLb', Gd, T, G)
            mat = T_new.reshape(cl * d * d, d * d * cr)
            U, S, Vh, disc = self._svd_truncate(mat)
            k = self._store_svd_result(s, U, S, Vh, (cl, d*d, -1), (-1, d*d, cr))

        self._total_discarded += disc
        self._n_truncations += 1
        self._max_chi_reached = max(self._max_chi_reached, k)
        return disc

    def apply_2site_diag(self, diag, site: int):
        """Pas een diagonale 2-site gate toe. GPU-transparant.

        diag: (d*d,) vector van diagonaal-elementen.
        """
        s = site
        _xp = self._xp
        d = self.d
        diag = self._cast_gate(diag)
        if self.gpu:
            diag = to_device(diag)
        cl = self.nodes[s].chi_left
        cr = self.nodes[s+1].chi_right

        if self.mode == 'schrodinger':
            Theta = xp_einsum('aib,bjc->aijc', self.nodes[s].tensor, self.nodes[s+1].tensor)
            dd = diag.reshape(d, d)
            Theta = Theta * dd[None, :, :, None]

            if self.use_sz and self._sz_svd is not None:
                Theta_cpu = to_numpy(Theta) if self.gpu else Theta
                q_left = self._bond_charges[s - 1] if s > 0 else np.array([0.0])
                q_right = self._bond_charges[s + 1] if s + 1 < self.n_sites - 1 else np.array([0.0])
                U, S, Vh, q_mid, disc = self._sz_svd(
                    Theta_cpu, cl, d, cr, q_left, q_right, self.max_chi)
                self._bond_charges[s] = q_mid
                if self.gpu:
                    U, S, Vh = to_device(U), to_device(S), to_device(Vh)
            else:
                mat = Theta.reshape(cl * d, d * cr)
                U, S, Vh, disc = self._svd_truncate(mat)

            k = self._store_svd_result(s, U, S, Vh, (cl, d, -1), (-1, d, cr))
        else:
            Theta = xp_einsum('aib,bjc->aijc', self.nodes[s].tensor, self.nodes[s+1].tensor)
            cd = _xp.conj(diag).reshape(d, d)
            dd = diag.reshape(d, d)
            T = Theta.reshape(cl, d, d, d, d, cr)
            T = T * cd[None, :, None, :, None, None] * dd[None, None, :, None, :, None]
            mat = T.reshape(cl * d * d, d * d * cr)
            U, S, Vh, disc = self._svd_truncate(mat)
            k = self._store_svd_result(s, U, S, Vh, (cl, d*d, -1), (-1, d*d, cr))
            self.nodes[s+1].tensor = (xp_diag(S) @ Vh).reshape(k, d*d, cr)

        self._total_discarded += disc
        self._n_truncations += 1
        self._max_chi_reached = max(self._max_chi_reached, k)
        return disc

    # -----------------------------------------------------------------
    # VERWACHTINGSWAARDEN
    # -----------------------------------------------------------------
    
    def contract_full(self) -> complex:
        """Contracteer de hele MPS tot een scalar (trace/overlap)."""
        L = np.ones((1,), dtype=complex)
        for node in self.nodes:
            if self.mode == 'schrodinger':
                # ⟨ψ|ψ⟩: contracteer over fysieke index (normalisatie)
                L = np.einsum('a,aib->b', L, node.tensor[:, 0, :])
            else:
                # Tr(O): contracteer over d² met I = δ_{bra,ket}
                d = self.d
                T = node.tensor.reshape(node.chi_left, d, d, node.chi_right)
                # Trace over bra=ket
                traced = np.einsum('aiib->ab', T)
                L = np.einsum('a,ab->b', L, traced)
        return L[0]
    
    def expectation_local(self, obs: np.ndarray, site: int) -> complex:
        """⟨ψ|O_site|ψ⟩ via MPS transfer-matrix contractie.

        L[α', α] = left environment (bra-bond × ket-bond).
        Per site: L_new = Σ_{α',α,i,(j)} L[α',α] · A*[α',i,β'] · (O[i,j]·) A[α,(j),β]
        Complexiteit: O(n · chi² · d).
        """
        L = np.ones((1, 1), dtype=complex)

        for i, node in enumerate(self.nodes):
            T = node.tensor  # (chi_L, d, chi_R)
            if i == site:
                # Operator site: T_op[a,i,b] = Σ_j O[i,j] · T[a,j,b]
                T_op = np.einsum('ij,ajb->aib', obs, T)
                L = np.einsum('pq,pir,qis->rs', L, T.conj(), T_op)
            else:
                # Non-operator: contract fysieke index
                L = np.einsum('pq,pir,qis->rs', L, T.conj(), T)

        return complex(np.trace(L))
    
    # -----------------------------------------------------------------
    # KWALITEITSMETRIEK
    # -----------------------------------------------------------------
    
    def split_norm_quality(self) -> float:
        """Split-norm gebaseerde kwaliteitsmeter.
        
        Meet de totale hoeveelheid weggegooid gewicht bij truncaties.
        Vuistregel uit B10h: split_norm < 0.01 → fout < 0.1%
        """
        if self._n_truncations == 0:
            return 0.0
        return self._total_discarded / self._n_truncations
    
    @property
    def max_chi_used(self) -> int:
        """Maximale chi die daadwerkelijk bereikt is."""
        return self._max_chi_reached
    
    @property
    def memory_bytes(self) -> int:
        """Geschat geheugengebruik in bytes."""
        total = 0
        for node in self.nodes:
            total += node.tensor.nbytes
        return total
    
    def chi_profile(self) -> List[int]:
        """Bond dimensies langs de keten."""
        return [self.nodes[i].chi_right for i in range(self.n_sites - 1)]
    
    # -----------------------------------------------------------------
    # INFO
    # -----------------------------------------------------------------
    
    def __repr__(self):
        mem = self.memory_bytes
        unit = 'B'
        if mem > 1024**2: mem /= 1024**2; unit = 'MB'
        elif mem > 1024: mem /= 1024; unit = 'KB'
        gpu_str = ", GPU" if self.gpu else ""
        sz_str = ", Sz" if self.use_sz else ""
        mp_str = ", fp32" if self.mixed_precision else ""
        adapt_str = f", eps={self.min_weight:.0e}" if self.min_weight is not None else ""
        return (f"ZornMPS(sites={self.n_sites}, d={self.d}, "
                f"max_chi={self.max_chi}, mode={self.mode}, "
                f"chi_max_used={self.max_chi_used}, mem={mem:.1f}{unit}"
                f"{sz_str}{gpu_str}{mp_str}{adapt_str})")


# =====================================================================
# HEISENBERG-MPO QAOA ENGINE (bewezen in B9/B10)
# =====================================================================

class HeisenbergQAOA:
    """QAOA MaxCut solver via Heisenberg-MPO. GPU-transparant.

    Bewezen: chi=2 exact voor 1D, chi=4 exact voor 2D p=1.
    10.000 qubits in 4ms (1D), 2500 qubits in 19s (2D).
    Met GPU: verwacht 10-100× sneller op 2D p≥2.
    """

    def __init__(self, Lx: int, Ly: int = 1, max_chi: int = 64, gpu: bool = False,
                 min_weight: Optional[float] = None,
                 mixed_precision: bool = False):
        self.Lx = Lx
        self.Ly = Ly
        self.d = 2**Ly if Ly > 1 else 2
        self.n_sites = Lx
        self.max_chi = max_chi
        self.min_weight = min_weight  # B15: adaptieve truncatie
        self.mixed_precision = mixed_precision  # B19
        self._cdtype = MP_COMPLEX if mixed_precision else np.complex128
        self.gpu = gpu and GPU_AVAILABLE
        self._xp = xp if self.gpu else np
        self.bp = self._bit_patterns()
        self._Hd = self._build_hadamard()
        self.n_edges = (Lx * (Ly - 1) + (Lx - 1) * Ly) if Ly > 1 else (Lx - 1)
        if self.mixed_precision:
            self._Hd = self._Hd.astype(self._cdtype)
        if self.gpu:
            self._Hd = to_device(self._Hd)
    
    def _bit_patterns(self):
        d, Ly = self.d, self.Ly
        if Ly == 1:
            return np.array([[0], [1]])
        return np.array([[(idx >> (Ly-1-q)) & 1 for q in range(Ly)] for idx in range(d)])
    
    def _build_hadamard(self):
        d, Ly, bp = self.d, self.Ly, self.bp
        H1 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
        Hd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Hd *= H1[bp[:, q:q+1], bp[:, q:q+1].T]
        return Hd
    
    def _to_cdtype(self, arr):
        """B19: cast naar tensor dtype."""
        return arr.astype(self._cdtype) if self.mixed_precision else arr

    def _zz_intra_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        diag = np.ones(d, dtype=complex)
        for y in range(Ly - 1):
            z1 = 1 - 2*bp[:, y].astype(float)
            z2 = 1 - 2*bp[:, y+1].astype(float)
            diag *= np.exp(-1j * gamma * z1 * z2)
        return self._to_cdtype(diag)

    def _zz_inter_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        iL = np.arange(d*d) // d
        iR = np.arange(d*d) % d
        diag = np.ones(d*d, dtype=complex)
        for y in range(Ly):
            z1 = 1 - 2*bp[iL, y].astype(float)
            z2 = 1 - 2*bp[iR, y].astype(float)
            diag *= np.exp(-1j * gamma * z1 * z2)
        return self._to_cdtype(diag)

    def _rx_col(self, beta):
        d, Ly, bp = self.d, self.Ly, self.bp
        c, s = np.cos(beta), np.sin(beta)
        rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
        Rxd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Rxd *= rx[bp[:, q:q+1], bp[:, q:q+1].T]
        return self._to_cdtype(Rxd)
    
    def _make_zz_mpo(self, x1, y1, x2, y2):
        """Maak ZZ-observable MPO. GPU-transparant. B19: mixed-precision."""
        _xp = self._xp
        d, bp = self.d, self.bp
        mpo = [_xp.eye(d, dtype=self._cdtype).reshape(1,d,d,1).copy()
               for _ in range(self.Lx)]
        cdt = self._cdtype
        if x1 == x2:
            z1 = 1 - 2*bp[:, y1].astype(float)
            z2 = 1 - 2*bp[:, y2].astype(float)
            zz = (z1 * z2).astype(cdt)
            if self.gpu:
                zz = to_device(zz)
            mpo[x1] = xp_diag(zz).reshape(1,d,d,1)
        else:
            for col, y in [(x1,y1), (x2,y2)]:
                dv = (1 - 2*bp[:, y].astype(float)).astype(cdt)
                if self.gpu:
                    dv = to_device(dv)
                mpo[col] = xp_diag(dv).reshape(1,d,d,1)
        return mpo
    
    def _ap1(self, mpo, s, U):
        Ud = U.conj().T
        mpo[s] = xp_einsum('ij,ajkb,kl->ailb', Ud, mpo[s], U)
        return mpo

    def _ap1_diag(self, mpo, s, diag):
        _xp = self._xp
        cd = _xp.conj(diag)
        mpo[s] = mpo[s] * cd[None,:,None,None] * diag[None,None,:,None]
        return mpo

    def _ap2_diag(self, mpo, s, diag_dd):
        _xp = self._xp
        d = self.d
        cl = mpo[s].shape[0]
        cr = mpo[s+1].shape[3]
        Th = xp_einsum('aijc,cklb->aijklb', mpo[s], mpo[s+1])
        cd = _xp.conj(diag_dd).reshape(d,d)
        dd = diag_dd.reshape(d,d)
        Th = Th * cd[None,:,None,:,None,None] * dd[None,None,:,None,:,None]
        mat = Th.reshape(cl*d*d, d*d*cr)
        if self.mixed_precision:
            U, S, V = xp_svd_mp(mat, full_matrices=False)
        else:
            U, S, V = xp_svd(mat, full_matrices=False)
        Sa = _xp.abs(S)
        k_nonzero = max(1, int(_xp.sum(Sa > 1e-12*Sa[0]))) if float(Sa[0]) > 1e-15 else 1
        if self.min_weight is not None:
            Sa_sq = Sa[:k_nonzero]**2
            total_sq = float(_xp.sum(Sa_sq))
            k = min(k_nonzero, self.max_chi)
            if total_sq > 0:
                running_disc = total_sq
                for j in range(min(k_nonzero, self.max_chi)):
                    running_disc -= float(Sa_sq[j])
                    if running_disc / total_sq < self.min_weight:
                        k = j + 1
                        break
        else:
            k = min(k_nonzero, self.max_chi)
        A = U[:,:k].reshape(cl,d,d,k)
        B = (xp_diag(S[:k]) @ V[:k,:]).reshape(k,d,d,cr)
        if self.mixed_precision:
            A = A.astype(self._cdtype) if A.dtype != self._cdtype else A
            B = B.astype(self._cdtype) if B.dtype != self._cdtype else B
        mpo[s] = A
        mpo[s+1] = B
        return mpo

    def _mpo_trace(self, mpo):
        _xp = self._xp
        L = _xp.ones((1,), dtype=self._cdtype)
        for W in mpo:
            L = xp_einsum('a,ab->b', L, W[:,0,0,:])
        return float(L[0].real) + 1j * float(L[0].imag) if self.gpu else L[0]
    
    def eval_edge(self, x1, y1, x2, y2, p, gammas, betas):
        """Evalueer <ZZ> op één edge via Heisenberg-MPO. GPU-transparant."""
        mpo = self._make_zz_mpo(x1, y1, x2, y2)

        # Build gate list (gates naar GPU als nodig)
        gates = []
        Hd = self._Hd
        for x in range(self.Lx):
            gates.append(('full', x, Hd))
        for l in range(p):
            zzi = self._zz_intra_diag(gammas[l])
            zze = self._zz_inter_diag(gammas[l])
            rxd = self._rx_col(betas[l])
            if self.gpu:
                zzi, zze, rxd = to_device(zzi), to_device(zze), to_device(rxd)
            for x in range(self.Lx):
                gates.append(('diag1', x, zzi))
            for x in range(self.Lx - 1):
                gates.append(('diag2', x, zze))
            for x in range(self.Lx):
                gates.append(('full', x, rxd))

        # Evolve in reverse (Heisenberg)
        for gt, s, data in reversed(gates):
            if gt == 'full':
                mpo = self._ap1(mpo, s, data)
            elif gt == 'diag1':
                mpo = self._ap1_diag(mpo, s, data)
            else:
                mpo = self._ap2_diag(mpo, s, data)

        result = self._mpo_trace(mpo)
        return float(result.real) if hasattr(result, 'real') else float(result)

    def eval_cost(self, p, gammas, betas):
        """Volledige MaxCut cost = Sigma_edges (1-<ZZ>)/2."""
        Lx, Ly = self.Lx, self.Ly
        total = 0.0
        for x in range(Lx):
            for y in range(Ly - 1):
                zz = self.eval_edge(x, y, x, y+1, p, gammas, betas)
                total += (1 - zz) / 2
        for x in range(Lx - 1):
            for y in range(Ly):
                zz = self.eval_edge(x, y, x+1, y, p, gammas, betas)
                total += (1 - zz) / 2
        return total

    def eval_ratio(self, p, gammas, betas):
        return self.eval_cost(p, gammas, betas) / self.n_edges


# =====================================================================
# 7-OPERATIE RECONSTRUCTIE (bewezen in B2/B3)
# =====================================================================

FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]

def fano_decompose(A: np.ndarray, d: int) -> np.ndarray:
    """Permuteer octonion-componenten volgens Fano-triplet d."""
    t = FANO[d]
    comp = [x for x in range(1, 8) if x not in t]
    return A[[0] + list(t) + comp]


# build_transfer_matrix en reconstruct_random verplaatst naar zorn_algebra.py
