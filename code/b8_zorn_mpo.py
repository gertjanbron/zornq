"""
B8: Zorn-MPO — Heisenberg-beeld operator in split-octonion representatie.

Strategie:
1. Groepeer 3 qubits per site → d=8 (= Zorn dimensie)
2. Bouw QAOA-circuit als intra-group (1-site) en inter-group (2-site) gates
3. Heisenberg-evolutie op grouped MPO
4. Test: blijft operator-chi = 1? → operator past in product van Zorn-elementen
5. Decomponeer 8×8 operator-matrices in Zorn-structuur
6. Bouw Zorn-native gate operaties
"""
import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import expm
import time, sys

# ============================================================
# ZORN ALGEBRA
# Split-octonion als 2×2 matrix over C³:
#   Z = [[alpha, a],    alpha,beta in C
#        [b,     beta]]  a,b in C³
#
# Vermenigvuldiging:
#   Z1 * Z2 = [[a1*a2 + <a1,b2>, a2*a1 + b2*b1 + a1×b2],  -- WRONG, let me use the real formula
#              [...]]
# ============================================================

class Zorn:
    """Split-octonion element als Zorn matrix.
    Representatie: (alpha, a, b, beta) met alpha,beta complex, a,b in C^3.
    """
    def __init__(self, alpha=0, a=None, b=None, beta=0):
        self.alpha = complex(alpha)
        self.beta = complex(beta)
        self.a = np.zeros(3, dtype=complex) if a is None else np.array(a, dtype=complex)
        self.b = np.zeros(3, dtype=complex) if b is None else np.array(b, dtype=complex)

    def to_array(self):
        """Flatten to 8 complex numbers: [alpha, a0, a1, a2, b0, b1, b2, beta]."""
        return np.array([self.alpha, *self.a, *self.b, self.beta], dtype=complex)

    @staticmethod
    def from_array(arr):
        return Zorn(arr[0], arr[1:4], arr[4:7], arr[7])

    def __mul__(self, other):
        """Zorn product (split-octonion multiplication).
        [α, a] [α', a']   [αα' + a·b', αa' + β'a + b×b']
        [b, β] [b', β'] = [α'b + βb' - a×a', ββ' + a'·b  ]
        where · = dot product, × = cross product
        """
        aa = self.alpha * other.alpha + np.dot(self.a, other.b)
        ab = self.alpha * other.a + other.beta * self.a + np.cross(self.b, other.b)
        ba = other.alpha * self.b + self.beta * other.b - np.cross(self.a, other.a)
        bb = self.beta * other.beta + np.dot(other.a, self.b)
        return Zorn(aa, ab, ba, bb)

    def conj(self):
        """Split-octonion conjugate: swap diagonal, negate off-diagonal."""
        return Zorn(self.beta, -self.a, -self.b, self.alpha)

    def norm_sq(self):
        """N(z) = z * z.conj() = (alpha*beta - a·b) * 1"""
        return self.alpha * self.beta - np.dot(self.a, self.b)

    def trace(self):
        return self.alpha + self.beta

    def to_matrix_8x8(self):
        """Left-multiplication matrix: z*x represented as 8×8 matrix acting on x.to_array()."""
        # Build by applying self * e_i for each basis element
        M = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            e = np.zeros(8, dtype=complex)
            e[i] = 1
            ei = Zorn.from_array(e)
            result = self * ei
            M[:, i] = result.to_array()
        return M

    def __repr__(self):
        return f"Zorn(α={self.alpha:.4f}, a={self.a}, b={self.b}, β={self.beta:.4f})"


# ============================================================
# QUBIT GATES
# ============================================================
I2 = np.eye(2, dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)

def Rx(theta):
    c = np.cos(theta/2); s = np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def ZZ_gate(gamma):
    return np.diag([np.exp(-1j*gamma), np.exp(1j*gamma),
                    np.exp(1j*gamma), np.exp(-1j*gamma)])


# ============================================================
# 3-QUBIT GROUPED GATES
# Group k contains qubits [3k, 3k+1, 3k+2]
# ============================================================

def embed_1q_in_group(U_1q, pos_in_group):
    """Embed a 1-qubit gate at position 0,1,2 within a 3-qubit group.
    Returns 8×8 unitary."""
    parts = [I2, I2, I2]
    parts[pos_in_group] = U_1q
    return np.kron(np.kron(parts[0], parts[1]), parts[2])

def embed_2q_in_group(U_2q, pos1, pos2):
    """Embed a 2-qubit gate at positions within a 3-qubit group.
    pos1, pos2 must be adjacent: (0,1) or (1,2).
    Returns 8×8 unitary."""
    if pos1 == 0 and pos2 == 1:
        return np.kron(U_2q, I2)
    elif pos1 == 1 and pos2 == 2:
        return np.kron(I2, U_2q)
    else:
        raise ValueError(f"Non-adjacent positions: {pos1},{pos2}")

def build_inter_group_gate(U_2q):
    """Build 64×64 gate for inter-group bond.
    Acts on qubit 2 of left group and qubit 0 of right group.
    In grouped space: (d_L=8) ⊗ (d_R=8), gate acts on
    qubit positions 2 (in left) and 0 (in right).
    Returns 64×64 unitary."""
    # The 2-qubit gate acts on the last qubit of left group
    # and first qubit of right group.
    # In the 6-qubit space [L0,L1,L2,R0,R1,R2], it acts on (L2,R0).
    # In grouped space: d_L=8 basis = |L0,L1,L2>, d_R=8 basis = |R0,R1,R2>
    # Gate on (L2, R0) in the 64-dim space:
    d = 8
    U_full = np.eye(d*d, dtype=complex)
    U_2q_4 = U_2q  # 4×4

    for l0 in range(2):
        for l1 in range(2):
            for r1 in range(2):
                for r2 in range(2):
                    for l2 in range(2):
                        for r0 in range(2):
                            # Input index in grouped space
                            idx_L = l0*4 + l1*2 + l2
                            idx_R = r0*4 + r1*2 + r2
                            idx_in = idx_L * d + idx_R

                            # 2q gate on (l2, r0)
                            for l2p in range(2):
                                for r0p in range(2):
                                    coeff = U_2q_4[l2p*2+r0p, l2*2+r0]
                                    if abs(coeff) < 1e-16:
                                        continue
                                    idx_Lp = l0*4 + l1*2 + l2p
                                    idx_Rp = r0p*4 + r1*2 + r2
                                    idx_out = idx_Lp * d + idx_Rp
                                    U_full[idx_out, idx_in] += coeff - (1.0 if idx_out==idx_in else 0.0)
    # Fix: we started with identity, need to subtract it and add gate
    # Actually, let me redo this properly
    U_full = np.zeros((d*d, d*d), dtype=complex)
    for l0 in range(2):
        for l1 in range(2):
            for l2 in range(2):
                for r0 in range(2):
                    for r1 in range(2):
                        for r2 in range(2):
                            idx_L = l0*4 + l1*2 + l2
                            idx_R = r0*4 + r1*2 + r2
                            idx_in = idx_L * d + idx_R
                            for l2p in range(2):
                                for r0p in range(2):
                                    coeff = U_2q_4[l2p*2+r0p, l2*2+r0]
                                    idx_Lp = l0*4 + l1*2 + l2p
                                    idx_Rp = r0p*4 + r1*2 + r2
                                    idx_out = idx_Lp * d + idx_Rp
                                    U_full[idx_out, idx_in] += coeff
    return U_full


# ============================================================
# GROUPED MPO
# ============================================================

def make_grouped_mpo_local(n_groups, group_idx, qubit_in_group, Op):
    """MPO with Op on one qubit within a group, identity elsewhere.
    Each tensor: (chi_L, d=8, d=8, chi_R) with chi=1."""
    d = 8
    I8 = np.eye(d, dtype=complex)
    mpo = []
    for g in range(n_groups):
        if g == group_idx:
            # Op on qubit_in_group, identity on others
            W = embed_1q_in_group(Op, qubit_in_group)
        else:
            W = I8.copy()
        mpo.append(W.reshape(1, d, d, 1))
    return mpo

def apply_1site_gate_grouped(mpo, group, U8):
    """Apply 8×8 unitary on one group: O -> U†·O·U."""
    Ud = U8.conj().T
    W = mpo[group]  # (chi_L, 8, 8, chi_R)
    W = np.einsum('ij,ajkb->aikb', Ud, W)
    W = np.einsum('ajkb,kl->ajlb', W, U8)
    mpo[group] = W
    return mpo

def apply_2site_gate_grouped(mpo, g1, U64, chi_max=64):
    """Apply 64×64 unitary on two groups: O -> U†·O·U.
    U64 acts on d_L⊗d_R = 8⊗8 = 64 dim space."""
    g2 = g1 + 1
    d = 8
    Ud = U64.conj().T
    Ud_4d = Ud.reshape(d, d, d, d)  # (bra_L, bra_R, ket_L, ket_R)... NO
    # Actually: Ud is 64×64, reshape to (8,8,8,8) = (out_L, out_R, in_L, in_R)
    # For MPO: U†[bra',bra] and U[ket',ket]
    # Ud = U†, so Ud[i,j] = U*[j,i], Ud reshaped as (i_L,i_R,j_L,j_R)

    cl = mpo[g1].shape[0]; cr = mpo[g2].shape[3]

    # Contract 2 MPO sites: Theta(cl, b_L, k_L, b_R, k_R, cr)
    Th = np.einsum('abce,edfg->abcdfg', mpo[g1], mpo[g2])

    # Apply U† on bra indices and U on ket indices
    Ud_r = Ud.reshape(d, d, d, d)  # (bra_L', bra_R', bra_L, bra_R)
    Uf_r = U64.reshape(d, d, d, d)  # (ket_L', ket_R', ket_L, ket_R)

    # U† on bra: contract bra_L (pos 1) and bra_R (pos 3) of Th
    Th = np.einsum('ijkl,akclef->aicjef', Ud_r, Th)
    # U on ket: contract ket positions
    Th = np.einsum('ijkl,abkdlf->aibjdf', Uf_r, Th)

    # Transpose to (cl, bra_L', ket_L', bra_R', ket_R', cr)
    Th = Th.transpose(0, 2, 1, 4, 3, 5)

    # SVD
    mat = Th.reshape(cl * d * d, d * d * cr)
    U_svd, S, V = svd(mat, full_matrices=False)
    Sa = np.abs(S)
    k = max(1, int(np.sum(Sa > 1e-14 * Sa[0]))) if Sa[0] > 1e-15 else 1
    k = min(k, chi_max)
    trunc = np.sum(Sa[k:]**2) if k < len(S) else 0.0

    mpo[g1] = U_svd[:, :k].reshape(cl, d, d, k)
    mpo[g2] = (np.diag(S[:k]) @ V[:k, :]).reshape(k, d, d, cr)
    return mpo, trunc

def mpo_expect_product_grouped(mpo, group_states):
    """<state|MPO|state> where group_states[g] is the 3-qubit index (0-7)."""
    L = np.ones((1,), dtype=complex)
    for g in range(len(mpo)):
        s = group_states[g]
        L = np.einsum('a,ab->b', L, mpo[g][:, s, s, :])
    return L[0]


# ============================================================
# BUILD GROUPED QAOA CIRCUIT
# ============================================================

def qaoa_grouped_gates(n_groups, p_layers, gamma, beta):
    """Build QAOA gate list for grouped representation.
    n_qubits = 3 * n_groups. 1D chain with ZZ on all bonds.
    Returns list of ('1g', group, U8) or ('2g', group, U64).
    """
    gates = []
    n_q = 3 * n_groups
    ZZ = ZZ_gate(gamma)
    Rxg = Rx(2 * beta)

    for layer in range(p_layers):
        # ZZ on all bonds
        for q in range(n_q - 1):
            g1 = q // 3;  pos1 = q % 3
            g2 = (q+1) // 3;  pos2 = (q+1) % 3

            if g1 == g2:
                # Intra-group: embed 2q gate as 8×8
                U8 = embed_2q_in_group(ZZ, pos1, pos2)
                gates.append(('1g', g1, U8))
            else:
                # Inter-group: build 64×64 gate
                U64 = build_inter_group_gate(ZZ)
                gates.append(('2g', g1, U64))

        # Rx on all qubits
        for q in range(n_q):
            g = q // 3; pos = q % 3
            U8 = embed_1q_in_group(Rxg, pos)
            gates.append(('1g', g, U8))

    return gates


def evolve_grouped_mpo(mpo, gates, chi_max=64):
    """Heisenberg evolution: reversed gates."""
    total_trunc = 0.0
    for entry in reversed(gates):
        gtype = entry[0]
        if gtype == '1g':
            _, g, U8 = entry
            mpo = apply_1site_gate_grouped(mpo, g, U8)
        elif gtype == '2g':
            _, g1, U64 = entry
            mpo, te = apply_2site_gate_grouped(mpo, g1, U64, chi_max)
            total_trunc += te
    return mpo, total_trunc


# ============================================================
# ZORN DECOMPOSITION OF 8×8 OPERATOR
# ============================================================

def matrix_to_zorn_basis(M8):
    """Decompose 8×8 matrix in terms of Zorn left-multiplication matrices.
    If M = L_z (left mult by z), find z.
    Returns Zorn element z such that z*x has matrix M8 acting on x.to_array().
    """
    # Method: apply M8 to the unit element e1 = Zorn(1,0,0,0)
    e1 = np.zeros(8, dtype=complex); e1[0] = 1
    z_arr = M8 @ e1
    return Zorn.from_array(z_arr)

def check_zorn_left_mult(M8):
    """Check if M8 is a left-multiplication matrix of some Zorn element.
    Returns (is_zorn, z, residual)."""
    z = matrix_to_zorn_basis(M8)
    M_reconstructed = z.to_matrix_8x8()
    residual = norm(M8 - M_reconstructed)
    return residual < 1e-10, z, residual

def check_zorn_right_mult(M8):
    """Check if M8 is a right-multiplication matrix."""
    # Right mult by z: x*z. Build R_z matrix.
    # R_z[i,j] = (e_j * z)[i]
    # Try: M8 applied to e1 gives the first column = ?, but for right mult
    # e_j * z, we need z first. Extract from e1 * z = z (since 1*z=z).
    e1 = np.zeros(8, dtype=complex); e1[0] = 1
    z_arr = M8 @ e1  # This gives column 0 = e_1 * z = z
    z = Zorn.from_array(z_arr)

    # Check: does R_z match M8?
    R = np.zeros((8, 8), dtype=complex)
    for j in range(8):
        ej = np.zeros(8, dtype=complex); ej[j] = 1
        ej_zorn = Zorn.from_array(ej)
        result = ej_zorn * z
        R[:, j] = result.to_array()
    residual = norm(M8 - R)
    return residual < 1e-10, z, residual


# ============================================================
# EXACT REFERENCE
# ============================================================

def exact_expect_sv(n_q, gates_flat, obs_qubit, obs_op, init_states):
    """Exact state-vector reference. gates_flat: list of (type, qubit(s), unitary)."""
    dim = 2**n_q
    psi = np.zeros(dim, dtype=complex)
    idx = 0
    for s in init_states:
        idx = idx * 2 + s
    psi[idx] = 1.0

    for gt, qubits, U in gates_flat:
        psi_r = psi.reshape([2]*n_q)
        if gt == '1q':
            q = qubits
            psi_r = np.tensordot(U, psi_r, axes=([1], [q]))
            psi_r = np.moveaxis(psi_r, 0, q)
        elif gt == '2q':
            q1, q2 = qubits
            U4 = U.reshape(2, 2, 2, 2)
            psi_r = np.tensordot(U4, psi_r, axes=([2, 3], [q1, q2]))
            psi_r = np.moveaxis(psi_r, [0, 1], [q1, q2])
        psi = psi_r.reshape(dim)

    Op_full = np.eye(1, dtype=complex)
    for i in range(n_q):
        Op_full = np.kron(Op_full, obs_op if i == obs_qubit else I2)
    return (psi.conj() @ Op_full @ psi).real

def qaoa_flat_gates(n_q, p_layers, gamma, beta):
    """Flat gate list for exact reference."""
    gates = []
    ZZ = ZZ_gate(gamma)
    Rxg = Rx(2 * beta)
    for layer in range(p_layers):
        for q in range(n_q - 1):
            gates.append(('2q', (q, q+1), ZZ))
        for q in range(n_q):
            gates.append(('1q', q, Rxg))
    return gates


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)

    print("="*70)
    print("  B8: Zorn-MPO — Heisenberg operator in split-octonion")
    print("="*70)

    # === PART 1: Zorn algebra verification ===
    print("\n--- Zorn algebra test ---")
    z1 = Zorn(1, [1, 0, 0], [0, 1, 0], 1)
    z2 = Zorn(0, [0, 1, 0], [0, 0, 1], 1)
    z3 = z1 * z2
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    print(f"z1*z2 = {z3}")
    print(f"N(z1) = {z1.norm_sq()}")

    # Verify associativity failure (octonions are non-associative)
    z_a = Zorn(1, [1,0,0], [0,0,0], 0)
    z_b = Zorn(0, [0,1,0], [0,0,0], 1)
    z_c = Zorn(0, [0,0,1], [1,0,0], 0)
    lhs = (z_a * z_b) * z_c
    rhs = z_a * (z_b * z_c)
    non_assoc = norm(lhs.to_array() - rhs.to_array())
    print(f"|(ab)c - a(bc)| = {non_assoc:.6f} {'(non-associative!)' if non_assoc > 1e-10 else '(associative)'}")

    # === PART 2: Grouped MPO on small system ===
    print("\n" + "="*70)
    print("  Grouped MPO test: n=9 (3 groups), p=1..5")
    print("="*70)

    n_groups = 3
    n_q = 3 * n_groups
    gamma = 0.3; beta = 0.7
    init_states = [0] * n_q
    group_states = [0] * n_groups  # all |000>
    obs_qubit = 0
    obs_group = 0
    obs_pos = 0

    for p in [1, 2, 3, 5]:
        # Exact reference
        flat_gates = qaoa_flat_gates(n_q, p, gamma, beta)
        exact_val = exact_expect_sv(n_q, flat_gates, obs_qubit, Z, init_states)

        # Grouped MPO
        grouped_gates = qaoa_grouped_gates(n_groups, p, gamma, beta)
        mpo = make_grouped_mpo_local(n_groups, obs_group, obs_pos, Z)
        mpo, trunc = evolve_grouped_mpo(mpo, grouped_gates, chi_max=64)
        val = mpo_expect_product_grouped(mpo, group_states).real

        chi = [mpo[g].shape[3] for g in range(n_groups-1)]
        err = abs(val - exact_val)

        print(f"\np={p}: exact={exact_val:+.12f}, grouped={val:+.12f}")
        print(f"  err={err:.2e}, trunc={trunc:.2e}, chi={chi}")

        # Check if chi=1 tensors have Zorn structure
        if all(c == 1 for c in chi):
            print(f"  CHI=1 OVERAL! Operator is product van 8×8 matrices.")
            for g in range(n_groups):
                M8 = mpo[g][0, :, :, 0]  # 8×8 matrix
                is_L, z_L, res_L = check_zorn_left_mult(M8)
                is_R, z_R, res_R = check_zorn_right_mult(M8)
                print(f"  Group {g}: left-Zorn={is_L} (res={res_L:.2e}), right-Zorn={is_R} (res={res_R:.2e})")
                if is_L:
                    print(f"    z = {z_L}")
        else:
            # For chi>1, examine the tensors
            print(f"  Chi>1 — operator has inter-group entanglement")
            for g in range(n_groups):
                shape = mpo[g].shape
                params = mpo[g].size
                print(f"  Group {g}: shape={shape}, params={params}")

    # === PART 3: Scale test with grouping ===
    print("\n" + "="*70)
    print("  SCHAALTEST: grouped Zorn-MPO")
    print("="*70)

    p = 5
    for n_g in [3, 10, 33, 100, 166]:
        n_q = 3 * n_g
        grouped_gates = qaoa_grouped_gates(n_g, p, gamma, beta)

        t0 = time.time()
        mpo = make_grouped_mpo_local(n_g, 0, 0, Z)
        mpo, trunc = evolve_grouped_mpo(mpo, grouped_gates, chi_max=64)
        val = mpo_expect_product_grouped(mpo, [0]*n_g).real
        dt = time.time() - t0

        chi = [mpo[g].shape[3] for g in range(n_g-1)]
        chi_max_val = max(chi) if chi else 1
        params = sum(w.size for w in mpo)
        mem_kb = params * 16 / 1024

        print(f"\nn={n_q:4d}q ({n_g} groups): <Z_0>={val:+.10f}")
        print(f"  tijd={dt:.2f}s, chi_max={chi_max_val}, params={params}, mem={mem_kb:.1f} KB")
        print(f"  trunc={trunc:.2e}")
        if n_q <= 9:
            flat_gates = qaoa_flat_gates(n_q, p, gamma, beta)
            ex = exact_expect_sv(n_q, flat_gates, obs_qubit, Z, [0]*n_q)
            print(f"  exact={ex:+.10f}, err={abs(val-ex):.2e}")

    sys.stdout.flush()
