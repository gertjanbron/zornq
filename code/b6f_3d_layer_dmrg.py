"""
3D Layer-grouped DMRG: Heisenberg op 2x2xLz kubus (periodiek x,y).

Groepeer elke xy-laag tot 1 MPS-site: d = 2^(Lx*Ly) = 16.
Vergelijk met standaard d=2 DMRG (snake ordering door 3D).

Reuse de DMRG-engine uit b6e_final.py (works for any d).
"""
import numpy as np
from numpy.linalg import eigh, svd
from scipy.sparse.linalg import eigsh, LinearOperator
import time, sys

Sp = np.array([[0,1],[0,0]], dtype=float)
Sm = np.array([[0,0],[1,0]], dtype=float)
Zd = np.array([[1,0],[0,-1]], dtype=float)
Id2 = np.eye(2, dtype=float)

# === Layer-grouped MPO for 2x2xLz (d=16) ===

def embed_1q(op, qubit, nq):
    """Embed 1-qubit op in nq-qubit space."""
    d = 2**nq
    ops = [Id2]*nq
    ops[qubit] = op
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r

def embed_2q(op1, q1, op2, q2, nq):
    ops = [Id2]*nq
    ops[q1] = op1
    ops[q2] = op2
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r

def h_intra_layer(Lx, Ly, periodic=True):
    """Intra-layer Heisenberg for Lx x Ly lattice (periodic in both)."""
    nq = Lx * Ly
    d = 2**nq
    h = np.zeros((d, d))

    def idx(x, y):
        return y * Lx + x

    for y in range(Ly):
        for x in range(Lx):
            # x-bond
            if x < Lx - 1 or (periodic and Lx > 2):
                x2 = (x + 1) % Lx
                q1, q2 = idx(x, y), idx(x2, y)
                h += 2 * embed_2q(Sp, q1, Sm, q2, nq)
                h += 2 * embed_2q(Sm, q1, Sp, q2, nq)
                h += embed_2q(Zd, q1, Zd, q2, nq)
            # y-bond
            if y < Ly - 1 or (periodic and Ly > 2):
                y2 = (y + 1) % Ly
                q1, q2 = idx(x, y), idx(x, y2)
                h += 2 * embed_2q(Sp, q1, Sm, q2, nq)
                h += 2 * embed_2q(Sm, q1, Sp, q2, nq)
                h += embed_2q(Zd, q1, Zd, q2, nq)
    return h

def build_mpo_layer(Lz, Lx, Ly, periodic_xy=True):
    """MPO for 3D Heisenberg with layer grouping.

    Each site = Lx*Ly qubits (one xy-layer). d = 2^(Lx*Ly).
    Inter-layer: Lx*Ly vertical bonds (one per xy-site).
    Intra-layer: h_layer as local term.

    MPO bond dim D = 2 + 3*Lx*Ly (same structure as 2D triplet).
    """
    nq = Lx * Ly
    d = 2**nq
    n_bonds = nq  # one z-bond per xy-site
    D = 2 + 3 * n_bonds

    h_layer = h_intra_layer(Lx, Ly, periodic_xy)

    # Inter-layer operators: for each xy-site q, operators S+_q, S-_q, Z_q
    start_ops = []
    end_ops = []
    for q in range(nq):
        start_ops.append([
            embed_1q(Sp, q, nq),
            embed_1q(Sm, q, nq),
            embed_1q(Zd, q, nq),
        ])
        end_ops.append([
            2 * embed_1q(Sm, q, nq),
            2 * embed_1q(Sp, q, nq),
            embed_1q(Zd, q, nq),
        ])

    Id_d = np.eye(d)
    W_bulk = np.zeros((D, d, d, D))
    W_bulk[0, :, :, 0] = Id_d       # accumulator
    W_bulk[D-1, :, :, D-1] = Id_d   # source
    W_bulk[D-1, :, :, 0] = h_layer  # intra-layer -> accum

    for q in range(nq):
        for k in range(3):
            ch = 3*q + k + 1
            W_bulk[D-1, :, :, ch] = start_ops[q][k]
            W_bulk[ch, :, :, 0] = end_ops[q][k]

    mpo = []
    for i in range(Lz):
        if i == 0:
            mpo.append(W_bulk[D-1:D, :, :, :].copy())
        elif i == Lz - 1:
            mpo.append(W_bulk[:, :, :, 0:1].copy())
        else:
            mpo.append(W_bulk.copy())
    return mpo

# === Standard 3D MPO (d=2, simple ordering) ===

def build_3d_std_bonds(Lx, Ly, Lz, periodic_xy=True):
    """Generate bonds for Lx x Ly x Lz, periodic in xy."""
    n = Lx * Ly * Lz

    def idx(x, y, z):
        return z * Lx * Ly + y * Lx + x

    bonds = set()
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if x < Lx - 1:
                    bonds.add((idx(x,y,z), idx(x+1,y,z)))
                elif periodic_xy and Lx > 2:
                    a, b = idx(0,y,z), idx(Lx-1,y,z)
                    bonds.add((min(a,b), max(a,b)))
                if y < Ly - 1:
                    bonds.add((idx(x,y,z), idx(x,y+1,z)))
                elif periodic_xy and Ly > 2:
                    a, b = idx(x,0,z), idx(x,Ly-1,z)
                    bonds.add((min(a,b), max(a,b)))
                if z < Lz - 1:
                    bonds.add((idx(x,y,z), idx(x,y,z+1)))
    return n, sorted(bonds)

def build_mpo_std_compressed(n, bonds):
    d = 2
    active_at_cut = []
    for k in range(n-1):
        active_at_cut.append([b for b,(i,j) in enumerate(bonds) if i <= k and j > k])
    cut_maps, cut_dims = [], []
    for k in range(n-1):
        active = active_at_cut[k]
        mapping = {bi: 3*li+1 for li, bi in enumerate(active)}
        cut_maps.append(mapping)
        cut_dims.append(2 + 3*len(active))
    mpo = []
    for site in range(n):
        DL = 1 if site == 0 else cut_dims[site-1]
        DR = 1 if site == n-1 else cut_dims[site]
        lm = {} if site == 0 else cut_maps[site-1]
        rm = {} if site == n-1 else cut_maps[site]
        W = np.zeros((DL, d, d, DR))
        if site == 0:
            if DR > 1: W[0,:,:,DR-1] = Id2
        elif site == n-1:
            W[0,:,:,0] = Id2
        else:
            W[0,:,:,0] = Id2
            W[DL-1,:,:,DR-1] = Id2
        for b,(i,j) in enumerate(bonds):
            if site == i:
                src = 0 if site == 0 else DL-1
                if b in rm:
                    ch = rm[b]
                    W[src,:,:,ch] = Sp
                    W[src,:,:,ch+1] = Sm
                    W[src,:,:,ch+2] = Zd
            elif site == j:
                if b in lm:
                    ch = lm[b]
                    W[ch,:,:,0] = 2*Sm
                    W[ch+1,:,:,0] = 2*Sp
                    W[ch+2,:,:,0] = Zd
            elif i < site < j:
                if b in lm and b in rm:
                    cl2, cr2 = lm[b], rm[b]
                    W[cl2,:,:,cr2] = Id2
                    W[cl2+1,:,:,cr2+1] = Id2
                    W[cl2+2,:,:,cr2+2] = Id2
        mpo.append(W)
    return mpo

# === DMRG engine (from b6e_final.py) ===

def random_mps(n, d, chi):
    mps = []
    for i in range(n):
        cl = 1 if i == 0 else min(chi, d**i, d**(n-i))
        cr = 1 if i == n-1 else min(chi, d**(i+1), d**(n-1-i))
        mps.append(np.random.randn(cl, d, cr))
    return mps

def right_canon(mps):
    for i in range(len(mps)-1, 0, -1):
        cl, d, cr = mps[i].shape
        U, S, Vh = svd(mps[i].reshape(cl, d*cr), full_matrices=False)
        mps[i] = Vh.reshape(-1, d, cr)
        mps[i-1] = np.einsum('ijk,kl->ijl', mps[i-1], U @ np.diag(S))
    mps[0] /= np.linalg.norm(mps[0])
    return mps

def upL(L, A, W):
    t = np.einsum('gmk,ksc->gmsc', L, A)
    t = np.einsum('gmsc,msrv->grvc', t, W)
    return np.einsum('grvc,gra->avc', t, A)

def upR(R, B, W):
    t = np.einsum('ksj,bvj->ksbv', B, R)
    t = np.einsum('msrv,ksbv->mrkb', W, t)
    return np.einsum('arb,mrkb->amk', B, t)

def solve_2site(L, Wi, Wj, R, shapes):
    cl, d1, d2, cr = shapes
    dim = cl * d1 * d2 * cr
    if dim <= 1500:
        WW = np.einsum('wabx,xcfv->wacbfv', Wi, Wj, optimize=True)
        LWW = np.einsum('gwk,wacbfv->gkacbfv', L, WW, optimize=True)
        H = np.einsum('gkacbfv,hvj->gkacbfhj', LWW, R, optimize=True)
        H_mat = H.transpose(0,4,5,6,1,2,3,7).reshape(dim, dim)
        H_mat = (H_mat + H_mat.T) / 2
        ev, vc = eigh(H_mat)
        return ev[0], vc[:,0]
    else:
        LW = np.einsum('pwa,wsqx->pasqx', L, Wi, optimize=True)
        WR = np.einsum('xrfv,hvb->xrfhb', Wj, R, optimize=True)
        def mv(v):
            x = v.reshape(cl, d1, d2, cr)
            t = np.einsum('pasqx,asrb->pqxrb', LW, x, optimize=True)
            t = np.einsum('pqxrb,xrfhb->pqfh', t, WR, optimize=True)
            return t.ravel()
        op = LinearOperator((dim, dim), matvec=mv, dtype=float)
        ev, vc = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=300)
        return ev[0], vc[:,0]

def dmrg(n, mpo, chi, nsw=20, tol=1e-10, verbose=False):
    d = mpo[0].shape[1]
    mps = random_mps(n, d, chi)
    mps = right_canon(mps)
    Rs = [None]*(n+1); Rs[n] = np.ones((1,1,1))
    for i in range(n-1, -1, -1):
        Rs[i] = upR(Rs[i+1], mps[i], mpo[i])
    Ls = [None]*(n+1); Ls[0] = np.ones((1,1,1))
    Eo = 1e10
    for sw in range(nsw):
        for i in range(n-1):
            cl, d1, _ = mps[i].shape
            _, d2, cr = mps[i+1].shape
            ev0, theta = solve_2site(Ls[i], mpo[i], mpo[i+1], Rs[i+2],
                                     (cl, d1, d2, cr))
            th = theta.reshape(cl*d1, d2*cr)
            U, S, Vh = svd(th, full_matrices=False)
            k = min(chi, len(S))
            mps[i] = U[:,:k].reshape(cl, d1, k)
            mps[i+1] = (np.diag(S[:k]) @ Vh[:k,:]).reshape(k, d2, cr)
            Ls[i+1] = upL(Ls[i], mps[i], mpo[i])
        for i in range(n-2, -1, -1):
            cl, d1, _ = mps[i].shape
            _, d2, cr = mps[i+1].shape
            ev0, theta = solve_2site(Ls[i], mpo[i], mpo[i+1], Rs[i+2],
                                     (cl, d1, d2, cr))
            th = theta.reshape(cl*d1, d2*cr)
            U, S, Vh = svd(th, full_matrices=False)
            k = min(chi, len(S))
            mps[i] = (U[:,:k] @ np.diag(S[:k])).reshape(cl, d1, k)
            mps[i+1] = Vh[:k,:].reshape(k, d2, cr)
            Rs[i+1] = upR(Rs[i+2], mps[i+1], mpo[i+1])
        En = ev0
        dE = abs(En - Eo)
        if verbose:
            print(f"  Sw{sw+1}: E={En:.10f} dE={dE:.2e}", flush=True)
        if dE < tol and sw > 0:
            break
        Eo = En
    return mps, En, sw+1

# === Exact reference ===

def exact_3d(Lx, Ly, Lz, periodic_xy=True):
    n, bonds = build_3d_std_bonds(Lx, Ly, Lz, periodic_xy)
    dim = 2**n
    H = np.zeros((dim, dim))
    for (i,j) in bonds:
        for (O1, O2) in [(Sp, 2*Sm), (Sm, 2*Sp), (Zd, Zd)]:
            op = np.eye(1)
            for k in range(n):
                if k == i: op = np.kron(op, O1)
                elif k == j: op = np.kron(op, O2)
                else: op = np.kron(op, Id2)
            H += op
    return np.linalg.eigvalsh(H)[0]

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 65)
    print("  3D Layer-grouped DMRG: 2x2xLz (d=16)")
    print("=" * 65)
    sys.stdout.flush()

    # 2x2x2 (8 qubits) — exact reference
    Lx, Ly, Lz = 2, 2, 2
    E_ex = exact_3d(Lx, Ly, Lz)
    n_std, bonds = build_3d_std_bonds(Lx, Ly, Lz)
    mpo_std = build_mpo_std_compressed(n_std, bonds)
    mpo_lay = build_mpo_layer(Lz, Lx, Ly)
    D_std = max(w.shape[0] for w in mpo_std)
    D_lay = max(w.shape[0] for w in mpo_lay)
    print(f"\n2x2x2 (8q), E_exact={E_ex:.10f}")
    print(f"  Std: {n_std} sites d=2 D={D_std}")
    print(f"  Layer: {Lz} sites d=16 D={D_lay}")
    sys.stdout.flush()

    for chi in [4, 8, 16]:
        _, Es, sws = dmrg(n_std, mpo_std, chi, nsw=20, tol=1e-12)
        _, El, swl = dmrg(Lz, mpo_lay, chi, nsw=20, tol=1e-12)
        print(f"  chi={chi:2d}: Std gap={Es-E_ex:.2e} | Layer gap={El-E_ex:.2e}")
        sys.stdout.flush()

    # 2x2x3 (12 qubits)
    Lx, Ly, Lz = 2, 2, 3
    E_ex = exact_3d(Lx, Ly, Lz)
    n_std, bonds = build_3d_std_bonds(Lx, Ly, Lz)
    mpo_std = build_mpo_std_compressed(n_std, bonds)
    mpo_lay = build_mpo_layer(Lz, Lx, Ly)
    D_std = max(w.shape[0] for w in mpo_std)
    D_lay = max(w.shape[0] for w in mpo_lay)
    print(f"\n2x2x3 (12q), E_exact={E_ex:.10f}")
    print(f"  Std: {n_std} sites d=2 D={D_std}")
    print(f"  Layer: {Lz} sites d=16 D={D_lay}")
    sys.stdout.flush()

    for chi in [4, 8, 16]:
        t0 = time.time()
        _, Es, sws = dmrg(n_std, mpo_std, chi, nsw=15, tol=1e-12)
        dts = time.time() - t0
        t0 = time.time()
        _, El, swl = dmrg(Lz, mpo_lay, chi, nsw=15, tol=1e-12)
        dtl = time.time() - t0
        print(f"  chi={chi:2d}: Std E={Es:.6f} gap={Es-E_ex:.2e} t={dts:.1f}s | "
              f"Layer E={El:.6f} gap={El-E_ex:.2e} t={dtl:.1f}s")
        sys.stdout.flush()

    # 2x2x4 (16 qubits)
    Lx, Ly, Lz = 2, 2, 4
    n_std, bonds = build_3d_std_bonds(Lx, Ly, Lz)
    mpo_std = build_mpo_std_compressed(n_std, bonds)
    mpo_lay = build_mpo_layer(Lz, Lx, Ly)
    D_std = max(w.shape[0] for w in mpo_std)
    D_lay = max(w.shape[0] for w in mpo_lay)
    print(f"\n2x2x4 (16q), no exact ref")
    print(f"  Std: {n_std} sites d=2 D={D_std}")
    print(f"  Layer: {Lz} sites d=16 D={D_lay}")
    sys.stdout.flush()

    for chi in [8, 16]:
        t0 = time.time()
        _, Es, sws = dmrg(n_std, mpo_std, chi, nsw=10, tol=1e-10)
        dts = time.time() - t0
        t0 = time.time()
        _, El, swl = dmrg(Lz, mpo_lay, chi, nsw=10, tol=1e-10)
        dtl = time.time() - t0
        print(f"  chi={chi:2d}: Std E={Es:.6f} t={dts:.1f}s | "
              f"Layer E={El:.6f} t={dtl:.1f}s")
        sys.stdout.flush()
