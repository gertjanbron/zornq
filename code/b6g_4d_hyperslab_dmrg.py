"""
4D Layer-grouped DMRG: Heisenberg op 2x2x2xLw hyperkubus.

Cross-sectie: 2x2x2 = 8 qubits per "hyperlaag" -> d = 2^8 = 256.
MPS loopt langs w-as: Lw sites.
Inter-hyperlaag: 8 w-bonds per cut.
Intra-hyperlaag: alle xyz-bonds exact in d=256.

Dit is het grensgebied: d=256 is groot maar hanteerbaar.
MPO D = 2 + 3*8 = 26.
"""
import numpy as np
from numpy.linalg import eigh, svd
from scipy.sparse.linalg import eigsh, LinearOperator
import time, sys

Sp = np.array([[0,1],[0,0]], dtype=float)
Sm = np.array([[0,0],[1,0]], dtype=float)
Zd = np.array([[1,0],[0,-1]], dtype=float)
Id2 = np.eye(2, dtype=float)

def embed_1q(op, qubit, nq):
    ops = [Id2]*nq; ops[qubit] = op
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def embed_2q(op1, q1, op2, q2, nq):
    ops = [Id2]*nq; ops[q1] = op1; ops[q2] = op2
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def h_intra_hyperslab(dims, periodic=True):
    """Intra-slab Heisenberg for a hypercubic slab.
    dims: tuple (Lx, Ly, Lz, ...) for the slab dimensions.
    All directions periodic if periodic=True.
    """
    ndim = len(dims)
    nq = 1
    for d in dims: nq *= d

    import itertools
    sites = list(itertools.product(*[range(d) for d in dims]))
    def to_idx(site):
        r = 0
        for k in range(ndim):
            r = r * dims[k] + site[k]
        return r

    d_hilbert = 2**nq
    h = np.zeros((d_hilbert, d_hilbert))

    for site in sites:
        for axis in range(ndim):
            # Bond in this direction
            neighbor = list(site)
            if site[axis] < dims[axis] - 1:
                neighbor[axis] += 1
            elif periodic and dims[axis] > 2:
                neighbor[axis] = 0
            else:
                continue
            q1, q2 = to_idx(tuple(site)), to_idx(tuple(neighbor))
            if q1 != q2:
                h += 2 * embed_2q(Sp, q1, Sm, q2, nq)
                h += 2 * embed_2q(Sm, q1, Sp, q2, nq)
                h += embed_2q(Zd, q1, Zd, q2, nq)
    return h

def build_mpo_hyperlayer(n_slabs, slab_dims, periodic_slab=True):
    """MPO for nD Heisenberg with hyperslab grouping.
    MPS flows along the last (ungrouped) dimension.
    Each site = product of slab_dims qubits.
    """
    nq_slab = 1
    for d in slab_dims: nq_slab *= d
    d = 2**nq_slab
    n_wbonds = nq_slab  # one w-bond per qubit in slab
    D = 2 + 3 * n_wbonds

    h_slab = h_intra_hyperslab(slab_dims, periodic_slab)

    start_ops, end_ops = [], []
    for q in range(nq_slab):
        start_ops.append([embed_1q(Sp,q,nq_slab), embed_1q(Sm,q,nq_slab), embed_1q(Zd,q,nq_slab)])
        end_ops.append([2*embed_1q(Sm,q,nq_slab), 2*embed_1q(Sp,q,nq_slab), embed_1q(Zd,q,nq_slab)])

    Id_d = np.eye(d)
    W_bulk = np.zeros((D, d, d, D))
    W_bulk[0,:,:,0] = Id_d
    W_bulk[D-1,:,:,D-1] = Id_d
    W_bulk[D-1,:,:,0] = h_slab
    for q in range(nq_slab):
        for k in range(3):
            ch = 3*q + k + 1
            W_bulk[D-1,:,:,ch] = start_ops[q][k]
            W_bulk[ch,:,:,0] = end_ops[q][k]

    mpo = []
    for i in range(n_slabs):
        if i == 0: mpo.append(W_bulk[D-1:D,:,:,:].copy())
        elif i == n_slabs-1: mpo.append(W_bulk[:,:,:,0:1].copy())
        else: mpo.append(W_bulk.copy())
    return mpo

# === DMRG engine ===

def random_mps(n, d, chi):
    mps = []
    for i in range(n):
        cl = 1 if i==0 else min(chi, d**i, d**(n-i))
        cr = 1 if i==n-1 else min(chi, d**(i+1), d**(n-1-i))
        mps.append(np.random.randn(cl, d, cr))
    return mps

def right_canon(mps):
    for i in range(len(mps)-1, 0, -1):
        cl,d,cr = mps[i].shape
        U,S,Vh = svd(mps[i].reshape(cl,d*cr), full_matrices=False)
        mps[i] = Vh.reshape(-1,d,cr)
        mps[i-1] = np.einsum('ijk,kl->ijl', mps[i-1], U@np.diag(S))
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
    dim = cl*d1*d2*cr
    if dim <= 1500:
        WW = np.einsum('wabx,xcfv->wacbfv', Wi, Wj, optimize=True)
        LWW = np.einsum('gwk,wacbfv->gkacbfv', L, WW, optimize=True)
        H = np.einsum('gkacbfv,hvj->gkacbfhj', LWW, R, optimize=True)
        H_mat = H.transpose(0,4,5,6,1,2,3,7).reshape(dim,dim)
        H_mat = (H_mat + H_mat.T)/2
        ev,vc = eigh(H_mat)
        return ev[0], vc[:,0]
    else:
        LW = np.einsum('pwa,wsqx->pasqx', L, Wi, optimize=True)
        WR = np.einsum('xrfv,hvb->xrfhb', Wj, R, optimize=True)
        def mv(v):
            x = v.reshape(cl,d1,d2,cr)
            t = np.einsum('pasqx,asrb->pqxrb', LW, x, optimize=True)
            t = np.einsum('pqxrb,xrfhb->pqfh', t, WR, optimize=True)
            return t.ravel()
        op = LinearOperator((dim,dim), matvec=mv, dtype=float)
        ev,vc = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=300)
        return ev[0], vc[:,0]

def dmrg(n, mpo, chi, nsw=20, tol=1e-10, verbose=False):
    d = mpo[0].shape[1]
    mps = random_mps(n, d, chi); mps = right_canon(mps)
    Rs = [None]*(n+1); Rs[n] = np.ones((1,1,1))
    for i in range(n-1,-1,-1): Rs[i] = upR(Rs[i+1], mps[i], mpo[i])
    Ls = [None]*(n+1); Ls[0] = np.ones((1,1,1))
    Eo = 1e10
    for sw in range(nsw):
        for i in range(n-1):
            cl,d1,_ = mps[i].shape; _,d2,cr = mps[i+1].shape
            ev0,theta = solve_2site(Ls[i],mpo[i],mpo[i+1],Rs[i+2],(cl,d1,d2,cr))
            th = theta.reshape(cl*d1,d2*cr)
            U,S,Vh = svd(th, full_matrices=False); k=min(chi,len(S))
            mps[i] = U[:,:k].reshape(cl,d1,k)
            mps[i+1] = (np.diag(S[:k])@Vh[:k,:]).reshape(k,d2,cr)
            Ls[i+1] = upL(Ls[i], mps[i], mpo[i])
        for i in range(n-2,-1,-1):
            cl,d1,_ = mps[i].shape; _,d2,cr = mps[i+1].shape
            ev0,theta = solve_2site(Ls[i],mpo[i],mpo[i+1],Rs[i+2],(cl,d1,d2,cr))
            th = theta.reshape(cl*d1,d2*cr)
            U,S,Vh = svd(th, full_matrices=False); k=min(chi,len(S))
            mps[i] = (U[:,:k]@np.diag(S[:k])).reshape(cl,d1,k)
            mps[i+1] = Vh[:k,:].reshape(k,d2,cr)
            Rs[i+1] = upR(Rs[i+2], mps[i+1], mpo[i+1])
        En = ev0; dE = abs(En-Eo)
        if verbose: print(f"  Sw{sw+1}: E={En:.10f} dE={dE:.2e}", flush=True)
        if dE < tol and sw > 0: break
        Eo = En
    return mps, En, sw+1

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 65)
    print("  4D Hyperslab DMRG: 2x2x2xLw (d=256)")
    print("=" * 65)
    sys.stdout.flush()

    # Build intra-slab Hamiltonian for 2x2x2 cube
    slab_dims = (2, 2, 2)
    nq_slab = 8
    d = 256
    print(f"\nSlab: {slab_dims}, nq={nq_slab}, d={d}")
    print(f"MPO D = {2 + 3*nq_slab}")
    sys.stdout.flush()

    # 2x2x2x2 = 16 qubits, 2 hyperslabs
    Lw = 2
    print(f"\n--- 2x2x2x2 ({nq_slab*Lw}q), {Lw} slabs ---")
    t0 = time.time()
    mpo = build_mpo_hyperlayer(Lw, slab_dims, periodic_slab=True)
    print(f"MPO built in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    for chi in [4, 8, 16]:
        t0 = time.time()
        _, E, sw = dmrg(Lw, mpo, chi, nsw=15, tol=1e-12, verbose=(chi==16))
        dt = time.time() - t0
        print(f"  chi={chi:2d}: E={E:.8f} sw={sw} t={dt:.1f}s")
        sys.stdout.flush()

    # 2x2x2x3 = 24 qubits
    Lw = 3
    print(f"\n--- 2x2x2x3 ({nq_slab*Lw}q), {Lw} slabs ---")
    mpo = build_mpo_hyperlayer(Lw, slab_dims, periodic_slab=True)
    for chi in [4, 8, 16]:
        t0 = time.time()
        _, E, sw = dmrg(Lw, mpo, chi, nsw=10, tol=1e-10)
        dt = time.time() - t0
        print(f"  chi={chi:2d}: E={E:.8f} sw={sw} t={dt:.1f}s")
        sys.stdout.flush()
