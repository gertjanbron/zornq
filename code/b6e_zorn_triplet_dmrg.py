"""
B6e: Zorn-triplet vs Standard DMRG on cylindrical lattices.
Key: 3 qubits per MPS site (d=8) matching Zorn structure.
Uses optimized 2-site Lanczos with precomputed LW/WR.
"""
import numpy as np
from numpy.linalg import svd, eigh
from scipy.sparse.linalg import eigsh, LinearOperator
import time, sys

Sp = np.array([[0,1],[0,0]], dtype=float)
Sm = np.array([[0,0],[1,0]], dtype=float)
Zd = np.array([[1,0],[0,-1]], dtype=float)
Id2 = np.eye(2, dtype=float)

def embed_1q(op, qubit, Ly=3):
    ops = [Id2]*Ly; ops[qubit] = op
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def embed_2q(op1, q1, op2, q2, Ly=3):
    ops = [Id2]*Ly; ops[q1] = op1; ops[q2] = op2
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def h_intra_column(Ly=3):
    d = 2**Ly; h = np.zeros((d,d))
    for y in range(Ly):
        y2 = (y+1) % Ly
        h += 2*embed_2q(Sp, y, Sm, y2, Ly)
        h += 2*embed_2q(Sm, y, Sp, y2, Ly)
        h += embed_2q(Zd, y, Zd, y2, Ly)
    return h

def build_mpo_triplet(Lx, Ly=3):
    d = 2**Ly; D = 2 + 3*Ly
    h_col = h_intra_column(Ly)
    start_ops, end_ops = [], []
    for y in range(Ly):
        start_ops.append([embed_1q(Sp,y,Ly), embed_1q(Sm,y,Ly), embed_1q(Zd,y,Ly)])
        end_ops.append([2*embed_1q(Sm,y,Ly), 2*embed_1q(Sp,y,Ly), embed_1q(Zd,y,Ly)])
    Id_d = np.eye(d)
    W_bulk = np.zeros((D,d,d,D))
    W_bulk[0,:,:,0] = Id_d; W_bulk[D-1,:,:,D-1] = Id_d; W_bulk[D-1,:,:,0] = h_col
    for y in range(Ly):
        for k in range(3):
            ch = 3*y+k+1
            W_bulk[D-1,:,:,ch] = start_ops[y][k]
            W_bulk[ch,:,:,0] = end_ops[y][k]
    mpo = []
    for i in range(Lx):
        if i == 0: mpo.append(W_bulk[D-1:D,:,:,:].copy())
        elif i == Lx-1: mpo.append(W_bulk[:,:,:,0:1].copy())
        else: mpo.append(W_bulk.copy())
    return mpo

def cylinder_bonds(Ly, Lx):
    n = Ly*Lx; coords = []
    for x in range(Lx):
        rng = range(Ly) if x%2==0 else range(Ly-1,-1,-1)
        for y in rng: coords.append((y,x))
    c2s = {c:i for i,c in enumerate(coords)}
    bonds = set()
    for x in range(Lx):
        for y in range(Ly):
            if x < Lx-1:
                s1,s2 = c2s[(y,x)],c2s[(y,x+1)]
                bonds.add((min(s1,s2),max(s1,s2)))
            y2 = (y+1)%Ly
            s1,s2 = c2s[(y,x)],c2s[(y2,x)]
            if s1 != s2: bonds.add((min(s1,s2),max(s1,s2)))
    return n, sorted(bonds)

def build_mpo_std_compressed(n, bonds):
    d = 2
    active_at_cut = []
    for k in range(n-1):
        active_at_cut.append([b for b,(i,j) in enumerate(bonds) if i<=k and j>k])
    cut_maps, cut_dims = [], []
    for k in range(n-1):
        active = active_at_cut[k]
        mapping = {bi: 3*li+1 for li,bi in enumerate(active)}
        cut_maps.append(mapping); cut_dims.append(2+3*len(active))
    mpo = []
    for site in range(n):
        DL = 1 if site==0 else cut_dims[site-1]
        DR = 1 if site==n-1 else cut_dims[site]
        lm = {} if site==0 else cut_maps[site-1]
        rm = {} if site==n-1 else cut_maps[site]
        W = np.zeros((DL,d,d,DR))
        if site==0:
            if DR>1: W[0,:,:,DR-1]=Id2
        elif site==n-1: W[0,:,:,0]=Id2
        else: W[0,:,:,0]=Id2; W[DL-1,:,:,DR-1]=Id2
        for b,(i,j) in enumerate(bonds):
            if site==i:
                src = 0 if site==0 else DL-1
                if b in rm:
                    ch=rm[b]; W[src,:,:,ch]=Sp; W[src,:,:,ch+1]=Sm; W[src,:,:,ch+2]=Zd
            elif site==j:
                if b in lm:
                    ch=lm[b]; W[ch,:,:,0]=2*Sm; W[ch+1,:,:,0]=2*Sp; W[ch+2,:,:,0]=Zd
            elif i<site<j:
                if b in lm and b in rm:
                    cl2,cr2=lm[b],rm[b]
                    W[cl2,:,:,cr2]=Id2; W[cl2+1,:,:,cr2+1]=Id2; W[cl2+2,:,:,cr2+2]=Id2
        mpo.append(W)
    return mpo

# ============================================================
# DMRG engines
# ============================================================

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
    """Solve 2-site problem: dense eigh for small, Lanczos for large."""
    cl, d1, d2, cr = shapes
    dim = cl*d1*d2*cr
    
    if dim <= 1500:
        # Dense
        WW = np.einsum('wabx,xcfv->wacbfv', Wi, Wj, optimize=True)
        LWW = np.einsum('gwk,wacbfv->gkacbfv', L, WW, optimize=True)
        H = np.einsum('gkacbfv,hvj->gkacbfhj', LWW, R, optimize=True)
        H_mat = H.transpose(0,4,5,6,1,2,3,7).reshape(dim,dim)
        H_mat = (H_mat + H_mat.T)/2
        ev,vc = eigh(H_mat)
        return ev[0], vc[:,0]
    else:
        # Lanczos with precomputed LW, WR
        LW = np.einsum('pwa,wsqx->pasqx', L, Wi, optimize=True)
        WR = np.einsum('xrfv,hvb->xrfhb', Wj, R, optimize=True)
        def mv(v):
            x = v.reshape(cl, d1, d2, cr)
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
            ev0, theta = solve_2site(Ls[i], mpo[i], mpo[i+1], Rs[i+2], (cl,d1,d2,cr))
            th = theta.reshape(cl*d1, d2*cr)
            U,S,Vh = svd(th, full_matrices=False); k = min(chi, len(S))
            mps[i] = U[:,:k].reshape(cl,d1,k)
            mps[i+1] = (np.diag(S[:k])@Vh[:k,:]).reshape(k,d2,cr)
            Ls[i+1] = upL(Ls[i], mps[i], mpo[i])
        for i in range(n-2,-1,-1):
            cl,d1,_ = mps[i].shape; _,d2,cr = mps[i+1].shape
            ev0, theta = solve_2site(Ls[i], mpo[i], mpo[i+1], Rs[i+2], (cl,d1,d2,cr))
            th = theta.reshape(cl*d1, d2*cr)
            U,S,Vh = svd(th, full_matrices=False); k = min(chi, len(S))
            mps[i] = (U[:,:k]@np.diag(S[:k])).reshape(cl,d1,k)
            mps[i+1] = Vh[:k,:].reshape(k,d2,cr)
            Rs[i+1] = upR(Rs[i+2], mps[i+1], mpo[i+1])
        En = ev0; dE = abs(En-Eo)
        if verbose: print(f"  Sw{sw+1}: E={En:.10f} dE={dE:.2e}", flush=True)
        if dE < tol and sw > 0: break
        Eo = En
    return mps, En, sw+1

def exact_cylinder(Ly, Lx):
    n, bonds = cylinder_bonds(Ly, Lx)
    dim = 2**n; H = np.zeros((dim,dim))
    for (i,j) in bonds:
        for (O1,O2) in [(Sp,2*Sm),(Sm,2*Sp),(Zd,Zd)]:
            op = np.eye(1)
            for k in range(n):
                if k==i: op=np.kron(op,O1)
                elif k==j: op=np.kron(op,O2)
                else: op=np.kron(op,Id2)
            H += op
    return np.linalg.eigvalsh(H)[0]

if __name__ == '__main__':
    pass
