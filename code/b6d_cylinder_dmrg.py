"""B6d v3: Cylindrical DMRG with correct compressed MPO."""
import numpy as np
from numpy.linalg import svd, eigh
import time

def cylinder_sites(Ly, Lx):
    n = Ly * Lx; coords = []
    for x in range(Lx):
        rng = range(Ly) if x % 2 == 0 else range(Ly-1, -1, -1)
        for y in rng: coords.append((y, x))
    c2s = {c: i for i, c in enumerate(coords)}
    bonds = set()
    for x in range(Lx):
        for y in range(Ly):
            if x < Lx-1:
                s1, s2 = c2s[(y,x)], c2s[(y,x+1)]
                bonds.add((min(s1,s2), max(s1,s2)))
            y2 = (y+1) % Ly
            s1, s2 = c2s[(y,x)], c2s[(y2,x)]
            if s1 != s2: bonds.add((min(s1,s2), max(s1,s2)))
    return n, coords, sorted(bonds)

def build_mpo_compressed(n, bonds):
    """Build MPO with site-dependent bond dimension.
    Convention: channel 0 = accumulator (left), channel D-1 = source (right).
    """
    d = 2
    Sp=np.array([[0,1],[0,0]]); Sm=np.array([[0,0],[1,0]])
    Zd=np.array([[1,0],[0,-1]]); Id=np.eye(2)
    
    # Active bonds at each cut
    active_at_cut = []
    for k in range(n-1):
        active = [b for b,(i,j) in enumerate(bonds) if i <= k and j > k]
        active_at_cut.append(active)
    
    # Channel mapping at each cut
    # Channel 0: accumulator
    # Channels 1..3*n_active: operator channels (3 per active bond)
    # Channel D-1 = 3*n_active + 1: source
    cut_maps = []  # bond_idx -> base channel offset at this cut
    cut_dims = []
    for k in range(n-1):
        active = active_at_cut[k]
        mapping = {}
        for local_idx, bond_idx in enumerate(active):
            mapping[bond_idx] = 3*local_idx + 1
        D = 2 + 3*len(active)  # 0=accum, 1..3n=ops, D-1=source
        cut_maps.append(mapping)
        cut_dims.append(D)
    
    mpo = []
    for site in range(n):
        # Dimensions
        DL = 1 if site == 0 else cut_dims[site-1]
        DR = 1 if site == n-1 else cut_dims[site]
        left_map = {} if site == 0 else cut_maps[site-1]
        right_map = {} if site == n-1 else cut_maps[site]
        
        W = np.zeros((DL, d, d, DR))
        
        # Source and accumulator pass-throughs
        # Site 0: DL=1, row 0 = source (NOT accumulator)
        # Site n-1: DR=1, col 0 = accumulator (NOT source)
        # Bulk: row 0 = accum→accum, row DL-1 = source→source
        
        if site == 0:
            # Row 0 IS the source. No accumulator input.
            if DR > 1:
                W[0, :, :, DR-1] = Id  # source pass-through
        elif site == n-1:
            # Col 0 IS the accumulator. No source output.
            W[0, :, :, 0] = Id  # accumulator pass-through
        else:
            # Bulk: both
            W[0, :, :, 0] = Id       # accumulator pass-through
            W[DL-1, :, :, DR-1] = Id  # source pass-through
        
        # Operator starts, ends, and pass-throughs
        for b, (i, j) in enumerate(bonds):
            if site == i:
                # Start: source row -> operator channels
                src_row = 0 if site == 0 else DL-1  # source row
                if b in right_map:
                    ch = right_map[b]
                    W[src_row, :, :, ch]   = Sp
                    W[src_row, :, :, ch+1] = Sm
                    W[src_row, :, :, ch+2] = Zd
            elif site == j:
                # End: operator channels -> accumulator col
                if b in left_map:
                    ch = left_map[b]
                    W[ch,   :, :, 0] = 2*Sm
                    W[ch+1, :, :, 0] = 2*Sp
                    W[ch+2, :, :, 0] = Zd
            elif i < site < j:
                # Pass-through
                if b in left_map and b in right_map:
                    ch_l, ch_r = left_map[b], right_map[b]
                    W[ch_l,   :, :, ch_r]   = Id
                    W[ch_l+1, :, :, ch_r+1] = Id
                    W[ch_l+2, :, :, ch_r+2] = Id
        
        mpo.append(W)
    return mpo

# DMRG engine (same as before)
def random_mps(n,d,chi):
    mps=[]
    for i in range(n):
        cl=1 if i==0 else min(chi,d**i,d**(n-i))
        cr=1 if i==n-1 else min(chi,d**(i+1),d**(n-1-i))
        mps.append(np.random.randn(cl,d,cr))
    return mps
def right_canon(mps):
    for i in range(len(mps)-1,0,-1):
        cl,d,cr=mps[i].shape
        U,S,Vh=svd(mps[i].reshape(cl,d*cr),full_matrices=False)
        mps[i]=Vh.reshape(-1,d,cr)
        mps[i-1]=np.einsum('ijk,kl->ijl',mps[i-1],U@np.diag(S))
    mps[0]/=np.linalg.norm(mps[0])
    return mps
def upL(L,A,W):
    t=np.einsum('gmk,ksc->gmsc',L,A)
    t=np.einsum('gmsc,msrv->grvc',t,W)
    return np.einsum('grvc,gra->avc',t,A)
def upR(R,B,W):
    t=np.einsum('ksj,bvj->ksbv',B,R)
    t=np.einsum('msrv,ksbv->mrkb',W,t)
    return np.einsum('arb,mrkb->amk',B,t)
def build_Heff(L,Wi,Wj,R,sh):
    cl,d1,d2,cr=sh
    WW=np.einsum('wabx,xcfv->wacbfv',Wi,Wj)
    LWW=np.einsum('gwk,wacbfv->gkacbfv',L,WW)
    H=np.einsum('gkacbfv,hvj->gkacbfhj',LWW,R)
    dim=cl*d1*d2*cr
    return (H.transpose(0,4,5,6,1,2,3,7).reshape(dim,dim)+
            H.transpose(0,4,5,6,1,2,3,7).reshape(dim,dim).T)/2
def dmrg(n,mpo,chi,nsw=20,tol=1e-10,verbose=False):
    mps=random_mps(n,2,chi); mps=right_canon(mps)
    Rs=[None]*(n+1); Rs[n]=np.ones((1,1,1))
    for i in range(n-1,-1,-1): Rs[i]=upR(Rs[i+1],mps[i],mpo[i])
    Ls=[None]*(n+1); Ls[0]=np.ones((1,1,1))
    Eo=1e10
    for sw in range(nsw):
        for i in range(n-1):
            cl,d1,_=mps[i].shape; _,d2,cr=mps[i+1].shape
            H=build_Heff(Ls[i],mpo[i],mpo[i+1],Rs[i+2],(cl,d1,d2,cr))
            ev,vc=eigh(H); th=vc[:,0].reshape(cl*d1,d2*cr)
            U,S,Vh=svd(th,full_matrices=False); k=min(chi,len(S))
            mps[i]=U[:,:k].reshape(cl,d1,k)
            mps[i+1]=(np.diag(S[:k])@Vh[:k,:]).reshape(k,d2,cr)
            Ls[i+1]=upL(Ls[i],mps[i],mpo[i])
        for i in range(n-2,-1,-1):
            cl,d1,_=mps[i].shape; _,d2,cr=mps[i+1].shape
            H=build_Heff(Ls[i],mpo[i],mpo[i+1],Rs[i+2],(cl,d1,d2,cr))
            ev,vc=eigh(H); th=vc[:,0].reshape(cl*d1,d2*cr)
            U,S,Vh=svd(th,full_matrices=False); k=min(chi,len(S))
            mps[i]=(U[:,:k]@np.diag(S[:k])).reshape(cl,d1,k)
            mps[i+1]=Vh[:k,:].reshape(k,d2,cr)
            Rs[i+1]=upR(Rs[i+2],mps[i+1],mpo[i+1])
        En=ev[0]; dE=abs(En-Eo)
        if verbose: print(f"  Sw{sw+1}: E={En:.8f} dE={dE:.2e}",flush=True)
        if dE<tol and sw>0: break
        Eo=En
    return mps,En,sw+1

np.random.seed(42)
print("B6d v3: Cylinder DMRG (compressed MPO)",flush=True)

# 3x3 cylinder (exact: -17.5746275041)
Ly,Lx=3,3; n,_,bonds=cylinder_sites(Ly,Lx)
mpo=build_mpo_compressed(n,bonds)
Ds=[w.shape[0] for w in mpo]
print(f"\n3x3 cyl: n={n}, {len(bonds)} bonds, Ds={Ds}",flush=True)
for chi in [8,16]:
    t0=time.time()
    _,E,sw=dmrg(n,mpo,chi,nsw=20)
    print(f"  chi={chi}: E={E:.10f} gap={E+17.5746275041:.2e} sw={sw} t={time.time()-t0:.2f}s",flush=True)

# 4x3 cylinder (exact: -30.1205486760)
Ly,Lx=4,3; n,_,bonds=cylinder_sites(Ly,Lx)
mpo=build_mpo_compressed(n,bonds)
Ds=[w.shape[0] for w in mpo]
print(f"\n4x3 cyl: n={n}, {len(bonds)} bonds, D_max={max(Ds)}",flush=True)
for chi in [16,32,64]:
    t0=time.time()
    _,E,sw=dmrg(n,mpo,chi,nsw=15,tol=1e-10)
    print(f"  chi={chi}: E={E:.10f} gap={E+30.1205486760:.2e} sw={sw} t={time.time()-t0:.1f}s",flush=True)

# 4x6 cylinder (24 sites)
Ly,Lx=4,6; n,_,bonds=cylinder_sites(Ly,Lx)
mpo=build_mpo_compressed(n,bonds)
print(f"\n4x6 cyl: n={n}, {len(bonds)} bonds, D_max={max(w.shape[0] for w in mpo)}",flush=True)
t0=time.time()
_,E,sw=dmrg(n,mpo,32,nsw=8,tol=1e-8,verbose=True)
print(f"  chi=32: E={E:.8f} E/site={E/n:.6f} sw={sw} t={time.time()-t0:.1f}s",flush=True)

