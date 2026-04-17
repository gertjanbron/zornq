"""B7 TEBD - complete run, 2x2x6, chi=32, dt=0.02, 40 steps."""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import time, sys

Sp = np.array([[0,1],[0,0]], dtype=complex)
Sm = np.array([[0,0],[1,0]], dtype=complex)
Zd = np.array([[1,0],[0,-1]], dtype=complex)
Id2 = np.eye(2, dtype=complex)

def embed_1q(op, q, nq):
    ops = [Id2]*nq; ops[q] = op
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def embed_2q(op1, q1, op2, q2, nq):
    ops = [Id2]*nq; ops[q1] = op1; ops[q2] = op2
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def h_intra_layer(Lx, Ly, periodic=True):
    nq = Lx*Ly; d = 2**nq
    h = np.zeros((d,d), dtype=complex)
    idx = lambda x,y: y*Lx+x
    for y in range(Ly):
        for x in range(Lx):
            for dx,dy in [(1,0),(0,1)]:
                x2,y2 = x+dx,y+dy
                if dx==1:
                    if x2>=Lx:
                        if periodic and Lx>2: x2=0
                        else: continue
                if dy==1:
                    if y2>=Ly:
                        if periodic and Ly>2: y2=0
                        else: continue
                q1,q2 = idx(x,y),idx(x2,y2)
                h += 2*embed_2q(Sp,q1,Sm,q2,nq) + 2*embed_2q(Sm,q1,Sp,q2,nq) + embed_2q(Zd,q1,Zd,q2,nq)
    return h

def h_inter_layer(Lx, Ly):
    nq = Lx*Ly; d = 2**nq
    h = np.zeros((d*d,d*d), dtype=complex)
    for q in range(nq):
        h += 2*np.kron(embed_1q(Sp,q,nq), embed_1q(Sm,q,nq))
        h += 2*np.kron(embed_1q(Sm,q,nq), embed_1q(Sp,q,nq))
        h += np.kron(embed_1q(Zd,q,nq), embed_1q(Zd,q,nq))
    return h

def apply_1site(mps, U):
    for i in range(len(mps)): mps[i] = np.einsum('ij,ajb->aib', U, mps[i])
    return mps

def apply_2site(mps, i, U, chi, d):
    cl,cr = mps[i].shape[0], mps[i+1].shape[2]
    theta = np.einsum('asc,crd->asrd', mps[i], mps[i+1]).reshape(cl,d*d,cr)
    theta = np.einsum('ij,ajb->aib', U, theta).reshape(cl*d, d*cr)
    Uv,S,Vh = svd(theta, full_matrices=False)
    k = min(chi, len(S))
    te = np.sum(S[k:]**2) if k<len(S) else 0.0
    mps[i] = Uv[:,:k].reshape(cl,d,k)
    mps[i+1] = (np.diag(S[:k])@Vh[:k,:]).reshape(k,d,cr)
    return mps, te, S[:k]

def tebd_step(mps, Uh, Ub, chi, d):
    n=len(mps); te=0
    mps = apply_1site(mps, Uh)
    for i in range(0,n-1,2): mps,e,_ = apply_2site(mps,i,Ub,chi,d); te+=e
    for i in range(1,n-1,2): mps,e,_ = apply_2site(mps,i,Ub,chi,d); te+=e
    mps = apply_1site(mps, Uh)
    return mps, te

def mps_norm(mps):
    L = np.ones((1,1), dtype=complex)
    for m in mps:
        t = np.einsum('ab,asc->bsc', L, m)
        L = np.einsum('bsc,bsd->cd', t, np.conj(m))
    return L[0,0].real

def normalize(mps):
    n = np.sqrt(abs(mps_norm(mps)))
    if n>1e-15: mps[0] /= n
    return mps

def measure_op(mps, site, Op):
    n = len(mps)
    L = np.ones((1,1), dtype=complex)
    for j in range(site):
        t = np.einsum('ab,asc->bsc', L, mps[j])
        L = np.einsum('bsc,bsd->cd', t, np.conj(mps[j]))
    t = np.einsum('ab,asc->bsc', L, mps[site])
    t_op = np.einsum('bsc,sr->brc', t, Op)
    L = np.einsum('brc,brd->cd', t_op, np.conj(mps[site]))
    for j in range(site+1, n):
        t = np.einsum('ab,asc->bsc', L, mps[j])
        L = np.einsum('bsc,bsd->cd', t, np.conj(mps[j]))
    return L[0,0]

def measure_E_mpo(mps, mpo):
    L = np.ones((1,1,1), dtype=complex)
    for i in range(len(mps)):
        t = np.einsum('gmk,ksc->gmsc', L, mps[i])
        t = np.einsum('gmsc,msrv->grvc', t, mpo[i])
        L = np.einsum('grvc,gra->avc', t, np.conj(mps[i]))
    return L[0,0,0].real

def build_mpo(Lz, Lx, Ly, periodic_xy=True):
    nq=Lx*Ly; d=2**nq; D=2+3*nq
    hl = h_intra_layer(Lx,Ly,periodic_xy)
    so,eo = [],[]
    for q in range(nq):
        so.append([embed_1q(Sp,q,nq),embed_1q(Sm,q,nq),embed_1q(Zd,q,nq)])
        eo.append([2*embed_1q(Sm,q,nq),2*embed_1q(Sp,q,nq),embed_1q(Zd,q,nq)])
    Id_d = np.eye(d,dtype=complex)
    W = np.zeros((D,d,d,D),dtype=complex)
    W[0,:,:,0]=Id_d; W[D-1,:,:,D-1]=Id_d; W[D-1,:,:,0]=hl
    for q in range(nq):
        for k in range(3):
            ch=3*q+k+1; W[D-1,:,:,ch]=so[q][k]; W[ch,:,:,0]=eo[q][k]
    mpo = []
    for i in range(Lz):
        if i==0: mpo.append(W[D-1:D,:,:,:].copy())
        elif i==Lz-1: mpo.append(W[:,:,:,0:1].copy())
        else: mpo.append(W.copy())
    return mpo

def bond_entropies(mps):
    mc = [m.copy() for m in mps]; ent = []
    for i in range(len(mc)-1):
        cl,d,cr = mc[i].shape
        U,S,Vh = svd(mc[i].reshape(cl*d,cr), full_matrices=False)
        mc[i] = U.reshape(cl,d,-1)
        mc[i+1] = np.einsum('ij,jkl->ikl', np.diag(S)@Vh, mc[i+1])
        s2 = (np.abs(S)**2); s2 = s2[s2>1e-20]; s2 /= s2.sum()
        ent.append(-np.sum(s2*np.log(s2)))
    return ent

if __name__ == '__main__':
    np.random.seed(42)
    Lx,Ly = 2,2; nq=4; d=16
    Lz = 6; chi = 32; dt = 0.02; n_steps = 40

    print("="*70)
    print(f"  B7: 3D TEBD — {Lx}x{Ly}x{Lz} ({nq*Lz}q), d={d}, chi={chi}, dt={dt}")
    print("="*70); sys.stdout.flush()

    Uh = expm(-1j*h_intra_layer(Lx,Ly,True)*dt/2)
    Ub = expm(-1j*h_inter_layer(Lx,Ly)*dt)
    mpo = build_mpo(Lz,Lx,Ly)

    Sz_op = np.zeros((d,d),dtype=complex)
    for q in range(nq): Sz_op += embed_1q(0.5*Zd,q,nq)

    kick = Lz//2
    mps = []
    for i in range(Lz):
        t = np.zeros((1,d,1),dtype=complex)
        if i==kick: t[0,d-1,0]=1  # all-down
        else: t[0,0,0]=1          # all-up
        mps.append(t)
    mps = normalize(mps)

    E0 = measure_E_mpo(mps, mpo)
    sz0 = [measure_op(mps,i,Sz_op).real for i in range(Lz)]
    S0 = bond_entropies(mps)

    print(f"Initieel: all-up + flip laag {kick}")
    print(f"  E0 = {E0:.6f}")
    print(f"  Sz = {['%+.2f'%s for s in sz0]}")
    sys.stdout.flush()

    print(f"\n{'stap':>5} {'t':>6} {'E':>12} {'dE/E':>10} {'S_max':>8} {'chi':>5} {'trunc':>10} {'cpu':>6}")
    print("-"*70)

    results = [{'t':0,'E':E0,'S':S0[:],'sz':sz0[:]}]
    t_total = time.time()

    for step in range(1, n_steps+1):
        tc = time.time()
        mps, terr = tebd_step(mps,Uh,Ub,chi,d)
        dtc = time.time()-tc

        if step%5==0 or step<=3:
            mps = normalize(mps)
            E = measure_E_mpo(mps,mpo)
            S = bond_entropies(mps)
            sz = [measure_op(mps,i,Sz_op).real for i in range(Lz)]
            chi_eff = max(mps[i].shape[2] for i in range(Lz-1))
            dEr = (E-E0)/abs(E0) if abs(E0)>1e-10 else E-E0
            results.append({'t':step*dt,'E':E,'S':S[:],'sz':sz[:]})
            print(f"{step:5d} {step*dt:6.2f} {E:12.6f} {dEr:10.2e} {max(S):8.4f} {chi_eff:5d} {terr:10.2e} {dtc:6.2f}s")
            sys.stdout.flush()

    T = time.time()-t_total
    print(f"\nTotale tijd: {T:.1f}s")
    Ef = results[-1]['E']
    print(f"\n{'='*70}")
    print(f"RESULTAAT: {Lx}x{Ly}x{Lz} ({nq*Lz}q), chi={chi}, dt={dt}")
    print(f"  E0={E0:.6f} -> Ef={Ef:.6f}, drift={abs(Ef-E0)/abs(E0):.4e}")
    print(f"  S_max(final)={max(results[-1]['S']):.4f}")

    print(f"\nEntanglement groei rond kick (bond {kick-1}-{kick}):")
    for r in results:
        s = r['S'][kick-1]
        bar = '#'*min(50,int(s*12))
        print(f"  t={r['t']:5.2f}: S={s:.4f} {bar}")

    print(f"\nSz-profiel (magnetisatie schokgolf):")
    for r in results:
        sz_bars = ''
        for s in r['sz']:
            if s > 0.1: sz_bars += f' +{s:.1f}'
            elif s < -0.1: sz_bars += f' {s:.1f}'
            else: sz_bars += f'  0.0'
        print(f"  t={r['t']:5.2f}: [{sz_bars} ]")
