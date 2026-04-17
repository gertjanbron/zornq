"""B7b: Adaptive vs Fixed chi — compact test on 2x2x6."""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import time, sys

Sp=np.array([[0,1],[0,0]],dtype=complex); Sm=Sp.T.copy(); Zd=np.diag([1.,-1.]).astype(complex); Id2=np.eye(2,dtype=complex)
def e1(op,q,nq):
    ops=[Id2]*nq;ops[q]=op;r=ops[0]
    for o in ops[1:]:r=np.kron(r,o)
    return r
def e2(o1,q1,o2,q2,nq):
    ops=[Id2]*nq;ops[q1]=o1;ops[q2]=o2;r=ops[0]
    for o in ops[1:]:r=np.kron(r,o)
    return r
def h_intra(Lx,Ly):
    nq=Lx*Ly;d=2**nq;h=np.zeros((d,d),dtype=complex)
    idx=lambda x,y:y*Lx+x
    for y in range(Ly):
        for x in range(Lx):
            for dx,dy in[(1,0),(0,1)]:
                x2,y2=x+dx,y+dy
                if dx==1 and x2>=Lx: continue
                if dy==1 and y2>=Ly: continue
                q1,q2=idx(x,y),idx(x2,y2)
                h+=2*e2(Sp,q1,Sm,q2,nq)+2*e2(Sm,q1,Sp,q2,nq)+e2(Zd,q1,Zd,q2,nq)
    return h
def h_inter(Lx,Ly):
    nq=Lx*Ly;d=2**nq;h=np.zeros((d*d,d*d),dtype=complex)
    for q in range(nq):
        h+=2*np.kron(e1(Sp,q,nq),e1(Sm,q,nq))+2*np.kron(e1(Sm,q,nq),e1(Sp,q,nq))+np.kron(e1(Zd,q,nq),e1(Zd,q,nq))
    return h

def a1(mps,U):
    for i in range(len(mps)):mps[i]=np.einsum('ij,ajb->aib',U,mps[i])
    return mps
def a2f(mps,i,U,chi,d):
    cl,cr=mps[i].shape[0],mps[i+1].shape[2]
    th=np.einsum('asc,crd->asrd',mps[i],mps[i+1]).reshape(cl,d*d,cr)
    th=np.einsum('ij,ajb->aib',U,th).reshape(cl*d,d*cr)
    Uv,S,Vh=svd(th,full_matrices=False);k=min(chi,len(S))
    te=np.sum(np.abs(S[k:])**2) if k<len(S) else 0.
    mps[i]=Uv[:,:k].reshape(cl,d,k);mps[i+1]=(np.diag(S[:k])@Vh[:k,:]).reshape(k,d,cr)
    return mps,te,k
def a2a(mps,i,U,eps,cmax,d):
    cl,cr=mps[i].shape[0],mps[i+1].shape[2]
    th=np.einsum('asc,crd->asrd',mps[i],mps[i+1]).reshape(cl,d*d,cr)
    th=np.einsum('ij,ajb->aib',U,th).reshape(cl*d,d*cr)
    Uv,S,Vh=svd(th,full_matrices=False)
    Sa=np.abs(S)
    if Sa[0]>1e-15: k=max(1,int(np.sum(Sa>eps*Sa[0])))
    else: k=1
    k=min(k,cmax,len(S))
    te=np.sum(Sa[k:]**2) if k<len(S) else 0.
    mps[i]=Uv[:,:k].reshape(cl,d,k);mps[i+1]=(np.diag(S[:k])@Vh[:k,:]).reshape(k,d,cr)
    return mps,te,k

def step_f(mps,Uh,Ub,chi,d):
    n=len(mps);te=0;cs=[0]*(n-1)
    mps=a1(mps,Uh)
    for i in range(0,n-1,2):mps,e,k=a2f(mps,i,Ub,chi,d);te+=e;cs[i]=k
    for i in range(1,n-1,2):mps,e,k=a2f(mps,i,Ub,chi,d);te+=e;cs[i]=k
    mps=a1(mps,Uh);return mps,te,cs

def step_a(mps,Uh,Ub,eps,cmax,d):
    n=len(mps);te=0;cs=[0]*(n-1)
    mps=a1(mps,Uh)
    for i in range(0,n-1,2):mps,e,k=a2a(mps,i,Ub,eps,cmax,d);te+=e;cs[i]=k
    for i in range(1,n-1,2):mps,e,k=a2a(mps,i,Ub,eps,cmax,d);te+=e;cs[i]=k
    mps=a1(mps,Uh);return mps,te,cs

def norm(mps):
    L=np.ones((1,1),dtype=complex)
    for m in mps:t=np.einsum('ab,asc->bsc',L,m);L=np.einsum('bsc,bsd->cd',t,np.conj(m))
    n=np.sqrt(abs(L[0,0].real))
    if n>1e-15:mps[0]/=n
    return mps

def mop(mps,site,Op):
    n=len(mps);L=np.ones((1,1),dtype=complex)
    for j in range(site):t=np.einsum('ab,asc->bsc',L,mps[j]);L=np.einsum('bsc,bsd->cd',t,np.conj(mps[j]))
    t=np.einsum('ab,asc->bsc',L,mps[site]);t2=np.einsum('bsc,sr->brc',t,Op);L=np.einsum('brc,brd->cd',t2,np.conj(mps[site]))
    for j in range(site+1,n):t=np.einsum('ab,asc->bsc',L,mps[j]);L=np.einsum('bsc,bsd->cd',t,np.conj(mps[j]))
    return L[0,0]

def mE(mps,mpo):
    L=np.ones((1,1,1),dtype=complex)
    for i in range(len(mps)):
        t=np.einsum('gmk,ksc->gmsc',L,mps[i]);t=np.einsum('gmsc,msrv->grvc',t,mpo[i]);L=np.einsum('grvc,gra->avc',t,np.conj(mps[i]))
    return L[0,0,0].real

def bmpo(Lz,Lx,Ly):
    nq=Lx*Ly;d=2**nq;D=2+3*nq;hl=h_intra(Lx,Ly)
    so,eo=[],[]
    for q in range(nq):so.append([e1(Sp,q,nq),e1(Sm,q,nq),e1(Zd,q,nq)]);eo.append([2*e1(Sm,q,nq),2*e1(Sp,q,nq),e1(Zd,q,nq)])
    Id_d=np.eye(d,dtype=complex);W=np.zeros((D,d,d,D),dtype=complex)
    W[0,:,:,0]=Id_d;W[D-1,:,:,D-1]=Id_d;W[D-1,:,:,0]=hl
    for q in range(nq):
        for k in range(3):ch=3*q+k+1;W[D-1,:,:,ch]=so[q][k];W[ch,:,:,0]=eo[q][k]
    mpo=[]
    for i in range(Lz):
        if i==0:mpo.append(W[D-1:D,:,:,:].copy())
        elif i==Lz-1:mpo.append(W[:,:,:,0:1].copy())
        else:mpo.append(W.copy())
    return mpo

def mem(mps):return sum(m.size for m in mps)

def init(Lz,d,kick):
    mps=[]
    for i in range(Lz):
        t=np.zeros((1,d,1),dtype=complex)
        if i==kick:t[0,d-1,0]=1
        else:t[0,0,0]=1
        mps.append(t)
    return norm(mps)

if __name__=='__main__':
    np.random.seed(42)
    Lx,Ly=2,2;nq=4;d=16;Lz=6;dt=0.02;NS=30;kick=Lz//2
    print("="*70)
    print(f"  B7b: Adaptive chi — {Lx}x{Ly}x{Lz} ({nq*Lz}q), dt={dt}, {NS} steps")
    print("="*70);sys.stdout.flush()
    Uh=expm(-1j*h_intra(Lx,Ly)*dt/2);Ub=expm(-1j*h_inter(Lx,Ly)*dt)
    mpo=bmpo(Lz,Lx,Ly)
    Sz=np.zeros((d,d),dtype=complex)
    for q in range(nq):Sz+=e1(0.5*Zd,q,nq)
    E0=mE(init(Lz,d,kick),mpo)

    configs = [
        ("Fixed chi=32", 'f', 32, None, None),
        ("Adaptive eps=1e-3 max=64", 'a', None, 1e-3, 64),
        ("Adaptive eps=1e-4 max=96", 'a', None, 1e-4, 96),
    ]
    all_res = {}
    for label,mode,chi_f,eps,cmax in configs:
        print(f"\n--- {label} ---");sys.stdout.flush()
        mps=init(Lz,d,kick);res=[];t0=time.time()
        for s in range(1,NS+1):
            if mode=='f': mps,te,cs=step_f(mps,Uh,Ub,chi_f,d)
            else: mps,te,cs=step_a(mps,Uh,Ub,eps,cmax,d)
            if s%5==0 or s==1:
                mps=norm(mps);E=mE(mps,mpo)
                sz=[mop(mps,i,Sz).real for i in range(Lz)]
                m=mem(mps)
                res.append({'s':s,'t':s*dt,'E':E,'te':te,'cs':cs[:],'sz':sz,'m':m})
                print(f"  t={s*dt:.2f} E={E:.4f} chi={cs} mem={m:6d} trunc={te:.3e}",flush=True)
        dtt=time.time()-t0
        all_res[label]=(res,dtt)
        print(f"  Tijd: {dtt:.1f}s")

    print("\n"+"="*70)
    print("VERGELIJKING")
    print("="*70)
    print(f"\n{'Methode':<32} {'Ef':>9} {'dE/E':>10} {'MemFin':>8} {'MemAvg':>8} {'Tijd':>6} {'SzKick':>8}")
    print("-"*82)
    for label,(res,dtt) in all_res.items():
        Ef=res[-1]['E'];mf=res[-1]['m'];ma=int(np.mean([r['m'] for r in res]))
        szk=res[-1]['sz'][kick]
        print(f"{label:<32} {Ef:9.4f} {(Ef-E0)/abs(E0):10.2e} {mf:8d} {ma:8d} {dtt:6.1f}s {szk:+8.4f}")

    print(f"\nChi-profiel (t={res[-1]['t']:.2f}):")
    for label,(res,dtt) in all_res.items():
        print(f"  {label}: {res[-1]['cs']}")

    print(f"\nSz-profiel (t={res[-1]['t']:.2f}):")
    for label,(res,dtt) in all_res.items():
        print(f"  {label}: {['%+.2f'%s for s in res[-1]['sz']]}")
