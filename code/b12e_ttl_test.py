"""B12e: TTL test — leeftijdstracking vs SVD"""
import numpy as np
from numpy.linalg import svd
import time

def heisenberg_gate(J, dt):
    X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.array([[1,0],[0,-1]])
    H2 = J*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z))
    ev,U = np.linalg.eigh(H2)
    return (U @ np.diag(np.exp(-1j*dt*ev)) @ U.conj().T)

def init_neel(n):
    mps=[]
    for i in range(n):
        T=np.zeros((1,2,1),dtype=complex); T[0,i%2,0]=1.0
        mps.append(T)
    return mps

def apply2(mps, gate, s, chi):
    cl,cr = mps[s].shape[0], mps[s+1].shape[2]
    Th = np.einsum('aib,bjc->aijc', mps[s], mps[s+1])
    G = gate.reshape(2,2,2,2)
    Th = np.einsum('ijkl,aklc->aijc', G, Th)
    mat = Th.reshape(cl*2, 2*cr)
    U,S,V = svd(mat, full_matrices=False)
    k = min(max(1,int(np.sum(np.abs(S)>1e-12*np.abs(S[0])))), chi)
    disc = np.sum(np.abs(S[k:])**2)/(np.sum(np.abs(S)**2)+1e-30)
    mps[s] = U[:,:k].reshape(cl,2,k)
    mps[s+1] = (np.diag(S[:k])@V[:k,:]).reshape(k,2,cr)
    return disc

def measure_sz(mps, site):
    Z=np.diag([1.0,-1.0]).astype(complex)
    L=np.ones((1,1),dtype=complex)
    for i,T in enumerate(mps):
        if i==site:
            L=np.einsum('xy,xiz,ij,yjw->zw', L, T.conj(), Z, T)
        else:
            L=np.einsum('xy,xiz,yiw->zw', L, T.conj(), T)
    return L[0,0].real

def tebd_step(mps, gate, chi):
    n=len(mps); d=0.0
    for s in range(0,n-1,2): d+=apply2(mps,gate,s,chi)
    for s in range(1,n-1,2): d+=apply2(mps,gate,s,chi)
    return d

n=16; J=1.0; dt=0.05; steps=40; chi=16
gate=heisenberg_gate(J,dt)

print("="*60)
print(f"TTL TEST: {n}q Heisenberg, dt={dt}, {steps} stappen, chi={chi}")
print("="*60)

# Standaard
mps0=init_neel(n)
t0=time.time()
d_tot=0
for _ in range(steps): d_tot+=tebd_step(mps0,gate,chi)
t_std=time.time()-t0
sz0=[measure_sz(mps0,i) for i in range(n)]
chi0=[mps0[i].shape[2] for i in range(n-1)]
print(f"\nStandaard SVD: {t_std:.3f}s, max_chi={max(chi0)}, disc={d_tot:.6f}")
print(f"  Sz[0:6]={[f'{s:.4f}' for s in sz0[:6]]}")

# TTL varianten
for max_age in [3, 7, 15, 30]:
    mps1=init_neel(n)
    ages=[0]*(n-1)
    d_tot=0
    t0=time.time()
    for step in range(steps):
        for s in range(0,n-1,2):
            ages[s]+=1
            c = min(chi,4) if ages[s]>max_age else chi
            d_tot+=apply2(mps1,gate,s,c)
            ages[s]=0
        for s in range(1,n-1,2):
            ages[s]+=1
            c = min(chi,4) if ages[s]>max_age else chi
            d_tot+=apply2(mps1,gate,s,c)
            ages[s]=0
    t_ttl=time.time()-t0
    sz1=[measure_sz(mps1,i) for i in range(n)]
    err=max(abs(a-b) for a,b in zip(sz0,sz1))
    spd=t_std/t_ttl if t_ttl>0 else 0
    print(f"\nTTL age={max_age:>2d}: {t_ttl:.3f}s ({spd:.2f}×), disc={d_tot:.6f}, Sz_err={err:.6f}")

# Nu de ECHTE vergelijking: wat als we gewoon een lagere chi gebruiken?
for c in [4, 8, 12]:
    mps2=init_neel(n)
    d_tot=0
    t0=time.time()
    for _ in range(steps): d_tot+=tebd_step(mps2,gate,c)
    t_lo=time.time()-t0
    sz2=[measure_sz(mps2,i) for i in range(n)]
    err2=max(abs(a-b) for a,b in zip(sz0,sz2))
    spd2=t_std/t_lo if t_lo>0 else 0
    print(f"\nLage chi={c:>2d}: {t_lo:.3f}s ({spd2:.2f}×), disc={d_tot:.6f}, Sz_err={err2:.6f}")

print("\n" + "="*60)
print("CONCLUSIE")
print("="*60)
