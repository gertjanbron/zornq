"""
B10b deel 2: Gamma hangt af van roostergrootte — waarom?

Hypothese: gamma* ~ 1/gemiddelde_graad
- 1D keten: graad ~2 (rand=1), gamma*=0.40
- 2D rooster: graad ~4 (rand=2-3), gamma* ~0.32-0.35
- Bij groter rooster: meer bulk (graad=4), minder rand → gamma daalt

Test: optimaliseer gamma voor verschillende roostermaten, meet convergentie.
"""
import numpy as np
from numpy.linalg import svd
import time, sys

# Hergebruik engine uit b10b_2d_optimizer.py (inline voor snelheid)
def bit_patterns(Ly):
    d=2**Ly
    return np.array([[(idx>>(Ly-1-q))&1 for q in range(Ly)] for idx in range(d)])

def Rx(t):
    c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-1j*s],[-1j*s,c]],dtype=complex)

class CG:
    def __init__(s,Lx,Ly,chi=256):
        s.Lx,s.Ly,s.d,s.chi=Lx,Ly,2**Ly,chi
        s.bp=bit_patterns(Ly)
        H1=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
        s.Hd=np.ones((s.d,s.d),dtype=complex)
        for q in range(Ly): s.Hd*=H1[s.bp[:,q:q+1],s.bp[:,q:q+1].T]
        s.ne=Lx*(Ly-1)+(Lx-1)*Ly
        # Average degree
        n=Lx*Ly; s.avg_deg = 2*s.ne/n
    
    def _zzi(s,g):
        diag=np.ones(s.d,dtype=complex)
        for y in range(s.Ly-1):
            diag*=np.exp(-1j*g*(1-2*s.bp[:,y].astype(float))*(1-2*s.bp[:,y+1].astype(float)))
        return diag
    def _zze(s,g):
        d=s.d; iL=np.arange(d*d)//d; iR=np.arange(d*d)%d
        diag=np.ones(d*d,dtype=complex)
        for y in range(s.Ly):
            diag*=np.exp(-1j*g*(1-2*s.bp[iL,y].astype(float))*(1-2*s.bp[iR,y].astype(float)))
        return diag
    def _rxc(s,b):
        rx=Rx(2*b); R=np.ones((s.d,s.d),dtype=complex)
        for q in range(s.Ly): R*=rx[s.bp[:,q:q+1],s.bp[:,q:q+1].T]
        return R
    def _gates(s,g,b):
        gt=[]
        for x in range(s.Lx): gt.append(('f',x,s.Hd))
        zi=s._zzi(g); ze=s._zze(g); rx=s._rxc(b)
        for x in range(s.Lx): gt.append(('d1',x,zi))
        for x in range(s.Lx-1): gt.append(('d2',x,ze))
        for x in range(s.Lx): gt.append(('f',x,rx))
        return gt
    def _id(s):
        return [np.eye(s.d,dtype=complex).reshape(1,s.d,s.d,1).copy() for _ in range(s.Lx)]
    def _obs(s,x1,y1,x2,y2):
        m=s._id()
        if x1==x2:
            dg=(1-2*s.bp[:,y1].astype(float))*(1-2*s.bp[:,y2].astype(float))
            m[x1]=np.diag(dg.astype(complex)).reshape(1,s.d,s.d,1)
        else:
            for c,y in [(x1,y1),(x2,y2)]:
                dg=(1-2*s.bp[:,y].astype(float)).astype(complex)
                m[c]=np.diag(dg).reshape(1,s.d,s.d,1)
        return m
    def _ap1(s,m,i,U):
        Ud=U.conj().T; W=np.einsum('ij,ajkb->aikb',Ud,m[i])
        m[i]=np.einsum('ajkb,kl->ajlb',W,U); return m
    def _ap1d(s,m,i,dg):
        cd=np.conj(dg)
        m[i]=m[i]*cd[None,:,None,None]*dg[None,None,:,None]; return m
    def _ap2d(s,m,i,dd):
        d=s.d; cl=m[i].shape[0]; cr=m[i+1].shape[3]
        Th=np.einsum('aijc,cklb->aijklb',m[i],m[i+1])
        cd=np.conj(dd).reshape(d,d); df=dd.reshape(d,d)
        Th=Th*cd[None,:,None,:,None,None]*df[None,None,:,None,:,None]
        mat=Th.reshape(cl*d*d,d*d*cr)
        U_s,S,V=svd(mat,full_matrices=False); Sa=np.abs(S)
        k=max(1,int(np.sum(Sa>1e-12*Sa[0]))) if Sa[0]>1e-15 else 1
        k=min(k,s.chi)
        m[i]=U_s[:,:k].reshape(cl,d,d,k)
        m[i+1]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,d,cr)
        return m
    def _evol(s,m,gt):
        for t,i,U in reversed(gt):
            if t=='f': m=s._ap1(m,i,U)
            elif t=='d1': m=s._ap1d(m,i,U)
            else: m=s._ap2d(m,i,U)
        return m
    def _exp(s,m):
        L=np.ones((1,),dtype=complex)
        for W in m: L=np.einsum('a,ab->b',L,W[:,0,0,:])
        return L[0]
    def ratio(s,g,b):
        gt=s._gates(g,b)
        tot=0.0
        for x in range(s.Lx):
            for y in range(s.Ly-1):
                m=s._obs(x,y,x,y+1); m=s._evol(m,gt)
                tot+=(1-s._exp(m).real)/2
        for x in range(s.Lx-1):
            for y in range(s.Ly):
                m=s._obs(x,y,x+1,y); m=s._evol(m,gt)
                tot+=(1-s._exp(m).real)/2
        return tot/s.ne

np.random.seed(42)
print("=" * 70)
print("  B10b: Gamma vs roostergrootte en gemiddelde graad")
print("=" * 70)

# Beta is universeel ~1.178. Zoek alleen gamma.
beta_fix = 1.1778

results = []
for Lx, Ly in [(2,2),(3,2),(4,2),(5,2),(6,2),(3,3),(4,3),(5,3),(4,4),(5,4),(6,3)]:
    n = Lx*Ly
    if Ly > 2 and n > 18: continue  # te traag
    
    eng = CG(Lx, Ly)
    
    # Golden section search for gamma
    a, b_gs = 0.15, 0.55
    gr = (np.sqrt(5)+1)/2
    for _ in range(20):
        c = b_gs - (b_gs-a)/gr
        d = a + (b_gs-a)/gr
        rc = eng.ratio(c, beta_fix)
        rd = eng.ratio(d, beta_fix)
        if rc > rd: b_gs = d
        else: a = c
    g_opt = (a+b_gs)/2
    r_opt = eng.ratio(g_opt, beta_fix)
    
    results.append((Lx, Ly, n, eng.ne, eng.avg_deg, g_opt, r_opt))
    print(f"  {Lx}x{Ly} ({n:2d}q, {eng.ne:2d}e): "
          f"avg_deg={eng.avg_deg:.2f}, gamma*={g_opt:.4f}, ratio={r_opt:.6f}")
    sys.stdout.flush()

# Analyse
print("\n--- Analyse ---")
print("Hypothese: gamma* ∝ 1/avg_degree")
for Lx,Ly,n,ne,deg,g,r in results:
    product = g * deg
    print(f"  {Lx}x{Ly}: gamma*×avg_deg = {product:.4f}")

# Check thermodynamic limit for Ly=2 (vary Lx)
print("\n--- Thermodynamic limiet Ly=2 ---")
ly2 = [(Lx,Ly,n,ne,d,g,r) for Lx,Ly,n,ne,d,g,r in results if Ly==2]
for Lx,Ly,n,ne,deg,g,r in ly2:
    print(f"  Lx={Lx}: gamma*={g:.4f}, ratio={r:.6f}, avg_deg={deg:.2f}")

print("\nDone.")
