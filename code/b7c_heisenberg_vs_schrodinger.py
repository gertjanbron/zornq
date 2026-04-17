"""
B7c: Heisenberg vs Schrodinger TEBD — 1D Heisenberg chain (d=2).
Compare operator entanglement (MPO) vs state entanglement (MPS).
n=10, d=2, chi_max=64, kick at site n//2.
"""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import time, sys

Sp=np.array([[0,1],[0,0]],dtype=complex);Sm=Sp.T.copy()
Zd=np.diag([1.,-1.]).astype(complex);Id2=np.eye(2,dtype=complex)
Sz=0.5*Zd; d=2

h2 = 2*np.kron(Sp,Sm)+2*np.kron(Sm,Sp)+np.kron(Zd,Zd)

# === MPS TEBD (Schrodinger) ===
def mps_step(mps,Ub,chi):
    n=len(mps);te=0
    for parity in [0,1]:
        for i in range(parity,n-1,2):
            cl,cr=mps[i].shape[0],mps[i+1].shape[2]
            th=np.einsum('asc,crd->asrd',mps[i],mps[i+1]).reshape(cl,d*d,cr)
            th=np.einsum('ij,ajb->aib',Ub,th).reshape(cl*d,d*cr)
            U,S,V=svd(th,full_matrices=False);k=min(chi,len(S))
            Sa=np.abs(S)
            if Sa[0]>1e-15:ka=max(1,int(np.sum(Sa>1e-12*Sa[0])));k=min(k,ka)
            te+=np.sum(Sa[k:]**2) if k<len(S) else 0
            mps[i]=U[:,:k].reshape(cl,d,k)
            mps[i+1]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,cr)
    return mps,te

def mps_expect(mps,site,Op):
    n=len(mps);L=np.ones((1,1),dtype=complex)
    for j in range(n):
        if j==site:
            t=np.einsum('ab,asc->bsc',L,mps[j])
            t2=np.einsum('bsc,sr->brc',t,Op)
            L=np.einsum('brc,brd->cd',t2,np.conj(mps[j]))
        else:
            t=np.einsum('ab,asc->bsc',L,mps[j])
            L=np.einsum('bsc,bsd->cd',t,np.conj(mps[j]))
    return L[0,0]

def norm_mps(mps):
    L=np.ones((1,1),dtype=complex)
    for m in mps:t=np.einsum('ab,asc->bsc',L,m);L=np.einsum('bsc,bsd->cd',t,np.conj(m))
    n2=np.sqrt(abs(L[0,0].real))
    if n2>1e-15:mps[0]/=n2
    return mps

def chi_profile(mps):
    return [mps[i].shape[2] for i in range(len(mps)-1)]

# === MPO TEBD (Heisenberg) ===
def mpo_step(mpo,Ub,chi):
    n=len(mpo);te=0
    Ub_d=Ub.conj().T
    Ud4=Ub_d.reshape(d,d,d,d); Uf4=Ub.reshape(d,d,d,d)
    for parity in [0,1]:
        for i in range(parity,n-1,2):
            cl=mpo[i].shape[0];cr=mpo[i+1].shape[3]
            Th=np.einsum('abce,edfg->abcdfg',mpo[i],mpo[i+1])
            Th=np.einsum('ijkl,akclef->aicjef',Ud4,Th)
            Th=np.einsum('ijkl,abkdlf->aibjdf',Uf4,Th)
            Th=Th.transpose(0,2,1,4,3,5)
            mat=Th.reshape(cl*d*d, d*d*cr)
            U,S,V=svd(mat,full_matrices=False);k=min(chi,len(S))
            Sa=np.abs(S)
            if Sa[0]>1e-15:ka=max(1,int(np.sum(Sa>1e-12*Sa[0])));k=min(k,ka)
            te+=np.sum(Sa[k:]**2) if k<len(S) else 0
            mpo[i]=U[:,:k].reshape(cl,d,d,k)
            mpo[i+1]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,d,cr)
    return mpo,te

def mpo_expect_product(mpo, states):
    L=np.ones((1,),dtype=complex)
    for i in range(len(mpo)):
        s=states[i]
        L=np.einsum('a,ab->b',L,mpo[i][:,s,s,:])
    return L[0]

def mpo_chi_profile(mpo):
    return [mpo[i].shape[3] for i in range(len(mpo)-1)]

# === Main ===
np.random.seed(42)
n=10; dt=0.05; NS=40; chi=64; kick=n//2

print("="*70)
print(f"  B7c: Heisenberg vs Schrodinger — 1D chain, n={n}, d=2")
print(f"  dt={dt}, {NS} steps, chi_max={chi}, kick site={kick}")
print("="*70);sys.stdout.flush()

# 1st order Trotter (simpler, sufficient for comparison)
Ub_half=expm(-1j*h2*dt*0.5)
Ub_full=expm(-1j*h2*dt)

# Initial: all up except kick (down)
states=[0]*n; states[kick]=1

# --- Exact ---
print("\n--- EXACT ---");sys.stdout.flush()
t0=time.time()
H_full=np.zeros((2**n,2**n),dtype=complex)
for i in range(n-1):
    for O1,O2 in [(Sp,2*Sm),(Sm,2*Sp),(Zd,Zd)]:
        op=np.eye(1,dtype=complex)
        for j in range(n):
            if j==i: op=np.kron(op,O1)
            elif j==i+1: op=np.kron(op,O2)
            else: op=np.kron(op,Id2)
        H_full+=op
Sz_full=np.eye(1,dtype=complex)
for i in range(n):
    if i==kick: Sz_full=np.kron(Sz_full,Sz)
    else: Sz_full=np.kron(Sz_full,Id2)
U_dt=expm(-1j*H_full*dt)
psi0=np.zeros(2**n,dtype=complex)
idx0=0
for i in range(n): idx0=idx0*2+states[i]
psi0[idx0]=1
psi=psi0.copy()
res_e=[]
sample=[0,1,2]+list(range(4,NS+1,4))
for step in range(NS+1):
    if step in sample:
        sz_ex=(psi.conj()@Sz_full@psi).real
        res_e.append({'step':step,'t':step*dt,'sz':sz_ex})
    if step<NS: psi=U_dt@psi
dt_e=time.time()-t0
print(f"  Done in {dt_e:.1f}s");sys.stdout.flush()

# --- Schrodinger ---
print("\n--- SCHRODINGER ---");sys.stdout.flush()
mps=[]
for i in range(n):
    t=np.zeros((1,d,1),dtype=complex);t[0,states[i],0]=1;mps.append(t)
mps=norm_mps(mps)

res_s=[]
t0=time.time()
for step in range(NS+1):
    if step in sample:
        mps=norm_mps(mps)
        sz=mps_expect(mps,kick,Sz).real
        cp=chi_profile(mps);cm=max(cp)
        res_s.append({'step':step,'t':step*dt,'sz':sz,'cm':cm,'cp':cp[:]})
    if step<NS:
        mps,te=mps_step(mps,Ub_full,chi)
dt_s=time.time()-t0
print(f"  Done in {dt_s:.1f}s");sys.stdout.flush()

# --- Heisenberg ---
print("\n--- HEISENBERG ---");sys.stdout.flush()
mpo=[]
for i in range(n):
    if i==kick: W=Sz.reshape(1,d,d,1)
    else: W=np.eye(d,dtype=complex).reshape(1,d,d,1)
    mpo.append(W.copy())

res_h=[]
t0=time.time()
for step in range(NS+1):
    if step in sample:
        sz=mpo_expect_product(mpo,states).real
        cp=mpo_chi_profile(mpo);cm=max(cp)
        res_h.append({'step':step,'t':step*dt,'sz':sz,'cm':cm,'cp':cp[:]})
    if step<NS:
        mpo,te=mpo_step(mpo,Ub_full,chi)
dt_h=time.time()-t0
print(f"  Done in {dt_h:.1f}s");sys.stdout.flush()

# === Compare ===
print("\n"+"="*70)
print("VERGELIJKING: Operator vs State entanglement")
print("="*70)
print(f"\n{'t':>6} {'Sz_exact':>10} {'Sz_Schr':>10} {'Sz_Heis':>10} {'err_S':>10} {'err_H':>10} {'chi_S':>7} {'chi_H':>7}")
print("-"*70)
for rs,rh,re in zip(res_s,res_h,res_e):
    es=abs(rs['sz']-re['sz']); eh=abs(rh['sz']-re['sz'])
    print(f"{rs['t']:6.2f} {re['sz']:10.6f} {rs['sz']:10.6f} {rh['sz']:10.6f} {es:10.2e} {eh:10.2e} {rs['cm']:7d} {rh['cm']:7d}")

print(f"\nTijd: Exact={dt_e:.1f}s, Schrodinger={dt_s:.1f}s, Heisenberg={dt_h:.1f}s")
print(f"\nChi-profiel bij t={res_s[-1]['t']:.2f}:")
print(f"  Schrodinger: {res_s[-1]['cp']}")
print(f"  Heisenberg:  {res_h[-1]['cp']}")
cs=sum(res_s[-1]['cp']);ch=sum(res_h[-1]['cp'])
print(f"  Totaal chi: S={cs}, H={ch}")
if ch>0 and cs>0:
    print(f"  Ratio H/S = {ch/cs:.3f}")
    if ch<cs: print(f"  ** HEISENBERG WINT: {cs/ch:.1f}x minder operator-entanglement **")
    elif ch>cs: print(f"  ** SCHRODINGER WINT: {ch/cs:.1f}x minder state-entanglement **")
    else: print(f"  ** GELIJK **")

# Chi evolution
print(f"\nChi-max evolutie:")
print(f"{'t':>6} {'chi_S':>7} {'chi_H':>7} {'ratio':>7}")
print("-"*30)
for rs,rh in zip(res_s,res_h):
    r=rh['cm']/rs['cm'] if rs['cm']>0 else 0
    print(f"{rs['t']:6.2f} {rs['cm']:7d} {rh['cm']:7d} {r:7.2f}")
