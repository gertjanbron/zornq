"""B7 verification: compare TEBD on 2x2x2 with exact time evolution."""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import time, sys

Sp = np.array([[0,1],[0,0]], dtype=complex)
Sm = np.array([[0,0],[1,0]], dtype=complex)
Zd = np.array([[1,0],[0,-1]], dtype=complex)
Id2 = np.eye(2, dtype=complex)

def embed_1q(op, q, nq):
    ops=[Id2]*nq; ops[q]=op; r=ops[0]
    for o in ops[1:]: r=np.kron(r,o)
    return r
def embed_2q(o1,q1,o2,q2,nq):
    ops=[Id2]*nq; ops[q1]=o1; ops[q2]=o2; r=ops[0]
    for o in ops[1:]: r=np.kron(r,o)
    return r

def h_intra_layer(Lx,Ly,periodic=True):
    nq=Lx*Ly; d=2**nq; h=np.zeros((d,d),dtype=complex)
    idx=lambda x,y: y*Lx+x
    for y in range(Ly):
        for x in range(Lx):
            for dx,dy in [(1,0),(0,1)]:
                x2,y2=x+dx,y+dy
                if dx==1:
                    if x2>=Lx:
                        if periodic and Lx>2: x2=0
                        else: continue
                if dy==1:
                    if y2>=Ly:
                        if periodic and Ly>2: y2=0
                        else: continue
                q1,q2=idx(x,y),idx(x2,y2)
                h+=2*embed_2q(Sp,q1,Sm,q2,nq)+2*embed_2q(Sm,q1,Sp,q2,nq)+embed_2q(Zd,q1,Zd,q2,nq)
    return h

def h_inter_layer(Lx,Ly):
    nq=Lx*Ly; d=2**nq; h=np.zeros((d*d,d*d),dtype=complex)
    for q in range(nq):
        h+=2*np.kron(embed_1q(Sp,q,nq),embed_1q(Sm,q,nq))
        h+=2*np.kron(embed_1q(Sm,q,nq),embed_1q(Sp,q,nq))
        h+=np.kron(embed_1q(Zd,q,nq),embed_1q(Zd,q,nq))
    return h

def apply_1site(mps, U):
    for i in range(len(mps)): mps[i]=np.einsum('ij,ajb->aib',U,mps[i])
    return mps

def apply_2site(mps,i,U,chi,d):
    cl,cr=mps[i].shape[0],mps[i+1].shape[2]
    theta=np.einsum('asc,crd->asrd',mps[i],mps[i+1]).reshape(cl,d*d,cr)
    theta=np.einsum('ij,ajb->aib',U,theta).reshape(cl*d,d*cr)
    Uv,S,Vh=svd(theta,full_matrices=False); k=min(chi,len(S))
    te=np.sum(S[k:]**2) if k<len(S) else 0.0
    mps[i]=Uv[:,:k].reshape(cl,d,k)
    mps[i+1]=(np.diag(S[:k])@Vh[:k,:]).reshape(k,d,cr)
    return mps,te

def tebd_step(mps,Uh,Ub,chi,d):
    n=len(mps); te=0
    mps=apply_1site(mps,Uh)
    for i in range(0,n-1,2): mps,e=apply_2site(mps,i,Ub,chi,d); te+=e
    for i in range(1,n-1,2): mps,e=apply_2site(mps,i,Ub,chi,d); te+=e
    mps=apply_1site(mps,Uh)
    return mps,te

def mps_to_vec(mps):
    """Convert MPS to full state vector."""
    v = mps[0][:,:]  # shape (1, d, chi) -> squeeze first dim
    v = v.reshape(v.shape[1], v.shape[2])  # (d, chi)
    for i in range(1, len(mps)):
        # v(d_tot, chi_l) x M(chi_l, d, chi_r) -> v(d_tot*d, chi_r)
        v = np.einsum('ia,ajb->ijb', v, mps[i]).reshape(-1, mps[i].shape[2])
    return v.ravel()

np.random.seed(42)
Lx,Ly=2,2; nq=4; d=16
Lz=2  # Just 2 layers = 8 qubits, fully exact

print("="*60)
print(f"VERIFICATIE: 2x2x2 (8q), TEBD vs exact time evolution")
print("="*60)

# Build full 8q Hamiltonian for exact reference
h_local = h_intra_layer(Lx,Ly,True)
h_inter = h_inter_layer(Lx,Ly)
Id_d = np.eye(d,dtype=complex)
H_full = np.kron(h_local, Id_d) + np.kron(Id_d, h_local) + h_inter
print(f"H_full: {H_full.shape}")

# Initial state: all-up + flip layer 1
psi0 = np.zeros(d*d, dtype=complex)
# |0000> x |1111> = index 0*16 + 15 = 15
psi0[0*d + (d-1)] = 1.0
E0_exact = (psi0.conj() @ H_full @ psi0).real
print(f"E0 = {E0_exact:.6f}")

# Exact evolution
dt = 0.02
n_steps = 40
U_exact = expm(-1j * H_full * dt)

# TEBD gates
Uh = expm(-1j*h_local*dt/2)
Ub = expm(-1j*h_inter*dt)

# MPS initial state (chi=16 = d -> exact)
mps = []
t0 = np.zeros((1,d,1),dtype=complex); t0[0,0,0]=1; mps.append(t0)
t1 = np.zeros((1,d,1),dtype=complex); t1[0,d-1,0]=1; mps.append(t1)

Sz_layer = np.zeros((d,d),dtype=complex)
for q in range(nq): Sz_layer += embed_1q(0.5*Zd,q,nq)
Sz_full_0 = np.kron(Sz_layer, Id_d)
Sz_full_1 = np.kron(Id_d, Sz_layer)

psi_exact = psi0.copy()

print(f"\n{'stap':>5} {'t':>6} {'Sz0_ex':>8} {'Sz0_tb':>8} {'Sz1_ex':>8} {'Sz1_tb':>8} {'|dPsi|':>10} {'E_exact':>10} {'E_tebd':>10}")
print("-"*85)

for step in range(n_steps+1):
    t = step*dt
    if step % 5 == 0:
        # Exact observables
        Sz0_ex = (psi_exact.conj() @ Sz_full_0 @ psi_exact).real
        Sz1_ex = (psi_exact.conj() @ Sz_full_1 @ psi_exact).real
        E_ex = (psi_exact.conj() @ H_full @ psi_exact).real
        
        # TEBD observables
        psi_tebd = mps_to_vec(mps)
        # Align global phase
        overlap = np.vdot(psi_exact, psi_tebd)
        phase = overlap / abs(overlap) if abs(overlap) > 1e-10 else 1.0
        psi_tebd_aligned = psi_tebd / phase
        diff = np.linalg.norm(psi_exact - psi_tebd_aligned)
        
        Sz0_tb = (psi_tebd.conj() @ Sz_full_0 @ psi_tebd).real
        Sz1_tb = (psi_tebd.conj() @ Sz_full_1 @ psi_tebd).real
        E_tb = (psi_tebd.conj() @ H_full @ psi_tebd).real
        
        print(f"{step:5d} {t:6.2f} {Sz0_ex:8.4f} {Sz0_tb:8.4f} {Sz1_ex:8.4f} {Sz1_tb:8.4f} {diff:10.2e} {E_ex:10.4f} {E_tb:10.4f}")
        sys.stdout.flush()
    
    if step < n_steps:
        # Exact evolution
        psi_exact = U_exact @ psi_exact
        # TEBD step (chi=d=16 -> no truncation -> exact)
        mps, te = tebd_step(mps, Uh, Ub, d, d)  # chi=d -> no truncation
        nrm = np.sqrt(abs(sum(abs(mps_to_vec(mps))**2)))
        mps[0] /= nrm

print(f"\nChi=d={d}: TEBD is exact (geen truncatie).")
print(f"Verschil komt puur van Trotter-fout (dt={dt}).")

# Now test with chi < d to see truncation effect
print(f"\n--- Zelfde evolutie, nu met chi=4 (truncatie) ---")
mps4 = []
t0 = np.zeros((1,d,1),dtype=complex); t0[0,0,0]=1; mps4.append(t0)
t1 = np.zeros((1,d,1),dtype=complex); t1[0,d-1,0]=1; mps4.append(t1)

psi_exact = psi0.copy()
for step in range(n_steps+1):
    t = step*dt
    if step % 5 == 0:
        Sz0_ex = (psi_exact.conj()@Sz_full_0@psi_exact).real
        Sz1_ex = (psi_exact.conj()@Sz_full_1@psi_exact).real
        psi_tb = mps_to_vec(mps4)
        Sz0_tb = (psi_tb.conj()@Sz_full_0@psi_tb).real
        Sz1_tb = (psi_tb.conj()@Sz_full_1@psi_tb).real
        E_tb = (psi_tb.conj()@H_full@psi_tb).real
        E_ex = (psi_exact.conj()@H_full@psi_exact).real
        print(f"  t={t:.2f}: Sz0 exact={Sz0_ex:+.4f} tebd={Sz0_tb:+.4f} | E exact={E_ex:.4f} tebd={E_tb:.4f}")
        sys.stdout.flush()
    if step < n_steps:
        psi_exact = U_exact @ psi_exact
        mps4, te = tebd_step(mps4, Uh, Ub, 4, d)  # chi=4
        nrm = np.sqrt(abs(sum(abs(mps_to_vec(mps4))**2)))
        mps4[0] /= nrm

print("\nDone.")
