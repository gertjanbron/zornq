"""B11: Hiërarchische Krachten — Lz=6, d=16, geoptimaliseerd"""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import time, sys

d = 16
I2=np.eye(2,dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)

def kron4(A,B,C,D): return np.kron(np.kron(A,B),np.kron(C,D))

def build_H_intra(J1d, J2d):
    H=np.zeros((d,d),dtype=complex)
    H+=J1d*(kron4(X,X,I2,I2)+kron4(Y,Y,I2,I2)+kron4(Z,Z,I2,I2))
    H+=J1d*(kron4(I2,I2,X,X)+kron4(I2,I2,Y,Y)+kron4(I2,I2,Z,Z))
    H+=J2d*(kron4(X,I2,X,I2)+kron4(Y,I2,Y,I2)+kron4(Z,I2,Z,I2))
    H+=J2d*(kron4(I2,X,I2,X)+kron4(I2,Y,I2,Y)+kron4(I2,Z,I2,Z))
    return H

def build_H_inter(J3d):
    H=np.zeros((d*d,d*d),dtype=complex)
    for q in range(4):
        for P in [X,Y,Z]:
            oL=[I2]*4; oL[q]=P; PL=kron4(*oL)
            oR=[I2]*4; oR[q]=P; PR=kron4(*oR)
            H+=J3d*np.kron(PL,PR)
    return H

# Observable: Sz per laag
Sz=np.zeros((d,d),dtype=complex)
for q in range(4):
    o=[I2]*4; o[q]=Z; Sz+=kron4(*o)/4

def mps_exp(mps, site, O):
    L=np.ones((1,1),dtype=complex)
    for i in range(len(mps)):
        A=mps[i]; Ac=A.conj()
        if i==site:
            OA=np.einsum('st,btd->bsd',O,A)
            L=np.einsum('ab,asc,bsd->cd',L,Ac,OA)
        else:
            L=np.einsum('ab,asc,bsd->cd',L,Ac,A)
    return L[0,0]

def run(name, J1d, J2d, J3d, Lz=6, dt=0.05, steps=10, chi=16):
    t_start = time.time()
    
    # Build gates (precompute)
    U_intra = expm(-1j*build_H_intra(J1d,J2d)*dt/2)
    U_inter_mat = expm(-1j*build_H_inter(J3d)*dt).reshape(d*d, d*d)
    
    mid=Lz//2
    mps=[]
    for i in range(Lz):
        A=np.zeros((1,d,1),dtype=complex)
        A[0, 8 if i==mid else 0, 0]=1.0  # |1000> of |0000>
        mps.append(A)
    
    snapshots = []
    
    for step in range(steps+1):
        prof=[mps_exp(mps,i,Sz).real for i in range(Lz)]
        snapshots.append((step*dt, prof))
        
        if step==steps: break
        
        # Trotter step
        for i in range(Lz):
            mps[i]=np.einsum('ij,ajb->aib',U_intra,mps[i])
        for parity in [0,1]:
            for i in range(parity,Lz-1,2):
                A=mps[i]; B=mps[i+1]
                cl,_,cr1=A.shape; _,_,cr2=B.shape
                Th=np.einsum('aib,bjc->aijc',A,B).reshape(cl,d*d,cr2)
                Th=np.einsum('ij,ajb->aib',U_inter_mat,Th)
                mat=Th.reshape(cl*d, d*cr2)
                U_s,S,V=svd(mat,full_matrices=False)
                k=min(len(S),chi)
                mps[i]=U_s[:,:k].reshape(cl,d,k)
                mps[i+1]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,cr2)
        for i in range(Lz):
            mps[i]=np.einsum('ij,ajb->aib',U_intra,mps[i])
    
    dt_total = time.time()-t_start
    
    # Analyse
    final=snapshots[-1][1]
    sz_mid=abs(final[mid])
    sz_total=sum(abs(x) for x in final)
    conf=sz_mid/sz_total if sz_total>1e-10 else 0
    
    weights=np.array([abs(x-np.mean(final)) for x in final])
    pos=np.arange(Lz).astype(float)
    if weights.sum()>1e-10:
        mu=np.average(pos,weights=weights)
        width=np.sqrt(np.average((pos-mu)**2,weights=weights))
    else: width=0
    
    mc = max(A.shape[2] for A in mps[:-1])
    
    print(f"  {name:<25s} conf={conf:.1%} width={width:.3f} chi={mc} ({dt_total:.1f}s)")
    print(f"    t=0.00: {[f'{x:.3f}' for x in snapshots[0][1]]}")
    for s_t, s_p in snapshots[1::2]:
        print(f"    t={s_t:.2f}: {[f'{x:.3f}' for x in s_p]}")
    print(f"    t={snapshots[-1][0]:.2f}: {[f'{x:.3f}' for x in final]}")
    
    return conf, width, snapshots

print("="*65)
print("  B11: Hiërarchische Krachten — Dimensionale Dynamica")
print("  2×2×6 (24q), excitatie op middelste laag, Heisenberg model")
print("  1D=horizontaal(sterk), 2D=verticaal(zwak), 3D=inter-laag(EM)")
print("="*65)
sys.stdout.flush()

configs = [
    ("UNIFORM (1:1:1)",       1.0,  1.0,  1.0),
    ("HIËRARCH (10:1:0.1)",  10.0,  1.0,  0.1),
    ("EXTREEM (100:1:0.01)", 100.0, 1.0,  0.01),
    ("OMGEKEERD (0.1:1:10)",  0.1,  1.0, 10.0),
    ("PURE 1D (10:0:0)",      10.0, 0.0,  0.0),
    ("PURE 3D (0:0:10)",       0.0, 0.0, 10.0),
]

results = []
for name, j1, j2, j3 in configs:
    c, w, snaps = run(name, j1, j2, j3, Lz=6, dt=0.05, steps=10, chi=16)
    results.append((name, c, w))
    sys.stdout.flush()

print(f"\n{'='*65}")
print(f"  SAMENVATTING")
print(f"{'='*65}")
print(f"  {'Model':<25s} | {'Confinement':>11s} | {'Breedte':>7s} | Gedrag")
print(f"  {'-'*25}-+-{'-'*11}-+-{'-'*7}-+-{'-'*25}")
for name, c, w in results:
    if c > 0.6: gedrag = "CONFINED"
    elif c > 0.3: gedrag = "GEDEELTELIJK verspreid"
    else: gedrag = "VRIJ verspreid"
    print(f"  {name:<25s} | {c:>10.1%} | {w:>7.3f} | {gedrag}")

print(f"\n  Interpretatie krachten-analogie:")
print(f"  - HIËRARCHISCH: excitatie bleef meer opgesloten dan UNIFORM?")
c_uni = results[0][1]; c_hier = results[1][1]; c_ext = results[2][1]
if c_hier > c_uni * 1.2:
    print(f"    JA: confinement {c_hier:.1%} vs {c_uni:.1%} (+{(c_hier/c_uni-1)*100:.0f}%)")
    print(f"    De sterke 1D-koppeling houdt excitatie lokaal vast.")
elif c_hier > c_uni * 0.95:
    print(f"    MARGINAAL: verschil te klein ({c_hier:.1%} vs {c_uni:.1%})")
else:
    print(f"    NEE: hiërarchie geeft MINDER confinement ({c_hier:.1%} vs {c_uni:.1%})")

if c_ext > c_hier * 1.1:
    print(f"  - EXTREEM versterkt confinement verder: {c_ext:.1%}")
else:
    print(f"  - EXTREEM geeft geen extra confinement: {c_ext:.1%}")
