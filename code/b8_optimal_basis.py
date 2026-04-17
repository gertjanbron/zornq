"""
B8: Zoek de optimale 8-dimensionale subruimte voor de operator.
Vraag: kan de operator in een 8-dim deelruimte van 64-dim Pauli-ruimte leven?
(Niet per se Zorn — maar welke 8-dim ruimte dan ook.)
"""
import numpy as np
from numpy.linalg import svd, norm
import sys

I2=np.eye(2,dtype=complex)
Z_op=np.array([[1,0],[0,-1]],dtype=complex)
X_op=np.array([[0,1],[1,0]],dtype=complex)
Y_op=np.array([[0,-1j],[1j,0]],dtype=complex)

def Rx(t):
    c=np.cos(t/2);s=np.sin(t/2)
    return np.array([[c,-1j*s],[-1j*s,c]],dtype=complex)

def ZZ_gate(g):
    return np.diag([np.exp(-1j*g),np.exp(1j*g),np.exp(1j*g),np.exp(-1j*g)])

def embed_1q(U,pos):
    p=[I2,I2,I2];p[pos]=U
    return np.kron(np.kron(p[0],p[1]),p[2])

def embed_2q(U,p1,p2):
    if p1==0 and p2==1: return np.kron(U,I2)
    elif p1==1 and p2==2: return np.kron(I2,U)

def build_inter(U2q):
    d=8;M=np.zeros((d*d,d*d),dtype=complex)
    for l0 in range(2):
     for l1 in range(2):
      for l2 in range(2):
       for r0 in range(2):
        for r1 in range(2):
         for r2 in range(2):
          iL=l0*4+l1*2+l2;iR=r0*4+r1*2+r2;idx_in=iL*d+iR
          for l2p in range(2):
           for r0p in range(2):
            c=U2q[l2p*2+r0p,l2*2+r0]
            oL=l0*4+l1*2+l2p;oR=r0p*4+r1*2+r2
            M[oL*d+oR,idx_in]+=c
    return M

paulis=[I2,X_op,Y_op,Z_op]
pnames=['I','X','Y','Z']
pauli_3q=[]
pauli_3q_names=[]
for i,p1 in enumerate(paulis):
    for j,p2 in enumerate(paulis):
        for k,p3 in enumerate(paulis):
            pauli_3q.append(np.kron(np.kron(p1,p2),p3))
            pauli_3q_names.append(f"{pnames[i]}{pnames[j]}{pnames[k]}")

gamma=0.3;beta=0.7;d=8;n_g=3

print("="*70)
print("  B8: Operator-dimensionaliteit in Pauli-ruimte")
print("="*70)

# Collect ALL operator matrices across all p-values
for p in [1,2,3,5]:
    ZZg=ZZ_gate(gamma);Rxg=Rx(2*beta)
    gates=[]
    for layer in range(p):
        for q in range(8):
            g1=q//3;p1=q%3;g2=(q+1)//3;p2=(q+1)%3
            if g1==g2: gates.append(('1g',g1,embed_2q(ZZg,p1,p2)))
            else: gates.append(('2g',g1,build_inter(ZZg)))
        for q in range(9):
            g=q//3;pos=q%3
            gates.append(('1g',g,embed_1q(Rxg,pos)))

    I8=np.eye(d,dtype=complex)
    mpo=[]
    for g in range(n_g):
        if g==0: mpo.append(embed_1q(Z_op,0).reshape(1,d,d,1))
        else: mpo.append(I8.reshape(1,d,d,1))

    for entry in reversed(gates):
        gt=entry[0]
        if gt=='1g':
            _,g,U8=entry;Ud=U8.conj().T
            W=mpo[g]
            W=np.einsum('ij,ajkb->aikb',Ud,W)
            W=np.einsum('ajkb,kl->ajlb',W,U8)
            mpo[g]=W
        elif gt=='2g':
            _,g1,U64=entry;g2=g1+1
            Ud=U64.conj().T
            Ud_r=Ud.reshape(d,d,d,d);Uf_r=U64.reshape(d,d,d,d)
            cl=mpo[g1].shape[0];cr=mpo[g2].shape[3]
            Th=np.einsum('abce,edfg->abcdfg',mpo[g1],mpo[g2])
            Th=np.einsum('ijkl,akclef->aicjef',Ud_r,Th)
            Th=np.einsum('ijkl,abkdlf->aibjdf',Uf_r,Th)
            Th=Th.transpose(0,2,1,4,3,5)
            mat=Th.reshape(cl*d*d,d*d*cr)
            U_s,S,V=svd(mat,full_matrices=False)
            Sa=np.abs(S)
            k=max(1,int(np.sum(Sa>1e-14*Sa[0]))) if Sa[0]>1e-15 else 1
            mpo[g1]=U_s[:,:k].reshape(cl,d,d,k)
            mpo[g2]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,d,cr)

    chi=[mpo[g].shape[3] for g in range(n_g-1)]

    # Collect all 8×8 matrix slices from all groups
    all_matrices = []
    for g in range(n_g):
        cL=mpo[g].shape[0]; cR=mpo[g].shape[3]
        for a in range(cL):
            for b in range(cR):
                M = mpo[g][a,:,:,b]
                if norm(M) > 1e-14:
                    all_matrices.append(M)

    # Decompose each in Pauli basis and stack
    pauli_vectors = []
    for M in all_matrices:
        coeffs = np.array([np.trace(P.conj().T @ M)/8 for P in pauli_3q])
        pauli_vectors.append(coeffs)
    pauli_matrix = np.array(pauli_vectors)  # (n_matrices, 64)

    # SVD of pauli_matrix to find the effective dimension
    U_p, S_p, V_p = svd(pauli_matrix, full_matrices=False)
    S_p_rel = S_p / S_p[0] if S_p[0] > 0 else S_p

    # How many significant dimensions?
    n_sig = np.sum(S_p_rel > 1e-12)

    print(f"\np={p}, chi={chi}, {len(all_matrices)} nonzero matrix slices")
    print(f"  Pauli-space singular values: {S_p_rel[:min(10,len(S_p_rel))]}")
    print(f"  Effective dimension: {n_sig}/64")
    print(f"  (Zorn dimension = 8)")

    # Which Paulis span this subspace?
    # Take top n_sig right singular vectors
    active_paulis = set()
    for sv_idx in range(n_sig):
        v = V_p[sv_idx]  # 64-dim
        for i in range(64):
            if abs(v[i]) > 1e-10:
                active_paulis.add(pauli_3q_names[i])

    print(f"  Active Paulis: {sorted(active_paulis)}")

    # Zorn coverage
    zorn_paulis = {'III','XIX','XIY','XXI','XXZ','XYI','XYX','XYY','XYZ',
                   'XZI','XZZ','YII','YIX','YIY','YIZ','YYX','YYY','ZII'}
    in_z = active_paulis & zorn_paulis
    out_z = active_paulis - zorn_paulis
    print(f"  In Zorn space: {len(in_z)}/{len(active_paulis)}: {sorted(in_z)}")
    if out_z:
        print(f"  OUTSIDE Zorn: {sorted(out_z)}")
    else:
        print(f"  *** VOLLEDIG IN ZORN SPACE ***")

# ======================
# Vergelijk: totaal parameters per representatie
# ======================
print("\n" + "="*70)
print("  VERGELIJKING: parameters per representatie")
print("="*70)
print(f"\nVoor n=498q (166 groepen), p=5, Z_0 observable:")
print(f"  Standaard d=2 MPO:   chi=2,  params = 166 × 2² × 2² = {166*16}")
print(f"  Grouped d=8 MPO:     chi=2,  params = 166 × 2² × 8² = {166*4*64}")
print(f"  Pauli-sparse repr:   ~7 coeff per groep = {166*7}")
print(f"  Pure Zorn (als 't past): 8 coeff per groep = {166*8}")
print(f"  d=2 MPO (feitelijk): chi_max=2, ~2056 params (gemeten)")
print(f"\n  Standaard state MPS: chi=16, params ~ {166*16*16*4}")

# Check: at which gamma/beta does the out-of-Zorn effect appear?
print("\n" + "="*70)
print("  Gamma/beta scan: wanneer breekt Zorn-structuur?")
print("="*70)

for gamma_test in [0.1, 0.3, 0.5, 0.8, 1.0]:
    for beta_test in [0.3, 0.7, 1.2]:
        ZZg=ZZ_gate(gamma_test);Rxg=Rx(2*beta_test)
        gates=[]
        for layer in range(3):  # p=3: where leakage starts
            for q in range(8):
                g1=q//3;p1=q%3;g2=(q+1)//3;p2=(q+1)%3
                if g1==g2: gates.append(('1g',g1,embed_2q(ZZg,p1,p2)))
                else: gates.append(('2g',g1,build_inter(ZZg)))
            for q in range(9):
                g=q//3;pos=q%3
                gates.append(('1g',g,embed_1q(Rxg,pos)))

        I8=np.eye(d,dtype=complex)
        mpo=[]
        for g in range(n_g):
            if g==0: mpo.append(embed_1q(Z_op,0).reshape(1,d,d,1))
            else: mpo.append(I8.reshape(1,d,d,1))
        for entry in reversed(gates):
            gt=entry[0]
            if gt=='1g':
                _,g,U8=entry;Ud=U8.conj().T;W=mpo[g]
                W=np.einsum('ij,ajkb->aikb',Ud,W);W=np.einsum('ajkb,kl->ajlb',W,U8);mpo[g]=W
            elif gt=='2g':
                _,g1,U64=entry;g2=g1+1;Ud=U64.conj().T
                Ud_r=Ud.reshape(d,d,d,d);Uf_r=U64.reshape(d,d,d,d)
                cl=mpo[g1].shape[0];cr=mpo[g2].shape[3]
                Th=np.einsum('abce,edfg->abcdfg',mpo[g1],mpo[g2])
                Th=np.einsum('ijkl,akclef->aicjef',Ud_r,Th)
                Th=np.einsum('ijkl,abkdlf->aibjdf',Uf_r,Th)
                Th=Th.transpose(0,2,1,4,3,5)
                mat=Th.reshape(cl*d*d,d*d*cr);U_s,S,V=svd(mat,full_matrices=False)
                Sa=np.abs(S);k=max(1,int(np.sum(Sa>1e-14*Sa[0]))) if Sa[0]>1e-15 else 1
                mpo[g1]=U_s[:,:k].reshape(cl,d,d,k)
                mpo[g2]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,d,cr)

        # Collect Pauli vectors
        all_m=[]
        for g in range(n_g):
            for a in range(mpo[g].shape[0]):
                for b in range(mpo[g].shape[3]):
                    M=mpo[g][a,:,:,b]
                    if norm(M)>1e-14: all_m.append(M)
        pv=np.array([np.array([np.trace(P.conj().T@M)/8 for P in pauli_3q]) for M in all_m])
        _,Sv,Vv=svd(pv,full_matrices=False)
        n_dim=np.sum(Sv/Sv[0]>1e-12)

        # Active paulis
        act=set()
        for si in range(n_dim):
            for i in range(64):
                if abs(Vv[si,i])>1e-10: act.add(pauli_3q_names[i])
        out=act-zorn_paulis
        print(f"  gamma={gamma_test:.1f} beta={beta_test:.1f}: dim={n_dim}, outside_zorn={len(out)}")
