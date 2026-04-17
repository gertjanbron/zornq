"""
B8: Welke Pauli-operatoren vormen de Zorn L-basis?
En: analyse bij p=5 (chi=2).
"""
import numpy as np
from numpy.linalg import svd, norm
import sys

# Zorn
class Zorn:
    def __init__(self, alpha=0, a=None, b=None, beta=0):
        self.alpha = complex(alpha)
        self.beta = complex(beta)
        self.a = np.zeros(3,dtype=complex) if a is None else np.array(a,dtype=complex)
        self.b = np.zeros(3,dtype=complex) if b is None else np.array(b,dtype=complex)
    def to_array(self):
        return np.array([self.alpha,*self.a,*self.b,self.beta],dtype=complex)
    @staticmethod
    def from_array(arr):
        return Zorn(arr[0],arr[1:4],arr[4:7],arr[7])
    def __mul__(self,o):
        aa=self.alpha*o.alpha+np.dot(self.a,o.b)
        ab=self.alpha*o.a+o.beta*self.a+np.cross(self.b,o.b)
        ba=o.alpha*self.b+self.beta*o.b-np.cross(self.a,o.a)
        bb=self.beta*o.beta+np.dot(o.a,self.b)
        return Zorn(aa,ab,ba,bb)
    def to_Lmatrix(self):
        M=np.zeros((8,8),dtype=complex)
        for i in range(8):
            e=np.zeros(8,dtype=complex);e[i]=1
            r=self*Zorn.from_array(e)
            M[:,i]=r.to_array()
        return M

I2=np.eye(2,dtype=complex)
Z_op=np.array([[1,0],[0,-1]],dtype=complex)
X_op=np.array([[0,1],[1,0]],dtype=complex)
Y_op=np.array([[0,-1j],[1j,0]],dtype=complex)

paulis=[I2,X_op,Y_op,Z_op]
pnames=['I','X','Y','Z']
pauli_3q=[]
pauli_3q_names=[]
for i,p1 in enumerate(paulis):
    for j,p2 in enumerate(paulis):
        for k,p3 in enumerate(paulis):
            pauli_3q.append(np.kron(np.kron(p1,p2),p3))
            pauli_3q_names.append(f"{pnames[i]}{pnames[j]}{pnames[k]}")

# ======================
# Part 1: Zorn L-basis in Pauli language
# ======================
print("="*70)
print("  Zorn L-basis elementen in Pauli-decompositie")
print("="*70)

zorn_names = ['1 (unit)', 'e1 (a[0])', 'e2 (a[1])', 'e3 (a[2])',
              'e4 (b[0])', 'e5 (b[1])', 'e6 (b[2])', 'e7 (beta)']

zorn_pauli_coverage = set()

for idx in range(8):
    e = np.zeros(8,dtype=complex); e[idx]=1
    z = Zorn.from_array(e)
    L = z.to_Lmatrix()

    # Pauli decomposition
    coeffs=[]
    for P in pauli_3q:
        c=np.trace(P.conj().T@L)/8
        coeffs.append(c)
    coeffs=np.array(coeffs)
    mag=np.abs(coeffs)

    print(f"\nZorn basis {zorn_names[idx]}:")
    top=np.argsort(mag)[::-1]
    for t in top:
        if mag[t]>1e-10:
            zorn_pauli_coverage.add(pauli_3q_names[t])
            print(f"  {pauli_3q_names[t]}: {coeffs[t]:.6f}")

print(f"\nTotaal Pauli-termen bedekt door Zorn L-basis: {len(zorn_pauli_coverage)}/64")
print(f"Pauli's in Zorn-ruimte: {sorted(zorn_pauli_coverage)}")

# Which Paulis are NOT in Zorn space?
all_paulis = set(pauli_3q_names)
outside = all_paulis - zorn_pauli_coverage
print(f"Pauli's BUITEN Zorn-ruimte: {len(outside)}/64")

# ======================
# Part 2: Rebuild operator at p=5 and analyze Pauli structure
# ======================
print("\n" + "="*70)
print("  Operator-analyse bij p=5 (chi=2)")
print("="*70)

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
    d=8; M=np.zeros((d*d,d*d),dtype=complex)
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

gamma=0.3;beta=0.7;d=8;n_g=3

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
            _,g,U8=entry; Ud=U8.conj().T
            W=mpo[g]
            W=np.einsum('ij,ajkb->aikb',Ud,W)
            W=np.einsum('ajkb,kl->ajlb',W,U8)
            mpo[g]=W
        elif gt=='2g':
            _,g1,U64=entry; g2=g1+1
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
    print(f"\np={p}, chi={chi}")

    # For chi=1 groups, do Pauli analysis
    for g in range(n_g):
        if mpo[g].shape[0]==1 and mpo[g].shape[3]==1:
            M=mpo[g][0,:,:,0]
            coeffs=[]
            for P in pauli_3q:
                c=np.trace(P.conj().T@M)/8
                coeffs.append(c)
            coeffs=np.array(coeffs)
            mag=np.abs(coeffs)
            nz=np.sum(mag>1e-10)

            # Zorn fit
            L_basis=[]
            for i in range(8):
                e=np.zeros(8,dtype=complex);e[i]=1
                L_basis.append(Zorn.from_array(e).to_Lmatrix())
            A=np.zeros((64,8),dtype=complex)
            for i in range(8): A[:,i]=L_basis[i].ravel()
            c_fit,_,_,_=np.linalg.lstsq(A,M.ravel(),rcond=None)
            recon=sum(c_fit[i]*L_basis[i] for i in range(8))
            res=norm(M-recon)/norm(M)*100

            # Which Paulis are nonzero?
            nz_paulis=[pauli_3q_names[i] for i in range(64) if mag[i]>1e-10]

            # Which are in Zorn space?
            in_zorn=[p for p in nz_paulis if p in zorn_pauli_coverage]
            out_zorn=[p for p in nz_paulis if p not in zorn_pauli_coverage]

            print(f"  Group {g}: {nz} Paulis, Zorn-miss={100-res:.0f}%→{res:.1f}% buiten")
            print(f"    In Zorn: {in_zorn}")
            print(f"    Buiten Zorn: {out_zorn}")
        else:
            # chi>1: analyse per bond-slice
            shape=mpo[g].shape
            total_params=mpo[g].size
            print(f"  Group {g}: shape={shape}, params={total_params} (chi>1)")

            # For chi=2 on one side: two 8×8 slices
            if shape[0]==1 and shape[3]==2:
                for s in range(2):
                    M=mpo[g][0,:,:,s]
                    coeffs=[np.trace(P.conj().T@M)/8 for P in pauli_3q]
                    mag=np.abs(np.array(coeffs))
                    nz=np.sum(mag>1e-10)
                    nz_p=[pauli_3q_names[i] for i in range(64) if mag[i]>1e-10]
                    print(f"    Slice chi_R={s}: {nz} Paulis: {nz_p}")
            elif shape[0]==2 and shape[3]==1:
                for s in range(2):
                    M=mpo[g][s,:,:,0]
                    coeffs=[np.trace(P.conj().T@M)/8 for P in pauli_3q]
                    mag=np.abs(np.array(coeffs))
                    nz=np.sum(mag>1e-10)
                    nz_p=[pauli_3q_names[i] for i in range(64) if mag[i]>1e-10]
                    print(f"    Slice chi_L={s}: {nz} Paulis: {nz_p}")
