"""B12c: Snelle perturbatieve test — enkele edge"""
import numpy as np
from numpy.linalg import svd
from itertools import combinations
from math import comb
import time

def bit_patterns(Ly):
    d = 2**Ly
    return np.array([[(idx >> (Ly-1-q)) & 1 for q in range(Ly)] for idx in range(d)])

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

class PertQAOA:
    def __init__(self, Lx, Ly, chi_max=256, level=None):
        self.Lx, self.Ly = Lx, Ly
        self.d = 2**Ly
        self.chi_max = chi_max
        self.level = level if level is not None else Ly
        self.bp = bit_patterns(Ly)
        H1 = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
        d, bp = self.d, self.bp
        self.Hd = np.ones((d,d), dtype=complex)
        for q in range(Ly):
            self.Hd *= H1[bp[:,q:q+1], bp[:,q:q+1].T]
    
    def _zzi_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        diag = np.ones(d, dtype=complex)
        for y in range(Ly-1):
            diag *= np.exp(-1j*gamma*(1-2*bp[:,y].astype(float))*(1-2*bp[:,y+1].astype(float)))
        return diag
    
    def _zze_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        level = min(self.level, Ly)
        z = 1 - 2*bp.astype(float)
        cg, sg = np.cos(gamma), -1j*np.sin(gamma)
        gate = np.zeros((d,d), dtype=complex)
        for k in range(level+1):
            for S in ([()] if k==0 else combinations(range(Ly), k)):
                coeff = cg**(Ly-k) * sg**k
                L = np.ones(d); R = np.ones(d)
                for y in (S if k > 0 else []):
                    L *= z[:,y]; R *= z[:,y]
                gate += coeff * np.outer(L, R)
        return gate.ravel()
    
    def _rx_col(self, beta):
        d, Ly, bp = self.d, self.Ly, self.bp
        rx = Rx(2*beta)
        R = np.ones((d,d), dtype=complex)
        for q in range(Ly):
            R *= rx[bp[:,q:q+1], bp[:,q:q+1].T]
        return R
    
    def eval_single_edge(self, p, gammas, betas):
        """Evalueer <ZZ> op edge (0,0)-(1,0) via Heisenberg-MPO."""
        Lx, Ly, d = self.Lx, self.Ly, self.d
        bp = self.bp
        
        # Observable: Z op qubit 0 van kolom 0 en kolom 1
        mpo = [np.eye(d, dtype=complex).reshape(1,d,d,1).copy() for _ in range(Lx)]
        for col in [0, 1]:
            diag = (1 - 2*bp[:,0].astype(float)).astype(complex)
            mpo[col] = np.diag(diag).reshape(1,d,d,1)
        
        # Build gate list
        gates = []
        for x in range(Lx):
            gates.append(('full', x, self.Hd))
        for l in range(p):
            zzi = self._zzi_diag(gammas[l])
            zze = self._zze_diag(gammas[l])
            rxd = self._rx_col(betas[l])
            for x in range(Lx): gates.append(('diag1', x, zzi))
            for x in range(Lx-1): gates.append(('diag2', x, zze))
            for x in range(Lx): gates.append(('full', x, rxd))
        
        # Evolve
        max_chi = 1
        for gt, s, data in reversed(gates):
            if gt == 'full':
                Ud = data.conj().T
                mpo[s] = np.einsum('ij,ajkb,kl->ailb', Ud, mpo[s], data)
            elif gt == 'diag1':
                cd = np.conj(data)
                mpo[s] = mpo[s] * cd[None,:,None,None] * data[None,None,:,None]
            else:  # diag2
                cl, cr = mpo[s].shape[0], mpo[s+1].shape[3]
                Th = np.einsum('aijc,cklb->aijklb', mpo[s], mpo[s+1])
                cd = np.conj(data).reshape(d,d)
                dd = data.reshape(d,d)
                Th = Th * cd[None,:,None,:,None,None] * dd[None,None,:,None,:,None]
                mat = Th.reshape(cl*d*d, d*d*cr)
                U, S, V = svd(mat, full_matrices=False)
                Sa = np.abs(S)
                k = max(1, int(np.sum(Sa > 1e-12*Sa[0])))
                k = min(k, self.chi_max)
                max_chi = max(max_chi, k)
                mpo[s] = U[:,:k].reshape(cl,d,d,k)
                mpo[s+1] = (np.diag(S[:k]) @ V[:k,:]).reshape(k,d,d,cr)
        
        # Contract
        L = np.ones((1,), dtype=complex)
        for W in mpo:
            L = np.einsum('a,ab->b', L, W[:,0,0,:])
        return L[0].real, max_chi

# === TESTS ===
print("="*65)
print("PERTURBATIEVE QAOA: SINGLE-EDGE TEST")
print("="*65)
beta = 1.1778

for Lx, Ly in [(4,2), (4,3), (3,4)]:
    n_e = Lx*(Ly-1) + (Lx-1)*Ly
    avg_d = 2*n_e/(Lx*Ly)
    gamma = 0.88/avg_d
    
    print(f"\n--- {Lx}×{Ly} grid (d={2**Ly}), p=1, γ={gamma:.4f} ---")
    print(f"  {'Level':>7s}  {'MaxRank':>7s}  {'<ZZ>':>10s}  {'Fout%':>7s}  {'Chi':>4s}  {'Tijd':>6s}")
    
    r_ref = None
    for lv in [Ly, 0, 1, 2, 3][:Ly+1]:
        tr = sum(comb(Ly,k) for k in range(min(lv,Ly)+1))
        t0 = time.time()
        eng = PertQAOA(Lx, Ly, chi_max=256, level=lv)
        zz, chi = eng.eval_single_edge(1, [gamma], [beta])
        dt = time.time() - t0
        if lv == Ly:
            r_ref = zz
            print(f"  exact   {2**Ly:>7d}  {zz:10.6f}  {'---':>7s}  {chi:>4d}  {dt:5.2f}s")
        else:
            fout = abs(zz - r_ref)/abs(r_ref)*100 if abs(r_ref)>1e-15 else 0
            print(f"  {lv:>7d}  ≤{tr:>6d}  {zz:10.6f}  {fout:6.3f}%  {chi:>4d}  {dt:5.2f}s")

# p=2 test
print(f"\n--- 4×2 grid (d=4), p=2, γ={0.88/2.67:.4f} ---")
Lx, Ly = 4, 2
gamma = 0.88 / 2.67
print(f"  {'Level':>7s}  {'<ZZ>':>10s}  {'Fout%':>7s}  {'Chi':>4s}  {'Tijd':>6s}")
r_ref = None
for lv in [Ly, 0, 1]:
    t0 = time.time()
    eng = PertQAOA(Lx, Ly, chi_max=256, level=lv)
    zz, chi = eng.eval_single_edge(2, [gamma]*2, [beta]*2)
    dt = time.time() - t0
    if lv == Ly:
        r_ref = zz
        print(f"  exact   {zz:10.6f}  {'---':>7s}  {chi:>4d}  {dt:5.2f}s")
    else:
        fout = abs(zz - r_ref)/abs(r_ref)*100 if abs(r_ref)>1e-15 else 0
        print(f"  {lv:>7d}  {zz:10.6f}  {fout:6.3f}%  {chi:>4d}  {dt:5.2f}s")

# p=2 op Ly=3
print(f"\n--- 3×3 grid (d=8), p=2 ---")
Lx, Ly = 3, 3
n_e = Lx*(Ly-1) + (Lx-1)*Ly
avg_d = 2*n_e/(Lx*Ly)
gamma = 0.88/avg_d
print(f"  γ={gamma:.4f}, avg_deg={avg_d:.2f}")
print(f"  {'Level':>7s}  {'<ZZ>':>10s}  {'Fout%':>7s}  {'Chi':>4s}  {'Tijd':>6s}")
r_ref = None
for lv in [Ly, 0, 1, 2]:
    t0 = time.time()
    eng = PertQAOA(Lx, Ly, chi_max=256, level=lv)
    zz, chi = eng.eval_single_edge(2, [gamma]*2, [beta]*2)
    dt = time.time() - t0
    if lv == Ly:
        r_ref = zz
        print(f"  exact   {zz:10.6f}  {'---':>7s}  {chi:>4d}  {dt:5.2f}s")
    else:
        fout = abs(zz - r_ref)/abs(r_ref)*100 if abs(r_ref)>1e-15 else 0
        print(f"  {lv:>7d}  {zz:10.6f}  {fout:6.3f}%  {chi:>4d}  {dt:5.2f}s")

print("\n" + "="*65)
print("CONCLUSIE")
print("="*65)
