import numpy as np
from scipy.optimize import minimize

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def kron_list(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def heisenberg_ham(n):
    dim = 2**n; H = np.zeros((dim,dim), dtype=complex)
    for i in range(n-1):
        for P in [X,Y,Z]:
            ops=[I2]*n; ops[i]=P; ops[i+1]=P; H += kron_list(ops)
    return H

def ising_ham(n, h=1.0):
    dim = 2**n; H = np.zeros((dim,dim), dtype=complex)
    for i in range(n-1):
        ops=[I2]*n; ops[i]=Z; ops[i+1]=Z; H -= kron_list(ops)
    for i in range(n):
        ops=[I2]*n; ops[i]=X; H -= h*kron_list(ops)
    return H

def product_state(params, n):
    ng = n//3; psi = None
    for g in range(ng):
        z = params[g*8:(g+1)*8]; z = z/(np.linalg.norm(z)+1e-15)
        psi = z if psi is None else np.kron(psi, z)
    return psi/(np.linalg.norm(psi)+1e-15)

def mps_state(params, n, chi):
    ng = n//3; d = 8; mps = []; idx = 0
    for g in range(ng):
        cl = 1 if g==0 else chi
        cr = 1 if g==ng-1 else chi
        sz = cl*d*cr
        mps.append(params[idx:idx+sz].reshape(cl,d,cr)); idx += sz
    result = mps[0]
    for i in range(1, len(mps)):
        c = np.einsum('ijk,kml->ijml', result, mps[i])
        s = c.shape; result = c.reshape(s[0], s[1]*s[2], s[3])
    psi = result.reshape(-1)
    n_val = np.linalg.norm(psi)
    return psi/n_val if n_val>1e-15 else psi

def e_prod(p, H, n): 
    psi = product_state(p, n); return np.real(psi.conj()@H@psi)
def e_mps(p, H, n, chi): 
    psi = mps_state(p, n, chi); return np.real(psi.conj()@H@psi)

print("="*70)
print("  TEST 1: ZORN-VQE OP VERSTRENGELDE TOESTANDEN")
print("="*70)

for label, ham_fn in [("Heisenberg XXX", heisenberg_ham), ("TF Ising h=J", lambda n: ising_ham(n,1.0))]:
    n = 6
    H = ham_fn(n)
    evals, evecs = np.linalg.eigh(H)
    E_ex = evals[0]; psi_ex = evecs[:,0]
    
    print(f"\n  {label}, n={n}, E_exact = {E_ex:.6f}")
    
    # Product
    np_ = (n//3)*8
    best = 999
    for t in range(8):
        x0 = np.random.default_rng(t).standard_normal(np_)
        r = minimize(e_prod, x0, args=(H,n), method='L-BFGS-B', options={'maxiter':300})
        if r.fun < best: best = r.fun; bx = r.x
    fid = abs(np.dot(psi_ex.conj(), product_state(bx,n)))**2
    print(f"    Product ({np_}p): E={best:.6f}, gap={best-E_ex:.4f}, fid={fid:.4f}")
    
    # MPS
    for chi in [2, 4]:
        ng = n//3
        np_m = (1*8*chi) + max(0,ng-2)*(chi*8*chi) + (chi*8*1)
        best_m = 999
        for t in range(8):
            x0 = np.random.default_rng(50*chi+t).standard_normal(np_m)*0.1
            r = minimize(e_mps, x0, args=(H,n,chi), method='L-BFGS-B', options={'maxiter':500})
            if r.fun < best_m: best_m = r.fun; bx_m = r.x
        fid = abs(np.dot(psi_ex.conj(), mps_state(bx_m,n,chi)))**2
        print(f"    MPS chi={chi} ({np_m}p): E={best_m:.6f}, gap={best_m-E_ex:.4f}, fid={fid:.4f}")

print("\nKLAAR")
