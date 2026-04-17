import numpy as np
from scipy.linalg import svd
import time

def mps_init_plus(n):
    """Product state |+>^n as MPS with chi=1."""
    return [np.ones((1,2,1), dtype=complex)/np.sqrt(2) for _ in range(n)]

def apply_2site_gate(mps, gate, site, chi_max):
    """Apply 2-site gate (4x4) to MPS at sites [site, site+1]."""
    M1, M2 = mps[site], mps[site+1]
    cl, d1, cm = M1.shape
    _, d2, cr = M2.shape
    # Merge: (cl, d1, d2, cr)
    theta = np.einsum('ijk,kml->ijml', M1, M2)
    # Apply gate: gate is (d1*d2, d1*d2), acts on physical indices
    # theta: (cl, d1, d2, cr) -> reshape to (cl, d1d2, cr)
    theta = theta.reshape(cl, d1*d2, cr)
    # gate[out, in] * theta[cl, in, cr] -> (cl, out, cr)
    theta = np.einsum('ij,kjl->kil', gate, theta)
    # SVD: (cl*d1, d2*cr)
    theta = theta.reshape(cl*d1, d2*cr)
    U, S, Vh = svd(theta, full_matrices=False)
    k = max(1, min(chi_max, int(np.sum(S > 1e-14))))
    mps[site] = U[:,:k].reshape(cl, d1, k)
    mps[site+1] = (np.diag(S[:k]) @ Vh[:k,:]).reshape(k, d2, cr)

def apply_1site_gate(mps, gate, site):
    """Apply 1-site gate (2x2) to MPS at site."""
    mps[site] = np.einsum('ij,kjl->kil', gate, mps[site])

def qaoa_mps(n, edges, gammas, betas, chi_max=16):
    mps = mps_init_plus(n)
    for layer in range(len(gammas)):
        g, b = gammas[layer], betas[layer]
        for (i, j) in edges:
            zz = np.diag([np.exp(1j*g), np.exp(-1j*g), np.exp(-1j*g), np.exp(1j*g)])
            apply_2site_gate(mps, zz, i, chi_max)
        Rx = np.array([[np.cos(b), -1j*np.sin(b)], [-1j*np.sin(b), np.cos(b)]])
        for q in range(n):
            apply_1site_gate(mps, Rx, q)
    return mps

def mps_expect_all_Z(mps):
    """Get <Z_q> for all qubits in one sweep."""
    n = len(mps)
    Zop = np.array([[1,0],[0,-1]], dtype=complex)
    lefts = [None]*(n+1); lefts[0] = np.ones((1,1), dtype=complex)
    for i in range(n):
        lefts[i+1] = np.einsum('ab,asc,bsd->cd', lefts[i], mps[i], mps[i].conj())
    rights = [None]*(n+1); rights[n] = np.ones((1,1), dtype=complex)
    for i in range(n-1,-1,-1):
        rights[i] = np.einsum('asc,bsd,cd->ab', mps[i], mps[i].conj(), rights[i+1])
    vals = {}
    for q in range(n):
        vals[q] = np.real(np.einsum('ab,asc,st,btd,cd->', lefts[q], mps[q], Zop, mps[q].conj(), rights[q+1]))
    return vals

def mps_ZZ_corr(mps, s1, s2):
    """<Z_{s1} Z_{s2}>"""
    n = len(mps)
    Zop = np.array([[1,0],[0,-1]], dtype=complex)
    L = np.ones((1,1), dtype=complex)
    for i in range(n):
        M = mps[i]
        if i == s1 or i == s2:
            L = np.einsum('ab,asc,st,btd->cd', L, M, Zop, M.conj())
        else:
            L = np.einsum('ab,asc,bsd->cd', L, M, M.conj())
    return np.real(L.item())

# === Exact QAOA for verification ===
def qaoa_exact(n, edges, gammas, betas):
    dim = 2**n; psi = np.ones(dim, dtype=complex)/np.sqrt(dim)
    indices = np.arange(dim)
    for layer in range(len(gammas)):
        g, b = gammas[layer], betas[layer]
        phase = np.zeros(dim)
        for (i,j) in edges:
            bi = (indices>>(n-1-i))&1; bj = (indices>>(n-1-j))&1
            phase += g*(1-2*(bi^bj))
        psi *= np.exp(1j*phase)
        c, s = np.cos(b), -1j*np.sin(b)
        for q in range(n):
            mask=1<<(n-1-q); i0=indices[indices&mask==0]; i1=i0|mask
            a, bb = psi[i0].copy(), psi[i1].copy()
            psi[i0]=c*a+s*bb; psi[i1]=s*a+c*bb
    return psi/np.linalg.norm(psi)

# === MAIN ===
print("="*70)
print("  B6b: LOKALE VERWACHTINGSWAARDEN — VERIFICATIE EN 500Q")
print("="*70)

rng = np.random.default_rng(42)

# Verify at n=12
print("\n--- Verificatie n=12, QAOA 3L ---")
n=12; edges=[(i,i+1) for i in range(n-1)]
gammas=[0.3+0.1*rng.standard_normal() for _ in range(3)]
betas=[0.5+0.1*rng.standard_normal() for _ in range(3)]

psi_ex = qaoa_exact(n, edges, gammas, betas)
for chi in [8, 16]:
    mps = qaoa_mps(n, edges, gammas, betas, chi_max=chi)
    vals = mps_expect_all_Z(mps)
    errs_z = []
    for q in range(n):
        idx = np.arange(2**n); bits = (idx>>(n-1-q))&1
        ex = np.real(np.sum(np.abs(psi_ex)**2*(1-2*bits)))
        errs_z.append(abs(vals[q]-ex))
    errs_zz = []
    for q in range(n-1):
        idx = np.arange(2**n)
        b1=(idx>>(n-1-q))&1; b2=(idx>>(n-1-q-1))&1
        ex=np.real(np.sum(np.abs(psi_ex)**2*(1-2*(b1^b2))))
        errs_zz.append(abs(mps_ZZ_corr(mps,q,q+1)-ex))
    print(f"  chi={chi}: max|err <Z>|={max(errs_z):.2e}, max|err <ZZ>|={max(errs_zz):.2e}")

# Scale up
print("\n--- SCHALING ---")
for n_big in [50, 100, 200, 500]:
    edges=[(i,i+1) for i in range(n_big-1)]
    gammas=[0.3+0.1*rng.standard_normal() for _ in range(3)]
    betas=[0.5+0.1*rng.standard_normal() for _ in range(3)]
    
    t0=time.time()
    mps = qaoa_mps(n_big, edges, gammas, betas, chi_max=16)
    t_build=time.time()-t0
    
    t0=time.time()
    vals = mps_expect_all_Z(mps)
    t_z=time.time()-t0
    
    t0=time.time()
    E = sum((1 - mps_ZZ_corr(mps,i,i+1))/2 for i,j in edges)
    t_E=time.time()-t0
    
    max_chi = max(m.shape[2] for m in mps[:-1])
    ram_kb = sum(m.nbytes for m in mps)/1024
    
    print(f"  n={n_big:4d}: build={t_build:.2f}s, <Z>={t_z*1000:.0f}ms, E_QAOA={E:.2f} "
          f"(cut={E/len(edges):.4f}), chi_max={max_chi}, RAM={ram_kb:.0f}KB")

print("\nKLAAR")
