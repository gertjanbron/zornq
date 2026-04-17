import numpy as np
from scipy.linalg import svd
import time

def state_to_mps(psi, n_sites, site_dim, chi_max=None):
    mps = []; C = psi.reshape(1, -1)
    for i in range(n_sites - 1):
        cl = C.shape[0]; C = C.reshape(cl * site_dim, -1)
        U, S, Vh = svd(C, full_matrices=False)
        k = max(1, min(chi_max or len(S), int(np.sum(S > 1e-14))))
        mps.append(U[:,:k].reshape(cl, site_dim, k))
        C = np.diag(S[:k]) @ Vh[:k,:]
    mps.append(C.reshape(C.shape[0], site_dim, 1))
    return mps

def mps_to_state(mps):
    result = mps[0]
    for i in range(1, len(mps)):
        c = np.einsum('ijk,kml->ijml', result, mps[i])
        s = c.shape; result = c.reshape(s[0], s[1]*s[2], s[3])
    return result.reshape(-1)

# === FAST local expectation values ===
def mps_local_expect(mps, O_dict):
    """Compute multiple local expectation values in one left-to-right sweep.
    O_dict: {site_index: O_matrix} where O is (d,d).
    Returns dict of expectation values.
    
    Strategy: build left environments once, store them.
    Then for each operator site, contract with operator and sweep right.
    """
    n = len(mps)
    
    # Build all left environments L[i] = contraction of sites 0..i-1
    lefts = [None] * (n + 1)
    lefts[0] = np.ones((1, 1), dtype=complex)
    for i in range(n):
        M = mps[i]  # (cl, d, cr)
        # L[i+1][b,b'] = sum_{a,a',s} L[i][a,a'] * M[a,s,b] * M*[a',s,b']
        # Efficient: reshape and matmul
        cl, d, cr = M.shape
        # M reshaped: (cl, d*cr) 
        # L @ M gives (cl, d*cr) for bra, need to contract with ket
        L = lefts[i]  # (cl, cl)
        # Contract: L[a,a'] M[a,s,b] conj(M)[a',s,b']
        # = (L @ M.reshape(cl, d*cr)).reshape(d,cr) contracted with conj(M)
        # Better: use einsum but with small dimensions
        lefts[i+1] = np.einsum('ab,asc,bsd->cd', L, M, M.conj())
    
    # Build all right environments R[i] = contraction of sites i..n-1
    rights = [None] * (n + 1)
    rights[n] = np.ones((1, 1), dtype=complex)
    for i in range(n-1, -1, -1):
        M = mps[i]
        rights[i] = np.einsum('asc,bsd,cd->ab', M, M.conj(), rights[i+1])
    
    # For each operator, compute expectation value
    results = {}
    for site, O in O_dict.items():
        M = mps[site]
        cl, d, cr = M.shape
        # <O> = L[site] * M * O * M* * R[site+1]
        val = np.einsum('ab,asc,st,btd,cd->', lefts[site], M, O, M.conj(), rights[site+1])
        results[site] = np.real(val)
    
    return results

def mps_2site_expect(mps, O_2site, site1, site2):
    """Compute <psi|O_{s1} O_{s2}|psi> for two single-site operators."""
    n = len(mps)
    O1, O2 = O_2site
    
    # Sweep left to right
    L = np.ones((1, 1), dtype=complex)
    for i in range(n):
        M = mps[i]
        if i == site1:
            L = np.einsum('ab,asc,st,btd->cd', L, M, O1, M.conj())
        elif i == site2:
            L = np.einsum('ab,asc,st,btd->cd', L, M, O2, M.conj())
        else:
            L = np.einsum('ab,asc,bsd->cd', L, M, M.conj())
    
    return np.real(L.item())

# === QAOA ===
def qaoa_exact(n, edges, gammas, betas):
    dim = 2**n; psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    indices = np.arange(dim)
    for layer in range(len(gammas)):
        g, b = gammas[layer], betas[layer]
        phase = np.zeros(dim)
        for (i,j) in edges:
            bi = (indices >> (n-1-i)) & 1; bj = (indices >> (n-1-j)) & 1
            phase += g * (1 - 2*(bi ^ bj))
        psi *= np.exp(1j * phase)
        c, s = np.cos(b), -1j*np.sin(b)
        for q in range(n):
            mask = 1<<(n-1-q); i0 = indices[indices&mask==0]; i1 = i0|mask
            a, bb = psi[i0].copy(), psi[i1].copy()
            psi[i0] = c*a + s*bb; psi[i1] = s*a + c*bb
    return psi / np.linalg.norm(psi)

# === Observables ===
I2 = np.eye(2, dtype=complex)
Zop = np.array([[1,0],[0,-1]], dtype=complex)

def Z_in_triplet(q):
    """Z on qubit q (0,1,2) within 3-qubit triplet → 8x8."""
    ops = [I2, I2, I2]; ops[q] = Zop
    return np.kron(np.kron(ops[0], ops[1]), ops[2])

def ZZ_in_triplet(q1, q2):
    ops = [I2, I2, I2]; ops[q1] = Zop; ops[q2] = Zop
    return np.kron(np.kron(ops[0], ops[1]), ops[2])

def exact_Z(psi, n, q):
    indices = np.arange(2**n); bits = (indices >> (n-1-q)) & 1
    return np.real(np.sum(np.abs(psi)**2 * (1 - 2*bits)))

def exact_ZZ(psi, n, q1, q2):
    indices = np.arange(2**n)
    b1 = (indices >> (n-1-q1)) & 1; b2 = (indices >> (n-1-q2)) & 1
    return np.real(np.sum(np.abs(psi)**2 * (1 - 2*(b1^b2))))

# === MAIN ===
print("="*70)
print("  B6b: LOKALE VERWACHTINGSWAARDEN UIT MPS")
print("="*70)

rng = np.random.default_rng(42)

# --- Test 1: d=2 MPS, n=12 ---
print("\n--- Test 1: d=2 MPS, n=12, QAOA 5L ---")
n=12; edges=[(i,i+1) for i in range(n-1)]
gammas=[0.3+0.1*rng.standard_normal() for _ in range(5)]
betas=[0.5+0.1*rng.standard_normal() for _ in range(5)]
psi = qaoa_exact(n, edges, gammas, betas)

for chi in [8, 16, 32]:
    mps = state_to_mps(psi, n, 2, chi_max=chi)
    O_dict = {q: Zop for q in range(n)}
    vals = mps_local_expect(mps, O_dict)
    
    errs = [abs(vals[q] - exact_Z(psi, n, q)) for q in range(n)]
    
    # Also test ZZ
    zz_errs = []
    for q in range(n-1):
        mps_zz = mps_2site_expect(mps, (Zop, Zop), q, q+1)
        zz_errs.append(abs(mps_zz - exact_ZZ(psi, n, q, q+1)))
    
    print(f"  chi={chi:3d}: max|err <Z>|={max(errs):.2e}, max|err <ZZ>|={max(zz_errs):.2e}")

# --- Test 2: Zorn-triplet d=8, n=12 ---
print("\n--- Test 2: Zorn-triplet d=8, n=12 ---")
for chi in [4, 8, 16]:
    mps_z = state_to_mps(psi, n//3, 8, chi_max=chi)
    
    results = []
    for g in range(n//3):
        for q in range(3):
            qubit = g*3 + q
            O = Z_in_triplet(q)
            site_vals = mps_local_expect(mps_z, {g: O})
            mps_val = site_vals[g]
            ex_val = exact_Z(psi, n, qubit)
            results.append((qubit, ex_val, mps_val, abs(ex_val-mps_val)))
    
    max_err = max(r[3] for r in results)
    
    # ZZ within triplet
    zz_intra_errs = []
    for g in range(n//3):
        for q1, q2 in [(0,1),(1,2)]:
            O = ZZ_in_triplet(q1, q2)
            v = mps_local_expect(mps_z, {g: O})[g]
            qubit1, qubit2 = g*3+q1, g*3+q2
            ex = exact_ZZ(psi, n, qubit1, qubit2)
            zz_intra_errs.append(abs(v - ex))
    
    # ZZ cross-triplet (qubit 2 of triplet g, qubit 0 of triplet g+1)
    zz_cross_errs = []
    for g in range(n//3 - 1):
        v = mps_2site_expect(mps_z, (Z_in_triplet(2), Z_in_triplet(0)), g, g+1)
        qubit1, qubit2 = g*3+2, (g+1)*3
        ex = exact_ZZ(psi, n, qubit1, qubit2)
        zz_cross_errs.append(abs(v - ex))
    
    print(f"  chi={chi:3d}: max|err <Z>|={max_err:.2e}, "
          f"max|err <ZZ> intra|={max(zz_intra_errs):.2e}, "
          f"max|err <ZZ> cross|={max(zz_cross_errs):.2e}")

# --- Test 3: Timing at n=15, 18 ---
print("\n--- Test 3: Timing ---")
for n_t in [12, 15, 18]:
    edges=[(i,i+1) for i in range(n_t-1)]
    gammas=[0.3+0.1*rng.standard_normal() for _ in range(3)]
    betas=[0.5+0.1*rng.standard_normal() for _ in range(3)]
    psi = qaoa_exact(n_t, edges, gammas, betas)
    
    # Exact
    t0=time.time()
    for q in range(n_t): exact_Z(psi, n_t, q)
    t_ex = time.time()-t0
    
    # d=2 chi=16
    mps = state_to_mps(psi, n_t, 2, chi_max=16)
    t0=time.time()
    mps_local_expect(mps, {q: Zop for q in range(n_t)})
    t_mps = time.time()-t0
    
    # Zorn d=8 chi=8
    mps_z = state_to_mps(psi, n_t//3, 8, chi_max=8)
    t0=time.time()
    for g in range(n_t//3):
        mps_local_expect(mps_z, {g: Z_in_triplet(0)})
    t_zorn = time.time()-t0
    
    print(f"  n={n_t}: exact={t_ex*1000:.1f}ms, MPS(d=2,chi=16)={t_mps*1000:.1f}ms, Zorn(d=8,chi=8)={t_zorn*1000:.1f}ms")

print("\nKLAAR")
