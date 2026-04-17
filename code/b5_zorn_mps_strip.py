import numpy as np
from scipy.linalg import svd
import time

def qaoa_exact(n, edges, gammas, betas):
    """Vectorized QAOA."""
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    indices = np.arange(dim)
    
    for layer in range(len(gammas)):
        gamma, beta = gammas[layer], betas[layer]
        # ZZ phase
        phase = np.zeros(dim)
        for (i,j) in edges:
            bi = (indices >> (n-1-i)) & 1
            bj = (indices >> (n-1-j)) & 1
            phase += gamma * (1 - 2*(bi ^ bj))
        psi *= np.exp(1j * phase)
        # Rx mixer
        c, s = np.cos(beta), -1j*np.sin(beta)
        for q in range(n):
            mask = 1 << (n-1-q)
            idx0 = indices[indices & mask == 0]
            idx1 = idx0 | mask
            a, b = psi[idx0].copy(), psi[idx1].copy()
            psi[idx0] = c*a + s*b
            psi[idx1] = s*a + c*b
    return psi / np.linalg.norm(psi)

def state_to_mps(psi, n, chi_max=None):
    mps = []
    C = psi.reshape(1, -1)
    for i in range(n-1):
        chi_l = C.shape[0]
        C = C.reshape(chi_l * 2, -1)
        U, S, Vh = svd(C, full_matrices=False)
        k = int(np.sum(S > 1e-14))
        if chi_max: k = min(k, chi_max)
        k = max(1, k)
        mps.append(U[:,:k].reshape(chi_l, 2, k))
        C = np.diag(S[:k]) @ Vh[:k,:]
    mps.append(C.reshape(C.shape[0], 2, 1))
    return mps

def mps_fidelity(psi_exact, mps):
    """Compute fidelity without reconstructing full state."""
    n = len(mps)
    # Contract <psi|MPS> using left-to-right sweep
    psi = psi_exact.reshape([2]*n)
    # Build overlap tensor
    # L[alpha] = sum over physical indices of psi * mps
    L = np.ones((1,), dtype=complex)
    dim = 2**n
    
    # Actually, for correctness let's just reconstruct
    result = mps[0]
    for i in range(1, n):
        contracted = np.einsum('ijk,kml->ijml', result, mps[i])
        s = contracted.shape
        result = contracted.reshape(s[0], s[1]*s[2], s[3])
    psi_mps = result.reshape(-1)
    
    p1 = psi_exact / np.linalg.norm(psi_exact)
    p2 = psi_mps / np.linalg.norm(psi_mps)
    return abs(np.dot(p1.conj(), p2))**2

def schmidt_ranks(psi, n):
    ranks = []
    for cut in range(1, n):
        mat = psi.reshape(2**cut, -1)
        S = svd(mat, compute_uv=False)
        ranks.append(int(np.sum(S > 1e-10)))
    return ranks

print("="*70)
print("  B5: ZORN-MPS STRIP VERIFICATIE")
print("="*70)

# Test n=12
print("\n--- QAOA fidelity, n=12 ---")
rng = np.random.default_rng(42)
edges12 = [(i,i+1) for i in range(11)]
chi_vals = [4, 8, 16, 32]
depths = [1, 3, 5, 8]

print(f"\n  {'Depth':<6s}", end="")
for chi in chi_vals:
    print(f"  {'chi='+str(chi):>10s}", end="")
print()
print("  " + "-"*(6 + 12*len(chi_vals)))

for d in depths:
    gammas = [0.3 + 0.1*rng.standard_normal() for _ in range(d)]
    betas = [0.5 + 0.1*rng.standard_normal() for _ in range(d)]
    t0 = time.time()
    psi = qaoa_exact(12, edges12, gammas, betas)
    
    print(f"  {d:<6d}", end="")
    for chi in chi_vals:
        mps = state_to_mps(psi, 12, chi_max=chi)
        f = mps_fidelity(psi, mps)
        if f > 0.999999:
            print(f"  {'EXACT':>10s}", end="")
        else:
            print(f"  {f*100:>9.4f}%", end="")
    print(f"  ({time.time()-t0:.1f}s)")

# Schmidt ranks
print("\n--- Schmidt rank profiel n=12 ---")
rng2 = np.random.default_rng(99)
for d in [1, 3, 5, 8]:
    gammas = [0.3 + 0.1*rng2.standard_normal() for _ in range(d)]
    betas = [0.5 + 0.1*rng2.standard_normal() for _ in range(d)]
    psi = qaoa_exact(12, edges12, gammas, betas)
    sr = schmidt_ranks(psi, 12)
    print(f"  Diepte {d}: max={max(sr):3d}, mid={sr[5]:3d}, profiel={sr}")

# Test n=15
print("\n--- QAOA fidelity, n=15 ---")
rng = np.random.default_rng(42)
edges15 = [(i,i+1) for i in range(14)]
chi_vals15 = [8, 16, 32]
depths15 = [1, 3, 5]

print(f"\n  {'Depth':<6s}", end="")
for chi in chi_vals15:
    print(f"  {'chi='+str(chi):>10s}", end="")
print()
print("  " + "-"*(6 + 12*len(chi_vals15)))

for d in depths15:
    gammas = [0.3 + 0.1*rng.standard_normal() for _ in range(d)]
    betas = [0.5 + 0.1*rng.standard_normal() for _ in range(d)]
    t0 = time.time()
    psi = qaoa_exact(15, edges15, gammas, betas)
    
    print(f"  {d:<6d}", end="")
    for chi in chi_vals15:
        mps = state_to_mps(psi, 15, chi_max=chi)
        f = mps_fidelity(psi, mps)
        if f > 0.999999:
            print(f"  {'EXACT':>10s}", end="")
        else:
            print(f"  {f*100:>9.4f}%", end="")
    print(f"  ({time.time()-t0:.1f}s)")

# Extrapolatie
print("\n--- Extrapolatie 500 qubits ---")
for chi in [8, 16, 32, 64]:
    params = 2*(1*2*chi) + 498*(chi*2*chi)
    ram_kb = params * 16 / 1024
    print(f"  chi={chi:3d}: {params:>10d} params, {ram_kb:>8.1f} KB")

print("\nKLAAR")
