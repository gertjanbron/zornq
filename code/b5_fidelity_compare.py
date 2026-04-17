import numpy as np
from scipy.linalg import svd
import time

def qaoa_exact(n, edges, gammas, betas):
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    indices = np.arange(dim)
    for layer in range(len(gammas)):
        gamma, beta = gammas[layer], betas[layer]
        phase = np.zeros(dim)
        for (i,j) in edges:
            bi = (indices >> (n-1-i)) & 1
            bj = (indices >> (n-1-j)) & 1
            phase += gamma * (1 - 2*(bi ^ bj))
        psi *= np.exp(1j * phase)
        c, s = np.cos(beta), -1j*np.sin(beta)
        for q in range(n):
            mask = 1 << (n-1-q)
            idx0 = indices[indices & mask == 0]
            idx1 = idx0 | mask
            a, b = psi[idx0].copy(), psi[idx1].copy()
            psi[idx0] = c*a + s*b
            psi[idx1] = s*a + c*b
    return psi / np.linalg.norm(psi)

def mps_svd(psi, n_sites, site_dim, chi_max):
    mps = []
    C = psi.reshape(1, -1)
    for i in range(n_sites - 1):
        chi_l = C.shape[0]
        rest = C.shape[1] // site_dim
        C = C.reshape(chi_l * site_dim, rest)
        U, S, Vh = svd(C, full_matrices=False)
        k = max(1, min(chi_max, int(np.sum(S > 1e-14))))
        mps.append(U[:,:k].reshape(chi_l, site_dim, k))
        C = np.diag(S[:k]) @ Vh[:k,:]
    mps.append(C.reshape(C.shape[0], site_dim, 1))
    return mps

def mps_to_state(mps):
    result = mps[0]
    for i in range(1, len(mps)):
        contracted = np.einsum('ijk,kml->ijml', result, mps[i])
        s = contracted.shape
        result = contracted.reshape(s[0], s[1]*s[2], s[3])
    return result.reshape(-1)

def fidelity(psi1, psi2):
    p1 = psi1 / np.linalg.norm(psi1)
    p2 = psi2 / np.linalg.norm(psi2)
    return abs(np.dot(p1.conj(), p2))**2

def truncation_error(psi, n_sites, site_dim, chi_max):
    """Compute sum of discarded singular values squared."""
    C = psi.reshape(1, -1)
    total_disc = 0.0
    for i in range(n_sites - 1):
        chi_l = C.shape[0]
        rest = C.shape[1] // site_dim
        C = C.reshape(chi_l * site_dim, rest)
        U, S, Vh = svd(C, full_matrices=False)
        k = max(1, min(chi_max, int(np.sum(S > 1e-14))))
        total_disc += np.sum(S[k:]**2)
        C = np.diag(S[:k]) @ Vh[:k,:]
    return total_disc

print("="*70)
print("  ZORN-TRIPLET vs STANDAARD: VOLLEDIGE VERGELIJKING")
print("="*70)

rng = np.random.default_rng(42)

# === Big comparison table ===
print(f"\n{'n':>3s} {'d':>3s} {'Method':<25s} {'chi':>4s} {'Fidelity':>12s} {'Params':>8s}")
print("-"*60)

for n in [12, 15, 18]:
    edges = [(i,i+1) for i in range(n-1)]
    for depth in [1, 3, 5, 8]:
        if n == 18 and depth == 8:
            continue  # might be slow
        gammas = [0.3 + 0.1*rng.standard_normal() for _ in range(depth)]
        betas = [0.5 + 0.1*rng.standard_normal() for _ in range(depth)]
        psi = qaoa_exact(n, edges, gammas, betas)
        
        configs = [
            ("Standard 1q/site", n, 2, 8),
            ("Standard 1q/site", n, 2, 16),
            (f"Zorn-triplet 3q/site", n//3, 8, 8),
            (f"Zorn-triplet 3q/site", n//3, 8, 16),
        ]
        
        for label, ns, sd, chi in configs:
            mps = mps_svd(psi, ns, sd, chi)
            f = fidelity(psi, mps_to_state(mps))
            params = sum(m.size for m in mps)
            fstr = "EXACT" if f > 0.999999 else f"{f*100:.4f}%"
            print(f"{n:3d} {depth:3d} {label:<25s} {chi:4d} {fstr:>12s} {params:8d}")
        print()

# === Key comparison: equal parameters ===
print("\n" + "="*70)
print("  GELIJKE PARAMETERS VERGELIJKING")
print("="*70)
print("\n  Bij gelijke parameter-budgetten:")
print(f"\n  {'Config':<35s} {'chi':>4s} {'d':>3s} {'#sites':>6s} {'~Params':>8s}")
print("  " + "-"*60)
# n=15: 
# Standard 1q: 15 sites, d=2, chi=8 → ~15*2*8*8 = 1920
# Zorn 3q: 5 sites, d=8, chi=8 → ~5*8*8*8 = 2560
# Standard 1q: 15 sites, d=2, chi=11 → ~15*2*11*11 = 3630
# Zorn 3q: 5 sites, d=8, chi=6 → ~5*8*6*6 = 1440
print("  Standard 1q/site, chi=8           8   2     15     1920")
print("  Zorn-triplet 3q/site, chi=8       8   8      5     2560")
print("  Standard 1q/site, chi=12         12   2     15     4320")
print("  Zorn-triplet 3q/site, chi=4       4   8      5      640")

print("\n  Fidelity bij n=15:")
n = 15
edges = [(i,i+1) for i in range(n-1)]
for depth in [5, 8]:
    gammas = [0.3 + 0.1*rng.standard_normal() for _ in range(depth)]
    betas = [0.5 + 0.1*rng.standard_normal() for _ in range(depth)]
    psi = qaoa_exact(n, edges, gammas, betas)
    
    configs = [
        ("Std chi=8", n, 2, 8),
        ("Std chi=12", n, 2, 12),
        ("Zorn chi=4", n//3, 8, 4),
        ("Zorn chi=8", n//3, 8, 8),
    ]
    print(f"\n  Diepte {depth}:")
    for label, ns, sd, chi in configs:
        mps = mps_svd(psi, ns, sd, chi)
        f = fidelity(psi, mps_to_state(mps))
        params = sum(m.size for m in mps)
        fstr = "EXACT" if f > 0.999999 else f"{f*100:.6f}%"
        print(f"    {label:<20s} {fstr:>14s}  ({params} params)")

# === Extrapolation ===
print("\n\n" + "="*70)
print("  EXTRAPOLATIE 500 QUBITS")
print("="*70)
print(f"\n  {'Method':<30s} {'chi':>4s} {'Sites':>6s} {'Params':>10s} {'RAM':>10s}")
print("  " + "-"*65)
for label, n_sites, d, chi in [
    ("Standard 1q/site", 500, 2, 8),
    ("Standard 1q/site", 500, 2, 16),
    ("Zorn-triplet 3q/site", 167, 8, 8),
    ("Zorn-triplet 3q/site", 167, 8, 16),
    ("Zorn-triplet 3q/site", 167, 8, 4),
]:
    params = 2*(1*d*chi) + (n_sites-2)*(chi*d*chi)
    ram = params * 16  # complex128
    ram_str = f"{ram/1024:.1f} KB" if ram < 1024*1024 else f"{ram/1024/1024:.1f} MB"
    print(f"  {label:<30s} {chi:4d} {n_sites:6d} {params:10d} {ram_str:>10s}")

print("\nKLAAR")
