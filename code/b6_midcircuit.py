import numpy as np
from scipy.linalg import svd

I2 = np.eye(2, dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
def kron_list(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def qaoa(n, edges, gammas, betas, psi_init=None):
    dim = 2**n; indices = np.arange(dim)
    psi = np.ones(dim, dtype=complex)/np.sqrt(dim) if psi_init is None else psi_init.copy()
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
    return psi/np.linalg.norm(psi)

def mps_compress(psi, n, chi_max):
    mps = []; C = psi.reshape(1,-1)
    for i in range(n-1):
        cl = C.shape[0]; C = C.reshape(cl*2,-1)
        U,S,Vh = svd(C, full_matrices=False)
        k = max(1, min(chi_max, int(np.sum(S>1e-14))))
        mps.append(U[:,:k].reshape(cl,2,k)); C = np.diag(S[:k])@Vh[:k,:]
    mps.append(C.reshape(C.shape[0],2,1))
    result = mps[0]
    for i in range(1, len(mps)):
        c = np.einsum('ijk,kml->ijml', result, mps[i]); s = c.shape
        result = c.reshape(s[0], s[1]*s[2], s[3])
    p = result.reshape(-1); return p/np.linalg.norm(p)

n = 12; edges = [(i,i+1) for i in range(n-1)]
rng = np.random.default_rng(42)
gammas = [0.3+0.1*rng.standard_normal() for _ in range(8)]
betas = [0.5+0.1*rng.standard_normal() for _ in range(8)]

psi_full = qaoa(n, edges, gammas, betas)
Z0 = kron_list([Z]+[I2]*(n-1))
ZZ01 = kron_list([Z,Z]+[I2]*(n-2))
Zmid = kron_list([I2]*(n//2-1)+[Z,Z]+[I2]*(n//2-1))

eZ0 = np.real(psi_full.conj()@Z0@psi_full)
eZZ = np.real(psi_full.conj()@ZZ01@psi_full)
eZm = np.real(psi_full.conj()@Zmid@psi_full)

print("="*70)
print("  TEST 3: MID-CIRCUIT COMPRESSIE")
print("="*70)
print(f"\n  QAOA 8L, n={n}")
print(f"  Exact: <Z0>={eZ0:.6f}, <Z0Z1>={eZZ:.6f}, <Zmid>={eZm:.6f}")

# Single compression after layer 4
print(f"\n  --- Enkele compressie na laag 4 ---")
psi_4 = qaoa(n, edges, gammas[:4], betas[:4])
print(f"  {'chi':>4s}  {'<Z0> err':>10s}  {'<ZZ01> err':>11s}  {'<Zmid> err':>11s}  {'Fidelity':>10s}")
print("  "+"-"*52)

for chi in [4, 8, 16, 32]:
    psi_c = mps_compress(psi_4, n, chi)
    psi_rest = qaoa(n, edges, gammas[4:], betas[4:], psi_init=psi_c)
    e0 = np.real(psi_rest.conj()@Z0@psi_rest)
    ezz = np.real(psi_rest.conj()@ZZ01@psi_rest)
    em = np.real(psi_rest.conj()@Zmid@psi_rest)
    fid = abs(np.dot(psi_full.conj(), psi_rest))**2
    print(f"  {chi:4d}  {abs(e0-eZ0):10.2e}  {abs(ezz-eZZ):11.2e}  {abs(em-eZm):11.2e}  {fid*100:9.4f}%")

# Double compression: after layer 2 and 5
print(f"\n  --- Dubbele compressie (na laag 2 en 5) ---")
print(f"  {'chi':>4s}  {'<Z0> err':>10s}  {'<Zmid> err':>11s}  {'Fidelity':>10s}")
print("  "+"-"*40)

for chi in [8, 16, 32]:
    psi = qaoa(n, edges, gammas[:2], betas[:2])
    psi = mps_compress(psi, n, chi)
    psi = qaoa(n, edges, gammas[2:5], betas[2:5], psi_init=psi)
    psi = mps_compress(psi, n, chi)
    psi = qaoa(n, edges, gammas[5:], betas[5:], psi_init=psi)
    e0 = np.real(psi.conj()@Z0@psi)
    em = np.real(psi.conj()@Zmid@psi)
    fid = abs(np.dot(psi_full.conj(), psi))**2
    print(f"  {chi:4d}  {abs(e0-eZ0):10.2e}  {abs(em-eZm):11.2e}  {fid*100:9.4f}%")

# Triple compression: every 2 layers
print(f"\n  --- Triple compressie (na laag 2, 4, 6) ---")
print(f"  {'chi':>4s}  {'<Z0> err':>10s}  {'<Zmid> err':>11s}  {'Fidelity':>10s}")
print("  "+"-"*40)

for chi in [8, 16, 32]:
    psi = qaoa(n, edges, gammas[:2], betas[:2])
    psi = mps_compress(psi, n, chi)
    psi = qaoa(n, edges, gammas[2:4], betas[2:4], psi_init=psi)
    psi = mps_compress(psi, n, chi)
    psi = qaoa(n, edges, gammas[4:6], betas[4:6], psi_init=psi)
    psi = mps_compress(psi, n, chi)
    psi = qaoa(n, edges, gammas[6:], betas[6:], psi_init=psi)
    e0 = np.real(psi.conj()@Z0@psi)
    em = np.real(psi.conj()@Zmid@psi)
    fid = abs(np.dot(psi_full.conj(), psi))**2
    print(f"  {chi:4d}  {abs(e0-eZ0):10.2e}  {abs(em-eZm):11.2e}  {fid*100:9.4f}%")

print("\nKLAAR")
