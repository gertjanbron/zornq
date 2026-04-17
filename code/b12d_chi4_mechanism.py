"""
B12d: Waarom is chi=4 genoeg bij p=1?

Hypothese: na de mixer-laag (RX gates) interfereren de hogere
componenten (|S|≥2) destructief, zodat effectief alleen
|S|=0 (rank 1) en |S|=1 (rank Ly) bijdragen.
Totaal rank: 1 + Ly ≈ 4 voor Ly=3.

We testen dit door de SINGULAR VALUES te meten van de MPO
na elke gate-laag, per component.
"""
import numpy as np
from numpy.linalg import svd
from itertools import combinations
import time

def bit_patterns(Ly):
    d = 2**Ly
    return np.array([[(idx >> (Ly-1-q)) & 1 for q in range(Ly)] for idx in range(d)])

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

# Parameters
Lx, Ly = 4, 3
d = 2**Ly
bp = bit_patterns(Ly)
beta = 1.1778
n_e = Lx*(Ly-1) + (Lx-1)*Ly
avg_d = 2*n_e/(Lx*Ly)
gamma = 0.88/avg_d

print("="*65)
print(f"CHI=4 MECHANISME: {Lx}×{Ly} (d={d}), γ={gamma:.4f}, β={beta:.4f}")
print("="*65)

# Bouw gates
H1 = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
Hd = np.ones((d,d), dtype=complex)
for q in range(Ly):
    Hd *= H1[bp[:,q:q+1], bp[:,q:q+1].T]

def zzi_diag(g):
    diag = np.ones(d, dtype=complex)
    for y in range(Ly-1):
        diag *= np.exp(-1j*g*(1-2*bp[:,y].astype(float))*(1-2*bp[:,y+1].astype(float)))
    return diag

def zze_diag(g):
    iL = np.arange(d*d) // d
    iR = np.arange(d*d) % d
    diag = np.ones(d*d, dtype=complex)
    for y in range(Ly):
        diag *= np.exp(-1j*g*(1-2*bp[iL,y].astype(float))*(1-2*bp[iR,y].astype(float)))
    return diag

def rx_col(b):
    rx = Rx(2*b)
    R = np.ones((d,d), dtype=complex)
    for q in range(Ly):
        R *= rx[bp[:,q:q+1], bp[:,q:q+1].T]
    return R

# Track singular values na elke gate-laag
# Start met ZZ_obs op edge (0,0)-(1,0)
mpo = [np.eye(d, dtype=complex).reshape(1,d,d,1).copy() for _ in range(Lx)]
for col in [0, 1]:
    diag = (1 - 2*bp[:,0].astype(float)).astype(complex)
    mpo[col] = np.diag(diag).reshape(1,d,d,1)

# Build full gate sequence for p=1
zzi = zzi_diag(gamma)
zze = zze_diag(gamma)
rxd = rx_col(beta)

gate_seq = []
for x in range(Lx):
    gate_seq.append(('H', x, Hd))
for x in range(Lx):
    gate_seq.append(('ZZi', x, zzi))
for x in range(Lx-1):
    gate_seq.append(('ZZe', x, zze))
for x in range(Lx):
    gate_seq.append(('RX', x, rxd))

def apply_gate(mpo, gt, s, data, chi_max=256):
    if gt in ('H', 'RX'):
        Ud = data.conj().T
        mpo[s] = np.einsum('ij,ajkb,kl->ailb', Ud, mpo[s], data)
    elif gt == 'ZZi':
        cd = np.conj(data)
        mpo[s] = mpo[s] * cd[None,:,None,None] * data[None,None,:,None]
    elif gt == 'ZZe':
        cl, cr = mpo[s].shape[0], mpo[s+1].shape[3]
        Th = np.einsum('aijc,cklb->aijklb', mpo[s], mpo[s+1])
        cd = np.conj(data).reshape(d,d)
        dd = data.reshape(d,d)
        Th = Th * cd[None,:,None,:,None,None] * dd[None,None,:,None,:,None]
        mat = Th.reshape(cl*d*d, d*d*cr)
        U, S, V = svd(mat, full_matrices=False)
        Sa = np.abs(S)
        k = max(1, int(np.sum(Sa > 1e-12*Sa[0])))
        k = min(k, chi_max)
        mpo[s] = U[:,:k].reshape(cl,d,d,k)
        mpo[s+1] = (np.diag(S[:k]) @ V[:k,:]).reshape(k,d,d,cr)
    return mpo

# Evolve in REVERSE en track chi na elke gate
print("\nSinguliere waarden na elke gate-laag (reversed):")
print(f"  {'Gate':>6s}  {'Site':>4s}  {'Chi':>4s}  {'Sing. waarden (top-5)':>40s}")

for gt, s, data in reversed(gate_seq):
    mpo = apply_gate(mpo, gt, s, data)
    # Report max chi en singular values
    max_chi = max(W.shape[0] for W in mpo)
    
    # Get singular values at middle bond
    mid = Lx // 2
    if mid < len(mpo):
        W = mpo[mid]
        cl, _, _, cr = W.shape
        mat = W.reshape(cl*d, d*cr)
        _, sv, _ = svd(mat, full_matrices=False)
        sv_str = ", ".join([f"{s:.4f}" for s in sv[:5]])
    else:
        sv_str = "---"
    
    print(f"  {gt:>6s}  {s:>4d}  {max_chi:>4d}  {sv_str}")

# Final result
L = np.ones((1,), dtype=complex)
for W in mpo:
    L = np.einsum('a,ab->b', L, W[:,0,0,:])
zz_val = L[0].real
print(f"\n<ZZ> = {zz_val:.6f}")
print(f"Max chi bereikt: {max(W.shape[0] for W in mpo)}")

# Nu: ANALYSEER de 2-site gate SVD in detail
print("\n" + "="*65)
print("ANALYSE VAN DE INTER-KOLOM ZZ-GATE SVD")
print("="*65)

# De ZZ inter gate is diagonaal: diag[i*d+j] = Π_y exp(-iγ z_y(i) z_y(j))
# Reshape als d×d matrix en doe SVD
zze_mat = zze.reshape(d, d)
U, S, Vt = svd(zze_mat, full_matrices=False)
print(f"\nSVD van exp(-iγ Σ_y Z_y Z_y') als {d}×{d} matrix:")
for i, s in enumerate(S):
    print(f"  σ_{i}: {s:.8f}  ({s/S[0]*100:.2f}%)")

# De EFFECTIEVE rank bij truncatie:
print(f"\nEffectieve rank bij truncatie-drempel:")
for thresh in [0.01, 0.001, 0.0001, 1e-6, 1e-10]:
    eff_rank = int(np.sum(S > thresh * S[0]))
    print(f"  ε={thresh:.0e}: rank={eff_rank}")

# Ontleed de singuliere waarden per component
# De gate factoriseert als: Π_y [cos(γ) I + (-i sin(γ)) Z_y ⊗ Z_y]
# Bij d=8 (Ly=3), zijn er 2^3=8 unieke diagonaal-elementen
print(f"\nDiagonaal van de inter-kolom gate ({d} unieke waarden):")
z = 1 - 2*bp.astype(float)
for i in range(d):
    for j in range(d):
        val = zze[i*d+j]
        zz_pattern = [z[i,y]*z[j,y] for y in range(Ly)]
        n_same = sum(1 for p in zz_pattern if p > 0)
        n_diff = sum(1 for p in zz_pattern if p < 0)
        if j == 0:  # print een paar voorbeelden
            print(f"  [{i},{j}]: {val:.6f}  (same={n_same}, diff={n_diff})")

# De sleutel: hoeveel UNIEKE diagonaalwaarden zijn er?
diag_vals = np.array([zze[i*d+j] for i in range(d) for j in range(d)])
unique_vals = np.unique(np.round(diag_vals, 10))
print(f"\nAantal unieke diagonaalwaarden: {len(unique_vals)}")
for v in unique_vals:
    count = np.sum(np.abs(diag_vals - v) < 1e-8)
    print(f"  {v:.6f}: {count}× (van {d*d})")

# De SVD rank is PRECIES het aantal unieke waarden
print(f"\n→ SVD rank = {np.sum(S > 1e-10)} = aantal unieke diag-waarden = {len(unique_vals)}")
print(f"  Dit verklaart chi=4 bij p=1: er zijn maar {len(unique_vals)} unieke waarden\!")
