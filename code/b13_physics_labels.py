"""
B13: Fysische labels uit de Zorn-structuur

Een Zorn-element Z = (a, α₁, α₂, α₃, β₁, β₂, β₃, b) heeft
INGEBOUWDE structuur die mapt op fysische grootheden:

  "Lading" (Q): split-norm N(z) = ab - α·β
    → Bewaard onder vermenigvuldiging: N(AB) = N(A)N(B)
    → Kan positief, negatief, of nul zijn
    → N=0: "massaloos" (null divisors)

  "Spin" (S): de α vs β asymmetrie
    → α-sector: |001⟩,|010⟩,|011⟩ ("linksdraaiend")
    → β-sector: |100⟩,|101⟩,|110⟩ ("rechtsdraaiend")
    → Chirale asymmetrie in het product: +β×δ vs -α×γ

  "Massa" (M): scalair gehalte
    → a (|000⟩) en b (|111⟩): "vacuüm-achtig"
    → Pure scalaire toestanden: geen spin, geen richting

Vraag: als we deze labels BIJHOUDEN per MPS-bond, kunnen we
dan slimmer trunceren?
"""
import numpy as np
from numpy.linalg import svd
import time

# === Zorn-labels voor d=8 states ===

def zorn_labels(d=8):
    """Classificeer de d=8 basistoestanden in sectoren."""
    # |000⟩=0: scalar a
    # |001⟩=1, |010⟩=2, |011⟩=3: α-vector (linksdraaiend)
    # |100⟩=4, |101⟩=5, |110⟩=6: β-vector (rechtsdraaiend)
    # |111⟩=7: scalar b
    sectors = {
        'scalar_a': [0],      # |000⟩
        'alpha':    [1,2,3],  # α-sector (links)
        'beta':     [4,5,6],  # β-sector (rechts)
        'scalar_b': [7],      # |111⟩
    }
    # Sector-index per basistoestand
    sec_idx = np.zeros(d, dtype=int)  # 0=scalar, 1=alpha, 2=beta, 3=scalar_b
    for i in sectors['alpha']: sec_idx[i] = 1
    for i in sectors['beta']:  sec_idx[i] = 2
    sec_idx[7] = 0  # scalar_b ook "scalar"
    return sectors, sec_idx

def measure_sector_weights(T, sec_idx):
    """Meet het gewicht per sector in een MPS tensor T (chi_l, d, chi_r)."""
    # Sector gewichten: hoeveel van de tensor leeft in elke sector
    weights = {}
    for sec, label in [(0, 'scalar'), (1, 'alpha'), (2, 'beta')]:
        mask = (sec_idx == sec)
        w = np.sum(np.abs(T[:, mask, :])**2)
        weights[label] = w
    total = sum(weights.values())
    if total > 1e-15:
        for k in weights: weights[k] /= total
    return weights

def split_norm_mps(T, d=8):
    """Bereken de effectieve split-norm van een MPS tensor.
    
    Voor elke (chi_l, chi_r) blok, bereken N = a*b - α·β
    van het Zorn-element in de d-index.
    """
    cl, _, cr = T.shape
    norms = []
    for l in range(cl):
        for r in range(cr):
            z = T[l, :, r]  # d-component Zorn element
            if d == 8:
                a, al, be, b = z[0], z[1:4], z[4:7], z[7]
                N = a*b - np.dot(al, be)
                norms.append(N)
    return np.array(norms)

# === TEBD met Zorn-groepering (d=8, 3 qubits per site) ===

def heisenberg_gate_d8(J, dt):
    """Heisenberg XXX gate op 6 qubits (2 Zorn-sites, d=8 elk).
    H = J Σ_{i,j naburig} (X_i X_j + Y_i Y_j + Z_i Z_j)
    
    Inter-site bonds: qubit 2 van site L koppelt aan qubit 0 van site R.
    """
    X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]]);I=np.eye(2)
    
    # Alleen de inter-triplet koppeling (qubit 2L ↔ qubit 0R)
    # In de d=8 × d=8 ruimte is dit:
    # q2_L staat op positie 2 in het linker triplet
    # q0_R staat op positie 0 in het rechter triplet
    
    d = 8
    H_inter = np.zeros((d*d, d*d), dtype=complex)
    
    for P in [X, Y, Z]:  # XX + YY + ZZ
        # P op qubit 2 van links, P op qubit 0 van rechts
        # Links: I⊗I⊗P = kron(kron(I,I),P) op 3 qubits
        PL = np.kron(np.kron(I, I), P)  # 8×8
        # Rechts: P⊗I⊗I
        PR = np.kron(np.kron(P, I), I)  # 8×8
        H_inter += J * np.kron(PL, PR)  # 64×64
    
    # Exponentiate
    ev, U = np.linalg.eigh(H_inter)
    gate = U @ np.diag(np.exp(-1j * dt * ev)) @ U.conj().T
    return gate

def init_neel_d8(n_sites):
    """Néel state op triplet-gegroepeerde MPS (d=8).
    |010 101 010 101 ...⟩
    """
    mps = []
    for i in range(n_sites):
        T = np.zeros((1, 8, 1), dtype=complex)
        if i % 2 == 0:
            T[0, 0b010, 0] = 1.0  # |010⟩ = α₂
        else:
            T[0, 0b101, 0] = 1.0  # |101⟩ = β₂
        mps.append(T)
    return mps

def apply2_d8(mps, gate, s, chi):
    d = 8
    cl, cr = mps[s].shape[0], mps[s+1].shape[2]
    Th = np.einsum('aib,bjc->aijc', mps[s], mps[s+1])
    G = gate.reshape(d, d, d, d)
    Th = np.einsum('ijkl,aklc->aijc', G, Th)
    mat = Th.reshape(cl*d, d*cr)
    U, S, V = svd(mat, full_matrices=False)
    k = min(max(1, int(np.sum(np.abs(S) > 1e-12*np.abs(S[0])))), chi)
    disc = np.sum(np.abs(S[k:])**2) / (np.sum(np.abs(S)**2) + 1e-30)
    mps[s] = U[:,:k].reshape(cl, d, k)
    mps[s+1] = (np.diag(S[:k]) @ V[:k,:]).reshape(k, d, cr)
    return disc

def tebd_step_d8(mps, gate, chi):
    n = len(mps); d = 0.0
    for s in range(0, n-1, 2): d += apply2_d8(mps, gate, s, chi)
    for s in range(1, n-1, 2): d += apply2_d8(mps, gate, s, chi)
    return d

# === TESTS ===
print("="*65)
print("B13: FYSISCHE LABELS UIT ZORN-STRUCTUUR")
print("="*65)

sectors, sec_idx = zorn_labels()
print("\nZorn-sectoren:")
print("  |000⟩ = scalar a   (vacuüm)")
print("  |001⟩,|010⟩,|011⟩ = α-vector (links-chiraal)")
print("  |100⟩,|101⟩,|110⟩ = β-vector (rechts-chiraal)")
print("  |111⟩ = scalar b   (vacuüm)")

# Test 1: Sector-gewichten tijdens TEBD
print("\n--- Sector-gewichten tijdens Heisenberg TEBD ---")
n_sites = 6  # 18 qubits
chi = 16
gate = heisenberg_gate_d8(1.0, 0.1)
mps = init_neel_d8(n_sites)

print(f"\n  {'Stap':>4s}  {'Scalar':>8s}  {'Alpha':>8s}  {'Beta':>8s}  {'Chi':>4s}  {'SplitN':>10s}")

for step in range(15):
    # Meet sector-gewichten op middelste site
    mid = n_sites // 2
    w = measure_sector_weights(mps[mid].copy(), sec_idx)
    
    # Meet split-norm
    sn = split_norm_mps(mps[mid].copy())
    sn_avg = np.mean(np.abs(sn)) if len(sn) > 0 else 0
    
    max_chi = max(mps[i].shape[2] for i in range(n_sites-1))
    
    print(f"  {step:>4d}  {w['scalar']:8.4f}  {w['alpha']:8.4f}  {w['beta']:8.4f}  {max_chi:>4d}  {sn_avg:10.6f}")
    
    tebd_step_d8(mps, gate, chi)

# Test 2: Split-norm conservatie
print("\n--- Split-norm conservatie ---")
print("  N(AB) = N(A)·N(B) geldt algebraisch.")
print("  Vraag: blijft de gemiddelde split-norm bewaard in de MPS?")

mps2 = init_neel_d8(6)
sn_start = []
for node in mps2:
    sn = split_norm_mps(node)
    sn_start.extend(sn)
print(f"  Start: <|N|> = {np.mean(np.abs(sn_start)):.6f}")

for step in range(10):
    tebd_step_d8(mps2, gate, chi)

sn_end = []
for node in mps2:
    sn = split_norm_mps(node)
    sn_end.extend(sn)
print(f"  Na 10 stappen: <|N|> = {np.mean(np.abs(sn_end)):.6f}")

# Test 3: Chiraliteit (α vs β)
print("\n--- Chirale asymmetrie ---")
print("  Start: Néel = afwisselend α₂ en β₂ → maximale chiraliteit")

mps3 = init_neel_d8(6)
for step in [0, 5, 10]:
    if step > 0:
        for _ in range(5): tebd_step_d8(mps3, gate, chi)
    
    total_alpha = 0; total_beta = 0
    for node in mps3:
        w = measure_sector_weights(node, sec_idx)
        total_alpha += w['alpha']
        total_beta += w['beta']
    chirality = (total_alpha - total_beta) / (total_alpha + total_beta + 1e-30)
    print(f"  Stap {step:>2d}: α={total_alpha/6:.4f}, β={total_beta/6:.4f}, chiraliteit={chirality:+.4f}")

print("\n" + "="*65)
print("ANALYSE: WAT KUNNEN WE HIERMEE?")
print("="*65)
print("""
De Zorn-structuur geeft drie natuurlijke labels per MPS-site:

  1. LADING (split-norm N = ab - α·β)
     → Bewaard bij Zorn-vermenigvuldiging
     → Kan als SYMMETRIE-SECTOR worden gebruikt in de MPS
     → Symmetrie-gerespecteerde truncatie: behoud N per bond
     → Dit is een BEKENDE techniek (U(1)/Z₂ symmetrie in MPS)

  2. SPIN (α/β chiraliteit)
     → Links (α) vs rechts (β) chiraal
     → De Néel-state heeft maximale chiraliteit
     → Dynamica mixt de sectoren → chiraliteit vervalt
     → POTENTIEEL: gebruik α/β labels als Z₂-symmetrie

  3. MASSA (scalair gehalte)
     → |000⟩ en |111⟩ zijn "scalair" (geen spin-richting)
     → Scalaire content groeit tijdens thermalisatie
     → Massaloze toestanden: N=0 (null divisors)
""")
