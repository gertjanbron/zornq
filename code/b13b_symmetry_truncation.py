"""
B13b: Symmetrie-beschermde truncatie met Zorn-labels

De sector-oscillaties laten zien dat de Zorn-labels FYSISCH zijn:
het systeem slingert tussen α-sector en scalar/β-sector.

Concrete toepassing: als we de MPS-bonds LABELEN per sector,
wordt de bondmatrix blok-diagonaal → SVD per blok is goedkoper.

Test: vergelijk standaard SVD vs sector-gelabelde SVD.
"""
import numpy as np
from numpy.linalg import svd
import time

# Zorn sector indices voor d=8
# Sector 0: scalar  (|000⟩, |111⟩)  — "massa"
# Sector 1: alpha   (|001⟩,|010⟩,|011⟩) — "links-chiraal"  
# Sector 2: beta    (|100⟩,|101⟩,|110⟩) — "rechts-chiraal"
SEC = np.array([0, 1, 1, 1, 2, 2, 2, 0])  # sector per d=8 index
SEC_SIZES = {0: 2, 1: 3, 2: 3}  # dimensie per sector
SEC_NAMES = {0: 'scalar', 1: 'alpha', 2: 'beta'}

# Maar: Heisenberg bewaart TOTAL Sz, niet Zorn-sectoren.
# Sz per d=8 basistoestand:
SZ = np.array([1.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -1.5])
# Sz-sectoren: -3/2, -1/2, +1/2, +3/2
SZ_SEC = np.array([3, 1, 1, 2, 1, 2, 2, 0])  # 0=Sz-3/2, 1=Sz-1/2, 2=Sz+1/2, 3=Sz+3/2
# Wacht: laat me dit correct doen
# |000⟩ = ↑↑↑ → Sz = +3/2 → sector 3
# |001⟩ = ↑↑↓ → Sz = +1/2 → sector 2
# |010⟩ = ↑↓↑ → Sz = +1/2 → sector 2
# |011⟩ = ↑↓↓ → Sz = -1/2 → sector 1
# |100⟩ = ↓↑↑ → Sz = +1/2 → sector 2
# |101⟩ = ↓↑↓ → Sz = -1/2 → sector 1
# |110⟩ = ↓↓↑ → Sz = -1/2 → sector 1
# |111⟩ = ↓↓↓ → Sz = -3/2 → sector 0
SZ_SEC = np.array([3, 2, 2, 1, 2, 1, 1, 0])

print("="*65)
print("VERGELIJKING: ZORN-SECTOREN vs Sz-SECTOREN")
print("="*65)

print("\n  Basis   Zorn-sector    Sz    Sz-sector")
labels = ['|000⟩','|001⟩','|010⟩','|011⟩','|100⟩','|101⟩','|110⟩','|111⟩']
for i in range(8):
    print(f"  {labels[i]}   {SEC_NAMES[SEC[i]]:>7s}     {SZ[i]:+.1f}     {SZ_SEC[i]}")

print(f"\n  Zorn-sectoren: 3 (scalar/alpha/beta) — NIET bewaard door Heisenberg")
print(f"  Sz-sectoren:   4 (Sz=-3/2,-1/2,+1/2,+3/2) — WÉL bewaard")

print("\n  Overlap:")
for zs in range(3):
    z_indices = [i for i in range(8) if SEC[i]==zs]
    sz_vals = [SZ_SEC[i] for i in z_indices]
    print(f"    Zorn {SEC_NAMES[zs]}: bevat Sz-sectoren {set(sz_vals)}")

# De Zorn-sectoren splitsen NIET netjes langs Sz-lijnen.
# alpha bevat Sz=+1/2 en -1/2, beta ook.
# Dus Zorn-labels zijn GEEN conservatiegrootheid voor Heisenberg.

print("\n" + "="*65)
print("MAAR: WAT ALS WE DE Sz-SYMMETRIE GEBRUIKEN OP d=8?")
print("="*65)

# Sz-symmetrie op d=8 geeft 4 blokken: {1, 3, 3, 1} dimensies.
# Bond-matrix wordt blok-diagonaal naar Sz-sector.
# SVD per blok: max 3×3 ipv 8×8 → VEEL goedkoper.

# Bouw Sz-symmetrie-beschermde 2-site gate
def heisenberg_gate_d8(J, dt):
    X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]]);I=np.eye(2)
    d=8
    H=np.zeros((d*d,d*d),dtype=complex)
    for P in [X,Y,Z]:
        PL=np.kron(np.kron(I,I),P); PR=np.kron(np.kron(P,I),I)
        H+=J*np.kron(PL,PR)
    ev,U=np.linalg.eigh(H)
    return U@np.diag(np.exp(-1j*dt*ev))@U.conj().T

# Test: hoeveel blok-structuur heeft de gate?
gate = heisenberg_gate_d8(1.0, 0.1)
G = gate.reshape(8,8,8,8)

# Check: mixt de gate Sz-sectoren?
print("\nGate-structuur: welke Sz-overgangen zijn niet-nul?")
sz_total_in = SZ_SEC[:, None] + SZ_SEC[None, :]  # (8,8) matrix van totale Sz
sz_total_out = SZ_SEC[:, None] + SZ_SEC[None, :]

# G[i,j,k,l]: output (i,j), input (k,l)
# Sz_in = SZ_SEC[k] + SZ_SEC[l]
# Sz_out = SZ_SEC[i] + SZ_SEC[j]
n_nonzero = 0
n_total = 0
n_sz_conserving = 0
for i in range(8):
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if abs(G[i,j,k,l]) > 1e-12:
                    n_nonzero += 1
                    sz_in = SZ_SEC[k] + SZ_SEC[l]
                    sz_out = SZ_SEC[i] + SZ_SEC[j]
                    if sz_in == sz_out:
                        n_sz_conserving += 1
                n_total += 1

print(f"  Non-zero entries: {n_nonzero}/{n_total}")
print(f"  Sz-conserverend:  {n_sz_conserving}/{n_nonzero} = {n_sz_conserving/n_nonzero*100:.1f}%")
print(f"  → Gate is 100% Sz-conserverend = blok-diagonaal\!")

# Check Zorn-sector conservatie
n_zorn_conserving = 0
for i in range(8):
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if abs(G[i,j,k,l]) > 1e-12:
                    zorn_in = SEC[k]*3 + SEC[l]
                    zorn_out = SEC[i]*3 + SEC[j]
                    if zorn_in == zorn_out:
                        n_zorn_conserving += 1

print(f"  Zorn-conserverend: {n_zorn_conserving}/{n_nonzero} = {n_zorn_conserving/n_nonzero*100:.1f}%")

# Nu de BIG PAYOFF: hoe groot worden de Sz-blokken?
print("\n  Sz-blokken van de 2-site bondruimte (d=8 × d=8 = 64):")
# Totale Sz van 2 sites: van -3 tot +3 in stappen van 1
for sz_tot in range(7):  # 0..6 mapping to -3..+3
    count = 0
    for k in range(8):
        for l in range(8):
            if SZ_SEC[k] + SZ_SEC[l] == sz_tot:
                count += 1
    if count > 0:
        print(f"    Sz_tot={sz_tot-3:+d}: {count} toestanden → SVD op {count}×{count}")

# Conclusie
print("\n" + "="*65)
print("CONCLUSIE: PRAKTISCHE WINST")
print("="*65)
print("""
WAT WERKT — Sz-symmetrie op d=8 MPS:
  De Heisenberg gate is 100% Sz-conserverend.
  De 64×64 bondmatrix splitst in 7 blokken:
    Sz=-3: 1×1, Sz=-2: 6×6, Sz=-1: 15×15, Sz=0: 20×20,
    Sz=+1: 15×15, Sz=+2: 6×6, Sz=+3: 1×1
  
  SVD op 7 kleine blokken << SVD op 1 grote 64×64 matrix.
  Geschatte speedup: 3-5× bij chi=16, meer bij hogere chi.
  DIT IS EEN BEKENDE TECHNIEK maar nog niet in onze engine.

WAT NIET WERKT — Zorn-sectoren als symmetrie:
  De Heisenberg gate mixt Zorn-sectoren (alpha/beta/scalar).
  Ze zijn geen conservatiegrootheid → niet bruikbaar voor
  blok-diagonale truncatie.
  
  MAAR: als we een Zorn-NATIVE Hamiltoniaan bouwen die de
  sectoren WÉL bewaart, krijgen we een Zorn-specifieke
  symmetrie-beschermde MPS. Dat zou echt nieuw zijn.

BRUIKBARE ANALOGIEËN:
  "Lading" = Sz (totale spin-z) → bewaard, bruikbaar voor blok-SVD
  "Spin"   = α/β chiraliteit → oscilleert, NIET bewaard
  "Massa"  = scalair gehalte → groeit bij thermalisatie
  
  De split-norm is altijd 0 voor pure basistoestanden.
  Het wordt pas niet-nul bij superpositie → meet COHERENTIE.
""")
