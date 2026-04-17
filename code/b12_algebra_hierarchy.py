"""
B12: Algebraïsche basis voor de kopplingshiërarchie

Kernvraag: produceert de Zorn-vermenigvuldiging NATUURLIJK
verschillende kopplingssterkte op verschillende dimensionale schalen?

De Zorn product Z1 * Z2 met Z = (a, α, β, b):
  scalar:  a*c + α·δ
  α-rij:   a*γ + d*α + β×δ
  β-rij:   c*β + b*δ - α×γ
  scalar:  β·γ + b*d

We ontleden dit in vier bijdragen:
  0D: scalar×scalar  (a*c, b*d)
  1D: scalar×vector  (a*γ, d*α, c*β, b*δ)  
  2D: cross products (β×δ, -α×γ)
  3D: dot products   (α·δ, β·γ)
"""
import numpy as np

# === Zorn algebra ===
def zmul(A, B):
    a, al, be, b = A[0], A[1:4], A[4:7], A[7]
    c, ga, de, d = B[0], B[1:4], B[4:7], B[7]
    return np.array([a*c + al@de,
                     *(a*ga + d*al + np.cross(be, de)),
                     *(c*be + b*de - np.cross(al, ga)),
                     be@ga + b*d])

# Ontleed het product in 4 dimensionale bijdragen
def zmul_0d(A, B):
    """Alleen scalar×scalar termen"""
    a, b = A[0], A[7]
    c, d = B[0], B[7]
    return np.array([a*c, 0,0,0, 0,0,0, b*d])

def zmul_1d(A, B):
    """Alleen scalar×vector termen"""
    a, al, be, b = A[0], A[1:4], A[4:7], A[7]
    c, ga, de, d = B[0], B[1:4], B[4:7], B[7]
    return np.array([0,
                     *(a*ga + d*al),
                     *(c*be + b*de),
                     0])

def zmul_2d(A, B):
    """Alleen cross product termen"""
    al, be = A[1:4], A[4:7]
    ga, de = B[1:4], B[4:7]
    return np.array([0,
                     *np.cross(be, de),
                     *(-np.cross(al, ga)),
                     0])

def zmul_3d(A, B):
    """Alleen dot product termen"""
    al, be = A[1:4], A[4:7]
    ga, de = B[1:4], B[4:7]
    return np.array([al@de, 0,0,0, 0,0,0, be@ga])

print("="*60)
print("DEEL 1: STRUCTUUR VAN HET ZORN-PRODUCT")
print("="*60)

# Bouw de volledige transfer matrix voor bilineair product
T_full = np.zeros((8, 64))
T_0d = np.zeros((8, 64))
T_1d = np.zeros((8, 64))
T_2d = np.zeros((8, 64))
T_3d = np.zeros((8, 64))

for j in range(8):
    for k in range(8):
        ej = np.zeros(8); ej[j] = 1
        ek = np.zeros(8); ek[k] = 1
        T_full[:, j*8+k] = zmul(ej, ek)
        T_0d[:, j*8+k] = zmul_0d(ej, ek)
        T_1d[:, j*8+k] = zmul_1d(ej, ek)
        T_2d[:, j*8+k] = zmul_2d(ej, ek)
        T_3d[:, j*8+k] = zmul_3d(ej, ek)

# Verificatie: de som moet het totaal geven
diff = np.max(np.abs(T_full - (T_0d + T_1d + T_2d + T_3d)))
print(f"\nVerificatie decompositie: max|T - (T0+T1+T2+T3)| = {diff:.2e}")

# Frobenius norm = totale "kopplingssterkte" per bijdrage
norm_full = np.linalg.norm(T_full, 'fro')
norm_0d = np.linalg.norm(T_0d, 'fro')
norm_1d = np.linalg.norm(T_1d, 'fro')
norm_2d = np.linalg.norm(T_2d, 'fro')
norm_3d = np.linalg.norm(T_3d, 'fro')

print(f"\nFrobenius normen (kopplingssterkte):")
print(f"  0D (scalar×scalar): {norm_0d:.4f}  ({norm_0d**2/norm_full**2*100:.1f}%)")
print(f"  1D (scalar×vector): {norm_1d:.4f}  ({norm_1d**2/norm_full**2*100:.1f}%)")
print(f"  2D (cross product): {norm_2d:.4f}  ({norm_2d**2/norm_full**2*100:.1f}%)")
print(f"  3D (dot product):   {norm_3d:.4f}  ({norm_3d**2/norm_full**2*100:.1f}%)")
print(f"  Totaal:             {norm_full:.4f}")

# Rank per bijdrage
for name, T in [("0D", T_0d), ("1D", T_1d), ("2D", T_2d), ("3D", T_3d), ("Full", T_full)]:
    r = np.linalg.matrix_rank(T, tol=1e-10)
    print(f"  Rank {name}: {r}/64")

print("\n" + "="*60)
print("DEEL 2: EFFECTIEVE KOPPELING TUSSEN QUBIT-DEELRUIMTEN")
print("="*60)

# De 8 Zorn-componenten corresponderen met 3 qubits:
# q0 q1 q2 -> index: |000>=0(a), |001>=1(α1), |010>=2(α2), |011>=3(α3),
#                     |100>=4(β1), |101>=5(β2), |110>=6(β3), |111>=7(b)
# 
# Structuur:
# q0=0: scalair + α-sector (a, α1, α2, α3)
# q0=1: β-sector + scalair (β1, β2, β3, b)
# q1,q2: selecteren component binnen sector

# Maak de 64x64 coupling matrix: C[i,j] = hoe sterk koppelt
# input-paar (j//8, j%8) aan output i via het Zorn-product
# Maar nuttiger: de effectieve 2-lichaam interactie matrix

# Bereken de effectieve Hamiltoniaan die het Zorn-product induceert
# op de 6-qubit ruimte (3 qubits per Zorn element)
# H_eff[a,b] = sum_out |T[out, a*8+b]|^2

print("\nEffectieve koppelingsmatrix ||zmul(e_i, e_j)||^2:")
C = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        C[i, j] = np.sum(T_full[:, i*8+j]**2)

# Groepeer per qubit-sector
labels = ['a=|000>', 'α1=|001>', 'α2=|010>', 'α3=|011>',
          'β1=|100>', 'β2=|101>', 'β3=|110>', 'b=|111>']

print("\n         ", "  ".join([f"{l[:3]:>5s}" for l in labels]))
for i in range(8):
    vals = "  ".join([f"{C[i,j]:5.2f}" for j in range(8)])
    print(f"  {labels[i]:>10s}: {vals}")

# Gemiddelde koppeling BINNEN en TUSSEN sectoren
# Sector A = {a, α1, α2, α3} (q0=0)
# Sector B = {β1, β2, β3, b} (q0=1)
secA = [0,1,2,3]
secB = [4,5,6,7]

intra_AA = np.mean([C[i,j] for i in secA for j in secA])
intra_BB = np.mean([C[i,j] for i in secB for j in secB])
cross_AB = np.mean([C[i,j] for i in secA for j in secB])
cross_BA = np.mean([C[i,j] for i in secB for j in secA])

print(f"\nGemiddelde koppeling per sector:")
print(f"  Intra A×A (q0=0, q0=0): {intra_AA:.4f}")
print(f"  Intra B×B (q0=1, q0=1): {intra_BB:.4f}")
print(f"  Cross A×B (q0=0, q0=1): {cross_AB:.4f}")
print(f"  Cross B×A (q0=1, q0=0): {cross_BA:.4f}")

# Fijnere analyse: binnen vector-sector vs scalar-vector 
vec_indices = [1,2,3]  # α-sector
bvec_indices = [4,5,6]  # β-sector
scalar_indices = [0, 7]  # a, b

# Cross product koppeling: β×δ koppelt β_i met δ_j (antisymmetrisch)
print(f"\nCross-product koppeling (β×δ term in α-output):")
C_cross = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        ei = np.zeros(8); ei[4+i] = 1  # β_i
        ej = np.zeros(8); ej[4+j] = 1  # δ_j
        result = zmul_2d(ei, ej)
        C_cross[i,j] = np.linalg.norm(result)

print("  β\\δ    δ1    δ2    δ3")
for i in range(3):
    print(f"  β{i+1}  {C_cross[i,0]:.3f}  {C_cross[i,1]:.3f}  {C_cross[i,2]:.3f}")

print(f"\n  Niet-nul entries: {np.count_nonzero(C_cross > 0.01)}/9")
print(f"  Patroon: elk β_i koppelt aan precies 2 δ_j (Levi-Civita)")

print("\n" + "="*60)
print("DEEL 3: DIMENSIONALE HIËRARCHIE IN OPERATIE-RANKS")
print("="*60)

# De 7 operaties hebben sterk verschillende ranks
# Dit is een bekende eigenschap van ZornQ
FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
def pdec(A, d):
    t = FANO[d]; comp = [x for x in range(1,8) if x not in t]
    return A[[0]+list(t)+comp]

def zconj(A): return np.array([A[7], *(-A[4:7]), *(-A[1:4]), A[0]])
def zinv(A):
    c = zconj(A); n = A[0]*A[7] - A[1:4]@A[4:7]
    return c/n if abs(n) > 1e-15 else c
def zhodge(A): return np.array([A[0], A[2],A[3],A[1], A[5],A[6],A[4], A[7]])
def zassoc(A,B,C): return zmul(zmul(A,B),C) - zmul(A,zmul(B,C))
def zjordan(A,B): return zmul(zmul(A,B),A)

rng = np.random.default_rng(42)
op_ranks = {}

for op in range(7):
    all_rows = []
    for d in range(7):
        if op == 0:  # bilineair
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:, j*8+k] = zmul(pdec(ej,d), pdec(ek,d))
            all_rows.append(T)
        elif op == 1:  # (A+B)xC
            C = pdec(rng.standard_normal(8), d)
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:, j*8+k] = zmul(pdec(ej,d)+pdec(ek,d), C)
            all_rows.append(T)
        elif op == 2:  # (A-B)xC
            C = pdec(rng.standard_normal(8), d)
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:, j*8+k] = zmul(pdec(ej,d)-pdec(ek,d), C)
            all_rows.append(T)
        elif op == 3:  # inv(A)xB
            Z0 = rng.standard_normal(8); eps = 1e-7
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ek=np.zeros(8);ek[k]=1
                    zp=Z0.copy();zp[j]+=eps;zm=Z0.copy();zm[j]-=eps
                    T[:,j*8+k]=(zmul(zinv(pdec(zp,d)),pdec(ek,d))-zmul(zinv(pdec(zm,d)),pdec(ek,d)))/(2*eps)
            all_rows.append(T)
        elif op == 4:  # Hodge
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:,j*8+k] = zmul(zhodge(pdec(ej,d)), pdec(ek,d))
            all_rows.append(T)
        elif op == 5:  # associator
            C = pdec(rng.standard_normal(8), d)
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:,j*8+k] = zassoc(pdec(ej,d), pdec(ek,d), C)
            all_rows.append(T)
        elif op == 6:  # Jordan ABA
            T = np.zeros((8, 64))
            for j in range(8):
                for k in range(8):
                    ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                    T[:,j*8+k] = zjordan(pdec(ej,d), pdec(ek,d))
            all_rows.append(T)
    
    Tall = np.vstack(all_rows)
    # Verwijder NaN/Inf rijen
    good = np.all(np.isfinite(Tall), axis=1) & (np.linalg.norm(Tall, axis=1) > 1e-15)
    Tall = Tall[good]
    r = np.linalg.matrix_rank(Tall, tol=1e-10)
    fnorm = np.linalg.norm(Tall, 'fro')
    op_ranks[op] = (r, fnorm)

SHORT = {0:'×', 1:'+', 2:'-', 3:'÷', 4:'H', 5:'[·]', 6:'ABA'}
print("\nRank en sterkte per operatie (enkele Fano-decompositie):")
print(f"  {'Op':>5s}  {'Rank':>4s}  {'||T||_F':>8s}  Interpretatie")
for op in range(7):
    r, fn = op_ranks[op]
    interp = {0:"bilineair product", 1:"additie", 2:"subtractie",
              3:"divisie (inversie)", 4:"Hodge dualiteit",
              5:"associator [A,B,C]", 6:"Jordan product ABA"}
    print(f"  {SHORT[op]:>5s}  {r:>4d}  {fn:>8.2f}  {interp[op]}")

print("\n" + "="*60)
print("DEEL 4: INTRA-TRIPLET vs INTER-TRIPLET KOPPELING")
print("="*60)

# In een MPS van Zorn-elementen:
# - Intra-triplet: 3 qubits BINNEN één Zorn element
# - Inter-triplet: koppeling TUSSEN twee Zorn elementen via het product
#
# De intra-triplet structuur is volledig bepaald door de 8D lokale ruimte.
# Elke lokale operatie op d=8 is GRATIS (geen bond dimension nodig).
#
# De inter-triplet koppeling gaat via de transfer matrix T (8×64).
# De SVD van T vertelt ons de effectieve koppelingsstructuur.

U, S, Vt = np.linalg.svd(T_full, full_matrices=False)
print(f"\nSinguliere waarden van het bilineaire Zorn product (8×64):")
for i, s in enumerate(S):
    print(f"  σ_{i}: {s:.6f}  ({s/S[0]*100:.1f}%)")

print(f"\n  Conditiegetal: {S[0]/S[-1]:.4f}")
print(f"  Rank: {np.sum(S > 1e-10)}")

# Ontleed de SVD per dimensionale bijdrage
for name, Tpart in [("0D scalar×scalar", T_0d), ("1D scalar×vector", T_1d), 
                     ("2D cross product", T_2d), ("3D dot product", T_3d)]:
    _, Sp, _ = np.linalg.svd(Tpart, full_matrices=False)
    Sp = Sp[Sp > 1e-10]
    print(f"\n  {name}:")
    print(f"    Rank: {len(Sp)}")
    print(f"    Singuliere waarden: {', '.join([f'{s:.4f}' for s in Sp])}")

print("\n" + "="*60)
print("DEEL 5: CHIRALE ASYMMETRIE")
print("="*60)

# Het Zorn-product heeft een cruciale asymmetrie:
# α-rij: β×δ (PLUS cross product)
# β-rij: -α×γ (MINUS cross product)
#
# Dit is de split-octonionische chirality.
# Hoe beïnvloedt dit de koppelingsstructuur?

# Test: vermenigvuldig random vectoren en meet de α vs β output
N = 10000
rng = np.random.default_rng(123)
alpha_norms = []
beta_norms = []
scalar_norms = []

for _ in range(N):
    A = rng.standard_normal(8)
    B = rng.standard_normal(8)
    result = zmul(A, B)
    scalar_norms.append(result[0]**2 + result[7]**2)
    alpha_norms.append(np.sum(result[1:4]**2))
    beta_norms.append(np.sum(result[4:7]**2))

print(f"\nGemiddelde output-energie (10000 random producten):")
print(f"  Scalair (a,b):   {np.mean(scalar_norms):.4f}")
print(f"  α-sector (1-3):  {np.mean(alpha_norms):.4f}")
print(f"  β-sector (4-6):  {np.mean(beta_norms):.4f}")
print(f"  α/β ratio:       {np.mean(alpha_norms)/np.mean(beta_norms):.6f}")
print(f"  → Chirality breekt NIET de α/β symmetrie op ensemble-niveau")

# Maar meet nu de CORRELATIESTRUCTUUR
# Hoeveel van de α-output hangt af van de cross product?
alpha_from_cross = []
alpha_from_linear = []
beta_from_cross = []
beta_from_linear = []

for _ in range(N):
    A = rng.standard_normal(8)
    B = rng.standard_normal(8)
    r_1d = zmul_1d(A, B)
    r_2d = zmul_2d(A, B)
    alpha_from_cross.append(np.sum(r_2d[1:4]**2))
    alpha_from_linear.append(np.sum(r_1d[1:4]**2))
    beta_from_cross.append(np.sum(r_2d[4:7]**2))
    beta_from_linear.append(np.sum(r_1d[4:7]**2))

print(f"\nBijdrage van cross product vs lineair aan vector-output:")
cr_a = np.mean(alpha_from_cross)
li_a = np.mean(alpha_from_linear)
cr_b = np.mean(beta_from_cross)
li_b = np.mean(beta_from_linear)
print(f"  α-sector: cross={cr_a:.4f}, lineair={li_a:.4f}, ratio={cr_a/li_a:.4f}")
print(f"  β-sector: cross={cr_b:.4f}, lineair={li_b:.4f}, ratio={cr_b/li_b:.4f}")

print("\n" + "="*60)
print("DEEL 6: DE KERNVRAAG — NATUURLIJKE HIËRARCHIE?")
print("="*60)

# De vraag: als we de Zorn-productstructuur interpreteren als een
# koppeling in een tensornetwerk, produceert dat AUTOMATISCH
# J_1D >> J_2D >> J_3D?
#
# Hypothese: 
# - Cross products (2D-achtig) creëren sterke, gestructureerde koppeling
# - Dot products (3D-achtig) creëren zwakke, uniforme koppeling
# - Scalair producten (0D-achtig) zijn de zwakste
#
# Test: bereken de "effectieve koppelingssterkte" op elke schaal

# Methode 1: Tel het aantal non-triviale termen
print("\nMethode 1: Telkracht (# non-triviale termen)")
nz_0d = np.count_nonzero(np.abs(T_0d) > 1e-10)
nz_1d = np.count_nonzero(np.abs(T_1d) > 1e-10)
nz_2d = np.count_nonzero(np.abs(T_2d) > 1e-10)
nz_3d = np.count_nonzero(np.abs(T_3d) > 1e-10)
nz_tot = np.count_nonzero(np.abs(T_full) > 1e-10)
print(f"  0D: {nz_0d:>4d} non-zero entries")
print(f"  1D: {nz_1d:>4d} non-zero entries")
print(f"  2D: {nz_2d:>4d} non-zero entries")
print(f"  3D: {nz_3d:>4d} non-zero entries")
print(f"  Tot: {nz_tot:>4d} non-zero entries")

# Methode 2: Spectrale koppelingssterkte
# Bereken de effectieve 2-qubit Hamiltoniaan die elk type induceert
print("\nMethode 2: Spectrale koppelingssterkte")
print("  (max singuliere waarde = sterkste koppelingsrichting)")
for name, Tpart in [("0D", T_0d), ("1D", T_1d), ("2D", T_2d), ("3D", T_3d)]:
    _, Sp, _ = np.linalg.svd(Tpart, full_matrices=False)
    print(f"  {name}: σ_max = {Sp[0]:.6f}")

# Methode 3: Effectieve interactie-Hamiltoniaan
# Het Zorn-product definieert een map |i>|j> -> |out>
# De effectieve Hamiltoniaan H_eff = T^T T / Tr(T^T T) 
# meet de "koppelingssterkte" in de 64-dim input ruimte.

H_eff_full = T_full.T @ T_full
H_eff_0d = T_0d.T @ T_0d
H_eff_1d = T_1d.T @ T_1d
H_eff_2d = T_2d.T @ T_2d
H_eff_3d = T_3d.T @ T_3d

print(f"\nMethode 3: Effectieve Hamiltoniaan Tr(H_eff)")
for name, H in [("0D", H_eff_0d), ("1D", H_eff_1d), ("2D", H_eff_2d), ("3D", H_eff_3d), ("Full", H_eff_full)]:
    tr = np.trace(H)
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    print(f"  {name}: Tr={tr:.4f}, λ_max={ev[0]:.4f}, λ_min(>0)={ev[ev>1e-10][-1] if np.any(ev>1e-10) else 0:.4f}")

# Methode 4: Hoeveel "quantuminformatie" draagt elk type?
# = entanglement dat het product creëert tussen twee qubits
print(f"\nMethode 4: Entanglement-genererende capaciteit")
print("  (Schmidt-rank van T als bipartiet operator)")
for name, Tpart in [("0D", T_0d), ("1D", T_1d), ("2D", T_2d), ("3D", T_3d)]:
    # Reshape T (8×64) als (8×8×8) en kijk naar Schmidt decomp
    Tresh = Tpart.reshape(8, 8, 8)
    # Partial trace over output → 8×8 reduced density matrix
    rho = np.einsum('oij,oik->jk', Tresh, Tresh)
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    ent = -np.sum(ev/np.sum(ev) * np.log2(ev/np.sum(ev)+1e-30)) if len(ev) > 0 else 0
    print(f"  {name}: Schmidt-rank={len(ev)}, entanglement={ent:.4f} bits")

print("\n" + "="*60)
print("DEEL 7: CONCLUSIE — KRACHT-HIËRARCHIE UIT ALGEBRA?")
print("="*60)

# Rangschik de 4 bijdragen naar sterkte
components = [
    ("0D (scalar×scalar)", norm_0d, np.linalg.matrix_rank(T_0d, tol=1e-10)),
    ("1D (scalar×vector)", norm_1d, np.linalg.matrix_rank(T_1d, tol=1e-10)),
    ("2D (cross product)", norm_2d, np.linalg.matrix_rank(T_2d, tol=1e-10)),
    ("3D (dot product)",   norm_3d, np.linalg.matrix_rank(T_3d, tol=1e-10)),
]
components.sort(key=lambda x: x[1], reverse=True)

print("\nRangschikking naar kopplingssterkte (Frobenius norm):")
for i, (name, norm, rank) in enumerate(components):
    print(f"  {i+1}. {name}: ||T||_F = {norm:.4f}, rank = {rank}")

# Bereken de ratio's
norms = [c[1] for c in components]
print(f"\nRatio's (genormaliseerd op sterkste):")
for i, (name, norm, rank) in enumerate(components):
    print(f"  {name}: {norm/norms[0]:.4f}")

print(f"\nDe hiërarchie is: {components[0][0].split('(')[1].split(')')[0]} > "
      f"{components[1][0].split('(')[1].split(')')[0]} > "
      f"{components[2][0].split('(')[1].split(')')[0]} > "
      f"{components[3][0].split('(')[1].split(')')[0]}")
