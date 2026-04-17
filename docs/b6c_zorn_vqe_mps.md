# B6c: Zorn-VQE MPS Prototype — DMRG 2-site sweep optimizer

## Status: 10 april 2026, BEWEZEN

---

## RESULTAAT

Variationele grondtoestandsvinding via DMRG-stijl 2-site sweeps op
MPS-ansatz. Geen state vector, geen exponentiële muur. Polynomiale
complexiteit in het aantal qubits.

**Heisenberg XXX, verificatie tegen exact:**

| n | χ | E_dmrg | E_exact | Gap | Sweeps | Tijd |
|---|---|--------|---------|-----|--------|------|
| 6 | 2 | -9.3759 | -9.9743 | 5.98e-1 | 20 | 0.03s |
| 6 | 4 | -9.9702 | -9.9743 | 4.14e-3 | 20 | 0.01s |
| 6 | 8 | -9.9743 | -9.9743 | <1e-14 | 11 | 0.02s |
| 12 | 4 | -20.529 | -20.568 | 3.89e-2 | 30 | 0.04s |
| 12 | 8 | -20.568 | -20.568 | 1.77e-4 | 30 | 0.2s |
| 12 | 16 | -20.568 | -20.568 | 2.37e-7 | 28 | 2.2s |

**Transverse-field Ising (h=1.0, kritiek punt):**

| n | χ | E_dmrg | E_exact | Gap |
|---|---|--------|---------|-----|
| 6 | 2 | -7.2943 | -7.2962 | 1.97e-3 |
| 6 | 4 | -7.2962 | -7.2962 | 8.24e-8 |
| 6 | 8 | -7.2962 | -7.2962 | <1e-14 |

**Schaling naar 500 qubits:**

| n | χ | E/site | E/bond | Sweeps | Tijd | RAM |
|---|---|--------|--------|--------|------|-----|
| 50 | 8 | -1.7573 | -1.7932 | 7 | 3.8s | 60 KB |
| 100 | 8 | -1.7643 | -1.7822 | 8 | 9.6s | 95 KB |
| 500 | 8 | -1.7698 | -1.7734 | 5 | 31s | 495 KB |

Bethe-ansatz referentie (oneindige keten): E/site ≈ -1.7726.

---

## METHODE

### DMRG 2-site sweep

1. **Initialisatie**: random MPS, rechts-canonicaliseer
2. **Rechts-sweep** (bond 0 → n-2):
   - Bouw effectieve Hamiltoniaan H_eff voor 2-site blok (i, i+1)
   - Diagonaliseer: θ = grondtoestand eigenvector
   - SVD-splits θ → mps[i] (links-canonicaal), mps[i+1]
   - Update linker-omgeving
3. **Links-sweep** (bond n-2 → 0):
   - Zelfde als rechts, maar splits rechts-canonicaal
   - Update rechter-omgeving
4. Herhaal tot convergentie (dE < tol)

### MPO-gebaseerde omgevingen

De Hamiltoniaan wordt gecodeerd als een Matrix Product Operator (MPO).
Links- en rechts-omgevingen bevatten de volledige Hamiltoniaan-bijdrage
van de contracteerde sites. Geen verlies van termen.

**Heisenberg MPO** (D=5):
```
W = | I    0    0    0    0  |
    | 0    0    0    0   2S- |
    | 0    0    0    0   2S+ |
    | 0    0    0    0    Z  |
    | 0    S+   S-   Z    I  |
```
Decomposie: XX + YY + ZZ = 2(S+S- + S-S+) + ZZ. Volledig reëel.

### Effectieve Hamiltoniaan

Voor bond (i, i+1) met bonddimensie χ en fysische dimensie d=2:
- H_eff heeft grootte (χ·d·d·χ) × (χ·d·d·χ) = (4χ²) × (4χ²)
- χ=8: 256 × 256 matrix → diagonalisatie in ~0.5ms
- χ=16: 1024 × 1024 matrix → diagonalisatie in ~10ms
- χ=32: 4096 × 4096 matrix → diagonalisatie in ~100ms

### Complexiteit

| Operatie | Kosten per bond | Per sweep (n bonds) |
|----------|----------------|---------------------|
| H_eff bouwen | O(χ² · D² · d²) | O(n · χ² · D² · d²) |
| Diagonalisatie | O(χ³ · d³) | O(n · χ³ · d³) |
| SVD | O(χ² · d²) | O(n · χ² · d²) |
| Omgeving update | O(χ² · D · d) | O(n · χ² · D · d) |

Totaal: O(n · χ³ · D² · d³) per sweep. Lineair in n!

---

## VERGELIJKING MET EERDERE POGINGEN

### Product-ansatz VQE (B6, Test 1)
- Globale optimalisatie met finite-difference gradient
- n_params = 8n/3 (Zorn-elementen)
- **FAALT** voor verstrengelde toestanden (fidelity 0.48 Heisenberg)

### MPS-ansatz met Adam optimizer (B6, Test 1)
- Werkt (χ=2: 99.7% Heisenberg)
- Maar: finite-difference gradient kost O(n_params) evaluaties
- n=12 χ=4: 96 params × 2 = 192 evaluaties per stap → te traag

### DMRG 2-site sweep (dit werk)
- Geen gradient nodig: exacte eigenwaarde-optimalisatie per bond
- Per stap: lokaal probleem (4χ² variabelen), niet globaal
- Convergeert in 5-10 sweeps
- **500 qubits in 31 seconden**

---

## COMBINATIE MET ZORN-ARCHITECTUUR

### Zorn-triplet grouping
Groepeer 3 qubits per MPS-site (d=8 i.p.v. d=2). Dit matcht de
Zorn-structuur: 3 qubits = 1 Zorn-element. Intra-triplet verstrengeling
wordt niet getrunceerd. Verwacht: 2-3× minder fout bij dezelfde χ.

H_eff grootte met triplet: (8χ)² × (8χ)² = 64χ² × 64χ².
χ=4: 1024 × 1024 → nog steeds snel.

### Mid-circuit compressie (B6, Test 3)
Voor diepe circuits: comprimeer periodiek naar χ=16.
- Enkele compressie: χ=8 exact
- Triple compressie: χ=16 exact
- Fidelity >99% na 20 lagen QAOA

### Lokale verwachtingswaarden (B6b)
Na DMRG-convergentie: meet alle <Z_q> en <Z_qZ_{q+1}> via
MPS-contractie in O(n · χ²) tijd. 500q: 76ms.

---

## VOLGENDE STAPPEN

1. **Zorn-triplet DMRG**: d=8 per site, χ=4-8 → verwacht betere
   nauwkeurigheid bij minder RAM
2. **Molecuulsimulatie**: H₂, LiH op 20-50 qubits als proof of concept
3. **2D systemen**: Zorn-PEPS of cilindrische DMRG
4. **Paper B4**: dit resultaat + B6b + mid-circuit = complete pipeline

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `docs/b6c_zorn_vqe_mps.md` | Dit document |
| `code/b6c_zorn_vqe_mps.py` | Volledige implementatie met verificatie |
