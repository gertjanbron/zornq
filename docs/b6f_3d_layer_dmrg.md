# B6f: 3D Layer-grouped DMRG — Heisenberg op 2×2×Lz

## Status: 10 april 2026, BEWEZEN

---

## RESULTAAT

Layer-groepering (hele xy-laag als 1 MPS-site) werkt in 3D net zo
dramatisch als de Zorn-triplet in 2D. Op 2×2×Lz kubussen:

- **EXACT bij χ=4** op 2×2×2 (standaard pas bij χ=16)
- **EXACT bij χ=16** op 2×2×3 (standaard gap=0.09 bij χ=16)
- 40 qubits (2×2×10) in 33 seconden bij χ=8

---

## NUMERIEKE RESULTATEN

### 2×2×2 kubus (8 qubits, periodiek xy), E_exact = -19.2803574975

| χ  | Standaard gap | Layer gap       | Verbetering |
|----|---------------|-----------------|-------------|
|  4 | 1.74          | <1e-14 (**EXACT**) | ∞        |
|  8 | 0.44          | <1e-14 (**EXACT**) | ∞        |
| 16 | <1e-14        | <1e-14          | Beide exact |

### 2×2×3 kubus (12 qubits), E_exact = -30.1205486760

| χ  | Standaard gap | Layer gap | Verbetering |
|----|---------------|-----------|-------------|
|  4 | 2.56          | 0.63      | 4.1×        |
|  8 | 0.85          | 0.33      | 2.6×        |
| 16 | 0.091         | <1e-10 (**EXACT**) | ∞  |

### 2×2×4 kubus (16 qubits)

| χ  | Standaard E | Layer E     |
|----|-------------|-------------|
|  8 | -39.754     | **-40.462** |
| 16 | -40.878     | **-41.036** |

### Schaalbaarheid (layer-grouped, χ=8)

| Systeem | Qubits | E         | E/qubit  | Tijd  |
|---------|--------|-----------|----------|-------|
| 2×2×2   | 8      | -19.280   | -2.410   | <0.1s |
| 2×2×3   | 12     | -29.795   | -2.483   | 3.1s  |
| 2×2×4   | 16     | -40.462   | -2.529   | 8.1s  |
| 2×2×6   | 24     | -61.865   | -2.578   | 21.2s |
| 2×2×10  | 40     | -104.879  | -2.622   | 32.8s |

---

## METHODE

### Layer-groepering

Elke xy-laag (2×2 = 4 qubits) wordt 1 MPS-site met d=16.
- Intra-laag bonds (x en y, periodiek): exact in d=16
- Inter-laag bonds (z): via MPS-bonds, 4 per cut
- MPO bonddimensie: D = 2 + 3×4 = 14

### Vergelijking met standaard

| Eigenschap      | Standaard (d=2) | Layer (d=16)  |
|----------------|-----------------|---------------|
| Sites          | 4×Lz            | Lz            |
| Physical dim   | 2               | 16            |
| MPO D          | 20              | 14            |
| Max bonds/cut  | ~9              | 4             |
| Chi voor exact | ~32+            | 16            |

### Bond-crossing reductie

Standaard (snake door 3D): max 9 bonds kruisen een cut
(4 intra-laag + 4 inter-laag + periodieke bonds).
Layer-grouped: exact 4 bonds kruisen (alleen z-richting).
Reductie: 2^9/2^4 = 32× in chi.

---

## IMPLICATIES

### 3D simulatie is haalbaar

Met layer-groepering op een 3×3 cross-sectie (d=512):
- MPO D = 2 + 3×9 = 29
- Chi~16-32 voor goede resultaten
- 3×3×100 (900 qubits) zou haalbaar zijn op een laptop

### Vergelijking met 2D triplet

| Dimensie | Cross-sectie | d    | Chi voor exact | Status     |
|----------|-------------|------|----------------|------------|
| 2D       | Ly=3        | 8    | 8              | BEWEZEN    |
| 2D       | Ly=4        | 16   | 16             | VERWACHT   |
| 3D       | 2×2         | 16   | 16             | **BEWEZEN** |
| 3D       | 3×3         | 512  | ~32?           | TE TESTEN  |
| 4D       | 2×2×2       | 256  | ~256           | GRENSGEBIED |

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b6f_3d_layer_dmrg.py` | Volledige implementatie |
| `docs/b6f_3d_layer_dmrg.md` | Dit document |
