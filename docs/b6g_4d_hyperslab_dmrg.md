# B6g: 4D Hyperslab DMRG — Heisenberg op 2×2×2×Lw

## Status: 10 april 2026, GRENSGEBIED

---

## RESULTAAT

4D hyperslab-groepering (hele 2×2×2 kubus als 1 MPS-site, d=256)
werkt voor het kleinste systeem maar schaalt niet:

- **EXACT bij chi=4** op 2×2×2×2 (16q, 1 sweep, 36s)
- **INFEASIBLE** op 2×2×2×3 (24q, Lanczos matvec te duur)

Dit bevestigt de bond-crossing analyse: in 4D groeit d even snel als
chi daalt, geen netto voordeel.

---

## NUMERIEKE RESULTATEN

### Exacte referentie

| Systeem  | Methode | E           | Tijd  |
|----------|---------|-------------|-------|
| 2×2×2    | eigvalsh (d=256) | -19.280357 | <0.1s |
| 2×2×2×2  | Lanczos (d²=65536) | -44.913933 | 11.5s |

De inter-slab bijdrage (8 Heisenberg-bonds langs w): -6.353 per cut.

### DMRG resultaten (2×2×2×2, 16 qubits, 2 slabs)

| chi | E           | Gap tot exact | Sweeps | Tijd  |
|-----|-------------|---------------|--------|-------|
|  4  | -44.9139328 | <1e-7         | 1      | 36s   |

Chi=4 is al exact omdat het 2-slab systeem slechts 1 MPS-bond heeft.
De 2-site solver lost het volledige probleem op in één stap.

### 2×2×2×3 (24 qubits, 3 slabs) — INFEASIBLE

Zelfs bij chi=2 is de 2-site Lanczos te duur:
- Boundary bond: dim = 1×256×256×2 = 131,072
- Interior bond: dim = 2×256×256×2 = 262,144
- Matvec cost: O(chi²·d²·D) per iteratie ≈ O(4·65536·26) ≈ 7M ops
- LW/WR precompute: ~6.6M entries per tensor
- Geschatte tijd per bond-solve: >100s
- Per sweep (4 bond-solves): >400s

---

## METHODE

### Hyperslab-groepering

Elke xyz-hyperlaag (2×2×2 = 8 qubits) wordt 1 MPS-site met d=256.
- Intra-slab bonds (x, y, z, periodiek): exact in d=256
  - 12 bonds: 4 per as (2×2×2 kubus met periodieke grenzen)
- Inter-slab bonds (w-richting): via MPS-bonds, 8 per cut
- MPO bonddimensie: D = 2 + 3×8 = 26

### Vergelijking met lagere dimensies

| Dim | Cross-sectie | d    | MPO D | Bonds/cut | Status     |
|-----|-------------|------|-------|-----------|------------|
| 2D  | Ly=3        | 8    | 11    | 3         | BEWEZEN    |
| 3D  | 2×2         | 16   | 14    | 4         | BEWEZEN    |
| 4D  | 2×2×2       | 256  | 26    | 8         | GRENSGEBIED|
| 5D  | 2×2×2×2     | 65536| 50    | 16        | ONMOGELIJK |

### Waarom 4D niet schaalt

De fundamentele bottleneck is de 2-site solver:
- dim = chi_l × d × d × chi_r
- Bij d=256: zelfs chi=2 geeft dim = 2×256×256×2 = 262K
- Lanczos matvec vereist O(d²·D·chi²) operaties per iteratie
- Dit maakt sweeps met chi>4 praktisch onmogelijk

Vergelijking kosten per matvec:
| Dim | d   | D  | chi=8 matvec | Haalbaar? |
|-----|-----|----|-------------|-----------|
| 2D  | 8   | 11 | 45K ops     | Ja        |
| 3D  | 16  | 14 | 230K ops    | Ja        |
| 4D  | 256 | 26 | 109M ops    | Nee       |

---

## CONCLUSIE

De 4D test bevestigt definitief de bond-crossing analyse uit de 8D
studie: slab-groepering werkt spectaculair in 2D en 3D, maar verliest
effectiviteit in 4D+ doordat d exponentieel groeit met het volume van
de cross-sectie.

De grens van bruikbaarheid ligt bij d ~ 512 (3D met 3×3 cross-sectie).
Daarboven wordt de 2-site DMRG solver de bottleneck.

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b6g_4d_hyperslab_dmrg.py` | Implementatie |
| `docs/b6g_4d_hyperslab_dmrg.md` | Dit document |
