# ZornMPS: Unified Engine Architectuur

## Overzicht

`zorn_mps.py` is de geconsolideerde engine die alle bewezen
resultaten uit B1-B12 integreert in één module. Het volgt de
MPS-architectuur uit de spec, maar met drie cruciale toevoegingen.

## Architectuur

### De Node (gecorrigeerd t.o.v. originele spec)

```
class ZornNode:
    tensor: ndarray (chi_left, d, chi_right)
    d = 8  (Zorn-element, niet 16)
```

De lokale dimensie d=8 correspondeert met het Zorn-element
(a, α₁, α₂, α₃, β₁, β₂, β₃, b). Voor column-grouped 2D
roosters wordt d=2^Ly (4, 8, 16, 32).

### Dual Mode

| Eigenschap | Schrödinger | Heisenberg |
|-----------|-------------|------------|
| Wat evolueert | Toestand |ψ⟩ | Operator O |
| Gate richting | Voorwaarts | Omgekeerd |
| d_effectief | d | d² |
| Typische chi | 4-64 | 2-4 |
| Use case | TEBD, DMRG | QAOA observabelen |

Heisenberg-mode is bewezen 4-32× compacter voor QAOA (B7d, B9).

### 7-Operatie Protocol

De module bevat alle 7 algebraïsche operaties:

| # | Operatie | Functie | Rank/64 |
|---|----------|---------|---------|
| 0 | Bilineair × | `zmul(A, B)` | 52 |
| 1 | Additie + | `A + B` | 8 |
| 2 | Subtractie − | `A - B` | 7 |
| 3 | Divisie ÷ | `zinv(A) × B` | 56 |
| 4 | Hodge H | `zhodge(A) × B` | 56 |
| 5 | Associator [·] | `(AB)C − A(BC)` | 28 |
| 6 | Jordan ABA | `zmul(zmul(A,B),A)` | 30 |

Samen: rank 64/64 = informatiecompleet (B2, B3).
Reconstructie via pseudo-inverse: F = 1.0 (machineprecisie).

### SVD-Truncatie

Identiek aan de spec, met twee toevoegingen:

1. **rSVD** (Halko-Martinsson-Tropp): O(k·m·n) voor d≥16.
   Bewezen 8-60× sneller op matrices ≥256×256 (B10d).

2. **Split-norm kwaliteitsmeter**: meet het totaal weggegooid
   gewicht. Vuistregel uit B10h: split_norm < 0.01 → fout < 0.1%.

### HeisenbergQAOA Subengine

Gespecialiseerde engine voor QAOA MaxCut:

- Column-grouped: Ly qubits per kolom als één d=2^Ly site
- Diagonale ZZ-gates: geen matmul, alleen element-wise
- Translatie-invariantie: bulk edges identiek, O(1) unieke evaluaties
- Lightcone-afsnijding: venster 2p+6 sites per edge

Bewezen prestatie:

| Configuratie | Qubits | Tijd | Chi |
|-------------|--------|------|-----|
| 1D p=1 | 10.000 | 4ms | 2 |
| 2D 50×50 p=1 | 2.500 | 19s | 4 |
| 2D 4×3 p=2 | 12 | 0.03s | 14 |

## Wat de spec miste (drie correcties)

### 1. d=8, niet d=16

De spec specificeert d=16 (4 qubits per node). De Zorn-structuur
heeft d=8 (3 qubits = 1 octonion). Dit geeft:
- Halvering van de lokale dimensie
- Structurele split-norm als kwaliteitsmeter
- 7-operatie informatiecompleetheid

### 2. Heisenberg-modus

De spec beschrijft alleen Schrödinger (evolueer de toestand).
De Heisenberg-route (evolueer de operator, reversed) is 4-32×
compacter voor observabelen. Verschil: "bouw het hele universum
na" vs "stuur alleen de thermometer door."

### 3. Algebraïsche hiërarchie

Het Zorn-product ontleedt in 4 componenten met natuurlijke
kopplingssterkte-hiërarchie (B12):

- Intra-triplet (16.0): alles lokaal, geen bond nodig
- Scalar×vector + cross product (8.5): rank 6 per type
- Dot product (3.5): rank 2
- Scalar×scalar (2.0): rank 2

Dit verklaart waarom chi=4 universeel volstaat bij p=1:
de gate heeft slechts Ly+1 unieke diagonaalwaarden.

## Bestanden

- `code/zorn_mps.py` — unified engine (alle componenten)
- `code/b12_algebra_hierarchy.py` — decompositie Zorn-product
- `code/b12_chirality_fano.py` — chiraliteitsanalyse
- `code/b12c_perturbative_qaoa.py` — perturbatieve expansie test
- `code/b12d_chi4_mechanism.py` — verklaring chi=4 fenomeen

## Aanbevolen bouwvolgorde

### Fase 1: Generieke MPS (1 weekend)
Implementeer de spec as-is met numpy/scipy. Of gebruik TeNPy.
Dit geeft direct 10.000+ qubits in 1D.

### Fase 2: Zorn-laag (1 week)
- d=8, 7 operaties, split-norm
- Heisenberg-modus
- Column-grouping voor 2D

### Fase 3: Optimalisatie (ongoing)
- rSVD voor d≥16
- GPU via cupy (drop-in replacement)
- Adaptive-chi TEBD
