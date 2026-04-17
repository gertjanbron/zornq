# B10b: 2D QAOA Parameter Optimalisatie

## Kernvraag

Zijn QAOA-parameters universeel overdraagbaar op 2D roosters,
net zoals in 1D (B9)? Of hangen ze af van de roostergeometrie?

## Methode

- Column-grouped Heisenberg-MPO engine (uit B10/B10c)
- Grid search (25×25) + gradient refinement voor p=1
- Optimalisatie op klein rooster (3×2, 6 qubits)
- Transfer test naar roosters tot 5×3 (15 qubits)
- Systematische gamma-scan over 8 roostermaten

## Hoofdresultaat

### Beta is universeel, gamma niet

**Beta* = 1.1778** — identiek aan 1D, onafhankelijk van roostermaat.
De mixer-parameter is universeel over alle grafenstructuren.

**Gamma* hangt af van de gemiddelde graad** van het rooster:

| Rooster | n  | edges | avg_deg | gamma* | ratio  |
|---------|----|-------|---------|--------|--------|
| 1D lang | —  | —     | 2.00    | 0.4021 | 0.7498 |
| 2×2     | 4  | 4     | 2.00    | 0.3927 | 0.7500 |
| 3×2     | 6  | 7     | 2.33    | 0.3542 | 0.7216 |
| 4×2     | 8  | 10    | 2.50    | 0.3392 | 0.7118 |
| 5×2     | 10 | 13    | 2.60    | 0.3315 | 0.7070 |
| 6×2     | 12 | 16    | 2.67    | 0.3268 | 0.7041 |
| 3×3     | 9  | 12    | 2.67    | 0.3252 | 0.7016 |
| 4×3     | 12 | 17    | 2.83    | 0.3138 | 0.6944 |
| 5×3     | 15 | 22    | 2.93    | 0.3077 | 0.6907 |

### Het product gamma* × avg_degree stijgt langzaam

| avg_deg | gamma* × deg |
|---------|-------------|
| 2.00    | 0.785       |
| 2.50    | 0.848       |
| 2.67    | 0.873       |
| 2.83    | 0.888       |
| 2.93    | 0.902       |

Dit is GEEN simpele 1/graad-relatie. Het product convergeert
naar ~0.90 voor grote 2D roosters.

### Convergentie Ly=2 (thermodynamic limit)

De gamma-stappen worden snel kleiner:
```
Lx=2→3: delta_gamma = -0.039
Lx=3→4: delta_gamma = -0.015
Lx=4→5: delta_gamma = -0.008
Lx=5→6: delta_gamma = -0.005
```

Gamma convergeert snel — bij Lx=10 is het nagenoeg stabiel.

### 2D parameters verslaan 1D parameters op 2D

Het verschil groeit met roostergrootte:
```
2×2:  +0.003  (2D-opt vs 1D-opt)
3×3:  +0.010
4×3:  +0.012
4×4:  +0.014
5×3:  +0.013
```

Bij 4×4 is het verschil 2.1% — significant voor optimalisatie.

## Interpretatie

De fysische verklaring is eenvoudig: gamma bepaalt de
fase-interactie per edge. Bij meer buren (hogere graad) draagt
elke edge minder bij aan de totale fase die een qubit voelt.
Te veel fase per edge → over-rotatie → slechtere approximatie.

In formule (benaderend): gamma* ≈ 0.9 / avg_degree voor
grote 2D roosters, en gamma* ≈ 0.8 / degree voor 1D.

## Praktische implicatie

Voor een onbekend 2D rooster van maat Lx × Ly:
1. Bereken avg_degree = 2 × n_edges / n_vertices
2. Schat gamma* ≈ 0.88 / avg_degree (interpolatie)
3. Gebruik beta* = 1.1778 (universeel)
4. Optioneel: fine-tune op klein sub-rooster

## Verificatie

Alle MPO-resultaten exact geverifieerd via state-vector
simulatie (error < 1e-15 op roosters tot 16 qubits).

## Bestanden

- `code/b10b_2d_optimizer.py` — optimizer met grid search + gradient
- `code/b10b_gamma_analysis.py` — gamma vs graad analyse
