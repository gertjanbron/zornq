# B9: QAOA MaxCut Optimizer via Heisenberg-MPO + Lightcone

## Status: BEWEZEN — 10 april 2026

## Samenvatting

Volledige QAOA MaxCut solver die schaalt naar 10.000+ qubits op een laptop.
Combineert drie technieken:

1. **Heisenberg-MPO** (uit B7d): evolueer de operator, niet de toestand
2. **Lightcone-afsnijding**: elke edge berekenen in O(1) via lokaal venster
3. **Translatie-invariantie**: bulk edges zijn identiek, slechts ~2p+4 unieke evaluaties
4. **Small-n optimalisatie**: parameters universeel, optimaliseer op n=20

## Probleem

MaxCut op path graph (1D keten):

    C = Σ_{i} (1 - Z_i Z_{i+1}) / 2

Maximale cut = n-1 (bipartiet graf). QAOA benadert dit met p lagen van
fase-separatie (exp(-iγ·C)) en mixing (exp(-iβ·Σ X_i)).

## Methode

### Heisenberg-MPO evaluatie

In plaats van de toestand |ψ⟩ = U|+⟩^n te berekenen (exponentieel geheugen),
evolueren we de observable O = Z_i⊗Z_{i+1}:

    O_H = U†·O·U

als MPO. Gates worden in **omgekeerde volgorde** toegepast:

    O_H = U_1†...U_n†·O·U_n...U_1

Voor QAOA p=1 op 1D: chi_max = 2 (operator blijft compact).

### Lightcone-afsnijding

⟨Z_i Z_{i+1}⟩ hangt alleen af van gates binnen afstand p+2 van edge (i,i+1).
We simuleren een **venster** van grootte w = 2p+6 in plaats van het hele systeem.

### Translatie-invariantie

Voor uniforme parameters op een path graph zijn alle bulk edges identiek.
Alleen de ~p+2 boundary edges aan elke kant zijn uniek.

Totale kosten per evaluatie: ~2p+4 venster-simulaties, elk O(w²) = O(p²).
Totaal: O(n) voor het optellen van bulk + boundary bijdragen.

### Small-n optimalisatie

Het gradient-probleem: finite-difference gradienten op groot n worden
gedomineerd door boundary-artefacten die met n schalen.

Oplossing: optimaliseer op klein n (n=20) waar de gradient betrouwbaar is.
De parameters zijn universeel door translatie-invariantie — dezelfde
gamma*, beta* werken voor elk n.

## Resultaten

### Verificatie (MPO vs exact state vector)

| n  | exact       | MPO         | fout    |
|----|-------------|-------------|---------|
| 8  | 5.36192856  | 5.36192856  | 8.9e-15 |
| 12 | 8.35950604  | 8.35950604  | 6.6e-14 |

### Optimale parameters (p=1, geoptimaliseerd op n=20)

- gamma* = 0.4021
- beta* = 1.1778
- Approximatie-ratio: 0.7556

### Multi-layer (n=20)

| p | ratio  | gamma           | beta           |
|---|--------|-----------------|----------------|
| 1 | 0.7556 | [0.402]         | [1.178]        |
| 2 | 0.7602 | [0.680, 1.227]  | [0.288, 0.969] |

### Schaaltest (p=1)

| n      | cut      | max   | ratio  | tijd    |
|--------|----------|-------|--------|---------|
| 100    | 74.3     | 99    | 0.7509 | 0.003s  |
| 500    | 374.3    | 499   | 0.7500 | 0.003s  |
| 1,000  | 749.2    | 999   | 0.7499 | 0.002s  |
| 2,000  | 1,499.0  | 1,999 | 0.7499 | 0.003s  |
| 5,000  | 3,748.5  | 4,999 | 0.7498 | 0.005s  |
| 10,000 | 7,497.6  | 9,999 | 0.7498 | 0.004s  |

## Complexiteitsanalyse

| Aspect       | Waarde                          |
|-------------|---------------------------------|
| Geheugen    | O(w · chi²) = O(p), onafhankelijk van n |
| Evaluatie   | O(n) per cost-evaluatie         |
| Chi (p=1)   | 2 (exact, geen truncatie nodig) |
| Venster     | 2p+6 qubits                    |

## Significantie

Dit is een werkende quantum-geïnspireerde combinatorische optimizer die:
- Geen state vector gebruikt (polynomiaal geheugen)
- Exact is (geen benadering, geen truncatie)
- Lineair schaalt in systeemgrootte
- In milliseconden draait op een laptop

De approximatie-ratio van ~0.75 is consistent met bekende QAOA p=1 resultaten
voor MaxCut. Hogere p kan dit verbeteren.

## Beperkingen

- **1D path graph**: translatie-invariantie is specifiek voor uniforme 1D
- **Hogere p**: venstergrootte groeit lineair met p, kosten kwadratisch
- **Willekeurige grafen**: vereist per-edge evaluatie, geen bulk-truc
  → B10 zal 2D connectiviteit testen

## Code

`code/b9_qaoa_optimizer.py` — 230 regels Python (numpy only)
