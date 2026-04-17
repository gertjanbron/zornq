# B10: 2D-connectiviteitstest — Heisenberg-MPO op vierkant rooster

## Status: BEWEZEN MET NUANCE — 10 april 2026

## Kernvraag

Schaalt de Heisenberg-MPO aanpak naar 2D grafen?
In 1D was chi_O=2 bij alle p. Wat gebeurt er in 2D?

## Methode

- Vierkant rooster Lx × Ly met snake-ordering naar 1D MPO
- QAOA MaxCut: ZZ gates op alle edges (horizontaal + verticaal)
- Verticale edges = long-range in MPO → SWAP-ladder nodig
- Meet operator chi na volledige Heisenberg-evolutie

## Resultaten

### Chi is CONSTANT in systeemgrootte (bij vast p)

| Rooster    | Qubits | Chi_max (p=1) | Truncatie  | Tijd    |
|------------|--------|---------------|------------|---------|
| 3×3        | 9      | 4             | ~0         | 0.00s   |
| 5×5        | 25     | 4             | ~0         | 0.01s   |
| 10×10      | 100    | 4             | ~0         | 0.15s   |
| 20×20      | 400    | 4             | ~0         | 0.70s   |
| 50×50      | 2500   | 4             | ~0         | 18.7s   |

**Chi groeit NIET met roostergrootte.** Exact, geen truncatie.

### Chi groeit EXPONENTIEEL met circuitdiepte p

| p  | Chi_max (3×3) | Tijd    |
|----|---------------|---------|
| 1  | 4             | 0.00s   |
| 2  | 64            | 0.05s   |
| 3  | 256           | 1.39s   |
| 5  | 256 (gekapt)  | 7.55s   |

De SWAP-ladder voor long-range edges vermengt operator-informatie
langs de hele MPO-keten. Bij elke laag verdubbelt het effectieve
bereik van de operator. In 2D raakt de operator meer buren per laag
dan in 1D → chi groeit sneller.

### Exacte verificatie

| Rooster | p | Exact cost | MPO cost   | Fout    | Chi |
|---------|---|------------|------------|---------|-----|
| 3×3     | 1 | 8.279519   | 8.279519   | 3.2e-14 | 6   |
| 4×4     | 1 | 16.040607  | 16.040607  | 4.7e-12 | 6   |
| 3×3     | 2 | 4.080422   | 4.080422   | 8.9e-16 | 64  |
| 4×4     | 2 | 8.663052   | 8.663052   | 3.1e-13 | 116 |

**Alles exact.** De methode is correct in 2D.

### Optimale QAOA parameters (2D, p=1)

- gamma* ≈ 0.325, beta* ≈ 1.178
- Approximatie-ratio: ~0.69 (2D vierkant rooster)
- Vergelijk: 1D path graph ratio ~0.75

## Analyse: twee regimes

### Regime 1: p=1 — VOLLEDIG SCHAALBAAR
- Chi = 4, onafhankelijk van n
- 2500 qubits in 19 seconden op een laptop
- Exact, geen benadering
- Bruikbaar voor QAOA p=1 op willekeurig grote 2D roosters

### Regime 2: p≥2 — BEGRENSD DOOR CHI
- Chi groeit exponentieel met p: ~4^p
- p=2: chi=64 (nog haalbaar, minuten voor klein rooster)
- p=3: chi=256 (grens van laptop, uren voor >10×10)
- p=5: vereist truncatie of GPU

## Conclusie

**Het antwoord op de kernvraag is genuanceerd:**

De operator chi groeit exponentieel in **circuitdiepte p**, maar NIET
in **systeemgrootte n**. Dit bevestigt precies wat voorspeld was:

- Voor vast p: de laptop schaalt naar willekeurig groot n
- De nauwkeurigheid/kwaliteit is begrensd door p, niet door n
- De grens is p~2-3 op een laptop, p~5-8 met GPU

**Vergelijking met Gemini's voorspelling:**
Gemini voorspelde dat operator spreading de methode zou breken in 2D.
Dat klopt NIET voor p=1 (chi=4, constant). Het klopt WEL voor
hoge p — maar dat is precies het circuit-diepte argument, niet het
systeemgrootte argument. De methode breekt pas bij hoge p, niet bij
grote n.

**Praktische waarde:**
QAOA p=1 op een 2D rooster is al een niet-triviale combinatorische
optimizer (ratio ~0.69). Met p=2 verbetert de ratio en is met
column-grouping (chi=16) haalbaar op willekeurig grote roosters.

## Ordening: Morton vs Snake vs Column-grouping

### Morton/Hilbert Z-curve
Getest als alternatief voor snake ordering. Resultaat: **helpt niet**.
Morton verdeelt de afstanden gelijkmatiger (V omlaag, H omhoog) maar
de gemiddelde afstand is identiek en de maximum afstand is zelfs slechter.
Chi bij p=2: snake=64, morton=53-81 (niet structureel beter).

Fundamenteel: elke 1D uitvouwing van een 2D rooster heeft edges met
afstand Ω(√n). Geen enkele curve kan dat vermijden — het is een
topologische beperking.

### Column-grouping: de oplossing
In plaats van het 2D rooster plat te vouwen, **rollen we het op**.
Elke kolom van Ly qubits wordt één MPO-site met d=2^Ly:

| Methode              | d  | p=1 | p=2 | p=3 |
|----------------------|----|-----|-----|-----|
| Snake (SWAP-ladder)  | 2  | 4   | 64  | 256 |
| Column-grouped Ly=2  | 4  | 4   | **16**  | 64  |
| Column-grouped Ly=3  | 8  | 4   | **32**  | —   |

Chi is **constant in Lx** (systeemlengte):
- 6×2, 10×2, 20×2, 50×2 bij p=2: allemaal chi=16

Waarom dit werkt: verticale edges worden **lokale** operaties
(binnen één site), horizontale edges zijn nearest-neighbor.
Geen SWAPs nodig → geen informatielek → lagere chi.

Dezelfde strategie als Zorn-triplet grouping (B6e), nu op de operator.

Trade-off: d groeit exponentieel met Ly. Laptop-limiet:
- Ly=2 (d=4): makkelijk
- Ly=3 (d=8): haalbaar
- Ly=4 (d=16): zwaar
- Ly=5 (d=32): grens

## Beperkingen

- Column-grouping beperkt tot smalle roosters (Ly ≤ 4-5)
- Hogere p vereist chi-truncatie → gecontroleerde benadering
- Breed vierkant rooster (Ly=Lx=groot) vereist 2D tensor netwerk (PEPO)

## Vervolgstappen

- B10b: Optimaliseer QAOA-parameters op 2D met column-grouped evaluator
- B10c: Chi-truncatie bij p=3+ met split-norm kwaliteitsmeter
- B10d: GPU-versnelling voor chi>64 en d>16
- B10e: PEPO (2D tensor netwerk) voor brede vierkante roosters
