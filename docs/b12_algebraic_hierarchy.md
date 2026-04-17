# B12: Algebraïsche basis voor de kopplingshiërarchie

## Kernvraag

Produceert de Zorn-vermenigvuldiging NATUURLIJK verschillende
kopplingssterkte op verschillende dimensionale schalen? Is er
een reden vanuit de octonionische algebra om de hiërarchie
J_1D >> J_2D >> J_3D te verwachten die we in B11 testten?

## Methode

### Decompositie van het Zorn-product

Het Zorn-product Z1 × Z2 met Z = (a, α, β, b):

    scalar-out:  a*c + α·δ
    α-output:    a*γ + d*α + β×δ
    β-output:    c*β + b*δ - α×γ
    scalar-out:  β·γ + b*d

We ontleden dit in vier dimensionale bijdragen:

- **0D** (scalar×scalar): a*c en b*d — 2 termen
- **1D** (scalar×vector): a*γ, d*α, c*β, b*δ — 12 termen
- **2D** (cross product): β×δ en -α×γ — 12 termen
- **3D** (dot product): α·δ en β·γ — 6 termen

Verificatie: de vier bijdragen sommeren exact tot het
volledige product (fout < 1e-15).

### Analyse-methoden

1. Frobenius norm van de transfer matrix per bijdrage
2. Rank (= aantal onafhankelijke koppelingskanalen)
3. Singuliere waarden (= spectrale kopplingssterkte)
4. Entanglement-genererende capaciteit (Schmidt-rank)
5. Chirale asymmetrie (commutator-analyse)
6. Fano-decompositie-invariantie

## Resultaten

### Kopplingssterkte per dimensionaal type

| Component       | ||T||_F | Energie% | Rank | σ_max  | Rank×σ |
|-----------------|---------|----------|------|--------|--------|
| 1D (scalar×vec) | 3.464   | 37.5%    | 6    | 1.414  | 8.485  |
| 2D (cross prod) | 3.464   | 37.5%    | 6    | 1.414  | 8.485  |
| 3D (dot prod)   | 2.449   | 18.7%    | 2    | 1.732  | 3.464  |
| 0D (scalar×scl) | 1.414   | 6.2%     | 2    | 1.000  | 2.000  |
| Totaal          | 5.657   | 100%     | 8    | 2.000  | 16.000 |

### Chirale asymmetrie

Het Zorn-product heeft een cruciale asymmetrie:
- α-output: +β×δ (positief cross product)
- β-output: -α×γ (negatief cross product)

Test op 10.000 random producten:
- α/β ratio commutator: 1.0037 (= gelijk)
- De chirality beïnvloedt het TEKEN maar niet de STERKTE

### Fano-planum invariantie

Alle 7 Fano-decomposities produceren IDENTIEKE normen en ranks.
De 1D/2D ratio is exact 1.0000 voor elke decompositie.
De Fano-structuur (Steiner tripel systeem) is perfect symmetrisch:
elk paar imaginaire eenheden zit in precies 1 triplet.

### Transfer matrix eigenwaarden

T·T^T heeft 8 gelijke eigenwaarden (= 4.0).
Het Zorn-product is een ISOTROPE koppeling op d=8.
Er is geen preferente richting in de bond-ruimte.

## Analyse

### De viervoudige hiërarchie

De Zorn-algebra produceert vier niveaus van kopplingssterkte
wanneer we de tensor-netwerkstructuur combineert met de
dimensionale decompositie:

1. **Intra-triplet** (16.0 = 100%): 3 qubits binnen één
   Zorn-element koppelen vrij in d=8. Geen truncatie nodig.
   → analogie: sterke kracht / confinement

2. **Adjacent 1D+2D** (8.5 = 53%): scalar×vector en cross
   product termen tussen naburige tripletten. Rank 6/8.
   → analogie: elektromagnetisme

3. **Adjacent 3D** (3.5 = 22%): dot product termen tussen
   naburige tripletten. Slechts rank 2/8.
   → analogie: zwakke kracht

4. **Adjacent 0D** (2.0 = 13%): scalar×scalar termen. Rank 2/8,
   geen structuur (pure getallen).
   → analogie: zwaartekracht

### De 1D/2D degeneratie: electroweak unificatie?

Het opvallendste resultaat: de 1D (scalar×vector) en 2D (cross
product) bijdragen zijn EXACT gelijk in sterkte, rank, en spectrale
structuur. De Fano-permutaties breken deze degeneratie niet.

Dit is suggestief voor de electroweak unificatie: in het Standaard
Model zijn de elektromagnetische en zwakke kracht gerelateerd via
SU(2)×U(1) symmetrie. Bij hoge energie zijn ze unified; bij lage
energie is de symmetrie gebroken door het Higgs-mechanisme.

In de Zorn-algebra is de 1D/2D symmetrie EXACT — er is geen
ingebouwd symmetrie-brekingsmechanisme. Als deze analogie klopt,
zou een extern mechanisme (analoog aan Higgs) nodig zijn.

### Wat de ratio's NIET verklaren

De algebraïsche ratio's zijn 1 : 0.53 : 0.22 : 0.13.
De echte krachten-ratio's zijn 1 : 10^-2 : 10^-6 : 10^-39.

Het verschil is vele ordes van grootte. De algebra levert de
TOPOLOGIE van de hiërarchie (vier niveaus, correct geordend)
maar niet de KWANTITATIEVE verhoudingen. Die zouden moeten
komen van:
- Renormalisatiegroep-effecten (running couplings)
- Symmetriebreking (Higgs-achtig mechanisme)
- Dimensionale compactificatie

## Eerlijke conclusie

### Wat het WÈL bewijst

1. De Zorn-algebra heeft een NATUURLIJKE viervoudige hiërarchie
   in kopplingssterkte
2. De rangschikking (intra > 1D+2D > 3D > 0D) is CORRECT
   ten opzichte van de natuurkrachten
3. De 1D/2D degeneratie is consistent met electroweak unificatie
4. De intra-triplet koppeling (d=8, exact) is structureel
   gescheiden van inter-triplet koppeling — dit IS confinement

### Wat het NIET bewijst

1. De kwantitatieve ratio's matchen niet (factor ~10 vs factor ~10^37)
2. De chirale asymmetrie breekt de 1D/2D degeneratie niet
3. Er is geen Higgs-achtig mechanisme in de pure algebra
4. De Fano-structuur is te symmetrisch om voorkeursrichtingen te creëren

### Status

De algebraïsche hiërarchie is een NOODZAKELIJKE maar ONVOLDOENDE
voorwaarde voor een theorie van de krachten. De topologie klopt;
de kwantitatieve verhoudingen vereisen aanvullende fysica.

Dit plaatst het resultaat in dezelfde categorie als B11: een
interessante structurele analogie die verdere studie verdient,
maar geen fysische theorie.

## Praktische toepassing: perturbatieve QAOA

### Idee

De 4-componentendecompositie leidt tot een perturbatieve expansie
van de inter-kolom ZZ-gate:

    exp(-iγ Σ_y Z_y Z_y') = Σ_{|S|≤k} cos(γ)^{Ly-|S|} (-i sin(γ))^{|S|} Π Z_y Z_y'

Door te trunceren op level k krijg je een lagere rank:

| Level | Max rank | Fout (Ly=4, p=1) |
|-------|----------|------------------|
| 0     | 1        | 100% (trivial)   |
| 1     | 1+Ly     | ~27%             |
| 2     | 1+Ly+C(Ly,2) | ~2.5%       |
| Ly-1  | 2^Ly - 1 | <0.1%           |
| Ly    | 2^Ly     | exact            |

### Resultaat

**De perturbatieve expansie verlaagt de chi NIET.** De exacte
diagonale gate houdt de productstructuur van de MPS intact
(chi=4 bij p=1), terwijl de benaderde gate die structuur breekt
en MEER chi nodig heeft (chi=13-24 bij level 1-2).

### Verklaring chi=4

De analyse onthulde WÉL waarom chi=4 universeel is bij p=1:
de inter-kolom ZZ-gate heeft slechts Ly+1 unieke diagonaalwaarden
(bepaald door het aantal matchende bits). De MPO-evolutie comprimeert
dit verder tot chi=4 door de Hadamard-initialisatie en observable-
structuur.

De 4 unieke waarden corresponderen met:
- exp(-iγ × (-Ly)): alle bits anders
- exp(-iγ × (-Ly+2)): 1 bit gelijk
- ...
- exp(-iγ × (+Ly)): alle bits gelijk

### Waar het WÉL nuttig is

1. **Parameter screening bij Ly≥4**: level-(Ly-1) geeft <0.1% fout
   met ~halve rank. Voor snelle grid-search voor de exacte run.
2. **TEBD/DMRG**: bij niet-diagonale gates (Heisenberg) zou de
   componentendecompositie de MPO bond dimension KUNNEN verlagen.
3. **Theoretisch begrip**: de algebraïsche structuur verklaart
   het chi=4 fenomeen dat we empirisch vonden in B10c.

## Bestanden

- `code/b12_algebra_hierarchy.py` — decompositie van het Zorn-product
- `code/b12_chirality_fano.py` — chiraliteits- en Fano-analyse
- `code/b12c_perturbative_qaoa.py` — perturbatieve QAOA engine + tests
- `code/b12d_chi4_mechanism.py` — analyse waarom chi=4 volstaat
