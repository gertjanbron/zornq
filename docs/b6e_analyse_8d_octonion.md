# Analyse: Waarom d=8, en wat betekent dat voor 8D?

## Status: 10 april 2026

---

## DE VRAAG

Werkt de Zorn-triplet compressie (3q → d=8) omdat:
(a) het toevallig past bij de cilindergeometrie, of
(b) er een diepere algebraische reden is die verbonden is met de 8D
    structuur van de octonionen?

## WAT WE WETEN (BEWEZEN)

### 1. Bond-crossing reductie

Op een 3×Lx cilinder met snake-ordering:
- Standaard (d=2): maximaal 5 bonds kruisen één cut
  → χ ~ 2^5 = 32 nodig voor exacte beschrijving
- Triplet (d=8): maximaal 3 bonds kruisen (alleen horizontaal)
  → χ ~ 2^3 = 8 nodig

De verticale bonds (2 per mid-column cut) worden geabsorbeerd in de
lokale d=8 Hilbertruimte. Dit verklaart de factor 4× reductie in χ.

### 2. Partitie-analyse (3×3 cilinder, 280 mogelijke 3×3×3 partities)

| Partitie    | Rang | Total S | Bonds kruisend |
|-------------|------|---------|----------------|
| Rijen       | #1   | 4.27    | 9              |
| Kolommen    | #64  | 6.16    | 6              |

**Rijen minimaliseren totale verstrengeling** (#1 van 280). Kolommen
staan op #64. Dit is contra-intuïtief: de "winnende" Zorn-triplet
(kolommen) is NIET de optimale partitie.

### 3. Waarom kolommen toch winnen voor DMRG

Het verschil: een MPS meet verstrengeling **sequentieel** (A|BC, AB|C),
niet over alle bipartities tegelijk.

- Kolom-MPS: lengte Lx, d=8 per site. Praktisch voor elke Lx.
- Rij-MPS: lengte Ly=3, d=2^Lx. Exponentieel in Lx → onbruikbaar.

De kolom-groepering is niet optimaal voor entropie, maar het is de
enige schaelbare optie die d vast houdt terwijl het systeem groeit.

### 4. Mutual Information structuur

Top MI-paren op 3×3 cilinder:
- Horizontale bonds: MI ≈ 0.50–0.63 (sterk)
- Verticale bonds: MI ≈ 0.14–0.35 (zwakker)

De triplet absorbeert de zwakkere verticale bonds. Dit is efficiënt
maar niet maximaal — de sterkere horizontale bonds moeten door de
MPS-bonds.

---

## WAT DIT BETEKENT VOOR 8D (SPECULATIEF)

### Het bewezen mechanisme

De Zorn-triplet compressie werkt via **dimensionele absorptie**: door
qubits langs één richting te groeperen in het lokale Hilbertruim,
elimineer je die richting uit de cut-oppervlakte.

In nD op een hyperkubus met zijde L:
- Zonder groepering: cut-oppervlakte = L^(n-1), χ ~ exp(L^(n-1))
- Met k-qubit groepering langs m richtingen: elimineer m dimensies
  → cut-oppervlakte = L^(n-1-m), χ ~ exp(L^(n-1-m))

### De octonion-connectie

In 8D: cut-oppervlakte = L^7. Groepering van 3 qubits (d=8) elimineert
1 richting → L^6. Dat is nuttig maar niet transformatief.

MAAR: als je de volle octonion-structuur gebruikt:
- 8 dimensies = 1 scalar + 7 imaginaire richtingen
- De 7 richtingen vormen het Fano-vlak: 7 punten, 7 lijnen, 3 per lijn
- Elke Fano-lijn = quaternionische subalgebra (d=4, 2 qubits)

**Hypothese**: groepeer niet 3 qubits willekeurig, maar groepeer langs
Fano-lijnen. Dan absorbeer je 7 richtingen tegelijk via de multiplicatieve
structuur van de octonionen, niet via brute-force dimensie-expansie.

- 7 richtingen geabsorbeerd → cut-oppervlakte daalt van L^7 naar L^0 = 1
- χ wordt CONSTANT, onafhankelijk van systeemgrootte

Dit zou betekenen: **octonionen zijn de maximale compressie-algebra voor
8D tensor networks**, precies omdat ze de maximale deling-algebra zijn
(Hurwitz).

### Status van deze hypothese

| Aspect | Status |
|--------|--------|
| d=8 absorbeert 1 richting | BEWEZEN (B6e) |
| Fano-lijnen als groeperings-eenheid | CONJECTUUR |
| G₂-symmetrie reduceert irrep-kanalen | CONJECTUUR |
| χ=constant in 8D met Zorn-groepering | ONBEWEZEN, testbaar |

### Verbinding met bestaande literatuur

1. **Freedman et al. (2018)**: Octonion-projecties geven universele
   quantum computing via meting. Verbinding met onze compressie: de
   octonion-projecties zijn precies de operatoren die de d=8 ruimte
   efficiënt doorzoeken.

2. **Black hole / qubit correspondentie**: Het Fano-vlak verschijnt in
   de drie-qubit verstrengeling (3-tangle) en zwarte-gat entropie.
   Onze 3-qubit groepering raakt dezelfde structuur.

3. **E₈ rooster**: De dichtste bolstapeling in 8D (Viazovska 2016) is
   gebouwd uit integrale octonionen. Als tensor networks op dit rooster
   worden gedefinieerd, zou de Zorn-structuur de natuurlijke compressie
   zijn.

---

## CONCREET TESTBARE VOORSPELLINGEN

1. **3D test (Ly=3, Lz=3, Lx variabel)**:
   Groepeer 3×3=9 qubits per kolom → d=512.
   Voorspelling: χ=512 geeft exacte resultaten waar standaard d=2
   DMRG χ=2^9=512 nodig heeft. Break-even, maar met 1/9 van de sites.

2. **Fano-lijn test**: Construeer een Hamiltoniaan op 7 qubits met
   interacties langs de 7 Fano-lijnen. Meet of de Zorn-triplet
   groepering optimaal is voor de verstrengeling-structuur.

3. **G₂-invariante Hamiltoniaan**: Bouw een Hamiltoniaan die commuteert
   met G₂ op d=8 sites. Meet of de irrep-decompositie de bonddimensie
   verder reduceert bovenop de dimensionele absorptie.

---

## TESTRESULTATEN (10 april 2026)

### Test 1: Fano-vlak Hamiltoniaan (7 qubits)

Fano-vlak: 7 lijnen × 3 punten = 21 paarinteracties = C(7,2).
Elke qubit-pair zit op precies één Fano-lijn → complete graaf.
**Geen structuur** te exploiteren met paarinteracties.

### Test 2: Bond-crossing in nD (geometrische analyse)

| Dim | Std max_cut | Grouped max_cut | d      | Netto voordeel |
|-----|-------------|-----------------|--------|----------------|
| 2D  | 5           | 3               | 8      | **4× in χ**    |
| 3D  | 25          | 9               | 512    | d=512 grensgebied |
| 4D  | 79          | 27              | 2^27   | d onhanteerbaar |
| 8D  | ~6000       | 2187            | 2^2187 | onmogelijk     |

**Slab-groepering schaalt niet voorbij 3D.** De d-groei compenseert
exact de chi-besparing.

### Test 3: G₂-invariante Hamiltoniaan (d=8 keten)

Octonion-interactie H = Σ_a L_a ⊗ L_a op d=8 sites:
- 3 sites: Schmidt rank **8/8** (vol), S=2.92
- 4 sites: Schmidt rank **64/64** (vol), S=3.24
- Heisenberg vergelijking: zelfde rang, lagere entropie

**G₂-symmetrie comprimeert entanglement NIET.**

---

## CONCLUSIE (NA TESTS)

De drie tests **falsificeren de sterke hypothese**.

**Wat BEWEZEN werkt:**
De Zorn-triplet compressie is geometrisch, niet algebraïsch.
Het werkt in 2D omdat d=8 klein genoeg is, groepering verticale
bonds elimineert, en max_cut daalt van 5→3 (chi reductie 4×).

**Wat NIET werkt:**
- Slab-groepering in ≥4D: d explodeert
- G₂-symmetrie: volle Schmidt-rang
- Fano-vlak: geen 2-lichaam structuur

**De octonion-connectie is reëel maar beperkt:**
d=8=2³ past bij Ly=3 cilinders. De Zorn-matrix geeft elegante MPO's.
Maar de waarde zit in 2D/3D simulatie, niet in universele 8D compressie.
