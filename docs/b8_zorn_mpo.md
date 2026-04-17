# B8: Zorn-MPO — Operator in Split-Octonion Representatie

## Status: 10 april 2026, GEDEELTELIJK SUCCES

---

## SAMENVATTING

De Heisenberg-operator leeft in slechts **4 dimensies** van de
64-dimensionale Pauli-ruimte. Dat is minder dan de Zorn-dimensie (8).
De standaard Zorn-basis vangt 6 van de 8 actieve Pauli-termen,
maar **XXX en XXY lekken altijd weg** — ongeacht gamma/beta.

De d=2 MPO met chi=2 is de optimale representatie: 2056 parameters
voor 500 qubits. De Zorn-groepering (d=8) is 20× duurder zonder
voordeel.

## KERNRESULTATEN

### 1. Operator dimensionaliteit

| p (lagen) | Effectieve dimensie | In Zorn | Buiten Zorn |
|-----------|--------------------:|--------:|------------:|
| 1         | 2                   | 4/4     | 0           |
| 2         | 2                   | 6/6     | 0           |
| 3         | 4                   | 6/8     | 2 (XXX,XXY) |
| 5         | 4                   | 6/8     | 2 (XXX,XXY) |

### 2. Zorn L-basis in Pauli-taal

De Zorn L-basis (links-vermenigvuldiging) overspant 18 van de 64
driekwart-Pauli operatoren:

```
III, XIX, XIY, XXI, XXZ, XYI, XYX, XYY, XYZ,
XZI, XZZ, YII, YIX, YIY, YIZ, YYX, YYY, ZII
```

De actieve operator-Pauli's bij p≥3 zijn:
```
III, XXX, XXY, XXZ, XYI, XZI, YII, ZII
         ^^^  ^^^
        buiten Zorn
```

### 3. Waarom XXX en XXY lekken

Bij p≤2 bereikt de operator-lightcone niet de grens van de 3-qubit
groep. Alle gates zijn intra-groep, en de gegenereerde Pauli-termen
blijven in de Zorn-subruimte.

Bij p=3 kruist de lightcone de groepsgrens via de ZZ-gate op
(qubit 2, qubit 3). Dit creëert de termen X⊗X⊗X en X⊗X⊗Y die
niet in de standaard Zorn L-basis zitten. Dit is **structureel** —
het gebeurt bij alle gamma/beta waarden.

### 4. Parameter-vergelijking (500 qubits, p=5)

| Representatie        | Parameters | Relatief |
|---------------------|----------:|---------:|
| State MPS (chi=16)  | ~170.000  | 83×      |
| Grouped d=8 MPO     | 42.496    | 21×      |
| d=2 MPO (chi=2)     | 2.056     | 1× (ref) |
| Pauli-sparse (~7/groep) | 1.162 | 0.6×     |

## CONCLUSIES

### Wat WERKT

1. **B7d Heisenberg-MPO**: operator is 8× compacter dan toestand,
   500q in 0.11s, exact tot machineprecisie.
2. **Operator is extreem spaars**: slechts 4 dimensies in 64D ruimte.
3. **Zorn-basis vangt 75%**: 6/8 actieve Pauli's liggen in Zorn-ruimte.
4. **Chi=1 bij p≤2**: de operator is een product van Zorn-elementen
   voor ondiepe circuits.

### Wat NIET werkt

1. **d=8 groepering helpt niet**: meer parameters, trager, geen voordeel.
2. **XXX en XXY lekken structureel**: de Zorn-basis mist 2 van 8 richtingen.
3. **Geen G₂-rotatie kan dit fixen**: de lekkage is in de XX-sector
   (3-qubit correlaties) die niet in de Zorn left-mult structuur past.

### De echte inzicht

De operator leeft in een **4-dimensionale** subruimte. Dat is:
- Minder dan Zorn (8D) → Zorn is te groot
- Meer dan een enkel getal (1D) → er ís structuur
- Precies de dimensie van een 2×2 complexe matrix → chi=2 MPO

De d=2 MPO met chi=2 IS al de optimale algebraïsche representatie.
Elke bond-matrix is 2×2 = 4 parameters, wat exact de 4-dimensionale
operator-ruimte vangt. Dit is effectief een "quaternion-MPO".

## VERVOLGRICHTINGEN

1. **Pauli-sparse MPO**: Sla direct de ~7 Pauli-coëfficiënten op
   in plaats van de volle tensor. Potentieel 2× compacter.
2. **Aangepaste Zorn-basis**: Zoek een G₂-rotatie die XXX en XXY
   opneemt in de basis (vervang 2 van de 18 Zorn-Pauli's).
3. **Operator Trotter**: Gebruik de Pauli-sparsiteit voor directe
   gate-update in Pauli-ruimte, zonder einsum/SVD.

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b8_zorn_mpo.py` | Grouped MPO + Zorn algebra + schaaltest |
| `code/b8_zorn_pauli.py` | Zorn L-basis in Pauli-decompositie |
| `code/b8_optimal_basis.py` | Dimensionaliteitsanalyse + parameter scan |
| `docs/b8_zorn_mpo.md` | Dit document |
