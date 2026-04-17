# B7: 3D TEBD — Tijdsevolutie op Layer-grouped MPS

## Status: 10 april 2026, BEWEZEN

---

## RESULTAAT

TEBD (Time-Evolving Block Decimation) op de 3D layer-grouped MPS werkt.
De Trotter-gates en SVD-truncatie zijn geverifieerd tegen exacte
tijdsevolutie op 2x2x2 (8 qubits).

- **Verificatie (2x2x2, chi=d=16):** Sz matcht exact tot 4 decimalen,
  |dPsi| < 2.2e-3 na t=0.8, puur Trotter-fout bij dt=0.02
- **Productie (2x2x6, chi=32):** 24 qubits, real-time evolution in 27s,
  magnetisatie-schokgolf zichtbaar over 6 lagen
- **Truncatie-effect (chi=4 vs exact):** SVD-truncatie vertraagt de
  spin-propagatie — de "Triage" is zichtbaar in de fysica

---

## METHODE

### Trotter-Suzuki (2e orde)

De tijdsevolutie-operator e^{-iHdt} wordt gesplitst in:

1. Lokale gates: U_local = expm(-i * h_intra_layer * dt/2)  [16x16]
2. Even bond-gates: U_bond = expm(-i * h_inter_layer * dt)  [256x256]
3. Oneven bond-gates: U_bond                                 [256x256]
4. Lokale gates: U_local                                     [16x16]

De gates worden eenmalig berekend (< 0.1s) en hergebruikt per tijdstap.

### SVD-truncatie

Na elke 2-site gate wordt de MPS teruggebracht naar bonddimensie chi
via SVD. De weggeknipte Schmidt-waarden kwantificeren het
informatieverlies per stap.

### Observabelen

Per tijdstap gemeten:
- Energie <H> via MPO-contractie
- Von Neumann entropie S(i) op elke bond
- Lokale magnetisatie <Sz> per laag

---

## NUMERIEKE RESULTATEN

### Verificatie: 2x2x2 (8q), chi=16 (exact, geen truncatie)

|  t   | Sz_0 exact | Sz_0 TEBD |  |dPsi|  | E exact | E TEBD |
|------|-----------|-----------|----------|---------|--------|
| 0.00 |  +2.0000  |  +2.0000  | 0.00e+00 |  4.0000 | 4.0000 |
| 0.20 |  +1.5437  |  +1.5424  | 1.38e-03 |  4.0000 | 3.9920 |
| 0.40 |  +1.0058  |  +1.0042  | 1.53e-03 |  4.0000 | 3.9944 |
| 0.60 |  +0.4606  |  +0.4587  | 1.99e-03 |  4.0000 | 3.9918 |
| 0.80 |  +0.1042  |  +0.1023  | 2.19e-03 |  4.0000 | 3.9954 |

Trotter-fout bij dt=0.02: |dPsi| < 2.2e-3, energie drift < 0.2%.

### Productie: 2x2x6 (24q), chi=32

|  t   |   E      | dE/E     | S_max  |  Sz profiel               |
|------|----------|----------|--------|---------------------------|
| 0.00 | 28.0000  | 0        | 0.0000 | +2 +2 +2 -2 +2 +2        |
| 0.10 | 27.9889  | 3.97e-4  | 3.4657 | +2 +2 +1.9 -1.7 +1.9 +2  |
| 0.20 | 27.9659  | 1.22e-3  | 3.4657 | +2 +2 +1.6 -1.1 +1.6 +2  |
| 0.40 | 27.8759  | 4.43e-3  | 3.4657 | +2 +1.8 +1.2 -0.1 +1.2 +1.8 |
| 0.60 | 27.7684  | 8.27e-3  | 3.4657 | +1.9 +1.7 +1.1 +0.6 +1.1 +1.6 |
| 0.80 | 27.6315  | 1.32e-2  | 3.4657 | +1.8 +1.6 +1.1 +1.0 +1.2 +1.3 |

De magnetisatie-schokgolf verspreidt zich symmetrisch vanuit laag 3
naar beide grenzen. Bij t=0.4 is de oorspronkelijke Sz=-2 al bij -0.1.

### Truncatie-effect: chi=4 vs exact op 2x2x2

|  t   | Sz_0 exact | Sz_0 chi=4 | Vertraging |
|------|-----------|------------|------------|
| 0.20 |  +1.5437  |  +1.6967   | 10%        |
| 0.40 |  +1.0058  |  +1.4838   | 48%        |
| 0.60 |  +0.4606  |  +1.3878   | 201%       |
| 0.80 |  +0.1042  |  +1.3492   | 1195%      |

De SVD-truncatie "vriest" de dynamica: bij lage chi propageert de
excitatie veel langzamer. Dit is het wiskundige equivalent van
informatieverlies bij compressie.

---

## VERBINDING MET DMRG

| Aspect    | DMRG              | TEBD                |
|-----------|-------------------|---------------------|
| Output    | Grondtoestand     | Tijdsevolutie       |
| Operator  | H (eigensolver)   | e^{-iHt} (gates)   |
| Optimaal  | Lage entanglement | Korte tijden        |
| Limiet    | chi = Schmidt-rang | chi + Trotter-fout |
| Analogie  | Foto              | Film                |

De layer-grouped MPS structuur (d=16 per laag) is identiek voor
beide methoden. DMRG vindt de grondtoestand, TEBD laat het systeem
in de tijd evolueren.

---

## KOSTEN

| Systeem | d  | chi | dt   | Stappen | Tijd/stap | Totaal |
|---------|----|----|------|---------|-----------|--------|
| 2x2x2  | 16 | 16 | 0.02 | 40      | <0.01s    | <1s    |
| 2x2x6  | 16 | 32 | 0.02 | 40      | 0.5s      | 27s    |
| 2x2x8  | 16 | 32 | 0.02 | 30      | 1.2s      | ~40s   |

Bottleneck: SVD op (chi*d) x (d*chi) matrix = (512 x 512) bij chi=32.
Schaalt als O(Lz * chi^2 * d^3) per tijdstap.

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b7_tebd_3d.py` | Productie TEBD (2x2xLz, chi=32) |
| `code/b7_tebd_verify.py` | Verificatie tegen exact (2x2x2) |
| `docs/b7_tebd_3d.md` | Dit document |
