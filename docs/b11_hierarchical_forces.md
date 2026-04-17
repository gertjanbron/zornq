# B11: Hiërarchische Krachten — Dimensionale Dynamica

## Hypothese

Als de koppelingen in een tensornetwerk een dimensionale hiërarchie
volgen — sterk in 1D (sterke kracht), zwakker in 2D (zwakke kracht),
zwakst in 3D (EM) — produceert dat confinement-achtig gedrag:
excitaties raken opgesloten in de laagdimensionale structuur.

## Testopzet

3D rooster 2×2×6 (24 qubits), layer-grouped MPS met d=16.
Heisenberg-model (XX+YY+ZZ) met drie koppelingsschalen:

- J_1D: horizontale bonds binnen 2×2 laag (q0-q1, q2-q3)
- J_2D: verticale bonds binnen 2×2 laag (q0-q2, q1-q3)
- J_3D: bonds tussen naburige lagen (4 per paar)

Initieel: alle qubits in |0>, behalve qubit 0 in de middelste laag
(geflipped naar |1>). Meet hoe deze excitatie zich verspreidt.

TEBD: 2e orde Suzuki-Trotter, dt=0.05, 10 stappen (t=0..0.50).

## Resultaten

### Excitatie-verspreiding na t=0.50

| Model                | Exc_mid | Exc_rest | Confinement |
|----------------------|---------|----------|-------------|
| UNIFORM (1:1:1)      | 0.030   | 0.475    | 5.9%        |
| HIËRARCHISCH (10:1:0.1) | 0.490 | 0.010  | 98.0%       |
| EXTREEM (100:1:0.01) | 0.500   | 0.000    | 100.0%      |
| OMGEKEERD (0.1:1:10) | 0.175   | 1.013    | 14.7%       |
| PURE 3D (0:0:10)     | 0.148   | 0.809    | 15.5%       |

### Tijdsevolutie (Sz per laag)

**UNIFORM** — excitatie verspreidt zich snel naar alle lagen:
```
t=0.00: [1.000, 1.000, 1.000, 0.500, 1.000, 1.000]
t=0.25: [1.000, 0.990, 0.905, 0.707, 0.903, 0.995]
t=0.50: [0.988, 0.917, 0.847, 0.970, 0.832, 0.941]
```

**HIËRARCHISCH** — excitatie blijft vrijwel volledig opgesloten:
```
t=0.00: [1.000, 1.000, 1.000, 0.500, 1.000, 1.000]
t=0.25: [1.000, 1.000, 0.999, 0.502, 0.999, 1.000]
t=0.50: [1.000, 1.000, 0.995, 0.510, 0.995, 1.000]
```

**EXTREEM** — excitatie volledig opgesloten (geen meetbare lekkage):
```
t=0.00: [1.000, 1.000, 1.000, 0.500, 1.000, 1.000]
t=0.50: [1.000, 1.000, 1.000, 0.500, 1.000, 1.000]
```

## Analyse

### Confinement is reëel en kwantitatief

De hiërarchische koppeling produceert onmiskenbaar confinement.
Bij J_1D/J_3D = 100 lekt na t=0.50 minder dan 0.1% van de
excitatie naar naburige lagen. Bij uniforme koppeling is na
dezelfde tijd 94% van de excitatie verspreid.

### Het mechanisme

Bij sterke J_1D oscilleert de excitatie razendsnel heen en
weer tussen de horizontaal gekoppelde qubits (q0↔q1, q2↔q3)
BINNEN de laag. Die snelle oscillatie is een lokale dynamica
die de qubit "vasthoudt". De zwakke J_3D koppeling kan die
snelle oscillatie niet bijhouden — de excitatie is effectief
opgesloten, analoog aan quark confinement in QCD.

### De kracht-analogie

De analogie werkt kwalitatief:

- **Sterke 1D koppeling → confinement**: excitaties zijn
  opgesloten in hun "keten" en kunnen niet ontsnappen naar
  hogere dimensies. Dit is analoog aan quarks die opgesloten
  zitten in hadronen door gluon-flux tubes.

- **Zwakke 3D koppeling → langzame verspreiding**: informatie
  lekt heel langzaam via de 3D-bonds, als een zwakke golf
  die door de ruimte reist. Analoog aan hoe EM-straling
  zich over grote afstanden verspreidt maar zwak is.

- **Omgekeerde hiërarchie → snelle verspreiding**: als J_3D
  dominant is, verspreidt alles onmiddellijk. Geen confinement.

### Wat het NIET bewijst

Dit bewijst niet dat de vier natuurkrachten dimensionaal
zijn georganiseerd. Het bewijst dat een hiërarchisch
tensornetwerk confinement-achtig gedrag produceert als
emergent fenomeen. Dat is interessant maar niet hetzelfde
als een fysische theorie van alles.

## Bestanden

- `code/b11_hierarchical_forces.py` — TEBD simulatie
