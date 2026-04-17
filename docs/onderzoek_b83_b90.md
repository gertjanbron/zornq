# Onderzoek: zijn B83-B90 nog haalbaar/nuttig?

*17 april 2026 — herwaardering na B10e KLAAR.*

Doel: per item nagaan of het oorspronkelijke GEPARKEERD/NIET HAALBAAR-verdict nog
klopt, gegeven de huidige ZornQ-stand (40+ KLAAR-items, robuuste solver-portfolio,
B130 dispatcher, B132 chemistry, B10e PEPS net afgerond, B103 ZX en B184 ML
gepland). Per item: **Origineel** verdict — **Herwaardering** — **Aanbeveling**.

---

## B83. G-QAOA — Grover Amplitude Amplification op QAOA

**Origineel:** NIET HAALBAAR. Grover-diffusie `2|ψ⟩⟨ψ|−I` vereist projectie op de
volledige entangled state — exponentieel duur in MPS. Threshold-orakel vereist
kennis van OPT.

**Herwaardering:** De architectonische argumenten staan nog. Sinds 2024 zijn er
echter twee varianten die wel haalbaar lijken:

1. *Filter-QAOA / threshold-QAOA* (Boulebnane-Montanaro 2024-stijl): vervang
   Grover-diffusie door post-selection bij sampling. Geen tensorprojectie nodig
   — alleen amplitude-versterking via klassieke rejection sampling. Maar dit
   levert geen extra signaal boven simpel sampling + B70 hotspot-repair, die we
   al hebben.
2. *Bound-driven oracle* via B159 (ILP-Oracle UB): zou een threshold leveren
   zonder OPT te kennen. Maar de winst zit dan bij de ILP-bound, niet bij Grover.

**Aanbeveling:** **Bevestig NIET HAALBAAR** als kwantumalgoritme. Het idee
"sampling + bound + repair" is al gerealiseerd via QAOA-sampling (B130) +
B159-bound + B70-repair. Geen aparte B83-implementatie nodig.

---

## B84. EPE-QAOA — Phase Estimation op QAOA-ansatz

**Origineel:** NIET HAALBAAR. QPE vereist controlled-U^(2^k); bond-dim
verdubbelt per controlled gate. Bij p=2, k=8: 256 gecontroleerde QAOA-lagen.

**Herwaardering:** Niets veranderd in de architecturale beperking. Iteratieve
QPE / Kitaev-variant met 1 ancilla lost het ancilla-aantal op, maar nog steeds
zijn controlled-U^(2^k) operaties nodig. Klassieke shadow-eigenvalue-schatting
(Huang-Preskill) is al gedekt door B10f (KLAAR).

**Aanbeveling:** **Bevestig NIET HAALBAAR.** Geen renovatie zinvol.

---

## B85. Local-Clifford Preconditioner

**Origineel:** GEPARKEERD; onduidelijke winst voor QAOA, ingrijpende wijziging.

**Herwaardering:** Voor 2D MaxCut-QAOA blijft de winst nul: ZZ is al diagonaal,
Hadamard-conjugatie doet niets nuttigs. MAAR — twee scenario's openen waar het
relevant kan worden:

1. *VQE/Heisenberg* (B132 chemistry-tak, KLAAR): lokale basiswissel kan XYZ↔ZZZ
   transformeren en gemiddeld-veld entanglement reduceren. Gertjan heeft H_2 al
   met FCI-precisie; uitbreiding naar grotere moleculen (LiH, BeH_2) zou kunnen
   profiteren.
2. *B162 UCC-ansatz* (MIDDEL OPEN): UCC heeft natuurlijke Clifford+T-decompositie,
   waar local-Clifford preconditioning T-count kan reduceren.

**Aanbeveling:** **Houd GEPARKEERD voor QAOA**, maar markeer als **kandidaat voor
revival samen met B162 UCC**. Niet zelfstandig oppakken.

---

## B86. Topological Gate Pruning / Crystallization Detection

**Origineel:** GEPARKEERD; TDQS (B41) doet pragmatisch alternatief.

**Herwaardering:** B41 (TDQS chi-aware triage, BEWEZEN +4-5%) plus B68
(BFS-diamant lichtkegel, KLAAR) plus B27 (graph-automorfisme, KLAAR) plus B21
(lightcone graph-stitching, KLAAR) plus B118/B119 (sparsifier + Schur)
verzamelen samen alle pragmatische gate-pruning-vormen die zinvol zijn op
laagdiepte-QAOA-circuits. Een formele "crystallization detector" zou
mathematisch elegant zijn maar voegt geen nieuwe winst-bronnen toe.

**Aanbeveling:** **Markeer REDUNDANT** (nieuwe categorie tussen GEPARKEERD en
DOOD). De functionaliteit is volledig gedekt door bestaande items. Verwijder
uit GEPARKEERD-tabel om backlog-ruis te verminderen.

---

## B87. ZX-Calculus Circuit Rewrite

**Origineel:** GEPARKEERD; beperkte winst op gestructureerde QAOA-circuits.

**Herwaardering:** Cruciaal punt — er staat al **B103 (ZX / Phase-Gadget
Rewrite Pass)** OPEN in MIDDEL. B87 en B103 hebben ~90% overlap. B103 is de
juiste framing (phase-gadget pass voor diepere circuits / Hamiltonian-compiler
output). Voor pure 2D-QAOA blijft de winst beperkt zoals oorspronkelijk
beoordeeld; voor B129 Hamiltonian-compiler output (Trotter-decomposities,
UCC-ansätze) kan ZX-rewrite 2-10× T-gate-count reductie geven (PyZX/quizx
literatuur 2024-2025).

**Aanbeveling:** **Markeer DUPLICATE-VAN-B103.** Schrap B87 uit GEPARKEERD;
verbreed B103-scope om expliciet de Hamiltonian-compiler-uitvoer (B129) als
primair doelwit te benoemen. Dat geeft B103 ook een concretere business-case.

---

## B88. Near-Clifford Hybrid Simulatie

**Origineel:** GEPARKEERD; pas interessant bij hoge p.

**Herwaardering:** Stabilizer-rank methoden (Bravyi 2019, Pashayan-Bartlett
2024) schalen als ~k^t met t = aantal T-achtige gates. Voor QAOA p=1 met
γ klein → veel near-Clifford gates → in principe winst mogelijk. **Maar** B133
(Scalability Benchmark, KLAAR) toont dat MPS al 5000 qubits in 0.23s aankan
en break-even pas bij n~14-16 ligt vs state-vector. Een stabilizer-rank
simulator zou dus moeten concurreren met gevestigde MPS-pijplijn op het
specifieke laag-magic-regime — niet duidelijk dat de winst de implementatie
rechtvaardigt.

Mogelijk relevanter voor **B162 UCC** waar de niet-Clifford-fractie expliciet
laag is per Trotter-stap. Connecteert dan ook met B85.

**Aanbeveling:** **Houd GEPARKEERD.** Activeren alleen als B162 UCC opgepakt
wordt EN T-count na B103 ZX-rewrite voldoende laag blijkt om stabilizer-rank
zinvol te maken. Dan een gecombineerde B85+B87→B103+B88 pass voor
chemistry-tak overwegen.

---

## B89. MIS / Rydberg Pivot

**Origineel:** GEPARKEERD; strategische afleiding.

**Herwaardering:** Twee duidelijke kanten:

1. *MIS-via-QUBO*: hoort thuis in **B153 (Beyond-MaxCut QUBO Suite, MIDDEL-HOOG
   OPEN)**. Gertjan heeft daar al de bedoeling MIS, weighted MaxCut, Max-k-Cut
   en portfolio te dekken. Geen aparte B89 nodig.
2. *Native Rydberg-platform* (unit-disk geometrie, Pasqal/QuEra hardware):
   strategisch een ander verhaal. ZornQ kan hier niet winnen op home-turf
   tegen native Rydberg-shots.

**Aanbeveling:** **Markeer ABSORBED-IN-B153** voor het MIS-deel; **schrap het
Rydberg-specifieke deel.** Niet zelfstandig nuttig.

---

## B90. Ervaringsgeheugen (ML-gestuurde Solver Selectie)

**Origineel:** GEPARKEERD; B57 embryonale versie.

**Herwaardering:** De ruimte is sindsdien bezet:
- **B57** (parameter library per graaftype): KLAAR
- **B130** (auto-dispatcher 3-tier, 5 solvers, 54 tests): KLAAR
- **B184** (Instance Difficulty Classifier, MIDDEL OPEN): exact de ML-rol
- **B186** (Solver-Selector als Gepubliceerd Benchmark, MIDDEL-HOOG OPEN):
  publiceerbare evaluatie

Een aparte "ervaringsgeheugen"-laag voegt niets toe boven B184+B186.

**Aanbeveling:** **Markeer ABSORBED-IN-B184+B186.** Schrap uit GEPARKEERD-tabel.

---

## Samenvatting

| # | Origineel | Nieuw verdict | Actie |
|---|-----------|---------------|-------|
| B83 | NIET HAALBAAR | **NIET HAALBAAR** (bevestigd) | Laat staan |
| B84 | NIET HAALBAAR | **NIET HAALBAAR** (bevestigd) | Laat staan |
| B85 | GEPARKEERD | GEPARKEERD-PLUS | Hint: "samen met B162 oppakken" |
| B86 | GEPARKEERD | **REDUNDANT** | Schrap uit backlog |
| B87 | GEPARKEERD | **DUPLICATE-VAN-B103** | Schrap; verbreed B103 |
| B88 | GEPARKEERD | GEPARKEERD-PLUS | Hint: "alleen i.c.m. B162+B103" |
| B89 | GEPARKEERD | **ABSORBED-IN-B153** | Schrap; B153 dekt MIS-QUBO |
| B90 | GEPARKEERD | **ABSORBED-IN-B184+B186** | Schrap; ML-laag is daar |

**Netto-effect:** 4 items kunnen geschrapt worden (B86, B87, B89, B90) → minder
backlog-ruis. 2 items blijven GEPARKEERD met een expliciete trigger (B85, B88).
2 items blijven NIET HAALBAAR (B83, B84).

**Belangrijkste structurele inzicht:** de "Clifford+ZX+near-Clifford"-cluster
(B85+B87+B88) hoort gebundeld te worden onder de **chemistry-tak** (B132 H_2
KLAAR → B162 UCC OPEN). Voor 2D-MaxCut-QAOA voegen ze structureel niets toe
boven wat MPS+TDQS+lightcone al levert; voor moleculaire VQE zijn ze potentieel
relevant. Dat is de natuurlijke trigger om ze te activeren.
