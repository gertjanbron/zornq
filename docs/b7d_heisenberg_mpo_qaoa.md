# B7d: Heisenberg-beeld MPO voor QAOA — BEWEZEN

## Status: 10 april 2026, DOORBRAAK

---

## RESULTAAT

Heisenberg-beeld via MPO werkt **exact** voor QAOA-circuits en is
**8× compacter** dan het Schrödinger-beeld.

| Aspect               | Schrödinger (MPS) | Heisenberg (MPO) |
|----------------------|-------------------|-------------------|
| Chi-max (n=9, p=5)   | 16                | 2                 |
| Sum chi              | 60                | 13                |
| Fout vs exact        | 4e-16             | 8e-16             |
| 500 qubits: tijd     | n.v.t. (te duur)  | 0.11s             |
| 500 qubits: geheugen | n.v.t.            | 32 KB             |
| 500 qubits: params   | n.v.t.            | 2056              |
| Truncatie-fout       | —                 | 8e-29 (nul)       |

Alle observabelen geverifieerd: Z_0, Z_4, Z_8, X_0, X_4, X_8 — allemaal
exact tot machineprecisie (fout < 3e-14).

## WAAROM WERKT HET

Bij QAOA-circuits (discrete gates) blijft de operator-entanglement
begrensd. De observable Z_i is lokaal en spreidt zich via nearest-neighbor
gates uit met een lightcone, maar de operator-Schmidt-rank groeit
langzamer dan de state-Schmidt-rank:

| n  | p | State chi_max | Operator chi_max | Ratio |
|----|---|---------------|------------------|-------|
| 5  | 5 | 4             | 2                | 0.5×  |
| 7  | 5 | 8             | 2                | 0.2×  |
| 9  | 5 | 16            | 2                | 0.1×  |

De ratio wordt BETER bij meer qubits: de operator-lightcone bereikt
maar ~5 bonds diep, terwijl de state-entanglement over de hele keten
groeit.

## VERSCHIL MET B7c

B7c testte Heisenberg op **continue Hamiltioniaan-evolutie** (TEBD).
Daar groeit de operator-entanglement onbegrensd: elke Trotter-stap
voegt d² = 4 aan de bond-dimensie toe, en na voldoende stappen
satureerd chi bij chi_max.

B7d test op **discrete QAOA-circuits**. Daar:
- ZZ-gates zijn diagonaal → veranderen bond-dim niet
- Rx-gates zijn 1-qubit → veranderen bond-dim niet
- Alleen de combinatie voegt langzaam entanglement toe
- De operator "vergeet" zijn staart: na de Rx-laag comprimeert SVD
  de operator weer terug naar chi=2

## SCHAALTEST

| n    | <Z_0>       | Tijd   | Chi_max | Params | Geheugen |
|------|-------------|--------|---------|--------|----------|
| 50   | +0.1662143  | 0.01s  | 2       | 256    | 4 KB     |
| 100  | +0.1662143  | 0.02s  | 2       | 456    | 7 KB     |
| 200  | +0.1662143  | 0.06s  | 2       | 856    | 13 KB    |
| 500  | +0.1662143  | 0.11s  | 2       | 2056   | 32 KB    |

Lineaire schaling in n. Geen state vector nodig. 500 qubits in 110ms.

## BUGFIX

De oorspronkelijke implementatie miste een transpose in de 2-site
MPO gate: na de twee einsums voor U^dag (bra) en U (ket) zijn de
indices (cl, k1', b1', k2', b2', cr). De reshape verwacht
(cl, b1', k1', b2', k2', cr). Fix: `Th.transpose(0,2,1,4,3,5)`.

## IMPLICATIES VOOR ZORNQ

De Heisenberg-route is **levend** voor QAOA-circuits:
- Operator chi=2 bij 5 lagen → past in één Zorn-element (dim 8)
- 500 qubits in 32 KB, 0.11s
- Alle verwachtingswaarden exact zonder state vector
- State vector voor 500 qubits zou 2^500 ≈ 10^150 getallen vereisen

Dit opent de weg voor:
1. Zorn-MPO: operator direct in split-octonion representatie
2. QAOA-optimalisatie: variational loop over gamma/beta met MPO-gradienten
3. Multi-observable: meerdere MPO's parallel evolueren

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b7d_mpo_heisenberg.py` | Volledige implementatie + tests |
| `docs/b7d_heisenberg_mpo_qaoa.md` | Dit document |
