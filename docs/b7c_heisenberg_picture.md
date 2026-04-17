# B7c: Heisenberg-beeld Operator-evolutie — GEFAALD

## Status: 10 april 2026, BEWEZEN DOOD

---

## RESULTAAT

Heisenberg-beeld TEBD (operator evolueren als MPO in plaats van toestand
als MPS) is **dramatisch slechter** dan Schrodinger-beeld voor dit type
systeem. De operator-entanglement groeit ~32x sneller dan de
state-entanglement.

| Aspect               | Schrodinger (MPS) | Heisenberg (MPO) |
|----------------------|-------------------|-------------------|
| Chi bij t=2.0        | 2                 | 64 (gesatureerd)  |
| Totaal chi           | 18                | 360 (20x meer)    |
| Fout t=0.6           | 4.1e-6            | 1.3e-2            |
| Fout t=2.0           | 2.4e-4            | 1.9e-1 (19%)      |
| Snelheid             | 0.0s              | 3.8s              |
| Kosten per stap      | O(n chi^2 d^2)    | O(n chi^2 d^4)    |

## WAAROM

De operator Sz(t) = e^{iHt} Sz e^{-iHt} wordt snel niet-lokaal.
Elke tijdstap past U_dag op de bra en U op de ket toe — dit creëert
correlaties in TWEE indices (bra en ket) tegelijk. De MPO-bonddimensie
groeit als d^2 = 4 per bond per laag, terwijl de MPS-bonddimensie
slechts d = 2 per laag groeit.

Voor dit systeem (single-magnon excitatie in Heisenberg-keten) blijft
de state-entanglement extreem laag (chi=2 volstaat voor machineprecisie),
terwijl de operator-entanglement onmiddellijk naar chi_max=64 satureerd
en dan fouten accumuleert.

### Chi-evolutie

|  t   | chi_S | chi_H | ratio |
|------|-------|-------|-------|
| 0.00 |   1   |    1  |  1.0  |
| 0.05 |   2   |    4  |  2.0  |
| 0.10 |   2   |   16  |  8.0  |
| 0.20 |   2   |   64  | 32.0  |
| 0.40 |   2   |   64  | 32.0  |

Na t=0.2 zit het Heisenberg-beeld al op chi_max en verliest informatie.

---

## CONCLUSIE

Heisenberg-beeld via MPO-evolutie is **geen bruikbare route** voor
ZornQ-simulatie. De d^4-kosten per stap (vs d^2 voor Schrodinger)
en de snellere entanglement-groei maken het in alle opzichten inferieur.

Dit sluit route B6-Heisenberg definitief af. De eerder in B6 gevonden
conclusie (ABA is kwadratisch, niet unitair) en deze MPO-test bevestigen
elkaar: het Heisenberg-beeld biedt geen voordeel voor tensor-netwerk
simulatie van dit type systeem.

---

## NUANCE — LATER HERROEPEN VOOR QAOA

**UPDATE 10 april 2026:** B7d heeft aangetoond dat het Heisenberg-beeld
WEL superieur is voor **discrete QAOA-circuits**. Het verschil:
- TEBD (continu): operator-entanglement groeit onbegrensd → DOOD
- QAOA (discreet): operator-entanglement blijft begrensd → DOORBRAAK

Zie `docs/b7d_heisenberg_mpo_qaoa.md` voor de doorbraak.

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `code/b7c_heisenberg_vs_schrodinger.py` | Vergelijking 3 methodes |
| `docs/b7c_heisenberg_picture.md` | Dit document |
