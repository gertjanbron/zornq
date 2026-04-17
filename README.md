# ZornQ — Octonionische Quantum Computing

## Overzicht
Split-octonionische algebra als informatiecompleet framework voor
quantum computing. Zeven algebraïsche operaties spannen de volledige
Hilbertruimte. Gebouwd op 8-9 april 2026.

## Hoofdresultaat
7 operaties (×, +, −, ÷, Hodge, associator, Jordan triple ABA)
→ rank 512/512 bij 9 qubits. Informatiecompleet.

## Structuur

### docs/
- `octonion_architectuur.md` — Definitief architectuurdocument
- `zornq_backlog.md` — Technische backlog
- `zorn_paper_v2.md` — Paper draft v2 (compleet, 9 apr 2026)
- `zorn_paper_v1.md` — Paper draft v1 (anyon bridge focus)
- `zorn_paper_v0_draft.md` — Eerste paper draft

### code/
- `zornq.py` — ZornQ simulator v3 (892 regels)
- `obo_v4.py` — OBO Optimizer v4
- `cayley_dickson_analysis.py` — 7 decomposities analyse
- `fibmps_zorn_benchmark.py` — FibMPS+Zorn benchmark
- `zorn_qubit_sim_v1.py` — Eerdere simulator versie

### transcripts/
- `journal.txt` — Sessie-index
- `sessie1.txt` — 8 apr: algebra, chiraliteit, 7 decomposities
- `sessie2.txt` — 9 apr ochtend: simulator, gates, Born-overlap
- `sessie3.txt` — 9 apr middag: MPS, BREC, pad-analyse
- `sessie4.txt` — 9 apr avond: doorbraak, rotscalar, 7 operaties

## Kerngetallen
| Metriek | Waarde |
|---|---|
| 3q = 1 Zorn, Born exact | fout = 0 |
| Rotscalar overleving 18q | 100% |
| 7 operaties rank 9q | 512/512 |
| 1 Zorn-product fidelity | 95.58% |
| Split-norm = concurrence | bewezen |
| Gate-compiler fidelity | F > 0.999 |
