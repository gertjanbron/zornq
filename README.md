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

## Installatie (B181 — 18 apr 2026)

ZornQ is nu pip-installable via de src-layout package in `src/zornq/`:

```bash
# Editable install (ontwikkelaars + reproduceerbare paper-runs)
pip install -e .

# Met optionele solver-backends
pip install -e ".[ilp,qiskit,gpu]"     # SCIP, Qiskit, CuPy
pip install -e ".[dev]"                # pytest, coverage
pip install -e ".[all]"                # alles (zonder gpu)
```

## CLI-gebruik

```bash
# Auto-dispatcher (B130) op DIMACS-bestand
python -m zornq solve graph.clq

# Forceer exacte ILP-oracle (B159)
python -m zornq solve graph.clq --method ilp

# Forceer Frank-Wolfe SDP sandwich (B176)
python -m zornq solve graph.clq --method fw --fw-iter 200

# B184 12-feature vector (JSON of pretty-table)
python -m zornq features graph.clq --pretty

# Gset-benchmark instance direct laden
python -m zornq solve gset:G14 --json

# Version
python -m zornq version
```

## Python API

```python
import zornq

# Load a DIMACS graph
g = zornq.load_dimacs("myciel3.clq")

# Auto-dispatch (B130 kiest ILP / FW / cograph-DP / greedy)
result = zornq.solve_maxcut(g, seed=42)
print(result.cut_value, result.strategy, result.is_exact)

# Primal-dual sandwich (B176 FW-SDP + GW-rounding)
sandwich = zornq.fw_sandwich(g, fw_iter=200)
print(sandwich["cut_lb"], "≤ OPT ≤", sandwich["ub"])

# Exact ILP (B159 HiGHS)
ilp = zornq.ilp_solve(g, time_limit=300.0)
print(ilp["cut_value"], ilp["optimal"])

# B184 difficulty-classifier
clf = zornq.DifficultyClassifier.train(panel="mixed", model="rf")
hint = clf.hint(g)   # 'ilp' | 'fw_sdp' | 'heuristic' | 'greedy'
```

## Documentatie & distributie (B181b — 18 apr 2026)

Sphinx-docs staan in `docs/sphinx/`. Lokaal bouwen:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html
```

Online versie wordt gebouwd door Read the Docs via `.readthedocs.yaml`
(Python 3.12, Ubuntu 22.04, PDF + htmlzip + HTML).

PyPI-publicatie loopt via GitHub Actions (`.github/workflows/publish.yml`):

```bash
# Release knippen + taggen
git tag v0.1.0
git push origin v0.1.0
# → build (sdist + wheel) → twine check → publish naar pypi.org
```

`PYPI_API_TOKEN` en optioneel `TEST_PYPI_API_TOKEN` moeten als repo-secret
zijn ingesteld. `workflow_dispatch` biedt een `testpypi`-pad voor
dry-run. Zie ook `.github/workflows/docs.yml` — iedere PR bouwt de
Sphinx-docs met warnings-as-errors.
