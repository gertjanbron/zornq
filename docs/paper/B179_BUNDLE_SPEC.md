# B179 Zenodo Bundle-Specification

**Datum:** 17 april 2026
**Doel:** alle inhoud die naar Zenodo moet voor DOI-geminte reproducibility-archive van ZornQ B4 paper-1.
**Triggering commit:** `037cd47` ("doc(paper): apply pass-1 review fixes") is de start-state; een nieuwe snapshot-commit is nodig (zie §0).

---

## 0. Pre-blocker — git-hygiëne vereist

**Huidige git-status:** 1 commit, 5 getracked files (alleen in `docs/paper/`), **264 Python-modules in `code/` untracked**, top-level config (`Dockerfile`, `requirements.txt`, `pyproject.toml`, `environment.yml`, `.dockerignore`, `README.md`) untracked, `docs/*.md` research-notities grotendeels untracked.

**Consequentie:** een `git archive paper-v1.0-2026-04-17` zou een bijna-lege tarball produceren. Er moet eerst een schone **snapshot-commit** komen.

**Plan voor git-hygiëne (stap 0 in B179-runbook):**

1. Installeer de `.gitignore` uit §6 hieronder.
2. `git add` alle bronbestanden minus geïgnoreerde.
3. `git commit -m "chore: snapshot all sources prior to Zenodo B179 archive"`
4. `git tag paper-v1.0-2026-04-17 <snapshot-hash>`
5. Daarna start het Zenodo-proces.

---

## 1. INCLUDE — verplicht in archive

### 1.1 Source code (`code/`)

Alle `.py` bestanden (~264). Expliciet ook:
- `code/auto_dispatcher.py` (post-duplicate-fix, 1094 regels)
- `code/b12_octonion_spinor.py`, `code/b49_anytime_plot.py`, `code/b80_mpqs.py`, `code/b130_*.py` (via auto_dispatcher), `code/b131_quality_certificate.py`, `code/b154_*.py`, `code/b159_ilp_oracle.py` (post-Dag-8b), `code/b160_qsvt.py`, `code/b165b_hardware_submit.py`, `code/b165b_noise_baselines.py`, `code/b165b_parse_results.py`, `code/b170_twin_width.py`, `code/b176_frank_wolfe_sdp.py`, `code/b177_figure_pipeline.py`, `code/b186_solver_selector_benchmark.py`
- `code/audit_trail.py`, `code/evidence_capsule.py`, `code/seed_ledger.py`
- Alle `code/test_*.py` (~65 test-files) — noodzakelijk voor reproduceerbare `pytest` run
- `code/zorn_solver/` pakket en `code/zorn_solver/tests/` subdir

### 1.2 Paper sources (`docs/paper/`)

- `main.tex` (750 regels, post-REVIEW_PASS)
- `refs.bib` (39+ entries, krijgt +1 `zornq2026code` entry na Zenodo-DOI)
- `Makefile`
- `figures/` alle `*.pdf` en `*.tex` (PGFPlots-bronnen)
- `tables/` alle `*.tex` en `*.md` (inclusief `b186_selector_table.*`, `b165b_hardware_table.*`)
- `hardware/` — subdir met `B165b_README.md` + `jobs/` directory (JSON job-bundles van `ibm_kingston` run)
- `data/` alle `*.json` en `*.csv` (B186 selector-results, B49 anytime-trace, B165b hardware-rows)
- `REVIEW_PASS_2026-04-17.md`
- `B179_BUNDLE_SPEC.md` (dit document)
- `ZENODO_CHECKSUMS.md` (wordt gegenereerd in stap 7)
- `main.pdf` (13 pagina's, compile-artifact — host als Zenodo-showcase maar niet in git-tag)

### 1.3 Research notities (`docs/`)

Alle `docs/b*.md` (B10-serie 2D connectivity, B12 algebraic hierarchy, etc.) — ~33 files. Dit is de intellectuele geschiedenis die het paper context geeft.

- `docs/zornq_backlog.md` (5134+ regels, full backlog)
- `docs/backlog_prioriteit.md` (prioriteiten-overzicht)
- `docs/octonion_architectuur.md`
- `docs/batch_benchmark_results.md`
- `docs/lightcone_vergelijking.html` (historic visualisatie)

### 1.4 Top-level config

- `README.md`
- `LICENSE` (MIT — **creëren als die nog niet bestaat**)
- `.gitignore` (nieuw, §6)
- `.dockerignore`
- `Dockerfile`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `.zenodo.json` (nieuw, §2)
- `CITATION.cff` (nieuw, §3 — optioneel maar aanbevolen)

### 1.5 Benchmark-inputs (`gset/`)

Alle Gset instance-files (DIMACS / rudy format) die de benchmarks gebruiken — deze zijn data-bron, geen code, maar essentieel voor replay. Check op licenties van upstream-instances; Gset is public domain.

---

## 2. EXCLUDE — absoluut NIET in archive

### 2.1 Cache / runtime-artifacts

- `__pycache__/` (alle, recursief)
- `*.pyc`, `*.pyo`, `*.pyd`
- `.pytest_cache/`
- `pytest-cache-files-*/`
- `.runtime/`
- `runtime_cuda/` (216 KB CUDA artifacts)
- `scratch_plain_test/`, `scratch_temp_test/`

### 2.2 Results + intermediate

- `results/` (3.3 MB, runtime result-dumps — laten users zelf regenereren voor replay-integriteit)
- `*.npz` (behalve als in `docs/paper/data/` en <1 MB)

### 2.3 LaTeX build-artifacts

- `main.aux`, `main.bbl`, `main.bcf`, `main.blg`, `main.log`, `main.out`, `main.run.xml`, `main.toc`, `main.synctex.gz`
- `*.fdb_latexmk`, `*.fls`
- Maar **WEL** `main.pdf` (die is de deliverable voor Zenodo-showcase)

### 2.4 Sensitieve / private

- `transcripts/` (conversatie-logs, kunnen tokens/context bevatten)
- Elke file matching `*token*`, `*secret*`, `*credential*`, `*.pem`, `*.key`, `.env`
- `code/**/token.txt`, `code/**/api_key*`
- Cowork-session-folders en Claude internal paths

### 2.5 Backup / temp

- `*.bak`, `*.bak-*`, `*.bak-truncated`
- `*~`, `*.swp`, `.DS_Store`, `Thumbs.db`

---

## 3. Zenodo metadata

Zie `.zenodo.json` in §4 voor machine-readable versie. Menselijke samenvatting:

| Veld | Waarde |
|---|---|
| Title | ZornQ: Anytime MaxCut with Sandwich Certificates and Real-Hardware QAOA Validation — Research Code and Paper Sources |
| Description | Companion artefact to the ZornQ B4 paper-1. Includes all Python source code (B12, B80, B130, B131, B154, B159, B160, B165b, B170, B176, B177, B186 modules + full test-suite), LaTeX paper sources, figures, tables, hardware job-bundles from an `ibm_kingston` QAOA run, and a pinned Docker container for reproducible replay. |
| Creators | Gertjan Bron (ORCID: **FILL IN**) |
| Keywords | MaxCut, Frank-Wolfe SDP, QAOA, octonion, reproducible research, twin-width, ILP oracle, sandwich certificate, IBM Quantum, ibm_kingston |
| License | MIT (code) + CC-BY-4.0 (paper/data) — **consistency with LICENSE file required** |
| Access right | open |
| Related identifiers | `arXiv:PENDING` (cite paper-1 arXiv-ID zodra beschikbaar) |
| Funding | (none / personal) |
| Version | 1.0.0 |
| Publication date | 2026-04-17 (of de datum van DOI-mint) |

---

## 4. `.zenodo.json` template

Plaats in repo-root. Zenodo-GitHub-integratie leest dit automatisch bij tag-release:

```json
{
  "title": "ZornQ: Anytime MaxCut with Sandwich Certificates and Real-Hardware QAOA Validation — Research Code and Paper Sources",
  "description": "Companion artefact to the ZornQ B4 paper-1. Includes all Python source code (B12, B80, B130, B131, B154, B159, B160, B165b, B170, B176, B177, B186 modules + full test-suite), LaTeX paper sources, figures, tables, hardware job-bundles from an ibm_kingston QAOA run, and a pinned Docker container for reproducible replay.",
  "creators": [
    {
      "name": "Bron, Gertjan",
      "affiliation": "Independent researcher",
      "orcid": "FILL-IN-BEFORE-PUBLISH"
    }
  ],
  "keywords": [
    "MaxCut",
    "Frank-Wolfe SDP",
    "QAOA",
    "octonion",
    "reproducible research",
    "twin-width",
    "ILP oracle",
    "sandwich certificate",
    "IBM Quantum",
    "ibm_kingston"
  ],
  "license": "MIT",
  "access_right": "open",
  "upload_type": "software",
  "version": "1.0.0",
  "publication_date": "2026-04-17",
  "related_identifiers": [
    {
      "identifier": "arXiv:PENDING",
      "relation": "isSupplementTo",
      "resource_type": "publication-article",
      "scheme": "arxiv"
    }
  ]
}
```

---

## 5. `CITATION.cff` template (optioneel — versterkt GitHub-citation)

Plaats in repo-root:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "ZornQ Research Code (paper-1 companion)"
authors:
  - family-names: Bron
    given-names: Gertjan
    orcid: "https://orcid.org/FILL-IN"
version: 1.0.0
date-released: 2026-04-17
license: MIT
doi: "10.5281/zenodo.FILL-IN-AFTER-MINT"
url: "https://github.com/FILL-IN/zornq"
preferred-citation:
  type: software
  title: "ZornQ: Anytime MaxCut with Sandwich Certificates and Real-Hardware QAOA Validation"
  authors:
    - family-names: Bron
      given-names: Gertjan
  year: 2026
  doi: "10.5281/zenodo.FILL-IN-AFTER-MINT"
```

---

## 6. `.gitignore` (nieuw, plaats in repo-root)

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
build/
dist/

# Pytest / mypy / ruff cache
.pytest_cache/
pytest-cache-files-*/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# LaTeX build
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc
docs/paper/main.log
docs/paper/main.aux
docs/paper/main.bbl
docs/paper/main.bcf
docs/paper/main.blg
docs/paper/main.out
docs/paper/main.run.xml

# Runtime / results
.runtime/
runtime_cuda/
scratch_plain_test/
scratch_temp_test/
results/
*.npz

# Backups
*.bak
*.bak-*
*.bak-truncated
*~
*.swp

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/

# Sensitive — never commit tokens
*token*
*secret*
*credential*
*.pem
*.key
.env
.env.*

# Cowork / Claude session paths (defensive)
transcripts/
.claude/
local-agent-mode-sessions/
```

---

## 7. `refs.bib` entry template

Voeg toe aan `docs/paper/refs.bib` na DOI-mint (stap 6 in runbook):

```bibtex
@misc{zornq2026code,
  author       = {Bron, Gertjan},
  title        = {{ZornQ: Anytime MaxCut with Sandwich Certificates and
                   Real-Hardware QAOA Validation --- Research Code and
                   Paper Sources}},
  year         = {2026},
  month        = apr,
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.FILL-IN-AFTER-MINT},
  url          = {https://doi.org/10.5281/zenodo.FILL-IN-AFTER-MINT},
  note         = {Companion archive to the ZornQ B4 paper-1}
}
```

---

## 8. `main.tex` §17 patch-stub

Zoek in `main.tex` de regel:

```latex
\section*{Data and code availability}

All code, pinned container specification, raw JSON/CSV artefacts,
LaTeX sources, and figure-generation pipelines are distributed with
this manuscript in the companion repository. The \textsc{ibm\_kingston}
job-bundles (\texttt{job\_id}, counts, and $(\gamma, \beta)$
parameters) are included in \texttt{docs/paper/hardware/jobs/}.
```

Vervang de eerste zin door:

```latex
All code, pinned container specification, raw JSON/CSV artefacts,
LaTeX sources, and figure-generation pipelines are publicly
distributed with this manuscript via Zenodo~\cite{zornq2026code},
with a GitHub mirror tagged \texttt{paper-v1.0-2026-04-17}.
```

---

## 9. Bundle-size estimate

| Component | Est. size |
|---|--:|
| `code/` (264 .py) | ~8-12 MB |
| `code/test_*.py` (65 tests) | incl. boven |
| `docs/paper/` (incl. figures/data) | ~5-8 MB |
| `docs/paper/main.pdf` | ~800 KB |
| `docs/*.md` research-notities | ~2-3 MB |
| `gset/` instance-files | dependent on contents — inspect |
| Top-level config (README, Dockerfile, ...) | ~50 KB |
| **Total estimated archive** | **~20-30 MB** |

Zenodo-limit per record is 50 GB, dus ruim voldoende marge.

---

## 10. Post-DOI verificatie-checklist

Na DOI-mint:
- [ ] DOI resolved naar Zenodo-record met correcte metadata
- [ ] `refs.bib` `@misc{zornq2026code}` entry ingevuld met echte DOI
- [ ] `main.tex` §17 bijgewerkt + `\cite{zornq2026code}` present
- [ ] `latexmk -pdf main.tex` recompile OK na `refs.bib` update
- [ ] PDF toont citation naar Zenodo in References sectie
- [ ] Zenodo-zip download geverifieerd tegen `ZENODO_CHECKSUMS.md`
- [ ] `.zenodo.json` + `CITATION.cff` DOI-velden ingevuld met echte DOI
- [ ] GitHub-repo-README linkt naar Zenodo-record
- [ ] `docs/backlog_prioriteit.md` header update: "B179 Zenodo open release KLAAR (DOI + commit)"
- [ ] `docs/zornq_backlog.md` B179-rij toegevoegd met KLAAR-status + DOI
