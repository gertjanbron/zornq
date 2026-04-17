# B179 Paper-patch

**Context:** na Zenodo-DOI-mint moeten `refs.bib` en `main.tex` worden bijgewerkt om naar de DOI te verwijzen. Dit document bevat de exacte patches.

---

## 1. `docs/paper/refs.bib` — voeg onderaan toe

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

**Vervang** `FILL-IN-AFTER-MINT` door het record-nummer dat Zenodo mint (bv. `15712345`) in beide velden (`doi` en `url`).

---

## 2. `docs/paper/main.tex` — §17 Data and code availability (regel ~741)

**Huidig blok:**
```latex
\section*{Data and code availability}

All code, pinned container specification, raw JSON/CSV artefacts,
LaTeX sources, and figure-generation pipelines are distributed with
this manuscript in the companion repository. The \textsc{ibm\_kingston}
job-bundles (\texttt{job\_id}, counts, and $(\gamma, \beta)$
parameters) are included in \texttt{docs/paper/hardware/jobs/}.
```

**Vervang door:**
```latex
\section*{Data and code availability}

All code, pinned container specification, raw JSON/CSV artefacts,
LaTeX sources, and figure-generation pipelines are publicly
distributed via Zenodo~\cite{zornq2026code} (DOI
\href{https://doi.org/10.5281/zenodo.FILL-IN-AFTER-MINT}{10.5281/zenodo.FILL-IN-AFTER-MINT})
and its GitHub mirror tagged \texttt{paper-v1.0-2026-04-17}. The
\textsc{ibm\_kingston} job-bundles (\texttt{job\_id}, counts, and
$(\gamma, \beta)$ parameters) are included in
\texttt{docs/paper/hardware/jobs/}.
```

**Let op** `\href` vereist `\usepackage{hyperref}` in de preamble. Check of `hyperref` al geladen is; zo niet, voeg toe. Anders laat `\href{...}{text}` vervallen en gebruik alleen `\cite{zornq2026code}` + de DOI als platte tekst.

---

## 3. Recompile-volgorde

```bash
cd docs/paper
latexmk -C                 # clean
latexmk -pdf main.tex      # full rebuild incl. biber
# of manueel:
pdflatex main && biber main && pdflatex main && pdflatex main
```

Verwacht resultaat: 13-page PDF met extra Zenodo-citation in References-sectie.

---

## 4. Snelle sed-commands (optioneel, als DOI bekend is)

```bash
export ZEN_DOI="10.5281/zenodo.15712345"   # <-- vervang met echte DOI

# refs.bib
sed -i "s|FILL-IN-AFTER-MINT|${ZEN_DOI#10.5281/zenodo.}|g" docs/paper/refs.bib

# main.tex — same replacement in the URL
sed -i "s|FILL-IN-AFTER-MINT|${ZEN_DOI#10.5281/zenodo.}|g" docs/paper/main.tex

# CITATION.cff
sed -i "s|FILL-IN-AFTER-MINT|${ZEN_DOI#10.5281/zenodo.}|g" CITATION.cff
```

---

## 5. Verificatie na recompile

- [ ] References-sectie bevat `[NN] Bron, G. ZornQ … Zenodo, doi:10.5281/zenodo.XXXXXXX`
- [ ] Klikbare hyperlink werkt in de PDF (via `hyperref`)
- [ ] Geen `??`-crossrefs
- [ ] `latexmk -pdf` loopt clean zonder undefined-citation warnings
- [ ] `git diff docs/paper/main.tex docs/paper/refs.bib CITATION.cff` toont alleen DOI-vervanging
