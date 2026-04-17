# B179 Runbook voor Antigravity

**Doel:** uitvoeren van B179 Zenodo Reproducibility Archive op Gertjan's laptop met volle git/GitHub/Zenodo-toegang en volledig TeX Live.

**Vereisten vooraf (user-side, niet Antigravity):**
1. GitHub-account met een lege of bestaande repo voor ZornQ (bv. `github.com/<username>/zornq`).
2. Zenodo-account, gekoppeld aan GitHub via `https://zenodo.org/account/settings/github/`.
3. ORCID-ID (niet verplicht maar sterk aangeraden).
4. Git-credentials geconfigureerd (`git config user.name` + `user.email`).
5. TeX Live volledig met biber (al aanwezig aangezien de paper-review-pass succesvol compileerde).

---

## Antigravity-prompt (kopieer-plak 1-op-1)

```
Taak: Uitvoeren van B179 Zenodo Reproducibility Archive voor het ZornQ B4
paper. Leidend document: docs/paper/B179_BUNDLE_SPEC.md. Ondersteunende
documenten: docs/paper/B179_PREFLIGHT_2026-04-17.md,
docs/paper/B179_PAPER_PATCH.md, docs/paper/ZENODO_CHECKSUMS.md.

Werkfolder: C:\Users\Me\Documents\Onderzoeksprojecten\103. ZornQ - Octonionische Quantum Computing
Doel: Zenodo-DOI minten, paper bijwerken met citation, alles committen.

USER-INPUT DIE JE NODIG HEBT VOORDAT JE BEGINT (vraag indien afwezig):
  - GitHub-repo-URL (of maak een nieuwe aan, bv. git@github.com:Gertjan/zornq.git)
  - ORCID-ID van Gertjan (voor .zenodo.json en CITATION.cff)
  - Bevestiging dat Zenodo-GitHub-integratie actief is voor de repo
  - Keuze of license MIT (code) + CC-BY-4.0 (paper) mag, of alternatief

STAP 0 — Lees eerst volledig:
  docs/paper/B179_BUNDLE_SPEC.md
  docs/paper/B179_PREFLIGHT_2026-04-17.md
  docs/paper/B179_PAPER_PATCH.md
  docs/paper/ZENODO_CHECKSUMS.md

STAP 1 — Git-hygiëne (kritisch blokker opheffen)
  1a. Verifieer dat sandbox-Claude `.gitignore`, `.zenodo.json`,
      `CITATION.cff`, `LICENSE` al heeft klaargezet in de repo-root.
      Als één ontbreekt: maak aan volgens de template in BUNDLE_SPEC §4/5/6.
  1b. Fix verified: `code/b7d_mpo_heisenberg.py` is na null-byte-strip
      14205 bytes. Als die fix nog niet is toegepast op jouw side:
        python3 -c "raw=open('code/b7d_mpo_heisenberg.py','rb').read();
        open('code/b7d_mpo_heisenberg.py','wb').write(raw.rstrip(b'\x00'))"
  1c. Controleer git-state:
        git status
        git log --oneline
      Verwachte state: HEAD op 037cd47 "doc(paper): apply pass-1 review fixes",
      werkmap met veel untracked files.
  1d. Stage alles wat door .gitignore NIET wordt uitgesloten:
        git add .
        git status --short   # verifieer: geen results/, runtime_cuda/, transcripts/
  1e. Snapshot-commit:
        git commit -m "chore: snapshot all sources prior to Zenodo B179 archive"
      Noteer de resulterende commit-hash als SNAPSHOT_HASH.

STAP 2 — GitHub-sync
  2a. Voeg remote toe (als nog niet):
        git remote add origin <GITHUB_URL>
      of check/update als er al één is.
  2b. Push:
        git push -u origin master
      Verwacht: beide commits (037cd47 + SNAPSHOT_HASH) gesynct.

STAP 3 — Vraag user om ORCID + bevestig Zenodo-GitHub-koppeling
  3a. Vraag user om zijn/haar ORCID-ID (bv. "0000-0001-2345-6789").
      Update .zenodo.json, CITATION.cff met de werkelijke ORCID
      (vervang "FILL-IN-BEFORE-PUBLISH" en "FILL-IN").
  3b. Vraag user te bevestigen dat op Zenodo de repo "on" staat
      in GitHub-integratie settings. Als niet: open
      https://zenodo.org/account/settings/github/ en schakel in.

STAP 4 — Tag + release + Zenodo-mint
  4a. Maak annotated git-tag:
        git tag -a paper-v1.0-2026-04-17 -m "ZornQ B4 paper-1 — Zenodo archive" <SNAPSHOT_HASH>
        git push origin paper-v1.0-2026-04-17
  4b. Maak GitHub-release: gh release create paper-v1.0-2026-04-17 \
        --title "ZornQ paper-v1.0 (2026-04-17)" \
        --notes-file docs/paper/REVIEW_PASS_2026-04-17.md
      (Of via GitHub-web-UI.)
  4c. Zenodo mint nu automatisch een DOI via de GitHub-integratie (kan
      tot ~5 minuten duren).
  4d. Haal de DOI op: navigeer naar https://zenodo.org/account/settings/github/
      kijk bij de repo, klik de release, noteer het DOI-nummer (bv.
      "10.5281/zenodo.15712345"). Sla op als ZEN_DOI.

STAP 5 — Metadata-review op Zenodo
  5a. Open het Zenodo-record. Klik "Edit".
  5b. Verifieer/update:
        - Title, description, creators (auto uit .zenodo.json)
        - Keywords
        - License: MIT (voor software-record)
        - Related identifiers: arXiv:PENDING kan later worden bijgewerkt
          zodra arXiv-preprint bestaat
  5c. Klik "Publish" — dit is definitief, DOI wordt onveranderbaar.

STAP 6 — Paper-patch doorvoeren
  6a. Pas docs/paper/B179_PAPER_PATCH.md toe:
        - Voeg @misc{zornq2026code, ...} toe aan docs/paper/refs.bib
          met de echte ZEN_DOI
        - Update docs/paper/main.tex §17 "Data and code availability"
          met de cite + URL
  6b. Vervang FILL-IN-AFTER-MINT overal:
        export ZEN_ID="${ZEN_DOI#10.5281/zenodo.}"
        sed -i "s|FILL-IN-AFTER-MINT|$ZEN_ID|g" docs/paper/refs.bib
        sed -i "s|FILL-IN-AFTER-MINT|$ZEN_ID|g" docs/paper/main.tex
        sed -i "s|FILL-IN-AFTER-MINT|$ZEN_ID|g" CITATION.cff
        sed -i "s|FILL-IN|$ZEN_ID|g" .zenodo.json  # alleen als nog ongezet
  6c. Verifieer geen FILL-IN meer in deze files:
        grep -n "FILL-IN" docs/paper/refs.bib docs/paper/main.tex CITATION.cff .zenodo.json
      Verwacht: geen output.

STAP 7 — Recompile + verifieer
  7a. Compile:
        cd docs/paper
        latexmk -C
        latexmk -pdf main.tex
  7b. Verifieer: main.pdf bevat nieuwe Zenodo-citation in References.
      Geen Undefined-citation warnings, geen ??-crossrefs, nog steeds 13 pagina's
      of +/- 1 page.
  7c. git commit:
        git add docs/paper/refs.bib docs/paper/main.tex docs/paper/main.pdf CITATION.cff .zenodo.json
        git commit -m "paper: cite Zenodo archive ${ZEN_DOI}"
        git push origin master

STAP 8 — Checksum-verificatie
  8a. Download het Zenodo-tarball (vanaf het record, "Files" tab,
      download de zip).
  8b. Extract en bereken sha256sum van alle bestanden, vergelijk met
      docs/paper/ZENODO_CHECKSUMS.md:
        unzip zornq-v1.0.0.zip -d /tmp/zenodo_check
        cd /tmp/zenodo_check/zornq-*/
        find . -type f -print0 | xargs -0 sha256sum | sort > /tmp/zenodo_hashes.txt
        # vergelijk handmatig of scripted tegen het manifest
  8c. Verschillen? Rapporteer aan user. Verwacht: alle files in het manifest
      matchen. Kleine extra's (bijv. .git/ info) zijn Zenodo-side normaal.

STAP 9 — Backlog-update
  9a. In docs/zornq_backlog.md, voeg onderaan de HOOG/DAG-9 sectie toe
      (of edit bestaande B179-stub):
        B179 KLAAR (17 apr 2026). Zenodo-DOI 10.5281/zenodo.${ZEN_ID}
        gemint via GitHub-release paper-v1.0-2026-04-17 op snapshot-commit
        ${SNAPSHOT_HASH}. Bundle 437 files, 15.18 MB (code/ 282 files
        3.21 MB, docs/ 75 files 1.37 MB, gset/ 71 files 10.59 MB, root 9
        files ~50 KB). Verificatie: docs/paper/ZENODO_CHECKSUMS.md matcht
        Zenodo-tarball. Paper-cite: main.tex §17 + refs.bib @zornq2026code.
        Repro-anchor voor venue-submission.
  9b. In docs/backlog_prioriteit.md, update de prose-header: vervang
      "B179 Zenodo open dataset release" met
      "B179 Zenodo open release KLAAR (DOI 10.5281/zenodo.${ZEN_ID})".
  9c. git commit -m "backlog: B179 Zenodo KLAAR ${ZEN_DOI}" && git push.

STAP 10 — Rapporteer terug
  Lever een samenvatting met:
    - SNAPSHOT_HASH (snapshot-commit)
    - paper-v1.0-2026-04-17 tag URL op GitHub
    - ZEN_DOI (10.5281/zenodo.XXXX)
    - Zenodo-record-URL
    - main.pdf: pagina-count, commit-hash van de post-cite commit
    - Checksum-verificatie: pass/fail (# matching, # mismatched)
    - Backlog-commit-hash
    - Afwijkingen van de runbook met motivering

CONSTRAINTS:
  - Raak geen files aan buiten scope (alleen wat in BUNDLE_SPEC §1 staat).
  - Push NIET tokens of secrets — controleer via git-diff voor elke commit
    dat er geen *token*/*secret*/*.env in staat.
  - Als Zenodo-DOI-mint faalt (bv. integratie niet actief): STOP en
    rapporteer. Niet zelfstandig via Zenodo-API oplossen tenzij user
    expliciete toestemming geeft.
  - Publieke zichtbaarheid: Zenodo "access_right=open" en GitHub-repo
    public — verifieer dat user dit bewust wil voordat push naar public.

SUCCESS-DEFINITIE:
  Zenodo-DOI resolved + onveranderbaar, main.pdf met werkende citation
  committed + gepusht, backlog geupdate, checksum-verificatie pass.
```

---

## Fallback-paden

### Als Zenodo-GitHub-integratie niet werkt

Gebruik in plaats daarvan de Zenodo-web-upload:

1. Maak op je laptop een tarball:
   ```bash
   cd <parent-dir>
   tar --exclude-vcs \
       --exclude-from="zornq/.gitignore" \
       -czf zornq-v1.0.0.tar.gz zornq/
   ```
2. Upload die tarball direct via `https://zenodo.org/deposit/new`.
3. Vul metadata handmatig in (kopieer uit `.zenodo.json`).
4. Publish. DOI wordt uitgegeven.

### Als checksum-mismatch

- Controleer of `.gitignore` consistent is tussen sandbox en laptop (verschil in welke files werden opgenomen).
- Regenereer `ZENODO_CHECKSUMS.md` op de laptop vóór Zenodo-upload zodat de manifest overeenkomt met de laptop-state.

### Als DOI-mint te lang duurt (>30 min)

- Check de Zenodo-webhook-log op GitHub: Settings → Webhooks → Recent Deliveries.
- Als de webhook rood is: klik "Redeliver".
- Als dat niet helpt: val terug op fallback-pad hierboven.

---

## Review-checklist voor Gertjan na Antigravity-run

- [ ] DOI resolved (klik de DOI-link in CITATION.cff, landt op Zenodo-record)
- [ ] main.pdf bevat zichtbare Zenodo-citation (References-sectie)
- [ ] GitHub-repo public + tag zichtbaar
- [ ] Zenodo-record public + published (niet Draft)
- [ ] Geen FILL-IN-placeholders meer in .zenodo.json / CITATION.cff / refs.bib / main.tex
- [ ] Geen tokens of secrets in git-history (`git log --all -p | grep -i token`)
- [ ] Backlog-files (`zornq_backlog.md`, `backlog_prioriteit.md`) bijgewerkt
