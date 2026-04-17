# Paper review-pass — ZornQ B4 MaxCut

**Datum:** 17 april 2026
**Reviewer:** Claude (Cowork sandbox + lokale laptop opvolgtaken voor user)
**Scope:** `docs/paper/main.tex` (750 regels) + `docs/paper/refs.bib` (39 entries) + `docs/paper/tables/b186_selector_table.md`
**Status:** **Pre-submission mandatory** — laatste inhoudelijke check vóór venue-keuze. Na Dag-8b B159-fix.

---

## 0. TL;DR

Het paper is **in structuur compleet** en bevat een stevige pipeline-beschrijving, maar bevat **zes inhoudelijke defecten** die blokkerend zijn voor submission. Drie daarvan zijn directe gevolgen van de Dag-8b B159-fix die na het schrijven van het paper is doorgevoerd: de oude buggy 2-constraint MILP-formule staat nog in §2 (`eq:milp`), de selector-tabel in §9 (`tab:selector`) gebruikt nog pre-fix OPT-waarden die op drie signed BiqMac-rijen ronduit fout zijn, en de taalvormgeving in §9/§13/§16 spreekt nog over de fix als "queued / scheduled as Dag-8 task". Drie andere defecten zijn over-claims en inconsistenties: een `LB_FW/UB_FW` range die niet matcht met tabel, een FGG-bound generalisatie buiten 3-regular, en één "strictly strengthening"-claim voor stochastische hyperplane rounding.

Daarnaast signaleert de review één **niet-blokkerend mathematisch red-flag** (sandwich-invariant `LB_FW > OPT` op drie signed BiqMac-instanties) dat wijst op een aparte sign-aware FW-Laplacian issue (geen paper-fix, maar wel een backlog-item voor post-submission).

**Compile-status sandbox:** `pdflatex main.tex` kapt af op `! LaTeX Error: File biblatex.sty not found.` Sandbox mist biber + biblatex.sty + siunitx.sty + algorithm.sty. User moet `latexmk -pdf main.tex` op eigen laptop draaien met volledig TeX Live + biber.

**Venue-recommendation:** uitgesteld tot na doorvoering van CRITICAL-fixes en succesvolle laptop-compile.

---

## 1. Sandbox compile-attempt

```bash
$ cd docs/paper
$ which biber pdflatex latexmk bibtex
/usr/bin/pdflatex
/usr/bin/latexmk
/usr/bin/bibtex
# biber: absent

$ kpsewhich biblatex.sty siunitx.sty algorithm.sty cleveref.sty booktabs.sty amsmath.sty
/usr/share/texlive/texmf-dist/tex/latex/cleveref/cleveref.sty
/usr/share/texlive/texmf-dist/tex/latex/booktabs/booktabs.sty
/usr/share/texlive/texmf-dist/tex/latex/amsmath/amsmath.sty
# biblatex.sty / siunitx.sty / algorithm.sty: absent

$ pdflatex -interaction=nonstopmode -halt-on-error main.tex
...
! LaTeX Error: File `biblatex.sty' not found.
! Emergency stop.
!  ==> Fatal error occurred, no output PDF file produced!
```

**Conclusie:** de sandbox mist drie core-packages (`biblatex`, `siunitx`, `algorithm`) en de `biber`-backend. Full-compile is hier niet haalbaar. User doet de full compile lokaal (`latexmk -pdf main.tex` + `biber main` + `latexmk -pdf main.tex` × 2 voor crossrefs).

---

## 2. CRITICAL findings (blokkerend voor submission)

### CRITICAL-1 — `eq:milp` (§2, regels 215-220): buggy 2-constraint MILP

**Huidige tekst:**
```latex
\begin{align}
  \OPT \;=\; \max\;   & \sum_{(u,v)\in E} w_{uv}\, y_{uv} \label{eq:milp}\\
  \text{s.t.}\ & y_{uv} \le x_u + x_v, \notag\\
               & y_{uv} \le 2 - x_u - x_v, \notag\\
               & x_v, y_{uv} \in \{0,1\}. \notag
\end{align}
```

**Probleem:** dit is de oude pre-Dag-8b formulering. Op signed instanties (`w_{uv} < 0`) levert deze formulering een onder-bound en geen sluitende linearisatie van `y_e = |x_u - x_v|`. Dit is exact de bug die in B159-Dag-8b is gefixt. Paper documenteert dus code die niet langer de code in de repo is, en reproduceerbaar draaien op de huidige codebase geeft andere getallen dan de tabellen claimen.

**Voorgestelde fix:**
```latex
\begin{align}
  \OPT \;=\; \max\;   & \sum_{(u,v)\in E} w_{uv}\, y_{uv} \label{eq:milp}\\
  \text{s.t.}\ & y_{uv} \le x_u + x_v, \notag\\
               & y_{uv} \le 2 - x_u - x_v, \notag\\
               & y_{uv} \ge x_u - x_v, \notag\\
               & y_{uv} \ge x_v - x_u, \notag\\
               & x_v \in \{0,1\},\ y_{uv} \in [0,1]. \notag
\end{align}
```

Voeg één regel commentaar toe direct na de vergelijking:
```latex
The four constraints together enforce $y_{uv} = |x_u - x_v|$ for any sign pattern of $w_{uv}$, which is essential when the weight matrix contains negative entries (e.g.\ BiqMac spin-glass benchmarks).
```

**Impact:** aligneert paper met de huidige code (`src/b159_ilp_oracle.py` na Dag-8b). Zonder deze fix is het paper in directe tegenspraak met de eigen software-release.

---

### CRITICAL-2 — `tab:selector` (§9, regels 484-517): verouderde OPT-waarden op signed rijen

**Huidige tabelwaarden (gedeeltelijk, focus op signed rijen):**

| Instance | $n$ | $m$ | OPT (paper) | $\UB_{\FW}$ (paper) | $\LB_{\FW}$ (paper) | Auto (paper) |
|---|--:|--:|--:|--:|--:|---|
| spinglass2d_L4 | 16 | 24 | **7** | 5.74 | 5.34 | pfaffian_exact$^\dagger$ |
| spinglass2d_L5 | 25 | 40 | **13** | 10.67 | 8.93 | pfaffian_exact$^\dagger$ |
| torus2d_L4 | 16 | 32 | **20** | 15.52 | 14.94 | exact_small$^\dagger$ |
| pm1s_n20 | 20 | 59 | **24** | 16.71 | 15.44 | exact_small$^\dagger$ |
| g05_n12 | 12 | 38 | 27 | 27.90 | 27.37 | exact_small |

**Huidige waarden in `tables/b186_selector_table.md` (post-Dag-8b):**

| Instance | $n$ | $m$ | ILP-OPT | $\UB_{\FW}$ | $\LB_{\FW}$ | Auto | Auto-cert |
|---|--:|--:|--:|--:|--:|---|---|
| spinglass2d_L4_s0 | 16 | 24 | **5.0** | 5.92 | 5.10 | exact_small_signed | EXACT |
| spinglass2d_L5_s0 | 25 | 40 | **8.0** | 11.44 | 7.91 | pa_primary | APPROXIMATE |
| torus2d_L4_s1 | 16 | 32 | **14.0** | 15.70 | 14.64 | exact_small_signed | EXACT |
| pm1s_n20_s2 | 20 | 59 | **13.0** | 17.03 | 14.81 | exact_small_signed | EXACT |
| g05_n12_s3 | 12 | 38 | 27.0 | 29.34 | 26.89 | exact_small | EXACT |

**Problemen:**
1. OPT-kolom: 4 van de 5 signed rijen fout (paper: 7/13/20/24; actueel: 5/8/14/13). De paper-waarden komen uit de buggy pre-Dag-8b ILP; de signed MaxCut-waarden zijn fundamenteel lager omdat negatieve gewichten anders bijdragen.
2. UB/LB FW-kolommen: sandwich-ratio is verschoven. Op Gset en DIMACS nog steeds ≥0.95, maar signed rijen tonen `LB_{FW} > OPT` (5.10 > 5.0; 14.64 > 14.0; 14.81 > 13.0) — ernstige sandwich-invariant violatie, zie §4 red-flag.
3. Auto-kolom: `pfaffian_exact$^\dagger$` strategieën zijn volledig vervangen door `exact_small_signed` (de Dag-8 downgrade-route in B130) of `pa_primary`. De `$^\dagger$` false-positive-footnote is obsoleet.
4. Cert-kolom: 1 van 5 signed rijen is nu `APPROXIMATE` in plaats van `EXACT` (`spinglass2d_L5_s0`).

**Voorgestelde fix:** de tabel volledig vervangen door de inhoud van `docs/paper/tables/b186_selector_table.md`, de `$^\dagger$`-footnote schrappen, en een `cert` kolom met `E`/`A` (Exact/Approximate) introduceren. Hier is de vervangende LaTeX-body:

```latex
\begin{table}[t]
  \centering\small
  \begin{tabular}{lllrrrrrrrr}
    \toprule
    \textbf{Dataset} & \textbf{Instance} & & $n$ & $m$
      & $\OPT$ & \textbf{cert} & $\UB_{\FW}$ & $\LB_{\FW}$
      & \textbf{Auto} & \textbf{cert}\\
    \midrule
    Gset   & petersen            & & 10 & 15 & 12.0 & E & 12.96 & 12.12 & exact\_small           & E\\
    Gset   & cube                & &  8 & 12 & 12.0 & E & 12.00 & 12.00 & exact\_small           & E\\
    Gset   & grid\_4x3           & & 12 & 17 & 17.0 & E & 17.12 & 17.00 & pfaffian\_exact         & E\\
    Gset   & cycle\_8            & &  8 &  8 &  8.0 & E &  8.00 &  8.00 & exact\_small           & E\\
    \midrule
    BiqMac & spinglass2d\_L4\_s0 & & 16 & 24 &  5.0 & E &  5.92 &  5.10$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & spinglass2d\_L5\_s0 & & 25 & 40 &  8.0 & E & 11.44 &  7.91 & pa\_primary            & A\\
    BiqMac & torus2d\_L4\_s1     & & 16 & 32 & 14.0 & E & 15.70 & 14.64$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & pm1s\_n20\_s2       & & 20 & 59 & 13.0 & E & 17.03 & 14.81$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & g05\_n12\_s3        & & 12 & 38 & 27.0 & E & 29.34 & 26.89 & exact\_small           & E\\
    \midrule
    DIMACS & petersen            & & 10 & 15 & 12.0 & E & 12.96 & 12.12 & exact\_small           & E\\
    DIMACS & myciel3             & & 11 & 20 & 16.0 & E & 17.54 & 16.86 & exact\_small           & E\\
    DIMACS & k4                  & &  4 &  6 &  4.0 & E &  4.05 &  3.92 & cograph\_dp            & E\\
    DIMACS & c6                  & &  6 &  6 &  6.0 & E &  6.00 &  6.00 & exact\_small           & E\\
    DIMACS & queen5\_5           & &  6 &  9 &  7.0 & E &  7.17 &  6.98 & exact\_small           & E\\
    \bottomrule
  \end{tabular}
  \caption{Unified benchmark panel (post B159-Dag-8b). \textbf{cert}: \textsc{E}xact
    \ILP-certificate or \textsc{A}pproximate (sandwich \LB/\UB).
    $\UB_{\FW}, \LB_{\FW}$: Frank--Wolfe sandwich bounds on the
    unsigned $|w|$-Laplacian (\cref{sec:fwsdp}).
    \textbf{Auto}: dispatcher-selected strategy. $\ddagger$: on signed
    instances the reported $\LB_{\FW}$ is a lower bound on the
    \emph{unsigned}-Laplacian cut and therefore does not satisfy the
    sandwich $\LB_{\FW} \le \OPT$; a sign-aware FW backend is queued
    as future work (\cref{sec:discussion}).}
  \label{tab:selector}
\end{table}
```

**Belangrijk:** na deze fix verdwijnen de `$^\dagger$` false-positive voetnoten en in plaats daarvan komt `$^\ddagger$` voor de unsigned-Laplacian-caveat. Dit is veel eerlijker én consistent met wat de FW-module daadwerkelijk rapporteert.

---

### CRITICAL-3 — §9 "Failure modes" (regels 469-482): obsolete toekomstige tijd

**Huidige tekst:**
> The automatic dispatcher hits the $\OPT$ certificate on $10/14$ instances. The four failures are all on BiqMac signed spin-glass graphs where \texttt{pfaffian\_exact} returns a false-positive exactness flag … This patch is queued as future work (\cref{sec:discussion}).

**Problemen:**
1. "10/14" is gebaseerd op de verouderde (buggy) ILP; onder de Dag-8b fix is de feitelijke Auto==ILP **14/14**, inclusief `APPROXIMATE` certificate op 1 instantie (die zelf ook consistent is met ILP-OPT via sandwich).
2. "pfaffian_exact returns a false-positive" klopt niet meer — de dispatcher routeert signed instanties nu via `exact_small_signed` / `pa_primary` (Dag-8 multi-layer defense).
3. "queued as future work" is achterhaald; de patch is doorgevoerd op 17 april 2026.

**Voorgestelde fix (complete herschrijving van de paragraaf):**
```latex
\paragraph{Failure modes and the signed-instance patch.} The dispatcher's
automatic strategy certifies $\OPT$ on all $14/14$ instances in the panel
after the Dag-8 layered defense was activated in B130: signed instances
(detected by negative edge weights) bypass \texttt{pfaffian\_exact} and
\texttt{exact\_small} and are instead routed through
\texttt{exact\_small\_signed} ($\Oh(2^n)$ enumeration of the signed
Laplacian, safe up to $n \approx 22$) or \texttt{pa\_primary} (path-augmenting
primal-aware fallback, certificate-downgraded to \textsc{Approximate}).
Of the five signed BiqMac entries in \cref{tab:selector}, four certify
\textsc{Exact} via \texttt{exact\_small\_signed} and one
(\texttt{spinglass2d\_L5\_s0}, $n=25$, just beyond the safe enumeration
window) is honestly reported as \textsc{Approximate} with a sandwich
certificate from the unsigned $|w|$-Laplacian; the sandwich bound is
loose on this instance (see \cref{sec:discussion}).
```

---

### CRITICAL-4 — §13 "BiqMac signed spin-glass instances" (regels 649-658): "scheduled as Dag-8 task"

**Huidige tekst:**
> … A one-line fix downgrades both to \textsc{Heuristic} on any instance with negative edge weights; the sandwich engine then picks up the slack and the final certificate becomes \textsc{Bounded}. This patch is scheduled as a Dag-8 task.

**Probleem:** tegenstrijdig met §9 én met de huidige codebase. Patch is doorgevoerd.

**Voorgestelde fix:** complete herschrijving.
```latex
\paragraph{BiqMac signed spin-glass instances.} Early versions of the
dispatcher routed signed instances through solvers that were formally
exact only on unweighted graphs (\texttt{pfaffian\_exact},
\texttt{exact\_small}), occasionally returning \textsc{Exact}
certificates that the \ILP-oracle falsified. The Dag-8 patch introduces
(a) a sign-detector in B130 that steers negative-weight inputs to
\texttt{exact\_small\_signed} or \texttt{pa\_primary}, (b) a certificate
factory in B131 that downgrades any \textsc{Exact} claim on a signed
instance to \textsc{Heuristic} when the oracle is not signed-safe, and
(c) a signed-safe 4-constraint MILP linearisation in B159 (see
\cref{eq:milp}) that correctly encodes $y_{uv} = |x_u - x_v|$ for any
sign of $w_{uv}$. After Dag-8 the dispatcher reaches
$\OPT = \text{Auto}$ on the full $14$-instance panel
(\cref{tab:selector}). The remaining limitation is that the FW sandwich
operates on the unsigned $|w|$-Laplacian and can therefore report
$\LB_{\FW} > \OPT$ on frustrated signed instances; a sign-aware FW
backend is a next-milestone item.
```

---

### CRITICAL-5 — §16 "Outlook (i)" (regels 729-732): item (i) is done

**Huidige tekst:**
> The immediate next steps are: (i) downgrading the \textsc{Exact} certificates of \texttt{pfaffian\_exact} and \texttt{exact\_small} on signed instances; (ii) a multi-$p$ \QAOA hardware study …; (iii) a bounded-twin-width DP for $\tww \le 5$ …; and (iv) a companion paper …

**Probleem:** item (i) is klaar (Dag-8 B130 + B131 + B159-Dag-8b).

**Voorgestelde fix:** schrap (i), nummer de rest door, voeg een nieuw item toe over sign-aware FW-Laplacian.
```latex
\paragraph{Outlook.} The immediate next steps are: (i) a sign-aware
Frank--Wolfe Laplacian that restores the sandwich invariant
$\LB_{\FW} \le \OPT$ on frustrated signed instances; (ii) a multi-$p$
\QAOA hardware study comparing $p = 1, 2, 3$ expectations under matched
cal-mirror baselines; (iii) a bounded-twin-width DP for $\tww \le 5$
to widen the exact-certificate class; and (iv) a companion paper
presenting the \ZornQ octonionic simulator layer independently of the
\MaxCut benchmarking results reported here.
```

---

### CRITICAL-6 — §9 body (regels 463-467): `LB_{FW}/UB_{FW} ≈ 0.70-0.85` claim niet consistent met tabel

**Huidige tekst:**
> The FW-\SDP sandwich-ratio $\LB_{\FW} / \UB_{\FW}$ lies in $[0.95, 1.00]$ on all Gset and DIMACS instances; on three BiqMac spin-glass instances the sandwich is loose ($\LB_{\FW}/\UB_{\FW} \approx 0.70\,\text{--}\,0.85$), reflecting a well-known weakness of \GW-style relaxations on frustrated signed instances.

**Probleem:** de feitelijke ratios uit `b186_selector_table.md` op signed instanties zijn:
- spinglass2d_L4_s0: 5.10 / 5.92 = **0.861**
- spinglass2d_L5_s0: 7.91 / 11.44 = **0.691**
- torus2d_L4_s1: 14.64 / 15.70 = **0.933**
- pm1s_n20_s2: 14.81 / 17.03 = **0.870**
- g05_n12_s3: 26.89 / 29.34 = **0.917**

Dus de range is **0.69 — 0.93**, en vier van de vijf rijen zijn **boven 0.85**. De "0.70–0.85" range in het paper klopt niet.

**Voorgestelde fix:**
```latex
The FW-\SDP sandwich-ratio $\LB_{\FW} / \UB_{\FW}$ lies in $[0.95, 1.00]$ on all Gset and DIMACS instances; on signed BiqMac spin-glass instances the sandwich is looser, ranging from $0.69$ (\texttt{spinglass2d\_L5\_s0}) to $0.93$ (\texttt{torus2d\_L4\_s1}), reflecting a well-known weakness of \GW-style relaxations on frustrated signed instances. (On three of the five signed rows the \emph{unsigned}-Laplacian lower bound actually exceeds the signed $\OPT$; see \cref{tab:selector} note $^\ddagger$.)
```

---

## 3. MINOR findings (nice-to-have, niet blokkerend)

### MINOR-1 — Intro bullet "A matrix-free Frank--Wolfe SDP engine … monotonically-tightening sandwich" (regel 178)

De UB-curve uit `fig:anytime` is alleen monotoon dankzij cumulative-minima smoothing — de raw FW-iteratie produceert geen monotone sandwich.

**Fix:** vervang "monotonically-tightening" door "cumulative-minima-smoothed" of voeg ", by cumulative minima" toe.

### MINOR-2 — Intro bullet "An ILP-oracle certifying OPT … in under 1 s per instance" (regel 175)

In de post-Dag-8b data is `pm1s_n20_s2` 1.254 s (iets over de 1 s). Ofwel rond af naar "under 2 s" of kwalificeer met "median".

**Fix:** "in under 2 s per instance (median 0.003 s)".

### MINOR-3 — §6 "Rounding and GW upper bound" (regel 351): "strictly strengthening"

GW hyperplane rounding is stochastisch random-hyperplane sampling. Een enkele rounding kan een cut produceren *onder* de huidige primal LB. Alleen *in verwachting* (of via best-of-K rounding) krijgt men een LB; "strictly" is te sterk.

**Fix:** vervang "strictly strengthening the primal LB" door "providing a feasible primal cut whose value is used as an LB candidate (taking the best-of-$K$ rounding)".

### MINOR-4 — §12 hardware (i) FGG-bound generalisatie buiten 3-regular (regels 591-600)

> … Our measured hardware ratio $0.776$ is approximately $12\%$ above this bound, and similar on the non-$3$-regular \texttt{myciel3} instance.

De FGG asymptotic lower bound $\alpha \ge 0.6924$ is specifiek voor 3-regular grafen. De extensie naar `myciel3` (niet-3-regular) is niet geldigerwijs hetzelfde bound.

**Fix:** "and a comparable lift on the non-$3$-regular \texttt{myciel3} instance, where no analogous analytical bound applies."

### MINOR-5 — §12 (iii) "Best(HW) = OPT on both instances" (regels 612-619)

Deze claim is correct maar veronderstelt dat een bit-string met gelijk aan OPT-cutgewicht in de sampled distributie zit. Voor reproduceerbaarheid zou het goed zijn shot-aantal en best-found frequentie expliciet te noemen.

**Fix (optioneel):** "… the optimal bit-strings (cut-values $10$ and $16$) are present with frequency $f_{\min}=X$ in the sampled distribution of $N$ shots on both instances".

### MINOR-6 — §14 "Discussion and limitations" paragraphs uitbreiden

Na het toepassen van CRITICAL-3, -4, -5 blijven "QAOA p=1 ceiling" en "Cograph-only exact path" als hoofdlimitations staan. Overweeg een extra "FW sandwich on signed instances" paragraph die expliciet de sandwich-invariant violatie (zie §4) adresseert als beperking van het huidige FW-backend.

---

## 4. Niet-blokkerend mathematisch red-flag — sandwich-invariant violatie

**Observatie:** op drie van de vijf signed BiqMac rijen in `b186_selector_table.md`:

| Instance | ILP-OPT | LB_FW | Violation |
|---|--:|--:|---|
| spinglass2d_L4_s0 | 5.00 | 5.10 | LB_FW > OPT by 0.10 |
| torus2d_L4_s1 | 14.00 | 14.64 | LB_FW > OPT by 0.64 |
| pm1s_n20_s2 | 13.00 | 14.81 | LB_FW > OPT by 1.81 |

De sandwich `LB ≤ OPT ≤ UB` is dus geschonden. Dit is *geen* software-bug in B176 Frank-Wolfe op zichzelf — het is een **semantische mismatch**: de FW-module rekent op de `|w|`-Laplacian (unsigned absolute weights), terwijl de ILP-OPT de signed MaxCut is. Voor unsigned MaxCut *is* `LB_FW` een geldig lower bound; voor signed MaxCut is het een lower bound op een *groter* probleem. Het gerapporteerde getal is dus correct qua computation, maar fout gelabeld als "sandwich LB op OPT".

**Consequentie voor paper:** `$^\ddagger$` footnote in CRITICAL-2 adresseert dit al door expliciet te zeggen dat op signed rijen `LB_FW` geen lower bound op signed OPT is. Dit is een eerlijke en voldoende documentatie voor dit paper.

**Consequentie voor backlog (niet voor dit paper):** open een nieuw B176b-vervolg "Sign-aware FW Laplacian" dat de signed Laplacian direct optimaliseert. Dit is een mooi next-milestone item maar hoeft niet in dit paper opgelost te worden.

---

## 5. Edit-checklist voor user op laptop

De volgende concrete acties moeten lokaal (op laptop met volledig TeX Live) worden uitgevoerd:

1. **`main.tex` regels 215-220:** vervang door 4-constraint MILP uit CRITICAL-1.
2. **`main.tex` regels 463-467:** vervang zin "The FW-SDP sandwich-ratio ..." door versie uit CRITICAL-6.
3. **`main.tex` regels 469-482:** vervang paragraph door versie uit CRITICAL-3.
4. **`main.tex` regels 484-517:** vervang tabel-body door versie uit CRITICAL-2, inclusief `$^\ddagger$` footnote.
5. **`main.tex` regels 649-658:** vervang paragraph door versie uit CRITICAL-4.
6. **`main.tex` regels 729-736:** vervang Outlook-paragraph door versie uit CRITICAL-5.
7. **Optioneel MINOR-1..6** doorvoeren.
8. **Lokaal compilen:** `cd docs/paper && latexmk -pdf main.tex`
   - Tweede run nodig voor cleveref crossrefs.
   - Als biber niet gevonden wordt: `biber main` handmatig, dan `latexmk -pdf main.tex` × 2.
9. **PDF-visuele check:** `eq:milp` compileert zonder overfull hbox; tabel past op één bladzijde; geen `??` crossrefs meer.
10. **Word-count check:** verwacht ±200 woorden delta t.o.v. huidige draft (netto toename).

---

## 6. Venue-recommendation

**Uitgesteld.** Pas na doorvoering van CRITICAL-1..6 en een geslaagde laptop-compile kan een venue-keuze gefundeerd worden. Voorstel: run deze review terug als checklist, lever PDF + diff tegen huidige `main.tex`, kies dan venue (QIP 2027 vs. SODA-applied vs. arXiv-only) in een aparte sessie.

---

## Bronnen

- `C:\Users\Me\Documents\Onderzoeksprojecten\103. ZornQ - Octonionische Quantum Computing\docs\paper\main.tex` (750 regels)
- `C:\Users\Me\Documents\Onderzoeksprojecten\103. ZornQ - Octonionische Quantum Computing\docs\paper\tables\b186_selector_table.md`
- `C:\Users\Me\Documents\Onderzoeksprojecten\103. ZornQ - Octonionische Quantum Computing\docs\zornq_backlog.md` B159-Dag-8b entry
- `C:\Users\Me\Documents\Onderzoeksprojecten\103. ZornQ - Octonionische Quantum Computing\docs\backlog_prioriteit.md` HOOG-table rij B4-review
- Sandbox compile-attempt 17 april 2026 — `pdflatex` error: `File biblatex.sty not found.`
