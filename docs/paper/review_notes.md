# Paper-1 Review Notes (B4 MaxCut, post-Zenodo build)

Build: `pdflatex+biber`, 14 pages, 40 bib entries (refs.bib), exit 0.
DOI link: `10.5281/zenodo.19637389` — verified in §Data-and-code-availability.

---

## Compilation diagnostics

- **14 pages**, 499 KB PDF — OK.
- **Bibliography renders**: 40 `@`-entries in `refs.bib`; biber exits clean.
- **Undefined references**: Many `\ref{sec:*}` / `\cref{tab:*}` resolve; some (`tab:selector`, `sec:ilp`, `sec:fwsdp`, `sec:dispatcher`, etc.) resolve on the second pdflatex pass. Final PDF shows all cross-refs working.
- **`zornq2026code`** cite renders correctly in §Data-and-code-availability.
- **Overfull hboxes**: 4 instances (lines ~412, 425, 563, 677). Non-critical but should be tightened before venue submission.

---

## Section-by-section review

### §Abstract (L93–115)
- ✅ Clear claims, all backed by later sections.
- ⚠️ **"certified MaxCut bounds"** — accurate but the *certification* is only ILP-exact for n≤50 and sandwich-bounded beyond. Consider adding "and provable sandwich bounds" to precision.
- ⚠️ **"all experiments are reproducible on a single workstation (GTX 1650-class GPU, 16 GB RAM)"** — this is true for *classical* experiments. The hardware run requires IBM Quantum access. The abstract doesn't distinguish. Minor.

### §1 Introduction (L118–199)
- ✅ Well-structured three-tensions framing.
- ✅ Contributions list is concrete and falsifiable.
- ⚠️ **"first publicly-documented sandwich on myciel3"** (L538) — this claim appears in §Anytime, not Introduction, but the Introduction's framing does not over-claim. Good.

### §2 Problem statement (L201–237)
- ✅ Notation is clean and standard.
- ✅ 4-constraint MILP correctly handles signed instances. Well-documented.

### §3 Pipeline overview (L240–271)
- ✅ Table 1 (tab:components) is a clear summary.
- ⚠️ **FW-SDP "Scales to n ≲ 500"**: With the B176b CGAL-SDP now validated to n=2000 (Task 2 data), this number is conservative. Consider updating to "n ≲ 2000" once B176b tables are integrated. → **Cross-ref to `tables/b176b_scale_table.tex` needed.**

### §4 ILP-oracle (L274–294)
- ✅ Solid. Claims are backed by fig:ilp-scaling.

### §5 FW-SDP sandwich (L297–369)
- ✅ Math is correct. Sandwich derivation clear.
- ⚠️ **"3.9× faster at n=200"** (L349) — no citation for this timing claim. Should reference a specific benchmark table or appendix.
- 🔴 **Missing B176b scaling subsection**: The CGAL-SDP (Yurtsever 2019) scale data is now available in `tables/b176b_scale_table.tex`. A subsection "SDP-scaling" should be added here as `\input{tables/b176b_scale_table.tex}` to document scaling to n=2000.

### §6 MPQS (L372–393)
- ✅ Correct description of BP + lightcone-QAOA.
- ⚠️ **"tree-exact"** (L377) — technically correct (BP is exact on trees). Not an over-claim.

### §7 Twin-width dispatcher (L396–419)
- ✅ Clean description of routing logic.
- ⚠️ **"~540× faster on K₁₈"** (L408) — a specific micro-benchmark claim. Defensible if the number comes from the test suite, but should cite the specific test or appendix.

### §8 QAOA simulation and hardware (L422–453)
- ✅ Three-path design (noiseless/depolarising/cal-mirror) is well-explained.

### §9 Solver-selector results (L456–519)
- ✅ Table 2 (tab:selector) is the paper's strongest empirical contribution.
- ✅ Signed-instance failures are honestly documented with the ‡ footnote.
- ⚠️ **"certifies OPT on 14/14 instances"** (L469) — but one (spinglass2d_L5_s0) is "Approximate". The text clarifies this, but the bold claim "14/14" in the dispatcher paragraph could be misread. Consider: "certifies Exact on 13/14 and Approximate on 1/14."

### §10 Anytime sandwich (L522–542)
- ⚠️ **"first publicly-documented sandwich on myciel3"** (L538) — unverifiable claim. Consider softening to "To our knowledge, the first…" to hedge.

### §11 Hardware validation (L545–619)
- ✅ Table 3 (tab:hardware) is concrete and falsifiable.
- ✅ Cal-mirror prediction agreement (2.1%–2.9%) is well-documented.
- ⚠️ **"Best(HW) = OPT on both instances"** (L613) — true but this is a *sampling* result, not QAOA-expectation optimality. The text clarifies this correctly (L615–619).

### §12 Combined leaderboard (L622–643)
- ✅ Straightforward reporting. No over-claims.

### §13 Discussion (L646–688)
- ✅ **Honest about limitations**: signed FW sandwich, p=1 ceiling, cograph-only exact path.
- ✅ **Octonionic layer disclaimer** (L683–688): Correctly positioned as future work.

### §14 Reproducibility (L691–720)
- ✅ Strong reproducibility section: Docker, seed-ledger, pinned deps, >500 tests.
- ⚠️ The Zenodo DOI is correctly cited and linked.

### §15 Conclusion (L723–745)
- ✅ Summarises without over-claiming.
- ✅ Outlook items are concrete and honest.

### §Data and code availability (L748–757)
- ✅ DOI `10.5281/zenodo.19637389` correctly cited via `\cite{zornq2026code}`.
- ✅ GitHub tag `paper-v1.0-2026-04-17` correctly referenced.

---

## Summary of flagged items

| # | Severity | Section | Issue |
|---|----------|---------|-------|
| 1 | 🔴 CRITICAL | §5 FW-SDP | Missing B176b scale subsection. `\input{tables/b176b_scale_table.tex}` should be added once Task 2 data is merged to main. |
| 2 | ⚠️ MEDIUM | §3 Pipeline | "Scales to n ≲ 500" in Table 1 is conservative; update to n ≲ 2000 with B176b evidence. |
| 3 | ⚠️ MEDIUM | §5 FW-SDP | "3.9× faster" timing claim (L349) needs citation or benchmark reference. |
| 4 | ⚠️ MEDIUM | §9 Selector | "14/14" dispatcher claim (L469) should clarify 13 Exact + 1 Approximate. |
| 5 | ⚠️ LOW | §10 Anytime | "first publicly-documented" (L538) — hedge with "To our knowledge". |
| 6 | ⚠️ LOW | §7 Dispatcher | "~540×" speedup claim (L408) — cite specific test. |
| 7 | ⚠️ LOW | Abstract | "reproducible on a single workstation" doesn't distinguish classical vs. quantum-hardware access. |
| 8 | 📝 STYLE | Multiple | 4 overfull hbox warnings — tighten before venue submission. |

---

## Cross-refs pending B176b integration

Once `b176b-scale` branch is merged to main:
- Add `\input{tables/b176b_scale_table.tex}` in §5 (after the "Rounding and GW upper bound" paragraph, ~L357).
- Update Table 1 (tab:components) FW-SDP "Scales to" column from "n ≲ 500" to "n ≲ 2000".
- Add a sentence: "The companion CGAL-SDP solver (B176b, Yurtsever et al. 2019) extends the sandwich to n = 2000 with provable dual certificates; see Table~\ref{tab:b176b-scale}."
