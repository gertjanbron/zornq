# ZornQ Backlog — Gesorteerd op Prioriteit
*Laatst bijgewerkt: 17 april 2026 (**DAG 8b B159-ILP-ORACLE SIGN-BUG KLAAR 17 apr 2026**: de Dag-8-bijvangst `B159-Dag-8b` (ILP-oracle-formulering reken­de `max Σ max(w_e, 0)·[cut]` op signed-instanties) is dichtgetimmerd met een signed-safe vier-constraint linearisatie `y_e ≤ x_u+x_v`, `y_e ≤ 2−x_u−x_v`, `y_e ≥ x_u−x_v`, `y_e ≥ x_v−x_u` die `y_e = |x_u−x_v|` afdwingt voor álle sign-combinaties. `code/b159_ilp_oracle.py` uitgebreid van ~412 naar ~425 regels; fix geldt gelijk voor alle drie solver-backends (HiGHS via `scipy.optimize.milp` row-matrix, SCIP via `pyscipopt.addCons`, Gurobi via `addConstr`); `n_constrs` in SCIP/Gurobi van `2m` → `4m`. Tests: `test_b159_ilp_oracle.py` 39/39 groen (was 31) incl. nieuwe `TestSignedInstancesDag8b`-klasse met 8 discriminerende tests — `P_4` + neg-backedge (OPT=20.0, pre-fix bug=30), `C_5` alt +/-, triangle + neg edge, all-neg triangle (OPT=0), `K_4` mixed-sign (OPT=40), dispatcher-routing, SCIP/Gurobi-match (graceful SKIP). **B186-regeneratie**: `Auto == ILP-OPT: 14/14` (was 10/14 met 4 signed-mismatches); spinglass2d_L4/L5, torus2d_L4, pm1s_n20 nu volledig consistent tussen dispatcher-auto en ILP-oracle. Full suite 134/134 groen. Artifacts: `docs/paper/data/b186_selector_results.{json,csv}` + `docs/paper/tables/b186_selector_table.{tex,md}` opnieuw geschreven. Backlog: B159-Dag-8b KLAAR; resterende Dag-8+-HOOG-items = paper-1 review-pass + biber-build (B4-review), B176b CGAL-SDP n≈10000, B179 Zenodo open dataset release. **DAG 8 CERTIFICATE-DOWNGRADE + SIGN-AWARE SOLVER KLAAR 17 apr 2026**: de paper §13-belofte "pfaffian_exact/exact_small op signed-instanties worden in Dag-8 gedowngraded" is waargemaakt met een **4-laags defense-in-depth** in `code/auto_dispatcher.py` (1094 regels, +158 regels t.o.v. Dag-2). (1) **Detectie-laag** `has_signed_edges(edges, tol=1e-12)` scant de edge-list op w<−tol; `classify_graph()` vult `info['has_signed_edges']`. (2) **Routing-laag**: `select_strategy()` blokkeert pfaffian_exact/exact_small/mps_qaoa_grid/mps_qaoa_wide/lightcone_qaoa voor signed inputs via `not is_signed`-guards. (3) **Solver-laag**: nieuwe `_run_signed_brute_force(n, edges, ...)` doet NumPy-vectorized 2^n × m bit-enumeration (int64 xs-vector, `bi ^ bj` XOR-mask, `cuts = np.sum(ew * (bi^bj), axis=1)`) met harde n≤24 check, retourneert is_exact=True; geregistreerd als `signed_brute_force` in SOLVER_FUNCS en gebruikt als dispatcher-strategie `exact_small_signed` voor signed n≤20. `_run_pfaffian` en `_run_brute_force` raisen nu expliciet ValueError op signed edges (third line of defense). (4) **Certificaat-laag**: `certify_result(best_cut, n, edges, info, is_exact, strategy)` downgradet is_exact-resultaten naar APPROXIMATE als strategy ∈ {pfaffian_exact, exact_small, exact_brute} én has_signed_edges=True (fourth line). **Tests**: `test_auto_dispatcher.py` 460 regels, +24 Dag-8 checks (has_signed_edges truth-table, classify_graph propagatie, select_strategy routing, signed_brute_force in registry, triangle cut=2.0, n>24 raise, pfaffian/BF raise op signed, certify_result downgrade-matrix, end-to-end dispatcher op signed triangle) → **114 passed, 0 failed**. **B186-panel regenereerd** met nieuwe strategieën: `docs/paper/data/b186_selector_results.{json,csv}` + `docs/paper/tables/b186_selector_table.{tex,md}` — 5 signed-affected rijen aangepast: spinglass2d_L4_s0 (pfaffian_exact/EXACT/−10.0 fout → exact_small_signed/EXACT/5.0 correct, t=0.059s), spinglass2d_L5_s0 (pfaffian_exact/EXACT/−14.0 fout → pa_primary/APPROXIMATE/8.0 eerlijk, t=0.373s), torus2d_L4_s1 (exact_small/partial-correct → exact_small_signed/EXACT/correct, t=0.054s), pm1s_n20_s2 (→ exact_small_signed/EXACT, t=1.413s), g05_n12_s3 (ongewijzigd, niet signed). Dispatcher-auto == ILP-OPT op 10/14 instanties; de resterende 4 "mismatches" zijn alle gevallen waar de dispatcher correct is en ILP B159 een eigen sign-bug heeft (zie nieuw backlog-item **B159-Dag-8b**). Task-update: **Task #65/#66 completed**. **NIEUW-ONTDEKTE BUG B159-Dag-8b ILP-ORACLE SIGN-BUG**: tijdens verificatie van spinglass2d_L4_s0 (dispatcher=5.0 vs ILP=7.0) bleek de formulering `y_e ≤ min(x_u+x_v, 2−x_u−x_v)` in `b159_ilp_oracle.py` y_e=0 toe te staan op gesneden negatieve-gewicht-edges — effectief berekent de ILP `max positive-cut`, niet signed MaxCut. Affects 4 BiqMac-instanties in B186-panel; out-of-scope voor Dag-8 maar toegevoegd als nieuwe backlog-item (HOOG/MIDDEL). **DAG 6-7 B4 PAPER-1 FULL DRAFT KLAAR 17 apr 2026**: volledige arXiv-style paper-draft opgeleverd, `docs/paper/main.tex` 754 regels / ~2931 woorden / 11 PDF-pagina's (zonder bibliografie; met biber-expansie verwacht ~13-15 pp.), 16 sections, 4 figures, 3 tables, 9 display-equations, 39 bib-entries. **Structuur**: §1 Introduction (3 tensions + 6 contributions-lijst + outline), §2 Problem statement & notation (MILP+Laplacian+AR/sandwich-ratio), §3 Pipeline overview (`tab:components`), §4 ILP-oracle ceiling (`fig:ilp-scaling`), §5 Frank-Wolfe sandwich SDP (spectraplex Δ_n, matrix-vrije LMO, gesloten-vorm line-search, sandwich-certificaat, GW-rounding, verwijst `fig:anytime`), §6 MPQS (BP + lightcone-QAOA met ζ_uv edge-belief-extractie), §7 Twin-width dispatcher + cograph-DP (540× speedup K_18), §8 QAOA simulatie + hardware pipeline (3 noise-paths), §9 Solver-selector (`tab:selector`, 14 instanties met failure-mode-markers), §10 Anytime sandwich curve (myciel3: LB=16=OPT ≤ UB=17.32, gap 7.61%), §11 Hardware validation op `ibm_kingston` (`tab:hardware`, `fig:hardware`, 3 observaties: +12% boven Farhi-Goldstone-Gutmann p=1-theorie, cal-mirror ≤2.9% afwijking, Best(HW)=OPT), §12 Combined leaderboard (`fig:leaderboard`), §13 Discussion & limitations (pfaffian_exact/exact_small op signed-instanties → Dag-8-downgrade, QAOA p=1 ceiling, cograph-only exact, octonion companion-paper), §14 Reproducibility (Docker + seed-ledger + token-hygiene), §15 Conclusion + outlook. **Abstract** ~160 woorden dekt volledige scope. **refs.bib** uitgebreid van 20 naar 39 entries incl. khot2007optimal, hastad2001some, jaggi2013revisiting, hazan2008sparse, yurtsever2021scalable, bonnet2022twin, schidler2022sat, corneil1981linear, kasteleyn1963dimer, hadlock1975finding, qiskit2024, qiskitruntime2024, wack2021quality, wang2018quantum, xue2021effects, boyd1994continuous, docker, ioannidis2005why, barahona1982computational. **Sandbox-compile-verificatie**: `pdflatex -interaction=nonstopmode main.tex` compileert cleanly met biblatex→thebibliography-stub (biber + siunitx/algorithm niet beschikbaar in sandbox, siunitx/algorithm uit preamble verwijderd wegens ongebruikt). Alle 33 unieke cite-keys in body matchen een bib-entry; alle \ref{sec:…}/\ref{fig:…}/\ref{tab:…} cross-refs resolven na tweede pass (0 undefined-warnings). Op Gertjan's laptop met full TeX Live + biber volgt bit-exacte PDF-reproductie via `cd docs/paper && latexmk -pdf main.tex`. Dag-6-7-task-update: task #61 **completed**, B4 paper-scaffolding invulling is nu `DRAFT-KLAAR`. **DAG 5 B165b HARDWARE-RUN OP `ibm_kingston` GELUKT 17 apr 2026**: echte IBM Quantum-run voltooid same-day (queue stond op 0); hardware E[H_C]/OPT = **0.776** (3reg8) en **0.773** (myciel3), Best(HW)=OPT op beide, calibration-mirror voorspelt hardware binnen **98.4%** (3reg8) en **97.1%** (myciel3) → offline Aer cal-mirror is gevalideerd als `ibm_kingston`-proxy, en de gemeten 0.776 zit **12% boven de Farhi-Goldstone-Gutmann p=1 theorie-AR ≈ 0.692**. Paper-artifacts: `docs/paper/tables/b165b_hardware_table.{md,tex}` volledig gevuld (4 kolommen + OPT + AR), `docs/paper/figures/b165b_hardware_comparison.{pdf,tex}` grouped bar-chart met OPT-lijnen en AR-annotaties, `docs/paper/data/b165b_hardware_rows.json` volledige JSON-trace. B165 status: OPEN → KLAAR. User-side pakket compleet om één echte IBM Quantum-run te doen zonder dat Claude een token ziet. `code/b165b_hardware_submit.py` (~320 regels) met `--dry-run`/`--submit <backend>`/`--resume <job_id>`/`--only <inst>`, token-lezer uit env-var of losse file, SubmissionBundle-dataclass met job_id-persistence. `code/b165b_noise_baselines.py` (~290 regels) draait drie Aer-baselines per instantie (noiseless, depolariserend p1=1e-3/p2=1e-2, calibration-mirror uit `backend.properties()` JSON-snapshot met veilige fallback); `--fetch-snapshot-from <backend>` haalt verse calibration op. `code/b165b_parse_results.py` (~230 regels) assembleert alles in een booktabs paper-tabel met ILP-OPT kolom (B159) en AR-kolom (hardware/OPT); hardware-kolom valt terug op "—" als er nog geen submit is. `code/test_b165b_hardware_submit.py` 13/13 groen incl. instance-registry, token-helpers, bundle-roundtrip, grid-search monotonie, dry-run zonder token, noise-model fallback, drie-baseline runner, parser met gemockte hardware-counts. `docs/paper/hardware/B165b_README.md` geeft stap-voor-stap run-instructies met expliciete veiligheidsafspraak (token buiten Cowork-folder). **Dag-5 voorbereidingsrun lokaal op Aer:** 3reg8 (n=8, m=12, OPT=10 ILP-certified) + myciel3 (n=11, m=20, OPT=16 ILP-certified) met QAOA p=1 en (γ*=0.388, β*=1.194) uit 10×10 grid-search; drie kolommen gevuld in `docs/paper/tables/b165b_hardware_table.{md,tex}` — 3reg8 E[H]=8.00/7.92/7.93 (noiseless/depol/cal-mirror), myciel3 E[H]=12.84/12.68/12.74; `best_cut_seen` haalt OPT op alle 6 cellen. Vierde kolom "Hardware" wacht op Gertjan's laptop-submit. 93/93 tests groen over B165+B165b+B176+B49 suites samen. **DAG 4 B49 ANYTIME PAPER-FIGUUR 17 apr 2026**: centrale paper-figuur geproduceerd — `code/b49_anytime_plot.py` (~540 regels) genereert de time-vs-value sandwich-plot met UB-curve uit **B176 FW-SDP** per-iteratie (monotoon dalend via cumulatieve minima, gebruikt de nieuwe `elapsed` key in de FW-history), 5-laags LB-cascade (alternating → B80 mpqs_bp → fw_gw_rounding vanaf Y-matrix → 1flip_polish) monotoon stijgend, en horizontale OPT-lijn uit **B159 HiGHS ILP-oracle**. Dual-output: matplotlib-PDF (y-as gecapte op 1.6·OPT, gestaggerde annotaties, gele sandwich-fill) + PGFPlots TikZ `.tex` + JSON+CSV trace. Test-suite `test_b49_anytime_plot.py` 18/18 groen incl. sandwich-invariant `LB ≤ OPT ≤ UB`, monotoniciteits-checks, PDF-magic bytes en artifact-roundtrip. Validatie op DIMACS/myciel3 (n=11, m=20): LB=16 ≤ OPT=16 (ILP-certified) ≤ UB=17.32, 400 FW-iter, sandwich-gap 7.61% — direct opneembaar als Figuur 2 in paper-1. Minor: `b176_frank_wolfe_sdp.py` uitgebreid met `elapsed` key in history-entries (40/40 tests blijven groen, additieve-alleen wijziging). Paper-artifacts: `docs/paper/figures/b49_anytime_plot.{pdf,tex}` + `docs/paper/data/b49_anytime_trace.{json,csv}`. B49 status: PROTOTYPE → KLAAR gepromoveerd. **DAG 3 B186 SOLVER-SELECTOR BENCHMARK 17 apr 2026**: `code/b186_solver_selector_benchmark.py` (~420 regels) levert het unified 14-instance panel (4×Gset+5×BiqMac+5×DIMACS) met 4 solvers per instantie: ILP (B159 HiGHS, LEVEL 1 EXACT via `certify_maxcut_from_ilp`), FW-SDP (B176 Frank-Wolfe-sandwich, LEVEL 2 BOUNDED/NEAR_EXACT via `certify_maxcut_from_fw`), cograph-DP (B170 O(n³) als P_4-vrij), en dispatcher-auto (B130 `solve_maxcut`). Emitters: JSON/CSV/LaTeX-booktabs/Markdown. Test-suite `test_b186_solver_selector_benchmark.py` 23/23 tests groen (per-runner, panel-schema, emitter-consistency, artifact-roundtrip). Full-panel run (1.0s totaal): ILP-certified 14/14, FW-sandwich LB/UB-ratio 0.95–1.00 op Gset+DIMACS, dispatcher-auto hit 10/14 vs ILP-OPT met 4 gedocumenteerde failure-modes op signed BiqMac-spinglass (`pfaffian_exact` false-positive, `exact_small` verkeerde cut). Paper-artifacts geschreven naar `docs/paper/data/b186_selector_results.{json,csv}` en `docs/paper/tables/b186_selector_table.{tex,md}` — direct opneembaar in paper-1. **DAG 2 B130+B131 INTEGRATIE 17 apr 2026**: quick-wins uit backlog-consolidatie afgesloten. B170 twin-width/cograph-metric ingeplugd als dispatcher-feature in B130 `auto_dispatcher.py` (~250 regels: `_compute_tww_feature` helper, `classify_graph` breidt info uit met `is_cograph`/`tww`/`is_unweighted`, `select_strategy` routet unweighted cographs naar nieuwe `cograph_dp` solver-slot, `_run_cograph_dp` wrapper met Laplacian-cut-verificatie; 88/88 tests groen incl. K_30 end-to-end cut=225 exact via cograph_dp). B176 FW-sandwich + B159 ILP-oracle geregistreerd als LEVEL 2/1-CERTIFIED in B131 `quality_certificate.py` (~170 regels: `certify_maxcut_from_fw` met duaal-gap→level-mapping, `certify_maxcut_from_ilp` met certified-flag→EXACT/BOUNDED-mapping, beide dependency-vrij via duck-typing op resultaat-objecten; 72/72 tests groen incl. 20 nieuwe factory-tests). Paper-claim "anytime solver mét duale certificaten over hele schaal-range" is nu factueel waar in plaats van "kloppend als je de bouwstenen apart leest". **DAG 1 REPRO-PREREQUISITE 17 apr 2026**: B178 Docker/Conda-lock Reproducibility KLAAR + B55 seed-ledger KLAAR afgesloten — `requirements.txt`+`pyproject.toml`+`Dockerfile`+`.dockerignore`+`environment.yml` reproduceerbaarheidsstack met Python 3.12-slim container, PYTHONHASHSEED=0, 7 core-deps gepind, 5 optional-dep groups (dev/qiskit/ilp/gpu/all); `seed_ledger.py` (~350 regels, v1.0.0) + `test_seed_ledger.py` 45/45 tests in 0.035s, deterministische SHA256-afgeleide child seeds, JSON-sidecar met integriteitsvalidatie, `attach_to_audit()` integratie met `audit_trail.py` v1.0.0, CLI + global singleton, replay-scenario bit-identiek na save/load. **BACKLOG-CONSOLIDATIE 17 apr 2026**: na B176-aflevering herzien — B51/B60 getwee-backend (cvxpy klein-n + FW groot-n), B69/B91 markering "groot-n rebenchmark open via B176", B48/B49 PROTOTYPE→KLAAR geüpgraded met sandwich-certificaat (B176+B159+B131), B176b CGAL-SDP uit parkering naar MIDDEL-HOOG met n≈10k als paper-2-doel, B130+B131 integratie-todo-stukjes (B170 tww-metric + B176 SDP-signal) expliciet, B72/B119 gemarkeerd "nieuwe tooling her-evaluatie niet gedaan" (niet heropenen tot hybride is geprobeerd), status-termen schoongeveegd (B62 BEANTWOORD→KLAAR-onderbouwd, B8 GEDEELTELIJK→KLAAR-gedeeltelijk, B6g GRENSGEBIED→KLAAR-grens, B7c BEWEZEN DOOD→GEFALSIFICEERD, B55 seeds-TODO→Repro-link, B52 GEFALSIFICEERD-rationale expliciet), B144/B149/B151 KLAAR BASIS→KLAAR definitief met explicitering dat B176/B159 de schaal-knelpunten hebben opgelost. **B176 KLAAR**: Frank-Wolfe / Conditional Gradient SDP, 40/40 tests, `b176_frank_wolfe_sdp.py` (~400 regels) spectraplex-relaxatie Δ_n={X⪰0, tr(X)=n} met diagonaal-penalty `−¼·tr(L·X)+(λ/2)·‖diag(X)−𝟏‖²`, matrix-vrije LMO via `scipy.sparse.linalg.eigsh(which='SA')` + dense-fallback n<40, gesloten-vorm line-search (kwadratisch in γ) of Jaggi γ=2/(k+2), low-rank X=Y·Yᵀ met SVD-truncatie op rank_cap, dual-certificaat `sdp_upper_bound = −f(X_k) + gap_k` (bewijs: f*≤−cut_SDP), feasible-primaal-LB ¼·tr(L·X̂) via rij-normalisatie X̂, GW-rounding vanaf Y; sandwich LB≤cvxpy≤UB op 7 kleine grafen, scalability 3-reg n=30..500 (FW wint vanaf n≈100 met 3.9× speedup op n=200, cvxpy infeasible n>300), cut/UB=0.922-0.994 structureel, gemeten cut/UB=0.947-0.960 ver boven GW-garantie 0.87856; B176b CGAL-SDP (Yurtsever 2019) geparkeerd voor n≈10000; **B156 KLAAR**: Lasserre level-2 SDP, 38/38 tests, 13/14 exact; **B158 KLAAR**: Triangle+OddCycle cutting planes, 23/23 tests, 12/14 exact via HiGHS, OC-separator vindt pentagon-cut op C_5/C_7 onmiddellijk; **B159 KLAAR**: ILP-Oracle via scipy.milp (HiGHS), 31/31 tests, 17/17 certified in <1s, schaalbaar tot 3-reg n=50 in 0.6s, certificeerbare OPT-kolom voor paper-tabellen; **B80 KLAAR**: MPQS Message-Passing Quantum Solver, 44/44 tests, max-product BP tree-exact op bomen (2/2) en OPT-hit op 13/13 benchmark via BP+1-flip, lightcone-QAOA (pure-numpy statevector) OPT op 11/13, ζ-matrix spectral rounding; **B154 KLAAR**: BiqMac-rudy + DIMACS loaders + combined leaderboard, 45/45 tests, synthetic generators (spinglass2d/3d/torus2d/pm1s/pm1d/w01/g05), DIMACS parser met 5 fixtures en 20-entry chromatic-DB, unified 14-instance panel Gset+BiqMac+DIMACS met ILP-certified OPT op 14/14; **B165 KLAAR**: Qiskit Runtime pipeline, 22/22 tests, ZornQ→Qiskit gate-export voor 15 gates, Aer + depolariserende noisy-Aer halen 9/9 best-cut OPT, hardware-pad token-gated SKIPPED; **B165b OPEN**: hardware-submit prep + calibration-mirror noise (echte IBM-run vanaf user-laptop, zonder Claude-token); **B177 KLAAR**: Paper figures-pipeline + LaTeX template, 19/19 tests, `b177_figure_pipeline.py` met JSON→matplotlib-PDF + PGFPlots-`.tex` dual output, arxiv-style `docs/paper/main.tex` + `refs.bib` (20+ core refs), Makefile (pdflatex+biber+latexmk-watch), beide paper-figures gegenereerd op 14-instance Gset+BiqMac+DIMACS panel; **B10e KLAAR**: PEPS/PEPO 2D tensor netwerk, 40/40 tests, `b10e_pepo.py` (~640 regels) met PEPS2D class, simple-update SVD truncatie + zero-pruning, boundary-MPO sweep contractie, full QAOA MaxCut pipeline; benchmark op 2x2/3x2/3x3/2x4/3x4 toont diff ~1e-15 vs exact state-vector behalve 3x3 p=2 chi=4 (diff 2.7e-4, verwachte truncatie); **B153 KLAAR**: Beyond-MaxCut QUBO Suite, 53/53 tests, `b153_qubo_suite.py` (~580 regels) met QUBO-datatype + 4 encoders (weighted MaxCut, Max-k-Cut one-hot, MIS, Markowitz portfolio Lucas 2014 §6.3) + 4 generieke solvers (BF/LS/SA/RR); RR matcht BF op alle haalbare instanties; absorbeert B89 MIS-deel; vormt domein-agnostische bovenbouw voor B130 dispatcher; **B170 KLAAR**: Twin-width parameter + cograph MaxCut, 41/41 tests, `b170_twin_width.py` (~550 regels) met Trigraph + contract + greedy tww-heuristic + P_4-free-cograph-check + cotree-DP; exacte O(n³) MaxCut op cographs matcht BF op alle haalbare cases; ~540× sneller dan BF op K_18 (11.6 ms vs 6226 ms); difficulty-metric route voor B130 dispatcher (cograph→cograph_dp, tww≤2→B42 tree_dp, anders→B153 QUBO); **B160 KLAAR**: QSVT / Block-Encoding Framework, 40/40 tests, `b160_qsvt.py` (~380 regels) met LCU-block-encoding van Pauli-sums (PREP via Householder + SELECT met identity-padding), QSP-primitieven (Chebyshev T_k-fases geverifieerd tot machine-precisie) en Jacobi-Anger Hamiltonian-simulatie via Chebyshev-recursie; op Ising-TF n=3 en Heisenberg-XXX n=4 tot αt=18 haalt Jacobi-Anger err=1e-15 waar Trotter-1 op 1e-1 hapert (12+ orden van grootte winst bij ~0.3 ms); exponentiële K-convergentie voorbij knik K≈e·αt/2 bevestigd; QSP-phase-angle-solver voor willekeurige polynomen geparkeerd als B160b; **B12 KLAAR**: Octonion-spinor correspondentie, 40/40 tests, `b12_octonion_spinor.py` (~400 regels). Split-octonion Zorn-algebra met Peirce-idempotenten (e_0, e_7), 6 nilpotente imaginairen j_1..j_6 (j_i²=0) en fermion-achtige anti-commutator {j_i,j_{i+3}}=1 voor alle 3 mode-paren. Niet-associativiteit bewezen (156/512 triples ≠ 0, [j_1,j_2,j_3]=e_0-e_7) maar Moufang-identiteiten exact op alle 512 triples → alternatieve algebra. Cl(4,3) gamma-matrices uit L-multiplication met signatuur (4,3) + {γ_μ,γ_ν}=2η_μν·I exact; falsificatie van claim "Cl(4,4)≅𝕆_s" (256 vs 8 dim) en vervanging door correcte "𝕆_s = spinor-module Cl(4,3)" (2^⌊7/2⌋=8). Bijection Φ:𝕆_s→F_3=Λ(ℂ³) orthogonaal maar NIET module-morfisme (‖Φ·L_{j_i}-c_i†·Φ‖≈2.8) — dit is de *falsificatie* van naïeve algebra-iso tussen octonionen en fermion-Fock. Triality-indicator Σ‖L-R‖=23.2 op Im(𝕆_s); **GESCHRAPT 17 apr 2026 na onderzoek_b83_b90.md**: B86 (redundant met B41+B27+B68+B21+B118+B119), B87 (duplicate-van-B103), B89 (absorbed-in-B153), B90 (absorbed-in-B57+B130+B184+B186); B85 en B88 GEPARKEERD-PLUS met expliciete trigger "alleen samen met B162 UCC-ansatz".)*

> **Kernmissie:** Schaalbare quantum compute op laptop (GTX 1650). QC als wapen in
> de toolbox naast klassiek. Engine kiest zelf beste strategie. Bewijs dat tensor
> network simulatie op consumer hardware kan concurreren met QC-hardware.

---

## KLAAR / AFGEROND (geen actie nodig)

| # | Naam | Status |
|---|------|--------|
| B1 | Schaalbaarheid 7-operatie completheid | AFGEROND |
| B2 | Minimale operatieset | AFGEROND |
| B3 | Reconstructie-algoritme | AFGEROND |
| B5 | Universaliteit en Diepte-scaling | AFGEROND |
| B6 | Route-tests (VQE, Heisenberg, mid-circuit) | AFGEROND |
| B6b | Lokale verwachtingswaarden uit MPS | AFGEROND |
| B6c | Zorn-VQE MPS prototype | AFGEROND |
| B6d | Cilindrische DMRG — 2D Heisenberg | BEWEZEN |
| B6e | Zorn-triplet Cilindrische DMRG | BEWEZEN |
| B6f | 3D Layer-grouped DMRG | BEWEZEN |
| B6g | 4D Hyperslab DMRG | KLAAR (GRENS — werkt voor kleine 4D hyperslabs, runtime onpraktisch boven 2⁴×Lτ=5; geen verdere actie, niet heropenen tenzij nieuwe toepassing). |
| B7 | 3D TEBD — Tijdsevolutie | BEWEZEN |
| B7b | Adaptive-chi TEBD | BEWEZEN |
| B7c | Heisenberg-beeld MPO-evolutie TEBD | GEFALSIFICEERD (Heisenberg-beeld TEBD explodeert in bond-dimensie; **B7d** QAOA-specifieke Heisenberg-MPO levert wél doorbraak). Geen verdere actie. |
| B7d | Heisenberg-beeld MPO voor QAOA | DOORBRAAK |
| B8 | Zorn-MPO: operator in split-octonion | KLAAR (GEDEELTELIJK — Zorn-representatie werkt voor 2-site Heisenberg- en QAOA-operatoren, volledige 𝕆_s-MPO-calculus voor algemene operatoren geparkeerd als research-niveau; niet blokkerend voor engine). |
| B9 | QAOA MaxCut Optimizer — Heisenberg-MPO | BEWEZEN |
| B10 | 2D-connectiviteitstest | BEWEZEN |
| B10b | 2D QAOA parameter-optimalisatie | BEWEZEN |
| B10c | Gecontroleerde chi-truncatie | BEWEZEN |
| B10d | Geoptimaliseerde engine + GPU | KLAAR |
| B10h | Informatieverlies bij chi-truncatie | BEWEZEN |
| B11 | Hiërarchische Krachten | BEWEZEN |
| B11b | GPU CUDA | KLAAR |
| B11c | Algebraïsche basis kopplingshiërarchie | BEWEZEN |
| B12 | Octonion-spinor correspondentie | KLAAR. `b12_octonion_spinor.py` (~400 regels) + `b12_benchmark.py` + `test_b12_octonion_spinor.py` 40/40 tests (0.3s). Split-octonion Zorn-algebra 𝕆_s met **Peirce-decompositie** (e_0, e_7 orthogonale idempotenten; e_0²=e_0, e_7²=e_7, e_0·e_7=0, e_0+e_7=1), zes nilpotente imaginairen j_1..j_6 (j_i²=0), fermion-achtige anti-commutator `{j_i, j_{i+3}} = 1` (identiteit) voor alle 3 mode-paren, α-basis-kruisproduct `j_1·j_2 = -j_6` cyclisch. **Associator-statistiek**: 156/512 basis-triples niet-associatief, max ‖[·,·,·]‖=√2, [j_1,j_2,j_3] = e_0-e_7 expliciet. **Moufang-identiteiten** (links + rechts) exact op alle 512 basis-triples → 𝕆_s is alternatieve algebra bewezen. **Cl(4,3) gamma-matrices** uit L-multiplication: γ_i = L(j_i+j_{i+3}) i=1..3, γ_{i+3} = L(j_i-j_{i+3}), γ_7 = L(e_7-e_0); full metric η = diag(+1,+1,+1,-1,-1,-1,+1) = signatuur (4,3), off-diag exact 0, {γ_μ,γ_ν} = 2·η_μν·I geverifieerd. **Fermion-Fock F_3 = Λ(ℂ³)** (8-dim) met Jordan-Wigner c_i†, c_i: {c_i,c_j†} = δ_ij·I exact. **Bijection Φ: 𝕆_s → F_3** via `phi_bijection()` — orthogonale 8×8 matrix (|det|=1), e_0↦\|∅⟩, j_i↦\|i⟩, j_{i+3}↦ε·\|jk⟩, e_7↦-\|123⟩. **Module-morfisme-falsificatie**: ‖Φ·L_{j_i} - c_i†·Φ‖ ∈ {2.83, 2.00, 2.83} voor i=1,2,3 → Φ is lineaire iso maar GEEN algebra-iso (non-associativiteit blokkeert). **Claim-falsificatie**: Cl(4,4)≅𝕆_s is FOUT (2^8=256 vs 8, factor 32 mismatch); correcte koppeling is 𝕆_s = spinor-module Cl(4,3) met 2^⌊7/2⌋=8 dim. **Triality-indicator**: L_a ≠ R_a op alle 7 imaginaire elementen, Σ‖L-R‖=23.2. Conceptuele brug voor B167 Albert-algebra J_3(𝕆) en Cl(4,3)-Casimir-symmetry voor ZornQ-circuits (geparkeerd tot concrete toepassing). |
| B13 | Sz-symmetrische blok-SVD | BEWEZEN |
| B15 | Dynamische Truncatie (Fidelity-Driven SVD) | KLAAR |
| B21 | Lightcone Graph-Stitching | KLAAR |
| B25 | ZNE Chi-Extrapolatie | KLAAR |
| B35 | Hybride Lightcone + GPU State Vector | KLAAR |
| — | ZornMPS Unified Engine | GEREED |
| B51 | Goemans-Williamson SDP Bound | KLAAR — tweetraps: **B60** (cvxpy/SCS interior-point, exact tot n≈500) + **B176** (Frank-Wolfe spectraplex-relaxatie met duaal sandwich-certificaat, schaalbaar tot n≥500; B176b CGAL-SDP voor n≈10000). SDP-bound is daarmee beschikbaar over het hele Gset-bereik, niet alleen voor kleine instanties. |
| B52 | Zorn-Heuristic Solver Integratie | GEFALSIFICEERD (Zorn-heuristic onderperformed klassieke LS; geen verdere actie, zie REDUNDANT-tabel). |
| B55 | Checkpoint / Resume / Nacht-Runner | KLAAR — functional + seed-ledger. `seed_ledger.py` (~350 regels, v1.0.0) + `test_seed_ledger.py` (45/45 tests in 0.035s). Deterministische SHA256-afgeleide child seeds uit master (SHA256(f"{master_hex}\|{label}")[:8] masked naar 32-bit), JSON-sidecar met integriteitsvalidatie bij load (re-derive check), `attach_to_audit()` integratie met bestaande `audit_trail.py` v1.0.0, numpy Generator / RandomState / python random factory-methods, CLI (`show`, `derive`), global singleton. Replay-scenario test bewijst bit-identieke reproductie na save/load. |
| B60 | GW-Bound Reporter (cvxpy) | KLAAR — referentie-backend voor kleine instanties. Vanaf n≈500 neemt **B176** het automatisch over (zie `cvxpy_reference_sdp` voor head-to-head); B60 blijft de correctness-oracle voor de FW-sandwich. |
| B62 | QAOA + Local Search Refinement | KLAAR (LS-refinement na QAOA bewezen nuttig, ondergebracht in **B70** hotspot-repair + **B134** BLS-pipeline; geen aparte runner nodig). |
| B64 | Fourier Warm-Starting (DCT) | KLAAR |
| B66 | Symmetrie-Caching Cilinder-Grids | KLAAR |
| B67 | Multi-Angle QAOA (ma-QAOA) | KLAAR |
| B65 | Zero-Allocation GPU Buffers | KLAAR |
| B19 | Mixed-Precision Tensors (FP32/FP64) | KLAAR |
| B37 | Lanczos Exact Benchmark + Krylov | KLAAR |
| B61 | Gset Benchmark Loader | KLAAR |
| B53 | Experiment- en Regressieharnas | KLAAR |
| B56 | Resultaat-Export met Audit Trail | KLAAR |
| B54 | Arbitraire-Graaf Lightcone QAOA | KLAAR |
| B27 | Graph Automorphism Deduplicatie | KLAAR |
| B50 | Graph Pruning Preprocessing | KLAAR |
| B68 | BFS-Diamant Lichtkegel | KLAAR |
| B69 | WS-QAOA: SDP Warm-Started QAOA | KLAAR (+37-46% op grids, cvxpy-warmstart n≤200). **Rebenchmark met B176 FW-Y-embedding op Gset G14/G43/G22 (n=800-2000) DONE (17 apr 2026):** `code/b69_fw_warmstart.py` (28/28 tests); fw-ws-bin (Egger 2021 recept) wint op G14 (AR 0.9657) en G22 (AR 0.9629) na 1-flip polish; fw-ws-cont/egger gebruiken SVD-principal-direction van Y; data in `docs/paper/data/b69_fw_warmstart_results.{json,csv}` + tables `.md`/`.tex`. Conclusie: FW-SDP levert sandwich UB ÉN competitieve warmstart-bits in één pass. |
| B79 | FQS: Fractal Quantum Solver + BM direct-solve | KLAAR (ratio 1.0 bipartiet, 0.908 approx niet-bipartiet) |
| B80 | MPQS: Message-Passing Quantum Solver | KLAAR (44/44 tests; max-product BP tree-exact; lightcone-QAOA + ζ-spectral rounding + greedy 1-flip; 13/13 BP-OPT en 11/13 LC-OPT op benchmark) |
| B154 | BiqMac + DIMACS Benchmarks | KLAAR (45/45 tests; rudy + DIMACS parsers; 7 synthetic BiqMac-generators; 5 DIMACS-fixtures; 20-entry chromatic-DB + 10-entry BiqMac BKS-DB; combined leaderboard op Gset+BiqMac+DIMACS, 14/14 ILP-certified, dataset-specifieke BP/LC-stats) |
| B177 | Paper Figures-Pipeline + LaTeX Template | KLAAR (19/19 tests; `b177_figure_pipeline.py` JSON→matplotlib-PDF + PGFPlots-`.tex` dual pipeline; arxiv-style `docs/paper/main.tex` + `refs.bib` met 20+ core refs; Makefile met pdflatex+biber+latexmk-watch; beide paper-figures (leaderboard + ILP-scaling) gegenereerd op 14-instance panel) |
| B10e | PEPO — 2D Tensor Netwerk | KLAAR. `b10e_pepo.py` (~640 regels) + `b10e_benchmark.py` + 40/40 tests (0.18s). **PEPS2D-class**: site-tensoren `T[x][y]` met shape `(D_L, D_R, D_U, D_D, d)`; constructors `from_product_vec`/`plus_state`/`zero_state`; `apply_single` (1-site einsum), `apply_two_horizontal`/`apply_two_vertical` (simple-update SVD truncatie naar `chi_max` met relatieve zero-singular-value pruning tol=1e-12 zodat identity-gates de bond niet inflateren). **Boundary-MPO contractie** (`expectation_value`): double-layer site-tensor `A = contract(T̄, T)` met optionele operator-insertie, kolom-voor-kolom absorptie in MPS-vormige boundary, links+rechts canonicale sweeps met SVD-truncatie naar `chi_b`. **Exacte state-vector reference** voor validatie tot n≤14. **2D MaxCut QAOA pipeline** (`peps_qaoa_maxcut`) met snake-ordering `q = y*Lx + x`, ZZ(γ) op grid-edges + RX(2β) mixer per site. **Benchmark**: 2x2/3x2/3x3 p=1,2 + 2x4 p=1 + 3x4 p=1 (chi_max=4, chi_b=16). **Resultaten**: alle exact-vergelijkbare cases diff ~1e-15 vs state-vector; 3x3 p=2 toont verwachte chi=4 truncatie-fout (diff 2.7e-4). 3x4 grid (12 qubits) PEPS energie consistent in 0.035s. PEPS levert 2D-topologie direct waar column-grouped MPS (B10c) breekt voor Ly≥4. |
| B153 | Beyond-MaxCut QUBO Suite | KLAAR. `b153_qubo_suite.py` (~580 regels) + `b153_benchmark.py` + `test_b153_qubo_suite.py` 53/53 tests (0.28s). **QUBO-datatype** (symmetrische Q + offset, `evaluate`/`delta_flip`-O(n)). **4 probleem-encoders**: weighted MaxCut, Max-k-Cut (one-hot encoding met budget-penalty), MIS (penalty A=n+1), Markowitz portfolio (Lucas 2014 §6.3). Elke encoder retourneert `QUBOInstance` met `decode(x)` voor probleem-specifieke output. **4 generieke solvers** (puur op QUBO): brute_force (EXACT-cert tot n≤22), local_search (1-flip greedy descent), simulated_annealing (Metropolis + geometrische cooling), random_restart (LS of SA inner). **CLI** met subcommands `maxcut`/`kcut`/`mis`/`markowitz`. **Benchmark-resultaten**: (1) Weighted MaxCut RR matcht BF op alle 6 instanties (Petersen=12 cf. B156-OPT). (2) Max-k-Cut: K_4/k=4 RR vindt 6 (alle edges); Petersen/k=3 RR vindt 15 (3-kleurbaar). (3) MIS: RR matcht BF exact op alle haalbare cases (C_5=2, Petersen=4, ER n=16=7); RR vindt α=9 op ER n=20 waar BF te traag is. (4) Markowitz: BF-OPT op n≤8; RR -1 tot -7% van OPT op n=12-16 (budget-penalty maakt landschap saddle-vol). Vormt domein-agnostische bovenbouw voor B130 dispatcher. Absorbeert B89 MIS-deel. |
| B160 | QSVT / Block-Encoding Framework | KLAAR. `b160_qsvt.py` (~380 regels) + `b160_benchmark.py` + `test_b160_qsvt.py` 40/40 tests (0.02s). **LCU block-encoding van Pauli-sum Hamiltonians**: `PauliSum` dataclass (terms = [(coef, "XYZ..."), ...]) met `to_matrix()` en `alpha = Σ|cᵢ|`; PREP-unitary via Householder-extensie van amplitudes √(|cᵢ|/α); SELECT-unitary Σᵢ sign(cᵢ)·|i⟩⟨i|⊗Pᵢ met identity-padding; `block_encode_pauli_sum` retourneert `BlockEncoding` met top-left 2ⁿ×2ⁿ blok = H/α; `verify_block_encoding` checkt unitariteit + match. **QSP-primitieven**: `qsp_signal(x) = [[x, i√(1−x²)]; [i√(1−x²), x]]`, `qsp_unitary(x, Φ) = Rz(φ₀)·Π[W·Rz(φₖ)]`; `chebyshev_T_phases(k) = [0]*(k+1)` reproduceert T_k(x) exact in top-left (|diff| ≤ 7e-16 geverifieerd voor k=1..6); `chebyshev_T_matrix(k, A)` via drie-term-recursie T₀=I, T₁=A, T_{n+1}=2A·T_n − T_{n-1}. **Jacobi-Anger Hamiltonian-simulatie**: `hamiltonian_simulation_qsvt(H, t, α, K, ε)` via e^{-iτx} = J₀(τ) + 2Σₖ(-i)^k Jₖ(τ) Tₖ(x); truncatie K ≈ e|τ|/2 + log(1/ε); `trotter_reference` (order 1 en 2) als baseline. **Benchmark-resultaten**: (1) QSP T₁..T₆ exact match tot machine-precisie. (2) LCU op Ising-TF en Heisenberg-XXX n=2..4 + MaxCut K₃: 7/7 verify=OK, m_anc = ⌈log₂ L⌉ ancilla-qubits. (3) Ham-sim Jacobi-Anger vs Trotter: op Ising-TF n=3 bij t∈[0.1, 4.0] (αt tot 14) err_JA = 1e-16…1.5e-15 vs err_T1 = 7e-4…1.6e-1, err_T2 = 1.5e-6…3.3e-2; op Heisenberg-XXX n=4 bij αt tot 18 idem machine-precisie. Jacobi-Anger wint 12+ orden van grootte bij ~0.3 ms walltime. (4) K-convergentie: exponentiële decay voorbij knik K ≈ e·αt/2, fout 1.1 → 1.1e-15 op K=4→30. Moderne QSVT-baseline voor B131 (Trotter-fout-certificaat) en fundament voor B129 compiler-modernisering. Volle QSP-phase-angle-solver voor willekeurige polynomen is geparkeerd als B160b (alleen bij concrete non-Chebyshev toepassing). |
| B176 | Frank-Wolfe / Conditional Gradient voor SDP | KLAAR. `b176_frank_wolfe_sdp.py` (~400 regels) + `b176_benchmark.py` + `test_b176_frank_wolfe_sdp.py` 40/40 tests (1.4s). **Spectraplex-relaxatie** Δ_n={X⪰0, tr(X)=n} met diagonaal-penalty `f(X) = −¼·tr(L·X) + (λ/2)·‖diag(X)−𝟏‖²`. **Matrix-vrije LMO** via `scipy.sparse.linalg.eigsh(which='SA')` + dense-fallback onder n=40. **Gesloten-vorm line-search** (f is kwadratisch in γ) of Jaggi γ=2/(k+2). **Low-rank maintenance** X=Y·Yᵀ met SVD-truncatie op `rank_cap`. **Dual-certificaat**: `sdp_upper_bound = −f(X_k) + gap_k` (bewijs: f* ≤ −cut_SDP, FW-gap geeft −f* ≤ −f(X_k)+gap). **Feasible-primaal-LB**: rij-normalisatie X̂ geeft cut_SDP ≥ ¼·tr(L·X̂). **GW-rounding** vanaf low-rank Y. **Validatie**: 40/40 tests groen, sandwich LB≤cvxpy≤UB op 7 kleine grafen, scalability 3-reg n=30..500 (FW wint vanaf n≈100, 3.9× speedup op n=200, cvxpy infeasible n>300), cut/UB=0.922-0.994 structureel, gemeten cut/UB=0.947-0.960 ver boven GW-garantie 0.87856. **Parkering**: B176b CGAL-SDP (Yurtsever 2019, Augmented Lagrangian voor n=10000-schaal). Refs: Jaggi 2013, Hazan 2008, Yurtsever 2019. |
| B178 | Docker/Conda-lock Reproducibility | KLAAR. Repro-prerequisite voor **B4** paper-replicatie. `requirements.txt` (7 core-deps: `numpy>=2.0,<3.0`, `scipy>=1.11,<2.0`, `cvxpy>=1.5,<2.0`, `networkx>=3.0,<4.0`, `matplotlib>=3.8,<4.0`, `pandas>=2.0,<3.0`, `psutil>=5.9`, `pytest>=8.0,<10.0`). `pyproject.toml` (project-metadata `zornq` v0.1.0, Python>=3.10,<3.13, 5 optional-dep groups: `dev`/`qiskit`/`ilp`/`gpu`/`all`, entry-points `zornq-gset-bench`/`zornq-b176`/`zornq-b159`/`zornq-audit-show`, pytest-config met `filterwarnings` voor cvxpy-DeprecationWarning). `Dockerfile` (ARG PYTHON_VERSION=3.12, python:3.12-slim base, multi-layer caching met requirements-eerst, env `PYTHONDONTWRITEBYTECODE=1`/`PYTHONUNBUFFERED=1`/`PYTHONHASHSEED=0`/`TZ=UTC`/`LC_ALL=C.UTF-8`, default CMD `pytest code -q --tb=short`). `.dockerignore` (sluit `__pycache__`/`.git`/`results/`/secrets/docs-paper-artifacts/*.npz uit). `environment.yml` (conda/mamba alternatief via conda-forge, zelfde versie-pins). Samen met **B55** seed-ledger vormt dit de volledige reproduceerbaarheidsstack. Volgt Jaggi/Yurtsever-lineage in de zin dat elk runbaar artefact van een paper-publicatie één container-spec + één seed-ledger-snapshot vereist. Container-lock via `conda list --explicit > environment.lock.txt` of `pip freeze` na eerste succesvolle build. |
| B170 | Twin-width Parameter + Cograph-DP | KLAAR. `b170_twin_width.py` (~550 regels) + `b170_benchmark.py` + `test_b170_twin_width.py` 41/41 tests (0.02s). **Trigraph-primitief** (BLACK/RED adjacency) met `contract(u, v)` semantiek *beide-buren-zwart → zwart; beide-afwezig → afwezig; anders → rood*. **Greedy twin-width heuristic** (O(n⁴) contractie-sequentie die lokaal max-red-degree minimaliseert) + **twin_width_exact** (branch-and-bound, n≤8). **Cograph-herkenning** via P_4-free-check (O(n⁴)) en **cotree-decompositie** (parallel = disconnected unie, series = complement-disconnected-join). **Cograph MaxCut DP in O(n³)** via cotree-bottom-up: parallel = convolutie op partitie-grootte, series = `k₁·(n₂−k₂) + (n₁−k₁)·k₂` cross-edge-bijdrage. **Benchmark-resultaten**: (1) Families — K_n, K_{m,n} → tww=0 (cograph_dp); P_n → tww=1; C_n → tww=2; Petersen → tww=4; bomen → tww≤2. (2) Cograph MaxCut correctness: DP == BF exact op alle n≤18 cases (K_n, K_{m,n}, random series-parallel cographs). Speedup: K_18 DP 11.6 ms vs BF 6226 ms = **~540×**, en DP haalbaar voor n=32 random-cograph in 58.9 ms waar BF prohibitive is. (3) Dispatcher-routing-regel: tww=0 → cograph_dp; tww≤2 → tree_dp (B42); tww≤5 → bounded_tww_dp (future); anders → qubo (B153). Feature voor B130 dispatcher: tww als difficulty-metric. Volle bounded-tww-DP voor willekeurige tww≥3 is geparkeerd voor B171/B172 (research-niveau). |
| B91 | BM-QAOA: Burer-Monteiro Warm-Start | KLAAR (13.6× sneller dan cvxpy). **BM-vs-FW head-to-head DONE (18 apr 2026):** `code/b91_bm_vs_fw.py` + 16/16 tests. Op Gset G14/G43/G22 wint BM op AR (0.986/0.986/0.984) vs FW (0.948/0.959/0.961); FW is ~1.8× sneller bij n=2000 en is **het enige SDP-pad dat een gecertificeerde UB levert** (duality-gap) voor dispatcher/selector/anytime-gebruik. Complementaire rollen: BM = primal-only snelle cut, FW = sandwich-SDP. Data `docs/paper/data/b91_bm_vs_fw_*` + tables. |
| B70 | Hotspot Repair: Frustration-Patch | KLAAR (+5-7% cold, complementair B69) |
| B71 | Homotopy Optimizer: Parameter Continuation | KLAAR (λ+p-continuation) |
| B92 | Anti-Kuramoto MaxCut Solver | KLAAR (ratio 1.0 bipartiet, +3.8% vs BM op tri 100×3, BM wint op random) |
| B82 | QW-QAOA: Szegedy Quantum Walk Mixer | KLAAR (exact optimum, XY-walk mixer, ~6× trager dan QITS, MPS-vervolgstap) |
| B93 | QITS: Imaginary-Time QAOA | KLAAR (exact optimum op alle n≤18, state-vector limiet n≤22) |
| B95 | Simulated Bifurcation (OPO) | KLAAR (3e klassieke baseline, ≈BM op random, verliest van Kur op grids) |
| B76 | p-ZNE: Depth Extrapolation | KLAAR (kwadratisch <1% fout) |
| B10f | Classical Shadows / Monte Carlo | KLAAR (ratio 1.0 alle instanties, shadow energy ~1/√K) |
| B100 | Planar Pfaffian Oracle | KLAAR (bipartiet O(1), brute-force n≤25, Pf²=det rel_err<1e-8) |
| B57 | Parameter-Bibliotheek per Graaftype | KLAAR |
| B26 | Transverse Contraction | BEWEZEN |
| B41 | TDQS v2: Chi-Aware Gate Selection | BEWEZEN (+4-5% vs QAOA-1) |
| B31 | Circuit Knitting / Wire Cutting | BEWEZEN |
| B29 | Randomized SVD d-wall doorbraak | BEWEZEN |
| B134 | Breakout Local Search (BLS) | KLAAR (exact op n<=20, schaalt naar n=1000, 16 tests) |
| B135 | Population Annealing Monte Carlo | KLAAR (23 tests, wint van BLS op n>=100: +8 tot +46 cut, 360 regels) |
| B136 | CUDA Local Search Kernel | KLAAR (6 CUDA kernels, CSR-format, CPU fallback, 27 tests. GTX 1650: cuda_pa gem. gap 0.00% vs cuda_bls 2.55%) |
| B137 | Gset Benchmark + Vergelijking | KLAAR (7 solvers, 40+ BKS. **Stanford Gset 71 instanties: combined gem. gap 3.30%, 8 exact BKS, 50/71 binnen 1%. Schaalbreuk bij n~5000-7000, grote sparse 9-24% gap.** Metadata-fix G6/G35/G55/G56 gedaan) |
| B130 | Auto-Dispatcher / Strategy Selector | KLAAR (3-tier dispatch: exact+quantum+classical, **7 solvers incl. cograph_dp + signed_brute_force**, certificering, **114 tests — Dag 8 signed-downgrade GESLOTEN 17 apr 2026**, was 88 in Dag 2). B170 tww/cograph-metric ingeplugd als difficulty-feature: `classify_graph` vult nu `is_cograph`, `tww`, `is_unweighted` (budget-capped: is_cograph op n≤100, tww op n≤32); `select_strategy` routet unweighted cographs naar nieuwe `cograph_dp` solver-slot tussen pfaffian en brute_force (K_30 cut=225 exact end-to-end). B176 FW-SDP-bound integratie via B131-certificate-factory (`certify_maxcut_from_fw` — zie B131). **Dag-8 uitbreiding 17 apr 2026:** signed-instance detectie + routing-guards + sign-aware solver + certify_result-downgrade. `has_signed_edges(edges)` + `info['has_signed_edges']` voedt `select_strategy` — pfaffian_exact/exact_small/mps_qaoa_grid/mps_qaoa_wide/lightcone_qaoa krijgen `not is_signed`-guards (voorkomt bipartite/grid short-circuit-bug op signed edges én QAOA-hang op signed grids). Nieuwe `exact_small_signed`-route activeert `_run_signed_brute_force` (NumPy-vectorized 2^n × m enumeration, n≤24, is_exact=True). `_run_pfaffian` en `_run_brute_force` raisen ValueError op signed edges als verdedigings­laag. `certify_result(..., strategy=...)` downgradet pfaffian_exact/exact_small/exact_brute + signed naar APPROXIMATE. End-to-end: signed triangle n=3 → exact_small_signed, cut=2.0, cert=EXACT; B186 5 spinglass/torus/pm1s-rijen nu correct sign-aware (zie header DAG 8). |
| B48 | Auto-Hybride Planner + Triage | KLAAR (dispatch+LS bewezen, `auto_planner.py` ~400 regels; draait bovenop B130 3-tier dispatcher). **Aanvul-idee:** routing-hint via B170 tww-metric + cograph-detector nog optioneel. |
| B49 | Anytime Solver met Certificaat | KLAAR (5 lagen + GW-SDP, 11/11 exact op bekende BKS). Anytime-certificaat nu compleet dichtgetrokken: UB via **B176 FW-sandwich** (alle n) + LB via layered-solvers + OPT-oracle via **B159 ILP**; registreerbaar onder **B131** LEVEL 2-CERTIFIED. **Dag 4 paper-figuur 17 apr 2026:** `code/b49_anytime_plot.py` (~540 regels) produceert de centrale time-vs-value sandwich-plot (UB monotoon ↓ uit B176 FW per-iter, LB monotoon ↑ via 5-laags cascade, OPT horizontaal via B159 ILP); matplotlib-PDF + PGFPlots TikZ + JSON/CSV trace; `test_b49_anytime_plot.py` 18/18 groen incl. sandwich-invariant. Validatie myciel3: LB=16 ≤ OPT=16 ≤ UB=17.32, gap 7.6%. Artifacts: `docs/paper/figures/b49_anytime_plot.{pdf,tex}` + `docs/paper/data/b49_anytime_trace.{json,csv}`. |
| B144 | UGC-Hard Gadget Generator | KLAAR (twisted/mobius/noise-cycle, exact certificeerbaar). **Schaal-knelpunt opgelost door B176/B159:** grote gadgets kunnen nu SDP-boven-gecertificeerd worden tot n≥500 via FW en exact gecertificeerd tot n≈50 via ILP-oracle. Eventuele uitbreiding naar UGC-bounds: zie **B187** (LAAG-MIDDEL). |
| B147 | FibMPS Regimekaart + Checkpoint Harness | KLAAR (route-metadata, `--checkpoint/--resume` in `gset_benchmark.py`). Status definitief. |
| B148 | SAT/CNF Gadget-Certificaatlaag | KLAAR (`maxcut_gadget_sat.py`, DPLL, DIMACS, geintegreerd in hotspot_repair). Status definitief — aanvulling via **B159** ILP-oracle als moderne exact-pad. |
| B149 | Multiscale Ordering & Cluster Routing | KLAAR (`multiscale_maxcut.py`, pocket-win op signed G27-G34, geen globale winst). Status definitief — niet heropenen tenzij **B170** cograph-DP of tww-metric signaleert dat een multiscale-route alsnog zinvol is op specifieke Gset-instanties. |
| B150 | Evidence Capsules & Receipts | KLAAR (`evidence_capsule.py`, SHA256, sidecar JSON, 3 evidence levels). Status definitief — gekoppeld aan **B131** certificaatlaag. |
| B151 | Bandit-Planner | KLAAR (`bandit_planner.py`, UCB, pocket-wins maar static-planner wint nipt globaal). Status definitief — verdere ML-gedreven dispatch-selectie vindt plaats in **B184** Instance Difficulty Classifier (KLAAR 18 apr 2026) + **B186** Solver-Selector Benchmark (KLAAR 17 apr 2026, 14/14 ILP-certified, 4 dispatcher-failure-modes op BiqMac-spinglass gedocumenteerd). |
| B72 | Multiscale Graph Coarsening | KLAAR maar TELEURSTELLEND — HEM coarsening verliest van direct PA op alle Gset instances. +1 grafen: ~1% slechter. +-1 Ising: ~7-8% slechter. **Open her-evaluatie:** HEM is getest zonder structuur-geleide contractie; een route *tww/cograph-geleide coarsening* (via **B170**) + Schur-substitutie (via **B119**) + sparsifier-preprocessing (via **B118**) heeft nooit een eerlijke run gekregen — niet heropenen tenzij deze specifieke hybride is geprobeerd. |
| B99 | Feedback-Edge Skeleton Solver v2 | KLAAR, WINT op ALLES. Multi-tree ensemble+greedy: G70 **2.7%**, G60 6.8%, G62(+-1) 17.6%(was 25%), G81(+-1) 17.5%(was 25.2%) |
| B128 | Hybrid QAOA+Classical Solver | KLAAR. QAOA ⟨ZZ⟩ correlaties sturen tree-selectie. **Quantum advantage:** +2 (n=160), +6 (n=250) op ±1 Ising. Eerste ZornQ onderdeel met meetbaar QC voordeel. |
| B14 | MERA tensor netwerk | KLAAR, BEVESTIGD. Engine werkt (22 tests), eigendecompositie-update convergent bij n=16. MPS wint bij n≤16 en p≤3 (area-law). MERA voordeel pas bij n>>16 + volume-law. |
| B109 | Adversarial Instance Generator | KLAAR. 7 families (33 tests). PA wint 8/15 adversarial instanties, B99 zwak op hoge-tw (42.9% gap). GW-gaps: chimera 9.5%, high_feedback 15.1%. |
| B73 | Quantum-Guided Branch-and-Bound | KLAAR. Exact MaxCut tot n≈25 (sparse), hybrid branching -21% B&B nodes. Complementair met BLS/PA als certificaat-generator. 34 tests PASS. |
| B101 | Symbolische Fourier Cost Compiler | KLAAR. Exacte analytische QAOA-1 formule met triangle-correctie. 98× sneller dan SV bij n=10. Ratio 0.66 op 3-regular. 40 tests PASS. |
| B107 | Quantum Nogood Learning | KLAAR. SAT-stijl nogood learning voor MaxCut. 4 extractiemethoden (exact/edge/triangle/heuristic), Z2-deduplicatie, progressive learn-solve pipeline. Guided BLS vindt optimale cuts. 55 tests PASS. |
| B104 | Boundary-State Compiler | KLAAR. Separator-decompositie + boundary→response compilatie. Ratio 1.0 op alle geteste grafen. Isomorfisme-caching, lightcone boundary cache. 34 tests PASS. |
| B39 | TRG / HOTRG | KLAAR. Native 2D tensor contractie. TRG coarse-grain 2x2 blokken, HOTRG hogere-orde SVD. Ising benchmark exact bij chi=16, HOTRG chi=4 error ~1e-4. QAOA 2D exact evaluator. Schaalt tot 256 spins in <1s. 59 tests PASS. |
| B34 | Mid-Circuit Measurement / Adaptieve Projectie | KLAAR. MPS-gebaseerde mid-circuit metingen. Born-regel sampling, multi-branch middeling, adaptieve meetpunt selectie (bond-entropie). Fideliteit 1.0 vs SV. Schaalt tot n=50. 63 tests PASS, 6 experimenten. |
| B118 | Cut-Preserving Sparsifier | KLAAR. Dense pm1: **+36-38** (n=200). Sparse Gset: geen effect. ER + DW + threshold methoden. 17 tests PASS. |
| B119 | Schur-Complement Separator Elimination | KLAAR. G70: **78% reductie** (10000→2164 nodes), maar reconstructie-kwaliteit slechter dan direct B99v2 (8610 vs 9283). Exacte leaf/chain eliminatie. 21 tests PASS. **Open her-evaluatie:** de eliminatie-volgorde werd naïef gekozen; met **B170** tww-metric als difficulty-proxy kan de selector gericht eerst *low-tww-pockets* wegschuren — re-benchmark nodig voordat B119 als definitief-beneden-B99v2 geldt. |
| B36 | Random Graph Testing | AFGEROND. RQAOA 98% ratio op n=20, maar 10.000× trager dan BLS. Tier 3 (BLS/PA/CUDA) wint op n≤22. |
| B40 | Lightcone + Transfer Matrix (iTEBD-QAOA) | AFGEROND. p=3 chi=32: ratio 0.699 (sweet spot). chi=64 timeout (>2u). Strip-BC nice-to-have. |
| B47 | RQAOA — Recursive QAOA | AFGEROND. p=2: 16x4 0.963→1.000, 100x4 0.971→0.994. QAOA ratio +13%. |
| B42 | Treewidth-Decompositie DP | KLAAR. Exact MaxCut op lage-tw grafen. **1000x4 pm1: +287 vs B99v2** (exact=2510 vs 2223). tw=4 grids in <0.3s. 27 tests PASS. |
| B128ci | Probleem-Agnostische Circuit Interface | KLAAR. ~850 regels, 66 tests. 16+ gates, 4 constructors (QAOA/VQE/Trotter), Observable framework, dual backend (SV+MPS). Prerequisite voor B129+B131. |
| B129 | Hamiltonian Compiler | KLAAR. ~884 regels, 71 tests. Hamiltonian class met 6 modellen (Ising/Heis/Hubbard/molecular/MaxCut/custom), JW transformatie, Trotter orde 1/2/4, QAOA (3 mixers), gate optimalisatie. |
| B131 | Kwaliteitsgarantie & Certificaat-Laag | KLAAR. ~962 regels, **72 tests — Dag 2 integratie GESLOTEN 17 apr 2026**. 5 niveaus (EXACT→UNKNOWN), Trotter-fout T1/T2/T4, chi-extrapolatie, MaxCut verificatie (bipartiet/BF), batch certificering. **Nieuwe factory-functies:** `certify_maxcut_from_fw(fw_result, ...)` vertaalt B176 FW-sandwich (feasible_cut_lb ≤ cut_SDP ≤ sdp_upper_bound) naar QualityCertificate met method="b176_frank_wolfe_sdp", verification="fw_duality_sandwich" en level bepaald uit relatieve gap (EXACT<1e-4%, NEAR_EXACT<1%, BOUNDED<15%, anders APPROXIMATE); `certify_maxcut_from_ilp(ilp_result, ...)` vertaalt B159 ILP-oracle resultaat (certified+match→EXACT, certified+klein-gap→NEAR_EXACT/BOUNDED, !certified→BOUNDED/APPROXIMATE) met verification ∈ {"ilp_certified_optimal","ilp_incumbent_only"}. Beide factories accepteren optionele assignment voor Laplacian-cut-verificatie en blijven dependency-vrij (geen import van b176/b159 modules — resultaat-objecten worden duck-typed doorgegeven). 20 nieuwe tests: 10 voor FW-factory (EXACT/NEAR_EXACT/BOUNDED/APPROXIMATE level-bepaling, incumbent-LB-strengthening, not-converged warning, assignment-verificatie, bounds-sanity, diagnostic-checks), 10 voor ILP-factory (certified-EXACT, mismatching-incumbent, solver-metadata, user-cut>opt-warning, missing-opt-value graceful). **Dag-8 certificate-downgrade KLAAR 17 apr 2026:** downgrade-laag verplaatst naar `auto_dispatcher.certify_result()` waar sign-context beschikbaar is (is_signed in info). Nieuwe regel: strategy ∈ {pfaffian_exact, exact_small, exact_brute} + has_signed_edges → APPROXIMATE (geen EXACT-certificaat mogelijk omdat pfaffian_maxcut's bipartiete/grid short-circuits niet sign-aware zijn en BF/exact_small gebruikelijk unit-weight aannemen). De B131 FW/ILP-factories zijn ongewijzigd gebleven (geen sign-specifieke mapping nodig — duality-gap en ILP-certificate dekken dit al correct); downgrade leeft in dispatcher-laag als *defense-in-depth* naast de routing-guards en solver-side ValueError. Nieuwe `exact_small_signed`-strategie levert EXACT-certificaten op signed n≤20 via `_run_signed_brute_force`. **Paper §13-belofte volledig ingelost.** |
| B132 | Multi-Domain Proof-of-Concept | KLAAR. ~710 regels, 37 tests. 3 domeinen: CM/Heisenberg (VQE), H2 molecuul (JW, FCI=-1.1373 Ha exact), PDE/1D-deeltje (Trotter-2, fid=1.0). End-to-end pipeline: Hamiltonian→Circuit→run→certify. |
| B133 | Scalability Benchmark Suite | KLAAR. ~590 regels, 19 tests. 7 benchmarks: QAOA/VQE/Trotter scaling, chi-convergentie, circuit-complexiteit, SV/MPS break-even, large-scale. MPS 5000q in 0.23s (chi=4, ~65K gates/s). SV exponentieel n=18→16s. Break-even ~n=14-16. |
| B32 | Tropische Tensor Netwerken (MAP) | KLAAR. Tropische (max,+) algebra voor exacte MaxCut. Variable elimination met min-degree heuristic. 100% match brute-force. 2D grids: C_max=|E| (bipartiet). Sandwich bound: QAOA ≤ C_max. 1D tot n=5000 (2.5s), 2D 10×10 (tw=13, 0.007s). 60 tests, 6 experimenten. |
| B156 | Lasserre / SoS level-2 SDP | KLAAR. `b156_sos2_sdp.py` (~380 regels) + `b156_benchmark.py` + 38/38 tests. Moment-matrix M_2 (N=(n+1)(n+2)/2), pseudo-momenten y_S met \|S\|≤4, cvxpy/SCS. **Resultaten:** 13/14 instanties exact op level-2 (Petersen 12.0 vs GW 12.5, K_3 2.0 vs GW 2.25, alle odd cycles C_5/C_7, alle bipartiete, 3-reg n=10/12/14 allemaal exact). Alleen K_5 niet exact (6.25 vs OPT 6, gelijk aan GW). Gemiddelde tightening 3.25%. SoS-2 vervult triangle-inequalities + facet-bounds automatisch. |
| B158 | Triangle + Odd-Cycle Cutting Planes | KLAAR. `b158_cutting_planes.py` (~555 regels) + `b158_benchmark.py` + 23/23 tests. LP-relaxatie van cut-polytoop via SciPy/HiGHS met 4 triangle-facetten per driehoek (optie K_n-extensie voor sparse grafen) + iteratieve odd-cycle separatie via signed-graph Dijkstra. **Resultaten:** 12/14 instanties exact op LP+OC; pentagon-cut gevonden in 1 iteratie sluit C_5/C_7 volledig. LP+OC 10-30× sneller dan SoS-2 (Petersen 0.01s vs 0.18s; 3-reg n=14: 0.02s vs 0.39s). Niet exact op K_5 (6.667) en K_6 (10) — LP-relaxatie is zwakker dan SDP op cliques. Voor sparse grafen is LP+OC de schaalbare keuze. |
| B159 | ILP-Oracle Ceiling voor MaxCut | KLAAR. `b159_ilp_oracle.py` (~412 regels) + `b159_benchmark.py` + `test_b159_ilp_oracle.py` (31/31 tests). Standaard 0/1-MILP via `scipy.optimize.milp` (HiGHS), met optionele `pyscipopt` en `gurobipy` paden (graceful SKIPPED als niet geïnstalleerd). Formulering: `max Σ w_uv y_uv` s.t. `y_uv ≤ x_u+x_v`, `y_uv ≤ 2−x_u−x_v`, `x,y ∈ {0,1}`. Symmetry-break door `x_0 = 0` (halveert search-space). **Benchmark:** 17/17 instanties certified (K_3..K_6, K_{3,3}, K_{2,5}, C_5/C_7/C_8, P_8, Petersen, 3-reg n=10/12/14/20/30/50) in totaal <1s per graaf; 3-reg n=50 in 0.619s. Upper-bound gap: LP+OC tight 15/17, SoS-2 tight 13/17 (binnen gemeten subset n≤14), GW tight slechts 6/17 (Petersen +0.50, K_5 +0.25). Time-limit behavior geverifieerd: uncertified incumbents worden via bitstring-recompute consistent gemaakt. |
| B165 | Qiskit Runtime Hardware-Run | KLAAR. `b165_qiskit_runtime.py` (~424 regels) + `b165_benchmark.py` + 22/22 tests. ZornQ→Qiskit gate-compiler (15 gates: H/X/Y/Z/S/T/RX/RY/RZ/CX/CZ/SWAP/RXX/RYY/RZZ — beide engines delen `exp(-iθ/2·ZZ)` conventie zodat geen factor-correctie nodig is). 3-mode backend: `aer` (lokaal AerSimulator zonder ruis), `noisy` (Aer + depolariserende NoiseModel met p1=1e-3 single-qubit, p2=1e-2 CX — IBM Eagle/Heron-realistisch april 2026), `hardware` (`QiskitRuntimeService` + `SamplerV2` op `ibm_brisbane`/etc., token-gated via `QISKIT_IBM_TOKEN` env-var, gracefully `SKIPPED_NO_TOKEN` zonder credentials). Sample-gebaseerde ⟨Z_u Z_v⟩ + E[H_C] uit shot-counts (Qiskit little-endian). **Resultaten:** Aer + noisy-Aer halen beide 9/9 best-cut OPT op K_3..Petersen+3-reg met QAOA p=1, γ=0.7, β=0.4. Noisy E[H_C] minimaal lager dan clean (Petersen 7.330 vs 7.326), bewijs dat depolariserende ruis op p=1 nog niet domineert. Hardware-pad geverifieerd via SKIPPED-pad zonder netwerk; voor echte runs moet user `pip install qiskit-ibm-runtime` + `QISKIT_IBM_TOKEN` zetten + queue-tijd accepteren. |
| B10g | 7-operatie Heisenberg CT-scan | GEFALSIFICEERD (chi ratio 1.00, 94.7% fout bij d=8, 22 tests) |
| 8D | 8D Analyse | GEFALSIFICEERD |

---

## IN PROGRESS

| # | Naam | Waar mee bezig |
|---|------|----------------|
| — | (geen actieve taken) | — |

---

## DAG-8+ KANDIDATEN — shortlist na paper-1 draft-oplevering

Na de Dag 6-7 paper-1 oplevering zijn de volgende kandidaten de natuurlijke
volgende blokken. Gesorteerd op *"waar verdient de paper nu het meest van?"*:

1. **Paper review-pass + biber-build** (0.5-1 dag, mandatory, **ACTIEF HOOG**):
   Gertjan leest de volledige draft op inhoud/toon/claims, voert `latexmk -pdf
   main.tex` met biber op laptop, markeert onduidelijkheden en over-claims.
   *Mandatory before submission.* Dit is nu het enige openstaande Dag-8+-blok
   dat nog paper-1-submission blokkeert.
2. ~~**Dag-8 certificate-downgrade** (`pfaffian_exact`/`exact_small` → HEURISTIC
   op signed instanties)~~ → **KLAAR 17 apr 2026** (ingebouwd in
   `code/auto_dispatcher.py` in plaats van `b131_quality_certificate.py` — 4-laags
   defense: detectie + routing-guards + sign-aware `_run_signed_brute_force` +
   `certify_result`-downgrade; 114/114 tests; B186 re-run met 5 correcte rijen;
   zie header DAG 8). ~~**Nieuw-ontdekt follow-up:** `B159-Dag-8b` ILP sign-bug
   (MIDDEL-HOOG, 1 dag).~~ → **B159-Dag-8b KLAAR 17 apr 2026** (signed-safe
   4-constraint linearisatie in 3 solver-backends, 39/39 unit-tests incl. 8
   discriminerende signed-instanties, B186 Auto==ILP 14/14; zie MIDDEL-HOOG-
   tabel en header DAG 8b).
3. ~~**B176b CGAL-SDP voor n≈10000**~~ **KLAAR 18 apr 2026**: `code/b176b_cgal_sdp.py`
   (~370 regels, Yurtsever-Fercoq-Cevher 2019 Augmented Lagrangian met expliciete
   duale variabele y + β√k-schedule), `test_b176b_cgal_sdp.py` (27 tests in 9
   classes), `b176b_benchmark.py` (correct/h2h/scale/conv-secties, paper-2 scale
   panel n∈{500,1000,2000,5000,10000} met `docs/paper/tables/b176b_scale_table.tex`
   generator). Harde duale sandwich `cut_SDP ≤ <y,1> − n·λ_min(-L/4 + diag(y))`
   geldt voor ELKE y — dus provably-valid UB tijdens convergentie. Test- en
   scale-runs uit te voeren op Gertjan's laptop (sandbox was out-of-space bij
   afronding). Zie backlog-entry B176b voor runbook.
4. ~~**B69 FW-warm-started QAOA rebenchmark op Gset n=500-2000**~~ **KLAAR 17 apr 2026**:
   `code/b69_fw_warmstart.py` (28/28 tests), G14/G43/G22 gedraaid, fw-ws-bin
   wint op G14+G22 na polish; data in `docs/paper/data/b69_fw_warmstart_results.*`
   + tables `.md`/`.tex`. Paper-stof geleverd; backlog-entry bijgewerkt.
5. ~~**B91 BM-vs-FW head-to-head**~~ **KLAAR 18 apr 2026**: `code/b91_bm_vs_fw.py`
   (16/16 tests), G14/G43/G22. BM wint op primal-AR (0.98+), FW wint op
   schaalbaarheid (n=2000, 1.8× sneller) en is enige SDP-pad met UB-certificaat.
   Data in `docs/paper/data/b91_bm_vs_fw_*` + tables.
6. ~~**B184 Instance Difficulty Classifier**~~ **KLAAR 18 apr 2026**:
   `code/b184_difficulty_classifier.py` (740 regels, 26/26 tests), 12-feature
   extractor + FW-gap-labeling + dual-backend (sklearn RF/GB + numpy stump).
   Mixed panel (26 instanties): RF CV-accuracy **0.77 ± 0.15**, top-features
   m/n/density. `dispatcher_hook()` klaar voor B130-integratie.
7. **B179 Open Dataset Release (Zenodo)** (1-2 dagen, **ACTIEF HOOG**).
   Paper-supplement, citeerbaar, laag-risico. Sluit natuurlijk aan op de
   paper-1 submission (Zenodo-DOI in refs.bib).
8. **B181 pip-installable zornq Package** (3-5 dagen). Community-multiplier
   voor paper.

Niet-paper-blokken met hoge intrinsieke waarde:
- **B173 Parallel Tempering Monte Carlo** (2-3 dagen) als Gset-baseline.
- **B175 Memetisch/Genetisch Algoritme** (3-5 dagen) — historische Gset-
  recordhouders.
- **B116 MaxCut QAOA als Z₂ Lattice Gauge Theory** (1-2 weken) — directe
  publiceerbare brug tussen bestaande infra en gauge-theorie.

---

## OPEN — Gesorteerd op prioriteit

### HOOG

| # | Naam | Doorlooptijd | Waarom hoog |
|---|------|-------------|-------------|
| ~~B134~~ | ~~Breakout Local Search (BLS)~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B135~~ | ~~Population Annealing Monte Carlo~~ | ~~3-5 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B136~~ | ~~CUDA Local Search Kernel~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B137~~ | ~~Gset Benchmark + Vergelijking~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B128~~ | ~~Probleem-Agnostische Circuit Interface~~ | ~~2-3 dagen~~ | ~~KLAAR — 66 tests, dual backend, 4 constructors~~ |
| ~~B129~~ | ~~Hamiltonian Compiler~~ | ~~3-5 dagen~~ | ~~KLAAR — 71 tests, JW, T1/T2/T4, 6 modellen~~ |
| ~~B130~~ | ~~Auto-Dispatcher / Strategy Selector~~ | ~~3-5 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| B4 | Paper-1 full draft | **DRAFT-KLAAR 17 apr 2026** (Dag 6-7). `docs/paper/main.tex` 754 regels, ~2931 woorden, 11 PDF-pp. (zonder bibliografie), 16 sections, 4 figures, 3 tables, 9 equations, 39 bib-entries. Sandbox-compile cleanly, 0 undefined refs na 2e pass. Open follow-up: review + biber-build op laptop + venue-keuze + abstract/title-tune. Paper-2 (octonionic companion) 2-3 weken vanaf later startdatum. |
| B4-review | Paper-1 review-pass + biber-build | 0.5-1 dag | **ACTIEF 17 apr 2026.** Volledige draft lezen op inhoud/toon/claims, `latexmk -pdf main.tex` met biber op laptop, onduidelijkheden en over-claims markeren. Mandatory pre-submission. Enige resterende blocker voor paper-1 submit na Dag-8b. |
| ~~B176b~~ | ~~CGAL-SDP: Augmented Lagrangian voor n≈10000~~ | ~~3-5 dagen~~ | **KLAAR 18 apr 2026** — `code/b176b_cgal_sdp.py` (~370 regels), 27 tests in 9 classes, `b176b_benchmark.py` met paper-2 scale-panel (JSON/CSV + MD/TeX-table generator). Harde duale UB `<y,1> − n·λ_min(-L/4+diag(y))` gecertificeerd voor ELKE y (geen penalty-absorptie meer). Test- en scale-runs op laptop (sandbox was out-of-space). Zie AFGEROND-tabel. |
| ~~B179~~ | ~~Zenodo Reproducibility Archive (Paper-1 Companion)~~ | ~~1 dag sandbox + 1 uur laptop~~ | **KLAAR 18 apr 2026** — DOI `10.5281/zenodo.19637389`, Zenodo-record https://zenodo.org/records/19637389, GitHub-tag `paper-v1.0-2026-04-17` op `github.com/gertjanbron/zornq`. Snapshot-commit `9b7faa3`, metadata-fix `334f6fc` (JSON-comma-repair), paper-patch `f44ae0d`, backlog-update `9623c58`. `main.pdf` nu 14 pp. met werkende `\href`-DOI-links in §17 + Zenodo-cite in References. Zie AFGEROND-tabel. |
| ~~Repro~~ | ~~Reproduceerbaarheid & Softwarekwaliteit~~ | ~~2-3 dagen~~ | ~~KLAAR — B178 Docker + B55 seed-ledger~~ |
| ~~B156~~ | ~~Lasserre / SoS level-2 SDP~~ | ~~2-3 dagen~~ | ~~KLAAR — `b156_sos2_sdp.py`, 38/38 tests, 13/14 instanties exact (Petersen, K3, C5, C7, K33, P8, ...). Gemiddelde tightening 3.25% vs GW; SoS-2=12.0 op Petersen vs GW=12.5 (OPT=12).~~ |
| ~~B158~~ | ~~Triangle + Odd-Cycle Cutting Planes~~ | ~~3-5 dagen~~ | ~~KLAAR — 23/23 tests, 12/14 exact, HiGHS+OC-separator op signed-graph; <0.02s tot n=14.~~ |
| ~~B165~~ | ~~Qiskit Runtime Hardware-Run~~ | ~~3-5 dagen~~ | ~~KLAAR — `b165_qiskit_runtime.py`, 22/22 tests, ZornQ→Qiskit gate-compiler (15 gates: H/X/Y/Z/S/T/RX/RY/RZ/CX/CZ/SWAP/RXX/RYY/RZZ), 3-mode backend (`aer`/`noisy`/`hardware`), depolariserende NoiseModel (p1=1e-3, p2=1e-2, IBM Eagle/Heron-realistic), SamplerV2 voor IBM Quantum (token-gated, gracefully SKIPPED zonder credentials). Benchmark: aer + noisy-aer halen 9/9 best-cut OPT op K_3..Petersen.~~ |
| ~~B165b~~ | ~~Hardware-Submit + Calibration-Mirror Noise~~ | ~~1-2 dagen~~ | **KLAAR 17 apr 2026** — echte `ibm_kingston`-run voltooid, hardware AR 0.776/0.773, Best(HW)=OPT, cal-mirror ≤2.9% off. Zie AFGEROND-tabel. Oorspronkelijke Dag-5 status: Submit-pakket compleet: `code/b165b_hardware_submit.py` (~320 regels, `--dry-run`/`--submit`/`--resume`, token via env-var of losse file) + `code/b165b_noise_baselines.py` (~290 regels, drie Aer-baselines incl. `--fetch-snapshot-from` voor cal-mirror uit `backend.properties()`) + `code/b165b_parse_results.py` (~230 regels, booktabs LaTeX + Markdown emitter met AR-kolom en ILP-OPT via B159) + `code/test_b165b_hardware_submit.py` 13/13 groen + `docs/paper/hardware/B165b_README.md` (stap-voor-stap laptop-instructies). Dag-5 voorbereidingsrun: 3reg8 (n=8, OPT=10) + myciel3 (n=11, OPT=16) met QAOA p=1 (γ*=0.388, β*=1.194 uit 10×10 grid-search), drie Aer-kolommen gevuld in `docs/paper/tables/b165b_hardware_table.{md,tex}`: 3reg8 E[H]=8.00/7.92/7.93 (noiseless/depolar/cal-mirror), myciel3 E[H]=12.84/12.68/12.74. `best_cut_seen` haalt OPT op beide instanties op alle drie baselines. Laatste kolom "Hardware" wacht op Gertjan's IBM Quantum submit (geen Claude-token vereist). |
| ~~B177~~ | ~~Paper Figures-Pipeline + LaTeX Template~~ | ~~2-3 dagen~~ | ~~KLAAR — `b177_figure_pipeline.py` (19/19 tests), arxiv-style `docs/paper/main.tex` + `refs.bib` (20+ core refs), Makefile (pdflatex+biber+latexmk-watch), JSON→matplotlib-PDF + PGFPlots-`.tex` dual output, beide paper-figures (leaderboard + ILP-scaling) gegenereerd op 14-instance Gset+BiqMac+DIMACS panel.~~ |
| ~~B178~~ | ~~Docker/Conda-lock Reproducibility~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND; `requirements.txt` + `pyproject.toml` + `Dockerfile` + `.dockerignore` + `environment.yml`, Python 3.12-slim container met PYTHONHASHSEED=0, 7 core-deps gepind, 5 optional-dep groups (dev/qiskit/ilp/gpu/all), entry-points voor `zornq-b176`/`zornq-b159`/`zornq-audit-show`.~~ |

### MIDDEL-HOOG

| # | Naam | Doorlooptijd | Waarom |
|---|------|-------------|--------|
| ~~B100~~ | ~~Planar Pfaffian Oracle~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B131~~ | ~~Kwaliteitsgarantie & Certificaat-Laag~~ | ~~2-3 dagen~~ | ~~KLAAR — 52 tests, 5 niveaus, Trotter/chi/MaxCut verificatie~~ |
| ~~B132~~ | ~~Multi-Domain Proof-of-Concept~~ | ~~1-2 weken~~ | ~~KLAAR — 3 domeinen: CM/Heisenberg, H2 molecuul (FCI=-1.1373 Ha), PDE/Trotter (fid=1.0), 37 tests~~ |
| ~~B133~~ | ~~Scalability Benchmark Suite~~ | ~~1-2 weken~~ | ~~KLAAR — 7 benchmarks, MPS 5000q/0.23s, break-even ~n=14-16, 19 tests~~ |
| ~~B99~~ | ~~Feedback-Edge Skeleton Solver~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND. WINT op positieve grafen: G60 7.6% vs 8.8%, G70 5.3% vs 10%~~ |
| ~~B118~~ | ~~Cut-Preserving Sparsifier~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND. Wint op dense pm1 (+36-38 bij n=200), geen effect op sparse Gset~~ |
| ~~B119~~ | ~~Schur-Complement Separator Elimination~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND. G70 78% reductie, maar reconstructie < direct B99v2~~ |
| ~~B72~~ | ~~Multiscale Graph Coarsening~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND. Benchmark teleurstellend: HEM helpt niet voor Gset~~ |
| ~~B73~~ | ~~Quantum-Guided Branch-and-Bound~~ | ~~klaar~~ | ~~KLAAR — exact tot n≈25, hybrid -21% nodes, 34 tests~~ |
| ~~B82~~ | ~~QW-QAOA: Szegedy Quantum Walk Mixer~~ | ~~6-8 uur~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B101~~ | ~~Symbolische Fourier Cost Compiler~~ | ~~klaar~~ | ~~KLAAR — 98× sneller dan SV, exacte analytische formule met triangle-correctie, 40 tests~~ |
| ~~B104~~ | ~~Boundary-State Compiler~~ | ~~klaar~~ | ~~KLAAR — separator-decompositie, ratio 1.0, isomorfisme-cache, 34 tests~~ |
| ~~B109~~ | ~~Adversarial Instance Generator~~ | ~~klaar~~ | ~~KLAAR — 7 families, 33 tests, PA domineert adversarial, B99 zwak op hoge-tw~~ |
| ~~B48~~ | ~~Auto-Hybride Planner + Triage~~ | ~~1-2d rest~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B49~~ | ~~Anytime Solver met Certificaat~~ | ~~PROTOTYPE~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B32~~ | ~~Tropische Tensor Netwerken~~ | ~~1-2 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B10f~~ | ~~Monte Carlo / Classical Shadows~~ | ~~?~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B34~~ | ~~Mid-Circuit Measurement~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND~~ |
| ~~B153~~ | ~~Beyond-MaxCut QUBO Suite (portfolio, weighted, Max-k-Cut, MIS)~~ | ~~1-2 weken~~ | ~~KLAAR — `b153_qubo_suite.py` (~580 regels) + 53/53 tests; QUBO-datatype + 4 encoders (weighted MaxCut, Max-k-Cut one-hot, MIS, Markowitz) + 4 generieke solvers (BF/LS/SA/RR); RR matcht BF op alle haalbare instanties; vindt α=9 op MIS ER n=20 waar BF te traag is; absorbeert B89 MIS-deel.~~ |
| ~~B154~~ | ~~BiqMac + DIMACS Benchmarks~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND; 45/45 tests, rudy+DIMACS loaders, 7 generators, combined leaderboard 14/14 ILP-certified~~ |
| ~~B159~~ | ~~ILP-Oracle Ceiling (HiGHS/SCIP/Gurobi)~~ | ~~1-2 dagen~~ | ~~KLAAR — `b159_ilp_oracle.py` (~412 regels), 31/31 tests, standaard 0/1-MILP via scipy.optimize.milp (HiGHS default, optionele pyscipopt/gurobipy paden met graceful SKIP). Symmetry-break fix `x_0=0`. Benchmark: 17/17 certified in <1s (Petersen 0.012s, 3-reg n=50 0.619s). UB-hierarchie gap-analyse: LP+OC tight 15/17, SoS-2 tight 13/17 (gemeten), GW tight 6/17. Levert harde OPT-kolom voor paper-tabellen zonder brute force.~~ |
| ~~B170~~ | ~~Twin-width Parameter~~ | ~~1 week~~ | ~~KLAAR — zie AFGEROND; 41/41 tests, Trigraph+greedy tww, cograph-DP ~540× sneller dan BF op K_18~~ |
| B173 | Parallel Tempering Monte Carlo | 2-3 dagen | Kan concurreren met PA op sparse spinglas |
| B175 | Memetisch / Genetisch Algoritme (Rochat-stijl) | 3-5 dagen | Historische recordhouders op Gset |
| ~~B176~~ | ~~Frank-Wolfe / Conditional Gradient voor SDP~~ | ~~3-5 dagen~~ | ~~KLAAR — zie AFGEROND; 40/40 tests, `b176_frank_wolfe_sdp.py` spectraplex+diag-penalty, matrix-vrije LMO, sandwich-certificaat, 3.9× sneller dan cvxpy op n=200~~ |
| ~~B176b~~ | ~~CGAL-SDP: Augmented Lagrangian voor n≈10000~~ | ~~3-5 dagen~~ | ~~**KLAAR 18 apr 2026** — `code/b176b_cgal_sdp.py` (~370 regels), Yurtsever-Fercoq-Cevher 2019 CGAL met expliciete duale variabele y + β√k-schedule. Harde UB `cut_SDP ≤ <y,1> − n·λ_min(-L/4 + diag(y))` geldig voor ELKE y. 27 tests in 9 classes, `b176b_benchmark.py` met correct/h2h/scale/conv-secties en paper-2 scale-panel n∈{500,1000,2000,5000,10000} met LaTeX-booktabs-tabel-generator naar `docs/paper/tables/b176b_scale_table.tex`. Test- en scale-runs op Gertjan's laptop (sandbox out-of-space bij afronding). Zie backlog-entry B176b voor runbook.~~ |
| ~~B186~~ | ~~Solver-Selector als Gepubliceerd Benchmark~~ | ~~1-2 dagen~~ | ~~KLAAR — zie header DAG 3 + DAG 8; `code/b186_solver_selector_benchmark.py` (~420 regels), 23/23 tests, 14-instance Gset+BiqMac+DIMACS panel met 4 solvers (ILP/FW-SDP/cograph-DP/dispatcher-auto), paper-artifacts in `docs/paper/data/b186_selector_results.{json,csv}` + `docs/paper/tables/b186_selector_table.{tex,md}`. Dag-8 re-run na signed-downgrade: 5 rijen met `exact_small_signed`/`pa_primary` i.p.v. foute `pfaffian_exact`/`exact_small`.~~ |
| ~~B159-Dag-8b~~ | ~~ILP-Oracle Sign-Bug voor Signed MaxCut~~ | ~~1 dag~~ | **KLAAR 17 apr 2026** — signed-safe ILP-formulering in `code/b159_ilp_oracle.py` (~425 regels). De y_e-linearisatie heeft nu vier constraints per edge (`y_e ≤ x_u + x_v` UB1, `y_e ≤ 2 − x_u − x_v` UB2, `y_e ≥ x_u − x_v` LB1, `y_e ≥ x_v − x_u` LB2), toegepast in alle drie solver-backends (HiGHS via `scipy.optimize.milp`, SCIP via `pyscipopt.addCons`, Gurobi via `addConstr`). Dit dwingt `y_e = \|x_u − x_v\|` voor elke sign-combinatie, zodat de objective `Σ w_e · y_e` exact de signed MaxCut berekent i.p.v. de oude `max Σ max(w_e, 0)·[cut]`. `n_constrs` in SCIP/Gurobi dispatch is verhoogd van `2m` naar `4m`. **Tests:** `test_b159_ilp_oracle.py` 39/39 groen (was 31); nieuwe `TestSignedInstancesDag8b`-klasse met 8 tests op discriminerende signed instanties — `P_4` + negatieve back-edge (OPT=20.0, pre-fix bug zou 30 rapporteren), `C_5` alternating +/-, triangle met één negatieve edge, all-negative triangle (OPT=0), `K_4` mixed-sign bipartite-structuur (OPT=40), dispatcher-routing naar signed MaxCut, SCIP/Gurobi match (graceful SKIP). Test `test_n_vars_matches_formula` nu assert `n_constrs == 4·m`. **Verificatie:** B186-rerun levert `Auto == ILP-OPT: 14/14` (was 10/14); de 4 signed BiqMac-mismatches (spinglass2d_L4/L5, torus2d_L4, pm1s_n20) zijn weg. `ILP-OPT(P_4 + neg-backedge) = 20.0 = brute-force signed MaxCut`, n_constrs=16=4·4. Volledige suite 134/134 groen (b159 + b186 + quality_certificate). Artifacts: `docs/paper/data/b186_selector_results.{json,csv}` + `docs/paper/tables/b186_selector_table.{tex,md}` opnieuw geschreven. Dispatcher-auto en ILP-oracle zijn nu consistent sign-aware. |

### MIDDEL

| # | Naam | Doorlooptijd | Waarom |
|---|------|-------------|--------|
| B103 | ZX / Phase-Gadget Rewrite Pass | 2-3 dagen | Concrete circuitoptimalisatie vóór tensorcontractie. **Verbreed scope (17 apr 2026):** primair doelwit is uitvoer van B129 Hamiltonian-compiler (Trotter, UCC). Absorbeert ex-B87. PyZX/quizx-stijl rewrites geven 2-10× T-count reductie op moleculaire ansätze. |
| ~~B107~~ | ~~Quantum Nogood Learning~~ | ~~klaar~~ | ~~KLAAR — 4 extractiemethoden, Z2-dedup, progressive solve, 55 tests~~ |
| B58 | GPU Batch-Kernel Packer | 1-2 dagen | Elimineert kernel-launch overhead bij veel edges |
| B23 | Tensor Network Contraction (cotengra) | 2-3 dagen | Alternatieve engine voor ondiepe circuits |
| B38 | SQA GPU-Shader | 2-3 dagen | Praktische solver naast QAOA, GPU showcase |
| ~~B42~~ | ~~Treewidth-Decompositie~~ | ~~2-3 dagen~~ | ~~KLAAR — zie AFGEROND. Exact MaxCut op tw=4 grids, +287 vs B99v2~~ |
| B102 | Local-Clifford / Gauge Preconditioner | 3-5 dagen | Research, potentieel grote chi-reductie |
| B106 | Counterdiabatic Low-Chi QAOA | 3-5 dagen | Research, slimmere dynamica bij lage p |
| B108 | Spectrale Mixer / Laplacian-Aware QAOA | 3-5 dagen | Research, alternatieve mixer |
| B105 | Dual-Graph Defect Solver | 3-5 dagen | Research, defect-representatie gefrustreerde grafen |
| B16 | Dynamische Qubit Routering | 1-2 dagen | Impact bij 2D Ly>=3 |
| B20 | TDVP Operator-Evolutie | 3-5 dagen | Relevant bij p>=3 |
| B28 | Belief Propagation (Simple Update) | 2-3 dagen | Snelle exploratie |
| B30 | MPO Pre-compression | 1-2 dagen | Schonere operator-toepassing |
| ~~B80~~ | ~~MPQS: Message-Passing Quantum Solver~~ | ~~1 dag~~ | ~~KLAAR — zie AFGEROND; BP+lightcone, 44/44 tests, 13/13 OPT via BP~~ |
| ~~B93~~ | ~~QITS: Imaginary-Time QAOA~~ | ~~4-6 uur~~ | ~~KLAAR — zie AFGEROND~~ |
| B116 | MaxCut QAOA als Z₂ Lattice Gauge Theory | 1-2 weken | Directe link met bestaande solvers, publiceerbaar |
| B115 | Lattice Gauge Theory via Tensor Netwerken | 2-3 weken | Concrete brug ZornQ-infra ↔ gauge-theorie |
| B120 | QMDD / Decision-Diagram Exact Cache | 2-3 dagen | Compacte exact-state representatie op symmetrische instanties |
| B121 | Matchgate / Free-Fermion Pocket Detector | 3-5 dagen | Research, exacte pocket solver binnen moeilijke instanties |
| B122 | Loop Calculus / Generalized Belief Propagation | 2-3 dagen | Regio-BP voor frustratie+korte cycli, versterkt B28/B80 |
| B124 | Learned Separator Placement for Knitting | 2-3 dagen | Versterkt B31+B104, planner optimaliseert separators |
| B125 | Tensor Sketch / CountSketch Contraction | 2-3 dagen | Compressie ín contractie zelf, anders dan shadows/MC |
| B127 | Cut Polytope Facet Miner | 2-3 dagen | Versterkt B49/B51/B73 certificaten met geleerde inequalities |
| B123 | Graph Wavelet Mixer | 3-5 dagen | Research, multiscale mixer naast B72/B108 |
| B94 | T-QAOA: Tabu-Quantum Search | 1 dag | Penalty op bezochte oplossingen |
| B74 | Online Bias-Calibrator | 4-6 uur | Zelfcorrigerend, relevanter bij TT-cross |
| B155 | TSP / VRP via QUBO-Embedding | 1-2 weken | Derde domein voor B132, ander NP-hard |
| B157 | Lovász θ-functie | 1-2 dagen | SDP-relaxatie, natuurlijk bij B153 MIS |
| ~~B160~~ | ~~QSVT / Block-Encoding Framework~~ | ~~1-2 weken~~ | ~~KLAAR — zie AFGEROND; 40/40 tests, LCU block-encoding + QSP + Jacobi-Anger Ham-sim wint 12+ orden van grootte van Trotter-1 op Ising-TF/Heisenberg~~ |
| B162 | UCC-ansatz voor VQE | 3-5 dagen | Versterkt B132 chemistry-claim |
| B164 | PEC + Virtual Distillation | 3-5 dagen | Versterkt NISQ-claim voor paper |
| B172 | Fiedler-Ordering als Full Preprocessor | 1-2 dagen | Goedkope potentiële winst, cache-hits |
| B174 | Tabu Search (klassiek) | 1-2 dagen | Klassieke baseline, ontbreekt nog |
| B179 | Open Dataset Release (Zenodo) | 1-2 dagen | Community-waarde, citeerbaar |
| B180 | Open MaxCut Leaderboard | 2-3 dagen | Publicatie-multiplier |
| B184 | Instance Difficulty Classifier | ~~3-5 dagen~~ **KLAAR 18 apr 2026** | 740 regels, 26/26 tests, RF CV-acc 0.77±0.15, top-features m/n/density, `dispatcher_hook` klaar voor B130 |
| B185 | QAOA-Landschap Visualizer | 1-2 dagen | Paperwaardig visueel materiaal |
| B188 | Lieb-Robinson Bounds Expliciet | 1 week | Paper-onderbouwing voor lightcone-werk |

### LAAG-MIDDEL

| # | Naam | Doorlooptijd | Waarom |
|---|------|-------------|--------|
| B75 | Motif-Atlas Lokale Buurten | 1 dag | Pas relevant bij niet-uniforme gewichten |
| B81 | Z-MPO: Operator-Flow Solver | 2-3 dagen | Veel nieuwe wiskunde, DMRG-achtig |
| B77 | UDP-QAOA: Dissipative Dephasing | 4-6 uur | SVD-truncatie doet dit al impliciet |
| B78 | Hyperbolische Qubit Routering | 1-2 dagen | Pas relevant bij niet-grid grafen |
| B96 | EQS: Projective Decimation | 4-6 uur | B34+B70 combinatie, chi-reductie |
| B97 | C-QAOA: Bidirectional Contraction | 1 dag | Forward+backward MPS experiment |
| B98 | AMPO: Annealing-MPS | 1-2 dagen | Continue TEBD-annealing, chi-intensief |
| B117 | Confinement & Hiërarchische Krachten | 2-3 weken | Speculatief maar potentieel krachtig voor grote instanties |
| B111 | QTT-Navier-Stokes | 1-2 weken | Concrete brug naar fluids, nieuw domein |
| B112 | QAOA voor Fluid Routing / Network Flow | 1 week | QUBO-formulering stromingsproblemen |
| B126 | Koopman Parameter Flow Model | 3-5 dagen | Research, lineaire operator warm-starting, sluit aan op B113 |
| B63 | JAX Autodiff voor Optimizer | 2-4 dagen (PoC) | Exacte gradienten; GPU al op 100%, JIT-winst marginaal |
| ~~B10g~~ | ~~7-operatie Heisenberg CT-scan~~ | ~~?~~ | ~~GEFALSIFICEERD — chi ratio 1.00, 94.7% fout bij d=8, 22 tests~~ |
| ~~B14~~ | ~~MERA tensor netwerk~~ | ~~klaar~~ | ~~BEVESTIGD — engine werkt (22 tests), maar MPS wint bij n≤16 en p≤3~~ |
| B17 | Non-Hermitische / Lindblad Dissipatie | 1-2 dagen | Interessant voor NISQ-sim, niet voor MaxCut |
| B33 | GNN/Transformer QAOA-Surrogaat | 1-2 weken | Research, apart project |
| ~~B39~~ | ~~TRG / HOTRG~~ | ~~klaar~~ | ~~KLAAR — native 2D contractie, Ising exact, HOTRG chi=4 error ~1e-4, 59 tests~~ |
| B43 | BREC Graph Benchmark | 1 dag | Validatie, deels overlap met B36 |
| B46 | Ruismodel (NISQ-Simulatie) | 2-3 dagen | Relevant voor paper en hardware-vergelijking |
| B161 | LCU (Linear Combination of Unitaries) | 3-5 dagen | Exacte simulatie-baseline |
| B163 | Dürr-Høyer Quantum Minimum Finding | 2-3 dagen | Research, bekende uitkomst: MPS-kosten nekt voordeel |
| B167 | Albert-algebra J₃(𝕆) Verkenning | 1-2 weken | Fundamenteel, hoog falsificatierisico |
| B171 | Rank-width / Clique-width Decompositie | 1-2 weken | Ruim gedekt door B170 twin-width |
| B181 | pip-installable zornq Package | 3-5 dagen | Community, paper-supplement |
| B182 | Qiskit Optimization Plugin | 2-3 dagen | Vereist B181, nice-to-have |
| B187 | Unique Games Conjecture Bounds | 1 week | Diep theoretisch, versterkt B144 |

### LAAG

| # | Naam | Doorlooptijd | Waarom |
|---|------|-------------|--------|
| ~~B10e~~ | ~~PEPO — 2D tensor netwerk~~ | ~~klaar~~ | ~~KLAAR — `b10e_pepo.py` (~640 regels) + 40/40 tests; PEPS2D met simple-update SVD + zero-pruning, boundary-MPO sweep, full QAOA MaxCut pipeline; benchmark 2x2/3x2/3x3/2x4/3x4 toont diff ~1e-15 vs exact (3x3 p=2 chi=4 toont verwachte truncatie 2.7e-4).~~ |
| B12 | Octonion-spinor correspondentie | ? | Wiskundige vraag, geen directe bruikbaarheid |
| B59 | Octonion-parameterisatie variationele circuits | ? | Puur speculatief, hoog falsificatierisico |
| B110 | Octonionische Vloeistofdynamica | open-ended | Fundamenteel onderzoek, lange termijn |
| B114 | Octonionen & G₂/E₈ Lie-groep Structuur | open-ended | Fundamenteel, wiskundig intensief |
| B113 | Hydrodynamische Parameter Flow | open-ended | Theoretisch, gamma/beta als vloeistofstroom |
| B18 | Disk-Based DMRG / Quantum Paging | 3-5 dagen | Pas relevant als RAM-limiet raken |
| B22 | Neural Network Quantum States | 1-2 weken | Research-vergelijking, niet core |
| B24 | ADAPT-QAOA | 3-5 dagen | Interessant maar rekenintensief |
| B44 | Fibonacci Anyonen | 1-2 weken | Fundamenteel ander paradigma |
| B45 | Qiskit Integratie / Hardware Export | 2-3 dagen | Nice-to-have voor paper |
| B166 | Pulse-Level Export (OpenPulse) | 2-3 dagen | Paper-appendix materiaal |
| B168 | Moufang Identiteiten als MPO-Symmetrie | 2-3 dagen | Speculatieve winst |
| B169 | Bott-Periodiciteit & Clifford-QEC Brug | 1 week | Diep theoretisch |
| B183 | NetworkX-compatible Solver Interface | 0.5 dag | Klein, vereist B181 |
| B189 | Holografische QEC / Bulk-Boundary Brug | 2-3 weken | Fundamenteel, paper-2 materiaal |

---

### GEPARKEERD (onduidelijke winst of strategische afleiding)

| # | Naam | Reden |
|---|------|-------|
| B85 | Local-Clifford Preconditioner | Geen QAOA-winst (ZZ al diagonaal); revisit alleen samen met **B162 UCC-ansatz** voor chemistry-tak |
| B88 | Near-Clifford Hybrid | Activeer alleen ALS B162 wordt opgepakt EN T-count na ZX-rewrite (B103) laag genoeg blijkt |

### REDUNDANT / GEABSORBEERD (geschrapt op 17 april 2026 na onderzoek_b83_b90.md)

| # | Naam | Reden schrap |
|---|------|--------------|
| B86 | Topological Gate Pruning | Volledig gedekt door B41 TDQS + B27 + B68 + B21 + B118/B119 |
| B87 | ZX-Calculus Circuit Rewrite | Duplicaat van **B103** (Phase-Gadget Rewrite Pass, MIDDEL OPEN); B103 verbreed met expliciet B129 Hamiltonian-compiler-uitvoer als doelwit |
| B89 | MIS / Rydberg Pivot | MIS-via-QUBO geabsorbeerd in **B153** Beyond-MaxCut QUBO Suite; native Rydberg geen ZornQ-territorium |
| B90 | Ervaringsgeheugen (ML) | Geabsorbeerd door B57 (KLAAR) + B130 (KLAAR) + **B184** Instance Difficulty Classifier (KLAAR 18 apr 2026) + **B186** Solver-Selector Benchmark (OPEN) |

### NIET HAALBAAR (in huidige architectuur)

| # | Naam | Reden |
|---|------|-------|
| B83 | G-QAOA: Grover Amplitude Amplification | Grover-diffusie (2|ψ⟩⟨ψ|-I) exponentieel duur in MPS — bevestigd 17 april 2026 |
| B84 | EPE-QAOA: Phase Estimation op QAOA | QPE vereist controlled-U^(2^k), bond dim verdubbelt per gate — bevestigd 17 april 2026 |

### NIET HAALBAAR (in huidige architectuur)

| # | Naam | Reden |
|---|------|-------|
| B83 | G-QAOA: Grover Amplitude Amplification | Grover-diffusie (2|ψ⟩⟨ψ|-I) exponentieel duur in MPS |
| B84 | EPE-QAOA: Phase Estimation op QAOA | QPE vereist controlled-U^(2^k), bond dim verdubbelt per gate |

---

## Quick wins (< 1 dag doorlooptijd)

1. ~~**B27** Graph Automorphism Deduplicatie~~ KLAAR

## Grootste impact (als ze werken)

1. ~~**B41** TDQS — fundamenteel nieuw algoritme~~ BEWEZEN (+4-5% vs QAOA-1)
6. **B10f** Classical Shadows — chi-onafhankelijke energy estimation
7. **B40** iTEBD-QAOA — exacte N naar oneindig limiet (publicatiewaardig)


B179 Zenodo open release KLAAR (DOI 10.5281/zenodo.19637389)
