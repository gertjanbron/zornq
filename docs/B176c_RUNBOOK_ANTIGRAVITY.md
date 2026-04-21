# B176c — Antigravity Runbook (laptop-GPU uitvoering)

**Doel.** De sandbox-implementatie van B176c (GPU-eigsh voor CGAL-SDP) is
klaar en groen op CPU. Dit runbook draait het GPU-deel op de laptop
(GTX 1650), meet de echte speedup, schrijft een results-document, en sluit
de backlog-entry. Optioneel: een kleine refactor in
`b176_frank_wolfe_sdp.lmo_spectraplex` zodat `cgal_maxcut_sdp` een
end-to-end schaalpanel kan draaien.

**Geschatte wall-time op laptop.** 45-90 min benchmark + 30 min results +
30 min commits + optioneel 60 min lmo_spectraplex refactor = 2-3 uur
totaal.

---

## 0. Waarschuwingen vooraf — lees dit EERST

Uit de NeurIPS26-fix-run van 20 apr bleek dat een agent een "template swap"
kan claimen terwijl het asset-bestand op disk nooit is vervangen. Voor
B176c moet je bij elke verificatie **letterlijk de tool-output pasten**, niet
parafraseren. De volgende hallucination-traps gelden:

- **Installatie ≠ werkend.** `pip install cupy-cuda12x` kan slagen terwijl
  CUDA-runtime ontbreekt. Altijd testen met `python -c "import cupy;
  cupy.cuda.Device(0).compute_capability"` — dit raises bij geen CUDA.
- **Import ≠ werkend eigsh.** CuPy importeert vaak op systemen zonder
  werkende CUDA (alleen bij het eerste kernel-launch klapt het). De
  `available_backends()`-helper in `b176c_gpu_eigsh.py` checkt dit goed —
  gebruik hem.
- **Benchmark-uitvoer ≠ correcte uitvoer.** Een CSV met 0 rijen betekent
  failure, niet success. Check rij-aantallen en kolommen.
- **Wall-time-claim ≠ gemeten.** Paste CSV-head + kolom-aggregates, niet
  een samenvatting.

Als een stap faalt, **stop** en rapporteer. **Claim geen success** zonder
alle V1-V6-verificaties letterlijk gepasted.

---

## 1. Preconditie-check

Werk in de repo-root:

```bash
cd "/c/Users/Me/Documents/Onderzoeksprojecten/103. ZornQ - Octonionische Quantum Computing"
git status
git switch main
git pull --ff-only
git switch -c b176c-gpu-eigsh main
```

**V1 (paste letterlijk):** output van

```bash
python -c "import sys; print(sys.version)"
python -c "import cupy" 2>&1 | head -3 || echo "CuPy not installed"
nvidia-smi 2>&1 | head -15 || echo "nvidia-smi not found"
```

Als CuPy al werkt en nvidia-smi een GPU toont: ga door naar stap 3.
Anders stap 2.

---

## 2. CuPy + CUDA installeren (alleen als nog niet aanwezig)

Detecteer eerst de CUDA-versie:

```bash
nvidia-smi | grep "CUDA Version"
```

Kies op basis daarvan het juiste pakket:

| CUDA Version | pip-pakket                       |
|--------------|----------------------------------|
| 11.x         | `pip install cupy-cuda11x`       |
| 12.x         | `pip install cupy-cuda12x`       |
| Geen/onduidelijk | STOP — vraag aan user         |

Installeer en verifieer:

```bash
pip install cupy-cuda12x              # OF cupy-cuda11x, aangepast op V1-output
python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.getDeviceCount())"
python -c "import cupy; print(cupy.cuda.Device(0).compute_capability)"
```

Verwachte output van de laatste: een string zoals `'75'` (Turing) of `'86'`
(Ampere). Voor GTX 1650 is `'75'` correct.

**V2 (paste letterlijk):** output van

```bash
python -c "import cupy; \
           print('cupy', cupy.__version__); \
           print('cuda runtime', cupy.cuda.runtime.runtimeGetVersion()); \
           print('device name', cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()); \
           print('compute cap', cupy.cuda.Device(0).compute_capability); \
           print('mem (MB)', cupy.cuda.runtime.memGetInfo()[1] // (1024*1024))"
```

**Let op:** `cupy.cuda.Device` heeft GEEN `.name` attribuut (in
tegenstelling tot bijv. PyTorch). De device-naam komt uit
`cupy.cuda.runtime.getDeviceProperties(id)['name']` (bytes → decode).
Vroegere versies van dit runbook hadden `.name` als typo; gecorrigeerd.

Als deze command faalt of een Python-exception werpt: **stop** en
rapporteer. Ga niet door met een fallback.

---

## 3. Unit-tests op laptop draaien

```bash
cd code
python -m pytest test_b176c_gpu_eigsh.py -v
cd ..
```

**V3 (paste letterlijk):** laatste 30 regels van de pytest-output. Verwacht:

- **22 passed** (alle CuPy-tests draaien nu ook, want GPU aanwezig)
- 0 failed
- 0 errors

Als een CuPy-test faalt: fix het pas als het een echte bug blijkt
(correctness-verschil > places=3). Als het een tolerance-probleem is,
rapporteer het en ga door.

**V3-kritiek — DO NOT PROCEED IF:**

- Aantal passed < 19 → iets is kapot, geen GPU-benchmark draaien
- `test_cupy_lanczos_matches_ground_truth` of `test_cupy_lobpcg_matches_ground_truth`
  faalt met een grote error → eigsh-backend is numeriek broken

---

## 4. Full micro-benchmark draaien

Dit is de kernmeting. Duurt 30-60 min op GTX 1650 voor de volledige panel.

```bash
mkdir -p docs/paper/data
python code/b176c_benchmark.py --only micro --seeds 3 2>&1 | tee /tmp/b176c_micro.log
```

Als het langer duurt dan 90 min: Ctrl-C, halveer `--seeds` naar 2 of
laat n=10000 weg (edit `ns_micro` in `main()`). Documenteer deze
aanpassing in stap 6.

**V4 (paste letterlijk):** output van

```bash
echo "=== CSV row count ==="
wc -l docs/paper/data/b176c_micro.csv
echo ""
echo "=== CSV header ==="
head -1 docs/paper/data/b176c_micro.csv
echo ""
echo "=== CSV unique backends ==="
awk -F, 'NR>1 {print $3}' docs/paper/data/b176c_micro.csv | sort -u
echo ""
echo "=== CSV unique n-values ==="
awk -F, 'NR>1 {print $1}' docs/paper/data/b176c_micro.csv | sort -un
echo ""
echo "=== PNG file ==="
ls -la docs/paper/data/b176c_walltime.png
echo ""
echo "=== Last 15 lines of benchmark log (samenvatting-tabel) ==="
tail -20 /tmp/b176c_micro.log
```

**V4-kritiek — DO NOT PROCEED IF:**

- CSV heeft minder dan 60 rijen (verwacht: 5 n × 3 seeds × 4 backends × 2
  cold/warm = 120; bij skips minimum 60)
- `cupy_lobpcg` of `cupy_lanczos` ontbreekt in de unique backends-lijst —
  de GPU-pad heeft dan niet gedraaid
- PNG ontbreekt of is < 20 KB

---

## 5. Speedup-analyse + results-document

Genereer een kleine analyse (zet dit in een Python-script of Jupyter,
niet in dit runbook inline):

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("docs/paper/data/b176c_micro.csv")
# Pivot: rows = n, cols = (backend, warm), value = mean wall_steady
piv = df.groupby(["n", "backend", "warm_start"])["wall_steady"].mean().unstack(["backend", "warm_start"])
print(piv.to_string())
print()
# Speedup t.o.v. scipy_arpack cold
base = piv.get(("scipy_arpack", 0))
if base is None:
    raise SystemExit("scipy_arpack cold ontbreekt")
for col in piv.columns:
    speedup = base / piv[col]
    print(f"{col} speedup (baseline=arpack cold):")
    print(speedup.to_string())
    print()
PY
```

**V5 (paste letterlijk):** de output van bovenstaand script — de
speedup-tabellen voor alle backend-combos vs `scipy_arpack/cold`.

Schrijf nu `docs/B176c_RESULTS.md` (maak het bestand, ~60-100 regels):

```markdown
# B176c — GPU-eigsh benchmark resultaten

**Datum:** YYYY-MM-DD
**Hardware:** <CPU-model>, <RAM>, NVIDIA <GPU-model> (compute cap <XX>)
**CuPy:** <version>   **CUDA runtime:** <version>
**OS:** <windows/linux/...>

## 1. Samenvatting

- Baseline (scipy_arpack, cold start) op n=10000: <X> s/call
- Best backend (cupy_lobpcg, warm start) op n=10000: <Y> s/call
- Speedup: **<X/Y>x**

Optioneel: [tabel met n vs speedup per backend]

## 2. Per-call wall-time vs n

Zie `docs/paper/data/b176c_walltime.png`.

Observaties:
- [korte bullets — welke backend wint waar, vanaf welke n, etc.]

## 3. Warm-start-effect

- Gemiddelde matvec-reductie door warm-start (over alle n en backends):
  <Z>%
- Specifiek voor cupy_lobpcg: [voor en na]

## 4. Conclusie + paper-2-impact

- Bevestigt / weerspreekt de "n=10000 haalbaar op laptop-GPU"-claim
  voor paper-2.
- [andere relevante observaties]

## 5. Raw data

- `docs/paper/data/b176c_micro.csv` (<X> rijen)
- `docs/paper/data/b176c_walltime.png`
```

Vul alle `<...>` velden met ECHTE getallen uit V5. **Niet met placeholder
of "TBD".**

---

## 6. Commits + backlog-update

```bash
cd "/c/Users/Me/Documents/Onderzoeksprojecten/103. ZornQ - Octonionische Quantum Computing"
git add code/b176c_gpu_eigsh.py code/test_b176c_gpu_eigsh.py code/b176c_benchmark.py
git add docs/paper/data/b176c_micro.csv docs/paper/data/b176c_walltime.png
git add docs/B176c_RESULTS.md docs/B176c_RUNBOOK_ANTIGRAVITY.md
git commit -m "B176c: GPU-eigsh module + 4 backends + micro-benchmark (sandbox-deel)"

# Eventuele benchmark-edits (als je --seeds verlaagd hebt etc) noteren
git add code/b176c_benchmark.py   # als je het aangepast hebt
git diff --cached | head          # zichtbaar maken
# commit alleen als er echt iets aangepast is

git commit -m "B176c: benchmark-resultaten op GTX 1650 + results-document" --allow-empty
```

Update nu `docs/backlog_prioriteit.md`:

- IN-PROGRESS tabel B176c-rij: status veranderen van
  "sandwich-deel KLAAR — laptop-GPU-benchmark pending" naar
  **"KLAAR YYYY-MM-DD — laptop-GPU-benchmark <X>x speedup op n=10000
  gemeten, zie docs/B176c_RESULTS.md"**. Verplaats de rij naar HOOG-tabel
  met `~~strikethrough~~` + "KLAAR" status.
- Update status-line in `docs/zornq_backlog.md` (regel 2): voeg aan het
  begin toe: `## Status: YYYY-MM-DD, na **B176c KLAAR (GPU-eigsh <X>x
  speedup op n=10000 met cupy_lobpcg+warm-start)** + ` gevolgd door
  bestaande inhoud.

```bash
git add docs/backlog_prioriteit.md docs/zornq_backlog.md
git commit -m "B176c: backlog-update KLAAR met gemeten speedup"
git push -u origin b176c-gpu-eigsh
```

**V6 (paste letterlijk):** output van

```bash
git log --oneline main..HEAD
echo "---"
git status
echo "---"
git ls-remote origin b176c-gpu-eigsh
```

Verwacht: 2-3 commits, working tree clean, remote heeft de branch.

---

## 7. PR openen (handmatig)

Plak deze URL in de browser om een PR te starten:

```
https://github.com/gertjanbron/zornq/pull/new/b176c-gpu-eigsh
```

Titel:

> B176c: GPU-eigsh voor CGAL-SDP (<X>x speedup op n=10000)

Beschrijving (paste deze template, vul in):

```markdown
## Wat

Nieuwe module `code/b176c_gpu_eigsh.py` met 4 eigsh-backends
(scipy_arpack, scipy_lobpcg, cupy_lanczos, cupy_lobpcg) + warm-start-
ondersteuning. Onblokkeert paper-2's "n=10000"-claim.

## Gemeten (GTX 1650)

- n=10000 per-call wall-time: **<baseline>s → <best>s (<speedup>x)**
- Warm-start matvec-reductie: gemiddeld <Z>%
- Best backend: **<cupy_lobpcg / cupy_lanczos> + warm-start**

Zie `docs/B176c_RESULTS.md` voor de volledige analyse.

## Changed files

- `code/b176c_gpu_eigsh.py` (+N)
- `code/test_b176c_gpu_eigsh.py` (+M)
- `code/b176c_benchmark.py` (+K)
- `docs/B176c_RESULTS.md` (+...)
- `docs/paper/data/b176c_{micro.csv, walltime.png}`
- `docs/backlog_prioriteit.md`, `docs/zornq_backlog.md`

## Integratie-follow-up (separate PR, optioneel)

Om CGAL-end-to-end op n=10000 te draaien is er een kleine refactor
nodig in `b176_frank_wolfe_sdp.lmo_spectraplex`: de matvec-callable
vervangen door een `(L, z, coef_L)`-tuple zodat `gpu_eigsh_smallest`
direct aangeroepen kan worden. Zie runbook sectie 8.

## Reviewer-checklist

- [ ] V3 pytest 22/22 lokaal geverifieerd
- [ ] V4 CSV row-count + unique backends OK
- [ ] V5 speedup-tabel klopt met `b176c_micro.csv`
- [ ] `docs/B176c_RESULTS.md` heeft geen `<...>` placeholders
```

---

## 8. OPTIONEEL — lmo_spectraplex refactor (aparte PR)

Dit is nodig om `cgal_maxcut_sdp` een end-to-end scale-panel te laten
draaien met cupy_lobpcg. Niet verplicht voor B176c-afsluiting; zet het
op de backlog als B176d indien tijd ontbreekt.

Kern van de refactor in `code/b176_frank_wolfe_sdp.py`:

```python
# was:
def lmo_spectraplex(matvec, n, tol=1e-8, maxiter=1000, dense_fallback_below=40,
                   dense_G=None): ...

# wordt (additief — oude API blijft werken):
def lmo_spectraplex(matvec, n, tol=1e-8, maxiter=1000, dense_fallback_below=40,
                   dense_G=None, L=None, z=None, coef_L=-0.25,
                   backend="auto", v_prev=None): ...
    if L is not None and z is not None:
        # nieuwe pad
        from b176c_gpu_eigsh import gpu_eigsh_smallest
        res = gpu_eigsh_smallest(L, z, coef_L=coef_L, v0=v_prev,
                                 tol=tol, backend=backend,
                                 dense_fallback_below=dense_fallback_below)
        return res.v, res.lam
    # oude pad (fall-through naar bestaande code)
    ...
```

In `b176b_cgal_sdp.cgal_maxcut_sdp` voeg `eigsh_backend: str = "auto"`
parameter toe, hergebruik `v_prev` via een locale variabele, en geef
`L=L, z=z, backend=eigsh_backend, v_prev=v_prev` door aan
`lmo_spectraplex`. Idem voor `dual_upper_bound`.

Schrijf dan een tweede sectie in `b176c_benchmark.py scale_panel` die
de nu-werkende backends draait. Commit als aparte PR
`b176c-integration`.

**V7 (alleen als je stap 8 doet) — paste letterlijk:**

```bash
python -m pytest code/test_b176_frank_wolfe_sdp.py code/test_b176b_cgal_sdp.py -q
echo "---"
python -c "from b176b_cgal_sdp import cgal_maxcut_sdp; import inspect; \
           print('eigsh_backend' in inspect.signature(cgal_maxcut_sdp).parameters)"
```

Moet `True` printen en alle bestaande tests moeten nog groen zijn.

---

## Final report format

Plak letterlijk in de thread (NIET parafraseren):

```
## B176c Antigravity-run -- FINAL REPORT

### V1 (Python + CuPy-pre-check)
<paste>

### V2 (CuPy + CUDA identification)
<paste>

### V3 (pytest test_b176c_gpu_eigsh.py)
<paste last 30 lines>

### V4 (CSV + PNG + benchmark tail)
<paste all 5 echo-blocks output>

### V5 (Pandas speedup-analyse)
<paste>

### V6 (git log + status + remote)
<paste>

### V7 (OPTIONAL, alleen bij sectie 8)
<paste>

### Summary
- Best backend op n=10000: <naam>
- Speedup: <X>x t.o.v. scipy_arpack cold
- PR URL: https://github.com/gertjanbron/zornq/pull/<NN>
- docs/B176c_RESULTS.md committed in <commit-sha>

### Claims die ik NIET gedaan heb zonder verificatie
- [x] Ik heb V1-V6 letterlijk gepasted, niet geparafraseerd.
- [x] De wall-time + speedup getallen komen uit de CSV, niet uit eigen
      interpretatie.
- [x] Als iets is misgegaan staat dat hierboven, niet verstopt.
```

**Klaar is het pas als alle V-secties gepasted zijn. Geen uitzonderingen.**
