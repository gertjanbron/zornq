# B165b — Hardware Run op IBM Quantum (run-instructies)

Deze map bevat een kant-en-klaar pakket om **één echte run** op IBM Quantum-hardware
te doen voor de paper-tabel. Claude heeft geen IBM-token, dus de echte submit
gebeurt op jouw eigen laptop. Wat hier al klaarstaat:

- `code/b165b_hardware_submit.py` — submit-script met `--dry-run`, `--submit`, `--resume`
- `code/b165b_noise_baselines.py` — drie Aer-baselines (noiseless, depolariserend, calibration-mirror)
- `code/b165b_parse_results.py` — produceert `docs/paper/tables/b165b_hardware_table.{md,tex}`
- `code/test_b165b_hardware_submit.py` — 13 unit-tests (100% groen)

De baselines op Aer zijn **al lokaal gedraaid**; alleen de hardware-kolom wacht
op jouw submit. Het parser-script vult die kolom automatisch in zodra je het
`job_id` op disk hebt.

---

## Stappenplan

### 1. Eenmalig: Python-environment

```bash
pip install qiskit>=1.0,<3.0 qiskit-aer>=0.14 qiskit-ibm-runtime>=0.20
# of:  pip install -e ".[qiskit]"   (vanuit de project-root)
```

### 2. IBM Quantum account + token

- Maak een gratis account op https://quantum.cloud.ibm.com/ (IBM Quantum Platform).
- Haal je API-token op via je account-pagina.
- Zet de token als env-var **buiten deze project-folder** (zo ziet Claude/Cowork hem niet):

  **macOS/Linux (zsh/bash):**
  ```bash
  echo 'export QISKIT_IBM_TOKEN="<jouw_token>"' >> ~/.zshrc
  source ~/.zshrc
  ```

  **Windows PowerShell:**
  ```powershell
  [Environment]::SetEnvironmentVariable("QISKIT_IBM_TOKEN", "<jouw_token>", "User")
  # herstart je terminal
  ```

  **Alternatief:** zet de token in een losse tekst-file ergens buiten deze repo,
  bijv. `~/.ibm_quantum_token`, en geef `--token-file ~/.ibm_quantum_token` mee
  aan het script. Dezelfde waarschuwing: niet binnen de Cowork-geselecteerde
  folder opslaan.

### 3. Dry-run (GEEN queue-tijd, GEEN kosten)

```bash
cd "<project_root>/code"
python b165b_hardware_submit.py --dry-run
```

Dit:
- checkt of je token gelezen kan worden,
- bouwt de twee circuits (3reg8 + myciel3) met een korte grid-search voor (γ, β),
- persist `docs/paper/hardware/jobs/prepared_3reg8.json` + `prepared_myciel3.json`
  zodat je later exact dezelfde circuits kunt submitten.

De output moet eindigen met `"token_found": true` als je token goed staat.

### 4. Kies een backend

```bash
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
import os
s = QiskitRuntimeService(channel='ibm_quantum', token=os.environ['QISKIT_IBM_TOKEN'])
for b in s.backends(operational=True, simulator=False):
    print(f'{b.name}  n_qubits={b.num_qubits}  pending_jobs={b.status().pending_jobs}')
"
```

Kies een backend met:
- `n_qubits >= 11` (voor myciel3; 8 is voldoende voor 3reg8),
- bescheiden `pending_jobs` (minder wachttijd).

**Suggestie:** `ibm_brisbane`, `ibm_sherbrooke`, of `ibm_kyoto` (Eagle/Heron).

### 5. (Optioneel maar aanbevolen) Calibration-snapshot ophalen

Hiermee bouwen de baselines een Aer-NoiseModel met de **echte T1/T2 + gate-errors**
van je gekozen backend, niet een generieke depolariserende benadering:

```bash
python b165b_noise_baselines.py --fetch-snapshot-from ibm_brisbane \
    --only 3reg8
```

Het script schrijft `docs/paper/hardware/baselines/ibm_brisbane_snapshot.json`
en gebruikt die meteen voor de baseline-run. Herhaal met `--only myciel3`.

### 6. Echte submit

```bash
python b165b_hardware_submit.py --submit ibm_brisbane --shots 4096
```

Wat er gebeurt:
1. 20×20 grid-search op Aer-noiseless voor optimale (γ, β) — ~10 s per instantie.
2. Transpile circuit tegen de gekozen backend.
3. `SamplerV2.run([circuit], shots=4096)` → `job_id` wordt direct naar
   `docs/paper/hardware/jobs/<job_id>.json` geschreven. **Sla dit job_id op!**
4. Het script pollt op de queue. Op free-tier kan dit **uren tot dagen** duren.

Tip: gebruik `--only 3reg8` voor een korte controle-run, of `--only myciel3`
voor de grotere instantie. Zonder `--only` submit het script sequentieel allebei.

### 7. (Als je laptop tussendoor uitgaat) Resume

Als het polling-proces wordt afgebroken:

```bash
python b165b_hardware_submit.py --resume <job_id>
```

Het script haalt de counts op en schrijft ze naar hetzelfde bundle-bestand.

### 8. Paper-tabel regenereren

Zodra één (of beide) job_id(s) status=COMPLETED heeft:

```bash
python b165b_parse_results.py
```

Dit schrijft `docs/paper/tables/b165b_hardware_table.{md,tex}` met 4 kolommen
(noiseless | depolariserend | calibration-mirror | hardware) + OPT + approx.ratio.

---

## Wat te doen als iets fout gaat

**"GEEN QISKIT_IBM_TOKEN gevonden"**
→ Env-var niet gezet. Herstart je terminal na het zetten. Check `echo $QISKIT_IBM_TOKEN`.

**"IBMQ credentials invalid"**
→ Token verlopen of wrong channel. Bij IBM Quantum Platform: `channel="ibm_quantum"`.
Bij IBM Cloud Qiskit Runtime: `channel="ibm_cloud"` + `instance="<CRN>"` → geef door
via `--instance-hub <crn>`.

**"Backend has no quantum jobs capability" / "Too large to transpile"**
→ Kies een grotere backend. myciel3 heeft 11 qubits nodig; niet alle 5-qubit
machines voldoen.

**Queue-tijd is te lang**
→ Free-tier kan soms uren duren. Eerst `--only 3reg8` (8 qubits, snellere queue
op small-qubit backends). Je kunt 24 uur wegblijven en later `--resume` gebruiken.

---

## Veiligheidsafspraken

- **De token gaat nooit in deze folder.** Zet hem in je OS-env-var of in een
  file buiten deze project-directory. Cowork/Claude ziet geen env-vars van jouw
  lokale shell.
- Als je `.env`-files gebruikt: **niet** in de folder die Claude/Cowork mag lezen.
- Bij het delen van resultaten naar de paper-tabel: de tabel bevat alleen counts,
  niet de token.

---

## Wat Claude al heeft gedraaid (lokaal op Aer)

Zie `docs/paper/hardware/baselines/` voor de JSONs:

- `3reg8_noiseless.json`           — Aer zonder ruis
- `3reg8_depolarising.json`        — Aer met generieke 1q/2q depolariserende ruis
- `3reg8_calibration_mirror.json`  — Aer met fallback cal-mirror (zonder snapshot)
- idem voor `myciel3_*.json`

En in `docs/paper/hardware/jobs/`:

- `prepared_3reg8.json`            — circuit + geoptimaliseerde (γ, β)
- `prepared_myciel3.json`          — idem

De tabel in `docs/paper/tables/b165b_hardware_table.md` heeft de hardware-kolom
nog leeg ("—"); die vult zichzelf zodra jij stap 6 + 8 hebt gedraaid.
