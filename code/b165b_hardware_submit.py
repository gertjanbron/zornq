#!/usr/bin/env python3
"""
B165b: Hardware-Submit Klaarzetter (IBM Quantum Runtime).

Doel
----
Dit script bereidt voor, maar voert NIET zelf uit op hardware tenzij de gebruiker
het expliciet doet. Claude heeft geen IBM-token nodig; de gebruiker draait dit
script op zijn/haar eigen laptop met een eigen token.

Ondersteunde instanties (Dag 5)
-------------------------------
- `3reg8`    : 3-reguliere random graaf n=8, seed=0   (8 qubits)
- `myciel3`  : DIMACS Mycielski M_3, n=11, m=20       (11 qubits)

Pad
---
1. Bouw ZornQ-Circuit (QAOA p=1) voor elke instantie met **vooraf-geoptimaliseerde**
   (γ, β) uit een 20×20 grid-search op Aer (noiseless).
2. Schrijf alle circuits + metadata naar disk (reproduceerbaar).
3. Bij `--submit <backend>`: submit sequentieel naar IBM Runtime SamplerV2,
   persist `job_id` per instantie, pol tot klaar, schrijf counts naar JSON.
4. Bij `--dry-run`: alleen token-check + backend-listing + transpile-estimate.
5. Bij `--resume <job_id>`: ophalen van eerder gesubmitte job.

CLI
---
  python b165b_hardware_submit.py --dry-run
  python b165b_hardware_submit.py --submit ibm_brisbane
  python b165b_hardware_submit.py --submit ibm_brisbane --only 3reg8
  python b165b_hardware_submit.py --resume <job_id>

Tokens en veiligheid
--------------------
* Token komt uit env-var `QISKIT_IBM_TOKEN` (of, als `--token-file <pad>` opgegeven,
  uit eerste regel van dat bestand).
* Claude/Cowork ziet de token NOOIT: zet hem buiten de project-folder of in je
  shell-profile, niet in een `.env` in deze repo.
* Zonder token + `--dry-run` is gewoon runbaar in CI / op Claude's kant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import SimpleGraph, random_3regular


# ============================================================
# Instance registry
# ============================================================

INSTANCE_SPECS = {
    "3reg8":   {"n": 8,  "loader": "random_3regular", "seed": 0},
    "myciel3": {"n": 11, "loader": "dimacs_fixture",  "name": "myciel3"},
}


def build_instance(name: str) -> SimpleGraph:
    """Reconstrueer een deterministische benchmarkgraaf uit de registry."""
    spec = INSTANCE_SPECS[name]
    if spec["loader"] == "random_3regular":
        return random_3regular(spec["n"], seed=spec["seed"])
    if spec["loader"] == "dimacs_fixture":
        from b154_dimacs_loader import load_fixture
        from b154_combined_leaderboard import to_simple
        g_w, _, _ = load_fixture(spec["name"])
        return to_simple(g_w)
    raise ValueError(f"Unknown loader for instance {name!r}")


# ============================================================
# QAOA-parameter presearch (Aer, noiseless, lokaal)
# ============================================================

def qaoa_grid_search(
    graph: SimpleGraph,
    p: int = 1,
    grid_size: int = 20,
    shots: int = 4096,
    seed: int = 42,
) -> tuple[list[float], list[float], float]:
    """Grid-search over (γ, β) ∈ (0, π) op Aer-noiseless voor QAOA-parameter.

    Geeft (γ*, β*, E*) met hoogste E[H_C]. Voor p=1; voor p>1 alleen γ_1,β_1.
    Dit draait LOKAAL — we willen goede angles **voor we hardware-queue-tijd gebruiken**.
    """
    assert p == 1, "Grid-search momenteel alleen voor p=1 (uitbreidbaar)."
    from b165_qiskit_runtime import run_aer, maxcut_value_from_counts
    from circuit_interface import Circuit

    edges = [(int(u), int(v), float(w)) for u, v, w in graph.edges]
    gammas = np.linspace(0.05, np.pi - 0.05, grid_size)
    betas = np.linspace(0.05, np.pi / 2 - 0.05, grid_size)

    best = (-np.inf, 0.0, 0.0)
    for g in gammas:
        for b in betas:
            qc = Circuit.qaoa_maxcut(graph.n, edges, p=1,
                                      gammas=[float(g)], betas=[float(b)])
            res = run_aer(qc, shots=shots, seed=seed)
            exp_val = maxcut_value_from_counts(res["counts"], graph)
            if exp_val > best[0]:
                best = (float(exp_val), float(g), float(b))
    return [best[1]], [best[2]], best[0]


# ============================================================
# Submission bundle & persistence
# ============================================================

@dataclass
class SubmissionBundle:
    """Alles wat we willen persisteren over één hardware-submit."""
    instance: str
    n_qubits: int
    p: int
    gammas: list[float]
    betas: list[float]
    expected_exp_value_aer: float
    shots: int
    backend_name: str = ""
    job_id: str = ""
    status: str = "PREPARED"   # PREPARED|SUBMITTED|COMPLETED|FAILED|SKIPPED_NO_TOKEN
    submit_unix: float = 0.0
    done_unix: float = 0.0
    queue_wait_s: float = 0.0
    execute_s: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    note: str = ""


def save_bundle(bundle: SubmissionBundle, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tag = bundle.job_id or f"prepared_{bundle.instance}"
    path = root / f"{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(bundle), f, indent=2, ensure_ascii=False)
    return path


def load_bundle(path: Path) -> SubmissionBundle:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SubmissionBundle(**data)


# ============================================================
# Token handling (env-var of losse file)
# ============================================================

def read_token(token_env: str = "QISKIT_IBM_TOKEN",
               token_file: str | None = None) -> str | None:
    """Haal IBM-token uit env-var of losse file. Nooit loggen."""
    if token_file:
        p = Path(token_file).expanduser()
        if p.exists():
            tok = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            return tok or None
    return os.environ.get(token_env) or None


# ============================================================
# Dry-run & submit
# ============================================================

def dry_run(token: str | None = None,
            instances: list[str] | None = None,
            shots: int = 4096,
            out_dir: Path | None = None) -> dict[str, Any]:
    """Geen hardware-call. Alleen: token-check, grid-search (klein), transpile-dep.

    Dit rapporteert voor de user wat er submit-wise klaarligt, zodat men vooraf
    weet wat er de wachtrij in gaat voor het echt wordt.
    """
    instances = instances or list(INSTANCE_SPECS.keys())
    report: dict[str, Any] = {
        "token_found": token is not None,
        "instances": {},
        "shots_per_instance": shots,
        "timestamp_unix": time.time(),
    }

    for inst in instances:
        g = build_instance(inst)
        # We doen een mini-grid-search (5×5) voor de dry-run om tijd te sparen.
        gammas, betas, exp_val = qaoa_grid_search(g, p=1, grid_size=5,
                                                  shots=min(shots, 2048))
        bundle = SubmissionBundle(
            instance=inst, n_qubits=g.n, p=1,
            gammas=gammas, betas=betas,
            expected_exp_value_aer=float(exp_val),
            shots=shots, status="PREPARED",
            note="dry-run pre-search (5x5 grid)",
        )
        if out_dir is not None:
            save_bundle(bundle, out_dir)
        report["instances"][inst] = {
            "n": g.n, "m": g.n_edges,
            "gamma": gammas[0], "beta": betas[0],
            "aer_expectation": float(exp_val),
            "prepared_path": str((out_dir / f"prepared_{inst}.json")
                                 if out_dir else "memory"),
        }

    if token is None:
        report["hint"] = ("Geen QISKIT_IBM_TOKEN gevonden — dit is OK voor een "
                          "dry-run; zet de env-var op je eigen laptop om echt "
                          "te submitten.")
    return report


def submit(backend_name: str,
           token: str,
           instances: list[str] | None = None,
           shots: int = 4096,
           out_dir: Path | None = None,
           grid_size: int = 20,
           instance_hub: str | None = None) -> list[SubmissionBundle]:
    """Echte IBM-submit — NIET door Claude uitvoeren.

    Per instantie:
      1. Grid-search → beste (γ, β) op Aer-noiseless
      2. Circuit.qaoa_maxcut → to_qiskit → transpile(backend)
      3. SamplerV2.run([qc]) → poll → counts
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile
    from b165_qiskit_runtime import (
        to_qiskit,
        add_measurements,
        maxcut_value_from_counts,
    )
    from circuit_interface import Circuit

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                    token=token, instance=instance_hub)
    backend = service.backend(backend_name)

    instances = instances or list(INSTANCE_SPECS.keys())
    bundles: list[SubmissionBundle] = []

    for inst in instances:
        g = build_instance(inst)
        gammas, betas, exp_val = qaoa_grid_search(g, p=1, grid_size=grid_size,
                                                  shots=shots)
        edges = [(int(u), int(v), float(w)) for u, v, w in g.edges]
        zqc = Circuit.qaoa_maxcut(g.n, edges, p=1, gammas=gammas, betas=betas)
        qc = to_qiskit(zqc)
        qc_meas = add_measurements(qc)
        qc_t = transpile(qc_meas, backend)

        bundle = SubmissionBundle(
            instance=inst, n_qubits=g.n, p=1,
            gammas=gammas, betas=betas,
            expected_exp_value_aer=float(exp_val),
            shots=shots,
            backend_name=backend_name,
        )

        sampler = SamplerV2(mode=backend)
        t_sub = time.time()
        job = sampler.run([qc_t], shots=shots)
        bundle.job_id = job.job_id()
        bundle.submit_unix = t_sub
        bundle.status = "SUBMITTED"

        if out_dir:
            save_bundle(bundle, out_dir)

        # Pol tot afgerond; voor free-tier kan dit uren duren.
        print(f"[B165b] Submitted {inst} -> job_id={bundle.job_id}, "
              f"backend={backend_name}. Wacht op resultaat...")

        result = job.result()
        t_done = time.time()
        pub = result[0]
        counts = dict(pub.data.c.get_counts())
        bundle.counts = counts
        bundle.status = "COMPLETED"
        bundle.done_unix = t_done
        bundle.execute_s = t_done - t_sub
        hw_exp = maxcut_value_from_counts(counts, g)
        bundle.note = f"hardware_expectation={hw_exp:.4f}"
        if out_dir:
            save_bundle(bundle, out_dir)
        bundles.append(bundle)

    return bundles


def resume(job_id: str,
           token: str,
           instance_hub: str | None = None,
           out_dir: Path | None = None,
           bundle_in: SubmissionBundle | None = None) -> SubmissionBundle:
    """Haal een eerder-gesubmitte job op, werk bundle bij."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                    token=token, instance=instance_hub)
    job = service.job(job_id)
    result = job.result()
    pub = result[0]
    counts = dict(pub.data.c.get_counts())

    if bundle_in is None:
        bundle_in = SubmissionBundle(instance="?", n_qubits=0, p=1,
                                      gammas=[], betas=[],
                                      expected_exp_value_aer=0.0, shots=0)
    bundle_in.job_id = job_id
    bundle_in.counts = counts
    bundle_in.status = "COMPLETED"
    bundle_in.done_unix = time.time()
    if out_dir:
        save_bundle(bundle_in, out_dir)
    return bundle_in


# ============================================================
# CLI
# ============================================================

def _default_out_dir() -> Path:
    here = Path(__file__).resolve().parent
    root = here.parent / "docs" / "paper" / "hardware" / "jobs"
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description="B165b: IBM Quantum hardware-submit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Geen hardware-call; alleen voorbereiding + grid-search.")
    ap.add_argument("--submit", type=str, default=None, metavar="BACKEND",
                    help="Submit naar IBM-backend (bv. ibm_brisbane).")
    ap.add_argument("--resume", type=str, default=None, metavar="JOB_ID",
                    help="Haal eerder-gesubmitte job op.")
    ap.add_argument("--only", type=str, default=None,
                    help="Beperk tot één instantie (3reg8 of myciel3).")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--grid-size", type=int, default=20)
    ap.add_argument("--token-env", default="QISKIT_IBM_TOKEN")
    ap.add_argument("--token-file", default=None,
                    help="Pad naar bestand met één regel token (alternatief "
                         "voor env-var).")
    ap.add_argument("--instance-hub", default=None,
                    help="Optioneel IBM hub/group/project (CRN/instance-string).")
    ap.add_argument("--out", default=None,
                    help="Output-folder voor bundles (default docs/paper/hardware/jobs).")
    args = ap.parse_args()

    out = Path(args.out) if args.out else _default_out_dir()
    instances = [args.only] if args.only else None

    if args.dry_run:
        token = read_token(args.token_env, args.token_file)
        rep = dry_run(token=token, instances=instances, shots=args.shots,
                       out_dir=out)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    if args.submit:
        token = read_token(args.token_env, args.token_file)
        if not token:
            print(f"[B165b] FOUT: geen token gevonden in ${args.token_env} "
                  f"en --token-file niet gezet. Submit afgebroken.")
            return 2
        bundles = submit(args.submit, token=token, instances=instances,
                          shots=args.shots, out_dir=out,
                          grid_size=args.grid_size,
                          instance_hub=args.instance_hub)
        for b in bundles:
            print(f"[B165b] {b.instance}: job_id={b.job_id}, status={b.status}")
        return 0

    if args.resume:
        token = read_token(args.token_env, args.token_file)
        if not token:
            print(f"[B165b] FOUT: --resume vereist token.")
            return 2
        b = resume(args.resume, token=token,
                    instance_hub=args.instance_hub, out_dir=out)
        print(f"[B165b] Resumed {args.resume}: status={b.status}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
