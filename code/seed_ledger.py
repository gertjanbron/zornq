#!/usr/bin/env python3
"""seed_ledger.py -- B55 follow-up: deterministisch seed-ledger voor ZornQ-runs.

Doel
----
Elke run van een benchmark-script gebruikt typisch meerdere seeds: een voor graaf-
generatie, een voor elke solver, een voor hyperplane-rounding, enzovoort. Zonder
centrale administratie is het onmogelijk om een run exact te reproduceren.

Dit ledger biedt:
  * `SeedLedger(master=...)` -- central registry; elke `derive(label)` geeft een
    deterministische child-seed uit de master via SHA256 (en onthoudt m).
  * `numpy_rng(label)` / `python_random(label)` -- convenience om direct een RNG
    te krijgen die al is gezaaid met de child-seed.
  * `to_dict()` / `save(path)` / `load(path)` -- JSON-sidecar-export voor audit.
  * Integratie met `audit_trail.AuditTrail` en `evidence_capsule` via `as_record()`.

Deterministische afleiding
--------------------------
Een child-seed wordt berekend als de eerste 8 bytes van
    SHA256(f"{master_hex}|{label}")
omgezet naar een 64-bits integer en daarna gemaskeerd naar 32 bits
(numpy's default seed-bereik). Zelfde master + label geeft altijd dezelfde
child-seed; verschillende labels geven statistisch onafhankelijke streams.

Gebruik
-------
    >>> from seed_ledger import SeedLedger
    >>> ledger = SeedLedger(master=42)
    >>> rng_graph = ledger.numpy_rng("graph_gen")
    >>> rng_gw    = ledger.numpy_rng("gw_rounding")
    >>> ledger.save("results/run_001.seeds.json")

Replay
------
    >>> from seed_ledger import SeedLedger
    >>> ledger = SeedLedger.load("results/run_001.seeds.json")
    >>> # zelfde master + labels -> bit-identieke streams

Integratie met AuditTrail
-------------------------
    >>> audit = AuditTrail(graph_desc="petersen", p=3, seed=ledger.master)
    >>> audit.extra_config = audit.extra_config or {}
    >>> audit.extra_config["seed_ledger"] = ledger.as_record()

Getest via `test_seed_ledger.py` (alle tests groen).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

__version__ = "1.0.0"
_SEED_BITS = 32  # numpy.random.default_rng accepteert 0..2**63-1, maar we
                 # houden het op 32 bits zodat ook python-random.seed() en
                 # scipy-solvers dezelfde waarde kunnen ontvangen zonder overflow.
_SEED_MASK = (1 << _SEED_BITS) - 1


# =====================================================================
# Helpers
# =====================================================================


def _master_to_hex(master: int | str) -> str:
    """Normaliseer master-seed naar deterministische hex-string."""
    if isinstance(master, str):
        if master.startswith("0x"):
            return hex(int(master, 16))
        return hex(int(hashlib.sha256(master.encode("utf-8")).hexdigest()[:16], 16))
    if isinstance(master, (np.integer,)):
        master = int(master)
    if not isinstance(master, int):
        raise TypeError(f"master must be int or str, got {type(master).__name__}")
    if master < 0:
        raise ValueError("master seed must be non-negative")
    return hex(master & 0xFFFFFFFFFFFFFFFF)


def _derive_child_seed(master_hex: str, label: str) -> int:
    """SHA256-based deterministic child-seed."""
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}")
    if not label:
        raise ValueError("label cannot be empty")
    payload = f"{master_hex}|{label}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Eerste 8 bytes als unsigned int
    child64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return child64 & _SEED_MASK


# =====================================================================
# SeedLedger
# =====================================================================


@dataclass
class SeedLedger:
    """Central registry for all random seeds used in a ZornQ run.

    Een master-seed bepaalt deterministisch alle child-seeds die via `derive()`
    worden opgehaald. Het ledger kan worden opgeslagen als JSON-sidecar en
    later worden geladen voor exacte replay.

    Attributes:
        master: master-seed (int >= 0). Als None, wordt `_random_master()` gebruikt.
        children: dict label -> child_seed (int). Read-only na derive; dubbele
            labels geven hetzelfde seed terug (consistent).
        metadata: vrije dict met context (commit-hash, machinenaam, etc.).
        created_at: ISO-8601-UTC-tijdstip van ledger-creatie.
    """

    master: int = 0
    children: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if isinstance(self.master, np.integer):
            self.master = int(self.master)
        if not isinstance(self.master, int):
            raise TypeError(
                f"master must be int, got {type(self.master).__name__}"
            )
        if self.master < 0:
            raise ValueError("master seed must be non-negative")
        self._master_hex = _master_to_hex(self.master)

    # ---------------- core API ----------------

    def derive(self, label: str) -> int:
        """Registreer een child-seed voor `label` en retourneer 'm.

        Deterministisch: zelfde master + label -> zelfde seed, ook over runs.
        Dubbele labels geven geen nieuwe seed; de eerder geregistreerde waarde
        wordt teruggegeven (idempotent).
        """
        if label in self.children:
            return self.children[label]
        child = _derive_child_seed(self._master_hex, label)
        self.children[label] = child
        return child

    def get(self, label: str) -> int | None:
        """Pure lookup: retourneer eerder-geregistreerde seed of None."""
        return self.children.get(label)

    def numpy_rng(self, label: str) -> np.random.Generator:
        """Retourneer een `numpy.random.Generator` gezaaid met de child-seed."""
        return np.random.default_rng(self.derive(label))

    def numpy_random_state(self, label: str) -> np.random.RandomState:
        """Retourneer een legacy `numpy.random.RandomState` voor scipy-compat."""
        return np.random.RandomState(self.derive(label))

    def python_random(self, label: str) -> random.Random:
        """Retourneer een stdlib `random.Random` gezaaid met de child-seed."""
        r = random.Random()
        r.seed(self.derive(label))
        return r

    def labels(self) -> list[str]:
        """Alle reeds-geregistreerde labels (stabiel gesorteerd)."""
        return sorted(self.children.keys())

    # ---------------- serialisatie ----------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict-representatie voor JSON-export."""
        return {
            "version": __version__,
            "kind": "zornq.seed_ledger",
            "master": int(self.master),
            "master_hex": self._master_hex,
            "children": dict(sorted(self.children.items())),
            "metadata": dict(sorted(self.metadata.items())),
            "created_at": self.created_at,
        }

    def as_record(self) -> dict[str, Any]:
        """Alias voor audit_trail / evidence_capsule integratie."""
        return self.to_dict()

    def save(self, path: str) -> str:
        """Schrijf JSON-sidecar. Maakt ouderdirectory aan als nodig."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)
            f.write("\n")
        return os.path.abspath(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedLedger":
        """Reconstrueer van canonical dict. Valideert dat kinderen matchen
        met wat opnieuw zou worden afgeleid (detecteert corruptie/tampering)."""
        kind = data.get("kind")
        if kind != "zornq.seed_ledger":
            raise ValueError(
                f"not a seed_ledger record (kind={kind!r})"
            )
        master = int(data["master"])
        children = {
            str(k): int(v) for k, v in dict(data.get("children", {})).items()
        }
        metadata = dict(data.get("metadata", {}))
        created_at = str(data.get("created_at", ""))

        ledger = cls(
            master=master,
            children={},  # leeg, we vullen hieronder en valideren
            metadata=metadata,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )

        # Valideer determinisme: elk opgeslagen child moet matchen met derive()
        mismatches = []
        for label, stored_seed in children.items():
            expected = _derive_child_seed(ledger._master_hex, label)
            if expected != stored_seed:
                mismatches.append((label, stored_seed, expected))
        if mismatches:
            details = "; ".join(
                f"{lbl!r}: stored={s} expected={e}"
                for lbl, s, e in mismatches
            )
            raise ValueError(
                f"seed-ledger integrity check failed: {details}"
            )

        ledger.children = children
        return ledger

    @classmethod
    def load(cls, path: str) -> "SeedLedger":
        """Laad ledger uit JSON-sidecar. Valideert determinisme."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ---------------- helpers voor audit-integratie ----------------

    def attach_to_audit(self, audit) -> None:
        """Plak de ledger als `seed_ledger`-veld in een AuditTrail.

        Werkt met het bestaande `audit_trail.AuditTrail` door het veld
        in `audit.data` te zetten zonder dat AuditTrail zelf SeedLedger hoeft
        te kennen. Roept geen AuditTrail-methoden aan, dus geen circulaire dep.
        """
        if not hasattr(audit, "data") or not isinstance(audit.data, dict):
            raise TypeError(
                "audit must expose a .data dict (AuditTrail-compatibel)"
            )
        audit.data["seed_ledger"] = self.to_dict()


# =====================================================================
# Convenience module-level singleton (optioneel gebruik)
# =====================================================================


_GLOBAL_LEDGER: SeedLedger | None = None


def get_global_ledger() -> SeedLedger | None:
    """Retourneer de globale ledger (of None als er geen is ingesteld)."""
    return _GLOBAL_LEDGER


def set_global_ledger(ledger: SeedLedger | None) -> None:
    """Zet de globale ledger (of wis met None)."""
    global _GLOBAL_LEDGER
    if ledger is not None and not isinstance(ledger, SeedLedger):
        raise TypeError("ledger must be a SeedLedger or None")
    _GLOBAL_LEDGER = ledger


def derive(label: str) -> int:
    """Shortcut die tegen de globale ledger werkt. ValueError als niet ingesteld."""
    if _GLOBAL_LEDGER is None:
        raise RuntimeError(
            "no global ledger set; call set_global_ledger(SeedLedger(master=...)) first"
        )
    return _GLOBAL_LEDGER.derive(label)


# =====================================================================
# CLI
# =====================================================================


def _cli_show(path: str) -> int:
    ledger = SeedLedger.load(path)
    data = ledger.to_dict()
    print(f"seed_ledger version:  {data['version']}")
    print(f"master:               {data['master']}")
    print(f"master_hex:           {data['master_hex']}")
    print(f"created_at:           {data['created_at']}")
    print(f"labels registered:    {len(data['children'])}")
    if data["children"]:
        print()
        print(f"  {'label':<40} {'child_seed':>12}")
        print(f"  {'-' * 40} {'-' * 12}")
        for label, seed in sorted(data["children"].items()):
            print(f"  {label:<40} {seed:>12}")
    if data["metadata"]:
        print()
        print("metadata:")
        for key, value in sorted(data["metadata"].items()):
            print(f"  {key}: {value}")
    return 0


def _cli_derive(master: int, labels: list[str]) -> int:
    ledger = SeedLedger(master=master)
    for label in labels:
        seed = ledger.derive(label)
        print(f"{label}: {seed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="ZornQ seed-ledger CLI (B55 follow-up)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="toon inhoud van een ledger-file")
    p_show.add_argument("path", help="pad naar JSON-ledger")

    p_derive = sub.add_parser("derive", help="leid seeds af voor labels")
    p_derive.add_argument("--master", type=int, required=True, help="master-seed (int)")
    p_derive.add_argument("labels", nargs="+", help="labels om af te leiden")

    args = parser.parse_args(argv)
    if args.cmd == "show":
        return _cli_show(args.path)
    if args.cmd == "derive":
        return _cli_derive(args.master, args.labels)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
