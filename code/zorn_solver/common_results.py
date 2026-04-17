from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunMeta:
    seed: int
    wall_clock_time: float
    peak_ram_mb: float
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimResult:
    energy_expectation: float
    best_sampled_cut: float | None
    most_likely_bitstring: tuple[int, ...] | None
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    meta: RunMeta


@dataclass(frozen=True)
class SolveResult:
    best_cut: float
    best_partition: tuple[int, ...]
    iterations: int
    converged: bool
    meta: RunMeta
