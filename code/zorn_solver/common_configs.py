from __future__ import annotations

from dataclasses import dataclass

from .graph_types import WeightedGraph


@dataclass(frozen=True)
class ResourceBudget:
    time_limit_sec: float
    max_ram_mb: int
    seed: int = 0


@dataclass(frozen=True)
class SimConfig:
    graph: WeightedGraph
    p: int
    mixer: str = "X"
    optimizer: str = "COBYLA"
    seed: int = 0
    maxiter: int = 64
    n_restarts: int = 3


@dataclass(frozen=True)
class SolveConfig:
    graph: WeightedGraph
    budget: ResourceBudget
    triage_threshold: float = 1e-8
    score_invalid_penalty: float = 10.0
    score_load_penalty: float = 0.1
    score_norm_penalty: float = 0.01
    recovery_scale: float = 0.25
    max_iterations: int = 32
