from __future__ import annotations

from typing import Protocol

from .results import SimResult, SolveResult


class SimulatorProtocol(Protocol):
    def evaluate(self, *args, **kwargs) -> SimResult: ...


class SolverProtocol(Protocol):
    def solve(self, *args, **kwargs) -> SolveResult: ...
