from __future__ import annotations

from zornq_common.configs import ResourceBudget, SolveConfig
from zornq_common.graph_types import WeightedGraph
from zornq_common.results import SolveResult

from .controller import ZornSearchController


class ZornHeuristicSolver:
    def __init__(
        self,
        graph: WeightedGraph,
        *,
        time_limit_sec: float = 1.0,
        max_ram_mb: int = 512,
        seed: int = 0,
        triage_threshold: float = 1e-8,
        max_iterations: int = 32,
        score_invalid_penalty: float = 10.0,
        score_load_penalty: float = 0.1,
        score_norm_penalty: float = 0.01,
        recovery_scale: float = 0.25,
    ) -> None:
        self.config = SolveConfig(
            graph=graph,
            budget=ResourceBudget(
                time_limit_sec=time_limit_sec,
                max_ram_mb=max_ram_mb,
                seed=seed,
            ),
            triage_threshold=triage_threshold,
            score_invalid_penalty=score_invalid_penalty,
            score_load_penalty=score_load_penalty,
            score_norm_penalty=score_norm_penalty,
            recovery_scale=recovery_scale,
            max_iterations=max_iterations,
        )
        self.controller = ZornSearchController(self.config)

    def solve(self) -> SolveResult:
        return self.controller.run()
