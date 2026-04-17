from __future__ import annotations

import time
from dataclasses import dataclass

from zornq_common.configs import SolveConfig
from zornq_common.profiling import PeakMemoryTracker
from zornq_common.results import RunMeta, SolveResult
from zornq_common.rng import make_rng

from .policy_triage import TriagePolicy
from .policy_tropical import Proposal, TropicalPolicy
from .zorn_core import Vec3, ZornState


@dataclass
class _NodeRuntime:
    state: ZornState[float]
    load: float = 0.0


class ZornSearchController:
    def __init__(self, config: SolveConfig) -> None:
        self.config = config
        self.graph = config.graph
        self.rng = make_rng(config.budget.seed)
        self.triage = TriagePolicy(eps=config.triage_threshold)
        self.tropical = TropicalPolicy(
            invalid_penalty=config.score_invalid_penalty,
            load_penalty=config.score_load_penalty,
            norm_penalty=config.score_norm_penalty,
        )

    def _random_state(self, node: int) -> ZornState[float]:
        deg = self.graph.weighted_degree(node)
        vec = self.rng.normal(0.0, self.config.recovery_scale, size=6)
        return ZornState(
            a=1.0 + deg,
            b=1.0,
            u=Vec3(float(vec[0]), float(vec[1]), float(vec[2])),
            v=Vec3(float(vec[3]), float(vec[4]), float(vec[5])),
        )

    def _init_runtime(self) -> list[_NodeRuntime]:
        return [_NodeRuntime(state=self._random_state(i), load=0.0) for i in range(self.graph.num_nodes)]

    def _decode_partition(self, runtime: list[_NodeRuntime]) -> list[int]:
        return [1 if node.state.a >= node.state.b else 0 for node in runtime]

    def _greedy_repair(self, partition: list[int], deadline: float) -> tuple[list[int], bool]:
        improved = False
        while time.perf_counter() < deadline:
            best_gain = 0.0
            best_node = None
            for node in range(self.graph.num_nodes):
                gain = self.graph.local_flip_gain(partition, node)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_node = node
            if best_node is None:
                break
            partition[best_node] ^= 1
            improved = True
        return partition, improved

    def _build_proposals(self, runtime: list[_NodeRuntime]) -> list[Proposal]:
        proposals: list[Proposal] = []
        for e in self.graph.edges:
            src_state = runtime[e.u].state
            dst_state = runtime[e.v].state

            tri_uv = self.triage.apply(src_state.multiply(dst_state))
            score_uv = self.tropical.score(
                base_cost=e.w,
                lambda_valid=tri_uv.lambda_valid,
                load=runtime[e.v].load,
                norm_abs=abs(tri_uv.norm_primal),
            )
            proposals.append(
                Proposal(
                    src_id=e.u,
                    dst_id=e.v,
                    score=score_uv,
                    norm_abs=abs(tri_uv.norm_primal),
                    lambda_valid=tri_uv.lambda_valid,
                    payload=tri_uv,
                )
            )

            tri_vu = self.triage.apply(dst_state.multiply(src_state))
            score_vu = self.tropical.score(
                base_cost=e.w,
                lambda_valid=tri_vu.lambda_valid,
                load=runtime[e.u].load,
                norm_abs=abs(tri_vu.norm_primal),
            )
            proposals.append(
                Proposal(
                    src_id=e.v,
                    dst_id=e.u,
                    score=score_vu,
                    norm_abs=abs(tri_vu.norm_primal),
                    lambda_valid=tri_vu.lambda_valid,
                    payload=tri_vu,
                )
            )
        return proposals

    def _commit(self, runtime: list[_NodeRuntime], proposals: list[Proposal]) -> None:
        winners = self.tropical.choose_min_per_target(proposals)
        touched = set()
        for winner in winners:
            touched.add(winner.dst_id)
            tri = winner.payload
            if tri.lambda_valid:
                runtime[winner.dst_id].state = tri.state
                runtime[winner.dst_id].load = max(0.0, runtime[winner.dst_id].load * 0.9 - 0.1)
            else:
                runtime[winner.dst_id].state = self._random_state(winner.dst_id)
                runtime[winner.dst_id].load += 1.0

        for idx, node in enumerate(runtime):
            if idx not in touched:
                node.load = max(0.0, node.load * 0.95)

    def run(self) -> SolveResult:
        deadline = time.perf_counter() + self.config.budget.time_limit_sec
        runtime = self._init_runtime()
        converged = False
        iteration = 0

        with PeakMemoryTracker() as tracker:
            for iteration in range(1, self.config.max_iterations + 1):
                if time.perf_counter() >= deadline:
                    break
                proposals = self._build_proposals(runtime)
                self._commit(runtime, proposals)

            partition = self._decode_partition(runtime)
            partition, improved = self._greedy_repair(partition, deadline)
            converged = not improved
            best_cut = self.graph.cut_value(partition)

        return SolveResult(
            best_cut=float(best_cut),
            best_partition=tuple(partition),
            iterations=iteration,
            converged=converged,
            meta=RunMeta(
                seed=self.config.budget.seed,
                wall_clock_time=tracker.elapsed_sec,
                peak_ram_mb=tracker.peak_mb,
                notes={
                    "backend": "zorn_heuristic",
                    "max_iterations": self.config.max_iterations,
                    "triage_threshold": self.config.triage_threshold,
                },
            ),
        )
