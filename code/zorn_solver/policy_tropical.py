from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Proposal:
    src_id: int
    dst_id: int
    score: float
    norm_abs: float
    lambda_valid: bool
    payload: Any


class TropicalPolicy:
    """
    Min-plus flavored selection layer.
    Lower score wins per target.
    """

    def __init__(
        self,
        invalid_penalty: float = 10.0,
        load_penalty: float = 0.1,
        norm_penalty: float = 0.01,
    ) -> None:
        self.invalid_penalty = float(invalid_penalty)
        self.load_penalty = float(load_penalty)
        self.norm_penalty = float(norm_penalty)

    def score(
        self,
        *,
        base_cost: float,
        lambda_valid: bool,
        load: float,
        norm_abs: float,
    ) -> float:
        penalty_invalid = 0.0 if lambda_valid else self.invalid_penalty
        return float(base_cost + penalty_invalid + self.load_penalty * load + self.norm_penalty * norm_abs)

    def choose_min_per_target(self, proposals: list[Proposal]) -> list[Proposal]:
        best: dict[int, Proposal] = {}
        for p in proposals:
            current = best.get(p.dst_id)
            if current is None or p.score < current.score:
                best[p.dst_id] = p
        return list(best.values())
