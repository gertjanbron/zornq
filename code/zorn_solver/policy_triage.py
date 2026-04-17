from __future__ import annotations

from dataclasses import dataclass

from .invariants import is_finite_state, norm_primal
from .zorn_core import ZornState


@dataclass(frozen=True)
class TriageResult:
    state: ZornState[float]
    lambda_valid: bool
    norm_primal: float
    reason: str


class TriagePolicy:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def apply(self, x: ZornState[float]) -> TriageResult:
        n = norm_primal(x)
        ok = is_finite_state(x) and abs(n) > self.eps
        if ok:
            return TriageResult(
                state=x,
                lambda_valid=True,
                norm_primal=n,
                reason="valid",
            )
        return TriageResult(
            state=ZornState.zero(),
            lambda_valid=False,
            norm_primal=n,
            reason="singular_or_nonfinite",
        )
