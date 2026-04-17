from __future__ import annotations

import math

from .zorn_core import ZornState


def norm_primal(x: ZornState[float]) -> float:
    return float(x.norm())


def is_finite_state(x: ZornState[float]) -> bool:
    vals = [x.a, x.b, x.u.x, x.u.y, x.u.z, x.v.x, x.v.y, x.v.z]
    return all(math.isfinite(float(v)) for v in vals)


def is_singular(x: ZornState[float], eps: float = 1e-8) -> bool:
    return abs(norm_primal(x)) <= eps
