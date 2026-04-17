from __future__ import annotations

from dataclasses import dataclass

from .zorn_core import Vec3, ZornState


@dataclass(frozen=True)
class Dual:
    real: float
    eps: float = 0.0

    def __add__(self, rhs: "Dual | float") -> "Dual":
        if isinstance(rhs, Dual):
            return Dual(self.real + rhs.real, self.eps + rhs.eps)
        return Dual(self.real + float(rhs), self.eps)

    __radd__ = __add__

    def __sub__(self, rhs: "Dual | float") -> "Dual":
        if isinstance(rhs, Dual):
            return Dual(self.real - rhs.real, self.eps - rhs.eps)
        return Dual(self.real - float(rhs), self.eps)

    def __rsub__(self, rhs: float) -> "Dual":
        return Dual(float(rhs) - self.real, -self.eps)

    def __mul__(self, rhs: "Dual | float") -> "Dual":
        if isinstance(rhs, Dual):
            return Dual(self.real * rhs.real, self.real * rhs.eps + self.eps * rhs.real)
        rhs = float(rhs)
        return Dual(self.real * rhs, self.eps * rhs)

    __rmul__ = __mul__

    def __truediv__(self, rhs: "Dual | float") -> "Dual":
        if isinstance(rhs, Dual):
            if rhs.real == 0.0:
                raise ZeroDivisionError
            return Dual(self.real / rhs.real, (self.eps * rhs.real - self.real * rhs.eps) / (rhs.real ** 2))
        rhs = float(rhs)
        if rhs == 0.0:
            raise ZeroDivisionError
        return Dual(self.real / rhs, self.eps / rhs)

    def __neg__(self) -> "Dual":
        return Dual(-self.real, -self.eps)

    def __float__(self) -> float:
        return self.real


def lift_state(x: ZornState[float], seed: str = "a") -> ZornState[Dual]:
    def d(val: float, hit: bool) -> Dual:
        return Dual(val, 1.0 if hit else 0.0)
    return ZornState(
        a=d(x.a, seed == "a"),
        b=d(x.b, seed == "b"),
        u=Vec3(d(x.u.x, seed == "ux"), d(x.u.y, seed == "uy"), d(x.u.z, seed == "uz")),
        v=Vec3(d(x.v.x, seed == "vx"), d(x.v.y, seed == "vy"), d(x.v.z, seed == "vz")),
    )


def dual_norm_eps(x: ZornState[Dual]) -> float:
    return x.norm().eps
