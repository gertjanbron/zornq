from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Vec3(Generic[T]):
    x: T
    y: T
    z: T

    def dot(self, rhs: "Vec3[T]") -> T:
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z  # type: ignore[operator]

    def cross(self, rhs: "Vec3[T]") -> "Vec3[T]":
        return Vec3(
            self.y * rhs.z - self.z * rhs.y,  # type: ignore[operator]
            self.z * rhs.x - self.x * rhs.z,  # type: ignore[operator]
            self.x * rhs.y - self.y * rhs.x,  # type: ignore[operator]
        )

    def scale(self, scalar: T) -> "Vec3[T]":
        return Vec3(
            self.x * scalar,  # type: ignore[operator]
            self.y * scalar,  # type: ignore[operator]
            self.z * scalar,  # type: ignore[operator]
        )

    def add(self, rhs: "Vec3[T]") -> "Vec3[T]":
        return Vec3(
            self.x + rhs.x,  # type: ignore[operator]
            self.y + rhs.y,  # type: ignore[operator]
            self.z + rhs.z,  # type: ignore[operator]
        )

    def sub(self, rhs: "Vec3[T]") -> "Vec3[T]":
        return Vec3(
            self.x - rhs.x,  # type: ignore[operator]
            self.y - rhs.y,  # type: ignore[operator]
            self.z - rhs.z,  # type: ignore[operator]
        )

    @classmethod
    def zero(cls) -> "Vec3[float]":
        return cls(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ZornState(Generic[T]):
    a: T
    b: T
    u: Vec3[T]
    v: Vec3[T]

    def norm(self) -> T:
        return self.a * self.b - self.u.dot(self.v)  # type: ignore[operator]

    def multiply(self, rhs: "ZornState[T]") -> "ZornState[T]":
        out_a = self.a * rhs.a + self.u.dot(rhs.v)  # type: ignore[operator]
        out_b = self.b * rhs.b + self.v.dot(rhs.u)  # type: ignore[operator]
        out_u = rhs.u.scale(self.a).add(self.u.scale(rhs.b)).sub(self.v.cross(rhs.v))
        out_v = self.v.scale(rhs.a).add(rhs.v.scale(self.b)).add(self.u.cross(rhs.u))
        return ZornState(a=out_a, b=out_b, u=out_u, v=out_v)

    @classmethod
    def identity(cls) -> "ZornState[float]":
        return cls(1.0, 1.0, Vec3.zero(), Vec3.zero())

    @classmethod
    def zero(cls) -> "ZornState[float]":
        return cls(0.0, 0.0, Vec3.zero(), Vec3.zero())
