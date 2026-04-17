from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Edge:
    u: int
    v: int
    w: float = 1.0


@dataclass(frozen=True)
class WeightedGraph:
    num_nodes: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        for e in self.edges:
            if not (0 <= e.u < self.num_nodes and 0 <= e.v < self.num_nodes):
                raise ValueError(f"edge {e} is out of range")
            if e.u == e.v:
                raise ValueError("self-loops are not supported in this scaffold")

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: Iterable[tuple[int, int, float] | tuple[int, int]],
    ) -> "WeightedGraph":
        normalized: list[Edge] = []
        for item in edges:
            if len(item) == 2:
                u, v = item
                w = 1.0
            else:
                u, v, w = item
            normalized.append(Edge(int(u), int(v), float(w)))
        return cls(num_nodes=num_nodes, edges=tuple(normalized))

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def neighbors(self, node: int) -> list[int]:
        out: list[int] = []
        for e in self.edges:
            if e.u == node:
                out.append(e.v)
            elif e.v == node:
                out.append(e.u)
        return out

    def weighted_degree(self, node: int) -> float:
        d = 0.0
        for e in self.edges:
            if e.u == node or e.v == node:
                d += e.w
        return d

    def adjacency(self) -> list[list[tuple[int, float]]]:
        adj = [[] for _ in range(self.num_nodes)]
        for e in self.edges:
            adj[e.u].append((e.v, e.w))
            adj[e.v].append((e.u, e.w))
        return adj

    def cut_value(self, partition: list[int] | tuple[int, ...]) -> float:
        if len(partition) != self.num_nodes:
            raise ValueError("partition length must equal num_nodes")
        total = 0.0
        for e in self.edges:
            if partition[e.u] != partition[e.v]:
                total += e.w
        return total

    def local_flip_gain(self, partition: list[int], node: int) -> float:
        current_side = partition[node]
        gain = 0.0
        for nbr, weight in self.adjacency()[node]:
            if partition[nbr] == current_side:
                gain += weight
            else:
                gain -= weight
        return gain
