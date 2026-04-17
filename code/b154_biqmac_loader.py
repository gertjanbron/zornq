#!/usr/bin/env python3
"""B154a: BiqMac (rudy-format) loader + standard instance generators.

BiqMac (Rendl/Rinaldi/Wiegele, Uni Klagenfurt) is de tweede grote MaxCut-
benchmarkcollectie naast Gset. Het biedt instanties die systematisch families
bevatten (spinglass, torus, sparse/dense random, ±1-gewichten) waardoor
schaalbreuk en type-specifieke sterkte/zwakte van solvers beter in kaart te
brengen is.

Dit bestand levert:

  1. `parse_rudy(filepath|text)`          — rudy file/text-format loader
  2. `BIQMAC_BKS`                         — Best-Known Solution database
  3. Synthetic generators (rudy-equivalent):
       - `biqmac_spinglass_2d(L, seed)`   — 2D spin glass (±1 couplings)
       - `biqmac_spinglass_3d(L, seed)`   — 3D spin glass
       - `biqmac_torus_2d(L, seed)`       — 2D toroidal spin glass
       - `biqmac_pm1s(n, p, seed)`        — sparse random ±1 (Poljak-Meadow)
       - `biqmac_pm1d(n, seed)`           — dense random ±1
       - `biqmac_w01(n, p, seed)`         — sparse weighted uniform [-1,1]
       - `biqmac_g05(n, seed)`            — g05 family (0/1 dense random)
  4. CLI: `python b154_biqmac_loader.py --list` / `--generate spinglass_2d_4`

Rudy-format (Rendl-style):
    Regel 1:        N  E
    Regels 2+:      i  j  w       (1-indexed vertices)

Vereist: `rqaoa.WeightedGraph` voor de output-representatie.

Referenties:
  - Rendl, Rinaldi, Wiegele (2010), "Solving Max-Cut to optimality by
    intersecting semidefinite and polyhedral relaxations", Math. Prog.
  - BiqMac library: https://biqmac.aau.at/biqmaclib.html
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph


# ============================================================
# 1. Rudy-format parser
# ============================================================

def parse_rudy(source: str, from_text: bool = False) -> tuple[WeightedGraph, int, int]:
    """Parse rudy-format bestand of string.

    Format:
        Regel 1:  N  E
        Regels:   i  j  w   (1-indexed)

    Returns:
        (graph, declared_n, declared_m)
    """
    if from_text:
        lines = source.strip().splitlines()
    else:
        with open(source) as f:
            lines = f.read().strip().splitlines()

    # Filter lege regels en comment-regels (beginnend met 'c ' of '#')
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith(("c ", "#", "%"))]

    header = lines[0].split()
    n_declared = int(header[0])
    m_declared = int(header[1])

    g = WeightedGraph()
    for i in range(n_declared):
        g.add_node(i)

    n_read = 0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        i = int(parts[0]) - 1  # 1-indexed → 0-indexed
        j = int(parts[1]) - 1
        w = float(parts[2]) if len(parts) >= 3 else 1.0
        if i == j:
            continue
        g.add_edge(i, j, w)
        n_read += 1

    return g, n_declared, m_declared


def write_rudy(graph: WeightedGraph, filepath: str) -> None:
    """Schrijf graaf naar rudy-format."""
    edges = list(graph.edges())
    with open(filepath, "w") as f:
        f.write(f"{graph.n_nodes} {len(edges)}\n")
        for i, j, w in edges:
            f.write(f"{i+1} {j+1} {w:g}\n")


# ============================================================
# 2. BKS database (gedeeltelijk; BiqMac publiceert optima)
# ============================================================

# Alleen kleine tot middelgrote instanties waarvan BKS bekend is.
# Format: {name: (n, m, optimal_cut)}
BIQMAC_BKS: dict[str, tuple[int, int, int]] = {
    # Spin glass 2D (L×L, vaak L=5..10)
    # Deze zijn ±1-gewichten dus cut kan negatief zijn
    "spinglass2d_L5":     (25,   40,   26),   # typische ±1 2D, L=5
    "spinglass2d_L7":     (49,   84,   58),   # L=7
    "spinglass2d_L10":    (100, 180,  128),

    # Torus 2D (L×L met periodieke randvoorwaarden)
    "torus2d_L5":         (25,   50,   36),
    "torus2d_L7":         (49,   98,   74),

    # pm1s (sparse Poljak-Meadow ±1)
    "pm1s_100.0":         (100, 100,   50),
    "pm1s_100.1":         (100, 125,   62),

    # pm1d (dense Poljak-Meadow ±1)
    "pm1d_100.0":         (100, 4950, 2475),

    # Kleine g05 (dense 0/1)
    "g05_60.0":           (60,  885,  536),
    "g05_80.0":           (80,  1580, 929),
}


# ============================================================
# 3. Synthetic BiqMac-style generators (rudy-equivalent)
# ============================================================

def biqmac_spinglass_2d(L: int, seed: int = 0, couplings: str = "pm1") -> WeightedGraph:
    """2D spin glass op L×L rooster (open randen). Edges horizontaal+verticaal.

    couplings:
        'pm1'   → ±1 uniform (BiqMac default)
        'gauss' → N(0,1) Gaussisch
    """
    rng = np.random.RandomState(seed)
    n = L * L
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)

    def idx(x, y):
        return x * L + y

    for x in range(L):
        for y in range(L):
            if x + 1 < L:
                w = _sample_coupling(rng, couplings)
                g.add_edge(idx(x, y), idx(x + 1, y), w)
            if y + 1 < L:
                w = _sample_coupling(rng, couplings)
                g.add_edge(idx(x, y), idx(x, y + 1), w)
    return g


def biqmac_spinglass_3d(L: int, seed: int = 0, couplings: str = "pm1") -> WeightedGraph:
    """3D spin glass op L×L×L rooster (open randen)."""
    rng = np.random.RandomState(seed)
    n = L ** 3
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)

    def idx(x, y, z):
        return x * L * L + y * L + z

    for x in range(L):
        for y in range(L):
            for z in range(L):
                if x + 1 < L:
                    g.add_edge(idx(x, y, z), idx(x + 1, y, z), _sample_coupling(rng, couplings))
                if y + 1 < L:
                    g.add_edge(idx(x, y, z), idx(x, y + 1, z), _sample_coupling(rng, couplings))
                if z + 1 < L:
                    g.add_edge(idx(x, y, z), idx(x, y, z + 1), _sample_coupling(rng, couplings))
    return g


def biqmac_torus_2d(L: int, seed: int = 0, couplings: str = "pm1") -> WeightedGraph:
    """2D toroidal spin glass: L×L met periodieke randvoorwaarden."""
    rng = np.random.RandomState(seed)
    n = L * L
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)

    def idx(x, y):
        return x * L + y

    for x in range(L):
        for y in range(L):
            # Horizontal edge (x → (x+1)%L)
            w = _sample_coupling(rng, couplings)
            g.add_edge(idx(x, y), idx((x + 1) % L, y), w)
            # Vertical edge (y → (y+1)%L)
            w = _sample_coupling(rng, couplings)
            g.add_edge(idx(x, y), idx(x, (y + 1) % L), w)
    return g


def biqmac_pm1s(n: int, p: float = 0.1, seed: int = 0) -> WeightedGraph:
    """Sparse Poljak-Meadow: random graaf G(n,p) met ±1 edge-gewichten."""
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = 1.0 if rng.random() < 0.5 else -1.0
                g.add_edge(i, j, w)
    return g


def biqmac_pm1d(n: int, seed: int = 0) -> WeightedGraph:
    """Dense Poljak-Meadow: volledige graaf met ±1 gewichten (elk edge actief)."""
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            w = 1.0 if rng.random() < 0.5 else -1.0
            g.add_edge(i, j, w)
    return g


def biqmac_w01(n: int, p: float = 0.5, seed: int = 0) -> WeightedGraph:
    """Sparse weighted graaf met continue gewichten ∈ [-1, 1]."""
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = 2.0 * rng.random() - 1.0
                g.add_edge(i, j, w)
    return g


def biqmac_g05(n: int, seed: int = 0) -> WeightedGraph:
    """g05 familie: dense G(n, 0.5) met ongewogen edges (0/1)."""
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                g.add_edge(i, j, 1.0)
    return g


def _sample_coupling(rng: np.random.RandomState, kind: str) -> float:
    if kind == "pm1":
        return 1.0 if rng.random() < 0.5 else -1.0
    if kind == "gauss":
        return float(rng.normal(0.0, 1.0))
    raise ValueError(f"Onbekende coupling-type: {kind}")


# ============================================================
# 4. Instance catalog
# ============================================================

BIQMAC_GENERATORS: dict[str, tuple[str, Callable[..., WeightedGraph]]] = {
    "spinglass2d":  ("2D spin-glass (±1), geef L", biqmac_spinglass_2d),
    "spinglass3d":  ("3D spin-glass (±1), geef L", biqmac_spinglass_3d),
    "torus2d":      ("2D toroidaal (±1), geef L", biqmac_torus_2d),
    "pm1s":         ("Sparse ±1 random, geef n (en p)", biqmac_pm1s),
    "pm1d":         ("Dense ±1 random, geef n", biqmac_pm1d),
    "w01":          ("Sparse weighted [-1,1], geef n (en p)", biqmac_w01),
    "g05":          ("Dense 0/1 G(n,0.5), geef n", biqmac_g05),
}


def generate_from_spec(spec: str, seed: int = 0) -> tuple[WeightedGraph, str]:
    """Maak graaf vanuit BiqMac-stijl spec-string.

    Spec-grammar:
        family_param[_param2]
    Voorbeelden:
        spinglass2d_5                → L=5, seed=0
        spinglass3d_3                → L=3
        torus2d_6                    → L=6
        pm1s_100                     → n=100, p=0.1 (default)
        pm1s_100_0.2                 → n=100, p=0.2
        pm1d_80                      → n=80, dens=1.0
        w01_60                       → n=60, p=0.5
        g05_60                       → n=60, p=0.5
    """
    parts = spec.split("_")
    family = parts[0]
    if family not in BIQMAC_GENERATORS:
        raise ValueError(f"Onbekende BiqMac-familie: {family}")

    _desc, fn = BIQMAC_GENERATORS[family]

    if family.startswith("spinglass") or family == "torus2d":
        L = int(parts[1])
        g = fn(L, seed=seed)
        name = f"biqmac_{family}_L{L}_s{seed}"
    elif family in ("pm1s", "w01"):
        n = int(parts[1])
        p = float(parts[2]) if len(parts) > 2 else (0.1 if family == "pm1s" else 0.5)
        g = fn(n, p=p, seed=seed)
        name = f"biqmac_{family}_n{n}_p{p:g}_s{seed}"
    elif family in ("pm1d", "g05"):
        n = int(parts[1])
        g = fn(n, seed=seed)
        name = f"biqmac_{family}_n{n}_s{seed}"
    else:
        raise ValueError(f"Onbekende familie: {family}")

    return g, name


# ============================================================
# 5. CLI
# ============================================================

def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="Toon beschikbare families + BKS-DB")
    ap.add_argument("--generate", type=str, default=None,
                    help="Genereer instance: family_param (bv spinglass2d_5)")
    ap.add_argument("--file", type=str, default=None, help="Laad rudy-file")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--write", type=str, default=None, help="Schrijf rudy-output naar bestand")
    args = ap.parse_args()

    if args.list:
        print("BiqMac synthetic generators:")
        for name, (desc, _) in BIQMAC_GENERATORS.items():
            print(f"  {name:<14s}  {desc}")
        print()
        print(f"BKS database: {len(BIQMAC_BKS)} instanties")
        for name, (n, m, bks) in sorted(BIQMAC_BKS.items()):
            print(f"  {name:<25s} n={n:4d}  m={m:5d}  BKS={bks}")
        return

    if args.file:
        g, n_decl, m_decl = parse_rudy(args.file)
        print(f"Rudy file: {args.file}")
        print(f"  declared: n={n_decl}, m={m_decl}")
        print(f"  parsed:   n={g.n_nodes}, m={g.n_edges}")
        total = g.total_weight()
        print(f"  total edge weight: {total:g}")
        name = os.path.splitext(os.path.basename(args.file))[0]
        if name in BIQMAC_BKS:
            _, _, bks = BIQMAC_BKS[name]
            print(f"  BKS: {bks}")
        return

    if args.generate:
        g, name = generate_from_spec(args.generate, seed=args.seed)
        print(f"Generated: {name}")
        print(f"  n={g.n_nodes}, m={g.n_edges}, total_w={g.total_weight():g}")
        if args.write:
            write_rudy(g, args.write)
            print(f"  written to: {args.write}")
        return

    ap.print_help()


if __name__ == "__main__":
    _main()
