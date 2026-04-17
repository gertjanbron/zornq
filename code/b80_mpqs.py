#!/usr/bin/env python3
"""B80: MPQS — Message-Passing Quantum Solver voor MaxCut.

Twee paden:
  1. `mpqs_classical_bp`   : max-product belief propagation in log-domein.
                             Exact op bomen, consistente heuristiek op loopy
                             grafen. Output: cut-bitstring + convergence info.
  2. `mpqs_lightcone`      : per-vertex BFS-lightcone + lokale p-laags QAOA
                             (pure-numpy statevector) + ζ-matrix spectral
                             rounding + greedy 1-flip refine.

Beide solvers zijn zelfvoorzienend (geen qiskit-afhankelijkheid; de QAOA wordt
met numpy-statevector gedraaid — prima tot |L| ≈ 16 qubits per lightcone).

De MPQS-filosofie: vervang de globale MaxCut-inferentie door een set lokale
quantum-berekeningen die gecommuniceerd worden via edge-beliefs ⟨Z_u Z_v⟩,
en round globaal via een signed-graph-spectral-methode.

Gebruik:
    python b80_mpqs.py --n 12 --reg 3                # random 3-reg n=12
    python b80_mpqs.py --petersen                    # Petersen graaf
    python b80_mpqs.py --n 8 --reg 3 --radius 3 --p 2
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import numpy as np

from b60_gw_bound import SimpleGraph, random_3regular, brute_force_maxcut


# ============================================================
# 1. Max-product BP in log-domein (tree-exact)
# ============================================================

def mpqs_classical_bp(
    graph: SimpleGraph,
    max_iters: int = 200,
    damping: float = 0.3,
    tol: float = 1e-7,
    seed: int = 42,
    pin_vertex: int = 0,
    pin_value: int = 0,
    verbose: bool = False,
) -> dict:
    """Max-product BP voor MaxCut.

    Factor langs edge (u,v):  f(s_u,s_v) = w_uv · [s_u ≠ s_v]
    (spin s_u ∈ {0,1}; hoger = beter).

    Messages m_{u→v}(s_v) in R^2, log-domein. Om de triviale Z_2-fixed point
    (messages = 0) te breken, pinnen we s_{pin_vertex} = pin_value via een
    groot extern veld op die vertex. Daarnaast initialiseren we de
    messages met kleine random ruis (seed-gedetermineerd).

    Update:
        incoming(s_u) = h_u(s_u) + Σ_{w∈N(u)\v} m_{w→u}(s_u)
        m_{u→v}(s_v) = max_{s_u} [ w_uv·[s_u≠s_v] + incoming(s_u) ]
    sweep → damping + normalisatie (trek mean af → stabiliteit).

    Beliefs: b(s_u) = h_u(s_u) + Σ_{w∈N(u)} m_{w→u}(s_u)
    Rounding: s_u = argmax b(s_u).

    Op bomen exact (convergentie in O(diameter) iteraties).
    """
    n = graph.n
    rng = np.random.RandomState(seed)

    # Voor elke knoop: buren in geordende lijst
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in graph.edges:
        neighbors[u].append((v, float(w)))
        neighbors[v].append((u, float(w)))

    # Externe velden: groot veld op pin_vertex, klein random op rest (breekt
    # Z_2 symmetrie en zorgt voor niet-triviaal BP-fixed point).
    field_scale = max(1.0, 0.5 * max((w for _, _, w in graph.edges), default=1.0))
    h = rng.normal(0.0, 1e-3 * field_scale, size=(n, 2))
    PIN_LARGE = 1e3 * field_scale
    h[pin_vertex, :] = 0.0
    h[pin_vertex, 1 - pin_value] = -PIN_LARGE  # verbied andere waarde

    # Messages met kleine random init
    messages: dict[tuple[int, int], np.ndarray] = {}
    for u, v, _ in graph.edges:
        messages[(u, v)] = rng.normal(0.0, 1e-3 * field_scale, size=2)
        messages[(v, u)] = rng.normal(0.0, 1e-3 * field_scale, size=2)

    converged = False
    n_iters = 0
    last_delta = float("inf")

    for it in range(max_iters):
        new_messages: dict[tuple[int, int], np.ndarray] = {}
        max_delta = 0.0

        for u in range(n):
            # Som van alle inkomende messages naar u + extern veld
            full_incoming = h[u].copy()
            for x, _w_ux in neighbors[u]:
                full_incoming += messages[(x, u)]

            for v, w_uv in neighbors[u]:
                # Incoming zonder v-bijdrage
                incoming = full_incoming - messages[(v, u)]

                # m_{u→v}(s_v) = max_{s_u} [ w_uv·[s_u≠s_v] + incoming(s_u) ]
                # Vectorized:
                # s_v=0: max over s_u van (w_uv·s_u + incoming[s_u])
                # s_v=1: max over s_u van (w_uv·(1-s_u) + incoming[s_u])
                opt_sv0 = max(incoming[0], w_uv + incoming[1])
                opt_sv1 = max(w_uv + incoming[0], incoming[1])
                new_m = np.array([opt_sv0, opt_sv1])

                # Normaliseer: trek mean af voor stabiliteit
                new_m -= new_m.mean()

                # Damping
                old_m = messages[(u, v)]
                damped = damping * old_m + (1.0 - damping) * new_m
                new_messages[(u, v)] = damped

                delta = float(np.max(np.abs(damped - old_m)))
                if delta > max_delta:
                    max_delta = delta

        messages = new_messages
        n_iters = it + 1
        last_delta = max_delta
        if verbose and (it % 20 == 0 or max_delta < tol):
            print(f"  BP iter {it:3d}: max |Δm| = {max_delta:.3e}")
        if max_delta < tol:
            converged = True
            break

    # Beliefs b(s_u) = h_u(s_u) + Σ_{w∈N(u)} m_{w→u}(s_u)
    beliefs = h.copy()
    for u in range(n):
        for w_n, _ in neighbors[u]:
            beliefs[u] += messages[(w_n, u)]

    # Rounding met harde pin (pin_vertex → pin_value)
    bits = ["0"] * n
    for u in range(n):
        if u == pin_vertex:
            bits[u] = str(pin_value)
        else:
            bits[u] = "1" if beliefs[u, 1] > beliefs[u, 0] else "0"
    bit_str = "".join(bits)
    cut_bp_raw = graph.cut_value(bit_str)

    # Greedy 1-flip refine
    refined_bits, cut_value = _greedy_1flip(graph, bit_str)

    return {
        "cut_bits": refined_bits,
        "cut_value": cut_value,
        "bp_cut_raw": cut_bp_raw,
        "converged": converged,
        "n_iters": n_iters,
        "last_delta": last_delta,
        "beliefs": beliefs,
    }


# ============================================================
# 2. Pure-numpy statevector QAOA
# ============================================================

def _qaoa_statevector(
    n_qubits: int,
    edges: Sequence[tuple[int, int, float]],
    gammas: Sequence[float],
    betas: Sequence[float],
) -> np.ndarray:
    """p-laag QAOA op n_qubits qubits. Hamiltonian:
        H_C = Σ (w_uv / 2) · (I − Z_u Z_v)    (MaxCut-objectief)
        H_B = Σ X_k

    Start in |+⟩^{⊗n}. Pas alternerend e^{-i γ H_C} en e^{-i β H_B} toe.

    Implementatie: computational basis, diagonaal H_C op indices, X-mixer via
    qubit-wise moveaxis rotaties (Rx(2β) per qubit).
    """
    assert len(gammas) == len(betas), "gammas en betas moeten even lang zijn"
    p = len(gammas)
    dim = 2 ** n_qubits

    # Bit-decompositie voor iedere basistoestand
    idx = np.arange(dim)
    bits = ((idx[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(np.int8)
    # spin_q = (1 − 2 bit_q) ∈ {+1,−1}
    spin = 1 - 2 * bits  # (dim, n_qubits)

    # H_C diagonaal: h_c[idx] = Σ (w/2)·(1 − s_u s_v)
    hc = np.zeros(dim, dtype=np.float64)
    for u, v, w in edges:
        hc += (float(w) / 2.0) * (1.0 - spin[:, u].astype(np.float64) * spin[:, v].astype(np.float64))

    # Start: |+⟩^{⊗n}
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    for gamma, beta in zip(gammas, betas):
        # e^{-i γ H_C}: element-wise fasefactor
        psi = psi * np.exp(-1j * float(gamma) * hc)

        # e^{-i β Σ X_k} = ⊗_k Rx(2β) = ⊗_k [[cosβ, −i sinβ],[−i sinβ, cosβ]]
        c = float(np.cos(beta))
        s_ = float(np.sin(beta))
        psi_tensor = psi.reshape([2] * n_qubits)
        for q in range(n_qubits):
            # Verplaats qubit q naar as 0
            t = np.moveaxis(psi_tensor, q, 0)
            shape_rest = t.shape[1:]
            t2 = t.reshape(2, -1)
            p0 = t2[0].copy()
            p1 = t2[1].copy()
            t2[0] = c * p0 - 1j * s_ * p1
            t2[1] = -1j * s_ * p0 + c * p1
            t = t2.reshape([2] + list(shape_rest))
            psi_tensor = np.moveaxis(t, 0, q)
        psi = psi_tensor.reshape(dim)

    return psi


def _expectation_zz(psi: np.ndarray, n_qubits: int, u: int, v: int) -> float:
    """⟨ψ|Z_u Z_v|ψ⟩ via basistoestand-diagonaal."""
    dim = len(psi)
    idx = np.arange(dim)
    bit_u = (idx >> u) & 1
    bit_v = (idx >> v) & 1
    # (−1)^{bit_u + bit_v} = +1 als gelijk, −1 als verschillend
    sign = np.where(bit_u == bit_v, 1.0, -1.0)
    probs = np.abs(psi) ** 2
    return float(np.sum(sign * probs))


# ============================================================
# 3. Lightcone-constructie
# ============================================================

def _build_lightcone(
    graph: SimpleGraph,
    center: int,
    radius: int,
) -> tuple[list[int], list[tuple[int, int, float]], dict[int, int]]:
    """BFS(center, radius). Return:
        - vertices: gesorteerde lijst van globale vertex-id's in lightcone
        - local_edges: (lu, lv, w) in lokale (her-indexeerde) id's
        - mapping: global_id → local_id
    """
    n = graph.n
    dist = [-1] * n
    dist[center] = 0
    frontier = [center]
    while frontier:
        new_frontier: list[int] = []
        for u in frontier:
            if dist[u] >= radius:
                continue
            for v, _ in graph.adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    new_frontier.append(v)
        frontier = new_frontier

    vertices = sorted(i for i in range(n) if dist[i] >= 0 and dist[i] <= radius)
    mapping = {g: l for l, g in enumerate(vertices)}
    vset = set(vertices)

    local_edges: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for u, v, w in graph.edges:
        if u in vset and v in vset:
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            local_edges.append((mapping[u], mapping[v], float(w)))

    return vertices, local_edges, mapping


# ============================================================
# 4. Greedy 1-flip refinement
# ============================================================

def _greedy_1flip(
    graph: SimpleGraph,
    bits: str,
    max_rounds: int = 100,
) -> tuple[str, float]:
    """Lokale zoekmethode: flip elke vertex die de cut verbetert.

    Eindigt als één volledige ronde geen flip oplevert.
    """
    n = graph.n
    current = list(bits)
    best_val = graph.cut_value("".join(current))

    for _ in range(max_rounds):
        improved = False
        for u in range(n):
            # Wissel bit u
            current[u] = "1" if current[u] == "0" else "0"
            new_val = graph.cut_value("".join(current))
            if new_val > best_val + 1e-12:
                best_val = new_val
                improved = True
            else:
                # Terugdraaien
                current[u] = "1" if current[u] == "0" else "0"
        if not improved:
            break
    return "".join(current), best_val


# ============================================================
# 5. MPQS-lightcone hoofdfunctie
# ============================================================

def mpqs_lightcone(
    graph: SimpleGraph,
    radius: int = 2,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    refine: bool = True,
    max_lightcone_qubits: int = 16,
    verbose: bool = False,
) -> dict:
    """MPQS met per-vertex BFS-lightcone + lokale QAOA + ζ-spectral rounding.

    Algoritme:
      1. Voor elke vertex v: bouw BFS(v, radius) lightcone.
      2. Draai p-laag QAOA op de lokale graaf.
      3. Lees ⟨Z_v Z_u⟩ af voor elke u in lightcone — dit is de *edge-belief*.
      4. Vul ζ-matrix: ζ[v,u] = ⟨Z_v Z_u⟩ (gemiddeld over beide kanten).
      5. Round: signed-graph spectral methode — teken van bottom eigenvector
         van de Laplaciaan op ζ bepaalt cut-partitie.
      6. Optioneel: greedy 1-flip refine.

    Parameters
    ----------
    radius : int
        BFS-diepte voor lightcone.
    gammas, betas : sequence of float
        QAOA-parameters (default: p=1 met γ=0.3, β=0.2).
    refine : bool
        Als True, voer greedy 1-flip uit op de spectral-rounding cut.
    max_lightcone_qubits : int
        Hard cap op lightcone-grootte (bescherm tegen blow-up).
    """
    if gammas is None:
        gammas = [0.3]
    if betas is None:
        betas = [0.2]

    t0 = time.time()
    n = graph.n
    zeta = np.zeros((n, n))
    count = np.zeros((n, n))
    lightcone_sizes: list[int] = []
    skipped: list[int] = []

    for center in range(n):
        vertices, local_edges, mapping = _build_lightcone(graph, center, radius)
        L = len(vertices)
        lightcone_sizes.append(L)

        if L > max_lightcone_qubits:
            # Te groot; val terug op lege belief (ζ = 0 langs deze edges)
            skipped.append(center)
            continue
        if L <= 1:
            continue

        psi = _qaoa_statevector(L, local_edges, gammas, betas)
        center_local = mapping[center]
        for global_u in vertices:
            if global_u == center:
                continue
            local_u = mapping[global_u]
            zz = _expectation_zz(psi, L, center_local, local_u)
            zeta[center, global_u] += zz
            count[center, global_u] += 1.0

    # Symmetriseer: ζ[u,v] = gemiddelde van beide perspectieven (waar beschikbaar)
    zeta_sym = np.zeros_like(zeta)
    for u in range(n):
        for v in range(u + 1, n):
            s = 0.0
            c = 0
            if count[u, v] > 0:
                s += zeta[u, v]
                c += 1
            if count[v, u] > 0:
                s += zeta[v, u]
                c += 1
            if c > 0:
                zeta_sym[u, v] = s / c
                zeta_sym[v, u] = s / c

    # Spectral rounding: bouw "signed adjacency" A waar A[u,v] = ζ_sym[u,v]
    # (in MaxCut: ζ ≈ -1 als u,v in verschillende partities, ≈ +1 als gelijk).
    # Bottom eigenvector van L_signed = D − A geeft de partities (frustratie-min).
    # Alternatief — en stabieler op lightcones — gebruik direct de eigenvector
    # van -A (grootst negatieve eigenwaarde zoekt sterkst-gefrustreerde knippen).
    # We kiezen eenvoudigste variant: min eigenvector van A (signed adjacency).

    eig_vals, eig_vecs = np.linalg.eigh(zeta_sym)
    # Bottom eigenvector
    v_min = eig_vecs[:, 0]
    bits = "".join("1" if v_min[i] > 0 else "0" for i in range(n))

    # Probeer ook de gespiegelde versie (bits en ~bits geven dezelfde cut,
    # maar greedy kan startafhankelijk zijn — we kiezen de beste).
    flipped = "".join("1" if b == "0" else "0" for b in bits)
    cut_a = graph.cut_value(bits)
    cut_b = graph.cut_value(flipped)
    if cut_b > cut_a:
        bits = flipped
        cut_value = cut_b
    else:
        cut_value = cut_a

    if refine:
        refined_bits, refined_val = _greedy_1flip(graph, bits)
        bits = refined_bits
        cut_value = refined_val

    wall = time.time() - t0

    if verbose:
        print(f"  lightcone-sizes: min={min(lightcone_sizes)} mean={np.mean(lightcone_sizes):.1f} max={max(lightcone_sizes)}")
        if skipped:
            print(f"  geskipped ({len(skipped)} vertices) wegens >{max_lightcone_qubits} qubits")
        print(f"  ζ-matrix eigen-spec: min={eig_vals[0]:+.3f}, max={eig_vals[-1]:+.3f}")
        print(f"  cut-value = {cut_value}")

    return {
        "cut_bits": bits,
        "cut_value": cut_value,
        "zeta_matrix": zeta_sym,
        "eigen_vals": eig_vals,
        "eigen_vecs": eig_vecs,
        "lightcone_sizes": lightcone_sizes,
        "n_skipped": len(skipped),
        "wall_time": wall,
    }


# ============================================================
# 6. CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=12, help="Aantal knopen (default: 12)")
    p.add_argument("--reg", type=int, default=3, help="Regulariteit random graaf (default: 3)")
    p.add_argument("--petersen", action="store_true", help="Gebruik Petersen graaf")
    p.add_argument("--seed", type=int, default=42, help="RNG-seed (default: 42)")
    p.add_argument("--radius", type=int, default=2, help="Lightcone-BFS-radius (default: 2)")
    p.add_argument("--p", type=int, default=1, help="QAOA-lagen (default: 1)")
    p.add_argument("--gamma", type=float, default=0.3, help="QAOA γ (default: 0.3)")
    p.add_argument("--beta", type=float, default=0.2, help="QAOA β (default: 0.2)")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.petersen:
        from b156_sos2_sdp import petersen_graph
        g = petersen_graph()
        name = "Petersen"
    else:
        g = random_3regular(args.n, seed=args.seed)
        name = f"random_{args.reg}-reg_n={args.n}_seed={args.seed}"

    print(f"Graaf: {name}  (n={g.n}, m={g.n_edges})")

    # Brute force referentie (alleen kleine n)
    bf = None
    if g.n <= 18:
        bf = brute_force_maxcut(g)
        print(f"  brute force OPT = {bf}")

    # BP
    print("\n[1] Classical max-product BP")
    t0 = time.time()
    bp_res = mpqs_classical_bp(g, verbose=True)
    print(f"  cut = {bp_res['cut_value']:.2f}"
          f"  (converged={bp_res['converged']}, iters={bp_res['n_iters']},"
          f"  Δ={bp_res['last_delta']:.2e},  t={time.time()-t0:.3f}s)")


    # Lightcone
    print(f"\n[2] MPQS-lightcone (radius={args.radius}, p={args.p})")
    gammas = [args.gamma] * args.p
    betas = [args.beta] * args.p
    lc_res = mpqs_lightcone(g, radius=args.radius, gammas=gammas, betas=betas, verbose=True)
    print(f"  cut = {lc_res['cut_value']:.2f}  (t={lc_res['wall_time']:.3f}s)")

    if bf is not None:
        print(f"\nratio(BP)        = {bp_res['cut_value']/bf:.4f}")
        print(f"ratio(lightcone) = {lc_res['cut_value']/bf:.4f}")


if __name__ == "__main__":
    _main()
