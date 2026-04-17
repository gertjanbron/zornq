#!/usr/bin/env python3
"""
B165: Qiskit Runtime Hardware-Run pipeline.

End-to-end pipeline om ZornQ-circuits te exporteren naar Qiskit en uit te voeren
op:
  1. AerSimulator (lokaal, geen ruis)            — referentie / unit tests
  2. AerSimulator + DepolarisingNoise           — hardware-proxy (geen credentials)
  3. IBM Quantum Runtime (Sampler V2)           — echte hardware (vereist token)

De ZornQ Circuit-class (B128) wordt term-voor-term naar `qiskit.QuantumCircuit`
gecompileerd. We ondersteunen alle gebruikte gates:
   H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ, SWAP, RXX, RYY, RZZ.

Voor MaxCut bepalen we ⟨H_C⟩ uit de gemeten sample-distributie:
   E[H_C] = Σ_{(u,v) ∈ E} w_uv · (1 − ⟨Z_u Z_v⟩) / 2
waarbij ⟨Z_u Z_v⟩ uit de bitstrings volgt.

Hardware-pad:
  Vereist QISKIT_IBM_TOKEN env-var (of `~/.qiskit/qiskit-ibm.json`); zonder
  raisen we expliciet `RuntimeError` met instructies. We gebruiken nooit
  blocking submit in tests — alleen Aer.

Gebruik:
  python b165_qiskit_runtime.py --aer --petersen
  python b165_qiskit_runtime.py --noisy --p1err 0.001 --random 8
  python b165_qiskit_runtime.py --hardware --petersen --backend ibm_brisbane
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

from b60_gw_bound import (
    SimpleGraph,
    brute_force_maxcut,
    random_3regular,
    random_erdos_renyi,
)


# ============================================================
# ZornQ Circuit → Qiskit QuantumCircuit
# ============================================================

def to_qiskit(circuit) -> "QuantumCircuit":  # noqa: F821
    """Vertaal een ZornQ `Circuit` naar een `qiskit.QuantumCircuit`.

    De ZornQ-Circuit gebruikt rzz/rxx/ryy met een conventie waarin het
    fasescheiding-argument θ direct in `exp(-i θ/2 · ZZ)` zit (zie B128
    `Gates.RZZ`). Qiskit gebruikt dezelfde conventie: `qc.rzz(theta)` =
    `exp(-i theta/2 · ZZ)`. Geen factor-correcties nodig.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(circuit.n_qubits, circuit.n_qubits)
    for op in circuit.ops:
        name = op.name
        qubits = op.qubits
        params = op.params if op.params is not None else ()

        if name == "H":
            qc.h(qubits[0])
        elif name == "X":
            qc.x(qubits[0])
        elif name == "Y":
            qc.y(qubits[0])
        elif name == "Z":
            qc.z(qubits[0])
        elif name == "S":
            qc.s(qubits[0])
        elif name == "T":
            qc.t(qubits[0])
        elif name == "RX":
            qc.rx(float(params[0]), qubits[0])
        elif name == "RY":
            qc.ry(float(params[0]), qubits[0])
        elif name == "RZ":
            qc.rz(float(params[0]), qubits[0])
        elif name == "CNOT":
            qc.cx(qubits[0], qubits[1])
        elif name == "CZ":
            qc.cz(qubits[0], qubits[1])
        elif name == "SWAP":
            qc.swap(qubits[0], qubits[1])
        elif name == "RXX":
            qc.rxx(float(params[0]), qubits[0], qubits[1])
        elif name == "RYY":
            qc.ryy(float(params[0]), qubits[0], qubits[1])
        elif name == "RZZ":
            qc.rzz(float(params[0]), qubits[0], qubits[1])
        else:
            raise NotImplementedError(
                f"B165 to_qiskit: gate '{name}' nog niet ondersteund.")
    return qc


def add_measurements(qc) -> "QuantumCircuit":  # noqa: F821
    """Voeg per qubit een measurement toe (in Z-basis)."""
    qc_meas = qc.copy()
    qc_meas.measure(range(qc.num_qubits), range(qc.num_qubits))
    return qc_meas


# ============================================================
# Sample-gebaseerde verwachtingswaarden
# ============================================================

def expectation_zz_from_counts(counts: dict[str, int], i: int, j: int,
                               n: int) -> float:
    """⟨Z_i Z_j⟩ uit shot-counts. Bitstrings in Qiskit-conventie (rechts→links)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    s = 0
    for bits, cnt in counts.items():
        # Qiskit bitstring is little-endian: bits[-(q+1)] = qubit q
        bi = bits[-(i + 1)]
        bj = bits[-(j + 1)]
        zi = 1 if bi == "0" else -1
        zj = 1 if bj == "0" else -1
        s += zi * zj * cnt
    return s / total


def maxcut_value_from_counts(counts: dict[str, int],
                              graph: SimpleGraph) -> float:
    """E[H_C] = Σ w_uv (1 − ⟨Z_u Z_v⟩)/2."""
    total = 0.0
    for u, v, w in graph.edges:
        zz = expectation_zz_from_counts(counts, u, v, graph.n)
        total += w * (1.0 - zz) / 2.0
    return total


def best_cut_from_counts(counts: dict[str, int],
                          graph: SimpleGraph) -> tuple[int, str]:
    """Beste cut-waarde over alle gemeten bitstrings (geen verwachting)."""
    best_val = -1.0
    best_bits = ""
    for bits, _cnt in counts.items():
        # Convert to bitstring index 0..n-1 via little-endian
        bs = [int(bits[-(q + 1)]) for q in range(graph.n)]
        v = 0.0
        for u, vtx, w in graph.edges:
            if bs[u] != bs[vtx]:
                v += w
        if v > best_val:
            best_val = v
            best_bits = bits
    return int(best_val), best_bits


# ============================================================
# Aer execution (lokaal)
# ============================================================

def run_aer(circuit, shots: int = 8192, seed: int = 42,
            noise_model=None) -> dict[str, Any]:
    """Voer een ZornQ-circuit uit op AerSimulator.

    Parameters
    ----------
    circuit : ZornQ Circuit
    shots   : aantal shots
    seed    : RNG seed (sampler reproducibiliteit)
    noise_model : Qiskit AerNoiseModel, optioneel
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc = to_qiskit(circuit)
    qc_meas = add_measurements(qc)

    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    qc_t = transpile(qc_meas, sim)

    t0 = time.time()
    job = sim.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    wall = time.time() - t0

    return {
        "counts": dict(counts),
        "shots": shots,
        "n_qubits": circuit.n_qubits,
        "depth": qc.depth(),
        "wall_time": wall,
        "backend": "AerSimulator",
        "noisy": noise_model is not None,
    }


def make_depolarising_noise(p1: float = 1e-3, p2: float = 1e-2):
    """Bouw een Aer NoiseModel met 1-qubit en 2-qubit depolariserende ruis.

    Realistische orde van grootte voor IBM Eagle/Heron rond april 2026:
    p1 ≈ 1e-4 (single-qubit), p2 ≈ 5e-3 tot 1e-2 (CX). Default conservatief.
    """
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
    )

    noise = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    # 1q gates die in to_qiskit voorkomen
    for g in ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]:
        noise.add_all_qubit_quantum_error(err1, [g])
    # 2q gates
    for g in ["cx", "cz", "swap", "rxx", "ryy", "rzz"]:
        noise.add_all_qubit_quantum_error(err2, [g])
    return noise


# ============================================================
# IBM Runtime Sampler V2 (echte hardware)
# ============================================================

def run_ibm_runtime(
    circuit,
    backend_name: str = "ibm_brisbane",
    shots: int = 4096,
    token_env: str = "QISKIT_IBM_TOKEN",
    instance: str | None = None,
    skip_if_no_token: bool = True,
) -> dict[str, Any]:
    """Submit een ZornQ-circuit naar echte IBM Quantum hardware.

    Vereist:
      - `QISKIT_IBM_TOKEN` env-var of opgeslagen account
      - netwerk-toegang (in deze sandbox NIET beschikbaar)
      - hardware-queue-tijd (uren tot dagen)

    Bij ontbrekende token + `skip_if_no_token=True` retourneren we een
    SKIPPED-dict zodat tests/CI niet falen.
    """
    token = os.environ.get(token_env)
    if not token:
        if skip_if_no_token:
            return {
                "status": "SKIPPED_NO_TOKEN",
                "backend": backend_name,
                "message": (f"Set ${token_env} en provider-instance om te "
                            f"submitten. Gebruik run_aer() voor lokale tests."),
            }
        raise RuntimeError(
            f"Geen ${token_env} env-var gevonden. Set token via "
            f"QiskitRuntimeService.save_account() of env-var.")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile

    service = QiskitRuntimeService(channel="ibm_quantum",
                                    token=token, instance=instance)
    backend = service.backend(backend_name)
    qc = to_qiskit(circuit)
    qc_meas = add_measurements(qc)
    qc_t = transpile(qc_meas, backend)

    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run([qc_t], shots=shots)
    result = job.result()
    wall = time.time() - t0

    pub = result[0]
    counts = pub.data.c.get_counts()
    return {
        "counts": dict(counts),
        "shots": shots,
        "n_qubits": circuit.n_qubits,
        "depth": qc.depth(),
        "wall_time": wall,
        "backend": backend_name,
        "job_id": job.job_id(),
    }


# ============================================================
# QAOA helper: bouw + run + rapporteer
# ============================================================

def qaoa_maxcut_run(
    graph: SimpleGraph,
    p: int = 1,
    gammas: list[float] | None = None,
    betas: list[float] | None = None,
    backend: str = "aer",
    shots: int = 8192,
    seed: int = 42,
    p1err: float = 1e-3,
    p2err: float = 1e-2,
    ibm_backend: str = "ibm_brisbane",
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end: bouw QAOA-MaxCut circuit en draai het.

    backend ∈ {"aer", "noisy", "hardware"}.
    """
    from circuit_interface import Circuit

    if gammas is None:
        gammas = [0.4] * p
    if betas is None:
        betas = [0.3] * p

    edges = [(int(u), int(v), float(w)) for u, v, w in graph.edges]
    qc = Circuit.qaoa_maxcut(graph.n, edges, p=p, gammas=gammas, betas=betas)

    if backend == "aer":
        result = run_aer(qc, shots=shots, seed=seed)
    elif backend == "noisy":
        nm = make_depolarising_noise(p1=p1err, p2=p2err)
        result = run_aer(qc, shots=shots, seed=seed, noise_model=nm)
    elif backend == "hardware":
        result = run_ibm_runtime(qc, backend_name=ibm_backend, shots=shots)
        if result.get("status") == "SKIPPED_NO_TOKEN":
            if verbose:
                print(f"  [hardware] {result['message']}")
            return result
    else:
        raise ValueError(f"Onbekende backend: {backend}")

    counts = result["counts"]
    expectation = maxcut_value_from_counts(counts, graph)
    best_cut, best_bs = best_cut_from_counts(counts, graph)
    opt = brute_force_maxcut(graph) if graph.n <= 18 else None

    result.update({
        "qaoa_expectation": expectation,
        "best_cut_seen": best_cut,
        "best_bitstring": best_bs,
        "opt": opt,
        "approx_ratio_expectation": expectation / opt if opt else None,
        "approx_ratio_best": best_cut / opt if opt else None,
        "p": p,
        "gammas": list(gammas),
        "betas": list(betas),
    })

    if verbose:
        print(f"\n=== QAOA p={p} on {backend} (n={graph.n}, m={graph.n_edges}) ===")
        print(f"  Backend:       {result['backend']}")
        print(f"  Shots:         {shots}, depth: {result.get('depth')}")
        print(f"  E[H_C]:        {expectation:.4f}")
        print(f"  Best cut seen: {best_cut} (bitstring={best_bs})")
        if opt is not None:
            print(f"  OPT:           {opt}")
            print(f"  Approx ratio:  E={result['approx_ratio_expectation']:.4f}, "
                  f"best={result['approx_ratio_best']:.4f}")
        print(f"  Wall-time:     {result.get('wall_time'):.2f}s")

    return result


# ============================================================
# CLI
# ============================================================

def _build_graph(args: argparse.Namespace) -> tuple[SimpleGraph, str]:
    if args.petersen:
        from b156_sos2_sdp import petersen_graph
        return petersen_graph(), "Petersen"
    if args.cycle is not None:
        from b156_sos2_sdp import cycle_graph
        return cycle_graph(args.cycle), f"C_{args.cycle}"
    if args.n is not None:
        from b156_sos2_sdp import complete_graph
        return complete_graph(args.n), f"K_{args.n}"
    if args.random is not None:
        return random_3regular(args.random, seed=args.seed), f"3reg_n{args.random}"
    if args.erdos is not None:
        return random_erdos_renyi(args.erdos, p=args.p, seed=args.seed), \
               f"ER_n{args.erdos}_p{args.p}"
    from b156_sos2_sdp import complete_graph
    return complete_graph(4), "K_4"


def main() -> None:
    parser = argparse.ArgumentParser(description="B165 Qiskit Runtime Pipeline")
    parser.add_argument("--n", type=int, help="K_n")
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--petersen", action="store_true")
    parser.add_argument("--random", type=int)
    parser.add_argument("--erdos", type=int)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qaoa-p", type=int, default=1, dest="qaoa_p")
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--aer", action="store_true",
                        help="Aer simulator (no noise)")
    parser.add_argument("--noisy", action="store_true",
                        help="Aer + depolarising noise")
    parser.add_argument("--p1err", type=float, default=1e-3)
    parser.add_argument("--p2err", type=float, default=1e-2)
    parser.add_argument("--hardware", action="store_true",
                        help="IBM Quantum Runtime (vereist token)")
    parser.add_argument("--ibm-backend", default="ibm_brisbane")
    args = parser.parse_args()

    graph, name = _build_graph(args)
    backend = "hardware" if args.hardware else ("noisy" if args.noisy else "aer")
    print(f"Graaf: {name}")
    qaoa_maxcut_run(
        graph, p=args.qaoa_p,
        gammas=[args.gamma] * args.qaoa_p,
        betas=[args.beta] * args.qaoa_p,
        backend=backend, shots=args.shots, seed=args.seed,
        p1err=args.p1err, p2err=args.p2err,
        ibm_backend=args.ibm_backend,
    )


if __name__ == "__main__":
    main()
