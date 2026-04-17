#!/usr/bin/env python3
"""B160 benchmark — QSVT/block-encoding vs Trotter op realistische Ham-simulatie.

Drie secties:

  1. QSP-polynomen T_1..T_6 — check dat onze minimale QSP-angle-set exact
     Chebyshev-polynomen T_k(x) reproduceert op het interval [-1, 1].

  2. LCU block-encoding van Ising- en Heisenberg-Hamiltonians — schaling van
     α vs ‖H‖_op, ancilla-count, verificatie van het top-left blok.

  3. Jacobi-Anger Hamiltonian-simulatie (exponentieel convergerende Chebyshev-
     expansie, ∼O(αt + log 1/ε) termen) versus:
       - order-1 Trotter met zelfde walltime-budget
       - order-2 Trotter (Strang-splitting)
     Toont waar QSVT significant beter is (grote αt, hoge accuracy-eisen).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b160_qsvt import (
    PauliSum,
    block_encode_pauli_sum,
    chebyshev_T_phases,
    hamiltonian_simulation_qsvt,
    jacobi_anger_truncation,
    pauli_matrix,
    qsp_unitary,
    trotter_reference,
    verify_block_encoding,
)


def _hdr(title: str) -> None:
    bar = "=" * 100
    print(bar); print(f"  {title}"); print(bar)


def _row(cols, widths) -> None:
    print("  ".join(str(c).rjust(w) for c, w in zip(cols, widths)))


# ============================================================
# 1. QSP-polynoom verificatie
# ============================================================

def bench_qsp_polynomials() -> None:
    _hdr("1. QSP-polynomen T_k(x) (fases Φ=(0,0,...,0), k+1 entries)")
    widths = [4, 10, 12, 12, 12]
    _row(["k", "x", "T_k(x) exact", "QSP top-left", "|diff|"], widths)
    print("-" * 80)
    xs = [-0.9, -0.3, 0.1, 0.6, 0.95]
    from numpy.polynomial.chebyshev import Chebyshev
    for k in range(1, 7):
        Tk = Chebyshev([0] * k + [1])
        phases = chebyshev_T_phases(k)
        for x in xs:
            U = qsp_unitary(x, phases)
            val = U[0, 0]
            exact = Tk(x)
            _row([k, f"{x:+.2f}", f"{exact:+.5f}",
                  f"{val.real:+.5f}", f"{abs(val.real - exact):.2e}"], widths)
    print()


# ============================================================
# 2. LCU block-encoding op klassieke Hamiltonians
# ============================================================

def _ising_tf_pauli_sum(n: int, J: float = 1.0, h: float = 0.5) -> PauliSum:
    terms = []
    for i in range(n - 1):
        label = ["I"] * n; label[i] = "Z"; label[i + 1] = "Z"
        terms.append((-J, "".join(label)))
    for i in range(n):
        label = ["I"] * n; label[i] = "X"
        terms.append((-h, "".join(label)))
    return PauliSum(n_qubits=n, terms=terms)


def _heisenberg_xxx_pauli_sum(n: int, J: float = 1.0) -> PauliSum:
    terms = []
    for i in range(n - 1):
        for P in "XYZ":
            label = ["I"] * n; label[i] = P; label[i + 1] = P
            terms.append((J, "".join(label)))
    return PauliSum(n_qubits=n, terms=terms)


def bench_block_encodings() -> None:
    _hdr("2. LCU block-encoding: alpha, ancilla-count, verificatie")
    widths = [28, 4, 6, 8, 10, 10, 10]
    _row(["model", "n", "terms", "m_anc", "alpha", "||H||_op", "verify"], widths)
    print("-" * 100)
    cases = []
    for n in [2, 3, 4]:
        cases.append((f"Ising-TF(J=1,h=0.5)", n, _ising_tf_pauli_sum(n)))
    for n in [2, 3, 4]:
        cases.append((f"Heisenberg-XXX(J=1)", n, _heisenberg_xxx_pauli_sum(n)))
    cases.append(("MaxCut K_3", 3, PauliSum(n_qubits=3, terms=[
        (-0.5, "ZZI"), (-0.5, "IZZ"), (-0.5, "ZIZ"),
    ])))
    for label, n, ps in cases:
        H = ps.to_matrix()
        be = block_encode_pauli_sum(ps)
        ok = verify_block_encoding(be, H)
        op_norm = float(np.linalg.norm(H, 2))
        _row([label, n, len(ps), be.m_ancilla,
              f"{be.alpha:.3f}", f"{op_norm:.3f}",
              "OK" if ok else "FAIL"], widths)
    print()


# ============================================================
# 3. Jacobi-Anger vs Trotter
# ============================================================

def bench_qsvt_vs_trotter() -> None:
    _hdr("3. Jacobi-Anger Ham-sim vs Trotter (order 1 en 2)")

    # Ising-TF n=3 — niet-commuterend, goede stress-test
    ps = _ising_tf_pauli_sum(3, J=1.0, h=0.5)
    H = ps.to_matrix()
    alpha = ps.alpha()
    # Converteer naar (coeff, matrix)-paren voor trotter_reference
    h_terms = [(c, pauli_matrix(lbl)) for c, lbl in ps.terms]

    widths = [5, 8, 8, 14, 14, 14, 14]
    _row(["t", "alpha*t", "K_JA",
          "err_JA", "err_T1 (s=20)", "err_T2 (s=20)",
          "time_JA"], widths)
    print("-" * 100)

    from scipy.linalg import expm
    for t in [0.1, 0.5, 1.0, 2.0, 4.0]:
        U_exact = expm(-1j * H * t)

        # Jacobi-Anger
        t0 = time.time()
        U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha)
        dt_ja = (time.time() - t0) * 1000
        err_ja = np.linalg.norm(U_ja - U_exact, 2)

        # Trotter-1 / Trotter-2 bij fixed 20 stappen
        U_t1 = trotter_reference(h_terms, t, 20, order=1)
        err_t1 = np.linalg.norm(U_t1 - U_exact, 2)
        U_t2 = trotter_reference(h_terms, t, 20, order=2)
        err_t2 = np.linalg.norm(U_t2 - U_exact, 2)

        K = jacobi_anger_truncation(alpha * t)
        _row([f"{t:.1f}", f"{alpha*t:.2f}", K,
              f"{err_ja:.2e}", f"{err_t1:.2e}", f"{err_t2:.2e}",
              f"{dt_ja:.1f}ms"], widths)
    print()

    # Heisenberg n=4 — grotere systeem
    _hdr("3b. Heisenberg-XXX n=4 — grotere Hilbert-ruimte")
    ps = _heisenberg_xxx_pauli_sum(4)
    H = ps.to_matrix()
    alpha = ps.alpha()
    h_terms = [(c, pauli_matrix(lbl)) for c, lbl in ps.terms]

    _row(["t", "alpha*t", "K_JA",
          "err_JA", "err_T1 (s=40)", "err_T2 (s=40)",
          "time_JA"], widths)
    print("-" * 100)
    for t in [0.2, 0.8, 2.0]:
        U_exact = expm(-1j * H * t)
        t0 = time.time()
        U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha)
        dt_ja = (time.time() - t0) * 1000
        err_ja = np.linalg.norm(U_ja - U_exact, 2)
        U_t1 = trotter_reference(h_terms, t, 40, order=1)
        err_t1 = np.linalg.norm(U_t1 - U_exact, 2)
        U_t2 = trotter_reference(h_terms, t, 40, order=2)
        err_t2 = np.linalg.norm(U_t2 - U_exact, 2)
        K = jacobi_anger_truncation(alpha * t)
        _row([f"{t:.1f}", f"{alpha*t:.2f}", K,
              f"{err_ja:.2e}", f"{err_t1:.2e}", f"{err_t2:.2e}",
              f"{dt_ja:.1f}ms"], widths)
    print()


# ============================================================
# 4. Accuracy-convergentie als functie van K
# ============================================================

def bench_K_convergence() -> None:
    _hdr("4. Convergentie in K — fout vs truncatie-graad (Ising-TF n=3, t=2.0)")
    ps = _ising_tf_pauli_sum(3, J=1.0, h=0.5)
    H = ps.to_matrix()
    alpha = ps.alpha()
    t = 2.0
    from scipy.linalg import expm
    U_exact = expm(-1j * H * t)
    print(f"    alpha*t = {alpha*t:.2f}   (empirische knik rond K ≈ e·αt/2 = {np.e*alpha*t/2:.1f})")
    widths = [6, 14]
    _row(["K", "||U_JA - U_exact||"], widths)
    print("-" * 40)
    for K in [4, 8, 12, 16, 20, 24, 30, 40]:
        U_ja = hamiltonian_simulation_qsvt(H, t, alpha=alpha, K=K)
        err = np.linalg.norm(U_ja - U_exact, 2)
        _row([K, f"{err:.3e}"], widths)
    print()


# ============================================================
# Hoofd
# ============================================================

def main() -> int:
    t0 = time.time()
    bench_qsp_polynomials()
    bench_block_encodings()
    bench_qsvt_vs_trotter()
    bench_K_convergence()
    print(f"\nTotaal walltime: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
