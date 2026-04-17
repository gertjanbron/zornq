#!/usr/bin/env python3
"""B12 benchmark — verificatie van octonion-spinor correspondentie.

Vijf secties:
  1. Split-octonion Zorn-algebra: basis-structuur, idempotenten, nilpotenten.
  2. Associator-statistiek en Moufang-identiteiten (alternatief-algebra bewijs).
  3. Cl(4,3) gamma-matrices uit L-multiplication + volledige metric.
  4. Fermion-algebra F_3 anti-commutatie-tabel.
  5. Bijection Φ: bewezen lineaire iso + falsifikatie module-morfisme.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b12_octonion_spinor import (
    BASIS_NAMES,
    L_matrix,
    R_matrix,
    anticommutator,
    associator,
    basis_vector,
    clifford_algebra_dim,
    clifford_generators_7d,
    clifford_metric,
    fock_annihilation,
    fock_creation,
    moufang_identity_left,
    moufang_identity_right,
    phi_bijection,
    triality_discrepancy,
    zorn_identity,
    zorn_mul,
    zorn_norm,
)


def _hdr(title: str) -> None:
    bar = "=" * 100
    print(bar); print(f"  {title}"); print(bar)


# ============================================================
# 1. Split-octonion structuur
# ============================================================


def bench_zorn_structure() -> None:
    _hdr("1. Zorn-algebra structuur: Peirce-idempotenten + nilpotenten")
    print("  Basis van 𝕆_s: ", ", ".join(f"{i}={n}" for i, n in enumerate(BASIS_NAMES)))
    print()
    print("  Zelfproduct-tabel (A*A):")
    print(f"  {'elt':>4}  {'N(A)':>7}  {'A*A':>28}")
    for k in range(8):
        v = basis_vector(k)
        sq = zorn_mul(v, v)
        lbl = "0" if np.allclose(sq, 0) else (BASIS_NAMES[k] if np.allclose(sq, v) else str(sq.astype(int).tolist()))
        print(f"  {BASIS_NAMES[k]:>4}  {zorn_norm(v):+.3f}   {lbl:>28}")
    print()
    print(f"  e0·e7 = {zorn_mul(basis_vector(0), basis_vector(7)).astype(int).tolist()}  (orthogonal)")
    print(f"  e0+e7 = {(basis_vector(0)+basis_vector(7)).astype(int).tolist()}  (= 1)")
    print()
    print("  Kruisproduct-tabel j_i · j_j (α-basis, i,j ∈ {1,2,3}):")
    print(f"  {'':>4}  " + "  ".join(f"{BASIS_NAMES[j]:>6}" for j in (1, 2, 3)))
    for i in (1, 2, 3):
        row = [f"{BASIS_NAMES[i]:>4}"]
        for j in (1, 2, 3):
            prod = zorn_mul(basis_vector(i), basis_vector(j))
            if np.allclose(prod, 0):
                row.append(f"{'0':>6}")
            else:
                # Zoek welke basisvector dit is (met teken)
                name = "?"
                for k in range(8):
                    if np.allclose(prod, basis_vector(k)):
                        name = f"+{BASIS_NAMES[k]}"; break
                    if np.allclose(prod, -basis_vector(k)):
                        name = f"-{BASIS_NAMES[k]}"; break
                row.append(f"{name:>6}")
        print("  " + "  ".join(row))
    print()
    print("  Fermion-mode-paren: {j_i, j_{i+3}} = 1 (identity):")
    one = zorn_identity()
    for i in (1, 2, 3):
        ac = zorn_mul(basis_vector(i), basis_vector(i + 3)) + \
             zorn_mul(basis_vector(i + 3), basis_vector(i))
        ok = "OK" if np.allclose(ac, one) else "FAIL"
        print(f"  {{j{i}, j{i+3}}}: {ac.astype(int).tolist()}  [{ok}]")
    print()


# ============================================================
# 2. Non-associativity + Moufang
# ============================================================


def bench_associator() -> None:
    _hdr("2. Associator-statistiek + Moufang-identiteiten")
    print("  Kernstelling: (𝕆_s, ·) is niet-associatief maar wel ALTERNATIEF.")
    print("  Toets: aantal basis-triples (i,j,k) met [e_i,e_j,e_k] ≠ 0  (/ 512):")
    cnt = 0; max_ass = 0.0; zero_count = 0
    for i, j, k in itertools.product(range(8), repeat=3):
        r = float(np.linalg.norm(associator(
            basis_vector(i), basis_vector(j), basis_vector(k))))
        if r > 1e-9:
            cnt += 1
            max_ass = max(max_ass, r)
        else:
            zero_count += 1
    print(f"    non-zero:   {cnt}/{8**3} triples")
    print(f"    zero:       {zero_count}/{8**3} triples")
    print(f"    max ||[·,·,·]||: {max_ass:.4f}")
    print()
    print("  Specifiek: [j1, j2, j3] = (j1·j2)·j3 - j1·(j2·j3)")
    a = associator(basis_vector(1), basis_vector(2), basis_vector(3))
    print(f"    → {a.astype(int).tolist()}   (= e0 - e7, bewijst niet-associativiteit)")
    print()
    print("  Moufang (alternatief-algebra): A(B(AC)) = ((AB)A)C  op 512 basis-triples:")
    max_left = 0.0; max_right = 0.0
    for i, j, k in itertools.product(range(8), repeat=3):
        max_left = max(max_left, moufang_identity_left(
            basis_vector(i), basis_vector(j), basis_vector(k)))
        max_right = max(max_right, moufang_identity_right(
            basis_vector(i), basis_vector(j), basis_vector(k)))
    print(f"    max |LHS - RHS|  (links) = {max_left:.2e}")
    print(f"    max |LHS - RHS|  (rechts)= {max_right:.2e}")
    print("  ⇒ 𝕆_s is alternatief (Moufang-identiteiten exact, tot machine-precisie)")
    print()


# ============================================================
# 3. Clifford gamma-matrices
# ============================================================


def bench_clifford() -> None:
    _hdr("3. Cl(4,3) gamma-matrices uit L-multiplication op 𝕆_s")
    print("  γ_i = L(j_i + j_{i+3})     i=1..3   [split-reëel]")
    print("  γ_{i+3} = L(j_i - j_{i+3}) i=1..3   [split-imaginair]")
    print("  γ_7 = L(e7 - e0)                   [chirale operator]")
    gens = clifford_generators_7d()
    eta = clifford_metric(gens)
    print()
    print("  Anti-commutator metric η (7×7) — {γ_μ,γ_ν} = 2·η_μν·I:")
    for row in eta.astype(int):
        print("    " + "  ".join(f"{v:+d}" for v in row))
    print()
    diag = np.diag(eta).astype(int)
    print(f"  Signatuur: diag(η) = {diag.tolist()}  ⇒ (p,q) = "
          f"({int(np.sum(diag>0))}, {int(np.sum(diag<0))})")
    off = eta - np.diag(np.diag(eta))
    print(f"  max |off-diag| = {np.max(np.abs(off)):.2e}  (exact 0 ⇒ orthogonale Clifford-basis)")
    print()
    print("  γ_i² verificatie:")
    for i, g in enumerate(gens):
        sq = g @ g
        val = sq[0, 0]
        ok = np.allclose(sq, val * np.eye(8))
        print(f"    γ_{i+1}² = {val:+.0f}·I   [{'OK' if ok else 'FAIL'}]")
    print()


# ============================================================
# 4. Fermion Fock F_3
# ============================================================


def bench_fock() -> None:
    _hdr("4. Fermion-algebra F_3 = Λ(ℂ³) — CAR-relaties")
    print("  {c_i, c_j†} ?= δ_ij · I   (canonical anti-commutation):")
    print(f"  {'':>5}   " + "  ".join(f"{'j='+str(j):>7}" for j in (1, 2, 3)))
    for i in (1, 2, 3):
        row = [f"  i={i:>2}"]
        for j in (1, 2, 3):
            AC = fock_annihilation(i) @ fock_creation(j) + \
                 fock_creation(j) @ fock_annihilation(i)
            val = np.diag(AC)[0]
            row.append(f"{val:>7.0f}")
        print("   " + " ".join(row))
    print()
    print("  (c_i†)² = 0  (Pauli-exclusie):")
    for i in (1, 2, 3):
        sq = fock_creation(i) @ fock_creation(i)
        ok = np.allclose(sq, 0)
        print(f"    (c_{i}†)²:  {'OK' if ok else 'FAIL'}")
    print()


# ============================================================
# 5. Bijection Φ + module-morfisme falsificatie
# ============================================================


def bench_bijection() -> None:
    _hdr("5. Bijection Φ: 𝕆_s → F₃ en module-morfisme falsificatie")
    Phi = phi_bijection()
    print(f"  Φ is orthogonaal:          Φ·Φᵀ - I  max = "
          f"{np.max(np.abs(Phi @ Phi.T - np.eye(8))):.2e}")
    print(f"  |det Φ|                    = {abs(np.linalg.det(Phi)):.3f}")
    print()
    print("  Φ · L_{j_i} - c_i† · Φ  (is Φ een module-iso? nee):")
    for i in (1, 2, 3):
        L = L_matrix(basis_vector(i))
        cd = fock_creation(i)
        d = np.linalg.norm(Phi @ L - cd @ Phi)
        print(f"    i={i}:  ||Φ·L_{{j{i}}} - c_{i}†·Φ|| = {d:.3f}  "
              f"{'(≠ 0 ⇒ niet intertwining)' if d > 1e-6 else ''}")
    print()
    print("  Conclusie: Φ is een LINEAIRE iso van 8-dim ℝ-ruimten,")
    print("             GEEN algebra/module-iso want 𝕆_s is niet-associatief")
    print("             en F_3 (via fermion-op-algebra) wel.")
    print()


# ============================================================
# 6. Falsificatie + triality
# ============================================================


def bench_claims() -> None:
    _hdr("6. Claim-toetsing: Cl(4,4) ≅? 𝕆_s + triality-indicatie")
    print("  Incorrecte claim:  Cl(4,4) ≅ 𝕆_s")
    print(f"    dim Cl(4,4) = 2^8 = {clifford_algebra_dim(4,4)}")
    print(f"    dim 𝕆_s     =       {8}")
    print(f"    → VERSCHIL FACTOR {clifford_algebra_dim(4,4) // 8} ⇒ claim FOUT.")
    print()
    print("  Correcte koppeling:  𝕆_s is spinor-module voor Cl(4,3)")
    print(f"    dim Cl(4,3)              = 2^7 = {clifford_algebra_dim(4,3)}")
    print(f"    dim spinor-module S      = 2^⌊7/2⌋ = {2**(7//2)}")
    print(f"    geconstrueerd op 𝕆_s     = 7 gamma-matrices 8×8  (zie sectie 3)")
    print()
    print("  Triality-indicatie: L_a ≠ R_a op Im(𝕆_s)")
    d = triality_discrepancy()
    print(f"    {'elt':>4}  {'||L-R||':>10}  {'||L-R^T||':>10}")
    for k in range(1, 8):
        d1, d2 = d[BASIS_NAMES[k]]
        print(f"    {BASIS_NAMES[k]:>4}  {d1:>10.3f}  {d2:>10.3f}")
    print(f"    totaal ||L-R|| (sum) = {d['__total_L_minus_R__']:.3f}")
    print("  ⇒ L en R zijn inequivalente 8-dim reps; samen met de adjoint-vector-rep")
    print("     geven ze de drie pootjes van Spin(4,4)-triality op 𝕆_s (Baez 2002).")
    print()


# ============================================================
# Hoofd
# ============================================================


def main() -> int:
    t0 = time.time()
    bench_zorn_structure()
    bench_associator()
    bench_clifford()
    bench_fock()
    bench_bijection()
    bench_claims()
    print(f"\nTotaal walltime: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
