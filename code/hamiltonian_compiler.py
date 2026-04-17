#!/usr/bin/env python3
"""
hamiltonian_compiler.py - B129 Hamiltonian Compiler

Gegeven een willekeurige Hamiltonian (Ising, Heisenberg, Hubbard, molecuul,
PDE-discretisatie), compileer automatisch naar QAOA-lagen, Trotter-stappen,
of VQE-ansatz. Ondersteunt:

- Hamiltonian class met model library (Ising, Heisenberg XXX/XXZ, Hubbard, molecuul)
- Jordan-Wigner transformatie (fermion -> qubit mapping)
- Hogere-orde Trotter (1e, 2e, 4e orde Suzuki)
- Commuterende groepen (term-groepering voor gereduceerde Trotter-fout)
- Gate-optimalisatie (merge rotaties, cancel inverses)
- Custom mixers (X, XY, Grover)

Gebruik:
    from hamiltonian_compiler import Hamiltonian

    # Bouw Hamiltonian
    H = Hamiltonian.heisenberg_xxz(n=10, Jxy=1.0, Jz=0.5)
    H = Hamiltonian.hubbard_1d(n_sites=4, t=1.0, U=4.0)
    H = Hamiltonian.molecular(one_body, two_body)

    # Compileer naar circuit
    qc = H.trotter(t=1.0, steps=10, order=2)
    qc = H.qaoa(p=3, gammas=[...], betas=[...])

    # Draai
    from circuit_interface import run_circuit, Observable
    result = run_circuit(qc, observables={'E': H.to_observable()})

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Set
from dataclasses import dataclass, field
from circuit_interface import Circuit, Gates, Observable, _append_pauli_rotation


# =====================================================================
# HAMILTONIAN CLASS
# =====================================================================

class Hamiltonian:
    """Willekeurige qubit Hamiltonian als gewogen som van Pauli strings.

    H = sum_k c_k * P_k, waar P_k = tensor product van {I, X, Y, Z}.

    Representatie: list van (coeff, pauli_string) tuples.
    pauli_string: dict {qubit_index: 'X'|'Y'|'Z'} (I is impliciet).
    """

    def __init__(self, terms=None, n_qubits=None, name="H"):
        self.terms: List[Tuple[complex, Dict[int, str]]] = terms or []
        self._n_qubits = n_qubits
        self.name = name

    @property
    def n_qubits(self):
        if self._n_qubits is not None:
            return self._n_qubits
        if not self.terms:
            return 0
        return max(
            (max(p.keys()) + 1 if p else 0) for _, p in self.terms
        )

    @property
    def n_terms(self):
        return len(self.terms)

    def add_term(self, coeff, pauli_string):
        """Voeg een term toe: coeff * Pauli_string."""
        self.terms.append((complex(coeff), dict(pauli_string)))
        return self

    def __add__(self, other):
        """Hamiltonianen optellen."""
        n = max(self.n_qubits, other.n_qubits)
        return Hamiltonian(self.terms + other.terms, n,
                           "%s+%s" % (self.name, other.name))

    def __mul__(self, scalar):
        """Scalaire vermenigvuldiging."""
        return Hamiltonian(
            [(c * scalar, dict(p)) for c, p in self.terms],
            self._n_qubits, self.name
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def simplify(self):
        """Combineer identieke Pauli strings, verwijder nul-termen."""
        combined = {}
        for coeff, pauli in self.terms:
            key = tuple(sorted(pauli.items()))
            combined[key] = combined.get(key, 0.0) + coeff
        self.terms = [
            (c, dict(k)) for k, c in combined.items()
            if abs(c) > 1e-15
        ]
        return self

    def norm(self):
        """Spectrale norm schatting: sum |c_k|."""
        return sum(abs(c) for c, _ in self.terms)

    def to_observable(self):
        """Converteer naar Observable (voor meting)."""
        return Observable([(float(c.real), dict(p)) for c, p in self.terms])

    # -----------------------------------------------------------------
    # MODEL CONSTRUCTORS
    # -----------------------------------------------------------------

    @classmethod
    def ising_transverse(cls, n, J=1.0, h=0.5, periodic=False):
        """Transverse-field Ising: H = -J * sum ZZ - h * sum X."""
        terms = []
        for i in range(n - 1 + (1 if periodic else 0)):
            j = (i + 1) % n
            terms.append((-J, {i: 'Z', j: 'Z'}))
        for i in range(n):
            terms.append((-h, {i: 'X'}))
        return cls(terms, n, "Ising-TF(n=%d,J=%.2f,h=%.2f)" % (n, J, h))

    @classmethod
    def heisenberg_xxx(cls, n, J=1.0, periodic=False):
        """Heisenberg XXX: H = J * sum (XX + YY + ZZ)."""
        terms = []
        for i in range(n - 1 + (1 if periodic else 0)):
            j = (i + 1) % n
            terms.append((J, {i: 'X', j: 'X'}))
            terms.append((J, {i: 'Y', j: 'Y'}))
            terms.append((J, {i: 'Z', j: 'Z'}))
        return cls(terms, n, "Heisenberg-XXX(n=%d,J=%.2f)" % (n, J))

    @classmethod
    def heisenberg_xxz(cls, n, Jxy=1.0, Jz=1.0, periodic=False, hz=0.0):
        """Heisenberg XXZ: H = Jxy*(XX+YY) + Jz*ZZ + hz*Z."""
        terms = []
        for i in range(n - 1 + (1 if periodic else 0)):
            j = (i + 1) % n
            terms.append((Jxy, {i: 'X', j: 'X'}))
            terms.append((Jxy, {i: 'Y', j: 'Y'}))
            terms.append((Jz, {i: 'Z', j: 'Z'}))
        if hz != 0:
            for i in range(n):
                terms.append((hz, {i: 'Z'}))
        return cls(terms, n,
                   "Heisenberg-XXZ(n=%d,Jxy=%.2f,Jz=%.2f)" % (n, Jxy, Jz))

    @classmethod
    def maxcut(cls, n, edges):
        """MaxCut Hamiltonian: C = sum w*(1 - ZZ)/2."""
        terms = []
        for i, j, w in edges:
            terms.append((float(w) / 2, {}))  # constante
            terms.append((-float(w) / 2, {int(i): 'Z', int(j): 'Z'}))
        return cls(terms, n, "MaxCut(n=%d,m=%d)" % (n, len(edges)))

    @classmethod
    def hubbard_1d(cls, n_sites, t=1.0, U=4.0, periodic=False, mu=0.0):
        """1D Fermi-Hubbard via Jordan-Wigner.

        H = -t sum_<ij,s> (c†_is c_js + h.c.) + U sum_i n_i↑ n_i↓ - mu sum n_i

        Qubit mapping: site i, spin s -> qubit 2*i + s (s=0: up, s=1: down).
        """
        n_qubits = 2 * n_sites
        terms = []

        # Hopping
        for i in range(n_sites - 1 + (1 if periodic else 0)):
            j = (i + 1) % n_sites
            for s in range(2):  # spin up, down
                qi, qj = 2 * i + s, 2 * j + s
                hop_terms = jordan_wigner_hopping(qi, qj, n_qubits)
                for c, p in hop_terms:
                    terms.append((-t * c, p))

        # On-site interaction: U * n_i↑ * n_i↓
        for i in range(n_sites):
            q_up, q_down = 2 * i, 2 * i + 1
            int_terms = jordan_wigner_interaction(q_up, q_down)
            for c, p in int_terms:
                terms.append((U * c, p))

        # Chemical potential: -mu * sum n_i
        if mu != 0:
            for i in range(n_sites):
                for s in range(2):
                    q = 2 * i + s
                    num_terms = jordan_wigner_number(q)
                    for c, p in num_terms:
                        terms.append((-mu * c, p))

        H = cls(terms, n_qubits,
                "Hubbard-1D(L=%d,t=%.2f,U=%.2f)" % (n_sites, t, U))
        H.simplify()
        return H

    @classmethod
    def molecular(cls, one_body, two_body, n_qubits=None):
        """Moleculaire Hamiltonian uit 1- en 2-electron integralen.

        Args:
            one_body: dict {(p,q): h_pq} of np.ndarray
            two_body: dict {(p,q,r,s): h_pqrs} of np.ndarray
                Chemist notation: (pq|rs) = int phi_p(1)phi_q(1) 1/r12 phi_r(2)phi_s(2)
            n_qubits: if None, inferred from indices

        Jordan-Wigner transformatie: fermionic -> qubit.
        """
        terms = []

        # Parse one-body
        if isinstance(one_body, np.ndarray):
            n = one_body.shape[0]
            ob_dict = {}
            for p in range(n):
                for q in range(n):
                    if abs(one_body[p, q]) > 1e-15:
                        ob_dict[(p, q)] = one_body[p, q]
            one_body = ob_dict
        else:
            n = max(max(p, q) for p, q in one_body.keys()) + 1 if one_body else 0

        # Parse two-body
        if isinstance(two_body, np.ndarray):
            n = max(n, two_body.shape[0])
            tb_dict = {}
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            if abs(two_body[p, q, r, s]) > 1e-15:
                                tb_dict[(p, q, r, s)] = two_body[p, q, r, s]
            two_body = tb_dict
        else:
            if two_body:
                n2 = max(max(idx) for idx in two_body.keys()) + 1
                n = max(n, n2)

        if n_qubits is None:
            n_qubits = n

        # One-body: sum h_pq c†_p c_q
        for (p, q), h_pq in one_body.items():
            if p == q:
                # n_p = (I - Z_p)/2
                num = jordan_wigner_number(p)
                for c, pauli in num:
                    terms.append((h_pq * c, pauli))
            else:
                hop = jordan_wigner_hopping(p, q, n_qubits)
                for c, pauli in hop:
                    terms.append((h_pq * c, pauli))

        # Two-body: 0.5 * sum h_pqrs c†_p c†_r c_s c_q
        for (p, q, r, s), h_pqrs in two_body.items():
            coeff = 0.5 * h_pqrs
            if abs(coeff) < 1e-15:
                continue
            # Chemist -> physicist: (pq|rs) -> <pr|qs>
            # c†_p c†_r c_s c_q via JW
            tw_terms = jordan_wigner_two_body(p, q, r, s, n_qubits)
            for c, pauli in tw_terms:
                terms.append((coeff * c, pauli))

        H = cls(terms, n_qubits, "Molecular(n=%d)" % n_qubits)
        H.simplify()
        return H

    @classmethod
    def custom(cls, n_qubits, pauli_strings):
        """Custom Hamiltonian uit lijst van (coeff, string) tuples.

        Args:
            pauli_strings: list van (coeff, str) waar str zoals "XZIY"
                I=identity, X/Y/Z=Pauli op die positie.
                Of (coeff, dict) zoals (1.0, {0:'X', 2:'Z'}).
        """
        terms = []
        for coeff, ps in pauli_strings:
            if isinstance(ps, str):
                pauli = {}
                for i, c in enumerate(ps):
                    if c in ('X', 'Y', 'Z'):
                        pauli[i] = c
                terms.append((complex(coeff), pauli))
            else:
                terms.append((complex(coeff), dict(ps)))
        return cls(terms, n_qubits, "Custom(n=%d)" % n_qubits)

    @classmethod
    def from_openfermion_str(cls, of_str, n_qubits=None):
        """Parse OpenFermion-achtige string representatie.

        Formaat: "0.5 [X0 Z1 Y2] + -0.3 [Z0 Z1]"
        """
        terms = []
        max_q = 0
        for part in of_str.split('+'):
            part = part.strip()
            if not part:
                continue
            # Split coeff en operators
            if '[' in part:
                coeff_str, ops_str = part.split('[')
                ops_str = ops_str.rstrip(']').strip()
            else:
                continue
            coeff = complex(coeff_str.strip())
            pauli = {}
            if ops_str:
                for op in ops_str.split():
                    if len(op) >= 2:
                        gate = op[0]
                        qubit = int(op[1:])
                        if gate in ('X', 'Y', 'Z'):
                            pauli[qubit] = gate
                            max_q = max(max_q, qubit)
            terms.append((coeff, pauli))

        if n_qubits is None:
            n_qubits = max_q + 1 if max_q >= 0 else 0
        return cls(terms, n_qubits, "OpenFermion(n=%d)" % n_qubits)

    # -----------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------

    def commuting_groups(self):
        """Verdeel termen in groepen van onderling commuterende Pauli strings.

        Twee Pauli strings commuteren als ze op een even aantal qubits
        verschillende niet-triviale Paulis hebben.

        Returns: list van lijsten van term-indices.
        """
        n = len(self.terms)
        if n == 0:
            return []

        def commutes(p1, p2):
            """Check of twee Pauli strings commuteren."""
            anti = 0
            all_qubits = set(p1.keys()) | set(p2.keys())
            for q in all_qubits:
                a = p1.get(q)
                b = p2.get(q)
                if a and b and a != b:
                    anti += 1
            return anti % 2 == 0

        # Greedy coloring
        groups = []
        assigned = [False] * n
        for i in range(n):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                # Check of j commuteert met alles in de groep
                ok = True
                for k in group:
                    if not commutes(self.terms[k][1], self.terms[j][1]):
                        ok = False
                        break
                if ok:
                    group.append(j)
                    assigned[j] = True
            groups.append(group)
        return groups

    def pauli_weight(self):
        """Gemiddeld gewicht (niet-triviale Paulis per term)."""
        if not self.terms:
            return 0.0
        return sum(len(p) for _, p in self.terms) / len(self.terms)

    def locality(self):
        """Maximale localiteit (max qubits per term)."""
        if not self.terms:
            return 0
        return max(len(p) for _, p in self.terms)

    def is_diagonal(self):
        """Check of H diagonaal is (alleen Z en I termen)."""
        for _, pauli in self.terms:
            for q, op in pauli.items():
                if op != 'Z':
                    return False
        return True

    # -----------------------------------------------------------------
    # COMPILATION
    # -----------------------------------------------------------------

    def qaoa(self, p, gammas, betas, mixer='X', n_qubits=None):
        """Compileer naar QAOA circuit.

        Args:
            p: QAOA diepte
            gammas: phase separator angles [p]
            betas: mixer angles [p]
            mixer: 'X' (standaard), 'XY' (particle-conserving), of
                   Hamiltonian object (custom mixer)
        """
        n = n_qubits or self.n_qubits
        qc = Circuit(n, name="QAOA-%s-p%d" % (self.name, p))
        qc.metadata['type'] = 'qaoa'
        qc.metadata['hamiltonian'] = self.name
        qc.metadata['p'] = p

        # |+> initialisatie (of custom voor XY mixer)
        if mixer == 'XY':
            # Half-filling: |0101...>
            for q in range(1, n, 2):
                qc.x(q)
            # Dan H op alle qubits voor superposition
            for q in range(n):
                qc.h(q)
        else:
            for q in range(n):
                qc.h(q)

        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Phase separator: exp(-i * gamma * H)
            self._compile_evolution(qc, gamma)

            # Mixer
            if mixer == 'X':
                for q in range(n):
                    qc.rx(q, 2 * beta)
            elif mixer == 'XY':
                # Particle-conserving XY mixer
                for q in range(n - 1):
                    qc.rxx(q, q + 1, 2 * beta)
                    qc.ryy(q, q + 1, 2 * beta)
            elif isinstance(mixer, Hamiltonian):
                mixer._compile_evolution(qc, beta)
            else:
                # String Pauli mixer
                for q in range(n):
                    qc.rx(q, 2 * beta)

        return qc

    def trotter(self, t, steps, order=1):
        """Trotter tijdsevolutie: exp(-iHt).

        Args:
            t: totale tijd
            steps: aantal Trotter stappen
            order: 1 (eerste orde), 2 (symmetrisch), 4 (Suzuki)
        """
        n = self.n_qubits
        qc = Circuit(n, name="Trotter%d-%s-t%.2f-s%d" % (order, self.name, t, steps))
        qc.metadata['type'] = 'trotter'
        qc.metadata['order'] = order
        qc.metadata['hamiltonian'] = self.name

        if order == 1:
            dt = t / steps
            for _ in range(steps):
                self._compile_evolution(qc, dt)
        elif order == 2:
            dt = t / steps
            self._compile_trotter2_step(qc, dt, steps)
        elif order == 4:
            dt = t / steps
            self._compile_trotter4_step(qc, dt, steps)
        else:
            raise ValueError("Trotter order %d niet ondersteund (1, 2, 4)" % order)

        return qc

    def trotter_grouped(self, t, steps, order=1):
        """Trotter met commuterende groepen (gereduceerde fout).

        Groepeert commuterende termen zodat ze exact tegelijk
        geevolueerd kunnen worden (geen Trotter-fout binnen groep).
        """
        groups = self.commuting_groups()
        n = self.n_qubits
        qc = Circuit(n, name="TrotterGrouped-%s-t%.2f-s%d" % (self.name, t, steps))
        qc.metadata['type'] = 'trotter_grouped'
        qc.metadata['n_groups'] = len(groups)

        dt = t / steps

        if order == 1:
            for _ in range(steps):
                for group in groups:
                    for idx in group:
                        coeff, pauli = self.terms[idx]
                        self._compile_single_term(qc, dt * coeff, pauli)
        elif order == 2:
            for step in range(steps):
                # Voorwaarts
                for group in groups:
                    for idx in group:
                        coeff, pauli = self.terms[idx]
                        self._compile_single_term(qc, 0.5 * dt * coeff, pauli)
                # Achterwaarts
                for group in reversed(groups):
                    for idx in group:
                        coeff, pauli = self.terms[idx]
                        self._compile_single_term(qc, 0.5 * dt * coeff, pauli)
        else:
            raise ValueError("Grouped Trotter order %d niet ondersteund" % order)

        return qc

    def vqe_ansatz(self, depth=3, params=None):
        """Hardware-efficient VQE ansatz met deze Hamiltonian als cost."""
        return Circuit.hardware_efficient(self.n_qubits, depth=depth, params=params)

    def _compile_evolution(self, qc, dt):
        """Compileer exp(-i * dt * H) op circuit qc."""
        for coeff, pauli in self.terms:
            if not pauli:
                continue  # constante term -> globale fase
            self._compile_single_term(qc, dt * coeff, pauli)

    def _compile_single_term(self, qc, angle, pauli):
        """Compileer exp(-i * angle * P) voor een enkele Pauli string P."""
        if not pauli:
            return

        qubits = sorted(pauli.keys())
        ops = [pauli[q] for q in qubits]

        # Speciale gevallen: efficienter dan generiek pad
        if len(qubits) == 1:
            q = qubits[0]
            if ops[0] == 'Z':
                qc.rz(q, 2 * angle)
            elif ops[0] == 'X':
                qc.rx(q, 2 * angle)
            elif ops[0] == 'Y':
                qc.ry(q, 2 * angle)
            return

        if len(qubits) == 2:
            q1, q2 = qubits
            if ops == ['Z', 'Z']:
                qc.rzz(q1, q2, 2 * angle)
                return
            if ops == ['X', 'X']:
                qc.rxx(q1, q2, 2 * angle)
                return
            if ops == ['Y', 'Y']:
                qc.ryy(q1, q2, 2 * angle)
                return

        # Generiek: basis change + CNOT cascade + RZ
        _append_pauli_rotation(qc, angle, pauli)

    def _compile_trotter2_step(self, qc, dt, steps):
        """Tweede-orde symmetrische Trotter: S2(dt) = prod_k e^{-i c_k P_k dt/2}
           * prod_k(reversed) e^{-i c_k P_k dt/2}."""
        for _ in range(steps):
            # Voorwaarts halve stap
            for coeff, pauli in self.terms:
                if pauli:
                    self._compile_single_term(qc, 0.5 * dt * coeff, pauli)
            # Achterwaarts halve stap
            for coeff, pauli in reversed(self.terms):
                if pauli:
                    self._compile_single_term(qc, 0.5 * dt * coeff, pauli)

    def _compile_trotter4_step(self, qc, dt, steps):
        """Vierde-orde Suzuki: S4(dt) = S2(p*dt)^2 * S2((1-4p)*dt) * S2(p*dt)^2.
        Met p = 1/(4 - 4^(1/3))."""
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        for _ in range(steps):
            # S2(p*dt) twee keer
            self._compile_trotter2_step(qc, p * dt, 1)
            self._compile_trotter2_step(qc, p * dt, 1)
            # S2((1-4p)*dt) een keer
            self._compile_trotter2_step(qc, (1.0 - 4.0 * p) * dt, 1)
            # S2(p*dt) twee keer
            self._compile_trotter2_step(qc, p * dt, 1)
            self._compile_trotter2_step(qc, p * dt, 1)


# =====================================================================
# JORDAN-WIGNER TRANSFORMATIE
# =====================================================================

def jordan_wigner_number(site):
    """Getaloperator n_p = c†_p c_p = (I - Z_p) / 2.

    Returns: list van (coeff, pauli_dict) tuples.
    """
    return [
        (0.5, {}),          # I/2
        (-0.5, {site: 'Z'}) # -Z_p/2
    ]


def jordan_wigner_hopping(p, q, n_qubits):
    """Hopping term c†_p c_q + c†_q c_p (reeel).

    Jordan-Wigner: c†_p = (X_p - iY_p)/2 * prod_{k<p} Z_k

    Voor p < q:
    c†_p c_q + h.c. = 0.5 * (X_p Z_{p+1}...Z_{q-1} X_q + Y_p Z_{p+1}...Z_{q-1} Y_q)

    Returns: list van (coeff, pauli_dict) tuples.
    """
    if p == q:
        return jordan_wigner_number(p)

    if p > q:
        p, q = q, p

    # Z-string tussen p en q
    terms = []

    # XX term
    pauli_xx = {p: 'X', q: 'X'}
    for k in range(p + 1, q):
        pauli_xx[k] = 'Z'
    terms.append((0.5, pauli_xx))

    # YY term
    pauli_yy = {p: 'Y', q: 'Y'}
    for k in range(p + 1, q):
        pauli_yy[k] = 'Z'
    terms.append((0.5, pauli_yy))

    return terms


def jordan_wigner_interaction(p, q):
    """Interactie n_p * n_q.

    n_p * n_q = (I - Z_p)(I - Z_q)/4
             = (I - Z_p - Z_q + Z_p Z_q) / 4

    Returns: list van (coeff, pauli_dict) tuples.
    """
    return [
        (0.25, {}),                      # I/4
        (-0.25, {p: 'Z'}),              # -Z_p/4
        (-0.25, {q: 'Z'}),              # -Z_q/4
        (0.25, {p: 'Z', q: 'Z'}),       # Z_p Z_q/4
    ]


def jordan_wigner_two_body(p, q, r, s, n_qubits):
    """Twee-deeltjes term c†_p c_q c†_r c_s via Jordan-Wigner.

    Simplified: als p=r en q=s -> n_p * n_q (al gedaan).
    Generiek: decompositie via ladder operatoren.

    Voor moleculaire Hamiltonianen zijn de meeste termen van de vorm:
    c†_p c†_r c_s c_q = c†_p c_q * c†_r c_s - delta_{qr} c†_p c_s

    Returns: list van (coeff, pauli_dict) tuples.
    """
    # Speciale gevallen
    if p == q and r == s:
        # n_p * n_r
        return jordan_wigner_interaction(p, r)

    if p == q:
        # n_p * c†_r c_s
        n_terms = jordan_wigner_number(p)
        hop_terms = jordan_wigner_hopping(r, s, n_qubits)
        return _multiply_pauli_terms(n_terms, hop_terms)

    if r == s:
        # c†_p c_q * n_r
        hop_terms = jordan_wigner_hopping(p, q, n_qubits)
        n_terms = jordan_wigner_number(r)
        return _multiply_pauli_terms(hop_terms, n_terms)

    # Generiek: c†_p c_q c†_r c_s
    # = c†_p c_q * c†_r c_s - delta_{q,r} * c†_p c_s
    hop_pq = jordan_wigner_hopping(p, q, n_qubits)
    hop_rs = jordan_wigner_hopping(r, s, n_qubits)
    terms = _multiply_pauli_terms(hop_pq, hop_rs)

    if q == r:
        hop_ps = jordan_wigner_hopping(p, s, n_qubits)
        for c, pauli in hop_ps:
            terms.append((-c, pauli))

    return terms


def _multiply_pauli_terms(terms_a, terms_b):
    """Vermenigvuldig twee Pauli-som representaties."""
    result = []
    for ca, pa in terms_a:
        for cb, pb in terms_b:
            coeff, pauli = _multiply_single_paulis(ca * cb, pa, pb)
            if abs(coeff) > 1e-15:
                result.append((coeff, pauli))
    return result


def _multiply_single_paulis(coeff, p1, p2):
    """Vermenigvuldig twee enkele Pauli strings.

    P1 * P2 = phase * P3 volgens Pauli algebra:
    XX = I, XY = iZ, XZ = -iY, etc.
    """
    phase = 1.0
    result = {}

    all_qubits = set(p1.keys()) | set(p2.keys())
    for q in all_qubits:
        a = p1.get(q)
        b = p2.get(q)

        if a and b:
            if a == b:
                # XX = YY = ZZ = I
                pass  # identity, nothing added
            else:
                # Levi-Civita: XY=iZ, YZ=iX, ZX=iY
                pauli_order = {'X': 0, 'Y': 1, 'Z': 2}
                ia, ib = pauli_order[a], pauli_order[b]
                if (ib - ia) % 3 == 1:
                    phase *= 1j
                else:
                    phase *= -1j
                # Resultaat Pauli
                ic = 3 - ia - ib  # remaining index
                result[q] = ['X', 'Y', 'Z'][ic]
        elif a:
            result[q] = a
        elif b:
            result[q] = b

    return coeff * phase, result


# =====================================================================
# GATE OPTIMALISATIE
# =====================================================================

class CircuitOptimizer:
    """Post-compilatie gate optimalisaties."""

    @staticmethod
    def merge_rotations(circuit):
        """Merge opeenvolgende rotaties op dezelfde qubit.

        RZ(a) RZ(b) -> RZ(a+b), RX(a) RX(b) -> RX(a+b), etc.
        """
        merged = Circuit(circuit.n_qubits, circuit.name + "-opt")
        merged.metadata = dict(circuit.metadata)

        i = 0
        while i < len(circuit.ops):
            op = circuit.ops[i]

            # Check of volgende op dezelfde qubit en zelfde type is
            if i + 1 < len(circuit.ops):
                next_op = circuit.ops[i + 1]
                if (op.name == next_op.name and
                    op.qubits == next_op.qubits and
                    op.name in ('RX', 'RY', 'RZ', 'RZZ', 'RXX', 'RYY') and
                    op.params and next_op.params):
                    # Merge
                    new_angle = op.params[0] + next_op.params[0]
                    if abs(new_angle) > 1e-12:
                        _add_rotation(merged, op.name, op.qubits, new_angle)
                    i += 2
                    continue

            merged.ops.append(op)
            i += 1

        return merged

    @staticmethod
    def cancel_inverses(circuit):
        """Verwijder opeenvolgende inverse paren: H H, X X, etc."""
        self_inverse = {'H', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'SWAP'}
        result = Circuit(circuit.n_qubits, circuit.name + "-opt")
        result.metadata = dict(circuit.metadata)

        i = 0
        while i < len(circuit.ops):
            op = circuit.ops[i]
            if (i + 1 < len(circuit.ops) and
                op.name in self_inverse and
                circuit.ops[i + 1].name == op.name and
                circuit.ops[i + 1].qubits == op.qubits):
                i += 2
                continue
            result.ops.append(op)
            i += 1

        return result

    @staticmethod
    def remove_small_rotations(circuit, threshold=1e-10):
        """Verwijder rotaties met verwaarloosbaar kleine hoeken."""
        result = Circuit(circuit.n_qubits, circuit.name)
        result.metadata = dict(circuit.metadata)

        for op in circuit.ops:
            if op.name in ('RX', 'RY', 'RZ', 'RZZ', 'RXX', 'RYY'):
                if op.params and abs(op.params[0]) < threshold:
                    continue
            result.ops.append(op)

        return result

    @staticmethod
    def optimize(circuit, passes=3):
        """Voer alle optimalisaties uit, meerdere passes."""
        qc = circuit
        for _ in range(passes):
            n_before = len(qc)
            qc = CircuitOptimizer.remove_small_rotations(qc)
            qc = CircuitOptimizer.merge_rotations(qc)
            qc = CircuitOptimizer.cancel_inverses(qc)
            if len(qc) == n_before:
                break
        return qc


def _add_rotation(circuit, name, qubits, angle):
    """Helper: voeg rotatie gate toe aan circuit."""
    if name == 'RX':
        circuit.rx(qubits[0], angle)
    elif name == 'RY':
        circuit.ry(qubits[0], angle)
    elif name == 'RZ':
        circuit.rz(qubits[0], angle)
    elif name == 'RZZ':
        circuit.rzz(qubits[0], qubits[1], angle)
    elif name == 'RXX':
        circuit.rxx(qubits[0], qubits[1], angle)
    elif name == 'RYY':
        circuit.ryy(qubits[0], qubits[1], angle)


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def compile_hamiltonian(hamiltonian, mode='trotter', **kwargs):
    """One-liner Hamiltonian -> Circuit.

    Args:
        hamiltonian: Hamiltonian object
        mode: 'trotter' (default), 'trotter2', 'trotter4',
              'trotter_grouped', 'qaoa', 'vqe'
        **kwargs: passed to compilation method
    """
    if mode == 'trotter':
        return hamiltonian.trotter(order=1, **kwargs)
    elif mode == 'trotter2':
        return hamiltonian.trotter(order=2, **kwargs)
    elif mode == 'trotter4':
        return hamiltonian.trotter(order=4, **kwargs)
    elif mode == 'trotter_grouped':
        return hamiltonian.trotter_grouped(**kwargs)
    elif mode == 'qaoa':
        return hamiltonian.qaoa(**kwargs)
    elif mode == 'vqe':
        return hamiltonian.vqe_ansatz(**kwargs)
    else:
        raise ValueError("Onbekende compilatie mode: %s" % mode)
