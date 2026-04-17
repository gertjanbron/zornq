#!/usr/bin/env python3
"""
circuit_interface.py - B128 Probleem-Agnostische Circuit Interface

Universele gate-set interface voor ZornQ. Ontkoppelt circuitspecificatie
van backend (MPS, state vector, lightcone). Ondersteunt:
- Willekeurige 1-qubit en 2-qubit gates
- Parametrische gates (RX, RY, RZ, RZZ, CNOT, etc.)
- Circuit als gate-sequence (append-based)
- Hamiltonian -> circuit compilatie (Trotter)
- Observable meting (lokaal, correlatie, energy)
- Backend auto-selectie (state vector vs MPS)

Gebruik:
    from circuit_interface import Circuit, Gates, run_circuit

    # Bouw circuit
    qc = Circuit(n_qubits=4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0, theta=0.5)
    qc.rzz(0, 1, gamma=0.3)

    # QAOA shortcut
    qc = Circuit.qaoa_maxcut(edges, p=1, gammas=[0.5], betas=[0.3])

    # VQE shortcut
    qc = Circuit.hardware_efficient(n_qubits=6, depth=3, params=theta)

    # Draai op beste backend
    result = run_circuit(qc, observables=[('ZZ', 0, 1), ('Z', 2)])

    # Of direct met MPS
    result = run_circuit(qc, backend='mps', max_chi=64)

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
import time


# =====================================================================
# GATE LIBRARY
# =====================================================================

class Gates:
    """Standaard gate library. Alle gates als numpy arrays."""

    @staticmethod
    def I():
        return np.eye(2, dtype=np.complex128)

    @staticmethod
    def X():
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    @staticmethod
    def Y():
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    @staticmethod
    def Z():
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @staticmethod
    def H():
        s = 1.0 / np.sqrt(2)
        return np.array([[s, s], [s, -s]], dtype=np.complex128)

    @staticmethod
    def S():
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    @staticmethod
    def T():
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

    @staticmethod
    def RX(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def RY(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def RZ(theta):
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)

    @staticmethod
    def CNOT():
        g = np.zeros((4, 4), dtype=np.complex128)
        g[0, 0] = g[1, 1] = g[2, 3] = g[3, 2] = 1
        return g

    @staticmethod
    def CZ():
        g = np.eye(4, dtype=np.complex128)
        g[3, 3] = -1
        return g

    @staticmethod
    def SWAP():
        g = np.zeros((4, 4), dtype=np.complex128)
        g[0, 0] = g[1, 2] = g[2, 1] = g[3, 3] = 1
        return g

    @staticmethod
    def RZZ(theta):
        """exp(-i * theta * Z_i Z_j / 2). Diagonaal."""
        d = np.array([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2)
        ], dtype=np.complex128)
        return np.diag(d)

    @staticmethod
    def RZZ_diag(theta):
        """Diagonaalvector van RZZ gate (voor geoptimaliseerd pad)."""
        return np.array([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2)
        ], dtype=np.complex128)

    @staticmethod
    def RXX(theta):
        """exp(-i * theta * X_i X_j / 2)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        g = np.zeros((4, 4), dtype=np.complex128)
        g[0, 0] = g[1, 1] = g[2, 2] = g[3, 3] = c
        g[0, 3] = g[3, 0] = -1j * s
        g[1, 2] = g[2, 1] = -1j * s
        return g

    @staticmethod
    def RYY(theta):
        """exp(-i * theta * Y_i Y_j / 2)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        g = np.zeros((4, 4), dtype=np.complex128)
        g[0, 0] = g[1, 1] = g[2, 2] = g[3, 3] = c
        g[0, 3] = g[3, 0] = 1j * s
        g[1, 2] = g[2, 1] = -1j * s
        return g

    @staticmethod
    def XXplusYY(theta, phi=0.0):
        """(XX+YY)/2 interactie gate, gebruikt in VQE."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        ep = np.exp(1j * phi)
        em = np.exp(-1j * phi)
        g = np.eye(4, dtype=np.complex128)
        g[1, 1] = c
        g[2, 2] = c
        g[1, 2] = -1j * s * em
        g[2, 1] = -1j * s * ep
        return g

    @staticmethod
    def custom_1q(matrix):
        """Willekeurige 1-qubit gate (2x2 matrix)."""
        return np.array(matrix, dtype=np.complex128).reshape(2, 2)

    @staticmethod
    def custom_2q(matrix):
        """Willekeurige 2-qubit gate (4x4 matrix)."""
        return np.array(matrix, dtype=np.complex128).reshape(4, 4)


# =====================================================================
# GATE INSTRUCTION
# =====================================================================

@dataclass
class GateOp:
    """Een gate-instructie in een circuit."""
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    matrix: Optional[np.ndarray] = None
    is_diagonal: bool = False


# =====================================================================
# CIRCUIT
# =====================================================================

class Circuit:
    """Universele circuit-specificatie.

    Append-based: bouw het circuit op door gates toe te voegen.
    Ondersteunt parametrische gates, barriers, en metadata.
    """

    def __init__(self, n_qubits: int, name: str = "circuit"):
        self.n_qubits = n_qubits
        self.name = name
        self.ops: List[GateOp] = []
        self.metadata: Dict = {}

    def __len__(self):
        return len(self.ops)

    def depth(self):
        """Schat circuit depth (simpele laag-telling)."""
        if not self.ops:
            return 0
        layers = [0] * self.n_qubits
        for op in self.ops:
            layer = max(layers[q] for q in op.qubits) + 1
            for q in op.qubits:
                layers[q] = layer
        return max(layers)

    # --- 1-qubit gates ---

    def h(self, qubit):
        self.ops.append(GateOp('H', (qubit,), matrix=Gates.H()))
        return self

    def x(self, qubit):
        self.ops.append(GateOp('X', (qubit,), matrix=Gates.X()))
        return self

    def y(self, qubit):
        self.ops.append(GateOp('Y', (qubit,), matrix=Gates.Y()))
        return self

    def z(self, qubit):
        self.ops.append(GateOp('Z', (qubit,), matrix=Gates.Z()))
        return self

    def s(self, qubit):
        self.ops.append(GateOp('S', (qubit,), matrix=Gates.S()))
        return self

    def t(self, qubit):
        self.ops.append(GateOp('T', (qubit,), matrix=Gates.T()))
        return self

    def rx(self, qubit, theta):
        self.ops.append(GateOp('RX', (qubit,), (theta,), matrix=Gates.RX(theta)))
        return self

    def ry(self, qubit, theta):
        self.ops.append(GateOp('RY', (qubit,), (theta,), matrix=Gates.RY(theta)))
        return self

    def rz(self, qubit, theta):
        self.ops.append(GateOp('RZ', (qubit,), (theta,),
                                matrix=Gates.RZ(theta), is_diagonal=True))
        return self

    # --- 2-qubit gates ---

    def cx(self, control, target):
        self.ops.append(GateOp('CNOT', (control, target), matrix=Gates.CNOT()))
        return self

    def cz(self, q1, q2):
        self.ops.append(GateOp('CZ', (q1, q2), matrix=Gates.CZ(), is_diagonal=True))
        return self

    def swap(self, q1, q2):
        self.ops.append(GateOp('SWAP', (q1, q2), matrix=Gates.SWAP()))
        return self

    def rzz(self, q1, q2, theta):
        self.ops.append(GateOp('RZZ', (q1, q2), (theta,),
                                matrix=Gates.RZZ(theta), is_diagonal=True))
        return self

    def rxx(self, q1, q2, theta):
        self.ops.append(GateOp('RXX', (q1, q2), (theta,), matrix=Gates.RXX(theta)))
        return self

    def ryy(self, q1, q2, theta):
        self.ops.append(GateOp('RYY', (q1, q2), (theta,), matrix=Gates.RYY(theta)))
        return self

    # --- Custom gates ---

    def apply_1q(self, qubit, matrix, name="U1"):
        mat = np.array(matrix, dtype=np.complex128).reshape(2, 2)
        self.ops.append(GateOp(name, (qubit,), matrix=mat))
        return self

    def apply_2q(self, q1, q2, matrix, name="U2", diagonal=False):
        mat = np.array(matrix, dtype=np.complex128).reshape(4, 4)
        self.ops.append(GateOp(name, (q1, q2), matrix=mat, is_diagonal=diagonal))
        return self

    # --- Circuit constructors ---

    @classmethod
    def qaoa_maxcut(cls, n_qubits, edges, p, gammas, betas):
        """QAOA MaxCut circuit.

        Args:
            n_qubits: aantal qubits
            edges: list van (i, j, w) tuples
            p: QAOA diepte
            gammas: phase angles per laag
            betas: mixer angles per laag
        """
        qc = cls(n_qubits, name="QAOA-MaxCut-p%d" % p)
        qc.metadata['type'] = 'qaoa'
        qc.metadata['p'] = p
        qc.metadata['edges'] = edges

        # |+> initialisatie
        for q in range(n_qubits):
            qc.h(q)

        # p lagen
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Phase separator: exp(-i * gamma * w * Z_i Z_j) per edge
            for i, j, w in edges:
                qc.rzz(int(i), int(j), 2 * gamma * float(w))

            # Mixer: exp(-i * beta * X_q) per qubit
            for q in range(n_qubits):
                qc.rx(q, 2 * beta)

        return qc

    @classmethod
    def qaoa_from_hamiltonian(cls, n_qubits, hamiltonian_terms, p, gammas, betas):
        """QAOA met willekeurige Hamiltonian.

        Args:
            hamiltonian_terms: list van (coeff, pauli_string) tuples
                pauli_string: dict {qubit: 'X'|'Y'|'Z'}
                Voorbeeld: [(1.0, {0:'Z', 1:'Z'}), (0.5, {0:'X'})]
        """
        qc = cls(n_qubits, name="QAOA-Ham-p%d" % p)
        qc.metadata['type'] = 'qaoa_hamiltonian'

        for q in range(n_qubits):
            qc.h(q)

        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Phase separator via Trotter
            for coeff, pauli in hamiltonian_terms:
                qubits_involved = sorted(pauli.keys())
                ops = [pauli[q] for q in qubits_involved]

                if len(qubits_involved) == 1 and ops[0] == 'Z':
                    q = qubits_involved[0]
                    qc.rz(q, 2 * gamma * coeff)
                elif len(qubits_involved) == 2 and ops == ['Z', 'Z']:
                    q1, q2 = qubits_involved
                    qc.rzz(q1, q2, 2 * gamma * coeff)
                elif len(qubits_involved) == 2 and ops == ['X', 'X']:
                    q1, q2 = qubits_involved
                    qc.rxx(q1, q2, 2 * gamma * coeff)
                elif len(qubits_involved) == 2 and ops == ['Y', 'Y']:
                    q1, q2 = qubits_involved
                    qc.ryy(q1, q2, 2 * gamma * coeff)
                else:
                    # Generiek: basis rotatie + diagonaal + terug
                    _append_pauli_rotation(qc, gamma * coeff, pauli)

            # Mixer
            for q in range(n_qubits):
                qc.rx(q, 2 * beta)

        return qc

    @classmethod
    def hardware_efficient(cls, n_qubits, depth, params=None):
        """Hardware-efficient VQE ansatz.

        Afwisselend RY+RZ per qubit en CNOT ladder.
        params: flat array van parameters (of random als None).
        """
        n_params = n_qubits * 2 * (depth + 1) + (n_qubits - 1) * depth
        if params is None:
            params = np.random.uniform(0, 2 * np.pi, n_params)
        params = np.asarray(params, dtype=float)

        qc = cls(n_qubits, name="HEA-d%d" % depth)
        qc.metadata['type'] = 'vqe'
        qc.metadata['n_params'] = n_params
        idx = 0

        # Initieel: RY + RZ per qubit
        for q in range(n_qubits):
            qc.ry(q, params[idx]); idx += 1
            qc.rz(q, params[idx]); idx += 1

        for d in range(depth):
            # CNOT ladder
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            # RY + RZ per qubit
            for q in range(n_qubits):
                qc.ry(q, params[idx]); idx += 1
                qc.rz(q, params[idx]); idx += 1

        return qc

    @classmethod
    def trotter_evolution(cls, n_qubits, hamiltonian_terms, t, n_steps):
        """Eerste-orde Trotter tijdsevolutie: exp(-iHt).

        Args:
            hamiltonian_terms: list van (coeff, pauli_string) tuples
            t: totale tijd
            n_steps: aantal Trotter stappen
        """
        dt = t / n_steps
        qc = cls(n_qubits, name="Trotter-t%.2f-s%d" % (t, n_steps))
        qc.metadata['type'] = 'trotter'

        for step in range(n_steps):
            for coeff, pauli in hamiltonian_terms:
                qubits_involved = sorted(pauli.keys())
                ops = [pauli[q] for q in qubits_involved]

                if len(qubits_involved) == 1 and ops[0] == 'Z':
                    qc.rz(qubits_involved[0], 2 * dt * coeff)
                elif len(qubits_involved) == 1 and ops[0] == 'X':
                    qc.rx(qubits_involved[0], 2 * dt * coeff)
                elif len(qubits_involved) == 2 and ops == ['Z', 'Z']:
                    qc.rzz(qubits_involved[0], qubits_involved[1], 2 * dt * coeff)
                elif len(qubits_involved) == 2 and ops == ['X', 'X']:
                    qc.rxx(qubits_involved[0], qubits_involved[1], 2 * dt * coeff)
                else:
                    _append_pauli_rotation(qc, dt * coeff, pauli)

        return qc

    def summary(self):
        """Kort overzicht van het circuit."""
        gate_counts = {}
        for op in self.ops:
            gate_counts[op.name] = gate_counts.get(op.name, 0) + 1
        return {
            'name': self.name,
            'n_qubits': self.n_qubits,
            'n_gates': len(self.ops),
            'depth': self.depth(),
            'gate_counts': gate_counts,
            'metadata': self.metadata,
        }


def _append_pauli_rotation(qc, angle, pauli):
    """Generieke Pauli-string rotatie: exp(-i * angle * P).

    Gebruikt basis-transformaties naar Z-basis, dan multi-qubit Z rotatie.
    """
    qubits = sorted(pauli.keys())

    # Basis transformatie naar Z
    for q in qubits:
        p = pauli[q]
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.apply_1q(q, Gates.RX(np.pi / 2).conj().T, "Sdg")

    # CNOT cascade
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])

    # RZ op laatste qubit
    qc.rz(qubits[-1], 2 * angle)

    # Undo CNOT cascade
    for i in range(len(qubits) - 2, -1, -1):
        qc.cx(qubits[i], qubits[i + 1])

    # Undo basis transformatie
    for q in qubits:
        p = pauli[q]
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.apply_1q(q, Gates.RX(np.pi / 2), "S")


# =====================================================================
# OBSERVABLES
# =====================================================================

@dataclass
class Observable:
    """Een observable om te meten."""
    terms: List[Tuple[float, Dict[int, str]]]  # [(coeff, {qubit: pauli}), ...]

    @classmethod
    def z(cls, qubit):
        return cls([(1.0, {qubit: 'Z'})])

    @classmethod
    def zz(cls, q1, q2):
        return cls([(1.0, {q1: 'Z', q2: 'Z'})])

    @classmethod
    def xx(cls, q1, q2):
        return cls([(1.0, {q1: 'X', q2: 'X'})])

    @classmethod
    def maxcut_cost(cls, edges):
        """MaxCut cost Hamiltonian: C = sum_e w_e (1 - Z_i Z_j) / 2."""
        terms = []
        for i, j, w in edges:
            terms.append((float(w) / 2, {}))  # constante
            terms.append((-float(w) / 2, {int(i): 'Z', int(j): 'Z'}))
        return cls(terms)

    @classmethod
    def heisenberg(cls, n_qubits, J=1.0, periodic=False):
        """Heisenberg XXX model: H = J * sum (XX + YY + ZZ)."""
        terms = []
        for i in range(n_qubits - 1 + (1 if periodic else 0)):
            j = (i + 1) % n_qubits
            terms.append((J, {i: 'X', j: 'X'}))
            terms.append((J, {i: 'Y', j: 'Y'}))
            terms.append((J, {i: 'Z', j: 'Z'}))
        return cls(terms)

    @classmethod
    def ising_transverse(cls, n_qubits, J=1.0, h=0.5, periodic=False):
        """Transverse-field Ising: H = -J * sum ZZ - h * sum X."""
        terms = []
        for i in range(n_qubits - 1 + (1 if periodic else 0)):
            j = (i + 1) % n_qubits
            terms.append((-J, {i: 'Z', j: 'Z'}))
        for i in range(n_qubits):
            terms.append((-h, {i: 'X'}))
        return cls(terms)


# =====================================================================
# BACKEND: STATE VECTOR
# =====================================================================

def _run_statevector(circuit, observables=None):
    """Exact state vector simulatie. Max ~22-24 qubits."""
    n = circuit.n_qubits
    if n > 26:
        raise ValueError("State vector te groot: n=%d (max ~26)" % n)

    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0...0>

    for op in circuit.ops:
        qubits = op.qubits
        mat = op.matrix

        if len(qubits) == 1:
            q = qubits[0]
            _apply_1q_sv(state, mat, q, n)
        elif len(qubits) == 2:
            q1, q2 = qubits
            _apply_2q_sv(state, mat, q1, q2, n)

    # Meet observables
    results = {}
    if observables:
        for name, obs in observables.items():
            val = _measure_observable_sv(state, obs, n)
            results[name] = val

    return state, results


def _apply_1q_sv(state, gate, qubit, n_qubits):
    """Pas 1-qubit gate toe op state vector (in-place)."""
    dim = 1 << n_qubits
    step = 1 << qubit
    for i in range(0, dim, 2 * step):
        for j in range(step):
            idx0 = i + j
            idx1 = idx0 + step
            a, b = state[idx0], state[idx1]
            state[idx0] = gate[0, 0] * a + gate[0, 1] * b
            state[idx1] = gate[1, 0] * a + gate[1, 1] * b


def _apply_2q_sv(state, gate, q1, q2, n_qubits):
    """Pas 2-qubit gate toe op state vector (in-place).

    Gate matrix in standaard Kronecker volgorde: |q1, q2> met
    index 0=|00>, 1=|01>(q2=1), 2=|10>(q1=1), 3=|11>.
    """
    if q1 > q2:
        q1, q2 = q2, q1
        # Swap qubit order in gate
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

    dim = 1 << n_qubits
    step1 = 1 << q1
    step2 = 1 << q2

    for i in range(dim):
        if (i >> q1) & 1 == 0 and (i >> q2) & 1 == 0:
            # Matrix index: 0=|00>, 1=|01>(q2 set), 2=|10>(q1 set), 3=|11>
            idx00 = i
            idx01 = i | step2          # q2=1
            idx10 = i | step1          # q1=1
            idx11 = i | step1 | step2  # both=1
            v = np.array([state[idx00], state[idx01], state[idx10], state[idx11]])
            w = gate @ v
            state[idx00] = w[0]
            state[idx01] = w[1]
            state[idx10] = w[2]
            state[idx11] = w[3]


def _measure_observable_sv(state, observable, n_qubits):
    """Meet een observable op een state vector."""
    total = 0.0
    for coeff, pauli in observable.terms:
        if not pauli:
            total += coeff
            continue
        # Bereken <psi| P |psi> voor Pauli string P
        psi_p = state.copy()
        for q, op in pauli.items():
            if op == 'Z':
                step = 1 << q
                dim = 1 << n_qubits
                for i in range(dim):
                    if (i >> q) & 1:
                        psi_p[i] *= -1
            elif op == 'X':
                _apply_1q_sv(psi_p, Gates.X(), q, n_qubits)
            elif op == 'Y':
                _apply_1q_sv(psi_p, Gates.Y(), q, n_qubits)
        total += coeff * np.real(np.vdot(state, psi_p))
    return float(total)


# =====================================================================
# BACKEND: MPS (via ZornMPS)
# =====================================================================

def _run_mps(circuit, observables=None, max_chi=64, gpu=False):
    """MPS backend via ZornMPS. Schaalt naar duizenden qubits."""
    from zorn_mps import ZornMPS

    n = circuit.n_qubits
    mps = ZornMPS(n_sites=n, d=2, max_chi=max_chi, mode='schrodinger', gpu=gpu)

    # Initialiseer |0...0> productstate
    basis_0 = np.zeros((n, 2), dtype=np.complex128)
    basis_0[:, 0] = 1.0
    mps.init_product_state(local_states=basis_0)

    for op in circuit.ops:
        qubits = op.qubits
        mat = op.matrix

        if len(qubits) == 1:
            mps.apply_1site_gate(mat, qubits[0])
        elif len(qubits) == 2:
            q1, q2 = qubits
            if abs(q1 - q2) == 1:
                site = min(q1, q2)
                if q1 > q2:
                    mat = mat.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
                if op.is_diagonal:
                    diag = np.diag(mat)
                    if q1 > q2:
                        # Herordenen diagonaal: |01> <-> |10>
                        diag = np.array([diag[0], diag[2], diag[1], diag[3]])
                    mps.apply_2site_diag(diag, site)
                else:
                    mps.apply_2site_gate(mat, site)
            else:
                # Non-adjacent qubits: SWAP route
                _apply_nonadjacent_2q(mps, mat, q1, q2, op.is_diagonal)

    results = {}
    if observables:
        for name, obs in observables.items():
            val = _measure_observable_mps(mps, obs)
            results[name] = val

    return mps, results


def _apply_nonadjacent_2q(mps, gate, q1, q2, is_diagonal):
    """2-qubit gate op niet-aangrenzende qubits via SWAP routing."""
    swap = Gates.SWAP()
    if q1 > q2:
        q1, q2 = q2, q1
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

    # SWAP q1 naar q2-1
    for i in range(q1, q2 - 1):
        mps.apply_2site_gate(swap, i)

    # Pas gate toe op (q2-1, q2)
    if is_diagonal:
        mps.apply_2site_diag(np.diag(gate), q2 - 1)
    else:
        mps.apply_2site_gate(gate, q2 - 1)

    # SWAP terug
    for i in range(q2 - 2, q1 - 1, -1):
        mps.apply_2site_gate(swap, i)


def _measure_observable_mps(mps, observable):
    """Meet observable op MPS state."""
    total = 0.0
    for coeff, pauli in observable.terms:
        if not pauli:
            total += coeff
            continue

        qubits = sorted(pauli.keys())
        if len(qubits) == 1:
            q = qubits[0]
            op_name = pauli[q]
            if op_name == 'Z':
                obs = Gates.Z()
            elif op_name == 'X':
                obs = Gates.X()
            elif op_name == 'Y':
                obs = Gates.Y()
            else:
                obs = Gates.I()
            val = mps.expectation_local(obs, q)
            total += coeff * np.real(val)
        elif len(qubits) == 2:
            # 2-punt correlatie via contractie
            q1, q2 = qubits
            op1 = _pauli_matrix(pauli[q1])
            op2 = _pauli_matrix(pauli[q2])
            val = _mps_2point_correlator(mps, op1, q1, op2, q2)
            total += coeff * np.real(val)
        else:
            # Generiek: TODO multi-point correlator
            raise NotImplementedError("Multi-point correlators (>2) niet geimplementeerd")
    return float(total)


def _pauli_matrix(name):
    if name == 'X': return Gates.X()
    if name == 'Y': return Gates.Y()
    if name == 'Z': return Gates.Z()
    return Gates.I()


def _mps_2point_correlator(mps, op1, q1, op2, q2):
    """<psi| op1(q1) op2(q2) |psi> via MPS transfer-matrix contractie.

    C[alpha', alpha] is de left-environment: bra-bond x ket-bond.
    Per site contracteren we de fysieke index (met evt operator) en
    propageren de bond-indices naar rechts.

    Correct einsum: 'pq,pir,qis->rs' (p,q=bonds links; i=fysiek; r,s=bonds rechts)
    Bij operator-site: T_op = O @ T op de fysieke index.
    """
    if q1 > q2:
        q1, q2, op1, op2 = q2, q1, op2, op1

    n = mps.n_sites

    # Initialiseer C vanuit site 0
    T = mps.nodes[0].tensor  # (chi_L=1, d, chi_R)
    if q1 == 0:
        T_op = np.einsum('ij,ajb->aib', op1, T)
        C = np.einsum('aib,aic->bc', T.conj(), T_op)
    elif q2 == 0:
        T_op = np.einsum('ij,ajb->aib', op2, T)
        C = np.einsum('aib,aic->bc', T.conj(), T_op)
    else:
        C = np.einsum('aib,aic->bc', T.conj(), T)

    for s in range(1, n):
        T = mps.nodes[s].tensor  # (chi_L, d, chi_R)
        if s == q1:
            T_op = np.einsum('ij,ajb->aib', op1, T)
            C = np.einsum('pq,pir,qis->rs', C, T.conj(), T_op)
        elif s == q2:
            T_op = np.einsum('ij,ajb->aib', op2, T)
            C = np.einsum('pq,pir,qis->rs', C, T.conj(), T_op)
        else:
            C = np.einsum('pq,pir,qis->rs', C, T.conj(), T)

    return np.trace(C)


# =====================================================================
# UNIFIED RUN INTERFACE
# =====================================================================

def run_circuit(circuit, observables=None, backend='auto',
                max_chi=64, gpu=False, verbose=False):
    """Draai een circuit op de beste backend.

    Args:
        circuit: Circuit object
        observables: dict {naam: Observable} of None
        backend: 'auto', 'statevector', 'mps'
        max_chi: MPS bond dimensie (voor MPS backend)
        gpu: gebruik GPU acceleratie
        verbose: print voortgang

    Returns:
        dict met 'observables', 'backend', 'time', en backend-specifieke data
    """
    t0 = time.time()

    # Auto-selectie
    if backend == 'auto':
        if circuit.n_qubits <= 22:
            backend = 'statevector'
        else:
            backend = 'mps'

    if verbose:
        s = circuit.summary()
        print("Circuit: %s, %d qubits, %d gates, depth %d" % (
            s['name'], s['n_qubits'], s['n_gates'], s['depth']))
        print("Backend: %s" % backend)

    if backend == 'statevector':
        state, results = _run_statevector(circuit, observables)
        elapsed = time.time() - t0
        return {
            'observables': results,
            'backend': 'statevector',
            'time': elapsed,
            'state': state,
        }
    elif backend == 'mps':
        mps, results = _run_mps(circuit, observables, max_chi=max_chi, gpu=gpu)
        elapsed = time.time() - t0
        return {
            'observables': results,
            'backend': 'mps',
            'time': elapsed,
            'max_chi_reached': mps._max_chi_reached,
            'total_discarded': mps._total_discarded,
        }
    else:
        raise ValueError("Onbekende backend: %s" % backend)
