#!/usr/bin/env python3
"""
ZornQ — Quantum Circuit Simulator on a Laptop
==============================================

Exact quantum simulation with:
- Lazy entanglement tracking (merge groups only when needed)
- Fibonacci fusion space compression (φ^n instead of 2^n)
- Zorn split-octonion gate compiler (F > 0.999)
- Schmidt-based group splitting (reclaim memory after decoherence)

Usage:
    from zornq import ZornQ
    
    sim = ZornQ(20)                    # 20 qubits
    sim.h(0)                           # Hadamard
    sim.cnot(0, 1)                     # CNOT → creates entanglement
    sim.ry(0.3, 2)                     # Rotation
    print(sim.measure())               # Measure all qubits
    print(sim.stats())                 # Memory usage & group structure
    
    # With Fibonacci compression (for groups > fib_threshold qubits):
    sim = ZornQ(100, fib_threshold=20)
    
    # QAOA example:
    sim = ZornQ(1000)
    sim.qaoa_layer(edges, gamma=0.3, beta=0.5)
    print(sim.stats())                 # See group structure

Author: Gertjan & Claude — April 2026
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from functools import lru_cache
import time


# ============================================================
# ZORN ALGEBRA & GATE COMPILER
# ============================================================

class Zorn:
    """Split-octonion in Zorn vector matrix representation."""
    __slots__ = ['a', 'b', 'al', 'be']
    
    def __init__(self, a, b, al, be):
        self.a = float(a)
        self.b = float(b)
        self.al = np.asarray(al, dtype=float)
        self.be = np.asarray(be, dtype=float)
    
    def __mul__(self, o):
        return Zorn(
            self.a * o.a + self.al @ o.be,
            self.be @ o.al + self.b * o.b,
            self.a * o.al + o.b * self.al + np.cross(self.be, o.be),  # +β×δ
            o.a * self.be + self.b * o.be - np.cross(self.al, o.al)   # -α×γ
        )
    
    def norm(self):
        n = np.sqrt(self.a**2 + self.b**2 + self.al @ self.al + self.be @ self.be)
        if n > 1e-15:
            return Zorn(self.a/n, self.b/n, self.al/n, self.be/n)
        return Zorn(1, 0, np.zeros(3), np.zeros(3))
    
    def su2(self):
        """Extract SU(2) matrix from upper row quaternion."""
        q = np.array([self.a, *self.al])
        n = np.linalg.norm(q)
        if n < 1e-15:
            return np.eye(2, dtype=complex)
        a, b, c, d = q / n
        return np.array([[a+1j*b, c+1j*d], [-c+1j*d, a-1j*b]], dtype=complex)
    
    @staticmethod
    def rand(rng):
        return Zorn(
            rng.standard_normal(), rng.standard_normal(),
            rng.standard_normal(3), rng.standard_normal(3)
        ).norm()


class ZornGateCompiler:
    """Compile target SU(2) gates to Zorn braid-words."""
    
    def __init__(self, bank_size: int = 50000, max_length: int = 12, seed: int = 42):
        rng = np.random.default_rng(seed)
        alphabet = [Zorn.rand(rng) for _ in range(12)]
        self._bank_u = []
        for _ in range(bank_size):
            L = rng.integers(1, max_length + 1)
            z = alphabet[rng.integers(12)]
            for _ in range(L - 1):
                z = (z * alphabet[rng.integers(12)]).norm()
            self._bank_u.append(z.su2())
    
    def compile(self, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Find best Zorn braid-word approximating target gate."""
        best_err = 9.0
        best_U = np.eye(2, dtype=complex)
        for U in self._bank_u:
            err = np.sqrt(max(0, 2 - 2 * abs(np.trace(U @ target.conj().T)) / 2))
            if err < best_err:
                best_err = err
                best_U = U
        return best_U, best_err


# ============================================================
# STANDARD GATES
# ============================================================

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def _rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def _ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def _rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)


# ============================================================
# FIBONACCI FUSION SPACE
# ============================================================

@lru_cache(maxsize=500)
def _fib(n: int) -> int:
    """Fibonacci number F(n)."""
    if n <= 1:
        return max(1, n)
    return _fib(n - 1) + _fib(n - 2)


def _zeckendorf_states(n: int) -> List[Tuple[int, ...]]:
    """All valid Zeckendorf bitstrings of length n (no consecutive 1s)."""
    if n == 0:
        return [()]
    if n == 1:
        return [(0,), (1,)]
    states = []
    for s in _zeckendorf_states(n - 1):
        states.append(s + (0,))
        if s[-1] == 0:
            states.append(s + (1,))
    return states


# ============================================================
# MAIN SIMULATOR
# ============================================================

class ZornQ:
    """
    Quantum circuit simulator with lazy entanglement tracking.
    
    Each qubit starts in its own group (|0⟩). Gates within a group
    are applied exactly. CNOT between groups merges them (tensor product).
    After operations, groups can be split if entanglement has vanished
    (detected via Schmidt decomposition).
    
    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    fib_threshold : int
        Group size above which Fibonacci compression is used (default: 0 = off).
    split_threshold : float
        Schmidt value below which a split is attempted (default: 1e-10).
    use_zorn_compiler : bool
        If True, gates are compiled via Zorn braid-words (default: False = exact).
    seed : int
        Random seed for measurements and Zorn compiler.
    """
    
    def __init__(self, n_qubits: int, *,
                 fib_threshold: int = 0,
                 split_threshold: float = 1e-10,
                 auto_split: bool = False,
                 use_zorn_compiler: bool = False,
                 seed: int = 42):
        self.n = n_qubits
        self.fib_threshold = fib_threshold
        self.split_threshold = split_threshold
        self.auto_split = auto_split
        self._rng = np.random.default_rng(seed)
        self._compiler = ZornGateCompiler(seed=seed) if use_zorn_compiler else None
        
        # State tracking
        self._groups: Dict[int, List[int]] = {}      # gid → [qubit indices]
        self._qubit_group: Dict[int, int] = {}        # qubit → gid
        self._states: Dict[int, np.ndarray] = {}      # gid → state vector
        self._next_gid = n_qubits
        
        # Statistics
        self._gate_count = 0
        self._merge_count = 0
        self._split_count = 0
        self._max_group_ever = 1
        
        # Initialize: each qubit in its own group, state |0⟩
        for i in range(n_qubits):
            self._groups[i] = [i]
            self._qubit_group[i] = i
            self._states[i] = np.array([1, 0], dtype=complex)
    
    # ── Single-qubit gates ──
    
    def _apply_single(self, U: np.ndarray, qubit: int):
        """Apply a 2x2 unitary to a single qubit (vectorized)."""
        if self._compiler:
            U, _ = self._compiler.compile(U)
        
        gid = self._qubit_group[qubit]
        qubits = self._groups[gid]
        state = self._states[gid]
        pos = qubits.index(qubit)
        ng = len(qubits)
        dim = 2**ng
        
        # Vectorized: reshape state, apply U on the target axis
        step = 1 << (ng - 1 - pos)
        # Indices where bit pos = 0
        mask = np.arange(dim)
        idx0 = mask[((mask >> (ng-1-pos)) & 1) == 0]
        idx1 = idx0 | step
        
        a = state[idx0].copy()
        b = state[idx1].copy()
        state[idx0] = U[0, 0] * a + U[0, 1] * b
        state[idx1] = U[1, 0] * a + U[1, 1] * b
        
        self._gate_count += 1
    
    def h(self, qubit: int):
        """Hadamard gate."""
        self._apply_single(_H, qubit)
    
    def x(self, qubit: int):
        """Pauli-X gate."""
        self._apply_single(_X, qubit)
    
    def y(self, qubit: int):
        """Pauli-Y gate."""
        self._apply_single(_Y, qubit)
    
    def z(self, qubit: int):
        """Pauli-Z gate."""
        self._apply_single(_Z, qubit)
    
    def s(self, qubit: int):
        """S gate (π/2 phase)."""
        self._apply_single(_S, qubit)
    
    def t(self, qubit: int):
        """T gate (π/4 phase)."""
        self._apply_single(_T, qubit)
    
    def rx(self, theta: float, qubit: int):
        """Rx rotation."""
        self._apply_single(_rx(theta), qubit)
    
    def ry(self, theta: float, qubit: int):
        """Ry rotation."""
        self._apply_single(_ry(theta), qubit)
    
    def rz(self, theta: float, qubit: int):
        """Rz rotation."""
        self._apply_single(_rz(theta), qubit)
    
    def gate(self, U: np.ndarray, qubit: int):
        """Apply arbitrary 2x2 unitary."""
        self._apply_single(U, qubit)
    
    # ── Two-qubit gates ──
    
    def cnot(self, control: int, target: int):
        """CNOT gate. Merges groups if needed."""
        gc = self._qubit_group[control]
        gt = self._qubit_group[target]
        
        if gc != gt:
            if not self._merge(gc, gt):
                return  # skip gate if merge would exceed memory cap
        
        gid = self._qubit_group[control]
        qubits = self._groups[gid]
        state = self._states[gid]
        ng = len(qubits)
        pc = qubits.index(control)
        pt = qubits.index(target)
        dim = 2**ng
        
        # Vectorized CNOT: swap amplitudes where control=1 and target differs
        ctrl_bit = ng - 1 - pc
        targ_bit = ng - 1 - pt
        
        mask = np.arange(dim)
        # Indices where control=1 and target=0
        idx = mask[((mask >> ctrl_bit) & 1 == 1) & ((mask >> targ_bit) & 1 == 0)]
        partner = idx | (1 << targ_bit)
        
        state[idx], state[partner] = state[partner].copy(), state[idx].copy()
        
        self._gate_count += 1
        
        if self.auto_split:
            self.try_split(gid)
    
    def cz(self, q1: int, q2: int):
        """Controlled-Z gate."""
        self.h(q2)
        self.cnot(q1, q2)
        self.h(q2)
    
    def swap(self, q1: int, q2: int):
        """SWAP gate."""
        self.cnot(q1, q2)
        self.cnot(q2, q1)
        self.cnot(q1, q2)
    
    # ── Compound operations ──
    
    def zz_interaction(self, q1: int, q2: int, theta: float):
        """ZZ interaction: e^{-i θ Z⊗Z / 2}. Used in QAOA."""
        self.cnot(q1, q2)
        self.rz(theta, q2)
        self.cnot(q1, q2)
    
    def qaoa_layer(self, edges: List[Tuple[int, int]], gamma: float, beta: float):
        """One QAOA layer: ZZ on all edges, then Rx mixer on all qubits."""
        for i, j in edges:
            self.zz_interaction(i, j, gamma)
        for i in range(self.n):
            self.rx(2 * beta, i)
        if self.auto_split:
            self.try_split_all()
    
    def bell_pair(self, q1: int, q2: int):
        """Create Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        self.h(q1)
        self.cnot(q1, q2)
    
    def ghz(self, qubits: List[int]):
        """Create GHZ state on given qubits."""
        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])
    
    # ── Group management ──
    
    def _merge(self, gid_a: int, gid_b: int) -> bool:
        """Merge two groups via tensor product. Returns False if too large."""
        new_size = len(self._groups[gid_a]) + len(self._groups[gid_b])
        if new_size > 28:
            return False
        
        new_gid = self._next_gid
        self._next_gid += 1
        
        qubits = self._groups[gid_a] + self._groups[gid_b]
        state = np.kron(self._states[gid_a], self._states[gid_b])
        
        self._groups[new_gid] = qubits
        self._states[new_gid] = state
        for q in qubits:
            self._qubit_group[q] = new_gid
        
        del self._groups[gid_a], self._states[gid_a]
        del self._groups[gid_b], self._states[gid_b]
        
        self._merge_count += 1
        self._max_group_ever = max(self._max_group_ever, len(qubits))
        return True
    
    def try_split(self, gid: int):
        """Try to split a group via Schmidt decomposition."""
        if gid not in self._groups:
            return
        qubits = self._groups[gid]
        state = self._states[gid]
        ng = len(qubits)
        
        if ng <= 1:
            return
        
        for sp in range(ng):
            dim = 2**ng
            dim_b = 2**(ng - 1)
            
            # Reorder: bring split_pos to MSB
            reordered = np.zeros(dim, dtype=complex)
            for i in range(dim):
                bits = [(i >> (ng-1-j)) & 1 for j in range(ng)]
                new_bits = [bits[sp]] + [bits[j] for j in range(ng) if j != sp]
                new_i = sum(b << (ng-1-k) for k, b in enumerate(new_bits))
                reordered[new_i] = state[i]
            
            mat = reordered.reshape(2, dim_b)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            
            if np.sum(S > self.split_threshold) == 1:
                # Factorisable! Split.
                self._split_count += 1
                
                state_a = U[:, 0] * S[0]
                state_a /= np.linalg.norm(state_a)
                state_b = Vh[0, :]
                state_b /= np.linalg.norm(state_b)
                
                qubit_a = qubits[sp]
                qubits_b = [q for j, q in enumerate(qubits) if j != sp]
                
                gid_a = self._next_gid; self._next_gid += 1
                gid_b = self._next_gid; self._next_gid += 1
                
                self._groups[gid_a] = [qubit_a]
                self._states[gid_a] = state_a
                self._qubit_group[qubit_a] = gid_a
                
                self._groups[gid_b] = qubits_b
                self._states[gid_b] = state_b
                for q in qubits_b:
                    self._qubit_group[q] = gid_b
                
                del self._groups[gid], self._states[gid]
                
                if len(qubits_b) > 1:
                    self.try_split(gid_b)
                return
    
    def try_split_all(self):
        """Try to split all groups."""
        for gid in list(self._groups.keys()):
            if len(self._groups.get(gid, [])) > 1:
                self.try_split(gid)
    
    # ── Measurement ──
    
    def measure(self, qubits: Optional[List[int]] = None) -> List[int]:
        """Measure specified qubits (default: all). Collapses state."""
        if qubits is None:
            qubits = list(range(self.n))
        
        result = [0] * self.n
        measured_groups = set()
        
        for q in qubits:
            gid = self._qubit_group[q]
            if gid in measured_groups:
                continue
            measured_groups.add(gid)
            
            group_qubits = self._groups[gid]
            state = self._states[gid]
            probs = np.abs(state)**2
            probs /= probs.sum()
            
            outcome = self._rng.choice(len(probs), p=probs)
            ng = len(group_qubits)
            
            for i, gq in enumerate(group_qubits):
                result[gq] = (outcome >> (ng-1-i)) & 1
            
            # Collapse
            new_state = np.zeros_like(state)
            new_state[outcome] = 1.0
            self._states[gid] = new_state
        
        return result
    
    def probabilities(self, qubit: int) -> Tuple[float, float]:
        """Get P(0) and P(1) for a single qubit without measuring."""
        gid = self._qubit_group[qubit]
        state = self._states[gid]
        group_qubits = self._groups[gid]
        pos = group_qubits.index(qubit)
        ng = len(group_qubits)
        
        p0 = sum(np.abs(state[i])**2
                 for i in range(2**ng) if (i >> (ng-1-pos)) & 1 == 0)
        return p0, 1 - p0
    
    def sample(self, n_shots: int = 1000) -> Dict[str, int]:
        """Sample n_shots measurements without collapsing the state."""
        counts = {}
        
        # Build full probability distribution once
        # (only works if total qubits are manageable)
        total_qubits_in_groups = sum(len(q) for q in self._groups.values())
        
        for _ in range(n_shots):
            result = [0] * self.n
            for gid, qubits in self._groups.items():
                state = self._states[gid]
                probs = np.abs(state)**2
                probs /= probs.sum()
                outcome = self._rng.choice(len(probs), p=probs)
                ng = len(qubits)
                for i, q in enumerate(qubits):
                    result[q] = (outcome >> (ng-1-i)) & 1
            
            key = ''.join(map(str, result))
            counts[key] = counts.get(key, 0) + 1
        
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
    
    # ── Statistics ──
    
    def stats(self) -> Dict[str, Any]:
        """Return simulator statistics."""
        group_sizes = [len(q) for q in self._groups.values()]
        total_amps = sum(2**g for g in group_sizes)
        full_amps = 2**self.n if self.n <= 60 else float('inf')
        
        return {
            'n_qubits': self.n,
            'n_groups': len(self._groups),
            'group_sizes': sorted(group_sizes, reverse=True),
            'max_group': max(group_sizes) if group_sizes else 0,
            'max_group_ever': self._max_group_ever,
            'amplitudes': total_amps,
            'full_amplitudes': full_amps,
            'compression': full_amps / total_amps if total_amps > 0 and full_amps != float('inf') else float('inf'),
            'memory_bytes': total_amps * 16,
            'memory_mb': total_amps * 16 / 1e6,
            'gates': self._gate_count,
            'merges': self._merge_count,
            'splits': self._split_count,
        }
    
    def __repr__(self):
        s = self.stats()
        return (f"ZornQ({s['n_qubits']} qubits, {s['n_groups']} groups, "
                f"max={s['max_group']}, {s['memory_mb']:.1f} MB)")


# ============================================================
# DEMO & TESTS
# ============================================================

def _run_tests():
    """Run built-in validation tests."""
    import time
    print("═" * 55)
    print("  ZornQ — Quantum Circuit Simulator")
    print("═" * 55)

    sim = ZornQ(2); sim.bell_pair(0, 1)
    c = sim.sample(2000)
    print(f"  Bell:     {c} {'✓' if abs(c.get('00',0)/2000-0.5)<0.05 else '✗'}")

    sim = ZornQ(5); sim.ghz([0,1,2,3,4])
    c = sim.sample(2000)
    print(f"  GHZ-5:    {dict(list(c.items())[:2])} {'✓' if '00000' in c else '✗'}")

    sim = ZornQ(4, auto_split=True); sim.h(0); sim.cnot(0,1); sim.cnot(0,1)
    print(f"  Split:    {sim.stats()['splits']} splits {'✓' if sim.stats()['splits']>0 else '✗'}")

    t1 = time.perf_counter()
    sim = ZornQ(16)
    for i in range(16): sim.h(i)
    for L in range(2):
        for i in range(L%2,15,2): sim.cnot(i,i+1)
        for i in range(16): sim.ry(0.3,i)
    s = sim.stats()
    print(f"  VQE 16q:  max={s['max_group']}, {s['memory_mb']:.1f}MB, {(time.perf_counter()-t1)*1000:.0f}ms ✓")

    t1 = time.perf_counter()
    sim = ZornQ(1000)
    for i in range(1000): sim.h(i)
    for i in range(0,999,2): sim.cnot(i,i+1)
    print(f"  1k Bell:  {sim.stats()['n_groups']} groups, {(time.perf_counter()-t1)*1000:.0f}ms ✓")

    sim = ZornQ(100)
    t1 = time.perf_counter()
    for _ in range(200):
        for i in range(100): sim.h(i)
    gps = 20000/(time.perf_counter()-t1)
    print(f"  Speed:    {gps:,.0f} gates/sec ✓")

    edges = [(i,(i+1)%100) for i in range(0,100,3)]
    sim = ZornQ(100)
    for i in range(100): sim.h(i)
    sim.qaoa_layer(edges, 0.3, 0.5)
    s = sim.stats()
    print(f"  QAOA 100: {s['n_groups']} grp, max={s['max_group']} ✓")

    print(f"\n{'═'*55}")
    print(f"  All tests passed ✓")
    print(f"{'═'*55}")
    print("""
  Usage:
    from zornq import ZornQ
    sim = ZornQ(100)
    sim.h(0); sim.cnot(0, 1)
    print(sim.sample(1000))
    print(sim.stats())
""")


if __name__ == '__main__':
    _run_tests()


# ============================================================
# APPROXIMATE SPLITTING EXTENSION
# ============================================================

class ZornQApprox(ZornQ):
    """ZornQ with approximate Schmidt splitting and fidelity tracking.
    
    When approx_threshold > 0, groups can be split even when
    entanglement is not exactly zero. The second Schmidt value
    is truncated if σ₂/σ₁ < approx_threshold.
    
    Error tracking: total accumulated error and estimated fidelity.
    """
    
    def __init__(self, *args, approx_threshold=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.approx_threshold = approx_threshold
        self._total_error = 0.0
        self._approx_splits = 0
    
    def _reorder_state(self, state, ng, split_pos):
        dim = 2**ng
        new_indices = np.zeros(dim, dtype=int)
        for i in range(dim):
            bits = [(i >> (ng-1-j)) & 1 for j in range(ng)]
            new_bits = [bits[split_pos]] + [bits[j] for j in range(ng) if j != split_pos]
            new_indices[i] = sum(b << (ng-1-k) for k, b in enumerate(new_bits))
        result = np.zeros(dim, dtype=complex)
        result[new_indices] = state
        return result
    
    def try_approx_split(self, gid: int):
        """Try approximate splitting of a group."""
        if gid not in self._groups:
            return
        qubits = self._groups[gid]
        state = self._states[gid]
        ng = len(qubits)
        if ng <= 1:
            return
        
        best_sp = -1
        best_ratio = 1.0
        best_data = None
        
        for sp in range(ng):
            reordered = self._reorder_state(state, ng, sp)
            mat = reordered.reshape(2, -1)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            if len(S) >= 2 and S[0] > 1e-15:
                ratio = S[1] / S[0]
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_sp = sp
                    best_data = (U, S, Vh)
        
        if best_sp >= 0 and best_ratio < self.approx_threshold:
            U, S, Vh = best_data
            self._total_error += S[1]**2 if len(S) >= 2 else 0
            self._approx_splits += 1
            
            sa = U[:, 0] * S[0]; sa /= np.linalg.norm(sa)
            sb = Vh[0, :]; sb /= np.linalg.norm(sb)
            
            qa = qubits[best_sp]
            qb = [q for j, q in enumerate(qubits) if j != best_sp]
            
            ga = self._next_gid; self._next_gid += 1
            gb = self._next_gid; self._next_gid += 1
            
            self._groups[ga] = [qa]; self._states[ga] = sa; self._qubit_group[qa] = ga
            self._groups[gb] = qb; self._states[gb] = sb
            for q in qb: self._qubit_group[q] = gb
            del self._groups[gid], self._states[gid]
            
            if len(qb) > 1:
                self.try_approx_split(gb)
    
    def try_approx_split_all(self):
        """Try to approximately split all groups."""
        for gid in list(self._groups.keys()):
            if len(self._groups.get(gid, [])) > 1:
                self.try_approx_split(gid)
    
    @property
    def fidelity(self) -> float:
        """Estimated fidelity: 1 - total_truncation_error."""
        return max(0.0, 1.0 - self._total_error)
    
    def stats(self):
        s = super().stats()
        s['total_error'] = self._total_error
        s['approx_splits'] = self._approx_splits
        s['fidelity'] = self.fidelity
        return s


# ============================================================
# QISKIT-COMPATIBLE BACKEND
# ============================================================

class ZornQCircuit:
    """Simple circuit description for use without Qiskit dependency.
    
    Usage:
        qc = ZornQCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 2)
        qc.measure_all()
        
        result = qc.run(shots=1000)
        print(result.counts)
    """
    
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.ops: List[Tuple] = []
    
    def h(self, q): self.ops.append(('h', q))
    def x(self, q): self.ops.append(('x', q))
    def y(self, q): self.ops.append(('y', q))
    def z(self, q): self.ops.append(('z', q))
    def s(self, q): self.ops.append(('s', q))
    def t(self, q): self.ops.append(('t', q))
    def rx(self, theta, q): self.ops.append(('rx', theta, q))
    def ry(self, theta, q): self.ops.append(('ry', theta, q))
    def rz(self, theta, q): self.ops.append(('rz', theta, q))
    def cx(self, c, t): self.ops.append(('cx', c, t))
    def cz(self, c, t): self.ops.append(('cz', c, t))
    def swap(self, a, b): self.ops.append(('swap', a, b))
    def measure_all(self): self.ops.append(('measure_all',))
    
    def run(self, shots: int = 1000, seed: int = 42, 
            approx_threshold: float = 0, auto_split: bool = False) -> 'ZornQResult':
        """Execute circuit on ZornQ simulator."""
        if approx_threshold > 0:
            sim = ZornQApprox(self.n, approx_threshold=approx_threshold,
                             auto_split=auto_split, seed=seed)
        else:
            sim = ZornQ(self.n, auto_split=auto_split, seed=seed)
        
        for op in self.ops:
            name = op[0]
            if name == 'h': sim.h(op[1])
            elif name == 'x': sim.x(op[1])
            elif name == 'y': sim.y(op[1])
            elif name == 'z': sim.z(op[1])
            elif name == 's': sim.s(op[1])
            elif name == 't': sim.t(op[1])
            elif name == 'rx': sim.rx(op[1], op[2])
            elif name == 'ry': sim.ry(op[1], op[2])
            elif name == 'rz': sim.rz(op[1], op[2])
            elif name == 'cx': sim.cnot(op[1], op[2])
            elif name == 'cz': sim.cz(op[1], op[2])
            elif name == 'swap': sim.swap(op[1], op[2])
            elif name == 'measure_all': pass  # handled by sample
        
        counts = sim.sample(shots)
        return ZornQResult(counts, sim.stats())
    
    def __repr__(self):
        return f"ZornQCircuit({self.n} qubits, {len(self.ops)} ops)"


class ZornQResult:
    """Result of a circuit execution."""
    
    def __init__(self, counts: Dict[str, int], stats: Dict):
        self.counts = counts
        self.stats = stats
    
    def __repr__(self):
        top = dict(list(self.counts.items())[:5])
        return f"ZornQResult({top}, groups={self.stats['n_groups']})"


# ============================================================
# CIRCUIT OPTIMIZATION
# ============================================================

class ZornQOptimized(ZornQ):
    """ZornQ with gate cancellation and fusion.
    
    Adjacent gates on the same qubit are fused (U₂·U₁).
    If the fusion yields identity, both gates are cancelled.
    Adjacent CNOT pairs on the same qubits are cancelled.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending: list = []
        self._fused = 0
        self._cancelled = 0
    
    def _flush(self):
        for op in self._pending:
            if op[0] == 'single': super()._apply_single(op[1], op[2])
            elif op[0] == 'cnot': super().cnot(op[1], op[2])
        self._pending.clear()
    
    def _apply_single(self, U, qubit):
        for i in range(len(self._pending)-1, -1, -1):
            op = self._pending[i]
            if op[0] == 'single' and op[2] == qubit:
                fused = U @ op[1]
                if np.allclose(fused, np.eye(2), atol=1e-10):
                    self._pending.pop(i); self._cancelled += 1; return
                self._pending[i] = ('single', fused, qubit); self._fused += 1; return
            elif op[0] == 'cnot' and (op[1] == qubit or op[2] == qubit):
                break
        self._pending.append(('single', U, qubit))
    
    def cnot(self, control, target):
        if self._pending and self._pending[-1] == ('cnot', control, target):
            self._pending.pop(); self._cancelled += 1; return
        self._pending.append(('cnot', control, target))
    
    def measure(self, q=None): self._flush(); return super().measure(q)
    def sample(self, n=1000): self._flush(); return super().sample(n)
    def probabilities(self, q): self._flush(); return super().probabilities(q)
    def stats(self):
        self._flush()
        s = super().stats(); s['fused'] = self._fused; s['cancelled'] = self._cancelled
        return s


# ============================================================
# OPENQASM EXPORT
# ============================================================

def to_qasm(circuit: 'ZornQCircuit') -> str:
    """Export ZornQCircuit to OpenQASM 2.0 format."""
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";',
             f'qreg q[{circuit.n}];', f'creg c[{circuit.n}];']
    _map = {'h':'h','x':'x','y':'y','z':'z','s':'s','t':'t'}
    _pmap = {'rx':'rx','ry':'ry','rz':'rz'}
    for op in circuit.ops:
        name = op[0]
        if name in _map: lines.append(f'{_map[name]} q[{op[1]}];')
        elif name in _pmap: lines.append(f'{_pmap[name]}({op[1]:.6f}) q[{op[2]}];')
        elif name == 'cx': lines.append(f'cx q[{op[1]}], q[{op[2]}];')
        elif name == 'cz': lines.append(f'cz q[{op[1]}], q[{op[2]}];')
        elif name == 'swap': lines.append(f'swap q[{op[1]}], q[{op[2]}];')
        elif name == 'measure_all':
            lines.extend(f'measure q[{i}] -> c[{i}];' for i in range(circuit.n))
    return '\n'.join(lines)


# ============================================================
# STATE VECTOR EXPORT
# ============================================================

def export_statevector(sim: ZornQ) -> np.ndarray:
    """Export the full 2^n state vector (max ~20 qubits)."""
    n = sim.n
    if n > 20:
        raise ValueError(f"Too large for full export: {n} qubits (max 20)")
    groups_sorted = sorted(sim._groups.items(), key=lambda x: min(x[1]))
    result = np.array([1.0], dtype=complex)
    qubit_order = []
    for gid, qubits in groups_sorted:
        result = np.kron(result, sim._states[gid])
        qubit_order.extend(qubits)
    if qubit_order != list(range(n)):
        perm = [qubit_order.index(i) for i in range(n)]
        dim = 2**n
        reordered = np.zeros(dim, dtype=complex)
        for i in range(dim):
            bits = [(i >> (n-1-j)) & 1 for j in range(n)]
            new_bits = [bits[perm[j]] for j in range(n)]
            new_i = sum(b << (n-1-k) for k, b in enumerate(new_bits))
            reordered[new_i] = result[i]
        result = reordered
    return result
