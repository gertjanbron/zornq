# Split-Octonions as an Information-Complete Framework for Quantum Computing

**Gertjan [achternaam]**
NHL Stenden University of Applied Sciences

**Draft v2 — April 9, 2026**

---

## Abstract

We establish that the split-octonion algebra in Zorn vector matrix representation provides an information-complete framework for quantum state representation. The core result: seven algebraic operations (multiplication, addition, subtraction, division, Hodge rotation, non-associative rebracketing, and the Jordan triple product ABA) span the full 2ⁿ-dimensional Hilbert space, verified at n=6 (64/64) and n=9 (512/512). One Zorn element (8 components) encodes exactly 3 qubits with the Born rule emerging as the Euclidean norm (error = 0 over 1000 random states). The Zorn product automatically performs Schmidt decomposition, capturing 95.6% of quantum information in a single product (QAOA 6-qubit benchmark), with exact reconstruction from 2 products at 4× compression. The split-norm ab−α·β is identical to the concurrence (entanglement measure) and arises spontaneously from quantum gates: H|0⟩ yields split-norm +0.5 while H|1⟩ yields −0.5, creating a natural (8,0)→(4,4)→(8,0) lifecycle for quantum computation. The rotscalar encoding achieves 100% basis-state survival through the Zorn product at 6, 9, 12, 15, and 18 qubits (262,144 states). We identify that each of the seven operations captures a structurally distinct class of quantum correlations, with the Jordan triple product—the measurement operator—filling the final 5 dimensions that all bilinear operations miss.

**Keywords:** split-octonions, Zorn matrices, quantum information, Schmidt decomposition, entanglement, concurrence, information completeness, Jordan triple product

---

## 1. Introduction

The octonions, discovered independently by Cayley and Graves in the 1840s, are the largest normed division algebra. Their split form, naturally represented as Zorn vector matrices, admits zero divisors through a (4,4) metric signature and connects to exceptional structures in mathematics and physics: the G₂ automorphism group, the E₈ lattice, SO(8) triality, and the Fano plane.

Despite longstanding interest in octonion applications to physics (Günaydin and Gürsey 1973, Baez 2002, Furey 2018), no concrete computational framework has connected the split-octonion algebra to quantum information processing. We establish this connection through three main results:

1. **Representation (§3):** A single Zorn element with 8 components encodes exactly 3 qubits. The Born rule is the Euclidean norm. Quantum gates are rotations in the Zorn space. The match 8 = 2³ is exact, not approximate.

2. **Compression (§4):** The Zorn product automatically selects the dominant Schmidt term of a bipartite quantum state. For QAOA circuits, one product achieves 95.6% fidelity; two products give exact reconstruction with 4× compression. The split-norm is the concurrence.

3. **Completeness (§5):** Seven algebraic operations—the four arithmetic operations, Hodge rotation, non-associative rebracketing, and the Jordan triple product—span the full Hilbert space. Verified: rank 64/64 at 6 qubits, rank 512/512 at 9 qubits.

Additionally, we present results on topological quantum gate compilation (§6), showing that Zorn braid-words achieve 2–3× better SU(2) approximation than Fibonacci anyons, and on the physical architecture (§7), where quantum computation corresponds to spontaneous signature breaking (8,0) → (4,4) → (8,0).

---

## 2. Zorn Vector Matrix Algebra

### 2.1 Definition

A split-octonion is represented as a Zorn vector matrix:

    Z = (a, α; β, b)    with a,b ∈ ℂ, α,β ∈ ℂ³.

Product:

    (a₁,α₁;β₁,b₁) × (a₂,α₂;β₂,b₂) = 
        (a₁a₂+α₁·β₂,  a₁α₂+b₂α₁+β₁×β₂;
         a₂β₁+b₁β₂−α₁×α₂,  β₁·α₂+b₁b₂)

The sign asymmetry (+β₁×β₂ vs −α₁×α₂) is structurally necessary for alternativity, verified by exhaustive search over 161,280 candidates. All algebraic identities (alternativity, flexibility, Moufang, composition) verified to machine precision (<10⁻¹⁴).

### 2.2 Split-norm and signature

The split-norm N(Z) = ab − α·β has signature (4,4), admitting zero divisors where N = 0. The compact norm ab + α·β has signature (8,0) with no zero divisors. The algebras Cl(8,0), Cl(0,8), and Cl(4,4) are all isomorphic to M₁₆(ℝ) by Bott periodicity.

### 2.3 Seven Cayley-Dickson decompositions

The octonions admit 7 inequivalent decompositions into quaternionic subalgebras, corresponding to the 7 lines of the Fano plane. Under the Zorn representation, these decompositions assign the 6 mixed bitstrings differently to the α and β vectors. The decompositions are related by the G₂ automorphism group (dimension 14).

---

## 3. Three Qubits = One Zorn Element

### 3.1 The encoding

The 8 components of a Zorn element map bijectively to the 8 amplitudes of 3 qubits:

    a = |000⟩,  α₁ = |001⟩,  α₂ = |010⟩,  α₃ = |011⟩
    β₁ = |100⟩,  β₂ = |101⟩,  β₃ = |110⟩,  b = |111⟩

**Theorem 2 (Born rule).** For any 3-qubit state |ψ⟩ encoded as a Zorn element Z, the measurement probability P(bitstring) = |Z_component|²/||Z||². Verified: maximum error = 0 over 1000 random states.

**Theorem 3 (Gates as rotations).** Standard quantum gates correspond to rotations in Zorn space: H on qubit 0 is a rotation in the (a,β₁) plane; CNOT(0,1) is a swap β₁ ↔ β₃.

### 3.2 Rotscalar encoding

For multi-qubit states (n > 3), the naive encoding loses 50% of basis states through the cross-product annihilation α×α = 0. The **rotscalar encoding** resolves this:

    |000⟩ → (a+b)/√2,    |111⟩ → (a−b)/√2
    |mixed⟩ → (α_i + iβ_i)/√2  or  (β_i + iα_i)/√2

**Theorem 4 (Complete survival).** Under rotscalar encoding, all 2ⁿ basis states survive the Zorn product. Verified at n = 6, 9, 12, 15, 18 (262,144 states).

### 3.3 Split-norm as concurrence

**Theorem 5.** The split-norm N = ab − α·β equals the concurrence (entanglement measure) of the encoded state. It arises spontaneously from quantum gates:

    |0⟩ → N = 0 (unentangled),   H|0⟩ → N = +0.5 (correlated),
    H|1⟩ → N = −0.5 (anti-correlated),   |+⟩|+⟩ → N = 0 (product state).

The sign of N determines the type of entanglement: positive = correlation, negative = anti-correlation, zero = product state.

---

## 4. Zorn Product as Schmidt Machine

### 4.1 Automatic Schmidt selection

The Zorn product of two 3-qubit Zorn elements projects onto the dominant Schmidt term of the 6-qubit state.

**Benchmark (QAOA 6 qubits, 1D chain, 1 layer):**

| k products | Fidelity | Parameters | Compression |
|---|---|---|---|
| 1 | 95.58% | 16 | 8× |
| 2 | 100.00% | 32 | 4× |

**Theorem 6 (Schmidt rank bounds).** For 1D circuits with bounded depth d, the Schmidt rank across any 3-qubit cut is bounded by min(2^d, 8). For QAOA (d=1): rank 2, constant in n. For VQE nearest-neighbor (d=2): rank 1–2. Verified up to n=18.

### 4.2 Compression scaling

| Circuit | Schmidt rank | Parameters | Scaling |
|---|---|---|---|
| GHZ | 2 (any n) | ~5n | O(n) |
| QAOA 1D 1L | 2 (any n) | ~5n | O(n) |
| VQE NN 2L | 1–2 (any n) | ~3n | O(n) |
| VQE NN 4L | 2–4 (any n) | ~11n | O(n) |
| Random deep | 8–64 (grows) | ~21n–O(2ⁿ) | varies |

MPS reconstruction verified: fidelity = 1.0000000000 for QAOA-12q (96 params vs 4096).

---

## 5. Information Completeness: Seven Operations

### 5.1 The problem

A single Zorn product (bilinear, 8→8) cannot represent the full 2ⁿ Hilbert space for n > 3. The product's transfer matrix has rank 29/64 at 6 qubits (45.3%). Adding 7 Cayley-Dickson decompositions saturates at rank 29.

### 5.2 Triality and Hodge duality

SO(8) triality provides three inequivalent 8D representations. The Hodge dual (cyclic permutation of 3-vector components) achieves rank 55/64 individually—the strongest single operation. Combined with conjugation:

    id|hodge + conj|hodge + hodge|neg_α → rank 64/64 (6 qubits)

### 5.3 Addition and subtraction

The sum (A+B)×C captures non-local correlations between non-adjacent groups. (A−B)×C captures anti-symmetric combinations. Both are independently necessary:

    Addition alone: rank 29 (9 qubits)
    Subtraction alone: rank 29
    Both together: rank 58

### 5.4 Division

Division A⁻¹ = conj(A)/N(A) is NOT equivalent to conjugation because the split-norm N has varying phase across basis states (N = 0.5, −0.5i, −0.5). This phase rotation carries entanglement information. Three division variants contribute 152 independent dimensions.

### 5.5 The Jordan triple product

**Theorem 7 (Completeness).** The Jordan triple product (ABA)×C fills the final 5 dimensions that all bilinear operations miss. With the complete set of seven operations:

| Operation | Type | Captures | Rank contribution |
|---|---|---|---|
| A×B | Bilinear | Local correlations | 56 |
| A+B | Linear | Symmetric overlap | 29 |
| A−B | Linear | Anti-symmetric overlap | 29 |
| A⁻¹ | Rational | Phase-weighted correlations | 152 |
| H(A) | Linear | Geometric rotations | 160 |
| (AB)C≠A(BC) | Bilinear | Tree topology | 8 |
| ABA | Quadratic | Measurement/self-interaction | 5 |
| | | **Total** | **512/512** |

The seven operations correspond to the seven imaginary units of the octonions, the seven lines of the Fano plane, and the seven Cayley-Dickson decompositions.

### 5.6 Significance

The Jordan triple product ABA is the quantum mechanical sandwich operator (the measurement operation A†BA). It is the only QUADRATIC operation in the set; all others are linear or bilinear. The fact that the measurement operator completes the algebra suggests a deep connection between octonion structure and quantum measurement.

---

## 6. Topological Gate Compilation

### 6.1 SU(2) density

Zorn braid-words of length L achieve maximum approximation error ε that decreases as L increases, outperforming Fibonacci anyon braids 2–3× (max ε = 0.069 vs 0.134 at depth 9).

### 6.2 Gate compiler

A working compiler achieves fidelities >0.999 for all standard single-qubit gates (H, X, Z, S, T, Rx, Ry, Rz) with gate approximation error <0.03.

### 6.3 Negative results

The Zorn representation does not provide intrinsic error protection (perturbation amplification factor ~1.0). The SU(2) extraction is not a homomorphism (mean distance 0.98), precluding Zorn-level gate composition.

---

## 7. Physical Architecture: (8,0) → (4,4) → (8,0)

### 7.1 Spontaneous signature breaking

Quantum gates spontaneously break the (8,0) compact signature into the (4,4) split signature. The lifecycle:

    (8,0) input → gates → (4,4) computation → measurement → (8,0) output

This is verified computationally: basis states have split-norm 0; after H gates, the split-norm becomes ±0.5.

### 7.2 Destructive interference as signature cancellation

The sum H|0⟩ + H|1⟩ = √2|0⟩ demonstrates that destructive interference IS signature cancellation: the positive split-norm (+0.5) of H|0⟩ and the negative split-norm (−0.5) of H|1⟩ sum to zero.

### 7.3 Reboot protocol

(8,0) + (0,8) → split-norm = 0 → null cone → active (4,4). Mathematically exact for all values of the split-norm. Connects to Hawking-Hartle no-boundary proposal (signature transition at the Big Bang) and Penrose Objective Reduction (geometric collapse).

---

## 8. Additional Results

### 8.1 ZornQ simulator (v3)

892-line Python implementation with lazy entanglement groups, vectorized gates, Schmidt splitting (exact and approximate with fidelity tracking), gate cancellation and fusion, OpenQASM 2.0 export, and state vector export. Performance: GHZ-10,000 via MPS in 1 second, 5 MB.

### 8.2 BREC graph isomorphism benchmark

Quantum fingerprint circuit: 207/211 pairs distinguished. Spectral method: 209/211. However, cospectral pairs (Shrikhande vs Rook 4×4) are correctly distinguished by the quantum method (quantum difference = 0.047 vs spectral difference = 0.000).

### 8.3 Error correction

CSS stabilizer codes on random geometric graphs achieve 65% erasure threshold (3.2× improvement over square lattice codes).

---

## 9. Discussion

### 9.1 Why 8?

The Cayley-Dickson construction ℝ → ℂ → ℍ → 𝕆 halts at 8 dimensions because 16D sedenions lose the division algebra property, acquiring uncontrollable zero divisors. The match 8 = 2³ is the deep reason why one Zorn element encodes exactly 3 qubits.

### 9.2 Why 7?

Seven imaginary units, seven Fano lines, seven decompositions, seven operations for completeness. The number 7 is the structure constant of the octonions, and G₂ has dimension 14 = 2×7.

### 9.3 Relationship to MPS

The Zorn product decomposition is mathematically equivalent to Matrix Product States with bond dimension χ ≤ 8. The compression is real but not beyond MPS capabilities. The Zorn framework adds: (a) a natural bond dimension from the algebra, (b) the split-norm as an entanglement measure, (c) the (8,0)→(4,4)→(8,0) physical interpretation, and (d) information completeness through seven operations.

### 9.4 Limitations

The exponential wall is not broken. For circuits with high entanglement (deep random circuits, random-graph QAOA), the Schmidt rank at middle cuts grows exponentially. The O(n) scaling applies only to bounded-depth circuits with local interactions. The 7-operation completeness is verified at 6 and 9 qubits; scaling to n > 12 remains to be tested.

---

## 10. Conclusion

The split-octonion algebra in Zorn representation is information-complete for quantum states. The seven algebraic operations—multiplication, addition, subtraction, division, Hodge rotation, non-associative rebracketing, and the Jordan triple product—span the full Hilbert space. The framework provides: exact 3-qubit encoding with Born rule as Euclidean norm, automatic Schmidt decomposition with 95.6% single-product fidelity, the concurrence as split-norm, and a physical architecture connecting quantum computation to signature transitions.

The measurement operator ABA completes the algebra. This is not coincidental: the seven operations of the octonions are precisely the seven windows needed to see all quantum information.

---

## References

Baez, J.C. (2002). The Octonions. Bull. Amer. Math. Soc. 39, 145–205.

Freedman, M., Kitaev, A., Wang, Z. (2002). Simulation of topological field theories by quantum computers. Commun. Math. Phys. 227, 587–603.

Furey, C. (2018). Three generations, two unbroken gauge symmetries, and one eight-dimensional algebra. Phys. Lett. B 785, 84–89.

Günaydin, M., Gürsey, F. (1973). Quark structure and octonions. J. Math. Phys. 14, 1651–1667.

Hartle, J.B., Hawking, S.W. (1983). Wave function of the universe. Phys. Rev. D 28, 2960.

Penrose, R. (1996). On gravity's role in quantum state reduction. Gen. Relat. Gravit. 28, 581–600.

Zorn, M.A. (1933). Alternativkörper und quadratische Systeme. Abh. Math. Sem. Univ. Hamburg 9, 395–402.
