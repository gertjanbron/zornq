#!/usr/bin/env python3
"""
multi_domain_poc.py - B132 Multi-Domain Proof-of-Concept

Demonstreert ZornQ-stack buiten MaxCut met 3 domeinen:
  1. Condensed matter: Heisenberg XXX grondtoestand (VQE + exact diag)
  2. Moleculaire simulatie: H2 molecuul (Jordan-Wigner + VQE + FCI)
  3. PDE: 1D kwantum deeltje / golfvergelijking (Trotter + expm referentie)

Elke demo vergelijkt het quantum-circuit resultaat met een klassieke referentie
en genereert een QualityCertificate.

Gebruik:
    from multi_domain_poc import run_all_demos
    results = run_all_demos(verbose=True)

    # Of individueel
    from multi_domain_poc import demo_condensed_matter, demo_molecular, demo_pde
    result = demo_condensed_matter(n_sites=4, depth=6, verbose=True)

Author: ZornQ project
Date: 16 april 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from typing import Dict, List, Tuple, Optional, Any
import time

from hamiltonian_compiler import Hamiltonian
from circuit_interface import Circuit, Gates, Observable, run_circuit
from quality_certificate import (
    certify_energy, certify_circuit_result, QualityCertificate,
    CertificateLevel
)


# =====================================================================
# HULPFUNCTIES
# =====================================================================

def exact_ground_state(hamiltonian, n_qubits=None):
    """Bereken exact grondtoestandsenergie via volledige diagonalisatie.

    Args:
        hamiltonian: Hamiltonian object
        n_qubits: override (anders van hamiltonian)

    Returns:
        (gs_energy, gs_state) tuple
    """
    n = n_qubits or hamiltonian.n_qubits
    dim = 1 << n
    H_mat = np.zeros((dim, dim), dtype=np.complex128)

    for coeff, pauli in hamiltonian.terms:
        # Bouw Pauli tensor product matrix
        term_mat = _pauli_string_to_matrix(pauli, n)
        H_mat += coeff * term_mat

    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    gs_energy = float(eigenvalues[0].real)
    gs_state = eigenvectors[:, 0]
    return gs_energy, gs_state


def _pauli_string_to_matrix(pauli_dict, n_qubits):
    """Converteer Pauli string dict naar volledige 2^n x 2^n matrix.

    Qubit ordering: qubit 0 = LSB (standaard conventie, consistent met
    circuit_interface.py state vector). Kronecker product is reversed:
    tensor product = q_{n-1} otimes ... otimes q_0.
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    if not pauli_dict:
        return np.eye(1 << n_qubits, dtype=np.complex128)

    # Bouw van hoog naar laag qubit: q_{n-1} otimes ... otimes q_0
    result = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits - 1, -1, -1):
        op = pauli_map.get(pauli_dict.get(q, 'I'))
        result = np.kron(result, op)
    return result


def vqe_optimize(hamiltonian, depth=4, n_restarts=3, maxiter=200,
                 verbose=False):
    """Draai VQE optimalisatie met hardware-efficient ansatz.

    Args:
        hamiltonian: Hamiltonian object
        depth: ansatz diepte
        n_restarts: aantal random restarts
        maxiter: max iteraties per restart
        verbose: print voortgang

    Returns:
        dict met 'energy', 'params', 'circuit', 'result', 'n_evals'
    """
    n = hamiltonian.n_qubits
    obs = hamiltonian.to_observable()
    n_params = n * 2 * (depth + 1) + (n - 1) * depth

    best_energy = np.inf
    best_params = None
    total_evals = 0

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + restart)
        x0 = rng.uniform(0, 2 * np.pi, n_params)

        eval_count = [0]

        def cost(params):
            eval_count[0] += 1
            qc = Circuit.hardware_efficient(n, depth=depth, params=params)
            result = run_circuit(qc, observables={'E': obs}, backend='statevector')
            return result['observables']['E']

        res = minimize(cost, x0, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 0.5})

        total_evals += eval_count[0]
        if verbose:
            print("  Restart %d: E=%.6f (%d evals)" % (
                restart + 1, res.fun, eval_count[0]))

        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x.copy()

    # Draai finale circuit
    qc = Circuit.hardware_efficient(n, depth=depth, params=best_params)
    result = run_circuit(qc, observables={'E': obs}, backend='statevector')

    return {
        'energy': best_energy,
        'params': best_params,
        'circuit': qc,
        'result': result,
        'n_evals': total_evals,
    }


# =====================================================================
# DEMO 1: CONDENSED MATTER - Heisenberg XXX Grondtoestand
# =====================================================================

def demo_condensed_matter(n_sites=4, depth=4, n_restarts=3,
                          maxiter=200, verbose=True):
    """Heisenberg XXX model grondtoestand via VQE.

    Vergelijkt met exact diagonalisatie.

    Args:
        n_sites: aantal spin-1/2 sites (qubits)
        depth: VQE ansatz diepte
        n_restarts: random restarts voor optimizer
        maxiter: max iteraties per restart
        verbose: print resultaten

    Returns:
        dict met resultaten en certificaat
    """
    t0 = time.time()
    if verbose:
        print("=" * 60)
        print("  DEMO 1: Condensed Matter - Heisenberg XXX")
        print("  %d sites, VQE depth=%d" % (n_sites, depth))
        print("=" * 60)

    # Bouw Hamiltonian
    H = Hamiltonian.heisenberg_xxx(n_sites, J=1.0)
    if verbose:
        print("  Hamiltonian: %s (%d termen)" % (H.name, H.n_terms))

    # Exact referentie
    if verbose:
        print("  Exacte diagonalisatie...")
    gs_energy, gs_state = exact_ground_state(H)
    if verbose:
        print("  Exact GS energie: %.6f" % gs_energy)

    # VQE
    if verbose:
        print("  VQE optimalisatie (depth=%d, %d restarts)..." % (
            depth, n_restarts))
    vqe = vqe_optimize(H, depth=depth, n_restarts=n_restarts,
                        maxiter=maxiter, verbose=verbose)
    vqe_energy = vqe['energy']

    # Fidelity met exact GS
    vqe_state = vqe['result']['state']
    fidelity = float(abs(np.dot(gs_state.conj(), vqe_state)) ** 2)

    # Relatieve fout
    rel_error = abs(vqe_energy - gs_energy) / abs(gs_energy) if gs_energy != 0 else abs(vqe_energy - gs_energy)

    # Per-site energie
    e_per_site_exact = gs_energy / n_sites
    e_per_site_vqe = vqe_energy / n_sites

    # Certificeer
    cert = certify_energy(vqe_energy, H, circuit_result=vqe['result'],
                          exact_gs=gs_energy)

    elapsed = time.time() - t0

    if verbose:
        print("\n  --- Resultaten ---")
        print("  VQE energie:     %.6f" % vqe_energy)
        print("  Exact energie:   %.6f" % gs_energy)
        print("  Relatieve fout:  %.2e" % rel_error)
        print("  Fidelity:        %.6f" % fidelity)
        print("  E/site (VQE):    %.6f" % e_per_site_vqe)
        print("  E/site (exact):  %.6f" % e_per_site_exact)
        print("  Circuit evals:   %d" % vqe['n_evals'])
        print("  Certificaat:     %s" % cert.level.value)
        print("  Tijd:            %.2fs" % elapsed)

    return {
        'domain': 'condensed_matter',
        'model': 'Heisenberg XXX',
        'n_sites': n_sites,
        'n_qubits': n_sites,
        'vqe_energy': vqe_energy,
        'exact_energy': gs_energy,
        'relative_error': rel_error,
        'fidelity': fidelity,
        'e_per_site_vqe': e_per_site_vqe,
        'e_per_site_exact': e_per_site_exact,
        'n_evals': vqe['n_evals'],
        'depth': depth,
        'certificate': cert,
        'time': elapsed,
    }


# =====================================================================
# DEMO 2: MOLECULAR - H2 Molecuul
# =====================================================================

def h2_qubit_hamiltonian(bond_length=0.74):
    """Qubit Hamiltonian voor H2 in STO-3G basis via Jordan-Wigner.

    Directe qubit-operator coefficienten (inclusief nucleaire repulsie).
    Referentie: O'Malley et al., PRX 6, 031007 (2016);
                Kandala et al., Nature 549, 242 (2017).

    4 qubits: 2 spatial orbitals x 2 spins.

    Args:
        bond_length: H-H afstand in Angstrom (momenteel alleen 0.74)

    Returns:
        (hamiltonian, nuclear_repulsion, n_qubits)
    """
    if abs(bond_length - 0.74) > 0.01:
        raise ValueError("H2 qubit Hamiltonian alleen beschikbaar voor R=0.74 A")

    # Coefficienten berekend uit STO-3G integralen via exacte JW decompositie.
    # Qubit ordering: q0=MO1-up, q1=MO1-down, q2=MO2-up, q3=MO2-down
    # NB: g0 bevat NIET de nucleaire repulsie (die wordt apart meegegeven).
    nuclear_repulsion = 0.7137539936876182

    terms = [
        # Identiteit (elektronisch, zonder E_nuc)
        (-0.8129230187, {}),
        # Enkele Z (qubit 0,1 = bonding MO; 2,3 = antibonding MO)
        (+0.1713584185, {0: 'Z'}),
        (+0.1713584185, {1: 'Z'}),
        (-0.2226344171, {2: 'Z'}),
        (-0.2226344171, {3: 'Z'}),
        # ZZ interacties
        (+0.1686265563, {0: 'Z', 1: 'Z'}),
        (+0.1204624594, {0: 'Z', 2: 'Z'}),
        (+0.1657843526, {1: 'Z', 2: 'Z'}),
        (+0.1657843526, {0: 'Z', 3: 'Z'}),
        (+0.1204624594, {1: 'Z', 3: 'Z'}),
        (+0.1743548356, {2: 'Z', 3: 'Z'}),
        # Dubbele excitatie (JW): X0X1Y2Y3 + Y0Y1X2X3 - X0Y1Y2X3 - Y0X1X2Y3
        (-0.0453218932, {0: 'X', 1: 'X', 2: 'Y', 3: 'Y'}),
        (-0.0453218932, {0: 'Y', 1: 'Y', 2: 'X', 3: 'X'}),
        (+0.0453218932, {0: 'X', 1: 'Y', 2: 'Y', 3: 'X'}),
        (+0.0453218932, {0: 'Y', 1: 'X', 2: 'X', 3: 'Y'}),
    ]

    H = Hamiltonian(terms, n_qubits=4, name="H2-STO3G(R=%.2f)" % bond_length)
    return H, nuclear_repulsion, 4


def demo_molecular(bond_length=0.74, depth=4, n_restarts=3,
                   maxiter=200, verbose=True):
    """H2 molecuul grondtoestand via Jordan-Wigner + VQE.

    Vergelijkt met FCI (exact diagonalisatie in de qubit-ruimte).

    Args:
        bond_length: H-H afstand in Angstrom
        depth: VQE ansatz diepte
        n_restarts: random restarts
        maxiter: max iteraties per restart
        verbose: print resultaten

    Returns:
        dict met resultaten en certificaat
    """
    t0 = time.time()
    if verbose:
        print("=" * 60)
        print("  DEMO 2: Molecular - H2 (STO-3G)")
        print("  Bond length: %.2f A, VQE depth=%d" % (bond_length, depth))
        print("=" * 60)

    # Qubit Hamiltonian (directe coefficienten, incl. E_nuc in constante term)
    H, E_nuc, n_qubits = h2_qubit_hamiltonian(bond_length)
    if verbose:
        print("  Qubits: %d (2 MOs x 2 spins)" % n_qubits)
        print("  Nuclear repulsion: %.6f Ha" % E_nuc)
        print("  Qubit Hamiltonian: %d termen" % H.n_terms)

    # Exact referentie (FCI in qubit ruimte)
    # H bevat elektronische energie; E_nuc wordt apart opgeteld
    if verbose:
        print("  Exacte diagonalisatie (FCI)...")
    gs_electronic, gs_state = exact_ground_state(H, n_qubits)
    gs_total = gs_electronic + E_nuc
    if verbose:
        print("  FCI elektronisch: %.6f Ha" % gs_electronic)
        print("  FCI totaal:       %.6f Ha" % gs_total)

    # Referentiewaarde H2 STO-3G bij R=0.74: ~-1.137 Ha
    ref_energy = -1.1373
    if verbose:
        print("  Literatuur ref:   ~%.4f Ha" % ref_energy)

    # VQE
    if verbose:
        print("  VQE optimalisatie...")
    vqe = vqe_optimize(H, depth=depth, n_restarts=n_restarts,
                        maxiter=maxiter, verbose=verbose)
    vqe_electronic = vqe['energy']
    vqe_total = vqe_electronic + E_nuc

    # Fidelity
    vqe_state = vqe['result']['state']
    fidelity = float(abs(np.dot(gs_state.conj(), vqe_state)) ** 2)

    # Chemische nauwkeurigheid: 1 kcal/mol = 0.0016 Ha
    chem_accuracy = 0.0016  # Ha
    error_ha = abs(vqe_total - gs_total)
    within_chem_acc = error_ha < chem_accuracy

    # Certificeer
    cert = certify_energy(vqe_electronic, H,
                          circuit_result=vqe['result'],
                          exact_gs=gs_electronic)

    elapsed = time.time() - t0

    if verbose:
        print("\n  --- Resultaten ---")
        print("  VQE totaal:      %.6f Ha" % vqe_total)
        print("  FCI totaal:      %.6f Ha" % gs_total)
        print("  Fout:            %.6f Ha (%.4f mHa)" % (
            error_ha, error_ha * 1000))
        print("  Chem. accuracy:  %s (< 1.6 mHa)" % (
            "JA" if within_chem_acc else "NEE"))
        print("  Fidelity:        %.6f" % fidelity)
        print("  Circuit evals:   %d" % vqe['n_evals'])
        print("  Certificaat:     %s" % cert.level.value)
        print("  Tijd:            %.2fs" % elapsed)

    return {
        'domain': 'molecular',
        'model': 'H2 STO-3G',
        'bond_length': bond_length,
        'n_qubits': n_qubits,
        'vqe_energy': vqe_total,
        'fci_energy': gs_total,
        'nuclear_repulsion': E_nuc,
        'error_ha': error_ha,
        'error_mha': error_ha * 1000,
        'within_chemical_accuracy': within_chem_acc,
        'fidelity': fidelity,
        'n_evals': vqe['n_evals'],
        'depth': depth,
        'certificate': cert,
        'time': elapsed,
    }


# =====================================================================
# DEMO 3: PDE - 1D Kwantum Deeltje op Rooster
# =====================================================================

def lattice_kinetic_hamiltonian(n_sites, dx=1.0, m=1.0, periodic=False):
    """Kinetische energie op 1D rooster via eindige-differentie.

    T = -(1/2m) * d^2/dx^2  -->  finite difference:
    T_ij = (1/2m*dx^2) * (2*delta_ij - delta_{i,j+1} - delta_{i,j-1})

    Equivalent aan tight-binding model: -t * sum (|i><j| + h.c.) + 2t * I
    met t = 1/(2*m*dx^2).

    Args:
        n_sites: roosterpunten (= qubits)
        dx: roosterspacing
        m: massa
        periodic: periodieke randvoorwaarden

    Returns:
        Hamiltonian object
    """
    t_hop = 1.0 / (2.0 * m * dx * dx)
    terms = []

    # Diagonaal: 2t per site
    for i in range(n_sites):
        terms.append((2.0 * t_hop, {i: 'Z'}))
        # 2t * I = 2t * (I) -> 2t * (I+Z)/2 + 2t * (I-Z)/2 = 2t * I
        # Maar we moeten Z gebruiken voor diag. Correct:
        # n_i = (I - Z_i)/2, dus |i><i| = n_i
        # We willen 2t * |i><i| = 2t * (I - Z_i)/2 = t*(I - Z_i)
        # Eigenlijk: T_ii = 2t, T_ij (adj) = -t
        # Hamiltonian = sum_i 2t |i><i| - t sum_<ij> (|i><j| + |j><i|)
        # = t * sum_i (I - Z_i) - t * sum_<ij> hopping
        pass

    # Gebruik eigenlijk de Heisenberg XY model structuur:
    # |i><j| + |j><i| = (X_i X_j + Y_i Y_j) / 2 (voor adjacent sites)
    # Maar dit is single-excitation subspace...
    # Nee - we werken in het computational basis = positie-basis.
    # Dit is een tight-binding model.

    # Correcte mapping: tight-binding op n_sites qubits
    # In de 1-excitation sector: |i> = |0...010...0> (qubit i = 1, rest 0)
    # Hopping: (X_i X_j + Y_i Y_j)/2
    # On-site: Z_i term
    terms = []

    # Diagonale bijdrage: 2t * n_i = 2t * (I - Z_i)/2 = t * (I - Z_i)
    for i in range(n_sites):
        terms.append((t_hop, {}))  # constante t per site
        terms.append((-t_hop, {i: 'Z'}))

    # Hopping: -t * (|i><j| + |j><i|) = -t * (XX + YY)/2
    for i in range(n_sites - 1 + (1 if periodic else 0)):
        j = (i + 1) % n_sites
        terms.append((-t_hop / 2.0, {i: 'X', j: 'X'}))
        terms.append((-t_hop / 2.0, {i: 'Y', j: 'Y'}))

    H = Hamiltonian(terms, n_sites,
                    "Lattice-T(n=%d,dx=%.2f)" % (n_sites, dx))
    H.simplify()
    return H


def lattice_potential_hamiltonian(n_sites, potential_fn, dx=1.0):
    """Potentiele energie op rooster.

    V(x_i) * |i><i| = V(x_i) * n_i = V(x_i) * (I - Z_i)/2

    Args:
        n_sites: roosterpunten
        potential_fn: callable V(x) -> float
        dx: roosterspacing

    Returns:
        Hamiltonian object
    """
    terms = []
    x0 = -(n_sites - 1) * dx / 2  # centreer rooster

    for i in range(n_sites):
        x = x0 + i * dx
        V = potential_fn(x)
        if abs(V) > 1e-15:
            # V * n_i = V * (I - Z_i)/2
            terms.append((V / 2.0, {}))
            terms.append((-V / 2.0, {i: 'Z'}))

    H = Hamiltonian(terms, n_sites, "Lattice-V(n=%d)" % n_sites)
    return H


def demo_pde(n_sites=6, t_evolve=1.0, steps=10, trotter_order=2,
             dx=1.0, verbose=True):
    """1D kwantum deeltje in harmonisch potentiaal via Trotter evolutie.

    Simuleert tijdsevolutie: |psi(t)> = exp(-iHt)|psi(0)>
    Vergelijkt Trotter-circuit met exacte matrix-exponentiaal.

    Het deeltje start op site n/2 (gaussisch golfpakket) en
    evolueert onder H = T + V met V = 0.5*k*x^2 (harmonisch).

    Args:
        n_sites: roosterpunten (qubits)
        t_evolve: evolutietijd
        steps: Trotter stappen
        trotter_order: 1, 2, of 4
        dx: roosterspacing
        verbose: print resultaten

    Returns:
        dict met resultaten en certificaat
    """
    t0 = time.time()
    if verbose:
        print("=" * 60)
        print("  DEMO 3: PDE - 1D Quantum Particle (Harmonic)")
        print("  %d sites, t=%.2f, Trotter-%d (%d stappen)" % (
            n_sites, t_evolve, trotter_order, steps))
        print("=" * 60)

    # Parameters
    m = 1.0
    k_spring = 0.5  # veerconstante

    # Hamiltonian
    H_T = lattice_kinetic_hamiltonian(n_sites, dx=dx, m=m)
    H_V = lattice_potential_hamiltonian(n_sites,
                                        potential_fn=lambda x: 0.5 * k_spring * x * x,
                                        dx=dx)
    H = H_T + H_V
    H.simplify()
    H.name = "Particle-1D(n=%d)" % n_sites

    if verbose:
        print("  Hamiltonian: %d termen" % H.n_terms)

    # Initialtoestand: deeltje op middelste site
    # In 1-excitation sector: |psi_0> = |0...010...0> (site n/2)
    mid = n_sites // 2

    # --- Exacte referentie: matrix exponentiaal ---
    if verbose:
        print("  Exacte tijdsevolutie (matrix exp)...")
    H_mat = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128)
    for coeff, pauli in H.terms:
        H_mat += coeff * _pauli_string_to_matrix(pauli, n_sites)

    # Initialtoestand: |0...010...0> met qubit mid=1
    psi0 = np.zeros(1 << n_sites, dtype=np.complex128)
    psi0[1 << mid] = 1.0  # qubit mid is |1>

    # Exact evolutie
    U_exact = expm(-1j * H_mat * t_evolve)
    psi_exact = U_exact @ psi0

    # Site-bezettingen (exact)
    occ_exact = np.zeros(n_sites)
    for i in range(n_sites):
        # <n_i> = <psi| (I-Z_i)/2 |psi>
        Z_mat = _pauli_string_to_matrix({i: 'Z'}, n_sites)
        occ_exact[i] = float(0.5 * (1.0 - (psi_exact.conj() @ Z_mat @ psi_exact).real))

    if verbose:
        print("  Exact bezetting: %s" % np.array2string(
            occ_exact, precision=4, suppress_small=True))

    # --- Trotter circuit ---
    if verbose:
        print("  Trotter-%d compilatie (%d stappen)..." % (
            trotter_order, steps))
    qc = H.trotter(t_evolve, steps=steps, order=trotter_order)

    # Voeg initialtoestand toe: X op qubit mid
    init_qc = Circuit(n_sites, name="init+trotter")
    init_qc.x(mid)  # |0> -> |1> op site mid
    # Append trotter gates
    for op in qc.ops:
        init_qc.ops.append(op)
    init_qc.metadata = qc.metadata.copy()

    if verbose:
        print("  Circuit: %d gates, depth %d" % (len(init_qc), init_qc.depth()))

    # Observables: bezetting per site
    obs = {}
    for i in range(n_sites):
        obs['n_%d' % i] = Observable([(0.5, {}), (-0.5, {i: 'Z'})])

    result = run_circuit(init_qc, observables=obs, backend='statevector')
    psi_trotter = result['state']

    # Site-bezettingen (Trotter)
    occ_trotter = np.array([result['observables']['n_%d' % i]
                             for i in range(n_sites)])

    if verbose:
        print("  Trotter bezetting: %s" % np.array2string(
            occ_trotter, precision=4, suppress_small=True))

    # Fidelity
    fidelity = float(abs(np.dot(psi_exact.conj(), psi_trotter)) ** 2)

    # RMS fout in bezettingen
    occ_rmse = float(np.sqrt(np.mean((occ_exact - occ_trotter) ** 2)))

    # Norm check
    norm_trotter = float(np.linalg.norm(psi_trotter))
    norm_exact = float(np.linalg.norm(psi_exact))

    # Positie-verwachtingswaarde
    positions = np.array([-(n_sites - 1) * dx / 2.0 + i * dx
                          for i in range(n_sites)])
    x_mean_exact = float(np.dot(occ_exact, positions))
    x_mean_trotter = float(np.dot(occ_trotter, positions))

    # Certificeer
    cert = certify_circuit_result(result, hamiltonian=H, circuit=init_qc,
                                   reference_state=psi_exact)

    elapsed = time.time() - t0

    if verbose:
        print("\n  --- Resultaten ---")
        print("  Fidelity:        %.6f" % fidelity)
        print("  Bezetting RMSE:  %.2e" % occ_rmse)
        print("  <x> Trotter:     %.4f" % x_mean_trotter)
        print("  <x> exact:       %.4f" % x_mean_exact)
        print("  Norm Trotter:    %.6f" % norm_trotter)
        print("  Certificaat:     %s" % cert.level.value)
        print("  Tijd:            %.2fs" % elapsed)

    return {
        'domain': 'pde',
        'model': '1D Quantum Particle (Harmonic)',
        'n_sites': n_sites,
        'n_qubits': n_sites,
        't_evolve': t_evolve,
        'trotter_order': trotter_order,
        'steps': steps,
        'fidelity': fidelity,
        'occ_rmse': occ_rmse,
        'occ_exact': occ_exact,
        'occ_trotter': occ_trotter,
        'x_mean_exact': x_mean_exact,
        'x_mean_trotter': x_mean_trotter,
        'n_gates': len(init_qc),
        'depth': init_qc.depth(),
        'certificate': cert,
        'time': elapsed,
    }


# =====================================================================
# RUN ALLE DEMOS
# =====================================================================

def run_all_demos(verbose=True):
    """Draai alle 3 demo-domeinen en geef samenvatting.

    Returns:
        dict met resultaten per domein
    """
    t0 = time.time()

    results = {}

    # Demo 1: Condensed Matter
    results['condensed_matter'] = demo_condensed_matter(
        n_sites=4, depth=4, n_restarts=3, maxiter=200, verbose=verbose)

    if verbose:
        print()

    # Demo 2: Molecular
    results['molecular'] = demo_molecular(
        bond_length=0.74, depth=4, n_restarts=3, maxiter=200, verbose=verbose)

    if verbose:
        print()

    # Demo 3: PDE
    results['pde'] = demo_pde(
        n_sites=6, t_evolve=1.0, steps=10, trotter_order=2, verbose=verbose)

    total_time = time.time() - t0

    if verbose:
        print()
        print("=" * 60)
        print("  SAMENVATTING - Multi-Domain PoC")
        print("=" * 60)
        for domain, r in results.items():
            cert = r['certificate']
            if domain == 'condensed_matter':
                print("  CM  (Heisenberg): E_err=%.2e, fid=%.4f, cert=%s" % (
                    r['relative_error'], r['fidelity'], cert.level.value))
            elif domain == 'molecular':
                print("  MOL (H2 STO-3G):  err=%.3f mHa, fid=%.4f, cert=%s" % (
                    r['error_mha'], r['fidelity'], cert.level.value))
            elif domain == 'pde':
                print("  PDE (1D Particle): fid=%.4f, RMSE=%.2e, cert=%s" % (
                    r['fidelity'], r['occ_rmse'], cert.level.value))
        print("  Totale tijd: %.2fs" % total_time)

    results['total_time'] = total_time
    return results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    run_all_demos(verbose=True)
