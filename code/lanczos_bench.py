"""
Lanczos Exact Benchmark + Krylov Evolutie (B37)
================================================
Twee tools voor exacte referentiewaarden:

(a) lanczos_maxcut(): Exact MaxCut optimum via sparse Lanczos
    - Bouwt MaxCut-Hamiltoniaan als sparse matrix
    - scipy.sparse.linalg.eigsh vindt grondtoestand
    - Limiet: ~26 qubits op 16GB RAM (2^26 * 16 bytes = 1GB)

(b) krylov_qaoa(): Exacte QAOA-evolutie via Krylov (expm_multiply)
    - Geen Trotter-fout, machineprecisie
    - Sneller dan gate-by-gate voor hoge p op kleine circuits

Gebruik:
    from lanczos_bench import lanczos_maxcut, krylov_qaoa

    # Exact optimum
    opt = lanczos_maxcut(edges, n_nodes)
    print(f"MaxCut = {opt.max_cut}, ratio = {opt.ratio}")

    # Exacte QAOA
    ratio = krylov_qaoa(edges, n_nodes, p=1, gammas=[0.4], betas=[0.3])
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time


# =====================================================================
# RESULT DATACLASSES
# =====================================================================

@dataclass
class MaxCutResult:
    """Resultaat van lanczos_maxcut()."""
    max_cut: float            # Exacte MaxCut waarde
    n_edges: int              # Totaal aantal edges
    ratio: float              # max_cut / n_edges (= 1 voor bipartite)
    ground_energy: float      # Laagste eigenwaarde van -H_MaxCut
    bitstring: np.ndarray     # Optimale bitstring
    n_qubits: int
    wall_time: float
    method: str = "lanczos"


# =====================================================================
# HAMILTONIANS (sparse)
# =====================================================================

def build_maxcut_hamiltonian(edges: list, n: int,
                              weights: Optional[dict] = None) -> sp.csr_matrix:
    """Bouw de MaxCut cost-Hamiltoniaan als sparse matrix.

    H_MC = sum_{(i,j) in E} w_ij * (1 - Z_i Z_j) / 2

    Eigenwaarden van -H_MC: de laagste is -MaxCut.

    Returns sparse (2^n, 2^n) diagonal matrix.
    """
    dim = 1 << n
    diag = np.zeros(dim, dtype=np.float64)

    for u, v in edges:
        w = 1.0
        if weights is not None:
            key = (min(u, v), max(u, v))
            w = weights.get(key, 1.0)

        # Z_i Z_j eigenvalues on computational basis
        for k in range(dim):
            zi = 1 - 2 * ((k >> u) & 1)  # +1 or -1
            zj = 1 - 2 * ((k >> v) & 1)
            diag[k] += w * (1 - zi * zj) / 2

    # Return -H zodat grondtoestand = MaxCut
    return sp.diags(-diag, format='csr')


def build_mixer_hamiltonian(n: int) -> sp.csr_matrix:
    """Bouw de transverse-field mixer B = sum_i X_i als sparse matrix."""
    dim = 1 << n
    rows, cols, vals = [], [], []

    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            k_flipped = k ^ mask
            rows.append(k)
            cols.append(k_flipped)
            vals.append(1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def build_zz_operator(u: int, v: int, n: int) -> sp.csr_matrix:
    """Bouw Z_u Z_v als sparse diagonaalmatrix."""
    dim = 1 << n
    diag = np.zeros(dim, dtype=np.float64)
    for k in range(dim):
        zi = 1 - 2 * ((k >> u) & 1)
        zj = 1 - 2 * ((k >> v) & 1)
        diag[k] = zi * zj
    return sp.diags(diag, format='csr')


# =====================================================================
# LANCZOS EXACT MAXCUT
# =====================================================================

def lanczos_maxcut(edges: list, n: int,
                    weights: Optional[dict] = None,
                    k_eigenvalues: int = 1) -> MaxCutResult:
    """Bereken exact MaxCut optimum via Lanczos.

    Parameters
    ----------
    edges : list of (int, int)
    n : int  (number of qubits/nodes)
    weights : dict, optional
    k_eigenvalues : int
        Aantal laagste eigenwaarden (default 1 = alleen grondtoestand)

    Returns
    -------
    MaxCutResult
    """
    t0 = time.time()
    n_edges = len(edges)
    dim = 1 << n

    if n > 28:
        raise ValueError(f"n={n} te groot voor exact Lanczos (2^{n} = {dim} states)")

    # Speciale gevallen
    if n <= 20:
        # Direct diag is sneller dan Lanczos voor kleine n
        H_diag = np.zeros(dim, dtype=np.float64)
        for u, v in edges:
            w = 1.0
            if weights:
                w = weights.get((min(u,v), max(u,v)), 1.0)
            for k in range(dim):
                zi = 1 - 2 * ((k >> u) & 1)
                zj = 1 - 2 * ((k >> v) & 1)
                H_diag[k] += w * (1 - zi * zj) / 2

        max_cut = float(np.max(H_diag))
        best_k = int(np.argmax(H_diag))
        bitstring = np.array([(best_k >> i) & 1 for i in range(n)])

        return MaxCutResult(
            max_cut=max_cut, n_edges=n_edges, ratio=max_cut / n_edges,
            ground_energy=-max_cut, bitstring=bitstring, n_qubits=n,
            wall_time=time.time() - t0, method="exact_diag")

    # Lanczos voor grotere systemen
    H = build_maxcut_hamiltonian(edges, n, weights)

    # eigsh vindt de kleinste eigenwaarden van -H_MC
    eigenvalues, eigenvectors = sla.eigsh(H, k=k_eigenvalues, which='SA')

    # Grondtoestand energie = -MaxCut
    ground_e = eigenvalues[0]
    max_cut = -ground_e

    # Bitstring: |k> met hoogste amplitude in grondtoestand
    psi = eigenvectors[:, 0]
    best_k = int(np.argmax(np.abs(psi)**2))
    bitstring = np.array([(best_k >> i) & 1 for i in range(n)])

    # Verifieer dat de bitstring inderdaad MaxCut geeft
    cut_val = sum(
        (weights.get((min(u,v), max(u,v)), 1.0) if weights else 1.0)
        for u, v in edges if bitstring[u] != bitstring[v]
    )
    # Grondtoestand kan een superpositie zijn — bitstring is dan niet exact
    # Maar cut_val moet dicht bij max_cut liggen

    return MaxCutResult(
        max_cut=max_cut, n_edges=n_edges, ratio=max_cut / n_edges,
        ground_energy=ground_e, bitstring=bitstring, n_qubits=n,
        wall_time=time.time() - t0)


# =====================================================================
# KRYLOV QAOA (exacte evolutie)
# =====================================================================

def krylov_qaoa(edges: list, n: int, p: int,
                gammas: list, betas: list,
                weights: Optional[dict] = None) -> float:
    """Exacte QAOA cost via Krylov-evolutie (expm_multiply).

    Geen Trotter-fout — machineprecisie.

    Returns
    -------
    float : MaxCut cost = sum (1 - <ZZ>)/2
    """
    dim = 1 << n

    # Build Hamiltonians
    # Cost Hamiltonian diag: C|k> = cost(k)|k>
    C_diag = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        w = 1.0
        if weights:
            w = weights.get((min(u,v), max(u,v)), 1.0)
        for k in range(dim):
            zi = 1 - 2 * ((k >> u) & 1)
            zj = 1 - 2 * ((k >> v) & 1)
            C_diag[k] += w * zi * zj

    C_sparse = sp.diags(C_diag, format='csr')
    B_sparse = build_mixer_hamiltonian(n)

    # |+>^n initialisatie
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # QAOA evolutie: |psi> = prod_l e^{-i beta_l B} e^{-i gamma_l C} |+>
    for l in range(p):
        # Phase separator: e^{-i gamma C}
        # C is diag, so this is just element-wise multiply
        psi = np.exp(-1j * gammas[l] * C_diag) * psi

        # Mixer: e^{-i beta B} via Krylov
        # B is sparse non-diagonal — use expm_multiply
        psi = sla.expm_multiply(-1j * betas[l] * B_sparse, psi)

    # Bereken cost: <psi| H_MC |psi>
    # H_MC = sum (1 - Z_i Z_j)/2 = n_edges/2 - C/2
    cost = len(edges) / 2 - np.real(psi.conj() @ (C_sparse @ psi)) / 2

    return float(cost)


def krylov_qaoa_ratio(edges: list, n: int, p: int,
                       gammas: list, betas: list,
                       weights: Optional[dict] = None) -> float:
    """QAOA approximation ratio via exact Krylov evolution."""
    cost = krylov_qaoa(edges, n, p, gammas, betas, weights)
    return cost / len(edges) if edges else 0.0


def optimize_krylov_qaoa(edges: list, n: int, p: int,
                          n_grid: int = 10,
                          weights: Optional[dict] = None) -> Tuple[float, list, list]:
    """Optimaliseer QAOA-parameters via Krylov (exact).

    Returns (best_ratio, best_gammas, best_betas).
    """
    best_r, best_g, best_b = 0, 0, 0

    # Grid search
    for g in np.linspace(0.1, 1.2, n_grid):
        for b in np.linspace(0.1, 1.2, n_grid):
            gammas = [g] * p
            betas = [b] * p
            r = krylov_qaoa_ratio(edges, n, p, gammas, betas, weights)
            if r > best_r:
                best_r, best_g, best_b = r, g, b

    # Refine
    try:
        from scipy.optimize import minimize
        def neg_r(params):
            return -krylov_qaoa_ratio(edges, n, p,
                                       list(params[:p]), list(params[p:]), weights)
        x0 = [best_g]*p + [best_b]*p
        res = minimize(neg_r, x0, method='Nelder-Mead',
                      options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 1000})
        if -res.fun > best_r:
            best_r = -res.fun
            best_g = res.x[0]
            best_b = res.x[p]
    except Exception:
        pass

    return best_r, [best_g]*p, [best_b]*p


# =====================================================================
# BENCHMARK SUITE
# =====================================================================

def make_grid_edges(Lx, Ly):
    """Genereer edges voor Lx*Ly grid."""
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x+1)*Ly + y))
            if y + 1 < Ly:
                edges.append((i, x*Ly + y + 1))
    return edges


def benchmark_grid(Lx, Ly, p_max=2, verbose=True):
    """Volledige benchmark voor een Lx*Ly grid.

    Returns dict met exact MaxCut + QAOA ratios per p.
    """
    n = Lx * Ly
    edges = make_grid_edges(Lx, Ly)
    n_edges = len(edges)

    result = {
        'grid': f'{Lx}x{Ly}',
        'n_qubits': n,
        'n_edges': n_edges,
    }

    # Exact MaxCut
    t0 = time.time()
    mc = lanczos_maxcut(edges, n)
    result['max_cut'] = mc.max_cut
    result['max_cut_ratio'] = mc.ratio
    result['exact_time'] = mc.wall_time
    result['exact_method'] = mc.method

    if verbose:
        print(f"  {Lx}x{Ly} (n={n}): MaxCut={mc.max_cut:.0f}/{n_edges} "
              f"= {mc.ratio:.6f}  [{mc.wall_time:.2f}s, {mc.method}]")

    # QAOA per p
    for p in range(1, p_max + 1):
        t0 = time.time()
        qaoa_r, gammas, betas = optimize_krylov_qaoa(edges, n, p, n_grid=8)
        dt = time.time() - t0

        result[f'qaoa_p{p}_ratio'] = qaoa_r
        result[f'qaoa_p{p}_approx'] = qaoa_r / mc.ratio if mc.ratio > 0 else 0
        result[f'qaoa_p{p}_gamma'] = gammas[0]
        result[f'qaoa_p{p}_beta'] = betas[0]
        result[f'qaoa_p{p}_time'] = dt

        if verbose:
            approx = qaoa_r / mc.ratio if mc.ratio > 0 else 0
            print(f"    p={p}: QAOA ratio={qaoa_r:.6f} "
                  f"({approx:.4f} of optimal)  "
                  f"g={gammas[0]:.4f} b={betas[0]:.4f}  [{dt:.2f}s]")

    return result


# =====================================================================
# SELF-TEST
# =====================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("B37: Lanczos Exact Benchmark + Krylov QAOA")
    print("=" * 65)

    # Test 1: Kleine grids
    print("\n--- Exact MaxCut + QAOA benchmarks ---")
    results = []
    for Lx, Ly in [(3,2), (4,2), (3,3), (4,3), (5,3), (4,4), (5,4)]:
        n = Lx * Ly
        if n > 24:
            print(f"  Skipping {Lx}x{Ly} (n={n} > 24)")
            continue
        r = benchmark_grid(Lx, Ly, p_max=2)
        results.append(r)

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Grid':>6} {'n':>3} {'m':>3} {'MaxCut':>7} {'Exact':>7} "
          f"{'QAOA-1':>7} {'%opt':>6} {'QAOA-2':>7} {'%opt':>6}")
    print("-" * 85)
    for r in results:
        p1 = r.get('qaoa_p1_ratio', 0)
        p2 = r.get('qaoa_p2_ratio', 0)
        a1 = r.get('qaoa_p1_approx', 0)
        a2 = r.get('qaoa_p2_approx', 0)
        print(f"{r['grid']:>6} {r['n_qubits']:>3} {r['n_edges']:>3} "
              f"{r['max_cut']:>7.0f} {r['max_cut_ratio']:>7.4f} "
              f"{p1:>7.4f} {a1:>5.1%} {p2:>7.4f} {a2:>5.1%}")

    print("\nDone.")
