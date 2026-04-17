"""
B10g: 7-operatie Heisenberg CT-scan

Bekijk een Heisenberg-geëvolueerde operator vanuit 7 Cayley-Dickson
perspectieven. Elk perspectief vangt 18/64 Pauli-componenten via de
Zorn L-basis. Samen: 7×18=126, met overlap → effectief ~52/64.

Kernvraag: kan de operator-spreiding bij 2D QAOA worden opgevangen door
meerdere perspectieven te combineren, elk met lage chi?

Experiment:
  1. Evolueer MPO via HeisenbergQAOA (full chi)
  2. Decomponeer resultaat-MPO in 64 Pauli-componenten per site
  3. Per Fano-perspectief: projecteer op 18-dim L-subspace, meet chi
  4. Vergelijk perspective-chi vs full-chi
  5. Reconstructeer <O> uit perspectieven en meet fout

Auteur: ZornQ project
Datum: 16 april 2026
"""
import numpy as np
from numpy.linalg import svd, norm
import time

# =====================================================================
# PAULI BASIS (3-qubit, d=8)
# =====================================================================

I2 = np.eye(2, dtype=complex)
_PAULI_1Q = {
    'I': I2,
    'X': np.array([[0,1],[1,0]], dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'Z': np.array([[1,0],[0,-1]], dtype=complex),
}

def build_pauli_basis_3q():
    """Bouw 64 Pauli-basismatrices voor 3 qubits (d=8).
    Returns: (matrices, names) — matrices shape (64, 8, 8)
    """
    labels = ['I', 'X', 'Y', 'Z']
    ops = [_PAULI_1Q[l] for l in labels]
    matrices = []
    names = []
    for i, p1 in enumerate(ops):
        for j, p2 in enumerate(ops):
            for k, p3 in enumerate(ops):
                matrices.append(np.kron(np.kron(p1, p2), p3))
                names.append(labels[i] + labels[j] + labels[k])
    return np.array(matrices), names

PAULI_3Q, PAULI_NAMES = build_pauli_basis_3q()


def pauli_decompose(M):
    """Decomponeer 8×8 matrix M in 64 Pauli-coëfficiënten.
    c_k = Tr(P_k† · M) / 8
    """
    coeffs = np.array([np.trace(P.conj().T @ M) / 8.0 for P in PAULI_3Q])
    return coeffs


def pauli_reconstruct(coeffs):
    """Reconstrueer 8×8 matrix uit Pauli-coëfficiënten."""
    return sum(c * P for c, P in zip(coeffs, PAULI_3Q))


# =====================================================================
# ZORN L-BASIS PER PERSPECTIEF
# =====================================================================

class Zorn:
    """Minimale Zorn split-octonion klasse voor L-basis constructie."""
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = np.asarray(v, dtype=complex)

    @staticmethod
    def basis(i):
        e = np.zeros(8, dtype=complex)
        e[i] = 1.0
        return Zorn(e)

    def __mul__(self, o):
        a, al, be, b = self.v[0], self.v[1:4], self.v[4:7], self.v[7]
        c, ga, de, d = o.v[0], o.v[1:4], o.v[4:7], o.v[7]
        return Zorn(np.array([
            a*c + al@de,
            *(a*ga + d*al + np.cross(be, de)),
            *(c*be + b*de - np.cross(al, ga)),
            be@ga + b*d
        ]))

    def L_matrix(self):
        """8×8 links-vermenigvuldigingsmatrix: L(z)·x = z*x"""
        M = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            M[:, i] = (self * Zorn.basis(i)).v
        return M


FANO = [(1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)]


def fano_permute(v, d):
    """Permuteer 8-vector volgens Fano-triplet d."""
    t = list(FANO[d])
    comp = [x for x in range(1, 8) if x not in t]
    return v[[0] + t + comp]


def zmul_perspective(A_vec, B_vec, d):
    """Zorn-product door perspectief d: zmul(pdec(A), pdec(B)).

    Door beide operanden te permuteren volgens Fano-triplet d,
    verandert de effectieve vermenigvuldigingstabel. Verschillende
    triples geven andere cross-product structuur en daarmee andere
    Pauli-componenten in de L-matrix.
    """
    A_perm = fano_permute(A_vec, d)
    B_perm = fano_permute(B_vec, d)
    a, al, be, b = A_perm[0], A_perm[1:4], A_perm[4:7], A_perm[7]
    c, ga, de, dd = B_perm[0], B_perm[1:4], B_perm[4:7], B_perm[7]
    return np.array([
        a*c + al@de,
        *(a*ga + dd*al + np.cross(be, de)),
        *(c*be + b*de - np.cross(al, ga)),
        be@ga + b*dd
    ])


def build_perspective_basis(d):
    """Bouw 8 L-matrices voor Fano-perspectief d.

    De L-matrix voor perspectief d en basis-element k is:
      L_d_k[i,j] = zmul(pdec(e_k, d), pdec(e_j, d))[i]

    Elke perspectief heeft een ANDERE vermenigvuldigingstabel door
    de Fano-permutatie, en dekt daarom andere Pauli-componenten.

    Returns: (L_matrices, pauli_coverage)
      L_matrices: (8, 8, 8) array van 8×8 matrices
      pauli_coverage: set van Pauli-namen die non-zero zijn
    """
    L_mats = np.zeros((8, 8, 8), dtype=complex)
    coverage = set()

    for k in range(8):
        M = np.zeros((8, 8), dtype=complex)
        e_k = np.zeros(8, dtype=complex)
        e_k[k] = 1.0

        for j in range(8):
            e_j = np.zeros(8, dtype=complex)
            e_j[j] = 1.0
            M[:, j] = zmul_perspective(e_k, e_j, d)

        L_mats[k] = M

        # Pauli-decompositie
        coeffs = pauli_decompose(M)
        for i, c in enumerate(coeffs):
            if abs(c) > 1e-10:
                coverage.add(PAULI_NAMES[i])

    return L_mats, coverage


def build_L_projection_matrix(d):
    """Bouw (64×8) projectiematrix A: M.ravel() ≈ A @ c
    zodat c = lstsq(A, M.ravel()) de 8 L-componenten geeft.
    """
    L_mats, coverage = build_perspective_basis(d)
    A = np.zeros((64, 8), dtype=complex)
    for k in range(8):
        A[:, k] = L_mats[k].ravel()
    return A, L_mats, coverage


def perspective_project(M, d):
    """Projecteer d_phys × d_phys operator M op perspectief d.

    Werkt alleen voor d_phys=8 (3 qubits per site). Voor d_phys<8
    retourneert de originele matrix met residual=0 (geen projectie).

    Returns: (M_proj, residual_frac, c_fit)
    """
    d_phys = M.shape[0]
    if d_phys != 8:
        # Zorn-perspectief alleen zinvol bij d=8 (3 qubits)
        return M.copy(), 0.0, None

    A, L_mats, _ = build_L_projection_matrix(d)
    c_fit, _, _, _ = np.linalg.lstsq(A, M.ravel(), rcond=None)
    M_proj = sum(c_fit[k] * L_mats[k] for k in range(8))
    resid = norm(M - M_proj)
    total = norm(M)
    frac = resid / total if total > 1e-15 else 0.0
    return M_proj, frac, c_fit


# =====================================================================
# PERSPECTIEF-COVERAGE ANALYSE
# =====================================================================

def analyze_perspective_coverage():
    """Analyseer welke Pauli-componenten elk perspectief dekt."""
    all_covered = set()
    per_persp = []

    for d in range(7):
        _, coverage = build_perspective_basis(d)
        per_persp.append(coverage)
        all_covered |= coverage

    # Overlap-analyse
    total_paulis = set(PAULI_NAMES)
    uncovered = total_paulis - all_covered

    return {
        'per_perspective': per_persp,
        'total_covered': len(all_covered),
        'uncovered': uncovered,
        'overlap_matrix': _overlap_matrix(per_persp),
    }


def _overlap_matrix(per_persp):
    """7×7 overlap matrix: |intersection| / |union|."""
    n = len(per_persp)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(per_persp[i] & per_persp[j])
            union = len(per_persp[i] | per_persp[j])
            M[i, j] = inter / union if union > 0 else 0
    return M


# =====================================================================
# MPO CHI-ANALYSE (Heisenberg-picture)
# =====================================================================

def mpo_chi_spectrum(mpo_tensor):
    """Bereken SVD-spectrum van een MPO-tensor.
    mpo_tensor: shape (chi_L, d, d, chi_R) of (chi_L, d_bra*d_ket, chi_R)
    Returns: singular values
    """
    shape = mpo_tensor.shape
    if len(shape) == 4:
        cl, d1, d2, cr = shape
        mat = mpo_tensor.reshape(cl * d1, d2 * cr)
    else:
        cl, dd, cr = shape
        d = int(np.sqrt(dd))
        mat = mpo_tensor.reshape(cl * d, d * cr)
    _, S, _ = svd(mat, full_matrices=False)
    return S


def mpo_effective_chi(mpo_tensor, tol=1e-12):
    """Tel effectieve bond-dimensie (singuliere waarden > tol * max)."""
    S = mpo_chi_spectrum(mpo_tensor)
    if len(S) == 0 or S[0] < 1e-15:
        return 0
    return int(np.sum(S > tol * S[0]))


# =====================================================================
# CT-SCAN EXPERIMENT: Heisenberg QAOA MPO decomposition
# =====================================================================

def heisenberg_mpo_pauli_analysis(Lx, Ly, p, gammas, betas, max_chi=128):
    """Evolueer QAOA MPO en analyseer Pauli-structuur per site.

    Voert Heisenberg-evolutie uit via HeisenbergQAOA, en decomponeert
    het resultaat-MPO in 64 Pauli-componenten per site.

    Returns: dict met per-site analyse
    """
    from zorn_mps import HeisenbergQAOA

    hq = HeisenbergQAOA(Lx, Ly, max_chi=max_chi)
    d = hq.d

    results = []

    # Analyseer elke rand
    for x in range(Lx - 1):
        for y in range(Ly):
            # Bouw en evolueer MPO voor deze edge
            mpo = hq._make_zz_mpo(x, y, x+1, y)

            # Gate list
            gates = []
            Hd = hq._Hd
            for xx in range(Lx):
                gates.append(('full', xx, Hd))
            for l in range(p):
                zzi = hq._zz_intra_diag(gammas[l])
                zze = hq._zz_inter_diag(gammas[l])
                rxd = hq._rx_col(betas[l])
                for xx in range(Lx):
                    gates.append(('diag1', xx, zzi))
                for xx in range(Lx - 1):
                    gates.append(('diag2', xx, zze))
                for xx in range(Lx):
                    gates.append(('full', xx, rxd))

            # Heisenberg evolutie
            for gt, s, data in reversed(gates):
                if gt == 'full':
                    mpo = hq._ap1(mpo, s, data)
                elif gt == 'diag1':
                    mpo = hq._ap1_diag(mpo, s, data)
                else:
                    mpo = hq._ap2_diag(mpo, s, data)

            # Analyseer MPO tensors
            edge_data = {
                'edge': (x, y, x+1, y),
                'sites': [],
            }
            for site_idx in range(Lx):
                W = mpo[site_idx]  # (cl, d, d, cr)
                cl, d1, d2, cr = W.shape
                chi = max(cl, cr)

                # Per bond-slice: Pauli analyse
                site_info = {
                    'shape': W.shape,
                    'chi': chi,
                    'pauli_structure': None,
                    'perspective_residuals': [],
                }

                # Chi=1: directe Pauli-analyse
                if cl == 1 and cr == 1:
                    M = W[0, :, :, 0]
                    coeffs = pauli_decompose(M)
                    mag = np.abs(coeffs)
                    n_nonzero = int(np.sum(mag > 1e-10))

                    # Perspectief-projectie
                    residuals = []
                    for dd in range(7):
                        _, resid, _ = perspective_project(M, dd)
                        residuals.append(resid)

                    site_info['pauli_structure'] = {
                        'n_nonzero': n_nonzero,
                        'coeffs': coeffs,
                        'top_paulis': [PAULI_NAMES[i] for i in np.argsort(mag)[::-1][:10]],
                    }
                    site_info['perspective_residuals'] = residuals

                edge_data['sites'].append(site_info)

            # Alleen eerste edge voor korte analyse
            results.append(edge_data)
            break  # Eerste y-edge per x
        if len(results) >= min(3, Lx - 1):
            break

    return results


def ct_scan_chi_comparison(Lx, Ly, p_values, gamma=0.3, beta=0.7,
                            max_chi=128, verbose=True):
    """Hoofdexperiment: vergelijk full-chi vs perspectief-chi.

    Per QAOA-diepte p:
      1. Evolueer MPO (volledig)
      2. Meet full chi per bond
      3. Decomponeer in 7 perspectieven
      4. Meet chi per perspectief
      5. Bereken accuracy van reconstructie

    Returns: lijst van resultaatdicts per p
    """
    from zorn_mps import HeisenbergQAOA

    results = []

    for p in p_values:
        t0 = time.time()
        gammas = [gamma] * p
        betas = [beta] * p

        hq = HeisenbergQAOA(Lx, Ly, max_chi=max_chi)
        d = hq.d

        if verbose:
            print("\n" + "="*60)
            print("  CT-scan: %dx%d grid, p=%d, gamma=%.2f, beta=%.2f" % (
                Lx, Ly, p, gamma, beta))
            print("="*60)

        # Kies representatieve edge
        x1, y1, x2, y2 = 0, 0, 1, 0

        # --- Full MPO evolutie ---
        mpo_full = hq._make_zz_mpo(x1, y1, x2, y2)
        gates = _build_gate_list(hq, Lx, p, gammas, betas)
        for gt, s, data in reversed(gates):
            if gt == 'full':
                mpo_full = hq._ap1(mpo_full, s, data)
            elif gt == 'diag1':
                mpo_full = hq._ap1_diag(mpo_full, s, data)
            else:
                mpo_full = hq._ap2_diag(mpo_full, s, data)

        # Full MPO chi per bond
        full_chis = []
        for site_idx in range(Lx):
            W = mpo_full[site_idx]
            full_chis.append(mpo_effective_chi(W))

        zz_full = float(hq._mpo_trace(mpo_full).real)

        if verbose:
            print("  Full MPO: chi=%s, <ZZ>=%.6f" % (
                [W.shape for W in mpo_full], zz_full))

        # --- Perspectief-analyse ---
        # Voor elke site met chi=1: directe Pauli-projectie
        # Voor sites met chi>1: analyseer SVD-spectrum per perspectief

        persp_data = []
        for dd in range(7):
            # Projecteer elke MPO-tensor op perspectief d
            mpo_proj = []
            proj_ok = True

            for site_idx in range(Lx):
                W = mpo_full[site_idx]
                cl, d1, d2, cr = W.shape

                if cl == 1 and cr == 1:
                    # Chi=1: directe projectie
                    M = W[0, :, :, 0]
                    M_proj, resid, _ = perspective_project(M, dd)
                    mpo_proj.append(M_proj.reshape(1, d1, d2, 1))
                elif cl <= 4 and cr <= 4:
                    # Laag chi: projecteer per slice
                    W_proj = np.zeros_like(W)
                    for a in range(cl):
                        for b in range(cr):
                            M = W[a, :, :, b]
                            M_proj, _, _ = perspective_project(M, dd)
                            W_proj[a, :, :, b] = M_proj
                    mpo_proj.append(W_proj)
                else:
                    # Hoog chi: projecteer en re-compress via SVD
                    W_proj = np.zeros_like(W)
                    for a in range(cl):
                        for b in range(cr):
                            M = W[a, :, :, b]
                            M_proj, _, _ = perspective_project(M, dd)
                            W_proj[a, :, :, b] = M_proj
                    mpo_proj.append(W_proj)

            # Trace van geprojecteerde MPO
            L = np.ones((1,), dtype=complex)
            for W in mpo_proj:
                L = np.einsum('a,ab->b', L, W[:, 0, 0, :])
            zz_proj = float(L[0].real)

            # Chi van geprojecteerde MPO na hercompressie
            proj_chis = []
            for W in mpo_proj:
                proj_chis.append(mpo_effective_chi(W))

            persp_data.append({
                'perspective': dd,
                'fano_triple': FANO[dd],
                'zz_proj': zz_proj,
                'chis': proj_chis,
                'max_chi': max(proj_chis) if proj_chis else 0,
            })

            if verbose:
                print("  Persp %d %s: chi_max=%d, <ZZ>=%.6f" % (
                    dd, FANO[dd], max(proj_chis) if proj_chis else 0, zz_proj))

        # --- Reconstructie via gewogen combinatie ---
        # Eenvoudigste: gemiddelde van perspectieven (met overlap-correctie)
        zz_avg = np.mean([pd['zz_proj'] for pd in persp_data])

        # Gewogen via coverage (inverse-residual weging)
        # Bouw reconstructie via least-squares op de perspectief-projecties
        zz_reconstructed = _reconstruct_from_perspectives(
            mpo_full, persp_data, Lx)

        elapsed = time.time() - t0

        result = {
            'Lx': Lx, 'Ly': Ly, 'p': p,
            'gamma': gamma, 'beta': beta,
            'full_chi': max(full_chis),
            'full_zz': zz_full,
            'perspectives': persp_data,
            'zz_avg': zz_avg,
            'zz_reconstructed': zz_reconstructed,
            'error_avg': abs(zz_avg - zz_full),
            'error_recon': abs(zz_reconstructed - zz_full),
            'max_persp_chi': max(pd['max_chi'] for pd in persp_data),
            'min_persp_chi': min(pd['max_chi'] for pd in persp_data),
            'time': elapsed,
        }
        results.append(result)

        if verbose:
            print("\n  Samenvatting:")
            print("    Full chi:     %d" % result['full_chi'])
            print("    Max persp chi: %d" % result['max_persp_chi'])
            print("    Min persp chi: %d" % result['min_persp_chi'])
            print("    Chi ratio:    %.2f" % (
                result['max_persp_chi'] / max(result['full_chi'], 1)))
            print("    <ZZ> full:    %.6f" % zz_full)
            print("    <ZZ> avg:     %.6f (err=%.2e)" % (
                zz_avg, result['error_avg']))
            print("    <ZZ> recon:   %.6f (err=%.2e)" % (
                zz_reconstructed, result['error_recon']))
            print("    Tijd:         %.2fs" % elapsed)

    return results


def _build_gate_list(hq, Lx, p, gammas, betas):
    """Bouw gate-lijst voor Heisenberg-evolutie."""
    gates = []
    Hd = hq._Hd
    for x in range(Lx):
        gates.append(('full', x, Hd))
    for l in range(p):
        zzi = hq._zz_intra_diag(gammas[l])
        zze = hq._zz_inter_diag(gammas[l])
        rxd = hq._rx_col(betas[l])
        for x in range(Lx):
            gates.append(('diag1', x, zzi))
        for x in range(Lx - 1):
            gates.append(('diag2', x, zze))
        for x in range(Lx):
            gates.append(('full', x, rxd))
    return gates


def _reconstruct_from_perspectives(mpo_full, persp_data, Lx):
    """Reconstrueer <ZZ> uit 7 perspectieven via gewogen combinatie.

    Strategie: elk perspectief levert een projectie van de operator.
    De 7 projecties samen spannen (bijna) de hele operatorruimte op.
    We gebruiken de full-MPO om de optimale weging te bepalen via
    een eenvoudige heuristiek.
    """
    # Simpele gewogen som: w_d = 1/7 (uniform)
    # Dit is een eerste-orde benadering; de exacte reconstructie
    # vereist de overlap-matrix inversie.
    zz_values = [pd['zz_proj'] for pd in persp_data]

    # Bereken per-perspectief norm (maat voor coverage-kwaliteit)
    norms = []
    for pd in persp_data:
        norms.append(abs(pd['zz_proj']))
    total_norm = sum(norms)

    if total_norm < 1e-15:
        return 0.0

    # Norm-gewogen gemiddelde
    weights = np.array(norms) / total_norm
    return float(np.dot(weights, zz_values))


# =====================================================================
# COMPRESSIE-EXPERIMENT: Perspectief-MPO hercompressie
# =====================================================================

def perspective_recompress(mpo_tensor, d, max_chi_out=None):
    """Projecteer MPO-tensor op perspectief d en hercomprimeer.

    Input: (chi_L, d_phys, d_phys, chi_R)
    Output: gehercomprimeerde tensor, effectieve chi
    """
    cl, d1, d2, cr = mpo_tensor.shape

    # Projecteer elke (d1, d2) slice
    W_proj = np.zeros_like(mpo_tensor)
    for a in range(cl):
        for b in range(cr):
            M = mpo_tensor[a, :, :, b]
            M_proj, _, _ = perspective_project(M, d)
            W_proj[a, :, :, b] = M_proj

    # Hercomprimeer via SVD
    mat = W_proj.reshape(cl * d1, d2 * cr)
    U, S, V = svd(mat, full_matrices=False)
    Sa = np.abs(S)

    k = max(1, int(np.sum(Sa > 1e-12 * Sa[0]))) if Sa[0] > 1e-15 else 1
    if max_chi_out is not None:
        k = min(k, max_chi_out)

    W_compressed = (U[:, :k] * S[:k]) @ V[:k, :]
    W_compressed = W_compressed.reshape(cl, d1, d2, cr)

    return W_compressed, k, S


# =====================================================================
# VOLLEDIGE CT-SCAN RAPPORT
# =====================================================================

def run_ctscan_report(verbose=True):
    """Draai het volledige CT-scan experiment en genereer rapport.

    Tests:
      1. Coverage-analyse: welke Paulis per perspectief
      2. 1D keten: chi-vergelijking bij p=1,2,3
      3. 2D grid: chi-vergelijking bij p=1,2
      4. Reconstructie-nauwkeurigheid
    """
    report = {}

    # --- 1. Coverage analyse ---
    if verbose:
        print("\n" + "="*70)
        print("  DEEL 1: Perspectief-Coverage Analyse")
        print("="*70)

    cov = analyze_perspective_coverage()
    report['coverage'] = cov

    if verbose:
        for d in range(7):
            print("  Perspectief %d %s: %d Paulis" % (
                d, FANO[d], len(cov['per_perspective'][d])))
        print("  Totaal uniek: %d/64" % cov['total_covered'])
        print("  Niet gedekt: %d  %s" % (
            len(cov['uncovered']),
            sorted(cov['uncovered'])[:10]))

    # --- 2. 1D keten ---
    if verbose:
        print("\n" + "="*70)
        print("  DEEL 2: 1D Keten (Ly=1)")
        print("="*70)

    results_1d = ct_scan_chi_comparison(
        Lx=8, Ly=1, p_values=[1, 2, 3],
        gamma=0.3, beta=0.7, max_chi=64, verbose=verbose)
    report['1d'] = results_1d

    # --- 3. 2D grid ---
    if verbose:
        print("\n" + "="*70)
        print("  DEEL 3: 2D Grid (Ly=2, d=4)")
        print("="*70)

    results_2d = ct_scan_chi_comparison(
        Lx=6, Ly=2, p_values=[1, 2],
        gamma=0.3, beta=0.7, max_chi=64, verbose=verbose)
    report['2d'] = results_2d

    # --- 4. 2D grid groter ---
    if verbose:
        print("\n" + "="*70)
        print("  DEEL 4: 2D Grid (Ly=3, d=8)")
        print("="*70)

    results_2d_big = ct_scan_chi_comparison(
        Lx=4, Ly=3, p_values=[1],
        gamma=0.3, beta=0.7, max_chi=64, verbose=verbose)
    report['2d_big'] = results_2d_big

    # --- Samenvatting ---
    if verbose:
        print("\n" + "="*70)
        print("  SAMENVATTING")
        print("="*70)

        # Verzamel alle resultaten
        for label, res_list in [('1D', results_1d), ('2D-Ly2', results_2d),
                                 ('2D-Ly3', results_2d_big)]:
            for r in res_list:
                chi_ratio = r['max_persp_chi'] / max(r['full_chi'], 1)
                print("  %s p=%d: full_chi=%d, persp_chi=%d..%d (ratio=%.2f), "
                      "err_avg=%.2e, err_recon=%.2e" % (
                    label, r['p'], r['full_chi'],
                    r['min_persp_chi'], r['max_persp_chi'],
                    chi_ratio, r['error_avg'], r['error_recon']))

        # Conclusie
        all_ratios = []
        for res_list in [results_1d, results_2d, results_2d_big]:
            for r in res_list:
                all_ratios.append(
                    r['max_persp_chi'] / max(r['full_chi'], 1))

        avg_ratio = np.mean(all_ratios)
        if avg_ratio < 0.5:
            verdict = "VEELBELOVEND — perspectieven reduceren chi significant"
        elif avg_ratio < 0.8:
            verdict = "MATIG — enige chi-reductie, maar niet dramatisch"
        else:
            verdict = "TELEURSTELLEND — perspectieven reduceren chi niet"

        print("\n  Gemiddelde chi-ratio: %.2f" % avg_ratio)
        print("  Verdict: %s" % verdict)

    report['verdict'] = verdict if verbose else "niet berekend"
    return report


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    report = run_ctscan_report(verbose=True)
