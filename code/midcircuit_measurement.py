#!/usr/bin/env python3
"""
B34: Mid-Circuit Measurement / Adaptieve Projectie

MPS-gebaseerde mid-circuit metingen voor QAOA op 1D-ketens.
Meet (projecteer) periodiek subsets van qubits naar klassieke staten
tijdens de QAOA-evolutie. De MPS-bond klapt naar chi=1, waardoor de
effectieve MPS kleiner wordt.

Kernfuncties:
  - MPS representatie met left-canonical / mixed-canonical vormen
  - Mid-circuit meting: project qubit naar |0> of |1>, collapse bond
  - MPS splitsen na meting in onafhankelijke stukken
  - Adaptieve meetpunt selectie op basis van bond-entropie
  - Multi-branch sampling met verwachtingswaarde-middeling
  - Integratie met QAOA-circuit (meting tussen lagen)

Referenties:
  - Noel et al. (2022): Mid-circuit measurements on superconducting qubits
  - Aaronson: Quantum Zeno effect — continu meten bevriest dynamiek
  - B15 min_weight: adaptieve chi-drempel als trigger voor meetbeslissing
"""

import numpy as np
from scipy.linalg import svd as scipy_svd


# ============================================================
# MPS Representatie
# ============================================================

class MPS:
    """Matrix Product State voor n qubits.

    Elke tensor M[i] heeft shape (chi_left, d, chi_right) met d=2.
    Randvoorwaarde: M[0] shape (1, d, chi), M[n-1] shape (chi, d, 1).
    """

    def __init__(self, tensors, classical_bits=None):
        """
        Parameters
        ----------
        tensors : list of ndarray
            Lijst van MPS-tensoren, elk shape (chi_l, d, chi_r).
        classical_bits : dict, optional
            {qubit_index: 0 of 1} voor reeds gemeten qubits.
        """
        self.tensors = list(tensors)
        self.n = len(tensors)
        self.d = 2  # qubit
        self.classical_bits = dict(classical_bits) if classical_bits else {}

    @classmethod
    def from_statevector(cls, psi, n, chi_max=None):
        """Construeer MPS uit volledige state vector via opeenvolgende SVDs.

        Parameters
        ----------
        psi : ndarray, shape (2**n,)
        n : int
        chi_max : int, optional
            Maximale bond dimensie. None = exact (geen truncatie).
        """
        tensors = []
        C = psi.reshape(1, -1)

        for i in range(n - 1):
            chi_l = C.shape[0]
            C = C.reshape(chi_l * 2, -1)
            U, S, Vh = np.linalg.svd(C, full_matrices=False)

            if chi_max is not None:
                k = min(chi_max, len(S))
            else:
                k = int(np.sum(S > 1e-15 * S[0])) if len(S) > 0 else 1
            k = max(1, k)

            tensors.append(U[:, :k].reshape(chi_l, 2, k))
            C = np.diag(S[:k]) @ Vh[:k, :]

        # Laatste tensor
        chi_l = C.shape[0]
        tensors.append(C.reshape(chi_l, 2, 1))

        return cls(tensors)

    def to_statevector(self):
        """Reconstrueer volledige state vector uit MPS (exponentieel in n)."""
        result = self.tensors[0]  # shape (1, d, chi)
        for i in range(1, self.n):
            # result shape (1, d^i, chi_prev), tensors[i] shape (chi_prev, d, chi_next)
            result = np.einsum('ijk,kml->ijml', result, self.tensors[i])
            s = result.shape
            result = result.reshape(s[0], s[1] * s[2], s[3])

        psi = result.reshape(-1)
        return psi

    def copy(self):
        """Diepe kopie van de MPS."""
        return MPS(
            [t.copy() for t in self.tensors],
            classical_bits=dict(self.classical_bits)
        )

    def norm(self):
        """Bereken de norm van de MPS via contractie."""
        # Contract van links naar rechts
        env = np.ones((1, 1), dtype=complex)
        for M in self.tensors:
            # env shape (chi, chi'), M shape (chi, d, chi_new)
            # M_conj shape (chi', d, chi_new')
            env = np.einsum('ab,adc,bde->ce', env, M, M.conj())
        return np.sqrt(abs(env[0, 0]))

    def normalize(self):
        """Normaliseer de MPS in-place."""
        nrm = self.norm()
        if nrm > 0:
            # Normaliseer laatste tensor
            self.tensors[-1] = self.tensors[-1] / nrm
        return self

    def max_bond_dim(self):
        """Maximale bond dimensie."""
        return max(self.tensors[i].shape[2] for i in range(self.n - 1))

    def bond_dims(self):
        """Lijst van bond dimensies."""
        return [self.tensors[i].shape[2] for i in range(self.n - 1)]

    def bond_entropy(self, site):
        """Bereken de von Neumann entropie op de bond tussen site en site+1.

        Brengt de MPS eerst in mixed-canonical vorm met het orthogonaliteitscentrum
        op de gevraagde bond, dan SVD voor Schmidt-waarden.

        Parameters
        ----------
        site : int
            Bond index (0 <= site < n-1). Entropie van de bond site | site+1.

        Returns
        -------
        float
            Von Neumann entropie S = -sum p*log2(p).
        """
        if site < 0 or site >= self.n - 1:
            raise ValueError(f"site {site} out of range [0, {self.n-2}]")

        # Maak kopie en breng in mixed-canonical vorm
        tensors = [t.copy() for t in self.tensors]

        # Left-canonicalize sites 0..site
        for i in range(site + 1):
            chi_l, d, chi_r = tensors[i].shape
            M = tensors[i].reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(M)
            new_chi = Q.shape[1]
            tensors[i] = Q.reshape(chi_l, d, new_chi)
            if i < self.n - 1:
                tensors[i + 1] = np.einsum('ij,jkl->ikl', R, tensors[i + 1])

        # Right-canonicalize sites n-1..site+1
        for i in range(self.n - 1, site, -1):
            chi_l, d, chi_r = tensors[i].shape
            M = tensors[i].reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(M.T)
            # Q.T is the right-canonical tensor
            new_chi = Q.shape[1]
            tensors[i] = Q.T.reshape(new_chi, d, chi_r)
            if i > 0:
                tensors[i - 1] = np.einsum('ijk,kl->ijl', tensors[i - 1], R.T)

        # Nu is het orthogonaliteitscentrum op bond site|site+1
        # SVD van de bond matrix
        chi_l, d_l, chi_mid = tensors[site].shape
        M = tensors[site].reshape(chi_l * d_l, chi_mid)
        _, S, _ = np.linalg.svd(M, full_matrices=False)

        # Schmidt-waarden zijn de singuliere waarden
        S = S[S > 1e-15]
        if len(S) == 0:
            return 0.0

        probs = S ** 2
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-30))
        return float(entropy)

    def all_bond_entropies(self):
        """Bereken entropie op elke bond.

        Returns
        -------
        list of float
            Entropie voor bonds 0|1, 1|2, ..., (n-2)|(n-1).
        """
        return [self.bond_entropy(i) for i in range(self.n - 1)]

    def schmidt_values(self, site):
        """Bereken Schmidt-waarden op bond site|site+1.

        Returns
        -------
        ndarray
            Gesorteerde Schmidt-waarden (aflopend).
        """
        if site < 0 or site >= self.n - 1:
            raise ValueError(f"site {site} out of range [0, {self.n-2}]")

        tensors = [t.copy() for t in self.tensors]

        # Left-canonicalize 0..site
        for i in range(site + 1):
            chi_l, d, chi_r = tensors[i].shape
            M = tensors[i].reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(M)
            new_chi = Q.shape[1]
            tensors[i] = Q.reshape(chi_l, d, new_chi)
            if i < self.n - 1:
                tensors[i + 1] = np.einsum('ij,jkl->ikl', R, tensors[i + 1])

        # Right-canonicalize n-1..site+1
        for i in range(self.n - 1, site, -1):
            chi_l, d, chi_r = tensors[i].shape
            M = tensors[i].reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(M.T)
            new_chi = Q.shape[1]
            tensors[i] = Q.T.reshape(new_chi, d, chi_r)
            if i > 0:
                tensors[i - 1] = np.einsum('ijk,kl->ijl', tensors[i - 1], R.T)

        chi_l, d_l, chi_mid = tensors[site].shape
        M = tensors[site].reshape(chi_l * d_l, chi_mid)
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        return S


def mps_compress(mps, chi_max, min_weight=None):
    """Comprimeer een MPS tot maximale bond dimensie chi_max.

    Parameters
    ----------
    mps : MPS
    chi_max : int
    min_weight : float, optional
        Drempel voor singuliere waarden (B15-stijl). Singuliere waarden
        kleiner dan min_weight * S_max worden weggegooid.

    Returns
    -------
    MPS
        Gecomprimeerde MPS (nieuwe instantie).
    """
    tensors = [t.copy() for t in mps.tensors]
    n = len(tensors)

    # Left-to-right sweep: QR
    for i in range(n - 1):
        chi_l, d, chi_r = tensors[i].shape
        M = tensors[i].reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(M)
        new_chi = Q.shape[1]
        tensors[i] = Q.reshape(chi_l, d, new_chi)
        tensors[i + 1] = np.einsum('ij,jkl->ikl', R, tensors[i + 1])

    # Right-to-left sweep: SVD met truncatie
    for i in range(n - 1, 0, -1):
        chi_l, d, chi_r = tensors[i].shape
        M = tensors[i].reshape(chi_l, d * chi_r)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        k = min(chi_max, len(S))
        if min_weight is not None and len(S) > 0:
            threshold = min_weight * S[0]
            k_thresh = int(np.sum(S > threshold))
            k = max(1, min(k, k_thresh))
        k = max(1, k)

        tensors[i] = Vh[:k, :].reshape(k, d, chi_r)
        tensors[i - 1] = np.einsum('ijk,kl,l->ijl',
                                    tensors[i - 1], U[:, :k], S[:k])

    result = MPS(tensors, classical_bits=dict(mps.classical_bits))
    result.normalize()
    return result


# ============================================================
# QAOA op MPS
# ============================================================

def apply_phase_gate(mps, qubit_i, qubit_j, gamma):
    """Pas e^{i*gamma*Z_i*Z_j} toe op de MPS.

    Dit is een diagonale 2-qubit gate. Voor naburige qubits (|i-j|=1)
    passen we het toe als lokale operatie op de bond.

    Parameters
    ----------
    mps : MPS
        In-place gemodificeerd.
    qubit_i, qubit_j : int
    gamma : float
    """
    # e^{i*gamma*ZZ} is diag(e^{ig}, e^{-ig}, e^{-ig}, e^{ig})
    # ZZ eigenvalues: |00>->+1, |01>->-1, |10>->-1, |11>->+1
    phases = np.array([np.exp(1j * gamma), np.exp(-1j * gamma),
                       np.exp(-1j * gamma), np.exp(1j * gamma)])

    i, j = min(qubit_i, qubit_j), max(qubit_i, qubit_j)

    # Als een van de qubits gemeten is, pas de fase klassiek toe
    if i in mps.classical_bits or j in mps.classical_bits:
        ci = mps.classical_bits.get(i, None)
        cj = mps.classical_bits.get(j, None)
        if ci is not None and cj is not None:
            # Beide gemeten: globale fase, skip
            return
        elif ci is not None:
            # Qubit i is gemeten: pas Z-fase toe op qubit j
            sign = 1 - 2 * ci  # +1 als 0, -1 als 1
            phase = np.exp(1j * gamma * sign)
            Z_phase = np.array([[phase, 0], [0, np.conj(phase)]], dtype=complex)
            mps.tensors[j] = np.einsum('ab,ibk->iak', Z_phase,
                                        mps.tensors[j].astype(complex))
        else:
            # Qubit j is gemeten: pas Z-fase toe op qubit i
            sign = 1 - 2 * cj
            phase = np.exp(1j * gamma * sign)
            Z_phase = np.array([[phase, 0], [0, np.conj(phase)]], dtype=complex)
            mps.tensors[i] = np.einsum('ab,ibk->iak', Z_phase,
                                        mps.tensors[i].astype(complex))
        return

    if j == i + 1:
        # Naburige qubits: absorbeer in tensor i
        # M[i] shape (chi_l, 2, chi_mid), M[j] shape (chi_mid, 2, chi_r)
        # Contract, apply gate, SVD terug
        Mi = mps.tensors[i]  # (chi_l, 2, chi_mid)
        Mj = mps.tensors[j]  # (chi_mid, 2, chi_r)

        # Theta = contract over chi_mid
        theta = np.einsum('ijk,kml->ijml', Mi, Mj)
        # theta shape (chi_l, 2, 2, chi_r)

        # Apply phase gate
        for si in range(2):
            for sj in range(2):
                theta[:, si, sj, :] *= phases[si * 2 + sj]

        # SVD terug
        chi_l, d1, d2, chi_r = theta.shape
        theta_mat = theta.reshape(chi_l * d1, d2 * chi_r)
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
        k = len(S)

        mps.tensors[i] = U.reshape(chi_l, d1, k)
        mps.tensors[j] = (np.diag(S) @ Vh).reshape(k, d2, chi_r)
    else:
        # Niet-naburige qubits: swap-network of directe toepassing
        # Voor eenvoud: pas diag gate toe via lokale fasen
        # Dit is exact want ZZ is diagonaal
        _apply_long_range_zz(mps, i, j, gamma)


def _apply_long_range_zz(mps, i, j, gamma):
    """Pas ZZ-gate toe op niet-naburige qubits via phase kickback.

    Omdat ZZ diagonaal is in de computationele basis, kunnen we het
    exact toepassen door de fasen in te bouwen op elke site.
    We gebruiken een CNOT-achtige decompositie via swap-netwerk.

    Eenvoudige implementatie: contract de hele keten i..j, apply, decompose.
    Voor kleine afstanden (|i-j| < 5) is dit effectief.
    """
    if j - i == 1:
        apply_phase_gate(mps, i, j, gamma)
        return

    # Voor langere afstand: swap qubit i naar j-1, apply naburig, swap terug
    # SWAP = product van 3 CNOT gates, maar voor MPS is het een permutatie
    # Efficienter: gebruik het feit dat ZZ diagonaal is
    # We absorberen de fase in de bond-structuur via een MPO
    # Simpelste correcte methode: volledige contractie van segment i..j

    # Contract tensors i..j into one big tensor
    T = mps.tensors[i]
    for k in range(i + 1, j + 1):
        T = np.einsum('...j,jkl->...kl', T, mps.tensors[k])
    # T shape (chi_l, d_i, d_{i+1}, ..., d_j, chi_r)

    # Reshape to (chi_l, d^(j-i+1), chi_r)
    shape_orig = T.shape
    chi_l = shape_orig[0]
    chi_r = shape_orig[-1]
    n_sites = j - i + 1
    d_total = 2 ** n_sites
    T = T.reshape(chi_l, d_total, chi_r)

    # Apply ZZ phase: iterate over computational basis
    for idx in range(d_total):
        bits = [(idx >> (n_sites - 1 - q)) & 1 for q in range(n_sites)]
        si, sj = bits[0], bits[-1]
        zz = 1 - 2 * (si ^ sj)  # +1 if same, -1 if different
        T[:, idx, :] *= np.exp(1j * gamma * zz)

    # Decompose back via SVD chain
    C = T.reshape(chi_l, d_total * chi_r)
    # Reshape: first separate out qubit i
    C = T  # (chi_l, d_total, chi_r)

    for k in range(n_sites - 1):
        chi_l_cur = C.shape[0]
        remaining = C.shape[1] // 2
        chi_r_cur = C.shape[2] if len(C.shape) > 2 else 1

        if len(C.shape) == 3:
            C = C.reshape(chi_l_cur * 2, remaining * chi_r_cur)
        else:
            C = C.reshape(chi_l_cur * 2, -1)

        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        bond = len(S)

        mps.tensors[i + k] = U.reshape(chi_l_cur, 2, bond)
        C = (np.diag(S) @ Vh).reshape(bond, remaining, -1)
        if k == n_sites - 2:
            # Laatste tensor
            C_shape = C.shape
            mps.tensors[j] = C.reshape(C_shape[0], 2, C_shape[2])


def apply_mixer_gate(mps, qubit, beta):
    """Pas e^{-i*beta*X} toe op een enkele qubit.

    X = [[0,1],[1,0]], dus e^{-i*beta*X} = cos(beta)*I - i*sin(beta)*X

    Parameters
    ----------
    mps : MPS
        In-place gemodificeerd.
    qubit : int
    beta : float
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    # Mixer matrix
    Rx = np.array([[c, s], [s, c]], dtype=complex)

    # Pas toe op de d-index van tensor[qubit]
    # M shape (chi_l, d_old, chi_r), Rx shape (d_new, d_old)
    # result[i, d_new, k] = sum_d_old Rx[d_new, d_old] * M[i, d_old, k]
    mps.tensors[qubit] = np.einsum('ab,ibk->iak',
                                    Rx, mps.tensors[qubit].astype(complex))


def apply_qaoa_layer(mps, edges, gamma, beta, chi_max=None, min_weight=None):
    """Pas een volledige QAOA-laag toe: fase + mixer.

    Parameters
    ----------
    mps : MPS
    edges : list of (int, int)
    gamma : float
    beta : float
    chi_max : int, optional
        Als gegeven, comprimeer na fase-stap.
    min_weight : float, optional
        Adaptieve SVD-drempel.
    """
    # Fase gates
    for (i, j) in edges:
        apply_phase_gate(mps, i, j, gamma)

    # Compressie na fase-stap (optioneel)
    if chi_max is not None:
        compressed = mps_compress(mps, chi_max, min_weight=min_weight)
        mps.tensors = compressed.tensors

    # Mixer gates
    for q in range(mps.n):
        apply_mixer_gate(mps, q, beta)


def qaoa_mps(n, edges, gammas, betas, chi_max=None, min_weight=None):
    """Voer QAOA uit op een MPS-representatie.

    Start vanuit |+>^n, pas p lagen toe.

    Parameters
    ----------
    n : int
        Aantal qubits.
    edges : list of (int, int)
    gammas, betas : list of float (lengte p)
    chi_max : int, optional
    min_weight : float, optional

    Returns
    -------
    MPS
    """
    # Start state |+>^n
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
    mps = MPS(tensors)

    p = len(gammas)
    for layer in range(p):
        apply_qaoa_layer(mps, edges, gammas[layer], betas[layer],
                        chi_max=chi_max, min_weight=min_weight)

    mps.normalize()
    return mps


# ============================================================
# Mid-Circuit Measurement
# ============================================================

def measure_qubit(mps, site, outcome=None, rng=None):
    """Meet een enkele qubit in de computationele basis.

    Projecteert qubit `site` naar |outcome>. Na meting klapt de
    bond dimensie op die plek naar 1 (of wordt de tensor een scalar
    in die richting).

    Parameters
    ----------
    mps : MPS
        Wordt in-place gemodificeerd.
    site : int
        Welke qubit te meten.
    outcome : int (0 of 1), optional
        Geforceerd meetresultaat. Als None, sample volgens Born-regel.
    rng : np.random.Generator, optional

    Returns
    -------
    int
        Het meetresultaat (0 of 1).
    float
        De Born-waarschijnlijkheid van de uitkomst.
    """
    if rng is None:
        rng = np.random.default_rng()

    if site in mps.classical_bits:
        # Al gemeten
        return mps.classical_bits[site], 1.0

    # Bereken de Born-waarschijnlijkheden voor |0> en |1>
    # via partieel contractie van de MPS
    probs = np.zeros(2)
    for s in range(2):
        # Project tensor op |s>
        # M[site] shape (chi_l, 2, chi_r) -> M[site][:, s, :] shape (chi_l, chi_r)
        projected_tensors = [t.copy() for t in mps.tensors]
        # Replace site tensor met projectie
        projected_tensors[site] = mps.tensors[site][:, s:s+1, :]  # (chi_l, 1, chi_r)

        # Bereken norm^2 van geprojecteerde MPS
        env = np.ones((1, 1), dtype=complex)
        for M in projected_tensors:
            env = np.einsum('ab,adc,bde->ce', env, M, M.conj())
        probs[s] = abs(env[0, 0])

    # Normaliseer
    total = probs.sum()
    if total < 1e-30:
        # Degeneraat geval
        probs = np.array([0.5, 0.5])
        total = 1.0
    probs /= total

    # Kies uitkomst
    if outcome is None:
        outcome = 0 if rng.random() < probs[0] else 1
    outcome = int(outcome)

    born_prob = probs[outcome]

    # Projecteer de MPS
    # Vervang tensor op site met geprojecteerde versie
    mps.tensors[site] = mps.tensors[site][:, outcome:outcome+1, :]
    # Shape is nu (chi_l, 1, chi_r)

    # Normaliseer
    mps.normalize()

    # Registreer klassiek bit
    mps.classical_bits[site] = outcome

    return outcome, born_prob


def measure_qubits(mps, sites, outcomes=None, rng=None):
    """Meet meerdere qubits sequentieel.

    Parameters
    ----------
    mps : MPS
    sites : list of int
    outcomes : list of int, optional
    rng : np.random.Generator, optional

    Returns
    -------
    list of int
        Meetresultaten.
    float
        Gecombineerde Born-waarschijnlijkheid (product).
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    combined_prob = 1.0

    for idx, site in enumerate(sites):
        out = outcomes[idx] if outcomes is not None else None
        result, prob = measure_qubit(mps, site, outcome=out, rng=rng)
        results.append(result)
        combined_prob *= prob

    return results, combined_prob


def split_mps_at_measured(mps, measured_site):
    """Splits een MPS in twee onafhankelijke stukken na meting op measured_site.

    Voorwaarde: qubit measured_site is al gemeten (d=1 op die plek).
    De MPS splitst in: sites [0..measured_site-1] en [measured_site+1..n-1].

    Parameters
    ----------
    mps : MPS
    measured_site : int

    Returns
    -------
    list of MPS
        Eén of twee MPS-stukken (afhankelijk van randpositie).
    """
    if measured_site not in mps.classical_bits:
        raise ValueError(f"Site {measured_site} is niet gemeten")

    n = mps.n
    pieces = []

    # Absorbeer de gemeten tensor in aangrenzende tensoren
    M_meas = mps.tensors[measured_site]  # (chi_l, 1, chi_r)
    mat = M_meas.reshape(M_meas.shape[0], M_meas.shape[2])  # (chi_l, chi_r)

    # SVD van de verbindingsmatrix
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    # In de meeste gevallen is dit (bijna) rank-1 na meting

    if measured_site > 0 and measured_site < n - 1:
        # Midden: splits in links en rechts
        # Absorbeer links: tensors[measured_site-1] *= U @ diag(S)
        left_factor = U @ np.diag(S)  # (chi_l, k)
        right_factor = Vh  # (k, chi_r)

        # Links stuk: sites 0..measured_site-1
        left_tensors = [t.copy() for t in mps.tensors[:measured_site]]
        left_tensors[-1] = np.einsum('ijk,kl->ijl', left_tensors[-1], left_factor)
        # Eindig met chi_r = k, maak het 1 door nog een SVD
        # Nee — gewoon laten, het is een geldige MPS

        # Rechts stuk: sites measured_site+1..n-1
        right_tensors = [t.copy() for t in mps.tensors[measured_site + 1:]]
        right_tensors[0] = np.einsum('ij,jkl->ikl', right_factor, right_tensors[0])

        left_mps = MPS(left_tensors,
                       {k: v for k, v in mps.classical_bits.items() if k < measured_site})
        right_mps = MPS(right_tensors,
                        {k - measured_site - 1: v for k, v in mps.classical_bits.items()
                         if k > measured_site})

        left_mps.normalize()
        right_mps.normalize()
        pieces = [left_mps, right_mps]

    elif measured_site == 0:
        # Randgeval: alleen rechts stuk
        right_tensors = [t.copy() for t in mps.tensors[1:]]
        # Absorbeer mat in eerste tensor van rechts stuk
        right_tensors[0] = np.einsum('ij,jkl->ikl', mat, right_tensors[0])
        # Hergroepeer zodat chi_l = 1
        chi_l, d, chi_r = right_tensors[0].shape
        if chi_l > 1:
            M = right_tensors[0].reshape(chi_l, d * chi_r)
            U2, S2, Vh2 = np.linalg.svd(M, full_matrices=False)
            right_tensors[0] = (np.diag(S2[:1]) @ Vh2[:1, :]).reshape(1, d, chi_r)

        right_mps = MPS(right_tensors,
                        {k - 1: v for k, v in mps.classical_bits.items() if k > 0})
        right_mps.normalize()
        pieces = [right_mps]

    elif measured_site == n - 1:
        # Randgeval: alleen links stuk
        left_tensors = [t.copy() for t in mps.tensors[:n - 1]]
        left_tensors[-1] = np.einsum('ijk,kl->ijl', left_tensors[-1], mat.T)
        # Hergroepeer zodat chi_r = 1
        chi_l, d, chi_r = left_tensors[-1].shape
        if chi_r > 1:
            M = left_tensors[-1].reshape(chi_l * d, chi_r)
            U2, S2, Vh2 = np.linalg.svd(M, full_matrices=False)
            left_tensors[-1] = (U2[:, :1] * S2[0]).reshape(chi_l, d, 1)

        left_mps = MPS(left_tensors,
                        {k: v for k, v in mps.classical_bits.items() if k < n - 1})
        left_mps.normalize()
        pieces = [left_mps]

    return pieces


# ============================================================
# Adaptieve Meetpunt Selectie
# ============================================================

def select_measurement_sites(mps, max_sites=None, entropy_threshold=0.1,
                             min_weight_threshold=None):
    """Selecteer qubits om te meten op basis van lage bond-entropie.

    Idee: bonds met lage entropie zijn al bijna klassiek — meting
    verstoort de toestand minimaal maar klapt chi naar 1.

    Parameters
    ----------
    mps : MPS
    max_sites : int, optional
        Maximaal aantal sites om te meten.
    entropy_threshold : float
        Meet sites waar aangrenzende bond-entropie < drempel.
    min_weight_threshold : float, optional
        Alternatieve drempel: meet sites waar de kleinste Schmidt-waarde
        op een aangrenzende bond > (1 - threshold) (bijna productstate).

    Returns
    -------
    list of int
        Sites geselecteerd voor meting.
    list of float
        Bijbehorende entropie-waarden.
    """
    n = mps.n
    already_measured = set(mps.classical_bits.keys())

    candidates = []

    for site in range(n):
        if site in already_measured:
            continue

        # Bereken de minimale entropie van aangrenzende bonds
        entropies_adj = []
        if site > 0:
            entropies_adj.append(mps.bond_entropy(site - 1))
        if site < n - 1:
            entropies_adj.append(mps.bond_entropy(site))

        if not entropies_adj:
            continue

        # Gebruik de maximale aangrenzende entropie als criterium
        # (we willen sites waar BEIDE bonds laag zijn)
        max_adj_entropy = max(entropies_adj)

        if max_adj_entropy < entropy_threshold:
            candidates.append((site, max_adj_entropy))

    # Sorteer op entropie (laagst eerst)
    candidates.sort(key=lambda x: x[1])

    if max_sites is not None:
        candidates = candidates[:max_sites]

    sites = [c[0] for c in candidates]
    entropies = [c[1] for c in candidates]

    return sites, entropies


def adaptive_measurement_schedule(mps, edges, gammas, betas,
                                  chi_max=None, min_weight=None,
                                  entropy_threshold=0.3,
                                  measure_every=1,
                                  max_measure_fraction=0.3,
                                  rng=None):
    """Voer QAOA uit met adaptieve mid-circuit metingen.

    Na elke `measure_every` lagen worden bonds geanalyseerd en
    qubits met lage entropie worden gemeten.

    Parameters
    ----------
    mps : MPS
    edges : list of (int, int)
    gammas, betas : list of float
    chi_max : int, optional
    min_weight : float, optional
    entropy_threshold : float
        Bond-entropie drempel voor meetbeslissing.
    measure_every : int
        Meet na elke zoveel QAOA-lagen.
    max_measure_fraction : float
        Maximale fractie van ongemeten qubits om per keer te meten.
    rng : np.random.Generator, optional

    Returns
    -------
    MPS
        Finale MPS na alle lagen en metingen.
    dict
        Statistieken over de metingen.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = len(gammas)
    stats = {
        'measurements_per_layer': [],
        'sites_measured': [],
        'bond_dims_history': [],
        'entropy_history': [],
    }

    for layer in range(p):
        apply_qaoa_layer(mps, edges, gammas[layer], betas[layer],
                        chi_max=chi_max, min_weight=min_weight)

        stats['bond_dims_history'].append(mps.bond_dims())

        if (layer + 1) % measure_every == 0 and layer < p - 1:
            # Bepaal hoeveel we mogen meten
            n_unmeasured = mps.n - len(mps.classical_bits)
            max_measure = max(1, int(max_measure_fraction * n_unmeasured))

            sites, entropies = select_measurement_sites(
                mps, max_sites=max_measure,
                entropy_threshold=entropy_threshold
            )

            stats['entropy_history'].append(mps.all_bond_entropies())

            if sites:
                outcomes, prob = measure_qubits(mps, sites, rng=rng)
                stats['measurements_per_layer'].append(len(sites))
                stats['sites_measured'].extend(sites)
            else:
                stats['measurements_per_layer'].append(0)

    mps.normalize()
    return mps, stats


# ============================================================
# Multi-Branch Sampling
# ============================================================

def multi_branch_expectation(n, edges, gammas, betas,
                             measure_sites, measure_after_layer,
                             observable_fn,
                             n_branches=32,
                             chi_max=None, min_weight=None,
                             rng=None):
    """Bereken verwachtingswaarde via multi-branch sampling.

    Voer QAOA uit, meet na opgegeven laag, en middel de observable
    over alle meetuitkomsten (gewogen met Born-waarschijnlijkheid).

    Parameters
    ----------
    n : int
        Aantal qubits.
    edges : list of (int, int)
    gammas, betas : list of float
    measure_sites : list of int
        Welke qubits te meten.
    measure_after_layer : int
        Na welke laag (0-indexed) te meten.
    observable_fn : callable
        Functie(MPS) -> float, berekent de observable.
    n_branches : int
        Aantal branches om te samplen.
    chi_max : int, optional
    min_weight : float, optional
    rng : np.random.Generator, optional

    Returns
    -------
    float
        Gewogen gemiddelde van de observable.
    float
        Standaardfout van het gemiddelde.
    dict
        Detail-statistieken.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = len(gammas)
    values = []
    weights = []

    for branch in range(n_branches):
        # Start vanuit |+>^n
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        tensors = [plus.reshape(1, 2, 1) for _ in range(n)]
        mps = MPS(tensors)

        # Pas lagen toe tot meetpunt
        for layer in range(min(measure_after_layer + 1, p)):
            apply_qaoa_layer(mps, edges, gammas[layer], betas[layer],
                            chi_max=chi_max, min_weight=min_weight)

        # Meet
        outcomes, prob = measure_qubits(mps, measure_sites, rng=rng)

        # Comprimeer na meting
        if chi_max is not None:
            mps = mps_compress(mps, chi_max, min_weight=min_weight)

        # Resterende lagen
        for layer in range(measure_after_layer + 1, p):
            apply_qaoa_layer(mps, edges, gammas[layer], betas[layer],
                            chi_max=chi_max, min_weight=min_weight)

        mps.normalize()

        # Bereken observable
        val = observable_fn(mps)
        values.append(val)
        weights.append(prob)

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    mean_val = np.sum(weights * values)
    # Gewogen variantie
    var = np.sum(weights * (values - mean_val) ** 2)
    stderr = np.sqrt(var / n_branches) if n_branches > 1 else 0.0

    stats = {
        'values': values,
        'weights': weights,
        'mean': mean_val,
        'stderr': stderr,
        'n_branches': n_branches,
    }

    return mean_val, stderr, stats


# ============================================================
# Observables
# ============================================================

def maxcut_cost_mps(mps, edges):
    """Bereken MaxCut cost <C> = sum_{(i,j)} (1 - <Z_i Z_j>) / 2.

    Parameters
    ----------
    mps : MPS
    edges : list of (int, int)

    Returns
    -------
    float
    """
    cost = 0.0
    for (i, j) in edges:
        zz = expectation_zz(mps, i, j)
        cost += (1.0 - zz) / 2.0
    return cost


def expectation_z(mps, site):
    """Bereken <Z_site> voor de MPS.

    Parameters
    ----------
    mps : MPS
    site : int

    Returns
    -------
    float
    """
    if site in mps.classical_bits:
        return 1 - 2 * mps.classical_bits[site]

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    env = np.ones((1, 1), dtype=complex)
    for idx in range(mps.n):
        M = mps.tensors[idx]
        if idx == site:
            # <M| Z |M>
            MZ = np.einsum('ijk,jl->ilk', M, Z)
            env = np.einsum('ab,adc,bde->ce', env, MZ, M.conj())
        else:
            env = np.einsum('ab,adc,bde->ce', env, M, M.conj())

    return float(np.real(env[0, 0]))


def expectation_zz(mps, site_i, site_j):
    """Bereken <Z_i Z_j> voor de MPS.

    Parameters
    ----------
    mps : MPS
    site_i, site_j : int

    Returns
    -------
    float
    """
    if site_i > site_j:
        site_i, site_j = site_j, site_i

    # Check klassieke bits
    if site_i in mps.classical_bits and site_j in mps.classical_bits:
        zi = 1 - 2 * mps.classical_bits[site_i]
        zj = 1 - 2 * mps.classical_bits[site_j]
        return float(zi * zj)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    env = np.ones((1, 1), dtype=complex)
    for idx in range(mps.n):
        M = mps.tensors[idx]
        if idx == site_i or idx == site_j:
            MZ = np.einsum('ijk,jl->ilk', M, Z)
            env = np.einsum('ab,adc,bde->ce', env, MZ, M.conj())
        else:
            env = np.einsum('ab,adc,bde->ce', env, M, M.conj())

    return float(np.real(env[0, 0]))


# ============================================================
# Helper: QAOA state vector (referentie)
# ============================================================

def qaoa_statevector(n, edges, gammas, betas):
    """QAOA via volledige state vector (referentie, exponentieel).

    Parameters
    ----------
    n : int
    edges : list of (int, int)
    gammas, betas : list of float

    Returns
    -------
    ndarray, shape (2**n,)
    """
    dim = 2 ** n
    indices = np.arange(dim)
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for layer in range(len(gammas)):
        g, b = gammas[layer], betas[layer]
        # Phase
        phase = np.zeros(dim)
        for (i, j) in edges:
            bi = (indices >> (n - 1 - i)) & 1
            bj = (indices >> (n - 1 - j)) & 1
            phase += g * (1 - 2 * (bi ^ bj))
        psi *= np.exp(1j * phase)
        # Mixer
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(n):
            mask = 1 << (n - 1 - q)
            i0 = indices[indices & mask == 0]
            i1 = i0 | mask
            a, bb = psi[i0].copy(), psi[i1].copy()
            psi[i0] = c * a + s * bb
            psi[i1] = s * a + c * bb

    return psi / np.linalg.norm(psi)


def maxcut_cost_statevector(psi, n, edges):
    """Bereken MaxCut cost vanuit state vector.

    Returns
    -------
    float
    """
    dim = 2 ** n
    indices = np.arange(dim)
    cost = 0.0
    for (i, j) in edges:
        bi = (indices >> (n - 1 - i)) & 1
        bj = (indices >> (n - 1 - j)) & 1
        zz_diag = 1 - 2 * (bi ^ bj)
        cost += np.real(np.sum(np.abs(psi) ** 2 * (1 - zz_diag) / 2))
    return float(cost)
