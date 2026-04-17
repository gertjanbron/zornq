"""
B10: 2D-connectiviteitstest — Heisenberg-MPO op vierkant rooster
=================================================================
Kernvraag: hoe groeit operator-chi met circuitdiepte p op een 2D rooster?

In 1D:  chi_O = 2 bij p=5 (exact, geen truncatie)
In 2D:  chi_O = ??? — dat meten we hier

Aanpak:
- Vierkant rooster Lx × Ly, snake-ordering naar 1D MPO
- QAOA MaxCut: ZZ gates op ALLE edges (horizontaal + verticaal)
- Verticale edges = long-range in MPO (afstand Lx)
- Meet chi per bond na elke QAOA-laag
- Vergelijk met exact state vector (kleine roosters)
"""
import numpy as np
from numpy.linalg import svd
import time, sys

I2 = np.eye(2, dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Hg = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def ZZg(g):
    return np.diag([np.exp(-1j*g), np.exp(1j*g), np.exp(1j*g), np.exp(-1j*g)])


# === 2D rooster → snake ordering ===

def snake_index(x, y, Lx):
    """Snake ordering: even rij L→R, oneven rij R→L."""
    if y % 2 == 0:
        return y * Lx + x
    else:
        return y * Lx + (Lx - 1 - x)

def build_2d_edges(Lx, Ly):
    """Alle edges van vierkant rooster in snake-ordering."""
    edges = []
    for y in range(Ly):
        for x in range(Lx):
            i = snake_index(x, y, Lx)
            # Horizontaal
            if x + 1 < Lx:
                j = snake_index(x+1, y, Lx)
                edges.append((min(i,j), max(i,j)))
            # Verticaal
            if y + 1 < Ly:
                j = snake_index(x, y+1, Lx)
                edges.append((min(i,j), max(i,j)))
    return sorted(set(edges))


# === MPO Heisenberg engine (hergebruik B7d/B9) ===

def make_identity_mpo(n):
    return [I2.reshape(1,2,2,1).copy() for _ in range(n)]

def make_zz_mpo(n, i, j):
    """MPO voor Z_i Z_j."""
    mpo = []
    for k in range(n):
        if k == i or k == j:
            mpo.append(Z.reshape(1,2,2,1).copy())
        else:
            mpo.append(I2.reshape(1,2,2,1).copy())
    return mpo

def ap1(mpo, s, U):
    Ud = U.conj().T
    W = mpo[s]
    W = np.einsum('ij,ajkb->aikb', Ud, W)
    W = np.einsum('ajkb,kl->ajlb', W, U)
    mpo[s] = W
    return mpo

def ap2_nn(mpo, s1, U4, chi_max=256):
    """2-site gate op NABURIGE sites s1, s1+1."""
    Ud = U4.conj().T
    Ud4 = Ud.reshape(2,2,2,2)
    Uf4 = U4.reshape(2,2,2,2)
    cl = mpo[s1].shape[0]
    cr = mpo[s1+1].shape[3]

    Th = np.einsum('abce,edfg->abcdfg', mpo[s1], mpo[s1+1])
    Th = np.einsum('ijkl,akclef->aicjef', Ud4, Th)
    Th = np.einsum('ijkl,abkdlf->aibjdf', Uf4, Th)
    Th = Th.transpose(0, 2, 1, 4, 3, 5)
    mat = Th.reshape(cl * 4, 4 * cr)

    U_s, S, V = svd(mat, full_matrices=False)
    Sa = np.abs(S)
    k = max(1, int(np.sum(Sa > 1e-12 * Sa[0]))) if Sa[0] > 1e-15 else 1
    k = min(k, chi_max)
    trunc = float(np.sum(Sa[k:]**2)) if k < len(Sa) else 0.0

    mpo[s1]   = U_s[:, :k].reshape(cl, 2, 2, k)
    mpo[s1+1] = (np.diag(S[:k]) @ V[:k, :]).reshape(k, 2, 2, cr)
    return mpo, trunc

def swap_gate():
    """SWAP gate als 4×4 matrix."""
    S = np.zeros((4,4), dtype=complex)
    S[0,0] = 1; S[1,2] = 1; S[2,1] = 1; S[3,3] = 1
    return S

def ap2_longrange(mpo, s1, s2, U4, chi_max=256):
    """2-site gate op NIET-naburige sites via SWAP-ladder.

    s1 < s2. We swappen site s2 naar s1+1, passen de gate toe,
    en swappen terug.
    """
    SWAP = swap_gate()
    total_trunc = 0.0

    # Swap s2 naar s1+1
    for k in range(s2-1, s1, -1):
        mpo, tr = ap2_nn(mpo, k-1, SWAP, chi_max)
        total_trunc += tr
        mpo, tr = ap2_nn(mpo, k, SWAP, chi_max)  # this is wrong, let me fix
        # Actually: swap k with k+1 means gate on (k, k+1)
        # But we need to bring s2 DOWN to s1+1
        # s2 is at position s2. Swap with s2-1, then s2-2, ..., down to s1+1

    # Actually, let me redo this properly.
    # We want to apply U on (s1, s2) where s2 > s1+1.
    # Strategy: swap s2 adjacent to s1, apply U, swap back.
    # But for Heisenberg picture, we can use the identity:
    # U†(s1,s2) · O · U(s1,s2) = SWAP†...SWAP† · U†(s1,s1+1) · SWAP...SWAP · O · SWAP†...SWAP† · U(s1,s1+1) · SWAP...SWAP
    # This is equivalent to: bring s2 next to s1, apply gate, bring back.

    # Simpler: just do SWAP ladder
    return mpo, total_trunc

def apply_long_range_gate(mpo, s1, s2, U4, chi_max=256):
    """Apply U on sites (s1, s2) where s2 > s1, via SWAP network."""
    if s2 == s1 + 1:
        return ap2_nn(mpo, s1, U4, chi_max)

    SWAP = swap_gate()
    total_trunc = 0.0

    # Bring s2 down to s1+1 via SWAPs
    for k in range(s2 - 1, s1, -1):
        mpo, tr = ap2_nn(mpo, k, SWAP, chi_max)
        total_trunc += tr

    # Apply the actual gate at (s1, s1+1)
    mpo, tr = ap2_nn(mpo, s1, U4, chi_max)
    total_trunc += tr

    # Bring back up via SWAPs
    for k in range(s1 + 1, s2):
        mpo, tr = ap2_nn(mpo, k, SWAP, chi_max)
        total_trunc += tr

    return mpo, total_trunc


def mpo_expect(mpo, states):
    L = np.ones((1,), dtype=complex)
    for i in range(len(mpo)):
        s = states[i]
        L = np.einsum('a,ab->b', L, mpo[i][:, s, s, :])
    return L[0]

def get_chi_profile(mpo):
    """Bond dimensions van alle bonds."""
    return [mpo[i].shape[3] for i in range(len(mpo)-1)]

def max_chi(mpo):
    return max(get_chi_profile(mpo)) if len(mpo) > 1 else 1


# === QAOA circuit op 2D rooster ===

def build_qaoa_gates_2d(n, edges, p, gammas, betas):
    """Bouw de volledige QAOA gate-lijst voor 2D rooster."""
    gates = []

    # Hadamard op alle qubits (|+>^n)
    for k in range(n):
        gates.append(('1', k, -1, Hg))

    for l in range(p):
        # Fase-separatie: ZZ op alle edges
        zz = ZZg(gammas[l])
        for (i, j) in edges:
            gates.append(('2', i, j, zz))

        # Mixer: Rx op alle qubits
        rx = Rx(2 * betas[l])
        for k in range(n):
            gates.append(('1', k, -1, rx))

    return gates


def evolve_heisenberg_2d(mpo, gates, chi_max=256):
    """Heisenberg evolutie: gates in omgekeerde volgorde."""
    total_trunc = 0.0
    for gt, s1, s2, U in reversed(gates):
        if gt == '1':
            mpo = ap1(mpo, s1, U)
        else:
            mpo, tr = apply_long_range_gate(mpo, s1, s2, U, chi_max)
            total_trunc += tr
    return mpo, total_trunc


def eval_cost_2d(Lx, Ly, p, gammas, betas, chi_max=256):
    """MaxCut cost op 2D rooster via Heisenberg-MPO."""
    n = Lx * Ly
    edges = build_2d_edges(Lx, Ly)
    gates = build_qaoa_gates_2d(n, edges, p, gammas, betas)

    total_cost = 0.0
    max_chi_seen = 1

    for (i, j) in edges:
        mpo = make_zz_mpo(n, i, j)
        mpo, trunc = evolve_heisenberg_2d(mpo, gates, chi_max)
        zz = mpo_expect(mpo, [0]*n).real
        total_cost += (1 - zz) / 2
        mc = max_chi(mpo)
        if mc > max_chi_seen:
            max_chi_seen = mc

    return total_cost, max_chi_seen


def eval_single_edge_2d(Lx, Ly, p, gammas, betas, ei, ej, chi_max=256):
    """Evalueer enkele edge en return cost + chi profiel."""
    n = Lx * Ly
    edges = build_2d_edges(Lx, Ly)
    gates = build_qaoa_gates_2d(n, edges, p, gammas, betas)

    mpo = make_zz_mpo(n, ei, ej)
    mpo, trunc = evolve_heisenberg_2d(mpo, gates, chi_max)
    zz = mpo_expect(mpo, [0]*n).real
    return (1 - zz) / 2, get_chi_profile(mpo), trunc


# === Exact reference ===

def exact_cost_2d(Lx, Ly, p, gammas, betas):
    """Brute force state vector voor verificatie (n <= 16)."""
    n = Lx * Ly
    if n > 16:
        return None
    dim = 2**n
    edges = build_2d_edges(Lx, Ly)

    # |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for l in range(p):
        # ZZ fase gates
        for (ia, ib) in edges:
            zz = ZZg(gammas[l])
            for idx in range(dim):
                bi_a = (idx >> (n-1-ia)) & 1
                bi_b = (idx >> (n-1-ib)) & 1
                state_2q = bi_a * 2 + bi_b
                psi[idx] *= zz[state_2q, state_2q]

        # Rx mixer
        rx = Rx(2 * betas[l])
        for k in range(n):
            pr = psi.reshape([2]*n)
            pr = np.tensordot(rx, pr, axes=([1],[k]))
            psi = np.moveaxis(pr, 0, k).reshape(dim)

    # Cost
    cost = 0
    for (ia, ib) in edges:
        for idx in range(dim):
            bi_a = (idx >> (n-1-ia)) & 1
            bi_b = (idx >> (n-1-ib)) & 1
            za = 1 - 2*bi_a
            zb = 1 - 2*bi_b
            cost += abs(psi[idx])**2 * (1 - za*zb) / 2
    return cost


# ============================================================
# MAIN: Chi-groei metingen
# ============================================================
np.random.seed(42)
print("=" * 70)
print("  B10: 2D Connectiviteitstest — Operator Chi op Vierkant Rooster")
print("=" * 70)

gamma_p1 = np.array([0.4021])
beta_p1  = np.array([1.1778])

# --- Test 1: Verificatie 3×3, p=1 ---
print("\n--- Test 1: Verificatie 3x3 (9q), p=1 ---")
t0 = time.time()
ex = exact_cost_2d(3, 3, 1, gamma_p1, beta_p1)
dt_ex = time.time() - t0
print(f"  Exact: {ex:.8f}  ({dt_ex:.2f}s)")
sys.stdout.flush()

t0 = time.time()
mc, chi = eval_cost_2d(3, 3, 1, gamma_p1, beta_p1, chi_max=256)
dt_mpo = time.time() - t0
print(f"  MPO:   {mc:.8f}  chi_max={chi}  ({dt_mpo:.2f}s)")
print(f"  Error: {abs(ex - mc):.2e}")
sys.stdout.flush()

# --- Test 2: Chi profiel voor enkele edge ---
print("\n--- Test 2: Chi-profiel per bond (3x3, p=1) ---")
edges_33 = build_2d_edges(3, 3)
print(f"  Edges: {edges_33}")
print(f"  Snake ordering: (0,0)=0 (1,0)=1 (2,0)=2 (2,1)=3 (1,1)=4 (0,1)=5 (0,2)=6 (1,2)=7 (2,2)=8")

# Test a horizontal (NN) edge and a vertical (long-range) edge
for label, ei, ej in [("horiz (0,1)", 0, 1), ("vert (2,3)", 2, 3), ("vert (1,4)", 1, 4)]:
    c, prof, tr = eval_single_edge_2d(3, 3, 1, gamma_p1, beta_p1, ei, ej, chi_max=256)
    print(f"  Edge {label}: cost={c:.6f}, chi={prof}, trunc={tr:.2e}")
sys.stdout.flush()

# --- Test 3: Chi vs diepte p ---
print("\n--- Test 3: Chi vs diepte p (3x3, 9q) ---")
for p in [1, 2, 3]:
    gammas = np.full(p, 0.4)
    betas = np.full(p, 1.18 / p)  # simple scaling
    t0 = time.time()

    # Meet chi op de "moeilijkste" edge (verticaal = long-range)
    n = 9
    edges = build_2d_edges(3, 3)
    gates = build_qaoa_gates_2d(n, edges, p, gammas, betas)

    # Middelste verticale edge
    mpo = make_zz_mpo(n, 1, 4)  # vert edge
    mpo, trunc = evolve_heisenberg_2d(mpo, gates, chi_max=512)
    chi_prof = get_chi_profile(mpo)
    zz = mpo_expect(mpo, [0]*n).real

    dt = time.time() - t0
    print(f"  p={p}: chi_max={max(chi_prof)}, chi_prof={chi_prof}, trunc={trunc:.2e}, {dt:.2f}s")
    sys.stdout.flush()

# --- Test 4: Chi vs roostergrootte ---
print("\n--- Test 4: Chi vs roostergrootte (p=1) ---")
for Lx, Ly in [(2,2), (3,3), (4,3), (4,4), (5,4)]:
    n = Lx * Ly
    if n > 20:
        chi_limit = 128
    else:
        chi_limit = 256

    t0 = time.time()
    edges = build_2d_edges(Lx, Ly)
    gates = build_qaoa_gates_2d(n, edges, 1, gamma_p1, beta_p1)

    # Measure chi on vertical edge near center
    # Find a vertical edge
    vert_edges = [(i,j) for (i,j) in edges if abs(i-j) > 1]
    if vert_edges:
        ei, ej = vert_edges[len(vert_edges)//2]
    else:
        ei, ej = edges[len(edges)//2]

    mpo = make_zz_mpo(n, ei, ej)
    mpo, trunc = evolve_heisenberg_2d(mpo, gates, chi_limit)
    chi_prof = get_chi_profile(mpo)
    zz = mpo_expect(mpo, [0]*n).real
    cost_edge = (1 - zz) / 2

    dt = time.time() - t0
    print(f"  {Lx}x{Ly} ({n:2d}q): chi_max={max(chi_prof):3d}, "
          f"edge({ei},{ej})={cost_edge:.4f}, trunc={trunc:.2e}, {dt:.2f}s")

    # Exact check for small systems
    if n <= 16:
        ex = exact_cost_2d(Lx, Ly, 1, gamma_p1, beta_p1)
        mc_total, mc_chi = eval_cost_2d(Lx, Ly, 1, gamma_p1, beta_p1, chi_limit)
        print(f"          total: exact={ex:.6f}, MPO={mc_total:.6f}, "
              f"err={abs(ex-mc_total):.2e}, chi_max={mc_chi}")

    sys.stdout.flush()

print("\n" + "=" * 70)
print("  CONCLUSIE")
print("=" * 70)
