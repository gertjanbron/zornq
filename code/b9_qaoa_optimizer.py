"""
B9: QAOA MaxCut Optimizer via Heisenberg-MPO + Lightcone
=========================================================
Volledige QAOA-solver op een laptop.
- Heisenberg-MPO: operator evolueren ipv toestand
- Lightcone-afsnijding: O(1) per edge, O(n) totaal
- Translatie-invariantie: bulk edges identiek
- Optimize-on-small-n: parameters universeel voor bulk,
  optimaliseer op n=20, pas toe op n=10.000
- Schaalt naar 10.000+ qubits

MaxCut op path graph: C = sum (1 - Z_i Z_{i+1})/2
Max cut = n-1 (bipartiet graf)
"""
import numpy as np
from numpy.linalg import svd
import time, sys

# === Basismatrices ===
I2 = np.eye(2, dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
Hg = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def ZZg(g):
    return np.diag([np.exp(-1j*g), np.exp(1j*g), np.exp(1j*g), np.exp(-1j*g)])


# === MPO Heisenberg engine ===

def ap1(mpo, s, U):
    """1-qubit gate in Heisenberg picture: W -> U dag W U"""
    Ud = U.conj().T
    W = mpo[s]
    W = np.einsum('ij,ajkb->aikb', Ud, W)
    W = np.einsum('ajkb,kl->ajlb', W, U)
    mpo[s] = W
    return mpo

def ap2(mpo, s1, U4, chi=32):
    """2-qubit gate in Heisenberg picture met SVD-truncatie."""
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
    k = max(1, int(np.sum(Sa > 1e-14 * Sa[0]))) if Sa[0] > 1e-15 else 1
    k = min(k, chi)

    mpo[s1]   = U_s[:, :k].reshape(cl, 2, 2, k)
    mpo[s1+1] = (np.diag(S[:k]) @ V[:k, :]).reshape(k, 2, 2, cr)
    return mpo

def mpo_expect(mpo, states):
    """Verwachtingswaarde <states|O|states> via MPO-contractie."""
    L = np.ones((1,), dtype=complex)
    for i in range(len(mpo)):
        s = states[i]
        L = np.einsum('a,ab->b', L, mpo[i][:, s, s, :])
    return L[0]


# === Lightcone-windowed evaluation ===

def eval_zz_window(edge_pos, wsize, p, gammas, betas):
    """<Z_e Z_{e+1}> via Heisenberg-MPO in een lokaal venster."""
    n = wsize
    gates = []
    for k in range(n):
        gates.append(('1', k, Hg))
    for l in range(p):
        zz = ZZg(gammas[l])
        for i in range(n-1):
            gates.append(('2', i, zz))
        rx = Rx(2 * betas[l])
        for k in range(n):
            gates.append(('1', k, rx))

    mpo = []
    for k in range(n):
        if k == edge_pos or k == edge_pos + 1:
            mpo.append(Z.reshape(1, 2, 2, 1).copy())
        else:
            mpo.append(I2.reshape(1, 2, 2, 1).copy())

    for gt, s1, U in reversed(gates):
        if gt == '1':
            mpo = ap1(mpo, s1, U)
        else:
            mpo = ap2(mpo, s1, U, 32)

    return mpo_expect(mpo, [0] * n).real


def eval_cost(n, p, gammas, betas):
    """MaxCut cost met lightcone-venster en translatie-invariantie."""
    w = min(2*p + 6, n)
    bd = min(p + 2, n - 1)

    bp = w // 2 - 1
    bulk = eval_zz_window(bp, w, p, gammas, betas)

    computed = {}
    for i in range(bd):
        if i < w - 1:
            computed[i] = eval_zz_window(i, w, p, gammas, betas)

    for i in range(bd):
        ei = n - 2 - i
        if ei not in computed and ei >= 0:
            pi_w = w - 2 - i
            if 0 <= pi_w < w - 1:
                computed[ei] = eval_zz_window(pi_w, w, p, gammas, betas)

    zz = sum(computed.get(i, bulk) for i in range(n - 1))
    return (n - 1) / 2 - zz / 2


# === Optimizer: small-n + grid search ===

def grid_search_p1(n_opt=20, g_range=(0.1, 1.5), b_range=(0.1, 1.6), steps=40):
    """2D grid search voor p=1 op klein n."""
    gammas = np.linspace(g_range[0], g_range[1], steps)
    betas  = np.linspace(b_range[0], b_range[1], steps)

    best_c = -1
    best_g, best_b = 0.5, 0.5

    for g in gammas:
        for b in betas:
            c = eval_cost(n_opt, 1, np.array([g]), np.array([b]))
            if c > best_c:
                best_c = c
                best_g, best_b = g, b

    return best_c, best_g, best_b


def refine_gradient(n_opt, p, gammas, betas, iters=30, lr=0.01, ep=5e-4):
    """Gradient descent verfijning op klein n."""
    g, b = gammas.copy(), betas.copy()
    best_c = -1
    best_g, best_b = g.copy(), b.copy()

    for it in range(iters):
        c0 = eval_cost(n_opt, p, g, b)
        if c0 > best_c:
            best_c = c0
            best_g, best_b = g.copy(), b.copy()

        gg = np.zeros(p)
        gb = np.zeros(p)
        for l in range(p):
            gp = g.copy(); gp[l] += ep
            gg[l] = (eval_cost(n_opt, p, gp, b) - c0) / ep
            bp = b.copy(); bp[l] += ep
            gb[l] = (eval_cost(n_opt, p, g, bp) - c0) / ep

        # Gradient ASCENT (maximaliseer cut)
        g += lr * gg
        b += lr * gb
        g = np.clip(g, 0.01, np.pi)
        b = np.clip(b, 0.01, np.pi / 2)

    c = eval_cost(n_opt, p, g, b)
    if c > best_c:
        best_c = c
        best_g, best_b = g.copy(), b.copy()

    return best_c, best_g, best_b


def optimize_p1(n_opt=20):
    """Volledige p=1 optimalisatie: grid search + gradient refinement."""
    _, g, b = grid_search_p1(n_opt, steps=40)
    c, g_arr, b_arr = refine_gradient(
        n_opt, 1, np.array([g]), np.array([b]), iters=40, lr=0.005
    )
    return c, g_arr, b_arr


def optimize_multi_p(n_opt, p, restarts=5, iters=60, lr=0.01):
    """Optimalisatie voor p>1 via random restarts + gradient descent."""
    best = -1
    best_p = None

    for r in range(restarts):
        g = np.random.uniform(0.2, 1.0, p)
        b = np.random.uniform(0.2, 0.8, p)
        c, g, b = refine_gradient(n_opt, p, g, b, iters=iters, lr=lr)
        if c > best:
            best = c
            best_p = (g.copy(), b.copy())

    return best, best_p


# === Exact reference (voor kleine n) ===

def exact_cost(n, p, gammas, betas):
    """Brute-force state vector simulatie. Alleen voor n <= 14."""
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for l in range(p):
        for i in range(n-1):
            for idx in range(dim):
                bi = [(idx >> (n-1-k)) & 1 for k in range(n)]
                psi[idx] *= np.exp(-1j * gammas[l] * (1 - 2*bi[i]) * (1 - 2*bi[i+1]))
        rx = Rx(2 * betas[l])
        for k in range(n):
            pr = psi.reshape([2] * n)
            pr = np.tensordot(rx, pr, axes=([1], [k]))
            psi = np.moveaxis(pr, 0, k).reshape(dim)

    cost = 0
    for i in range(n-1):
        for idx in range(dim):
            bi = [(idx >> (n-1-k)) & 1 for k in range(n)]
            cost += abs(psi[idx])**2 * (1 - (1 - 2*bi[i]) * (1 - 2*bi[i+1])) / 2
    return cost


# ============================================================
# MAIN
# ============================================================
np.random.seed(42)
print("=" * 70)
print("  B9: QAOA MaxCut Solver — Heisenberg-MPO + Lightcone")
print("=" * 70)

# --- Stap 1: Verificatie MPO vs exact ---
print("\n--- Verificatie tegen exact ---")
for n in [8, 12]:
    g = np.array([0.41])
    b = np.array([1.18])
    ex = exact_cost