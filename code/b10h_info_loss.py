"""
B10h: Informatieverlies bij chi-truncatie — diepteanalyse
==========================================================
Kernvraag: voor welke problemen is chi-truncatie veilig,
en wanneer gooi je cruciale informatie weg?

Drie testproblemen op 1D keten (Heisenberg-MPO, schoon meetbaar):
1. MaxCut op regulier rooster — combinatorisch, verwacht VEILIG
2. Random-gewogen MaxCut — ongeordend, verwacht MATIG
3. Gefrustreerd model (antiferro + next-nearest) — verwacht GEVAARLIJK

Voor elk meten we:
- Exacte cost (geen truncatie)
- Cost bij chi=2,4,8,16 (met truncatie)
- Relatieve fout per chi
- Totale truncatie-fout (split-norm)
- Correlatie tussen split-norm en fysische fout

Plus: QAOA op MaxCut vs QAOA op gefrustreerd model
"""
import numpy as np
from numpy.linalg import svd
import time, sys

# === 1D Heisenberg-MPO engine (bewezen in B9) ===

I2 = np.eye(2, dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def ZZg(g):
    return np.diag([np.exp(-1j*g), np.exp(1j*g), np.exp(1j*g), np.exp(-1j*g)])

def Hg():
    return np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)

def make_identity_mpo(n):
    return [I2.reshape(1,2,2,1).copy() for _ in range(n)]

def make_zz_mpo(n, i, j):
    mpo = []
    for k in range(n):
        if k == i or k == j:
            mpo.append(Z.reshape(1,2,2,1).copy())
        else:
            mpo.append(I2.reshape(1,2,2,1).copy())
    return mpo

def ap1(mpo, s, U):
    Ud = U.conj().T
    W = np.einsum('ij,ajkb->aikb', Ud, mpo[s])
    mpo[s] = np.einsum('ajkb,kl->ajlb', W, U)
    return mpo

def ap2_nn(mpo, s1, U4, chi_max=256):
    Ud = U4.conj().T
    Ud4 = Ud.reshape(2,2,2,2); Uf4 = U4.reshape(2,2,2,2)
    cl = mpo[s1].shape[0]; cr = mpo[s1+1].shape[3]
    Th = np.einsum('abce,edfg->abcdfg', mpo[s1], mpo[s1+1])
    Th = np.einsum('ijkl,akclef->aicjef', Ud4, Th)
    Th = np.einsum('ijkl,abkdlf->aibjdf', Uf4, Th)
    Th = Th.transpose(0,2,1,4,3,5)
    mat = Th.reshape(cl*4, 4*cr)
    U_s, S, V = svd(mat, full_matrices=False)
    Sa = np.abs(S)
    k = max(1, int(np.sum(Sa > 1e-12*Sa[0]))) if Sa[0]>1e-15 else 1
    k = min(k, chi_max)
    trunc = float(np.sum(Sa[k:]**2)) if k < len(Sa) else 0.0
    mpo[s1] = U_s[:,:k].reshape(cl,2,2,k)
    mpo[s1+1] = (np.diag(S[:k])@V[:k,:]).reshape(k,2,2,cr)
    return mpo, trunc

def mpo_expect(mpo, n):
    L = np.ones((1,), dtype=complex)
    for i in range(n):
        L = np.einsum('a,ab->b', L, mpo[i][:,0,0,:])
    return L[0]

def max_chi(mpo):
    return max(mpo[i].shape[3] for i in range(len(mpo)-1)) if len(mpo)>1 else 1


# === QAOA builder met gewogen edges ===

def build_qaoa_gates_weighted(n, edges, weights, p, gammas, betas):
    """QAOA gates met per-edge gewichten w_ij."""
    gates = []
    h = Hg()
    for k in range(n): gates.append(('1', k, -1, h))
    for l in range(p):
        for idx, (i, j) in enumerate(edges):
            w = weights[idx]
            zz = ZZg(gammas[l] * w)  # gewogen gamma
            gates.append(('2', i, j, zz))
        rx = Rx(2 * betas[l])
        for k in range(n): gates.append(('1', k, -1, rx))
    return gates

def evolve_heisenberg(mpo, gates, chi_max=256):
    total_trunc = 0.0
    for gt, s1, s2, U in reversed(gates):
        if gt == '1':
            mpo = ap1(mpo, s1, U)
        else:
            if s2 == s1 + 1:
                mpo, tr = ap2_nn(mpo, s1, U, chi_max)
                total_trunc += tr
    return mpo, total_trunc

def eval_cost_weighted(n, edges, weights, p, gammas, betas, chi_max=256):
    """Evalueer gewogen MaxCut cost."""
    gates = build_qaoa_gates_weighted(n, edges, weights, p, gammas, betas)
    total_cost = 0.0
    total_trunc = 0.0
    for idx, (i, j) in enumerate(edges):
        w = weights[idx]
        mpo = make_zz_mpo(n, i, j)
        mpo, tr = evolve_heisenberg(mpo, gates, chi_max)
        zz = mpo_expect(mpo, n).real
        total_cost += w * (1 - zz) / 2
        total_trunc += tr
    return total_cost, total_trunc


# === Exact state vector reference ===

def exact_qaoa_cost(n, edges, weights, p, gammas, betas):
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for l in range(p):
        for idx, (i, j) in enumerate(edges):
            w = weights[idx]
            zz = ZZg(gammas[l] * w)
            for s in range(dim):
                bi = (s >> (n-1-i)) & 1
                bj = (s >> (n-1-j)) & 1
                state_2q = bi*2 + bj
                psi[s] *= zz[state_2q, state_2q]
        rx = Rx(2 * betas[l])
        for k in range(n):
            pr = psi.reshape([2]*n)
            pr = np.tensordot(rx, pr, axes=([1],[k]))
            psi = np.moveaxis(pr, 0, k).reshape(dim)
    
    cost = 0
    for idx, (i, j) in enumerate(edges):
        w = weights[idx]
        for s in range(dim):
            bi = (s >> (n-1-i)) & 1
            bj = (s >> (n-1-j)) & 1
            za, zb = 1-2*bi, 1-2*bj
            cost += w * abs(psi[s])**2 * (1 - za*zb) / 2
    return cost


# ============================================================
# TEST 1: Uniform MaxCut (alle gewichten = 1)
# ============================================================
np.random.seed(42)
print("=" * 70)
print("  B10h: Informatieverlies bij chi-truncatie")
print("=" * 70)

n = 16
edges_nn = [(i, i+1) for i in range(n-1)]

print(f"\n{'='*70}")
print(f"  TEST 1: Uniform MaxCut — {n}q ketting, {len(edges_nn)} edges")
print(f"{'='*70}")

weights_uniform = np.ones(len(edges_nn))

for p in [1, 2, 3, 4]:
    gammas = np.full(p, 0.4021)
    betas = np.full(p, 1.1778 / p)
    
    # Exact
    cost_ex = exact_qaoa_cost(n, edges_nn, weights_uniform, p, gammas, betas)
    
    print(f"\n  p={p}: exact cost = {cost_ex:.6f}")
    for chi in [2, 4, 8, 16]:
        cost_tr, trunc = eval_cost_weighted(n, edges_nn, weights_uniform, 
                                             p, gammas, betas, chi)
        err = abs(cost_tr - cost_ex) / abs(cost_ex) * 100
        print(f"    chi={chi:2d}: cost={cost_tr:.6f}, "
              f"err={err:6.3f}%, split_norm={trunc:.2e}")
    sys.stdout.flush()


# ============================================================
# TEST 2: Random-gewogen MaxCut
# ============================================================
print(f"\n{'='*70}")
print(f"  TEST 2: Random-gewogen MaxCut — {n}q, gewichten U(0.1, 2.0)")
print(f"{'='*70}")

weights_random = np.random.uniform(0.1, 2.0, len(edges_nn))
print(f"  Gewichten: {weights_random.round(2)}")

for p in [1, 2, 3, 4]:
    gammas = np.full(p, 0.4021)
    betas = np.full(p, 1.1778 / p)
    
    cost_ex = exact_qaoa_cost(n, edges_nn, weights_random, p, gammas, betas)
    
    print(f"\n  p={p}: exact cost = {cost_ex:.6f}")
    for chi in [2, 4, 8, 16]:
        cost_tr, trunc = eval_cost_weighted(n, edges_nn, weights_random,
                                             p, gammas, betas, chi)
        err = abs(cost_tr - cost_ex) / abs(cost_ex) * 100
        print(f"    chi={chi:2d}: cost={cost_tr:.6f}, "
              f"err={err:6.3f}%, split_norm={trunc:.2e}")
    sys.stdout.flush()


# ============================================================
# TEST 3: Gefrustreerd model — NN + NNN (next-nearest-neighbor)
# ============================================================
print(f"\n{'='*70}")
print(f"  TEST 3: Gefrustreerd model — NN + NNN (J1-J2)")
print(f"{'='*70}")

# NN edges (ferromagnetisch, J1=1)
# NNN edges (antiferromagnetisch, J2=-0.5) — veroorzaakt frustratie
edges_j1j2 = []
weights_j1j2 = []

# J1: nearest-neighbor
for i in range(n-1):
    edges_j1j2.append((i, i+1))
    weights_j1j2.append(1.0)

# J2: next-nearest-neighbor (ALLEEN als dat NN is in MPO)
for i in range(n-2):
    edges_j1j2.append((i, i+2))
    weights_j1j2.append(-0.5)  # negatief = frustratie

weights_j1j2 = np.array(weights_j1j2)
print(f"  {len(edges_j1j2)} edges (NN + NNN)")
print(f"  J1=1.0 (NN), J2=-0.5 (NNN, frustrerend)")

# NNN edges zijn NIET nearest-neighbor in MPO → we moeten SWAP-ladder gebruiken
# Maar dat is complex. Laten we in plaats daarvan alleen NN edges gebruiken
# maar met AFWISSELENDE tekens (ook frustrerend).

print(f"\n  Alternatief: afwisselende gewichten (frustratie zonder NNN)")
edges_frust = [(i, i+1) for i in range(n-1)]
# Afwisselend +1 en -0.8 → frustratie: buren willen tegengestelde dingen
weights_frust = np.array([1.0 if i%2==0 else -0.8 for i in range(n-1)])
print(f"  Gewichten: {weights_frust}")

for p in [1, 2, 3, 4]:
    gammas = np.full(p, 0.4021)
    betas = np.full(p, 1.1778 / p)
    
    cost_ex = exact_qaoa_cost(n, edges_frust, weights_frust, p, gammas, betas)
    
    print(f"\n  p={p}: exact cost = {cost_ex:.6f}")
    for chi in [2, 4, 8, 16]:
        cost_tr, trunc = eval_cost_weighted(n, edges_frust, weights_frust,
                                             p, gammas, betas, chi)
        err = abs(cost_tr - cost_ex) / abs(cost_ex) * 100
        print(f"    chi={chi:2d}: cost={cost_tr:.6f}, "
              f"err={err:6.3f}%, split_norm={trunc:.2e}")
    sys.stdout.flush()


# ============================================================
# TEST 4: Sterk gefrustreerd — random +/- gewichten
# ============================================================
print(f"\n{'='*70}")
print(f"  TEST 4: Spin glass — random ±1 gewichten")
print(f"{'='*70}")

weights_glass = np.random.choice([-1.0, 1.0], size=len(edges_nn))
print(f"  Gewichten: {weights_glass.astype(int)}")

for p in [1, 2, 3, 4]:
    gammas = np.full(p, 0.4021)
    betas = np.full(p, 1.1778 / p)
    
    cost_ex = exact_qaoa_cost(n, edges_nn, weights_glass, p, gammas, betas)
    
    print(f"\n  p={p}: exact cost = {cost_ex:.6f}")
    for chi in [2, 4, 8, 16]:
        cost_tr, trunc = eval_cost_weighted(n, edges_nn, weights_glass,
                                             p, gammas, betas, chi)
        err = abs(cost_tr - cost_ex) / abs(cost_ex) * 100
        print(f"    chi={chi:2d}: cost={cost_tr:.6f}, "
              f"err={err:6.3f}%, split_norm={trunc:.2e}")
    sys.stdout.flush()


# ============================================================
# SAMENVATTING: split-norm vs fysische fout correlatie
# ============================================================
print(f"\n{'='*70}")
print(f"  SAMENVATTING: split-norm als kwaliteitsmeter")
print(f"{'='*70}")

# Collect all (split_norm, physical_error) pairs
all_pairs = []
labels = ["uniform", "random_w", "frustrated", "spin_glass"]
all_weights = [weights_uniform, weights_random, weights_frust, weights_glass]

for label, weights in zip(labels, all_weights):
    edges = edges_nn
    for p in [2, 3, 4]:
        gammas = np.full(p, 0.4021)
        betas = np.full(p, 1.1778 / p)
        cost_ex = exact_qaoa_cost(n, edges, weights, p, gammas, betas)
        for chi in [2, 4, 8]:
            cost_tr, trunc = eval_cost_weighted(n, edges, weights, p, gammas, betas, chi)
            phys_err = abs(cost_tr - cost_ex) / abs(cost_ex)
            if trunc > 1e-15:
                all_pairs.append((label, p, chi, trunc, phys_err))

print(f"\n  {'type':<12} {'p':>2} {'chi':>4} {'split_norm':>12} {'phys_err':>12} {'ratio':>10}")
print(f"  {'-'*56}")
for label, p, chi, sn, pe in sorted(all_pairs, key=lambda x: x[3]):
    ratio = pe / sn if sn > 1e-20 else 0
    print(f"  {label:<12} {p:>2} {chi:>4} {sn:>12.2e} {pe:>12.2e} {ratio:>10.2f}")

print("\nDone.")
