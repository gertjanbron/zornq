"""
B10b: 2D QAOA Parameter Optimalisatie
======================================
Kernvraag: zijn QAOA-parameters universeel overdraagbaar op 2D roosters,
net zoals in 1D (B9)?

Aanpak:
1. Optimaliseer gamma, beta op klein 2D rooster (Lx×Ly)
2. Test of die parameters werken op groter rooster
3. Vergelijk met 1D-optimale parameters (gamma*=0.4021, beta*=1.1778)

Engine: column-grouped Heisenberg-MPO (uit B10/B10c)
- Elke kolom van Ly qubits = 1 MPO-site met d=2^Ly
- Verticale edges = lokale operaties (geen SWAPs\!)
- Horizontale edges = nearest-neighbor 2-site gates
"""
import numpy as np
from numpy.linalg import svd
import time, sys

# === Column-grouped MPO engine (bewezen in B10c) ===

def bit_patterns(Ly):
    d = 2**Ly
    return np.array([[(idx >> (Ly-1-q)) & 1 for q in range(Ly)] for idx in range(d)])

def Rx(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

class ColumnGroupedQAOA:
    """Column-grouped Heisenberg-MPO engine voor 2D QAOA MaxCut."""
    
    def __init__(self, Lx, Ly, chi_max=256):
        self.Lx = Lx
        self.Ly = Ly
        self.d = 2**Ly
        self.n = Lx * Ly
        self.chi_max = chi_max
        self.bp = bit_patterns(Ly)
        self._build_hadamard()
        self._build_edges()
    
    def _build_hadamard(self):
        d, Ly, bp = self.d, self.Ly, self.bp
        H1 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
        self.Hd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            self.Hd *= H1[bp[:, q:q+1], bp[:, q:q+1].T]
    
    def _build_edges(self):
        """Tel alle edges voor normalisatie."""
        Lx, Ly = self.Lx, self.Ly
        self.n_edges = Lx * (Ly - 1) + (Lx - 1) * Ly  # vert + horiz
    
    def _build_zz_intra_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        diag = np.ones(d, dtype=complex)
        for y in range(Ly - 1):
            diag *= np.exp(-1j * gamma * (1 - 2*bp[:, y].astype(float))
                          * (1 - 2*bp[:, y+1].astype(float)))
        return diag
    
    def _build_zz_inter_diag(self, gamma):
        d, Ly, bp = self.d, self.Ly, self.bp
        iL = np.arange(d*d) // d
        iR = np.arange(d*d) % d
        diag = np.ones(d*d, dtype=complex)
        for y in range(Ly):
            diag *= np.exp(-1j * gamma * (1 - 2*bp[iL, y].astype(float))
                          * (1 - 2*bp[iR, y].astype(float)))
        return diag
    
    def _build_rx_col(self, beta):
        d, Ly, bp = self.d, self.Ly, self.bp
        rx = Rx(2 * beta)
        Rxd = np.ones((d, d), dtype=complex)
        for q in range(Ly):
            Rxd *= rx[bp[:, q:q+1], bp[:, q:q+1].T]
        return Rxd
    
    def _build_gates(self, p, gammas, betas):
        Lx = self.Lx
        gates = []
        for x in range(Lx):
            gates.append(('full', x, self.Hd))
        for l in range(p):
            zzi = self._build_zz_intra_diag(gammas[l])
            zze = self._build_zz_inter_diag(gammas[l])
            rxd = self._build_rx_col(betas[l])
            for x in range(Lx):
                gates.append(('diag1', x, zzi))
            for x in range(Lx - 1):
                gates.append(('diag2', x, zze))
            for x in range(Lx):
                gates.append(('full', x, rxd))
        return gates
    
    def _make_id_mpo(self):
        d = self.d
        return [np.eye(d, dtype=complex).reshape(1, d, d, 1).copy()
                for _ in range(self.Lx)]
    
    def _make_zz_obs(self, x1, y1, x2, y2):
        d, Ly, bp = self.d, self.Ly, self.bp
        mpo = self._make_id_mpo()
        if x1 == x2:
            diag = ((1 - 2*bp[:, y1].astype(float))
                   * (1 - 2*bp[:, y2].astype(float)))
            mpo[x1] = np.diag(diag.astype(complex)).reshape(1, d, d, 1)
        else:
            for col, y in [(x1, y1), (x2, y2)]:
                diag = (1 - 2*bp[:, y].astype(float)).astype(complex)
                mpo[col] = np.diag(diag).reshape(1, d, d, 1)
        return mpo
    
    def _ap1(self, mpo, s, U):
        Ud = U.conj().T
        W = np.einsum('ij,ajkb->aikb', Ud, mpo[s])
        mpo[s] = np.einsum('ajkb,kl->ajlb', W, U)
        return mpo
    
    def _ap1_diag(self, mpo, s, diag):
        cd = np.conj(diag)
        mpo[s] = mpo[s] * cd[None, :, None, None] * diag[None, None, :, None]
        return mpo
    
    def _ap2_diag(self, mpo, s1, diag_dd):
        d, chi_max = self.d, self.chi_max
        cl = mpo[s1].shape[0]
        cr = mpo[s1+1].shape[3]
        Th = np.einsum('aijc,cklb->aijklb', mpo[s1], mpo[s1+1])
        cd = np.conj(diag_dd).reshape(d, d)
        dd = diag_dd.reshape(d, d)
        Th = Th * cd[None, :, None, :, None, None] * dd[None, None, :, None, :, None]
        mat = Th.reshape(cl * d * d, d * d * cr)
        U_s, S, V = svd(mat, full_matrices=False)
        Sa = np.abs(S)
        k = max(1, int(np.sum(Sa > 1e-12 * Sa[0]))) if Sa[0] > 1e-15 else 1
        k = min(k, chi_max)
        mpo[s1] = U_s[:, :k].reshape(cl, d, d, k)
        mpo[s1+1] = (np.diag(S[:k]) @ V[:k, :]).reshape(k, d, d, cr)
        return mpo
    
    def _evolve(self, mpo, gates):
        for gt, s, data in reversed(gates):
            if gt == 'full':
                mpo = self._ap1(mpo, s, data)
            elif gt == 'diag1':
                mpo = self._ap1_diag(mpo, s, data)
            else:
                mpo = self._ap2_diag(mpo, s1=s, diag_dd=data)
        return mpo
    
    def _mpo_exp(self, mpo):
        L = np.ones((1,), dtype=complex)
        for W in mpo:
            L = np.einsum('a,ab->b', L, W[:, 0, 0, :])
        return L[0]
    
    def eval_cost(self, p, gammas, betas):
        """Evalueer MaxCut cost = sum_edges (1 - <ZZ>)/2."""
        Lx, Ly = self.Lx, self.Ly
        gates = self._build_gates(p, gammas, betas)
        total = 0.0
        # Vertical edges (intra-column)
        for x in range(Lx):
            for y in range(Ly - 1):
                mpo = self._make_zz_obs(x, y, x, y+1)
                mpo = self._evolve(mpo, gates)
                zz = self._mpo_exp(mpo).real
                total += (1 - zz) / 2
        # Horizontal edges (inter-column)
        for x in range(Lx - 1):
            for y in range(Ly):
                mpo = self._make_zz_obs(x, y, x+1, y)
                mpo = self._evolve(mpo, gates)
                zz = self._mpo_exp(mpo).real
                total += (1 - zz) / 2
        return total
    
    def eval_ratio(self, p, gammas, betas):
        """MaxCut approximation ratio = cost / n_edges."""
        return self.eval_cost(p, gammas, betas) / self.n_edges


# === Exact state vector reference ===

def exact_cost_2d(Lx, Ly, p, gammas, betas):
    n = Lx * Ly
    if n > 16: return None
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            if x+1 < Lx: edges.append((x*Ly+y, (x+1)*Ly+y))  # horiz
            if y+1 < Ly: edges.append((x*Ly+y, x*Ly+y+1))      # vert
    
    # Actually, need to be careful about qubit ordering.
    # In column-grouped, qubit (x,y) is bit y within site x.
    # For state vector, let's use linear index q = x*Ly + y
    
    for l in range(p):
        # ZZ phase gates
        for (qa, qb) in edges:
            g = gammas[l]
            for idx in range(dim):
                ba = (idx >> (n-1-qa)) & 1
                bb = (idx >> (n-1-qb)) & 1
                za, zb = 1-2*ba, 1-2*bb
                psi[idx] *= np.exp(-1j * g * za * zb)
        # Rx mixer
        rx = Rx(2 * betas[l])
        for k in range(n):
            pr = psi.reshape([2]*n)
            pr = np.tensordot(rx, pr, axes=([1], [k]))
            psi = np.moveaxis(pr, 0, k).reshape(dim)
    
    cost = 0
    for (qa, qb) in edges:
        for idx in range(dim):
            ba = (idx >> (n-1-qa)) & 1
            bb = (idx >> (n-1-qb)) & 1
            za, zb = 1-2*ba, 1-2*bb
            cost += abs(psi[idx])**2 * (1 - za*zb) / 2
    return cost


# ============================================================
# STAP 1: Grid search p=1 op klein rooster
# ============================================================
np.random.seed(42)
print("=" * 70)
print("  B10b: 2D QAOA Parameter Optimalisatie")
print("=" * 70)

# Optimaliseer op 3×2 (6 qubits, 7 edges) — snel genoeg voor grid search
Lx_opt, Ly_opt = 3, 2
eng = ColumnGroupedQAOA(Lx_opt, Ly_opt, chi_max=256)
print(f"\nOptimalisatie-rooster: {Lx_opt}x{Ly_opt} ({eng.n}q, {eng.n_edges} edges)")

# --- Grid search ---
print("\n--- Grid search p=1 ---")
N_grid = 25
gammas_grid = np.linspace(0.05, 1.5, N_grid)
betas_grid = np.linspace(0.05, 2.0, N_grid)

best_ratio = 0
best_g, best_b = 0, 0
t0 = time.time()

for i, g in enumerate(gammas_grid):
    for j, b in enumerate(betas_grid):
        r = eng.eval_ratio(1, [g], [b])
        if r > best_ratio:
            best_ratio = r
            best_g, best_b = g, b
    if (i+1) % 5 == 0:
        dt = time.time() - t0
        print(f"  {i+1}/{N_grid} rijen, beste ratio={best_ratio:.6f} "
              f"(g={best_g:.4f}, b={best_b:.4f}), {dt:.1f}s")
        sys.stdout.flush()

dt_grid = time.time() - t0
print(f"\nGrid search klaar in {dt_grid:.1f}s")
print(f"  Beste: gamma={best_g:.4f}, beta={best_b:.4f}, ratio={best_ratio:.6f}")
sys.stdout.flush()

# --- Gradient refinement ---
print("\n--- Gradient refinement ---")
g_cur, b_cur = best_g, best_b
step = 0.02
for iteration in range(30):
    r0 = eng.eval_ratio(1, [g_cur], [b_cur])
    
    # Finite differences
    dg = (eng.eval_ratio(1, [g_cur + step], [b_cur])
        - eng.eval_ratio(1, [g_cur - step], [b_cur])) / (2 * step)
    db = (eng.eval_ratio(1, [g_cur], [b_cur + step])
        - eng.eval_ratio(1, [g_cur], [b_cur - step])) / (2 * step)
    
    norm = np.sqrt(dg**2 + db**2)
    if norm < 1e-8:
        break
    
    # Line search
    best_lr = 0
    best_r = r0
    for lr in [0.01, 0.02, 0.05, 0.1, 0.2]:
        g_try = g_cur + lr * dg / norm
        b_try = b_cur + lr * db / norm
        r_try = eng.eval_ratio(1, [g_try], [b_try])
        if r_try > best_r:
            best_r = r_try
            best_lr = lr
    
    if best_lr == 0:
        step *= 0.5
        if step < 1e-5:
            break
        continue
    
    g_cur += best_lr * dg / norm
    b_cur += best_lr * db / norm
    
    if iteration % 5 == 0:
        print(f"  iter {iteration}: gamma={g_cur:.6f}, beta={b_cur:.6f}, "
              f"ratio={best_r:.8f}")
        sys.stdout.flush()

r_final = eng.eval_ratio(1, [g_cur], [b_cur])
print(f"\nGeoptimaliseerd p=1:")
print(f"  gamma* = {g_cur:.6f}")
print(f"  beta*  = {b_cur:.6f}")
print(f"  ratio  = {r_final:.8f}")
print(f"  (1D referentie: gamma=0.4021, beta=1.1778, ratio≈0.7498)")
sys.stdout.flush()

# === Exact verificatie ===
print("\n--- Exact verificatie op optimalisatie-rooster ---")
ex = exact_cost_2d(Lx_opt, Ly_opt, 1, [g_cur], [b_cur])
if ex is not None:
    n_edges = eng.n_edges
    print(f"  Exact cost:  {ex:.8f} (ratio={ex/n_edges:.8f})")
    print(f"  MPO cost:    {r_final*n_edges:.8f} (ratio={r_final:.8f})")
    print(f"  Error:       {abs(ex/n_edges - r_final):.2e}")
sys.stdout.flush()

# ============================================================
# STAP 2: Test parameter transfer naar grotere roosters
# ============================================================
print("\n" + "=" * 70)
print("  STAP 2: Parameter transfer")
print("=" * 70)

g_opt, b_opt = g_cur, b_cur
g_1d, b_1d = 0.4021, 1.1778  # 1D optimaal

test_lattices = [(2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (4, 4), (5, 3)]

for Lx, Ly in test_lattices:
    n = Lx * Ly
    n_e = Lx*(Ly-1) + (Lx-1)*Ly
    
    eng_test = ColumnGroupedQAOA(Lx, Ly, chi_max=256)
    
    t0 = time.time()
    r_2d = eng_test.eval_ratio(1, [g_opt], [b_opt])
    dt = time.time() - t0
    
    r_1d = eng_test.eval_ratio(1, [g_1d], [b_1d])
    
    # Exact check if small enough
    ex_str = ""
    if n <= 16:
        ex = exact_cost_2d(Lx, Ly, 1, [g_opt], [b_opt])
        if ex is not None:
            ex_str = f" exact={ex/n_e:.6f}"
    
    print(f"  {Lx}x{Ly} ({n:2d}q, {n_e:2d}e): "
          f"2D-opt={r_2d:.6f} 1D-opt={r_1d:.6f} "
          f"verschil={r_2d-r_1d:+.6f}{ex_str} ({dt:.1f}s)")
    sys.stdout.flush()

# ============================================================
# STAP 3: Zijn 2D parameters ANDERS dan 1D?
# ============================================================
print("\n" + "=" * 70)
print("  STAP 3: Parametervergelijking 2D vs 1D")
print("=" * 70)
print(f"  1D optimaal:  gamma={g_1d:.4f}, beta={b_1d:.4f}")
print(f"  2D optimaal:  gamma={g_opt:.4f}, beta={b_opt:.4f}")
delta_g = abs(g_opt - g_1d)
delta_b = abs(b_opt - b_1d)
print(f"  Verschil:     delta_gamma={delta_g:.4f}, delta_beta={delta_b:.4f}")

# Optimaliseer ook op 4×3 voor verificatie
print("\n--- Extra check: optimaliseer op 4x3 ---")
eng43 = ColumnGroupedQAOA(4, 3, chi_max=128)
print(f"  Rooster: 4x3 ({eng43.n}q, {eng43.n_edges} edges)")

# Coarse grid around known optimum
best_r43 = 0; best_g43 = g_opt; best_b43 = b_opt
for dg in np.linspace(-0.15, 0.15, 11):
    for db in np.linspace(-0.15, 0.15, 11):
        g, b = g_opt + dg, b_opt + db
        if g <= 0 or b <= 0: continue
        r = eng43.eval_ratio(1, [g], [b])
        if r > best_r43:
            best_r43 = r; best_g43 = g; best_b43 = b
    sys.stdout.flush()

print(f"  Optimaal op 4x3: gamma={best_g43:.4f}, beta={best_b43:.4f}, "
      f"ratio={best_r43:.6f}")
print(f"  Verschil met 3x2 optimum: "
      f"delta_g={abs(best_g43-g_opt):.4f}, delta_b={abs(best_b43-b_opt):.4f}")
sys.stdout.flush()

# ============================================================
# CONCLUSIE
# ============================================================
print("\n" + "=" * 70)
print("  CONCLUSIE B10b")
print("=" * 70)
print(f"  2D-optimale p=1 params: gamma={g_opt:.4f}, beta={b_opt:.4f}")
print(f"  1D-optimale p=1 params: gamma=0.4021, beta=1.1778")
print(f"  Parameters zijn {'GELIJK' if delta_g < 0.05 and delta_b < 0.05 else 'VERSCHILLEND'} "
      f"(binnen tolerantie 0.05)")
