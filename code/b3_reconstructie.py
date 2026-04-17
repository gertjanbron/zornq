"""
B3: Reconstructie-algoritme
Gegeven 7 operaties x 7 decomposities meetuitkomsten,
reconstrueer de state vector via pseudo-inverse.
"""
import numpy as np
import time

# === Zorn algebra ===
def zmul(A, B):
    a,al,be,b = A[0],A[1:4],A[4:7],A[7]
    c,ga,de,d = B[0],B[1:4],B[4:7],B[7]
    return np.array([a*c+al@de, *(a*ga+d*al+np.cross(be,de)),
                     *(c*be+b*de-np.cross(al,ga)), be@ga+b*d])
def zconj(A): return np.array([A[7],*(-A[4:7]),*(-A[1:4]),A[0]])
def zinv(A):
    c=zconj(A); n=A[0]*A[7]-A[1:4]@A[4:7]
    return c/n if abs(n)>1e-15 else c
def zhodge(A): return np.array([A[0],A[2],A[3],A[1],A[5],A[6],A[4],A[7]])
def zassoc(A,B,C): return zmul(zmul(A,B),C)-zmul(A,zmul(B,C))
def zjordan(A,B): return zmul(zmul(A,B),A)

FANO=[(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
def pdec(A,d):
    t=FANO[d]; comp=[x for x in range(1,8) if x not in t]
    return A[[0]+list(t)+comp]

SHORT={0:'x',1:'+',2:'-',3:'/',4:'H',5:'[.]',6:'ABA'}

# === Bouw transfer matrix ===
def build_transfer_matrix(seed=42):
    """Bouw de volledige transfer matrix T voor 6 qubits (64D).
    T heeft rank 64 en kan gebruikt worden voor state reconstructie.
    Retourneert: T (N x 64), en beschrijving van elke rij."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    
    for op in range(7):
        for d in range(7):
            if op == 0:  # x: bilineair
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                        T[:,j*8+k]=zmul(pdec(ej,d),pdec(ek,d))
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
                
            elif op in (1,2):  # +/- : (A+/-B)xC
                C = pdec(rng.standard_normal(8), d)
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                        if op==1: T[:,j*8+k]=zmul(pdec(ej,d)+pdec(ek,d),C)
                        else: T[:,j*8+k]=zmul(pdec(ej,d)-pdec(ek,d),C)
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
                
            elif op == 3:  # /: Jacobiaan van inv(A)xB
                Z0=rng.standard_normal(8); eps=1e-7
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ek_v=np.zeros(8);ek_v[k]=1
                        zp=Z0.copy();zp[j]+=eps;zm=Z0.copy();zm[j]-=eps
                        T[:,j*8+k]=(zmul(zinv(pdec(zp,d)),pdec(ek_v,d))-zmul(zinv(pdec(zm,d)),pdec(ek_v,d)))/(2*eps)
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
                
            elif op == 4:  # H: Hodge varianten
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                        T[:,j*8+k]=zmul(zhodge(pdec(ej,d)),pdec(ek,d))
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
                
            elif op == 5:  # [.]: associator
                C = pdec(rng.standard_normal(8), d)
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1;ek=np.zeros(8);ek[k]=1
                        T[:,j*8+k]=zassoc(pdec(ej,d),pdec(ek,d),C)
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
                
            elif op == 6:  # ABA: Jacobiaan
                Z0=rng.standard_normal(8); eps=1e-7
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ek_v=np.zeros(8);ek_v[k]=1
                        zp=Z0.copy();zp[j]+=eps;zm=Z0.copy();zm[j]-=eps
                        T[:,j*8+k]=(zjordan(pdec(zp,d),pdec(ek_v,d))-zjordan(pdec(zm,d),pdec(ek_v,d)))/(2*eps)
                rows.append(T)
                labels.extend([(SHORT[op],d,f'out{i}') for i in range(8)])
    
    T_full = np.vstack(rows)
    # Verwijder slechte rijen
    good = ~np.any(np.isnan(T_full)|np.isinf(T_full), axis=1)
    good &= np.linalg.norm(T_full, axis=1) > 1e-15
    return T_full[good], [l for l,g in zip(labels, good) if g]


def reconstruct(T, measurements):
    """Reconstrueer state vector uit meetuitkomsten via pseudo-inverse.
    T: transfer matrix (M x N)
    measurements: vector (M,) van meetuitkomsten
    Retourneert: gereconstrueerde state vector (N,)
    """
    T_pinv = np.linalg.pinv(T)
    return T_pinv @ measurements


def measure_state(T, psi):
    """Simuleer meetuitkomsten: y = T @ psi"""
    return T @ psi


def fidelity(psi_true, psi_recon):
    """Bereken fidelity |<psi_true|psi_recon>|^2"""
    psi_t = psi_true / np.linalg.norm(psi_true)
    psi_r = psi_recon / np.linalg.norm(psi_recon)
    return abs(np.dot(psi_t.conj(), psi_r))**2


# === Quantum toestanden genereren ===
def state_computational(n_qubits, bitstring):
    """Computationele basistoestand |bitstring>"""
    dim = 2**n_qubits
    idx = int(bitstring, 2)
    psi = np.zeros(dim)
    psi[idx] = 1.0
    return psi

def state_ghz(n_qubits):
    """GHZ: (|000...0> + |111...1>) / sqrt(2)"""
    dim = 2**n_qubits
    psi = np.zeros(dim)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return psi

def state_bell():
    """Bell |Phi+> = (|00> + |11>) / sqrt(2), embedded in 6q"""
    # Eerste 2 qubits Bell, rest |0>
    psi = np.zeros(64)
    psi[0] = 1/np.sqrt(2)        # |000000>
    psi[0b110000] = 1/np.sqrt(2)  # |110000>
    return psi

def state_w(n_qubits):
    """W-state: (|100..0> + |010..0> + ... + |000..1>) / sqrt(n)"""
    dim = 2**n_qubits
    psi = np.zeros(dim)
    for i in range(n_qubits):
        idx = 1 << (n_qubits - 1 - i)
        psi[idx] = 1/np.sqrt(n_qubits)
    return psi

def state_random(n_qubits, seed=123):
    """Random Haar-verdeelde toestand"""
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    psi = rng.standard_normal(dim) + 1j*rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)

def state_qaoa_1d(n_qubits, gamma=0.3, beta=0.5):
    """QAOA 1-laag op 1D keten"""
    dim = 2**n_qubits
    # Start: |+>^n
    psi = np.ones(dim) / np.sqrt(dim)
    # ZZ interactie op naburige paren
    for i in range(n_qubits-1):
        for idx in range(dim):
            bi = (idx >> (n_qubits-1-i)) & 1
            bj = (idx >> (n_qubits-2-i)) & 1
            phase = gamma * (1 - 2*(bi ^ bj))
            psi[idx] *= np.exp(1j * phase)
    # Rx mixer op alle qubits
    H = np.array([[np.cos(beta), -1j*np.sin(beta)],
                   [-1j*np.sin(beta), np.cos(beta)]])
    for q in range(n_qubits):
        psi_new = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            b = (idx >> (n_qubits-1-q)) & 1
            idx_flip = idx ^ (1 << (n_qubits-1-q))
            psi_new[idx] += H[b, 0] * psi[idx & ~(1 << (n_qubits-1-q))]
            psi_new[idx] += H[b, 1] * psi[idx ^ (1 << (n_qubits-1-q))]
        psi = psi_new
    return psi / np.linalg.norm(psi)

def state_vqe_simple(n_qubits, seed=77):
    """Simpele VQE-achtige toestand: Ry rotaties + CNOT ladder"""
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    # Ry rotaties
    for q in range(n_qubits):
        theta = rng.uniform(0, np.pi)
        Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                        [np.sin(theta/2), np.cos(theta/2)]])
        psi_new = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            b = (idx >> (n_qubits-1-q)) & 1
            idx0 = idx & ~(1 << (n_qubits-1-q))  # bit q = 0
            idx1 = idx0 | (1 << (n_qubits-1-q))   # bit q = 1
            psi_new[idx] = Ry[b,0]*psi[idx0] + Ry[b,1]*psi[idx1]
        psi = psi_new
    # CNOT ladder
    for q in range(n_qubits-1):
        psi_new = psi.copy()
        for idx in range(dim):
            ctrl = (idx >> (n_qubits-1-q)) & 1
            if ctrl:
                tgt_bit = 1 << (n_qubits-2-q)
                idx_flip = idx ^ tgt_bit
                psi_new[idx], psi_new[idx_flip] = psi[idx_flip], psi[idx]
        psi = psi_new
    return psi / np.linalg.norm(psi)


# === CHSH test ===
def chsh_from_state(psi_64):
    """Bereken CHSH waarde uit een 6-qubit state.
    Gebruik qubits 0,1 als het Bell-paar.
    CHSH = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    met a=Z, a'=X, b=(Z+X)/sqrt2, b'=(Z-X)/sqrt2
    """
    dim = len(psi_64)
    n = int(np.log2(dim))
    
    # Reduceer naar 2-qubit density matrix (trace over qubits 2-5)
    psi_mat = psi_64.reshape([2]*n)
    # Partial trace: sum over qubits 2,3,4,5
    rho_2q = np.zeros((4,4), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for rest in np.ndindex(*([2]*(n-2))):
                        idx_bra = (i,j) + rest
                        idx_ket = (k,l) + rest
                        rho_2q[i*2+j, k*2+l] += psi_mat[idx_bra] * psi_mat[idx_ket].conj()
    
    # Pauli matrices
    I2 = np.eye(2)
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])
    
    def expect(A, B):
        M = np.kron(A, B)
        return np.real(np.trace(rho_2q @ M))
    
    # Alice: a=Z, a'=X
    # Bob: b=(Z+X)/sqrt2, b'=(Z-X)/sqrt2
    b_plus = (Z + X) / np.sqrt(2)
    b_minus = (Z - X) / np.sqrt(2)
    
    S = expect(Z, b_plus) - expect(Z, b_minus) + expect(X, b_plus) + expect(X, b_minus)
    return abs(S)


# === MAIN ===
if __name__ == '__main__':
    print("="*60)
    print("  B3: RECONSTRUCTIE-ALGORITME")
    print("="*60)
    
    # Stap 1: bouw transfer matrix
    print("\nStap 1: Bouw transfer matrix...")
    t0 = time.time()
    T, labels = build_transfer_matrix(seed=42)
    rank = np.linalg.matrix_rank(T, tol=1e-10)
    print(f"  T shape: {T.shape}, rank: {rank}/64")
    
    # Bereken pseudo-inverse
    T_pinv = np.linalg.pinv(T)
    print(f"  T_pinv shape: {T_pinv.shape}")
    print(f"  Conditiegetal: {np.linalg.cond(T):.2e}")
    print(f"  ({time.time()-t0:.1f}s)")
    
    # Stap 2: Test op diverse toestanden
    print(f"\n{'='*60}")
    print("  RECONSTRUCTIE FIDELITY (6 qubits, 64D)")
    print(f"{'='*60}")
    
    states = {
        '|000000>': state_computational(6, '000000'),
        '|111111>': state_computational(6, '111111'),
        '|101010>': state_computational(6, '101010'),
        'GHZ-6': state_ghz(6),
        'Bell(01)+|0000>': state_bell(),
        'W-6': state_w(6),
        'QAOA 1D 1L': state_qaoa_1d(6),
        'VQE simple': state_vqe_simple(6),
        'Random Haar': state_random(6, seed=123),
        'Random Haar 2': state_random(6, seed=456),
        'Random Haar 3': state_random(6, seed=789),
    }
    
    print(f"\n  {'Toestand':<20s}  {'Fidelity':>12s}  {'Fout':>12s}  {'Norm ratio':>10s}")
    print(f"  {'-'*58}")
    
    all_fidelities = []
    for name, psi in states.items():
        # Neem reëel deel voor de transfer matrix (die is reëel)
        psi_real = np.real(psi) if np.max(np.abs(np.imag(psi))) < 1e-14 else np.abs(psi)
        
        # Meet
        y = T @ psi_real
        # Reconstrueer
        psi_hat = T_pinv @ y
        # Fidelity
        f = fidelity(psi_real, psi_hat)
        err = np.linalg.norm(psi_real - psi_hat) / np.linalg.norm(psi_real)
        norm_ratio = np.linalg.norm(psi_hat) / np.linalg.norm(psi_real)
        
        all_fidelities.append(f)
        print(f"  {name:<20s}  {f:12.10f}  {err:12.2e}  {norm_ratio:10.6f}")
    
    print(f"\n  Gemiddelde fidelity: {np.mean(all_fidelities):.10f}")
    print(f"  Minimum fidelity:   {np.min(all_fidelities):.10f}")
    
    # Stap 3: Complex-waardige toestanden
    print(f"\n{'='*60}")
    print("  COMPLEX-WAARDIGE TOESTANDEN")
    print(f"{'='*60}")
    print("\n  De transfer matrix T is reeel-waardig.")
    print("  Voor complexe psi: reconstrueer Re(psi) en Im(psi) apart.")
    
    complex_states = {
        'QAOA (complex)': state_qaoa_1d(6),
        'VQE (complex)': state_vqe_simple(6),
        'Random complex': state_random(6, seed=999),
    }
    
    print(f"\n  {'Toestand':<20s}  {'Fidelity':>12s}  {'Fout':>12s}")
    print(f"  {'-'*48}")
    
    for name, psi in complex_states.items():
        # Splits in Re en Im
        psi_re = np.real(psi)
        psi_im = np.imag(psi)
        
        y_re = T @ psi_re
        y_im = T @ psi_im
        
        psi_hat_re = T_pinv @ y_re
        psi_hat_im = T_pinv @ y_im
        psi_hat = psi_hat_re + 1j * psi_hat_im
        
        # Complex fidelity
        f = abs(np.dot(psi.conj(), psi_hat / np.linalg.norm(psi_hat)))**2
        err = np.linalg.norm(psi - psi_hat) / np.linalg.norm(psi)
        print(f"  {name:<20s}  {f:12.10f}  {err:12.2e}")
    
    # Stap 4: CHSH test
    print(f"\n{'='*60}")
    print("  CHSH TEST")
    print(f"{'='*60}")
    
    # Maak Bell-toestand en reconstrueer
    psi_bell = state_bell()
    y_bell = T @ psi_bell
    psi_bell_hat = T_pinv @ y_bell
    
    chsh_orig = chsh_from_state(psi_bell)
    chsh_recon = chsh_from_state(psi_bell_hat)
    
    print(f"\n  Bell state (origineel):       CHSH = {chsh_orig:.6f}")
    print(f"  Bell state (gereconstrueerd): CHSH = {chsh_recon:.6f}")
    print(f"  Tsirelson-grens:              CHSH = {2*np.sqrt(2):.6f}")
    print(f"  Klassieke grens:              CHSH = 2.000000")
    
    # QAOA state
    psi_qaoa = state_qaoa_1d(6)
    psi_qaoa_real = np.real(psi_qaoa) if np.max(np.abs(np.imag(psi_qaoa))) < 1e-14 else np.abs(psi_qaoa)
    y_qaoa = T @ psi_qaoa_real
    psi_qaoa_hat = T_pinv @ y_qaoa
    chsh_qaoa = chsh_from_state(psi_qaoa_hat)
    print(f"  QAOA (gereconstrueerd):       CHSH = {chsh_qaoa:.6f}")
    
    # Stap 5: Conditie en stabiliteit
    print(f"\n{'='*60}")
    print("  STABILITEIT (ruis-tolerantie)")
    print(f"{'='*60}")
    
    psi_test = state_ghz(6)
    y_clean = T @ psi_test
    
    print(f"\n  {'SNR (dB)':<12s}  {'Fidelity':>12s}  {'Fout':>12s}")
    print(f"  {'-'*40}")
    for snr_db in [100, 80, 60, 40, 30, 20, 10]:
        noise_std = np.linalg.norm(y_clean) * 10**(-snr_db/20)
        y_noisy = y_clean + np.random.default_rng(42).standard_normal(len(y_clean)) * noise_std
        psi_hat = T_pinv @ y_noisy
        f = fidelity(psi_test, psi_hat)
        err = np.linalg.norm(psi_test - psi_hat) / np.linalg.norm(psi_test)
        print(f"  {snr_db:<12d}  {f:12.10f}  {err:12.2e}")

    print(f"\n{'='*60}")
    print("  KLAAR")
    print(f"{'='*60}")
