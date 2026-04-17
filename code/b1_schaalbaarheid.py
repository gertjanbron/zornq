"""
B1: Schaalbaarheid 7-operatie informatiecompleetheid
Test of rank = 2^n voor n = 6, 9, 12 (en hoger indien haalbaar).

Methode: bilineaire transfer matrix per groepenpaar,
getensord met identiteit op overige groepen.
"""
import numpy as np
from itertools import combinations
import time, sys

# === Zorn algebra ===
def zmul(A, B):
    a,al,be,b = A[0],A[1:4],A[4:7],A[7]
    c,ga,de,d = B[0],B[1:4],B[4:7],B[7]
    return np.array([a*c+al@de, *(a*ga+d*al+np.cross(be,de)),
                     *(c*be+b*de-np.cross(al,ga)), be@ga+b*d])
def zadd(A,B): return A+B
def zsub(A,B): return A-B
def zconj(A): return np.array([A[7], *(-A[4:7]), *(-A[1:4]), A[0]])
def znorm(A): return A[0]*A[7] - A[1:4]@A[4:7]
def zinv(A):
    c=zconj(A); n=znorm(A)
    return c/n if abs(n)>1e-15 else c
def zhodge(A): return np.array([A[0], A[2],A[3],A[1], A[5],A[6],A[4], A[7]])
def zassoc(A,B,C): return zmul(zmul(A,B),C) - zmul(A,zmul(B,C))
def zjordan(A,B): return zmul(zmul(A,B),A)

FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
def pdec(A, d):
    t=FANO[d]; comp=[x for x in range(1,8) if x not in t]
    return A[[0]+list(t)+comp]

SHORT = {0:'x', 1:'+', 2:'-', 3:'/', 4:'H', 5:'[.]', 6:'ABA'}
rng = np.random.default_rng(42)

def build_local_transfer_matrices():
    """Bouw de 8xN transfer matrices voor elke operatie, alle 7 decomps.
    Retourneert dict: op_idx -> np.array van shape (rows, 64)
    """
    OP = {}
    for op in range(7):
        matrices = []
        for d in range(7):
            if op == 0:  # x bilineair
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1; ek=np.zeros(8);ek[k]=1
                        T[:, j*8+k] = zmul(pdec(ej,d), pdec(ek,d))
                matrices.append(T)
            elif op == 1:  # (A+B)xC
                for _ in range(2):
                    C = pdec(rng.standard_normal(8), d)
                    T = np.zeros((8, 64))
                    for j in range(8):
                        for k in range(8):
                            ej=np.zeros(8);ej[j]=1; ek=np.zeros(8);ek[k]=1
                            T[:, j*8+k] = zmul(pdec(ej,d)+pdec(ek,d), C)
                    matrices.append(T)
            elif op == 2:  # (A-B)xC
                for _ in range(2):
                    C = pdec(rng.standard_normal(8), d)
                    T = np.zeros((8, 64))
                    for j in range(8):
                        for k in range(8):
                            ej=np.zeros(8);ej[j]=1; ek=np.zeros(8);ek[k]=1
                            T[:, j*8+k] = zmul(pdec(ej,d)-pdec(ek,d), C)
                    matrices.append(T)
            elif op == 3:  # inv
                for _ in range(3):
                    Z0 = rng.standard_normal(8); eps=1e-7
                    T = np.zeros((8, 64))
                    for j in range(8):
                        for k in range(8):
                            ek_v=np.zeros(8);ek_v[k]=1
                            zp=Z0.copy();zp[j]+=eps; zm=Z0.copy();zm[j]-=eps
                            T[:,j*8+k]=(zmul(zinv(pdec(zp,d)),pdec(ek_v,d))-zmul(zinv(pdec(zm,d)),pdec(ek_v,d)))/(2*eps)
                    matrices.append(T)
            elif op == 4:  # Hodge
                T1=np.zeros((8,64)); T2=np.zeros((8,64)); T3=np.zeros((8,64))
                for j in range(8):
                    for k in range(8):
                        ej=np.zeros(8);ej[j]=1; ek=np.zeros(8);ek[k]=1
                        T1[:,j*8+k]=zmul(zhodge(pdec(ej,d)),pdec(ek,d))
                        T2[:,j*8+k]=zmul(pdec(ej,d),zhodge(pdec(ek,d)))
                        T3[:,j*8+k]=zmul(zhodge(zhodge(pdec(ej,d))),pdec(ek,d))
                matrices.extend([T1,T2,T3])
            elif op == 5:  # Associator
                for _ in range(2):
                    C = pdec(rng.standard_normal(8), d)
                    T = np.zeros((8, 64))
                    for j in range(8):
                        for k in range(8):
                            ej=np.zeros(8);ej[j]=1; ek=np.zeros(8);ek[k]=1
                            T[:,j*8+k] = zassoc(pdec(ej,d), pdec(ek,d), C)
                    matrices.append(T)
            elif op == 6:  # Jordan ABA
                for _ in range(3):
                    Z0=rng.standard_normal(8); eps=1e-7
                    T = np.zeros((8, 64))
                    for j in range(8):
                        for k in range(8):
                            ek_v=np.zeros(8);ek_v[k]=1
                            zp=Z0.copy();zp[j]+=eps;zm=Z0.copy();zm[j]-=eps
                            T[:,j*8+k]=(zjordan(pdec(zp,d),pdec(ek_v,d))-zjordan(pdec(zm,d),pdec(ek_v,d)))/(2*eps)
                    matrices.append(T)
        M = np.vstack(matrices)
        M = M[~np.any(np.isnan(M)|np.isinf(M), axis=1)]
        M = M[np.linalg.norm(M, axis=1) > 1e-15]
        OP[op] = M
    return OP

def rank_at_scale(n_qubits, OP_LOCAL, ops=None, verbose=True):
    """
    Bereken de rank van operaties in de n-qubit Hilbertruimte.
    
    Methode: voor n_groups Zorn-groepen, pas operaties toe op elk
    aangrenzend paar. De lokale transfer matrix T_local (r x 64) wordt
    uitgebreid naar de globale ruimte via tensor product met identiteit
    op de overige groepen.
    
    Globale meetvector: voor rij t van T_local op paar (g, g+1),
    en basisvector e_k van de overige groepen:
      v = I_left tensor t tensor I_right
    
    Dit is equivalent aan: v[left*64*right + pair_idx] = t[pair_idx]
    voor elke combinatie van left en right indices.
    """
    assert n_qubits % 3 == 0
    n_groups = n_qubits // 3
    dim = 8 ** n_groups
    
    if ops is None:
        ops = list(range(7))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  B1: {n_qubits} QUBITS ({n_groups} groepen, {dim}D)")
        print(f"{'='*60}")
    
    # Verzamel alle globale meetvectoren
    # Strategie: voor elk paar, neem de lokale T matrix en 
    # breid uit met tensorproduct identiteit.
    # 
    # De globale matrix is opgebouwd uit blokken:
    # Voor paar (g, g+1): T_local kron I_(other_dim)
    # waarbij other_dim = 8^(n_groups - 2)
    
    n_pairs = n_groups - 1
    other_dim = 8 ** (n_groups - 2) if n_groups > 2 else 1
    
    # Combineer lokale T matrices voor gevraagde operaties
    T_local_combined = np.vstack([OP_LOCAL[op] for op in ops])
    T_local_combined = T_local_combined[np.linalg.norm(T_local_combined, axis=1) > 1e-15]
    local_rank = np.linalg.matrix_rank(T_local_combined, tol=1e-10)
    
    if verbose:
        print(f"  Lokale rank (6q basis): {local_rank}/64")
        print(f"  Paren: {n_pairs}, overige dim: {other_dim}")
    
    if n_groups == 2:
        # Speciaal geval: precies 6 qubits, directe berekening
        r = np.linalg.matrix_rank(T_local_combined, tol=1e-10)
        if verbose:
            print(f"  Globale rank: {r}/{dim}")
        return r, dim
    
    # Voor grotere systemen: de globale rank is begrensd door
    # local_rank * other_dim per paar, maar paren overlappen (gedeelde groepen).
    # We moeten de DAADWERKELIJKE globale matrix bouwen.
    
    # Efficiënte aanpak: gebruik SVD op de lokale matrix om 
    # een compacte basis te vinden, dan breid uit.
    
    # Stap 1: lokale basis via SVD
    U, S, Vt = np.linalg.svd(T_local_combined, full_matrices=False)
    # Behoud significante singuliere waarden
    sig_mask = S > 1e-10
    local_basis = Vt[sig_mask]  # local_rank x 64 matrix
    
    if verbose:
        print(f"  Lokale basis: {local_basis.shape[0]} vectoren")
    
    # Stap 2: voor elk paar, bouw globale basisvectoren
    # via Kronecker product met identiteit
    # 
    # Voor paar (g, g+1) in n_groups groepen:
    #   left_dim = 8^g
    #   pair_dim = 64
    #   right_dim = 8^(n_groups - g - 2)
    #
    # Globale vector: I_left kron local_basis_row kron I_right
    # Dit geeft left_dim * local_rank * right_dim vectoren per paar.
    
    # Maar we hoeven niet ALLE vectoren te materialiseren.
    # We bouwen de globale matrix slim met Kronecker structuur.
    
    # Als dim <= 4096: bouw expliciet
    # Als dim > 4096: gebruik steekproef + iteratieve rank schatting
    
    if dim <= 4096:
        # Expliciet bouwen
        all_global = []
        
        for pair_g in range(n_pairs):
            left_dim = 8 ** pair_g
            right_dim = 8 ** (n_groups - pair_g - 2)
            
            # Voor elke lokale basisvector en elke combinatie van
            # left/right indices
            for lb in local_basis:
                lb_mat = lb.reshape(8, 8)  # pair_dim = 8 x 8
                for li in range(left_dim):
                    for ri in range(right_dim):
                        gv = np.zeros(dim)
                        for pi in range(8):
                            for pj in range(8):
                                global_idx = (li * 64 * right_dim + 
                                            (pi * 8 + pj) * right_dim + ri)
                                gv[global_idx] = lb_mat[pi, pj]
                        all_global.append(gv)
        
        G = np.array(all_global)
        G = G[np.linalg.norm(G, axis=1) > 1e-15]
        r = np.linalg.matrix_rank(G, tol=1e-10)
        
    else:
        # Voor dim > 4096: gebruik random projectie methode
        # Project globale vectoren naar lagere dimensie en schat rank
        if verbose:
            print(f"  Dim {dim} > 4096: gebruik random projectie + sampling")
        
        # Random projectie matrix (dim -> target_dim)
        target_dim = min(dim, 8192)
        proj = rng.standard_normal((target_dim, dim)) / np.sqrt(target_dim)
        
        projected = []
        for pair_g in range(n_pairs):
            left_dim = 8 ** pair_g
            right_dim = 8 ** (n_groups - pair_g - 2)
            
            for lb in local_basis:
                lb_mat = lb.reshape(8, 8)
                # Sample een subset van left/right indices
                n_samples_lr = min(left_dim * right_dim, 200)
                if left_dim * right_dim <= n_samples_lr:
                    lr_indices = [(li, ri) for li in range(left_dim) for ri in range(right_dim)]
                else:
                    lr_indices = [(rng.integers(left_dim), rng.integers(right_dim)) 
                                 for _ in range(n_samples_lr)]
                
                for li, ri in lr_indices:
                    gv = np.zeros(dim)
                    for pi in range(8):
                        for pj in range(8):
                            global_idx = (li * 64 * right_dim + 
                                        (pi * 8 + pj) * right_dim + ri)
                            gv[global_idx] = lb_mat[pi, pj]
                    pv = proj @ gv
                    projected.append(pv)
        
        G = np.array(projected)
        G = G[np.linalg.norm(G, axis=1) > 1e-15]
        r = np.linalg.matrix_rank(G, tol=1e-10)
    
    if verbose:
        print(f"  Matrix: {G.shape}")
        print(f"  Globale rank: {r}/{dim}")
    
    return r, dim


# === MAIN ===
print("B1: SCHAALBAARHEID 7-OPERATIE INFORMATIECOMPLEETHEID")
print("=" * 60)

t0 = time.time()
print("\nStap 1: Bouw lokale transfer matrices (6q basis)...")
OP_LOCAL = build_local_transfer_matrices()

# Individuele ranks
print("\n  Lokale individuele ranks:")
for op in range(7):
    r = np.linalg.matrix_rank(OP_LOCAL[op], tol=1e-10)
    print(f"    {SHORT[op]:5s}: {r}/64")

# Gecombineerde lokale rank
M_all = np.vstack([OP_LOCAL[op] for op in range(7)])
M_all = M_all[np.linalg.norm(M_all, axis=1) > 1e-15]
r_all = np.linalg.matrix_rank(M_all, tol=1e-10)
print(f"    ALL:  {r_all}/64")

print(f"\n  ({time.time()-t0:.1f}s)")

# Test op verschillende schalen
for nq in [6, 9, 12]:
    t1 = time.time()
    r, dim = rank_at_scale(nq, OP_LOCAL, ops=list(range(7)))
    dt = time.time() - t1
    status = "COMPLEET" if r == dim else f"INCOMPLEET (mist {dim-r})"
    print(f"  Status: {status}")
    print(f"  Tijd: {dt:.1f}s")

# Individuele operaties bij 9q en 12q
for nq in [9, 12]:
    print(f"\n  Individuele operaties bij {nq}q:")
    dim = 8 ** (nq // 3)
    for op in range(7):
        r, _ = rank_at_scale(nq, OP_LOCAL, ops=[op], verbose=False)
        print(f"    {SHORT[op]:5s}: {r}/{dim} ({100*r/dim:.1f}%)")
    # Alle samen
    r, _ = rank_at_scale(nq, OP_LOCAL, ops=list(range(7)), verbose=False)
    print(f"    ALL:  {r}/{dim} ({100*r/dim:.1f}%)")

print(f"\nTotale tijd: {time.time()-t0:.1f}s")
