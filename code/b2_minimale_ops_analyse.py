import numpy as np
from itertools import combinations
import sys, time

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

# Pre-build ALL transfer matrices once
rng = np.random.default_rng(42)
OP_MATRICES = {}

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
        elif op == 3:  # inv(A)xB
            for _ in range(3):
                Z0 = rng.standard_normal(8); eps=1e-7
                T = np.zeros((8, 64))
                for j in range(8):
                    for k in range(8):
                        ek=np.zeros(8);ek[k]=1
                        zp=Z0.copy();zp[j]+=eps; zm=Z0.copy();zm[j]-=eps
                        T[:,j*8+k]=(zmul(zinv(pdec(zp,d)),pdec(ek,d))-zmul(zinv(pdec(zm,d)),pdec(ek,d)))/(2*eps)
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
                        ek=np.zeros(8);ek[k]=1
                        zp=Z0.copy();zp[j]+=eps;zm=Z0.copy();zm[j]-=eps
                        T[:,j*8+k]=(zjordan(pdec(zp,d),pdec(ek,d))-zjordan(pdec(zm,d),pdec(ek,d)))/(2*eps)
                matrices.append(T)
    OP_MATRICES[op] = np.vstack(matrices)

def rank_for_ops(ops):
    M = np.vstack([OP_MATRICES[op] for op in ops])
    M = M[~np.any(np.isnan(M)|np.isinf(M), axis=1)]
    M = M[np.linalg.norm(M, axis=1) > 1e-15]
    return np.linalg.matrix_rank(M, tol=1e-10) if M.shape[0]>0 else 0

print("="*60)
print("  B2: MINIMALE OPERATIESET (6q = 64D)")
print("="*60)

# Individueel
print("\n  INDIVIDUEEL:")
for op in range(7):
    r = rank_for_ops([op])
    print(f"  {SHORT[op]:5s}  rank={r:3d}/64  ({100*r/64:.1f}%)")

# Paren
print("\n  PAREN:")
full2 = []
for combo in combinations(range(7), 2):
    r = rank_for_ops(list(combo))
    mark = " *" if r==64 else ""
    if r==64: full2.append(combo)
    print(f"  {'+'.join(SHORT[c] for c in combo):12s}  rank={r:3d}/64{mark}")

# Triples
print("\n  TRIPLES:")
full3 = []
for combo in combinations(range(7), 3):
    r = rank_for_ops(list(combo))
    mark = " *" if r==64 else ""
    if r==64: full3.append(combo)
    print(f"  {'+'.join(SHORT[c] for c in combo):12s}  rank={r:3d}/64{mark}")

# Onmisbaarheid
print("\n  ONMISBAARHEID (6 van 7):")
for op in range(7):
    without = [x for x in range(7) if x != op]
    r = rank_for_ops(without)
    print(f"  Zonder {SHORT[op]:5s}  rank={r:3d}/64  {'ONMISBAAR' if r<64 else 'vervangbaar'}")

# Samenvatting
print(f"\n{'='*60}")
print("  SAMENVATTING")
print(f"{'='*60}")
print(f"\n  Volledige paren ({len(full2)}):")
for p in full2:
    print(f"    {{{', '.join(SHORT[c] for c in p)}}}")

not_full_triples = [c for c in combinations(range(7),3) if rank_for_ops(list(c))<64]
print(f"\n  Triples NIET volledig ({len(not_full_triples)}):")
for c in not_full_triples:
    r = rank_for_ops(list(c))
    print(f"    {'+'.join(SHORT[x] for x in c):15s}  rank={r}/64  (mist {64-r})")

full3_no_pair = [c for c in full3 if not any(set(p).issubset(set(c)) for p in full2)]
print(f"\n  Triples volledig ZONDER volledig paar-subset ({len(full3_no_pair)}):")
for c in full3_no_pair:
    print(f"    {{{', '.join(SHORT[x] for x in c)}}}")
