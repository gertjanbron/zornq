"""
B10h deel 2: Informatieverlies op 2D rooster
Hier treedt echte truncatie op bij p>=2.
Test: uniform vs random vs gefrustreerd gewichten.
"""
import numpy as np
from numpy.linalg import svd
import time, sys

Ly_test = 2; Lx_test = 4  # 8 qubits, d=4

def bit_patterns(Ly):
    d=2**Ly
    return np.array([[(idx>>(Ly-1-q))&1 for q in range(Ly)] for idx in range(d)])

def Rx(t):
    c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-1j*s],[-1j*s,c]],dtype=complex)

bp = bit_patterns(Ly_test)
d = 2**Ly_test

def build_hadamard():
    H1=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
    Hd=np.ones((d,d),dtype=complex)
    for q in range(Ly_test): Hd*=H1[bp[:,q:q+1],bp[:,q:q+1].T]
    return Hd

def build_rx(beta):
    rx=Rx(2*beta); R=np.ones((d,d),dtype=complex)
    for q in range(Ly_test): R*=rx[bp[:,q:q+1],bp[:,q:q+1].T]
    return R

def build_zzi(gamma, vert_weights):
    """Intra-column ZZ with per-edge weights."""
    diags = []
    for x in range(Lx_test):
        dg = np.ones(d, dtype=complex)
        for y in range(Ly_test-1):
            w = vert_weights[x*(Ly_test-1) + y]
            dg *= np.exp(-1j * gamma * w * (1-2*bp[:,y].astype(float))*(1-2*bp[:,y+1].astype(float)))
        diags.append(dg)
    return diags

def build_zze(gamma, horiz_weights):
    """Inter-column ZZ with per-edge weights for each y."""
    diags = []
    for x in range(Lx_test-1):
        dg = np.ones(d*d, dtype=complex)
        iL=np.arange(d*d)//d; iR=np.arange(d*d)%d
        for y in range(Ly_test):
            w = horiz_weights[(x*Ly_test)+y]
            dg *= np.exp(-1j * gamma * w * (1-2*bp[iL,y].astype(float))*(1-2*bp[iR,y].astype(float)))
        diags.append(dg)
    return diags

Hd = build_hadamard()

def ap1(mpo,s,U):
    Ud=U.conj().T; W=np.einsum('ij,ajkb->aikb',Ud,mpo[s])
    mpo[s]=np.einsum('ajkb,kl->ajlb',W,U); return mpo

def ap1d(mpo,s,dg):
    cd=np.conj(dg)
    mpo[s]=mpo[s]*cd[None,:,None,None]*dg[None,None,:,None]; return mpo

def ap2d(mpo,s1,dd,chi_max):
    cl=mpo[s1].shape[0]; cr=mpo[s1+1].shape[3]
    Th=np.einsum('aijc,cklb->aijklb',mpo[s1],mpo[s1+1])
    cd=np.conj(dd).reshape(d,d); df=dd.reshape(d,d)
    Th=Th*cd[None,:,None,:,None,None]*df[None,None,:,None,:,None]
    mat=Th.reshape(cl*d*d,d*d*cr)
    U_s,S,V=svd(mat,full_matrices=False); Sa=np.abs(S)
    k=max(1,int(np.sum(Sa>1e-12*Sa[0]))) if Sa[0]>1e-15 else 1
    k=min(k,chi_max)
    trunc=float(np.sum(Sa[k:]**2)) if k<len(Sa) else 0.0
    mpo[s1]=U_s[:,:k].reshape(cl,d,d,k)
    mpo[s1+1]=(np.diag(S[:k])@V[:k,:]).reshape(k,d,d,cr)
    return mpo, trunc

def mpo_exp(mpo):
    L=np.ones((1,),dtype=complex)
    for W in mpo: L=np.einsum('a,ab->b',L,W[:,0,0,:])
    return L[0]

def max_chi(mpo):
    return max(W.shape[3] for W in mpo[:-1]) if len(mpo)>1 else 1

def get_edges():
    """Return (type, x1, y1, x2, y2) and indexing for weights."""
    vert = []; horiz = []
    for x in range(Lx_test):
        for y in range(Ly_test-1):
            vert.append((x, y, x, y+1))
    for x in range(Lx_test-1):
        for y in range(Ly_test):
            horiz.append((x, y, x+1, y))
    return vert, horiz

def make_obs(x1,y1,x2,y2):
    mpo=[np.eye(d,dtype=complex).reshape(1,d,d,1).copy() for _ in range(Lx_test)]
    if x1==x2:
        dg=(1-2*bp[:,y1].astype(float))*(1-2*bp[:,y2].astype(float))
        mpo[x1]=np.diag(dg.astype(complex)).reshape(1,d,d,1)
    else:
        for c,y in [(x1,y1),(x2,y2)]:
            dg=(1-2*bp[:,y].astype(float)).astype(complex)
            mpo[c]=np.diag(dg).reshape(1,d,d,1)
    return mpo

def eval_cost_2d(p, gammas, betas, vert_w, horiz_w, chi_max):
    """Full cost with weighted edges."""
    vert_edges, horiz_edges = get_edges()
    
    # Build gates
    gates = []
    for x in range(Lx_test): gates.append(('f', x, Hd))
    for l in range(p):
        zzi_list = build_zzi(gammas[l], vert_w)
        zze_list = build_zze(gammas[l], horiz_w)
        rxd = build_rx(betas[l])
        for x in range(Lx_test): gates.append(('d1', x, zzi_list[x]))
        for x in range(Lx_test-1): gates.append(('d2', x, zze_list[x]))
        for x in range(Lx_test): gates.append(('f', x, rxd))
    
    total = 0.0; total_trunc = 0.0
    # Vertical edges
    for idx, (x1,y1,x2,y2) in enumerate(vert_edges):
        w = vert_w[idx]
        mpo = make_obs(x1,y1,x2,y2)
        for gt,s,data in reversed(gates):
            if gt=='f': mpo=ap1(mpo,s,data)
            elif gt=='d1': mpo=ap1d(mpo,s,data)
            else: mpo,tr=ap2d(mpo,s,data,chi_max); total_trunc+=tr
        zz = mpo_exp(mpo).real
        total += w * (1-zz)/2
    # Horizontal edges
    for idx, (x1,y1,x2,y2) in enumerate(horiz_edges):
        w = horiz_w[idx]
        mpo = make_obs(x1,y1,x2,y2)
        for gt,s,data in reversed(gates):
            if gt=='f': mpo=ap1(mpo,s,data)
            elif gt=='d1': mpo=ap1d(mpo,s,data)
            else: mpo,tr=ap2d(mpo,s,data,chi_max); total_trunc+=tr
        zz = mpo_exp(mpo).real
        total += w * (1-zz)/2
    
    return total, total_trunc

np.random.seed(42)
vert_edges, horiz_edges = get_edges()
n_vert = len(vert_edges)
n_horiz = len(horiz_edges)
n_edges = n_vert + n_horiz
print(f"Rooster: {Lx_test}x{Ly_test} ({Lx_test*Ly_test}q), d={d}")
print(f"Edges: {n_vert} vert + {n_horiz} horiz = {n_edges}")

# === Problem types ===
problems = {
    "uniform": (np.ones(n_vert), np.ones(n_horiz)),
    "random_w": (np.random.uniform(0.1, 2.0, n_vert),
                 np.random.uniform(0.1, 2.0, n_horiz)),
    "frustrated": (np.array([1.0 if i%2==0 else -0.8 for i in range(n_vert)]),
                   np.array([1.0 if i%2==0 else -0.8 for i in range(n_horiz)])),
    "spin_glass": (np.random.choice([-1.0, 1.0], n_vert),
                   np.random.choice([-1.0, 1.0], n_horiz)),
}

gamma_2d = 0.34  # 2D optimaal
beta_fix = 1.1778

all_data = []

for name, (vw, hw) in problems.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    for p in [1, 2, 3]:
        gammas = np.full(p, gamma_2d)
        betas = np.full(p, beta_fix/p)
        
        # Exact (chi=1024)
        cost_ex, _ = eval_cost_2d(p, gammas, betas, vw, hw, 1024)
        
        print(f"  p={p}: exact={cost_ex:.6f}")
        for chi in [4, 8, 16, 32, 64]:
            cost_tr, trunc = eval_cost_2d(p, gammas, betas, vw, hw, chi)
            err = abs(cost_tr - cost_ex) / max(abs(cost_ex), 1e-10) * 100
            star = "*" if err < 1 else " "
            print(f"    chi={chi:3d}: cost={cost_tr:.6f} err={err:7.3f}%{star} "
                  f"split={trunc:.2e}")
            all_data.append((name, p, chi, trunc, err))
        sys.stdout.flush()

# === Correlatie analyse ===
print(f"\n{'='*60}")
print(f"  CORRELATIE: split-norm vs fysische fout")
print(f"{'='*60}")
print(f"  Is split-norm een betrouwbare kwaliteitsmeter?")
print(f"\n  {'type':<12} {'p':>2} {'chi':>4} {'split':>10} {'err%':>8} {'betrouwbaar':>12}")
print(f"  {'-'*52}")

# Group by problem: is error proportional to split-norm?
for name in problems:
    subset = [(p,c,s,e) for n,p,c,s,e in all_data if n==name and s>1e-15]
    for p,c,s,e in subset:
        reliable = "JA" if (s < 1e-10 and e < 0.01) or (s > 0.01 and e > 0.1) else "MATIG" if e < 5 else "NEE"
        print(f"  {name:<12} {p:>2} {c:>4} {s:>10.2e} {e:>7.3f}% {reliable:>12}")

print("\nDone.")
