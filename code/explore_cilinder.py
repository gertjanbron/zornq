#!/usr/bin/env python3
"""
ZornQ Explorer: Maximale cilinder op GTX 1650
==============================================
Schaalt Lx op bij vaste Ly=4 (d=16) — de sweet spot voor je GPU.
Doel: hoe ver komen we in qubits bij chi=32 op je hardware?

Gebruik:
    python explore_cilinder.py           # GPU
    python explore_cilinder.py --cpu     # CPU vergelijking
"""
import sys
import time

cpu_mode = '--cpu' in sys.argv

if cpu_mode:
    try:
        import gpu_backend, numpy as np
        gpu_backend.GPU_AVAILABLE = False
        gpu_backend.xp = np
    except ImportError:
        pass

from zorn_mps import HeisenbergQAOA

try:
    from gpu_backend import GPU_AVAILABLE, GPU_NAME, gpu_memory_info
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "none"
    gpu_memory_info = lambda: None

USE_GPU = GPU_AVAILABLE and not cpu_mode


def run(Lx, Ly, chi, min_weight=None):
    n_q = Lx * Ly
    beta = 1.1778
    n_e = Lx * (Ly - 1) + (Lx - 1) * Ly
    avg_d = 2 * n_e / n_q
    gamma = 0.88 / avg_d

    tag = f"eps={min_weight:.0e}" if min_weight else "vast"
    label = f"{Lx}x{Ly}={n_q:>3d}q  chi={chi:>3d}  {tag:>10s}"
    print(f"  {label}  ", end="", flush=True)

    try:
        t0 = time.time()
        eng = HeisenbergQAOA(Lx=Lx, Ly=Ly, max_chi=chi,
                             gpu=USE_GPU, min_weight=min_weight)
        ratio = eng.eval_ratio(1, [gamma], [beta])
        dt = time.time() - t0
        print(f"ratio={ratio:.5f}  {dt:9.1f}s")
        return ratio, dt
    except MemoryError:
        print("OUT OF MEMORY")
        return None, None
    except Exception as e:
        msg = str(e)[:60]
        print(f"FOUT: {msg}")
        return None, None


if __name__ == '__main__':
    print("=" * 70)
    print("  ZornQ CILINDER EXPLORER: Lx opschalen bij Ly=4 (d=16)")
    print("=" * 70)

    if USE_GPU:
        print(f"  GPU:  {GPU_NAME}")
        mem = gpu_memory_info()
        if mem:
            print(f"  VRAM: {mem[0]:.0f} / {mem[1]:.0f} MB")
    else:
        print(f"  Mode: CPU (numpy)")
    print()

    # JIT warmup
    if USE_GPU:
        print("--- JIT Warmup ---")
        run(3, 2, 8)
        print()

    # ---------------------------------------------------------------
    # Blok 1: Ly=4 cilinders, chi=32 — de hoofdlijn
    # ---------------------------------------------------------------
    print("--- Cilinder Ly=4, chi=32 (GPU sweet spot) ---")
    results = []
    for Lx in [5, 8, 10, 15, 20, 30, 50]:
        r, dt = run(Lx, 4, 32)
        if dt is not None:
            results.append((Lx, 4, 32, r, dt))
        if dt is not None and dt > 120:
            print("    (>2min, stoppen met opschalen)")
            break
    print()

    # ---------------------------------------------------------------
    # Blok 2: Ly=4, chi=16 — sneller, lichtere verstrengeling
    # ---------------------------------------------------------------
    print("--- Cilinder Ly=4, chi=16 (sneller, lichter) ---")
    for Lx in [20, 50, 100]:
        r, dt = run(Lx, 4, 16)
        if dt is not None and dt > 120:
            print("    (>2min, stoppen)")
            break
    print()

    # ---------------------------------------------------------------
    # Blok 3: Ly=3 cilinders — d=8, veel sneller
    # ---------------------------------------------------------------
    print("--- Cilinder Ly=3, chi=32 (d=8, snelste 2D) ---")
    for Lx in [20, 50, 100, 200]:
        r, dt = run(Lx, 3, 32)
        if dt is not None and dt > 120:
            print("    (>2min, stoppen)")
            break
    print()

    # ---------------------------------------------------------------
    # Blok 4: B15 adaptieve truncatie op de zwaarste cases
    # ---------------------------------------------------------------
    print("--- B15 Adaptief (eps=1e-3) op Ly=4 ---")
    for Lx in [10, 20, 30]:
        r1, dt1 = run(Lx, 4, 32, min_weight=None)
        r2, dt2 = run(Lx, 4, 32, min_weight=1e-3)
        if dt1 and dt2:
            speedup = dt1 / dt2 if dt2 > 0 else 0
            delta = abs(r1 - r2) if r1 and r2 else -1
            print(f"    -> B15 speedup: {speedup:.1f}x, delta={delta:.1e}")
        print()
        if dt1 and dt1 > 120:
            break

    # ---------------------------------------------------------------
    # Samenvatting
    # ---------------------------------------------------------------
    print("=" * 70)
    print("  SAMENVATTING")
    print("=" * 70)
    if results:
        print(f"  {'Systeem':>12s}  {'Qubits':>6s}  {'Ratio':>8s}  {'Tijd':>10s}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*10}")
        for Lx, Ly, chi, r, dt in results:
            print(f"  {Lx}x{Ly} chi={chi:>2d}  {Lx*Ly:>5d}q  {r:>8.5f}  {dt:>9.1f}s")
    print()
    print("  Sweet spot: Ly=4 (d=16) met chi=32 op GPU")
    print("  Ly=3 (d=8) voor snelle exploratie van lange ketens")
    print("  B15 eps=1e-3 voor CPU-modus of parameter-sweeps")
    print("=" * 70)
