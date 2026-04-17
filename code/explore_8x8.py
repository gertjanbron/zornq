#!/usr/bin/env python3
"""
ZornQ Explorer: Opschaling naar 8×8 = 64 qubits
=================================================
Schaalt stap voor stap op van 4×4 naar 8×8 om te zien
waar de grenzen liggen op jouw GTX 1650 (4GB VRAM).

Bij Ly=8 is d=256 per kolom — dat is zwaar.
We testen met oplopende chi en rapporteren ratio + tijd.

Gebruik:
    python explore_8x8.py           # GPU (aanbevolen)
    python explore_8x8.py --cpu     # CPU vergelijking
"""
import sys
import time

# CPU mode?
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


def run_test(Lx, Ly, chi, min_weight=None, timeout=300):
    """Draai QAOA en meet ratio + tijd. Geeft None bij timeout/error."""
    beta = 1.1778
    n_q = Lx * Ly

    if Ly > 1:
        n_e = Lx * (Ly - 1) + (Lx - 1) * Ly
        avg_d = 2 * n_e / n_q
        gamma = 0.88 / avg_d
    else:
        gamma = 0.4021

    d = 2**Ly
    label = f"{Lx}x{Ly}={n_q:>3d}q  d={d:>4d}  chi={chi:>3d}"
    if min_weight:
        label += f"  eps={min_weight:.0e}"

    print(f"  {label:>45s}  ", end="", flush=True)

    try:
        t0 = time.time()
        eng = HeisenbergQAOA(Lx=Lx, Ly=Ly, max_chi=chi,
                             gpu=USE_GPU, min_weight=min_weight)
        ratio = eng.eval_ratio(1, [gamma], [beta])
        dt = time.time() - t0

        if dt > timeout:
            print(f"TIMEOUT ({dt:.0f}s)")
            return None

        print(f"ratio={ratio:.5f}  {dt:8.1f}s")
        return (ratio, dt)

    except MemoryError:
        print("OUT OF MEMORY")
        return None
    except Exception as e:
        print(f"FOUT: {e}")
        return None


if __name__ == '__main__':
    print("=" * 70)
    print("  ZornQ EXPLORER: Opschaling naar 8x8 = 64 qubits")
    print("=" * 70)

    if USE_GPU:
        print(f"  GPU:  {GPU_NAME}")
        mem = gpu_memory_info()
        if mem:
            print(f"  VRAM: {mem[0]:.0f} / {mem[1]:.0f} MB")
        print(f"  Mode: GPU (cupy)")
    else:
        print(f"  Mode: CPU (numpy)")
    print()

    # ---------------------------------------------------------------
    # Stap 1: JIT warmup (GPU)
    # ---------------------------------------------------------------
    if USE_GPU:
        print("--- JIT Warmup ---")
        run_test(3, 2, 8)
        print()

    # ---------------------------------------------------------------
    # Stap 2: Bekende systemen (referentie)
    # ---------------------------------------------------------------
    print("--- Referentie (bekende systemen) ---")
    run_test(4, 3, 16)      # 12q, d=8   — snel
    run_test(5, 4, 32)      # 20q, d=16  — de 12s GPU case
    print()

    # ---------------------------------------------------------------
    # Stap 3: Opschalen Ly (diepte kolom = meer verstrengeling)
    # ---------------------------------------------------------------
    print("--- Opschalen Ly (kolom-diepte) ---")
    run_test(4, 4, 16)      # 16q, d=16
    run_test(4, 5, 16)      # 20q, d=32
    run_test(4, 5, 32)      # 20q, d=32, hogere chi
    run_test(4, 6, 16)      # 24q, d=64
    run_test(4, 6, 32)      # 24q, d=64, hogere chi
    print()

    # ---------------------------------------------------------------
    # Stap 4: Opschalen Lx (lengte keten)
    # ---------------------------------------------------------------
    print("--- Opschalen Lx (ketenlengte) bij Ly=4 ---")
    run_test(6, 4, 32)      # 24q
    run_test(8, 4, 32)      # 32q
    run_test(10, 4, 32)     # 40q
    print()

    # ---------------------------------------------------------------
    # Stap 5: Richting 8×8
    # ---------------------------------------------------------------
    print("--- Richting 8x8 ---")
    run_test(6, 6, 16)      # 36q, d=64
    run_test(6, 6, 32)      # 36q, d=64, hogere chi
    run_test(8, 6, 16)      # 48q, d=64
    run_test(8, 6, 32)      # 48q, d=64, hogere chi
    print()

    print("--- De grote test: 8x8 ---")
    run_test(8, 8, 4)       # 64q, d=256, minimale chi
    run_test(8, 8, 8)       # 64q, d=256, lage chi
    run_test(8, 8, 16)      # 64q, d=256, mid chi
    run_test(8, 8, 16, min_weight=1e-3)  # met B15
    run_test(8, 8, 32)      # 64q, d=256, hoge chi (zwaar!)
    run_test(8, 8, 32, min_weight=1e-3)  # met B15

    print()
    print("=" * 70)
    print("  NOTITIES:")
    print("  - d = 2^Ly = lokale dimensie per MPS-site (kolom)")
    print("  - Bij d=256 (Ly=8) is de 2-site tensor d^2 = 65536")
    print("  - SVD van (chi*d^2) x (d^2*chi) matrix is de bottleneck")
    print("  - GPU wint vooral bij chi >= 32, d >= 32")
    print("=" * 70)
