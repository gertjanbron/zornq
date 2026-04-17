#!/usr/bin/env python3
"""
lightcone_qaoa.py - B21: Lightcone Graph-Stitching voor QAOA MaxCut.

Idee: knip de MaxCut-graaf in lokale sub-circuits per edge, gebaseerd
op de lichtkegel bij diepte p. Simuleer elk sub-circuit exact via
state vector. Stik verwachtingswaarden klassiek aan elkaar.

Bij p=1 op een Lx x Ly grid:
  - Lichtkegel per edge: ~(2p+2) kolommen x Ly qubits
  - Bij Ly=4, p=1: 4 kolommen x 4 qubits = 16 qubits per edge
  - State vector: 2^16 = 65536 amplitudes -> microseconden
  - 8x8 grid: 112 edges x ~1ms = 112ms TOTAAL

Geen chi-muur. Geen truncatie. Wiskundig exact.

B35 Hybride modus: wanneer de lichtkegel > SV_THRESHOLD qubits is,
val terug op een mini-MPS (HeisenbergQAOA) per sub-circuit.
Combineert lightcone-decompositie met MPS-compressie.

Gebruik:
  python lightcone_qaoa.py                          # 8x8 p=1
  python lightcone_qaoa.py --Lx 8 --Ly 8 --p 1     # 8x8 p=1
  python lightcone_qaoa.py --Lx 20 --Ly 4 --p 2 --chi 32  # hybride!
  python lightcone_qaoa.py --Lx 100 --Ly 4 --p 1   # 100x4 = 400 qubits!
"""

import numpy as np
import math
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Threshold: boven dit aantal qubits valt state vector af
# Configureerbaar via constructor sv_threshold_gpu/cpu.
# Defaults: GTX 1650 (4GB) comfortabel bij 26q GPU / 22q CPU.
# Bij upgrade naar bijv. RTX 4070 Ti (12GB): 28q GPU haalbaar.
# Bij RTX 4090 (24GB): 30q GPU.
SV_THRESHOLD_GPU_DEFAULT = 26
SV_THRESHOLD_CPU_DEFAULT = 22


class LightconeQAOA:
    """QAOA MaxCut via lightcone decomposition.

    Voor elke edge berekenen we <ZZ> exact via state vector,
    maar alleen op de qubits binnen de lichtkegel van die edge.
    """

    def __init__(self, Lx, Ly=4, verbose=True, chi=None, gpu=False, fp32=False,
                 sv_threshold_gpu=None, sv_threshold_cpu=None):
        self.Lx = Lx
        self.Ly = Ly
        self.verbose = verbose
        self.chi = chi  # MPS chi voor hybride modus
        self.gpu = gpu
        self.fp32 = fp32  # single precision (2x minder VRAM, sneller op GPU)
        self._mps_cache = {}  # cache mini-MPS instanties per n_cols

        # Configureerbare SV thresholds (voor GPU upgrade)
        self.sv_threshold_gpu = sv_threshold_gpu or SV_THRESHOLD_GPU_DEFAULT
        self.sv_threshold_cpu = sv_threshold_cpu or SV_THRESHOLD_CPU_DEFAULT

        # B65: Pre-allocated GPU/CPU buffers voor zero-allocation eval_edge_exact
        # Worden lazy geinitialiseerd bij eerste gebruik (dan kennen we max dim)
        self._buf_max_dim = 0
        self._buf_state = None    # complex, dim entries
        self._buf_hphase = None   # float, dim entries
        self._buf_z = None        # float, dim entries (herbruikbaar scratch)

        # Bouw edge-lijst
        self.edges = []
        # Horizontale edges (inter-column): (x,y)-(x+1,y)
        for x in range(Lx - 1):
            for y in range(Ly):
                self.edges.append(('h', x, y))
        # Verticale edges (intra-column): (x,y)-(x,y+1)
        for x in range(Lx):
            for y in range(Ly - 1):
                self.edges.append(('v', x, y))
        self.n_edges = len(self.edges)

    def lightcone_columns(self, edge_type, edge_x, p):
        """Bepaal welke kolommen binnen de lichtkegel van een edge liggen.

        Bij p lagen QAOA:
          - ZZ-gates raken kolommen x en x+1 (inter) of alleen x (intra)
          - Mixer raakt alle qubits in een kolom
          - Per laag groeit de lichtkegel 1 kolom in elke richting

        Returns: (col_min, col_max) inclusief
        """
        if edge_type == 'h':
            # Horizontale edge: raakt kolom edge_x en edge_x+1
            center_min, center_max = edge_x, edge_x + 1
        else:
            # Verticale edge: raakt alleen kolom edge_x
            center_min, center_max = edge_x, edge_x

        # Lichtkegel groeit met p in elke richting
        col_min = max(0, center_min - p)
        col_max = min(self.Lx - 1, center_max + p)
        return col_min, col_max

    def lightcone_diamond(self, edge_type, edge_x, edge_y, p):
        """BFS-diamant lichtkegel: exacte set (x,y) posities.

        In plaats van hele kolommen pakt dit alleen de grid-posities die
        daadwerkelijk bereikbaar zijn via p stappen Manhattan-afstand
        vanaf de edge-endpoints. Filtert "dode qubits" in de hoeken eruit.

        Bij p=1, Ly=4, horizontale edge:
          Kolom-methode: 4 kolommen × 4 rijen = 16 qubits
          BFS-diamant:   8 qubits (alleen directe buren)
          Besparing:     8 qubits = 256× minder VRAM!

        Returns: (positions, edges_in_cone)
          positions: list van (x,y) tuples, gesorteerd
          edges_in_cone: list van ((x1,y1),(x2,y2)) edges binnen de cone
        """
        Lx, Ly = self.Lx, self.Ly

        # Start-nodes van de edge
        if edge_type == 'h':
            seeds = {(edge_x, edge_y), (edge_x + 1, edge_y)}
        else:
            seeds = {(edge_x, edge_y), (edge_x, edge_y + 1)}

        # BFS op grid
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(p):
            next_frontier = set()
            for (x, y) in frontier:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < Lx and 0 <= ny < Ly and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        next_frontier.add((nx, ny))
            frontier = next_frontier

        # Sorteer posities voor deterministische qubit-indexering
        positions = sorted(visited)

        # Vind alle grid-edges binnen de cone
        pos_set = set(positions)
        edges_in_cone = []
        for (x, y) in positions:
            # Horizontale edge naar rechts
            if (x + 1, y) in pos_set:
                edges_in_cone.append(((x, y), (x + 1, y)))
            # Verticale edge naar beneden
            if (x, y + 1) in pos_set:
                edges_in_cone.append(((x, y), (x, y + 1)))

        return positions, edges_in_cone

    def eval_edge_diamond(self, edge_type, edge_x, edge_y, p, gammas, betas):
        """Bereken <ZZ> exact via BFS-diamant lichtkegel.

        Zoals eval_edge_exact, maar met de minimale qubit-set (BFS-diamant)
        i.p.v. volledige kolommen. Gebruikt dezelfde B65 buffer-strategie.

        Besparing: bij p=1 Ly=4: 16→8 qubits (256× minder VRAM).
                   bij p=2 Ly=4: 24→16 qubits (256× minder VRAM).
                   bij p=4 Ly=4: 32→28 qubits (16× minder VRAM).
        """
        use_gpu = self.gpu and GPU_AVAILABLE
        xp = cp if use_gpu else np

        if self.fp32:
            fdtype = xp.float32
            cdtype = xp.complex64
        else:
            fdtype = xp.float64
            cdtype = xp.complex128

        # BFS-diamant
        positions, edges_in_cone = self.lightcone_diamond(edge_type, edge_x, edge_y, p)
        n_qubits = len(positions)
        dim = 2 ** n_qubits

        sv_limit = self.sv_threshold_gpu if use_gpu else self.sv_threshold_cpu
        if n_qubits > sv_limit:
            raise ValueError(
                "BFS-diamant %d qubits > %d, gebruik --gpu of hogere threshold"
                % (n_qubits, sv_limit))

        # Qubit index mapping: (x,y) → compact index
        qi = {pos: idx for idx, pos in enumerate(positions)}

        # B65: Lazy buffer init / resize
        if dim > self._buf_max_dim:
            self._buf_max_dim = dim
            self._buf_state = xp.empty(dim, dtype=cdtype)
            self._buf_hphase = xp.empty(dim, dtype=fdtype)
            self._buf_z = xp.empty(dim, dtype=fdtype)
            if self.verbose:
                bytes_per = 8 if self.fp32 else 16
                mb = dim * (bytes_per + 4 + 4) / 1e6
                print("    [BFS] Buffers: dim=%d (%d qubits, %.1f MB)" % (dim, n_qubits, mb))

        state = self._buf_state[:dim]
        H_phase = self._buf_hphase[:dim]

        bitstrings = xp.arange(dim)

        # Z-diagonalen per qubit
        z_cache = {}
        for pos in positions:
            idx = qi[pos]
            z_cache[pos] = (1 - 2 * ((bitstrings >> idx) & 1)).astype(fdtype)
        del bitstrings

        # Fase-Hamiltoniaan: som van ZZ over alle edges in de cone
        H_phase[:] = 0
        for (pos_a, pos_b) in edges_in_cone:
            H_phase += z_cache[pos_a] * z_cache[pos_b]

        # Target edge observatie
        if edge_type == 'h':
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x + 1, edge_y)]
        else:
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x, edge_y + 1)]
        del z_cache

        # Rx mixer
        def apply_rx_all(state, beta):
            cb = fdtype(math.cos(float(beta)))
            msb = cdtype(-1j * math.sin(float(beta)))

            if use_gpu and hasattr(xp, 'fuse'):
                @xp.fuse()
                def _rx_fused(s0, s1):
                    return cb * s0 + msb * s1, msb * s0 + cb * s1

                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    new0, new1 = _rx_fused(s[:, 0, :], s[:, 1, :])
                    s[:, 0, :] = new0
                    s[:, 1, :] = new1
                    state = s.reshape(-1)
            else:
                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    tmp = cb * s[:, 0, :] + msb * s[:, 1, :]
                    s[:, 1, :] = msb * s[:, 0, :] + cb * s[:, 1, :]
                    s[:, 0, :] = tmp
                    state = s.reshape(-1)
            return state

        # |+>^n
        state[:] = cdtype(1.0 / math.sqrt(dim))

        # QAOA circuit
        for layer in range(p):
            gamma_scaled = fdtype(gammas[layer])
            state *= xp.exp(cdtype(-1j) * gamma_scaled * H_phase)
            state = apply_rx_all(state, betas[layer])

        # <ZZ>
        zz_obs = z_obs_a * z_obs_b
        probs = xp.abs(state)**2
        result = float(xp.dot(probs, zz_obs))

        del probs, zz_obs, z_obs_a, z_obs_b
        return result

    def eval_edge_exact(self, edge_type, edge_x, edge_y, p, gammas, betas):
        """Bereken <ZZ> voor een edge exact via state vector op de lichtkegel.

        Gebruikt GPU (CuPy) wanneer beschikbaar en self.gpu=True.
        Returns: float <ZZ> waarde
        """
        use_gpu = self.gpu and GPU_AVAILABLE
        xp = cp if use_gpu else np

        # Precision: fp32 halveert VRAM en benut fp32-cores op consumer GPU's
        if self.fp32:
            fdtype = xp.float32
            cdtype = xp.complex64
        else:
            fdtype = xp.float64
            cdtype = xp.complex128

        Ly = self.Ly
        col_min, col_max = self.lightcone_columns(edge_type, edge_x, p)
        n_cols = col_max - col_min + 1
        n_qubits = n_cols * Ly
        dim = 2 ** n_qubits

        sv_limit = self.sv_threshold_gpu if use_gpu else self.sv_threshold_cpu
        if n_qubits > sv_limit:
            raise ValueError(
                "Lichtkegel %d qubits > %d, gebruik --chi of --gpu"
                % (n_qubits, sv_limit))

        # B65: Lazy buffer init / resize (eenmalig per maximale dim)
        if dim > self._buf_max_dim:
            self._buf_max_dim = dim
            self._buf_state = xp.empty(dim, dtype=cdtype)
            self._buf_hphase = xp.empty(dim, dtype=fdtype)
            self._buf_z = xp.empty(dim, dtype=fdtype)
            if self.verbose:
                mb = dim * (8 if self.fp32 else 16 + 4 + 4) / 1e6
                print("    [B65] Buffers gealloceerd: dim=%d (%.1f MB)" % (dim, mb))

        # Views op pre-allocated buffers (geen allocatie!)
        state = self._buf_state[:dim]
        H_phase = self._buf_hphase[:dim]
        z_scratch = self._buf_z[:dim]

        bitstrings = xp.arange(dim)

        def qi(col, row):
            return (col - col_min) * Ly + row

        # Precache alle Z-diagonalen (vermijdt herberekening per laag)
        # Gebruik z_scratch als tijdelijke opslag; z_cache refs wijzen naar
        # aparte arrays (onvermijdelijk: n_qubits entries tegelijk nodig)
        z_cache = {}
        for x in range(col_min, col_max + 1):
            for y in range(Ly):
                idx = qi(x, y)
                z_cache[(x, y)] = (1 - 2 * ((bitstrings >> idx) & 1)).astype(fdtype)

        # bitstrings niet meer nodig na z_cache
        del bitstrings

        # Precache fase-Hamiltoniaan (gamma-onafhankelijk)
        # Schrijf direct in pre-allocated H_phase buffer
        H_phase[:] = 0
        for x in range(col_min, col_max + 1):
            for y in range(Ly - 1):
                H_phase += z_cache[(x, y)] * z_cache[(x, y + 1)]
        for x in range(col_min, col_max):
            for y in range(Ly):
                H_phase += z_cache[(x, y)] * z_cache[(x + 1, y)]

        # Bewaar alleen de 2 z_cache entries voor de target-edge observatie
        # Dit geeft ~1.4 GB VRAM vrij (22 van 24 entries x 64MB elk)
        if edge_type == 'h':
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x + 1, edge_y)]
        else:
            z_obs_a = z_cache[(edge_x, edge_y)]
            z_obs_b = z_cache[(edge_x, edge_y + 1)]
        del z_cache

        def apply_rx_all(state, beta):
            """Rx(beta) op alle qubits. Twee paden:
            - CuPy/GPU: fused kernel per qubit (1 launch i.p.v. ~7)
            - NumPy/CPU: nocopy variant (1 temp i.p.v. 2 copies)
            """
            cb = fdtype(math.cos(float(beta)))
            msb = cdtype(-1j * math.sin(float(beta)))

            if use_gpu and hasattr(xp, 'fuse'):
                # CuPy fuse: merge multiply+add into single kernel per qubit
                @xp.fuse()
                def _rx_fused(s0, s1):
                    return cb * s0 + msb * s1, msb * s0 + cb * s1

                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    new0, new1 = _rx_fused(s[:, 0, :], s[:, 1, :])
                    s[:, 0, :] = new0
                    s[:, 1, :] = new1
                    state = s.reshape(-1)
            else:
                # CPU / fallback: nocopy (1 temp allocation per qubit)
                for q in range(n_qubits):
                    s = state.reshape(2**(n_qubits-q-1), 2, 2**q)
                    tmp = cb * s[:, 0, :] + msb * s[:, 1, :]
                    s[:, 1, :] = msb * s[:, 0, :] + cb * s[:, 1, :]
                    s[:, 0, :] = tmp
                    state = s.reshape(-1)

            return state

        # Start: |+>^n (schrijf in pre-allocated buffer)
        state[:] = cdtype(1.0 / math.sqrt(dim))

        # QAOA circuit: p lagen
        for layer in range(p):
            gamma_scaled = fdtype(gammas[layer])
            state *= xp.exp(cdtype(-1j) * gamma_scaled * H_phase)
            state = apply_rx_all(state, betas[layer])

        # Meet <ZZ> op de target edge
        zz_obs = z_obs_a * z_obs_b

        probs = xp.abs(state)**2
        result = float(xp.dot(probs, zz_obs))

        # B65: state en H_phase zijn views op persistente buffers,
        # dus geen del nodig. Alleen tijdelijke refs opruimen.
        del probs, zz_obs, z_obs_a, z_obs_b

        return result

    def eval_edge_mps(self, edge_type, edge_x, edge_y, p, gammas, betas):
        """Bereken <ZZ> via mini-MPS op de lichtkegel (B35 hybride).

        Maakt een HeisenbergQAOA instantie aan voor alleen de lichtkegel-
        kolommen en evalueert de edge via Heisenberg-MPO met chi-truncatie.
        """
        from zorn_mps import HeisenbergQAOA as HQAOA

        Ly = self.Ly
        col_min, col_max = self.lightcone_columns(edge_type, edge_x, p)
        n_cols = col_max - col_min + 1

        # Hergebruik mini-MPS instantie als dezelfde grootte
        chi = self.chi if self.chi else 32
        cache_key = (n_cols, chi)
        if cache_key not in self._mps_cache:
            self._mps_cache[cache_key] = HQAOA(
                Lx=n_cols, Ly=Ly, max_chi=chi, gpu=self.gpu)
        mini = self._mps_cache[cache_key]

        # Remap edge-coordinaten naar mini-grid
        local_x = edge_x - col_min
        if edge_type == 'h':
            zz = mini.eval_edge(local_x, edge_y, local_x + 1, edge_y,
                                p, gammas, betas)
        else:
            zz = mini.eval_edge(local_x, edge_y, local_x, edge_y + 1,
                                p, gammas, betas)
        return zz

    def eval_edge(self, edge_type, edge_x, edge_y, p, gammas, betas):
        """Kies automatisch: BFS diamond, kolom state vector, of MPS.

        Prioriteit:
        1. BFS diamond state vector (minimale qubits, exact)
        2. Kolom state vector (als diamond niet veel helpt)
        3. Column-grouped MPS via HeisenbergQAOA (als chi opgegeven)
        """
        use_gpu = self.gpu and GPU_AVAILABLE
        sv_limit = self.sv_threshold_gpu if use_gpu else self.sv_threshold_cpu

        # Bereken kolom-qubits
        col_min, col_max = self.lightcone_columns(edge_type, edge_x, p)
        n_qubits_col = (col_max - col_min + 1) * self.Ly

        # Bereken BFS-diamant qubits
        positions, _ = self.lightcone_diamond(edge_type, edge_x, edge_y, p)
        n_qubits_bfs = len(positions)

        # Kies de methode met minste qubits die past
        if n_qubits_bfs <= sv_limit:
            # BFS-diamant past: gebruik het (altijd <= kolom-methode)
            return self.eval_edge_diamond(edge_type, edge_x, edge_y,
                                          p, gammas, betas)
        elif n_qubits_col <= sv_limit:
            # Kolom-methode past (theoretisch onmogelijk als BFS niet past,
            # maar voor robuustheid)
            return self.eval_edge_exact(edge_type, edge_x, edge_y,
                                        p, gammas, betas)
        elif self.chi is not None:
            return self.eval_edge_mps(edge_type, edge_x, edge_y,
                                      p, gammas, betas)
        else:
            raise ValueError(
                "BFS-diamant %d qubits > %d (kolom: %d). "
                "Gebruik --gpu (tot %d) of --chi N"
                % (n_qubits_bfs, sv_limit, n_qubits_col,
                   self.sv_threshold_gpu))

    def eval_cost(self, p, gammas, betas):
        """Bereken volledige MaxCut cost via lightcone per edge."""
        total = 0.0
        t0 = time.time()

        # B65: Initieel mempool cleanup (eenmalig per eval_cost, niet per edge)
        if self.gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        # Groepeer edges per lichtkegel (deduplicatie voor translatie-invariantie)
        # Bij uniforme parameters zijn bulk-edges identiek!
        cache = {}
        n_cached = 0
        n_computed = 0

        # B66: Symmetrie-caching voor cilinder-grids
        # Op een open-boundary grid is y spiegelsymmetrisch: y <-> Ly-1-y
        # Horizontale edges: y=0 en y=Ly-1 geven identieke <ZZ>
        # Verticale edges: v-edge bij (x,y) verbindt y↔y+1;
        #   v-edge bij (x, Ly-2-y) verbindt (Ly-2-y)↔(Ly-1-y)
        #   Door spiegeling is dit symmetrisch equivalent
        Ly = self.Ly

        for idx, (etype, ex, ey) in enumerate(self.edges):
            col_min, col_max = self.lightcone_columns(etype, ex, p)

            # B66: map ey naar symmetrische equivalent
            if etype == 'h':
                ey_sym = min(ey, Ly - 1 - ey)
            else:  # 'v': edge verbindt ey↔ey+1
                ey_sym = min(ey, Ly - 2 - ey)

            # Cache key: edge-type, afstand tot linkerrand, afstand tot rechterrand,
            # en de y-positie (symmetrie-gereduceerd)
            dist_left = ex - col_min
            dist_right = col_max - (ex + (1 if etype == 'h' else 0))
            n_cols = col_max - col_min + 1

            # Bulk edge: niet geraakt door randen
            is_bulk = (col_min > 0 and col_max < self.Lx - 1)
            if is_bulk:
                cache_key = (etype, ey_sym, n_cols, 'bulk')
            else:
                cache_key = (etype, ey_sym, dist_left, dist_right)

            if cache_key in cache:
                zz = cache[cache_key]
                n_cached += 1
            else:
                zz = self.eval_edge(etype, ex, ey, p, gammas, betas)
                cache[cache_key] = zz
                n_computed += 1
                # B65: Geen free_all_blocks() meer per edge nodig —
                # state en H_phase zijn persistente buffers, z_cache entries
                # worden vrijgegeven door del z_cache in eval_edge_exact

            total += (1 - zz) / 2

            if self.verbose and (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print("    %d/%d edges (%.1fs, %d berekend, %d cached)" % (
                    idx + 1, self.n_edges, elapsed, n_computed, n_cached))

        elapsed = time.time() - t0
        if self.verbose:
            # Toon vergelijking kolom vs BFS
            mid_x = self.Lx // 2
            mid_y = self.Ly // 2
            col_min, col_max = self.lightcone_columns('h', mid_x, p)
            n_q_col = (col_max - col_min + 1) * self.Ly
            pos_bfs, _ = self.lightcone_diamond('h', mid_x, mid_y, p)
            n_q_bfs = len(pos_bfs)
            saved = n_q_col - n_q_bfs
            print("    Klaar: %d edges in %.3fs (%d uniek, %d cached)" % (
                self.n_edges, elapsed, n_computed, n_cached))
            print("    Lichtkegel: %dq BFS-diamant (was %dq kolom, %dq bespaard = %dx minder VRAM)" % (
                n_q_bfs, n_q_col, saved, 2**saved if saved > 0 else 1))

        return total

    def eval_ratio(self, p, gammas, betas):
        return self.eval_cost(p, gammas, betas) / self.n_edges

    def optimize(self, p=1, n_gamma=20, n_beta=20, refine=True):
        """Twee-fase optimalisatie: grid search + scipy verfijning.

        Fase 1: grove gridsearch over gamma/beta (uniforme params per laag)
        Fase 2: scipy L-BFGS-B verfijning met per-laag vrijheidsgraden

        Returns: (ratio, gammas_list, betas_list, info_dict)
        """
        old_verbose = self.verbose
        self.verbose = False
        t0 = time.time()

        # --- Fase 1: Grid search (uniforme params) ---
        gammas_grid = np.linspace(0.05, np.pi / 2, n_gamma)
        betas_grid = np.linspace(0.05, np.pi / 2, n_beta)

        best_ratio = 0
        best_g, best_b = 0.4, 1.18
        n_evals = 0

        for gi, g in enumerate(gammas_grid):
            for bi, b in enumerate(betas_grid):
                r = self.eval_ratio(p, [g] * p, [b] * p)
                n_evals += 1
                if r > best_ratio:
                    best_ratio = r
                    best_g, best_b = g, b
            if old_verbose and (gi + 1) % 5 == 0:
                elapsed = time.time() - t0
                print("    Grid: %d/%d gamma, best=%.6f (%.1fs)" % (
                    gi + 1, n_gamma, best_ratio, elapsed))

        grid_ratio = best_ratio
        grid_time = time.time() - t0
        if old_verbose:
            print("    Grid klaar: ratio=%.6f, gamma=%.4f, beta=%.4f (%.1fs, %d evals)" % (
                grid_ratio, best_g, best_b, grid_time, n_evals))

        # --- Fase 2: Scipy verfijning ---
        best_gammas = [best_g] * p
        best_betas = [best_b] * p

        if refine:
            has_scipy = True
            try:
                from scipy.optimize import minimize
            except ImportError:
                has_scipy = False
                if old_verbose:
                    print("    (scipy niet beschikbaar, skip verfijning)")

            if has_scipy:
                def neg_ratio(params):
                    gammas = list(params[:p])
                    betas = list(params[p:])
                    return -self.eval_ratio(p, gammas, betas)

                x0 = np.array([best_g] * p + [best_b] * p)
                bounds = [(0.01, np.pi)] * p + [(0.01, np.pi)] * p

                t1 = time.time()
                result = minimize(neg_ratio, x0, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
                n_evals += result.nfev
                refined_ratio = -result.fun
                refined_gammas = list(result.x[:p])
                refined_betas = list(result.x[p:])
                refine_time = time.time() - t1

                if old_verbose:
                    improvement = refined_ratio - grid_ratio
                    print("    Scipy klaar: ratio=%.6f (+%.6f) (%.1fs, %d extra evals)" % (
                        refined_ratio, improvement, refine_time, result.nfev))

                if refined_ratio > grid_ratio:
                    best_ratio = refined_ratio
                    best_gammas = refined_gammas
                    best_betas = refined_betas

        total_time = time.time() - t0
        self.verbose = old_verbose

        total_time = time.time() - t0

        info = {}
        info['grid_ratio'] = grid_ratio
        info['grid_gamma'] = best_g
        info['grid_beta'] = best_b
        info['grid_time'] = grid_time
        info['total_time'] = total_time
        info['n_evals'] = n_evals

        return best_ratio, best_gammas, best_betas, info

    @staticmethod
    def warmstart_params(gammas, betas, method='fourier'):
        """Genereer startparameters voor p+1 vanuit optimale p-parameters.

        Methoden:
          'fourier': DCT-gebaseerde interpolatie (Zhou et al. 2020)
            Transformeer naar frequentiedomein, pad met nul, transformeer terug.
            QAOA-parameters vertonen Fourier-concentratie; de nieuwe hoge-
            frequentie component start op nul, wat wiskundig correcter is
            dan lineaire interpolatie.
          'interp': lineaire interpolatie (volgt adiabatische curve)
            p=1: [g1] -> [g1, g1]  (dupliceer)
            p>=2: [g1, g2] -> [g1, (g1+g2)/2, g2]
          'append': kopieer laatste waarde
            [g1, g2] -> [g1, g2, g2]
          'zero': voeg kleine waarde toe
            [g1, g2] -> [g1, g2, 0.01]

        Literatuur: Zhou et al. (2020) "QAOA parameter concentration"
        """
        p = len(gammas)
        if method == 'fourier':
            # DCT warm-start: frequentie-domein padding
            # Voor p=1->p=2: DCT niet zinvol (1 coeff), val terug op interp
            if p >= 2:
                from scipy.fft import dct, idct
                # Type-II DCT (standaard), ortho-normalisatie
                g_freq = dct(np.array(gammas, dtype=np.float64), type=2, norm='ortho')
                b_freq = dct(np.array(betas, dtype=np.float64), type=2, norm='ortho')
                # Pad met nul: nieuwe hoge-frequentie component = 0
                g_freq_new = np.zeros(p + 1)
                b_freq_new = np.zeros(p + 1)
                g_freq_new[:p] = g_freq
                b_freq_new[:p] = b_freq
                # Schaalfactor: iDCT van lengte p+1 vs p
                # De ortho DCT normaliseert met sqrt(2/N), dus bij N-verandering
                # moeten we herschalen: sqrt((p+1)/p)
                scale = np.sqrt((p + 1) / p)
                g_freq_new *= scale
                b_freq_new *= scale
                # Inverse DCT terug naar tijddomein
                new_gammas = list(idct(g_freq_new, type=2, norm='ortho'))
                new_betas = list(idct(b_freq_new, type=2, norm='ortho'))
            else:
                # p=1->p=2: dupliceer (DCT op 1 punt is triviaal)
                new_gammas = list(gammas) + [gammas[-1]]
                new_betas = list(betas) + [betas[-1]]
        elif method == 'interp':
            if p >= 2:
                old_x = np.linspace(0, 1, p)
                new_x = np.linspace(0, 1, p + 1)
                new_gammas = list(np.interp(new_x, old_x, gammas))
                new_betas = list(np.interp(new_x, old_x, betas))
            else:
                # p=1->p=2: dupliceer (behoud waarde, niet halveren!)
                new_gammas = list(gammas) + [gammas[-1]]
                new_betas = list(betas) + [betas[-1]]
        elif method == 'append':
            new_gammas = list(gammas) + [gammas[-1]]
            new_betas = list(betas) + [betas[-1]]
        else:  # 'zero'
            new_gammas = list(gammas) + [0.01]
            new_betas = list(betas) + [0.01]
        return new_gammas, new_betas

    def optimize_progressive(self, p_max, n_gamma=10, n_beta=10,
                             refine=True, method='fourier',
                             checkpoint_file=None):
        """Progressieve optimalisatie: p=1 -> p=2 -> ... -> p_max.

        Elke stap warm-start vanuit de vorige optimale parameters.
        Grid search alleen bij p=1; hogere p starten direct bij scipy.
        Met checkpoint_file worden resultaten na elke p-stap opgeslagen
        en bij herstart hervat vanaf het laatst voltooide p-niveau.

        Args:
            p_max: maximale circuitdiepte
            n_gamma, n_beta: gridgrootte voor p=1
            refine: gebruik scipy verfijning
            method: 'fourier', 'interp', of 'append' voor warm-starting
            checkpoint_file: pad naar JSON checkpoint (None = geen checkpoint)

        Returns: dict met resultaten per p-niveau
        """
        old_verbose = self.verbose
        results = {}

        # --- Checkpoint laden als beschikbaar ---
        resumed_from = 0
        if checkpoint_file:
            try:
                import json
                with open(checkpoint_file, 'r') as f:
                    saved = json.load(f)
                # Herstel voltooide p-niveaus
                for k, v in saved.get('results', {}).items():
                    results[int(k)] = v
                if results:
                    resumed_from = max(results.keys())
                    if old_verbose:
                        print("\n  Checkpoint geladen: p=1..%d voltooid" % resumed_from)
                        print("  Hervatten vanaf p=%d" % (resumed_from + 1))
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass  # Geen geldig checkpoint, begin opnieuw

        # --- p=1: volledige grid search + scipy ---
        if old_verbose:
            print("\n  === Progressive optimizer: p=1 -> p=%d ===" % p_max)
            print("  Warm-start methode: %s" % method)
            if checkpoint_file:
                print("  Checkpoint: %s" % checkpoint_file)
            print()

        if resumed_from >= 1:
            if old_verbose:
                r1 = results[1]
                print("  p=1: ratio=%.6f (uit checkpoint)\n" % r1['ratio'])
        else:
            self.verbose = old_verbose
            ratio, gammas, betas, info = self.optimize(
                p=1, n_gamma=n_gamma, n_beta=n_beta, refine=refine)

            results[1] = {
                'ratio': ratio, 'gammas': list(gammas),
                'betas': list(betas), 'info': info
            }
            if old_verbose:
                print("  p=1: ratio=%.6f (%.1fs)\n" % (ratio, info['total_time']))
            # Checkpoint opslaan na p=1
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, results, method)

        # --- p=2 t/m p_max: warm-start + scipy only ---
        for p in range(2, p_max + 1):
            # Skip als al in checkpoint
            if p <= resumed_from:
                if old_verbose:
                    rp = results[p]
                    print("  p=%d: ratio=%.6f (uit checkpoint)\n" % (p, rp['ratio']))
                continue

            t0 = time.time()
            if old_verbose:
                print("  p=%d: warm-start vanuit p=%d..." % (p, p - 1))

            # Warm-start parameters
            prev_g = results[p - 1]['gammas']
            prev_b = results[p - 1]['betas']
            init_g, init_b = self.warmstart_params(prev_g, prev_b, method)

            if old_verbose:
                g_str = ", ".join("%.4f" % v for v in init_g)
                b_str = ", ".join("%.4f" % v for v in init_b)
                print("    Init gammas: [%s]" % g_str)
                print("    Init betas:  [%s]" % b_str)

            # Evalueer startpunt
            self.verbose = False
            init_ratio = self.eval_ratio(p, init_g, init_b)
            n_evals = 1

            if old_verbose:
                print("    Init ratio: %.6f" % init_ratio)

            best_ratio = init_ratio
            best_gammas = list(init_g)
            best_betas = list(init_b)

            # --- Mini grid search rond warm-start punt ---
            # Perturbeer uniforme parameters over een klein grid
            # Dit vangt het geval dat de warm-start in een dal landt
            n_mini = 5  # 5x5 = 25 punten
            g_center = np.mean(init_g)
            b_center = np.mean(init_b)
            g_lo = max(0.05, g_center - 0.3)
            g_hi = min(np.pi, g_center + 0.3)
            b_lo = max(0.05, b_center - 0.3)
            b_hi = min(np.pi / 2, b_center + 0.3)
            g_range = np.linspace(g_lo, g_hi, n_mini)
            b_range = np.linspace(b_lo, b_hi, n_mini)

            for gi in g_range:
                for bi in b_range:
                    r = self.eval_ratio(p, [gi] * p, [bi] * p)
                    n_evals += 1
                    if r > best_ratio:
                        best_ratio = r
                        best_gammas = [gi] * p
                        best_betas = [bi] * p

            if old_verbose:
                print("    Mini grid (%dx%d): ratio=%.6f (+%.6f vs init)" % (
                    n_mini, n_mini, best_ratio, best_ratio - init_ratio))

            # --- Multi-restart scipy (5 startpunten) ---
            if refine:
                try:
                    from scipy.optimize import minimize

                    def neg_ratio(params):
                        gs = list(params[:p])
                        bs = list(params[p:])
                        return -self.eval_ratio(p, gs, bs)

                    bounds = [(0.01, np.pi)] * p + [(0.01, np.pi / 2)] * p

                    starts = [
                        np.array(init_g + init_b),
                        np.array(best_gammas + best_betas),
                    ]
                    rng = np.random.RandomState(42)
                    for _ in range(3):
                        perturb = rng.uniform(-0.15, 0.15, size=2 * p)
                        cand = np.array(best_gammas + best_betas) + perturb
                        cand[:p] = np.clip(cand[:p], 0.01, np.pi)
                        cand[p:] = np.clip(cand[p:], 0.01, np.pi / 2)
                        starts.append(cand)

                    scipy_best = best_ratio
                    scipy_best_x = np.array(best_gammas + best_betas)

                    for si, x0 in enumerate(starts):
                        result = minimize(neg_ratio, x0, method='L-BFGS-B',
                                          bounds=bounds,
                                          options={'maxiter': 100, 'ftol': 1e-8})
                        n_evals += result.nfev
                        refined = -result.fun
                        if refined > scipy_best:
                            scipy_best = refined
                            scipy_best_x = result.x.copy()
                            if old_verbose:
                                print("    Scipy restart %d: %.6f (nieuw best!)" % (
                                    si, refined))

                    if scipy_best > best_ratio:
                        best_ratio = scipy_best
                        best_gammas = list(scipy_best_x[:p])
                        best_betas = list(scipy_best_x[p:])

                except ImportError:
                    if old_verbose:
                        print("    scipy niet beschikbaar, skip verfijning")

        self.verbose = old_verbose
        elapsed = time.time() - t0
        if self.verbose:
            print("    Optimalisatie klaar: ratio=%.6f, %d evals, %.1fs" % (
                best_ratio, n_evals, elapsed))

        info = {
            'n_evals': n_evals,
            'wall_time': elapsed,
            'phase': 'grid+scipy' if refine else 'grid',
        }
        return best_ratio, best_gammas, best_betas, info


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lightcone QAOA MaxCut')
    parser.add_argument('--Lx', type=int, default=8, help='Grid breedte')
    parser.add_argument('--Ly', type=int, default=4, help='Grid hoogte')
    parser.add_argument('--p', type=int, default=1, help='QAOA diepte')
    parser.add_argument('--chi', type=int, default=None, help='MPS chi (hybride)')
    parser.add_argument('--gpu', action='store_true', help='Gebruik GPU')
    parser.add_argument('--fp32', action='store_true', help='Single precision')
    parser.add_argument('--sv-gpu', type=int, default=None,
                        help='SV threshold GPU qubits (default: 26)')
    parser.add_argument('--sv-cpu', type=int, default=None,
                        help='SV threshold CPU qubits (default: 22)')
    args = parser.parse_args()

    print("=" * 60)
    print("Lightcone QAOA MaxCut — %dx%d grid, p=%d" % (args.Lx, args.Ly, args.p))
    if args.chi:
        print("Hybride modus: chi=%d (MPS fallback)" % args.chi)
    if args.gpu:
        print("GPU modus: %s" % ("CuPy beschikbaar" if GPU_AVAILABLE else "NIET beschikbaar"))
    if args.fp32:
        print("Single precision (fp32)")
    print("=" * 60)

    lc = LightconeQAOA(args.Lx, args.Ly, verbose=True, chi=args.chi,
                        gpu=args.gpu, fp32=args.fp32,
                        sv_threshold_gpu=args.sv_gpu,
                        sv_threshold_cpu=args.sv_cpu)

    ratio, gammas, betas, info = lc.optimize(p=args.p)
    n_edges = lc.n_edges
    print("\nResultaat:")
    print("  MaxCut ratio: %.6f" % ratio)
    print("  Geschatte cut: %.1f / %d" % (ratio * n_edges, n_edges))
    print("  Optimale params: gamma=%s, beta=%s" % (
        [round(g, 4) for g in gammas], [round(b, 4) for b in betas]))
    print("  Evaluaties: %d, tijd: %.1fs" % (info['n_evals'], info['wall_time']))