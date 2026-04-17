#!/usr/bin/env python3
"""
reservoir_qaoa.py — Physical Reservoir Computing voor QAOA MaxCut.

Architectuur:
  1. ENCODING: QAOA-parameters (gamma, beta) → fysieke variabelen
     - gamma → zwaartekracht (kostenfunctie-sterkte)
     - beta  → startsnelheid (mixer-sterkte)
     - Graafstructuur → startposities deeltjes
  2. RESERVOIR: Chaotische deeltjessimulatie (de "Hilbert-ruimte")
     - Botsingen = niet-lineaire vermenging van input
     - Chaos = exponentiële gevoeligheid voor parameters
     - Geheugengebruik: O(N) ongeacht "verstrengeling"
  3. READOUT: Lineaire regressie op snapshot
     - Features: posities + snelheden na T stappen
     - Target: exacte QAOA-energie (uit state-vector)
     - Training: milliseconden

Pipeline:
  1. Genereer trainingsdata: (gamma, beta) → exact QAOA energy
  2. Voor elke (gamma, beta): draai chaos-sim, neem snapshot
  3. Train lineaire readout: snapshot → energy
  4. Test op onbekende (gamma, beta) waarden

Gebruik:
  python reservoir_qaoa.py                    # 8 qubits, standaard graaf
  python reservoir_qaoa.py --nodes 12 --train 200 --test 50
  python reservoir_qaoa.py --nodes 10 --particles 500 --frames 150
"""

import numpy as np
import time
import argparse
import os


# =====================================================================
# 1. EXACTE QAOA STATE-VECTOR (trainingsdata-generator)
# =====================================================================

def generate_random_graph(n_nodes: int, edge_prob: float = 0.5, seed: int = 42):
    """Genereer een random Erdos-Renyi graaf."""
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((i, j, 1))
    return edges


def generate_graph_from_particles(n_nodes: int, box_size: float = 60.0,
                                  radius: float = 3.0, steps: int = 200,
                                  seed: int = 42):
    """Genereer graaf via deeltjesbotsingen (hergebruik van particle_qaoa)."""
    from particle_qaoa import ParticleBox
    box = ParticleBox(n_nodes, box_size=box_size, radius=radius,
                      speed=3.0, seed=seed)
    box.simulate(steps, verbose=False)
    _, edges = box.to_graph(min_collisions=1)
    return edges


def exact_qaoa_energy(n_nodes: int, edges: list, gamma: float, beta: float):
    """Bereken exacte QAOA p=1 energie via state vector."""
    dim = 2 ** n_nodes
    bitstrings = np.arange(dim)

    # Z-diagonalen
    z_diag = {}
    for i in range(n_nodes):
        z_diag[i] = 1 - 2 * ((bitstrings >> i) & 1).astype(float)

    # ZZ-fase per edge
    zz_phase = np.zeros(dim)
    for i, j, w in edges:
        zz_phase += w * z_diag[i] * z_diag[j]

    # Cost diagonaal
    cost_diag = np.zeros(dim)
    for i, j, w in edges:
        cost_diag += w * (1 - z_diag[i] * z_diag[j]) / 2

    # Start: |+>^n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # ZZ-gate
    state *= np.exp(-1j * gamma * zz_phase)

    # X-mixer op alle qubits
    cb, sb = np.cos(beta), np.sin(beta)
    for q in range(n_nodes):
        mask = 1 << q
        partner = bitstrings ^ mask
        state = cb * state - 1j * sb * state[partner]

    # <C>
    return float(np.real(np.sum(np.abs(state)**2 * cost_diag)))


# =====================================================================
# 2. CHAOS RESERVOIR (deeltjessimulatie)
# =====================================================================

class ChaosReservoir:
    """Deeltjes-reservoir dat QAOA-parameters vertaalt naar chaos-dynamica.

    Encoding:
      - gamma → zwaartekracht (schaal: gamma * gravity_scale)
      - beta  → startsnelheid (schaal: beta * speed_scale)
      - Graafstructuur → startposities (nodes op cirkel, edges bepalen afstanden)
    """

    def __init__(self, n_nodes: int, edges: list, n_particles: int = 200,
                 box_size: float = 80.0, radius: float = 2.0,
                 n_frames: int = 100, gravity_scale: float = 0.3,
                 speed_scale: float = 4.0, base_seed: int = 0):
        self.n_nodes = n_nodes
        self.edges = edges
        self.n_particles = n_particles
        self.box = box_size
        self.radius = radius
        self.n_frames = n_frames
        self.gravity_scale = gravity_scale
        self.speed_scale = speed_scale
        self.base_seed = base_seed

        # Vaste startposities gebaseerd op graafstructuur
        # Nodes op een cirkel, extra deeltjes gevuld rondom edges
        self._init_base_positions()

    def _init_base_positions(self):
        """Bereken basis-posities uit graafstructuur."""
        n = self.n_nodes
        cx, cy = self.box / 2, self.box / 2
        r = self.box * 0.3

        # Nodes op cirkel
        self.node_pos = np.zeros((n, 2))
        for i in range(n):
            angle = 2 * np.pi * i / n
            self.node_pos[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]

        # Extra deeltjes: random rond het centrum
        rng = np.random.default_rng(self.base_seed)
        n_extra = self.n_particles - n
        if n_extra > 0:
            self.extra_pos = np.column_stack([
                cx + rng.normal(0, r * 0.5, n_extra),
                cy + rng.normal(0, r * 0.5, n_extra)
            ])
            # Clamp to box
            self.extra_pos = np.clip(self.extra_pos, self.radius, self.box - self.radius)
        else:
            self.extra_pos = np.zeros((0, 2))

    def run(self, gamma: float, beta: float):
        """Draai reservoir met gegeven QAOA-parameters. Return snapshot features."""
        n = self.n_particles
        pos = np.zeros((n, 2))
        vel = np.zeros((n, 2))

        # Posities: graaf-nodes + extra deeltjes
        pos[:self.n_nodes] = self.node_pos.copy()
        if self.n_particles > self.n_nodes:
            pos[self.n_nodes:] = self.extra_pos.copy()

        # Snelheden: beta bepaalt de magnitude
        speed = beta * self.speed_scale
        rng = np.random.default_rng(self.base_seed + hash((gamma, beta)) % (2**31))
        angles = rng.uniform(0, 2 * np.pi, n)
        speeds = speed * (0.5 + 0.5 * rng.uniform(size=n))
        vel[:, 0] = np.cos(angles) * speeds
        vel[:, 1] = np.sin(angles) * speeds

        # Zwaartekracht: gamma bepaalt de sterkte
        gravity = gamma * self.gravity_scale

        # Simuleer
        r = self.radius
        box = self.box
        for frame in range(self.n_frames):
            # Zwaartekracht
            vel[:, 1] += gravity * 0.1

            # Beweeg
            pos += vel

            # Muurbotsingen
            for dim in range(2):
                low = pos[:, dim] < r
                high = pos[:, dim] > box - r
                pos[low, dim] = r
                vel[low, dim] = np.abs(vel[low, dim])
                pos[high, dim] = box - r
                vel[high, dim] = -np.abs(vel[high, dim])

            # Deeltje-deeltje botsingen (grid-accelerated voor snelheid)
            self._collide(pos, vel, r)

        # Snapshot features: statistieken over posities en snelheden
        # Gebruik ALLEEN statistische features (niet individuele posities)
        # Dit voorkomt overfitting en maakt het reservoir-onafhankelijk van n_particles
        sp = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        features = np.array([
            # Positie-statistieken
            np.mean(pos[:, 0]) / box,
            np.mean(pos[:, 1]) / box,
            np.std(pos[:, 0]) / box,
            np.std(pos[:, 1]) / box,
            np.median(pos[:, 0]) / box,
            np.median(pos[:, 1]) / box,
            # Snelheid-statistieken
            np.mean(vel[:, 0]) / max(speed, 0.1),
            np.mean(vel[:, 1]) / max(speed, 0.1),
            np.std(vel[:, 0]) / max(speed, 0.1),
            np.std(vel[:, 1]) / max(speed, 0.1),
            np.mean(sp) / max(speed, 0.1),
            np.std(sp) / max(speed, 0.1),
            np.max(sp) / max(speed, 0.1),
            np.min(sp) / max(speed, 0.1),
            # Ruimtelijke correlaties
            np.corrcoef(pos[:, 0], pos[:, 1])[0, 1] if len(pos) > 2 else 0,
            np.corrcoef(vel[:, 0], vel[:, 1])[0, 1] if len(vel) > 2 else 0,
            # Energie-maat
            np.mean(sp**2),
            np.std(sp**2),
            # Verdeling: hoeveel in elke kwadrant
            np.sum((pos[:, 0] < box/2) & (pos[:, 1] < box/2)) / len(pos),
            np.sum((pos[:, 0] >= box/2) & (pos[:, 1] < box/2)) / len(pos),
            np.sum((pos[:, 0] < box/2) & (pos[:, 1] >= box/2)) / len(pos),
            np.sum((pos[:, 0] >= box/2) & (pos[:, 1] >= box/2)) / len(pos),
            # Pairwise afstand statistieken (sample)
            np.mean(np.sqrt(np.sum((pos[:20, None] - pos[None, :20])**2, axis=-1))),
            # Input echo: hoe goed "herinnert" het systeem gamma en beta
            gamma,  # directe echo
            beta,   # directe echo
        ])
        return features

    @staticmethod
    def _collide(pos, vel, radius):
        """Gevectoriseerde botsingsdetectie via numpy."""
        n = len(pos)
        if n < 2:
            return
        min_dist = 2 * radius
        min_dist_sq = min_dist * min_dist

        # Pairwise afstanden (vectorized)
        dx = pos[:, 0, None] - pos[None, :, 0]  # (n, n)
        dy = pos[:, 1, None] - pos[None, :, 1]
        dist_sq = dx**2 + dy**2

        # Vind botsende paren (boven-driehoek)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask &= (dist_sq < min_dist_sq) & (dist_sq > 1e-10)
        pairs = np.argwhere(mask)

        for idx in range(len(pairs)):
            i, j = pairs[idx]
            d = np.sqrt(dist_sq[i, j])
            nx_v = (pos[j, 0] - pos[i, 0]) / d
            ny_v = (pos[j, 1] - pos[i, 1]) / d
            dvn = (vel[i, 0] - vel[j, 0]) * nx_v + (vel[i, 1] - vel[j, 1]) * ny_v
            if dvn > 0:
                vel[i, 0] -= dvn * nx_v
                vel[i, 1] -= dvn * ny_v
                vel[j, 0] += dvn * nx_v
                vel[j, 1] += dvn * ny_v
                overlap = min_dist - d
                pos[i, 0] -= nx_v * overlap * 0.5
                pos[i, 1] -= ny_v * overlap * 0.5
                pos[j, 0] += nx_v * overlap * 0.5
                pos[j, 1] += ny_v * overlap * 0.5


# =====================================================================
# 3. READOUT (lineaire regressie)
# =====================================================================

class LinearReadout:
    """Ridge regressie readout voor reservoir computing."""

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0
        self.train_r2 = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train de readout. X = (n_samples, n_features), y = (n_samples,)."""
        n, d = X.shape
        # Ridge regression: w = (X^T X + alpha I)^{-1} X^T y
        # Met bias: voeg kolom enen toe
        X_bias = np.column_stack([X, np.ones(n)])
        A = X_bias.T @ X_bias + self.alpha * np.eye(d + 1)
        b = X_bias.T @ y
        params = np.linalg.solve(A, b)
        self.weights = params[:-1]
        self.bias = params[-1]

        # R² score
        y_pred = X_bias @ params
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.train_r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


# =====================================================================
# 4. PIPELINE
# =====================================================================

def run_reservoir_qaoa(n_nodes: int = 8, n_train: int = 150, n_test: int = 50,
                       n_particles: int = 100, n_frames: int = 100,
                       edge_prob: float = 0.5, graph_seed: int = 42,
                       verbose: bool = True):
    """Volledige Reservoir Computing QAOA pipeline."""

    print("=" * 60)
    print("  RESERVOIR COMPUTING QAOA — Chaos als Kwantumprocessor")
    print("=" * 60)

    # 1. Genereer graaf
    print(f"\n[1] Genereer graaf: {n_nodes} nodes, p_edge={edge_prob}")
    edges = generate_random_graph(n_nodes, edge_prob, seed=graph_seed)
    n_edges = len(edges)
    total_weight = sum(w for _, _, w in edges)
    print(f"    {n_edges} edges, totaal gewicht: {total_weight}")

    if n_edges == 0:
        print("    GEEN EDGES — verhoog edge_prob")
        return

    # 2. Genereer parameter-punten
    print(f"\n[2] Genereer {n_train + n_test} parameter-punten (gamma, beta)")
    rng = np.random.default_rng(123)
    all_gammas = rng.uniform(0.05, np.pi / 2, n_train + n_test)
    all_betas = rng.uniform(0.05, np.pi / 2, n_train + n_test)

    # 3. Bereken exacte QAOA-energieën (ground truth)
    print(f"\n[3] Bereken exacte QAOA-energieën (state vector, dim=2^{n_nodes}={2**n_nodes})...")
    t0 = time.time()
    all_energies = np.array([
        exact_qaoa_energy(n_nodes, edges, g, b)
        for g, b in zip(all_gammas, all_betas)
    ])
    t_exact = time.time() - t0
    print(f"    {n_train + n_test} evaluaties in {t_exact:.2f}s "
          f"({t_exact/(n_train+n_test)*1000:.1f}ms/eval)")
    print(f"    Energie range: [{all_energies.min():.3f}, {all_energies.max():.3f}]")

    # 4. Draai reservoir voor alle parameters
    print(f"\n[4] Draai chaos-reservoir: {n_particles} deeltjes, "
          f"{n_frames} frames per evaluatie...")
    reservoir = ChaosReservoir(
        n_nodes=n_nodes, edges=edges,
        n_particles=n_particles, n_frames=n_frames,
    )

    t0 = time.time()
    all_features = []
    for idx, (g, b) in enumerate(zip(all_gammas, all_betas)):
        features = reservoir.run(g, b)
        all_features.append(features)
        if verbose and (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {idx+1}/{n_train+n_test} reservoir-runs, {elapsed:.1f}s")
    t_reservoir = time.time() - t0
    all_features = np.array(all_features)
    print(f"    Feature matrix: {all_features.shape} "
          f"({t_reservoir:.2f}s, "
          f"{t_reservoir/(n_train+n_test)*1000:.1f}ms/run)")

    # 5. Train/test split
    X_train = all_features[:n_train]
    y_train = all_energies[:n_train]
    X_test = all_features[n_train:]
    y_test = all_energies[n_train:]

    # 6. Train readout
    print(f"\n[5] Train lineaire readout (ridge regressie)...")
    t0 = time.time()
    readout = LinearReadout(alpha=1.0)
    readout.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"    Training R²: {readout.train_r2:.4f} ({t_train*1000:.1f}ms)")

    # 7. Test
    print(f"\n[6] Test op {n_test} onbekende parameters...")
    y_pred = readout.predict(X_test)
    errors = np.abs(y_pred - y_test)
    relative_errors = errors / np.maximum(np.abs(y_test), 1e-10)

    # R²
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    test_r2 = 1 - ss_res / max(ss_tot, 1e-10)

    print(f"    Test R²:          {test_r2:.4f}")
    print(f"    MAE:              {np.mean(errors):.4f}")
    print(f"    Max error:        {np.max(errors):.4f}")
    print(f"    Gem. rel. error:  {np.mean(relative_errors):.2%}")

    # 8. Vind optimale parameters via reservoir
    print(f"\n[7] Zoek optimale (gamma, beta) via reservoir...")
    n_search = 400
    search_gammas = np.linspace(0.05, np.pi/2, 20)
    search_betas = np.linspace(0.05, np.pi/2, 20)

    best_pred_energy = -1
    best_g, best_b = 0, 0
    for g in search_gammas:
        for b in search_betas:
            feat = reservoir.run(g, b)
            pred = readout.predict(feat.reshape(1, -1))[0]
            if pred > best_pred_energy:
                best_pred_energy = pred
                best_g, best_b = g, b

    # Vergelijk met exact
    exact_at_best = exact_qaoa_energy(n_nodes, edges, best_g, best_b)
    print(f"    Reservoir optimum: gamma={best_g:.3f}, beta={best_b:.3f}")
    print(f"    Voorspelde energie: {best_pred_energy:.3f}")
    print(f"    Werkelijke energie: {exact_at_best:.3f}")
    print(f"    Ratio (gewogen):    {exact_at_best/total_weight:.4f}")

    # Vergelijk met exact grid search optimum
    best_exact = 0
    best_exact_g, best_exact_b = 0, 0
    for g in search_gammas:
        for b in search_betas:
            e = exact_qaoa_energy(n_nodes, edges, g, b)
            if e > best_exact:
                best_exact = e
                best_exact_g, best_exact_b = g, b

    print(f"\n    Exact optimum:     gamma={best_exact_g:.3f}, beta={best_exact_b:.3f}")
    print(f"    Exacte energie:    {best_exact:.3f}")
    print(f"    Exacte ratio:      {best_exact/total_weight:.4f}")
    print(f"    Reservoir/Exact:   {exact_at_best/max(best_exact, 1e-10):.4f}")

    # 9. Samenvatting
    print(f"\n{'=' * 60}")
    print(f"  SAMENVATTING")
    print(f"{'=' * 60}")
    print(f"  Graaf: {n_nodes} nodes, {n_edges} edges")
    print(f"  Reservoir: {n_particles} deeltjes, {n_frames} frames")
    print(f"  Training: {n_train} samples -> R2 = {readout.train_r2:.4f}")
    print(f"  Test:     {n_test} samples -> R2 = {test_r2:.4f}")
    quality_pct = exact_at_best/max(best_exact,1e-10)
    print("  Reservoir vindt " + str(round(quality_pct * 100, 1)) + "%% van exact optimum")
    print("")
    print("  Snelheid:")
    t_per_exact = t_exact/(n_train+n_test)*1000
    t_per_res = t_reservoir/(n_train+n_test)*1000
    print("    Exact state-vector: %.1fms/eval" % t_per_exact)
    print("    Reservoir + readout: %.1fms/eval" % t_per_res)
    print("=" * 60)

    return {
        'test_r2': test_r2,
        'train_r2': readout.train_r2,
        'mae': float(np.mean(errors)),
        'reservoir_ratio': exact_at_best / total_weight,
        'exact_ratio': best_exact / total_weight,
        'quality': exact_at_best / max(best_exact, 1e-10),
    }


def main():
    parser = argparse.ArgumentParser(description='Reservoir Computing QAOA')
    parser.add_argument('--nodes', '-n', type=int, default=8)
    parser.add_argument('--train', type=int, default=150)
    parser.add_argument('--test', type=int, default=50)
    parser.add_argument('--particles', type=int, default=100)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--edge-prob', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_reservoir_qaoa(
        n_nodes=args.nodes, n_train=args.train, n_test=args.test,
        n_particles=args.particles, n_frames=args.frames,
        edge_prob=args.edge_prob, graph_seed=args.seed,
    )


if __name__ == '__main__':
    main()
