#!/usr/bin/env python3
"""
wave_qaoa.py - Golf-Reservoir Computing voor QAOA MaxCut.

Architectuur (RGBA-model):
  R-kanaal: huidige golfhoogte (amplitude, t0)
  G-kanaal: vorige golfhoogte (geheugen, t-1)
  B-kanaal: landschap/obstakels (MaxCut graaf)
  A-kanaal: bronnen (QAOA parameters gamma, beta)

De graaf wordt gecodeerd als een 2D-obstakelpatroon.
Golfbronnen op nodes pompen energie het systeem in.
Het interferentiepatroon na T stappen bevat informatie
over de graafstructuur - precies wat QAOA nodig heeft.

Cruciale verbetering t.o.v. deeltjes-reservoir:
  - Golven HEBBEN fase -> destructieve interferentie mogelijk
  - Golfvergelijking is lineair -> superpositie werkt
  - Laplaciaan = lokale operatie -> GPU-native
  - Fourier-spectra van het patroon relateren aan graph Laplacian eigenwaarden
"""

import numpy as np
import time
import argparse


# =====================================================================
# 1. EXACTE QAOA (hergebruik)
# =====================================================================

def generate_random_graph(n_nodes, edge_prob=0.5, seed=42):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((i, j, 1))
    return edges


def exact_qaoa_energy(n_nodes, edges, gamma, beta):
    dim = 2 ** n_nodes
    bitstrings = np.arange(dim)
    z_diag = {}
    for i in range(n_nodes):
        z_diag[i] = 1 - 2 * ((bitstrings >> i) & 1).astype(float)
    zz_phase = np.zeros(dim)
    for i, j, w in edges:
        zz_phase += w * z_diag[i] * z_diag[j]
    cost_diag = np.zeros(dim)
    for i, j, w in edges:
        cost_diag += w * (1 - z_diag[i] * z_diag[j]) / 2
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    state *= np.exp(-1j * gamma * zz_phase)
    cb, sb = np.cos(beta), np.sin(beta)
    for q in range(n_nodes):
        mask = 1 << q
        partner = bitstrings ^ mask
        state = cb * state - 1j * sb * state[partner]
    return float(np.real(np.sum(np.abs(state)**2 * cost_diag)))


# =====================================================================
# 2. GOLF-RESERVOIR
# =====================================================================

class WaveReservoir:
    """2D golfvergelijking-reservoir voor QAOA.

    De graaf wordt op een 2D grid getekend:
    - Nodes = golfbronnen (oscillerend met frequentie ~ gamma)
    - Edges = corridors waar golven doorheen stromen
    - Niet-verbonden gebieden = muren (demping)

    Na T tijdstappen wordt het interferentiepatroon uitgelezen.
    """

    def __init__(self, n_nodes, edges, grid_size=64, n_steps=100,
                 damping=0.002, wave_speed=0.3):
        self.n_nodes = n_nodes
        self.edges = edges
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.damping = damping
        self.c = wave_speed  # wave speed squared, for stability < 0.5

        # Bouw het statische landschap (B-kanaal)
        self.landscape = self._build_landscape()

        # Node posities op het grid
        self.node_positions = self._place_nodes()

    def _place_nodes(self):
        """Plaats nodes op een cirkel in het grid."""
        n = self.n_nodes
        gs = self.grid_size
        cx, cy = gs // 2, gs // 2
        r = gs * 0.3
        positions = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            x = np.clip(x, 2, gs - 3)
            y = np.clip(y, 2, gs - 3)
            positions.append((x, y))
        return positions

    def _build_landscape(self):
        """Bouw het B-kanaal: graafstructuur als golfmedium.

        1.0 = vrij medium (golf kan stromen)
        0.0 = muur (golf wordt gedempt)
        Edges worden getekend als corridors tussen node-posities.
        """
        gs = self.grid_size
        landscape = np.ones((gs, gs)) * 0.3  # achtergrond: licht dempend

        # Node-posities
        positions = self._place_nodes()

        # Teken nodes als vrije zones (cirkels)
        for x, y in positions:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < gs and 0 <= ny < gs:
                            landscape[ny, nx] = 1.0

        # Teken edges als corridors
        for i, j, w in self.edges:
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            # Bresenham-achtige lijn met breedte
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for t in range(steps + 1):
                frac = t / steps
                px = int(x1 + frac * (x2 - x1))
                py = int(y1 + frac * (y2 - y1))
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < gs and 0 <= ny < gs:
                            landscape[ny, nx] = max(landscape[ny, nx], 0.8 * w)

        return landscape

    def run(self, gamma, beta):
        """Simuleer golfvoortplanting en return features.

        gamma -> bronfrequentie (hoe snel de bronnen oscilleren)
        beta  -> bronamplitude (hoe hard de bronnen pompen)
        """
        gs = self.grid_size

        # R-kanaal (huidige golf) en G-kanaal (vorige golf)
        R = np.zeros((gs, gs))
        G = np.zeros((gs, gs))

        c_sq = self.c  # wave speed parameter
        damp = 1.0 - self.damping

        # Bron-frequenties per node (gemoduleerd door gamma)
        base_freq = gamma * 2.0 + 0.5  # frequentie uit gamma
        amplitude = beta * 0.5 + 0.1    # amplitude uit beta

        # Geef elke node een licht verschillende frequentie
        node_freqs = np.array([base_freq * (1 + 0.1 * i / max(1, self.n_nodes - 1))
                               for i in range(self.n_nodes)])

        # Tijdevolutie van de golfvergelijking
        for t in range(self.n_steps):
            # Laplaciaan (finite differences, 4-punts stencil)
            lap = (np.roll(R, 1, axis=0) + np.roll(R, -1, axis=0) +
                   np.roll(R, 1, axis=1) + np.roll(R, -1, axis=1) - 4 * R)

            # Golfvergelijking: R_new = 2*R - G + c^2 * lap * landscape
            R_new = (2 * R - G + c_sq * lap * self.landscape) * damp

            # Bronnen: nodes pompen golven het systeem in
            for idx, (x, y) in enumerate(self.node_positions):
                source = amplitude * np.sin(node_freqs[idx] * t * 0.1)
                R_new[y, x] += source

            # Absorberende randvoorwaarden (sponge layer)
            R_new[0:2, :] *= 0.5
            R_new[-2:, :] *= 0.5
            R_new[:, 0:2] *= 0.5
            R_new[:, -2:] *= 0.5

            # Schuif: G <- R, R <- R_new
            G = R.copy()
            R = R_new

        # Feature extractie uit het interferentiepatroon
        features = self._extract_features(R, G, gamma, beta)
        return features

    def _extract_features(self, R, G, gamma, beta):
        """Extraheer features uit het golfpatroon.

        Cruciaal: we gebruiken Fourier-features die gerelateerd zijn
        aan de spectrale eigenschappen van de graaf.
        """
        gs = self.grid_size

        # 1. Ruimtelijke statistieken
        spatial = [
            np.mean(R), np.std(R), np.max(R), np.min(R),
            np.mean(np.abs(R)), np.median(np.abs(R)),
            np.mean(G), np.std(G),
            np.mean(R * G),  # temporele correlatie
        ]

        # 2. Energie per kwadrant (ruimtelijke verdeling)
        half = gs // 2
        quadrants = [
            np.mean(R[:half, :half]**2),
            np.mean(R[:half, half:]**2),
            np.mean(R[half:, :half]**2),
            np.mean(R[half:, half:]**2),
        ]

        # 3. Fourier-features (de kern van de zaak!)
        # Het 2D power spectrum is gerelateerd aan de graph Laplacian eigenwaarden
        fft = np.fft.fft2(R)
        power = np.abs(fft)**2
        # Radieel gemiddeld power spectrum (8 bins)
        cx, cy = gs // 2, gs // 2
        ky, kx = np.mgrid[0:gs, 0:gs]
        kr = np.sqrt((kx - cx)**2 + (ky - cy)**2)
        power_shifted = np.fft.fftshift(power)
        radial_bins = 8
        max_r = gs // 2
        radial_power = []
        for b in range(radial_bins):
            r_low = b * max_r / radial_bins
            r_high = (b + 1) * max_r / radial_bins
            mask = (kr >= r_low) & (kr < r_high)
            if np.any(mask):
                radial_power.append(np.mean(power_shifted[mask]))
            else:
                radial_power.append(0.0)
        # Normaliseer
        total_power = sum(radial_power) + 1e-10
        radial_power = [p / total_power for p in radial_power]

        # 4. Node-specifieke metingen (golf op node-posities)
        node_amplitudes = []
        node_energies = []
        for x, y in self.node_positions:
            node_amplitudes.append(R[y, x])
            node_energies.append(R[y, x]**2 + G[y, x]**2)

        # Statistieken over nodes
        node_stats = [
            np.mean(node_amplitudes), np.std(node_amplitudes),
            np.mean(node_energies), np.std(node_energies),
            np.max(node_amplitudes) - np.min(node_amplitudes),
        ]

        # 5. Correlaties tussen verbonden en niet-verbonden nodes
        connected_corr = []
        unconnected_corr = []
        connected_set = set()
        for i, j, w in self.edges:
            connected_set.add((i, j))
            connected_set.add((j, i))

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                corr = node_amplitudes[i] * node_amplitudes[j]
                if (i, j) in connected_set:
                    connected_corr.append(corr)
                else:
                    unconnected_corr.append(corr)

        corr_features = [
            np.mean(connected_corr) if connected_corr else 0,
            np.mean(unconnected_corr) if unconnected_corr else 0,
            (np.mean(connected_corr) if connected_corr else 0) -
            (np.mean(unconnected_corr) if unconnected_corr else 0),
        ]

        # 6. Input echo
        echo = [gamma, beta, gamma * beta, gamma**2, beta**2]

        all_features = spatial + quadrants + radial_power + node_stats + corr_features + echo
        return np.array(all_features, dtype=float)


# =====================================================================
# 3. READOUT
# =====================================================================

class RidgeReadout:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0
        self.train_r2 = 0.0

    def fit(self, X, y):
        n, d = X.shape
        Xb = np.column_stack([X, np.ones(n)])
        A = Xb.T @ Xb + self.alpha * np.eye(d + 1)
        params = np.linalg.solve(A, Xb.T @ y)
        self.w = params[:-1]
        self.b = params[-1]
        y_pred = Xb @ params
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.train_r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return self

    def predict(self, X):
        return X @ self.w + self.b


# =====================================================================
# 4. PIPELINE
# =====================================================================

def run_wave_qaoa(n_nodes=8, n_train=100, n_test=30, grid_size=48,
                  n_steps=80, edge_prob=0.5, graph_seed=42):

    sep = "=" * 60
    print(sep)
    print("  WAVE RESERVOIR QAOA - Interferentie als Kwantumprocessor")
    print(sep)

    # 1. Graaf
    print("\n[1] Genereer graaf: %d nodes, p_edge=%.1f" % (n_nodes, edge_prob))
    edges = generate_random_graph(n_nodes, edge_prob, seed=graph_seed)
    n_edges = len(edges)
    total_weight = sum(w for _, _, w in edges)
    print("    %d edges, totaal gewicht: %d" % (n_edges, total_weight))
    if n_edges == 0:
        print("    GEEN EDGES")
        return

    # 2. Parameters
    n_total = n_train + n_test
    print("\n[2] Genereer %d parameter-punten" % n_total)
    rng = np.random.default_rng(123)
    all_gammas = rng.uniform(0.05, np.pi / 2, n_total)
    all_betas = rng.uniform(0.05, np.pi / 2, n_total)

    # 3. Exacte energieen
    print("\n[3] Bereken exacte QAOA-energieen (dim=2^%d=%d)..." % (n_nodes, 2**n_nodes))
    t0 = time.time()
    all_energies = np.array([
        exact_qaoa_energy(n_nodes, edges, g, b)
        for g, b in zip(all_gammas, all_betas)
    ])
    t_exact = time.time() - t0
    print("    %d evaluaties in %.2fs (%.1fms/eval)" % (
        n_total, t_exact, t_exact / n_total * 1000))
    print("    Energie range: [%.3f, %.3f]" % (all_energies.min(), all_energies.max()))

    # 4. Golf-reservoir
    print("\n[4] Draai golf-reservoir: %dx%d grid, %d stappen..." % (
        grid_size, grid_size, n_steps))
    reservoir = WaveReservoir(n_nodes, edges, grid_size=grid_size, n_steps=n_steps)

    t0 = time.time()
    all_features = []
    for idx, (g, b) in enumerate(zip(all_gammas, all_betas)):
        feat = reservoir.run(g, b)
        all_features.append(feat)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print("    %d/%d runs, %.1fs" % (idx + 1, n_total, elapsed))
    t_wave = time.time() - t0
    all_features = np.array(all_features)
    print("    Feature matrix: %s (%.2fs, %.1fms/run)" % (
        str(all_features.shape), t_wave, t_wave / n_total * 1000))

    # 5. Train/test
    X_train, y_train = all_features[:n_train], all_energies[:n_train]
    X_test, y_test = all_features[n_train:], all_energies[n_train:]

    # 6. Readout
    print("\n[5] Train ridge readout...")
    readout = RidgeReadout(alpha=0.5)
    readout.fit(X_train, y_train)
    print("    Train R2: %.4f" % readout.train_r2)

    # 7. Test
    y_pred = readout.predict(X_test)
    errors = np.abs(y_pred - y_test)
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    test_r2 = 1 - ss_res / max(ss_tot, 1e-10)

    print("\n[6] Test op %d onbekende parameters:" % n_test)
    print("    Test R2:         %.4f" % test_r2)
    print("    MAE:             %.4f" % np.mean(errors))
    print("    Max error:       %.4f" % np.max(errors))
    print("    Gem. rel. error: %.2f%%" % (np.mean(errors / np.maximum(np.abs(y_test), 1e-10)) * 100))

    # 8. Optimum zoeken
    print("\n[7] Zoek optimum via reservoir...")
    search_gammas = np.linspace(0.05, np.pi / 2, 20)
    search_betas = np.linspace(0.05, np.pi / 2, 20)

    best_pred = -1
    best_g, best_b = 0, 0
    for g in search_gammas:
        for b in search_betas:
            feat = reservoir.run(g, b)
            pred = readout.predict(feat.reshape(1, -1))[0]
            if pred > best_pred:
                best_pred = pred
                best_g, best_b = g, b

    exact_at_best = exact_qaoa_energy(n_nodes, edges, best_g, best_b)
    print("    Reservoir: gamma=%.3f, beta=%.3f" % (best_g, best_b))
    print("    Voorspeld: %.3f" % best_pred)
    print("    Werkelijk: %.3f (ratio=%.4f)" % (exact_at_best, exact_at_best / total_weight))

    # Exact optimum
    best_exact = 0
    best_eg, best_eb = 0, 0
    for g in search_gammas:
        for b in search_betas:
            e = exact_qaoa_energy(n_nodes, edges, g, b)
            if e > best_exact:
                best_exact = e
                best_eg, best_eb = g, b

    print("    Exact:     gamma=%.3f, beta=%.3f" % (best_eg, best_eb))
    print("    Exact:     %.3f (ratio=%.4f)" % (best_exact, best_exact / total_weight))
    quality = exact_at_best / max(best_exact, 1e-10)
    print("    Kwaliteit: %.1f%%" % (quality * 100))

    # Samenvatting
    print("\n" + sep)
    print("  SAMENVATTING")
    print(sep)
    print("  Graaf: %d nodes, %d edges" % (n_nodes, n_edges))
    print("  Golf-reservoir: %dx%d grid, %d stappen" % (grid_size, grid_size, n_steps))
    print("  Features: %d (incl. Fourier-spectrum)" % all_features.shape[1])
    print("  Train R2: %.4f | Test R2: %.4f" % (readout.train_r2, test_r2))
    print("  Kwaliteit: %.1f%% van exact optimum" % (quality * 100))
    print("  Snelheid: exact=%.1fms, golf=%.1fms per eval" % (
        t_exact / n_total * 1000, t_wave / n_total * 1000))
    print(sep)

    return {
        'test_r2': test_r2,
        'train_r2': readout.train_r2,
        'quality': quality,
        'n_features': all_features.shape[1],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wave Reservoir QAOA')
    parser.add_argument('--nodes', '-n', type=int, default=8)
    parser.add_argument('--train', type=int, default=100)
    parser.add_argument('--test', type=int, default=30)
    parser.add_argument('--grid', type=int, default=48)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--edge-prob', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_wave_qaoa(
        n_nodes=args.nodes, n_train=args.train, n_test=args.test,
        grid_size=args.grid, n_steps=args.steps,
        edge_prob=args.edge_prob, graph_seed=args.seed,
    )
