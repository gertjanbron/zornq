#!/usr/bin/env python3
"""
particle_qaoa.py — Deeltjessimulatie als graaf-generator voor QAOA MaxCut.

Pipeline:
  1. Simuleer N deeltjes met elastische botsingen in een 2D doos
  2. Bouw een graaf: nodes = deeltjes, edges = botsingen (gewogen op frequentie)
  3. Los MaxCut op via brute-force (kleine N) of ZornQ MPS-engine (grote N)
  4. Visualiseer: deeltjesposities + MaxCut-partitie

Gebruik:
  python particle_qaoa.py                      # defaults: 12 deeltjes, 500 stappen
  python particle_qaoa.py --particles 20 --steps 1000 --speed 3.0
  python particle_qaoa.py --particles 8 --exact  # brute-force voor verificatie

Analogie:
  - Elke botsing = een 2-qubit ZZ-gate
  - Botsingsgrafiek = het MaxCut-probleem
  - De chaos genereert de probleeminstantie
  - QAOA lost het op alsof het een kwantumcomputer is
"""

import numpy as np
import argparse
import time
import sys
import os

# Probeer ZornQ engine te laden
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from zorn_mps import ZornMPS
    ZORNQ_AVAILABLE = True
except ImportError:
    ZORNQ_AVAILABLE = False


# =====================================================================
# 1. DEELTJESSIMULATIE
# =====================================================================

class ParticleBox:
    """2D elastische deeltjessimulatie in een doos.

    Deeltjes botsen onderling en met muren. Botsingen worden
    geregistreerd als edges in een graaf.
    """

    def __init__(self, n_particles: int, box_size: float = 100.0,
                 speed: float = 3.0, radius: float = 2.0,
                 elasticity: float = 1.0, gravity: float = 0.0,
                 seed: int = 42):
        self.n = n_particles
        self.box = box_size
        self.radius = radius
        self.elasticity = elasticity
        self.gravity = gravity
        self.rng = np.random.default_rng(seed)

        # Random posities (niet overlappend)
        self.pos = np.zeros((n_particles, 2))
        for i in range(n_particles):
            for attempt in range(1000):
                p = self.rng.uniform(radius, box_size - radius, size=2)
                if i == 0 or np.min(np.linalg.norm(self.pos[:i] - p, axis=1)) > 2 * radius:
                    self.pos[i] = p
                    break
            else:
                self.pos[i] = self.rng.uniform(radius, box_size - radius, size=2)

        # Random snelheden
        angles = self.rng.uniform(0, 2 * np.pi, n_particles)
        speeds = speed * (0.5 + self.rng.uniform(size=n_particles))
        self.vel = np.column_stack([np.cos(angles) * speeds, np.sin(angles) * speeds])

        # Botsingsregistratie
        self.collision_count = np.zeros((n_particles, n_particles), dtype=int)
        self.total_collisions = 0
        self.history = []  # [(stap, i, j, energie), ...]

    def step(self, dt: float = 1.0):
        """Eén tijdstap: beweeg en detecteer botsingen."""
        # Zwaartekracht
        self.vel[:, 1] += self.gravity * dt

        # Beweeg
        self.pos += self.vel * dt

        # Muurbotsingen
        for dim in range(2):
            low = self.pos[:, dim] < self.radius
            high = self.pos[:, dim] > self.box - self.radius
            self.pos[low, dim] = self.radius
            self.vel[low, dim] = np.abs(self.vel[low, dim]) * self.elasticity
            self.pos[high, dim] = self.box - self.radius
            self.vel[high, dim] = -np.abs(self.vel[high, dim]) * self.elasticity

        # Deeltje-deeltje botsingen
        step_collisions = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(dx)
                min_dist = 2 * self.radius
                if dist < min_dist and dist > 1e-10:
                    # Elastische botsing
                    n = dx / dist
                    dv = self.vel[i] - self.vel[j]
                    dvn = np.dot(dv, n)
                    if dvn > 0:  # Naderen
                        self.vel[i] -= dvn * n * self.elasticity
                        self.vel[j] += dvn * n * self.elasticity
                        # Overlap corrigeren
                        overlap = min_dist - dist
                        self.pos[i] -= n * overlap * 0.5
                        self.pos[j] += n * overlap * 0.5
                        # Registreer
                        energy = 0.5 * dvn**2
                        self.collision_count[i, j] += 1
                        self.collision_count[j, i] += 1
                        self.total_collisions += 1
                        step_collisions.append((i, j, energy))

        return step_collisions

    def simulate(self, n_steps: int, dt: float = 1.0, verbose: bool = True):
        """Draai de volledige simulatie."""
        t0 = time.time()
        for step in range(n_steps):
            collisions = self.step(dt)
            for i, j, e in collisions:
                self.history.append((step, i, j, e))
            if verbose and (step + 1) % (n_steps // 5) == 0:
                elapsed = time.time() - t0
                print(f"  stap {step+1}/{n_steps}: "
                      f"{self.total_collisions} botsingen, {elapsed:.2f}s")
        return self

    def to_graph(self, min_collisions: int = 1):
        """Converteer botsingsdata naar een gewogen edge-lijst.

        Returns:
            nodes: int (aantal nodes)
            edges: list of (i, j, weight)
        """
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.collision_count[i, j]
                if w >= min_collisions:
                    edges.append((i, j, w))
        return self.n, edges

    def graph_stats(self, edges):
        """Bereken grafiekstatistieken."""
        n = self.n
        degrees = np.zeros(n, dtype=int)
        for i, j, w in edges:
            degrees[i] += 1
            degrees[j] += 1
        return {
            'nodes': n,
            'edges': len(edges),
            'max_edges': n * (n-1) // 2,
            'density': len(edges) / max(1, n * (n-1) // 2),
            'avg_degree': np.mean(degrees),
            'max_degree': int(np.max(degrees)) if n > 0 else 0,
            'total_weight': sum(w for _, _, w in edges),
            'total_collisions': self.total_collisions,
        }


# =====================================================================
# 2. MAXCUT SOLVERS
# =====================================================================

def maxcut_brute_force(n_nodes: int, edges: list):
    """Exact MaxCut via brute force. Alleen voor n <= 20."""
    if n_nodes > 20:
        raise ValueError(f"Brute force te duur voor n={n_nodes} (max 20)")

    best_cut = 0
    best_partition = 0

    for partition in range(1, 2**(n_nodes - 1)):  # symmetrie: skip complement
        cut = 0
        for i, j, w in edges:
            bi = (partition >> i) & 1
            bj = (partition >> j) & 1
            if bi != bj:
                cut += w
        if cut > best_cut:
            best_cut = cut
            best_partition = partition

    # Decodeer partitie
    labels = np.array([(best_partition >> i) & 1 for i in range(n_nodes)])
    return best_cut, labels


def maxcut_qaoa_statevec(n_nodes: int, edges: list,
                         p: int = 1, n_gamma: int = 20, n_beta: int = 20):
    """QAOA MaxCut via exacte state-vector simulatie voor willekeurige grafen.

    Werkt voor n <= ~18 qubits (2^18 = 262144 amplitudes, past in RAM).
    Gebruikt de volledige QAOA-circuit: Hadamard → (gamma·ZZ, beta·X)^p → <C>.
    """
    if n_nodes > 18:
        print(f"  [WARN] State vector te groot voor n={n_nodes} (max 18), skip QAOA")
        return None, None

    total_weight = sum(w for _, _, w in edges)
    if total_weight == 0:
        return 0, 0

    dim = 2 ** n_nodes

    # Pre-bereken Z-operatoren per qubit als diagonaal (bitstring-basis)
    bitstrings = np.arange(dim)
    z_diag = {}  # qubit i -> array van +1/-1
    for i in range(n_nodes):
        z_diag[i] = 1 - 2 * ((bitstrings >> i) & 1).astype(float)

    # Pre-bereken ZZ-diagonaal per edge (gewogen)
    zz_phase_template = np.zeros(dim)
    for i, j, w in edges:
        zz_phase_template += w * z_diag[i] * z_diag[j]

    # Cost operator diagonaal: C = sum_edges w*(1 - ZiZj)/2
    cost_diag = np.zeros(dim)
    for i, j, w in edges:
        cost_diag += w * (1 - z_diag[i] * z_diag[j]) / 2

    # X-rotatie op qubit q: exp(-i*beta*X_q)
    # In computationele basis: Rx(2*beta) = cos(beta)*I - i*sin(beta)*X
    def apply_rx_all(state, beta):
        """Pas exp(-i*beta*X) toe op alle qubits."""
        cb = np.cos(beta)
        sb = np.sin(beta)
        result = state.copy()
        for q in range(n_nodes):
            # X op qubit q: flip bit q
            mask = 1 << q
            partner = bitstrings ^ mask  # indices met bit q geflipt
            new_result = cb * result - 1j * sb * result[partner]
            result = new_result
        return result

    def eval_qaoa(gammas_list, betas_list):
        """Evalueer QAOA verwachtingswaarde <C> voor gegeven parameters."""
        # Start: |+>^n
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)

        for layer in range(len(gammas_list)):
            gamma = gammas_list[layer]
            beta = betas_list[layer]
            # ZZ-fase gate: exp(-i*gamma*sum_edges w*ZiZj)
            state *= np.exp(-1j * gamma * zz_phase_template)
            # X-mixer: exp(-i*beta*sum_i Xi)
            state = apply_rx_all(state, beta)

        # <C> = <state|C|state>
        return float(np.real(np.sum(np.abs(state)**2 * cost_diag)))

    # Grid search
    gammas = np.linspace(0.05, np.pi/2, n_gamma)
    betas = np.linspace(0.05, np.pi/2, n_beta)

    best_cost = 0
    best_g, best_b = 0, 0
    print(f"  QAOA state-vector (n={n_nodes}, dim={dim}): "
          f"{n_gamma}x{n_beta} grid search...")

    t0 = time.time()
    for gamma in gammas:
        for beta in betas:
            cost = eval_qaoa([gamma], [beta])
            if cost > best_cost:
                best_cost = cost
                best_g, best_b = gamma, beta
    t1 = time.time()

    ratio = best_cost / total_weight if total_weight > 0 else 0
    print(f"  QAOA p={p}: cost={best_cost:.3f}/{total_weight} "
          f"(ratio={ratio:.4f})")
    print(f"  gamma*={best_g:.3f}, beta*={best_b:.3f}, tijd={t1-t0:.2f}s")

    return best_cost, ratio


# =====================================================================
# 3. VISUALISATIE (ASCII + optioneel matplotlib)
# =====================================================================

def ascii_graph(n_nodes: int, edges: list, labels=None, max_width: int = 60):
    """Teken de graaf als adjacency-achtige ASCII art."""
    print(f"\n  Graaf: {n_nodes} nodes, {len(edges)} edges")
    print(f"  {'─' * max_width}")

    # Adjacency
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j, w in edges:
        adj[i, j] = w
        adj[j, i] = w

    # Header
    hdr = "     " + "".join(f"{j:3d}" for j in range(min(n_nodes, 20)))
    print(hdr)
    for i in range(min(n_nodes, 20)):
        row = f"  {i:2d} |"
        for j in range(min(n_nodes, 20)):
            if i == j:
                row += "  ."
            elif adj[i, j] > 0:
                if labels is not None and labels[i] != labels[j]:
                    row += f" \033[91m{adj[i,j]:2d}\033[0m"  # rood = geknipte edge
                else:
                    row += f" {adj[i,j]:2d}"
            else:
                row += "  ."
        if labels is not None:
            row += f"  {'A' if labels[i] == 0 else 'B'}"
        print(row)

    if labels is not None:
        a_count = np.sum(labels == 0)
        b_count = np.sum(labels == 1)
        print(f"\n  Partitie: A={a_count} nodes, B={b_count} nodes")


def plot_results(box: ParticleBox, edges: list, labels=None,
                 filename: str = 'particle_qaoa_result.html'):
    """Genereer een interactieve HTML-visualisatie van het resultaat."""
    n = box.n
    pos = box.pos

    # Normaliseer posities naar [50, 550]
    if n == 0:
        return
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    span = pmax - pmin
    span[span < 1e-10] = 1
    norm_pos = 50 + 500 * (pos - pmin) / span

    # Bouw SVG
    svg_lines = []
    svg_lines.append('<svg width="600" height="600" xmlns="http://www.w3.org/2000/svg">')
    svg_lines.append('<rect width="600" height="600" fill="#0a0a0f"/>')

    # Edges
    max_w = max((w for _, _, w in edges), default=1)
    for i, j, w in edges:
        x1, y1 = norm_pos[i]
        x2, y2 = norm_pos[j]
        opacity = 0.15 + 0.6 * (w / max_w)
        color = "#ff6666" if labels is not None and labels[i] != labels[j] else "#3a5a8a"
        width = 0.5 + 2.5 * (w / max_w)
        svg_lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                        f'x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color}" stroke-width="{width:.1f}" '
                        f'opacity="{opacity:.2f}"/>')

    # Nodes
    for i in range(n):
        x, y = norm_pos[i]
        if labels is not None:
            color = "#7eb8ff" if labels[i] == 0 else "#ff9944"
        else:
            color = "#7eb8ff"
        svg_lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" '
                        f'fill="{color}" stroke="white" stroke-width="0.5"/>')
        svg_lines.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" '
                        f'fill="white" font-size="9" font-family="monospace">{i}</text>')

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    # Stats
    stats = box.graph_stats(edges)
    cut_value = 0
    if labels is not None:
        for i, j, w in edges:
            if labels[i] != labels[j]:
                cut_value += w

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Particle QAOA Result</title>
<style>
body {{ background: #0a0a0f; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
       display: flex; flex-direction: column; align-items: center; padding: 20px; }}
h1 {{ color: #7eb8ff; font-size: 20px; }}
.stats {{ background: #12121a; border-radius: 8px; padding: 16px; margin: 12px 0;
          font-family: monospace; font-size: 13px; line-height: 1.8;
          border: 1px solid #2a2a3a; max-width: 600px; width: 100%; }}
.stats b {{ color: #7eb8ff; }}
.cut {{ color: #ff6666; font-weight: bold; }}
.legend {{ display: flex; gap: 20px; margin: 8px 0; font-size: 12px; }}
.legend span {{ display: flex; align-items: center; gap: 4px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style></head><body>
<h1>Particle QAOA — Botsingsgraaf als MaxCut</h1>
<div class="legend">
  <span><span class="dot" style="background:#7eb8ff"></span> Partitie A</span>
  <span><span class="dot" style="background:#ff9944"></span> Partitie B</span>
  <span style="color:#ff6666">━ Geknipte edge (MaxCut)</span>
  <span style="color:#3a5a8a">━ Interne edge</span>
</div>
{svg_content}
<div class="stats">
  <b>Deeltjessimulatie</b><br>
  Deeltjes (qubits): {stats['nodes']}<br>
  Totale botsingen (gates): {stats['total_collisions']}<br>
  Unieke botsings-paren (edges): {stats['edges']} / {stats['max_edges']}<br>
  Graafdichtheid: {stats['density']:.2%}<br>
  Gem. graad: {stats['avg_degree']:.1f} | Max graad: {stats['max_degree']}<br>
  <br>
  <b>MaxCut Resultaat</b><br>
  Cut-waarde: <span class="cut">{cut_value}</span> / {stats['total_weight']} totaal gewicht<br>
  Cut-ratio: <span class="cut">{cut_value/max(1, stats['total_weight']):.4f}</span><br>
  Geknipte edges: {sum(1 for i,j,w in edges if labels is not None and labels[i] != labels[j])} / {len(edges)}<br>
  <br>
  <b>Analogie</b><br>
  Elke botsing = een 2-qubit quantum gate<br>
  De graaf = het verstrenglings-netwerk<br>
  MaxCut = de optimale partitie van het kwantumsysteem<br>
  Rode lijnen = de "geknipte" verstrengeling (cf. Circuit Knitting B31)
</div>
</body></html>"""

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Visualisatie opgeslagen: {filepath}")
    return filepath


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Particle → QAOA MaxCut pipeline')
    parser.add_argument('--particles', '-n', type=int, default=12,
                       help='Aantal deeltjes/qubits (default: 12)')
    parser.add_argument('--steps', '-s', type=int, default=500,
                       help='Simulatiestappen (default: 500)')
    parser.add_argument('--speed', type=float, default=3.0,
                       help='Deeltjessnelheid (default: 3.0)')
    parser.add_argument('--radius', type=float, default=3.0,
                       help='Deeltjesstraal (default: 3.0)')
    parser.add_argument('--box', type=float, default=50.0,
                       help='Doosgrootte (default: 50.0)')
    parser.add_argument('--gravity', type=float, default=0.0,
                       help='Zwaartekracht (default: 0.0)')
    parser.add_argument('--elasticity', type=float, default=1.0,
                       help='Elasticiteit 0-1 (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--min-collisions', type=int, default=1,
                       help='Min botsingen voor edge (default: 1)')
    parser.add_argument('--exact', action='store_true',
                       help='Brute-force exact (max ~20 nodes)')
    parser.add_argument('--chi', type=int, default=32,
                       help='Max bond dimension voor QAOA (default: 32)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip HTML visualisatie')
    args = parser.parse_args()

    print("=" * 60)
    print("  PARTICLE → QAOA MAXCUT PIPELINE")
    print("=" * 60)

    # Stap 1: Deeltjessimulatie
    print(f"\n[1] Deeltjessimulatie: {args.particles} deeltjes, "
          f"{args.steps} stappen, seed={args.seed}")
    box = ParticleBox(
        n_particles=args.particles,
        box_size=args.box,
        speed=args.speed,
        radius=args.radius,
        elasticity=args.elasticity,
        gravity=args.gravity,
        seed=args.seed,
    )
    box.simulate(args.steps, verbose=True)

    # Stap 2: Bouw graaf
    print(f"\n[2] Bouw botsingsgraaf (min {args.min_collisions} botsingen per edge)")
    n_nodes, edges = box.to_graph(min_collisions=args.min_collisions)
    stats = box.graph_stats(edges)
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']} / {stats['max_edges']} "
          f"(dichtheid: {stats['density']:.1%})")
    print(f"  Gem. graad: {stats['avg_degree']:.1f} | "
          f"Max graad: {stats['max_degree']}")
    print(f"  Totaal gewicht: {stats['total_weight']}")

    if len(edges) == 0:
        print("\n  GEEN EDGES — verhoog --steps, --speed of --radius")
        return

    # Stap 3: MaxCut
    labels = None
    exact_cut = None
    qaoa_cost = None

    if args.exact or n_nodes <= 20:
        print(f"\n[3a] Exact MaxCut (brute force, n={n_nodes})")
        t0 = time.time()
        exact_cut, labels = maxcut_brute_force(n_nodes, edges)
        t1 = time.time()
        total_weight = stats['total_weight']
        print(f"  Optimale cut: {exact_cut} / {total_weight} "
              f"(ratio: {exact_cut/total_weight:.4f})")
        print(f"  Tijd: {t1-t0:.3f}s")
        print(f"  Partitie: A={np.sum(labels==0)}, B={np.sum(labels==1)}")

    if n_nodes <= 18:
        print(f"\n[3b] QAOA MaxCut (state-vector, p=1)")
        qaoa_cost, qaoa_ratio = maxcut_qaoa_statevec(n_nodes, edges)

    # Vergelijking
    if exact_cut is not None and qaoa_cost is not None:
        total_weight = stats['total_weight']
        print(f"\n[4] Vergelijking:")
        print(f"  Exact:  {exact_cut}/{total_weight} = {exact_cut/total_weight:.4f}")
        print(f"  QAOA:   {qaoa_cost:.1f}/{total_weight} = {qaoa_cost/total_weight:.4f}")
        print(f"  QAOA/Exact: {qaoa_cost/max(1,exact_cut):.4f}")

    # Visualisatie
    if not args.no_plot and labels is not None:
        ascii_graph(n_nodes, edges, labels)
        print(f"\n[5] HTML visualisatie genereren...")
        plot_results(box, edges, labels)

    print(f"\n{'=' * 60}")
    print(f"  Deeltjes → Botsingsgraaf → MaxCut: KLAAR")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
