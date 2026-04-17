#!/usr/bin/env python3
"""
gset_loader.py - B61: Gset Benchmark Loader voor MaxCut.

Laadt Gset benchmark-instanties en biedt standaard vergelijkingsgrafen
voor publicatie. Ondersteunt:

1. Gset file parser (edge-list formaat van Stanford OR Library)
2. Best-Known Solutions (BKS) database voor directe vergelijking
3. Ingebouwde kleine benchmark-grafen voor lightcone-QAOA
4. Graaf-generatoren: 3-regulier, Erdos-Renyi, toroidal grid

Gebruik:
    python gset_loader.py --list                    # toon beschikbare grafen
    python gset_loader.py --graph petersen          # laad Petersen graaf
    python gset_loader.py --graph grid_6x3          # 6x3 grid
    python gset_loader.py --graph reg3_20 --seed 42 # random 3-reg, 20 nodes
    python gset_loader.py --file path/to/G14.txt    # laad Gset bestand
    python gset_loader.py --graph petersen --solve   # brute-force MaxCut

Gset bronnen:
    https://web.stanford.edu/~yyye/yyye/Gset/
    Format: eerste regel "N E", daarna "i j w" per edge
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqaoa import WeightedGraph, brute_force_maxcut


# =====================================================================
# Gset file parser
# =====================================================================

def parse_gset_file(filepath):
    """Parse een Gset-formaat bestand naar WeightedGraph.

    Formaat:
        Regel 1: N E    (aantal knopen, aantal edges)
        Regels 2+: i j w  (edge van i naar j, gewicht w)

    Knopen in Gset zijn 1-indexed, we converteren naar 0-indexed.
    """
    g = WeightedGraph()
    with open(filepath) as f:
        header = f.readline().strip().split()
        n_nodes = int(header[0])
        n_edges = int(header[1])

        # Voeg alle knopen toe (ook als ze geen edges hebben)
        for i in range(n_nodes):
            g.add_node(i)

        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            i = int(parts[0]) - 1  # 0-indexed
            j = int(parts[1]) - 1
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            if i != j:  # geen self-loops
                g.add_edge(i, j, w)

    return g, n_nodes, n_edges


# =====================================================================
# Best-Known Solutions database
# =====================================================================

# Bron: BLS (Benlic & Hao 2013), DSDP, recente papers
# Format: {naam: (n_nodes, n_edges, best_known_cut)}
GSET_BKS = {
    'G1':  (800, 19176, 11624),
    'G2':  (800, 19176, 11620),
    'G3':  (800, 19176, 11622),
    'G4':  (800, 19176, 11646),
    'G5':  (800, 19176, 11631),
    'G6':  (800, 19176, 2178),
    'G7':  (2000, 19176, 2006),
    'G8':  (2000, 19176, 2005),
    'G9':  (2000, 19176, 2054),
    'G10': (2000, 19176, 2000),
    'G11': (800, 1600, 564),
    'G12': (800, 1600, 556),
    'G13': (800, 1600, 582),
    'G14': (800, 4694, 3064),
    'G15': (800, 4661, 3050),
    'G16': (800, 4672, 3052),
    'G17': (800, 4667, 3047),
    'G18': (800, 4694, 992),
    'G19': (800, 4661, 906),
    'G20': (800, 4672, 941),
    'G21': (800, 4667, 931),
    'G22': (2000, 19990, 13359),
    'G23': (2000, 19990, 13344),
    'G24': (2000, 19990, 13337),
    'G25': (2000, 19990, 13340),
    'G26': (2000, 19990, 13328),
    'G27': (2000, 19990, 3341),
    'G28': (2000, 19990, 3298),
    'G29': (2000, 19990, 3405),
    'G30': (2000, 19990, 3413),
    'G31': (2000, 19990, 3310),
    'G32': (2000, 4000, 1410),
    'G33': (2000, 4000, 1382),
    'G34': (2000, 4000, 1384),
    'G35': (2000, 11778, 7687),
    'G36': (2000, 4000, 7680),
    'G37': (2000, 4000, 7691),
    'G38': (2000, 4000, 7688),
    'G39': (2000, 4000, 2408),
    'G40': (2000, 4000, 2400),
    'G41': (2000, 4000, 2405),
    'G42': (2000, 4000, 2481),
    'G43': (1000, 9990, 6660),
    'G44': (1000, 9990, 6650),
    'G45': (1000, 9990, 6654),
    'G46': (1000, 9990, 6649),
    'G47': (1000, 9990, 6657),
    'G48': (3000, 6000, 6000),
    'G49': (3000, 6000, 6000),
    'G50': (3000, 6000, 5880),
    'G51': (1000, 5909, 3848),
    'G52': (1000, 5916, 3851),
    'G53': (1000, 5914, 3850),
    'G54': (1000, 5916, 3852),
    'G55': (5000, 12498, 10294),
    'G56': (5000, 12498, 4012),
    'G57': (5000, 10000, 3492),
    'G58': (5000, 29570, 19263),
    'G59': (5000, 29532, 6078),
    'G60': (7000, 17148, 14176),
    'G61': (7000, 17148, 5789),
    'G62': (7000, 14000, 4868),
    'G63': (7000, 41459, 27045),
    'G64': (7000, 41459, 8735),
    'G65': (8000, 16000, 5558),
    'G66': (9000, 18000, 6360),
    'G67': (10000, 20000, 6940),
    # Grote instanties (Rinaldi et al., Dunning et al.)
    'G70': (10000, 9999, 9541),
    'G72': (10000, 20000, 6998),
    'G77': (14000, 28000, 9926),
    'G81': (20000, 40000, 14030),
}


# =====================================================================
# Ingebouwde kleine benchmark-grafen (voor lightcone-QAOA)
# =====================================================================

def make_petersen():
    """Petersen graaf: 10 nodes, 15 edges, 3-regulier, MaxCut = 12."""
    g = WeightedGraph()
    # Buitenring
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    # Binnenring (pentagram)
    for i in range(5):
        g.add_edge(5 + i, 5 + (i + 2) % 5)
    # Spaken
    for i in range(5):
        g.add_edge(i, 5 + i)
    return g, 12  # BKS = 12


def make_cube():
    """Kubus (3D hypercube): 8 nodes, 12 edges, 3-regulier, MaxCut = 12."""
    g = WeightedGraph()
    # 3-bit hypercube: edge als Hamming-afstand 1
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << bit)
            if j > i:
                g.add_edge(i, j)
    return g, 12  # bipartiet, MaxCut = alle edges


def make_dodecahedron():
    """Dodecaeder: 20 nodes, 30 edges, 3-regulier, MaxCut = 25.

    Frustratie: bevat oneven cykels (5-ringen), dus niet bipartiet.
    """
    g = WeightedGraph()
    # Standaard dodecaeder adjacency
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),           # buitenring
        (0,5),(1,6),(2,7),(3,8),(4,9),            # spaken naar ring 2
        (5,10),(6,11),(7,12),(8,13),(9,14),       # spaken naar ring 3
        (10,11),(11,12),(12,13),(13,14),(14,10),  # binnenring
        (5,15),(6,16),(7,17),(8,18),(9,19),       # spaken naar kern
        (15,16),(16,17),(17,18),(18,19),(19,15),  # kern ring
    ]
    for i, j in edges:
        g.add_edge(i, j)
    return g, 27  # Gecorrigeerd: Lanczos exact bewijs (was 25)


def make_complete(n):
    """Complete graaf K_n. MaxCut = n^2/4 (afgerond)."""
    g = WeightedGraph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    bks = (n * n) // 4
    return g, bks


def make_cycle(n):
    """Cykel C_n. MaxCut = n als n even, n-1 als n oneven."""
    g = WeightedGraph()
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    bks = n if n % 2 == 0 else n - 1
    return g, bks


def make_grid(Lx, Ly):
    """Lx x Ly grid graaf. Bipartiet, MaxCut = alle edges."""
    g = WeightedGraph.grid(Lx, Ly)
    n_edges = (Lx - 1) * Ly + Lx * (Ly - 1)
    return g, n_edges  # bipartiet -> BKS = n_edges


def make_toroidal_grid(Lx, Ly):
    """Toroidaal Lx x Ly grid (periodieke randvoorwaarden).

    Bipartiet als Lx en Ly beide even. MaxCut = 2*Lx*Ly als bipartiet.
    """
    g = WeightedGraph()
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            g.add_node(node)
            # Horizontaal (periodiek)
            right = ((x + 1) % Lx) * Ly + y
            g.add_edge(node, right)
            # Verticaal (periodiek)
            down = x * Ly + (y + 1) % Ly
            g.add_edge(node, down)
    n_edges = 2 * Lx * Ly
    # Bipartiet als Lx en Ly beide even
    if Lx % 2 == 0 and Ly % 2 == 0:
        bks = n_edges
    else:
        bks = None  # niet triviaal te bepalen
    return g, bks


def make_random_regular(n, d=3, seed=None):
    """Random d-reguliere graaf op n nodes.

    Gebruikt pairing model. Vereist n*d even.
    """
    if (n * d) % 2 != 0:
        raise ValueError("n*d moet even zijn, kreeg n=%d d=%d" % (n, d))

    rng = np.random.RandomState(seed)
    max_attempts = 200

    for _ in range(max_attempts):
        points = []
        for i in range(n):
            points.extend([i] * d)
        rng.shuffle(points)

        g = WeightedGraph()
        for i in range(n):
            g.add_node(i)

        valid = True
        for k in range(0, len(points), 2):
            i, j = points[k], points[k + 1]
            if i == j:
                valid = False
                break
            if j in g.adj.get(i, {}):
                valid = False
                break
            g.add_edge(i, j)

        if valid and all(len(g.adj[i]) == d for i in range(n)):
            return g, None  # BKS onbekend

    raise RuntimeError("Kon geen geldige %d-reguliere graaf genereren" % d)


def make_erdos_renyi(n, p=0.5, seed=None):
    """Erdos-Renyi random graaf G(n, p)."""
    rng = np.random.RandomState(seed)
    g = WeightedGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j)
    return g, None  # BKS onbekend


def make_triangular_grid(Lx, Ly):
    """Triangulair grid op cilinder: vierkant grid + diagonale edges.

    Niet bipartiet door driehoeken. Interessant voor frustratie-effecten.
    """
    g = WeightedGraph()
    for x in range(Lx):
        for y in range(Ly):
            node = x * Ly + y
            g.add_node(node)
            if x + 1 < Lx:
                g.add_edge(node, (x + 1) * Ly + y)
            if y + 1 < Ly:
                g.add_edge(node, x * Ly + y + 1)
            # Diagonaal: (x,y) -> (x+1, y+1)
            if x + 1 < Lx and y + 1 < Ly:
                g.add_edge(node, (x + 1) * Ly + y + 1)
    return g, None  # niet triviaal


# =====================================================================
# Catalogus van ingebouwde grafen
# =====================================================================

BUILTIN_GRAPHS = {
    # Klassieke kleine grafen
    'petersen':     {'desc': 'Petersen (10n, 15e, 3-reg)', 'func': make_petersen},
    'cube':         {'desc': 'Kubus/Q3 (8n, 12e, 3-reg, bipartiet)', 'func': make_cube},
    'dodecahedron': {'desc': 'Dodecaeder (20n, 30e, 3-reg)', 'func': make_dodecahedron},

    # Cykels
    'cycle_8':      {'desc': 'Cykel C8 (even, bipartiet)', 'func': lambda: make_cycle(8)},
    'cycle_11':     {'desc': 'Cykel C11 (oneven, gefrustreerd)', 'func': lambda: make_cycle(11)},
    'cycle_20':     {'desc': 'Cykel C20 (even, bipartiet)', 'func': lambda: make_cycle(20)},

    # Complete grafen
    'K5':           {'desc': 'Complete K5 (10e)', 'func': lambda: make_complete(5)},
    'K8':           {'desc': 'Complete K8 (28e)', 'func': lambda: make_complete(8)},
    'K10':          {'desc': 'Complete K10 (45e)', 'func': lambda: make_complete(10)},

    # Grid grafen (onze specialiteit)
    'grid_4x3':     {'desc': 'Grid 4x3 (12n, bipartiet)', 'func': lambda: make_grid(4, 3)},
    'grid_6x3':     {'desc': 'Grid 6x3 (18n, bipartiet)', 'func': lambda: make_grid(6, 3)},
    'grid_8x3':     {'desc': 'Grid 8x3 (24n, bipartiet)', 'func': lambda: make_grid(8, 3)},
    'grid_8x4':     {'desc': 'Grid 8x4 (32n, bipartiet)', 'func': lambda: make_grid(8, 4)},
    'grid_10x3':    {'desc': 'Grid 10x3 (30n, bipartiet)', 'func': lambda: make_grid(10, 3)},
    'grid_20x3':    {'desc': 'Grid 20x3 (60n, bipartiet)', 'func': lambda: make_grid(20, 3)},

    # Triangulaire grids (gefrustreerd)
    'tri_4x3':      {'desc': 'Triangulair 4x3 (gefrustreerd)', 'func': lambda: make_triangular_grid(4, 3)},
    'tri_6x3':      {'desc': 'Triangulair 6x3 (gefrustreerd)', 'func': lambda: make_triangular_grid(6, 3)},

    # Toroidale grids
    'torus_4x4':    {'desc': 'Torus 4x4 (32e, bipartiet)', 'func': lambda: make_toroidal_grid(4, 4)},
    'torus_6x4':    {'desc': 'Torus 6x4 (48e, bipartiet)', 'func': lambda: make_toroidal_grid(6, 4)},
}

# Dynamische grafen met parameters
PARAM_GRAPHS = {
    'reg3': {'desc': 'Random 3-regulier (geef _N, bv reg3_20)', 'func': make_random_regular, 'd': 3},
    'reg4': {'desc': 'Random 4-regulier (geef _N, bv reg4_20)', 'func': make_random_regular, 'd': 4},
    'er':   {'desc': 'Erdos-Renyi p=0.3 (geef _N, bv er_16)', 'func': make_erdos_renyi, 'p': 0.3},
    'grid': {'desc': 'Grid LxxLy (geef _LxxLy, bv grid_12x4)', 'func': make_grid},
    'tri':  {'desc': 'Triangulair LxxLy (geef _LxxLy)', 'func': make_triangular_grid},
}


def load_graph(name, seed=None):
    """Laad een graaf op naam. Returns (WeightedGraph, bks_or_None, info_dict).

    Ondersteunde namen:
        - Ingebouwde: 'petersen', 'cube', 'grid_6x3', etc.
        - Parametrisch: 'reg3_20', 'er_16', 'grid_12x4', etc.
        - Gset: 'G14' (vereist bestand in gset/ directory)
    """
    info = {'name': name, 'source': 'builtin'}

    # Check ingebouwde grafen
    if name in BUILTIN_GRAPHS:
        g, bks = BUILTIN_GRAPHS[name]['func']()
        info['description'] = BUILTIN_GRAPHS[name]['desc']
        info['n_nodes'] = g.n_nodes
        info['n_edges'] = g.n_edges
        info['bks'] = bks
        return g, bks, info

    # Check parametrische grafen (naam_param)
    for prefix, spec in PARAM_GRAPHS.items():
        if name.startswith(prefix + '_'):
            param_str = name[len(prefix) + 1:]
            if 'x' in param_str:
                # Grid formaat: LxXLy
                parts = param_str.split('x')
                Lx, Ly = int(parts[0]), int(parts[1])
                g, bks = spec['func'](Lx, Ly)
            else:
                n = int(param_str)
                if 'reg' in prefix:
                    g, bks = spec['func'](n, d=spec['d'], seed=seed)
                elif prefix == 'er':
                    g, bks = spec['func'](n, p=spec['p'], seed=seed)
                else:
                    g, bks = spec['func'](n)

            info['source'] = 'generated'
            info['description'] = spec['desc']
            info['n_nodes'] = g.n_nodes
            info['n_edges'] = g.n_edges
            info['bks'] = bks
            if seed is not None:
                info['seed'] = seed
            return g, bks, info

    # Check Gset bestanden
    if name.upper().startswith('G') and name[1:].isdigit():
        return load_gset(name, seed)

    raise ValueError("Onbekende graaf: '%s'. Gebruik --list voor opties." % name)


def load_gset(name, seed=None):
    """Zoek en laad een Gset bestand."""
    name_upper = name.upper()
    info = {'name': name_upper, 'source': 'gset'}

    # Zoek het bestand in mogelijke locaties
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, '..', 'gset', name_upper),
        os.path.join(script_dir, '..', 'gset', name_upper + '.txt'),
        os.path.join(script_dir, 'gset', name_upper),
        os.path.join(script_dir, 'gset', name_upper + '.txt'),
        os.path.join(script_dir, '..', 'data', 'gset', name_upper),
    ]

    filepath = None
    for p in search_paths:
        if os.path.exists(p):
            filepath = p
            break

    if filepath is None:
        # Check of we BKS info hebben
        if name_upper in GSET_BKS:
            n, e, bks = GSET_BKS[name_upper]
            raise FileNotFoundError(
                "Gset bestand '%s' niet gevonden.\n"
                "  Verwacht: gset/%s of gset/%s.txt\n"
                "  Download van: https://web.stanford.edu/~yyye/yyye/Gset/\n"
                "  Info: %d nodes, %d edges, BKS=%d" % (
                    name_upper, name_upper, name_upper, n, e, bks))
        else:
            raise FileNotFoundError(
                "Gset bestand '%s' niet gevonden en niet in BKS database." % name)

    g, n_nodes, n_edges = parse_gset_file(filepath)
    bks = GSET_BKS.get(name_upper, (None, None, None))[2]

    info['filepath'] = filepath
    info['n_nodes'] = g.n_nodes
    info['n_edges'] = g.n_edges
    info['bks'] = bks
    info['gset_header'] = (n_nodes, n_edges)
    return g, bks, info


# =====================================================================
# Graaf-naar-grid conversie (voor lightcone_qaoa.py)
# =====================================================================

def graph_to_grid_edges(graph):
    """Converteer een WeightedGraph naar (Lx, Ly, edge_list) als het
    een grid-structuur heeft. Returns None als het geen grid is.

    Dit is een heuristiek: check of de graaf isomorf is met een
    rechthoekig grid. Bruikbaar voor ingebouwde grid-instanties.
    """
    n = graph.n_nodes
    # Probeer alle (Lx, Ly) combinaties
    for Ly in range(2, min(n, 10)):
        if n % Ly != 0:
            continue
        Lx = n // Ly
        if Lx < 2:
            continue

        # Bouw verwachte grid en vergelijk edges
        expected = WeightedGraph.grid(Lx, Ly)
        if expected.n_edges != graph.n_edges:
            continue

        # Simpele check: dezelfde edge-set (na sortering van nodes)
        exp_edges = set((min(i, j), max(i, j)) for i, j, _ in expected.edges())
        act_edges = set((min(i, j), max(i, j)) for i, j, _ in graph.edges())
        if exp_edges == act_edges:
            return Lx, Ly

    return None


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='B61: Gset Benchmark Loader')
    parser.add_argument('--graph', type=str, default=None,
                        help='Graaf naam (bv petersen, grid_6x3, reg3_20, G14)')
    parser.add_argument('--file', type=str, default=None,
                        help='Pad naar Gset-formaat bestand')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed voor gegenereerde grafen')
    parser.add_argument('--list', action='store_true',
                        help='Toon beschikbare grafen')
    parser.add_argument('--solve', action='store_true',
                        help='Bereken exact MaxCut (brute force, max ~25 nodes)')
    parser.add_argument('--json', action='store_true',
                        help='Output als JSON')
    args = parser.parse_args()

    sep = "=" * 60

    if args.list:
        print(sep)
        print("  B61: Beschikbare Benchmark-Grafen")
        print(sep)
        print("\n  Ingebouwde grafen:")
        for name, spec in BUILTIN_GRAPHS.items():
            print("    %-20s %s" % (name, spec['desc']))
        print("\n  Parametrische grafen (naam_N of naam_LxxLy):")
        for name, spec in PARAM_GRAPHS.items():
            print("    %-20s %s" % (name + '_N', spec['desc']))
        print("\n  Gset (vereist download):")
        print("    G1..G67            Stanford Gset (800-10000 nodes)")
        print("    Download: https://web.stanford.edu/~yyye/yyye/Gset/")
        print("\n  Gset BKS database: %d instanties geladen" % len(GSET_BKS))
        print(sep)
        return

    # Laad graaf
    if args.file:
        g, n_nodes, n_edges = parse_gset_file(args.file)
        bks = None
        info = {'name': os.path.basename(args.file), 'source': 'file',
                'n_nodes': g.n_nodes, 'n_edges': g.n_edges, 'bks': None}
    elif args.graph:
        g, bks, info = load_graph(args.graph, seed=args.seed)
    else:
        print("Geef --graph <naam>, --file <pad>, of --list")
        return

    # Output
    if args.json:
        edge_list = [(i, j, w) for i, j, w in g.edges()]
        data = {
            'info': info,
            'edges': edge_list,
        }
        print(json.dumps(data, indent=2))
        return

    print(sep)
    print("  Graaf: %s" % info['name'])
    print(sep)
    print("  Nodes:    %d" % g.n_nodes)
    print("  Edges:    %d" % g.n_edges)
    if info.get('bks') is not None:
        print("  BKS:      %d (ratio %.6f)" % (
            info['bks'], info['bks'] / max(g.total_weight(), 1)))
    if info.get('description'):
        print("  Type:     %s" % info['description'])
    if info.get('seed') is not None:
        print("  Seed:     %d" % info['seed'])

    # Grid detectie
    grid = graph_to_grid_edges(g)
    if grid:
        print("  Grid:     %dx%d (compatible met lightcone_qaoa.py)" % grid)

    # Graad-verdeling
    degrees = sorted([len(g.adj[n]) for n in g.nodes])
    deg_min, deg_max = degrees[0], degrees[-1]
    deg_avg = sum(degrees) / len(degrees)
    if deg_min == deg_max:
        print("  Graad:    %d-regulier" % deg_min)
    else:
        print("  Graad:    min=%d, max=%d, gem=%.1f" % (deg_min, deg_max, deg_avg))

    # Exact MaxCut
    if args.solve:
        if g.n_nodes > 25:
            print("\n  Brute force niet haalbaar voor %d nodes (max ~25)" % g.n_nodes)
        else:
            import time
            print("\n  Brute-force MaxCut...")
            t0 = time.time()
            cut_val, partition = brute_force_maxcut(g)
            elapsed = time.time() - t0
            ratio = cut_val / g.total_weight() if g.total_weight() > 0 else 0
            print("  MaxCut:   %d / %d edges (ratio %.6f)" % (
                cut_val, g.n_edges, ratio))
            print("  Tijd:     %.3fs" % elapsed)
            if info.get('bks') is not None:
                gap = abs(info['bks'] - cut_val)
                if gap > 0:
                    print("  BKS gap:  %d (BKS=%d)" % (gap, info['bks']))
                else:
                    print("  Bevestigd: exact = BKS")

    print(sep)


if __name__ == '__main__':
    main()
