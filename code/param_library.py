#!/usr/bin/env python3
"""
Parameter-Bibliotheek per Graaftype (B57)
==========================================
Slaat optimale QAOA-parameters (gamma, beta) op per graafklasse en
QAOA-diepte p. Gebruikt als warm-start voor de optimizer.

Graafklassen worden bepaald door eenvoudige heuristieken:
    - is_grid + dimensies (Ly)
    - is_bipartite
    - is_regular + graad
    - gemiddelde graad
    - dichtheid

De bibliotheek groeit mee met gebruik: na elke succesvolle
optimalisatie worden betere parameters automatisch opgeslagen.

Gebruik:
    from param_library import ParamLibrary
    lib = ParamLibrary()

    # Lookup warm-start
    init = lib.lookup(graph_stats, p=1)
    # init = {'gammas': [...], 'betas': [...], 'ratio': 0.87, 'source': 'grid_Ly3'}

    # Update na optimalisatie
    lib.update(graph_stats, p=1, gammas=[0.88], betas=[1.18], ratio=0.95)
"""
import json
import os
import numpy as np
from typing import Optional, Dict, List, Tuple


# =====================================================================
# GRAPH CLASS KEYS
# =====================================================================

def classify_for_params(stats: Dict) -> str:
    """Bepaal een graafklasse-sleutel voor parameter lookup.

    Input: stats dict van auto_planner.classify_graph()
    Output: string key zoals 'grid_Ly3', 'reg3', 'bipartite_sparse', etc.
    """
    n = stats.get('n_nodes', 0)
    is_grid = stats.get('is_grid', False)
    grid_dims = stats.get('grid_dims', None)
    is_bip = stats.get('is_bipartite', False)
    avg_deg = stats.get('avg_degree', 0)
    min_deg = stats.get('min_degree', 0)
    max_deg = stats.get('max_degree', 0)
    density = stats.get('density', 0)

    # Grid grafen — specifiek per Ly
    if is_grid and grid_dims is not None:
        Lx, Ly = grid_dims
        if Ly == 1:
            return 'chain'
        return f'grid_Ly{Ly}'

    # Reguliere grafen
    if min_deg == max_deg and min_deg > 0:
        d = min_deg
        if is_bip:
            return f'reg{d}_bip'
        return f'reg{d}'

    # Bipartiet sparse
    if is_bip and density < 0.2:
        return 'bipartite_sparse'

    # Bipartiet dense
    if is_bip:
        return 'bipartite_dense'

    # Sparse non-bipartite
    if density < 0.1:
        deg_bucket = int(round(avg_deg))
        return f'sparse_d{deg_bucket}'

    # Dense
    if density > 0.5:
        return 'dense'

    # Medium density
    return f'medium_d{int(round(avg_deg))}'


# =====================================================================
# INITIAL PARAMETER DATABASE
# =====================================================================

# Gebaseerd op analytische resultaten en optimalisatie-runs.
# Format: {class_key: {p: {'gammas': [...], 'betas': [...], 'ratio': float}}}
#
# Referenties:
# - Grid p=1: gamma = pi/(4*d_avg) ≈ 0.393 voor d=2, beta ≈ pi*3/8 ≈ 1.178
# - 3-regulier p=1: Farhi et al. (2014), gamma ≈ 0.616, beta ≈ 0.393*pi ≈ 1.234
# - Complete K_n: gamma → 0 voor n→∞, beta ≈ pi/4

INITIAL_PARAMS = {
    # ── Grids ──────────────────────────────────────────────────
    'chain': {
        1: {'gammas': [0.785], 'betas': [0.785], 'ratio': 1.0,
            'note': '1D chain, bipartiet, altijd exact'},
    },
    'grid_Ly2': {
        1: {'gammas': [0.55], 'betas': [1.10], 'ratio': 0.95,
            'note': 'Ly=2 ladder'},
        2: {'gammas': [0.50, 0.65], 'betas': [1.05, 0.85], 'ratio': 0.98},
    },
    'grid_Ly3': {
        1: {'gammas': [0.45], 'betas': [1.18], 'ratio': 0.90,
            'note': 'Onze workhorse: 8x3, 10x3, 20x3'},
        2: {'gammas': [0.40, 0.55], 'betas': [1.15, 0.90], 'ratio': 0.95},
    },
    'grid_Ly4': {
        1: {'gammas': [0.40], 'betas': [1.18], 'ratio': 0.85,
            'note': 'Ly=4 grids'},
        2: {'gammas': [0.35, 0.50], 'betas': [1.12, 0.88], 'ratio': 0.92},
    },
    'grid_Ly5': {
        1: {'gammas': [0.38], 'betas': [1.15], 'ratio': 0.80},
    },
    'grid_Ly6': {
        1: {'gammas': [0.35], 'betas': [1.12], 'ratio': 0.75},
    },

    # ── Reguliere grafen ──────────────────────────────────────
    'reg3': {
        1: {'gammas': [0.616], 'betas': [1.234], 'ratio': 0.69,
            'note': 'Farhi et al. analytisch optimum'},
        2: {'gammas': [0.55, 0.70], 'betas': [1.20, 0.95], 'ratio': 0.76},
    },
    'reg3_bip': {
        1: {'gammas': [0.60], 'betas': [1.20], 'ratio': 0.80,
            'note': '3-reg bipartiet (bv cube)'},
    },
    'reg4': {
        1: {'gammas': [0.49], 'betas': [1.18], 'ratio': 0.67,
            'note': 'gamma ≈ pi/(4*d) voor d=4'},
    },
    'reg4_bip': {
        1: {'gammas': [0.45], 'betas': [1.15], 'ratio': 0.78},
    },
    'reg5': {
        1: {'gammas': [0.39], 'betas': [1.15], 'ratio': 0.65},
    },

    # ── Bipartiet ──────────────────────────────────────────────
    'bipartite_sparse': {
        1: {'gammas': [0.50], 'betas': [1.18], 'ratio': 0.85,
            'note': 'Generic bipartiet sparse, beter dan random init'},
    },
    'bipartite_dense': {
        1: {'gammas': [0.35], 'betas': [1.10], 'ratio': 0.80},
    },

    # ── Sparse non-bipartiet ──────────────────────────────────
    'sparse_d3': {
        1: {'gammas': [0.60], 'betas': [1.20], 'ratio': 0.65},
    },
    'sparse_d4': {
        1: {'gammas': [0.45], 'betas': [1.15], 'ratio': 0.63},
    },
    'sparse_d5': {
        1: {'gammas': [0.38], 'betas': [1.12], 'ratio': 0.60},
    },

    # ── Dense ──────────────────────────────────────────────────
    'dense': {
        1: {'gammas': [0.20], 'betas': [1.00], 'ratio': 0.55,
            'note': 'Dense grafen: kleine gamma, grote beta'},
    },

    # ── Medium density ─────────────────────────────────────────
    'medium_d3': {
        1: {'gammas': [0.55], 'betas': [1.18], 'ratio': 0.62},
    },
    'medium_d4': {
        1: {'gammas': [0.42], 'betas': [1.15], 'ratio': 0.60},
    },
    'medium_d5': {
        1: {'gammas': [0.35], 'betas': [1.10], 'ratio': 0.58},
    },
}

# Universal fallback (weighted average across classes)
UNIVERSAL_PARAMS = {
    1: {'gammas': [0.45], 'betas': [1.18], 'ratio': 0.60},
    2: {'gammas': [0.40, 0.55], 'betas': [1.15, 0.90], 'ratio': 0.65},
    3: {'gammas': [0.35, 0.45, 0.60], 'betas': [1.10, 0.95, 0.80], 'ratio': 0.68},
}


# =====================================================================
# PARAM LIBRARY CLASS
# =====================================================================

class ParamLibrary:
    """Parameter-bibliotheek met warm-start lookup en auto-update.

    Parameters worden opgeslagen in een JSON-bestand dat groeit met gebruik.
    De ingebouwde INITIAL_PARAMS dienen als basis; gebruikersdata
    overschrijft ze als de ratio beter is.

    Parameters
    ----------
    filepath : str or None
        Pad naar het JSON-bestand. None = alleen in-memory.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self._db = {}

        # Start met ingebouwde params
        for cls_key, p_dict in INITIAL_PARAMS.items():
            self._db[cls_key] = {}
            for p, params in p_dict.items():
                self._db[cls_key][str(p)] = params.copy()

        # Laad gebruikersdata (overschrijft als ratio beter)
        if filepath and os.path.exists(filepath):
            self._load(filepath)

    def _load(self, filepath):
        """Laad parameters uit JSON en merge met ingebouwde."""
        try:
            with open(filepath) as f:
                user_db = json.load(f)
            for cls_key, p_dict in user_db.items():
                if cls_key not in self._db:
                    self._db[cls_key] = {}
                for p_str, params in p_dict.items():
                    existing = self._db[cls_key].get(p_str, {})
                    if params.get('ratio', 0) >= existing.get('ratio', 0):
                        self._db[cls_key][p_str] = params
        except (json.JSONDecodeError, IOError):
            pass

    def _save(self):
        """Sla parameters op naar JSON."""
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._db, f, indent=2, default=_json_default)

    def lookup(self, stats: Dict, p: int = 1) -> Optional[Dict]:
        """Zoek warm-start parameters voor een graafklasse.

        Parameters
        ----------
        stats : dict
            Graph stats van classify_graph().
        p : int
            QAOA diepte.

        Returns
        -------
        dict or None
            {'gammas': [...], 'betas': [...], 'ratio': float, 'source': str}
            None als geen match.
        """
        cls_key = classify_for_params(stats)
        p_str = str(p)

        # Exact match
        if cls_key in self._db and p_str in self._db[cls_key]:
            entry = self._db[cls_key][p_str].copy()
            entry['source'] = cls_key
            entry['match'] = 'exact'
            return entry

        # Fallback: probeer parent class
        parent = self._parent_class(cls_key)
        if parent and parent in self._db and p_str in self._db[parent]:
            entry = self._db[parent][p_str].copy()
            entry['source'] = f'{parent} (fallback from {cls_key})'
            entry['match'] = 'parent'
            return entry

        # Universal fallback
        if p in UNIVERSAL_PARAMS:
            entry = UNIVERSAL_PARAMS[p].copy()
            entry['source'] = f'universal (no match for {cls_key})'
            entry['match'] = 'universal'
            return entry

        # Construct from p=1
        if 1 in UNIVERSAL_PARAMS:
            base = UNIVERSAL_PARAMS[1]
            entry = {
                'gammas': base['gammas'] * p,
                'betas': base['betas'] * p,
                'ratio': base['ratio'],
                'source': f'universal_repeated (p={p})',
                'match': 'constructed',
            }
            return entry

        return None

    def _parent_class(self, cls_key: str) -> Optional[str]:
        """Zoek een parent-klasse voor fallback."""
        # reg3_bip → reg3
        if '_bip' in cls_key:
            return cls_key.replace('_bip', '')
        # grid_Ly7 → grid_Ly6 → grid_Ly5 ...
        if cls_key.startswith('grid_Ly'):
            try:
                ly = int(cls_key[7:])
                if ly > 2:
                    return f'grid_Ly{ly - 1}'
            except ValueError:
                pass
        # sparse_d6 → sparse_d5
        if cls_key.startswith('sparse_d'):
            try:
                d = int(cls_key[8:])
                if d > 3:
                    return f'sparse_d{d - 1}'
            except ValueError:
                pass
        # medium_d6 → medium_d5
        if cls_key.startswith('medium_d'):
            try:
                d = int(cls_key[8:])
                if d > 3:
                    return f'medium_d{d - 1}'
            except ValueError:
                pass
        return None

    def update(self, stats: Dict, p: int, gammas: list, betas: list,
               ratio: float, force: bool = False) -> bool:
        """Update de bibliotheek als de ratio beter is.

        Parameters
        ----------
        stats : dict
            Graph stats.
        p : int
            QAOA diepte.
        gammas, betas : list
            Optimale parameters.
        ratio : float
            Behaalde cut-ratio.
        force : bool
            Overschrijf zelfs als ratio slechter is.

        Returns
        -------
        bool
            True als de entry is bijgewerkt.
        """
        cls_key = classify_for_params(stats)
        p_str = str(p)

        if cls_key not in self._db:
            self._db[cls_key] = {}

        existing = self._db[cls_key].get(p_str, {})
        if force or ratio > existing.get('ratio', 0):
            self._db[cls_key][p_str] = {
                'gammas': [float(g) for g in gammas],
                'betas': [float(b) for b in betas],
                'ratio': float(ratio),
                'n_nodes': stats.get('n_nodes', 0),
                'n_edges': stats.get('n_edges', 0),
            }
            self._save()
            return True
        return False

    def get_init_range(self, stats: Dict, p: int = 1,
                       spread: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Geef een search range gecentreerd rond de warm-start.

        Returns (gamma_range, beta_range) als 1D arrays voor grid search.
        """
        entry = self.lookup(stats, p)
        if entry is None:
            # Default range
            return (np.linspace(0.1, 1.2, 8),
                    np.linspace(0.1, 1.5, 8))

        g0 = entry['gammas'][0]
        b0 = entry['betas'][0]

        # Smaller focused range around warm-start
        g_lo = max(0.01, g0 - spread)
        g_hi = g0 + spread
        b_lo = max(0.01, b0 - spread)
        b_hi = b0 + spread

        return (np.linspace(g_lo, g_hi, 8),
                np.linspace(b_lo, b_hi, 8))

    def summary(self) -> str:
        """Human-readable samenvatting van de bibliotheek."""
        lines = ["Parameter Library Summary", "=" * 40]
        for cls_key in sorted(self._db.keys()):
            p_dict = self._db[cls_key]
            for p_str in sorted(p_dict.keys()):
                entry = p_dict[p_str]
                g = entry.get('gammas', [])
                b = entry.get('betas', [])
                r = entry.get('ratio', 0)
                note = entry.get('note', '')
                g_str = ', '.join(f'{x:.3f}' for x in g)
                b_str = ', '.join(f'{x:.3f}' for x in b)
                lines.append(f"  {cls_key:>20} p={p_str}: "
                           f"γ=[{g_str}] β=[{b_str}] "
                           f"ratio={r:.3f}"
                           f"{' — ' + note if note else ''}")
        lines.append(f"\nTotal: {sum(len(v) for v in self._db.values())} entries "
                    f"across {len(self._db)} classes")
        return '\n'.join(lines)

    def __len__(self):
        return sum(len(v) for v in self._db.values())

    def __repr__(self):
        return f"ParamLibrary({len(self)} entries, {len(self._db)} classes)"


def _json_default(obj):
    """JSON serializer voor numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =====================================================================
# CONVENIENCE: warm-started grid search
# =====================================================================

def warm_grid_search(eval_fn, stats: Dict, p: int = 1,
                     n_points: int = 8, spread: float = 0.3,
                     library: Optional[ParamLibrary] = None
                     ) -> Tuple[float, list, list, str]:
    """Grid search met warm-start uit de parameter-bibliotheek.

    Parameters
    ----------
    eval_fn : callable(gammas, betas) -> ratio
        Evaluatiefunctie die een QAOA-ratio teruggeeft.
    stats : dict
        Graph stats van classify_graph().
    p : int
        QAOA diepte.
    n_points : int
        Aantal punten per dimensie in de grid search.
    spread : float
        Zoekradius rond warm-start.
    library : ParamLibrary, optional
        Bibliotheek om te gebruiken. Default: globale instantie.

    Returns
    -------
    (best_ratio, best_gammas, best_betas, source)
    """
    if library is None:
        library = ParamLibrary()

    entry = library.lookup(stats, p)
    source = 'default'

    if entry is not None:
        source = entry.get('source', 'library')
        g0 = entry['gammas']
        b0 = entry['betas']

        # First: evaluate the warm-start directly
        try:
            warm_ratio = eval_fn(g0, b0)
        except Exception:
            warm_ratio = 0

        best_ratio = warm_ratio
        best_gammas = list(g0)
        best_betas = list(b0)

        # Then: focused grid search around warm-start
        g_range = np.linspace(max(0.01, g0[0] - spread),
                              g0[0] + spread, n_points)
        b_range = np.linspace(max(0.01, b0[0] - spread),
                              b0[0] + spread, n_points)
    else:
        best_ratio = 0
        best_gammas = [0.45] * p
        best_betas = [1.18] * p

        # Default wide grid search
        g_range = np.linspace(0.1, 1.2, n_points)
        b_range = np.linspace(0.1, 1.5, n_points)

    # Grid search (p=1 only searches first gamma/beta)
    for g_val in g_range:
        for b_val in b_range:
            gammas = [g_val] * p
            betas = [b_val] * p
            try:
                ratio = eval_fn(gammas, betas)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_gammas = gammas
                    best_betas = betas
            except Exception:
                continue

    # Auto-update library
    if best_ratio > 0:
        library.update(stats, p, best_gammas, best_betas, best_ratio)

    return best_ratio, best_gammas, best_betas, source


# =====================================================================
# SELF-TEST
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Parameter-Bibliotheek per Graaftype (B57)")
    print("=" * 60)

    lib = ParamLibrary()
    print(lib.summary())
    print()

    # Test lookup
    from auto_planner import classify_graph, make_grid_edges

    # Grid 8x3
    edges = make_grid_edges(8, 3)
    stats = classify_graph(24, edges)
    entry = lib.lookup(stats, p=1)
    print(f"grid_8x3 → class={classify_for_params(stats)}, "
          f"warm-start: γ={entry['gammas']}, β={entry['betas']}, "
          f"source={entry['source']}")

    # Random 3-regular
    stats_reg = {'n_nodes': 30, 'n_edges': 45, 'avg_degree': 3.0,
                 'max_degree': 3, 'min_degree': 3, 'density': 0.1,
                 'is_grid': False, 'grid_dims': None,
                 'is_bipartite': False, 'is_sparse': True,
                 'treewidth_upper': 5, 'classification': 'medium'}
    entry = lib.lookup(stats_reg, p=1)
    print(f"reg3_30 → class={classify_for_params(stats_reg)}, "
          f"warm-start: γ={entry['gammas']}, β={entry['betas']}, "
          f"source={entry['source']}")

    # Dense graph
    stats_dense = {'n_nodes': 20, 'n_edges': 100, 'avg_degree': 10.0,
                   'max_degree': 15, 'min_degree': 5, 'density': 0.53,
                   'is_grid': False, 'grid_dims': None,
                   'is_bipartite': False, 'is_sparse': False,
                   'treewidth_upper': 12, 'classification': 'hard'}
    entry = lib.lookup(stats_dense, p=1)
    print(f"dense_20 → class={classify_for_params(stats_dense)}, "
          f"warm-start: γ={entry['gammas']}, β={entry['betas']}, "
          f"source={entry['source']}")

    # Grid Ly=7 (falls back to Ly=6)
    stats_g7 = {'n_nodes': 56, 'n_edges': 97, 'avg_degree': 3.46,
                'max_degree': 4, 'min_degree': 2, 'density': 0.063,
                'is_grid': True, 'grid_dims': (8, 7),
                'is_bipartite': True, 'is_sparse': True,
                'treewidth_upper': 7, 'classification': 'hard'}
    entry = lib.lookup(stats_g7, p=1)
    print(f"grid_8x7 → class={classify_for_params(stats_g7)}, "
          f"warm-start: γ={entry['gammas']}, β={entry['betas']}, "
          f"source={entry['source']}")

    print(f"\n{lib}")
