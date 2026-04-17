#!/usr/bin/env python3
"""
audit_trail.py - B56: Resultaat-Export met Audit Trail.

Schrijft na elke run een standaard artefact dat alle beslissingen vastlegt.
Goud voor papers, foutenanalyse en reproduceerbaarheid.

Elk artefact bevat:
  - run_id: sha256(graaf + p + seed + code_version)
  - environment: GPU, RAM, hostname, OS, Python-versie, packages
  - route: gekozen methode per edge (lightcone/MPS/RQAOA)
  - parameters: alle optimizer-instellingen en resultaten
  - timing: per-fase breakdown
  - bounds: lower/upper bounds indien beschikbaar
  - diagnostics: max lightcone qubits, chi, truncatie-fout

Gebruik:
  # Standalone
  from audit_trail import AuditTrail
  audit = AuditTrail(graph_desc="grid_20x3", p=3, seed=42)
  audit.log_phase("grid_search", ratio=0.75, time_s=12.3, n_evals=100)
  audit.log_phase("scipy", ratio=0.80, time_s=8.1, n_evals=50)
  audit.set_result(ratio=0.8037, gammas=[...], betas=[...])
  audit.save("results/run_001.json")
  audit.save_html("results/run_001.html")

  # Via bench harnas (B53)
  python zornq_bench.py --suite small --audit

  # Bekijk bestaand artefact
  python audit_trail.py --show results/run_001.json
  python audit_trail.py --compare run_001.json run_002.json

Bouwt voort op: zornq_bench.py (B53), lightcone_qaoa.py, ma_qaoa.py (B67)
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

__version__ = "1.0.0"


# =====================================================================
# Environment helpers
# =====================================================================

def _get_git_info():
    """Haal git commit, branch en dirty status."""
    info = {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}
    code_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5, cwd=code_dir)
        if r.returncode == 0:
            info['commit'] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5, cwd=code_dir)
        if r.returncode == 0:
            info['branch'] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5, cwd=code_dir)
        if r.returncode == 0:
            info['dirty'] = len(r.stdout.strip()) > 0
    except Exception:
        pass
    return info


def _get_gpu_info():
    """Detecteer GPU via nvidia-smi of CuPy."""
    gpu = {'name': 'none', 'vram_mb': 0, 'driver': 'unknown'}
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = r.stdout.strip().split(', ')
            if len(parts) >= 3:
                gpu['name'] = parts[0].strip()
                gpu['vram_mb'] = int(float(parts[1].strip()))
                gpu['driver'] = parts[2].strip()
    except Exception:
        pass
    return gpu


def _get_package_versions():
    """Versies van kernpakketten."""
    versions = {'python': platform.python_version()}
    for pkg in ['numpy', 'scipy', 'cupy']:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass
    return versions


def get_environment():
    """Verzamel volledige environment-informatie."""
    ram_mb = 0
    try:
        import psutil
        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        try:
            # Fallback: /proc/meminfo op Linux
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        ram_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass

    return {
        'hostname': socket.gethostname(),
        'os': platform.platform(),
        'cpu': platform.processor() or platform.machine(),
        'ram_mb': ram_mb,
        'gpu': _get_gpu_info(),
        'python': platform.python_version(),
        'packages': _get_package_versions(),
        'git': _get_git_info(),
        'audit_trail_version': __version__,
    }


# =====================================================================
# Run ID generatie
# =====================================================================

def make_run_id(graph_desc, p, seed=None, code_version=None):
    """Genereer deterministische run_id via SHA256.

    Args:
        graph_desc: string beschrijving van de graaf (bijv. "grid_20x3")
        p: circuit diepte
        seed: random seed (None wordt "noseed")
        code_version: git hash (auto-detected als None)

    Returns:
        12-karakter hex string
    """
    if code_version is None:
        code_version = _get_git_info()['commit']
    seed_str = str(seed) if seed is not None else "noseed"
    payload = "%s|p=%d|seed=%s|code=%s" % (graph_desc, p, seed_str, code_version)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# =====================================================================
# AuditTrail klasse
# =====================================================================

class AuditTrail:
    """Verzamelt en exporteert audit-informatie voor een enkele run.

    Typisch gebruik:
        audit = AuditTrail("grid_20x3", p=3, seed=42)
        audit.log_phase("grid_search", ratio=0.75, ...)
        audit.set_result(ratio=0.80, gammas=[...], betas=[...])
        audit.save("output.json")
    """

    def __init__(self, graph_desc, p, seed=None, method="lightcone",
                 extra_config=None):
        """Initialiseer audit trail voor een run.

        Args:
            graph_desc: graafbeschrijving (bijv. "grid_20x3", "petersen")
            p: circuit diepte
            seed: random seed
            method: primaire methode ("lightcone", "rqaoa", "ma-qaoa", "mps")
            extra_config: dict met extra configuratie-parameters
        """
        self.start_time = time.time()
        self.start_datetime = datetime.now(timezone.utc).isoformat()

        self.graph_desc = graph_desc
        self.p = p
        self.seed = seed
        self.method = method

        env = get_environment()
        self.run_id = make_run_id(graph_desc, p, seed, env['git']['commit'])

        self.data = {
            'run_id': self.run_id,
            'timestamp': self.start_datetime,
            'graph': graph_desc,
            'p': p,
            'seed': seed,
            'method': method,
            'config': extra_config or {},
            'environment': env,
            'phases': [],
            'result': None,
            'bounds': {},
            'diagnostics': {},
            'warnings': [],
        }

    def log_phase(self, name, **kwargs):
        """Log een optimalisatiefase.

        Args:
            name: fase-naam ("grid_search", "scipy", "ma_optimize", ...)
            **kwargs: willekeurige key-value paren (ratio, time_s, n_evals, ...)
        """
        phase = {'name': name, 'wall_time': time.time() - self.start_time}
        phase.update(kwargs)
        self.data['phases'].append(phase)

    def set_result(self, ratio, gammas=None, betas=None, cut_value=None,
                   n_edges=None, **kwargs):
        """Sla het eindresultaat op.

        Args:
            ratio: behaalde approximatie-ratio
            gammas: optimale gamma-parameters
            betas: optimale beta-parameters
            cut_value: absolute cut-waarde
            n_edges: totaal aantal edges
            **kwargs: extra resultaat-info
        """
        self.data['result'] = {
            'ratio': ratio,
            'gammas': list(gammas) if gammas is not None else None,
            'betas': list(betas) if betas is not None else None,
            'cut_value': cut_value,
            'n_edges': n_edges,
            'total_time_s': time.time() - self.start_time,
        }
        self.data['result'].update(kwargs)

    def set_ma_result(self, ratio, gamma_per_class, betas, class_names=None,
                      **kwargs):
        """Sla ma-QAOA resultaat op met per-klasse gamma's.

        Args:
            ratio: behaalde ratio
            gamma_per_class: list van dicts [{class_id: gamma}, ...]
            betas: list van beta-waarden
            class_names: optioneel dict {class_id: naam}
        """
        # Converteer class_id tuples naar strings voor JSON
        gpc_json = []
        for layer_dict in gamma_per_class:
            gpc_json.append({str(k): v for k, v in layer_dict.items()})

        cn_json = None
        if class_names:
            cn_json = {str(k): v for k, v in class_names.items()}

        self.data['result'] = {
            'ratio': ratio,
            'gamma_per_class': gpc_json,
            'betas': list(betas),
            'class_names': cn_json,
            'method': 'ma-qaoa',
            'total_time_s': time.time() - self.start_time,
        }
        self.data['result'].update(kwargs)

    def set_bounds(self, lower=None, upper=None, bks=None, greedy=None,
                   gw_sdp=None):
        """Sla grenzen op voor context.

        Args:
            lower: ondergrens (bijv. greedy cut)
            upper: bovengrens (bijv. BKS of brute force)
            bks: best known solution
            greedy: greedy heuristiek resultaat
            gw_sdp: Goemans-Williamson SDP relaxatie
        """
        bounds = {}
        if lower is not None:
            bounds['lower'] = lower
        if upper is not None:
            bounds['upper'] = upper
        if bks is not None:
            bounds['bks'] = bks
        if greedy is not None:
            bounds['greedy'] = greedy
        if gw_sdp is not None:
            bounds['gw_sdp'] = gw_sdp
        self.data['bounds'] = bounds

    def set_diagnostics(self, **kwargs):
        """Sla diagnostische informatie op.

        Typische keys: max_lightcone_qubits, avg_chi, max_truncation_error,
        n_edges_computed, n_edges_cached, cache_hit_rate
        """
        self.data['diagnostics'].update(kwargs)

    def add_warning(self, message):
        """Voeg een waarschuwing toe (bijv. 'dirty git', 'low ratio')."""
        self.data['warnings'].append({
            'message': message,
            'wall_time': time.time() - self.start_time,
        })

    def finalize(self):
        """Rond de audit af met totaaltijd en checks."""
        self.data['total_time_s'] = time.time() - self.start_time

        # Auto-warnings
        env = self.data['environment']
        if env['git']['dirty']:
            self.add_warning("Git working directory is dirty (uncommitted changes)")
        if self.data['result'] is None:
            self.add_warning("Geen resultaat gezet — run mogelijk mislukt")
        elif self.data['result'].get('ratio', 0) < 0.5:
            self.add_warning("Ratio < 0.5 — mogelijk convergentieprobleem")

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def to_dict(self):
        """Retourneer volledige audit als dict."""
        self.finalize()
        return self.data

    def save(self, filepath):
        """Schrijf audit als JSON.

        Args:
            filepath: pad naar output JSON bestand
        """
        self.finalize()
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False,
                      default=str)
        return filepath

    def save_html(self, filepath):
        """Schrijf 1-pagina HTML summary.

        Args:
            filepath: pad naar output HTML bestand
        """
        self.finalize()
        html = _render_html(self.data)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return filepath


# =====================================================================
# HTML renderer
# =====================================================================

def _render_html(data):
    """Genereer compact 1-pagina HTML summary."""
    result = data.get('result') or {}
    env = data.get('environment', {})
    gpu = env.get('gpu', {})
    git = env.get('git', {})
    phases = data.get('phases', [])
    bounds = data.get('bounds', {})
    diag = data.get('diagnostics', {})
    warnings = data.get('warnings', [])

    # Ratio kleur
    ratio = result.get('ratio', 0)
    if ratio >= 0.8:
        ratio_color = '#2d8a4e'
    elif ratio >= 0.7:
        ratio_color = '#b8860b'
    else:
        ratio_color = '#cc3333'

    # Phases tabel
    phase_rows = ""
    for ph in phases:
        name = ph.get('name', '?')
        r = ph.get('ratio', '')
        t = ph.get('time_s', ph.get('wall_time', ''))
        evals = ph.get('n_evals', '')
        r_str = "%.6f" % r if isinstance(r, float) else str(r)
        t_str = "%.1fs" % t if isinstance(t, (int, float)) else str(t)
        phase_rows += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (
            name, r_str, t_str, evals)

    # Parameters
    params_html = ""
    if result.get('gammas'):
        params_html += "<p><b>Gammas:</b> %s</p>\n" % (
            ", ".join("%.4f" % g for g in result['gammas']))
    if result.get('betas'):
        params_html += "<p><b>Betas:</b> %s</p>\n" % (
            ", ".join("%.4f" % b for b in result['betas']))
    if result.get('gamma_per_class'):
        params_html += "<p><b>Gamma per klasse:</b></p>\n<ul>\n"
        cnames = result.get('class_names', {})
        for i, layer in enumerate(result['gamma_per_class']):
            parts = []
            for cid, g in layer.items():
                label = cnames.get(cid, cid)
                parts.append("%s=%.4f" % (label, g))
            params_html += "<li>Laag %d: %s</li>\n" % (i + 1, ", ".join(parts))
        params_html += "</ul>\n"

    # Bounds
    bounds_html = ""
    if bounds:
        parts = []
        for k, v in bounds.items():
            if isinstance(v, float):
                parts.append("%s=%.4f" % (k, v))
            else:
                parts.append("%s=%s" % (k, v))
        bounds_html = "<p><b>Bounds:</b> %s</p>" % ", ".join(parts)

    # Diagnostics
    diag_html = ""
    if diag:
        parts = []
        for k, v in diag.items():
            if isinstance(v, float):
                parts.append("%s=%.4f" % (k, v))
            else:
                parts.append("%s=%s" % (k, v))
        diag_html = "<p><b>Diagnostics:</b> %s</p>" % ", ".join(parts)

    # Warnings
    warn_html = ""
    if warnings:
        warn_html = '<div class="warn">\n'
        for w in warnings:
            warn_html += "<p>%s</p>\n" % w['message']
        warn_html += "</div>\n"

    dirty_badge = ' <span class="dirty">DIRTY</span>' if git.get('dirty') else ''

    html = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<title>ZornQ Audit — %(run_id)s</title>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 800px;
         margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
  h1 { font-size: 1.4em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { font-size: 1.1em; color: #555; margin-top: 1.5em; }
  .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 0.3em 2em;
          background: #f5f5f5; padding: 0.8em 1em; border-radius: 6px; font-size: 0.9em; }
  .meta dt { font-weight: 600; }
  .meta dd { margin: 0; }
  .ratio { font-size: 2em; font-weight: 700; color: %(ratio_color)s; }
  table { border-collapse: collapse; width: 100%%; margin: 0.5em 0; }
  th, td { text-align: left; padding: 0.4em 0.8em; border-bottom: 1px solid #ddd; }
  th { background: #f0f0f0; font-weight: 600; }
  .warn { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
          padding: 0.5em 1em; margin: 1em 0; }
  .dirty { background: #dc3545; color: white; padding: 0.1em 0.4em;
           border-radius: 3px; font-size: 0.8em; }
  .footer { color: #888; font-size: 0.8em; margin-top: 2em; border-top: 1px solid #ddd;
            padding-top: 0.5em; }
</style>
</head>
<body>
<h1>ZornQ Audit Trail</h1>

<dl class="meta">
  <dt>Run ID</dt><dd><code>%(run_id)s</code></dd>
  <dt>Timestamp</dt><dd>%(timestamp)s</dd>
  <dt>Graaf</dt><dd>%(graph)s</dd>
  <dt>p</dt><dd>%(p)d</dd>
  <dt>Methode</dt><dd>%(method)s</dd>
  <dt>Seed</dt><dd>%(seed)s</dd>
  <dt>Git</dt><dd><code>%(git_commit)s</code> (%(git_branch)s)%(dirty)s</dd>
  <dt>GPU</dt><dd>%(gpu_name)s (%(gpu_vram)d MB)</dd>
  <dt>Host</dt><dd>%(hostname)s</dd>
  <dt>RAM</dt><dd>%(ram_mb)d MB</dd>
</dl>

%(warn_html)s

<h2>Resultaat</h2>
<p class="ratio">%(ratio_str)s</p>
<p>Totale tijd: %(total_time)s</p>
%(params_html)s
%(bounds_html)s

<h2>Optimalisatie-fasen</h2>
<table>
<tr><th>Fase</th><th>Ratio</th><th>Tijd</th><th>Evaluaties</th></tr>
%(phase_rows)s
</table>

%(diag_html)s

<div class="footer">
  Gegenereerd door audit_trail.py v%(version)s
</div>
</body>
</html>""" % {
        'run_id': data.get('run_id', '?'),
        'timestamp': data.get('timestamp', '?'),
        'graph': data.get('graph', '?'),
        'p': data.get('p', 0),
        'method': data.get('method', '?'),
        'seed': data.get('seed', 'none'),
        'git_commit': git.get('commit', '?'),
        'git_branch': git.get('branch', '?'),
        'dirty': dirty_badge,
        'gpu_name': gpu.get('name', 'none'),
        'gpu_vram': gpu.get('vram_mb', 0),
        'hostname': env.get('hostname', '?'),
        'ram_mb': env.get('ram_mb', 0),
        'ratio_color': ratio_color,
        'ratio_str': "%.6f" % ratio if ratio else "N/A",
        'total_time': "%.1fs" % data.get('total_time_s', 0),
        'params_html': params_html,
        'bounds_html': bounds_html,
        'phase_rows': phase_rows,
        'diag_html': diag_html,
        'warn_html': warn_html,
        'version': __version__,
    }
    return html


# =====================================================================
# Vergelijking van twee audit trails
# =====================================================================

def compare_audits(path_a, path_b):
    """Vergelijk twee audit-JSONs en print verschil-rapport.

    Args:
        path_a: pad naar eerste JSON
        path_b: pad naar tweede JSON

    Returns:
        dict met vergelijkingsresultaten
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    ra = (a.get('result') or {}).get('ratio', 0)
    rb = (b.get('result') or {}).get('ratio', 0)
    ta = a.get('total_time_s', 0)
    tb = b.get('total_time_s', 0)

    diff = {
        'ratio_a': ra,
        'ratio_b': rb,
        'ratio_delta': rb - ra,
        'ratio_pct': (rb - ra) / max(ra, 1e-10) * 100,
        'time_a': ta,
        'time_b': tb,
        'time_ratio': tb / max(ta, 1e-10),
        'same_graph': a.get('graph') == b.get('graph'),
        'same_p': a.get('p') == b.get('p'),
        'method_a': a.get('method'),
        'method_b': b.get('method'),
    }

    return diff


def print_comparison(path_a, path_b):
    """Print vergelijkingsrapport naar stdout."""
    d = compare_audits(path_a, path_b)
    sep = "=" * 56
    print(sep)
    print("  Audit Trail Vergelijking")
    print(sep)
    print()
    print("  %-20s  %-18s  %-18s" % ("", "Run A", "Run B"))
    print("  " + "-" * 52)
    print("  %-20s  %-18s  %-18s" % ("Methode", d['method_a'], d['method_b']))
    print("  %-20s  %-18.6f  %-18.6f" % ("Ratio", d['ratio_a'], d['ratio_b']))
    print("  %-20s  %-18.1f  %-18.1f" % ("Tijd (s)", d['time_a'], d['time_b']))
    print()
    print("  Ratio verschil:  %+.6f (%+.2f%%)" % (d['ratio_delta'], d['ratio_pct']))
    print("  Snelheidsratio:  %.2fx" % d['time_ratio'])
    if not d['same_graph']:
        print("  LET OP: Verschillende grafen!")
    if not d['same_p']:
        print("  LET OP: Verschillende p-waarden!")
    print(sep)


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='B56: Resultaat-Export met Audit Trail')
    parser.add_argument('--show', metavar='FILE',
                        help='Toon audit trail JSON in leesbaar formaat')
    parser.add_argument('--compare', nargs=2, metavar='FILE',
                        help='Vergelijk twee audit trail JSONs')
    parser.add_argument('--to-html', nargs=2, metavar=('JSON', 'HTML'),
                        help='Converteer JSON audit naar HTML summary')
    parser.add_argument('--demo', action='store_true',
                        help='Genereer demo audit trail')
    args = parser.parse_args()

    if args.show:
        with open(args.show) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    if args.to_html:
        with open(args.to_html[0]) as f:
            data = json.load(f)
        html = _render_html(data)
        with open(args.to_html[1], 'w', encoding='utf-8') as f:
            f.write(html)
        print("HTML geschreven naar %s" % args.to_html[1])
        return

    if args.demo:
        audit = AuditTrail("grid_6x3", p=2, seed=42, method="lightcone")
        audit.log_phase("grid_search", ratio=0.6697, time_s=0.3, n_evals=100)
        audit.log_phase("scipy_p1", ratio=0.6884, time_s=0.1, n_evals=18)
        audit.log_phase("warmstart_p2", ratio=0.7058, time_s=0.0, n_evals=1)
        audit.log_phase("scipy_p2", ratio=0.7907, time_s=2.9, n_evals=361)
        audit.set_result(ratio=0.7907, gammas=[0.314, 0.314], betas=[1.178, 1.178],
                         cut_value=21.35, n_edges=27)
        audit.set_bounds(bks=27, greedy=22)
        audit.set_diagnostics(
            max_lightcone_qubits=12,
            n_edges_computed=27,
            n_edges_cached=15,
            cache_hit_rate=0.556,
        )

        json_path = "demo_audit.json"
        html_path = "demo_audit.html"
        audit.save(json_path)
        audit.save_html(html_path)
        print("Demo audit geschreven naar:")
        print("  JSON: %s" % json_path)
        print("  HTML: %s" % html_path)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
