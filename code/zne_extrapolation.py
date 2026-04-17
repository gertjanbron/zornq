#!/usr/bin/env python3
"""
zne_extrapolation.py - B25: Zero-Noise Extrapolation voor QAOA MaxCut.

Idee: truncatie-ruis in MPS is analoog aan hardware-ruis in NISQ.
Draai QAOA bij meerdere chi-waarden, fit een polynoom, extrapoleer
naar chi=oneindig. Voorspel het exacte antwoord zonder de zware
berekening uit te voeren.

Gebruik:
  python zne_extrapolation.py                          # standaard 5x4 cilinder
  python zne_extrapolation.py --Lx 10 --Ly 4          # 10x4 cilinder
  python zne_extrapolation.py --Lx 20 --Ly 3 --gpu    # 20x3 op GPU
  python zne_extrapolation.py --Lx 8 --Ly 4 --gpu --p 2  # p=2, hogere diepte

Output:
  - Tabel met chi -> ratio
  - Lineaire + kwadratische fit
  - Richardson-extrapolatie
  - Betrouwbaarheidsinterval
  - Optioneel: plot als HTML
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zorn_mps import HeisenbergQAOA


def zne_extrapolate(Lx, Ly=4, p=1, gammas=None, betas=None,
                    chi_values=None, gpu=False, min_weight=None):
    """Draai QAOA bij meerdere chi en extrapoleer naar chi=inf.

    Returns:
        dict met resultaten per chi + extrapolaties
    """
    if chi_values is None:
        chi_values = [4, 8, 16, 32, 64]

    # Standaard QAOA parameters (bewezen optimaal voor 2D p=1)
    if gammas is None:
        if Ly > 1:
            avg_degree = 2 * (1 - 1/Lx) * (1 - 1/Ly) / (1 - 1/(Lx*Ly)) if Lx > 1 else 1
            # Vereenvoudigd: gebruik bekende goede waarden
            gammas = [0.40] if p == 1 else [0.35, 0.25][:p]
        else:
            gammas = [0.40] * p
    if betas is None:
        betas = [1.18] * p

    n_qubits = Lx * Ly
    n_edges = (Lx * (Ly - 1) + (Lx - 1) * Ly) if Ly > 1 else (Lx - 1)

    print("=" * 60)
    print("  B25: ZNE Chi-Extrapolatie")
    print("=" * 60)
    print("  Systeem: %dx%d = %d qubits, %d edges" % (Lx, Ly, n_qubits, n_edges))
    print("  p=%d, gamma=%s, beta=%s" % (p, gammas, betas))
    print("  GPU: %s" % ("ja" if gpu else "nee"))
    print("  Chi-waarden: %s" % chi_values)
    print()

    # Draai QAOA voor elke chi
    results = []
    for chi in chi_values:
        print("  chi=%d ..." % chi, end=" ", flush=True)
        t0 = time.time()
        try:
            qaoa = HeisenbergQAOA(Lx, Ly, max_chi=chi, gpu=gpu, min_weight=min_weight)
            ratio = qaoa.eval_ratio(p, gammas, betas)
            cost = ratio * n_edges
            elapsed = time.time() - t0
            print("ratio=%.6f, cost=%.3f, %.2fs" % (ratio, cost, elapsed))
            results.append({
                'chi': chi,
                'ratio': ratio,
                'cost': cost,
                'time': elapsed,
                'success': True,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print("FOUT: %s (%.2fs)" % (str(e)[:50], elapsed))
            results.append({
                'chi': chi,
                'ratio': None,
                'cost': None,
                'time': elapsed,
                'success': False,
            })

    # Filter succesvolle runs
    good = [r for r in results if r['success']]
    if len(good) < 2:
        print("\n  Te weinig datapunten voor extrapolatie")
        return {'results': results}

    chis = np.array([r['chi'] for r in good])
    ratios = np.array([r['ratio'] for r in good])
    costs = np.array([r['cost'] for r in good])

    # X-as: 1/chi (truncatie-ruis neemt af met chi)
    x = 1.0 / chis

    print("\n  Resultaten:")
    print("  %6s  %10s  %10s  %8s" % ("chi", "ratio", "cost", "tijd"))
    print("  " + "-" * 40)
    for r in results:
        if r['success']:
            print("  %6d  %10.6f  %10.3f  %7.2fs" % (
                r['chi'], r['ratio'], r['cost'], r['time']))
        else:
            print("  %6d  %10s  %10s  %7.2fs" % (r['chi'], "FOUT", "FOUT", r['time']))

    # =========================================
    # EXTRAPOLATIES
    # =========================================
    print("\n  Extrapolaties naar chi -> oneindig (1/chi -> 0):")

    extrapolations = {}

    # 1. Lineaire fit: ratio = a + b/chi
    if len(good) >= 2:
        coeffs_lin = np.polyfit(x, ratios, 1)
        ratio_lin = coeffs_lin[-1]  # waarde bij x=0
        print("  Lineair:      ratio(inf) = %.6f" % ratio_lin)
        extrapolations['linear'] = {
            'ratio_inf': float(ratio_lin),
            'coeffs': coeffs_lin.tolist(),
        }

    # 2. Kwadratische fit: ratio = a + b/chi + c/chi^2
    if len(good) >= 3:
        coeffs_quad = np.polyfit(x, ratios, 2)
        ratio_quad = coeffs_quad[-1]
        print("  Kwadratisch:  ratio(inf) = %.6f" % ratio_quad)
        extrapolations['quadratic'] = {
            'ratio_inf': float(ratio_quad),
            'coeffs': coeffs_quad.tolist(),
        }

    # 3. Richardson-extrapolatie (paren van opeenvolgende chi's)
    if len(good) >= 2:
        # Neem de twee hoogste chi-waarden
        idx_sorted = np.argsort(chis)
        i1, i2 = idx_sorted[-2], idx_sorted[-1]
        chi1, chi2 = chis[i1], chis[i2]
        r1, r2 = ratios[i1], ratios[i2]
        # Aanname: fout ~ 1/chi, dus Richardson = (chi2*r2 - chi1*r1) / (chi2 - chi1)
        if chi2 != chi1:
            ratio_rich = (chi2 * r2 - chi1 * r1) / (chi2 - chi1)
            print("  Richardson:   ratio(inf) = %.6f  (chi=%d,%d)" % (
                ratio_rich, chi1, chi2))
            extrapolations['richardson'] = {
                'ratio_inf': float(ratio_rich),
                'chi_pair': [int(chi1), int(chi2)],
            }

    # 4. Betrouwbaarheidsinterval
    ext_values = [v['ratio_inf'] for v in extrapolations.values()]
    if len(ext_values) >= 2:
        mean_ext = np.mean(ext_values)
        spread = np.max(ext_values) - np.min(ext_values)
        print("\n  Gemiddelde extrapolatie: %.6f +/- %.6f" % (mean_ext, spread / 2))
        print("  Spread tussen methoden: %.6f (%.2f%%)" % (
            spread, spread / max(abs(mean_ext), 1e-10) * 100))
    elif len(ext_values) == 1:
        mean_ext = ext_values[0]
        spread = 0
        print("\n  Extrapolatie (alleen lineair): %.6f" % mean_ext)

    # 5. Convergentie-analyse
    if len(good) >= 3:
        diffs = np.diff(ratios[np.argsort(chis)])
        print("\n  Convergentie (delta ratio bij chi-verhoging):")
        sorted_chis = chis[np.argsort(chis)]
        for i in range(len(diffs)):
            print("    chi %d->%d: delta=%.6f" % (
                sorted_chis[i], sorted_chis[i+1], diffs[i]))

    # 6. Diagnostiek: hoe goed is chi=32?
    chi32_data = [r for r in good if r['chi'] == 32]
    if chi32_data and len(ext_values) > 0:
        chi32_ratio = chi32_data[0]['ratio']
        best_est = np.mean(ext_values)
        gap = best_est - chi32_ratio
        print("\n  Diagnostiek chi=32:")
        print("    chi=32 ratio:       %.6f" % chi32_ratio)
        print("    Geschatte exact:    %.6f" % best_est)
        print("    Gap:                %.6f (%.2f%%)" % (gap, gap / max(abs(best_est), 1e-10) * 100))

    # Genereer HTML plot
    html_path = _generate_plot(Lx, Ly, p, results, extrapolations, n_edges)

    print("\n" + "=" * 60)

    return {
        'results': results,
        'extrapolations': extrapolations,
        'html': html_path,
    }


def _generate_plot(Lx, Ly, p, results, extrapolations, n_edges):
    """Genereer een HTML-plot van de ZNE-resultaten."""
    good = [r for r in results if r['success']]
    if len(good) < 2:
        return None

    chis = [r['chi'] for r in good]
    ratios = [r['ratio'] for r in good]
    x_data = [1.0 / c for c in chis]

    # SVG plot
    w, h = 500, 350
    margin = {'top': 40, 'right': 30, 'bottom': 50, 'left': 70}
    pw = w - margin['left'] - margin['right']
    ph = h - margin['top'] - margin['bottom']

    x_min, x_max = 0, max(x_data) * 1.15
    y_min = min(ratios) - 0.01
    y_max = max(ratios) + 0.02
    if 'linear' in extrapolations:
        y_max = max(y_max, extrapolations['linear']['ratio_inf'] + 0.01)

    def sx(v):
        return margin['left'] + (v - x_min) / (x_max - x_min) * pw
    def sy(v):
        return margin['top'] + (1 - (v - y_min) / (y_max - y_min)) * ph

    svg = []
    svg.append('<svg width="%d" height="%d" xmlns="http://www.w3.org/2000/svg">' % (w, h))
    svg.append('<rect width="%d" height="%d" fill="#0a0a0f"/>' % (w, h))

    # Grid
    for i in range(5):
        yv = y_min + i * (y_max - y_min) / 4
        y_px = sy(yv)
        svg.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#1a1a2a" stroke-width="0.5"/>' % (
            margin['left'], y_px, w - margin['right'], y_px))
        svg.append('<text x="%d" y="%.1f" fill="#888" font-size="10" text-anchor="end">%.4f</text>' % (
            margin['left'] - 5, y_px + 3, yv))

    for c in chis:
        xv = 1.0 / c
        x_px = sx(xv)
        svg.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#1a1a2a" stroke-width="0.5"/>' % (
            x_px, margin['top'], x_px, h - margin['bottom']))
        svg.append('<text x="%.1f" y="%d" fill="#888" font-size="10" text-anchor="middle">1/%d</text>' % (
            x_px, h - margin['bottom'] + 15, c))

    # Fit lijnen
    if 'linear' in extrapolations:
        coeffs = extrapolations['linear']['coeffs']
        x0, x1_pt = 0, max(x_data) * 1.1
        y0 = coeffs[0] * x0 + coeffs[1]
        y1_v = coeffs[0] * x1_pt + coeffs[1]
        svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#44aa66" stroke-width="1.5" stroke-dasharray="6,3"/>' % (
            sx(x0), sy(y0), sx(x1_pt), sy(y1_v)))

    if 'quadratic' in extrapolations:
        coeffs = extrapolations['quadratic']['coeffs']
        pts = []
        for i in range(50):
            xv = i * max(x_data) * 1.1 / 49
            yv = coeffs[0] * xv**2 + coeffs[1] * xv + coeffs[2]
            pts.append("%.1f,%.1f" % (sx(xv), sy(yv)))
        svg.append('<polyline points="%s" fill="none" stroke="#aa8844" stroke-width="1.5" stroke-dasharray="3,3"/>' % " ".join(pts))

    # Extrapolatie punten op y-as (x=0)
    colors = {'linear': '#44aa66', 'quadratic': '#aa8844', 'richardson': '#aa4466'}
    labels = {'linear': 'Lin', 'quadratic': 'Quad', 'richardson': 'Rich'}
    y_offset = 0
    for key, ext in extrapolations.items():
        yv = ext['ratio_inf']
        svg.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="white" stroke-width="1"/>' % (
            sx(0), sy(yv), colors.get(key, '#888')))
        svg.append('<text x="%.1f" y="%.1f" fill="%s" font-size="9">%s=%.4f</text>' % (
            sx(0) + 8, sy(yv) + 3 + y_offset, colors.get(key, '#888'),
            labels.get(key, key), yv))
        y_offset += 12

    # Datapunten
    for xv, yv in zip(x_data, ratios):
        svg.append('<circle cx="%.1f" cy="%.1f" r="5" fill="#7eb8ff" stroke="white" stroke-width="1"/>' % (
            sx(xv), sy(yv)))

    # Labels
    svg.append('<text x="%d" y="20" fill="#7eb8ff" font-size="14" font-weight="bold">ZNE Chi-Extrapolatie %dx%d p=%d</text>' % (
        margin['left'], Lx, Ly, p))
    svg.append('<text x="%d" y="%d" fill="#888" font-size="11" text-anchor="middle">1/chi (truncatie-ruis)</text>' % (
        w // 2, h - 5))
    svg.append('<text x="15" y="%d" fill="#888" font-size="11" transform="rotate(-90,15,%d)">QAOA ratio</text>' % (
        h // 2, h // 2))

    svg.append('</svg>')

    # Bouw HTML
    stats_html = ""
    for r in results:
        if r['success']:
            stats_html += "<tr><td>%d</td><td>%.6f</td><td>%.3f</td><td>%.2fs</td></tr>\n" % (
                r['chi'], r['ratio'], r['cost'], r['time'])

    ext_html = ""
    for key, ext in extrapolations.items():
        ext_html += "<tr><td>%s</td><td>%.6f</td></tr>\n" % (
            labels.get(key, key), ext['ratio_inf'])

    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>ZNE Chi-Extrapolatie %dx%d</title>
<style>
body { background: #0a0a0f; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
       display: flex; flex-direction: column; align-items: center; padding: 20px; }
h1 { color: #7eb8ff; font-size: 18px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
td, th { padding: 4px 12px; border: 1px solid #2a2a3a; }
th { background: #12121a; color: #7eb8ff; }
.box { background: #12121a; border-radius: 8px; padding: 16px; margin: 10px 0;
       border: 1px solid #2a2a3a; max-width: 520px; width: 100%%; }
.green { color: #44aa66; } .yellow { color: #aa8844; } .red { color: #aa4466; }
</style></head><body>
<h1>B25: ZNE Chi-Extrapolatie - %dx%d cilinder (p=%d)</h1>
%s
<div class="box">
<h3 style="color:#7eb8ff;margin:0 0 8px">Meetpunten</h3>
<table><tr><th>chi</th><th>ratio</th><th>cost</th><th>tijd</th></tr>
%s</table></div>
<div class="box">
<h3 style="color:#7eb8ff;margin:0 0 8px">Extrapolaties (chi -> inf)</h3>
<table><tr><th>Methode</th><th>ratio(inf)</th></tr>
%s</table></div>
</body></html>""" % (Lx, Ly, Lx, Ly, p, '\n'.join(svg), stats_html, ext_html)

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'zne_result_%dx%d_p%d.html' % (Lx, Ly, p))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print("\n  Plot opgeslagen: %s" % filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description='B25: ZNE Chi-Extrapolation')
    parser.add_argument('--Lx', type=int, default=5)
    parser.add_argument('--Ly', type=int, default=4)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--chi', type=str, default='4,8,16,32,64',
                       help='Komma-gescheiden chi waarden')
    parser.add_argument('--gamma', type=float, nargs='+', default=None)
    parser.add_argument('--beta', type=float, nargs='+', default=None)
    args = parser.parse_args()

    chi_values = [int(c) for c in args.chi.split(',')]

    zne_extrapolate(
        Lx=args.Lx, Ly=args.Ly, p=args.p,
        gammas=args.gamma, betas=args.beta,
        chi_values=chi_values, gpu=args.gpu,
    )


if __name__ == '__main__':
    main()
