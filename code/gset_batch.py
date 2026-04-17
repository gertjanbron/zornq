#!/usr/bin/env python3
"""
gset_batch.py — Batch benchmark runner via B48 Auto-Hybride Planner.

Draait alle ingebouwde benchmark-grafen + parametrische generatoren
door ZornSolver en vergelijkt resultaten met BKS (Best Known Solutions).

Output: tabel op stdout + optioneel Markdown-export voor paper.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gset_loader import (
    BUILTIN_GRAPHS, load_graph, make_random_regular as gset_make_rr,
    make_erdos_renyi, make_complete, make_cycle
)
from auto_planner import ZornSolver, SolverResult


def wg_to_edges(wg):
    """Convert WeightedGraph to (n_nodes, edge_list) for ZornSolver."""
    edges = []
    seen = set()
    for u, v, w in wg.edges():
        key = (min(u, v), max(u, v))
        if key not in seen:
            edges.append(key)
            seen.add(key)
    return wg.n_nodes, edges


# =====================================================================
# Benchmark suite — diverse graph families
# =====================================================================

def build_benchmark_suite(seed=42):
    """Build list of (name, n_nodes, edges, bks_or_None)."""
    suite = []

    # 1. All builtin graphs
    for name in BUILTIN_GRAPHS:
        g, bks, info = load_graph(name, seed=seed)
        n, edges = wg_to_edges(g)
        suite.append((name, n, edges, bks))

    # 2. Extra parametric: random regular
    for n_nodes in [14, 16, 20]:
        gname = f'reg3_{n_nodes}'
        g, bks, info = load_graph(gname, seed=seed)
        n, edges = wg_to_edges(g)
        suite.append((gname, n, edges, bks))

    # 3. Erdos-Renyi
    for n_nodes in [12, 16]:
        g, bks = make_erdos_renyi(n_nodes, p=0.3, seed=seed)
        n, edges = wg_to_edges(g)
        suite.append((f'ER_{n_nodes}_0.3', n, edges, bks))

    # 4. Complete graphs (dense, stress test for classifier)
    for n_nodes in [6, 8, 10]:
        g, bks = make_complete(n_nodes)
        n, edges = wg_to_edges(g)
        suite.append((f'K{n_nodes}', n, edges, bks))

    # 5. Odd cycles (frustrated)
    for n_nodes in [7, 13]:
        g, bks = make_cycle(n_nodes)
        n, edges = wg_to_edges(g)
        suite.append((f'C{n_nodes}', n, edges, bks))

    # Deduplicate by name (builtins may overlap with extras)
    seen = set()
    deduped = []
    for item in suite:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    return deduped


# =====================================================================
# Main batch run
# =====================================================================

def run_batch(p_values=(1, 2), chi_budget=32, verbose_solver=False):
    """Run full batch and return results list."""
    suite = build_benchmark_suite()
    solver = ZornSolver(chi_budget=chi_budget, gpu=False,
                        mixed_precision=False, verbose=verbose_solver)

    results = []
    for name, n, edges, bks in suite:
        m = len(edges)
        row = {'name': name, 'n': n, 'm': m, 'bks': bks}

        for p in p_values:
            t0 = time.time()
            try:
                res = solver.solve(n, edges, p=p)
                row[f'cut_p{p}'] = res.cut_value
                row[f'ratio_p{p}'] = res.ratio
                row[f'method_p{p}'] = res.method
                row[f'time_p{p}'] = res.wall_time
                if bks is not None and bks > 0:
                    row[f'approx_p{p}'] = res.cut_value / bks
                else:
                    row[f'approx_p{p}'] = None
                row[f'notes_p{p}'] = ', '.join(res.notes) if res.notes else ''
            except Exception as e:
                row[f'cut_p{p}'] = None
                row[f'ratio_p{p}'] = None
                row[f'method_p{p}'] = f'ERROR: {e}'
                row[f'time_p{p}'] = time.time() - t0
                row[f'approx_p{p}'] = None
                row[f'notes_p{p}'] = str(e)

        results.append(row)
    return results


def print_table(results, p_values=(1, 2)):
    """Print results as aligned table."""
    # Header
    cols = ['Graph', 'n', 'm', 'BKS']
    for p in p_values:
        cols += [f'Cut(p={p})', f'%BKS', f'Method', f'Time']

    sep = '=' * 130
    print(sep)
    print("ZornSolver Batch Benchmark — Auto-Hybride Planner (B48)")
    print(sep)

    hdr = f"{'Graph':>16} {'n':>4} {'m':>4} {'BKS':>6}"
    for p in p_values:
        hdr += f" | {'Cut':>6} {'%BKS':>6} {'Method':>16} {'Time':>6}"
    print(hdr)
    print('-' * 130)

    for row in results:
        line = f"{row['name']:>16} {row['n']:>4} {row['m']:>4} "
        if row['bks'] is not None:
            line += f"{row['bks']:>6}"
        else:
            line += f"{'?':>6}"

        for p in p_values:
            cut = row.get(f'cut_p{p}')
            approx = row.get(f'approx_p{p}')
            method = row.get(f'method_p{p}', '?')
            t = row.get(f'time_p{p}', 0)

            if cut is not None:
                line += f" | {cut:>6.0f}"
                if approx is not None:
                    line += f" {approx:>5.1%}"
                else:
                    line += f" {'—':>6}"
                # Shorten method name
                short_m = method.replace('_mpo', '').replace('_exact', '')[:16]
                line += f" {short_m:>16} {t:>5.1f}s"
            else:
                line += f" | {'ERR':>6} {'—':>6} {method[:16]:>16} {t:>5.1f}s"

        print(line)

    print(sep)

    # Summary stats
    for p in p_values:
        approxes = [r[f'approx_p{p}'] for r in results if r.get(f'approx_p{p}') is not None]
        if approxes:
            avg = np.mean(approxes)
            minv = np.min(approxes)
            print(f"  p={p}: avg %BKS = {avg:.1%}, min = {minv:.1%} ({len(approxes)} instances with BKS)")

    times = [r.get('time_p1', 0) for r in results if r.get('time_p1') is not None]
    print(f"  Total wall time (p=1): {sum(times):.1f}s")


def export_markdown(results, p_values=(1, 2), filepath=None):
    """Export results as Markdown table for paper."""
    lines = []
    lines.append("# ZornSolver Batch Benchmark Results")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Results")
    lines.append("")

    # Header
    hdr = "| Graph | n | m | BKS |"
    sep_line = "|---|---|---|---|"
    for p in p_values:
        hdr += f" Cut(p={p}) | %BKS | Method | Time |"
        sep_line += "---|---|---|---|"
    lines.append(hdr)
    lines.append(sep_line)

    for row in results:
        bks_str = str(row['bks']) if row['bks'] is not None else '?'
        line = f"| {row['name']} | {row['n']} | {row['m']} | {bks_str} |"

        for p in p_values:
            cut = row.get(f'cut_p{p}')
            approx = row.get(f'approx_p{p}')
            method = row.get(f'method_p{p}', '?')
            t = row.get(f'time_p{p}', 0)

            if cut is not None:
                a_str = f"{approx:.1%}" if approx is not None else "—"
                line += f" {cut:.0f} | {a_str} | {method} | {t:.1f}s |"
            else:
                line += f" ERR | — | {method[:20]} | {t:.1f}s |"

        lines.append(line)

    lines.append("")
    lines.append("## Summary")
    for p in p_values:
        approxes = [r[f'approx_p{p}'] for r in results if r.get(f'approx_p{p}') is not None]
        if approxes:
            lines.append(f"- p={p}: avg %BKS = {np.mean(approxes):.1%}, "
                        f"min = {np.min(approxes):.1%}, "
                        f"max = {np.max(approxes):.1%} "
                        f"({len(approxes)} instances)")

    content = '\n'.join(lines) + '\n'
    if filepath:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\nMarkdown exported to: {filepath}")
    return content


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='B48 Batch Benchmark')
    parser.add_argument('--chi', type=int, default=32, help='Chi budget')
    parser.add_argument('--p-max', type=int, default=1, help='Max QAOA depth')
    parser.add_argument('--export', type=str, default=None, help='Export Markdown to file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose solver output')
    args = parser.parse_args()

    p_values = list(range(1, args.p_max + 1))
    results = run_batch(p_values=p_values, chi_budget=args.chi,
                        verbose_solver=args.verbose)
    print_table(results, p_values=p_values)

    if args.export:
        export_markdown(results, p_values=p_values, filepath=args.export)
