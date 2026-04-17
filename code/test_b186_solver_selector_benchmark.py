#!/usr/bin/env python3
"""Unit-tests voor b186_solver_selector_benchmark."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b60_gw_bound import SimpleGraph
from b186_solver_selector_benchmark import (
    SolverResult,
    run_ilp,
    run_fw_sdp,
    run_cograph_dp,
    run_dispatcher,
    run_panel,
    emit_latex_table,
    emit_markdown_table,
    emit_csv,
    save_artifacts,
    _graph_edges,
    _fmt,
)


# ============================================================
# Small helper fixtures
# ============================================================

def _triangle() -> SimpleGraph:
    g = SimpleGraph(3)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(0, 2, 1.0)
    return g


def _cycle_4() -> SimpleGraph:
    g = SimpleGraph(4)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(2, 3, 1.0)
    g.add_edge(3, 0, 1.0)
    return g


def _k4() -> SimpleGraph:
    """Complete graph K_4 — ook een cograph."""
    g = SimpleGraph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j, 1.0)
    return g


# ============================================================
# Primitive runners
# ============================================================

class TestGraphHelpers(unittest.TestCase):
    def test_graph_edges_format(self):
        g = _triangle()
        edges = _graph_edges(g)
        self.assertEqual(len(edges), 3)
        for e in edges:
            self.assertEqual(len(e), 3)
            self.assertIsInstance(e[2], float)

    def test_fmt_none(self):
        self.assertEqual(_fmt(None), "--")
        self.assertEqual(_fmt(None, dash="N/A"), "N/A")

    def test_fmt_float(self):
        self.assertEqual(_fmt(3.14, ".2f"), "3.14")
        self.assertEqual(_fmt(0, ".1f"), "0.0")


class TestRunIlp(unittest.TestCase):
    def test_triangle_opt_is_2(self):
        r = run_ilp(_triangle(), time_limit=5.0)
        self.assertEqual(r.solver, "ilp")
        self.assertAlmostEqual(r.cut_value, 2.0, places=6)
        self.assertTrue(r.extra.get("certified"))
        self.assertEqual(r.level, "EXACT")

    def test_cycle4_opt_is_4(self):
        r = run_ilp(_cycle_4(), time_limit=5.0)
        self.assertAlmostEqual(r.cut_value, 4.0, places=6)
        self.assertTrue(r.extra.get("certified"))

    def test_ilp_has_bounds(self):
        r = run_ilp(_cycle_4(), time_limit=5.0)
        self.assertIsNotNone(r.upper_bound)
        self.assertIsNotNone(r.lower_bound)
        self.assertGreaterEqual(r.wall_time, 0.0)


class TestRunFwSdp(unittest.TestCase):
    def test_triangle_sandwich_sound(self):
        r = run_fw_sdp(_triangle(), max_iter=200)
        self.assertEqual(r.solver, "fw_sdp")
        # Sandwich-invariant: UB moet LB dominante (beide bounds op cut_SDP).
        self.assertGreaterEqual(r.upper_bound + 1e-6, r.lower_bound)
        # UB is geldige bovengrens op MaxCut integer OPT (=2 voor triangle);
        # mag nooit onder OPT liggen.
        self.assertGreaterEqual(r.upper_bound, 2.0 - 1e-3)
        self.assertIn(r.level, {"EXACT", "NEAR_EXACT", "BOUNDED", "APPROXIMATE"})

    def test_fw_produces_numeric_bounds(self):
        r = run_fw_sdp(_cycle_4(), max_iter=200)
        self.assertIsNotNone(r.upper_bound)
        self.assertIsNotNone(r.lower_bound)
        self.assertGreaterEqual(r.extra["iterations"], 1)
        # UB is geldige bovengrens op MaxCut integer OPT (=4 voor C_4).
        self.assertGreaterEqual(r.upper_bound, 4.0 - 1e-3)


class TestRunCographDp(unittest.TestCase):
    def test_k4_is_cograph_exact(self):
        """K_4 is een cograph → exact in O(n^3)."""
        r = run_cograph_dp(_k4())
        self.assertFalse(r.skipped)
        self.assertAlmostEqual(r.cut_value, 4.0, places=6)  # K_4 maxcut = 4
        self.assertEqual(r.level, "EXACT")
        self.assertAlmostEqual(r.gap_pct, 0.0, places=6)

    def test_c4_is_cograph(self):
        """C_4 = complement van 2*K_2 → cograph, maxcut = 4."""
        r = run_cograph_dp(_cycle_4())
        self.assertFalse(r.skipped)
        self.assertAlmostEqual(r.cut_value, 4.0, places=6)

    def test_triangle_is_cograph(self):
        """K_3 is een cograph."""
        r = run_cograph_dp(_triangle())
        self.assertFalse(r.skipped)
        self.assertAlmostEqual(r.cut_value, 2.0, places=6)

    def test_p4_not_cograph_skipped(self):
        """Path P_4 bevat P_4 dus geen cograph → skip met reden."""
        g = SimpleGraph(4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 3, 1.0)
        r = run_cograph_dp(g)
        self.assertTrue(r.skipped)
        self.assertIn("cograph", r.skip_reason.lower())


class TestRunDispatcher(unittest.TestCase):
    def test_triangle_dispatch(self):
        r = run_dispatcher(_triangle(), time_budget=3.0)
        self.assertEqual(r.solver, "dispatcher_auto")
        self.assertIsNotNone(r.cut_value)
        self.assertAlmostEqual(r.cut_value, 2.0, places=6)
        self.assertIn("strategy", r.extra)

    def test_k4_dispatch_is_exact(self):
        r = run_dispatcher(_k4(), time_budget=3.0)
        self.assertAlmostEqual(r.cut_value, 4.0, places=6)


# ============================================================
# Panel + artifact emitters
# ============================================================

class TestRunPanelStub(unittest.TestCase):
    def test_small_panel_two_instances(self):
        """Mini-panel: 2 instanties zodat het test-suite niet 10 minuten duurt."""
        panel = [
            ("TEST", "triangle", _triangle(), None),
            ("TEST", "c4", _cycle_4(), None),
        ]
        rows = run_panel(panel, ilp_time=3.0, fw_iters=100,
                         dispatcher_budget=1.0)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("dataset", r)
            self.assertIn("name", r)
            self.assertIn("n", r)
            self.assertIn("m", r)
            self.assertIn("solvers", r)
            for key in ("ilp", "fw_sdp", "cograph_dp", "dispatcher_auto"):
                self.assertIn(key, r["solvers"])


class TestEmitters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel = [
            ("TEST", "triangle", _triangle(), None),
            ("TEST", "k4", _k4(), None),
        ]
        cls.rows = run_panel(panel, ilp_time=3.0, fw_iters=100,
                             dispatcher_budget=1.0)

    def test_latex_table_has_booktabs(self):
        tex = emit_latex_table(self.rows)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\toprule", tex)
        self.assertIn(r"\midrule", tex)
        self.assertIn(r"\bottomrule", tex)
        self.assertIn(r"\end{tabular}", tex)

    def test_latex_table_has_instances(self):
        tex = emit_latex_table(self.rows)
        self.assertIn("triangle", tex)
        self.assertIn("k4", tex)

    def test_latex_underscores_escaped(self):
        """Onderstreepjes in instance-namen moeten geescaped zijn."""
        panel = [("TEST", "pm1s_n20", _triangle(), None)]
        rows = run_panel(panel, ilp_time=3.0, fw_iters=50,
                         dispatcher_budget=1.0)
        tex = emit_latex_table(rows)
        # Raw "_" niet toegestaan buiten command-context; \_ moet er staan.
        self.assertIn(r"pm1s\_n20", tex)

    def test_markdown_table_header(self):
        md = emit_markdown_table(self.rows)
        self.assertIn("| Dataset |", md)
        self.assertIn("| Instance |", md)
        self.assertIn("ILP-OPT", md)

    def test_csv_has_header_and_rows(self):
        csv = emit_csv(self.rows)
        lines = csv.strip().split("\n")
        self.assertEqual(len(lines), 1 + len(self.rows))
        self.assertIn("idx", lines[0])
        self.assertIn("dataset", lines[0])


class TestSaveArtifacts(unittest.TestCase):
    def test_writes_all_four_files(self):
        panel = [("TEST", "triangle", _triangle(), None)]
        rows = run_panel(panel, ilp_time=3.0, fw_iters=50,
                         dispatcher_budget=1.0)

        # Use a manually-managed temp dir to avoid mount-recursion edge cases.
        tmp = os.path.join(tempfile.gettempdir(), f"b186_test_{uuid.uuid4().hex}")
        os.makedirs(tmp, exist_ok=True)
        try:
            paths = save_artifacts(rows, tmp)
            for key in ("json", "csv", "tex", "md"):
                self.assertIn(key, paths)
                self.assertTrue(os.path.exists(paths[key]),
                                f"Missing artifact: {key} at {paths[key]}")
            # JSON roundtrip
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["rows"]), 1)
            self.assertIn("meta", data)
            self.assertEqual(data["meta"]["n_instances"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Schema-invariant
# ============================================================

class TestResultSchema(unittest.TestCase):
    """Waarborg dat de per-instance dict altijd de verwachte keys heeft —
    downstream-tooling (B177 paper-figuur-pipeline) leunt hierop."""

    def test_solver_result_dataclass_fields(self):
        r = SolverResult(solver="stub")
        d = r.__dict__
        for key in ("solver", "cut_value", "upper_bound", "lower_bound",
                    "gap_pct", "level", "wall_time", "extra",
                    "skipped", "skip_reason"):
            self.assertIn(key, d)

    def test_ilp_result_has_certified_flag(self):
        r = run_ilp(_triangle(), time_limit=2.0)
        self.assertIn("certified", r.extra)
        self.assertIn("status", r.extra)

    def test_fw_result_has_iterations_and_converged(self):
        r = run_fw_sdp(_triangle(), max_iter=50)
        self.assertIn("iterations", r.extra)
        self.assertIn("converged", r.extra)

    def test_dispatcher_result_has_strategy_and_tier(self):
        r = run_dispatcher(_triangle(), time_budget=1.0)
        self.assertIn("strategy", r.extra)
        self.assertIn("tier", r.extra)


if __name__ == "__main__":
    unittest.main()
