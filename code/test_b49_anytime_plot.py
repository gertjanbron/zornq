#!/usr/bin/env python3
"""Unit-tests voor b49_anytime_plot."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from b60_gw_bound import SimpleGraph
from b49_anytime_plot import (
    AnytimeTrace,
    cut_value,
    one_flip_polish,
    collect_ub_trace,
    collect_lb_trace,
    collect_opt,
    run_anytime_pipeline,
    trace_to_json,
    trace_to_csv,
    emit_pgfplots_tex,
    save_artifacts,
)


# ============================================================
# Fixtures
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


def _petersen() -> SimpleGraph:
    """10-vertex Petersen graph (buitenring 0-4 + binnenring 5-9 + spokes)."""
    g = SimpleGraph(10)
    outer = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    inner = [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
    spokes = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    for (u, v) in outer + inner + spokes:
        g.add_edge(u, v, 1.0)
    return g


# ============================================================
# Cut-helpers
# ============================================================

class TestCutValue(unittest.TestCase):
    def test_triangle_alternating_cut(self):
        g = _triangle()
        # all-same → 0
        self.assertEqual(cut_value(g, [0, 0, 0]), 0)
        # 0-1 split: 0 en 1 tegenover 2 → 2 edges cut (0-2 en 1-2)
        self.assertEqual(cut_value(g, [0, 0, 1]), 2)
        # max-cut triangle = 2
        self.assertEqual(cut_value(g, [0, 1, 0]), 2)

    def test_c4_max_cut_is_4(self):
        g = _cycle_4()
        self.assertEqual(cut_value(g, [0, 1, 0, 1]), 4)
        self.assertEqual(cut_value(g, [0, 0, 0, 0]), 0)


class TestOneFlipPolish(unittest.TestCase):
    def test_polish_never_decreases_cut(self):
        g = _petersen()
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=g.n).astype(np.int8)
        cut0 = cut_value(g, bits)
        bits1 = one_flip_polish(g, bits)
        cut1 = cut_value(g, bits1)
        self.assertGreaterEqual(cut1, cut0)

    def test_polish_reaches_local_optimum(self):
        g = _cycle_4()
        bits = np.zeros(4, dtype=np.int8)
        bits1 = one_flip_polish(g, bits)
        self.assertEqual(cut_value(g, bits1), 4)  # C_4 maxcut = 4


# ============================================================
# UB collector
# ============================================================

class TestUBTrace(unittest.TestCase):
    def test_ub_trace_is_nonempty(self):
        g = _petersen()
        tr, fw = collect_ub_trace(g, max_iter=30, seed=0)
        self.assertGreater(len(tr.ub_trace), 0)
        self.assertEqual(tr.n, 10)
        self.assertEqual(tr.m, 15)
        self.assertGreaterEqual(tr.fw_iterations, 1)

    def test_ub_monotone_non_increasing(self):
        g = _petersen()
        tr, _ = collect_ub_trace(g, max_iter=50, seed=0)
        mono = [m for (_, _, m) in tr.ub_trace]
        for i in range(1, len(mono)):
            self.assertLessEqual(mono[i], mono[i - 1] + 1e-9,
                                 f"UB mono niet non-stijgend bij i={i}")

    def test_ub_elapsed_monotone(self):
        g = _petersen()
        tr, _ = collect_ub_trace(g, max_iter=50, seed=0)
        ts = [t for (t, _, _) in tr.ub_trace]
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i], ts[i - 1] - 1e-9,
                                    f"UB elapsed niet monotone bij i={i}")


# ============================================================
# LB cascade
# ============================================================

class TestLBTrace(unittest.TestCase):
    def test_lb_trace_has_all_layers(self):
        g = _petersen()
        _, fw = collect_ub_trace(g, max_iter=50, seed=0)
        lb = collect_lb_trace(g, fw, bp_iters=50, gw_trials=10, seed=0)
        # minimaal 4 snapshots: alternating, mpqs_bp, fw_gw_rounding, 1flip_polish
        self.assertGreaterEqual(len(lb), 4)
        sources = [src for (_, _, _, src) in lb]
        self.assertIn("alternating", sources)

    def test_lb_monotone_non_decreasing(self):
        g = _petersen()
        _, fw = collect_ub_trace(g, max_iter=50, seed=0)
        lb = collect_lb_trace(g, fw, bp_iters=50, gw_trials=10, seed=0)
        mono = [m for (_, _, m, _) in lb]
        for i in range(1, len(mono)):
            self.assertGreaterEqual(mono[i], mono[i - 1] - 1e-9,
                                    f"LB mono niet niet-dalend bij i={i}")

    def test_lb_elapsed_ordered(self):
        g = _petersen()
        _, fw = collect_ub_trace(g, max_iter=30, seed=0)
        lb = collect_lb_trace(g, fw, bp_iters=30, gw_trials=5, seed=0)
        ts = [t for (t, _, _, _) in lb]
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i], ts[i - 1] - 1e-9)


# ============================================================
# OPT via ILP
# ============================================================

class TestCollectOpt(unittest.TestCase):
    def test_triangle_opt_is_2(self):
        g = _triangle()
        opt, cert = collect_opt(g, time_limit=3.0)
        self.assertAlmostEqual(opt, 2.0, places=6)
        self.assertTrue(cert)

    def test_petersen_opt_is_12(self):
        g = _petersen()
        opt, cert = collect_opt(g, time_limit=5.0)
        self.assertAlmostEqual(opt, 12.0, places=6)
        self.assertTrue(cert)


# ============================================================
# Full pipeline
# ============================================================

class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _petersen()
        cls.trace = run_anytime_pipeline(
            "TEST", "petersen_test", cls.g,
            fw_iters=40, ilp_time=3.0, bp_iters=40, gw_trials=10, seed=0,
        )

    def test_sandwich_respects_opt(self):
        """Na afloop: LB <= OPT <= UB (sandwich-validiteit)."""
        tr = self.trace
        self.assertIsNotNone(tr.opt_value)
        if tr.ub_trace:
            ub_final = tr.ub_trace[-1][2]
            # UB moet OPT dominantezijn (modulo kleine numerieke slack).
            self.assertGreaterEqual(ub_final, tr.opt_value - 1e-2)
        if tr.lb_trace:
            lb_final = tr.lb_trace[-1][2]
            # LB mag OPT niet overschrijden voor unweighted positieve grafen.
            self.assertLessEqual(lb_final, tr.opt_value + 1e-6)

    def test_trace_fields_populated(self):
        tr = self.trace
        self.assertEqual(tr.instance_name, "petersen_test")
        self.assertEqual(tr.dataset, "TEST")
        self.assertEqual(tr.n, 10)
        self.assertGreater(tr.fw_iterations, 0)


# ============================================================
# Serializers
# ============================================================

class TestSerializers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = _petersen()
        cls.trace = run_anytime_pipeline(
            "TEST", "pet_ser", g,
            fw_iters=20, ilp_time=3.0, bp_iters=20, gw_trials=5, seed=0,
        )

    def test_trace_to_json_schema(self):
        d = trace_to_json(self.trace)
        for key in ("meta", "instance", "opt", "fw", "ub_trace", "lb_trace"):
            self.assertIn(key, d)
        self.assertEqual(d["instance"]["name"], "pet_ser")
        self.assertIsInstance(d["ub_trace"], list)
        self.assertIsInstance(d["lb_trace"], list)
        # Ieder punt in ub_trace heeft de juiste keys
        if d["ub_trace"]:
            self.assertIn("t", d["ub_trace"][0])
            self.assertIn("ub_mono", d["ub_trace"][0])

    def test_trace_to_csv_has_header(self):
        csv = trace_to_csv(self.trace)
        lines = csv.strip().split("\n")
        self.assertIn("idx", lines[0])
        self.assertIn("t_ub", lines[0])
        self.assertIn("t_lb", lines[0])
        self.assertIn("source", lines[0])
        self.assertGreater(len(lines), 1)


# ============================================================
# Artifact emission
# ============================================================

class TestSaveArtifacts(unittest.TestCase):
    def test_writes_all_four_artifacts(self):
        g = _petersen()
        tr = run_anytime_pipeline(
            "TEST", "pet_art", g,
            fw_iters=20, ilp_time=3.0, bp_iters=20, gw_trials=5, seed=0,
        )
        tmp = os.path.join(tempfile.gettempdir(),
                           f"b49_test_{uuid.uuid4().hex}")
        os.makedirs(tmp, exist_ok=True)
        try:
            paths = save_artifacts(tr, tmp)
            for key in ("json", "csv", "pdf", "tex"):
                self.assertIn(key, paths)
                self.assertTrue(os.path.exists(paths[key]),
                                f"Missing: {key} at {paths[key]}")
                self.assertGreater(os.path.getsize(paths[key]), 0)

            # PDF-magic
            with open(paths["pdf"], "rb") as f:
                head = f.read(5)
            self.assertEqual(head, b"%PDF-", "PDF header incorrect")

            # JSON roundtrip
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["instance"]["name"], "pet_art")

            # Tex is PGFPlots-compatible
            with open(paths["tex"], encoding="utf-8") as f:
                tex = f.read()
            self.assertIn(r"\begin{tikzpicture}", tex)
            self.assertIn(r"\begin{axis}", tex)
            self.assertIn(r"\addplot", tex)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# PGFPlots emitter
# ============================================================

class TestPgfplotsEmitter(unittest.TestCase):
    def test_pgfplots_contains_opt_dashed_line(self):
        tr = AnytimeTrace(
            instance_name="demo", dataset="TEST", n=5, m=6,
            opt_value=3.0, opt_certified=True,
            ub_trace=[(0.001, 4.0, 4.0), (0.01, 3.5, 3.5)],
            lb_trace=[(0.002, 2.0, 2.0, "greedy"),
                      (0.02, 2.8, 2.8, "polish")],
        )
        tmp = os.path.join(tempfile.gettempdir(),
                           f"b49_pg_{uuid.uuid4().hex}.tex")
        try:
            emit_pgfplots_tex(tr, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            self.assertIn(r"\draw[dashed", content)
            self.assertIn("OPT = 3", content)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    unittest.main()
