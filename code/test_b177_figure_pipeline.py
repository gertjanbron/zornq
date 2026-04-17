#!/usr/bin/env python3
"""Tests for B177 Paper Figure Pipeline.

Exercises:
  - JSON dump/load round-trip.
  - Leaderboard + scaling data collection.
  - Matplotlib figure rendering (PDF).
  - PGFPlots CSV + .tex emission.
  - CLI entrypoint.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b177_figure_pipeline import (
    dump_json,
    load_json,
    collect_leaderboard_data,
    collect_ilp_scaling_data,
    plot_leaderboard_ratio,
    plot_ilp_scaling,
    emit_leaderboard_csv,
    emit_ilp_scaling_csv,
    emit_pgfplots_leaderboard,
    emit_pgfplots_ilp_scaling,
    run_pipeline,
    main as cli_main,
)


# ============================================================
# Helpers
# ============================================================

def _tiny_leaderboard() -> dict:
    """Build a small synthetic leaderboard dict mimicking real output."""
    return {
        "meta": {"timestamp": "2026-04-17 00:00:00", "ilp_time_limit": 10.0,
                 "n_instances": 3},
        "rows": [
            {"dataset": "Gset", "name": "petersen", "n": 10, "m": 15,
             "opt": 12.0, "bp": 12.0, "lc": 12.0, "gw": 13.5,
             "t_ilp": 0.02, "t_bp": 0.01, "t_lc": 0.05, "certified": True,
             "known_opt": 12.0},
            {"dataset": "BiqMac", "name": "pm1s_n20", "n": 20, "m": 59,
             "opt": 24.0, "bp": 13.0, "lc": 10.0, "gw": 26.0,
             "t_ilp": 0.01, "t_bp": 0.005, "t_lc": 0.04, "certified": True,
             "known_opt": None},
            {"dataset": "DIMACS", "name": "k4", "n": 4, "m": 6,
             "opt": 4.0, "bp": 4.0, "lc": 4.0, "gw": 4.5,
             "t_ilp": 0.005, "t_bp": 0.001, "t_lc": 0.01, "certified": True,
             "known_opt": None},
        ],
    }


def _tiny_scaling() -> dict:
    return {
        "meta": {"timestamp": "2026-04-17 00:00:00", "ilp_time_limit": 10.0,
                 "graph_type": "random_3reg", "sizes": [10, 14],
                 "seeds": [0, 1]},
        "points": [
            {"n": 10, "m": 15, "seed": 0, "t_ilp": 0.012, "opt": 12.0, "certified": True},
            {"n": 10, "m": 15, "seed": 1, "t_ilp": 0.010, "opt": 11.0, "certified": True},
            {"n": 14, "m": 21, "seed": 0, "t_ilp": 0.040, "opt": 15.0, "certified": True},
            {"n": 14, "m": 21, "seed": 1, "t_ilp": 0.035, "opt": 16.0, "certified": True},
        ],
    }


# ============================================================
# Suite 1: JSON I/O
# ============================================================

class TestJsonIO(unittest.TestCase):

    def test_roundtrip_dict(self):
        obj = {"a": 1, "b": [2, 3], "c": {"d": "x"}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            dump_json(obj, p)
            back = load_json(p)
            self.assertEqual(obj, back)

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "deeply" / "nested" / "path" / "t.json"
            dump_json({"x": 1}, p)
            self.assertTrue(p.is_file())

    def test_roundtrip_leaderboard(self):
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lb.json"
            dump_json(data, p)
            back = load_json(p)
            self.assertEqual(back["meta"]["n_instances"], 3)
            self.assertEqual(len(back["rows"]), 3)
            self.assertEqual(back["rows"][0]["name"], "petersen")


# ============================================================
# Suite 2: Data collection
# ============================================================

class TestDataCollection(unittest.TestCase):

    def test_collect_leaderboard_smoke(self):
        """End-to-end collection should produce non-empty rows."""
        data = collect_leaderboard_data(ilp_time=5.0)
        self.assertIn("meta", data)
        self.assertIn("rows", data)
        self.assertGreater(len(data["rows"]), 0)
        row0 = data["rows"][0]
        self.assertIn("dataset", row0)
        self.assertIn("name", row0)
        self.assertIn("opt", row0)
        # At least one row should have certified ILP
        self.assertTrue(any(r.get("certified") for r in data["rows"]))

    def test_collect_ilp_scaling_smoke(self):
        """Scaling collection on tiny sweep."""
        data = collect_ilp_scaling_data(sizes=[6, 10], seeds=[0, 1], ilp_time=5.0)
        self.assertIn("meta", data)
        self.assertEqual(len(data["points"]), 4)
        for p in data["points"]:
            self.assertIn("n", p)
            self.assertIn("t_ilp", p)
            self.assertGreaterEqual(p["t_ilp"], 0.0)


# ============================================================
# Suite 3: Matplotlib rendering
# ============================================================

class TestMatplotlibFigures(unittest.TestCase):

    def test_leaderboard_pdf(self):
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "lb.pdf"
            plot_leaderboard_ratio(data, pdf)
            self.assertTrue(pdf.is_file())
            self.assertGreater(pdf.stat().st_size, 1000)  # > 1KB

    def test_scaling_pdf(self):
        data = _tiny_scaling()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "sc.pdf"
            plot_ilp_scaling(data, pdf)
            self.assertTrue(pdf.is_file())
            self.assertGreater(pdf.stat().st_size, 1000)

    def test_leaderboard_handles_missing_values(self):
        """None for bp/lc should not crash."""
        data = _tiny_leaderboard()
        data["rows"][1]["bp"] = None
        data["rows"][1]["lc"] = None
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "lb.pdf"
            plot_leaderboard_ratio(data, pdf)
            self.assertTrue(pdf.is_file())


# ============================================================
# Suite 4: PGFPlots CSV + .tex emission
# ============================================================

class TestPgfplotsEmission(unittest.TestCase):

    def test_leaderboard_csv(self):
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lb.csv"
            emit_leaderboard_csv(data, p)
            self.assertTrue(p.is_file())
            text = p.read_text()
            self.assertIn("idx dataset name", text)
            self.assertIn("petersen", text)
            self.assertIn("k4", text)
            # Three rows + header
            self.assertEqual(len(text.strip().splitlines()), 4)

    def test_scaling_csv(self):
        data = _tiny_scaling()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sc.csv"
            emit_ilp_scaling_csv(data, p)
            text = p.read_text()
            self.assertIn("n m seed t_ilp", text)
            self.assertEqual(len(text.strip().splitlines()), 5)  # header + 4

    def test_leaderboard_tex(self):
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lb.tex"
            emit_pgfplots_leaderboard(data, p)
            tex = p.read_text()
            self.assertIn("tikzpicture", tex)
            self.assertIn("addplot", tex)
            self.assertIn("MPQS-BP", tex)

    def test_scaling_tex(self):
        data = _tiny_scaling()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sc.tex"
            emit_pgfplots_ilp_scaling(data, p)
            tex = p.read_text()
            self.assertIn("tikzpicture", tex)
            self.assertIn("ymode=log", tex)


# ============================================================
# Suite 5: run_pipeline end-to-end
# ============================================================

class TestRunPipeline(unittest.TestCase):

    def test_full_pipeline_fast(self):
        """Run the full pipeline in fast mode."""
        with tempfile.TemporaryDirectory() as tmp:
            out_fig = Path(tmp) / "fig"
            out_data = Path(tmp) / "data"
            summary = run_pipeline(
                out_fig=out_fig,
                out_data=out_data,
                do_figures=True,
                do_scaling=True,
                scaling_sizes=[8, 12],
                verbose=False,
            )
            # Leaderboard artifacts
            self.assertTrue((out_data / "b154_leaderboard.json").is_file())
            self.assertTrue((out_data / "b154_leaderboard.csv").is_file())
            self.assertTrue((out_data / "b154_leaderboard.tex").is_file())
            self.assertTrue((out_fig / "b154_leaderboard_ratio.pdf").is_file())
            # Scaling artifacts
            self.assertTrue((out_data / "ilp_scaling.json").is_file())
            self.assertTrue((out_data / "ilp_scaling.csv").is_file())
            self.assertTrue((out_data / "ilp_scaling.tex").is_file())
            self.assertTrue((out_fig / "ilp_scaling.pdf").is_file())
            # Summary consistency
            self.assertIn("leaderboard", summary)
            self.assertIn("scaling", summary)

    def test_data_only_run(self):
        """--no-figures should skip PDFs but keep JSON/CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            out_fig = Path(tmp) / "fig"
            out_data = Path(tmp) / "data"
            summary = run_pipeline(
                out_fig=out_fig,
                out_data=out_data,
                do_figures=False,
                do_scaling=False,
                verbose=False,
            )
            self.assertTrue((out_data / "b154_leaderboard.json").is_file())
            self.assertFalse((out_fig / "b154_leaderboard_ratio.pdf").exists())
            self.assertNotIn("scaling", summary)


# ============================================================
# Suite 6: CLI
# ============================================================

class TestCli(unittest.TestCase):

    def test_cli_fast_no_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_fig = Path(tmp) / "fig"
            out_data = Path(tmp) / "data"
            rc = cli_main([
                "--out-fig", str(out_fig),
                "--out-data", str(out_data),
                "--no-scaling",
                "--fast",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_data / "b154_leaderboard.json").is_file())
            self.assertTrue((out_fig / "b154_leaderboard_ratio.pdf").is_file())

    def test_cli_data_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_fig = Path(tmp) / "fig"
            out_data = Path(tmp) / "data"
            rc = cli_main([
                "--out-fig", str(out_fig),
                "--out-data", str(out_data),
                "--no-figures",
                "--no-scaling",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_data / "b154_leaderboard.json").is_file())


# ============================================================
# Suite 7: JSON content sanity
# ============================================================

class TestJsonContentSanity(unittest.TestCase):

    def test_leaderboard_values_are_numeric(self):
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lb.json"
            dump_json(data, p)
            back = load_json(p)
            for row in back["rows"]:
                self.assertIsInstance(row["opt"], (int, float))

    def test_pgfplots_ratios_in_range(self):
        """r_bp and r_lc should be between 0 and ~1.2."""
        data = _tiny_leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lb.csv"
            emit_leaderboard_csv(data, p)
            for line in p.read_text().strip().splitlines()[1:]:
                parts = line.split()
                r_bp = float(parts[-2])
                r_lc = float(parts[-1])
                self.assertGreaterEqual(r_bp, 0.0)
                self.assertLessEqual(r_bp, 1.2)
                self.assertGreaterEqual(r_lc, 0.0)
                self.assertLessEqual(r_lc, 1.2)


# ============================================================
# Suite 8: Artifact path independence
# ============================================================

class TestPathIndependence(unittest.TestCase):

    def test_deep_nested_output_paths(self):
        """Outputs in deeply nested dirs should be created transparently."""
        with tempfile.TemporaryDirectory() as tmp:
            out_fig = Path(tmp) / "a" / "b" / "c" / "fig"
            out_data = Path(tmp) / "x" / "y" / "z" / "data"
            summary = run_pipeline(
                out_fig=out_fig,
                out_data=out_data,
                do_figures=False,
                do_scaling=False,
                verbose=False,
            )
            self.assertTrue(out_data.exists())
            self.assertIn("leaderboard", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
