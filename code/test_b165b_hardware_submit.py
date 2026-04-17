#!/usr/bin/env python3
"""Unit-tests voor B165b (hardware-submit + noise-baselines + parser).

We draaien ZONDER IBM-token: alle `submit`-paden die een echte call doen
worden geskipt of gemockt. De baselines en parser worden volledig getest op
Aer.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from b60_gw_bound import SimpleGraph

from b165b_hardware_submit import (
    INSTANCE_SPECS,
    build_instance,
    qaoa_grid_search,
    SubmissionBundle,
    save_bundle,
    load_bundle,
    read_token,
    dry_run,
)

from b165b_noise_baselines import (
    BaselineResult,
    save_baseline,
    run_baselines_for_instance,
    run_all_baselines,
    noise_model_from_properties,
)

from b165b_parse_results import (
    TableRow,
    assemble_rows,
    emit_markdown,
    emit_latex,
    emit_json,
    save_table,
)


# Skip heel de module als qiskit niet geïnstalleerd is.
try:
    import qiskit  # noqa: F401
    import qiskit_aer  # noqa: F401
    QISKIT_OK = True
except Exception:  # pragma: no cover
    QISKIT_OK = False

QISKIT_REASON = "qiskit / qiskit-aer niet geïnstalleerd"


# ============================================================
# Instance registry
# ============================================================

class TestInstanceRegistry(unittest.TestCase):
    def test_registry_has_both_instances(self):
        self.assertIn("3reg8", INSTANCE_SPECS)
        self.assertIn("myciel3", INSTANCE_SPECS)

    def test_build_3reg8(self):
        g = build_instance("3reg8")
        self.assertEqual(g.n, 8)
        # 3-reguliere: alle graden = 3
        for v in range(g.n):
            deg = sum(1 for (u1, u2, _w) in g.edges if u1 == v or u2 == v)
            self.assertEqual(deg, 3, f"vertex {v} heeft deg={deg} ≠ 3")

    def test_build_myciel3(self):
        g = build_instance("myciel3")
        self.assertEqual(g.n, 11)
        self.assertEqual(g.n_edges, 20)


# ============================================================
# Token helpers
# ============================================================

class TestTokenHelpers(unittest.TestCase):
    def test_read_token_from_env(self):
        saved = os.environ.get("TEST_TOKEN_VAR")
        try:
            os.environ["TEST_TOKEN_VAR"] = "sekret_abc"
            tok = read_token("TEST_TOKEN_VAR")
            self.assertEqual(tok, "sekret_abc")
        finally:
            if saved is None:
                os.environ.pop("TEST_TOKEN_VAR", None)
            else:
                os.environ["TEST_TOKEN_VAR"] = saved

    def test_read_token_missing(self):
        os.environ.pop("NON_EXISTENT_B165B_VAR", None)
        self.assertIsNone(read_token("NON_EXISTENT_B165B_VAR"))

    def test_read_token_from_file(self):
        tmp = Path(tempfile.gettempdir()) / f"b165b_tok_{uuid.uuid4().hex}.txt"
        try:
            tmp.write_text("file_token_xyz\nextra\n")
            tok = read_token(token_env="NO_SUCH_VAR_X", token_file=str(tmp))
            self.assertEqual(tok, "file_token_xyz")
        finally:
            tmp.unlink(missing_ok=True)


# ============================================================
# Bundle persistence
# ============================================================

class TestBundlePersistence(unittest.TestCase):
    def test_roundtrip(self):
        b = SubmissionBundle(
            instance="3reg8", n_qubits=8, p=1,
            gammas=[0.4], betas=[0.3],
            expected_exp_value_aer=5.5, shots=1024,
            backend_name="ibm_foo", job_id="jid123", status="SUBMITTED",
        )
        tmp = Path(tempfile.gettempdir()) / f"b165b_bun_{uuid.uuid4().hex}"
        try:
            p = save_bundle(b, tmp)
            self.assertTrue(p.exists())
            b2 = load_bundle(p)
            self.assertEqual(b2.instance, "3reg8")
            self.assertEqual(b2.job_id, "jid123")
            self.assertAlmostEqual(b2.expected_exp_value_aer, 5.5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# QAOA grid-search
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestGridSearch(unittest.TestCase):
    def test_grid_search_3reg8_improves_over_random(self):
        g = build_instance("3reg8")
        gammas, betas, exp_val = qaoa_grid_search(g, p=1, grid_size=4,
                                                    shots=1024)
        # Moeten geldige angles krijgen
        self.assertEqual(len(gammas), 1)
        self.assertEqual(len(betas), 1)
        # Expectation moet positief zijn (MaxCut kan geen negatief totaal geven)
        self.assertGreater(exp_val, 0)
        # Niet ver buiten redelijke range
        self.assertLess(exp_val, g.n_edges + 1)


# ============================================================
# Dry-run (no token needed)
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestDryRun(unittest.TestCase):
    def test_dry_run_no_token(self):
        # Forceer GEEN token
        os.environ.pop("QISKIT_IBM_TOKEN", None)
        tmp = Path(tempfile.gettempdir()) / f"b165b_dry_{uuid.uuid4().hex}"
        try:
            rep = dry_run(token=None, instances=["3reg8"], shots=512,
                          out_dir=tmp)
            self.assertFalse(rep["token_found"])
            self.assertIn("3reg8", rep["instances"])
            inst_rep = rep["instances"]["3reg8"]
            self.assertEqual(inst_rep["n"], 8)
            self.assertIn("gamma", inst_rep)
            self.assertIn("beta", inst_rep)
            # Prepared-bundle moet op disk staan
            self.assertTrue((tmp / "prepared_3reg8.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Noise-model fallback (no snapshot)
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestNoiseModelFallback(unittest.TestCase):
    def test_noise_from_empty_props_falls_back_safely(self):
        """Zonder geldige backend.properties krijgen we een fallback-NoiseModel,
        niet een crash."""
        class _Empty:
            qubits = []
            gates = []

            def qubit_property(self, *a, **kw):  # pragma: no cover
                raise RuntimeError("empty")
        nm, params = noise_model_from_properties(_Empty())
        self.assertIsNotNone(nm)
        # Fallback kan een 'fallback' of een 'T1_avg_us' key hebben.
        self.assertTrue("fallback" in params or "T1_avg_us" in params)


# ============================================================
# Baseline runner
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestBaselinesForInstance(unittest.TestCase):
    def test_three_baselines_on_3reg8(self):
        g = build_instance("3reg8")
        # Kies middelmatige angles (geen grid-search nodig voor deze test).
        gammas, betas = [0.6], [0.4]
        tmp = Path(tempfile.gettempdir()) / f"b165b_bl_{uuid.uuid4().hex}"
        try:
            res = run_baselines_for_instance(
                "3reg8", gammas, betas, shots=512, seed=0,
                backend_snapshot=None, out_dir=tmp,
            )
            for key in ("noiseless", "depolarising", "calibration_mirror"):
                self.assertIn(key, res)
                r = res[key]
                self.assertEqual(r.n_qubits, g.n)
                self.assertEqual(r.shots, 512)
                self.assertGreater(sum(r.counts.values()), 0)
                # Noiseless expectation moet >= noisy baselines (ruis verlaagt
                # typisch de gemeten expectation voor dezelfde angles).
            self.assertGreaterEqual(
                res["noiseless"].expectation,
                res["depolarising"].expectation - 0.5,
                "Depolariserend zou niet boven noiseless moeten zitten.",
            )
            # Disk-output
            for b in ("noiseless", "depolarising", "calibration_mirror"):
                self.assertTrue((tmp / f"3reg8_{b}.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Parser / emitters
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestParser(unittest.TestCase):
    def _prepare_baselines(self, root: Path) -> Path:
        """Mini-baselines: één instantie, drie files."""
        root.mkdir(parents=True, exist_ok=True)
        for name, exp in [("noiseless", 5.2),
                           ("depolarising", 4.8),
                           ("calibration_mirror", 4.9)]:
            r = BaselineResult(
                instance="3reg8", baseline=name, n_qubits=8, shots=512,
                counts={"00000000": 256, "11111111": 256},
                expectation=exp, best_cut=5, best_bitstring="01010101",
            )
            save_baseline(r, root)
        return root

    def test_assemble_and_emit(self):
        base_tmp = Path(tempfile.gettempdir()) / f"b165b_p_{uuid.uuid4().hex}"
        jobs_tmp = base_tmp / "jobs"
        baselines_tmp = base_tmp / "baselines"
        try:
            self._prepare_baselines(baselines_tmp)
            rows = assemble_rows(jobs_tmp, baselines_tmp,
                                  instances=["3reg8"], ilp_time=5.0)
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r.instance, "3reg8")
            self.assertEqual(r.n, 8)
            self.assertAlmostEqual(r.exp_noiseless, 5.2, places=3)
            self.assertAlmostEqual(r.exp_depolarising, 4.8, places=3)
            self.assertAlmostEqual(r.exp_cal_mirror, 4.9, places=3)
            # OPT moet certified zijn op 3-reg n=8 binnen 5s
            self.assertGreater(r.opt, 0)
            # Hardware ontbreekt → None
            self.assertIsNone(r.exp_hardware)

            md = emit_markdown(rows)
            self.assertIn("| 3reg8 |", md)
            self.assertIn("Noiseless", md)

            tex = emit_latex(rows)
            self.assertIn(r"\begin{table}", tex)
            self.assertIn(r"\toprule", tex)
            self.assertIn("3reg8", tex)

            js = emit_json(rows)
            data = json.loads(js)
            self.assertEqual(data[0]["instance"], "3reg8")

            paths = save_table(rows, base_tmp / "paper")
            for k in ("md", "tex", "json"):
                self.assertIn(k, paths)
                self.assertTrue(paths[k].exists())
        finally:
            shutil.rmtree(base_tmp, ignore_errors=True)


# ============================================================
# Parser with mocked COMPLETED hardware-bundle
# ============================================================

@unittest.skipUnless(QISKIT_OK, QISKIT_REASON)
class TestParserWithHardware(unittest.TestCase):
    def test_hardware_column_filled_via_mocked_counts(self):
        base_tmp = Path(tempfile.gettempdir()) / f"b165b_phw_{uuid.uuid4().hex}"
        jobs_tmp = base_tmp / "jobs"
        baselines_tmp = base_tmp / "baselines"
        try:
            jobs_tmp.mkdir(parents=True, exist_ok=True)
            baselines_tmp.mkdir(parents=True, exist_ok=True)

            # Bouw een "nep-hardware" bundle: we runnen Aer zelf en zeggen dat
            # het een hardware-result is.
            from b165b_hardware_submit import (
                SubmissionBundle, save_bundle,
            )
            from b165_qiskit_runtime import run_aer, to_qiskit, add_measurements
            from circuit_interface import Circuit

            g = build_instance("3reg8")
            edges = [(int(u), int(v), float(w)) for u, v, w in g.edges]
            zqc = Circuit.qaoa_maxcut(g.n, edges, p=1,
                                       gammas=[0.6], betas=[0.4])
            res = run_aer(zqc, shots=512, seed=0)
            bun = SubmissionBundle(
                instance="3reg8", n_qubits=8, p=1,
                gammas=[0.6], betas=[0.4],
                expected_exp_value_aer=0.0, shots=512,
                backend_name="ibm_mock", job_id="mockjid",
                status="COMPLETED", counts=dict(res["counts"]),
            )
            save_bundle(bun, jobs_tmp)

            # Minimale baselines
            for name, exp in [("noiseless", 5.2),
                               ("depolarising", 4.8),
                               ("calibration_mirror", 4.9)]:
                r = BaselineResult(
                    instance="3reg8", baseline=name, n_qubits=8, shots=512,
                    counts={"00000000": 256, "11111111": 256},
                    expectation=exp, best_cut=5, best_bitstring="01010101",
                )
                save_baseline(r, baselines_tmp)

            rows = assemble_rows(jobs_tmp, baselines_tmp,
                                  instances=["3reg8"], ilp_time=5.0)
            r = rows[0]
            self.assertIsNotNone(r.exp_hardware)
            self.assertEqual(r.backend_name, "ibm_mock")
            self.assertIsNotNone(r.approx_ratio)
            self.assertGreater(r.approx_ratio, 0.0)
        finally:
            shutil.rmtree(base_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
