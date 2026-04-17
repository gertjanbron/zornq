#!/usr/bin/env python3
"""Tests voor seed_ledger.py (B55 follow-up)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from seed_ledger import (
    SeedLedger,
    _derive_child_seed,
    _master_to_hex,
    derive,
    get_global_ledger,
    set_global_ledger,
)


# =====================================================================
# Helpers
# =====================================================================


class TestMasterToHex(unittest.TestCase):
    def test_int_master_is_hex(self):
        self.assertEqual(_master_to_hex(0), "0x0")
        self.assertEqual(_master_to_hex(42), "0x2a")
        self.assertEqual(_master_to_hex(255), "0xff")

    def test_string_master_is_deterministic(self):
        h1 = _master_to_hex("gertjan-2026")
        h2 = _master_to_hex("gertjan-2026")
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("0x"))

    def test_string_master_differs_from_other_strings(self):
        self.assertNotEqual(_master_to_hex("a"), _master_to_hex("b"))

    def test_hex_string_master_parsed(self):
        self.assertEqual(_master_to_hex("0xdeadbeef"), "0xdeadbeef")

    def test_negative_master_raises(self):
        with self.assertRaises(ValueError):
            _master_to_hex(-1)

    def test_numpy_int_master_accepted(self):
        # numpy ints komen voor in benchmark-helpers
        self.assertEqual(_master_to_hex(np.int64(42)), "0x2a")

    def test_non_int_non_str_raises(self):
        with self.assertRaises(TypeError):
            _master_to_hex(3.14)  # type: ignore[arg-type]


class TestDeriveChildSeed(unittest.TestCase):
    def test_deterministic(self):
        a = _derive_child_seed("0x2a", "gw_rounding")
        b = _derive_child_seed("0x2a", "gw_rounding")
        self.assertEqual(a, b)

    def test_different_labels_differ(self):
        a = _derive_child_seed("0x2a", "graph_gen")
        b = _derive_child_seed("0x2a", "gw_rounding")
        self.assertNotEqual(a, b)

    def test_different_masters_differ(self):
        a = _derive_child_seed("0x2a", "graph_gen")
        b = _derive_child_seed("0x2b", "graph_gen")
        self.assertNotEqual(a, b)

    def test_in_uint32_range(self):
        seed = _derive_child_seed("0x2a", "anything")
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_empty_label_raises(self):
        with self.assertRaises(ValueError):
            _derive_child_seed("0x2a", "")

    def test_non_str_label_raises(self):
        with self.assertRaises(TypeError):
            _derive_child_seed("0x2a", 42)  # type: ignore[arg-type]


# =====================================================================
# SeedLedger basics
# =====================================================================


class TestSeedLedgerCore(unittest.TestCase):
    def test_init_default_master(self):
        ledger = SeedLedger()
        self.assertEqual(ledger.master, 0)
        self.assertEqual(ledger.children, {})

    def test_init_with_master(self):
        ledger = SeedLedger(master=42)
        self.assertEqual(ledger.master, 42)

    def test_derive_registers_seed(self):
        ledger = SeedLedger(master=42)
        seed = ledger.derive("graph_gen")
        self.assertEqual(ledger.children["graph_gen"], seed)

    def test_derive_is_deterministic_across_instances(self):
        l1 = SeedLedger(master=42)
        l2 = SeedLedger(master=42)
        self.assertEqual(l1.derive("x"), l2.derive("x"))

    def test_derive_idempotent(self):
        ledger = SeedLedger(master=42)
        a = ledger.derive("x")
        b = ledger.derive("x")
        self.assertEqual(a, b)
        self.assertEqual(len(ledger.children), 1)

    def test_get_returns_none_if_not_registered(self):
        ledger = SeedLedger(master=42)
        self.assertIsNone(ledger.get("never_derived"))

    def test_get_returns_seed_after_derive(self):
        ledger = SeedLedger(master=42)
        s = ledger.derive("x")
        self.assertEqual(ledger.get("x"), s)

    def test_labels_is_sorted(self):
        ledger = SeedLedger(master=42)
        ledger.derive("z")
        ledger.derive("a")
        ledger.derive("m")
        self.assertEqual(ledger.labels(), ["a", "m", "z"])

    def test_negative_master_raises(self):
        with self.assertRaises(ValueError):
            SeedLedger(master=-1)

    def test_non_int_master_raises(self):
        with self.assertRaises(TypeError):
            SeedLedger(master=3.14)  # type: ignore[arg-type]

    def test_numpy_int_master_accepted(self):
        ledger = SeedLedger(master=np.int64(42))
        self.assertEqual(ledger.master, 42)


# =====================================================================
# RNG-factories
# =====================================================================


class TestLedgerRNGs(unittest.TestCase):
    def test_numpy_rng_is_deterministic(self):
        l1 = SeedLedger(master=42)
        l2 = SeedLedger(master=42)
        r1 = l1.numpy_rng("gw_rounding")
        r2 = l2.numpy_rng("gw_rounding")
        a = r1.standard_normal(100)
        b = r2.standard_normal(100)
        np.testing.assert_array_equal(a, b)

    def test_different_labels_give_different_streams(self):
        ledger = SeedLedger(master=42)
        r1 = ledger.numpy_rng("a")
        r2 = ledger.numpy_rng("b")
        a = r1.standard_normal(100)
        b = r2.standard_normal(100)
        # Statistisch onafhankelijk -> max-abs-diff moet groot zijn
        self.assertGreater(float(np.max(np.abs(a - b))), 0.5)

    def test_numpy_rng_type(self):
        ledger = SeedLedger(master=42)
        self.assertIsInstance(ledger.numpy_rng("x"), np.random.Generator)

    def test_numpy_random_state_deterministic(self):
        l1 = SeedLedger(master=42)
        l2 = SeedLedger(master=42)
        r1 = l1.numpy_random_state("a")
        r2 = l2.numpy_random_state("a")
        np.testing.assert_array_equal(r1.rand(50), r2.rand(50))

    def test_python_random_deterministic(self):
        l1 = SeedLedger(master=42)
        l2 = SeedLedger(master=42)
        r1 = l1.python_random("x")
        r2 = l2.python_random("x")
        a = [r1.random() for _ in range(100)]
        b = [r2.random() for _ in range(100)]
        self.assertEqual(a, b)


# =====================================================================
# Serialisatie
# =====================================================================


class TestLedgerSerialization(unittest.TestCase):
    def test_to_dict_has_required_fields(self):
        ledger = SeedLedger(master=42, metadata={"host": "laptop"})
        ledger.derive("x")
        d = ledger.to_dict()
        for key in ("version", "kind", "master", "master_hex", "children",
                    "metadata", "created_at"):
            self.assertIn(key, d)
        self.assertEqual(d["kind"], "zornq.seed_ledger")

    def test_to_dict_children_sorted(self):
        ledger = SeedLedger(master=42)
        for lbl in ("z", "a", "m"):
            ledger.derive(lbl)
        d = ledger.to_dict()
        # dict insertion-order bewaren; we controleren dat keys gesorteerd zijn
        self.assertEqual(list(d["children"].keys()), ["a", "m", "z"])

    def test_save_and_load_roundtrip(self):
        ledger = SeedLedger(master=42, metadata={"purpose": "test"})
        for lbl in ("graph_gen", "gw_rounding", "bls"):
            ledger.derive(lbl)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ledger.save(path)
            loaded = SeedLedger.load(path)
            self.assertEqual(loaded.master, 42)
            self.assertEqual(loaded.children, ledger.children)
            self.assertEqual(loaded.metadata, ledger.metadata)
        finally:
            os.unlink(path)

    def test_save_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "deep", "ledger.json")
            SeedLedger(master=42).save(path)
            self.assertTrue(os.path.exists(path))

    def test_load_validates_integrity(self):
        # Sla op, knoei met children, load moet exceptie geven
        ledger = SeedLedger(master=42)
        ledger.derive("a")
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ledger.save(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Corrumpeer: zet willekeurig ander getal
            data["children"]["a"] = 12345
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with self.assertRaises(ValueError):
                SeedLedger.load(path)
        finally:
            os.unlink(path)

    def test_load_rejects_foreign_record(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
            json.dump({"kind": "other_thing", "master": 0}, f)
        try:
            with self.assertRaises(ValueError):
                SeedLedger.load(path)
        finally:
            os.unlink(path)


# =====================================================================
# AuditTrail-integratie
# =====================================================================


class _FakeAudit:
    """Minimale stand-in voor AuditTrail -- we hebben alleen .data nodig."""
    def __init__(self):
        self.data = {"run_id": "test"}


class TestAuditIntegration(unittest.TestCase):
    def test_attach_to_audit(self):
        ledger = SeedLedger(master=42)
        ledger.derive("a")
        audit = _FakeAudit()
        ledger.attach_to_audit(audit)
        self.assertIn("seed_ledger", audit.data)
        self.assertEqual(audit.data["seed_ledger"]["master"], 42)
        self.assertIn("a", audit.data["seed_ledger"]["children"])

    def test_attach_rejects_non_audit_object(self):
        ledger = SeedLedger(master=42)
        with self.assertRaises(TypeError):
            ledger.attach_to_audit(object())

    def test_as_record_alias(self):
        ledger = SeedLedger(master=42)
        self.assertEqual(ledger.as_record(), ledger.to_dict())


# =====================================================================
# Globale ledger
# =====================================================================


class TestGlobalLedger(unittest.TestCase):
    def setUp(self):
        set_global_ledger(None)  # reset

    def tearDown(self):
        set_global_ledger(None)

    def test_default_is_none(self):
        self.assertIsNone(get_global_ledger())

    def test_set_and_get(self):
        ledger = SeedLedger(master=42)
        set_global_ledger(ledger)
        self.assertIs(get_global_ledger(), ledger)

    def test_set_to_none_clears(self):
        set_global_ledger(SeedLedger(master=1))
        set_global_ledger(None)
        self.assertIsNone(get_global_ledger())

    def test_set_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            set_global_ledger(42)  # type: ignore[arg-type]

    def test_derive_shortcut_works(self):
        set_global_ledger(SeedLedger(master=42))
        s1 = derive("x")
        s2 = derive("x")
        self.assertEqual(s1, s2)

    def test_derive_without_global_raises(self):
        with self.assertRaises(RuntimeError):
            derive("x")


# =====================================================================
# End-to-end replay scenario
# =====================================================================


class TestReplayScenario(unittest.TestCase):
    def test_full_roundtrip_produces_same_samples(self):
        # Schrijf een ledger
        ledger_a = SeedLedger(master=2026, metadata={"run": "alpha"})
        rng_a_gw = ledger_a.numpy_rng("gw_rounding")
        samples_a = rng_a_gw.standard_normal(200)
        rng_a_graph = ledger_a.numpy_rng("graph_gen")
        samples_graph = rng_a_graph.integers(0, 1000, size=50)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ledger_a.save(path)

            # Laad, haal RNG's opnieuw op
            ledger_b = SeedLedger.load(path)
            rng_b_gw = ledger_b.numpy_rng("gw_rounding")
            samples_b = rng_b_gw.standard_normal(200)
            rng_b_graph = ledger_b.numpy_rng("graph_gen")
            samples_graph_b = rng_b_graph.integers(0, 1000, size=50)

            # Bit-identiek
            np.testing.assert_array_equal(samples_a, samples_b)
            np.testing.assert_array_equal(samples_graph, samples_graph_b)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
