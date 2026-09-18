"""The guard constant must match the coordination hash recorded for the pause."""
from pathlib import Path
import ast
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESERVATION = ROOT / "locomotion" / "reservation.py"
POLICY = ROOT / "docs" / "SPARK_COMPUTE_COORDINATION.md"
PAUSE_HASH = "c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c"


def module_constant(name):
    tree = ast.parse(RESERVATION.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing module constant: " + name)


class ReservationBindingTests(unittest.TestCase):
    def test_guard_binds_the_recorded_pause_coordination_bytes(self):
        recorded = re.findall(r"`([0-9a-f]{64})`", POLICY.read_text())
        self.assertIn(PAUSE_HASH, recorded)
        self.assertEqual(module_constant("COORDINATION_SHA256"), PAUSE_HASH)

    def test_old_coordination_hash_is_not_bound(self):
        self.assertNotEqual(module_constant("COORDINATION_SHA256"),
                            "fc1da7bf07d1db3022b1b9a3ec41d261a69e7a19c562457ea4e4326a043893b0")
