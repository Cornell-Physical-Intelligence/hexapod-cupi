"""The accepted-controller file cannot disagree with its evidence or its pins."""

import hashlib
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "configs/accepted_controller.json"
REQUIREMENTS = ROOT / "configs/walking_requirements.json"


def pinned(name):
    """A string constant from src/locomotion/evaluation/checkpoint.py without importing torch."""
    for line in (ROOT / "src/locomotion/evaluation/checkpoint.py").read_text().splitlines():
        if line.startswith(name + " = "):
            return line.split('"')[1]
    raise AssertionError(name + " is not pinned")


class AcceptedControllerTests(unittest.TestCase):
    def setUp(self):
        self.accepted = json.loads(ACCEPTED.read_text())

    def test_checkpoint_hashes_equal_the_loader_pins(self):
        controller = self.accepted["controller"]
        self.assertEqual(controller["checkpoint_sha256"], pinned("CHECKPOINT_SHA"))
        self.assertEqual(controller["declaration_sha256"], pinned("DECLARATION_SHA"))

    def test_requirements_set_and_hash_match_the_file_on_disk(self):
        requirements = json.loads(REQUIREMENTS.read_text())
        self.assertIn(self.accepted["requirements_set"], requirements["sets"])
        self.assertEqual(
            self.accepted["requirements_sha256"],
            hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
            "the requirements changed; re-approve the accepted controller",
        )

    def test_limits_equal_the_requirement_rows(self):
        rules = json.loads(REQUIREMENTS.read_text())["sets"][self.accepted["requirements_set"]]
        judged = self.accepted["judged"]
        self.assertEqual(
            judged["stopping_distance_m"]["limit"], rules["controller"]["stopping_distance_max_m"]
        )
        self.assertEqual(
            judged["drift_per_hold_m"]["limit"], rules["controller"]["drift_per_hold_max_m"]
        )
        self.assertEqual(
            judged["stop_position_tolerance_m"]["limit"], rules["stop"]["position_tolerance_m"]
        )

    def test_judged_values_equal_their_evidence(self):
        judged = self.accepted["judged"]
        judgement = json.loads((ROOT / judged["stopping_distance_m"]["evidence"]).read_text())
        self.assertEqual(
            judgement["checkpoint_sha256"], [self.accepted["controller"]["checkpoint_sha256"]]
        )
        stop = next(c for c in judgement["cases"] if c["stop"] is not None)
        self.assertAlmostEqual(
            judged["stopping_distance_m"]["value"], stop["stop"]["final_distance_m"], places=4
        )
        worst = max(
            c["checks"]["drift_per_hold_m"]["value"]
            for c in judgement["cases"]
            if "drift_per_hold_m" in c["checks"]
        )
        self.assertAlmostEqual(judged["drift_per_hold_m"]["value"], worst, places=3)
        closed = json.loads((ROOT / judged["stop_position_tolerance_m"]["evidence"]).read_text())
        worst_stop = max(r["final_distance_to_last_goal_m"] for r in closed.values())
        self.assertAlmostEqual(judged["stop_position_tolerance_m"]["value"], worst_stop, places=3)
        self.assertFalse(any(r["fell"] for r in closed.values()))

    def test_pass_flags_follow_from_values_and_limits(self):
        for name, row in self.accepted["judged"].items():
            if isinstance(row["value"], (int, float)):
                self.assertEqual(row["pass"], row["value"] <= row["limit"], name)
                self.assertTrue(math.isfinite(row["value"]), name)

    def test_receipt_hash_matches_the_file(self):
        receipt = self.accepted["compatibility_receipt"]
        digest = hashlib.sha256((ROOT / receipt["path"]).read_bytes()).hexdigest()
        self.assertEqual(receipt["sha256"], digest)
