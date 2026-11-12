"""The accepted-controller file cannot disagree with its evidence or its pins."""

import hashlib
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from tools import controller_evidence, project_site

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

    def evidence_fixture(self, root):
        """Copy only the selected records; mutations never touch the published evidence."""
        paths = {
            "configs/accepted_controller.json",
            "configs/walking_requirements.json",
            "robot/active_model.json",
            self.accepted["compatibility_receipt"]["path"],
        }
        for row in self.accepted["judged"].values():
            paths.update([row["evidence"]] if isinstance(row["evidence"], str) else row["evidence"])
        for name in paths:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())

    def repin(self, accepted, reference, path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in accepted["judged"].values():
            if row["evidence"] == reference:
                row["evidence_sha256"] = digest
            elif isinstance(row["evidence"], list) and reference in row["evidence"]:
                row["evidence_sha256"][row["evidence"].index(reference)] = digest

    def test_changed_registry_values_limits_verdicts_evidence_and_identities_reject(self):
        for name, row in self.accepted["judged"].items():
            mutations = {
                "value": row["value"] + " changed" if isinstance(row["value"], str) else 0.0,
                "pass": not row["pass"],
                "evidence": "site/assets/c48_surrogate_requirements_20261108_001/files.json",
                "evidence_sha256": ["0" * 64] * len(row["evidence_sha256"])
                if isinstance(row["evidence_sha256"], list)
                else "0" * 64,
            }
            if "limit" in row:
                mutations["limit"] = 100.0
            for field, changed in mutations.items():
                with self.subTest(row=name, field=field):
                    accepted = deepcopy(self.accepted)
                    accepted["judged"][name][field] = changed
                    with self.assertRaises(ValueError):
                        controller_evidence.validate(ROOT, accepted)
        for key in ("checkpoint_sha256", "declaration_sha256", "robot_model_sha256"):
            with self.subTest(identity=key):
                accepted = deepcopy(self.accepted)
                accepted["controller"][key] = "0" * 64
                with self.assertRaises(ValueError):
                    controller_evidence.validate(ROOT, accepted)
        for name, row in self.accepted["judged"].items():
            if not isinstance(row["value"], (int, float)):
                continue
            with self.subTest(fabricated_pass=name):
                accepted = deepcopy(self.accepted)
                accepted["judged"][name].update(value=0.0, **{"pass": True})
                with self.assertRaises(ValueError):
                    controller_evidence.validate(ROOT, accepted)

    def test_torque_projection_keeps_its_explicit_quiet_window_scope(self):
        for field, value in (("case_id", "learning:quiet_32s"), ("scope", "all_native_motion")):
            with self.subTest(field=field):
                accepted = deepcopy(self.accepted)
                accepted["judged"]["motor_torque_nm"][field] = value
                with self.assertRaises(ValueError):
                    controller_evidence.validate(ROOT, accepted)

    def test_evidence_semantics_still_validate_when_an_editor_updates_its_hash(self):
        mutations = {
            "checkpoint": lambda value: value.update(checkpoint_sha256=["0" * 64]),
            "requirements": lambda value: value.update(requirements_sha256="0" * 64),
            "duplicate case": lambda value: value["cases"][1].update(case_id="stand"),
            "missing tilt": lambda value: value["cases"][0]["checks"].pop("tilt_deg"),
            "wrong tilt bound": lambda value: value["cases"][0]["checks"]["tilt_deg"].update(
                bound=100
            ),
            "wrong tilt verdict": lambda value: value["cases"][0]["checks"]["tilt_deg"].update(
                {"pass": False}
            ),
            "nonfinite tilt": lambda value: value["cases"][0]["checks"]["tilt_deg"].update(
                value=float("nan")
            ),
            "missing route": lambda value: value["cases"].pop(),
        }
        reference = self.accepted["judged"]["tilt_deg"]["evidence"]
        original = json.loads((ROOT / reference).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.evidence_fixture(root)
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    accepted, value = deepcopy(self.accepted), deepcopy(original)
                    mutate(value)
                    (root / reference).write_text(json.dumps(value))
                    self.repin(accepted, reference, root / reference)
                    with self.assertRaises(ValueError):
                        controller_evidence.validate(root, accepted)

    def test_verdict_uses_the_unrounded_measurement(self):
        accepted = deepcopy(self.accepted)
        reference = accepted["judged"]["tilt_deg"]["evidence"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.evidence_fixture(root)
            value = json.loads((root / reference).read_text())
            for row in value["cases"]:
                row["checks"]["tilt_deg"].update(value=1.0, **{"pass": True})
            value["cases"][0]["checks"]["tilt_deg"].update(value=1.7004, **{"pass": False})
            (root / reference).write_text(json.dumps(value))
            self.repin(accepted, reference, root / reference)
            accepted["judged"]["tilt_deg"].update(value=1.7, **{"pass": False})
            controller_evidence.validate(root, accepted)
            accepted["judged"]["tilt_deg"]["pass"] = True
            with self.assertRaisesRegex(ValueError, "verdict differs: tilt_deg"):
                controller_evidence.validate(root, accepted)

    def test_publication_runs_the_same_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.evidence_fixture(root)
            accepted = deepcopy(self.accepted)
            accepted["judged"]["tilt_deg"].update(value=0.0, **{"pass": True})
            (root / "configs/accepted_controller.json").write_text(json.dumps(accepted))
            with (
                patch.object(project_site, "ROOT", root),
                patch.object(project_site, "registry", return_value={}),
                patch.object(project_site, "records", return_value=[]),
                self.assertRaisesRegex(ValueError, "value differs: tilt_deg"),
            ):
                project_site.check()

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
        routes = json.loads((ROOT / judged["stop_position_tolerance_m"]["evidence"]).read_text())
        self.assertEqual(routes["requirements_sha256"], self.accepted["requirements_sha256"])
        stops = [
            c["checks"]["stop_position_m"]
            for c in routes["cases"]
            if "route_m" in c and c["route_m"]
        ]
        self.assertEqual(len(stops), 5)
        worst_stop = max(s["value"] for s in stops)
        self.assertAlmostEqual(judged["stop_position_tolerance_m"]["value"], worst_stop, places=3)
        self.assertTrue(all(s["pass"] for s in stops))

    def test_pass_flags_follow_from_unrounded_evidence(self):
        controller_evidence.validate(ROOT, self.accepted)
        for name, row in self.accepted["judged"].items():
            if isinstance(row["value"], (int, float)):
                self.assertTrue(math.isfinite(row["value"]), name)

    def test_receipt_hash_matches_the_file(self):
        receipt = self.accepted["compatibility_receipt"]
        digest = hashlib.sha256((ROOT / receipt["path"]).read_bytes()).hexdigest()
        self.assertEqual(receipt["sha256"], digest)
