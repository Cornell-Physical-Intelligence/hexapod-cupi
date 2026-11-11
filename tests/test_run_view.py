"""Check the viewer's run files: leg points, the ghost, the run's source and the shared test fixture.

The fixture viewer/test/fixtures/sample_run.json is the run file of the synthetic attempt below; the viewer's
node test reads it. The comparison decompresses the data block, since zlib versions may deflate differently. After a deliberate format change, rewrite it with
``uv run python -c "from tests.test_run_view import write_fixture; write_fixture()"``.
"""

import base64
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from locomotion.evaluation import runner
from locomotion.tests.test_evaluate import FakeCapture, FakeNative, fixture_files
from viewer import run_view

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "viewer/test/fixtures/sample_run.json"
STEP_M = 0.05 * 0.02


def batch(folder, case_id, command, poses, passed=True, failed=()):
    """One evaluation batch whose body takes ``poses`` (x, y, yaw after each control) under ``command``."""
    folder.mkdir(parents=True)
    n = len(poses)
    pose = np.zeros((n, 1, 7))
    for k, (x, y, heading) in enumerate(poses):
        pose[k, 0] = [x, y, 0.1, 0.0, 0.0, np.sin(heading / 2), np.cos(heading / 2)]
    toe = np.zeros((n, 1, 6, 3))
    toe[..., 2] = 0.01
    contact = np.zeros((n, 1, 6), bool)
    contact[:, 0, ::2] = True
    np.savez(
        folder / "control_trace.npz",
        root_pose_xyzw=pose,
        joint_position_rad=np.zeros((n, 1, 18)),
        toe_xyz_world_m=toe,
        distal_contact=contact,
        command=np.asarray(command, float)[:, None],
        terminated=np.zeros((n, 1), bool),
    )
    result = {
        "pass": passed,
        "failed_bounds": list(failed),
        "checks": {name: {"value": 0.1, "bound": 0.025} for name in failed},
        "metrics": {"mean_velocity_mps": [0.05, 0.0, 0.0], "mean_yaw_rate_rad_s": 0.0},
    }
    (folder / "report.json").write_text(
        json.dumps({"assigned_case_ids": [case_id], "results": [result]})
    )


def synthetic_attempt(root):
    """A native-layout attempt: a forward walk that follows its command, then a walk that stops."""
    evaluation = root / "attempt_fixture/run/standing/evaluation"
    forward = [[0.05, 0.0, 0.0]] * 6
    # A heading-zero body walks forward along world -y.
    batch(
        evaluation / "batch_000",
        "learning:forward",
        forward,
        [(0, -STEP_M * (k + 1), 0) for k in range(6)],
    )
    stop = [[0.05, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]] * 3
    poses = [(0, -STEP_M * min(k + 1, 3), 0) for k in range(6)]
    batch(evaluation / "batch_001", "learning:stop", stop, poses, False, ["planar_error_mps"])
    (evaluation / "batch_002").mkdir()
    (evaluation / "batch_002/report.json").write_text(
        json.dumps({"assigned_case_ids": ["learning:no_trace"], "results": [{"pass": True}]})
    )
    return root / "attempt_fixture"


def decoded(value):
    """The run file with its data block as the uncompressed bytes in hexadecimal."""
    return {**value, "data": gzip.decompress(base64.b64decode(value["data"])).hex()}


def write_fixture():
    with tempfile.TemporaryDirectory() as directory:
        value = run_view.build(synthetic_attempt(Path(directory)))
    FIXTURE.write_text(json.dumps(value, indent=1) + "\n")


class RunViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.attempt = synthetic_attempt(Path(self.directory.name))
        self.value = run_view.build(self.attempt)

    def tearDown(self):
        self.directory.cleanup()

    def test_the_run_file_matches_the_viewer_fixture(self):
        self.assertEqual(
            decoded(json.loads(json.dumps(self.value))), decoded(json.loads(FIXTURE.read_text()))
        )

    def test_a_body_that_follows_its_command_stays_on_the_ghost_and_a_new_command_restarts_it(self):
        forward, stop = self.value["cases"]
        self.assertEqual(forward["ghost_distance_mm"], [0.0, 0.0, 0.0])
        self.assertEqual(forward["ghost_distance_end_mm"], 0.0)
        # The ghost moves forward in navigation axes (forward, left) by 1 mm per control.
        self.assertEqual(forward["ghost_mm"], [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        self.assertEqual(stop["ghost_distance_mm"], [0.0, 0.0, 0.0])
        self.assertEqual(stop["command"][-1], [0.0, 0.0, 0.0])

    def test_leg_points_come_from_the_robot_model_in_navigation_axes(self):
        model = json.loads(
            (ROOT / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json").read_text()
        )
        coxa = next(j for j in model["joints"] if j["name"] == "lf_coxa_yaw")["xyz"]
        evaluation = self.attempt / "run/standing/evaluation"
        trace = np.load(evaluation / "batch_000/control_trace.npz")
        joints = {j["name"]: j for j in model["joints"]}
        points = run_view.leg_points(
            trace["root_pose_xyzw"][:, 0],
            trace["joint_position_rad"][:, 0],
            trace["toe_xyz_world_m"][:, 0],
            joints,
        )
        root = trace["root_pose_xyzw"][0, 0, :3]
        np.testing.assert_allclose(points[0, 0, 0], root + coxa, atol=1e-12)
        nav = run_view.navigation(points[0, 0, 0, :2])
        np.testing.assert_allclose(nav, [-(root[1] + coxa[1]), root[0] + coxa[0]], atol=1e-12)

    def test_the_run_names_its_source_attempt_and_untraced_cases(self):
        self.assertEqual((self.value["source"], self.value["name"]), ("native", "attempt_fixture"))
        self.assertEqual(self.value["cases_without_trace"], ["learning:no_trace"])
        self.assertEqual(
            self.value["cases"][1]["failed"],
            [{"check": "planar_error_mps", "value": 0.1, "bound": 0.025}],
        )
        surrogate = Path(self.directory.name) / "surrogate_output"
        surrogate.mkdir()
        (self.attempt / "run/standing/evaluation").rename(surrogate / "evaluation")
        (surrogate / "surrogate_summary.json").write_text("{}")
        value = run_view.build(surrogate)
        self.assertEqual((value["source"], value["name"]), ("surrogate", "surrogate_output"))

    def recorded_judgement(self):
        """Recorded verdicts differ from the old fixture reports and use a different order."""
        return {
            "schema": "hexapod_walking_requirements_judgement_v2",
            "cases": [
                {"case_id": "learning:stop", "pass": True, "checks": {}, "failed_checks": []},
                {
                    "case_id": "learning:forward",
                    "pass": False,
                    "checks": {"drift_per_hold_m": {"value": 0.2, "bound": 0.1, "pass": False}},
                    "failed_checks": ["drift_per_hold_m"],
                },
            ],
        }

    def save_judgement(self, value):
        path = self.attempt / "run/standing/evaluation/judgement.json"
        path.write_text(json.dumps(value))
        return path

    def test_judgement_verdicts_match_by_case_id_and_preserve_recorded_checks(self):
        for version in (1, 2):
            with self.subTest(version=version):
                judgement = self.recorded_judgement()
                judgement["schema"] = f"hexapod_walking_requirements_judgement_v{version}"
                if version == 1:
                    for row in judgement["cases"]:
                        del row["failed_checks"]
                path = self.save_judgement(judgement)
                before = path.read_bytes()
                value = run_view.build(self.attempt)
                self.assertEqual([row["pass"] for row in value["cases"]], [False, True])
                self.assertEqual(
                    value["cases"][0]["failed"],
                    [{"check": "drift_per_hold_m", "value": 0.2, "bound": 0.1}],
                )
                self.assertEqual(value["cases"][1]["failed"], [])
                self.assertEqual(value["judgement_sha256"], hashlib.sha256(before).hexdigest())
                self.assertEqual(path.read_bytes(), before)

    def test_a_present_judgement_cannot_fall_back_to_old_report_verdicts(self):
        for kind in (
            "missing",
            "duplicate",
            "nonboolean",
            "no_verdict",
            "checks",
            "failed",
            "schema",
        ):
            with self.subTest(kind=kind):
                value = self.recorded_judgement()
                if kind == "missing":
                    value["cases"].pop()
                elif kind == "duplicate":
                    value["cases"].append(dict(value["cases"][0]))
                elif kind == "nonboolean":
                    value["cases"][0]["pass"] = "false"
                elif kind == "no_verdict":
                    del value["cases"][0]["pass"]
                elif kind == "checks":
                    value["cases"][0]["checks"] = []
                elif kind == "failed":
                    value["cases"][0]["failed_checks"] = ["unrecorded"]
                else:
                    value["schema"] = "unknown"
                self.save_judgement(value)
                with self.assertRaisesRegex(ValueError, "[Jj]udgement"):
                    run_view.build(self.attempt)

    def test_judgement_failure_list_must_match_the_recorded_check_flags(self):
        for version in (1, 2):
            for failed in ([], ["drift_per_hold_m", "tilt_deg"]):
                with self.subTest(version=version, failed=failed):
                    value = self.recorded_judgement()
                    value["schema"] = f"hexapod_walking_requirements_judgement_v{version}"
                    row = value["cases"][1]
                    row["checks"]["tilt_deg"] = {"value": 0.0, "bound": 1.7, "pass": True}
                    row["failed_checks"] = failed
                    self.save_judgement(value)
                    with self.assertRaisesRegex(ValueError, "Judgement.*disagree"):
                        run_view.build(self.attempt)

    def test_judgement_verdict_must_agree_with_whether_checks_failed(self):
        for version in (1, 2):
            for index in (0, 1):
                with self.subTest(version=version, index=index):
                    value = self.recorded_judgement()
                    value["schema"] = f"hexapod_walking_requirements_judgement_v{version}"
                    row = value["cases"][index]
                    row["pass"] = not row["pass"]
                    if version == 1:
                        for case in value["cases"]:
                            del case["failed_checks"]
                    self.save_judgement(value)
                    with self.assertRaisesRegex(ValueError, "Judgement.*disagree"):
                        run_view.build(self.attempt)

    def test_current_evaluator_output_exports_its_recorded_failure(self):
        root = Path(self.directory.name)
        geometry, checkpoint = fixture_files(root)
        evaluation = root / "current/evaluation"
        with patch.object(runner, "ExactEvaluationCapture", FakeCapture):
            judgement = runner.run_cases(
                FakeNative(terminate_at=2),
                lambda observation: torch.zeros(1, 18),
                evaluation,
                geometry,
                checkpoint,
                case_ids=["stand"],
            )
        recorded = judgement["cases"][0]
        self.assertFalse(recorded["pass"])
        self.assertTrue(recorded["failed_checks"])
        report = json.loads((evaluation / "batch_000/report.json").read_text())
        self.assertNotIn("pass", report["results"][0])
        value = run_view.build(evaluation)
        row = value["cases"][0]
        self.assertEqual((row["case_id"], row["pass"]), (recorded["case_id"], recorded["pass"]))
        self.assertEqual(
            row["failed"],
            [
                {
                    "check": name,
                    "value": recorded["checks"][name]["value"],
                    "bound": recorded["checks"][name]["bound"],
                }
                for name in recorded["failed_checks"]
            ],
        )
        self.assertEqual(
            value["judgement_sha256"],
            hashlib.sha256((evaluation / "judgement.json").read_bytes()).hexdigest(),
        )
        (evaluation / "judgement.json").unlink()
        with self.assertRaisesRegex(ValueError, "judgement.json is required"):
            run_view.build(evaluation)


if __name__ == "__main__":
    unittest.main()
