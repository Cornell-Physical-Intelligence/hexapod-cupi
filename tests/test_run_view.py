"""Check the viewer's run files: leg points, the ghost, the run's source and the shared test fixture.

The fixture viewer/test/fixtures/sample_run.json is the run file of the synthetic attempt below; the viewer's
node test reads it. The comparison decompresses the data block, since zlib versions may deflate differently. After a deliberate format change, rewrite it with
``uv run python -c "from tests.test_run_view import write_fixture; write_fixture()"``.
"""

import base64
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
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

    def test_a_judgement_for_another_checkpoint_cannot_label_a_trace(self):
        evaluation = self.attempt / "run/standing/evaluation"
        report_path = evaluation / "batch_000/report.json"
        report = json.loads(report_path.read_text())
        report_path.write_text(json.dumps({**report, "checkpoint_sha256": "a" * 64}))
        judgement = self.recorded_judgement()
        self.save_judgement({**judgement, "checkpoint_sha256": ["b" * 64]})
        with self.assertRaisesRegex(ValueError, "Judgement checkpoint differs"):
            run_view.build(self.attempt)

    def test_backend_uses_capture_metadata_and_historical_surrogate_markers(self):
        standalone = Path(self.directory.name) / "standalone"
        (self.attempt / "run/standing/evaluation").rename(standalone)
        self.assertEqual(run_view.build(standalone)["source"], "unknown")
        for name in ("cycle_summary.json", "surrogate_summary.json"):
            with self.subTest(marker=name):
                marker = standalone / name
                marker.write_text("{}")
                self.assertEqual(run_view.build(standalone)["source"], "surrogate")
                marker.unlink()
        source = standalone / "batch_000/native400hz/source.json"
        source.parent.mkdir()
        source.write_text(json.dumps({"backend": "mujoco 3.14.0"}))
        self.assertEqual(run_view.build(standalone)["source"], "surrogate")
        source.parent.rename(standalone / "capture_backup")
        parent = Path(self.directory.name) / "surrogate_parent"
        variant = parent / "contact_consensus/variant/evaluation"
        variant.parent.mkdir(parents=True)
        standalone.rename(variant)
        native = variant.parent / "native"
        native.mkdir()
        (native / "native_readback.json").write_text(
            json.dumps({"surrogate": {"engine": "mujoco 3.14.0"}})
        )
        self.assertEqual(run_view.build(variant)["source"], "surrogate")
        (native / "native_readback.json").unlink()
        (parent / "surrogate_summary.json").write_text("{}")
        self.assertEqual(run_view.build(variant)["source"], "surrogate")

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

    def test_preserved_v1_judgement_keeps_its_four_unjudged_cases(self):
        original = ROOT / "site/assets/c48_requirements_20261107_001/judgement.json"
        raw = original.read_bytes()
        judgement = json.loads(raw)
        evaluation = Path(self.directory.name) / "historical/evaluation"
        for index, row in enumerate(judgement["cases"]):
            batch(
                evaluation / f"batch_{index:03d}",
                row["case_id"],
                [row["command"]] * 2,
                [(0.0, 0.0, 0.0)] * 2,
                passed=True,
            )
        (evaluation / "judgement.json").write_bytes(raw)
        output = Path(self.directory.name) / "historical.json"
        with redirect_stdout(io.StringIO()) as printed:
            run_view.main([str(evaluation), "--output", str(output)])
        value = json.loads(output.read_text())
        self.assertEqual(
            [row["pass"] for row in value["cases"]],
            [row["pass"] for row in judgement["cases"]],
        )
        unjudged = [row for row in value["cases"] if row["pass"] is None]
        self.assertEqual(len(unjudged), 4)
        self.assertTrue(all(row["failed"] == [] for row in unjudged))
        self.assertIn("4 not judged", printed.getvalue())
        self.assertEqual(value["judgement_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(original.read_bytes(), raw)

    def test_null_verdict_requires_an_explicit_v1_verdict_without_checks_or_failures(self):
        for kind in ("v2", "failed_check", "passed_check", "failed_list", "missing_verdict"):
            with self.subTest(kind=kind):
                value = self.recorded_judgement()
                value["schema"] = "hexapod_walking_requirements_judgement_v1"
                row = value["cases"][0]
                row["pass"] = None
                if kind == "v2":
                    value["schema"] = "hexapod_walking_requirements_judgement_v2"
                elif kind in ("failed_check", "passed_check"):
                    passed = kind == "passed_check"
                    row["checks"] = {"tilt_deg": {"pass": passed}}
                    row["failed_checks"] = [] if passed else ["tilt_deg"]
                elif kind == "failed_list":
                    row["failed_checks"] = ["unrecorded"]
                else:
                    del row["pass"]
                self.save_judgement(value)
                with self.assertRaisesRegex(ValueError, "[Jj]udgement"):
                    run_view.build(self.attempt)

    def test_displayed_check_values_and_bounds_must_be_finite(self):
        for version in (1, 2):
            for field in ("value", "bound"):
                for number in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(version=version, field=field, number=number):
                        value = self.recorded_judgement()
                        value["schema"] = f"hexapod_walking_requirements_judgement_v{version}"
                        value["cases"][1]["checks"]["drift_per_hold_m"][field] = number
                        self.save_judgement(value)
                        with self.assertRaisesRegex(ValueError, "Nonfinite displayed check"):
                            run_view.build(self.attempt)

    def test_export_refuses_nonfinite_legacy_metrics_without_writing_a_file(self):
        report_path = self.attempt / "run/standing/evaluation/batch_000/report.json"
        report = json.loads(report_path.read_text())
        report["results"][0]["metrics"]["mean_yaw_rate_rad_s"] = float("inf")
        report_path.write_text(json.dumps(report))
        output = Path(self.directory.name) / "invalid.json"
        with self.assertRaisesRegex(ValueError, "Out of range float"):
            run_view.main([str(self.attempt), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_a_pass_from_another_run_cannot_label_a_short_or_terminal_stand(self):
        root = Path(self.directory.name)
        geometry, checkpoint = fixture_files(root)
        evaluations = [root / name / "evaluation" for name in ("pass", "terminal")]
        with patch.object(runner, "ExactEvaluationCapture", FakeCapture):
            for evaluation, terminate in zip(evaluations, (None, 2)):
                runner.run_cases(
                    FakeNative(terminate_at=terminate),
                    lambda observation: torch.zeros(1, 18),
                    evaluation,
                    geometry,
                    checkpoint,
                    case_ids=["stand"],
                )
        successful, failed = (folder / "judgement.json" for folder in evaluations)
        self.assertTrue(run_view.build(evaluations[0])["cases"][0]["pass"])
        self.assertFalse(run_view.build(evaluations[1])["cases"][0]["pass"])
        judgement = json.loads(successful.read_bytes())
        self.assertEqual(judgement["allocation"], json.loads(failed.read_bytes())["allocation"])
        failed.write_bytes(successful.read_bytes())
        with self.assertRaisesRegex(ValueError, "Judgement batch or trace identity differs"):
            run_view.build(evaluations[1])
        judgement["schema"] = "hexapod_walking_requirements_judgement_v2"
        for key in ("batch", "report_sha256", "control_trace_sha256"):
            del judgement["cases"][0][key]
        successful.write_text(json.dumps(judgement))
        self.assertTrue(run_view.build(evaluations[0])["cases"][0]["pass"])
        failed.write_text(json.dumps(judgement))
        with self.assertRaisesRegex(ValueError, "incomplete or terminal trace"):
            run_view.build(evaluations[1])

    def test_published_legacy_judgement_uses_its_own_archive_subgroup(self):
        root = Path(self.directory.name)
        evaluation = self.attempt / "run/standing/evaluation"
        path = self.save_judgement(self.recorded_judgement())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prefix = "contact_consensus/pyramidal_cone/evaluation"
        raw_files = {
            f"{prefix}/judgement.json": {"sha256": digest},
            "evaluation/batch_000/control_trace.npz": {"sha256": "0" * 64},
        }
        for batch_dir in sorted(evaluation.glob("batch_*"))[:2]:
            for name in ("report.json", "control_trace.npz"):
                raw = (batch_dir / name).read_bytes()
                raw_files[f"{prefix}/{batch_dir.name}/{name}"] = {
                    "sha256": hashlib.sha256(raw).hexdigest()
                }
        manifest = root / "site/assets/fixture/files.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"raw_files": raw_files}))
        with patch.object(run_view, "ROOT", root):
            archived = run_view.archived_batch_digests(digest)
        self.assertNotEqual(archived["batch_000/control_trace.npz"], "0" * 64)
        with patch.object(run_view, "archived_batch_digests", return_value=archived):
            self.assertEqual(run_view.build(evaluation)["judgement_binding"], "archive_hashes")
            report_path = evaluation / "batch_000/report.json"
            report_path.write_bytes(report_path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                ValueError, "Published judgement batch or trace identity differs"
            ):
                run_view.build(evaluation)

    def test_unindexed_legacy_replay_preserves_both_historical_tilt_conventions(self):
        evaluation = Path(self.directory.name) / "legacy/evaluation"
        folder = evaluation / "batch_000"
        case_id = "legacy:forward"
        forward = [0.05, 0.0, 0.0]
        batch(folder, case_id, [forward] * 4 + [[0.0, 0.0, 0.0]] * 2, [(0.0, 0.0, 0.0)] * 6)
        trace_path = folder / "control_trace.npz"
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = dict(archive)
        trace["root_pose_xyzw"][1, 0, 3:] = [np.sin(0.25), 0.0, 0.0, np.cos(0.25)]
        np.savez(trace_path, **trace)
        report_path = folder / "report.json"
        report = json.loads(report_path.read_bytes())
        report["cases"] = [
            {"case_id": case_id, "command": forward, "stop_control": 4, "controls": 6}
        ]
        report["files"] = {trace_path.name: hashlib.sha256(trace_path.read_bytes()).hexdigest()}
        report_path.write_text(json.dumps(report))
        from locomotion.evaluation.requirements import tilt_degrees

        whole_trace = float(tilt_degrees(trace["root_pose_xyzw"][:, 0]).max())
        for tilt, expected_pass in ((whole_trace, False), (0.0, True)):
            with self.subTest(tilt=tilt):
                row = {
                    "case_id": case_id,
                    "pass": expected_pass,
                    "checks": {"tilt_deg": {"value": tilt, "bound": 1.7, "pass": expected_pass}},
                }
                judgement = {"schema": "hexapod_walking_requirements_judgement_v2", "cases": [row]}
                path = evaluation / "judgement.json"
                for indent in (None, 2):
                    path.write_text(json.dumps(judgement, indent=indent))
                    exported = run_view.build(evaluation)
                    self.assertEqual(exported["judgement_binding"], "legacy")
                    self.assertIs(exported["cases"][0]["pass"], expected_pass)

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
