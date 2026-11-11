"""Native recording: uninterrupted trials, preserved prefixes and the judged case run."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import locomotion.evaluation.requirements as requirements
import locomotion.evaluation.runner as run
from locomotion.amp.features import extract_features
from locomotion.omniwalk.env_cfg import EnvConfig


class FakeNative:
    def __init__(self, terminate_at=None):
        self.num_envs = 1
        self.device = "cpu"
        self.cfg = EnvConfig(num_envs=1, episode_seconds=60.0)
        self.commands = torch.zeros(1, 3)
        self.current = {
            "toe_world": torch.zeros(1, 6, 3),
            "root": torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]),
        }
        self.counter = 0
        self.reset_count = 0
        self.capture = None
        self.native_errors = []
        self.terminate_at = terminate_at
        self.steps = 0
        self.sim = SimpleNamespace(get_physics_step_count=lambda: self.counter)

    def reset(self):
        self.reset_count += 1
        return self.observations(self.current)

    def observations(self, state):
        obs = torch.zeros(1, 231)
        obs[:, 210:213] = self.commands
        return {"obs": obs, "critic": torch.cat((obs, torch.zeros(1, 3)), -1)}

    def step(self, action):
        self.steps += 1
        self.counter += 8
        self.capture.count += 8
        return {
            **self.observations(self.current),
            "terminated": torch.tensor([self.steps == self.terminate_at]),
            "truncated": torch.zeros(1, dtype=torch.bool),
        }

    def verify_native_recipe(self, phase):
        pass


class MovingNative(FakeNative):
    """Measure every motion-feature field and move the joints 0.01 rad each control."""

    def __init__(self, terminate_at=None):
        super().__init__(terminate_at)
        self.current = {
            "q": torch.zeros(1, 18),
            "dq": torch.zeros(1, 18),
            "linear": torch.zeros(1, 3),
            "angular": torch.zeros(1, 3),
            "root": torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]),
            "toe_world": torch.zeros(1, 6, 3),
            "toe_body": torch.zeros(1, 6, 3),
        }

    def step(self, action):
        self.current = {**self.current, "q": self.current["q"] + 0.01}
        return super().step(action)


class FakeCapture:
    def __init__(self, env, output, geometry):
        self.env = env
        self.count = 0
        env.capture = self
        self.last_control = [{"distal_contact": np.ones((1, 6), bool)}]

    def control_record(self, result, command, control):
        n = 1
        root = np.zeros((n, 7))
        root[:, 2] = 0.1
        root[:, 6] = 1.0
        row = {
            "root_pose_xyzw": root,
            "time_s": np.array((control + 1) * 0.02),
            "command": np.asarray(command),
            "requested_command": np.asarray(command),
            "velocity_navigation_mps": np.zeros((n, 3)),
            "velocity_world_mps": np.zeros((n, 3)),
            "gyro_body_rad_s": np.zeros((n, 3)),
            "terminated": result["terminated"].numpy(),
            "truncated": result["truncated"].numpy(),
            "reset": np.zeros(n, bool),
            "minimum_non_toe_floor_m": np.full(n, 0.01),
        }
        for key in (
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_target_rad",
            "computed_torque_nm",
            "applied_torque_nm",
            "saturation_count_400hz",
        ):
            row[key] = np.zeros((n, 18))
        for key in (
            "applied_torque_abs_max_400hz",
            "nonfoot_contact",
            "nonfoot_contact_count_400hz",
            "missing_six_toe_count_400hz",
        ):
            row[key] = np.zeros(n)
        return row

    def close(self):
        self.env.capture = None
        return {
            "steps": self.count,
            "failure": None,
            "joint_bound_violation_steps": [0],
            "speed_bound_violation_steps": [0],
            "nonfoot_contact_steps_400hz": [0],
            "maximum_applied_nm": [0.0],
            "minimum_non_toe_floor_m": [0.01],
            "minimum_plate_height_m": [0.1],
        }


STAND = {"case_id": "stand", "command": [0.0, 0.0, 0.0], "controls": 1000}


def fixture_files(temporary):
    geometry = Path(temporary) / "geometry.npz"
    geometry.write_bytes(b"CPU fixture")
    checkpoint = Path(temporary) / "checkpoint"
    checkpoint.write_bytes(b"CPU fixture")
    return geometry, checkpoint


class RecordingTests(unittest.TestCase):
    def test_policy_evaluation_disables_autograd_and_restores_caller(self):
        native = FakeNative(terminate_at=2)
        parameter = torch.nn.Parameter(torch.ones(1, 18))
        observed = []

        def policy(observation):
            action = parameter * 0.0
            observed.append((torch.is_inference_mode_enabled(), action.requires_grad))
            return action

        with TemporaryDirectory() as temporary, torch.enable_grad():
            geometry, _ = fixture_files(temporary)
            with patch.object(run, "ExactEvaluationCapture", FakeCapture):
                report = run.run_batch(
                    native,
                    policy,
                    [STAND],
                    Path(temporary) / "run",
                    geometry,
                    checkpoint_sha256="CPU fixture",
                    source_sha256="CPU fixture",
                )
            self.assertEqual(report["controls"], 2)
            self.assertEqual(observed, [(True, False), (True, False)])
            self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())

    def test_failed_terminal_prefix_preserved_without_reset(self):
        native = FakeNative(terminate_at=2)
        with TemporaryDirectory() as temporary:
            geometry, _ = fixture_files(temporary)
            with patch.object(run, "ExactEvaluationCapture", FakeCapture):
                report = run.run_batch(
                    native,
                    lambda obs: torch.zeros(1, 18),
                    [STAND],
                    Path(temporary) / "run",
                    geometry,
                    checkpoint_sha256="CPU fixture",
                    source_sha256="CPU fixture",
                )
            self.assertEqual(native.reset_count, 1)
            self.assertEqual(native.steps, 2)
            self.assertEqual(report["controls"], 2)
            self.assertEqual(report["recorded_physics_steps"], 16)
            self.assertEqual(report["failure_kind"], "native_terminal_prefix")
            self.assertFalse(report["results"][0]["native_capture_complete"])
            with np.load(Path(temporary) / "run/control_trace.npz") as raw:
                self.assertTrue(raw["terminated"][1, 0])
                self.assertEqual(raw["reset"].sum(), 0)
                self.assertEqual(raw["policy_observation"].shape, (2, 1, 231))
                np.testing.assert_array_equal(
                    raw["policy_observation"][:, :, 210:213], raw["command"]
                )
                self.assertEqual(raw["toe_xyz_world_m"].shape, (2, 1, 6, 3))

    def test_command_holds_until_the_stop_control_then_zero(self):
        case = {"case_id": "walk", "command": [0.05, 0.0, 0.0], "controls": 4, "stop_control": 2}
        with TemporaryDirectory() as temporary:
            geometry, _ = fixture_files(temporary)
            with patch.object(run, "ExactEvaluationCapture", FakeCapture):
                report = run.run_batch(
                    FakeNative(),
                    lambda obs: torch.zeros(1, 18),
                    [case],
                    Path(temporary) / "run",
                    geometry,
                    checkpoint_sha256="CPU fixture",
                    source_sha256="CPU fixture",
                )
            self.assertTrue(report["acquisition_complete"])
            self.assertTrue(report["results"][0]["native_capture_complete"])
            with np.load(Path(temporary) / "run/control_trace.npz") as raw:
                np.testing.assert_array_equal(
                    raw["command"][:, 0], [[0.05, 0, 0], [0.05, 0, 0], [0, 0, 0], [0, 0, 0]]
                )

    def trace(self, **keywords):
        with TemporaryDirectory() as temporary:
            geometry, _ = fixture_files(temporary)
            with patch.object(run, "ExactEvaluationCapture", FakeCapture):
                report = run.run_batch(
                    MovingNative(terminate_at=2),
                    lambda obs: torch.zeros(1, 18),
                    [STAND],
                    Path(temporary) / "run",
                    geometry,
                    checkpoint_sha256="CPU fixture",
                    source_sha256="CPU fixture",
                    **keywords,
                )
            self.assertEqual(report["controls"], 2)
            with np.load(Path(temporary) / "run/control_trace.npz") as raw:
                return {key: raw[key] for key in raw.files}

    def test_motion_features_record_the_state_before_and_after_each_control(self):
        trace = self.trace(motion_features=extract_features)
        for key in ("amp_state_before", "amp_state_after"):
            self.assertEqual(trace[key].shape, (2, 1, 61))
        np.testing.assert_allclose(trace["amp_state_before"][:, 0, 0], [0.0, 0.01], atol=1e-7)
        np.testing.assert_allclose(trace["amp_state_after"][:, 0, 0], [0.01, 0.02], atol=1e-7)
        np.testing.assert_array_equal(trace["amp_state_after"][0], trace["amp_state_before"][1])
        np.testing.assert_allclose(trace["amp_state_before"][:, 0, 42], [0.1, 0.1], atol=1e-7)

    def test_trace_without_motion_features_holds_no_motion_state(self):
        trace = self.trace()
        self.assertFalse({"amp_state_before", "amp_state_after"} & set(trace))
        self.assertEqual(trace["policy_observation"].shape, (2, 1, 231))

    def test_native_capture_retains_mid_hold_events_and_com_frame(self):
        native = FakeNative()
        native.current = {"q": torch.zeros(1, 18)}
        native.geometry_meta = {}
        native.native_body_names = ["body"]
        native.sensor_map = []
        native.contact = SimpleNamespace(get_contact_data=lambda dt: [])
        native.native_readback = {
            "limits": np.tile([[-1.0, 1.0]], (1, 18, 1)).tolist(),
            "native_max_velocity": np.full((1, 18), 20.0).tolist(),
        }
        fake_geometry = SimpleNamespace(clearance=lambda poses: (np.zeros(1), np.full(1, 0.01)))
        root = np.array([[0.0, 0.0, 0.1, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], np.float32)
        state = {
            "q": torch.zeros(1, 18),
            "dq": torch.zeros(1, 18),
            "root": root,
            "root_velocity": np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.2]], np.float32),
            "link": root[:, None, :],
            "toe_world": torch.zeros(1, 6, 3),
            "toe_body": torch.zeros(1, 6, 3),
        }

        def classify(*args):
            feet = np.ones((1, 6), bool)
            if native.counter == 3:
                feet[0, 1] = False
            return {
                "distal_contact": feet,
                "distal_force_world": np.zeros((1, 6, 3)),
                "nonfoot_contact": np.array([native.counter == 5]),
                "nonfoot_force_world": np.zeros((1, 4, 3)),
                "patches": [],
            }

        with TemporaryDirectory() as temporary:
            geometry = Path(temporary) / "geometry.npz"
            np.savez(geometry, cloud=np.zeros(1))
            with (
                patch.object(run, "DiagnosticGeometry", return_value=fake_geometry),
                patch.object(run, "classify_patches", side_effect=classify),
            ):
                capture = run.ExactEvaluationCapture(native, Path(temporary) / "raw", geometry)
                for substep in range(8):
                    native.counter += 1
                    q = np.zeros((1, 18), np.float32)
                    target = q.copy()
                    if substep == 3:
                        target[0, 0] = 0.2
                    requested, applied, ceiling = run.diagnostic_servo(
                        q, q, target, np.full(18, 12.0, np.float32), np.asarray(run.KD, np.float32)
                    )
                    capture(
                        native,
                        state,
                        q,
                        q,
                        requested,
                        applied,
                        ceiling,
                        target,
                        None,
                        substep,
                        applied,
                    )
                result = {
                    "terminated": torch.zeros(1, dtype=torch.bool),
                    "truncated": torch.zeros(1, dtype=torch.bool),
                }
                row = capture.control_record(result, [[0.0, 0.0, 0.0]], 0)
                receipt = capture.close()
            self.assertEqual(row["missing_six_toe_count_400hz"].tolist(), [1])
            self.assertEqual(row["nonfoot_contact_count_400hz"].tolist(), [1])
            self.assertEqual(row["nonfoot_contact"].tolist(), [False])
            self.assertEqual(row["saturation_count_400hz"][0, 0], 1)
            self.assertAlmostEqual(row["computed_torque_abs_max_400hz"][0], 2.4, places=6)
            self.assertEqual(row["computed_torque_nm"].max(), 0.0)
            np.testing.assert_allclose(row["velocity_navigation_mps"], [[1.0, 2.0, 3.0]], atol=1e-6)
            self.assertEqual(receipt["steps"], 8)
            self.assertEqual(receipt["nonfoot_contact_steps_400hz"], [1])
            with np.load(Path(temporary) / "raw/substeps_000.npz") as saved:
                self.assertEqual(len(saved["sequence"]), 8)
                self.assertFalse(saved["distal_contact"][2, 0, 1])


class CaseRunTests(unittest.TestCase):
    def run_cases(self, native, temporary, **keywords):
        geometry, checkpoint = fixture_files(temporary)
        with patch.object(run, "ExactEvaluationCapture", FakeCapture):
            return run.run_cases(
                native,
                lambda obs: torch.zeros(1, 18),
                Path(temporary) / "run",
                geometry,
                checkpoint,
                **keywords,
            )

    def test_custom_requirement_set_reaches_the_route_follower(self):
        rules, _ = requirements.load_requirements()
        rules["commands"]["translate_speeds_mps"] = [0.025]
        rules["stop"]["routes_m"] = {"custom": [[0.2, 0.0]]}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "requirements.json"
            path.write_text(json.dumps({"sets": {"custom": rules}}))
            result = self.run_cases(
                FakeNative(),
                temporary,
                requirements_path=path,
                requirements_set="custom",
                case_ids=["route_custom"],
            )
            self.assertIsNone(result["failure"])
            with np.load(Path(temporary) / "run/batch_000/control_trace.npz") as trace:
                expected = np.tile(
                    np.array([0.025, 0.0, 0.0], dtype=trace["command"].dtype),
                    (len(trace["command"]), 1, 1),
                )
                np.testing.assert_array_equal(trace["command"], expected)

    def test_every_selected_case_is_recorded_and_judged_against_the_requirements_file(self):
        native = FakeNative()
        with TemporaryDirectory() as temporary:
            judgement = self.run_cases(native, temporary, case_ids=["stand", "walk_0.050_0deg"])
            saved = json.loads((Path(temporary) / "run/judgement.json").read_text())
            self.assertEqual(saved, judgement)
            self.assertEqual(native.reset_count, 2)
            self.assertEqual(native.steps, 2 * 1200)
            allocation = judgement["allocation"]
            self.assertEqual(allocation["requirements_sha256"], requirements.load_requirements()[1])
            self.assertEqual(allocation["judge_sha256"], run._sha(requirements.__file__))
            self.assertEqual(
                [c["case_id"] for c in allocation["cases"]], ["stand", "walk_0.050_0deg"]
            )
            self.assertIsNone(judgement["failure"])
            stand, walk = judgement["cases"]
            self.assertTrue(stand["pass"])
            self.assertEqual(set(stand["checks"]), {"tilt_deg", "applied_torque_nm"})
            # The fixture body never moves, so the walk drifts from its ghost and stops in place.
            self.assertEqual(walk["failed_checks"], ["drift_per_hold_m"])
            self.assertAlmostEqual(
                walk["checks"]["drift_per_hold_m"]["value"], 0.05 * 999 * 0.02, places=6
            )
            self.assertEqual(walk["stop"]["stop_control"], 1000)
            self.assertEqual(walk["checks"]["stopping_distance_m"]["value"], 0.0)
            self.assertEqual(judgement["failed_cases"], ["walk_0.050_0deg"])
            self.assertFalse(judgement["pass"])
            for index in range(2):
                self.assertTrue((Path(temporary) / f"run/batch_{index:03d}/report.json").exists())

    def test_invalid_selection_or_video_without_cameras_creates_no_output(self):
        with TemporaryDirectory() as temporary:
            for keywords in (
                {"case_ids": ["stand", "unknown"]},
                {"case_ids": ["stand", "stand"]},
                {"case_ids": []},
                {"video_case_id": "stand"},
                {"case_ids": ["stand"], "video_case_id": "walk_0.050_0deg"},
            ):
                with self.subTest(keywords=keywords), self.assertRaises(ValueError):
                    self.run_cases(FakeNative(), temporary, **keywords)
            self.assertFalse((Path(temporary) / "run").exists())

    def test_deadline_preserves_prefix_and_does_not_start_the_next_case(self):
        native = FakeNative()
        with TemporaryDirectory() as temporary:
            with patch.object(run.time, "monotonic", side_effect=lambda: float(native.steps)):
                judgement = self.run_cases(
                    native, temporary, case_ids=["stand", "walk_0.050_0deg"], max_wall_seconds=50.0
                )
            self.assertEqual(native.steps, 100)
            self.assertEqual(native.reset_count, 1)
            self.assertIn("allocation exhausted", judgement["failure"])
            self.assertEqual(len(judgement["cases"]), 1)
            self.assertEqual(judgement["cases"][0]["failed_checks"], ["complete_trace"])
            self.assertFalse(judgement["pass"])
            with np.load(Path(temporary) / "run/batch_000/control_trace.npz") as saved:
                self.assertEqual(len(saved["time_s"]), 100)

    def test_a_route_case_takes_its_commands_from_the_follower_and_judges_the_stop_position(self):
        native = FakeNative()
        with TemporaryDirectory() as temporary:
            rules = json.loads(requirements.REQUIREMENTS.read_text())
            rules["sets"]["stage2_flat"]["stop"]["routes_m"] = {"forward_0.5m": [[0.5, 0.0]]}
            path = Path(temporary) / "requirements.json"
            path.write_text(json.dumps(rules))
            judgement = self.run_cases(
                native, temporary, requirements_path=path, case_ids=["route_forward_0.5m"]
            )
            case = judgement["allocation"]["cases"][0]
            self.assertEqual(case["route_m"], [[0.5, 0.0]])
            self.assertEqual(case["controls"], round((0.5 / 0.025 + 20.0 + 4.0) / 0.02))
            self.assertEqual(native.steps, case["controls"])
            with np.load(Path(temporary) / "run/batch_000/control_trace.npz") as saved:
                # The fixture body never moves, so the follower keeps asking for the forward walk.
                np.testing.assert_allclose(
                    saved["command"][:, 0], [[0.05, 0.0, 0.0]] * case["controls"]
                )
            route = judgement["cases"][0]
            self.assertFalse(route["route"]["arrived"])
            self.assertAlmostEqual(route["route"]["final_distance_m"], 0.5, places=9)
            # A 44 s forward hold is also a drift case, and the static body drifts from its ghost.
            self.assertEqual(route["failed_checks"], ["drift_per_hold_m", "stop_position_m"])
            self.assertEqual(route["checks"]["stop_position_m"]["bound"], 0.12)

    def test_a_fall_ends_its_case_and_the_next_case_still_runs(self):
        native = FakeNative(terminate_at=2)
        with TemporaryDirectory() as temporary:
            judgement = self.run_cases(native, temporary, case_ids=["walk_0.050_0deg", "stand"])
            self.assertIsNone(judgement["failure"])
            self.assertEqual(native.reset_count, 2)
            self.assertEqual(native.steps, 2 + 1200)
            fallen, stand = judgement["cases"]
            self.assertEqual(fallen["checks"]["complete_trace"]["value"], 2)
            self.assertFalse(fallen["pass"])
            self.assertTrue(stand["pass"])
            self.assertEqual(judgement["failed_cases"], ["walk_0.050_0deg"])


if __name__ == "__main__":
    unittest.main()
