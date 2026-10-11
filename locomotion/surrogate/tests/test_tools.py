"""Check replay scoring, the calibration helpers and the boundary to the native pack."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from locomotion.prepare import prepare
from locomotion.surrogate import calibrate, replay, self_check
from locomotion.surrogate.env import ROOT, make_env
from locomotion.task import TaskConfig, TrainingTask
from locomotion.task_v2 import TrainingTaskV2

PACKAGE = ROOT / "locomotion/surrogate"
REMOTE = Path("/home/orionh/HEXAPOD_runs/restart_20260914/test_surrogate_boundary")


def reward_with_a_missing_field(telemetry, command, terminated):
    raise ValueError("Missing telemetry: absent_field")


def imported_modules(source, package="locomotion"):
    """Each absolute module name that a source file of ``package`` can bind through an import statement.

    ``from .x import y`` and ``from package import x`` name ``package.x`` as well as their own
    module, so a relative import of a subpackage counts.
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = ([package] if node.level else []) + ([node.module] if node.module else [])
            base = ".".join(parts)
            names.add(base)
            names.update(base + "." + alias.name for alias in node.names)
    return names


def forbidden_imports(source):
    """The MuJoCo and surrogate imports of one kernel source file."""
    return sorted(name for name in imported_modules(source)
                  if name.split(".")[0] == "mujoco" or (name + ".").startswith("locomotion.surrogate."))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "replay"
        env = make_env(2, threads=0, episode_seconds=60., substep_state=False)
        env.commands[:] = torch.tensor([.05, 0., 0.])
        generator = torch.Generator().manual_seed(3)
        cls.summary = replay.run(env, lambda control: .15 * torch.randn(2, 18, generator=generator), 10, cls.output,
                                 rewards=("v1", "v2"))
        env.close()
        with np.load(cls.output / "telemetry.npz") as file:
            cls.data = {key: file[key] for key in file.files}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_scripted_run_records_telemetry_and_rewards(self):
        self.assertEqual((self.summary["controls"], self.summary["replicas"]), (10, 2))
        self.assertEqual(self.data["action"].shape, (10, 2, 18))
        self.assertEqual(sorted(self.summary["reward_component_means"]), ["v1", "v2"])
        with np.load(self.output / "rewards.npz") as file:
            self.assertEqual(file["v2/reward"].shape, (10, 2))
        self.assertTrue(json.loads((self.output / "summary.json").read_text())["surrogate"])

    def test_stored_record_scores_the_same_reward_again(self):
        scores = replay.score_arrays(self.data, ["v1", "v2"])
        means = replay.reward_summary(scores)
        for name in ("v1", "v2"):
            self.assertEqual(scores[name]["reward"].shape, (10, 2))
            self.assertAlmostEqual(means[name]["reward"], self.summary["reward_component_means"][name]["reward"], places=6)

    def test_offline_scores_equal_the_live_task_rewards(self):
        # The live task steps the physics and the scorer replays its record; neither side reads the other.
        for name, task_class in (("v1", TrainingTask), ("v2", TrainingTaskV2)):
            env = make_env(4, threads=0, episode_seconds=60., substep_state=False)
            task = task_class(env, TaskConfig(seed=5))
            task.reset()
            generator = torch.Generator().manual_seed(11)
            records, rewards = [], []
            for _ in range(40):
                output = task.step(.3 * torch.randn(4, 18, generator=generator))
                self.assertFalse(bool((output["terminated"] | output["truncated"]).any()))
                records.append({key: value.numpy().copy() for key, value in env.telemetry.items()})
                rewards.append(output["reward"].numpy().copy())
            env.close()
            data = {key: np.stack([row[key] for row in records]) for key in records[0]}
            self.assertTrue(np.abs(data["command"]).max() > 0, name)
            scores = replay.score_arrays(data, [name])[name]["reward"]
            self.assertLess(float(np.abs(scores - np.stack(rewards)).max()), 1e-6, name)

    def test_score_command_keeps_a_fresh_output_name(self):
        target = Path(self.temporary.name) / "scores"
        arguments = ["score", "--telemetry", str(self.output / "telemetry.npz"), "--rewards", "v1", "--output", str(target)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main(arguments), 0)
        with np.load(target.with_name("scores.npz")) as file:
            self.assertEqual(file["v1/reward"].shape, (10, 2))
        with self.assertRaises(FileExistsError):
            replay.main(arguments)

    def test_player_rebuilds_the_recorded_toe_positions(self):
        player = replay.TelemetryPlayer(self.data)
        try:
            player.cursor = 4
            player.step(torch.from_numpy(self.data["action"][4]))
            self.assertTrue(np.allclose(player.current["toe_world"].numpy(), self.data["toe_xyz_world"][4], atol=1e-5))
            self.assertTrue(np.array_equal(player.telemetry["joint_target_rad"].numpy(), self.data["joint_target_rad"][4]))
        finally:
            player.close()

    def test_reward_components_name_a_reward_without_its_fields(self):
        scores, unscored = calibrate.reward_components(self.data, 2)
        self.assertEqual((sorted(scores), unscored), (sorted(calibrate.REWARDS), {}))
        self.assertEqual(scores["v1"]["reward"].shape, (8, 2))
        with mock.patch.object(calibrate, "REWARDS", ("v1", f"{__name__}:reward_with_a_missing_field")):
            scores, unscored = calibrate.reward_components(self.data, 2)
        self.assertEqual(sorted(scores), ["v1"])
        self.assertEqual(unscored, {f"{__name__}:reward_with_a_missing_field": "Missing telemetry: absent_field"})

    def test_action_traces_and_means_parse(self):
        probe = make_env(1, threads=0)
        trace = Path(self.temporary.name) / "trace.npz"
        target = probe.neutral.numpy() + .07 * np.ones((6, 18), dtype=np.float32)
        np.savez(trace, policy_action=np.zeros((6, 18), dtype=np.float32), joint_target_rad=target)
        actions, commands = replay.load_actions(trace, "policy_action", probe)
        self.assertEqual((actions.shape, commands), ((6, 1, 18), None))
        actions, _ = replay.load_actions(trace, "target", probe)
        probe.close()
        self.assertTrue(np.allclose(actions, .2, atol=1e-6))
        self.assertEqual(replay.parse_mean("0.5").tolist(), [.5] * 18)
        with self.assertRaises(ValueError):
            replay.parse_mean("1,2")


class CalibrationTests(unittest.TestCase):
    def test_native_noise_ranges_span_the_extract(self):
        def run(speed, linear):
            comp = {"vertical_velocity": -.1, "roll_pitch_rate": -.2, "joint_torque": -.1, "joint_acceleration": -.1,
                    "action_rate": -.2, "linear_tracking": linear, "yaw_tracking": .4}
            return {"aspd": speed, "sat": .002, "comp": comp, "u1": {"aspd": speed, "sat": .004, "comp": comp}}
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(calibrate.native_noise_ranges(temporary), {})
            (Path(temporary) / "native_noise_updates_6_20.json").write_text(json.dumps(
                {"A": run(.048, .20), "C": run(.045, .21), "D": run(.043, .08)}))
            ranges = calibrate.native_noise_ranges(temporary)
        self.assertEqual(ranges["updates_6_20"]["achieved_planar_speed_mps"], [.043, .048])
        self.assertEqual(ranges["updates_6_20"]["tracking_v2"]["linear_tracking"], [.20, .21])
        self.assertEqual(ranges["updates_6_20"]["tracking_v3"]["linear_tracking"], [.08, .08])
        self.assertEqual(ranges["tracking_runs"], {"v2": ["A", "C"], "v3": ["D"]})

    def test_ratio_guards_a_zero_reference(self):
        self.assertEqual((calibrate.ratio(1., 2.), calibrate.ratio(1., 0.)), (.5, None))

    def test_result_names_stay_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trial.json"
            self.assertEqual(calibrate.new_file(path), path)
            path.write_text("{}")
            with self.assertRaises(FileExistsError):
                calibrate.new_file(path)
            (Path(temporary) / "sweep_a_train").mkdir()
            with self.assertRaises(FileExistsError):
                calibrate.sweep(temporary, temporary, ["a=noslip_iterations=0"])
            with self.assertRaisesRegex(ValueError, "distinct"):
                calibrate.sweep(temporary, temporary, ["b=noslip_iterations=0", "b=noslip_iterations=1"])


class SelfCheckTests(unittest.TestCase):
    RECORD = {"shaft_contact_classification": {"pass": True},
              "training_smoke_v2": {"status": "completed", "expected_status": "completed"},
              "training_smoke_v4_without_clock": {"status": "failed", "expected_status": "failed"},
              "training_start_vs_native_run_A": {"run": {"status": "completed", "expected_status": "completed"}},
              "evaluation_smoke_learner_options": {"returncode": 0}, "total_mass_kg": 7.4}

    def test_a_record_that_holds_each_check_has_no_failure(self):
        self.assertEqual(self_check.failures(self.RECORD), [])
        self.assertEqual(self_check.failures({"shaft_contact_classification": {"pass": True}}), [])

    def test_each_failed_check_reaches_the_failure_list(self):
        self.assertEqual(self_check.failures({}), ["shaft_contact_classification"])
        for key, value, name in (
                ("shaft_contact_classification", {"pass": False, "reason": "none"}, "shaft_contact_classification"),
                ("training_smoke_v2", {"status": "failed", "expected_status": "completed"}, "training_smoke_v2"),
                ("training_smoke_v4_without_clock", {"status": "completed", "expected_status": "failed"},
                 "training_smoke_v4_without_clock"),
                ("training_start_vs_native_run_A", {"run": {"status": "failed", "expected_status": "completed"}},
                 "training_start_vs_native_run_A.run"),
                ("evaluation_smoke_learner_options", {"returncode": 1, "error": "x"}, "evaluation_smoke_learner_options")):
            self.assertEqual(self_check.failures({**self.RECORD, key: value}), [name])

    def test_messages_hold_no_machine_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            text = self_check.portable(f"FileExistsError('{temporary}/smoke') in {ROOT}/locomotion/surrogate/train.py", temporary)
        self.assertEqual(text, "FileExistsError('<output>/smoke') in <repository>/locomotion/surrogate/train.py")

    def test_shaft_contact_reaches_the_repository_classifier(self):
        report = self_check.shaft_contact()
        self.assertTrue(report["pass"], report)
        self.assertIn("shaft", report["patch_categories_with_force"])
        self.assertEqual(report["toe_cap_force_n"], 0.)


class BoundaryTests(unittest.TestCase):
    def test_native_pack_holds_no_surrogate_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepare(Path(temporary) / "ppo", REMOTE, mode="train")
            frozen = json.loads((Path(temporary) / "ppo/source/FREEZE_SHA256.json").read_text())
        self.assertTrue(frozen)
        self.assertFalse([name for name in frozen if "surrogate" in name])

    def test_import_scan_reads_relative_and_absolute_forms(self):
        for statement in ("from .surrogate import env", "from . import surrogate", "from .surrogate.env import make_env",
                          "from locomotion import surrogate", "import locomotion.surrogate.env as e",
                          "from locomotion.surrogate import train", "import mujoco", "from mujoco import rollout",
                          "import mujoco.rollout"):
            self.assertTrue(forbidden_imports(statement), statement)
        for statement in ("from . import env", "from .task import TaskConfig", "from locomotion import task_v2",
                          "import mujoco_like", "from .surrogates import x", "import torch"):
            self.assertEqual(forbidden_imports(statement), [], statement)

    def test_kernel_imports_neither_the_surrogate_nor_mujoco(self):
        paths = sorted((ROOT / "locomotion").glob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(forbidden_imports(path.read_text()), [], path.name)

    def test_package_holds_no_absolute_user_path(self):
        for path in sorted(PACKAGE.glob("*.py")):
            text = path.read_text()
            self.assertNotIn("/Users/", text, path.name)
            self.assertNotIn("/private/tmp", text, path.name)
            self.assertNotIn("sys.path", text, path.name)


if __name__ == "__main__":
    unittest.main()
