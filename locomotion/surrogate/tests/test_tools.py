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

from locomotion.spark.prepare import prepare
from locomotion.surrogate import replay
from locomotion.surrogate.env import ROOT, make_env
from locomotion.omni.task import TaskConfig, TrainingTask

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
                                 rewards=("c48_long",))
        env.close()
        with np.load(cls.output / "telemetry.npz") as file:
            cls.data = {key: file[key] for key in file.files}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_scripted_run_records_telemetry_and_rewards(self):
        self.assertEqual((self.summary["controls"], self.summary["replicas"]), (10, 2))
        self.assertEqual(self.data["action"].shape, (10, 2, 18))
        self.assertEqual(sorted(self.summary["reward_component_means"]), ["c48_long"])
        with np.load(self.output / "rewards.npz") as file:
            self.assertEqual(file["c48_long/reward"].shape, (10, 2))
        self.assertTrue(json.loads((self.output / "summary.json").read_text())["surrogate"])

    def test_stored_record_scores_the_same_reward_again(self):
        scores = replay.score_arrays(self.data, ["c48_long"])
        means = replay.reward_summary(scores)
        for name in ("c48_long",):
            self.assertEqual(scores[name]["reward"].shape, (10, 2))
            self.assertAlmostEqual(means[name]["reward"], self.summary["reward_component_means"][name]["reward"], places=6)

    def test_offline_scores_equal_the_live_task_rewards(self):
        # The live task steps the physics and the scorer replays its record; neither side reads the other.
        for name, task_class in (("c48_long", TrainingTask),):
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
        arguments = ["score", "--telemetry", str(self.output / "telemetry.npz"), "--rewards", "c48_long", "--output", str(target)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main(arguments), 0)
        with np.load(target.with_name("scores.npz")) as file:
            self.assertEqual(file["c48_long/reward"].shape, (10, 2))
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
