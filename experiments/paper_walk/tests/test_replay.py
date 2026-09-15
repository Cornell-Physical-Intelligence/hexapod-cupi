"""CPU orchestration checks; the mock is explicitly not simulation evidence."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.paper_walk import replay
from experiments.paper_walk.env_config import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "artifacts/restart_2026-09-14/paper_tripod_prior_001"


class MockCapture:
    def __init__(self, env):
        self.env, self.rows = env, []
        env.capture = self

    def close(self):
        self.env.capture = None


class MockEnv:
    """Deterministic ABI fixture only. Never used by the replay implementation."""
    def __init__(self, fail_at=None, stopped=False):
        self.num_envs, self.device = 32, "cpu"
        self.joint_names = list(JOINT_NAMES)
        self.neutral = torch.tensor([0., -.3, .4]*6)
        self.commands = torch.zeros(32, 3)
        self.cfg = SimpleNamespace(control_dt=.02, decimation=8, target_slew_rad=.04,
                                   action_scale_rad=.35, episode_seconds=20.)
        self.sim = SimpleNamespace(get_physics_step_count=lambda: self.counter)
        self.counter, self.controls, self.resets = 0, 0, 0
        self.capture, self.fail_at, self.stopped = None, fail_at, stopped

    def reset(self):
        self.resets += 1
        self.current = torch.zeros(32, 61)
        self.current[:, :18] = self.neutral
        self.current[:, 42] = .09
        return self._observations(self.current)

    def _observations(self, state):
        obs = torch.zeros(32, 231)
        obs[:, 210:213] = self.commands
        obs[:, :18] = state[:, :18]
        return {"amp": state.clone(), "obs": obs}

    def step(self, action):
        self.controls += 1
        target = self.neutral + .35*action
        self.current[:, :18] = target + .001
        self.current[:, 18:36] = self.controls*.0001
        self.current[:, 42] = .087 + self.controls*.000001
        if not self.stopped:
            self.current[:, 36] = self.commands[:, 1]*.5
            self.current[:, 37] = -self.commands[:, 0]*.5
            self.current[:, 41] = self.commands[:, 2]*.5
        for _ in range(8):
            self.counter += 1
            self.capture.rows.append({
                "joint_position_rad": self.current[:, :18].numpy().copy(),
                "joint_target_rad": target.numpy().copy(),
                "requested_torque_nm": np.zeros((32, 18), np.float32),
                "applied_torque_nm": np.zeros((32, 18), np.float32),
                "nonfoot_contact": np.zeros(32, bool),
                "fall_or_joint_violation": np.zeros(32, bool),
                "distal_contact": np.ones((32, 6), bool)})
        self.telemetry = {"joint_target_rad": target}
        result = self._observations(self.current)
        result["terminated"] = torch.zeros(32, dtype=torch.bool)
        if self.controls == self.fail_at:
            result["terminated"][0] = True
        result["truncated"] = torch.zeros(32, dtype=torch.bool)
        return result


class ReplayTests(unittest.TestCase):
    def run_replay(self, env, output):
        with patch.object(replay, "NativeReplayCapture", MockCapture):
            return replay.realize_prior(env, PRIOR/"tripod_prior.npz", PRIOR/"prior_metadata.json", output)

    def test_actual_pairs_phase_split_and_no_reset(self):
        env = MockEnv()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"replay"
            report = self.run_replay(env, output)
            self.assertTrue(report["all_commands_accepted"])
            self.assertEqual((env.resets, report["controls"], report["physics_steps"]), (1, 380, 3040))
            self.assertFalse(report["independent_heldout_variant"])
            with np.load(output/"replay_controls.npz") as raw:
                np.testing.assert_array_equal(raw["states"][1:], raw["next_states"][:-1])
                np.testing.assert_array_equal(raw["cycle_index"][:200], -1)
                self.assertFalse(np.array_equal(raw["next_states"][260, :, :18], raw["requested_joint_target_rad"][260]))
                with np.load(output/"realized_prior.npz") as train:
                    self.assertEqual(train["states"].shape, (1920, 61))
                    np.testing.assert_array_equal(train["states"], raw["states"][260:320].reshape(-1, 61))
                    np.testing.assert_array_equal(train["next_states"], raw["next_states"][260:320].reshape(-1, 61))
                    np.testing.assert_allclose(-train["next_states"][:, 37], train["commands"][:, 0]*.5)
                    np.testing.assert_array_equal(train["phase_index"], np.repeat(np.arange(60), 32))
                    np.testing.assert_array_equal(train["observations"][:, 210:213], train["commands"])
                    np.testing.assert_array_equal(train["observations"][:, :18], train["states"][:, :18])
                with np.load(output/"heldout_prior.npz") as heldout:
                    np.testing.assert_array_equal(heldout["states"], raw["states"][320:380].reshape(-1, 61))

    def test_motionless_attempt_preserved_but_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"replay"
            report = self.run_replay(MockEnv(stopped=True), output)
            self.assertTrue(report["complete"])
            self.assertFalse(report["all_commands_accepted"])
            self.assertTrue((output/"replay_controls.npz").is_file())
            self.assertFalse((output/"realized_prior.npz").exists())
            self.assertIn("commanded_motion_not_realized", report["replicas"][1]["rejected_reasons"])

    def test_fall_aborts_without_reset_or_expert_export(self):
        env = MockEnv(fail_at=205)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"replay"
            report = self.run_replay(env, output)
            self.assertFalse(report["complete"])
            self.assertEqual(env.resets, 1)
            self.assertEqual(report["controls"], 205)
            self.assertFalse((output/"realized_prior.npz").exists())
            with np.load(output/"replay_controls.npz") as raw:
                self.assertTrue(raw["terminated"][-1, 0])


if __name__ == "__main__":
    unittest.main()
