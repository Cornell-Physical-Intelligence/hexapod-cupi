"""Replay the pre-change fixed-action protocol: ``--randomization none`` must reproduce the Phase 0 oracle bit for bit.

The oracle ``step7_baseline_20261013_001`` was captured at main 7478fca8 before any Step 7 physics edit and stays
outside the checkout. ``fixtures/`` holds its state rows and protocol record; the test regenerates the actions
from the recorded seed and compares every state row, including the selected reset of replica 1 after control 40.
"""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from locomotion.surrogate.env import make_env

FIXTURES = Path(__file__).parent / "fixtures"
EXTRACT = json.loads((FIXTURES / "step7_baseline_20261013_001.json").read_text())
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replay(protocol):
    """Run the capture protocol: one state row after each control and after each reset call."""
    reset = protocol["selected_reset"]
    env = make_env(protocol["num_envs"], threads=protocol["threads"], seed=protocol["env_seed"])
    generator = torch.Generator().manual_seed(protocol["action_seed"])
    actions, states, resets = [], [], []
    try:
        env.reset()
        states.append(env._state.copy())
        for control in range(protocol["controls"]):
            action = protocol["action_std"] * torch.randn(protocol["num_envs"], 18, generator=generator)
            output = env.step(action)
            actions.append(action.numpy().copy())
            states.append(env._state.copy())
            done = output["terminated"] | output["truncated"]
            rows = sorted(set(done.nonzero().flatten().tolist()) | ({reset["row"]} if control == reset["after_control"] else set()))
            if rows:
                env.reset(rows)
                resets.append({"after_control": control, "rows": rows})
                states.append(env._state.copy())
        counters = {"physics_steps": env.physics_steps, "speed_clamp_events": env.speed_clamp_events,
                    "touchdown_events": env.touchdown_events, "overshoot_events": env.overshoot_events,
                    "peak_joint_speed": env.peak_joint_speed}
    finally:
        env.close()
    return np.stack(actions), np.stack(states), resets, counters


class OracleTests(unittest.TestCase):
    def test_none_reproduces_the_pre_change_trajectory_including_the_selected_reset(self):
        expected = np.load(FIXTURES / "step7_baseline_20261013_001.npz")["state_after_each_control_or_reset"]
        recorded = EXTRACT["arrays"]["state_after_each_control_or_reset"]
        self.assertEqual(sha(np.ascontiguousarray(expected).tobytes()), recorded["sha256"], "fixture changed")
        self.assertEqual(list(expected.shape), recorded["shape"])
        threads = torch.get_num_threads()
        torch.set_num_threads(EXTRACT["protocol"]["torch_threads"])
        try:
            actions, states, resets, counters = replay(EXTRACT["protocol"])
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(sha(np.ascontiguousarray(actions).tobytes()), EXTRACT["arrays"]["actions"]["sha256"])
        self.assertEqual(resets, EXTRACT["resets"])
        self.assertEqual(states.shape, expected.shape)
        differing = np.argwhere(states != expected)
        self.assertEqual(len(differing), 0, f"{len(differing)} state values differ; first at (row, replica, column) "
                                            f"{differing[0].tolist() if len(differing) else None}")
        self.assertEqual(counters, EXTRACT["counters"])


class PreservedBytesTests(unittest.TestCase):
    def test_surrogate_records_and_the_approved_model_keep_their_bytes(self):
        for name in ("locomotion/surrogate/records/self_check.json", "locomotion/surrogate/records/equivalence.json",
                     "robot/active_model.json"):
            with self.subTest(file=name):
                self.assertEqual(sha((ROOT / name).read_bytes()), EXTRACT["source_files"][name])


if __name__ == "__main__":
    unittest.main()
