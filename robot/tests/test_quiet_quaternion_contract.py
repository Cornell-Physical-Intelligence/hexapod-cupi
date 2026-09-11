"""Actual capture-to-quiet scoring, using the installed SDK's XYZW convention."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from experiments.c_length_study.tools.omni_diagnostics import capture_step
from experiments.c_length_study.tools.omni_quiet_review import quiet_metrics, QUIET_GATES


class QuietQuaternionContractTests(unittest.TestCase):
    def capture(self, axis, degrees):
        count = 150
        angles = np.linspace(0., degrees, count)[:, None]
        raw = Rotation.from_euler(axis, angles, degrees=True).as_quat()
        proxy = lambda value: SimpleNamespace(torch=value)
        vector = torch.zeros(count, 3, dtype=torch.float64)
        joint = torch.zeros(count, 2, dtype=torch.float64)
        quaternion = torch.from_numpy(raw.copy())
        d = SimpleNamespace(root_pos_w=proxy(vector.clone()), root_quat_w=proxy(quaternion),
            root_lin_vel_b=proxy(vector.clone()), root_ang_vel_b=proxy(vector.clone()),
            projected_gravity_b=proxy(torch.tensor([0., 0., -1.]).expand(count, 3)),
            joint_pos=proxy(joint.clone()), joint_vel=proxy(joint.clone()),
            computed_torque=proxy(joint.clone()), applied_torque=proxy(joint.clone()))
        env = SimpleNamespace(_robot=SimpleNamespace(data=d), step_dt=.02, device="cpu", num_envs=count,
            omni_diagnostic_start_position=vector.clone(),
            omni_diagnostic_start_quaternion=torch.tensor([0., 0., 0., 1.]).expand(count, 4),
            _base_contact_sensor=SimpleNamespace(data=SimpleNamespace(
                net_forces_w_history=proxy(torch.zeros(count, 3, 1, 3)))),
            cfg=SimpleNamespace(terminate_on_computed_torque_demand_duration_s=.1),
            _episode_elapsed_s=torch.zeros(count), _commands=vector.clone(), omni_targets=vector.clone(),
            _processed_actions=joint.clone(), _actions=joint.clone(),
            reset_terminated=torch.zeros(count, dtype=torch.bool), reset_time_outs=torch.zeros(count, dtype=torch.bool),
            _torque_demand_excess_duration_s=torch.zeros(count), _joint_target_slew_limited_fraction=torch.zeros(count),
            _vector_in_command_frame=lambda value: value)
        def rotate(q, value, inverse=False):
            return torch.from_numpy(Rotation.from_quat(q.numpy()).apply(value.numpy(), inverse=inverse))
        fake_math = SimpleNamespace(quat_apply=rotate, quat_apply_inverse=lambda q, value: rotate(q, value, True))
        with patch.dict(sys.modules, {"isaaclab.utils.math": fake_math}):
            capture_step(env, {"nonfoot": torch.zeros(count)})
        sample = env.omni_diagnostic_sample
        # Capture copied the raw SDK data; later simulator writes cannot alter it.
        quaternion.zero_()
        np.testing.assert_array_equal(sample["quaternion_world_xyzw"], raw)
        np.testing.assert_array_equal(sample["quaternion_world_wxyz"], raw[:, [3, 0, 1, 2]])
        return {key: value[:, None, ...] for key, value in sample.items()}

    def test_actual_ten_degree_sdk_yaw_rejects_unchanged_two_degree_quiet_bound(self):
        data = self.capture("z", 10.)
        score = quiet_metrics(data, 0, 0, ["joint_b", "joint_a"], .02)
        self.assertEqual(QUIET_GATES["max_heading_excursion_deg"], 2.)
        self.assertAlmostEqual(score["max_heading_excursion_deg"], 10., places=10)
        self.assertFalse(score["pass"])
        self.assertIn("max_heading_excursion_deg", score["failed_bounds"])
        # The archived bug would falsely admit the same10-degree yaw as zero.
        legacy = dict(data, quaternion_world_wxyz=data["quaternion_world_xyzw"])
        self.assertAlmostEqual(quiet_metrics(legacy, 0, 0, ["joint_b", "joint_a"], .02)["max_heading_excursion_deg"], 0.)

    def test_roll_is_not_heading_and_quaternion_sign_does_not_change_heading(self):
        data = self.capture("x", 10.)
        self.assertAlmostEqual(quiet_metrics(data, 0, 0, ["joint_b", "joint_a"], .02)["max_heading_excursion_deg"], 0.)
        data = self.capture("z", -10.)
        original = quiet_metrics(data, 0, 0, ["joint_b", "joint_a"], .02)
        data["quaternion_world_wxyz"] *= -1
        flipped = quiet_metrics(data, 0, 0, ["joint_b", "joint_a"], .02)
        self.assertEqual(original["max_heading_excursion_deg"], flipped["max_heading_excursion_deg"])


if __name__ == "__main__":
    unittest.main()
