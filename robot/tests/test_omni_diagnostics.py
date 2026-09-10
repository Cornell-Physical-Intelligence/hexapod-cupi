"""Guard against a diagnostic hiding failures or changing the torque contract."""
from pathlib import Path
import ast
import __future__
import math
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from omni_diagnostics import (apply_omni_overrides, diagnostic_options,
                              diagnostic_scenarios, sample_masks, _summary, capture_step)
from omni_action_filter import filter_joint_targets


class DiagnosticContracts(unittest.TestCase):
    def cfg(self):
        return SimpleNamespace(processed_joint_target_slew_limit_rad_per_20ms=.06,
                               omni_reward_weights={"torque_excess": -.3, "linear_progress": 1.5,
                                                    "stand_joint_velocity": 0., "stand_target_velocity": 0.},
                               sim=SimpleNamespace(physics=SimpleNamespace()))

    def test_resets_restart_exclusion_and_terminal_samples_are_not_erased(self):
        age = np.array([[.02, 2.2], [2.2, 3.], [.02, 3.1]])
        terminal = np.array([[False, False], [True, False], [False, True]])
        masks = sample_masks(age, terminal, 2.)
        self.assertEqual(int(masks["all"].sum()), 6)
        self.assertEqual(int(masks["post_settle"].sum()), 4)
        self.assertEqual(int(masks["post_settle_nonterminal"].sum()), 2)
        self.assertFalse(masks["post_settle"][2, 0])
        self.assertTrue(masks["post_settle"][1, 0])

    def test_only_bounded_explicit_overrides_are_allowed(self):
        cfg = self.cfg()
        apply_omni_overrides(cfg, {"target_slew_rad_per_20ms": .03,
                                  "reward_weights": {"torque_excess": -.6, "linear_progress": 0}})
        self.assertEqual(cfg.processed_joint_target_slew_limit_rad_per_20ms, .03)
        self.assertEqual(cfg.omni_reward_weights["torque_excess"], -.6)
        for change in ({"torque_limit": 4}, {"target_slew_rad_per_20ms": .1},
                       {"reward_weights": {"unknown": 0}}, {"reward_weights": {"torque_excess": .3}},
                       {"reward_weights": {"torque_excess": float("nan")}},
                       {"reward_weights": {"stand_joint_velocity": .1}},
                       {"reward_weights": {"stand_target_velocity": -20}},
                       {"observation_noise_scale": -1}, {"target_filter_time_constant_s": .5},
                       {"external_forces_every_iteration": True}):
            with self.assertRaises(ValueError):
                apply_omni_overrides(self.cfg(), change)

    def test_target_filter_has_exact_time_constant_no_overshoot_and_zero_passthrough(self):
        previous = torch.zeros(1, 2)
        requested = torch.tensor([[.5, -.5]])
        one = filter_joint_targets(requested, previous, .02, .08)
        two = filter_joint_targets(requested, filter_joint_targets(requested, previous, .01, .08), .01, .08)
        self.assertTrue(torch.allclose(one, two))
        self.assertAlmostEqual(float(one[0, 0]), .5*(1-math.exp(-.25)), places=6)
        self.assertTrue(torch.all(one.abs() <= requested.abs()))
        self.assertIs(filter_joint_targets(requested, previous, .02, 0), requested)
        # Reset is represented by explicitly replacing previous filter state.
        reset_pose = torch.tensor([[.1, .2]])
        self.assertTrue(torch.allclose(filter_joint_targets(reset_pose, reset_pose, .02, .08), reset_pose))

    def test_filter_followed_by_existing_slew_remains_bounded_under_reversals(self):
        tree = ast.parse((Path(__file__).resolve().parents[2] / "isaaclab/hexapod_rl/env.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "limit_processed_joint_target_slew")
        namespace = {"torch": torch, "math": math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "limit_processed_joint_target_slew", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        limiter = namespace["limit_processed_joint_target_slew"]
        filtered = previous = torch.zeros(1, 2)
        for step in range(400):
            request = torch.full((1, 2), (.5 if step % 2 else -.5) if step < 100 else 0.)
            filtered = filter_joint_targets(request, filtered, .02, .08)
            next_target, _ = limiter(filtered, previous, torch.tensor([True]), max_delta_rad_per_20ms=.03, step_dt=.02)
            self.assertTrue(torch.all((next_target-previous).abs() <= .030001))
            previous = next_target
        self.assertLess(float(previous.abs().max()), 1e-6)

    def test_stand_motion_costs_are_zero_for_moving_commands_and_rest(self):
        # Load only the actual pure function from the Isaac-backed module.
        tree = ast.parse((Path(__file__).resolve().parents[2] / "tools/omni_flat_env.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "quiet_stand_terms")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "quiet_stand_terms", "exec"), namespace)
        call = namespace["quiet_stand_terms"]
        commands = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [0., 0., -.2]])
        terms = call(commands, torch.ones(3, 18), torch.full((3, 18), .02), torch.zeros(3, 18), .02)
        for value in terms.values():
            self.assertTrue(torch.allclose(value, torch.tensor([1., 0., 0.])))
        rest = call(torch.zeros_like(commands), torch.zeros(3, 18), torch.ones(3, 18), torch.ones(3, 18), .02)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in rest.values()))

    def test_diagnostic_duration_is_bounded_and_zero_probe_is_explicit(self):
        self.assertEqual(len(diagnostic_scenarios()), 12)
        self.assertEqual(diagnostic_options({})["controller"], "policy")
        self.assertEqual(diagnostic_options({"controller": "zero"})["controller"], "zero")
        for options in ({"duration_s": 300}, {"settle_s": 12}, {"controller": "reference"}, {"seed": .5}):
            with self.assertRaises(ValueError):
                diagnostic_options(options)

    def test_snapshot_survives_automatic_reset_and_keeps_failure_causes(self):
        # Identity quaternion makes both navigation conversions straightforward;
        # this tests the timing and copying contract without importing Isaac.
        proxy = lambda value: SimpleNamespace(torch=value)
        vector = torch.zeros(1, 3)
        joint = torch.zeros(1, 2)
        quaternion = torch.tensor([[1., 0., 0., 0.]])
        d = SimpleNamespace(root_pos_w=proxy(torch.tensor([[.02, 0., .12]])),
                            root_quat_w=proxy(quaternion), root_lin_vel_b=proxy(vector.clone()),
                            root_ang_vel_b=proxy(vector.clone()), projected_gravity_b=proxy(torch.tensor([[0., 0., -1.]])),
                            joint_pos=proxy(joint.clone()), joint_vel=proxy(joint.clone()),
                            computed_torque=proxy(torch.tensor([[6., 1.]])),
                            applied_torque=proxy(torch.tensor([[1.6, 1.]])))
        env = SimpleNamespace(_robot=SimpleNamespace(data=d), step_dt=.02, device="cpu", num_envs=1,
                              omni_diagnostic_start_position=torch.tensor([[0., 0., .12]]),
                              omni_diagnostic_start_quaternion=quaternion,
                              _base_contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=proxy(torch.zeros(1, 3, 1, 3)))),
                              cfg=SimpleNamespace(terminate_on_computed_torque_demand_duration_s=.1),
                              _episode_elapsed_s=torch.tensor([3.]), _commands=vector.clone(), omni_targets=vector.clone(),
                              _processed_actions=joint.clone(), _actions=joint.clone(),
                              reset_terminated=torch.tensor([True]), reset_time_outs=torch.tensor([False]),
                              _torque_demand_excess_duration_s=torch.tensor([.1]),
                              _joint_target_slew_limited_fraction=torch.zeros(1),
                              _vector_in_command_frame=lambda value: value)
        fake_math = SimpleNamespace(quat_apply_inverse=lambda q, value: value, quat_apply=lambda q, value: value)
        with patch.dict(sys.modules, {"isaaclab.utils.math": fake_math}):
            capture_step(env, {"torque": torch.ones(1)})
        d.computed_torque.torch.zero_()
        env.reset_terminated.zero_()
        env._episode_elapsed_s.zero_()
        snapshot = env.omni_diagnostic_sample
        self.assertEqual(float(snapshot["computed_torque_nm"][0, 0]), 6.)
        self.assertTrue(snapshot["terminated"][0])
        self.assertTrue(snapshot["reason_torque_duration"][0])
        self.assertAlmostEqual(float(snapshot["age_s"][0]), 3.02, places=5)
        self.assertEqual(float(snapshot["finite_difference_velocity_navigation_mps"][0, 0]), 1.)
        self.assertEqual(float(snapshot["finite_difference_heading_rate_rad_s"][0]), 0.)

    def test_report_preserves_runtime_names_and_distinguishes_applied_torque(self):
        # Runtime order is deliberately non-alphabetic. Reset-transient values
        # exceed steady values so mistaken pooling is visible.
        shape = (2, 1)
        zeros = np.zeros((*shape, 3))
        data = {key: zeros.copy() for key in ("velocity_navigation_mps", "finite_difference_velocity_navigation_mps",
                                              "gyro_navigation_rad_s", "command", "projected_gravity")}
        data["projected_gravity"][..., 2] = -1
        for key in ("joint_position_rad", "joint_target_rad", "joint_velocity_rad_s"):
            data[key] = np.zeros((*shape, 2))
        data["computed_torque_nm"] = np.array([[[40., 40.]], [[2., 1.]]])
        data["applied_torque_nm"] = np.array([[[1.6, 1.6]], [[1.6, 1.]]])
        data["reward_term_torque"] = np.ones(shape)
        summary = _summary(data, np.array([[False], [True]]), ["right_knee", "left_hip"])
        self.assertEqual(summary["computed_torque_abs_max_nm"], 2.)
        self.assertEqual(summary["applied_torque_abs_max_nm"], 1.6)
        self.assertEqual(summary["joints"]["right_knee"]["saturation_fraction"], 1.)
        self.assertEqual(summary["joints"]["left_hip"]["saturation_fraction"], 0.)
        self.assertEqual(_summary(data, np.zeros(shape, dtype=bool), ["a", "b"]), {"samples": 0})


if __name__ == "__main__":
    unittest.main()
