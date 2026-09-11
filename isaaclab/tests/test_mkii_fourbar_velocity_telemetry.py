"""Physics-rate velocity observations, with independent moving-frame checks."""
from __future__ import annotations

import ast
import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_validate_mkii_fourbar_metrics as metrics_tests
from test_validate_mkii_fourbar_metrics import ROOT, FakeRobot, kin, matrix_from_quat, validator
from tools.assets.audit_mkii_stance import _rotation


class FourbarVelocityTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = kin.make_contract()

    def capture_context(self):
        module = types.ModuleType("isaaclab.utils.math")
        module.matrix_from_quat = matrix_from_quat
        return patch.dict(sys.modules, {"isaaclab.utils.math": module})

    def test_rotated_angular_lever_arms_and_common_translation(self):
        rotation = torch.eye(3, dtype=torch.float64).repeat(1, 2, 1, 1)
        rotation[0, 0] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        a, b = torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
        a[:3, 3], b[:3, 3] = torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.])
        frames = [[(0, a), (1, b)]]
        linear = torch.tensor([[[1., 3., 0.], [-2., 3., 0.]]], dtype=torch.float64)
        angular = torch.tensor([[[0., 0., 2.], [0., 0., -1.]]], dtype=torch.float64)
        # Both pins move (-1,3,0), despite different link-origin velocities.
        result = validator.closure_relative_point_velocities(rotation, linear, angular, frames)
        torch.testing.assert_close(result, torch.zeros(1, 1, 3, dtype=torch.float64), rtol=0, atol=0)
        angular[0, 1, 2] = -2.
        expected = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64)
        torch.testing.assert_close(validator.closure_relative_point_velocities(rotation, linear, angular, frames), expected)
        torch.testing.assert_close(validator.closure_relative_point_velocities(rotation, linear + 20., angular, frames), expected)

    def test_actual_cad_link_velocity_matches_differentiated_closed_motion_under_root_rotation(self):
        raw = FakeRobot(self.contract, tilt=True)
        names = raw._robot.joint_names
        q = self.contract["default_joint_positions_rad"]
        qd = {name: .3*math.sin(index+.4) for index, name in enumerate(names)}
        for name, relation in self.contract["passive_relations"].items():
            qd[name] = relation["multiplier"]*qd[relation["source_joint"]]
        epsilon = 1e-5
        poses_pm = [kin.forward_kinematics(self.contract["joint_frames"],
                    {name: q[name]+sign*epsilon*qd[name] for name in names}) for sign in (-1, 1)]
        omega = np.array([.2, -.1, .3])
        speed = np.array([.4, -.2, .1])
        linear, angular = [], []
        for env in range(raw.num_envs):
            root = raw.poses[env]["body"]
            worlds = []
            for sign in (-1, 1):
                world = root.copy()
                world[:3, :3] = _rotation(omega/np.linalg.norm(omega), sign*epsilon*np.linalg.norm(omega)) @ root[:3, :3]
                world[:3, 3] += sign*epsilon*speed
                worlds.append(world)
            linear_env, angular_env = [], []
            for name in raw.names:
                minus, plus = [world @ poses[name] for world, poses in zip(worlds, poses_pm)]
                linear_env.append((plus[:3, 3]-minus[:3, 3])/(2*epsilon))
                rotation_dot = (plus[:3, :3]-minus[:3, :3])/(2*epsilon)
                skew = rotation_dot @ raw.poses[env][name][:3, :3].T
                angular_env.append([skew[2, 1], skew[0, 2], skew[1, 0]])
            linear.append(linear_env)
            angular.append(angular_env)
        data = raw._robot.data
        data.body_link_lin_vel_w = torch.tensor(np.asarray(linear), dtype=torch.float32)
        data.body_link_ang_vel_w = torch.tensor(np.asarray(angular), dtype=torch.float32)
        data.joint_vel = torch.tensor([[qd[name] for name in names]]*raw.num_envs)
        metrics = validator.PhysicalMetrics(raw, self.contract)
        with self.capture_context():
            for _ in range(validator.DECIMATION):
                metrics.capture()
        metrics.drain()
        result = metrics.windows["startup"]
        self.assertLess(result["max_closure_relative_point_velocity_m_s"], 2e-7)
        self.assertLess(result["max_passive_velocity_relation_error_rad_s"], 1e-7)

    def test_shuffled_actual_joint_names_and_early_substep_spike_have_exact_window_rms(self):
        raw = FakeRobot(self.contract)
        raw._robot.joint_names.reverse()
        data = raw._robot.data
        data.joint_pos = data.joint_pos.flip(-1)
        metrics = validator.PhysicalMetrics(raw, self.contract)
        names = raw._robot.joint_names
        for window, amplitude in (("startup", 4.), ("settled", 2.), ("driven", 8.)):
            metrics.window = window
            with self.capture_context():
                for step in range(validator.DECIMATION):
                    data.joint_vel.zero_()
                    data.body_link_lin_vel_w.zero_()
                    if step == 0:
                        data.joint_vel[1, names.index("lf_tibia_rod_pivot")] = amplitude
                        # One endpoint of one C pin moves (3,4,0); all q stay at stance.
                        body = self.contract["joint_frames"]["lf_tibia_loop_closure"]["body0"]
                        data.body_link_lin_vel_w[1, raw.names.index(body)] = torch.tensor([3., 4., 0.])
                    metrics.capture()
            # Reduction tensors must retain the early spike after backing buffers change.
            data.joint_vel.zero_()
            data.body_link_lin_vel_w.zero_()
            metrics.drain()
            result = metrics.windows[window]
            self.assertEqual(result["max_passive_velocity_relation_error_rad_s"], amplitude)
            self.assertEqual(result["max_closure_relative_point_velocity_m_s"], 5.)
            self.assertEqual(result["passive_velocity_relation_samples"], validator.DECIMATION*2*12)
            self.assertEqual(result["closure_relative_point_velocity_samples"], validator.DECIMATION*2*6)
            self.assertAlmostEqual(result["rms_passive_velocity_relation_error_rad_s"], amplitude/math.sqrt(validator.DECIMATION*2*12))
            self.assertAlmostEqual(result["rms_closure_relative_point_velocity_m_s"], 5/math.sqrt(validator.DECIMATION*2*6))
            self.assertEqual(result["max_passive_relation_error_rad"], 0.)
        # Add a quiet policy step: peak persists and RMS uses the entire window.
        with self.capture_context():
            for _ in range(validator.DECIMATION):
                metrics.capture()
        metrics.drain()
        self.assertEqual(metrics.windows["driven"]["max_passive_velocity_relation_error_rad_s"], 8.)
        self.assertAlmostEqual(metrics.windows["driven"]["rms_passive_velocity_relation_error_rad_s"],
                               8/math.sqrt(2*validator.DECIMATION*2*12))
        self.assertEqual(metrics.velocity_telemetry_description["acceptance_use"], "observational_only_no_velocity_thresholds")
        report = metrics_tests.FourbarPhysicalMetricsTests().report(metrics.windows["settled"])
        self.assertEqual(validator.grade(report), [])  # New velocity values are not acceptance bounds.

    def test_velocity_signs_are_physical_relations_and_nonfinite_motion_is_reported(self):
        qd = torch.tensor([[2., 2., -2.], [2., 2.5, -1.25]])
        residual = validator.passive_joint_velocity_residuals(qd, [(1, 0, 1.), (2, 0, -1.)])
        torch.testing.assert_close(residual, torch.tensor([[0., 0.], [.5, .75]]))
        raw = FakeRobot(self.contract)
        raw._robot.data.body_link_ang_vel_w[0, raw.names.index("lf_tibia_pushrod"), 0] = float("nan")
        metrics = validator.PhysicalMetrics(raw, self.contract)
        with self.capture_context():
            for _ in range(validator.DECIMATION):
                metrics.capture()
        with self.assertRaisesRegex(ValueError, "Nonfinite physical metric"):
            metrics.drain()

    def test_both_report_paths_publish_observational_metadata_and_diagnostic_reuses_math(self):
        for filename in ("validate_mkii_fourbar.py", "diagnose_mkii_fourbar.py"):
            source = (ROOT/"isaaclab"/filename).read_text()
            tree = ast.parse(source)
            assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                           and ast.unparse(node.targets[0]) == "report['velocity_constraint_telemetry']"]
            self.assertEqual(len(assignments), 1)
            self.assertEqual(ast.unparse(assignments[0].value), "metrics.velocity_telemetry_description")
        source = (ROOT/"isaaclab/diagnose_mkii_fourbar.py").read_text()
        self.assertIn("closure_relative_point_velocities(rot, lin, ang, self.metrics.frames)", source)


if __name__ == "__main__":
    unittest.main()
