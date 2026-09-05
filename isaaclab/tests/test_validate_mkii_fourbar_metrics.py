"""Independent CPU checks of actual-pose closure and ground support metrics."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pxr import Gf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import mkii_fourbar_kinematics as kin
from audit_mkii_stance import _rotation

spec = importlib.util.spec_from_file_location('fourbar_validator_metrics_under_test', ROOT/'isaaclab/validate_mkii_fourbar.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def matrix_from_quat(q):
    """Independent normalized wxyz conversion for the fake SDK math module."""
    w, x, y, z = q.unbind(-1)
    s = 2./q.square().sum(-1)
    return torch.stack((1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
                        s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
                        s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def quaternion(rotation):
    q = Gf.Matrix3d(*map(float, rotation.T.flat)).ExtractRotation().GetQuat()
    return [q.GetReal(), *q.GetImaginary()]


class FakeRobot:
    def __init__(self, contract, *, tilt=False):
        self.contract = contract
        self.device = 'cpu'
        self.num_envs = 2
        self.origins = np.array([[4., -3., 1.2], [-2., 8., -.4]])
        self._terrain = types.SimpleNamespace(env_origins=torch.tensor(self.origins, dtype=torch.float32))
        self.names = list(reversed(contract['body_paths']))
        relative = kin.forward_kinematics(contract['joint_frames'], contract['default_joint_positions_rad'])
        self.poses = []
        for index in range(self.num_envs):
            world = np.eye(4)
            world[:3, :3] = _rotation(np.array([0., 0., 1.]), .63 if index else -.27)
            if tilt:
                world[:3, :3] = world[:3, :3] @ _rotation(np.array([0., 1., 0.]), -.24) @ _rotation(np.array([1., 0., 0.]), .13)
            world[:3, 3] = self.origins[index]+np.array([0., 0., .5 if tilt else contract['reset_root_height_m']])
            self.poses.append({name: world @ pose for name, pose in relative.items()})
        data = types.SimpleNamespace(
            joint_pos=torch.tensor([[contract['default_joint_positions_rad'][n] for n in contract['tree_joint_names']]]*self.num_envs),
            joint_vel=torch.zeros(self.num_envs, 30))
        self._robot = types.SimpleNamespace(body_names=self.names, data=data)
        self.sync()
        self._body_contact_sensors = {}
        for name in self.names:
            force = torch.zeros(self.num_envs, 1, 3)
            if name.endswith('_tibia'):
                force[:, :, 2] = 2.
            self._body_contact_sensors[name] = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=force))

    def sync(self):
        data = self._robot.data
        data.body_link_pos_w = torch.tensor(np.array([[poses[name][:3, 3] for name in self.names] for poses in self.poses]), dtype=torch.float32)
        data.body_link_quat_w = torch.tensor([[quaternion(poses[name][:3, :3]) for name in self.names] for poses in self.poses], dtype=torch.float32)
        data.root_pos_w = data.body_link_pos_w[:, self.names.index('body')]
        # Deliberately different COM fields: metrics must use the link frame.
        data.body_com_pos_w = data.body_link_pos_w+10.
        data.body_com_quat_w = torch.tensor([1., 0., 0., 0.]).expand_as(data.body_link_quat_w)

    def motor_state(self, name):
        return torch.full((self.num_envs, 18), .8 if name == 'applied_torque' else .9)

    def motor_telemetry(self, name):
        value = {'instantaneous_limit_nm': 1.2, 'continuous_limit_nm': 1.2, 'burst_headroom': 1., 'invalid_input': False}[name]
        return torch.full((self.num_envs, 18), value)

    def closure_coordinate_error(self):
        # The pose corruption tests keep encoder coordinates unchanged. Actual
        # pin residuals must catch errors that a q_D-q_A check cannot detect.
        return torch.zeros(self.num_envs, 12)


class FourbarPhysicalMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = kin.make_contract()
        cls.root = kin.load_model()[0]

    def capture(self, raw):
        metrics = validator.PhysicalMetrics(raw, self.contract)
        metrics.window = 'settled'
        sdk_math = types.ModuleType('isaaclab.utils.math')
        sdk_math.matrix_from_quat = matrix_from_quat
        with patch.dict(sys.modules, {'isaaclab.utils.math': sdk_math}):
            for _ in range(4):
                metrics.capture()
        metrics.drain()
        self.assertEqual(metrics.windows['settled']['substeps'], 4)
        return metrics.windows['settled']

    def report(self, window):
        return {'cpu_asset_pass': True, 'kit_asset_pass': True, 'body_count': 31,
                'joint_count': 30, 'active_motor_count': 18, 'steps_completed': 1,
                'steps_requested': 1, 'terminated_count': 0, 'truncated_count': 0,
                'reset_max_joint_error_rad': 0., 'anatomical_frame_pass': True,
                'physics_substeps': 4, 'driven_steps': 0, 'windows': {'settled': window}}

    def test_named_link_frames_and_translated_terrain_origins(self):
        raw = FakeRobot(self.contract)
        result = self.capture(raw)
        self.assertLess(result['max_closure_point_m'], 2e-6)
        self.assertLess(result['max_closure_axis_chord'], 2e-6)
        self.assertAlmostEqual(result['mean_height_m'], self.contract['reset_root_height_m'], places=6)
        self.assertEqual(result['min_support'], 6)
        self.assertEqual(result['nonfoot_contact_env_substeps'], 0)
        self.assertEqual(validator.grade(self.report(result)), [])

    def test_tilted_sphere_box_and_cylinder_support_matches_numpy_geometry(self):
        raw = FakeRobot(self.contract, tilt=True)
        result = self.capture(raw)
        rows = [row for i, poses in enumerate(raw.poses) for row in kin.collision_heights(self.root, poses, -raw.origins[i, 2])]
        expected = min(row['bottom_z_m'] for row in rows if not row['foot'])
        self.assertAlmostEqual(result['min_nonfoot_clearance_m'], expected, places=6)
        self.assertAlmostEqual(result['mean_height_m'], .5, places=6)

    def test_actual_one_mm_closure_error_rejected_with_unchanged_joint_positions(self):
        raw = FakeRobot(self.contract)
        raw.poses[1]['lf_tibia_pushrod'][0, 3] += .001
        raw.sync()
        result = self.capture(raw)
        self.assertAlmostEqual(result['max_closure_point_m'], .001, places=5)
        self.assertEqual(result['max_passive_relation_error_rad'], 0.)
        self.assertTrue(any('max_closure_point_m' in x for x in validator.grade(self.report(result))))

    def test_reversed_hinge_axis_rejected_even_when_both_pin_origins_match(self):
        raw = FakeRobot(self.contract)
        frame = self.contract['joint_frames']['lf_tibia_loop_closure']
        name = frame['body0']
        local = np.array(frame['body0_from_hinge_matrix'])
        pose = raw.poses[0][name]
        hinge = pose @ local
        perpendicular = np.cross(hinge[:3, 2], np.array([0., 0., 1.]))
        perpendicular /= np.linalg.norm(perpendicular)
        rotation = _rotation(perpendicular, math.pi) @ pose[:3, :3]
        pose[:3, :3] = rotation
        pose[:3, 3] = hinge[:3, 3]-rotation @ local[:3, 3]
        raw.sync()
        result = self.capture(raw)
        self.assertLess(result['max_closure_point_m'], 2e-6)
        self.assertAlmostEqual(result['max_closure_axis_chord'], 2., places=6)
        self.assertTrue(any('max_closure_axis_chord' in x for x in validator.grade(self.report(result))))

    def test_ground_penetration_rejected_without_any_reported_contact(self):
        raw = FakeRobot(self.contract)
        for pose in raw.poses[0].values():
            pose[2, 3] -= .05
        raw.sync()
        result = self.capture(raw)
        self.assertLess(result['max_closure_point_m'], 2e-6)
        self.assertLess(result['min_nonfoot_clearance_m'], 0.)
        self.assertEqual(result['nonfoot_contact_env_substeps'], 0)
        self.assertTrue(any('nonfoot geometry' in x for x in validator.grade(self.report(result))))

    def test_missing_driven_direction_result_cannot_admit_long_probe(self):
        report = self.report(self.capture(FakeRobot(self.contract)))
        report.update(steps_requested=1000, steps_completed=1000, physics_substeps=4000)
        self.assertTrue(any('Driven motor-coordinate' in x for x in validator.grade(report)))
        report['driven_coordinate_pass'] = False
        self.assertTrue(any('Driven motor-coordinate' in x for x in validator.grade(report)))

    def test_missing_or_failed_reset_and_anatomical_checks_are_rejected(self):
        baseline = self.report(self.capture(FakeRobot(self.contract)))
        for key, value in [('reset_max_joint_error_rad', 6e-6), ('anatomical_frame_pass', False)]:
            report = dict(baseline)
            report[key] = value
            self.assertTrue(any('Live reset or anatomical' in x for x in validator.grade(report)))
            del report[key]
            self.assertTrue(any('Live reset or anatomical' in x for x in validator.grade(report)))


if __name__ == '__main__':
    unittest.main()
