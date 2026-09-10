import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from physics_telemetry import *


def fixture():
    manifest = {'link_joint_mapping': {leg: {
        'joints': {part: f'{leg}_{part}_joint' for part in ('coxa', 'femur', 'tibia')},
        'links': {part: f'{leg}_{part}' for part in ('coxa', 'femur', 'tibia')},
    } for leg in LEGS}}
    names = tuple(f'{leg}_{part}_joint' for leg in LEGS for part in ('coxa', 'femur', 'tibia'))[::-1]
    bodies = tuple(sorted({'body_mock'} | {f'{leg}_{part}' for leg in LEGS for part in ('coxa', 'femur', 'tibia')}, reverse=True))
    sensors = []
    for leg in LEGS:
        sensors.append(SimpleNamespace(body_names=[f'{leg}_tibia'], data=SimpleNamespace(
            contact_pos_w=np.full((2, 1, 1, 3), np.nan), force_matrix_w=np.zeros((2, 1, 1, 3)))))
    sensors[0].data.contact_pos_w[:, 0, 0] = [1., 2., 0.]
    sensors[0].data.force_matrix_w[:, 0, 0] = [0., 0., 15.]
    q = Rotation.from_euler('z', [[90.], [-90.]], degrees=True).as_quat()
    data = SimpleNamespace(
        root_pos_w=np.array([[1., 2., .13], [3., 4., .13]]), root_quat_w=q,
        root_lin_vel_w=np.zeros((2, 3)), root_ang_vel_w=np.zeros((2, 3)),
        root_link_lin_vel_w=np.zeros((2, 3)), root_link_ang_vel_w=np.zeros((2, 3)),
        root_lin_vel_b=np.zeros((2, 3)), root_ang_vel_b=np.zeros((2, 3)),
        projected_gravity_b=np.tile([0., 0., -1.], (2, 1)),
        body_link_pos_w=np.ones((2, 19, 3)),
        body_link_quat_w=np.tile(q[:, None], (1, 19, 1)),
        body_link_lin_vel_w=np.tile([.2, .3, 0.], (2, 19, 1)),
        body_link_ang_vel_w=np.tile([0., 0., 2.], (2, 19, 1)),
        joint_pos=np.tile(np.arange(18), (2, 1)) * .01,
        soft_joint_pos_limits=np.tile([-1., 1.], (2, 18, 1)),
        joint_vel=np.zeros((2, 18)), computed_torque=np.zeros((2, 18)), applied_torque=np.zeros((2, 18)))
    distal = np.zeros((2, 6), bool); distal[:, 0] = True
    sensor = lambda count: SimpleNamespace(data=SimpleNamespace(net_forces_w_history=np.zeros((2, 3, count, 3))))
    env = SimpleNamespace(_robot=SimpleNamespace(joint_names=names, body_names=bodies, data=data),
                          _processed_actions=data.joint_pos.copy() + .02,
                          _feet_contact_sensors=sensors, _coxa_contact_sensor=sensor(6),
                          _femur_contact_sensors=[sensor(1) for _ in LEGS], _base_contact_sensor=sensor(1))
    env._get_foot_contact_state = lambda **kwargs: (distal, np.zeros_like(distal), np.zeros((2, 6)),
                                                   np.zeros((2, 6, 3)), np.zeros((2, 6, 3)), np.zeros((2, 6)))
    layout = named_layout(manifest, names, bodies, [s.body_names for s in sensors])
    return env, manifest, layout


class MeasuredTelemetryTests(unittest.TestCase):
    def test_sdk_xyzw_matches_independent_scipy_and_preserves_upright_yaw(self):
        rotation = Rotation.from_euler('xyz', [[12., -3., 10.], [0., 0., 90.]], degrees=True)
        np.testing.assert_allclose(rotation_xyzw(rotation.as_quat()), rotation.as_matrix(), atol=1e-14)
        np.testing.assert_allclose(rotation_xyzw(-rotation.as_quat()), rotation.as_matrix(), atol=1e-14)
        np.testing.assert_allclose(rotation_xyzw(rotation.as_quat())[1] @ [0., -1., 0.], [1., 0., 0.], atol=1e-14)

    def test_arbitrary_named_runtime_order_and_sensor_order_rejection(self):
        env, manifest, layout = fixture()
        sample = capture_measured_state(env, layout, np.zeros((6, 3)), time_s=.02)
        np.testing.assert_allclose(sample['joint_position_leg_major_rad'][0].ravel(), np.arange(18)[::-1]*.01)
        np.testing.assert_allclose(sample['joint_target_leg_major_rad'][0].ravel(), np.arange(18)[::-1]*.01+.02)
        with self.assertRaisesRegex(ValueError, 'sensor'):
            named_layout(manifest, env._robot.joint_names, env._robot.body_names,
                         [s.body_names for s in env._feet_contact_sensors][::-1])

    def test_actual_link_transform_and_angular_velocity_for_reference_point(self):
        env, _, layout = fixture()
        points = np.tile([0., .1, 0.], (6, 1))
        sample = capture_measured_state(env, layout, points, time_s=.02)
        np.testing.assert_allclose(sample['reference_point_world_m'][0, 0], [.9, 1., 1.], atol=1e-14)
        np.testing.assert_allclose(sample['reference_point_world_m'][1, 0], [1.1, 1., 1.], atol=1e-14)
        np.testing.assert_allclose(sample['reference_point_velocity_world_mps'][0, 0], [.2, .1, 0.], atol=1e-14)
        np.testing.assert_allclose(sample['reference_point_velocity_world_mps'][1, 0], [.2, .5, 0.], atol=1e-14)

    def test_unknown_contact_stays_unknown_and_arrays_do_not_alias_sdk(self):
        env, _, layout = fixture()
        sample = capture_measured_state(env, layout, np.zeros((6, 3)), time_s=.02)
        self.assertFalse(sample['contact_point_valid'][:, 1:].any())
        self.assertTrue(np.isnan(sample['contact_point_world_m'][:, 1:]).all())
        saved = sample['position_world_m'].copy()
        env._robot.data.root_pos_w[:] = 999.
        np.testing.assert_array_equal(sample['position_world_m'], saved)

    def test_unknown_contact_cannot_be_reclassified_as_support(self):
        env, _, layout = fixture()
        env._feet_contact_sensors[0].data.contact_pos_w[:] = np.nan
        with self.assertRaisesRegex(ValueError, 'unknown'):
            capture_measured_state(env, layout, np.zeros((6, 3)), time_s=.02)

    def test_nonfinite_physics_and_invalid_quaternion_rejected(self):
        env, _, layout = fixture()
        env._robot.data.computed_torque[1, 7] = np.nan
        with self.assertRaisesRegex(ValueError, 'computed_torque'):
            capture_measured_state(env, layout, np.zeros((6, 3)), time_s=.02)
        with self.assertRaisesRegex(ValueError, 'unit'):
            rotation_xyzw([0., 0., 0., 0.])

    def test_pre_reset_capture_preserves_predicate_once_and_final_failure_pose(self):
        env, _, layout = fixture()
        calls = []
        def original():
            calls.append(1)
            return np.array([True, False]), np.array([False, False])
        env._get_dones = original
        expected = env._robot.data.root_pos_w.copy()
        with pre_reset_capture(env, lambda: capture_measured_state(env, layout, np.zeros((6, 3)), time_s=.02)) as samples:
            term, trunc = env._get_dones()
            env._robot.data.root_pos_w[:] = 999. # mimic parent auto-reset after dones
            sample = require_single_pre_reset_sample(samples, 0, term, trunc)
            np.testing.assert_array_equal(sample['position_world_m'], expected)
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                require_single_pre_reset_sample(samples, -1, term, trunc)
        self.assertIs(env._get_dones, original)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
