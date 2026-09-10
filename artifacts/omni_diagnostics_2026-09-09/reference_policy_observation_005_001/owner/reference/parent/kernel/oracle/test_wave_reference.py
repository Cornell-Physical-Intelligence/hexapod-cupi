"""Synthetic kinematic/contact fixtures only; no physics pass is implied."""
import unittest
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from wave_reference import WaveContactReference, WaveConfig, SerialGeometry, tensor


class Fixture:
    def __init__(self, names=None, preload=.0):
        self.g = SerialGeometry()
        self.leg_names = tuple(n for leg in self.g.names for n in leg)
        self.names = tuple(names or self.leg_names)
        self.to_runtime = [self.leg_names.index(n) for n in self.names]
        self.to_leg = [self.names.index(n) for n in self.leg_names]
        q = self.g.q0.numpy().copy()
        self.qtarget = q.reshape(18)[self.to_runtime].copy()
        self.q = q.copy(); self.q[:, 1] -= preload
        self.p = np.array([0., 0., .13053251856352807]); self.R = np.eye(3)
        self.time = 0.; self.v = np.zeros(18); self.cmd = np.zeros(3)
        self.previous_feet = None; self.force_no_touchdown = False; self.force_never_flight = False
        feet = self.g.fk(tensor(self.q))[0].numpy() + self.p
        self.floor = feet[:, 2].copy()

    def snapshot(self):
        local = self.g.fk(tensor(self.q))[0].numpy()
        feet = (self.R @ local.T).T+self.p
        velocity = np.zeros((6, 3)) if self.previous_feet is None else (feet-self.previous_feet)/.02
        contact = feet[:, 2] < self.floor+1e-5
        if self.force_never_flight:
            contact[:] = True
        if self.force_no_touchdown and self.time > .2:
            contact[0] = False
        points = feet.copy(); points[~contact] = np.nan
        limits = np.stack((self.g.lower.numpy(), self.g.upper.numpy()), axis=-1).reshape(18, 2)[self.to_runtime]
        return dict(time_s=np.array([self.time]), position_world_m=self.p[None].copy(),
            quaternion_world_xyzw=Rotation.from_matrix(self.R).as_quat()[None], rotation_world_from_body=self.R[None].copy(),
            velocity_body_mps=np.array([[self.cmd[1], -self.cmd[0], 0.]]), gyro_body_rad_s=np.array([[0., 0., self.cmd[2]]]),
            joint_position_rad=self.q.reshape(18)[self.to_runtime][None].copy(), joint_target_rad=self.qtarget[None].copy(),
            executable_target_velocity_rad_s=self.v[None].copy(), soft_joint_pos_limits_rad=limits[None],
            reference_point_world_m=feet[None], reference_point_velocity_world_mps=velocity[None],
            contact_point_world_m=points[None], contact_point_valid=contact[None].copy(), distal_contact=contact[None].copy(),
            shaft_contact=np.zeros((1, 6), bool), coxa_contact=np.zeros((1, 6), bool), femur_contact=np.zeros((1, 6), bool),
            base_contact=np.array([False]), terminated=np.array([False]), truncated=np.array([False]))

    def advance(self, output):
        self.previous_feet = self.snapshot()['reference_point_world_m'][0].copy()
        self.qtarget = output['q_ref'][0].copy(); self.q = self.qtarget[self.to_leg].reshape(6, 3)
        self.v = output['v_ref'][0].copy(); self.time = output['target_time_s']
        self.p = output['state']['desired_position_world_m'].copy()
        self.R = output['state']['desired_rotation_world_from_body'].copy()
        self.cmd = output['admitted_command'].copy()


class WaveTests(unittest.TestCase):
    def test_initial_executed_target_preload_preserved_in_named_reverse_order(self):
        g = SerialGeometry(); names = tuple(reversed([n for leg in g.names for n in leg]))
        f = Fixture(names, preload=.01); r = WaveContactReference(names)
        initial = f.snapshot(); out = r.reset(initial)
        np.testing.assert_array_equal(out['q_ref'], initial['joint_target_rad'])
        self.assertGreater(np.max(np.abs(out['state']['initial_joint_preload_leg_major_rad'])), .009)
        out = r.step(initial, [0, 0, 0])
        self.assertTrue(out['valid'][0], out['failure_reason'])
        np.testing.assert_array_equal(out['q_ref'], initial['joint_target_rad'])

    def test_one_full_forward_wave_then_stop_preserves_budget_and_contacts(self):
        f = Fixture(); r = WaveContactReference(f.names); r.reset(f.snapshot())
        max_v = max_a = 0.; liftoffs_at_stop = None
        for k in range(1600):
            requested = [.005, 0, 0] if k < 1200 else [0, 0, 0]
            if k == 1200:
                liftoffs_at_stop = r.liftoffs
            out = r.step(f.snapshot(), requested)
            self.assertTrue(out['valid'][0], f'{k}: {out["failure_reason"]}')
            max_v = max(max_v, abs(out['v_ref']).max()); max_a = max(max_a, abs(out['a_ref']).max())
            f.advance(out)
        self.assertGreaterEqual(r.touchdowns, 6)
        self.assertEqual(r.liftoffs, liftoffs_at_stop)
        self.assertEqual(r.current_leg, None)
        self.assertLess(abs(out['admitted_command']).max(), 1e-6)
        np.testing.assert_array_equal(out['admitted_command'], [0., 0., 0.])
        self.assertEqual(out['state']['mode'], 'reference_quiet_hold')
        self.assertIsNotNone(out['state']['reference_quiet_time_s'])
        self.assertLessEqual(max_v, 1.75+1e-5); self.assertLessEqual(max_a, 6.+1e-5)

    def test_missing_touchdown_fails_and_emits_no_target(self):
        f = Fixture(); f.force_no_touchdown = True
        r = WaveContactReference(f.names); r.reset(f.snapshot())
        for _ in range(160):
            out = r.step(f.snapshot(), [.005, 0, 0])
            if not out['valid'][0]: break
            f.advance(out)
        self.assertFalse(out['valid'][0]); self.assertIsNone(out['q_ref'])
        self.assertIn('touchdown missing', out['failure_reason'])
        self.assertEqual(r.touchdowns, 0)

    def test_scheduled_motion_without_actual_flight_cannot_count_a_step(self):
        f = Fixture(); f.force_never_flight = True
        r = WaveContactReference(f.names); r.reset(f.snapshot())
        for _ in range(110):
            out = r.step(f.snapshot(), [.005, 0, 0])
            if not out['valid'][0]: break
            f.advance(out)
        self.assertFalse(out['valid'][0]); self.assertIn('Liftoff not observed', out['failure_reason'])
        self.assertEqual(r.touchdowns, 0)

    def test_separated_single_sample_contact_glitches_do_not_confirm_flight(self):
        f = Fixture(); f.force_never_flight = True
        r = WaveContactReference(f.names); r.reset(f.snapshot())
        for k in range(30):
            measured = f.snapshot()
            if k in (10, 20):
                measured['distal_contact'][0, 0] = False
            out = r.step(measured, [.005, 0, 0])
            self.assertTrue(out['valid'][0], out['failure_reason'])
            self.assertFalse(out['state']['flight_seen'])
            f.advance(out)
        self.assertEqual(r.flight_count, 0)

    def test_measured_support_loss_or_pose_convention_mismatch_rejected(self):
        f = Fixture(); r = WaveContactReference(f.names); r.reset(f.snapshot())
        measured = f.snapshot(); measured['distal_contact'][0, :2] = False
        out = r.step(measured, [.005, 0, 0]); self.assertFalse(out['valid'][0])
        self.assertIn('five measured support', out['failure_reason'])
        measured = f.snapshot(); measured['quaternion_world_xyzw'] = np.array([[1., 0., 0., 0.]])
        with self.assertRaises(ValueError):
            WaveContactReference(f.names).reset(measured)

    def test_actual_body_tracking_failure_not_hidden_by_virtual_trajectory(self):
        f = Fixture(); r = WaveContactReference(f.names); r.reset(f.snapshot())
        measured = f.snapshot(); measured['position_world_m'][0, 0] += .1
        out = r.step(measured, [0, 0, 0]); self.assertFalse(out['valid'][0])
        self.assertIn('Actual body', out['failure_reason'])

    def test_overbudget_reference_fails_instead_of_clipping(self):
        f = Fixture(); r = WaveContactReference(f.names, replace(WaveConfig(), reference_velocity_rad_s=.001)); r.reset(f.snapshot())
        for _ in range(20):
            out = r.step(f.snapshot(), [.005, 0, 0])
            if not out['valid'][0]: break
            f.advance(out)
        self.assertFalse(out['valid'][0]); self.assertIn('budget exceeded', out['failure_reason'])
        self.assertIsNone(out['q_ref'])

    def test_command_derating_preserves_arcs_and_is_explicit(self):
        f = Fixture(); r = WaveContactReference(f.names); r.reset(f.snapshot())
        out = r.step(f.snapshot(), [.05, 0, .10])
        self.assertTrue(out['valid'][0], out['failure_reason'])
        self.assertAlmostEqual(out['command_derating_factor'], .1)
        np.testing.assert_allclose(out['admitted_target_command'], [.005, 0, .01])


if __name__ == '__main__':
    unittest.main()
