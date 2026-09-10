import unittest
import numpy as np
from screen_metrics import physical_metrics, standing_screen, measured_flight_touchdowns, measured_progress


def trace(steps=1000, envs=1):
    return dict(computed_torque_nm=np.zeros((steps, envs, 18)), applied_torque_nm=np.zeros((steps, envs, 18)),
        coxa_contact=np.zeros((steps, envs, 6), bool), femur_contact=np.zeros((steps, envs, 6), bool),
        shaft_contact=np.zeros((steps, envs, 6), bool), base_contact=np.zeros((steps, envs), bool),
        distal_contact=np.ones((steps, envs, 6), bool), velocity_body_mps=np.zeros((steps, envs, 3)),
        position_world_m=np.tile([0., 0., .13], (steps, envs, 1)),
        rotation_world_from_body=np.tile(np.eye(3), (steps, envs, 1, 1)),
        velocity_world_mps=np.zeros((steps, envs, 3)),
        reference_point_world_m=np.zeros((steps, envs, 6, 3)),
        terminated=np.zeros((steps, envs), bool), truncated=np.zeros((steps, envs), bool),
        reference_to_executable_lag_rad=np.zeros((steps, envs, 18)),
        position_target_cast_error_rad=np.zeros((steps, envs, 18)))


class ScreenMetricTests(unittest.TestCase):
    def test_single_motor_saturation_cannot_hide_in_mean(self):
        data = trace(envs=32)
        data['computed_torque_nm'][200:210, 0, 7] = 1.7
        result = standing_screen(data)
        self.assertTrue(result['original_basic_standing_physics_pass'])
        self.assertAlmostEqual(result['post_settle_max_joint_saturation_fraction'], .0125)
        self.assertFalse(result['passed'])

    def test_any_termination_retained_even_during_settling(self):
        data = trace()
        data['terminated'][1, 0] = True
        self.assertFalse(standing_screen(data)['passed'])

    def test_numerical_target_cast_tolerance_does_not_allow_reference_lag(self):
        data = trace()
        data['position_target_cast_error_rad'][:] = 1e-7
        self.assertTrue(standing_screen(data)['passed'])
        data['reference_to_executable_lag_rad'][300, 0, 0] = 1e-5
        self.assertFalse(standing_screen(data)['passed'])

    def test_timed_touchdown_and_contact_glitch_cannot_count_as_completed_step(self):
        data = trace(steps=30)
        data['distal_contact'][5, 0, 0] = False # one-sample contact glitch
        data['distal_contact'][10:14, 0, 1] = False
        data['reference_point_world_m'][10:14, 0, 1, 2] = .003
        data['reference_point_world_m'][14:, 0, 1, 0] = .007
        data['distal_contact'][25:, 0, 2] = False # still missing at end
        events = measured_flight_touchdowns(data, start_step=1)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['leg_index'], 1)
        self.assertTrue(events[0]['confirmed_measured_touchdown'])
        self.assertAlmostEqual(events[0]['measured_reference_point_lift_m'], .003)
        self.assertAlmostEqual(events[0]['measured_reference_point_planar_step_m'], .007)
        self.assertFalse(events[1]['confirmed_measured_touchdown'])

    def test_unknown_and_nonfinite_values_do_not_pass(self):
        data = trace()
        data['computed_torque_nm'][1, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            physical_metrics(data)

    def test_actual_link_progress_requires_position_velocity_consistency(self):
        data = trace(steps=30)
        data['position_world_m'][:, 0, 1] = -np.arange(30)*.02*.005
        data['velocity_world_mps'][:, 0, 1] = -.005
        row = measured_progress(data, start_step=0, end_step=29)
        self.assertAlmostEqual(row['measured_forward_displacement_m'], .0029)
        self.assertLess(row['displacement_integral_difference_m'], 1e-12)
        data['position_world_m'][20:, 0, 1] -= .1
        row = measured_progress(data, start_step=0, end_step=29)
        self.assertAlmostEqual(row['displacement_integral_difference_m'], .1)


if __name__ == '__main__':
    unittest.main()
