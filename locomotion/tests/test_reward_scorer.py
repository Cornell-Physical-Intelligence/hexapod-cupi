"""Offline reward scoring rebuilds training telemetry from a recorded native trace."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from locomotion.evaluation import reward_scorer as reward_scorer
from locomotion.backends.isaac.env import LocomotionEnv
from locomotion.tasks.task import REWARD_VERSION, TaskConfig, measured_reward

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
NOMINAL_HEIGHT = 0.09780231400684256


def edit_file(directory, overrides, *, version=REWARD_VERSION):
    path = Path(directory)/'reward_edit.json'
    path.write_text(json.dumps({'schema': reward_scorer.REWARD_CONFIG_SCHEMA, 'base_reward_version': version,
                                'overrides': overrides}))
    return path


class RewardConfigTests(unittest.TestCase):
    def test_exported_edit_scores_through_measured_reward(self):
        with tempfile.TemporaryDirectory() as temporary:
            reward = reward_scorer.load_reward_config(edit_file(temporary, {'yaw_tracking_weight': .5}))
        trace = reward_scorer.load_trace(TRACE)
        com = reward_scorer.root_com_local()
        edited, components = reward_scorer.evaluate(trace, reward, TaskConfig(), NOMINAL_HEIGHT, com)
        current, current_components = reward_scorer.evaluate(trace, measured_reward, TaskConfig(), NOMINAL_HEIGHT, com)
        torch.testing.assert_close(components['yaw_tracking'], current_components['yaw_tracking'] * (.5/.3))
        torch.testing.assert_close(edited - current, components['yaw_tracking'] - current_components['yaw_tracking'])

    def test_edits_that_training_would_reject_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'Positive finite'):
                reward_scorer.load_reward_config(edit_file(temporary, {'tilt_weight': 0}))
            with self.assertRaisesRegex(ValueError, 'Not reward coefficients'):
                reward_scorer.load_reward_config(edit_file(temporary, {'seed': 1}))
            with self.assertRaisesRegex(ValueError, 'not the current'):
                reward_scorer.load_reward_config(edit_file(temporary, {}, version='older_reward'))

    def test_command_line_names_each_reward_once(self):
        parser = argparse.ArgumentParser()
        reward_scorer.add_reward_arguments(parser)
        with tempfile.TemporaryDirectory() as temporary:
            path = edit_file(temporary, {'tilt_weight': .7})
            args = parser.parse_args(['--reward-config', f'tilt={path}'])
            self.assertEqual(list(reward_scorer.selected_rewards(parser, args)), [f'current ({REWARD_VERSION})', 'tilt'])
            args = parser.parse_args(['--reward-config', f'x={path}', '--reward', 'x=locomotion.tasks.task:measured_reward'])
            with self.assertRaises(SystemExit):
                reward_scorer.selected_rewards(parser, args)


class RewardScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = reward_scorer.load_trace(TRACE)
        cls.com = reward_scorer.root_com_local()

    def test_origin_velocity_matches_the_recorded_critic_observation(self):
        self.assertLess(reward_scorer.reconstruction_error(self.trace, self.com), 1e-6)

    def test_centre_of_mass_velocity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'critic observation'):
            reward_scorer.score_trace(self.trace, {'current': measured_reward}, TaskConfig(),
                                      NOMINAL_HEIGHT, [0., 0., 0.])

    def test_batched_rows_equal_rewards_from_recorded_observations(self):
        reward, _ = reward_scorer.evaluate(self.trace, measured_reward, TaskConfig(), NOMINAL_HEIGHT, self.com)
        data = self.trace.data
        rows = slice(1, -1)
        critic = torch.as_tensor(data['critic_observation'][2:, 0,
                                 LocomotionEnv.observation_width:LocomotionEnv.critic_width])
        telemetry = {'linear_velocity_nav': critic,
                     'angular_velocity_body': torch.as_tensor(data['gyro_body_rad_s'][rows, 0], dtype=torch.float32),
                     'root_pose_xyzw': torch.as_tensor(data['root_pose_xyzw'][rows, 0]),
                     'joint_velocity_rad_s': torch.as_tensor(data['joint_velocity_rad_s'][rows, 0]),
                     'joint_target_rad': torch.as_tensor(data['joint_target_rad'][rows, 0]),
                     'torque_square_sum_400hz': torch.as_tensor(data['applied_torque_squared_sum_400hz'][rows, 0]),
                     'other_body_force_max_400hz': torch.zeros(len(critic)),
                     'command': torch.as_tensor(data['command'][rows, 0])}
        expected, _ = measured_reward(telemetry, telemetry['command'], torch.as_tensor(data['joint_target_rad'][:-2, 0]),
                                      torch.as_tensor(data['terminated'][rows, 0]), TaskConfig(), NOMINAL_HEIGHT)
        torch.testing.assert_close(reward[:-1], expected, rtol=0, atol=1e-6)

    def test_motionless_counterfactual_gives_the_stationary_tracking_floor(self):
        _, components = reward_scorer.evaluate(self.trace, measured_reward, TaskConfig(), NOMINAL_HEIGHT,
                                               self.com, motionless=True)
        linear = math.exp(-.05**2/.0009)
        torch.testing.assert_close(components['linear_tracking'].double(),
                                   torch.full_like(components['linear_tracking'].double(), linear), rtol=0, atol=1e-6)
        torch.testing.assert_close(components['yaw_tracking'], torch.full_like(components['yaw_tracking'], .3))
        # docs/TRAINING.md: a motionless robot earns 28% of combined tracking at 0.05 m/s.
        combined = (components['linear_tracking'] + components['yaw_tracking']).double().mean() / 1.3
        self.assertAlmostEqual(float(combined), .28, places=2)

    def test_first_control_is_excluded_for_its_missing_previous_target(self):
        result = reward_scorer.score([TRACE], {'current': measured_reward}, NOMINAL_HEIGHT)
        self.assertEqual(result['traces'][0]['controls_scored'], len(self.trace.data['command']) - 1)

    def test_candidates_are_scored_and_ranked_beside_the_current_reward(self):
        def forward_only(telemetry, commands, previous_target, terminated, config, nominal_height):
            forward = telemetry['linear_velocity_nav'][:, 0]
            return forward, {'forward': forward}
        result = reward_scorer.score([TRACE, TRACE], {'current': measured_reward, 'forward': forward_only},
                                     NOMINAL_HEIGHT)
        group = result['traces'][0]['groups'][0]
        self.assertEqual(set(group['rewards']), {'current', 'forward'})
        self.assertAlmostEqual(group['rewards']['forward']['motionless']['mean'], 0., places=12)
        self.assertGreater(group['rewards']['forward']['recorded']['mean'], .04)
        self.assertEqual(set(result['ranking']), {'current', 'forward'})

    def test_reward_specification_requires_module_and_function(self):
        self.assertIs(reward_scorer.load_reward('locomotion.tasks.task:measured_reward'), measured_reward)
        with self.assertRaisesRegex(ValueError, 'MODULE:FUNCTION'):
            reward_scorer.load_reward('locomotion.tasks.task')

    def test_trace_without_a_required_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'control_trace.npz'
            np.savez(path, **{k: v for k, v in self.trace.data.items() if k != 'command'})
            with self.assertRaisesRegex(ValueError, 'command'):
                reward_scorer.load_trace(path)

    def test_command_line_writes_the_json_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'score.json'
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                reward_scorer.main([str(TRACE), '--nominal-height', str(NOMINAL_HEIGHT), '--json', str(output)])
            report = json.loads(output.read_text())
        self.assertEqual(report['schema'], reward_scorer.SCHEMA)
        self.assertEqual(report['declared_gaps'], list(reward_scorer.DECLARED_GAPS))
        self.assertIn('ranking by mean recorded reward', printed.getvalue())


if __name__ == '__main__':
    unittest.main()
