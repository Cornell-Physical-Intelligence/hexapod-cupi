"""The workbench's live arithmetic equals measured_reward, and exported edits score through it."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

from locomotion import reward_scorer, reward_workbench
from locomotion.task import REWARD_VERSION, TaskConfig, measured_reward

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
NOMINAL_HEIGHT = 0.09780231400684256
EDITED = {'linear_tracking_weight': 1.7, 'linear_error_variance': .0004, 'yaw_tracking_weight': .5,
          'yaw_error_variance': .01, 'quiet_joint_rate_weight': .2, 'quiet_joint_rate_scale_rad_s': .05,
          'quiet_target_step_weight': .07, 'quiet_target_step_scale_rad': .004, 'tilt_weight': .9,
          'angular_xy_weight': .08, 'vertical_velocity_weight': 1., 'height_weight': 2.,
          'normalized_effort_weight': .03, 'target_movement_weight': .02, 'nonfoot_event_weight': .5,
          'nonfoot_force_weight': .1, 'terminal_penalty': 3.}


def edit_file(directory, overrides, *, version=REWARD_VERSION):
    path = Path(directory)/'reward_edit.json'
    path.write_text(json.dumps({'schema': reward_scorer.REWARD_CONFIG_SCHEMA, 'base_reward_version': version,
                                'overrides': overrides}))
    return path


class RewardWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trace = reward_scorer.load_trace(TRACE)
        telemetry, previous, terminated = reward_scorer.reward_inputs(trace, reward_scorer.root_com_local())
        # Standing commands on alternate rows exercise the quiet terms and their logarithmic tail.
        telemetry['command'] = telemetry['command'].clone()
        telemetry['command'][::2] = 0
        telemetry['other_body_force_max_400hz'] = torch.linspace(0, 3, len(terminated))
        terminated = terminated.clone()
        terminated[-3:] = True
        cls.inputs = telemetry, previous, terminated

    def assert_recombination_matches(self, config):
        telemetry, previous, terminated = self.inputs
        _, expected = measured_reward(telemetry, telemetry['command'], previous, terminated, config, NOMINAL_HEIGHT)
        actual = reward_workbench.recombine(reward_workbench.features(telemetry, previous, terminated, NOMINAL_HEIGHT), config)
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value.double().numpy(), rtol=1e-5, atol=1e-6, err_msg=name)

    def test_recombination_equals_measured_reward_at_current_coefficients(self):
        self.assert_recombination_matches(TaskConfig())

    def test_recombination_equals_measured_reward_under_edited_coefficients(self):
        self.assert_recombination_matches(replace(TaskConfig(), **EDITED))

    def test_standing_rows_reach_the_logarithmic_tail(self):
        telemetry, previous, terminated = self.inputs
        f = reward_workbench.features(telemetry, previous, terminated, NOMINAL_HEIGHT)
        u = f['joint_rate_ms'][f['quiet'] == 1] / TaskConfig().quiet_joint_rate_scale_rad_s**2
        self.assertTrue((u > 1).any())

    def test_every_term_and_coefficient_is_described(self):
        telemetry, previous, terminated = self.inputs
        _, components = measured_reward(telemetry, telemetry['command'], previous, terminated, TaskConfig(), NOMINAL_HEIGHT)
        self.assertEqual(set(reward_workbench.TERMS), set(components))
        described = {field for *_, fields in reward_workbench.TERMS.values() for field in fields}
        self.assertEqual(described, set(reward_scorer.REWARD_FIELDS))

    def test_page_embeds_python_rewards_for_its_self_check(self):
        html = reward_workbench.build_page([TRACE], NOMINAL_HEIGHT)
        payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
        trace = payload['traces'][0]
        recombined = reward_workbench.recombine({k: np.array(v) for k, v in trace['recorded'].items()}, TaskConfig())
        np.testing.assert_allclose(sum(recombined.values()), trace['python_reward'], atol=1e-5)
        self.assertEqual(payload['defaults'], {f: getattr(TaskConfig(), f) for f in reward_scorer.REWARD_FIELDS})
        self.assertEqual(payload['reward_version'], REWARD_VERSION)

    def test_page_without_traces_still_explains_the_reward(self):
        html = reward_workbench.build_page([], NOMINAL_HEIGHT)
        self.assertIn('linear_tracking_weight', html)


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
            args = parser.parse_args(['--reward-config', f'x={path}', '--reward', 'x=locomotion.task:measured_reward'])
            with self.assertRaises(SystemExit):
                reward_scorer.selected_rewards(parser, args)


if __name__ == '__main__':
    unittest.main()
