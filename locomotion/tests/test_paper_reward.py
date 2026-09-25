"""Table I task and penalty terms, with their declared adaptations, on synthetic and recorded telemetry."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import unittest

import torch

from locomotion import paper_reward as paper, reward_scorer
from locomotion.task import TaskConfig

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'


def telemetry(n=1, **overrides):
    values = {'linear_velocity_nav': torch.zeros(n, 3), 'angular_velocity_body': torch.zeros(n, 3),
              'torque_square_sum_400hz': torch.zeros(n, 18), 'joint_velocity_rad_s': torch.zeros(n, 18),
              'previous_joint_velocity_rad_s': torch.zeros(n, 18), 'action': torch.zeros(n, 18),
              'previous_action': torch.zeros(n, 18), 'other_body_force_max_400hz': torch.zeros(n),
              'requested_torque_abs_max_400hz': torch.zeros(n, 18), 'command': torch.zeros(n, 3)}
    values.update(overrides)
    return values


def score(values):
    return paper.paper_reward(values, values['command'], None, None)


def joint(value, index=0):
    row = torch.zeros(1, 18)
    row[0, index] = value
    return row


class TrackingTests(unittest.TestCase):
    def test_exact_tracking_pays_each_weight(self):
        values = telemetry(command=torch.tensor([[.05, 0., .2]]), linear_velocity_nav=torch.tensor([[.05, 0., 0.]]),
                           angular_velocity_body=torch.tensor([[0., 0., .2]]))
        _, c = score(values)
        self.assertAlmostEqual(float(c['linear_tracking']), 1., places=6)
        self.assertAlmostEqual(float(c['yaw_tracking']), .5, places=6)

    def test_tracking_decreases_with_an_unsquared_error(self):
        values = telemetry(command=torch.tensor([[.05, 0., 0.]]), linear_velocity_nav=torch.tensor([[.02, .04, 0.]]))
        _, c = score(values)
        self.assertAlmostEqual(float(c['linear_tracking']), math.exp(-.05/.15), places=6)

    def test_motionless_robot_keeps_most_of_the_paper_tracking_reward(self):
        # The 0.15 kernel was sized for a faster robot; at 0.05 m/s standing still keeps 72%.
        _, c = score(telemetry(command=torch.tensor([[.05, 0., 0.]])))
        self.assertAlmostEqual(float(c['linear_tracking']), math.exp(-1/3), places=6)


class PenaltyTests(unittest.TestCase):
    def test_rest_costs_nothing(self):
        _, c = score(telemetry())
        for name, value in c.items():
            if not name.endswith('tracking'):
                self.assertEqual(float(value), 0., name)

    def test_printed_weights_and_norms(self):
        values = telemetry(linear_velocity_nav=torch.tensor([[0., 0., .5]]), angular_velocity_body=torch.tensor([[.3, .4, 0.]]),
                           torque_square_sum_400hz=joint(8.), joint_velocity_rad_s=joint(.02),
                           action=joint(.3), previous_action=joint(-.1), other_body_force_max_400hz=torch.tensor([2.]))
        _, c = score(values)
        self.assertAlmostEqual(float(c['vertical_velocity']), -.25, places=7)
        self.assertAlmostEqual(float(c['roll_pitch_rate']), -.08*.5, places=7)
        self.assertAlmostEqual(float(c['joint_torque']), -2e-6*1., places=12)
        self.assertAlmostEqual(float(c['joint_acceleration']), -1.5e-7*1., places=12)
        self.assertAlmostEqual(float(c['action_rate']), -.01*.4, places=7)
        self.assertAlmostEqual(float(c['collisions']), -.05, places=7)

    def test_contact_below_the_force_threshold_is_not_a_collision(self):
        _, c = score(telemetry(other_body_force_max_400hz=torch.tensor([1.])))
        self.assertEqual(float(c['collisions']), 0.)

    def test_limit_hinges_engage_only_beyond_their_limits(self):
        _, inside = score(telemetry(requested_torque_abs_max_400hz=joint(1.6), joint_velocity_rad_s=joint(50.)))
        self.assertEqual(float(inside['torque_limit']), 0.)
        self.assertEqual(float(inside['velocity_limit']), 0.)
        _, beyond = score(telemetry(requested_torque_abs_max_400hz=joint(2.), joint_velocity_rad_s=joint(-51.26548245743669)))
        self.assertAlmostEqual(float(beyond['torque_limit']), -.05*.4, places=6)
        self.assertAlmostEqual(float(beyond['velocity_limit']), -.5*1., places=5)

    def test_velocity_limit_is_the_approved_model_limit(self):
        selector = json.loads((ROOT/'robot/active_model.json').read_text())
        model = json.loads((ROOT/selector['model']['path']).read_text())
        self.assertEqual(paper.PAPER_CONFIG.velocity_limit_rad_s, model['actuator_specification']['urdf_max_velocity_rad_s'])

    def test_missing_telemetry_is_named(self):
        values = telemetry()
        del values['previous_action']
        with self.assertRaisesRegex(ValueError, 'previous_action'):
            score(values)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = reward_scorer.load_trace(TRACE)
        cls.com = reward_scorer.root_com_local()

    def evaluate(self, config):
        def reward(*args):
            return paper.paper_reward(*args, paper_config=config)
        return reward_scorer.evaluate(self.trace, reward, TaskConfig(), None, self.com)[1]

    def test_calibrated_penalties_meet_the_rule_on_their_source(self):
        result = paper.calibrate([self.trace], ratio=.8, com_local=self.com)
        components = self.evaluate(replace(paper.PAPER_CONFIG, **result['weights']))
        tracking = sum(float(components[name].double().mean()) for name in paper.TRACKING_TERMS)
        shares = [float(components[name].double().abs().mean()) for name in paper.CALIBRATED_TERMS]
        for share in shares:
            self.assertAlmostEqual(share / (.8 * tracking / len(shares)), 1., delta=.005)

    def test_frozen_weights_define_the_calibrated_config(self):
        for field, weight in paper.CALIBRATION['weights'].items():
            self.assertEqual(getattr(paper.CALIBRATED_CONFIG, field), weight)
        untouched = set(asdict(paper.PAPER_CONFIG)) - set(paper.CALIBRATION['weights'])
        self.assertEqual({f: getattr(paper.CALIBRATED_CONFIG, f) for f in untouched},
                         {f: getattr(paper.PAPER_CONFIG, f) for f in untouched})

    def test_calibration_rescales_only_the_continuous_penalties(self):
        literal, calibrated = self.evaluate(paper.PAPER_CONFIG), self.evaluate(paper.CALIBRATED_CONFIG)
        for term, value in literal.items():
            field = paper.CALIBRATED_TERMS.get(term)
            factor = getattr(paper.CALIBRATED_CONFIG, field) / getattr(paper.PAPER_CONFIG, field) if field else 1.
            torch.testing.assert_close(calibrated[term], value * factor, rtol=1e-5, atol=1e-9, msg=term)

    def test_calibrated_reward_uses_the_frozen_config(self):
        values = telemetry(action=joint(.3))
        torch.testing.assert_close(paper.paper_reward_calibrated(values, values['command'], None, None)[0],
                                   paper.paper_reward(values, values['command'], None, None,
                                                      paper_config=paper.CALIBRATED_CONFIG)[0])


class RecordedTraceTests(unittest.TestCase):
    def test_scores_a_recorded_trace_and_its_motionless_counterfactual(self):
        trace = reward_scorer.load_trace(TRACE)
        com = reward_scorer.root_com_local()
        reward, c = reward_scorer.evaluate(trace, paper.paper_reward, TaskConfig(), .0978, com)
        self.assertEqual(len(reward), len(trace.data['command']) - 1)
        self.assertLess(float(c['joint_acceleration'].max()), 0.)
        still, s = reward_scorer.evaluate(trace, paper.paper_reward, TaskConfig(), .0978, com, motionless=True)
        for name in ('joint_acceleration', 'action_rate', 'roll_pitch_rate', 'vertical_velocity'):
            self.assertTrue(bool((s[name] == 0).all()), name)
        self.assertTrue(bool(torch.isfinite(still).all()))


if __name__ == '__main__':
    unittest.main()
