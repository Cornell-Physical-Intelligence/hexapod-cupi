"""Checks behind docs/REWARD_V2_TABLE1_AUDIT.md: the sign, the stationary criterion and the tracking variants.

Version 1 is ``task.measured_reward``; the paper variant is
``paper_reward.paper_reward``. The command-scaled kernel below is the audit's
proposal for reward v2 and is not yet in ``task.py``.
"""
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from locomotion import paper_reward, reward_scorer
from locomotion.task import TaskConfig, command_bank, measured_reward

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
CRITERION = .10
K = .4
C_MIN = {'translation': .025, 'yaw': .15}


def command_scaled(error, command, kind):
    """Audit variant B: exp(-|e| / (k max(|c|, c_min)))."""
    return math.exp(-abs(error) / (K * max(abs(command), C_MIN[kind])))


def v1_motionless_components(command):
    telemetry = {'linear_velocity_nav': torch.zeros(1, 3), 'angular_velocity_body': torch.zeros(1, 3),
                 'root_pose_xyzw': torch.tensor([[0., 0., .0978, 0., 0., 0., 1.]]),
                 'joint_velocity_rad_s': torch.zeros(1, 18), 'joint_target_rad': torch.zeros(1, 18),
                 'torque_square_sum_400hz': torch.zeros(1, 18), 'other_body_force_max_400hz': torch.zeros(1),
                 'command': torch.tensor([command], dtype=torch.float32)}
    _, components = measured_reward(telemetry, telemetry['command'], torch.zeros(1, 18),
                                    torch.zeros(1, dtype=torch.bool), TaskConfig(), .0978)
    return {k: float(v) for k, v in components.items()}


def paper_motionless_components(command):
    n = torch.zeros(1, 18)
    telemetry = {'linear_velocity_nav': torch.zeros(1, 3), 'angular_velocity_body': torch.zeros(1, 3),
                 'torque_square_sum_400hz': n, 'joint_velocity_rad_s': n, 'previous_joint_velocity_rad_s': n,
                 'action': n, 'previous_action': n, 'other_body_force_max_400hz': torch.zeros(1),
                 'requested_torque_abs_max_400hz': n, 'command': torch.tensor([command], dtype=torch.float32)}
    return {k: float(v) for k, v in paper_reward.paper_reward(telemetry, telemetry['command'], None, None)[1].items()}


def translation_commands():
    return [c for c in command_bank(TaskConfig()) if any(c[:2])]


class SignTests(unittest.TestCase):
    def test_resolved_tracking_kernel_decreases_with_error(self):
        errors = [0., .01, .05, .2]
        rewards = [math.exp(-e/.15) for e in errors]
        self.assertEqual(rewards, sorted(rewards, reverse=True))
        self.assertEqual(rewards[0], 1.)
        self.assertGreater(math.exp(errors[-1]/.15), 1., 'the printed sign would reward error')


class StationaryCriterionTests(unittest.TestCase):
    """Share of the commanded tracking term that a motionless robot keeps (audit section 4)."""

    def test_version_one_fails_at_the_slow_translation_command(self):
        slow = v1_motionless_components([.025, 0., 0.])['linear_tracking']
        fast = v1_motionless_components([.05, 0., 0.])['linear_tracking']
        self.assertAlmostEqual(slow, math.exp(-.025**2/.0009), places=6)
        self.assertGreater(slow, CRITERION)
        self.assertLess(fast, CRITERION)

    def test_version_one_combined_figures_match_training(self):
        # docs/TRAINING.md: about 62% at 0.025 m/s and 28% at 0.05 m/s, before penalties.
        for speed, expected in ((.025, .615), (.05, .279)):
            c = v1_motionless_components([speed, 0., 0.])
            self.assertAlmostEqual((c['linear_tracking'] + c['yaw_tracking']) / 1.3, expected, places=3)

    def test_version_one_fails_for_yaw(self):
        self.assertAlmostEqual(v1_motionless_components([0., 0., .2])['yaw_tracking'] / .3, math.exp(-1), places=6)

    def test_paper_scale_fails_for_every_bank_command(self):
        for command in translation_commands():
            share = paper_motionless_components(command)['linear_tracking']
            self.assertGreater(share, CRITERION, command)
        self.assertAlmostEqual(.15 * math.log(10), .345, places=3)

    def test_command_scaled_kernel_passes_for_every_bank_command(self):
        for command in command_bank(TaskConfig()):
            translation = math.hypot(*command[:2])
            if translation:
                self.assertLess(command_scaled(translation, translation, 'translation'), CRITERION, command)
            if command[2]:
                self.assertLess(command_scaled(command[2], command[2], 'yaw'), CRITERION, command)
        self.assertLess(K, 1/math.log(10))
        self.assertAlmostEqual(math.exp(-1/K), .082, places=3)


class CommandScaledTests(unittest.TestCase):
    def test_zero_command_pays_standing_still_in_full(self):
        self.assertEqual(command_scaled(0., 0., 'translation'), 1.)
        self.assertAlmostEqual(command_scaled(.01, 0., 'translation'), math.exp(-1), places=12)

    def test_scale_is_continuous_at_the_smallest_command(self):
        below = command_scaled(.02, C_MIN['translation'] - 1e-9, 'translation')
        at = command_scaled(.02, C_MIN['translation'], 'translation')
        self.assertAlmostEqual(below, at, places=6)

    def test_exact_tracking_pays_in_full_at_every_speed(self):
        for command in (.025, .05, .3):
            self.assertEqual(command_scaled(0., command, 'translation'), 1.)


class RecordedTraceTests(unittest.TestCase):
    """The fixture is a smooth recorded tripod walk at the 0.05 m/s command (audit section 5)."""

    def test_both_command_scaled_variants_pay_a_smooth_walk(self):
        trace = reward_scorer.load_trace(TRACE)
        v = reward_scorer.origin_velocity_nav(trace, reward_scorer.root_com_local())[1:, 0, :2].numpy()
        error = np.linalg.norm(v - [.05, 0.], axis=-1)
        window = 60
        mean = np.stack([np.convolve(v[:, i], np.ones(window) / window, 'valid') for i in (0, 1)], -1)
        mean_error = np.linalg.norm(mean - [.05, 0.], axis=-1)
        per_control = float(np.exp(-error / (K * .05)).mean())
        averaged = float(np.exp(-mean_error / (K * .05)).mean())
        self.assertLess(v[:, 0].std(), .01)
        self.assertAlmostEqual(per_control, .858, places=3)
        self.assertAlmostEqual(averaged, .970, places=3)


if __name__ == '__main__':
    unittest.main()
