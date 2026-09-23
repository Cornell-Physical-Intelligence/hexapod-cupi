"""Reference formulas for the Reward v2 Table I audit, with CPU checks.

This module does not touch the live training reward in
:mod:`locomotion.task`. Per issue #18's first bounded task, the audit in
`docs/REWARD_V2_TABLE1_AUDIT.md` records the chosen formulas and named
adaptations, with CPU checks, before any reward-schema change lands in
`locomotion/task.py`. These are that record: a standalone, dependency-free
(no torch, no simulator) statement of the resolved Table I formulas from
Liu et al., arXiv:2511.03167, Table I, p. 3, plus the two sign/shape
resolutions the audit proposes:

* The printed task-tracking exponent has no minus sign (``exp(x/0.15)``,
  growing with error). This module implements the decreasing form,
  ``exp(-x/0.15)``, that a tracking reward requires (audit Finding 1).
* Table I's task-tracking and several penalty rows use an unsquared L2 norm
  (``||error||_2``), not a squared error -- distinct from the variance-based
  Gaussian forms currently used in ``locomotion/task.py`` (audit Finding 2).

Vectors are plain ``list[float]`` / ``tuple[float, ...]``; only :mod:`math` is
used, so this file documents the formulas independent of the training stack.
"""
from __future__ import annotations

import math
import unittest


def _norm(values):
    return math.sqrt(sum(v * v for v in values))


def linear_tracking_reward(velocity_xy, desired_xy, *, weight=1.0, scale=0.15):
    """Table I, Task r^g, Linear velocity -- resolved decreasing form."""
    error = [a - b for a, b in zip(velocity_xy, desired_xy)]
    return weight * math.exp(-_norm(error) / scale)


def yaw_tracking_reward(yaw_rate, desired_yaw_rate, *, weight=0.5, scale=0.15):
    """Table I, Task r^g, Angular velocity -- resolved decreasing form."""
    return weight * math.exp(-abs(yaw_rate - desired_yaw_rate) / scale)


def vertical_velocity_penalty(v_z, *, weight=1.0):
    """Table I, Penalty r^l, Linear velocity: -1 * v_z^2."""
    return -weight * v_z * v_z


def roll_pitch_rate_penalty(angular_velocity_xy, *, weight=0.08):
    """Table I, Penalty r^l, Angular velocity: -0.08 * ||omega_xy||_2."""
    return -weight * _norm(angular_velocity_xy)


def joint_torque_penalty(torque, *, weight=2e-6):
    """Table I, Penalty r^l, Joint torque: -2e-6 * ||tau||_2 (raw N*m)."""
    return -weight * _norm(torque)


def joint_acceleration_penalty(joint_acceleration, *, weight=1.5e-7):
    """Table I, Penalty r^l, Joint acceleration: -1.5e-7 * ||q_ddot||_2."""
    return -weight * _norm(joint_acceleration)


def action_rate_penalty(action, previous_action, *, weight=0.01):
    """Table I, Penalty r^l, Action rate: -0.01 * ||a_t - a_t-1||_2."""
    delta = [a - b for a, b in zip(action, previous_action)]
    return -weight * _norm(delta)


def collision_penalty(collision_count, *, weight=0.05):
    """Table I, Penalty r^l, Collisions: -0.05 * n_collision."""
    return -weight * collision_count


def hinge_limit_penalty(measured, limit, *, weight):
    """Shared form for the three Table I limit-violation rows: torque,
    joint-velocity and contact-force limits each fit
    ``-weight * ||max(|measured| - limit, 0)||_2``, differing only in
    ``measured``/``limit`` units and ``weight`` (0.05, 0.5 and 0.1
    respectively). Exactly zero inside the limit; the hinge only engages
    once a component's magnitude exceeds its named limit.
    """
    excess = [max(abs(m) - l, 0.0) for m, l in zip(measured, limit)]
    return -weight * _norm(excess)


class TaskTrackingTests(unittest.TestCase):
    def test_zero_error_gives_the_declared_weight(self):
        self.assertAlmostEqual(linear_tracking_reward((0.05, 0.0), (0.05, 0.0)), 1.0, places=12)
        self.assertAlmostEqual(yaw_tracking_reward(0.2, 0.2), 0.5, places=12)

    def test_reward_decreases_monotonically_with_error(self):
        rewards = [linear_tracking_reward((e, 0.0), (0.0, 0.0)) for e in (0.0, 0.01, 0.02, 0.05, 0.1)]
        self.assertEqual(rewards, sorted(rewards, reverse=True))
        self.assertTrue(all(r > 0.0 for r in rewards))

    def test_reward_never_exceeds_its_weight(self):
        for e in (0.0, -0.03, 0.2, -1.0, 5.0):
            self.assertLessEqual(linear_tracking_reward((e, 0.0), (0.0, 0.0)), 1.0 + 1e-12)
            self.assertLessEqual(yaw_tracking_reward(e, 0.0), 0.5 + 1e-12)

    def test_known_value_regression(self):
        # exp(-0.03/0.15) = exp(-0.2); pinned so an accidental sign or scale
        # change is caught even if the monotonicity checks above still pass.
        self.assertAlmostEqual(linear_tracking_reward((0.03, 0.0), (0.0, 0.0)), math.exp(-0.2), places=12)
        self.assertAlmostEqual(yaw_tracking_reward(0.03, 0.0, weight=0.5), 0.5 * math.exp(-0.2), places=12)


class PenaltyFormTests(unittest.TestCase):
    def test_zero_state_gives_zero_penalty(self):
        self.assertEqual(vertical_velocity_penalty(0.0), 0.0)
        self.assertEqual(roll_pitch_rate_penalty((0.0, 0.0)), 0.0)
        self.assertEqual(joint_torque_penalty([0.0] * 18), 0.0)
        self.assertEqual(joint_acceleration_penalty([0.0] * 18), 0.0)
        self.assertEqual(action_rate_penalty([0.0] * 18, [0.0] * 18), 0.0)
        self.assertEqual(collision_penalty(0), 0.0)

    def test_penalties_are_never_positive(self):
        self.assertLess(vertical_velocity_penalty(0.4), 0.0)
        self.assertLess(vertical_velocity_penalty(-0.4), 0.0)  # sign-independent: squared term
        self.assertLess(roll_pitch_rate_penalty((0.1, -0.2)), 0.0)
        self.assertLess(joint_torque_penalty([0.05] * 18), 0.0)
        self.assertLess(collision_penalty(3), 0.0)

    def test_known_value_regression(self):
        self.assertAlmostEqual(vertical_velocity_penalty(0.5), -0.25, places=12)
        self.assertAlmostEqual(roll_pitch_rate_penalty((0.3, 0.4), weight=0.08), -0.08 * 0.5, places=12)
        self.assertAlmostEqual(collision_penalty(4, weight=0.05), -0.2, places=12)


class HingeLimitPenaltyTests(unittest.TestCase):
    def test_zero_inside_the_limit(self):
        measured, limit = [0.5] * 18, [1.0] * 18
        self.assertEqual(hinge_limit_penalty(measured, limit, weight=0.05), 0.0)

    def test_zero_exactly_at_the_limit(self):
        measured, limit = [1.0] * 18, [1.0] * 18
        self.assertEqual(hinge_limit_penalty(measured, limit, weight=0.05), 0.0)

    def test_negative_once_a_single_joint_exceeds_its_limit(self):
        measured = [0.5] * 17 + [1.6]
        limit = [1.0] * 18
        penalty = hinge_limit_penalty(measured, limit, weight=0.05)
        self.assertLess(penalty, 0.0)
        self.assertAlmostEqual(penalty, -0.05 * 0.6, places=12)

    def test_sign_of_measured_value_does_not_matter(self):
        limit = [1.0] * 18
        over_positive = hinge_limit_penalty([0.0] * 17 + [1.6], limit, weight=0.5)
        over_negative = hinge_limit_penalty([0.0] * 17 + [-1.6], limit, weight=0.5)
        self.assertAlmostEqual(over_positive, over_negative, places=12)

    def test_matches_each_table1_named_row(self):
        # Torque limits: weight 0.05. Velocity limits: weight 0.5. Contact
        # force: weight 0.1. Same shape, three declared weights.
        excess_source = [0.0] * 17 + [0.4]
        limit = [0.0] * 18
        self.assertAlmostEqual(hinge_limit_penalty(excess_source, limit, weight=0.05), -0.05 * 0.4, places=12)
        self.assertAlmostEqual(hinge_limit_penalty(excess_source, limit, weight=0.5), -0.5 * 0.4, places=12)
        self.assertAlmostEqual(hinge_limit_penalty(excess_source, limit, weight=0.1), -0.1 * 0.4, places=12)


if __name__ == "__main__":
    unittest.main()
