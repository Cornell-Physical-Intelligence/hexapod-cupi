"""Tests for adapter.py (pure NumPy, unittest)."""
import unittest

import numpy as np

import adapter as A
from adapter import (ActorFrameInputs, JointConfig, ObservationHistory, ObservationScales,
                     TargetPipeline, command_to_native_body_frame, native_body_to_command_frame)

J = A.NUM_JOINTS
F = A.FRAME_SIZE
L = A.FRAME_LAYOUT
NAMES = tuple(f"leg{i // 3}_{('coxa', 'femur', 'tibia')[i % 3]}" for i in range(J))


def make_config():
    lower = np.full(J, -1.0)
    upper = np.full(J, 1.0)
    lower[0] = -0.2          # asymmetric limits on joint 0
    neutral = np.zeros(J)
    neutral[1] = 0.3         # non-zero neutral on joint 1
    scale = np.ones(J)
    scale[2] = 0.5           # smaller action scale on joint 2
    return JointConfig(NAMES, lower, upper, neutral, scale)


def make_scales():
    return ObservationScales(
        angular_velocity=np.full(3, 2.0),
        projected_gravity=np.ones(3),
        command=np.array([1.0, 1.0, 0.5]),
        joint_position_error=np.full(J, 0.5),
        joint_rate=np.full(J, 10.0),
        previous_action=np.ones(J),
        target_error=np.full(J, 0.5),
        linear_velocity=np.full(3, 0.5),
    )


def make_inputs(k, neutral, *, vx=0.0, act=0.0):
    return ActorFrameInputs(
        native_angular_velocity=np.zeros((k, 3)),
        native_projected_gravity=np.tile([0.0, 0.0, -1.0], (k, 1)),
        command=np.tile([vx, 0.0, 0.0], (k, 1)),
        joint_position=np.tile(neutral + 0.0, (k, 1)),
        joint_rate=np.zeros((k, J)),
        previous_clipped_action=np.full((k, J), act),
        executed_target=np.tile(neutral + 0.0, (k, 1)),
    )


class FrameMapTest(unittest.TestCase):
    def test_direction_fixtures(self):
        native_forward, native_left, up = [0, -1, 0], [1, 0, 0], [0, 0, 1]
        np.testing.assert_array_equal(native_body_to_command_frame(native_forward), [1, 0, 0])
        np.testing.assert_array_equal(native_body_to_command_frame(native_left), [0, 1, 0])
        np.testing.assert_array_equal(native_body_to_command_frame(up), [0, 0, 1])
        np.testing.assert_array_equal(command_to_native_body_frame([1, 0, 0]), native_forward)

    def test_angular_velocity_roll_and_yaw(self):
        # roll about the native forward axis (-Y) becomes roll about command +X; yaw about +Z is unchanged
        np.testing.assert_array_equal(native_body_to_command_frame([0, -1.5, 0]), [1.5, 0, 0])
        np.testing.assert_array_equal(native_body_to_command_frame([0, 0, 0.7]), [0, 0, 0.7])

    def test_proper_rotation_inverse_and_batching(self):
        M = A.NATIVE_TO_COMMAND
        self.assertAlmostEqual(np.linalg.det(M), 1.0)
        np.testing.assert_allclose(M @ M.T, np.eye(3))
        v = np.random.default_rng(0).normal(size=(4, 2, 3))
        np.testing.assert_allclose(command_to_native_body_frame(native_body_to_command_frame(v)), v)
        self.assertEqual(native_body_to_command_frame(v).shape, (4, 2, 3))

    def test_rejects_wrong_vector_width(self):
        with self.assertRaises(ValueError):
            native_body_to_command_frame(np.zeros((2, 4)))


class JointConfigTest(unittest.TestCase):
    def test_rejects_duplicate_and_wrong_count_names(self):
        b = make_config()
        dup = list(NAMES)
        dup[5] = dup[4]
        with self.assertRaises(ValueError):
            JointConfig(dup, b.lower, b.upper, b.neutral, b.action_scale)
        with self.assertRaises(ValueError):
            JointConfig(NAMES[:-1], b.lower, b.upper, b.neutral, b.action_scale)

    def test_rejects_misaligned_or_invalid_arrays(self):
        b = make_config()
        bad = [
            dict(lower=b.lower[:-1]),                                   # misaligned length
            dict(lower=b.lower.reshape(1, J)),                          # wrong rank
            dict(upper=np.where(np.arange(J) == 3, np.inf, b.upper)),   # non-finite limit
            dict(lower=b.upper),                                        # lower not below upper
            dict(neutral=np.full(J, 1.5)),                              # neutral outside limits
            dict(action_scale=np.where(np.arange(J) == 7, 0.0, 1.0)),   # non-positive scale
        ]
        for kw in bad:
            args = dict(lower=b.lower, upper=b.upper, neutral=b.neutral, action_scale=b.action_scale)
            args.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                JointConfig(NAMES, **args)

    def test_config_owns_its_arrays(self):
        lower = np.full(J, -1.0)
        cfg = JointConfig(NAMES, lower, np.ones(J), np.zeros(J), np.ones(J))
        lower[0] = -9.0
        self.assertEqual(cfg.lower[0], -1.0)


class TargetPipelineTest(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.pipe = TargetPipeline(self.cfg, num_envs=3, dt=0.02, max_target_step=0.04)

    def test_constructor_rejects_bad_scalars(self):
        for kw in (dict(num_envs=0), dict(dt=0.0), dict(max_target_step=-0.04),
                   dict(max_target_step=np.nan)):
            args = dict(num_envs=2, dt=0.02, max_target_step=0.04)
            args.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                TargetPipeline(self.cfg, **args)
        self.assertAlmostEqual(self.pipe.max_target_rate, 2.0)

    def test_clip_request_limit_and_first_slew(self):
        snap = self.pipe.step(np.full((3, J), 2.0))
        np.testing.assert_array_equal(snap.raw_action, 2.0)            # raw preserved
        np.testing.assert_array_equal(snap.clipped_action, 1.0)
        self.assertAlmostEqual(snap.requested_target[0, 0], 1.0)       # neutral 0 + scale 1
        self.assertAlmostEqual(snap.requested_target[0, 1], 1.0)       # 0.3 + 1 limited to upper 1.0
        self.assertAlmostEqual(snap.requested_target[0, 2], 0.5)       # scale 0.5
        np.testing.assert_allclose(snap.executed_target[:, 0], 0.04)   # slewed from neutral 0
        np.testing.assert_allclose(snap.executed_target[:, 1], 0.34)   # slewed from neutral 0.3
        np.testing.assert_allclose(self.pipe.executed_target, snap.executed_target)
        np.testing.assert_array_equal(self.pipe.previous_clipped_action, 1.0)

    def test_asymmetric_limits(self):
        for _ in range(10):
            snap = self.pipe.step(np.full((3, J), -1.0))
        self.assertAlmostEqual(snap.executed_target[0, 0], -0.2)   # lower limit reached, never below
        self.assertAlmostEqual(snap.executed_target[0, 3], -0.4)   # symmetric joint still slewing
        for _ in range(40):
            snap = self.pipe.step(np.full((3, J), 1.0))
        self.assertAlmostEqual(snap.executed_target[0, 0], 1.0)
        self.assertAlmostEqual(snap.executed_target[0, 3], 1.0)

    def test_slew_after_saturation(self):
        up = np.full((3, J), 1.0)
        for _ in range(24):
            snap = self.pipe.step(up)
        self.assertAlmostEqual(snap.executed_target[1, 3], 0.96)
        snap = self.pipe.step(up)
        self.assertAlmostEqual(snap.executed_target[1, 3], 1.0)
        for _ in range(5):
            snap = self.pipe.step(up)                     # held at the limit
        self.assertAlmostEqual(snap.executed_target[1, 3], 1.0)
        snap = self.pipe.step(-up)                        # reversal must slew, not jump
        self.assertAlmostEqual(snap.requested_target[1, 3], -1.0)
        self.assertAlmostEqual(snap.executed_target[1, 3], 0.96)
        snap = self.pipe.step(-up)
        self.assertAlmostEqual(snap.executed_target[1, 3], 0.92)

    def test_invalid_actions_reject_before_mutation(self):
        self.pipe.step(np.full((3, J), 0.5))
        before_t, before_a = self.pipe.executed_target, self.pipe.previous_clipped_action
        bad = np.full((3, J), 1.0)
        bad[1, 5] = np.nan
        for action in (bad, np.ones(J), np.ones((4, J)), np.ones((3, J - 1))):
            with self.assertRaises(ValueError):
                self.pipe.step(action)
            np.testing.assert_array_equal(self.pipe.executed_target, before_t)
            np.testing.assert_array_equal(self.pipe.previous_clipped_action, before_a)

    def test_snapshots_are_owned(self):
        action = np.full((3, J), 0.5)
        snap = self.pipe.step(action)
        action[:] = 9.0                                   # caller mutates its input afterwards
        np.testing.assert_array_equal(snap.raw_action, 0.5)
        expected = self.pipe.executed_target
        snap.executed_target[:] = 7.0
        snap.clipped_action[:] = 7.0
        self.pipe.executed_target[:] = 7.0                # properties return copies too
        self.pipe.previous_clipped_action[:] = 7.0
        np.testing.assert_array_equal(self.pipe.executed_target, expected)
        np.testing.assert_array_equal(self.pipe.previous_clipped_action, 0.5)

    def test_reset_rows_touches_only_selected_rows(self):
        for _ in range(2):
            self.pipe.step(np.full((3, J), 1.0))
        before = self.pipe.executed_target
        measured = np.full((1, J), 0.1)
        measured[0, 0] = -0.5                             # below joint 0 lower limit -0.2
        measured[0, 3] = 1.5                              # above upper limit 1.0
        held = self.pipe.reset_rows(np.array([1]), measured)
        self.assertAlmostEqual(held[0, 0], -0.2)          # explicit out-of-limit rule: clamp
        self.assertAlmostEqual(held[0, 3], 1.0)
        self.assertAlmostEqual(held[0, 4], 0.1)
        np.testing.assert_array_equal(self.pipe.executed_target[1], held[0])
        np.testing.assert_array_equal(self.pipe.previous_clipped_action[1], 0.0)
        for row in (0, 2):
            np.testing.assert_array_equal(self.pipe.executed_target[row], before[row])
            np.testing.assert_array_equal(self.pipe.previous_clipped_action[row], 1.0)
        held[:] = 5.0                                     # returned array is owned
        self.assertAlmostEqual(self.pipe.executed_target[1, 4], 0.1)
        snap = self.pipe.step(np.zeros((3, J)))           # next slew starts from the reset pose
        self.assertAlmostEqual(snap.executed_target[1, 4], 0.06)
        self.assertAlmostEqual(snap.executed_target[0, 4], 0.04)   # row 0 was at 0.08

    def test_reset_rejects_bad_rows_or_measurements_without_mutation(self):
        before = self.pipe.executed_target
        nan = np.zeros((1, J))
        nan[0, 2] = np.nan
        cases = [
            (np.array([1, 1]), np.zeros((2, J))),         # duplicate rows
            (np.array([3]), np.zeros((1, J))),            # out of range
            (np.array([0.0]), np.zeros((1, J))),          # non-integer rows
            (np.array([[0]]), np.zeros((1, J))),          # wrong rank
            (np.array([0, 1]), np.zeros((1, J))),         # misaligned measurement rows
            (np.array([0]), nan),                         # non-finite measurement
        ]
        for rows, measured in cases:
            with self.assertRaises(ValueError):
                self.pipe.reset_rows(rows, measured)
            np.testing.assert_array_equal(self.pipe.executed_target, before)


class ObservationHistoryTest(unittest.TestCase):
    def setUp(self):
        self.neutral = make_config().neutral
        self.hist = ObservationHistory(self.neutral, make_scales(), num_envs=3)

    def test_scales_must_be_positive(self):
        kw = dict(angular_velocity=np.ones(3), projected_gravity=np.ones(3), command=np.ones(3),
                  joint_position_error=np.ones(J), joint_rate=np.ones(J), previous_action=np.ones(J),
                  target_error=np.ones(J), linear_velocity=np.ones(3))
        kw["joint_rate"] = np.where(np.arange(J) == 2, 0.0, 1.0)
        with self.assertRaises(ValueError):
            ObservationScales(**kw)

    def test_frame_layout_normalization_and_frame_map(self):
        inputs = ActorFrameInputs(
            native_angular_velocity=np.tile([0.5, -1.0, 0.25], (3, 1)),
            native_projected_gravity=np.tile([0.0, -0.6, -0.8], (3, 1)),  # nose-down toward native forward (-Y)
            command=np.tile([0.3, -0.2, 0.4], (3, 1)),
            joint_position=np.tile(self.neutral + 0.1, (3, 1)),
            joint_rate=np.full((3, J), 5.0),
            previous_clipped_action=np.full((3, J), 0.5),
            executed_target=np.tile(self.neutral - 0.1, (3, 1)),
        )
        self.hist.reset_rows(np.arange(3), inputs)
        obs = self.hist.actor_observation()
        self.assertEqual(obs.shape, (3, A.ACTOR_OBS_SIZE))
        frame = obs[2, -F:]                                                   # newest frame of row 2
        np.testing.assert_allclose(frame[L["angular_velocity"]], [0.5, 0.25, 0.125])  # mapped, then / 2
        np.testing.assert_allclose(frame[L["projected_gravity"]], [0.6, 0.0, -0.8])   # forward tilt lands on +X
        np.testing.assert_allclose(frame[L["command"]], [0.3, -0.2, 0.8])            # not remapped; wz / 0.5
        np.testing.assert_allclose(frame[L["joint_position_error"]], 0.2)
        np.testing.assert_allclose(frame[L["joint_rate"]], 0.5)
        np.testing.assert_allclose(frame[L["previous_action"]], 0.5)
        np.testing.assert_allclose(frame[L["target_error"]], -0.2)
        np.testing.assert_allclose(obs.reshape(3, A.HISTORY_LENGTH, F)[2], np.tile(frame, (5, 1)))

    def test_history_order_and_selective_reset(self):
        n = self.neutral
        c = L["command"].start
        self.hist.reset_rows(np.arange(3), make_inputs(3, n, vx=0.1))
        self.hist.push(make_inputs(3, n, vx=0.2))
        self.hist.push(make_inputs(3, n, vx=0.3))
        np.testing.assert_allclose(self.hist.history[:, :, c], [[0.1, 0.1, 0.1, 0.2, 0.3]] * 3)
        obs = self.hist.actor_observation()
        self.assertAlmostEqual(obs[0, c], 0.1)                  # oldest first
        self.assertAlmostEqual(obs[0, 4 * F + c], 0.3)          # newest last
        before = self.hist.history
        self.hist.reset_rows(np.array([1]), make_inputs(1, n, vx=0.9))
        after = self.hist.history
        np.testing.assert_allclose(after[1, :, c], 0.9)         # all five slots repeat the reset frame
        for row in (0, 2):
            np.testing.assert_array_equal(after[row], before[row])   # neighbors untouched

    def test_observation_is_owned(self):
        self.hist.reset_rows(np.arange(3), make_inputs(3, self.neutral, vx=0.1))
        self.hist.actor_observation()[:] = 42.0
        self.hist.history[:] = 42.0
        self.assertAlmostEqual(self.hist.actor_observation()[0, L["command"].start], 0.1)

    def test_critic_velocity_is_privileged_and_does_not_touch_actor(self):
        self.hist.reset_rows(np.arange(3), make_inputs(3, self.neutral, vx=0.1))
        actor_before = self.hist.actor_observation()
        critic = self.hist.critic_observation(np.tile([0.0, -0.4, 0.1], (3, 1)))   # native: moving forward
        self.assertEqual(critic.shape, (3, A.CRITIC_OBS_SIZE))
        np.testing.assert_array_equal(critic[:, :A.ACTOR_OBS_SIZE], actor_before)
        np.testing.assert_allclose(critic[:, A.ACTOR_OBS_SIZE:], [[0.8, 0.0, 0.2]] * 3)  # mapped, / 0.5
        np.testing.assert_array_equal(self.hist.actor_observation(), actor_before)
        np.testing.assert_array_equal(self.hist.history[:, :, :], actor_before.reshape(3, 5, F))
        with self.assertRaises(ValueError):
            self.hist.critic_observation(np.zeros((2, 3)))

    def test_invalid_inputs_reject_before_mutation(self):
        n = self.neutral
        with self.assertRaises(RuntimeError):
            self.hist.push(make_inputs(3, n))                    # no valid first frame yet
        with self.assertRaises(RuntimeError):
            self.hist.actor_observation()
        self.hist.reset_rows(np.arange(3), make_inputs(3, n, vx=0.1))
        before = self.hist.history
        bad_rate = make_inputs(3, n)
        bad_rate.joint_rate[1, 4] = np.nan
        for inputs in (bad_rate, make_inputs(3, n, act=1.5), make_inputs(2, n)):
            with self.assertRaises(ValueError):
                self.hist.push(inputs)
            np.testing.assert_array_equal(self.hist.history, before)
        with self.assertRaises(ValueError):
            self.hist.reset_rows(np.array([0, 0]), make_inputs(2, n))   # duplicate rows
        np.testing.assert_array_equal(self.hist.history, before)


class TimingContractTest(unittest.TestCase):
    """An executed target appears only in the frame pushed after the interval that held it."""

    def test_executed_target_lands_in_next_frame_only(self):
        cfg = make_config()
        pipe = TargetPipeline(cfg, num_envs=2)
        hist = ObservationHistory(cfg.neutral, make_scales(), num_envs=2)
        measured = np.tile(cfg.neutral, (2, 1))
        pipe.reset_rows(np.arange(2), measured)

        def frame(joint_position):
            return ActorFrameInputs(
                native_angular_velocity=np.zeros((2, 3)),
                native_projected_gravity=np.tile([0.0, 0.0, -1.0], (2, 1)),
                command=np.zeros((2, 3)),
                joint_position=joint_position,
                joint_rate=np.zeros((2, J)),
                previous_clipped_action=pipe.previous_clipped_action,  # describes the completed interval
                executed_target=pipe.executed_target,
            )

        hist.reset_rows(np.arange(2), frame(measured))             # tick 0: nothing executed yet
        snap = pipe.step(np.full((2, J), 1.0))                     # target T0, held during interval 0
        newest = hist.actor_observation()[:, -F:]
        np.testing.assert_array_equal(newest[:, L["target_error"]], 0.0)     # must not claim T0 yet
        np.testing.assert_array_equal(newest[:, L["previous_action"]], 0.0)

        hist.push(frame(measured + 0.01))                          # tick 1: interval 0 completed
        obs = hist.actor_observation()
        newest, older = obs[:, -F:], obs[:, -2 * F:-F]
        np.testing.assert_allclose(newest[:, L["target_error"]], (snap.executed_target - cfg.neutral) / 0.5)
        np.testing.assert_allclose(newest[:, L["target_error"]][:, 3], 0.08)   # 0.04 rad / 0.5 scale
        np.testing.assert_allclose(newest[:, L["previous_action"]], 1.0)
        np.testing.assert_allclose(newest[:, L["joint_position_error"]], 0.02)
        np.testing.assert_array_equal(older[:, L["target_error"]], 0.0)     # earlier frame unchanged


if __name__ == "__main__":
    unittest.main()
