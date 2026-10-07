"""Trace both yaw command signs from the command bank through observations, reward v4 and evaluation.

Positive yaw is a left turn: counterclockwise about native +Z seen from above. Each fixture turns a
level body at a known world rate, so a sign inversion anywhere in the chain fails one of these tests.
"""
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion import evaluation, ppo, task_v4
from locomotion.env import LocomotionEnv, _diagnostic_rotation, inverse_rotate, navigation
from locomotion.evaluate import learning_probe_cases
from locomotion.task import TaskConfig, command_bank
from locomotion.task_v4 import REWARD_V4_CONFIG as CONFIG, TrainingTaskV4
from locomotion.tests.test_evaluation import fixture, score
from locomotion.tests.test_task_v4 import GaitDouble

RATE = .2
SIGNS = (-1., 1.)


def yawed(heading):
    """XYZW quaternion of a level body turned counterclockwise by ``heading`` about +Z."""
    return torch.tensor([0., 0., math.sin(heading / 2), math.cos(heading / 2)])


class YawDouble(GaitDouble):
    """GaitDouble whose level root turns at a declared world yaw rate; the gyro reads that rate in body axes."""

    def __init__(self, n, heading):
        super().__init__(n)
        self.turn = torch.zeros(n)
        self.heading = torch.full((n,), float(heading))
        self.current["root"][:, 3:] = torch.stack([yawed(float(h)) for h in self.heading])

    def step(self, action):
        self.heading = self.heading + self.turn * self.cfg.control_dt
        quaternion = torch.stack([yawed(float(h)) for h in self.heading])
        self.current["root"][:, 3:] = quaternion
        output = super().step(action)
        world = torch.stack((torch.zeros_like(self.turn), torch.zeros_like(self.turn), self.turn), -1)
        self.telemetry["angular_velocity_body"] = inverse_rotate(quaternion, world)
        return output


class YawSignTests(unittest.TestCase):
    def test_command_bank_and_probes_hold_both_signs_in_the_navigation_frame(self):
        bank = command_bank(TaskConfig())
        for sign in SIGNS:
            self.assertIn([0., 0., sign * TaskConfig().yaw_rate_rad_s], bank)
        probes = {case["case_id"]: case["command"] for case in learning_probe_cases()}
        self.assertEqual(probes["learning:yaw_-1"], [0., 0., -RATE])
        self.assertEqual(probes["learning:yaw_+1"], [0., 0., RATE])
        # Navigation forward is native -Y and left is native +X; both frames share +Z, so yaw keeps its sign.
        torch.testing.assert_close(navigation(torch.tensor([0., -1., 0.])), torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(navigation(torch.tensor([1., 0., 0.])), torch.tensor([0., 1., 0.]))
        torch.testing.assert_close(navigation(torch.tensor([0., 0., RATE])), torch.tensor([0., 0., RATE]))
        forward, left = torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.])
        self.assertEqual(float(torch.linalg.cross(forward, left)[2]), 1.)

    def test_observation_carries_the_signed_command_and_body_yaw_rate(self):
        for sign in SIGNS:
            for heading in (0., 2.5, -2.5):
                with self.subTest(sign=sign, heading=heading):
                    quaternion = yawed(heading)[None]
                    world = torch.tensor([[0., 0., sign * RATE]])
                    state = {"angular": inverse_rotate(quaternion, world), "gravity": torch.tensor([[0., 0., -1.]]),
                             "q": torch.zeros(1, 18), "dq": torch.zeros(1, 18), "linear": torch.zeros(1, 3)}
                    env = SimpleNamespace(neutral=torch.zeros(18), commands=torch.tensor([[0., 0., sign * RATE]]),
                                          previous_action=torch.zeros(1, 18), cfg=SimpleNamespace(record_motion_features=False))
                    frame = LocomotionEnv._proprio(env, state)
                    self.assertAlmostEqual(float(frame[0, 2]), sign * RATE, places=6)
                    env.history = frame[:, None].expand(-1, 5, -1).clone()
                    observation = LocomotionEnv._observations(env, state)["obs"]
                    self.assertEqual(observation.shape, (1, 231))
                    torch.testing.assert_close(observation[0, 210:213], torch.tensor([0., 0., sign * RATE]))
                    scaled = ppo.scale_observation(observation, "fixed")
                    self.assertAlmostEqual(float(scaled[0, 212]), sign * RATE * ppo.COMMAND_SCALES[2], places=6)
                    self.assertAlmostEqual(float(scaled[0, 168 + 2]), sign * RATE * ppo.FRAME_SCALES[2], places=5)
                    # A yaw command counts as motion, so the gait clock runs for both signs.
                    clock = ppo.clock_features(torch.tensor([15]), 60, observation[:, 210:213])
                    torch.testing.assert_close(clock, torch.tensor([[1., 0.]]), atol=1e-6, rtol=0)

    def test_reward_pays_the_commanded_turn_and_charges_the_opposite_turn(self):
        for sign in SIGNS:
            # Start near +pi so that a left turn crosses the heading wrap; start near -pi for a right turn.
            native = YawDouble(2, heading=sign * (math.pi - .002))
            task = TrainingTaskV4(native, reward_config=CONFIG.__class__(forward_draw_fraction=0.))
            task.reset()
            native.commands[:] = torch.tensor([0., 0., sign * RATE])
            task.window.restart(torch.arange(2), native.commands)
            task.remaining_controls[:] = 1000
            native.turn = torch.tensor([sign * RATE, -sign * RATE])
            for _ in range(CONFIG.tracking_window_controls + 2):
                task.step(torch.zeros(2, 18))
            parts = task.last_components
            with self.subTest(sign=sign):
                self.assertLess(float(task_v4.heading(native.current["root"][:1, 3:])[0]) * sign, 0., "the turn crossed the wrap")
                torch.testing.assert_close(task.window.values.mean(1)[:, 2], torch.tensor([sign * RATE, -sign * RATE]),
                                           atol=1e-4, rtol=0)
                self.assertAlmostEqual(float(parts["yaw_tracking"][0]), CONFIG.yaw_tracking_weight, places=3)
                self.assertAlmostEqual(float(parts["yaw_tracking"][1]), -CONFIG.yaw_tracking_weight, places=3)
                self.assertAlmostEqual(float(parts["yaw_rate"][0]), 0., places=4)
                self.assertAlmostEqual(float(parts["yaw_rate"][1]),
                                       -CONFIG.yaw_rate_weight * (2 * RATE / CONFIG.yaw_rate_scale_rad_s) ** 2, places=3)
                # The training record reads gyro minus command: zero for the commanded turn, twice the rate otherwise.
                interval = task.status(reset_interval=True)["interval_metrics"]
                self.assertGreater(interval["command_classes"]["yaw"]["environment_controls"], 0)
                self.assertAlmostEqual(interval["signed_yaw_error_rad_s"], -sign * RATE, places=5)

    def test_swing_travel_asks_each_toe_to_move_with_the_commanded_turn(self):
        # A left turn moves a toe ahead of the body (navigation +x) to the left, and a toe on the left to the rear.
        toes = torch.tensor([[[.2, 0.], [0., .2], [-.2, 0.], [0., -.2], [.2, .2], [-.2, -.2]]])
        for sign in SIGNS:
            commands = torch.tensor([[0., 0., sign * RATE]])
            wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack((-toes[..., 1], toes[..., 0]), -1)
            lifted = torch.zeros(1, 6, 3)
            forward = task_v4.measured_reward_v4(
                {**_telemetry(commands), "toe_xyz_nav": toes, "toe_velocity_nav": wanted,
                 "tibia_floor_force_world_n": lifted}, commands, torch.zeros(1, dtype=torch.bool), CONFIG, .1)[1]
            backward = task_v4.measured_reward_v4(
                {**_telemetry(commands), "toe_xyz_nav": toes, "toe_velocity_nav": -wanted,
                 "tibia_floor_force_world_n": lifted}, commands, torch.zeros(1, dtype=torch.bool), CONFIG, .1)[1]
            with self.subTest(sign=sign):
                self.assertAlmostEqual(float(wanted[0, 0, 1]), sign * RATE * .2, places=6)
                self.assertAlmostEqual(float(wanted[0, 1, 0]), -sign * RATE * .2, places=6)
                self.assertGreater(float(forward["swing_travel"]), 0.)
                self.assertLess(float(backward["swing_travel"]), 0.)

    def test_evaluation_scores_each_probe_against_its_signed_command(self):
        for sign in SIGNS:
            command = (0., 0., sign * RATE)
            with self.subTest(sign=sign):
                # The evaluator's gyro is R^T times the world angular velocity, for any heading.
                rotation = _diagnostic_rotation(yawed(2.9 * sign).numpy().astype(float))
                gyro = np.einsum("ji,j->i", rotation, np.array([0., 0., sign * RATE]))
                self.assertAlmostEqual(gyro[2], sign * RATE, places=6)
                matched = score(fixture(command=command), "omni_static", command=list(command))
                self.assertTrue(matched["pass"], matched["failed_bounds"])
                self.assertAlmostEqual(matched["metrics"]["mean_yaw_rate_rad_s"], sign * RATE, places=6)
                self.assertAlmostEqual(matched["metrics"]["yaw_error_rad_s"], 0., places=6)
                opposite = fixture(command=command)
                opposite["gyro_body_rad_s"][:, 2] = -sign * RATE
                result = score(opposite, "omni_static", command=list(command))
                self.assertIn("yaw_error_rad_s", result["failed_bounds"])
                self.assertAlmostEqual(result["metrics"]["yaw_error_rad_s"], 2 * RATE, places=6)
                self.assertAlmostEqual(result["metrics"]["mean_yaw_rate_rad_s"], -sign * RATE, places=6)


def _telemetry(commands):
    from locomotion.tests.test_task_v4 import telemetry
    return telemetry(commands)


if __name__ == "__main__":
    unittest.main()
