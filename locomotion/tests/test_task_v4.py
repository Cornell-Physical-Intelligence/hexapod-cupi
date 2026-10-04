"""Check reward version 4 without a simulator: noise behaviour, stepping, executed-target terms."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import task_v2, task_v4, train
from locomotion.prepare import prepare
from locomotion.task_v4 import REWARD_V4_CONFIG as CONFIG, TrainingTaskV4, measured_reward_v4
from locomotion.tests.test_task_v2 import RewardDouble

REMOTE = "/srv/cupi/hexapod/runs/test/reward_v4_fixture"
HEIGHT = .1


class GaitDouble(RewardDouble):
    """RewardDouble plus the six toe forces version 4 reads."""

    def __init__(self, n=2):
        super().__init__(n)
        self.forces = torch.zeros(n, 6, 3)
        self.forces[..., 2] = 12.

    def step(self, action):
        output = super().step(action)
        self.telemetry["tibia_floor_force_world_n"] = self.forces.clone()
        return output


def telemetry(commands, **overrides):
    n = len(commands)
    zeros = torch.zeros(n, 18)
    pose = torch.zeros(n, 7)
    pose[:, 2], pose[:, 6] = HEIGHT, 1.
    result = {"linear_velocity_nav": torch.zeros(n, 3), "angular_velocity_body": torch.zeros(n, 3),
        "root_pose_xyzw": pose, "joint_velocity_rad_s": zeros, "previous_joint_velocity_rad_s": zeros,
        "joint_target_rad": zeros, "previous_joint_target_rad": zeros, "torque_square_sum_400hz": zeros,
        "requested_torque_abs_max_400hz": zeros, "other_body_force_max_400hz": torch.zeros(n),
        "tracked_velocity_nav": torch.zeros(n, 3), "touchdown": torch.zeros(n, 6, dtype=torch.bool),
        "air_time_s": torch.zeros(n, 6), "command": commands.clone(), "action": zeros}
    result.update(overrides)
    return result


def score(commands, config=CONFIG, terminated=None, **overrides):
    terminated = torch.zeros(len(commands), dtype=torch.bool) if terminated is None else terminated
    return measured_reward_v4(telemetry(commands, **overrides), commands, terminated, config, HEIGHT)


class RewardV4Tests(unittest.TestCase):
    def test_motionless_robot_earns_no_tracking_on_a_moving_command_and_all_of_it_at_rest(self):
        commands = torch.tensor([[.05, 0., 0.], [.025, 0., 0.], [0., 0., .2], [0., 0., 0.]])
        _, parts = score(commands)
        torch.testing.assert_close(parts["linear_tracking"], torch.tensor([0., 0., 1., 1.]) * CONFIG.linear_tracking_weight)
        torch.testing.assert_close(parts["yaw_tracking"], torch.tensor([1., 1., 0., 1.]) * CONFIG.yaw_tracking_weight)
        tracked = torch.cat((commands[:, :2], commands[:, 2:]), -1)
        reward, parts = score(commands, tracked_velocity_nav=tracked)
        torch.testing.assert_close(parts["linear_tracking"], torch.full((4,), CONFIG.linear_tracking_weight))
        torch.testing.assert_close(reward, torch.full((4,), CONFIG.linear_tracking_weight + CONFIG.yaw_tracking_weight))

    def test_error_scale_follows_the_command_and_is_fixed_for_a_zero_component(self):
        commands = torch.tensor([[.05, 0., 0.], [.025, 0., .15], [0., 0., 0.]])
        tracked = torch.tensor([[.04, 0., .02], [.02, 0., .12], [.01, 0., .02]])
        linear, yaw = task_v4.tracking_errors(tracked, commands)
        torch.testing.assert_close(linear, torch.tensor([.2, .2, .01 / CONFIG.fixed_speed_scale_mps]))
        torch.testing.assert_close(yaw, torch.tensor([.02 / CONFIG.fixed_yaw_scale_rad_s, .2, .02 / CONFIG.fixed_yaw_scale_rad_s]))
        fixed = replace(CONFIG, tracking_scale="fixed")
        torch.testing.assert_close(task_v4.tracking_errors(tracked, commands, fixed)[0],
                                   torch.tensor([.01, .005, .01]) / CONFIG.fixed_speed_scale_mps)

    def test_zero_mean_velocity_noise_leaves_the_gain_from_walking_unchanged(self):
        generator = torch.Generator().manual_seed(4)
        commands = torch.tensor([[.05, 0., 0.]]).expand(200000, -1)
        noise = torch.cat((torch.randn(200000, 2, generator=generator) * .01, torch.zeros(200000, 1)), -1)
        gains = []
        for jitter in (torch.zeros_like(noise), noise):
            still = score(commands, tracked_velocity_nav=jitter)[1]["linear_tracking"].mean()
            walking = score(commands, tracked_velocity_nav=commands + jitter)[1]["linear_tracking"].mean()
            gains.append(float(walking - still))
        self.assertAlmostEqual(gains[0], CONFIG.linear_tracking_weight, places=6)
        self.assertAlmostEqual(gains[1], gains[0], delta=.01)
        # The exponential kernel of versions 2 and 3 loses part of that gain to the same noise.
        old = []
        for jitter in (torch.zeros_like(noise), noise):
            still = task_v2.tracking_components(jitter[:, :2], jitter[:, 2], commands)["linear_tracking"].mean()
            walking = task_v2.tracking_components((commands + jitter)[:, :2], jitter[:, 2], commands)["linear_tracking"].mean()
            old.append(float(walking - still))
        self.assertLess(old[1], .7 * old[0])

    def test_displacement_velocity_reads_body_frame_progress_and_heading_change(self):
        previous = torch.zeros(2, 7)
        previous[:, 6] = 1.
        pose = previous.clone()
        # Navigation forward is native -Y; one millimetre per control is 0.05 m/s.
        pose[0, 1] = -.001
        half = .5 * .004
        pose[1, 3:] = torch.tensor([0., 0., math.sin(half), math.cos(half)])
        velocity = task_v4.displacement_velocity(pose, previous)
        torch.testing.assert_close(velocity[0], torch.tensor([.05, 0., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(velocity[1], torch.tensor([0., 0., .2]), atol=1e-5, rtol=0)
        # A quarter turn of the body: the same world displacement now points to the robot's left.
        turned = previous.clone()
        turned[:, 3:] = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
        moved = turned.clone()
        moved[0, 1] = -.001
        rotated = task_v4.displacement_velocity(moved, turned)[0]
        self.assertAlmostEqual(abs(float(rotated[1])), .05, places=5)
        self.assertAlmostEqual(float(rotated[0]), 0., places=5)

    def test_touchdown_pays_air_time_above_threshold_on_moving_commands_alone(self):
        commands = torch.tensor([[.05, 0., 0.], [.05, 0., 0.], [.05, 0., 0.], [0., 0., 0.]])
        touchdown = torch.zeros(4, 6, dtype=torch.bool)
        touchdown[:, 0] = True
        air = torch.zeros(4, 6)
        air[:, 0] = torch.tensor([.3, .02, 5., .3])
        _, parts = score(commands, touchdown=touchdown, air_time_s=air)
        c = CONFIG
        expected = torch.tensor([.3 - c.air_time_threshold_s, .02 - c.air_time_threshold_s,
                                 c.air_time_cap_s - c.air_time_threshold_s, 0.]) * c.air_time_weight
        torch.testing.assert_close(parts["air_time"], expected)
        self.assertLess(float(parts["air_time"][1]), 0.)
        # A foot held in the air earns nothing until it lands.
        _, held = score(commands, air_time_s=air)
        self.assertEqual(float(held["air_time"].abs().sum()), 0.)

    def test_penalties_read_executed_motion_and_ignore_the_raw_sample(self):
        commands = torch.tensor([[.05, 0., 0.]])
        quiet_action, _ = score(commands, action=torch.zeros(1, 18))
        loud_action, _ = score(commands, action=torch.full((1, 18), 7.))
        torch.testing.assert_close(quiet_action, loud_action, rtol=0, atol=0)
        step = torch.full((1, 18), CONFIG.target_rate_scale_rad)
        _, parts = score(commands, joint_target_rad=step)
        self.assertAlmostEqual(float(parts["target_rate"]), -CONFIG.target_rate_weight, places=6)
        _, parts = score(commands, linear_velocity_nav=torch.tensor([[0., 0., CONFIG.vertical_velocity_scale_mps]]),
            torque_square_sum_400hz=torch.full((1, 18), 8 * 1.6**2), other_body_force_max_400hz=torch.tensor([2.]),
            requested_torque_abs_max_400hz=torch.full((1, 18), 3.2))
        self.assertAlmostEqual(float(parts["vertical_velocity"]), -CONFIG.vertical_velocity_weight, places=6)
        self.assertAlmostEqual(float(parts["joint_torque"]), -CONFIG.torque_weight, places=6)
        self.assertAlmostEqual(float(parts["collisions"]), -CONFIG.collision_weight, places=6)
        self.assertAlmostEqual(float(parts["torque_limit"]), -CONFIG.torque_limit_weight, places=6)
        pose = telemetry(commands)["root_pose_xyzw"]
        pose[:, 2] = HEIGHT + CONFIG.height_scale_m
        self.assertAlmostEqual(float(score(commands, root_pose_xyzw=pose)[1]["height"]), -CONFIG.height_weight, places=5)

    def test_quiet_terms_apply_to_the_zero_command_alone(self):
        commands = torch.tensor([[0., 0., 0.], [.05, 0., 0.]])
        rate = torch.full((2, 18), CONFIG.quiet_joint_rate_scale_rad_s)
        step = torch.full((2, 18), CONFIG.quiet_target_scale_rad)
        _, parts = score(commands, joint_velocity_rad_s=rate, previous_joint_velocity_rad_s=rate, joint_target_rad=step)
        torch.testing.assert_close(parts["quiet_joint_rate"], torch.tensor([-CONFIG.quiet_joint_rate_weight, 0.]))
        torch.testing.assert_close(parts["quiet_target_motion"], torch.tensor([-CONFIG.quiet_target_weight, 0.]))

    def test_a_fall_costs_more_than_the_discounted_reward_it_avoids(self):
        commands = torch.tensor([[.05, 0., 0.]])
        _, parts = score(commands, terminated=torch.tensor([True]))
        ceiling = CONFIG.linear_tracking_weight + CONFIG.yaw_tracking_weight
        self.assertGreaterEqual(-float(parts["termination"]), 10 * ceiling)
        with self.assertRaisesRegex(ValueError, "termination weight"):
            replace(CONFIG, termination_weight=1.).validate()
        for bad in (dict(tracking_kernel="other"), dict(tracking_window_controls=0), dict(air_time_cap_s=.05),
                    dict(vertical_velocity_scale_mps=0.), dict(torque_weight=-1.)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(CONFIG, **bad).validate()

    def test_missing_or_mismatched_telemetry_is_rejected(self):
        commands = torch.tensor([[.05, 0., 0.]])
        fields = telemetry(commands)
        for name in ("tracked_velocity_nav", "touchdown", "previous_joint_target_rad"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                measured_reward_v4({k: v for k, v in fields.items() if k != name}, commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        with self.assertRaisesRegex(ValueError, "command differs"):
            measured_reward_v4(dict(fields, command=torch.zeros(1, 3)), commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        with self.assertRaisesRegex(ValueError, "nominal plate height"):
            measured_reward_v4(fields, commands, torch.zeros(1, dtype=torch.bool), CONFIG)


class TrainingTaskV4Tests(unittest.TestCase):
    def test_task_times_each_foot_and_restarts_its_memory(self):
        native = GaitDouble(2)
        task = TrainingTaskV4(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [0., 0., 0.]])
        task.window.restart(torch.arange(2), native.commands)
        task.remaining_controls[:] = 1000
        task.step(torch.zeros(2, 18))
        native.forces[:, 0] = 0.
        for _ in range(10):
            task.step(torch.zeros(2, 18))
        self.assertAlmostEqual(float(task.air_time[0, 0]), .2, places=6)
        self.assertEqual(float(task.last_components["air_time"].abs().sum()), 0.)
        native.forces[:, 0, 2] = 12.
        task.step(torch.zeros(2, 18))
        expected = (.2 - CONFIG.air_time_threshold_s) * CONFIG.air_time_weight
        self.assertAlmostEqual(float(task.last_components["air_time"][0]), expected, places=6)
        self.assertEqual(float(task.last_components["air_time"][1]), 0.)
        gait = task.status(reset_interval=True)["interval_gait"]
        self.assertEqual(gait["moving_command_rows"], 12.)
        self.assertAlmostEqual(gait["foot_contact_fraction"][0], 2/12)
        self.assertEqual(gait["foot_contact_fraction"][1], 1.)
        self.assertAlmostEqual(gait["mean_air_time_s"][0], .2, places=6)
        self.assertNotIn("interval_gait", task.status())
        native.forces[:, 1] = 0.
        task.step(torch.zeros(2, 18))
        task.reset(torch.tensor([0]))
        self.assertEqual(float(task.air_time[0].sum()), 0.)
        self.assertTrue(bool(task.in_contact[0].all()))
        self.assertEqual(int(task.window.count[0]), 0)
        self.assertGreater(float(task.air_time[1, 1]), 0.)

    def test_declaration_lists_each_departure_and_keeps_version_one_fields_unused(self):
        with tempfile.TemporaryDirectory() as directory:
            task = TrainingTaskV4(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertEqual(declared["reward_version"], task_v4.REWARD_VERSION)
        self.assertEqual(declared["reward"]["departures_from_table_1"], list(task_v4.DEPARTURES))
        self.assertEqual(set(declared["reward"]["formulas"]) - {"tracked_velocity", "error", "kernel"},
                         set(score(torch.tensor([[.05, 0., 0.]]))[1]))
        self.assertIn("linear_tracking_weight", declared["unused_version1_reward_fields"])
        self.assertFalse(declared["stage2_gates_changed"])
        self.assertEqual(task.status()["reward_version"], task_v4.REWARD_VERSION)

    def test_variants_override_named_coefficients(self):
        variant = task_v4.variant("tracking_window_controls=1,air_time_weight=0,tracking_kernel=absolute")
        self.assertEqual((variant.reward_config.tracking_window_controls, variant.reward_config.air_time_weight,
                          variant.reward_config.tracking_kernel), (1, 0., "absolute"))
        self.assertTrue(issubclass(variant, TrainingTaskV4))
        for bad in ("unknown=1", "air_time_weight", "air_time_weight=1,air_time_weight=2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                task_v4.reward_config(bad)

    def test_entry_and_pack_accept_version_four_for_training_alone(self):
        self.assertEqual(train.checkpoint_reward_version({"identity": {"reward_version": "4"}}), "4")
        with tempfile.TemporaryDirectory() as directory:
            binding = prepare(Path(directory)/"v4", REMOTE, mode="train", reward_version="4")
            args = binding["command_args"]
            self.assertEqual(args[args.index("--reward-version")+1], "4")
            self.assertIn("locomotion/task_v4.py", json.loads((Path(directory)/"v4/source/FREEZE_SHA256.json").read_text()))
            with self.assertRaisesRegex(ValueError, "training only"):
                prepare(Path(directory)/"bad", REMOTE, mode="diagnostic", reward_version="4")


if __name__ == "__main__":
    unittest.main()
