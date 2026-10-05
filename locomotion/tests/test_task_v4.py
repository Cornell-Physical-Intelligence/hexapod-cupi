"""Check reward version 4 without a simulator: noise behaviour, the contact schedule, executed-motion terms."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import task_v2, task_v4, train
from locomotion.prepare import prepare
from locomotion.task import TaskConfig
from locomotion.task_v4 import REWARD_V4_CONFIG as CONFIG, TrainingTaskV4, measured_reward_v4
from locomotion.tests.test_task_v2 import RewardDouble

REMOTE = "/srv/cupi/hexapod/runs/test/reward_v4_fixture"
HEIGHT = .1
LOADED = torch.tensor([[0., 0., 24.]]).expand(6, -1)


class GaitDouble(RewardDouble):
    """RewardDouble plus the toe forces, toe positions, limits and episode count version 4 reads."""

    def __init__(self, n=2):
        super().__init__(n)
        self.forces = LOADED.expand(n, -1, -1).clone()
        self.toes = torch.zeros(n, 6, 3)
        self.world = torch.zeros(n, 6, 3)
        self.current["toe_body"] = self.toes.clone()
        self.current["toe_world"] = self.world.clone()
        self.episode_steps = torch.zeros(n, dtype=torch.long)
        self.lower, self.upper = torch.full((18,), -1.), torch.full((18,), 1.)
        self.neutral = torch.zeros(18)
        self.cfg.action_scale_rad = .35

    def step(self, action):
        output = super().step(action)
        self.episode_steps += 1
        self.telemetry.update(tibia_floor_force_world_n=self.forces.clone(), toe_xyz_body=self.toes.clone(),
                              toe_xyz_world=self.world.clone(), computed_torque_nm=torch.zeros(self.num_envs, 18))
        self.current["toe_body"] = self.toes.clone()
        self.current["toe_world"] = self.world.clone()
        return output


def telemetry(commands, **overrides):
    """One control of a robot at rest on six loaded feet, far from every limit."""
    n = len(commands)
    zeros = torch.zeros(n, 18)
    pose = torch.zeros(n, 7)
    pose[:, 2], pose[:, 6] = HEIGHT, 1.
    result = {"linear_velocity_nav": torch.zeros(n, 3), "angular_velocity_body": torch.zeros(n, 3),
        "root_pose_xyzw": pose, "joint_velocity_rad_s": zeros, "previous_joint_velocity_rad_s": zeros,
        "joint_target_rad": zeros, "previous_joint_target_rad": zeros, "torque_square_sum_400hz": zeros,
        "requested_torque_abs_max_400hz": zeros, "computed_torque_nm": zeros,
        "other_body_force_max_400hz": torch.zeros(n), "tibia_floor_force_world_n": LOADED.expand(n, -1, -1).clone(),
        "tracked_velocity_nav": torch.zeros(n, 3), "scheduled_swing": torch.zeros(n, 6, dtype=torch.bool),
        "toe_velocity_nav": torch.zeros(n, 6, 2), "toe_xyz_nav": torch.zeros(n, 6, 2),
        "joint_limit_margin_rad": torch.ones(n, 18), "executed_action": zeros, "command": commands.clone(), "action": zeros}
    result.update(overrides)
    return result


def score(commands, config=CONFIG, terminated=None, **overrides):
    terminated = torch.zeros(len(commands), dtype=torch.bool) if terminated is None else terminated
    return measured_reward_v4(telemetry(commands, **overrides), commands, terminated, config, HEIGHT)


def lifted(rows, feet):
    force = LOADED.expand(rows, -1, -1).clone()
    force[:, feet] = 0.
    return force


class RewardV4Tests(unittest.TestCase):
    def test_motionless_robot_earns_no_tracking_on_a_moving_command_and_all_of_it_at_rest(self):
        commands = torch.tensor([[.05, 0., 0.], [.025, 0., 0.], [0., 0., .2], [0., 0., 0.]])
        reward, parts = score(commands)
        torch.testing.assert_close(parts["linear_tracking"], torch.tensor([0., 0., 1., 1.]) * CONFIG.linear_tracking_weight)
        torch.testing.assert_close(parts["yaw_tracking"], torch.tensor([1., 1., 0., 1.]) * CONFIG.yaw_tracking_weight)
        # Standing on six loaded feet earns no gait term on any command.
        self.assertEqual(float(parts["gait_schedule"].abs().sum() + parts["swing_travel"].abs().sum()), 0.)
        torch.testing.assert_close(reward, parts["linear_tracking"] + parts["yaw_tracking"] + parts["yaw_rate"])
        tracked = torch.cat((commands[:, :2], commands[:, 2:]), -1)
        _, parts = score(commands, tracked_velocity_nav=tracked)
        torch.testing.assert_close(parts["linear_tracking"], torch.full((4,), CONFIG.linear_tracking_weight))

    def test_error_scale_follows_the_command_and_is_fixed_for_a_zero_component(self):
        commands = torch.tensor([[.05, 0., 0.], [.025, 0., .15], [0., 0., 0.]])
        tracked = torch.tensor([[.04, 0., .02], [.02, 0., .12], [.01, 0., .02]])
        linear, yaw = task_v4.tracking_errors(tracked, commands)
        torch.testing.assert_close(linear, torch.tensor([.2, .2, .01 / CONFIG.fixed_speed_scale_mps]))
        torch.testing.assert_close(yaw, torch.tensor([.02 / CONFIG.fixed_yaw_scale_rad_s, .2, .02 / CONFIG.fixed_yaw_scale_rad_s]))

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
        # A quarter turn of the body: the same world displacement now points to the robot's side.
        turned = previous.clone()
        turned[:, 3:] = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
        moved = turned.clone()
        moved[0, 1] = -.001
        rotated = task_v4.displacement_velocity(moved, turned)[0]
        self.assertAlmostEqual(abs(float(rotated[1])), .05, places=5)
        self.assertAlmostEqual(float(rotated[0]), 0., places=5)

    def test_contact_schedule_pays_unloaded_swing_feet_and_charges_unloaded_stance_feet(self):
        wanted = task_v4.scheduled_swing(torch.tensor([0, 23, 24, 30, 53, 54, 60]))
        self.assertEqual(wanted.int().tolist(), [[1, 0, 1, 0, 1, 0]]*2 + [[0]*6] + [[0, 1, 0, 1, 0, 1]]*2 + [[0]*6] + [[1, 0, 1, 0, 1, 0]])
        commands = torch.tensor([[.05, 0., 0.]]).expand(4, -1).clone()
        force = LOADED.expand(4, -1, -1).clone()                      # every foot carries twice its even share
        force[1, [0, 2, 4], 2] = 0.                                    # scheduled tripod fully lifted
        force[2, [0, 2, 4], 2] = CONFIG.schedule_load_n / 2            # the same feet half unloaded
        force[3, [1, 3, 5], 2] = 0.                                    # the other tripod lifted instead
        swing = wanted[:1].expand(4, -1)
        _, parts = score(commands, tibia_floor_force_world_n=force, scheduled_swing=swing)
        torch.testing.assert_close(parts["gait_schedule"], torch.tensor([0., 1., .5, -1.]) * CONFIG.schedule_weight, atol=1e-6, rtol=0)
        _, parts = score(torch.zeros(4, 3), tibia_floor_force_world_n=force, scheduled_swing=swing)
        self.assertEqual(float(parts["gait_schedule"].abs().sum()), 0.)

    def test_swing_travel_pays_unloaded_feet_that_advance_with_the_command_up_to_a_ceiling(self):
        commands = torch.tensor([[.05, 0., 0.]]).expand(4, -1).clone()
        velocity = torch.zeros(4, 6, 2)
        velocity[0, :3, 0] = .03      # three lifted feet advance over the floor at 0.6 times the command
        velocity[1, :3, 0] = -.03     # the same feet move back through the air
        velocity[2, :3, 0] = 5.       # a spike saturates at the per-foot clip and the ceiling
        velocity[3, 3:, 0] = .03      # loaded feet earn nothing
        _, parts = score(commands, tibia_floor_force_world_n=lifted(4, [0, 1, 2]), toe_velocity_nav=velocity)
        unit = 3 * .6 / CONFIG.swing_travel_full * CONFIG.swing_travel_weight
        torch.testing.assert_close(parts["swing_travel"], torch.tensor([unit, -unit, CONFIG.swing_travel_weight, 0.]), atol=1e-6, rtol=0)
        # A yaw command asks each toe to move along its lever arm.
        turning = torch.tensor([[0., 0., .2]])
        toe = torch.zeros(1, 6, 2)
        toe[0, 0] = torch.tensor([.2, 0.])
        tangent = torch.zeros(1, 6, 2)
        tangent[0, 0, 1] = .04
        _, parts = score(turning, tibia_floor_force_world_n=lifted(1, [0]), toe_velocity_nav=tangent, toe_xyz_nav=toe)
        self.assertAlmostEqual(float(parts["swing_travel"]), CONFIG.swing_travel_weight / CONFIG.swing_travel_full, places=5)
        _, parts = score(torch.zeros(1, 3), tibia_floor_force_world_n=lifted(1, [0]), toe_velocity_nav=tangent, toe_xyz_nav=toe)
        self.assertEqual(float(parts["swing_travel"]), 0.)

    def test_penalties_read_executed_motion_and_ignore_the_raw_sample(self):
        commands = torch.tensor([[.05, 0., 0.]])
        quiet_action, _ = score(commands, action=torch.zeros(1, 18))
        loud_action, _ = score(commands, action=torch.full((1, 18), 7.))
        torch.testing.assert_close(quiet_action, loud_action, rtol=0, atol=0)
        c = CONFIG
        step = torch.full((1, 18), c.target_rate_scale_rad)
        self.assertAlmostEqual(float(score(commands, joint_target_rad=step)[1]["target_rate"]), -c.target_rate_weight, places=6)
        gyro = torch.tensor([[c.roll_pitch_scale_rad_s, c.roll_pitch_scale_rad_s, c.yaw_rate_scale_rad_s]])
        _, parts = score(commands, linear_velocity_nav=torch.tensor([[0., 0., c.vertical_velocity_scale_mps]]),
            angular_velocity_body=gyro, torque_square_sum_400hz=torch.full((1, 18), 8 * 1.6**2),
            other_body_force_max_400hz=torch.tensor([2.]), requested_torque_abs_max_400hz=torch.full((1, 18), 3.2),
            computed_torque_nm=torch.full((1, 18), -1.5), joint_limit_margin_rad=torch.full((1, 18), c.joint_margin_rad / 2),
            executed_action=torch.full((1, 18), -(1 + c.action_limit_onset) / 2))
        for name, weight in (("vertical_velocity", c.vertical_velocity_weight), ("roll_pitch_rate", c.roll_pitch_weight),
                             ("yaw_rate", c.yaw_rate_weight), ("joint_torque", c.torque_weight),
                             ("collisions", c.collision_weight), ("torque_limit", c.torque_limit_weight),
                             ("over_rating", c.over_rating_weight / 2), ("joint_margin", c.joint_margin_weight * 18 / 4),
                             ("action_limit", c.action_limit_weight / 4)):
            self.assertAlmostEqual(float(parts[name]), -weight, places=5, msg=name)
        pose = telemetry(commands)["root_pose_xyzw"]
        pose[:, 2] = HEIGHT + c.height_scale_m
        self.assertAlmostEqual(float(score(commands, root_pose_xyzw=pose)[1]["height"]), -c.height_weight, places=5)
        # Requested torque below the onset, joints far from their limits and targets inside the onset cost nothing.
        _, parts = score(commands, computed_torque_nm=torch.full((1, 18), 1.39),
                         executed_action=torch.full((1, 18), c.action_limit_onset))
        self.assertEqual(float(parts["over_rating"] + parts["joint_margin"] + parts["action_limit"]), 0.)

    def test_quiet_terms_apply_to_the_zero_command_alone(self):
        commands = torch.tensor([[0., 0., 0.], [.05, 0., 0.]])
        rate = torch.full((2, 18), CONFIG.quiet_joint_rate_scale_rad_s)
        step = torch.full((2, 18), CONFIG.quiet_target_scale_rad)
        _, parts = score(commands, joint_velocity_rad_s=rate, previous_joint_velocity_rad_s=rate, joint_target_rad=step,
                         tibia_floor_force_world_n=lifted(2, [0, 1, 2]))
        torch.testing.assert_close(parts["quiet_joint_rate"], torch.tensor([-CONFIG.quiet_joint_rate_weight, 0.]))
        torch.testing.assert_close(parts["quiet_target_motion"], torch.tensor([-CONFIG.quiet_target_weight, 0.]))
        torch.testing.assert_close(parts["quiet_contact"], torch.tensor([-CONFIG.quiet_contact_weight / 2, 0.]))

    def test_a_fall_costs_more_than_the_discounted_reward_it_avoids(self):
        commands = torch.tensor([[.05, 0., 0.]])
        _, parts = score(commands, terminated=torch.tensor([True]))
        ceiling = CONFIG.linear_tracking_weight + CONFIG.yaw_tracking_weight
        self.assertGreaterEqual(-float(parts["termination"]), 10 * ceiling)
        with self.assertRaisesRegex(ValueError, "termination weight"):
            replace(CONFIG, termination_weight=1.).validate()
        for bad in (dict(tracking_window_controls=0), dict(schedule_period_controls=5), dict(schedule_swing_fraction=.6),
                    dict(vertical_velocity_scale_mps=0.), dict(torque_weight=-1.), dict(over_rating_onset_nm=1.6),
                    dict(forward_draw_fraction=1.5), dict(swing_travel_full=0.), dict(action_limit_onset=1.)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(CONFIG, **bad).validate()

    def test_missing_or_mismatched_telemetry_is_rejected(self):
        commands = torch.tensor([[.05, 0., 0.]])
        fields = telemetry(commands)
        for name in ("tracked_velocity_nav", "scheduled_swing", "previous_joint_target_rad", "toe_velocity_nav",
                     "joint_limit_margin_rad", "computed_torque_nm"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                measured_reward_v4({k: v for k, v in fields.items() if k != name}, commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        with self.assertRaisesRegex(ValueError, "command differs"):
            measured_reward_v4(dict(fields, command=torch.zeros(1, 3)), commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        with self.assertRaisesRegex(ValueError, "nominal plate height"):
            measured_reward_v4(fields, commands, torch.zeros(1, dtype=torch.bool), CONFIG)


class TrainingTaskV4Tests(unittest.TestCase):
    def moving_task(self, n=2, **options):
        native = GaitDouble(n)
        task = TrainingTaskV4(native, reward_config=replace(CONFIG, forward_draw_fraction=0., **options))
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [0., 0., 0.]])[:n]
        task.window.restart(torch.arange(n), native.commands)
        task.remaining_controls[:] = 1000
        return native, task

    def test_task_follows_the_schedule_times_each_foot_and_restarts_its_memory(self):
        native, task = self.moving_task()
        # Controls 0 to 23 schedule tripod lf, lr, rm; lift those feet for ten controls.
        native.forces[:, [0, 2, 4]] = 0.
        for _ in range(10):
            task.step(torch.zeros(2, 18))
        self.assertAlmostEqual(float(task.last_components["gait_schedule"][0]), CONFIG.schedule_weight, places=6)
        self.assertEqual(float(task.last_components["gait_schedule"][1]), 0.)
        self.assertAlmostEqual(float(task.last_components["quiet_contact"][1]), -CONFIG.quiet_contact_weight / 2, places=6)
        self.assertAlmostEqual(float(task.air_time[0, 0]), .2, places=6)
        native.forces[:] = LOADED
        task.step(torch.zeros(2, 18))
        gait = task.status(reset_interval=True)["interval_gait"]
        self.assertEqual(gait["moving_command_rows"], 11.)
        self.assertAlmostEqual(gait["foot_contact_fraction"][0], 1/11)
        self.assertEqual(gait["foot_contact_fraction"][1], 1.)
        self.assertAlmostEqual(gait["mean_air_time_s"][0], .2, places=6)
        self.assertNotIn("interval_gait", task.status())
        # After control 24 the same lift falls outside the swing window and costs the schedule term.
        native.episode_steps[:] = 30
        native.forces[:, [0, 2, 4]] = 0.
        task.step(torch.zeros(2, 18))
        self.assertAlmostEqual(float(task.last_components["gait_schedule"][0]), -CONFIG.schedule_weight, places=6)
        task.reset(torch.tensor([0]))
        self.assertEqual(float(task.air_time[0].sum()), 0.)
        self.assertTrue(bool(task.in_contact[0].all()))
        self.assertEqual(int(task.window.count[0]), 0)
        self.assertGreater(float(task.air_time[1, 0]), 0.)

    def test_swing_travel_reads_toe_motion_over_the_floor_and_the_margin_reads_joint_limits(self):
        native, task = self.moving_task(1)
        task.step(torch.zeros(1, 18))
        native.forces[0, 0] = 0.
        native.world[0, 0, 1] -= .05 * .02            # navigation forward is native -Y: 0.05 m/s over the floor
        native.position[0, 0] = .95                   # 0.05 rad from the upper limit of the double
        task.step(torch.zeros(1, 18))
        self.assertAlmostEqual(float(task.last_components["swing_travel"]),
                               CONFIG.swing_travel_weight / CONFIG.swing_travel_full, places=5)
        self.assertAlmostEqual(float(task.last_components["joint_margin"]), -CONFIG.joint_margin_weight * .25, places=5)

    def test_forward_draw_share_replaces_moving_draws_and_keeps_the_zero_quota(self):
        commands = {}
        for share in (0., 1.):
            native = GaitDouble(64)
            task = TrainingTaskV4(native, TaskConfig(seed=5), reward_config=replace(CONFIG, forward_draw_fraction=share))
            task.reset()
            commands[share] = native.commands.clone()
            self.assertEqual(task.zero_command_draws, math.ceil(.15 * 64))
        zero = (commands[1.] == 0).all(-1)
        self.assertEqual(int(zero.sum()), math.ceil(.15 * 64))
        torch.testing.assert_close(commands[1.][~zero], torch.tensor([[.05, 0., 0.]]).expand(int((~zero).sum()), -1))
        self.assertGreater(len(commands[0.][~(commands[0.] == 0).all(-1)].unique(dim=0)), 5)

    def test_declaration_lists_each_departure_and_keeps_version_one_fields_unused(self):
        with tempfile.TemporaryDirectory() as directory:
            task = TrainingTaskV4(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertEqual(declared["reward_version"], task_v4.REWARD_VERSION)
        self.assertEqual(declared["reward"]["departures_from_table_1"], list(task_v4.DEPARTURES))
        self.assertEqual(set(declared["reward"]["formulas"]) - {"tracked_velocity", "error"},
                         set(score(torch.tensor([[.05, 0., 0.]]))[1]))
        self.assertIn("linear_tracking_weight", declared["unused_version1_reward_fields"])
        self.assertFalse(declared["stage2_gates_changed"])
        self.assertEqual(task.status()["reward_version"], task_v4.REWARD_VERSION)

    def test_variants_override_named_coefficients(self):
        variant = task_v4.variant("tracking_window_controls=1,schedule_weight=0,forward_draw_fraction=0.5")
        self.assertEqual((variant.reward_config.tracking_window_controls, variant.reward_config.schedule_weight,
                          variant.reward_config.forward_draw_fraction), (1, 0., .5))
        self.assertTrue(issubclass(variant, TrainingTaskV4))
        for bad in ("unknown=1", "schedule_weight", "schedule_weight=1,schedule_weight=2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                task_v4.reward_config(bad)

    def test_entry_and_pack_accept_version_four_for_training_alone(self):
        self.assertEqual(train.checkpoint_reward_version({"identity": {"reward_version": "4"}}), "4")
        with tempfile.TemporaryDirectory() as directory:
            binding = prepare(Path(directory)/"v4", REMOTE, mode="train", reward_version="4", gait_clock=60)
            args = binding["command_args"]
            self.assertEqual(args[args.index("--reward-version")+1], "4")
            self.assertEqual(args[args.index("--gait-clock")+1], "60")
            self.assertIn("locomotion/task_v4.py", json.loads((Path(directory)/"v4/source/FREEZE_SHA256.json").read_text()))
            with self.assertRaisesRegex(ValueError, "training only"):
                prepare(Path(directory)/"bad", REMOTE, mode="diagnostic", reward_version="4")


if __name__ == "__main__":
    unittest.main()
