"""Reward version 2 beside an unchanged version 1, without a simulator."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import paper_reward, reward_scorer, task, task_v2, train
from locomotion.prepare import prepare
from locomotion.task import TaskConfig, TrainingTask, command_bank
from locomotion.task_v2 import RewardV2Config, StrideWindow, TrainingTaskV2, measured_reward_v2
from locomotion.tests.test_task import NativeDouble

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/"locomotion/tests/fixtures/control_trace.npz"
SCALED = replace(task_v2.REWARD_V2_CONFIG, tracking_kernel="scaled")
REMOTE = "/home/orionh/HEXAPOD_runs/restart_20260914/reward_v2_fixture"


class RewardDouble(NativeDouble):
    """NativeDouble plus the joint velocity, raw action and requested torque version 2 reads."""

    def __init__(self, n=3):
        super().__init__(n)
        self.current["dq"] = torch.zeros(n, 18)
        self.velocity = None
        self.requested = torch.zeros(n, 18)

    def reset(self, indices):
        output = super().reset(indices)
        self.current["dq"][indices] = self.rate[indices]
        return output

    def step(self, action):
        output = super().step(action)
        if self.velocity is not None:
            self.telemetry["linear_velocity_nav"] = self.velocity[:, :3].clone()
            self.telemetry["angular_velocity_body"][:, 2] = self.velocity[:, 3]
        self.telemetry.update(action=torch.as_tensor(action, dtype=torch.float32).clone(),
                              requested_torque_abs_max_400hz=self.requested.clone())
        self.current["dq"] = self.rate.clone()
        return output


def telemetry(commands, velocity=None, yaw=None, **overrides):
    n = len(commands)
    zeros = torch.zeros(n, 18)
    result = {"linear_velocity_nav": torch.zeros(n, 3), "angular_velocity_body": torch.zeros(n, 3),
              "joint_velocity_rad_s": zeros, "previous_joint_velocity_rad_s": zeros, "action": zeros,
              "previous_action": zeros, "torque_square_sum_400hz": zeros, "requested_torque_abs_max_400hz": zeros,
              "other_body_force_max_400hz": torch.zeros(n), "command": commands.clone()}
    if velocity is not None:
        result["linear_velocity_nav"][:, :2] = velocity
    if yaw is not None:
        result["angular_velocity_body"][:, 2] = yaw
    result["tracked_planar_velocity_nav"] = result["linear_velocity_nav"][:, :2].clone()
    result["tracked_yaw_rate_rad_s"] = result["angular_velocity_body"][:, 2].clone()
    result.update(overrides)
    return result


def run(native, commands, steps, velocity=None):
    reward_task = TrainingTaskV2(native)
    reward_task.reset()
    native.commands[:] = commands
    reward_task.stride.restart(torch.arange(native.num_envs), native.commands)
    rewards = []
    for _ in range(steps):
        native.velocity = velocity
        rewards.append(reward_task.step(torch.zeros(native.num_envs, 18))["reward"])
    return reward_task, rewards


class RewardV2FormulaTests(unittest.TestCase):
    def test_tracking_error_follows_the_command_scaled_kernel(self):
        commands = torch.tensor([[.05, 0., 0.], [.025, 0., 0.], [0., 0., .2], [.04, 0., .15]])
        velocity = torch.tensor([[.04, .0075], [.02, 0.], [0., 0.], [.04, 0.]])
        yaw = torch.tensor([0., 0., .15, .1])
        _, components = measured_reward_v2(telemetry(commands, velocity, yaw), commands, SCALED)
        linear = torch.tensor([math.exp(-.0125/(.4*.05)), math.exp(-.005/(.4*.025)), 1., 1.])
        yaw_expected = .5*torch.tensor([math.exp(0.), math.exp(0.), math.exp(-.05/(.4*.2)), math.exp(-.05/(.4*.15))])
        torch.testing.assert_close(components["linear_tracking"], linear)
        torch.testing.assert_close(components["yaw_tracking"], yaw_expected)

    def test_motionless_robot_keeps_under_ten_percent_of_every_commanded_component(self):
        bank = torch.tensor(command_bank(TaskConfig()))
        for config in (SCALED, task_v2.REWARD_V2_CONFIG):
            _, components = measured_reward_v2(telemetry(bank), bank, config)
            translating = torch.linalg.vector_norm(bank[:, :2], dim=-1) > 0
            turning = bank[:, 2] != 0
            linear = components["linear_tracking"]/config.linear_tracking_weight
            yaw = components["yaw_tracking"]/config.yaw_tracking_weight
            with self.subTest(kernel=config.tracking_kernel):
                self.assertTrue(bool((linear[translating] < .1).all()))
                self.assertTrue(bool((yaw[turning] < .1).all()))
                torch.testing.assert_close(linear[translating], torch.full((int(translating.sum()),), math.exp(-2.5)))

    def test_zero_command_pays_standing_still_and_costs_motion(self):
        commands = torch.zeros(2, 3)
        velocity = torch.tensor([[0., 0.], [.05, 0.]])
        _, components = measured_reward_v2(telemetry(commands, velocity), commands, SCALED)
        torch.testing.assert_close(components["linear_tracking"], torch.tensor([1., math.exp(-.05/(.4*.025))]))
        torch.testing.assert_close(components["yaw_tracking"], torch.tensor([.5, .5]))

    def test_penalties_read_previous_control_memory_and_limits(self):
        commands = torch.zeros(1, 3)
        rate = torch.zeros(1, 18)
        rate[0, 0] = 51.26548245743669
        action = torch.zeros(1, 18)
        action[0, :2] = torch.tensor([.3, .4])
        requested = torch.full((1, 18), 1.)
        requested[0, 3] = 2.6
        values = telemetry(commands, joint_velocity_rad_s=rate, action=action, requested_torque_abs_max_400hz=requested,
                           torque_square_sum_400hz=torch.full((1, 18), 8.), other_body_force_max_400hz=torch.tensor([1.5]))
        values["linear_velocity_nav"][0, 2] = .1
        values["angular_velocity_body"][0, :2] = torch.tensor([.3, .4])
        _, c = measured_reward_v2(values, commands, SCALED)
        w = SCALED
        expected = {"vertical_velocity": -w.vertical_velocity_weight*.01, "roll_pitch_rate": -w.roll_pitch_weight*.5,
                    "joint_torque": -w.torque_weight*math.sqrt(18.), "joint_acceleration": -w.acceleration_weight*rate[0, 0]/.02,
                    "action_rate": -w.action_rate_weight*.5, "collisions": -w.collision_weight,
                    "torque_limit": -w.torque_limit_weight, "velocity_limit": -w.velocity_limit_weight}
        for name, value in expected.items():
            with self.subTest(term=name):
                self.assertAlmostEqual(float(c[name]), float(value), places=5)

    def test_missing_memory_or_stride_input_is_rejected(self):
        commands = torch.zeros(1, 3)
        for key in ("previous_action", "previous_joint_velocity_rad_s", "tracked_planar_velocity_nav"):
            values = telemetry(commands)
            del values[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                measured_reward_v2(values, commands, task_v2.REWARD_V2_CONFIG)

    def test_configuration_checks(self):
        for changes in ({"tracking_kernel": "paper"}, {"scale_k": 1/math.log(10)}, {"stride_controls": 0},
                        {"minimum_speed_mps": .03}, {"minimum_yaw_rate_rad_s": .2}, {"torque_weight": -1.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(task_v2.REWARD_V2_CONFIG, **changes).validate(TaskConfig())
        self.assertEqual(task_v2.reward_config("tracking_kernel=scaled,scale_k=.3").scale_k, .3)
        with self.assertRaises(ValueError):
            task_v2.reward_config("unknown=1")


class RewardV2TaskTests(unittest.TestCase):
    def test_version_one_remains_the_frozen_default(self):
        self.assertEqual((ROOT/"locomotion/task.py").read_bytes(), (ROOT/"locomotion/tests/fixtures/baseline/task.py").read_bytes())
        self.assertEqual(TrainingTask(NativeDouble(1)).declaration()["reward_version"], task.REWARD_VERSION)
        self.assertNotEqual(task_v2.REWARD_VERSION, task.REWARD_VERSION)

    def test_commands_and_observations_match_version_one(self):
        old, new = NativeDouble(2), RewardDouble(2)
        v1, v2 = TrainingTask(old), TrainingTaskV2(new)
        torch.testing.assert_close(v1.reset()["obs"], v2.reset()["obs"])
        for _ in range(130):
            a, b = v1.step(torch.zeros(2, 18)), v2.step(torch.zeros(2, 18))
            torch.testing.assert_close(a["obs"], b["obs"])
            torch.testing.assert_close(a["critic"], b["critic"])
        self.assertEqual(v1.command_draws, v2.command_draws)
        self.assertNotEqual(v1.status()["reward_version"], v2.status()["reward_version"])
        self.assertEqual(set(v2.last_components), set(paper_reward.CALIBRATED_TERMS) | {
            "linear_tracking", "yaw_tracking", "collisions", "torque_limit", "velocity_limit"})

    def test_command_change_restarts_the_stride_window(self):
        native = RewardDouble(1)
        command = torch.tensor([[.05, 0., 0.]])
        reward_task, _ = run(native, command, 10, velocity=torch.tensor([[.05, 0., 0., 0.]]))
        self.assertEqual(int(reward_task.stride.count[0]), 10)
        native.commands[:] = torch.tensor([[0., .05, 0.]])
        native.velocity = torch.tensor([[0., .05, 0., 0.]])
        reward_task.remaining_controls[:] = 50
        reward_task.step(torch.zeros(1, 18))
        self.assertEqual(int(reward_task.stride.count[0]), 1)
        # Without the restart the window would still hold ten forward controls.
        self.assertAlmostEqual(float(reward_task.last_components["linear_tracking"][0]), 1., places=6)

    def test_window_caps_at_stride_controls(self):
        native = RewardDouble(1)
        reward_task, _ = run(native, torch.tensor([[.05, 0., 0.]]), 61, velocity=torch.tensor([[.05, 0., 0., 0.]]))
        self.assertEqual(int(reward_task.stride.count[0]), 60)

    def test_selected_reset_clears_only_that_replicas_memory(self):
        native = RewardDouble(2)
        reward_task = TrainingTaskV2(native)
        reward_task.reset()
        native.rate[:] = .5
        for value in (.2, .6):
            reward_task.step(torch.full((2, 18), value))
        before = {"action": reward_task.previous_action[1].clone(), "rate": reward_task.previous_joint_velocity[1].clone(),
                  "count": int(reward_task.stride.count[1]), "values": reward_task.stride.values[1].clone()}
        native.rate[0] = .25
        reward_task.reset(torch.tensor([0]))
        self.assertTrue(bool((reward_task.previous_action[0] == 0).all()))
        self.assertTrue(bool((reward_task.previous_joint_velocity[0] == .25).all()))
        self.assertEqual(int(reward_task.stride.count[0]), 0)
        torch.testing.assert_close(reward_task.previous_action[1], before["action"])
        torch.testing.assert_close(reward_task.previous_joint_velocity[1], before["rate"])
        self.assertEqual(int(reward_task.stride.count[1]), before["count"])
        torch.testing.assert_close(reward_task.stride.values[1], before["values"])

    def test_action_rate_and_acceleration_use_the_previous_control(self):
        native = RewardDouble(1)
        reward_task = TrainingTaskV2(native)
        reward_task.reset()
        native.rate[:] = .02
        reward_task.step(torch.full((1, 18), .1))
        w = task_v2.REWARD_V2_CONFIG
        first = reward_task.last_components
        self.assertAlmostEqual(float(first["action_rate"][0]), -w.action_rate_weight*math.sqrt(18*.01), places=6)
        self.assertAlmostEqual(float(first["joint_acceleration"][0]), -w.acceleration_weight*math.sqrt(18)*1., places=6)
        reward_task.step(torch.full((1, 18), .1))
        self.assertEqual(float(reward_task.last_components["action_rate"][0]), 0.)
        self.assertEqual(float(reward_task.last_components["joint_acceleration"][0]), 0.)

    def test_declaration_records_version_config_adaptations_and_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            TrainingTaskV2(RewardDouble(1), output_dir=temporary)
            declaration = json.loads((Path(temporary)/"task_definition.json").read_text())
        self.assertEqual(declaration["reward_version"], task_v2.REWARD_VERSION)
        self.assertNotIn("quiet_cost_tail", declaration)
        reward = declaration["reward"]
        self.assertEqual(reward["config"]["tracking_kernel"], "stride")
        self.assertEqual(reward["kernel_audit_label"], "C")
        self.assertAlmostEqual(reward["stationary_translation_fraction_at_or_above_floor"], math.exp(-2.5))
        self.assertEqual(reward["review"]["status"], "provisional")
        self.assertEqual(len(reward["adaptations"]), len(task_v2.ADAPTATIONS))
        self.assertIn("linear_tracking_weight", declaration["unused_version1_reward_fields"])
        self.assertFalse(declaration["physics_changed"])


class RewardV2ScorerTests(unittest.TestCase):
    def test_training_window_matches_the_scorer_stride_average(self):
        trace = reward_scorer.load_trace(TRACE)
        values, _, _ = reward_scorer.reward_inputs(trace, reward_scorer.root_com_local())
        rows = torch.cat((values["linear_velocity_nav"][:, :2], values["angular_velocity_body"][:, 2:]), -1)
        commands = values["command"]
        replicas = values["replicas"]
        expected = paper_reward.stride_average(rows, commands, replicas, 60)
        window = StrideWindow(replicas, 60, 3)
        window.restart(torch.arange(replicas), commands[:replicas])
        actual = torch.cat([window.update(rows[i:i+replicas], commands[i:i+replicas])
                            for i in range(0, len(rows), replicas)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_offline_scores_match_the_audited_candidate_variants(self):
        trace = reward_scorer.load_trace(TRACE)
        com = reward_scorer.root_com_local()
        for spec, variant in (("", "kernel=stride,penalties=calibrated"),
                              ("tracking_kernel=scaled", "kernel=scaled,penalties=calibrated")):
            with self.subTest(spec=spec):
                ours, _ = reward_scorer.evaluate(trace, task_v2.scorer_reward(spec), TaskConfig(), None, com)
                theirs, _ = reward_scorer.evaluate(trace, paper_reward.variant(variant), TaskConfig(), None, com)
                torch.testing.assert_close(ours, theirs)
        self.assertIs(reward_scorer.load_reward("locomotion.task_v2:scorer_default"), task_v2.scorer_default)


class RewardV2LaunchTests(unittest.TestCase):
    def test_prepare_binds_version_two_for_training_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = prepare(Path(temporary)/"v2", REMOTE, mode="train", reward_version="2")["command_args"]
            self.assertEqual(args[args.index("--reward-version")+1], "2")
            self.assertEqual(json.loads((Path(temporary)/"v2/PACK.json").read_text())["reward_version"], "2")
            self.assertNotIn("--reward-version", prepare(Path(temporary)/"v1", REMOTE, mode="train")["command_args"])
            self.assertTrue((Path(temporary)/"v2/source/locomotion/task_v2.py").exists())
            for mode, version in (("diagnostic", "2"), ("train", "3")):
                with self.subTest(mode=mode, version=version), self.assertRaises(ValueError):
                    prepare(Path(temporary)/"bad", REMOTE, mode=mode, reward_version=version)

    def test_train_rejects_version_two_outside_training(self):
        args = ["--asset", "a", "--model", "m", "--geometry", "g", "--geometry-extrema", "e", "--stance", "s",
                "--output", "o", "--source-freeze-sha256", "f"*64, "--num-envs", "1", "--preflight-only",
                "--mode", "evaluate", "--reward-version", "2"]
        with self.assertRaisesRegex(ValueError, "training only"):
            train.main(args)


if __name__ == "__main__":
    unittest.main()
