"""The yaw-gated contact schedule: reward v4 plus one term that withholds schedule pay on a yaw error."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import task_v4, task_v4_yaw_gate as gated
from dataclasses import replace

from locomotion.task_v4_yaw_gate import REWARD_V4_YAW_GATE_CONFIG as CONFIG, measured_reward_v4_yaw_gate, yaw_gate

ALL = replace(CONFIG, yaw_gate_on="all")
from locomotion.tests.test_task_v4 import HEIGHT, LOADED, GaitDouble, telemetry

SWING_A = torch.tensor([True, False, True, False, True, False])


def scores(commands, stride_yaw, force=None, config=CONFIG):
    """Both rewards for rows whose scheduled tripod lf, lr, rm is lifted unless ``force`` says otherwise."""
    n = len(commands)
    if force is None:
        force = LOADED.expand(n, -1, -1).clone()
        force[:, [0, 2, 4], 2] = 0.
    values = telemetry(commands, tibia_floor_force_world_n=force, scheduled_swing=SWING_A.expand(n, -1),
                       gate_yaw_rate_rad_s=torch.as_tensor(stride_yaw, dtype=torch.float32))
    alive = torch.zeros(n, dtype=torch.bool)
    return (task_v4.measured_reward_v4(values, commands, alive, config, HEIGHT),
            measured_reward_v4_yaw_gate(values, commands, alive, config, HEIGHT))


class YawGateTests(unittest.TestCase):
    def test_gate_follows_the_yaw_error_relative_to_the_command_or_its_floor(self):
        commands = torch.tensor([[0., 0., .2], [0., 0., .2], [0., 0., .2], [.05, 0., 0.], [.05, 0., 0.]])
        gate = yaw_gate(torch.tensor([.2, .1, 0., 0., .065]), commands)
        torch.testing.assert_close(gate, torch.tensor([1., .75, 0., 1., 1 - .65**2]))

    def test_stepping_without_turning_earns_no_schedule_pay_on_a_turn_command(self):
        commands = torch.tensor([[0., 0., .2]]).expand(3, -1).clone()
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, .1, 0.]))
        pay = CONFIG.schedule_weight  # the scheduled tripod fully unloaded
        torch.testing.assert_close(v4_parts["gait_schedule"], torch.full((3,), pay))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -.25, -1.]) * pay)
        torch.testing.assert_close(reward - v4_reward, parts["schedule_yaw_gate"])

    def test_default_gate_leaves_translations_and_covers_turns_and_arcs(self):
        self.assertEqual(CONFIG.yaw_gate_on, "turn")
        commands = torch.tensor([[.05, 0., 0.], [-.035, .035, 0.], [0., 0., .2], [.04, 0., .15]])
        _, (_, parts) = scores(commands, torch.zeros(4))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., 0., -1., -1.]) * CONFIG.schedule_weight)

    def test_heading_drift_on_a_translation_command_costs_part_of_the_pay_when_gating_all(self):
        commands = torch.tensor([[.05, 0., 0.]]).expand(2, -1).clone()
        _, (_, parts) = scores(commands, torch.tensor([0., .065]), config=ALL)
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -(.65**2)]) * CONFIG.schedule_weight)
        _, (_, default) = scores(commands, torch.tensor([0., .065]))
        self.assertEqual(default["schedule_yaw_gate"].tolist(), [0., 0.])

    def test_the_gate_never_lowers_the_charge_for_lifting_stance_feet(self):
        commands = torch.tensor([[0., 0., .2]]).expand(2, -1).clone()
        force = LOADED.expand(2, -1, -1).clone()
        force[:, [1, 3, 5], 2] = 0.  # the wrong tripod lifted
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, 0.]), force)
        torch.testing.assert_close(v4_parts["gait_schedule"], torch.full((2,), -CONFIG.schedule_weight))
        self.assertEqual(parts["schedule_yaw_gate"].tolist(), [0., 0.])
        torch.testing.assert_close(reward, v4_reward)

    def test_zero_command_and_every_v4_component_are_unchanged(self):
        commands = torch.tensor([[0., 0., 0.], [.05, 0., 0.]])
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.3, .3]))
        self.assertEqual(float(parts["schedule_yaw_gate"][0]), 0.)
        for name, value in v4_parts.items():
            torch.testing.assert_close(parts[name], value, msg=name)
        self.assertEqual(set(parts) - set(v4_parts), {"schedule_yaw_gate"})

    def test_missing_stride_yaw_is_rejected(self):
        commands = torch.tensor([[0., 0., .2]])
        with self.assertRaisesRegex(ValueError, "gate_yaw_rate_rad_s"):
            measured_reward_v4_yaw_gate(telemetry(commands), commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)

    def test_task_averages_yaw_wobble_over_a_stride_and_keeps_drift(self):
        native = GaitDouble(2)
        task = gated.variant("forward_draw_fraction=0")(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [.05, 0., 0.]])
        task.window.restart(torch.arange(2), native.commands)
        task.gate_window.restart(torch.arange(2), native.commands)
        task.remaining_controls[:] = 1000
        heading = torch.zeros(2)
        original = native.step
        def turning(action):
            output = original(action)
            k = native.steps
            # Row 0 wobbles +-0.1 rad/s within each 60-control stride; row 1 also drifts at 0.065 rad/s.
            rate = torch.tensor([.1, .1]) * math.sin(2 * math.pi * k / 60) + torch.tensor([0., .065])
            heading.add_(rate * .02)
            pose = native.current["root"].clone()
            pose[:, 3:] = torch.stack((torch.zeros(2), torch.zeros(2), torch.sin(heading / 2), torch.cos(heading / 2)), -1)
            native.current["root"] = pose
            native.telemetry["root_pose_xyzw"] = pose.clone()
            return output
        native.step = turning
        for _ in range(120):
            task.step(torch.zeros(2, 18))
        stride = task.gate_window.update(torch.zeros(2, 1), native.commands)  # one more zero sample
        self.assertLess(abs(float(stride[0])), .01)
        self.assertAlmostEqual(float(stride[1]), .065, delta=.01)

    def test_task_applies_the_gate_and_declares_its_experimental_version(self):
        native = GaitDouble(2)
        task = gated.variant("forward_draw_fraction=0")(native)
        task.reset()
        native.commands[:] = torch.tensor([[0., 0., .2], [.05, 0., 0.]])
        task.window.restart(torch.arange(2), native.commands)
        task.remaining_controls[:] = 1000
        native.forces[:, [0, 2, 4]] = 0.  # follow the schedule without turning
        task.step(torch.zeros(2, 18))
        gate_term = task.last_components["schedule_yaw_gate"]
        self.assertLess(float(gate_term[0]), 0.)
        self.assertEqual(float(gate_term[1]), 0.)
        with tempfile.TemporaryDirectory() as directory:
            gated.TrainingTaskV4YawGate(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertEqual(declared["reward_version"], gated.REWARD_VERSION)
        self.assertIn("schedule_yaw_gate", declared["reward"]["formulas"])
        self.assertEqual(declared["reward"]["config"]["yaw_gate_scale_rad_s"], CONFIG.yaw_gate_scale_rad_s)
        self.assertEqual(task.status()["reward_version"], gated.REWARD_VERSION)

    def test_command_threshold_leaves_arcs_and_keeps_turns_in_place(self):
        commands = torch.tensor([[.04, 0., .15], [0., 0., .2], [0., 0., -.2]])
        _, (_, parts) = scores(commands, torch.zeros(3), config=replace(CONFIG, yaw_gate_min_command_rad_s=.18))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -1., -1.]) * CONFIG.schedule_weight)

    def test_bonus_pays_a_matched_turn_above_v4_and_nothing_without_a_turn(self):
        bonus = replace(CONFIG, yaw_gate_bonus=.5)
        commands = torch.tensor([[0., 0., .2]]).expand(4, -1).clone()
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, .2*(1-(1/3)**.5), .1, 0.]), config=bonus)
        pay = CONFIG.schedule_weight
        # gate 1 -> +50 %; gate 2/3 (the break-even 1/(1+b)) -> 0; gate 0.75 -> +12.5 %; no turn -> -100 %.
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([.5, 0., .125, -1.]) * pay, atol=1e-5, rtol=0)
        torch.testing.assert_close(reward - v4_reward, parts["schedule_yaw_gate"])
        torch.testing.assert_close(v4_parts["gait_schedule"] + parts["schedule_yaw_gate"], torch.tensor([1.5, 1., 1.125, 0.]) * pay, atol=1e-5, rtol=0)

    def test_floor_keeps_part_of_the_pay_and_still_rewards_turning_more(self):
        floor = replace(CONFIG, yaw_gate_bonus=.5, yaw_gate_floor=.5)
        commands = torch.tensor([[0., 0., .2]]).expand(3, -1).clone()
        (v4_reward, v4_parts), (_, parts) = scores(commands, torch.tensor([0., .1, .2]), config=floor)
        pay = CONFIG.schedule_weight
        # gate 0 -> half the v4 pay; gate 0.75 -> 0.5 + 1.0 * 0.75 = 1.25; gate 1 -> 1.5.
        torch.testing.assert_close(v4_parts["gait_schedule"] + parts["schedule_yaw_gate"], torch.tensor([.5, 1.25, 1.5]) * pay, atol=1e-5, rtol=0)
        self.assertEqual(CONFIG.yaw_gate_floor, 0.)
        for bad in ("yaw_gate_floor=-0.1", "yaw_gate_floor=1.5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)

    def test_new_options_default_to_the_withholding_gate(self):
        self.assertEqual((CONFIG.yaw_gate_min_command_rad_s, CONFIG.yaw_gate_bonus, CONFIG.gate_window_controls), (0., 0., 60))
        for bad in ("yaw_gate_bonus=-1", "yaw_gate_min_command_rad_s=-0.1", "gate_window_controls=300"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)
        self.assertEqual(gated.reward_config("gate_window_controls=120").gate_window_controls, 120)

    def test_gate_off_adds_nothing(self):
        commands = torch.tensor([[0., 0., .2], [.05, 0., 0.]])
        (v4_reward, _), (reward, parts) = scores(commands, torch.zeros(2), config=replace(CONFIG, yaw_gate_on="none"))
        self.assertEqual(parts["schedule_yaw_gate"].tolist(), [0., 0.])
        torch.testing.assert_close(reward, v4_reward)

    def test_turn_draws_replace_a_share_of_moving_draws_and_keep_the_zero_quota(self):
        def draws(share, n=400):
            native = GaitDouble(n)
            task = gated.variant(f"forward_draw_fraction=0,turn_draw_fraction={share}")(native)
            task.reset()
            return native.commands.clone(), task
        base, _ = draws(0.)
        turned, task = draws(.5)
        zero = (base == 0).all(-1)
        self.assertTrue(torch.equal((turned == 0).all(-1), zero))
        pure_turn = (turned[:, :2] == 0).all(-1) & (turned[:, 2] != 0)
        share = float(pure_turn[~zero].float().mean())
        self.assertGreater(share, .45)
        self.assertEqual({round(x, 6) for x in turned[pure_turn, 2].tolist()}, {-.2, .2})
        default, _ = draws(0.)
        self.assertTrue(torch.equal(default, base))  # no extra generator draws at the default
        with self.assertRaises(ValueError):
            gated.reward_config("turn_draw_fraction=1.5")

    def test_variants_override_v4_and_gate_options(self):
        variant = gated.variant("yaw_gate_scale_rad_s=0.05,forward_draw_fraction=0")
        self.assertEqual((variant.reward_config.yaw_gate_scale_rad_s, variant.reward_config.forward_draw_fraction), (.05, 0.))
        self.assertTrue(issubclass(variant, task_v4.TrainingTaskV4))
        self.assertEqual(gated.variant("yaw_gate_on=all").reward_config.yaw_gate_on, "all")
        for bad in ("yaw_gate_scale_rad_s=0", "unknown=1", "yaw_gate_on=sometimes"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)


class NativeSelectionTests(unittest.TestCase):
    OPTIONS = "forward_draw_fraction=0,turn_draw_fraction=0.3,gate_window_controls=120,yaw_gate_bonus=0.25"

    def test_prepare_selects_the_yaw_gate_task_for_version_four_training_alone(self):
        from locomotion.prepare import prepare
        from locomotion.tests.test_task_v4 import REMOTE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare(root/"gate", REMOTE, mode="train", reward_version="4", gait_clock=60,
                           reward_options=self.OPTIONS, reward_task="v4_yaw_gate")["command_args"]
            self.assertEqual(args[args.index("--reward-task")+1], "v4_yaw_gate")
            pack = json.loads((root/"gate/PACK.json").read_text())
            self.assertEqual((pack["reward_task"], pack["reward_options"]), ("v4_yaw_gate", self.OPTIONS))
            self.assertIn("locomotion/task_v4_yaw_gate.py", json.loads((root/"gate/source/FREEZE_SHA256.json").read_text()))
            plain = prepare(root/"plain", REMOTE, mode="train", reward_version="4")["command_args"]
            self.assertNotIn("--reward-task", plain)
            self.assertNotIn("reward_task", json.loads((root/"plain/PACK.json").read_text()))
            for index, bad in enumerate((dict(reward_version="2", reward_task="v4_yaw_gate"),
                                         dict(reward_version="4", reward_task="v5"))):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "reward task"):
                    prepare(root/f"bad{index}", REMOTE, mode="train", **bad)

    def test_train_checks_the_task_and_its_options_before_launch(self):
        from locomotion import train
        base = ["--asset", "a", "--model", "m", "--geometry", "g", "--geometry-extrema", "e", "--stance", "s",
                "--output", "o", "--source-freeze-sha256", "f"*64, "--num-envs", "1", "--preflight-only"]
        with self.assertRaisesRegex(ValueError, "reward task"):
            train.main(base + ["--mode", "train", "--reward-version", "2", "--reward-task", "v4_yaw_gate"])
        with self.assertRaisesRegex(ValueError, "unique RewardV4YawGateConfig"):
            train.main(base + ["--mode", "train", "--reward-version", "4", "--reward-task", "v4_yaw_gate",
                               "--reward-options", "not_a_field=0"])
        # Gate options name no RewardV4Config field, so plain v4 refuses them.
        with self.assertRaisesRegex(ValueError, "unique RewardV4Config"):
            train.main(base + ["--mode", "train", "--reward-version", "4", "--reward-options", "turn_draw_fraction=0.3"])


if __name__ == "__main__":
    unittest.main()
