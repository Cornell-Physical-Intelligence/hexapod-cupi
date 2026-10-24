"""Experiment destination_v1: the destination, its pay, the five inputs, evaluation and the widened start."""

import math
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch

from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import EXPERIMENT_INPUTS, ppo_config
from locomotion.omniwalk.destination_task import (
    EXPERIMENTS,
    DestinationRewardConfig,
    DestinationTrainingTask,
)
from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.destination import (
    DestinationTracker,
    advance_destination,
    destination_features,
)
from locomotion.omniwalk.mdp.rewards import REWARD_CONFIG
from locomotion.rl.destination_wrapper import DestinationVecEnv
from locomotion.rl.ppo import initialize_learner, widen_inputs
from locomotion.tests.task_double import GaitDouble

FORWARD, LEFT = torch.tensor([[0.05, 0.0, 0.0]]), torch.tensor([[0.0, 0.05, 0.0]])
SETTINGS = EXPERIMENTS["destination_v1"]


def plain(value, **kw):
    return value


def pose_at(x, y, yaw):
    return torch.tensor([[x, y, 0.1, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])


def moving_task(motion, command=FORWARD, reward=None, fall_on=None, remaining=1000):
    """A one-row task whose body follows ``motion(k) -> (x, y, yaw)`` after control k."""
    native = GaitDouble(1)
    native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
    task = DestinationTrainingTask(native, TaskConfig(), None, reward or SETTINGS["reward"])
    task.reset()
    native.commands[:] = command
    task.remaining_controls[:] = remaining
    count, original = {"k": 0}, native.step

    def step(action):
        output = original(action)
        count["k"] += 1
        pose = pose_at(*motion(count["k"]))
        native.current["root"], native.telemetry["root_pose_xyzw"] = pose, pose.clone()
        if count["k"] == fall_on:
            output = {**output, "terminated": torch.tensor([True])}
        return output

    native.step = step
    return task, native


def forward(k, slide=0.0):
    # A forward walk at heading 0 moves along world -y; a slide moves along world +x (left).
    return (slide * 0.02 * k, -0.05 * 0.02 * k, 0.0)


class DestinationTests(unittest.TestCase):
    def test_the_destination_follows_the_command_in_the_turning_body_axes(self):
        start = torch.zeros(3, 3)
        start[2, 2] = math.pi / 2
        commands = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.05, 0.0]])
        destination = start
        for _ in range(100):
            destination = advance_destination(destination, commands)
        expected = torch.tensor([[0.0, -0.1, 0.0], [0.0, 0.0, 0.4], [0.0, 0.1, math.pi / 2]])
        torch.testing.assert_close(destination, expected, atol=1e-5, rtol=0)
        arc = torch.zeros(1, 3)
        for _ in range(500):
            arc = advance_destination(arc, torch.tensor([[0.05, 0.0, 0.2]]))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(arc[0, :2])), 0.5 * math.sin(1.0), places=4
        )

    def test_the_gap_reads_in_body_navigation_axes(self):
        cases = (
            ((0.0, -0.1, 0.0), (0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0]),
            ((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), [0.0, 1.0, 0.0, 1.0]),
            ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), [0.0, 0.0, math.sin(0.3), math.cos(0.3)]),
            ((0.1, 0.0, math.pi / 2), (0.0, 0.0, math.pi / 2), [1.0, 0.0, 0.0, 1.0]),
        )
        for destination, body, expected in cases:
            with self.subTest(destination=destination):
                features = destination_features(
                    pose_at(*body), torch.tensor([destination]), torch.tensor([125])
                )
                torch.testing.assert_close(
                    features[0], torch.tensor(expected + [0.5]), atol=1e-5, rtol=0
                )

    def test_the_tracker_restarts_on_a_new_command_or_episode(self):
        tracker, steps = DestinationTracker(1, "cpu"), torch.zeros(1, dtype=torch.long)
        torch.testing.assert_close(
            tracker.inputs(pose_at(0.0, 0.0, 0.0), FORWARD, steps)[0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        for _ in range(50):
            steps += 1
            features = tracker.inputs(pose_at(0.0, 0.0, 0.0), FORWARD, steps)
        torch.testing.assert_close(
            features[0], torch.tensor([0.5, 0.0, 0.0, 1.0, 0.2]), atol=1e-5, rtol=0
        )
        steps += 1
        torch.testing.assert_close(
            tracker.inputs(pose_at(0.0, 0.0, 0.0), LEFT, steps)[0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        )


class TaskTests(unittest.TestCase):
    def test_the_experiment_differs_from_the_canonical_reward_only_in_its_fields(self):
        base, chosen = asdict(REWARD_CONFIG), asdict(SETTINGS["reward"])
        changed = {key: chosen[key] for key in chosen if base.get(key) != chosen[key]}
        self.assertEqual(
            changed,
            {
                "termination_weight": 32.0,
                "destination_weight": 1.0,
                "destination_distance_m": 0.03,
                "destination_kernel": "exponential",
                "command_controls": 1000,
            },
        )
        self.assertEqual(
            (EXPERIMENT_INPUTS["destination_v1"], SETTINGS["episode_seconds"]), ("ghost", 20.0)
        )

    def test_destination_v2_only_halves_the_pay_every_centimetre(self):
        first, second = EXPERIMENTS["destination_v1"], EXPERIMENTS["destination_v2"]
        changed = {
            key: value
            for key, value in asdict(second["reward"]).items()
            if asdict(first["reward"])[key] != value
        }
        self.assertEqual(list(changed), ["destination_distance_m"])
        self.assertEqual(
            {key: value for key, value in second.items() if key != "reward"},
            {key: value for key, value in first.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v2"], "ghost")
        scale = changed["destination_distance_m"]
        for distance_m, pay in ((0.0, 1.0), (0.005, 0.7071), (0.01, 0.5), (0.02, 0.25)):
            self.assertAlmostEqual(math.exp(-distance_m / scale), pay, places=4)
        second["reward"].validate()

    def test_destination_v3_only_concentrates_the_pay_with_the_rational_kernel(self):
        second, third = EXPERIMENTS["destination_v2"], EXPERIMENTS["destination_v3"]
        changed = {
            key: value
            for key, value in asdict(third["reward"]).items()
            if asdict(second["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "destination_kernel": "rational",
                "destination_distance_m": 0.0037,
                "destination_power": 1.2,
                "destination_quiet": "flat",
                "hold_weight": 2.0,
                "hold_grace_controls": 100,
                "hold_deadband_rad": 0.005,
                "hold_span_rad": 0.045,
                "hold_smoothing_controls": 50,
            },
        )
        self.assertEqual(
            {key: value for key, value in third.items() if key != "reward"},
            {key: value for key, value in second.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v3"], "ghost")
        task, _ = moving_task(lambda k: forward(k, slide=0.02), reward=third["reward"])
        for k in range(1, 26):
            task.step(torch.zeros(1, 18))
            if k in (1, 5, 13, 25):  # 0.4, 2, 5.2 and 10 mm to the side of the destination
                self.assertAlmostEqual(
                    float(task.last_components["destination"][0]),
                    1 / (1 + (0.02 * 0.02 * k / 0.0037) ** 1.2),
                    places=4,
                )
        for bad in (
            dict(destination_power=None),
            dict(destination_power=0.0),
            dict(destination_kernel="exponential"),
            dict(destination_quiet="track"),
            dict(hold_span_rad=None),
            dict(hold_weight=0.0),
            dict(hold_grace_controls=0),
            dict(hold_deadband_rad=-0.001),
            dict(hold_smoothing_controls=101),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(third["reward"], **bad).validate()

    def test_a_zero_command_pays_the_full_destination_and_charges_joint_drift_after_the_grace(self):
        reward = EXPERIMENTS["destination_v3"]["reward"]
        smoothing = reward.hold_smoothing_controls

        def standing(position):
            """Charges and running-mean drift while standing; joint 4 follows ``position(k)``."""
            task, native = moving_task(
                lambda k: (0.001 * k, 0.0, 0.0), command=torch.zeros(1, 3), reward=reward
            )
            start, out, mean, anchor = native.position.clone(), {}, 0.0, None
            for k in range(1, 301):
                native.position[:] = start
                native.position[0, 4] += position(k)
                mean = position(k) if k == 1 else mean + (position(k) - mean) / smoothing
                anchor = mean if k == 100 else anchor
                task.step(torch.zeros(1, 18))
                # The body slides 1 mm per control, yet a zero command pays the full destination.
                self.assertEqual(float(task.last_components["destination"][0]), 1.0)
                out[k] = (float(task.last_components["hold"][0]), mean - (anchor or mean))
            return out

        # Exploration-sized jitter of +-20 mrad averages out and is never charged.
        jitter = standing(lambda k: 0.02 * (-1) ** k)
        self.assertTrue(all(charge == 0.0 for charge, _ in jitter.values()))
        # A creep of 0.2 mrad per control is charged once its running mean leaves the deadband.
        creep = standing(lambda k: 0.0002 * k)
        self.assertTrue(all(creep[k][0] == 0.0 for k in range(1, 125)))
        for k in (150, 200, 300):
            expected = -2.0 * min(max((abs(creep[k][1]) - 0.005) / 0.045, 0), 1) / 18
            self.assertAlmostEqual(creep[k][0], expected, places=5)
        self.assertLess(creep[300][0], -0.07)
        # A moving command pays by distance and charges nothing.
        walking, native = moving_task(lambda k: forward(k, slide=0.02), reward=reward)
        for _ in range(130):
            native.position[0, 4] += 0.01
            walking.step(torch.zeros(1, 18))
        self.assertEqual(float(walking.last_components["hold"][0]), 0.0)
        self.assertLess(float(walking.last_components["destination"][0]), 0.1)

    def test_destination_v4_drops_the_hold_charge_and_pays_nothing_under_a_zero_command(self):
        third, fourth = EXPERIMENTS["destination_v3"], EXPERIMENTS["destination_v4"]
        changed = {
            key: value
            for key, value in asdict(fourth["reward"]).items()
            if asdict(third["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "destination_quiet": "off",
                "hold_weight": None,
                "hold_grace_controls": None,
                "hold_deadband_rad": None,
                "hold_span_rad": None,
                "hold_smoothing_controls": None,
            },
        )
        self.assertEqual(
            {key: value for key, value in fourth.items() if key != "reward"},
            {key: value for key, value in third.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v4"], "ghost")
        # Standing exactly on the destination pays nothing and charges no hold.
        standing, _ = moving_task(
            lambda k: (0.0, 0.0, 0.0), command=torch.zeros(1, 3), reward=fourth["reward"]
        )
        for _ in range(150):
            standing.step(torch.zeros(1, 18))
            self.assertEqual(float(standing.last_components["destination"][0]), 0.0)
            self.assertNotIn("hold", standing.last_components)
        # A moving command still pays by the rational kernel.
        walking, _ = moving_task(lambda k: forward(k, slide=0.02), reward=fourth["reward"])
        for k in range(1, 6):
            walking.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(walking.last_components["destination"][0]),
            1 / (1 + (0.02 * 0.02 * 5 / 0.0037) ** 1.2),
            places=4,
        )

    def test_destination_v5_pays_the_destination_under_a_zero_command_too(self):
        fourth, fifth = EXPERIMENTS["destination_v4"], EXPERIMENTS["destination_v5"]
        changed = {
            key: value
            for key, value in asdict(fifth["reward"]).items()
            if asdict(fourth["reward"])[key] != value
        }
        self.assertEqual(changed, {"destination_quiet": None})
        self.assertEqual(
            {key: value for key, value in fifth.items() if key != "reward"},
            {key: value for key, value in fourth.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v5"], "ghost")
        # Standing 1 mm per control off the destination pays by the rational kernel.
        standing, _ = moving_task(
            lambda k: (0.001 * k, 0.0, 0.0), command=torch.zeros(1, 3), reward=fifth["reward"]
        )
        for _ in range(4):
            standing.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(standing.last_components["destination"][0]),
            1 / (1 + (0.004 / 0.0037) ** 1.2),
            places=4,
        )
        self.assertNotIn("hold", standing.last_components)

    def test_destination_v6_follows_a_share_of_walks_with_a_stop_for_the_rest_of_the_command(self):
        fifth, sixth = EXPERIMENTS["destination_v5"], EXPERIMENTS["destination_v6"]
        changed = {
            key: value
            for key, value in asdict(sixth["reward"]).items()
            if asdict(fifth["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {"stop_fraction": 0.2, "stop_after_min_controls": 200, "stop_after_max_controls": 600},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v6"], "ghost")
        for bad in (
            dict(stop_fraction=0.0),
            dict(stop_after_max_controls=None),
            dict(stop_after_min_controls=700),
            dict(stop_after_max_controls=1000),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(sixth["reward"], **bad).validate()
        # Every moving command stops: it walks 10 to 20 controls, then stands for the rest of 60.
        reward = replace(
            sixth["reward"],
            stop_fraction=1.0,
            stop_after_min_controls=10,
            stop_after_max_controls=20,
            command_controls=60,
        )
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, reward)
        task.reset()
        moving = (native.commands != 0).any(-1)
        self.assertTrue(bool(moving.any()) and bool((~moving).any()))
        walked = task.remaining_controls.clone()
        self.assertTrue(bool(((walked >= 10) & (walked <= 20))[moving].all()))
        self.assertTrue(bool((walked[~moving] == 60).all()))
        first = native.commands.clone()
        # Training truncates the episode when the stop ends at control 60; check up to there.
        for k in range(1, 60):
            task.step(torch.zeros(64, 18))
            stopped = moving & (walked <= k)
            self.assertTrue(bool((native.commands[stopped] == 0).all()))
            self.assertTrue(
                torch.equal(native.commands[~stopped & moving], first[~stopped & moving])
            )
            self.assertTrue(bool((task.remaining_controls[stopped] == 60 - k).all()))
        # A reset drops a stop that its episode no longer reaches.
        task.reset()
        task.stop_pending[:] = True
        task.reset()
        self.assertTrue(bool((task.remaining_controls[(native.commands == 0).all(-1)] == 60).all()))

    def test_destination_v7_pays_for_the_destination_heading_like_the_destination(self):
        fifth, seventh = EXPERIMENTS["destination_v5"], EXPERIMENTS["destination_v7"]
        changed = {
            key: value
            for key, value in asdict(seventh["reward"]).items()
            if asdict(fifth["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "heading_weight": 1.0,
                "heading_width_rad": math.radians(1.0),
                "heading_power": 1.2,
                "termination_weight": 42.0,
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v7"], "ghost")
        reward = seventh["reward"]
        with self.assertRaises(ValueError):
            replace(reward, termination_weight=41.0).validate()
        for bad in (dict(heading_power=None), dict(heading_width_rad=0.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(reward, **bad).validate()
        width = math.radians(1.0)
        # Walking forward, the body turns 0.1 deg per control while the destination keeps its heading.
        walking, _ = moving_task(lambda k: (0.0, -0.001 * k, math.radians(0.1 * k)), reward=reward)
        for k in range(1, 31):
            walking.step(torch.zeros(1, 18))
            if k in (5, 10, 30):
                self.assertAlmostEqual(
                    float(walking.last_components["heading"][0]),
                    1 / (1 + (math.radians(0.1 * k) / width) ** 1.2),
                    places=4,
                )
        # A turn command turns the destination: a body that turns with it is paid in full.
        turning, _ = moving_task(
            lambda k: (0.0, 0.0, 0.2 * 0.02 * k),
            command=torch.tensor([[0.0, 0.0, 0.2]]),
            reward=reward,
        )
        for _ in range(20):
            turning.step(torch.zeros(1, 18))
            self.assertAlmostEqual(float(turning.last_components["heading"][0]), 1.0, places=4)
        # Standing pays too, and the heading gap wraps around +-180 deg.
        standing, _ = moving_task(
            lambda k: (0.0, 0.0, math.radians(359.5)), command=torch.zeros(1, 3), reward=reward
        )
        for _ in range(3):
            standing.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(standing.last_components["heading"][0]),
            1 / (1 + (math.radians(0.5) / width) ** 1.2),
            places=4,
        )

    def test_destination_v8_doubles_every_joint_penalty_of_v7(self):
        seventh, eighth = EXPERIMENTS["destination_v7"], EXPERIMENTS["destination_v8"]
        changed = {
            key: value
            for key, value in asdict(eighth["reward"]).items()
            if asdict(seventh["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "torque_weight": 0.1,
                "acceleration_weight": 0.1,
                "target_rate_weight": 0.1,
                "torque_limit_weight": 1.0,
                "over_rating_weight": 20.0,
                "joint_margin_weight": 2.0,
                "action_limit_weight": 12.0,
                "quiet_joint_rate_weight": 1.0,
                "quiet_target_weight": 1.0,
                "quiet_strain_weight": 20.0,
            },
        )
        self.assertEqual(
            {key: value for key, value in eighth.items() if key != "reward"},
            {key: value for key, value in seventh.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v8"], "ghost")
        eighth["reward"].validate()

    def test_destination_v9_adds_destination_v6_stops_to_v8(self):
        eighth, ninth = EXPERIMENTS["destination_v8"], EXPERIMENTS["destination_v9"]
        changed = {
            key: value
            for key, value in asdict(ninth["reward"]).items()
            if asdict(eighth["reward"])[key] != value
        }
        stops = ("stop_fraction", "stop_after_min_controls", "stop_after_max_controls")
        self.assertEqual(
            changed, {key: getattr(EXPERIMENTS["destination_v6"]["reward"], key) for key in stops}
        )
        self.assertEqual(
            {key: value for key, value in ninth.items() if key != "reward"},
            {key: value for key, value in eighth.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v9"], "ghost")
        ninth["reward"].validate()

    def test_every_control_pays_the_exponential_of_the_distance_and_a_fall_pays_nothing(self):
        task, _ = moving_task(lambda k: forward(k, slide=0.02))
        for k in range(1, 301):
            task.step(torch.zeros(1, 18))
            if k in (1, 75, 300):  # 0.4 mm, 3 cm and 12 cm to the side of the destination
                self.assertAlmostEqual(
                    float(task.last_components["destination"][0]),
                    math.exp(-0.02 * 0.02 * k / 0.03),
                    places=4,
                )
        fell, _ = moving_task(forward, fall_on=3)
        for _ in range(3):
            fell.step(torch.zeros(1, 18))
        self.assertEqual(float(fell.last_components["destination"][0]), 0.0)

    def test_commands_last_the_declared_length_and_the_fall_cost_counts_the_weight(self):
        native = GaitDouble(3)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, SETTINGS["reward"])
        task.reset()
        self.assertEqual(task.remaining_controls.tolist(), [1000] * 3)
        with self.assertRaises(ValueError):
            replace(SETTINGS["reward"], termination_weight=31.0).validate()
        for bad in (dict(destination_kernel="gaussian"), dict(command_controls=5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(SETTINGS["reward"], **bad).validate()
        with self.assertRaises(ValueError):
            DestinationTrainingTask(native, TaskConfig(), None, REWARD_CONFIG)
        self.assertIsInstance(SETTINGS["reward"], DestinationRewardConfig)


class CurriculumTests(unittest.TestCase):
    def test_destination_v10_is_v9_with_a_walking_speed_curriculum(self):
        nine, ten = (
            EXPERIMENTS["destination_v9"]["reward"],
            EXPERIMENTS["destination_v10"]["reward"],
        )
        changed = {key for key, value in asdict(ten).items() if asdict(nine)[key] != value}
        self.assertEqual(
            changed,
            {
                "speed_curriculum_start_mps",
                "speed_curriculum_step_mps",
                "speed_curriculum_final_mps",
                "speed_curriculum_threshold",
                "speed_curriculum_window_controls",
                "speed_curriculum_new_share",
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v10"], "ghost")
        for bad in (
            dict(speed_curriculum_final_mps=0.16),
            dict(speed_curriculum_threshold=1.5),
            dict(speed_curriculum_window_controls=0),
            dict(speed_curriculum_new_share=None),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(ten, **bad).validate()

    def test_walks_take_the_top_or_a_slower_speed_and_the_top_rises_past_the_threshold(self):
        native = GaitDouble(4096)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        reward = EXPERIMENTS["destination_v10"]["reward"]
        task = DestinationTrainingTask(native, TaskConfig(), None, reward)
        task.reset()
        command = native.commands
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        walks = (speed > 0) & (command[:, 2] == 0)
        levels = torch.round(speed[walks] / 0.025).long()
        self.assertEqual(task.curriculum_level, 0.075)
        self.assertEqual(set(levels.tolist()), {1, 2, 3})
        self.assertAlmostEqual(float((levels == 3).double().mean()), 0.5, delta=0.05)
        # Walks point every way; arcs keep v9's 0.04 m/s.
        angle = torch.atan2(command[walks, 1], command[walks, 0])
        self.assertEqual(int(torch.histc(angle, 4, -math.pi, math.pi).min() > 0), 1)
        self.assertTrue(
            torch.allclose(speed[(speed > 0) & (command[:, 2] != 0)], torch.tensor(0.04))
        )
        # A window below the threshold holds the level; one above it raises the level and refills.
        window = reward.speed_curriculum_window_controls
        for _ in range(window):
            task._curriculum_step(torch.full((4096,), 0.79))
        self.assertEqual(task.curriculum_status()["level_mps"], 0.075)
        for level in (0.1, 0.125, 0.15, 0.15):
            for _ in range(window):
                task._curriculum_step(torch.full((4096,), 0.85))
            self.assertEqual(task.curriculum_status()["level_mps"], level)
        status = task.curriculum_status()
        self.assertTrue(status["complete"])
        self.assertEqual([a["level_mps"] for a in status["advances"]], [0.075, 0.1, 0.125, 0.15])


class BackoffTests(unittest.TestCase):
    def test_destination_v11_rehearses_more_and_backs_off_a_falling_level(self):
        ten, eleven = (EXPERIMENTS[n]["reward"] for n in ("destination_v10", "destination_v11"))
        changed = {k for k, v in asdict(eleven).items() if asdict(ten)[k] != v}
        self.assertEqual(
            changed, {"speed_curriculum_new_share", "speed_curriculum_backoff_threshold"}
        )
        with self.assertRaises(ValueError):
            replace(eleven, speed_curriculum_backoff_threshold=0.9).validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, eleven)
        window = eleven.speed_curriculum_window_controls
        # The first level never backs off; a raised level that falls below 0.65 drops one step.
        for _ in range(window):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.075)
        for pay, level in ((0.85, 0.1), (0.85, 0.125), (0.6, 0.1)):
            for _ in range(window):
                task._curriculum_step(torch.full((64,), pay))
            self.assertEqual(task.curriculum_level, level)
        self.assertTrue(task.curriculum_status()["advances"][-1]["back_off"])

    def test_destination_v13_walks_everywhere_and_pays_0_8_at_5_mm(self):
        twelve, thirteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v12", "destination_v13")
        )
        changed = {k for k, v in asdict(thirteen).items() if asdict(twelve)[k] != v}
        self.assertEqual(
            changed,
            {
                "destination_distance_m",
                "speed_curriculum_walks_only",
                "speed_curriculum_window_controls",
            },
        )
        self.assertEqual(thirteen.speed_curriculum_window_controls, 360)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v13"], "ghost")
        width, power = thirteen.destination_distance_m, thirteen.destination_power
        self.assertAlmostEqual(1 / (1 + (0.005 / width) ** power), 0.8)
        with self.assertRaises(ValueError):
            replace(thirteen, speed_curriculum_walks_only=False).validate()
        native = GaitDouble(2048)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, thirteen)
        task.reset()
        command = native.commands
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        # Every row walks without yaw; about a fifth are due to stop.
        self.assertTrue(bool((speed > 0).all()) and bool((command[:, 2] == 0).all()))
        self.assertAlmostEqual(float(task.stop_pending.double().mean()), 0.2, delta=0.03)
        stopping = torch.nonzero(task.stop_pending).flatten()
        task._resample(stopping)
        self.assertTrue(bool((native.commands[stopping] == 0).all()))

    def test_destination_v14_backs_off_only_after_50_updates_at_a_level(self):
        thirteen, fourteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v13", "destination_v14")
        )
        changed = {k for k, v in asdict(fourteen).items() if asdict(thirteen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_backoff_window_controls"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v14"], "ghost")
        with self.assertRaises(ValueError):
            replace(fourteen, speed_curriculum_backoff_window_controls=100).validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, fourteen)
        # 10 updates above 0.8 raise the level each time, up to 0.125 m/s; the window is read every 7
        # controls, so the first read after 360 controls comes at 364.
        for level in (0.1, 0.125):
            for _ in range(364):
                task._curriculum_step(torch.full((64,), 0.9))
            self.assertEqual(task.curriculum_level, level)
        # Low pay holds the level until 1800 controls have passed there, then lowers it once.
        for _ in range(1799):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.125)
        for _ in range(7):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertTrue(task.curriculum_status()["advances"][-1]["back_off"])

    def test_destination_v15_rises_only_after_a_window_without_falls(self):
        fourteen, fifteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v14", "destination_v15")
        )
        changed = {k for k, v in asdict(fifteen).items() if asdict(fourteen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_advance_without_falls"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v15"], "ghost")
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, fifteen)
        none, one = torch.zeros(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool)
        one[3] = True
        # High pay with one fall at control 100 holds the level at the read after 364 controls; the fall
        # leaves the 360-control window at control 460 and the level rises at the next read (462).
        for k in range(364):
            task._curriculum_step(torch.full((64,), 0.9), one if k == 99 else none)
        self.assertEqual(task.curriculum_level, 0.075)
        self.assertEqual(task.curriculum_status()["window_falls"], 1)
        for _ in range(98):
            task._curriculum_step(torch.full((64,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.1)

    def test_destination_v12_is_v10_with_the_back_off(self):
        ten, twelve = (EXPERIMENTS[n]["reward"] for n in ("destination_v10", "destination_v12"))
        changed = {k for k, v in asdict(twelve).items() if asdict(ten)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_backoff_threshold"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v12"], "ghost")


class GaitClockTests(unittest.TestCase):
    def test_speed_v1_is_the_fixed_clock_up_to_0_075_and_shortens_the_cycle_above(self):
        from locomotion.omniwalk.mdp.gait import GaitPhase, gait_cycle
        from locomotion.omniwalk.mdp.rewards import scheduled_swing
        from locomotion.rl.wrapper import clock_features

        command = torch.tensor([[0.0, 0.05, 0.0], [-0.075, 0.0, 0.0], [0.0, 0.0, 0.2]])
        gait = GaitPhase(3, "speed_v1", "cpu")
        for k in range(600):
            steps = torch.full((3,), k)
            swing, period = gait.swing(command)
            self.assertTrue(bool((swing == scheduled_swing(steps)).all()), k)
            torch.testing.assert_close(period, torch.full((3,), 1.2, dtype=torch.float64))
            torch.testing.assert_close(
                clock_features(steps, 60, command, gait.phase),
                clock_features(steps, 60, command),
                atol=1e-5,
                rtol=0,
            )
            gait.advance(command)
        walks = torch.tensor([[0.0875, 0.0, 0.0], [0.0, -0.1, 0.0], [0.125, 0.0, 0.0], [0.3, 0, 0]])
        period, swing = gait_cycle(walks, "speed_v1")
        torch.testing.assert_close(
            period, torch.tensor([56.0, 52.0, 44.0, 40.0], dtype=torch.float64)
        )
        torch.testing.assert_close(swing, torch.tensor([0.44, 0.48, 0.5, 0.5], dtype=torch.float64))

    def test_evaluation_follows_the_phase_the_training_task_kept(self):
        from locomotion.omniwalk.mdp.gait import GaitPhase

        trained, followed = GaitPhase(1, "speed_v1", "cpu"), GaitPhase(1, "speed_v1", "cpu")
        speeds = [0.1] * 70 + [0.125] * 50 + [0.0] * 30 + [0.15] * 40
        for k, speed in enumerate(speeds):
            command = torch.tensor([[speed, 0.0, 0.0]])
            seen = followed.follow(command, torch.tensor([k])).clone()
            torch.testing.assert_close(seen, trained.phase)
            trained.advance(command)
        # A 0.125 m/s walk crosses half its 44-control cycle in 22 controls.
        gait = GaitPhase(1, "speed_v1", "cpu")
        for _ in range(22):
            gait.advance(torch.tensor([[0.125, 0.0, 0.0]]))
        self.assertAlmostEqual(float(gait.phase[0]), 0.5)

    def test_destination_v16_scales_the_clock_and_backs_off_a_robot_5_cm_from_its_ghost(self):
        fifteen, sixteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v15", "destination_v16")
        )
        changed = {k for k, v in asdict(sixteen).items() if asdict(fifteen)[k] != v}
        self.assertEqual(changed, {"gait_clock_scaling", "speed_curriculum_backoff_distance_m"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v16"], "ghost")
        with self.assertRaises(ValueError):
            replace(sixteen, gait_clock_scaling="speed_v0").validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, sixteen)
        near, none = torch.zeros(64), torch.zeros(64, dtype=torch.bool)
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, near)
        self.assertEqual(task.curriculum_level, 0.1)
        # A fallen robot's distance does not count (its fall holds the level); a standing one 6 cm away
        # lowers it at the next read of a full window, however high the pay.
        far, fell = near.clone(), none.clone()
        far[5], fell[5] = 0.2, True
        for k in range(364):
            task._curriculum_step(
                torch.full((64,), 0.9), fell if k == 10 else none, far if k == 10 else near
            )
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertEqual(task.curriculum_status()["window_lost"], 0)
        far[5] = 0.06
        for _ in range(7):
            task._curriculum_step(torch.full((64,), 0.9), none, far)
        self.assertEqual(task.curriculum_level, 0.075)
        # The first level holds while any robot stands that far away, and rises once none has for a window.
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, far)
        self.assertEqual(task.curriculum_level, 0.075)
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, near)
        self.assertEqual(task.curriculum_level, 0.1)
        record = task.curriculum_status()["advances"][-2]
        self.assertEqual((record["back_off"], record["reason"]), (True, "distance"))

    def test_the_wrapper_shows_the_task_phase_and_records_the_scaling(self):
        sixteen = EXPERIMENTS["destination_v16"]["reward"]
        task, native = moving_task(lambda k: (0.0, -0.125 * 0.02 * k, 0.0), reward=sixteen)
        with self.assertRaises(ValueError):
            DestinationVecEnv(task, plain, observation_scaling="fixed", gait_clock=60)
        env = DestinationVecEnv(
            task, plain, observation_scaling="fixed", gait_clock=60, gait_clock_scaling="speed_v1"
        )
        native.commands[:] = torch.tensor([[0.125, 0.0, 0.0]])
        task.remaining_controls[:] = 1000
        env.get_observations()
        for _ in range(22):
            observation, *_ = env.step(torch.zeros(1, 18))
        torch.testing.assert_close(
            observation["policy"][0, 231:233], torch.tensor([0.0, -1.0]), atol=1e-5, rtol=0
        )
        # The reward scheduled tripod B's swing at the half cycle, not the fixed clock's control 22.
        self.assertEqual(task.last_components["gait_schedule"].shape, (1,))
        swing, _ = task.gait.swing(native.commands)
        self.assertEqual(swing[0].tolist(), [False, True, False, True, False, True])


class WrapperTests(unittest.TestCase):
    def wrapped(self, **options):
        task, native = moving_task(lambda k: (0.0, -0.03 * 0.02 * k, 0.0))  # 0.02 m/s short
        env = DestinationVecEnv(task, plain, observation_scaling="fixed", gait_clock=60, **options)
        native.commands[:] = FORWARD
        task.remaining_controls[:] = 1000
        return env, task, native

    def test_both_inputs_gain_five_values_that_track_the_lag(self):
        env, task, native = self.wrapped()
        first = env.get_observations()
        self.assertEqual(
            (first["policy"].shape[-1], first["critic"].shape[-1]), (231 + 2 + 5, 234 + 2 + 5)
        )
        torch.testing.assert_close(first["policy"][0, -5:], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        for _ in range(100):
            observation, *_ = env.step(torch.zeros(1, 18))
        torch.testing.assert_close(
            observation["policy"][0, -5:],
            torch.tensor([0.4, 0.0, 0.0, 1.0, 0.4]),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(observation["critic"][0, -5:], observation["policy"][0, -5:])

    def test_a_command_switch_bootstraps_the_held_destination_and_restarts_the_inputs(self):
        env, task, native = self.wrapped(command_segments="bootstrap")
        env.get_observations()
        for _ in range(50):
            env.step(torch.zeros(1, 18))
        task.bank = LEFT.clone()  # the next draw is a left walk or a stop, either a new command
        task.remaining_controls[:] = 1
        observation, _, done, extras = env.step(torch.zeros(1, 18))
        critic = extras["command_bootstrap_observation"]["critic"]
        # 51 controls at 0.02 m/s short of the command: 0.0204 m behind the held destination.
        torch.testing.assert_close(
            critic[0, -5:], torch.tensor([0.204, 0.0, 0.0, 1.0, 51 / 250]), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            observation["critic"][0, -5:], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        )

    def test_evaluation_rebuilds_the_training_inputs(self):
        from locomotion.scripts.play import actor_input

        env, task, native = self.wrapped()
        args = SimpleNamespace(
            observation_scaling="fixed", gait_clock=60, experiment="destination_v1"
        )
        tracker, seen, trained = DestinationTracker(1, "cpu"), [], []
        for _ in range(30):
            raw = env.current["obs"].clone()
            raw[:, 210:213] = native.commands  # evaluation's observation carries the live command
            seen.append(actor_input(native, args, raw, tracker)[0, -5:].clone())
            observation, *_ = env.step(torch.zeros(1, 18))
            trained.append(observation["policy"][0, -5:].clone())
        for k in range(29):
            torch.testing.assert_close(seen[k + 1], trained[k], atol=1e-5, rtol=0)

    def test_the_learner_configuration_records_the_input(self):
        base = ppo_config(7, gait_clock=60)
        ghost = ppo_config(7, gait_clock=60, destination_input="ghost")
        self.assertEqual(
            ghost["environment_wrapper"],
            dict(base["environment_wrapper"], destination_input="ghost"),
        )
        self.assertEqual(ppo_config(7, gait_clock=60, destination_input="none"), base)


class SurrogateTests(unittest.TestCase):
    def test_surrogate_evaluation_rebuilds_a_native_experiment_checkpoint(self):
        from locomotion.surrogate.evaluate import DEFAULT_TASK, task_and_wrapper
        from locomotion.surrogate.train import resolve

        plain_args = SimpleNamespace(task=None, wrapper=None)
        task, wrapper = task_and_wrapper(plain_args, {"experiment": "destination_v9"})
        self.assertIs(resolve(wrapper), DestinationVecEnv)
        self.assertIsInstance(resolve(task)(GaitDouble(1)), DestinationTrainingTask)
        self.assertEqual(
            resolve(task)(GaitDouble(1)).reward_config, EXPERIMENTS["destination_v9"]["reward"]
        )
        self.assertEqual(
            task_and_wrapper(plain_args, {}), (DEFAULT_TASK, "locomotion.rl.wrapper:OmniVecEnv")
        )
        chosen = SimpleNamespace(task="a:B", wrapper="c:D")
        self.assertEqual(task_and_wrapper(chosen, {"experiment": "destination_v9"}), ("a:B", "c:D"))


class StartTests(unittest.TestCase):
    def test_a_widened_start_computes_what_the_checkpoint_did_and_restarts_the_rate(self):
        torch.manual_seed(0)

        def learner(actor_in, critic_in):
            algorithm = SimpleNamespace(
                actor=torch.nn.Sequential(
                    torch.nn.Linear(actor_in, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)
                ),
                critic=torch.nn.Sequential(
                    torch.nn.Linear(critic_in, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
                ),
            )
            algorithm.optimizer = torch.optim.Adam(
                [*algorithm.actor.parameters(), *algorithm.critic.parameters()], lr=1e-5
            )

            def load(loaded, cfg, strict):
                algorithm.actor.load_state_dict(loaded["actor_state_dict"], strict=strict)
                algorithm.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
                algorithm.optimizer.load_state_dict(loaded["optimizer_state_dict"])

            algorithm.load = load
            return algorithm

        narrow = learner(4, 5)
        loss = (
            narrow.actor(torch.randn(8, 4)).square().sum()
            + narrow.critic(torch.randn(8, 5)).square().sum()
        )
        loss.backward()
        narrow.optimizer.step()
        saved = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        wide = learner(6, 7)
        widened, rate = initialize_learner(wide, saved, 3e-4, "configured")
        self.assertEqual(
            (widened, rate), ([["actor.0.weight", 4, 6], ["critic.0.weight", 5, 7]], 3e-4)
        )
        x = torch.randn(3, 4)
        torch.testing.assert_close(
            wide.actor(torch.cat((x, torch.randn(3, 2)), -1)), narrow.actor(x)
        )
        moment = wide.optimizer.state_dict()["state"][0]["exp_avg"]
        self.assertEqual(tuple(moment.shape), (3, 6))
        torch.testing.assert_close(moment[:, 4:], torch.zeros(3, 2))
        self.assertEqual(wide.optimizer.param_groups[0]["lr"], 3e-4)
        bad = {
            **saved,
            "actor_state_dict": {**saved["actor_state_dict"], "2.weight": torch.zeros(5, 3)},
        }
        with self.assertRaises(ValueError):
            widen_inputs(bad, learner(6, 7))

    def test_a_widened_normalizer_passes_the_added_inputs_unscaled_and_a_missing_one_is_named(self):
        from rsl_rl.modules.normalization import EmpiricalNormalization

        class Actor(torch.nn.Module):
            def __init__(self, inputs, normalize=True):
                super().__init__()
                if normalize:
                    self.obs_normalizer = EmpiricalNormalization(inputs)
                self.mlp = torch.nn.Linear(inputs, 2)

        def learner(inputs, normalize=True):
            algorithm = SimpleNamespace(
                actor=Actor(inputs, normalize),
                critic=torch.nn.Sequential(torch.nn.Linear(inputs, 1)),
            )
            algorithm.optimizer = torch.optim.Adam(
                [*algorithm.actor.parameters(), *algorithm.critic.parameters()]
            )
            return algorithm

        narrow = learner(4)
        narrow.actor.obs_normalizer.update(torch.randn(64, 4) * 3 + 1)
        saved = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        widened = widen_inputs(saved, learner(6))
        self.assertEqual(widened, [["actor.mlp.weight", 4, 6], ["critic.0.weight", 4, 6]])
        state = saved["actor_state_dict"]
        for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            column = state[f"obs_normalizer.{name}"]
            self.assertEqual(tuple(column.shape), (1, 6))
            torch.testing.assert_close(column[:, 4:], torch.full((1, 2), fill))
            torch.testing.assert_close(column[:, :4], getattr(narrow.actor.obs_normalizer, name))
        empirical = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "obs_normalizer._mean"):
            widen_inputs(empirical, learner(6, normalize=False))


class NativeStartTests(unittest.TestCase):
    def test_native_training_starts_only_from_a_declared_checkpoint_of_the_same_robot(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import locomotion.rl.ppo as ppo
        from locomotion.evaluation.checkpoint import STARTING_KEYS
        from locomotion.scripts.train import sha, start_from_checkpoint

        robot = {key: key[0] * 64 for key in STARTING_KEYS}
        identity = {**robot, "physics_config": {"floor": "isaaclab_mesh"}}
        parent = {
            **robot,
            "physics_config": {"floor": "earlier"},
            "experiment": "destination_v8",
            "reward_version": "4",
            "reward_task": "v4_yaw_gate",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            def write(record_identity, infos_identity=None):
                torch.save(
                    {"infos": {"identity": infos_identity or record_identity, "updates": 3500}},
                    path,
                )
                record = {
                    "identity": record_identity,
                    "updates": 3500,
                    "checkpoint_sha256": sha(path),
                }
                path.with_suffix(".json").write_text(json.dumps(record))
                return SimpleNamespace(
                    initialize_from=path,
                    initialize_from_sha256=sha(path),
                    initialize_from_declaration_sha256=sha(path.with_suffix(".json")),
                    initialize_rate="configured",
                    device="cpu",
                )

            config = {"algorithm": {"learning_rate": 3e-4}}
            with patch.object(
                ppo, "initialize_learner", return_value=([["actor.0.weight", 4, 6]], 3e-4)
            ):
                args = write(parent)
                started = dict(identity)
                start_from_checkpoint(SimpleNamespace(alg=None), args, config, started)
                start = started["initialized_from"]
                self.assertEqual(
                    (start["declaration_sha256"], start["experiment"], start["updates"]),
                    (args.initialize_from_declaration_sha256, "destination_v8", 3500),
                )
                self.assertFalse(start["physics_config_equal"])
                cases = {
                    "declaration bytes": (parent, None, "initialize_from_declaration_sha256"),
                    "robot differs: usd_sha256": (dict(parent, usd_sha256="0" * 64), None, None),
                    "Embedded checkpoint declaration": (parent, dict(parent, seed=1), None),
                }
                for message, (record, infos, broken) in cases.items():
                    with self.subTest(message=message):
                        args = write(record, infos)
                        if broken:
                            setattr(args, broken, "0" * 64)
                        with self.assertRaisesRegex(ValueError, message):
                            start_from_checkpoint(
                                SimpleNamespace(alg=None), args, config, dict(identity)
                            )


class PackTests(unittest.TestCase):
    def test_a_training_pack_names_the_experiment_and_mounts_its_starting_checkpoint(self):
        import json
        import tempfile
        from pathlib import Path

        from locomotion.spark.prepare import prepare

        remote = "/srv/cupi/hexapod/runs/julian/destination_fixture"
        start = "/srv/cupi/hexapod/runs/julian/c48/run/standing/checkpoint_update002500.pt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(
                root / "train",
                remote,
                mode="train",
                updates=3500,
                extended_updates=True,
                experiment="destination_v1",
                initialize_from=start,
                initialize_from_sha="c" * 64,
                initialize_from_declaration_sha="d" * 64,
                steps_per_env=36,
                allocation_profile="flat_pilot_v1",
                max_wall_seconds=21600,
            )
            args = binding["command_args"]
            self.assertEqual(args[args.index("--experiment") + 1], "destination_v1")
            self.assertEqual(
                args[args.index("--initialize-from") + 1], "/checkpoint/checkpoint_update002500.pt"
            )
            self.assertEqual(args[args.index("--initialize-from-sha256") + 1], "c" * 64)
            self.assertEqual(args[args.index("--initialize-from-declaration-sha256") + 1], "d" * 64)
            self.assertNotIn("--initialize-rate", args)  # the configured rate is the default
            self.assertIn([str(Path(start).parent), "/checkpoint"], binding["extra_mounts"])
            self.assertEqual(binding["input_files"][start], "c" * 64)
            self.assertEqual(binding["input_files"][start[: -len(".pt")] + ".json"], "d" * 64)
            pack = json.loads((root / "train/PACK.json").read_text())
            self.assertEqual(
                (pack["experiment"], pack["initialize_from_sha256"]), ("destination_v1", "c" * 64)
            )
            evaluation = prepare(
                root / "evaluate",
                remote,
                mode="evaluate",
                experiment="destination_v1",
                checkpoint=start,
                checkpoint_sha="a" * 64,
                checkpoint_declaration_sha="b" * 64,
                steps_per_env=36,
            )
            self.assertIn("--experiment", evaluation["command_args"])
            for index, bad in enumerate(
                (
                    dict(mode="train", experiment="unknown"),
                    dict(mode="diagnostic", experiment="destination_v1"),
                    dict(mode="train", initialize_from=start),
                    dict(mode="train", initialize_from=start, initialize_from_sha="c" * 64),
                    dict(
                        mode="train",
                        initialize_from="/tmp/x.pt",
                        initialize_from_sha="c" * 64,
                        initialize_from_declaration_sha="d" * 64,
                    ),
                    dict(
                        mode="train",
                        initialize_from=start,
                        initialize_from_sha="c" * 64,
                        initialize_from_declaration_sha="not a hash",
                    ),
                    dict(
                        mode="evaluate",
                        checkpoint="/tmp/x.pt",
                        checkpoint_sha="a" * 64,
                        checkpoint_declaration_sha="b" * 64,
                    ),
                )
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root / f"bad{index}", remote, **bad)
                # A refusal writes nothing, so the same output can be prepared again.
                self.assertFalse((root / f"bad{index}").exists())


class LaunchMountTests(unittest.TestCase):
    def test_every_mount_a_training_pack_declares_is_one_the_launcher_accepts(self):
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "spark/launch.py"
        allowed = {
            node.value
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("/")
        }
        self.assertIn("/checkpoint", allowed)


if __name__ == "__main__":
    unittest.main()
