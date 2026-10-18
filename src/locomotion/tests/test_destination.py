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
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(third["reward"], **bad).validate()

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


if __name__ == "__main__":
    unittest.main()


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
            self.assertNotIn("--initialize-rate", args)  # the configured rate is the default
            self.assertIn([str(Path(start).parent), "/checkpoint"], binding["extra_mounts"])
            self.assertEqual(binding["input_files"][start], "c" * 64)
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
                    dict(mode="train", initialize_from="/tmp/x.pt", initialize_from_sha="c" * 64),
                )
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root / f"bad{index}", remote, **bad)


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
