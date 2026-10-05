"""Check the surrogate trainer entry and the evaluator's no-noise policy path."""
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch

from locomotion import ppo
from locomotion.surrogate import calibrate, evaluate, train
from locomotion.task_v2 import REWARD_V2_CONFIG, TrainingTaskV2
from locomotion.task_v4 import REWARD_V4_CONFIG

NATIVE_ROW_KEYS = {"update", "transitions", "collection_seconds", "learning_seconds", "loss", "learning_rate",
                   "mean_action_std", "actions", "task", "policy_update"}


class HalfActionRateTask(TrainingTaskV2):
    """Reward version 2 with half the action-rate weight: an example ``--task`` class."""

    def __init__(self, env, config=None, output_dir=None):
        super().__init__(env, config, output_dir,
                         reward_config=replace(REWARD_V2_CONFIG, action_rate_weight=REWARD_V2_CONFIG.action_rate_weight / 2))


def task_factory(text):
    return {"half": HalfActionRateTask}[text]


class PlainWrapper:
    def __init__(self, task):
        self.task, self.options = task, {}


class OpenWrapper:
    def __init__(self, task, **options):
        self.task, self.options = task, options


class RecordingActor:
    """Return queued actions and keep each policy input."""

    def __init__(self, actions):
        self.actions, self.inputs = list(actions), []

    def __call__(self, observations):
        self.inputs.append(observations["policy"].clone())
        return self.actions.pop(0)


def run(directory, name, *arguments):
    output = Path(directory) / name
    # The trainer prints its progress and the traceback of a failed run.
    with contextlib.redirect_stdout(io.StringIO()):
        code = train.main(["--output", str(output), "--num-envs", "4", "--threads", "1", "--updates", "1", *arguments])
    return code, output, json.loads((output / "state.json").read_text())


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.code, cls.output, cls.state = run(cls.temporary.name, "default")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_one_update_writes_the_native_record_layout(self):
        self.assertEqual((self.code, self.state["status"]), (0, "completed"))
        rows = [json.loads(line) for line in (self.output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), NATIVE_ROW_KEYS)
        self.assertEqual(rows[0]["transitions"], 24 * 4)
        record = json.loads((self.output / "checkpoint_update000001.json").read_text())
        self.assertEqual(record["checkpoint_sha256"], train_sha(self.output / "checkpoint_update000001.pt"))
        identity = self.state["identity"]
        self.assertTrue(identity["surrogate"] and identity["backend"].startswith("mujoco "))
        package = Path(train.__file__).resolve().parent
        self.assertEqual(sorted(identity["surrogate_files"]), sorted(path.name for path in package.glob("*.py")))
        self.assertNotIn("environment_wrapper", identity["ppo_config"])
        self.assertEqual((identity["action_smoothing"], identity["velocity_noise"]), ("none", 0.))

    def test_newest_learner_options_reach_the_wrapper(self):
        clock = REWARD_V4_CONFIG.schedule_period_controls if REWARD_V4_CONFIG.schedule_weight else 60
        code, output, state = run(
            self.temporary.name, "options", "--task", "locomotion.task_v4:TrainingTaskV4", "--gait-clock", str(clock),
            "--action-mean", "tanh", "--observation-normalization", "none", "--observation-scaling", "fixed",
            "--command-segments", "bootstrap", "--learning-rate-max", "3e-4", "--action-std", "0.1",
            "--action-std-final", "0.05", "--action-smoothing", "mean2", "--velocity-noise", "0.5")
        self.assertEqual(code, 0, state.get("traceback"))
        config = json.loads((output / "ppo_config.json").read_text())
        self.assertEqual(config["environment_wrapper"], {
            "observation_scaling": "fixed", "command_segments": "bootstrap", "gait_clock": clock,
            "action_smoothing": "mean2", "velocity_noise": .5})
        self.assertEqual(config["exploration"], {"action_std_final": .05})
        self.assertEqual(config["algorithm"]["learning_rate_max"], 3e-4)
        identity = state["identity"]
        self.assertEqual((identity["action_smoothing"], identity["velocity_noise"], identity["gait_clock"]), ("mean2", .5, clock))
        row = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
        self.assertTrue(.05 - 1e-6 <= row["mean_action_std"] <= .1 + 1e-6)

    def test_schedule_reward_requires_its_gait_clock(self):
        if not REWARD_V4_CONFIG.schedule_weight:
            self.skipTest("Reward version 4 holds no contact-schedule term at this revision")
        code, _, state = run(self.temporary.name, "no_clock", "--task", "locomotion.task_v4:TrainingTaskV4")
        self.assertEqual((code, state["status"]), (1, "failed"))
        self.assertIn("gait clock", state["errors"][0])

    def test_entry_restores_the_callers_thread_settings(self):
        variable, count = os.environ.pop("OMP_NUM_THREADS", None), torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            code, _, state = run(self.temporary.name, "threads")
            self.assertEqual((code, state["identity"]["threads"]), (0, 1))
            self.assertEqual((os.environ.get("OMP_NUM_THREADS"), torch.get_num_threads()), (None, 2))
            os.environ["OMP_NUM_THREADS"] = "3"
            with train.process_threads(1):
                self.assertEqual((os.environ["OMP_NUM_THREADS"], torch.get_num_threads()), ("3", 1))
            self.assertEqual((os.environ["OMP_NUM_THREADS"], torch.get_num_threads()), ("3", 2))
        finally:
            torch.set_num_threads(count)
            os.environ.pop("OMP_NUM_THREADS", None)
            if variable is not None:
                os.environ["OMP_NUM_THREADS"] = variable

    def test_thread_count_stays_within_one_to_six(self):
        self.assertEqual([train.thread_count(value) for value in (1, 6)], [1, 6])
        for value in (0, 7, 16):
            with self.assertRaises(ValueError):
                train.thread_count(value)
        with self.assertRaises(ValueError):
            train.main(["--output", str(Path(self.temporary.name) / "many"), "--threads", "16"])
        self.assertFalse((Path(self.temporary.name) / "many").exists())

    def test_documented_factory_example_resolves(self):
        example = re.search(r"``(locomotion\.task_v4:variant\([^`]*\))``", train.resolve.__doc__)
        self.assertIsNotNone(example)
        self.assertTrue(issubclass(train.resolve(example.group(1)), train.resolve("locomotion.task_v4:TrainingTaskV4")))

    def test_example_task_and_factory_resolve(self):
        self.assertIs(train.resolve("locomotion.task_v2:TrainingTaskV2"), TrainingTaskV2)
        self.assertIs(train.resolve(f"{__name__}:task_factory(half)"), HalfActionRateTask)
        for bad in ("locomotion.task_v2", f"{__name__}:task_factory(half", f"{__name__}:run"):
            with self.assertRaises(ValueError):
                train.resolve(bad)
        code, _, state = run(self.temporary.name, "example", "--task", f"{__name__}:HalfActionRateTask")
        self.assertEqual(code, 0, state.get("traceback"))

    def test_learner_configuration_rejects_an_unknown_selected_option(self):
        old = SimpleNamespace(ppo_config=lambda seed, *, action_std=.15: {"seed": seed, "action_std": action_std})
        self.assertEqual(train.learner_configuration(old, 3, {"action_std": .2, "velocity_noise": 0., "gait_clock": 0}),
                         {"seed": 3, "action_std": .2})
        with self.assertRaisesRegex(ValueError, "velocity_noise"):
            train.learner_configuration(old, 3, {"velocity_noise": .5})
        config = train.learner_configuration(ppo, 3, {"action_smoothing": "mean2", "velocity_noise": .5})
        self.assertEqual(config["environment_wrapper"], {"action_smoothing": "mean2", "velocity_noise": .5})

    def test_wrapper_receives_recorded_options(self):
        self.assertEqual(train.wrapper_options({}), {"observation_scaling": "none", "command_segments": "continuous"})
        options = train.wrapper_options({"environment_wrapper": {"gait_clock": 60, "velocity_noise": .5}})
        self.assertEqual(options, {"observation_scaling": "none", "command_segments": "continuous",
                                   "gait_clock": 60, "velocity_noise": .5})
        self.assertEqual(train.build_wrapper(OpenWrapper, "task", options).options, options)
        self.assertEqual(train.build_wrapper(PlainWrapper, "task", train.wrapper_options({})).options, {})
        with self.assertRaisesRegex(ValueError, "gait_clock"):
            train.build_wrapper(PlainWrapper, "task", options)

    def test_training_start_compares_compact_rows(self):
        rows = calibrate.surrogate_rows(self.output)
        self.assertEqual([row["u"] for row in rows], [1])
        extract = Path(self.temporary.name) / "extract.jsonl"
        extract.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result = calibrate.training_start(self.output, extract, first=1, last=1)
        self.assertEqual(result["update 1"]["surrogate"], result["update 1"]["native"])
        self.assertIn("ratio 1.000", "\n".join(calibrate.start_lines(result)))

    def test_extract_writes_a_reference_directory_once(self):
        reference = Path(self.temporary.name) / "reference"
        metrics = self.output / "metrics.jsonl"
        written = calibrate.extract({"A": metrics, "D": metrics}, reference, first=1, last=1)
        self.assertEqual([path.name for path in written], ["metrics_A.jsonl", "metrics_D.jsonl", "native_noise_updates_6_20.json"])
        result = calibrate.training_start(self.output, reference / "metrics_A.jsonl", first=1, last=1)
        self.assertEqual(result["update 1"]["surrogate"], result["update 1"]["native"])
        ranges = calibrate.native_noise_ranges(reference)
        self.assertEqual((ranges["runs"], ranges["tracking_runs"]), (["A", "D"], {"v2": ["A"], "v3": ["D"]}))
        speed = calibrate.surrogate_rows(self.output)[0]["aspd"]
        self.assertEqual(ranges["updates_6_20"]["achieved_planar_speed_mps"], [speed, speed])
        with self.assertRaises(FileExistsError):
            calibrate.extract({"A": metrics}, reference, first=1, last=1)


def train_sha(path):
    from locomotion.surrogate.env import sha
    return sha(path)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(num_envs=2, episode_steps=torch.tensor([7, 7]))
        generator = torch.Generator().manual_seed(1)
        self.observation = torch.randn(2, 231, generator=generator)
        self.observation[:, 210:213] = torch.tensor([[.05, 0., 0.], [0., 0., 0.]])
        self.velocity = torch.zeros(231, dtype=torch.bool)
        for frame in range(5):
            self.velocity[42 * frame + 24:42 * frame + 42] = True

    def policy(self, options, actions=(), **keywords):
        return evaluate.DeterministicPolicy(RecordingActor(actions), self.env, options, **keywords)

    def test_velocity_gain_scales_the_ninety_velocity_inputs(self):
        self.assertEqual(int(self.velocity.sum()), 90)
        inputs = self.policy({}, velocity_input_gain=2.).inputs(self.observation)
        self.assertTrue(torch.equal(inputs[:, self.velocity], 2 * self.observation[:, self.velocity]))
        self.assertTrue(torch.equal(inputs[:, ~self.velocity], self.observation[:, ~self.velocity]))
        self.assertTrue(torch.equal(self.policy({}).inputs(self.observation), self.observation))

    def test_inputs_follow_the_recorded_scales_and_clock(self):
        options = {"observation_scaling": "fixed", "gait_clock": 60}
        inputs = self.policy(options).inputs(self.observation)
        self.assertEqual(tuple(inputs.shape), (2, 233))
        self.assertTrue(torch.equal(inputs[:, :231], ppo.scale_observation(self.observation, "fixed")))
        clock = ppo.clock_features(self.env.episode_steps, 60, self.observation[:, 210:213])
        self.assertTrue(torch.equal(inputs[:, 231:], clock))
        self.assertEqual(clock[1].tolist(), [0., 0.])

    def test_recorded_velocity_noise_adds_no_evaluation_noise(self):
        policy = self.policy({"velocity_noise": .5, "observation_scaling": "fixed"})
        first, second = policy.inputs(self.observation), policy.inputs(self.observation)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, ppo.scale_observation(self.observation, "fixed")))

    def test_two_control_mean_cancels_an_alternating_action(self):
        a = torch.full((2, 18), .4)
        for options, forced in (({}, True), ({"action_smoothing": "mean2"}, False)):
            policy = self.policy(options, [a, -a, a, -a], action_mean2=forced)
            self.assertTrue(policy.mean2)
            self.env.episode_steps = torch.tensor([0, 0])
            self.assertTrue(torch.equal(policy(self.observation), .5 * a))
            self.env.episode_steps = torch.tensor([1, 1])
            self.assertEqual(float(policy(self.observation).abs().max()), 0.)
            self.env.episode_steps = torch.tensor([2, 2])
            self.assertEqual(float(policy(self.observation).abs().max()), 0.)
            # A new episode pairs its first action with the neutral action.
            self.env.episode_steps = torch.tensor([0, 0])
            self.assertTrue(torch.equal(policy(self.observation), -.5 * a))
        plain = self.policy({}, [a, -a])
        self.assertFalse(plain.mean2)
        self.assertTrue(torch.equal(plain(self.observation), a))
        self.assertTrue(torch.equal(plain(self.observation), -a))

    def test_wrapper_adapter_replaces_the_path(self):
        adapter = lambda actor, observation: observation[:, :18] * 2
        policy = self.policy({"gait_clock": 60, "action_smoothing": "mean2", "velocity_noise": .5}, adapter=adapter)
        self.assertTrue(torch.equal(policy(self.observation), self.observation[:, :18] * 2))
        self.assertEqual(policy.actor.inputs, [])
        # The record claims no option that the adapter path skips.
        record = policy.record({"action_smoothing": "mean2", "velocity_noise": .5})
        self.assertEqual((record["source"], record["action_smoothing_applied"], record["velocity_noise_applied"]),
                         ("wrapper policy_for_evaluation", None, None))
        for keywords in ({"velocity_input_gain": 1.25}, {"action_mean2": True}):
            with self.assertRaisesRegex(ValueError, "policy_for_evaluation"):
                self.policy({}, adapter=adapter, **keywords)

    def test_policy_record_names_the_applied_options(self):
        record = self.policy({"velocity_noise": .5}, velocity_input_gain=1.25, action_mean2=True).record({"velocity_noise": .5})
        self.assertEqual({key: record[key] for key in ("source", "velocity_input_gain", "action_mean2_forced",
                                                       "action_smoothing_applied", "velocity_noise_recorded", "velocity_noise_applied")},
                         {"source": "evaluation branch of locomotion/train.py", "velocity_input_gain": 1.25,
                          "action_mean2_forced": True, "action_smoothing_applied": "mean2",
                          "velocity_noise_recorded": .5, "velocity_noise_applied": 0.})
        recorded = self.policy({"action_smoothing": "mean2"}).record({"action_smoothing": "mean2"})
        self.assertEqual((recorded["action_mean2_forced"], recorded["action_smoothing_applied"]), (False, "mean2"))

    def test_foot_load_rows_follow_each_control(self):
        env = SimpleNamespace(total_controls=0, telemetry={})
        loads = evaluate.FootLoad(env)
        loads.record("a")

        def control(value):
            env.total_controls += 1
            env.telemetry = {"tibia_floor_force_world_n": torch.tensor([[[0., 3., 4.]] * 6]) * value}

        control(1.)
        loads.record("a")
        loads.record("a")
        control(2.)
        # The call at the start of probe b reads the last control of probe a.
        loads.record("a")
        loads.record("b")
        control(3.)
        loads.record("b")
        loads.record(None)
        self.assertEqual(loads.rows, {"a": [[5.] * 6, [10.] * 6], "b": [[15.] * 6]})
        self.assertEqual(loads.rest_load(), {"a": [7.5] * 6, "b": [15.] * 6})

    def test_consensus_reports_disagreement_as_unresolved(self):
        check = lambda status: {"pass": status == "pass", "failed_bounds": [] if status == "pass" else ["bound"],
                                "checks": {"bound": {"status": status}}}
        agreed = evaluate.consensus({"default": check("pass"), "pyramidal_cone": check("pass"), "friction_0.9": check("pass")})
        self.assertEqual((agreed["verdict"], agreed["checks"]), ("pass", {"bound": "pass"}))
        split = evaluate.consensus({"default": check("fail"), "pyramidal_cone": check("pass"), "friction_0.9": check("pass")})
        self.assertEqual((split["verdict"], split["checks"], split["failed_bounds_in_all"]), ("unresolved", {"bound": "unresolved"}, []))


if __name__ == "__main__":
    unittest.main()
