"""Compare actor-only loading with a synthetic stock RSL-RL actor."""

import copy
import unittest
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import locomotion.rl.ppo as ppo
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import ppo_config


class ActorCheckpoint(dict):
    def __getitem__(self, key):
        if key not in ("actor_state_dict", "infos"):
            raise AssertionError("Inference read training state: " + key)
        return super().__getitem__(key)


class PolicyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = ppo_config(
            20260917,
            action_mean="tanh",
            observation_normalization="none",
            observation_scaling="fixed",
            gait_clock=60,
            action_smoothing="mean2",
            command_segments="bootstrap",
            learning_rate_max=0.0003,
            action_std_final=0.05,
            velocity_noise=0.5,
            steps_per_env=48,
        )
        generator = torch.Generator().manual_seed(71)
        self.observations = TensorDict({"policy": torch.randn(4, 233, generator=generator)}, [4])
        cfg = copy.deepcopy(self.config["actor"])
        cfg.pop("class_name")
        with torch.random.fork_rng():
            torch.manual_seed(43)
            self.baseline = MLPModel(
                self.observations, self.config["obs_groups"], "actor", 18, **cfg
            )
        self.baseline.eval()
        self.infos = {"identity": {"synthetic": True}, "updates": 2500}
        self.checkpoint = ActorCheckpoint(
            actor_state_dict=self.baseline.state_dict(), infos=self.infos
        )

    def test_load_preserves_stock_actor_outputs_and_configuration(self):
        original = copy.deepcopy(self.config)
        rng = torch.random.get_rng_state().clone()
        with patch(
            "torch.optim.Adam", side_effect=AssertionError("Inference constructed an optimizer")
        ):
            actor, infos = ppo.load_actor(self.checkpoint, self.config, self.observations, "cpu")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(self.config, original)
        self.assertIs(infos, self.infos)
        self.assertFalse(actor.training)
        self.assertEqual(tuple(actor.state_dict()), tuple(self.baseline.state_dict()))
        with torch.inference_mode():
            self.assertTrue(torch.equal(actor(self.observations), self.baseline(self.observations)))
        self.assertTrue(bool((actor(self.observations).abs() <= 1).all()))

    def test_evaluation_smooths_either_actor_branch_like_the_training_wrapper(self):
        import ast
        import inspect
        from types import SimpleNamespace

        from locomotion.rl.wrapper import smoothed_action
        from locomotion.scripts import play

        env = SimpleNamespace(episode_steps=torch.zeros(1, dtype=torch.long))
        output = torch.full((1, 18), 0.1)
        policy = play.smoothed_policy(lambda observation: output.clone(), env, "mean2")
        self.assertTrue(
            torch.equal(
                policy(None), smoothed_action(output, torch.zeros_like(output), env.episode_steps)
            )
        )
        env.episode_steps += 1
        self.assertTrue(torch.equal(policy(None), output))
        self.assertTrue(
            torch.equal(
                play.smoothed_policy(lambda observation: output.clone(), env, "none")(None), output
            )
        )
        # Both actor branches flow through the one smoothing wrapper; neither returns an unsmoothed action.
        tree = next(
            node
            for node in ast.parse(inspect.getsource(play)).body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluation_policy"
        )
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        self.assertEqual(
            [
                ast.unparse(node.value.func)
                for node in returns
                if isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "smoothed_policy"
            ],
            ["smoothed_policy"],
        )
        self.assertTrue(
            all(
                ast.unparse(node.value).startswith(("smoothed_policy(", "actor("))
                for node in returns
            )
        )

    def test_load_rejects_missing_and_extra_actor_parameters(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                state = dict(self.checkpoint["actor_state_dict"])
                if mutation == "missing":
                    state.pop(next(iter(state)))
                else:
                    state["unexpected_parameter"] = torch.zeros(1)
                checkpoint = ActorCheckpoint(actor_state_dict=state, infos=self.infos)
                with self.assertRaises(RuntimeError):
                    ppo.load_actor(checkpoint, self.config, self.observations, "cpu")
