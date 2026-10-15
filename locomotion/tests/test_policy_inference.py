"""Compare actor-only loading with a synthetic stock RSL-RL actor."""
import copy
import unittest
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from locomotion.gait_clock.config import ppo_config
import locomotion.gait_clock.ppo as ppo


class ActorCheckpoint(dict):
    def __getitem__(self, key):
        if key not in ('actor_state_dict', 'infos'):
            raise AssertionError('Inference read training state: ' + key)
        return super().__getitem__(key)


class PolicyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = ppo_config(20260917, action_mean='tanh',
            observation_normalization='none', observation_scaling='fixed', gait_clock=60,
            action_smoothing='mean2', command_segments='bootstrap', learning_rate_max=.0003,
            action_std_final=.05, velocity_noise=.5, steps_per_env=48)
        generator = torch.Generator().manual_seed(71)
        self.observations = TensorDict({'policy': torch.randn(4, 233, generator=generator)}, [4])
        cfg = copy.deepcopy(self.config['actor'])
        cfg.pop('class_name')
        with torch.random.fork_rng():
            torch.manual_seed(43)
            self.baseline = MLPModel(self.observations, self.config['obs_groups'], 'actor', 18, **cfg)
        self.baseline.eval()
        self.infos = {'identity': {'synthetic': True}, 'updates': 2500}
        self.checkpoint = ActorCheckpoint(actor_state_dict=self.baseline.state_dict(), infos=self.infos)

    def test_load_preserves_stock_actor_outputs_and_configuration(self):
        original = copy.deepcopy(self.config)
        rng = torch.random.get_rng_state().clone()
        with patch('torch.optim.Adam', side_effect=AssertionError('Inference constructed an optimizer')):
            actor, infos = ppo.load_actor(self.checkpoint, self.config, self.observations, 'cpu')
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(self.config, original)
        self.assertIs(infos, self.infos)
        self.assertFalse(actor.training)
        self.assertEqual(tuple(actor.state_dict()), tuple(self.baseline.state_dict()))
        with torch.inference_mode():
            self.assertTrue(torch.equal(actor(self.observations), self.baseline(self.observations)))
        self.assertTrue(bool((actor(self.observations).abs() <= 1).all()))

    def test_load_rejects_missing_and_extra_actor_parameters(self):
        for mutation in ('missing', 'extra'):
            with self.subTest(mutation=mutation):
                state = dict(self.checkpoint['actor_state_dict'])
                if mutation == 'missing':
                    state.pop(next(iter(state)))
                else:
                    state['unexpected_parameter'] = torch.zeros(1)
                checkpoint = ActorCheckpoint(actor_state_dict=state, infos=self.infos)
                with self.assertRaises(RuntimeError):
                    ppo.load_actor(checkpoint, self.config, self.observations, 'cpu')
