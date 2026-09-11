"""Execute only the actual pinned RSL method on CPU with transparent doubles.

This is an interface counterexample, not a PPO training or reset regression.
The original method body is compiled without edits or importing its package.
"""
import ast
import __future__
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

SOURCE_SHA = 'a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c'


def actual_method():
    path = Path(__file__).parent / 'inputs/RSL_ppo.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('Actual source binding changed')
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPO')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_env_step')
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), namespace)
    return namespace[fn.name]


class Model:
    def __init__(self):
        self.normalized = []
        self.resets = []

    def update_normalization(self, obs):
        self.normalized.append(obs.clone())

    def reset(self, dones):
        self.resets.append(dones.clone())


class Storage:
    def add_transition(self, t):
        self.rewards = t.rewards.clone()
        self.dones = t.dones.clone()


def invoke_actual():
    actor, critic, storage = Model(), Model(), Storage()
    transition = SimpleNamespace(values=torch.tensor([[2.], [2.]]), clear=lambda: None)
    alg = SimpleNamespace(actor=actor, critic=critic, storage=storage,
                          transition=transition, rnd=None, gamma=.99, device='cpu')
    # Row0 is a true terminal; row1 is an artificial time limit.
    # Both next observations here are reset/recovery placeholders, not s_final.
    obs = torch.tensor([[100.], [999.]])
    actual_method()(alg, obs, torch.ones(2), torch.tensor([True, True]),
                    {'time_outs': torch.tensor([False, True])})
    return alg, obs


class ActualRSLBoundary(unittest.TestCase):
    def test_actual_timeout_uses_stored_pre_action_value_not_final_packet(self):
        alg, _ = invoke_actual()
        actual = alg.storage.rewards
        torch.testing.assert_close(actual, torch.tensor([1., 2.98]))
        # Required project convention for V(s_final)=7: terminal0, timeout7.
        desired = torch.tensor([1., 1. + .99 * 7.])
        self.assertAlmostEqual(float(desired[1] - actual[1]), 4.95, places=5)
        self.assertEqual(float(actual[0]), 1.)

    def test_actual_next_packet_normalization_has_no_recovery_mask(self):
        alg, obs = invoke_actual()
        for model in (alg.actor, alg.critic):
            self.assertEqual(len(model.normalized), 1)
            torch.testing.assert_close(model.normalized[0], obs)
            self.assertEqual(model.normalized[0].shape[0], 2)
            self.assertTrue(model.resets[0].all())


if __name__ == '__main__':
    unittest.main()
