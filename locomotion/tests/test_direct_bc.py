"""CPU contracts for the Direct candidate's shared-actor BC experiment."""
import copy
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from locomotion.backends.isaac.config import JOINT_NAMES
from locomotion.learning.bc import ScriptedTripodTeacher, collect, fit, reset_ppo_optimizer, run_bc
from locomotion.learning.config import ppo_config


class DirectBCTests(unittest.TestCase):
    def test_teacher_command_and_reset_rows(self):
        teacher = ScriptedTripodTeacher(JOINT_NAMES, 60)
        commands = torch.tensor([[.05, 0., 0.], [-.05, 0., 0.], [0., 0., 0.]])
        controls = torch.tensor([0, 0, 21])
        action = teacher.actions(commands, controls)
        torch.testing.assert_close(action[0, ::3], -action[1, ::3])
        self.assertEqual(float(action[2].abs().sum()), 0.)
        controls[1] = 17
        controls[1] = 0
        commands[1, 0] = .05
        torch.testing.assert_close(teacher.actions(commands, controls)[1], action[0])
        commands[0, 1] = .01
        with self.assertRaisesRegex(ValueError, 'forward/backward'):
            teacher.actions(commands, controls)

    def test_collection_reads_commands_after_step(self):
        teacher = ScriptedTripodTeacher(JOINT_NAMES, 60)
        task = SimpleNamespace(commands=torch.tensor([[.05, 0., 0.]]), episode_steps=torch.zeros(1, dtype=torch.long))
        task.reset = lambda: None
        obs = TensorDict({'policy': torch.zeros(1, 233)}, [1])
        env = SimpleNamespace(task=task, previous_action=torch.zeros(1, 18), get_observations=lambda: obs)
        def step(action):
            task.commands[0, 0] *= -1
            task.episode_steps.zero_()
            return obs, None, None, None
        env.step = step
        _, labels = collect(env, None, teacher, 2, 0.)
        torch.testing.assert_close(labels[0, ::3], -labels[1, ::3])

    def test_deadline_restores_bank_and_reset(self):
        bank = torch.tensor([[0., .05, 0.]])
        task = SimpleNamespace(bank=bank, joint_names=JOINT_NAMES)
        reset_banks = []
        def reset():
            reset_banks.append(task.bank.clone())
            return None
        task.reset = reset
        env = SimpleNamespace(task=task, gait_clock=60, previous_action=torch.ones(1, 18),
                              get_observations=lambda: None)
        actor = torch.nn.Linear(1, 1)
        runner = SimpleNamespace(alg=SimpleNamespace(actor=actor))
        def expired():
            raise TimeoutError('BC wall limit')
        with self.assertRaisesRegex(TimeoutError, 'wall limit'):
            run_bc(runner, env, check_deadline=expired)
        self.assertIs(task.bank, bank)
        torch.testing.assert_close(reset_banks[0], torch.tensor([[.05, 0., 0.], [-.05, 0., 0.], [0., 0., 0.]]))
        torch.testing.assert_close(reset_banks[-1], bank)
        self.assertEqual(float(env.previous_action.sum()), 0.)
        self.assertTrue(actor.training)

    def test_fit_preserves_shared_actor_keys_shapes_and_ppo_reset(self):
        for normalization in ('empirical', 'none'):
            for width in (231, 233):
                config = ppo_config(1, observation_normalization=normalization, action_mean='tanh')
                obs = TensorDict({'policy': torch.randn(8, width), 'critic': torch.randn(8, width + 3)}, [8])
                actor_config = copy.deepcopy(config['actor'])
                actor_config.pop('class_name')
                actor = MLPModel(obs, config['obs_groups'], 'actor', 18, **actor_config)
                keys = {key: value.shape for key, value in actor.state_dict().items()}
                labels = torch.zeros(8, 18)
                self.assertTrue(torch.isfinite(torch.tensor(fit(actor, obs, labels, 1, .001))))
                self.assertEqual(keys, {key: value.shape for key, value in actor.state_dict().items()})
                clone = MLPModel(obs, config['obs_groups'], 'actor', 18,
                                 **{k: copy.deepcopy(v) for k, v in config['actor'].items() if k != 'class_name'})
                clone.load_state_dict(actor.state_dict(), strict=True)
                torch.testing.assert_close(clone(obs), actor(obs))
                optimizer = torch.optim.Adam(actor.parameters())
                actor(obs).sum().backward()
                optimizer.step()
                self.assertTrue(optimizer.state)
                runner = SimpleNamespace(alg=SimpleNamespace(optimizer=optimizer), current_learning_iteration=7)
                reset_ppo_optimizer(runner)
                self.assertFalse(optimizer.state)
                self.assertEqual(runner.current_learning_iteration, 0)


if __name__ == '__main__':
    unittest.main()
