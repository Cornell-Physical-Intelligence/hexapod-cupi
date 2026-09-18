"""Compare the kernel with recorded native controls and frozen source arithmetic."""
import importlib
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

from locomotion import env, env_config, evaluation, ppo, task

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'locomotion/tests/fixtures/baseline'
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'


class RecordedParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if env_config.sha(TRACE) != '15d89c43a9537a0d45bac58b1f94e99d67804b45d5198e8f08d51441eeadb9ad':
            raise ValueError('Native regression recording changed')
        expected = json.loads((FROZEN/'FREEZE_SHA256.json').read_text())
        package = types.ModuleType('_kernel_recorded_baseline')
        package.__path__ = [str(FROZEN)]
        sys.modules[package.__name__] = package
        cls.old = {}
        for name in ('env_config', 'env', 'task', 'vanilla'):
            if env_config.sha(FROZEN/(name+'.py')) != expected[name+'.py']:
                raise ValueError('Frozen source changed: '+name)
            cls.old[name] = importlib.import_module(package.__name__+'.'+name)
        with np.load(TRACE, allow_pickle=False) as data:
            cls.trace = {key: torch.from_numpy(data[key][:, 0].copy()) for key in data.files if data[key].ndim > 1}

    def test_recorded_commands_produce_the_same_executed_targets(self):
        trace = self.trace
        model = json.loads((ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())
        joints = {joint['name']: joint for joint in model['joints']}
        lower = torch.tensor([joints[name]['lower'] for name in env_config.JOINT_NAMES])
        upper = torch.tensor([joints[name]['upper'] for name in env_config.JOINT_NAMES])
        neutral = torch.tensor([0., -.3, .4]*6)
        held = torch.cat((neutral[None], trace['joint_target_rad'][:-1]))
        args = (trace['policy_action'], held, neutral, lower, upper)
        actual = env.emitted_target(*args)
        torch.testing.assert_close(actual, trace['joint_target_rad'], rtol=0, atol=0)
        torch.testing.assert_close(actual, self.old['env'].emitted_target(*args), rtol=0, atol=0)
        state = trace['amp_state_before']
        args = (state[:, :18], state[:, 18:36], actual, torch.tensor(env_config.KD))
        for current, old in zip(env.motor_force(*args), self.old['env'].motor_force(*args)):
            torch.testing.assert_close(current, old, rtol=0, atol=0)

    def test_recorded_actor_and_critic_inputs_survive_observation_cleanup(self):
        trace = self.trace
        native = env.LocomotionEnv.__new__(env.LocomotionEnv)
        native.history = trace['policy_observation'][:, :210].reshape(-1, 5, 42)
        native.commands = trace['policy_observation'][:, 210:213]
        native.previous_action = trace['policy_observation'][:, 213:]
        native.cfg = env_config.EnvConfig(record_motion_features=True)
        amp = trace['amp_state_before']
        state = {'q': amp[:, :18], 'dq': amp[:, 18:36], 'linear': amp[:, 36:39],
                 'angular': amp[:, 39:42], 'root': torch.cat((torch.zeros(len(amp), 2), amp[:, 42:43]), -1),
                 'toe_body': amp[:, 43:].reshape(-1, 6, 3)}
        actual = native._observations(state)
        baseline = self.old['env'].PaperWalkEnv._observations(native, state)
        for key in ('obs', 'critic', 'amp'):
            torch.testing.assert_close(actual[key], baseline[key], atol=0, rtol=0)
        torch.testing.assert_close(actual['obs'], trace['policy_observation'], atol=0, rtol=0)
        torch.testing.assert_close(actual['critic'], trace['critic_observation'], atol=0, rtol=0)
        native.cfg = env_config.EnvConfig(record_motion_features=False)
        training = native._observations(state)
        self.assertEqual(set(training), {'obs', 'critic'})
        for key in training:
            torch.testing.assert_close(training[key], actual[key], atol=0, rtol=0)

    def test_reward_commands_ppo_settings_and_numerical_gates_keep_their_bytes(self):
        self.assertEqual((ROOT/'locomotion/task.py').read_bytes(), (FROZEN/'task.py').read_bytes())
        self.assertEqual((ROOT/'locomotion/evaluation.py').read_bytes(), (FROZEN/'evaluation.py').read_bytes())
        self.assertEqual(ppo.ppo_config(20260914), self.old['vanilla'].ppo_config(20260914))
        self.assertEqual(task.command_bank(task.TaskConfig()),
                         self.old['task'].command_bank(self.old['task'].TaskConfig()))


if __name__ == '__main__':
    unittest.main()
