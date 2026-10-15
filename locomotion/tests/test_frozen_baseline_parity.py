"""Compare task, wrapper and optimizer arithmetic with frozen source archives."""
import hashlib
import importlib
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest

import torch
from tensordict import TensorDict

from locomotion.tests.test_learner_options import CommandTask, construct
from locomotion.tests.task_double import GaitDouble
from locomotion.tests.test_task import SPEC

FIXTURE = Path(__file__).parent / 'fixtures/refactor_b2a6168c.tar.gz'
FIXTURE_SHA = 'f0afe51e180bc1d5c5f6099b8225e77cd63afdbed78e4830dd837767ae43b6fc'
BASELINE = '_frozen_baseline'


class FrozenBaselineParityTests(unittest.TestCase):
    fixture = FIXTURE
    fixture_sha = FIXTURE_SHA
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(cls.fixture.read_bytes()).hexdigest() != cls.fixture_sha:
            raise ValueError('Frozen baseline source changed')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with tarfile.open(cls.fixture) as archive:
            archive.extractall(cls.directory.name, filter='data')
        def clear_modules():
            for name in list(sys.modules):
                if name == BASELINE or name.startswith(BASELINE + '.'):
                    del sys.modules[name]
        clear_modules()
        cls.addClassCleanup(clear_modules)
        package = types.ModuleType(BASELINE)
        package.__path__ = [cls.directory.name]
        sys.modules[package.__name__] = package

    def module(self, name, baseline=False):
        if baseline:
            return importlib.import_module(BASELINE + '.' + name)
        path = 'omni.ppo' if name == 'ppo' else 'omni.' + name
        return importlib.import_module('locomotion.' + path)

    def exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_active_reward_command_rng_and_selected_resets_match_frozen_baseline(self):
        traces = []
        for baseline in (True, False):
            module = self.module('task_v4_yaw_gate', True) if baseline else self.module('task')
            task = (module.variant(SPEC)(GaitDouble(3), module.task_v4.TaskConfig(seed=20260917)) if baseline else
                    module.TrainingTask(GaitDouble(3), self.module('commands').TaskConfig(seed=20260917)))
            task.reset()
            trace = []
            for step in range(12):
                task.remaining_controls[step % 3] = 1
                if step == 6:
                    task.reset(torch.tensor([1]))
                output = task.step(torch.full((3, 18), step / 20.))
                trace.append(({key: value.clone() for key, value in output.items()},
                              task.rng.get_state().clone(), task.remaining_controls.clone(),
                              {key: value.clone() for key, value in task.last_components.items()}))
            traces.append(trace)
        self.exact(traces[1], traces[0])

    def test_adapter_noise_bootstrap_and_reset_rng_match(self):
        traces = []
        for baseline in (True, False):
            torch.manual_seed(239)
            vec_env = self.module('ppo', True).VanillaVecEnv if baseline else self.module('wrapper').OmniVecEnv
            wrapper = vec_env(
                CommandTask(), tensor_dict=lambda data, **kw: data, observation_scaling='fixed',
                command_segments='bootstrap', gait_clock=60, action_smoothing='mean2', velocity_noise=.5)
            traces.append(([wrapper.step(torch.full((3, 18), .2)) for _ in range(4)], torch.get_rng_state()))
        self.exact(traces[1], traces[0])

    def test_optimizer_parameters_state_losses_and_rng_match(self):
        results = []
        for baseline in (True, False):
            torch.manual_seed(20260917)
            config = (self.module('ppo', True) if baseline else self.module('ppo_cfg')).ppo_config(20260917, action_mean='tanh',
                observation_normalization='none', learning_rate_max=.0003, action_noise_correlation=.5)
            if baseline:
                for node in (config['algorithm'], config['actor']['distribution_cfg']):
                    node['class_name'] = node['class_name'].replace('locomotion.', BASELINE + '.')
            algorithm = construct(config)
            algorithm.num_learning_epochs, algorithm.num_mini_batches = 1, 2
            obs = TensorDict({'policy': torch.randn(8, 231), 'critic': torch.randn(8, 234)}, batch_size=[8])
            with torch.inference_mode():
                for _ in range(4):
                    algorithm.act(obs)
                    algorithm.process_env_step(obs, torch.randn(8), torch.zeros(8, dtype=torch.long), {})
                algorithm.compute_returns(obs)
            loss = algorithm.update()
            results.append((algorithm.actor.state_dict(), algorithm.critic.state_dict(),
                            algorithm.optimizer.state_dict(), loss, torch.get_rng_state()))
        self.exact(results[1], results[0])

    def test_reward_archive_and_numerical_gate_bytes_remain_frozen(self):
        root = Path(__file__).resolve().parents[1]
        with tarfile.open(FIXTURE) as archive:
            # The immutable archive holds retired reward source bytes.
            self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA)
            self.assertEqual((root/'evaluation/scoring.py').read_bytes(), archive.extractfile('evaluation.py').read())


class PackageSnapshotParityTests(FrozenBaselineParityTests):
    fixture = Path(__file__).parent / 'fixtures/packages_02f551fd.tar.gz'
    fixture_sha = '5857bb0e60f1785e770b6c35dc01cdb1737fe24f70d029b6da61331f74bb16d2'


if __name__ == '__main__':
    unittest.main()
