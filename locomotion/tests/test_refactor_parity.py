"""Compare candidate arithmetic with source frozen from main b2a6168c."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest

import torch
from tensordict import TensorDict

from locomotion.tests.test_learner_options import CommandTask, construct
from locomotion.tests.test_task_v4 import GaitDouble

FIXTURE = Path(__file__).parent / 'fixtures/refactor_b2a6168c.tar.gz'
FIXTURE_SHA = 'f0afe51e180bc1d5c5f6099b8225e77cd63afdbed78e4830dd837767ae43b6fc'


class RefactorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA:
            raise ValueError('Frozen b2a6168c source changed')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with tarfile.open(FIXTURE) as archive:
            archive.extractall(cls.directory.name, filter='data')
        package = types.ModuleType('_refactor_baseline')
        package.__path__ = [cls.directory.name]
        sys.modules[package.__name__] = package

    def module(self, name, baseline=False):
        return importlib.import_module(('_refactor_baseline.' if baseline else 'locomotion.') + name)

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

    def test_reward_command_rng_and_selected_resets_match_all_versions(self):
        for name, class_name in (('task', 'TrainingTask'), ('task_v2', 'TrainingTaskV2'),
                                 ('task_v3', 'TrainingTaskV3'), ('task_v4', 'TrainingTaskV4'),
                                 ('task_v4_yaw_gate', 'TrainingTaskV4YawGate')):
            with self.subTest(task=name):
                traces = []
                for baseline in (True, False):
                    task = getattr(self.module(name, baseline), class_name)(GaitDouble(3))
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
            wrapper = self.module('ppo', baseline).VanillaVecEnv(
                CommandTask(), tensor_dict=lambda data, **kw: data, observation_scaling='fixed',
                command_segments='bootstrap', gait_clock=60, action_smoothing='mean2', velocity_noise=.5)
            traces.append(([wrapper.step(torch.full((3, 18), .2)) for _ in range(4)], torch.get_rng_state()))
        self.exact(traces[1], traces[0])

    def test_optimizer_parameters_state_losses_and_rng_match(self):
        results = []
        for baseline in (True, False):
            torch.manual_seed(20260917)
            config = self.module('ppo', baseline).ppo_config(20260917, action_mean='tanh',
                observation_normalization='none', learning_rate_max=.0003, action_noise_correlation=.5)
            if baseline:
                for node in (config['algorithm'], config['actor']['distribution_cfg']):
                    node['class_name'] = node['class_name'].replace('locomotion.', '_refactor_baseline.')
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


class RefactorBoundaryTests(unittest.TestCase):
    def test_compatibility_imports_keep_class_identity(self):
        for old, new, name in (
            ('env', 'backends.isaac.env', 'LocomotionEnv'),
            ('task', 'tasks.task', 'TrainingTask'),
            ('task_v4_yaw_gate', 'tasks.task_v4_yaw_gate', 'TrainingTaskV4YawGate'),
            ('ppo', 'learning.ppo', 'VanillaVecEnv'),
            ('evaluate', 'evaluation.runner', 'ExactEvaluationCapture'),
            ('prepare', 'runs.prepare', 'prepare')):
            with self.subTest(module=old):
                self.assertIs(getattr(importlib.import_module('locomotion.' + old), name),
                              getattr(importlib.import_module('locomotion.' + new), name))

    def test_nested_pack_is_complete_and_detects_dependency_tampering(self):
        from locomotion.prepare import prepare
        from locomotion.runs.sources import verify_runtime
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            prepare(output, '/srv/cupi/hexapod/runs/james/refactor_fixture', updates=2)
            source = output/'source'
            manifest = verify_runtime(source)
            self.assertEqual(manifest['schema'], 'hexapod_runtime_sources_v2')
            for path in ('locomotion/core/motor.py', 'locomotion/tasks/task_v4_yaw_gate.py',
                         'locomotion/backends/isaac/env.py', 'locomotion/learning/ppo.py'):
                self.assertIn(path, manifest['files'])
            self.assertFalse(any('/tests/' in name for name in manifest['files']))
            path = source/'locomotion/core/motor.py'
            path.write_text(path.read_text() + '\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'Runtime dependency'):
                verify_runtime(source)

    def test_prepare_and_training_use_shared_validation(self):
        from locomotion import prepare, train
        from locomotion.runs import validation
        for name in ('update_limit', 'validate_decay', 'validate_deadline'):
            self.assertIs(getattr(prepare, name), getattr(validation, name))
            self.assertIs(getattr(train, name), getattr(validation, name))


if __name__ == '__main__':
    unittest.main()
