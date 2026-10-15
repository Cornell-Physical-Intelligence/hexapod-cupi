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
from locomotion.tests.task_double import GaitDouble
from locomotion.tests.test_task import SPEC

FIXTURE = Path(__file__).parent / 'fixtures/refactor_b2a6168c.tar.gz'
FIXTURE_SHA = 'f0afe51e180bc1d5c5f6099b8225e77cd63afdbed78e4830dd837767ae43b6fc'


class RefactorParityTests(unittest.TestCase):
    fixture = FIXTURE
    fixture_sha = FIXTURE_SHA
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(cls.fixture.read_bytes()).hexdigest() != cls.fixture_sha:
            raise ValueError('Frozen b2a6168c source changed')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with tarfile.open(cls.fixture) as archive:
            archive.extractall(cls.directory.name, filter='data')
        def clear_modules():
            for name in list(sys.modules):
                if name == '_refactor_baseline' or name.startswith('_refactor_baseline.'):
                    del sys.modules[name]
        clear_modules()
        cls.addClassCleanup(clear_modules)
        package = types.ModuleType('_refactor_baseline')
        package.__path__ = [cls.directory.name]
        sys.modules[package.__name__] = package

    def module(self, name, baseline=False):
        if baseline:
            return importlib.import_module('_refactor_baseline.' + name)
        path = 'learning.ppo' if name == 'ppo' else 'tasks.' + name
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

    def test_active_reward_command_rng_and_selected_resets_match_frozen_c48(self):
        traces = []
        for baseline in (True, False):
            module = self.module('task_v4_yaw_gate', True) if baseline else self.module('task')
            task = (module.variant(SPEC)(GaitDouble(3), module.task_v4.TaskConfig(seed=20260917)) if baseline else
                    module.TrainingTask(GaitDouble(3), module.TaskConfig(seed=20260917)))
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

    def test_historical_reward_archive_and_numerical_gate_bytes_remain_frozen(self):
        root = Path(__file__).resolve().parents[1]
        with tarfile.open(FIXTURE) as archive:
            # The immutable archive holds retired reward source bytes.
            self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA)
            self.assertEqual((root/'evaluation/scoring.py').read_bytes(), archive.extractfile('evaluation.py').read())


class RefactorBoundaryTests(unittest.TestCase):
    def test_implementations_replace_retired_import_paths(self):
        for old, new, name in (
            ('env', 'backends.isaac.env', 'LocomotionEnv'),
            ('task', 'tasks.task', 'TrainingTask'),
            ('ppo', 'learning.ppo', 'VanillaVecEnv'),
            ('evaluate', 'evaluation.runner', 'ExactEvaluationCapture'),
            ('prepare', 'runs.prepare', 'prepare')):
            with self.subTest(module=old):
                self.assertIsNone(importlib.util.find_spec('locomotion.' + old))
                self.assertTrue(hasattr(importlib.import_module('locomotion.' + new), name))

    def test_nested_pack_is_complete_and_detects_dependency_tampering(self):
        from locomotion.runs.prepare import prepare
        from locomotion.runs.sources import verify_runtime
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            prepare(output, '/srv/cupi/hexapod/runs/james/refactor_fixture', updates=2)
            source = output/'source'
            manifest = verify_runtime(source)
            self.assertEqual(manifest['schema'], 'hexapod_runtime_sources_v2')
            for path in ('locomotion/core/motor.py', 'locomotion/tasks/task.py',
                         'locomotion/backends/isaac/env.py', 'locomotion/learning/ppo.py'):
                self.assertIn(path, manifest['files'])
            self.assertFalse(any('/tests/' in name for name in manifest['files']))
            path = source/'locomotion/core/motor.py'
            path.write_text(path.read_text() + '\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'Runtime dependency'):
                verify_runtime(source)

    def test_prepare_and_training_use_shared_validation(self):
        from locomotion.runs import prepare as prepare
        from locomotion.runs import train as train
        from locomotion.runs import validation
        for name in ('update_limit', 'validate_decay', 'validate_deadline'):
            self.assertIs(getattr(prepare, name), getattr(validation, name))
            self.assertIs(getattr(train, name), getattr(validation, name))

    def test_nested_physics_dependencies_cannot_reuse_standing_identity(self):
        from locomotion.runs.sources import physics_files, source_files
        root = Path(__file__).resolve().parents[1]
        files = source_files(root)
        physics = physics_files(files)
        for name in ('core/motor.py', 'core/transforms.py', 'backends/isaac/native.py', 'backends/isaac/motor.py',
                     'evaluation/capture.py', 'evaluation/scoring.py'):
            self.assertIn(name, physics)
            changed = dict(files, **{name: '0'*64})
            self.assertNotEqual(physics, physics_files(changed))
        self.assertNotIn('learning/config.py', physics)

    def test_legacy_identity_checks_remain_strict(self):
        from copy import deepcopy
        from contracts.release import COMPATIBILITY_KEYS
        from locomotion.runs.identity import require_checkpoint_identity
        identity = dict.fromkeys(COMPATIBILITY_KEYS, 'same')
        identity['schema'] = 'hexapod_locomotion_ppo_v1'
        require_checkpoint_identity({'identity': deepcopy(identity)}, identity)
        for key in COMPATIBILITY_KEYS:
            record = {'identity': deepcopy(identity)}
            record['identity'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_checkpoint_identity(record, identity)
        identity.update(schema='hexapod_locomotion_ppo_v2', runtime_sources={'schema': 'hexapod_runtime_sources_v2',
                        'files': {'locomotion/core/motor.py': 'a'*64}})
        record = {'identity': deepcopy(identity)}
        require_checkpoint_identity(record, identity)
        record['identity']['runtime_sources']['files']['locomotion/core/motor.py'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'runtime dependencies'):
            require_checkpoint_identity(record, identity)


if __name__ == '__main__':
    unittest.main()
