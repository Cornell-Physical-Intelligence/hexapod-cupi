"""Check runtime packs, the physics dependency closure and checkpoint identity keys."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

from contracts.release import COMPATIBILITY_KEYS
from locomotion.evaluation.checkpoint import require_checkpoint_identity
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import ppo_config
from locomotion.spark.prepare import prepare
from locomotion.spark.sources import physics_files, source_files, verify_runtime

ROOT = Path(__file__).resolve().parents[1]


class DependencyIdentityTests(unittest.TestCase):
    def test_nested_pack_is_complete_and_detects_dependency_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            prepare(output, '/srv/cupi/hexapod/runs/james/dependency_fixture', updates=2)
            source = output/'source'
            manifest = verify_runtime(source)
            self.assertEqual(manifest['schema'], 'hexapod_runtime_sources_v2')
            for path in ('locomotion/omniwalk/motor.py', 'locomotion/omniwalk/task.py',
                         'locomotion/omniwalk/env.py', 'locomotion/omniwalk/ppo.py'):
                self.assertIn(path, manifest['files'])
            self.assertFalse(any('/tests/' in name for name in manifest['files']))
            path = source/'locomotion/omniwalk/motor.py'
            path.write_text(path.read_text() + '\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'Runtime dependency'):
                verify_runtime(source)

    def test_nested_physics_dependencies_cannot_reuse_standing_identity(self):
        files = source_files(ROOT)
        physics = physics_files(files)
        for name in ('omniwalk/motor.py', 'omniwalk/frames.py', 'omniwalk/isaac.py', 'omniwalk/actuator.py',
                     'evaluation/capture.py', 'evaluation/scoring.py'):
            self.assertIn(name, physics)
            changed = dict(files, **{name: '0'*64})
            self.assertNotEqual(physics, physics_files(changed))
        self.assertNotIn('omniwalk/ppo.py', physics)

    def test_physics_closure_starts_at_the_environment_and_its_configuration(self):
        paths = physics_files(source_files(ROOT))
        self.assertIn('omniwalk/env.py', paths)
        self.assertIn('omniwalk/env_cfg.py', paths)

    def test_identity_checks_reject_each_changed_key_and_runtime_source(self):
        identity = dict.fromkeys(COMPATIBILITY_KEYS, 'same')
        identity['schema'] = 'hexapod_locomotion_ppo_v1'
        require_checkpoint_identity({'identity': deepcopy(identity)}, identity)
        for key in COMPATIBILITY_KEYS:
            record = {'identity': deepcopy(identity)}
            record['identity'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_checkpoint_identity(record, identity)
        identity.update(schema='hexapod_locomotion_ppo_v2', runtime_sources={'schema': 'hexapod_runtime_sources_v2',
                        'files': {'locomotion/omniwalk/motor.py': 'a'*64}})
        record = {'identity': deepcopy(identity)}
        require_checkpoint_identity(record, identity)
        record['identity']['runtime_sources']['files']['locomotion/omniwalk/motor.py'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'runtime dependencies'):
            require_checkpoint_identity(record, identity)

    def test_learner_configuration_uses_package_classes(self):
        config = ppo_config(17, action_mean='tanh', learning_rate_max=.0003)
        self.assertEqual(config['algorithm']['class_name'],
                         'locomotion.omniwalk.ppo:CappedRatePPO')
        self.assertEqual(config['actor']['distribution_cfg']['class_name'],
                         'locomotion.omniwalk.ppo:BoundedMeanGaussian')

    def test_checkpoint_module_adds_no_global_import_hook(self):
        before = list(sys.meta_path)
        importlib.import_module('locomotion.evaluation.checkpoint')
        self.assertEqual(before, sys.meta_path)


if __name__ == '__main__':
    unittest.main()


class SharedValidationTests(unittest.TestCase):
    def test_prepare_and_training_share_one_validation(self):
        """The host preparer and the container trainer accept the same packs only if they share one rule set."""
        import locomotion.omniwalk.train as train
        import locomotion.spark.prepare as prepare_module
        import locomotion.spark.validation as validation
        for name in ('update_limit', 'validate_decay', 'validate_deadline'):
            self.assertIs(getattr(prepare_module, name), getattr(validation, name))
            self.assertIs(getattr(train, name), getattr(validation, name))
