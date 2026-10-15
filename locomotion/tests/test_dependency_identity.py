"""Check runtime packs, the physics dependency closure and checkpoint identity keys."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from contracts.release import COMPATIBILITY_KEYS
from locomotion.evaluation.checkpoint import require_checkpoint_identity
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import ppo_config
import locomotion.scripts.cli as cli
import locomotion.scripts.play as play
import locomotion.scripts.train as train
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
            for path in ('locomotion/robots/rs05.py', 'locomotion/omniwalk/task.py',
                         'locomotion/omniwalk/adapter.py', 'locomotion/rl/ppo.py'):
                self.assertIn(path, manifest['files'])
            self.assertFalse(any('/tests/' in name for name in manifest['files']))
            path = source/'locomotion/robots/rs05.py'
            path.write_text(path.read_text() + '\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'Runtime dependency'):
                verify_runtime(source)

    def test_nested_physics_dependencies_cannot_reuse_standing_identity(self):
        files = source_files(ROOT)
        physics = physics_files(files)
        for name in ('robots/rs05.py', 'omniwalk/mdp/actions.py', 'omniwalk/mdp/frames.py', 'omniwalk/env.py', 'robots/actuator.py',
                     'evaluation/capture.py', 'evaluation/scoring.py'):
            self.assertIn(name, physics)
            changed = dict(files, **{name: '0'*64})
            self.assertNotEqual(physics, physics_files(changed))
        self.assertNotIn('rl/ppo.py', physics)

    def test_physics_closure_starts_at_the_environment_and_its_configuration(self):
        paths = physics_files(source_files(ROOT))
        self.assertIn('omniwalk/adapter.py', paths)
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
                        'files': {'locomotion/robots/rs05.py': 'a'*64}})
        record = {'identity': deepcopy(identity)}
        require_checkpoint_identity(record, identity)
        record['identity']['runtime_sources']['files']['locomotion/robots/rs05.py'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'runtime dependencies'):
            require_checkpoint_identity(record, identity)

    def test_training_and_evaluation_scripts_bind_the_same_entry(self):
        """A checkpoint that train.py declares passes play.py's identity check; a changed entry hash fails it."""
        remote = '/srv/cupi/hexapod/runs/james/entry_fixture'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'stance.json').write_text(json.dumps({'nominal_joint_position_rad': [0.]*18, 'reset_root_height_m': .2}))
            for name in ('geometry.json', 'geometry_extrema.npz', 'checkpoint.pt'):
                (root/name).write_bytes(name.encode())
            local = {'--stance': root/'stance.json', '--geometry': root/'geometry.json',
                     '--geometry-extrema': root/'geometry_extrema.npz', '--checkpoint': root/'checkpoint.pt'}
            checkpoint_sha = cli.sha(root/'checkpoint.pt')

            def command(mode, **options):
                """Return the pack's command line with local inputs in place of the container paths."""
                args = prepare(root/mode, remote, mode=mode, updates=2000, **options)['command_args']
                for option, value in local.items():
                    if option in args:
                        args[args.index(option)+1] = str(value)
                return args

            def preflight(script, args):
                """Parse a command line with the script's parser and return the identity its preflight prints."""
                # A checkout has no frozen pack manifests or standing captures; the identity path runs unchanged.
                with patch.object(cli, 'validate', return_value=None), \
                        patch.object(cli, 'verify_runtime', return_value={'schema': 'fixture'}), \
                        patch.object(cli, 'require_admission', return_value={'fixture': True}), \
                        redirect_stdout(io.StringIO()) as printed:
                    self.assertEqual(script.main(args + ['--output', str(root/'unused'), '--preflight-only']), 0)
                return json.loads(printed.getvalue())

            trained = preflight(train, command('train'))
            self.assertEqual(trained['entry_sha256'], cli.sha(ROOT/'scripts/train.py'))
            evaluation = command('evaluate', checkpoint=remote+'/checkpoint.pt', checkpoint_sha=checkpoint_sha,
                                 checkpoint_declaration_sha='b'*64)
            declaration = {'identity': trained, 'updates': 2000, 'checkpoint_sha256': checkpoint_sha}
            (root/'checkpoint.json').write_text(json.dumps(declaration))
            # play.py binds the checkpoint through require_checkpoint_identity before its preflight prints.
            evaluated = preflight(play, evaluation)
            self.assertEqual(evaluated['entry_sha256'], trained['entry_sha256'])
            require_checkpoint_identity(declaration, evaluated)
            declaration['identity'] = dict(trained, entry_sha256='0'*64)
            (root/'checkpoint.json').write_text(json.dumps(declaration))
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                require_checkpoint_identity(declaration, evaluated)
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                preflight(play, evaluation)

    def test_learner_configuration_uses_package_classes(self):
        config = ppo_config(17, action_mean='tanh', learning_rate_max=.0003)
        self.assertEqual(config['algorithm']['class_name'],
                         'locomotion.rl.ppo:CappedRatePPO')
        self.assertEqual(config['actor']['distribution_cfg']['class_name'],
                         'locomotion.rl.ppo:BoundedMeanGaussian')

    def test_checkpoint_module_adds_no_global_import_hook(self):
        before = list(sys.meta_path)
        importlib.import_module('locomotion.evaluation.checkpoint')
        self.assertEqual(before, sys.meta_path)


if __name__ == '__main__':
    unittest.main()


class SharedValidationTests(unittest.TestCase):
    def test_prepare_and_training_share_one_validation(self):
        """The host preparer and the container scripts accept the same packs only if they share one rule set."""
        import locomotion.scripts.cli as cli
        import locomotion.spark.prepare as prepare_module
        import locomotion.spark.validation as validation
        for name in ('update_limit', 'validate_decay', 'validate_deadline'):
            self.assertIs(getattr(prepare_module, name), getattr(validation, name))
            self.assertIs(getattr(cli, name), getattr(validation, name))
