"""Pure supervisor checks; no Docker, GPU, SSH or lock acquisition."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'isaaclab/deploy/validate-mkii-v2'
loader = importlib.machinery.SourceFileLoader('_validate_mkii_v2_supervisor_tests', str(LAUNCHER))
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec); loader.exec_module(module)


class SupervisorTests(unittest.TestCase):
    def test_process_ownership_does_not_exclude_unrelated_siblings(self):
        processes = module.parse_processes('10 1 bash launch\n11 10 python supervisor\n12 11 docker client\n20 1 python nowcast_run.py\n21 10 python score_clip.py\n22 1 tail -f nowcast_run.py.log\n')
        owned = module.descendants(processes, [11])
        self.assertEqual(owned, {11, 12})
        self.assertEqual(module.ancestors(processes, 11), {10, 11})
        self.assertEqual(module.blocking_producers(processes, owned | module.ancestors(processes, 11)), [20, 21])

    def test_gpu_capable_containers_fail_even_if_current_gpu_usage_is_idle(self):
        cpu = {'runtime': 'runc', 'device_requests': None}
        self.assertFalse(module.container_gpu_capable(cpu))
        for value in (
            dict(cpu, runtime='nvidia'), dict(cpu, privileged=True),
            dict(cpu, devices=[{'PathOnHost': '/dev/nvidia0'}]),
            dict(cpu, device_requests=[{'Driver': '', 'Capabilities': [['gpu']]}]),
        ):
            self.assertTrue(module.container_gpu_capable(value))

    def test_cleanup_identity_needs_exact_name_and_unpredictable_owner_label(self):
        identity = {'id': 'a'*64, 'name': '/owned', 'labels': {module.OWNER_LABEL: 'nonce'}}
        self.assertTrue(module.check_identity(identity, 'owned', 'nonce'))
        self.assertFalse(module.check_identity(identity, 'other', 'nonce'))
        self.assertFalse(module.check_identity(identity, 'owned', 'other'))
        self.assertFalse(module.check_identity(None, 'owned', 'nonce'))

    def test_compose_preserves_paths_and_validator_args_as_argv_elements(self):
        argv = module.compose_argv(Path('/source with spaces'), Path('/out with spaces'), 'owned', 'nonce', 32, 1000)
        self.assertIn('/source with spaces:/workspace/hexapod:ro', argv)
        self.assertIn('/out with spaces:/workspace/validation_artifacts:rw', argv)
        self.assertIn('--no-deps', argv)
        self.assertNotIn('--rm', argv)
        self.assertNotIn('--resume', argv)
        self.assertEqual(argv[-4:], ['--viz', 'none', '--device', 'cuda:0'])
        self.assertEqual(argv[argv.index('--report')+1], '/workspace/validation_artifacts/report.json')
        self.assertIn('exec "$@"', argv[argv.index('-c')+1])

    def test_combined_telemetry_mitigation_is_one_unquoted_argv_element(self):
        argv = module.compose_argv(Path('/source'), Path('/output'), 'owned', 'nonce', 32, 1000)
        kit_args = [argument for argument in argv if argument.startswith('--kit_args=')]
        self.assertEqual(kit_args, ['--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry'])
        self.assertNotIn('"', kit_args[0])

    def test_source_manifest_records_exact_bytes_and_refuses_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); source = directory/'source'; source.mkdir()
            (source/'a.py').write_text('example\n'); manifest = directory/'source.SHA256SUMS'
            report = module.snapshot_source(source, manifest)
            self.assertEqual(report['files'], 1)
            self.assertTrue(manifest.read_text().endswith('  a.py\n'))
            (source/'escape').symlink_to(manifest)
            with self.assertRaisesRegex(module.Blocked, 'escapes'):
                module.snapshot_source(source, manifest)

    def test_zero_exit_requires_explicit_successful_standing_report(self):
        base = {'pass': True, 'standing_gate_pass': True, 'errors': [],
                'task_id': 'Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0',
                'run_kind': 'short_probe', 'steps_requested': 100, 'steps_completed': 100, 'num_envs': 32}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with self.assertRaisesRegex(module.Blocked, 'without its required report'):
                module.validate_written_report(path, steps=100, num_envs=32)
            path.write_text(json.dumps(base))
            self.assertEqual(module.validate_written_report(path, steps=100, num_envs=32)['run_kind'], 'short_probe')
            for changed in ({'pass': False}, {'standing_gate_pass': False}, {'pass': 1}, {'pass': 'true'},
                            {'errors': ['standing torque failed']}, {'task_id': 'old_mock'},
                            {'steps_completed': 99}, {'run_kind': 'acceptance'}, {'num_envs': 1}):
                with self.subTest(changed=changed):
                    path.write_text(json.dumps(dict(base, **changed)))
                    with self.assertRaises(module.Blocked):
                        module.validate_written_report(path, steps=100, num_envs=32)
            path.write_text(json.dumps(dict(base, run_kind='acceptance', steps_requested=1000, steps_completed=1000)))
            self.assertEqual(module.validate_written_report(path, steps=1000, num_envs=32)['run_kind'], 'acceptance')
            for invalid in ('not json', '[]', '{"pass": NaN}'):
                path.write_text(invalid)
                with self.assertRaises(module.Blocked):
                    module.validate_written_report(path, steps=100, num_envs=32)

    def test_dry_run_uses_no_site_packages_docker_gpu_or_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'; (source/'isaaclab').mkdir(parents=True)
            (source/'isaaclab/validate_mkii_v2.py').write_text('raise RuntimeError("must not execute")\n')
            result = subprocess.run([sys.executable, '-S', str(LAUNCHER), '--source-dir', str(source), '--source-commit', 'a0f0b39', '--dry-run'], text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report['execution'], 'not_started')
            self.assertFalse(Path(report['output']).exists())
            self.assertEqual(report['timeout_seconds'], 900)


if __name__ == '__main__':
    unittest.main()
