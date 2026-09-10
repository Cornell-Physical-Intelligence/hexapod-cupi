"""Exercise admission-only ownership and immutable prerequisites without dispatch."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('learning_admission_guard', HERE/'launch_guarded_remote.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        shutil.copytree(HERE/'previous_owner', self.base, dirs_exist_ok=True)
        self.patch = patch.object(guard, 'BASE', self.base)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.unit = 'ActiveState=failed\nSubState=failed\nInvocationID='+guard.PREVIOUS_INVOCATION

    def absent(self, *args, **kwargs):
        return subprocess.CompletedProcess(args, 1, '', 'Error: No such object')

    def test_unbound_guard_refuses_before_any_process_or_pause(self):
        with patch.object(guard, 'PAUSE', self.base/'new-pause'), patch.object(guard, 'OUTPUT', self.base/'new-output'), \
             patch.object(guard, 'HOST_SHA256', 'PENDING'), patch.object(guard.subprocess, 'run') as run, \
             patch.object(guard.subprocess, 'check_output') as call:
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                guard.main()
        run.assert_not_called()
        call.assert_not_called()
        self.assertFalse((self.base/'new-pause').exists())

    def test_all_three_final_bindings_are_required(self):
        for field in ('HOST_SHA256', 'HOST_FREEZE_SHA256', 'CONSUMER_FREEZE_SHA256'):
            with self.subTest(field=field), patch.object(guard, 'HOST_SHA256', 'a'*64), \
                 patch.object(guard, 'HOST_FREEZE_SHA256', 'b'*64), \
                 patch.object(guard, 'CONSUMER_FREEZE_SHA256', 'c'*64), patch.object(guard, field, 'PENDING'):
                with self.assertRaisesRegex(RuntimeError, 'pending'):
                    guard.require_final_bindings()

    def test_exact_failed_prior_owner_with_four_absent_tokens_is_eligible(self):
        with patch.object(guard, 'call', return_value=self.unit), patch.object(guard.subprocess, 'run', side_effect=self.absent) as run:
            guard.verify_previous_owner()
        self.assertEqual(run.call_count, 4)
        identifiers = {call.args[0][-1] for call in run.call_args_list}
        expected = set()
        for p in (self.base/'reference_pair_motion_001/jobs').glob('*.json'):
            job = json.loads(p.read_text())
            expected.update((job['container_name'], job['container_id']))
        self.assertEqual(identifiers, expected)

    def test_restarted_or_active_prior_owner_is_rejected_before_inspect(self):
        for unit in (self.unit.replace('ActiveState=failed', 'ActiveState=active'),
                     self.unit.replace(guard.PREVIOUS_INVOCATION, 'different')):
            with self.subTest(unit=unit), patch.object(guard, 'call', return_value=unit), patch.object(guard.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    guard.verify_previous_owner()
                run.assert_not_called()

    def test_changed_prior_restoration_is_rejected_before_inspect(self):
        (self.base/'forecast_pause_050/restored.json').write_text('{}')
        with patch.object(guard, 'call', return_value=self.unit), patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'receipt changed'):
                guard.verify_previous_owner()
            run.assert_not_called()

    def test_live_or_unknown_previous_container_is_not_absence(self):
        for result in (subprocess.CompletedProcess([], 0, 'exact /owner false', ''),
                       subprocess.CompletedProcess([], 1, '', 'Cannot connect to Docker daemon')):
            with self.subTest(result=result), patch.object(guard, 'call', return_value=self.unit), \
                 patch.object(guard.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'not proven absent'):
                    guard.verify_previous_owner()

    def test_frozen_bundle_rejects_payload_mutation_addition_and_symlink(self):
        folder = self.base/'bundle'; folder.mkdir()
        payload = folder/'runtime.py'; payload.write_text('exact')
        manifest = folder/'FREEZE_SHA256.json'
        manifest.write_text(json.dumps({'runtime.py': hashlib.sha256(payload.read_bytes()).hexdigest()}))
        bound = hashlib.sha256(manifest.read_bytes()).hexdigest()
        guard.verify_frozen(folder, bound)
        payload.write_text('changed')
        with self.assertRaises(RuntimeError): guard.verify_frozen(folder, bound)
        payload.write_text('exact')
        extra = folder/'extra'; extra.write_text('extra')
        with self.assertRaises(RuntimeError): guard.verify_frozen(folder, bound)
        extra.unlink(); extra.symlink_to(payload)
        with self.assertRaises(RuntimeError): guard.verify_frozen(folder, bound)

    def test_dispatch_is_only_admission_with_declared_deadline_and_fallback(self):
        tree = ast.parse((HERE/'launch_guarded_remote.py').read_text())
        commands = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'cmd' for t in n.targets)]
        self.assertEqual(len(commands), 1)
        names = {k: getattr(guard, k) for k in ('UNIT','HOST','SOURCE','RUN','DEVICE_RUN','BRIDGE','CONSUMER','OBSERVATION','OUTPUT')}
        names['restorer'] = Path('/exact/resume.py')
        command = eval(compile(ast.Expression(commands[0]), '<guard dispatch expression>', 'eval'), {'str': str}, names)
        self.assertIn('--property=RuntimeMaxSec=1920', command)
        self.assertIn('--property=TimeoutStopSec=180', command)
        self.assertEqual(command[command.index('--phase-group')+1], 'admission')
        self.assertNotIn('--decision-receipt', command)
        self.assertNotIn('learn', command)
        self.assertNotIn('train_10', command)
        fallback = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == '--on-active=40m']
        self.assertEqual(len(fallback), 1)


if __name__ == '__main__':
    unittest.main()
