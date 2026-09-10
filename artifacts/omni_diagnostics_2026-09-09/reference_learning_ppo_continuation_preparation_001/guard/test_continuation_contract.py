"""Check exact external admission and bounded continuation without GPU or services."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('continuation_guard', HERE/'launch_guarded_remote.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ContinuationContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base/'reference_learning_ppo_001'
        self.output.mkdir()
        self.decision = self.base/'decision.json'
        self.decision.write_text('{"accepted":true}')
        self.decision_hash = hashlib.sha256(self.decision.read_bytes()).hexdigest()
        self.pins = {}
        for name in guard.PRIOR_PINS:
            file = self.base/name; file.parent.mkdir(parents=True, exist_ok=True)
            value = {'status':'completed'}
            if '/jobs/' in name:
                value.update(cleanup_checked=True, container_name=file.stem+'-owned', container_id=file.stem+'-id')
            file.write_text(json.dumps(value))
            self.pins[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for key, value in {'BASE':self.base, 'OUTPUT':self.output, 'PAUSE':self.base/'pause052',
                           'DECISION_RECEIPT':self.decision, 'DECISION_SHA256':self.decision_hash,
                           'PRIOR_PINS':self.pins, 'PREVIOUS_INVOCATION':'a'*32}.items():
            p = patch.object(guard, key, value); p.start(); self.addCleanup(p.stop)
        self.unit = 'ActiveState=inactive\nSubState=dead\nInvocationID='+'a'*32

    def test_each_pending_binding_refuses_before_any_service_action(self):
        with patch.object(guard, 'DECISION_SHA256', 'PENDING'), patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'pending'): guard.main()
            run.assert_not_called()
        with patch.object(guard, 'PREVIOUS_INVOCATION', 'PENDING'):
            with self.assertRaisesRegex(RuntimeError, 'pending'): guard.require_final_bindings()
        for name in self.pins:
            with self.subTest(name=name), patch.object(guard, 'PRIOR_PINS', {**self.pins, name:'PENDING'}):
                with self.assertRaisesRegex(RuntimeError, 'pending'): guard.require_final_bindings()
        self.assertFalse(guard.PAUSE.exists())

    def test_binding_cannot_omit_an_admission_job(self):
        with patch.object(guard, 'PRIOR_PINS', {k:v for k,v in self.pins.items() if 'calibration' not in k}):
            with self.assertRaisesRegex(RuntimeError, 'three jobs'): guard.require_final_bindings()

    def test_exact_prior_three_jobs_require_six_absent_tokens(self):
        result = subprocess.CompletedProcess([], 1, '', 'No such object')
        with patch.object(guard, 'call', return_value=self.unit), patch.object(guard.subprocess, 'run', return_value=result) as run:
            guard.verify_previous_owner()
        self.assertEqual(run.call_count, 6)
        self.assertEqual(len({c.args[0][-1] for c in run.call_args_list}), 6)

    def test_active_or_restarted_admission_owner_is_rejected(self):
        for unit in (self.unit.replace('inactive', 'active'), self.unit.replace('a'*32, 'b'*32)):
            with self.subTest(unit=unit), patch.object(guard, 'call', return_value=unit), patch.object(guard.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError): guard.verify_previous_owner()
                run.assert_not_called()

    def test_collected_inactive_unit_allows_only_empty_live_invocation_with_all_proofs(self):
        unit = self.unit.replace('a'*32, '')
        result = subprocess.CompletedProcess([], 1, '', 'No such object')
        with patch.object(guard, 'call', return_value=unit), patch.object(guard.subprocess, 'run', return_value=result) as run:
            guard.require_final_bindings()
            guard.verify_previous_owner()
        self.assertEqual(run.call_count, 6)
        with patch.object(guard, 'call', return_value=unit.replace('inactive','active')), patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'not exited'): guard.verify_previous_owner()
            run.assert_not_called()

    def test_missing_restoration_or_unknown_container_never_allows_pause(self):
        with patch.object(guard, 'call', return_value=self.unit), patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'permission denied')):
            with self.assertRaisesRegex(RuntimeError, 'not proven absent'): guard.verify_previous_owner()
        (self.base/'forecast_pause_051/restored.json').unlink()
        with patch.object(guard, 'call', return_value=self.unit), patch.object(guard.subprocess, 'run') as run:
            with self.assertRaises(FileNotFoundError): guard.verify_previous_owner()
            run.assert_not_called()
        self.assertFalse(guard.PAUSE.exists())

    def test_exact_host_admission_is_called_with_the_same_args_before_prior_check(self):
        args = SimpleNamespace(decision_sha256=self.decision_hash)
        baseline = {'exact':'identity'}
        host = Mock()
        host.admit_existing_campaign.return_value = ({'asset':'sha'}, {'decision_sha256':self.decision_hash})
        result = guard.admit_learning_before_pause(host, args, baseline)
        self.assertEqual(result[0], {'asset':'sha'})
        self.assertEqual([c[0] for c in host.mock_calls], ['admit_existing_campaign','verify_prior_files'])
        host.admit_existing_campaign.assert_called_once_with(args, baseline)
        host.verify_prior_files.assert_called_once_with(args)
        self.assertFalse(guard.PAUSE.exists())

    def test_host_rejection_and_decision_substitution_are_not_bypassed(self):
        args = SimpleNamespace(decision_sha256=self.decision_hash)
        host = Mock(); host.admit_existing_campaign.side_effect = ValueError('actual admission failed')
        with self.assertRaisesRegex(ValueError, 'actual admission failed'):
            guard.admit_learning_before_pause(host, args, {})
        host.verify_prior_files.assert_not_called()
        self.decision.write_text('changed')
        host.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'decision changed'):
            guard.admit_learning_before_pause(host, args, {})
        host.admit_existing_campaign.assert_not_called()
        self.assertFalse(guard.PAUSE.exists())

    def test_host_cannot_return_a_different_review_receipt(self):
        args = SimpleNamespace(decision_sha256=self.decision_hash)
        host = Mock(); host.admit_existing_campaign.return_value = ({}, {'decision_sha256':'b'*64})
        with self.assertRaisesRegex(RuntimeError, 'different external'):
            guard.admit_learning_before_pause(host, args, {})
        host.verify_prior_files.assert_not_called()

    def test_dispatch_is_only_learn_with_external_receipt_and_bounded_time(self):
        tree = ast.parse((HERE/'launch_guarded_remote.py').read_text())
        commands = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'cmd' for t in n.targets)]
        self.assertEqual(len(commands), 1)
        values = {k:getattr(guard,k) for k in ('UNIT','HOST','SOURCE','RUN','DEVICE_RUN','BRIDGE','CONSUMER','OBSERVATION','OUTPUT','DECISION_RECEIPT')}
        values['restorer'] = Path('/exact/restorer.py')
        command = eval(compile(ast.Expression(commands[0]), '<actual guard command>', 'eval'), {'str':str}, values)
        self.assertEqual(command[command.index('--phase-group')+1], 'learn')
        self.assertEqual(command[command.index('--decision-receipt')+1], str(self.decision))
        self.assertIn('--property=RuntimeMaxSec=3720', command)
        self.assertIn('--property=TimeoutStopSec=180', command)
        self.assertNotIn('train_25', command)
        self.assertEqual(sum(isinstance(n, ast.Constant) and n.value=='--on-active=70m' for n in ast.walk(tree)), 1)

    def test_full_host_admission_precedes_pause_and_rechecks_before_dispatch(self):
        tree = ast.parse((HERE/'launch_guarded_remote.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        admission = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id=='admit_learning_before_pause')
        pause = next(n.lineno for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value,ast.Name)
                     and n.func.value.id=='PAUSE' and n.func.attr=='mkdir')
        self.assertLess(admission, pause)
        checks = [n.lineno for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value,ast.Name)
                  and n.func.value.id=='learning_host' and n.func.attr=='verify_prior_files']
        self.assertEqual(len(checks), 3)
        self.assertTrue(any(admission < n < pause for n in checks))
        self.assertTrue(any(n > pause for n in checks))

    def test_post_pause_integrity_failure_uses_known_not_launched_restoration(self):
        tree = ast.parse((HERE/'launch_guarded_remote.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
        pause_try = next(n for n in main.body if isinstance(n, ast.Try) and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name) and c.func.value.id=='PAUSE'
            and c.func.attr=='mkdir' for c in ast.walk(n)))
        self.assertTrue(any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr=='verify_prior_files' for stmt in pause_try.body for c in ast.walk(stmt)))
        self.assertEqual(len(pause_try.handlers), 1)
        handler_literals = [n.value for n in ast.walk(pause_try.handlers[0]) if isinstance(n, ast.Constant)]
        self.assertNotIn('--stop-owner', handler_literals)
        self.assertIn('/usr/bin/python3', handler_literals)


if __name__ == '__main__':
    unittest.main()
