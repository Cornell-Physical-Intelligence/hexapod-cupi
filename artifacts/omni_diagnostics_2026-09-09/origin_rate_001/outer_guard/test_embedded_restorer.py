"""Exercise the actual Python embedded in the dispatch guard, without services."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


GUARD = Path(__file__).with_name('launch_guarded_remote.py')


def embedded_restorer():
    tree = ast.parse(GUARD.read_text())
    values = [n.args[0].value for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.value.id == 'restorer'
              and n.func.attr == 'write_text' and n.args
              and isinstance(n.args[0], ast.Constant)]
    assert len(values) == 1
    return compile(values[0], '<actual embedded restorer>', 'exec')


class EmbeddedRestorerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pause = Path(self.tmp.name) / 'pause'
        self.output = Path(self.tmp.name) / 'output'
        self.pause.mkdir()
        (self.output / 'jobs').mkdir(parents=True)
        self.script = self.pause / 'resume_forecasting.py'
        self.record = dict(unit='exact-owner.service', output=str(self.output), units={
            'previous-active.timer': 'ActiveState=active\nSubState=waiting',
            'previous-inactive.timer': 'ActiveState=inactive\nSubState=dead',
            'previous-active.service': 'ActiveState=active\nSubState=running'})
        (self.pause / 'pause.json').write_text(json.dumps(self.record))
        self.job = dict(container_name='exact-container', container_id='a' * 64)
        (self.output / 'jobs/standing.json').write_text(json.dumps(self.job))
        self.calls = []

    def result(self, text='', error='', code=0):
        return subprocess.CompletedProcess([], code, text, error)

    def run_code(self, docker_results, stop_owner=False, fail_stop=False):
        responses = iter(docker_results)
        def run(args, **kwargs):
            self.calls.append(args)
            if args[:2] == ['docker', 'inspect']:
                return next(responses)
            if args[:2] == ['docker', 'stop'] and fail_stop:
                raise subprocess.CalledProcessError(1, args)
            return self.result()
        with patch('subprocess.run', side_effect=run), patch.object(
                sys, 'argv', [str(self.script)] + (['--stop-owner'] if stop_owner else [])):
            exec(embedded_restorer(), {'__file__': str(self.script), '__name__': '__main__'})

    def assert_not_restored(self):
        self.assertFalse((self.pause / 'restored.json').exists())
        self.assertFalse(any(c[:3] == ['systemctl', '--user', 'start'] for c in self.calls))

    def inspect(self, running, identity=None, name='exact-container'):
        return self.result(f"{identity or self.job['container_id']} /{name} {running}\n")

    def test_wrong_identity_or_name_never_signalled_or_restored(self):
        for result in [self.inspect('true', identity='b' * 64), self.inspect('true', name='other')]:
            with self.subTest(result=result.stdout), self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                self.run_code([result])
            self.assert_not_restored()
            self.assertFalse(any(c[:2] == ['docker', 'stop'] for c in self.calls))

    def test_daemon_permission_and_malformed_results_do_not_restore(self):
        for result in [self.result(error='Cannot connect to Docker daemon', code=1),
                       self.result(error='permission denied', code=1), self.result('bad fields')]:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                self.run_code([result])
            self.assert_not_restored()

    def test_owned_running_container_stopped_then_checked_before_only_active_timer(self):
        self.run_code([self.inspect('true'), self.inspect('false')], stop_owner=True)
        self.assertEqual(self.calls[0], ['systemctl', '--user', 'stop', 'exact-owner.service'])
        self.assertEqual(self.calls[2], ['docker', 'stop', '--time', '20', self.job['container_id']])
        self.assertEqual(self.calls[-1], ['systemctl', '--user', 'start', 'previous-active.timer'])
        self.assertEqual(json.loads((self.pause / 'restored.json').read_text())['owned_cleanup_checked'],
                         [self.job['container_id']])

    def test_explicit_absence_allows_restore_without_signalling_another_container(self):
        self.run_code([self.result(error='Error: No such object: ' + self.job['container_id'], code=1)])
        self.assertFalse(any(c[:2] == ['docker', 'stop'] for c in self.calls))
        self.assertTrue((self.pause / 'restored.json').exists())

    def test_post_stop_unknown_running_or_wrong_identity_does_not_restore(self):
        for result in [self.result(error='permission denied', code=1), self.inspect('true'),
                       self.inspect('false', identity='b' * 64)]:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                self.run_code([self.inspect('true'), result])
            self.assert_not_restored()

    def test_docker_stop_failure_does_not_restore(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_code([self.inspect('true')], fail_stop=True)
        self.assert_not_restored()

    def test_name_only_startup_job_recovers_exact_identity_before_stopping(self):
        self.job['container_id'] = None
        (self.output / 'jobs/standing.json').write_text(json.dumps(self.job))
        self.run_code([self.inspect('true', identity='c' * 64), self.inspect('false', identity='c' * 64)])
        self.assertEqual(self.calls[0][-1], 'exact-container')
        self.assertEqual(self.calls[1][-1], 'c' * 64)

    def test_ambiguous_dispatch_failure_stops_owner_before_restoration(self):
        tree = ast.parse(GUARD.read_text())
        cases = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
                 and isinstance(n.test.operand, ast.Name) and n.test.operand.id == 'launched']
        self.assertEqual(len(cases), 1)
        calls = [n for n in ast.walk(cases[0]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'run']
        self.assertEqual(len(calls), 1)
        self.assertIn('--stop-owner', [n.value for n in ast.walk(calls[0].args[0])
                                     if isinstance(n, ast.Constant)])


if __name__ == '__main__':
    unittest.main()
