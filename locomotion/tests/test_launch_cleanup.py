"""Exercise container/client exit races without starting remote processes."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from locomotion import launch


class CleanupTests(unittest.TestCase):
    def run_failed_client(self, owned_states):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root/'logs').mkdir()
            (root/'jobs').mkdir()
            process = Mock(returncode=1)
            process.poll.return_value = 1
            for name, result in (('preflight', {}),):
                stack.enter_context(patch.object(launch, name, return_value=result))
            stack.enter_context(patch.object(launch.guard, 'verify_tree', return_value=None))
            stack.enter_context(patch.object(launch.os, 'open', return_value=10))
            stack.enter_context(patch.object(launch.os, 'close'))
            stack.enter_context(patch.object(launch.fcntl, 'flock'))
            stack.enter_context(patch.object(launch, 'command', return_value=['fixture']))
            stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=process))
            inspect = stack.enter_context(patch.object(launch, 'owned_container', side_effect=owned_states))
            stop = stack.enter_context(patch.object(launch.subprocess, 'run'))
            binding = {'mode': 'diagnostic', 'max_seconds': 120, 'source_freeze_sha256': 'a'*64}
            with self.assertRaises(RuntimeError):
                launch.run_owned(binding, {'source': root/'source', 'output': root})
            report = json.loads((root/'jobs/standing.json').read_text())
            return stop.call_args_list, inspect.call_args_list, report

    def test_client_exit_before_first_poll_still_stops_its_container(self):
        calls, inspections, report = self.run_failed_client([('a'*64, True), None])
        self.assertEqual(calls[0].args[0], ['docker', 'stop', '--time', '20', 'a'*64])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(inspections), 2)
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['cleanup_checked'])

    def test_container_creation_racing_client_exit_gets_second_inspection(self):
        calls, _, report = self.run_failed_client([None, ('b'*64, True)])
        self.assertEqual(calls[0].args[0][-1], 'b'*64)
        self.assertTrue(report['cleanup_checked'])

    def test_unknown_identity_prevents_signalling_and_records_owner_review(self):
        calls, _, report = self.run_failed_client([RuntimeError('identity differs')])
        self.assertEqual(calls, [])
        self.assertFalse(report['cleanup_checked'])
        self.assertTrue(report['cleanup_requires_owner_review'])

    def test_other_gpu_work_is_recorded_without_stopping_the_run(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root/'logs').mkdir()
            (root/'jobs').mkdir()
            process = Mock(returncode=1)
            process.poll.side_effect = [None, 1, 1]
            other = {'process': '999, /app/llama-server', 'cgroup': '0::/docker/other'}
            stack.enter_context(patch.object(launch, 'preflight', return_value={}))
            stack.enter_context(patch.object(launch.guard, 'verify_tree', return_value=None))
            stack.enter_context(patch.object(launch.os, 'open', return_value=10))
            stack.enter_context(patch.object(launch.os, 'close'))
            stack.enter_context(patch.object(launch.fcntl, 'flock'))
            stack.enter_context(patch.object(launch.time, 'sleep'))
            stack.enter_context(patch.object(launch, 'command', return_value=['fixture']))
            stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=process))
            stack.enter_context(patch.object(launch, 'resources', return_value=('999, /app/llama-server', 32*1024**3)))
            stack.enter_context(patch.object(launch, 'live_competitors', return_value=[other]))
            stack.enter_context(patch.object(launch, 'owned_container',
                                             side_effect=[('a'*64, True), ('a'*64, True), None]))
            stack.enter_context(patch.object(launch.subprocess, 'run',
                                             return_value=SimpleNamespace(returncode=0, stdout='PID\n111\n', stderr='')))
            binding = {'mode': 'diagnostic', 'max_seconds': 120, 'source_freeze_sha256': 'a'*64}
            # The run fails later only because this fixture writes no native state.
            with self.assertRaisesRegex(RuntimeError, 'did not complete'):
                launch.run_owned(binding, {'source': root/'source', 'output': root})
            report = json.loads((root/'jobs/standing.json').read_text())
        self.assertEqual(report['other_gpu_processes_seen'], [other])
        self.assertTrue(report['cleanup_checked'])

    def run_main(self, root, argv, run_owned=None):
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv', ['launch', '--bindings', str(root/'binding.json'),
                                                                  '--bindings-sha256', 'a'*64, *argv]))
            stack.enter_context(patch.object(launch.guard, 'pinned_file'))
            stack.enter_context(patch.object(launch.guard, 'read', return_value={'command_args': [], 'mode': 'train'}))
            stack.enter_context(patch.object(launch, 'verify', return_value={'output': root/'attempt'}))
            stack.enter_context(patch.object(launch, 'lock_files'))
            stack.enter_context(patch.object(launch, 'require_wandb_key'))
            stack.enter_context(patch.object(launch, 'preflight', return_value={}))
            stack.enter_context(patch.object(launch.signal, 'signal'))
            stack.enter_context(patch.object(launch.guard, 'both_locks'))
            stack.enter_context(patch.object(launch.guard, 'cleanup_owned', return_value={'reservation_released': False}))
            stack.enter_context(patch.object(launch, 'run_owned', side_effect=run_owned))
            stack.enter_context(patch.object(launch, 'host_snapshot', side_effect=RuntimeError('nvidia-smi failed')))
            stack.enter_context(patch('builtins.print'))
            launch.main()

    def test_failed_host_snapshot_still_saves_the_cleanup_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'run failed'):
                self.run_main(root, [], run_owned=RuntimeError('run failed'))
            cleanup = json.loads((root/'attempt/cleanup.json').read_text())
            self.assertEqual(cleanup['host'], {'error': repr(RuntimeError('nvidia-smi failed'))})
            self.assertFalse(cleanup['reservation_released'])
            (root/'attempt/cleanup.json').unlink()
            self.run_main(root, ['--cleanup-only'])
            receipt = json.loads((root/'attempt/cleanup.json').read_text())
            self.assertEqual(receipt['host'], {'error': repr(RuntimeError('nvidia-smi failed'))})

    def test_missing_container_and_inspection_failure_are_distinct(self):
        with patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stderr='No such object: owned', stdout='')):
            self.assertIsNone(launch.owned_container('owned'))
        with patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stderr='daemon unavailable', stdout='')):
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                launch.owned_container('owned')

    def test_reused_name_cannot_replace_owned_identity(self):
        with patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stderr='', stdout='different /owned true')):
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                launch.owned_container('owned', 'original')

    def test_contact_truncation_invalidates_successful_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'native.log'
            path.write_text('Warning: incomplete contact data maxContactDataCount = 4096\n')
            result = launch.audit_contact_log(path)
            self.assertFalse(result['passed'])
            self.assertEqual(result['reported_capacities'], [4096])


if __name__ == '__main__':
    unittest.main()
