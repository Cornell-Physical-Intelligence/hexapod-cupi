"""Canonical sharing control without prose-triggered interruption; no processes."""
import ast
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_run_mkii_fourbar_supervisor as existing

host = existing.supervisor
SOURCE = Path(host.__file__)


class CoordinationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.note = self.root/'coordination.md'
        override = patch.object(host, 'COORDINATION', self.note)
        override.start()
        self.addCleanup(override.stop)

    def write(self, text):
        self.note.write_text(text)

    def test_none_has_separate_semantic_token_and_raw_note_evidence(self):
        self.write('HEXAPOD_SHARE_STATUS=NONE\nOriginal status\n')
        first = host.read_coordination_control()
        self.assertEqual(first['semantic_token'], 'HEXAPOD_SHARE_STATUS=NONE')
        self.assertEqual(first['status'], 'NONE')
        self.assertEqual(first['raw_note_sha256'], hashlib.sha256(self.note.read_bytes()).hexdigest())
        self.assertEqual(host.coordination_snapshot(), first['raw_note_sha256'])
        self.write('Historical SHARING REQUESTED discussion\nHEXAPOD_SHARE_STATUS=NONE\nProgress updated\n')
        second = host.read_coordination_control()
        self.assertEqual(second['semantic_token'], first['semantic_token'])
        self.assertNotEqual(second['raw_note_sha256'], first['raw_note_sha256'])
        host.require_coordination_none(second)

    def test_initial_requested_missing_duplicate_and_malformed_controls_block(self):
        malformed = ['', 'HEXAPOD_SHARE_STATUS=REQUESTED\n',
            'HEXAPOD_SHARE_STATUS=NONE\nHEXAPOD_SHARE_STATUS=NONE\n',
            'HEXAPOD_SHARE_STATUS=NONE\nHEXAPOD_SHARE_STATUS=REQUESTED\n',
            'HEXAPOD_SHARE_STATUS=NONE\nHEXAPOD_SHARE_STATUS=UNKNOWN\n',
            ' HEXAPOD_SHARE_STATUS=NONE\n', 'HEXAPOD_SHARE_STATUS=NONE \n',
            'HEXAPOD_SHARE_STATUS =NONE\n', 'HEXAPOD_SHARE_STATUS=none\n',
            'HEXAPOD_SHARE_STATUS=NONE # comment\n',
            'HEXAPOD_SHARE_STATUS=NONE\nhexapod_share_status=REQUESTED\n']
        for text in malformed:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(host.Blocked): host.coordination_snapshot()
        self.note.write_bytes(b'HEXAPOD_SHARE_STATUS=NONE\n\xff')
        self.assertEqual(host.read_coordination_control()['status'], 'INVALID')
        with self.assertRaises(host.Blocked): host.coordination_snapshot()
        self.note.unlink()
        self.assertIsNone(host.read_coordination_control()['raw_note_sha256'])
        with self.assertRaises(host.Blocked): host.coordination_snapshot()

    def test_active_prose_changes_do_not_pause_any_mode(self):
        for mode in ('validate', 'diagnose', 'train'):
            report = {}
            for index in range(3):
                self.write(f'HEXAPOD_SHARE_STATUS=NONE\nOrdinary status append {index}; SHARING REQUESTED was historical.\n')
                self.assertIsNone(host.poll_coordination(report, self.root, mode, None, 100.+index))
                self.assertEqual(report['coordination_latest']['status'], 'NONE')
                self.assertNotIn('pause_requested', report)
                self.assertFalse((self.root/'stop_requested').exists())

    def test_training_request_keeps_120_second_checkpoint_grace_and_latches(self):
        self.write('HEXAPOD_SHARE_STATUS=REQUESTED\nWeather asks for a handoff.\n')
        report = {}
        deadline = host.poll_coordination(report, self.root, 'train', None, 100.)
        self.assertEqual(deadline, 220.)
        self.assertEqual(report['checkpoint_pause_grace_seconds'], 120)
        self.assertEqual(report['coordination_pause_trigger']['semantic_token'], 'HEXAPOD_SHARE_STATUS=REQUESTED')
        marker = (self.root/'stop_requested').read_text()
        self.assertIn('checkpoint and pause', marker)
        self.write('HEXAPOD_SHARE_STATUS=NONE\nRequest consumed for a later run.\n')
        self.assertEqual(host.poll_coordination(report, self.root, 'train', deadline, 219.), 220.)
        self.assertEqual((self.root/'stop_requested').read_text(), marker)
        self.assertEqual(report['coordination_pause_trigger']['status'], 'REQUESTED')
        with self.assertRaisesRegex(host.Blocked, 'yielded compute'):
            host.poll_coordination(report, self.root, 'train', deadline, 220.)

    def test_validation_yields_immediately_and_invalid_active_control_fails_closed(self):
        for status in ('HEXAPOD_SHARE_STATUS=REQUESTED\n', 'HEXAPOD_SHARE_STATUS=UNKNOWN\n', ''):
            for mode in ('validate', 'diagnose'):
                self.write(status)
                report = {}
                with self.assertRaisesRegex(host.Blocked, 'yielded compute'):
                    host.poll_coordination(report, self.root, mode, None, 10.)
                self.assertIn('coordination_pause_trigger', report)
            self.write(status)
            report = {}
            self.assertEqual(host.poll_coordination(report, self.root, 'train', None, 10.), 130.)
            with self.assertRaises(host.Blocked):
                host.poll_coordination(report, self.root, 'train', 130., 130.)

    def main_nodes(self):
        tree = ast.parse(SOURCE.read_text())
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')

    def test_actual_initial_guard_records_raw_hash_and_semantic_token_before_blocking(self):
        main = self.main_nodes()
        block = next(node for node in ast.walk(main) if isinstance(node, ast.Try)
                     and node.body and ast.unparse(node.body[0]).startswith('shared_control = read_coordination_control()'))
        selected = block.body[:5]
        self.assertEqual(ast.unparse(selected[-1]), 'require_coordination_none(shared_control)')
        code = compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec')
        for state in ('NONE', 'REQUESTED'):
            self.write(f'HEXAPOD_SHARE_STATUS={state}\nordinary notes\n')
            namespace = dict(vars(host), report={})
            if state == 'REQUESTED':
                with self.assertRaises(host.Blocked): exec(code, namespace)
            else: exec(code, namespace)
            report = namespace['report']
            self.assertEqual(report['coordination_sha256'], hashlib.sha256(self.note.read_bytes()).hexdigest())
            self.assertEqual(report['coordination_initial']['semantic_token'], f'HEXAPOD_SHARE_STATUS={state}')
            self.assertEqual(report['coordination_protocol'], 'canonical_share_status_v2')

    def test_actual_pre_barrier_reread_blocks_new_request_before_gpu_launch(self):
        main = self.main_nodes()
        block = next(node for node in ast.walk(main) if isinstance(node, ast.Try)
                     and any(isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == "report['coordination_before_barrier']"
                             for item in node.body))
        index = next(i for i, item in enumerate(block.body) if isinstance(item, ast.Assign)
                     and ast.unparse(item.targets[0]) == "report['coordination_before_barrier']")
        code = compile(ast.Module(body=block.body[index:index+3], type_ignores=[]), str(SOURCE), 'exec')
        for text in ('HEXAPOD_SHARE_STATUS=REQUESTED\n', 'HEXAPOD_SHARE_STATUS=BAD\n'):
            self.write(text)
            with self.assertRaises(host.Blocked):
                exec(code, dict(vars(host), report={}, output=self.root))
            self.assertFalse((self.root/'admitted').exists())
        self.write('HEXAPOD_SHARE_STATUS=NONE\nHarmless updated note\n')
        exec(code, dict(vars(host), report={}, output=self.root))
        self.assertTrue((self.root/'admitted').exists())

    def test_actual_active_poll_preserves_resource_checks_during_training_grace(self):
        main = self.main_nodes()
        branch = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == 'time.monotonic() >= next_gate')
        code = compile(ast.Module(body=branch.body, type_ignores=[]), str(SOURCE), 'exec')
        gate, save = Mock(return_value={'gpu_pids': [123]}), Mock()
        current = {'id': 'owned-container-only'}
        namespace = dict(vars(host), report={}, output=self.root, pause_deadline=None,
            args=SimpleNamespace(mode='train'), current=current, resource_gate=gate, atomic_json=save,
            time=SimpleNamespace(monotonic=lambda: 50.))
        self.write('HEXAPOD_SHARE_STATUS=NONE\nProgress\n')
        exec(code, namespace)
        gate.assert_called_once_with(owned_container=current, allow_owned_gpu=True)
        self.assertIsNone(namespace['pause_deadline'])
        self.write('HEXAPOD_SHARE_STATUS=REQUESTED\n')
        exec(code, namespace)
        self.assertEqual(namespace['pause_deadline'], 170.)
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(save.call_count, 2)
        self.assertEqual(namespace['next_gate'], 55.)


if __name__ == '__main__':
    unittest.main()
