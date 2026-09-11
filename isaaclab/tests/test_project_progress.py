"""Progress cannot promote evidence, invent ready work, or drift between outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('progress_site', ROOT / 'tools/project_site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.project = json.loads((ROOT / 'site/project.json').read_text())

    def marker(self, identifier):
        return next(m for m in self.project['milestones'] if m['id'] == identifier)

    def test_current_progress_metrics_match_actual_preserved_payload(self):
        site.validate_progress(self.project)
        metric = self.project['progress']['facts'][0]['metric']
        metric['value'] = metric['total']
        with self.assertRaisesRegex(ValueError, 'differs from source'):
            site.validate_progress(self.project)

    def test_accepted_marker_requires_human_and_scope(self):
        self.marker('walking')['acceptance'] = None
        with self.assertRaisesRegex(ValueError, 'human acceptance'):
            site.validate_progress(self.project)

    def test_implementation_cannot_be_marked_ready_without_defined_step(self):
        self.marker('stage2')['status'] = 'ready'
        with self.assertRaisesRegex(ValueError, 'defined next step'):
            site.validate_progress(self.project)

    def test_unknown_requirements_and_dependencies_reject(self):
        for key, value, message in [('requirements', ['R-99'], 'architecture'),
                                    ('depends_on', ['missing'], 'dependency')]:
            with self.subTest(key=key):
                project = copy.deepcopy(self.project)
                project['milestones'][1][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    site.validate_progress(project)

    def test_cycles_reject_even_when_no_work_is_ready(self):
        for m in self.project['milestones']:
            m['status'] = 'needs_definition'
        self.marker('walking')['depends_on'] = ['mission']
        with self.assertRaisesRegex(ValueError, 'cycle'):
            site.validate_progress(self.project)

    def test_ready_step_cannot_bypass_blocked_dependencies(self):
        m = self.marker('mission')
        m['status'] = 'ready'
        m['next_step'] = dict(outcome='fixture', owner='owner', reviewer='reviewer',
                              issue='https://example.com/issue', acceptance='recorded output', baseline='fixture')
        with self.assertRaisesRegex(ValueError, 'unmet dependencies'):
            site.validate_progress(self.project)

    def test_attempt_status_is_distinct_from_capability_state(self):
        self.project['progress']['facts'][0]['result'] = 'accepted'
        with self.assertRaisesRegex(ValueError, 'attempt result'):
            site.validate_progress(self.project)

    def test_generated_status_is_deterministic_and_does_not_read_status(self):
        before = site.status_text(self.project)
        with patch.object(Path, 'read_text', side_effect=AssertionError('STATUS cannot be an input')):
            self.assertEqual(before, site.status_text(self.project))
        self.assertIn('Next step: **not defined**', before)
        self.assertIn('10/32', before)
        self.assertIn('Historical', before)

    def test_committed_status_matches_registry(self):
        self.assertEqual((ROOT / 'STATUS.md').read_text(), site.status_text(self.project))

    def test_unknown_fields_in_source_are_errors_not_zeroes(self):
        self.project['progress']['facts'][0]['metric']['field'] = ['unavailable']
        with self.assertRaisesRegex(ValueError, 'Missing metric field'):
            site.validate_progress(self.project)

    def test_nonfinite_metric_is_rejected(self):
        self.project['progress']['facts'][0]['metric']['value'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            site.validate_progress(self.project)


if __name__ == '__main__':
    unittest.main()
