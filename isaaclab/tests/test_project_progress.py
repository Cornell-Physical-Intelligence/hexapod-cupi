"""Progress cannot promote evidence, invent ready work, or drift between outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('progress_site', ROOT / 'tools/project_site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        (root / 'ARCHITECTURE.md').write_text('R-01 fixture requirement')
        (root / 'evidence.json').write_text(json.dumps({'passed': 10, 'total': 32}))
        (root / 'evidence.md').write_text('Fixture evidence')
        patcher = patch.object(site, 'ROOT', root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = {'progress': {
            'as_of': '2026-09-11T00:00:00Z', 'source_commit': 'a'*40,
            'architecture': 'ARCHITECTURE.md', 'execution_source': 'evidence.md',
            'prior_snapshot': 'https://example.com/snapshot', 'summary': 'Fixture: 10/32.',
            'definition_policy': 'Define the next step with the team.',
            'facts': [dict(id='fixture', title='Fixture', backend='Synthetic test',
                          result='failed', text='Fixture result', source='evidence.md',
                          metric=dict(path='evidence.json', field=['passed'], value=10,
                                      total_field=['total'], total=32, units='cases', window='fixture window'))]},
            'milestones': []}
        for identifier, dependencies in [('walking', []), ('stage2', ['walking']),
                                          ('stage3', ['stage2']), ('mission', ['stage2', 'stage3'])]:
            self.project['milestones'].append(dict(
                id=identifier, title='Historical fixture' if identifier=='walking' else identifier,
                status='accepted' if identifier=='walking' else 'blocked',
                description='Fixture description', gate='Fixture proof', backend='Synthetic test',
                owner='Fixture owner', question='What next?', depends_on=dependencies,
                requirements=['R-01'], source='evidence.md', blocker='Fixture blocker',
                next_step=None, acceptance=dict(by='Fixture reviewer', scope='Fixture only',
                                               source='evidence.md') if identifier=='walking' else None))

    def marker(self, identifier):
        return next(m for m in self.project['milestones'] if m['id'] == identifier)

    def test_progress_cannot_promote_a_failed_payload(self):
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

    def test_approved_scope_is_visible_but_does_not_assign_or_accept_work(self):
        marker = self.marker('mission')
        marker['definition'] = dict(id='M1', title='Scan reload', by='Reviewer',
                                    summary='Save and independently reload a scan.',
                                    fixture='Known floor and wall.', criteria=['Preserve points exactly.'])
        site.validate_progress(self.project)
        output = site.status_text(self.project)
        self.assertIn('M1 — Scan reload', output)
        self.assertIn('Preserve points exactly.', output)
        self.assertIn('Assignment pending.', output)
        marker['status'] = 'ready'
        with self.assertRaisesRegex(ValueError, 'defined next step'):
            site.validate_progress(self.project)

    def test_approved_definition_requires_explicit_acceptance_checks(self):
        self.marker('mission')['definition'] = dict(id='M1', title='Scan reload', by='Reviewer',
                                                    summary='Reload scan.', fixture='Known scene.', criteria=[])
        with self.assertRaisesRegex(ValueError, 'acceptance criteria'):
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
        project = json.loads((ROOT / 'site/project.json').read_text())
        self.assertEqual((ROOT / 'STATUS.md').read_text(), site.status_text(project))

    def test_current_registry_validates_against_its_actual_evidence(self):
        with patch.object(site, 'ROOT', ROOT):
            site.validate_progress(json.loads((ROOT / 'site/project.json').read_text()))

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
