"""Progress cannot promote evidence, misplace the roadmap, or drift between outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('progress_site', ROOT / 'tools/project_site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        (root / 'evidence.json').write_text(json.dumps(
            {'passed': 10, 'probes': {str(i): {} for i in range(13)}, 'count': 13}))
        (root / 'evidence.md').write_text('Fixture evidence')
        patcher = patch.object(site, 'ROOT', root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = {
            'mission': 'Fixture mission.',
            'latest': {'date': '2026-10-11', 'title': 'Fixture result', 'text': 'Fixture text.',
                       'next': 'Fixture next step.', 'media_ids': ['clip'],
                       'metric': {'value': 10, 'total': 13, 'label': 'tests passed',
                                  'sources': [{'path': 'evidence.json', 'field': ['passed'],
                                               'total_field': ['probes']}]}},
            'roadmap': [dict(id='design', title='Design', state='done', accepted_by='Fixture reviewer', source='evidence.md'),
                        dict(id='stage2', title='Walk', state='now'),
                        dict(id='stage3', title='Terrain', state='next'),
                        dict(id='mission', title='Survey', state='later')],
            'accomplishments': [dict(date='2026-10-11', text='Newest.', source='evidence.md'),
                                dict(date='2026-09-14', text='Oldest.', source='evidence.md')],
            'nodes': [dict(id='a', title='A', text='Part A.', state='working', column=0, row=0, source='evidence.md'),
                      dict(id='b', title='B', text='Part B.', state='not_started', column=1, row=0, source='evidence.md')],
            'edges': [['a', 'b']],
            'robot': dict(media_id='clip', caption='It has 18 motors and weighs 7.47 kg.',
                          mass_kg=7.466088235225788, motors=18, source='evidence.md'),
            'media': [dict(id='clip', type='video', caption='Fixture clip.')],
            'glossary': [dict(term='Term', meaning='Meaning.')]}

    def test_fixture_is_valid(self):
        site.validate_progress(self.project)

    def test_progress_cannot_promote_a_failed_payload(self):
        self.project['latest']['metric']['value'] = 13
        with self.assertRaisesRegex(ValueError, 'differs from source'):
            site.validate_progress(self.project)

    def test_total_counts_the_named_table_or_reads_the_number(self):
        metric = self.project['latest']['metric']
        metric['total'] = 12
        with self.assertRaisesRegex(ValueError, 'differs from source'):
            site.validate_progress(self.project)
        metric['total'] = 13
        metric['sources'][0]['total_field'] = ['count']
        site.validate_progress(self.project)

    def test_every_metric_source_must_agree(self):
        (site.ROOT / 'other.json').write_text(json.dumps({'passed': 9, 'probes': list(range(13))}))
        self.project['latest']['metric']['sources'].append(
            {'path': 'other.json', 'field': ['passed'], 'total_field': ['probes']})
        with self.assertRaisesRegex(ValueError, 'differs from source'):
            site.validate_progress(self.project)

    def test_roadmap_needs_exactly_one_current_step(self):
        for states in (['done', 'next', 'next', 'later'], ['done', 'now', 'now', 'later']):
            with self.subTest(states=states):
                project = copy.deepcopy(self.project)
                for step, state in zip(project['roadmap'], states):
                    step['state'] = state
                with self.assertRaisesRegex(ValueError, 'exactly one current step'):
                    site.validate_progress(project)

    def test_completed_steps_come_before_the_current_step(self):
        for states in (['next', 'now', 'done', 'later'], ['done', 'now', 'done', 'later']):
            with self.subTest(states=states):
                project = copy.deepcopy(self.project)
                for step, state in zip(project['roadmap'], states):
                    step['state'] = state
                with self.assertRaisesRegex(ValueError, 'Completed steps'):
                    site.validate_progress(project)

    def test_done_step_requires_human_acceptance_and_evidence(self):
        for key in ('accepted_by', 'source'):
            with self.subTest(missing=key):
                project = copy.deepcopy(self.project)
                project['roadmap'][0].pop(key)
                with self.assertRaisesRegex(ValueError, 'human acceptance'):
                    site.validate_progress(project)
        self.project['roadmap'][0]['source'] = 'missing.md'
        with self.assertRaisesRegex(ValueError, 'Missing'):
            site.validate_progress(self.project)
        # A step that moves to done must carry its own acceptance.
        project = copy.deepcopy(self.project)
        project['roadmap'][0]['source'] = 'evidence.md'
        project['roadmap'][1]['state'] = 'done'
        project['roadmap'][2]['state'] = 'now'
        with self.assertRaisesRegex(ValueError, 'human acceptance'):
            site.validate_progress(project)

    def test_unknown_states_reject(self):
        self.project['roadmap'][2]['state'] = 'blocked'
        with self.assertRaisesRegex(ValueError, 'roadmap state'):
            site.validate_progress(self.project)
        self.project['roadmap'][2]['state'] = 'next'
        self.project['nodes'][0]['state'] = 'implemented'
        with self.assertRaisesRegex(ValueError, 'system state'):
            site.validate_progress(self.project)

    def test_graph_rejects_unknown_endpoints_and_shared_positions(self):
        self.project['edges'].append(['a', 'missing'])
        with self.assertRaisesRegex(ValueError, 'graph endpoint'):
            site.validate_progress(self.project)
        self.project['edges'].pop()
        self.project['nodes'][1]['column'] = 0
        with self.assertRaisesRegex(ValueError, 'share a position'):
            site.validate_progress(self.project)

    def test_accomplishments_are_newest_first(self):
        self.project['accomplishments'].reverse()
        with self.assertRaisesRegex(ValueError, 'newest first'):
            site.validate_progress(self.project)

    def test_latest_and_robot_media_must_exist(self):
        self.project['latest']['media_ids'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'known media'):
            site.validate_progress(self.project)
        self.project['latest']['media_ids'] = ['clip']
        self.project['robot']['media_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'Robot section'):
            site.validate_progress(self.project)

    def test_robot_caption_states_the_recorded_mass_and_motors(self):
        self.project['robot']['caption'] = 'It has 18 motors and weighs 7.5 kg.'
        with self.assertRaisesRegex(ValueError, 'mass and motor count'):
            site.validate_progress(self.project)

    def test_generated_status_is_deterministic_and_does_not_read_status(self):
        before = site.status_text(self.project)
        with patch.object(Path, 'read_text', side_effect=AssertionError('STATUS cannot be an input')):
            self.assertEqual(before, site.status_text(self.project))
        self.assertIn('10 of 13 tests passed', before)
        self.assertIn('- Walk: Now', before)
        self.assertIn('Fixture next step.', before)

    def test_generated_status_matches_registry(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(site, 'ROOT', ROOT), \
                patch.object(site, 'SITE', ROOT / 'site'):
            project = site.registry()
            site.write_status(Path(temp) / 'STATUS.md')
            self.assertEqual((Path(temp) / 'STATUS.md').read_text(), site.status_text(project))

    def test_current_registry_validates_against_its_actual_evidence(self):
        with patch.object(site, 'ROOT', ROOT), patch.object(site, 'SITE', ROOT / 'site'):
            site.registry()

    def test_unknown_fields_in_source_are_errors_not_zeroes(self):
        self.project['latest']['metric']['sources'][0]['field'] = ['unavailable']
        with self.assertRaisesRegex(ValueError, 'Missing metric field'):
            site.validate_progress(self.project)

    def test_nonfinite_metric_is_rejected(self):
        self.project['latest']['metric']['value'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            site.validate_progress(self.project)


if __name__ == '__main__':
    unittest.main()
