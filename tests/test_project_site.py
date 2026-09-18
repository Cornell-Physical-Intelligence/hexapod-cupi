"""Protect the user-mandated public research update and evidence contract."""
import importlib.util, json, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'tools/project_site.py'
spec = importlib.util.spec_from_file_location('project_site', PATH)
site = importlib.util.module_from_spec(spec); spec.loader.exec_module(site)

class PosterContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.patches = [patch.object(site, 'ROOT', self.root), patch.object(site, 'SITE', self.root/'site')]
        for p in self.patches:p.start()
        (self.root/'site/updates').mkdir(parents=True)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def run_git(self,*args):
        return subprocess.check_output(['git',*args],cwd=self.root,text=True,stderr=subprocess.DEVNULL).strip()
    def init(self):
        self.run_git('init');self.run_git('config','user.email','test@example.invalid');self.run_git('config','user.name','test')
        (self.root/'code.py').write_text('first\n');self.run_git('add','.');self.run_git('commit','-m','base')
        return self.run_git('rev-parse','HEAD')
    def row(self, **extra):
        return dict(id='20260910T160000_test',date='2026-09-10T16:00:00Z',title='Test',summary='The implementation changes.',areas=['locomotion'],changes=['code.py'],evidence=[],next='Run the actual comparison.',no_project_impact=False,**extra)
    def save(self,row):
        (self.root/'site/updates'/f"{row['id']}.json").write_text(json.dumps(row))
    def test_missing_update_rejects_code_change(self):
        base=self.init();self.save(self.row());self.run_git('add','.');self.run_git('commit','-m','old update');base=self.run_git('rev-parse','HEAD')
        (self.root/'code.py').write_text('changed\n')
        with patch.object(site,'registry',return_value=({}, {}, {})):
            with self.assertRaisesRegex(ValueError,'lack a new central update'):site.check(base)
    def test_new_bounded_record_covers_change(self):
        base=self.init();(self.root/'code.py').write_text('changed\n');self.save(self.row())
        with patch.object(site,'registry',return_value=({}, {}, {})):site.check(base)
    def test_changed_paths_preserve_unicode_and_newlines(self):
        self.init()
        name = 'café mesh\nscan.stl'
        (self.root/name).write_text('mesh')
        self.run_git('add', name)
        self.run_git('commit', '-m', 'CAD filename fixture')
        base = self.run_git('rev-parse', 'HEAD')
        (self.root/name).unlink()
        (self.root/'新 mesh.stl').write_text('new mesh')
        self.assertEqual(site.changed_paths(base), [name, '新 mesh.stl'])
    def test_old_record_deletion_rejected(self):
        self.init();r=self.row();self.save(r);self.run_git('add','.');self.run_git('commit','-m','record');base=self.run_git('rev-parse','HEAD')
        (self.root/'site/updates'/f"{r['id']}.json").unlink();r['id']='20260910T170000_next';self.save(r)
        with patch.object(site,'registry',return_value=({}, {}, {})):
            with self.assertRaisesRegex(ValueError,'append-only'):site.check(base)
    def test_no_impact_needs_reason(self):
        r=self.row();r['no_project_impact']=True;self.save(r)
        with self.assertRaisesRegex(ValueError,'specific no-impact'):site.records()
    def test_references_cannot_escape_or_embed_credentials(self):
        for value in ['../secret','/tmp/secret','https://user:password@example.com/file','javascript:alert(1)']:
            with self.subTest(value=value),self.assertRaises(ValueError):site.reference(value)
    def test_case_insensitive_host_cannot_admit_a_linux_missing_reference(self):
        directory = self.root / 'Evidence';directory.mkdir()
        receipt = directory / 'REVIEW.json';receipt.write_text('{}')
        self.assertEqual(site.reference('Evidence/REVIEW.json'),receipt)
        # Simulate the successful exists() lookup that macOS permits; actual
        # directory entries retain their spelling on both macOS and Linux.
        with patch.object(Path,'exists',return_value=True):
            for value in ['Evidence/review.json','evidence/REVIEW.json']:
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'capitalization'):
                    site.reference(value)
    def test_catchall_does_not_waive_changes(self):
        r=self.row();r['changes']=['**'];self.save(r)
        with self.assertRaisesRegex(ValueError,'bounded'):site.records()
    def checkpoint_fixture(self):
        path=self.root/'policy.pt';path.write_bytes(b'actual checkpoint fixture')
        digest=site.sha(path)
        project={'checkpoint':{'path':'policy.pt','video_matches':True},'primary_video_id':'latest',
                 'media':[{'id':'latest','type':'video','controller':digest}]}
        return path,project,{'final_checkpoint_sha256':digest}
    def test_changed_checkpoint_bytes_are_rejected(self):
        path,project,receipt=self.checkpoint_fixture()
        site.checkpoint_media(project,receipt)
        path.write_bytes(b'a different checkpoint')
        with self.assertRaisesRegex(ValueError,'bytes differ'):site.checkpoint_media(project,receipt)
    def test_older_video_cannot_claim_latest_weights(self):
        path,project,receipt=self.checkpoint_fixture()
        project['media'][0]['controller']='0'*64
        with self.assertRaisesRegex(ValueError,'Matching video'):site.checkpoint_media(project,receipt)
        project['checkpoint']['video_matches']=False
        site.checkpoint_media(project,receipt)

class EvidenceCorrectionTests(unittest.TestCase):
    setUp = PosterContractTests.setUp
    tearDown = PosterContractTests.tearDown
    row = PosterContractTests.row
    save = PosterContractTests.save

    def fixture(self):
        receipt = self.root / 'Evidence/REVIEW.json'
        receipt.parent.mkdir()
        receipt.write_text('{"result":"failed"}\n')
        old = self.row()
        old['evidence'] = ['Evidence/review.json']
        self.save(old)
        old_path = self.root / 'site/updates' / (old['id'] + '.json')
        correction = self.row()
        correction.update(id='20260910T170000_correction', date='2026-09-10T17:00:00Z',
                          evidence=['Evidence/REVIEW.json'], evidence_reference_corrections=[{
                              'old_record_id': old['id'], 'old_record_sha256': site.sha(old_path),
                              'old_reference': 'Evidence/review.json',
                              'replacement_reference': 'Evidence/REVIEW.json',
                              'replacement_sha256': site.sha(receipt)}])
        self.save(correction)
        return old, old_path, receipt, correction

    def test_exact_pins_resolve_only_presentation_and_preserve_provenance(self):
        old, old_path, receipt, correction = self.fixture()
        before = (old_path.read_bytes(), receipt.read_bytes())
        rows = site.records()
        resolved = next(r for r in rows if r['id'] == old['id'])
        self.assertEqual(resolved['evidence'], ['Evidence/REVIEW.json'])
        self.assertEqual(resolved['original_evidence'], old['evidence'])
        self.assertEqual(resolved['original_record'], str(old_path.relative_to(self.root)))
        self.assertEqual(resolved['original_record_sha256'], site.sha(old_path))
        self.assertEqual(resolved['evidence_corrections_applied'][0]['correction_record_id'], correction['id'])
        resolved['evidence'][0] = 'presentation-only'
        self.assertEqual(site.records()[0]['evidence'], ['Evidence/REVIEW.json'])
        self.assertEqual(before, (old_path.read_bytes(), receipt.read_bytes()))

    def test_without_correction_bad_reference_still_fails(self):
        _, _, _, correction = self.fixture()
        del correction['evidence_reference_corrections']
        self.save(correction)
        with self.assertRaises(ValueError):
            site.records()

    def test_new_record_cannot_borrow_an_old_reference_correction(self):
        self.fixture()
        new = self.row()
        new.update(id='20260910T180000_new', date='2026-09-10T18:00:00Z',
                   evidence=['Evidence/review.json'])
        self.save(new)
        with self.assertRaises(ValueError):
            site.records()

    def test_registry_reference_does_not_use_record_corrections(self):
        self.fixture()
        site.records()
        (self.root / 'site/project.json').write_text(json.dumps({
            'schema_version': 2, 'summary_tags': ['Test'],
            'status_source': 'Evidence/review.json'}))
        with self.assertRaises(ValueError):
            site.registry()

    def test_changed_old_record_and_replacement_bytes_reject(self):
        old, old_path, receipt, _ = self.fixture()
        old_bytes = old_path.read_bytes()
        old['summary'] = 'An edited historical conclusion.'
        self.save(old)
        with self.assertRaisesRegex(ValueError, 'old record SHA'):
            site.records()
        old_path.write_bytes(old_bytes)
        receipt.write_text('{"result":"passed"}\n')
        with self.assertRaisesRegex(ValueError, 'replacement file or SHA'):
            site.records()

    def test_malformed_unknown_stale_or_unmatched_correction_rejects(self):
        _, _, _, correction = self.fixture()
        cases = [
            ('old_record_id', 'nonexistent'),
            ('old_record_sha256', '0' * 64),
            ('old_record_sha256', 'malformed'),
            ('old_reference', 'Evidence/another.json'),
            ('replacement_sha256', '0' * 64),
            ('replacement_reference', 'https://example.com/evidence.json'),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                edited = json.loads(json.dumps(correction))
                edited['evidence_reference_corrections'][0][key] = value
                self.save(edited)
                with self.assertRaises(ValueError):
                    site.records()
        for declarations in [None, {}, [], [{}], [{'unexpected': 'field'}]]:
            with self.subTest(declarations=declarations):
                edited = dict(correction, evidence_reference_corrections=declarations)
                self.save(edited)
                with self.assertRaises(ValueError):
                    site.records()

    def test_replacement_requires_exact_case_and_own_canonical_evidence(self):
        _, _, _, correction = self.fixture()
        edited = json.loads(json.dumps(correction))
        edited['evidence'] = []
        self.save(edited)
        with self.assertRaisesRegex(ValueError, 'cite its distinct canonical'):
            site.records()
        edited = json.loads(json.dumps(correction))
        edited['evidence'] = ['evidence/REVIEW.json']
        edited['evidence_reference_corrections'][0]['replacement_reference'] = 'evidence/REVIEW.json'
        self.save(edited)
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'capitalization'):
                site.records()

    def test_duplicate_or_conflicting_declarations_reject(self):
        _, _, _, correction = self.fixture()
        other = self.root / 'Evidence/OTHER.json'
        other.write_text('{"other":"evidence"}\n')
        for conflict in [False, True]:
            with self.subTest(conflict=conflict):
                edited = json.loads(json.dumps(correction))
                duplicate = dict(edited['evidence_reference_corrections'][0])
                if conflict:
                    duplicate['replacement_reference'] = 'Evidence/OTHER.json'
                    duplicate['replacement_sha256'] = site.sha(other)
                    edited['evidence'].append('Evidence/OTHER.json')
                edited['evidence_reference_corrections'].append(duplicate)
                self.save(edited)
                with self.assertRaisesRegex(ValueError, 'Duplicate or conflicting'):
                    site.records()

    def test_correction_cannot_target_same_or_later_timestamp(self):
        _, _, _, correction = self.fixture()
        for when in ['2026-09-10T16:00:00Z', '2026-09-10T15:00:00Z']:
            with self.subTest(when=when):
                self.save(dict(correction, date=when))
                with self.assertRaisesRegex(ValueError, 'older update'):
                    site.records()


class ArchivedDocumentTests(unittest.TestCase):
    setUp = PosterContractTests.setUp
    tearDown = PosterContractTests.tearDown
    run_git = PosterContractTests.run_git
    init = PosterContractTests.init

    def test_git_retirement_preserves_links_and_rejects_changed_identity(self):
        for storage in ('retired_sources', 'archived_documents'):
            with self.subTest(storage=storage):
                path = self.root/'old.md'
                path.write_text('Recorded failed attempt.\n')
                digest = site.sha(path)
                commit = self.init()
                path.unlink()
                inventory = dict(baseline=commit, tools=[])
                if storage == 'retired_sources':
                    inventory[storage] = dict(source_commit=commit, files={'old.md': digest})
                else:
                    inventory[storage] = [dict(previous_path='old.md', storage='git',
                                               source_commit=commit, sha256=digest)]
                config = self.root/'configs/source_inventory.json'
                config.parent.mkdir(exist_ok=True)
                config.write_text(json.dumps(inventory))
                self.assertEqual(site.historical_source('old.md'),
                    'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/'+commit+'/old.md')
                config.write_text(config.read_text().replace(digest, '0'*64))
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    site.reference('old.md')

    def archive_fixture(self):
        destination = self.root / 'docs/archive/old.md'
        destination.parent.mkdir(parents=True)
        destination.write_text('Preserved failed attempt.\n')
        (self.root / 'configs').mkdir()
        item = dict(previous_path='HANDOFF.md', path='docs/archive/old.md',
                    source_commit='a' * 40, sha256=site.sha(destination))
        (self.root / 'configs/source_inventory.json').write_text(json.dumps(
            dict(baseline='b' * 40, tools=[], archived_documents=[item])))
        return destination

    def test_archived_reference_uses_original_revision(self):
        self.archive_fixture()
        self.assertIsNone(site.reference('HANDOFF.md'))
        self.assertEqual(site.historical_source('HANDOFF.md'),
            'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/'
            + 'a' * 40 + '/HANDOFF.md')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            site.reference('invented.md')

    def test_archive_corruption_or_missing_copy_rejects_reference(self):
        destination = self.archive_fixture()
        destination.write_text('Changed conclusion.\n')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            site.reference('HANDOFF.md')
        destination.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or unsafe'):
            site.reference('HANDOFF.md')

    def test_maintained_reference_remains_current(self):
        self.archive_fixture()
        current = self.root / 'HANDOFF.md'
        current.write_text('Current reference.\n')
        self.assertEqual(site.reference('HANDOFF.md'), current)
        self.assertIsNone(site.historical_source('HANDOFF.md'))

if __name__=='__main__':unittest.main()
