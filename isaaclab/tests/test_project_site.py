"""Protect the user-mandated public research update and evidence contract."""
import importlib.util, json, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / 'tools/project_site.py'
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

class ArchivedDocumentTests(unittest.TestCase):
    setUp = PosterContractTests.setUp
    tearDown = PosterContractTests.tearDown

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
