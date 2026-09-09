"""CPU-only tests against the exact pinned campaign, without launching phases."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('per_job_campaign_wrapper_test', HERE/'run_campaign.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class WrapperTests(unittest.TestCase):
    def test_every_phase_has_exact_per_job_prefix_and_unchanged_original_arguments(self):
        campaign, _ = wrapper.load_campaign(ROOT, wrapper.SOURCE_COMMIT)
        args = SimpleNamespace(source_dir=ROOT, source_commit=wrapper.SOURCE_COMMIT,
            asset_model='mkii_fourbar_v5', environment_layout='coincident_flat_origin_v1',
            full_timeout_seconds=21600, phase_timeout_seconds=7200)
        phases = campaign.phases(args.asset_model, args.environment_layout)
        original = campaign.phase_argv
        expected = [original(p, args, Path('/tmp/result'), admission=Path('/tmp/admission.json'),
                             checkpoint=Path('/tmp/checkpoint.pt')) for p in phases]
        restore = wrapper.install_per_job_flock(campaign)
        try:
            with patch('subprocess.Popen', side_effect=AssertionError('No process starts while composing phases')):
                actual = [campaign.phase_argv(p, args, Path('/tmp/result'), admission=Path('/tmp/admission.json'),
                          checkpoint=Path('/tmp/checkpoint.pt')) for p in phases]
            self.assertEqual(actual, [[*wrapper.FLOCK_PREFIX, *argv] for argv in expected])
            self.assertEqual([x[x.index('--timeout-seconds')+1] for x in actual], ['7200']*4+['21600'])
        finally:
            restore()
        self.assertIs(campaign.phase_argv, original)

    def test_dry_run_never_calls_campaign_or_opens_any_lock(self):
        with patch.object(wrapper, 'install_per_job_flock', side_effect=AssertionError('Dry run cannot wrap/launch')):
            self.assertEqual(wrapper.main(['--source-dir', str(ROOT), '--source-commit', wrapper.SOURCE_COMMIT,
                '--output-root', '/tmp/hexapod_dry_run_no_campaign', '--dry-run']), 0)
        self.assertFalse(Path('/tmp/hexapod_dry_run_no_campaign').exists())

    def test_source_commit_mismatch_fails_before_loading_campaign(self):
        with self.assertRaisesRegex(ValueError, 'exact pinned source commit'):
            wrapper.load_campaign(ROOT, '0'*40)


if __name__ == '__main__':
    unittest.main()
