"""The selected placement must survive host/campaign boundaries and admission."""
import importlib.machinery
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader(
    '_layout_campaign_tests', str(ROOT/'isaaclab/deploy/run-mkii-fourbar-campaign'))
spec = importlib.util.spec_from_loader(loader.name, loader)
campaign = importlib.util.module_from_spec(spec)
loader.exec_module(campaign)
host = campaign.host


class LayoutLauncherTests(unittest.TestCase):
    def test_campaign_keeps_layout_across_validation_and_both_training_sizes(self):
        args = SimpleNamespace(source_dir=ROOT, source_commit='a'*40,
            asset_model='mkii_fourbar_v5', environment_layout='coincident_flat_origin_v1',
            phase_timeout_seconds=7200, full_timeout_seconds=21600)
        plan = campaign.phases(args.asset_model, args.environment_layout)
        self.assertEqual([p['num_envs'] for p in plan], [1, 32, 32, 64, 512])
        for phase in plan:
            argv = campaign.phase_argv(phase, args, Path('/output'),
                admission=Path('/admission.json'), checkpoint=Path('/checkpoint.pt'))
            self.assertEqual(argv[argv.index('--environment-layout')+1], args.environment_layout)
        with self.assertRaisesRegex(campaign.CampaignFailed, 'layout differs'):
            campaign.phase_argv(dict(plan[0], environment_layout='grid_2m_v1'), args, Path('/output'))

    def test_container_receives_explicit_layout_even_for_default_grid(self):
        for layout in host.ENVIRONMENT_LAYOUTS:
            args = SimpleNamespace(mode='validate', steps=1000, solver_multiplier=1,
                asset_model='mkii_fourbar_v5', environment_layout=layout, num_envs=32)
            argv = host.compose_argv(ROOT, Path('/output'), 'owned', 'fixture', args)
            matches = [item for item in argv if item.startswith('HEXAPOD_MKII_ENVIRONMENT_LAYOUT=')]
            self.assertEqual(matches, ['HEXAPOD_MKII_ENVIRONMENT_LAYOUT='+layout])
            self.assertIn(str(ROOT)+':/workspace/hexapod:ro', argv)

    def test_historical_grid_cannot_admit_coincident_layout(self):
        legacy = {'runtime_manifest': {}}
        host.require_requested_layout(legacy, 'grid_2m_v1')
        with self.assertRaises(host.Blocked):
            host.require_requested_layout(legacy, 'coincident_flat_origin_v1')
        with self.assertRaises(host.Blocked):
            host.require_requested_layout(legacy, 'unverified_layout')

    def test_diagnostic_rejects_changed_layout_before_any_gpu_operation(self):
        with self.assertRaises(SystemExit) as caught:
            host.main(['diagnose', '--source-dir', str(ROOT),
                '--environment-layout', 'coincident_flat_origin_v1', '--dry-run'])
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
