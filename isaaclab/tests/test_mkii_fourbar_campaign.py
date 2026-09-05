"""No-GPU campaign sequencing, resource waiting, pause and resume lineage checks."""
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader('_fourbar_campaign_tests', str(ROOT/'isaaclab/deploy/run-mkii-fourbar-campaign'))
spec = importlib.util.spec_from_loader(loader.name, loader)
campaign = importlib.util.module_from_spec(spec)
loader.exec_module(campaign)
CONTRACT = {'task_id': 'Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0', 'sha256': 'fixture'}


def report_fixture(directory, phase, parent=None):
    directory.mkdir(parents=True, exist_ok=True)
    report = {'pass': True, 'errors': [], 'contract': CONTRACT, 'task_id': CONTRACT['task_id'],
              'num_envs': phase['num_envs'], 'mode': phase['mode']}
    if phase['mode'] == 'validate':
        report.update(steps_requested=phase['steps'], steps_completed=phase['steps'],
            solver_multiplier=phase['solver_multiplier'], solver_iterations=[32*phase['solver_multiplier'], 4*phase['solver_multiplier']],
            driven_steps=2400 if phase['steps'] >= 1000 else 0, driven_coordinate_pass=True,
            windows={window: {'mean_height_m': .138, 'max_applied_nm': 1.2} for window in ('settled', 'driven')})
    else:
        checkpoint = directory/'checkpoint.pt'
        checkpoint.write_bytes(phase['name'].encode())
        start = parent['next_iteration'] if parent else 0
        report.update(iterations_requested=phase['iterations'], iterations_completed=phase['iterations'],
            paused=False, checkpoint_verified=True, checkpoint_roundtrip_pass=True,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            next_iteration=start+phase['iterations'], start_iteration=start,
            resumed_from_sha256=parent['checkpoint_sha256'] if parent else None,
            policy_before_sha256=parent['policy_after_sha256'] if parent else 'a'*64,
            algorithm_before_sha256=parent['algorithm_after_sha256'] if parent else 'b'*64,
            policy_after_sha256='c'*64, algorithm_after_sha256='d'*64, policy_changed=True,
            inference_probe={'steps': 100, 'finite': True})
        checkpoint.with_suffix('.pt.json').write_text(json.dumps({key: report[key] for key in ('contract', 'checkpoint_sha256', 'next_iteration')}))
    path = directory/'report.json'
    path.write_text(json.dumps(report))
    (directory/'supervisor.json').write_text(json.dumps({'supervisor_exit_code': 0, 'execution': 'finished',
        'validator_report_status': 'passed', 'contract': CONTRACT, 'source_identity_unchanged_at_finish': True,
        'cleanup': 'removed_exact_id'}))
    return path, report


class CampaignTests(unittest.TestCase):
    def test_resource_busy_waits_then_proceeds_but_inspection_error_is_not_retried(self):
        recorded, sleeps = [], []
        with patch.object(campaign, 'resource_ready'):
            gate = unittest.mock.Mock(side_effect=[campaign.host.Blocked('Unrelated GPU producer processes remain: PIDs [10]'), {'idle': True}])
            campaign.wait_for_resources(30, recorded.append, gate=gate, clock=lambda: 0, sleep=sleeps.append)
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(sleeps, [15.])
        self.assertEqual(recorded[-1]['state'], 'resources_ready')
        broken = unittest.mock.Mock(side_effect=campaign.host.Blocked('Cannot establish GPU utilization'))
        with self.assertRaises(campaign.host.Blocked):
            campaign.wait_for_resources(30, recorded.append, gate=broken, clock=lambda: 0, sleep=sleeps.append)
        self.assertEqual(broken.call_count, 1)

    def test_resource_wait_is_bounded_without_launching_any_phase(self):
        current = [0.]
        def sleep(seconds):
            current[0] += seconds
        gate = unittest.mock.Mock(side_effect=campaign.host.Blocked('GPU utilization is nonzero before admission'))
        with self.assertRaisesRegex(campaign.CampaignFailed, 'wait expired'):
            campaign.wait_for_resources(22, lambda value: None, gate=gate, clock=lambda: current[0], sleep=sleep)
        self.assertEqual(current[0], 22.)
        self.assertEqual(gate.call_count, 3)

    def test_reused_probe_needs_successful_supervisor_and_exact_current_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = report_fixture(Path(directory), campaign.phases()[0])
            campaign.verify_phase(path, campaign.phases()[0], CONTRACT)
            with self.assertRaises((campaign.CampaignFailed, campaign.host.Blocked)):
                campaign.verify_phase(path, campaign.phases()[0], dict(CONTRACT, sha256='stale'))
            supervision = json.loads(path.with_name('supervisor.json').read_text())
            for changed in ({'supervisor_exit_code': 1}, {'cleanup': 'FAILED'}, {'source_identity_unchanged_at_finish': False}):
                path.with_name('supervisor.json').write_text(json.dumps(dict(supervision, **changed)))
                with self.assertRaises(campaign.CampaignFailed):
                    campaign.verify_phase(path, campaign.phases()[0], CONTRACT)

    def test_exact_separate_process_policy_and_optimizer_continuity_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent = report_fixture(root/'scratch', campaign.phases()[3])
            path, report = report_fixture(root/'full', campaign.phases()[4], parent)
            campaign.verify_phase(path, campaign.phases()[4], CONTRACT, parent=parent)
            for changed in ({'resumed_from_sha256': 'wrong'}, {'start_iteration': 0},
                            {'policy_before_sha256': 'wrong'}, {'algorithm_before_sha256': 'wrong'}):
                path.write_text(json.dumps(dict(report, **changed)))
                with self.assertRaises(campaign.CampaignFailed):
                    campaign.verify_phase(path, campaign.phases()[4], CONTRACT, parent=parent)

    def test_paused_scratch_stops_without_automatic_resume_and_inference_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = report_fixture(Path(directory), campaign.phases()[3])
            path.write_text(json.dumps(dict(report, paused=True)))
            with self.assertRaises(campaign.CampaignPaused):
                campaign.verify_phase(path, campaign.phases()[3], CONTRACT)
            path.write_text(json.dumps(dict(report, inference_probe={'steps': 99, 'finite': True})))
            with self.assertRaises(campaign.CampaignFailed):
                campaign.verify_phase(path, campaign.phases()[3], CONTRACT)

    def run_fixture_campaign(self, directory, *, fail=None, reuse=False, pause=False):
        root = Path(directory)
        args = SimpleNamespace(source_dir=root/'source', source_commit='a'*40, probe_report=None,
                               wait_seconds=0, phase_timeout_seconds=1800, full_timeout_seconds=7200)
        state = {'phases': [], 'pass': False, 'host_entrypoint_sha256': {}}
        if reuse:
            args.probe_report, _ = report_fixture(root/'existing-probe', campaign.phases()[0])
        calls, results = [], {}
        def launch(argv, log, row):
            phase = row['requested']
            calls.append(phase['name'])
            if phase['name'] == fail:
                return 1
            path, result = report_fixture(log.parent/('hexapod-fourbar-'+phase['name']), phase, results.get('scratch') if phase['name'] == 'full' else None)
            if phase['name'] == 'scratch' and pause:
                result['paused'] = True
                path.write_text(json.dumps(result))
            results[phase['name']] = result
            if phase['name'] == 'full':
                self.assertEqual(argv[argv.index('--checkpoint')+1], str(root/'scratch/hexapod-fourbar-scratch/checkpoint.pt'))
            return 0
        with patch.object(campaign, 'wait_for_resources') as wait, patch.object(campaign, 'require_campaign_source'):
            try:
                campaign.run_campaign(args, root, CONTRACT, state, lambda: None, launch=launch)
            except Exception as exc:
                return state, calls, wait.call_count, exc
        return state, calls, wait.call_count, None

    def test_campaign_runs_exact_sequence_and_full_job_resumes_verified_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            state, calls, waits, error = self.run_fixture_campaign(directory)
            self.assertIsNone(error)
            self.assertEqual(calls, ['probe', 'nominal', 'refined', 'scratch', 'full'])
            self.assertEqual(waits, 1)
            self.assertTrue(state['pass'])
            self.assertTrue(state['separate_process_resume_verified'])
            self.assertEqual(state['phases'][-1]['requested']['num_envs'], 512)
            self.assertEqual(state['phases'][-1]['requested']['iterations'], 1000)

    def test_no_retry_or_following_phase_after_first_physical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            state, calls, waits, error = self.run_fixture_campaign(directory, fail='nominal')
            self.assertIsInstance(error, campaign.CampaignFailed)
            self.assertEqual(calls, ['probe', 'nominal'])
            self.assertEqual(waits, 1)
            self.assertFalse(state['pass'])

    def test_probe_reuse_skips_only_probe_and_paused_scratch_stops_full_job(self):
        with tempfile.TemporaryDirectory() as directory:
            state, calls, _, error = self.run_fixture_campaign(directory, reuse=True, pause=True)
            self.assertIsInstance(error, campaign.CampaignPaused)
            self.assertEqual(calls, ['nominal', 'refined', 'scratch'])
            self.assertEqual(state['phases'][0]['state'], 'reused_verified')
            self.assertEqual(state['phases'][-1]['state'], 'paused')


if __name__ == '__main__':
    unittest.main()
