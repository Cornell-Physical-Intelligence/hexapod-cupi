"""CPU-only tests for bounded decisions, exact ownership and checkpoint chains."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuation as c


class Decisions(unittest.TestCase):
    def test_only_measured_eta_over_deadline_inside_checkpoint_margin_pauses(self):
        h = [{"completed": 10, "elapsed": 260.}, {"completed": 20, "elapsed": 520.}]
        self.assertFalse(c.pause_decision([], 1000, 200)["request_pause"])
        self.assertFalse(c.pause_decision(h[:1], 1000, 200)["request_pause"])
        self.assertFalse(c.pause_decision(h, 1000, 1000)["request_pause"])
        result = c.pause_decision(h, 1000, 350)
        self.assertTrue(result["request_pause"])
        self.assertEqual(result["checkpoint_margin_seconds"], 352)
        self.assertFalse(c.pause_decision(h, 21, 350)["request_pause"])
        h[-1] = {"completed": 1000, "elapsed": 26000.}
        self.assertFalse(c.pause_decision(h, 1000, 50)["request_pause"])

    def test_strict_progress_and_nonregressing_timing(self):
        good = {"iterations_completed": 5, "last_iteration": 7, "elapsed_seconds": 100.,
                "collect_seconds": 19., "learn_seconds": 1.}
        sample = c.progress_sample(good, start_iteration=3, requested=1000)
        for field, value in (("iterations_completed", True), ("iterations_completed", 1001),
                ("last_iteration", 6), ("elapsed_seconds", float("nan")), ("learn_seconds", float("inf"))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                c.progress_sample({**good, field: value}, start_iteration=3, requested=1000)
        self.assertEqual(c.update_timing([sample], sample), [sample])
        for changed in ({"completed": 4, "elapsed": 105.}, {"completed": 5, "elapsed": 105.},
                        {"completed": 6, "elapsed": 99.}):
            with self.assertRaises(ValueError): c.update_timing([sample], changed)

    def test_chunk_count_reserves_inference_capture_cleanup_and_total_time(self):
        count = c.choose_chunk(300, 26., 20000)
        self.assertEqual(count, 300)
        self.assertLess(c.choose_chunk(1000, 26., 64800), 1000)
        self.assertLess(c.choose_chunk(1000, 26., 10000), c.choose_chunk(1000, 26., 64800))
        for remaining, rate, seconds in ((1000, 26., 2500), (0, 26., 20000), (1000, float('inf'), 20000)):
            with self.assertRaises(ValueError): c.choose_chunk(remaining, rate, seconds)

    def test_job_command_keeps_frozen_host_limits_and_nonblocking_job_lock(self):
        spawn, expected = c.segment_argv({"source_dir": "/source"}, "/out", "/admission", "/checkpoint", 123)
        self.assertEqual(spawn, c.FLOCK + expected)
        self.assertEqual(expected[:3], ['/usr/bin/python3', '/source/isaaclab/deploy/run-mkii-fourbar', 'train'])
        self.assertEqual(expected[expected.index('--iterations') + 1], '123')
        self.assertEqual(expected[expected.index('--timeout-seconds') + 1], '21600')
        self.assertEqual(expected[expected.index('--source-commit') + 1], c.SOURCE_COMMIT)
        with self.assertRaises(ValueError): c.segment_argv({'source_dir': '/source'}, '/o', '/a', '/c', 1001)

    def test_chain_excludes_scratch_and_rejects_gap_overlap_overshoot_or_fourth_segment(self):
        rows = [{'start_iteration': 3, 'completed': 400, 'next_iteration': 403},
                {'start_iteration': 403, 'completed': 600, 'next_iteration': 1003}]
        self.assertEqual(c.chain_totals(rows, 3), 1000)
        for changed in ([{**rows[0], 'start_iteration': 0}, rows[1]],
                        [rows[0], {**rows[1], 'start_iteration': 402}],
                        [rows[0], {**rows[1], 'completed': 601, 'next_iteration': 1004}], rows * 2):
            with self.assertRaises(ValueError): c.chain_totals(changed, 3)


class Evidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'segment' / 'hexapod-fourbar-train-test'; self.output.mkdir(parents=True)
        self.report_path = self.output / 'report.json'
        self.checkpoint = self.output / 'checkpoint.pt'; self.checkpoint.write_bytes(b'actual fixture checkpoint')
        self.contract = {'sha256': c.SOURCE_SHA, 'task_id': 'test_task'}
        self.parent = {'next_iteration': 403, 'checkpoint_sha256': 'a'*64,
                       'policy_after_sha256': 'b'*64, 'algorithm_after_sha256': 'c'*64}
        self.admission = {'runtime_manifest': {'actual': 'same physical runtime'}}
        self.report = {'task_id': 'test_task', 'contract': self.contract, 'mode': 'provisional_physical_fourbar_ppo',
            'pass': True, 'errors': [], 'hardware_admission': False, 'paused': False,
            'num_envs': 512, 'iterations_requested': 600, 'iterations_completed': 600,
            'start_iteration': 403, 'next_iteration': 1003, 'checkpoint_verified': True,
            'checkpoint_roundtrip_pass': True, 'policy_changed': True,
            'checkpoint_sha256': c.sha(self.checkpoint), 'resumed_from_sha256': 'a'*64,
            'policy_before_sha256': 'b'*64, 'algorithm_before_sha256': 'c'*64,
            'policy_after_sha256': 'd'*64, 'algorithm_after_sha256': 'e'*64,
            'runtime_manifest': self.admission['runtime_manifest'], 'inference_probe': {'finite': True, 'steps': 100}}
        self.supervisor = {'source': str(self.source), 'source_commit': c.SOURCE_COMMIT,
            'supervisor_exit_code': 0, 'execution': 'finished', 'validator_report_status': 'passed',
            'cleanup': 'removed_exact_id', 'source_identity_unchanged_at_finish': True, 'contract': self.contract,
            'owner': 'f'*32, 'container_id': 'a'*64, 'container_name': 'hexapod-fourbar-train-test'}
        self.sidecar = {'next_iteration': 1003, 'checkpoint_sha256': c.sha(self.checkpoint)}
        self.common = SimpleNamespace(TASK_ID='test_task', read_json=lambda p: json.loads(Path(p).read_text()),
            canonical=lambda value: json.dumps(value, sort_keys=True),
            source_api=lambda source: SimpleNamespace(require_checkpoint=self.require_checkpoint))
        self.host = SimpleNamespace(validate_written_report=Mock(), require_checkpoint=self.require_checkpoint)
        self.write()

    def require_checkpoint(self, path, contract):
        if c.sha(path) != self.sidecar['checkpoint_sha256'] or contract != self.contract:
            raise ValueError('Checkpoint bytes mismatch')
        return dict(self.sidecar)

    def write(self):
        self.report_path.write_text(json.dumps(self.report))
        self.report_path.with_name('supervisor.json').write_text(json.dumps(self.supervisor))
        Path(str(self.checkpoint)+'.json').write_text(json.dumps(self.sidecar))

    def verify(self, requested=600):
        self.write()
        return c.verify_segment(self.report_path, requested=requested, parent=self.parent, contract=self.contract,
            admission=self.admission, source=self.source, common=self.common, host=self.host)

    def test_complete_segment_retains_requested_completed_and_all_state_handoffs(self):
        value = self.verify()
        self.assertEqual((value['requested'], value['completed'], value['start_iteration'], value['next_iteration']),
                         (600, 600, 403, 1003))
        self.assertEqual(value['checkpoint_sha256'], self.sidecar['checkpoint_sha256'])
        self.assertEqual(value['algorithm_after_sha256'], 'e'*64)
        self.host.validate_written_report.assert_called_once()

    def test_paused_segment_honestly_keeps_600_requested_and_200_completed(self):
        self.report.update(paused=True, iterations_completed=200, next_iteration=603)
        self.report.pop('inference_probe'); self.sidecar['next_iteration'] = 603
        value = self.verify()
        self.assertEqual((value['requested'], value['completed'], value['paused']), (600, 200, True))

    def test_physics_failure_missing_cleanup_or_source_error_never_continues(self):
        original = dict(self.supervisor)
        for update in ({'cleanup': 'FAILED'}, {'source_identity_unchanged_at_finish': False},
                       {'validator_report_status': 'rejected'}, {'supervisor_exit_code': 1},
                       {'source': '/foreign'}, {'source_commit': 'f'*40}):
            self.supervisor = {**original, **update}
            with self.subTest(update=update), self.assertRaises(ValueError): self.verify()
        self.supervisor = original
        self.report.update(pass_=False)
        self.host.validate_written_report.side_effect = ValueError('physical guard rejected')
        with self.assertRaisesRegex(ValueError, 'physical guard'): self.verify()

    def test_policy_optimizer_checkpoint_iteration_mismatches_rejected(self):
        original = copy.deepcopy(self.report)
        for update in ({'algorithm_before_sha256': 'f'*64}, {'policy_before_sha256': 'f'*64},
            {'resumed_from_sha256': 'f'*64}, {'start_iteration': 402}, {'next_iteration': 1004},
            {'checkpoint_sha256': 'f'*64}, {'iterations_completed': 601}, {'iterations_requested': True},
            {'num_envs': 512.0}, {'inference_probe': {'steps': 99, 'finite': True}}, {'paused': True, 'errors': ['failure']}):
            self.report = {**original, **update}
            with self.subTest(update=update), self.assertRaises(ValueError): self.verify()

    def test_pause_file_requires_fresh_exact_ownership_and_cannot_be_reused(self):
        checker = Mock(side_effect=ValueError('foreign PID'))
        with self.assertRaisesRegex(ValueError, 'foreign PID'):
            c.request_checkpoint(self.output, {'request_pause': True}, 'owner', verify_owned=checker)
        self.assertFalse((self.output/'stop_requested').exists())
        with self.assertRaisesRegex(ValueError, 'No measured'):
            c.request_checkpoint(self.output, {'request_pause': False}, 'owner', verify_owned=Mock())
        checker = Mock()
        record = c.request_checkpoint(self.output, {'request_pause': True}, 'owner', verify_owned=checker)
        self.assertEqual(record['sha256'], c.sha(self.output/'stop_requested')); checker.assert_called_once()
        with self.assertRaises(FileExistsError):
            c.request_checkpoint(self.output, {'request_pause': True}, 'owner', verify_owned=checker)

    def test_discovery_binds_recorded_owner_and_rejects_later_container_identity_changes(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.source, obj.contract, obj.common = self.source, self.contract, self.common
        observed = {'id': 'a'*64, 'name': '/hexapod-fourbar-train-test', 'labels': {'owner': 'f'*32}}
        obj.host = SimpleNamespace(OWNER_LABEL='owner', inspect_container=lambda _: observed,
            check_identity=lambda value, name, owner: value['name']=='/'+name and value['labels']['owner']==owner)
        job = {'output_root': str(self.output.parent)}
        obj.discover_output(job)
        self.assertEqual(job['container']['owner'], 'f'*32)
        self.supervisor['owner'] = 'b'*32; self.write()
        with self.assertRaisesRegex(ValueError, 'ownership changed'): obj.discover_output(job)
        with self.assertRaisesRegex(ValueError, 'Cannot bind'):
            obj.discover_output({'output_root': str(self.output.parent)})

    def test_pause_on_final_update_is_not_captured_and_never_adds_update_1001(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.recheck_chain = Mock(return_value=1000)
        obj.launch = Mock()
        with self.assertRaisesRegex(ValueError, 'unpaused final inference'):
            obj.capture({'paused': True}, {}, {}, Path('/admission'))
        obj.launch.assert_not_called()

    def test_complete_original_never_launches_continuation_or_duplicate_capture(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.state = {'segments': [], 'launch_reserved': None}; obj.check = Mock(); obj.save = Mock()
        obj.campaign = Mock(return_value={'state': 'complete'}); obj.launch = Mock(); obj.capture = Mock()
        obj.verify_original_completion = Mock()
        self.assertEqual(obj.run(), 0)
        self.assertEqual(obj.state['state'], 'original_completed')
        obj.launch.assert_not_called(); obj.capture.assert_not_called()
        obj.verify_original_completion.assert_called_once_with({'state': 'complete'})

    def test_reserved_launch_restart_fails_closed_without_duplicate_dispatch(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.state = {'launch_reserved': {'argv': ['reserved']}}; obj.save = Mock(); obj.launch = Mock()
        self.assertEqual(obj.run(), 1)
        self.assertEqual(obj.state['state'], 'launch_uncertain'); obj.launch.assert_not_called()

    def test_rejected_supervisor_cannot_be_hidden_by_original_inner_report_pass(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.state = {'segments': [], 'launch_reserved': None}; obj.check = Mock(); obj.save = Mock()
        obj.campaign = Mock(return_value={'state': 'complete', 'pass': True})
        obj.verify_original_completion = Mock(side_effect=ValueError('supervisor rejected'))
        obj.launch = Mock(); obj.capture = Mock()
        with self.assertRaisesRegex(ValueError, 'supervisor rejected'): obj.run()
        obj.launch.assert_not_called(); obj.capture.assert_not_called()
        self.assertNotEqual(obj.state.get('state'), 'original_completed')

    def test_admission_is_pinned_once_and_change_rejected_before_acceptance(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.state = {}; obj.save = Mock()
        path = self.root/'admission.json'; path.write_text('{"pass":true}')
        expected = c.sha(path); obj.pin_admission(path, expected)
        original = copy.deepcopy(obj.state['admission'])
        path.write_text('{"pass":true,"extra":"replacement"}')
        with self.assertRaisesRegex(ValueError, 'admission bytes changed'):
            obj.pin_admission(path, expected)
        with self.assertRaisesRegex(ValueError, 'admission binding changed'):
            obj.pin_admission(path, c.sha(path))
        with self.assertRaisesRegex(ValueError, 'admission bytes changed'):
            obj.accept_segment('/unused', {}, {}, {}, path)
        self.assertEqual(obj.state['admission'], original)

    def test_total_budget_uses_same_boot_monotonic_deadline_not_wall_clock(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.state = {'boot_id': 'same', 'deadline_boottime': 100., 'next_source_check_boottime': 1000.}
        obj.interrupted = False
        obj.host = SimpleNamespace(require_coordination_none=Mock(), read_coordination_control=Mock(return_value={'status':'NONE'}))
        with patch.object(c, 'boot_id', return_value='same'), patch.object(c, 'boot_seconds', return_value=101.), \
                patch.object(c.time, 'time', return_value=-100000.):
            with self.assertRaisesRegex(ValueError, '18-hour deadline'): obj.check()
        with patch.object(c, 'boot_id', return_value='different'), patch.object(c, 'boot_seconds', return_value=0.):
            with self.assertRaisesRegex(ValueError, 'different Linux boot'): obj.check()
        with patch.object(c, 'boot_id', return_value='same'), patch.object(c, 'boot_seconds', return_value=99.), \
                patch.object(c.time, 'time', return_value=100000000.):
            obj.check()

    def test_old_follower_must_be_no_video_and_exited_without_any_capture_reservation(self):
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.binding = {'campaign_file':'/campaign.json','source_sha256':c.SOURCE_SHA,'campaign_pid':42,'campaign_start_ticks':50}
        obj.state = {'deadline_boottime': 1000.}; obj.check = Mock(); obj.save = Mock()
        obj.config = {'state_dir':str(self.root),'campaign_follower_process':{'pid':51}}
        state = {'binding':dict(obj.binding),'state':'no_video'}
        obj.common = SimpleNamespace(read_json=lambda _: state)
        obj.exact_process = Mock(return_value=True)
        with patch.object(c, 'boot_seconds', side_effect=[0.,0.,121.]), patch.object(c.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'did not reach no_video'): obj.wait_previous_follower()
        obj.exact_process.return_value=False
        with patch.object(c, 'boot_seconds', return_value=0.): obj.wait_previous_follower()
        state['capture_argv']=['reserved']
        with patch.object(c, 'boot_seconds', return_value=0.):
            with self.assertRaisesRegex(ValueError, 'never duplicate'): obj.wait_previous_follower()

    def test_modified_imported_tool_is_rejected_before_its_sentinel_executes(self):
        follower = self.root/'follower'; capture = self.root/'capture'; follower.mkdir(); capture.mkdir()
        sentinel = self.root/'executed'
        victim = follower/'follow_campaign.py'
        victim.write_text('# original reviewed file\n')
        fm = follower/'SHA256SUMS'; fm.write_text(c.sha(victim)+'  follow_campaign.py\n')
        source_manifest = self.source/'release.sha256'
        source_manifest.write_text('# fixture manifest\n')
        cm = capture/'SHA256SUMS'; cm.write_text('')
        config = {'source_dir':str(self.source),'source_sha256':c.SOURCE_SHA,'source_commit':c.SOURCE_COMMIT,
            'source_manifest_sha256':c.sha(source_manifest),'source_manifest':'release.sha256',
            'follower_tools':str(follower),'follower_manifest_sha256':c.sha(fm),
            'capture_tools':str(capture),'capture_manifest_sha256':c.sha(cm)}
        victim.write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
        with patch.object(c, 'MANIFEST_SHA', c.sha(source_manifest)):
            with self.assertRaisesRegex(ValueError, 'Bootstrap source/tool bytes changed'):
                c.Coordinator(config, self.root/'state')
        self.assertFalse(sentinel.exists())

    def test_capture_common_accepts_final_unpaused_remaining_segment_but_not_paused(self):
        path = HERE.parent/'policy_capture_tools_v2/capture_common.py'
        spec = importlib.util.spec_from_file_location('actual_capture_common_for_chain_test', path)
        common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
        runtime = {'schema':'hexapod.physical_fourbar_runtime.v1','task_id':common.TASK_ID,'model_id':'mkii_fourbar_v5',
                   'observation_dim':84,'active_motor_names':[f'motor_{i}' for i in range(18)],
                   'tree_joint_names':[f'joint_{i}' for i in range(30)]}
        report = {**self.report,'task_id':common.TASK_ID,'runtime_manifest':runtime}
        self.assertEqual(common.validate_training(report, {'runtime_manifest':runtime}, self.sidecar, self.contract), runtime)
        with self.assertRaisesRegex(ValueError, 'paused run'):
            common.validate_training({**report,'paused':True}, {'runtime_manifest':runtime}, self.sidecar, self.contract)

    def coordinator_lifecycle(self, *, restart=None):
        """Use real run/segment/attestation logic; substitute process/GPU effects."""
        folder = self.root/'coordinator'; folder.mkdir()
        admission_path = self.root/'admission.json'; admission_path.write_text(json.dumps(self.admission))
        scratch = {**self.parent, 'next_iteration':3}
        original_path = self.root/'original'/'report.json'; original_path.parent.mkdir()
        original_checkpoint = original_path.with_name('checkpoint.pt'); original_checkpoint.write_bytes(b'original partial')
        original = {**self.report, 'iterations_requested':1000, 'iterations_completed':400,
            'start_iteration':3, 'next_iteration':403, 'paused':True,
            'checkpoint_sha256':c.sha(original_checkpoint)}
        original.pop('inference_probe')
        original_path.write_text(json.dumps(original))
        original_path.with_name('supervisor.json').write_text(json.dumps(self.supervisor))
        Path(str(original_checkpoint)+'.json').write_text(json.dumps({'checkpoint_sha256':c.sha(original_checkpoint),'next_iteration':403}))
        self.report.update(resumed_from_sha256=c.sha(original_checkpoint), policy_before_sha256='d'*64,
                           algorithm_before_sha256='e'*64, policy_after_sha256='f'*64, algorithm_after_sha256='1'*64)
        self.write()
        pause = original_path.parent/'stop_requested'; pause.write_text('owned pause request')
        original_job = {'requested':1000,'origin':'original_campaign','history':[{'completed':10,'elapsed':260.},
            {'completed':400,'elapsed':10400.}], 'pause_request':{'path':str(pause),'sha256':c.sha(pause)},
            'process':{'pid':42},'output':str(original_path.parent)}
        train_job = {'requested':600,'origin':'coordinator','kind':'train','history':[{'completed':10,'elapsed':260.},
            {'completed':600,'elapsed':15600.}], 'pause_request':None,'process':{'pid':43},'output':str(self.output)}
        captured = folder/'capture_outputs'/'hexapod-policy-capture-test'
        def make_capture_output():
            captured.mkdir(parents=True,exist_ok=True)
            (captured/'supervisor.json').write_text('{"pass":true}')
            (captured/'policy.mp4').write_bytes(b'fake video')
        spec = importlib.util.spec_from_file_location('actual_follower_for_chain_test',
            HERE.parent/'campaign_capture_followup_v2/follow_campaign.py')
        follower = importlib.util.module_from_spec(spec); spec.loader.exec_module(follower)
        calls = []
        obj = c.Coordinator.__new__(c.Coordinator)
        obj.folder, obj.source, obj.contract = folder, self.source, self.contract
        obj.config = {'campaign':str(self.root/'campaign.json'),'source_dir':str(self.source),
                      'capture_tools':str(self.root/'capture_tools')}
        obj.binding = {'source':str(self.source),'source_sha256':c.SOURCE_SHA}
        obj.state = {'segments':[],'active_job':None,'completed_full_updates':0,'launch_reserved':None,
            'deadline_boottime':100000.,'launch_history':[],'admission':{'path':str(admission_path),'sha256':c.sha(admission_path)}}
        obj.common = SimpleNamespace(**vars(self.common))
        obj.common.verify_inputs = lambda *args: {'contract':self.contract,'input_sha256':{'checkpoint':c.sha(args[1])}}
        obj.host = SimpleNamespace(validate_written_report=Mock(),require_checkpoint=lambda path, contract:
            json.loads(Path(str(path)+'.json').read_text()))
        obj.follower = SimpleNamespace(durable_json=follower.durable_json,verify_campaign_process=Mock(),
            process_identity=Mock(),verify_completed_capture=lambda *a: calls.append('verify_capture') or captured/'supervisor.json')
        obj.config['campaign_pid']=42
        obj.check=Mock(); obj.save=Mock(); obj.verify_source=Mock(); obj.child=None
        obj.exact_process=Mock(return_value=False)
        obj.wait_previous_follower=lambda: calls.append('old_follower_no_video_exited')
        obj.original_parent_and_admission=Mock(return_value=(scratch,self.admission,admission_path))
        paused={'state':'paused','phases':[{'name':'full','state':'paused','report_sha256':c.sha(original_path)}]}
        obj.campaign=Mock(side_effect=[{'state':'running'},paused,paused,paused] if restart is None else None)
        if restart is not None:obj.campaign.return_value=paused
        obj.capture_original_job=Mock(return_value=original_job)
        def launch(spawn, expected, *, kind, output_root, requested=None):
            calls.append(('launch',kind,requested))
            if kind=='capture':
                make_capture_output()
                job={'process':{'pid':44,'start_ticks':0},'origin':'coordinator','kind':'capture'}
            else:
                job=copy.deepcopy(train_job)
                self.assertEqual(requested,600)
                self.assertEqual(spawn,c.FLOCK+expected)
            obj.state['active_job']=job
            return job
        obj.launch=Mock(side_effect=launch)
        obj.monitor_job=Mock(side_effect=[original_path,self.report_path])
        if restart in ('train','capture'):
            record=c.verify_segment(original_path,requested=1000,parent=scratch,contract=self.contract,
                admission=self.admission,source=self.source,common=obj.common,host=obj.host)
            record['owned_job']=original_job;obj.state['segments']=[record];obj.state['completed_full_updates']=400
            obj.state['active_job']=copy.deepcopy(train_job)
            obj.monitor_job=Mock(return_value=self.report_path)
        if restart=='capture':
            parent=json.loads(original_path.read_text())
            record=c.verify_segment(self.report_path,requested=600,parent=parent,contract=self.contract,
                admission=self.admission,source=self.source,common=obj.common,host=obj.host)
            record['owned_job']=train_job;obj.state['segments'].append(record);obj.state['completed_full_updates']=1000
            make_capture_output()
            obj.state['active_job']={'process':{'pid':44,'start_ticks':0},'origin':'coordinator','kind':'capture','deadline_boottime':100000.}
        return obj,calls

    def test_full_mocked_pause_resume_chain_then_exactly_one_capture(self):
        obj,calls=self.coordinator_lifecycle()
        with patch.object(c,'boot_seconds',return_value=1000.):self.assertEqual(obj.run(),0)
        self.assertEqual(obj.state['state'],'video_complete')
        self.assertEqual([(s['requested'],s['completed']) for s in obj.state['segments']],[(1000,400),(600,600)])
        self.assertEqual(obj.state['completed_full_updates'],1000)
        self.assertEqual([v for v in calls if isinstance(v,tuple)],[('launch','train',600),('launch','capture',None)])
        self.assertLess(calls.index('old_follower_no_video_exited'),calls.index(('launch','train',600)))
        attestation=json.loads((obj.folder/'chain_attestation.json').read_text())
        self.assertEqual((attestation['full_updates_completed'],attestation['scratch_updates_excluded'],attestation['final_next_iteration']),(1000,3,1003))

    def test_restart_bound_training_job_does_not_launch_another_training_process(self):
        obj,calls=self.coordinator_lifecycle(restart='train')
        with patch.object(c,'boot_seconds',return_value=1000.):self.assertEqual(obj.run(),0)
        self.assertEqual([v for v in calls if isinstance(v,tuple)],[('launch','capture',None)])
        self.assertEqual(obj.state['state'],'video_complete')

    def test_restart_bound_capture_job_does_not_launch_second_capture(self):
        obj,calls=self.coordinator_lifecycle(restart='capture')
        with patch.object(c,'boot_seconds',return_value=1000.):self.assertEqual(obj.run(),0)
        obj.launch.assert_not_called()
        self.assertEqual(calls.count('verify_capture'),1)

    def test_spawned_pid_is_durable_before_post_exec_identity_confirmation(self):
        obj=c.Coordinator.__new__(c.Coordinator)
        obj.folder=self.root;obj.state={'segments':[],'launch_reserved':None,'launch_history':[]}
        obj.verify_source=Mock();obj.check=Mock();saved=[]
        obj.save=lambda:saved.append(copy.deepcopy(obj.state))
        command=['/usr/bin/python3','/reviewed_host']
        process={'pid':123,'start_ticks':999,'argv':command,'boot_id':'same'}
        obj.follower=SimpleNamespace(process_identity=Mock(return_value=process))
        child=SimpleNamespace(pid=123,poll=lambda:None)
        with patch.object(c.subprocess,'Popen',return_value=child):
            obj.launch(c.FLOCK+command,command,kind='train',output_root=self.root/'fresh',requested=600)
        self.assertNotIn('spawned_pid',saved[0]['launch_reserved'])
        self.assertEqual(saved[1]['launch_reserved']['spawned_pid'],123)
        self.assertEqual(saved[2]['active_job']['process'],process)


if __name__ == '__main__':
    unittest.main()
