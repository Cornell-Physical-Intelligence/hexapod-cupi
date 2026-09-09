import copy
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
import evaluation_common as e
import run_evaluation as host

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FROZEN=ROOT/'artifacts/mkii_fourbar_2026-09-06/policy_capture_tools_v2'

class ScheduleTests(unittest.TestCase):
    def test_exact_timing_bounds_and_signed_coverage(self):
        self.assertEqual([e.schedule(i)[0] for i in (0,49,50,199,200,299,300,449)],
            ['settle','settle','motion','motion','stop','stop','reverse','reverse'])
        self.assertEqual(len(e.CASES),12)
        allcmd=np.stack([e.schedule(i)[1] for i in range(450)])
        self.assertTrue((np.abs(allcmd)<=np.array([.150001,.150001,.300001])).all())
        np.testing.assert_array_equal(allcmd[50],-allcmd[300]);self.assertTrue((allcmd[:50]==0).all())
        self.assertTrue((allcmd[200:300]==0).all())
        for i in (-1,450,True):
            with self.assertRaises(ValueError):e.schedule(i)

    def test_frame_transform(self):
        np.testing.assert_array_equal(e.body_to_navigation(np.array([[.2,-.1,.3]])),[[.1,.2,.3]])

    def test_injected_command_reaches_fresh_observation_and_reset_callback(self):
        class Raw:
            def __init__(self):
                self._commands=torch.zeros(12,3);self._command_time_left_s=torch.zeros(12)
                self.cfg=SimpleNamespace(episode_length_s=20.);self.random_calls=0
            def _sample_commands(self,ids):self.random_calls+=1;self._commands[ids]=999
        raw=Raw()
        class Wrapped:
            def get_observations(self):
                obs=torch.zeros(12,84);obs[:,9:12]=raw._commands;return {'policy':obs}
        with e.ScheduledCommands(raw) as override:
            for index in (49,50,200,300):
                phase,obs=override.prepare(index,Wrapped())
                np.testing.assert_array_equal(obs['policy'][:,9:12],e.schedule(index)[1])
                raw._commands[3]=999;raw._sample_commands(torch.tensor([3]))
                np.testing.assert_array_equal(raw._commands,e.schedule(index)[1])
            self.assertEqual(raw.random_calls,0);self.assertEqual(override.calls,4)
        self.assertNotIn('_sample_commands',vars(raw))
        raw._sample_commands(torch.tensor([3]));self.assertEqual(raw.random_calls,1)
        with self.assertRaisesRegex(ValueError,'consumed policy observation'):
            e.assert_command_observation(np.ones((12,3)),np.zeros((12,84)))

    def test_pre_reset_end_state_hook_preserves_original_done_result(self):
        class Raw:
            position=5
            def _get_dones(self):return self.result
        raw=Raw();raw.result=(torch.tensor([True]),torch.tensor([False]))
        with e.EndStateCapture(raw,lambda:{'position':raw.position}) as capture:
            returned=raw._get_dones();raw.position=0
            self.assertIs(returned,raw.result);self.assertEqual(capture.latest,{'position':5})
            self.assertEqual(capture.calls,1)
        self.assertNotIn('_get_dones',vars(raw))

class MetricsTests(unittest.TestCase):
    def fixture(self):
        command=np.stack([e.schedule(i)[1] for i in range(450)])
        obs=np.zeros((450,12,84));obs[:,:,9:12]=command
        pos=np.zeros((450,12,3));pos[:,:,2]=.14
        result={'command_navigation':command,'observation_policy':obs,'end_com_velocity_navigation':command.copy(),
            'done':np.zeros((450,12),dtype=bool),'pre_plate_pos_w_m':pos,'end_plate_pos_w_m':pos.copy(),
            'end_foot_pad':np.ones((450,12,6),dtype=bool),'end_foot_slip_speed_mps':np.zeros((450,12,6))}
        for name in ('end_applied_motor_torque_nm','end_raw_motor_demand_nm','end_motor_clipping_nm','end_burst_headroom','end_joint_acc_rad_s2'):
            result[name]=np.zeros((450,12,18))
        return result
    def test_reset_transition_excluded_but_counted_without_erasing_raw_state(self):
        states=self.fixture();states['done'][60,1]=True
        states['end_com_velocity_navigation'][60,1]=[100,100,100];states['end_plate_pos_w_m'][60,1]=[100,100,100]
        result=e.summarize(states);case=result['cases'][1]
        self.assertEqual(case['reset_transition_indices'],[60]);phase=case['phases']['motion']
        self.assertEqual(phase['transitions_used'],149);np.testing.assert_allclose(phase['tracking_rmse'],0)
        self.assertEqual(phase['plate_planar_path_length_m'],0);self.assertIsNone(result['policy_skill_pass'])
        self.assertEqual(states['end_plate_pos_w_m'][60,1,0],100)
    def test_stationary_policy_not_masked_by_reward(self):
        states=self.fixture();states['end_com_velocity_navigation'][:]=0
        result=e.summarize(states);phase=result['cases'][1]['phases']['motion']
        self.assertAlmostEqual(phase['tracking_mae'][0],.15,places=6);self.assertEqual(phase['near_stationary_fraction'],1.)
        yaw=result['cases'][9]['phases']['motion'];self.assertAlmostEqual(yaw['tracking_mae'][2],.30,places=6)
    def test_stale_observation_and_changed_schedule_fail(self):
        states=self.fixture();states['observation_policy'][50,1,9]=0
        with self.assertRaisesRegex(ValueError,'stale'):e.summarize(states)
        states=self.fixture();states['command_navigation'][50,1,0]=.10
        with self.assertRaisesRegex(ValueError,'schedule'):e.summarize(states)

class IntegrityTests(unittest.TestCase):
    def test_frozen_helper_and_source_guard_are_reused(self):
        helper=e.helpers(FROZEN)
        self.assertTrue(callable(helper.require_same_inputs));self.assertTrue(callable(helper.validate_training))
        wrong=SimpleNamespace(verify_inputs=lambda *args:{'contract':{'sha256':'bad'}})
        with self.assertRaisesRegex(ValueError,'frozen 1600'):e.verify_inputs(wrong,ROOT,None,None,None)
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'capture_common.py').write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError,'hash differs'):e.helpers(d)

    def test_changed_source_is_rejected_before_real_source_api_executes(self):
        helper=e.helpers(FROZEN)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';script=source/'tools/mkii_training_contract.py'
            script.parent.mkdir(parents=True);script.write_text('# reviewed CPU source fixture\n')
            manifest=source/e.SOURCE_MANIFEST;manifest.parent.mkdir(parents=True)
            manifest.write_text(e.digest(script)+'  tools/mkii_training_contract.py\n')
            reviewed_manifest=e.digest(manifest);sentinel=root/'unreviewed_source_executed'
            script.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("executed")\n')
            with patch.object(e,'SOURCE_MANIFEST_SHA256',reviewed_manifest), \
                 patch.object(helper,'source_api',wraps=helper.source_api) as source_api:
                with self.assertRaisesRegex(ValueError,'bootstrap bytes differ'):
                    e.verify_inputs(helper,source,None,None,None)
                source_api.assert_not_called()
            self.assertFalse(sentinel.exists())

    def test_rewritten_manifest_cannot_rebase_source_before_import(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d);script=source/'tools/mkii_training_contract.py'
            script.parent.mkdir();script.write_text('# reviewed fixture\n')
            manifest=source/e.SOURCE_MANIFEST;manifest.parent.mkdir(parents=True)
            manifest.write_text(e.digest(script)+'  tools/mkii_training_contract.py\n')
            reviewed_manifest=e.digest(manifest)
            script.write_text('raise RuntimeError("unreviewed")\n')
            manifest.write_text(e.digest(script)+'  tools/mkii_training_contract.py\n')
            common=SimpleNamespace(verify_inputs=Mock(side_effect=AssertionError('Source cannot execute')))
            with patch.object(e,'SOURCE_MANIFEST_SHA256',reviewed_manifest):
                with self.assertRaisesRegex(ValueError,'bootstrap manifest differs'):
                    e.verify_inputs(common,source,None,None,None)
            common.verify_inputs.assert_not_called()

    def test_manifest_member_cannot_escape_frozen_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();outside=root/'outside.py'
            outside.write_text('# outside the reviewed source\n')
            link=source/'escaped.py';link.symlink_to(outside)
            manifest=source/e.SOURCE_MANIFEST;manifest.parent.mkdir(parents=True)
            manifest.write_text(e.digest(outside)+'  escaped.py\n')
            common=SimpleNamespace(verify_inputs=Mock(side_effect=AssertionError('Source cannot execute')))
            with patch.object(e,'SOURCE_MANIFEST_SHA256',e.digest(manifest)):
                with self.assertRaisesRegex(ValueError,'bootstrap bytes differ'):
                    e.verify_inputs(common,source,None,None,None)
            common.verify_inputs.assert_not_called()
    def test_request_and_report_guards(self):
        verified={'contract':{'sha256':e.SOURCE_SHA256},'input_sha256':{'checkpoint':'a'*64},'runtime_manifest':{'schema':'CPU fixture'}}
        tools=e.tool_identity(HERE)
        request={'schema':e.SCHEMA,**verified,'evaluation_tools_sha256':tools,
            'frozen_capture_helper_sha256':e.CAPTURE_HELPER_SHA256,'schedule':e.schedule_descriptor()}
        e.require_request(request,verified,tools)
        wrong=copy.deepcopy(request);wrong['schedule']['controls']=449
        with self.assertRaises(ValueError):e.require_request(wrong,verified,tools)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);artifacts={}
            for name in ('states.npz','metadata.json','metrics.json'):
                (out/name).write_bytes(b'CPU report integrity fixture, not policy data')
                artifacts[name]={'sha256':e.digest(out/name),'bytes':(out/name).stat().st_size}
            report={'schema':e.SCHEMA,'pass':True,'errors':[],'controls_completed':450,**verified,
                'evaluation_tools_sha256':tools,'policy_skill_pass':None,'hardware_admission':False,
                'navigation_or_terrain_qualification':False,'physics_coverage_pass':True,'policy_state_unchanged':True,
                'checkpoint_unchanged':True,'artifacts':artifacts,'physics_substeps':14400,'num_envs':12,'policy_dt_s':.02,
                'schedule':e.schedule_descriptor(),'admitted_runtime_comparison':{'pass':True}}
            (out/'report.json').write_text(json.dumps(report));e.validate_report(out/'report.json',verified,tools)
            for key,value in (('controls_completed',449),('physics_substeps',7200),('admitted_runtime_comparison',{'pass':False}),('policy_skill_pass',True),('checkpoint_unchanged',False)):
                wrong=dict(report,**{key:value});(out/'report.json').write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):e.validate_report(out/'report.json',verified,tools)
            (out/'report.json').write_text(json.dumps(report));(out/'states.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):e.validate_report(out/'report.json',verified,tools)
    def test_host_dry_run_does_not_open_locks_or_start_processes(self):
        methods=('read_coordination_control','require_coordination_none','resource_gate','inspect_container',
                 'check_identity','require_unchanged_source','snapshot_source')
        guard=SimpleNamespace(**{name:lambda *a,**k:None for name in methods},OWNER_LABEL='owner',TELEMETRY_ARGUMENT='')
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);paths=[]
            for name in ('checkpoint.pt','admission.json','training.json'):
                path=base/name;path.write_text('CPU dry-run fixture');paths.append(path)
            verified={'contract':{'sha256':e.SOURCE_SHA256},'input_sha256':{},
                'runtime_manifest':{'usd_path_relative':'robot/test.usda'}}
            with patch.object(host,'verify_inputs',return_value=verified),patch.object(host,'load_module',return_value=guard), \
                 patch.object(host,'environment_layout',return_value='coincident_flat_origin_v1'), \
                 patch('os.open',side_effect=AssertionError('dry run cannot open locks')), \
                 patch('subprocess.Popen',side_effect=AssertionError('dry run cannot launch')),patch('sys.stdout',new_callable=io.StringIO):
                result=host.main(['--source-dir',str(ROOT),'--capture-tools-dir',str(FROZEN),'--checkpoint',str(paths[0]),
                    '--admission',str(paths[1]),'--training-report',str(paths[2]),'--output-root',str(base/'output'),'--dry-run'])
            self.assertEqual(result,0);self.assertFalse((base/'output').exists())

    def test_host_argv_is_headless_readonly_and_preserves_helper_flag(self):
        args=SimpleNamespace(checkpoint=Path('/input/checkpoint.pt'),admission=Path('/input/admission.json'),
            training_report=Path('/input/training.json'),capture_tools_dir=FROZEN,environment_layout='coincident_flat_origin_v1')
        guard=SimpleNamespace(OWNER_LABEL='hexapod.owner',TELEMETRY_ARGUMENT='--/privacy/userConsent/enableTelemetry=false')
        cmd=host.compose_argv(guard,ROOT,Path('/output'),HERE,'owned-name','owner',args,{'usd_path_relative':'robot/test.usda'})
        self.assertIn('--capture-tools-dir',cmd);self.assertNotIn('--enable_cameras',cmd)
        self.assertIn('/workspace/policy_evaluation_tools/evaluate_policy.py',cmd)
        mounts=[cmd[i+1] for i,value in enumerate(cmd) if value=='-v']
        self.assertTrue(all(item.endswith(':ro') for item in mounts if not item.startswith('/output:')))
        self.assertEqual(cmd[-4:],['--viz','none','--device','cuda:0'])

class SupervisorTests(unittest.TestCase):
    def run_host(self,primary_pass=True,yield_requested=False,log_error=None,log_open_error=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lab=root/'lab';(lab/'docker').mkdir(parents=True)
            for name in ('.env.base','docker-compose.yaml'):(lab/'docker'/name).write_text('CPU fixture only')
            paths=[]
            for name in ('checkpoint.pt','admission.json','training.json'):
                path=root/name;path.write_text('CPU fixture only');paths.append(path)
            (root/'checkpoint.pt.json').write_text('CPU fixture only')
            helper=e.helpers(FROZEN)
            verified={'contract':{'sha256':e.SOURCE_SHA256},'input_sha256':{'checkpoint':'a'*64},
                'runtime_manifest':{'usd_path_relative':'robot/test.usda'}}
            guard=SimpleNamespace(OWNER_LABEL='cpu.owner',LAB=lab,LOCK=root/'hex.lock',TELEMETRY_ARGUMENT='',Blocked=RuntimeError)
            calls={'coordination':0,'inspect':0};events=[];current={}
            def coordination():
                calls['coordination']+=1
                return {'status':'REQUESTED' if yield_requested and calls['coordination']>=3 else 'NONE'}
            def require_none(value):
                if value['status']!='NONE':raise RuntimeError('Sharing requested')
            guard.read_coordination_control=coordination;guard.require_coordination_none=require_none
            guard.require_unchanged_source=lambda *a:True;guard.snapshot_source=lambda *a:{'files':0}
            guard.resource_gate=lambda **k:{}
            def inspect(identifier):
                calls['inspect']+=1
                return dict(current,running=calls['inspect']<=2,exit_code=0) if current else None
            guard.inspect_container=inspect
            guard.check_identity=lambda obj,name,owner:bool(obj and obj['name']=='/'+name and obj['labels'].get('cpu.owner')==owner)
            def command(argv,**kw):
                events.append(tuple(argv[:2]));stdout=''
                if argv[:2]==['docker','compose']:
                    name=argv[argv.index('--name')+1];owner=argv[argv.index('--label')+1].split('=',1)[1]
                    current.update(id='c'*64,name='/'+name,labels={'cpu.owner':owner})
                    output=root/'output'/name
                    self.assertFalse((output/'admitted').exists())
                    report={'schema':e.SCHEMA,'pass':primary_pass,'errors':[] if primary_pass else ['CPU failure fixture'],
                        'controls_completed':450,'physics_substeps':14400,'num_envs':12,'policy_dt_s':.02,
                        'schedule':e.schedule_descriptor(),**verified,'evaluation_tools_sha256':e.tool_identity(HERE),
                        'admitted_runtime_comparison':{'pass':True},'policy_skill_pass':None,
                        'hardware_admission':False,'navigation_or_terrain_qualification':False,
                        'physics_coverage_pass':True,'policy_state_unchanged':True,'checkpoint_unchanged':True,'artifacts':{}}
                    for name in ('states.npz','metadata.json','metrics.json'):
                        (output/name).write_bytes(b'Temporary CPU integrity fixture only')
                        report['artifacts'][name]={'sha256':e.digest(output/name),'bytes':(output/name).stat().st_size}
                    helper.write_json(output/'report.json',report)
                elif argv[:2]==['docker','inspect']:stdout='no\n'
                elif argv[:2] in (['docker','stop'],['docker','rm']):self.assertEqual(argv[-1],'c'*64)
                return SimpleNamespace(returncode=0,stdout=stdout,stderr='')
            guard.command=command
            def opened(path,flags,*args):
                events.append(('open',str(path),flags))
                return 11 if str(path)=='/opt/wx/gpu.lock' else 9
            original_path_open=Path.open
            def path_open(path,*args,**kwargs):
                if path.name=='container.log' and log_open_error is not None:
                    raise log_open_error
                return original_path_open(path,*args,**kwargs)
            def retrieve_logs(argv,**kwargs):
                events.append(('docker','logs'))
                self.assertEqual(argv,['docker','logs','--timestamps','c'*64])
                if log_error is not None:raise log_error
                return SimpleNamespace(returncode=0)
            with ExitStack() as stack:
                for context in (patch.object(host,'verify_inputs',return_value=verified),patch.object(host,'load_module',return_value=guard),
                    patch.object(host,'environment_layout',return_value='coincident_flat_origin_v1'),patch('os.open',side_effect=opened),
                    patch('os.close',side_effect=lambda fd:events.append(('close',fd))),
                    patch('fcntl.flock',side_effect=lambda fd,op:events.append(('flock',fd,op))),
                    patch('signal.signal',return_value=host.signal.SIG_DFL),patch('time.sleep'),
                    patch.object(Path,'open',path_open),patch('subprocess.run',side_effect=retrieve_logs),
                    patch('sys.stdout',new_callable=io.StringIO)):
                    stack.enter_context(context)
                code=host.main(['--source-dir',str(ROOT),'--capture-tools-dir',str(FROZEN),'--checkpoint',str(paths[0]),
                    '--admission',str(paths[1]),'--training-report',str(paths[2]),'--output-root',str(root/'output')])
            output=next((root/'output').iterdir());report=json.loads((output/'supervisor.json').read_text())
            self.assertIn(('open','/opt/wx/gpu.lock',host.os.O_RDONLY),events)
            self.assertIn(('docker','rm'),events,msg=str(report))
            self.assertLess(events.index(('docker','rm')),events.index(('close',11)))
            self.assertLess(events.index(('docker','rm')),events.index(('flock',9,host.fcntl.LOCK_UN)))
            self.assertEqual(report['cleanup'],'removed_exact_id')
            return code,report,(output/'stop_requested').exists()
    def test_completed_job_verified_and_locks_released_after_exact_cleanup(self):
        code,report,_=self.run_host();self.assertEqual(code,0);self.assertTrue(report['evaluation_artifacts_verified'])
    def test_zero_exit_does_not_mask_failed_primary_report(self):
        code,report,_=self.run_host(primary_pass=False);self.assertEqual(code,1);self.assertFalse(report['pass'])
    def test_coordination_stops_only_own_job_and_releases_both_locks(self):
        code,report,stop=self.run_host(yield_requested=True);self.assertEqual(code,1);self.assertTrue(stop)
    def test_log_command_failure_preserves_exact_cleanup_but_fails_acceptance(self):
        for error in (host.subprocess.TimeoutExpired(['docker','logs'],20),
                      host.subprocess.CalledProcessError(1,['docker','logs']),OSError('Cannot invoke Docker logs')):
            with self.subTest(error=type(error).__name__):
                code,report,_=self.run_host(log_error=error)
                self.assertEqual(code,1);self.assertFalse(report['pass'])
                self.assertFalse(report['container_log_retrieved'])
                self.assertIn(type(error).__name__,report['container_log_error'])
                self.assertNotIn('evaluation_artifacts_verified',report)
    def test_log_file_open_failure_does_not_skip_exact_cleanup(self):
        code,report,_=self.run_host(log_open_error=OSError('Log path cannot be opened'))
        self.assertEqual(code,1);self.assertFalse(report['pass'])
        self.assertFalse(report['container_log_retrieved'])
        self.assertIn('Log path cannot be opened',report['container_log_error'])

if __name__=='__main__':unittest.main()
