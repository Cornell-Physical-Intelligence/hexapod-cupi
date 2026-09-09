"""CPU-only capture lineage, corruption, command and supervisor regression tests."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout
import io
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_common as common
import capture_policy as host
import record_admitted_policy as recorder


class EvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        repo = HERE.parents[2]
        for relative, data in {
            "tools/mkii_training_contract.py": (repo / "tools/mkii_training_contract.py").read_bytes(),
            "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda": b"USD fixture\n",
            "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf": b"URDF fixture\n",
            "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json": b"{}\n",
        }.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        api = common.load_module(self.source / "tools/mkii_training_contract.py", "test_capture_source_"+uuid.uuid4().hex)
        self.addCleanup(sys.modules.pop, api.__name__, None)
        patcher = patch.object(common, "source_api", return_value=api)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.contract = api.identity(self.source)
        self.runtime = {"schema": "hexapod.physical_fourbar_runtime.v1", "task_id": common.TASK_ID,
            "model_id": "mkii_fourbar_v5", "observation_dim": 84,
            "active_motor_names": [f"motor_{i}" for i in range(18)],
            "tree_joint_names": [f"coordinate_{i}" for i in range(30)],
            "usd_path_relative": "robot/physical.usda", "solver_iterations": 64}
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"test checkpoint identity, not executable pickle")
        self.sidecar = {"contract": self.contract, "checkpoint_sha256": common.digest(self.checkpoint), "next_iteration": 12}
        self.admission = {"pass": True, "simulation_training_admission": True, "errors": [],
            "task_id": common.TASK_ID, "contract": self.contract, "num_envs": 32, "steps_completed": 1000,
            "runtime_manifest": self.runtime}
        self.training = {"pass": True, "errors": [], "task_id": common.TASK_ID,
            "mode": "provisional_physical_fourbar_ppo", "contract": self.contract,
            "paused": False, "hardware_admission": False, "checkpoint_verified": True,
            "checkpoint_roundtrip_pass": True, "policy_changed": True, "iterations_requested": 3,
            "iterations_completed": 3, "start_iteration": 9, "next_iteration": 12,
            "checkpoint_sha256": common.digest(self.checkpoint), "policy_after_sha256": "a"*64,
            "algorithm_after_sha256": "b"*64, "inference_probe": {"finite": True, "steps": 100},
            "seed": 42, "runtime_manifest": copy.deepcopy(self.runtime)}
        self.admission_path, self.training_path = self.root/"admission.json", self.root/"training.json"
        self.persist()

    def persist(self):
        common.write_json(str(self.checkpoint)+".json", self.sidecar)
        common.write_json(self.admission_path, self.admission)
        common.write_json(self.training_path, self.training)

    def verify(self):
        return common.verify_inputs(self.source, self.checkpoint, self.admission_path, self.training_path)

    def test_complete_matching_training_accepts(self):
        self.assertEqual(self.verify()["input_sha256"]["checkpoint"], common.digest(self.checkpoint))

    def test_checkpoint_corruption_rejected(self):
        self.checkpoint.write_bytes(self.checkpoint.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "Checkpoint bytes"):
            self.verify()

    def test_source_change_rejected(self):
        (self.source/"tools/new_runtime.py").write_text("changed=True\n")
        with self.assertRaisesRegex(ValueError, "admission"):
            self.verify()

    def test_external_capture_artifacts_do_not_change_training_identity(self):
        before = self.verify()['contract']
        folder = self.source/'artifacts/policy_capture_tools'
        folder.mkdir(parents=True)
        (folder/'record_admitted_policy.py').write_text('# separately reviewed artifact\n')
        self.assertEqual(self.verify()['contract'],before)

    def test_paused_incomplete_and_unverified_training_rejected(self):
        original = copy.deepcopy(self.training)
        for changes in ({"paused": True}, {"iterations_completed": 2}, {"next_iteration": 11},
                        {"checkpoint_roundtrip_pass": False}, {"pass": 1}, {"errors": ["failure"]},
                        {"inference_probe": {"finite": True, "steps": 99}}, {"iterations_requested": True}):
            with self.subTest(changes=changes):
                self.training = {**copy.deepcopy(original), **changes}
                self.persist()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_mock_or_scalar_type_runtime_changes_rejected(self):
        for key, value in (("model_id", "mkii_fourbar_v3"), ("solver_iterations", 64.0)):
            with self.subTest(key=key):
                self.training["runtime_manifest"] = {**self.runtime, key: value}
                self.persist()
                with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                    self.verify()

    def test_short_admission_cannot_be_used(self):
        self.admission["num_envs"] = 1
        self.persist()
        with self.assertRaisesRegex(ValueError, "Short probes"):
            self.verify()

    def test_changed_reports_rejected_at_finish(self):
        before = self.verify()
        self.training["new_information"] = "report bytes changed"
        self.persist()
        with self.assertRaisesRegex(ValueError, "changed during"):
            common.require_same_inputs(before, self.verify())

    def run_mock_supervisor(self, *, primary_pass=True, yield_requested=False):
        verified = self.verify()
        guard_path = self.source/'isaaclab/deploy/run-mkii-fourbar'
        guard_path.parent.mkdir(parents=True)
        guard_path.write_text('# frozen helper fixture\n')
        lab = self.root/'lab'
        (lab/'docker').mkdir(parents=True)
        for name in ('.env.base','docker-compose.yaml'):
            (lab/'docker'/name).write_text('fixture only\n')
        events, current = [], {}
        probe = {"streams":[{"codec_name":"h264","pix_fmt":"yuv420p","width":1280,"height":720,
                  "nb_read_frames":"750","avg_frame_rate":"50/1"}],"format":{"duration":"15.000000"}}
        guard = SimpleNamespace(OWNER_LABEL='test.owner', LAB=lab, LOCK=self.root/'gpu.lock',
            TELEMETRY_ARGUMENT='--kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"',
            Blocked=RuntimeError)
        control_calls = 0
        def coordination():
            nonlocal control_calls
            control_calls += 1
            return {'status':'REQUESTED' if yield_requested and control_calls >= 3 else 'NONE'}
        def require_none(value):
            if value['status'] != 'NONE':
                raise ValueError('sharing requested')
        guard.read_coordination_control = coordination
        guard.require_coordination_none = require_none
        guard.require_unchanged_source = lambda *a: True
        guard.snapshot_source = lambda source, manifest: {'files':0}
        guard.resource_gate = lambda **kw: events.append(('gate',bool(kw.get('owned_container')))) or {}
        inspections = 0
        def inspect(identifier):
            nonlocal inspections
            inspections += 1
            return {**current,'running': inspections <= 2, 'exit_code':0} if current else None
        guard.inspect_container = inspect
        guard.check_identity = lambda value,name,owner: bool(value and value.get('name')=='/'+name
            and value.get('labels',{}).get(guard.OWNER_LABEL)==owner)
        def command(argv, **kwargs):
            events.append(tuple(argv[:2]))
            stdout = ''
            if argv[:2] == ['docker','compose']:
                name = argv[argv.index('--name')+1]
                owner = argv[argv.index('--label')+1].split('=',1)[1]
                current.update(id='a'*64,name='/'+name,pid=123,labels={guard.OWNER_LABEL:owner})
                output = self.root/'output'/name
                self.assertFalse((output/'admitted').exists(), 'GPU command must still be behind CPU barrier')
                report = {'schema':common.SCHEMA,'pass':primary_pass,'errors':[] if primary_pass else ['failed'],
                    'contract':verified['contract'],'runtime_manifest':verified['runtime_manifest'],
                    'input_sha256':verified['input_sha256'],'capture_tools_sha256':common.tool_identity(HERE),
                    'frames_written':750,'controls_completed':750,'encoder_finalized':True,
                    'checkpoint_unchanged':True,'policy_state_unchanged':True,'physics_coverage_pass':True,'artifacts':{}}
                for artifact in ('policy.mp4','states.npz','metadata.json'):
                    (output/artifact).write_bytes(b'mocked capture output')
                    report['artifacts'][artifact]={'sha256':common.digest(output/artifact),'bytes':21}
                common.write_json(output/'report.json',report)
            elif argv[:2] == ['docker','inspect']:
                stdout = 'no\n'
            elif argv[0] == 'ffprobe' and '-count_frames' in argv:
                stdout = json.dumps(probe)
            elif argv[:2] in (['docker','rm'],['docker','stop']):
                self.assertEqual(argv[-1], 'a'*64, 'Cleanup must use exact immutable ID')
            return SimpleNamespace(returncode=0,stdout=stdout,stderr='')
        guard.command = command
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(host,'load_module',return_value=guard))
            stack.enter_context(patch.object(host.os,'open',return_value=9))
            stack.enter_context(patch.object(host.os,'close',side_effect=lambda fd:events.append(('close',fd))))
            stack.enter_context(patch.object(host.fcntl,'flock',side_effect=lambda fd,op:events.append(('flock',op))))
            stack.enter_context(patch.object(host.signal,'signal',return_value=host.signal.SIG_DFL))
            stack.enter_context(patch.object(host.time,'sleep'))
            stack.enter_context(patch.object(host.subprocess,'run',return_value=SimpleNamespace(returncode=0)))
            result = host.main(['--source-dir',str(self.source),'--checkpoint',str(self.checkpoint),
                '--admission',str(self.admission_path),'--training-report',str(self.training_path),
                '--output-root',str(self.root/'output')])
        output = next((self.root/'output').iterdir())
        self.assertLess(events.index(('docker','rm')),events.index(('flock',host.fcntl.LOCK_UN)))
        self.assertEqual(events[-1],('close',9))
        return result, common.read_json(output/'supervisor.json'), output

    def test_guarded_supervisor_decodes_then_passes_and_removes_exact_id(self):
        result, report, output = self.run_mock_supervisor()
        self.assertEqual(result,0)
        self.assertTrue(report['decoded_video_verified'])
        self.assertEqual(report['cleanup'],'removed_exact_id')
        self.assertTrue((output/'ffprobe.json').exists())

    def test_guarded_supervisor_rejects_zero_exit_failed_primary_report(self):
        result, report, _ = self.run_mock_supervisor(primary_pass=False)
        self.assertEqual(result,1)
        self.assertFalse(report['pass'])
        self.assertEqual(report['cleanup'],'removed_exact_id')

    def test_guarded_supervisor_yields_without_touching_other_jobs(self):
        result, report, output = self.run_mock_supervisor(yield_requested=True)
        self.assertEqual(result,1)
        self.assertEqual(report['cleanup'],'removed_exact_id')
        self.assertTrue((output/'stop_requested').exists())


class PureTests(unittest.TestCase):
    def test_runner_descriptor_preserves_normalization_and_distribution(self):
        # Shape observed from the installed Lab 3 config after its RSL5 adapter.
        descriptor={'actor':{'class_name':'rsl_rl.models:MLPModel','hidden_dims':[256,256,128],
            'activation':'elu','obs_normalization':True,
            'distribution_cfg':{'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.15,'std_type':'scalar'}},
            'critic':{'class_name':'rsl_rl.models:MLPModel','hidden_dims':[256,256,128],
                      'activation':'elu','obs_normalization':True,'distribution_cfg':None},'check_for_nan':True}
        cfg=object()
        calls=[]
        api=SimpleNamespace(runner_config_dict=lambda actual,version:calls.append((actual,version)) or copy.deepcopy(descriptor))
        result=recorder.make_runner_config(api,cfg)
        self.assertEqual(calls,[(cfg,'5.0.1')])
        self.assertEqual(result,descriptor)
        self.assertEqual(result['actor']['distribution_cfg']['init_std'],.15)
        self.assertTrue(result['actor']['obs_normalization'])
        self.assertIsNone(result['critic']['distribution_cfg'])

    def test_runner_rejects_legacy_source_and_obsolete_actual_descriptor_fields(self):
        with self.assertRaisesRegex(ValueError,'runner_config_dict'):
            recorder.make_runner_config(SimpleNamespace(),object())
        for name in ('actor','critic'):
            for obsolete in ('stochastic','init_noise_std','noise_std_type','state_dependent_std'):
                descriptor={'actor':{},'critic':{},'check_for_nan':True}
                descriptor[name][obsolete]=None
                api=SimpleNamespace(runner_config_dict=lambda cfg,version:descriptor)
                with self.subTest(model=name,field=obsolete),self.assertRaisesRegex(ValueError,'unsupported'):
                    recorder.make_runner_config(api,object())

    def test_layout_selection_retains_exact_admitted_descriptor(self):
        repo = HERE.parents[2]
        helper = common.load_module(repo/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/environment_layout.py',
                                    'capture_source_environment_layout')
        self.assertEqual(common.environment_layout(repo,{}),'grid_2m_v1')
        runtime = {'resolved_environment_layout':helper.layout_runtime_descriptor()}
        self.assertEqual(common.environment_layout(repo,runtime),'coincident_flat_origin_v1')
        runtime['resolved_environment_layout']['origin_world_m']=[6.,0.,0.]
        with self.assertRaises(ValueError):
            common.environment_layout(repo,runtime)

    def test_state_command_time_and_quaternion_alignment(self):
        import numpy as np
        count = 3
        widths = {'joint_pos_rad':30,'joint_vel_rad_s':30,'root_pos_w_m':3,'root_quat_w_xyzw':4,'command_navigation':3}
        states = {prefix+name:np.zeros((count,width)) for prefix in ('pre_','post_') for name,width in widths.items()}
        for prefix in ('pre_','post_'):
            states[prefix+'root_quat_w_xyzw'][:,3]=1.
        for key,width in {'observation_policy':84,'policy_action':18,'processed_target_endpoint_rad':18,
                          'post_applied_motor_torque_nm':18,'camera_eye_target_w_m':6}.items():
            states[key]=np.zeros((count,width))
        for key in ('reward','done'):
            states[key]=np.zeros(count)
        states['frame_index']=np.arange(count)
        states['simulation_time_s']=(np.arange(count)+1)*.02
        common.validate_states(states,count,.02)
        for key,index,value in [('observation_policy',(0,9),.1),('simulation_time_s',0,.04),
                                ('pre_root_quat_w_xyzw',(0,3),.5)]:
            bad = {name:array.copy() for name,array in states.items()}
            bad[key][index]=value
            with self.assertRaises(ValueError):
                common.validate_states(bad,count,.02)

    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.json"
            for value in ('{"pass":false,"pass":true}', '{"x": NaN}'):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    common.read_json(path)

    def test_dimensions_and_frames(self):
        import numpy as np
        self.assertEqual(common.validate_dimensions(15., 1280, 720), 750)
        for seconds, width, height in ((9.,1280,720),(21.,1280,720),(15.,1279,720),(float('nan'),1280,720)):
            with self.assertRaises(ValueError):
                common.validate_dimensions(seconds, width, height)
        frame = np.zeros((256,256,3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "blank"):
            common.check_frame(frame,256,256)
        frame[0,0] = 255
        self.assertIs(common.check_frame(frame,256,256),frame)
        with self.assertRaisesRegex(ValueError, "uint8"):
            common.check_frame(frame.astype(float),256,256)

    def test_safe_readonly_compose_arguments_and_exact_telemetry(self):
        guard = SimpleNamespace(OWNER_LABEL="test.owner", TELEMETRY_ARGUMENT=
            '--kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"')
        args = SimpleNamespace(checkpoint=Path('/inputs/a space$(touch NEVER).pt'), admission=Path('/inputs/admission.json'),
                               training_report=Path('/inputs/training.json'), seconds=15.,width=1280,height=720,
                               environment_layout='coincident_flat_origin_v1')
        result = host.compose_argv(guard, Path('/source with space'), Path('/output'), Path('/tools'),
            'capture-unique','owner123',args,{"usd_path_relative":"robot/physical.usda"})
        self.assertIn('/inputs/a space$(touch NEVER).pt:/workspace/capture_inputs/checkpoint.pt:ro', result)
        self.assertIn('/source with space:/workspace/hexapod:ro',result)
        self.assertIn('/tools:/workspace/policy_capture_tools:ro',result)
        self.assertIn('HEXAPOD_MKII_ENVIRONMENT_LAYOUT=coincident_flat_origin_v1',result)
        self.assertEqual([v for v in result if v.startswith('--kit_args=')],
            ['--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry'])
        self.assertEqual(sum(v.endswith(':rw') for v in result),1)
        barrier = result[result.index('-c')+1]
        self.assertIn('exec "$@"',barrier)
        self.assertNotIn('checkpoint',barrier)
        self.assertEqual(result[-5:],['--viz','none','--device','cuda:0','--enable_cameras'])
        with self.assertRaises(ValueError):
            host.compose_argv(guard,Path('/source'),Path('/output'),Path('/tools'),'name','owner',args,
                              {'usd_path_relative':'../../other.usda'})

    def test_mount_control_characters_rejected(self):
        for path in ('/tmp/a:b','/tmp/a\nb'):
            with self.assertRaises(ValueError):
                host.mount_path(path,exists=False)

    def test_ffprobe_requires_actual_frame_count_and_duration(self):
        probe = {"streams":[{"codec_name":"h264","pix_fmt":"yuv420p","width":1280,"height":720,
                  "nb_read_frames":"750","avg_frame_rate":"50/1"}],"format":{"duration":"15.000000"}}
        self.assertEqual(host.verify_video_probe(probe,frames=750,width=1280,height=720,seconds=15.),probe)
        for changed in ('nb_read_frames','avg_frame_rate','codec_name'):
            bad = copy.deepcopy(probe)
            bad['streams'][0][changed] = {'nb_read_frames':'749','avg_frame_rate':'25/1','codec_name':'mpeg4'}[changed]
            with self.assertRaises(ValueError):
                host.verify_video_probe(bad,frames=750,width=1280,height=720,seconds=15.)

    def test_exit_zero_cannot_mask_failed_or_corrupt_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verified = {"input_sha256":{"checkpoint":"c"*64},"contract":{"sha256":"s"},"runtime_manifest":{"x":1}}
            report = {"schema":common.SCHEMA,"pass":True,"errors":[],**verified,"frames_written":750,
                      "controls_completed":750,"encoder_finalized":True,"checkpoint_unchanged":True,
                      "policy_state_unchanged":True,"physics_coverage_pass":True,"artifacts":{}}
            for name in ('policy.mp4','states.npz','metadata.json'):
                (root/name).write_bytes(b'fixture')
                report['artifacts'][name]={'sha256':common.digest(root/name),'bytes':7}
            path = root/'report.json'
            common.write_json(path,report)
            common.validate_capture_report(path,verified,750)
            for key,value in (('pass',False),('encoder_finalized',False),('frames_written',749)):
                common.write_json(path,{**report,key:value})
                with self.assertRaises(ValueError):
                    common.validate_capture_report(path,verified,750)
            common.write_json(path,report)
            (root/'policy.mp4').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'artifact is missing or changed'):
                common.validate_capture_report(path,verified,750)


if __name__ == '__main__':
    unittest.main()
