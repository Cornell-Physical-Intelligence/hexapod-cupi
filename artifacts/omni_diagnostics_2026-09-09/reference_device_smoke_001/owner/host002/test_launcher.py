"""Host guards use real recorded values; generated receipts remain CPU fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import launch_device_smoke_spark as host
from numeric_evidence import numeric

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent/'reference_device_smoke_001'
sys.path.insert(0,str(BRIDGE))
import test_recorded_seam as recorded


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recorded.RecordedSeamTests.setUpClass();replay=recorded.RecordedSeamTests()
        encoder,packet=replay.run_sequence(1)
        cls.encoded=encoder.build(packet);cls.measurement=packet['measurement']
        cls.schema=encoder.spec()
        provenance=json.loads((HERE/'inputs/PROVENANCE.json').read_text())
        cls.rawpath=HERE/'inputs'/provenance['derived']
        assert host.sha(cls.rawpath)==provenance['derived_sha256']

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)
        self.identity={'replicas':1,'source_manifest_sha256':host.SOURCE_MAP,'actor_width':846,'critic_width':849}
        self.sub={'samples_including_initial':2113,'completed_controls':264,'error':None,'method_restored':True,
            'physical_acceptance_gate_changed':False,'decimation':8,'physics_dt_s':.0025,'physics_handles_decimation':False,
            'postsettle_worst_joint_substep_saturation_fraction':0.,'joint_names_runtime':list(self.schema['joint_names_runtime'])}
        self.state=dict(status='completed',identity=self.identity,controls=264,bridge_passed=True,source_inputs_unchanged=True,
            actor_width=846,critic_width=849,substep_torque_checked=True,short_hold_is_not_quiet_admission=True,
            policy_training_started=False,stage2_complete=False,legacy_env_step_includes_original_CPU_telemetry_copies=True,
            runtime_binding={'runtime_tree_sha256':host.RUNTIME_TREE},substep_review=self.sub,
            comparison_keys=sorted(host.READER_COMPARISON_KEYS))
        with np.load(self.rawpath) as data:
            self.state['postsettle_max_requested_nm']=float(np.abs(data['computed_torque_nm'][1601:]).max())
            self.state['postsettle_max_applied_nm']=float(np.abs(data['applied_torque_nm'][1601:]).max())
        shutil.copyfile(self.rawpath,self.p/'physics_substeps.npz')
        self.observation={k:self.encoded[k].numpy().copy() for k in ('policy','critic','raw_sdk_joint_velocity_rad_s','interval_joint_rate_rad_s','interval_rate_valid')}
        self.sample={k:v.numpy().copy() for k,v in self.measurement.items()}
        clocks=[];current=np.zeros((1,14),dtype=np.float32)
        for _ in range(264):
            for __ in range(8):current=current+np.float32(.0025)
            clocks.append(current.copy())
        clocks=np.stack(clocks)
        self.clocks={'sensor_timestamp_s':clocks,'sensor_last_update_s':clocks.copy(),'expected_timestamp_s':clocks.copy(),
            'sensor_outdated':np.zeros((264,1,14),dtype=bool),'sensor_age_s':np.zeros((264,1,14)),
            'all_sensors_valid':np.ones((264,1),dtype=bool),'contact_valid':np.ones((264,1,6),dtype=bool),
            'contact_age_s':np.zeros((264,1,6))}
        self.timing={'samples':[dict(control=k,reference_and_target_set_ms=.1,env_step_including_capture_ms=1.,
            nested_device_capture_ms=.1,observation_history_ms=.1) for k in range(201,265)]}
        for name in ('environment.yaml','solver_comparison.json','last_reference_state.npz','physics_control_integrals.npz'):
            (self.p/name).write_bytes(b'CPU fixture; these auxiliary bytes are hashed, not physically replayed here')
        self.store()

    def store(self):
        for name,data in [('state.json',self.state),('physics_substep_review.json',self.sub),('observation_schema.json',self.schema),('timings.json',self.timing)]:
            (self.p/name).write_text(json.dumps(data))
        np.savez_compressed(self.p/'final_observation.npz',**self.observation)
        np.savez_compressed(self.p/'last_device_sample.npz',**self.sample)
        np.savez_compressed(self.p/'sensor_clocks.npz',**self.clocks)

    def test_recorded_values_and_schema_fixture_pass_without_runtime_claim(self):
        result=host.validate_result(self.p,self.identity)
        self.assertTrue(result['passed']);self.assertFalse(result['PPO_admitted']);self.assertFalse(result['fresh_quiet_admission'])
        self.assertEqual(len(result['files_sha256']),12)

    def test_missing_wrong_or_failed_state_cannot_promote(self):
        baseline=deepcopy(self.state)
        for key,value in [('identity',{'replicas':32}),('controls',263),('substep_torque_checked',False),
            ('source_inputs_unchanged',False),('actor_width',740),('critic_width',743),('bridge_passed',False),
            ('comparison_keys',[]),
            ('postsettle_max_requested_nm',float('nan')),('postsettle_max_applied_nm',1.6001),('policy_training_started',True)]:
            self.state=deepcopy(baseline);self.state[key]=value;self.store()
            with self.subTest(key=key),self.assertRaises(ValueError):host.validate_result(self.p,self.identity)

    def test_stale_interior_clock_and_contact_or_terminal_reject(self):
        self.clocks['all_sensors_valid'][50,0]=False;self.store()
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)
        self.clocks['all_sensors_valid'][50,0]=True;self.clocks['sensor_last_update_s'][51,0,3]-=.0025;self.store()
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)
        self.clocks['sensor_last_update_s']=self.clocks['sensor_timestamp_s'].copy();self.sample['terminated'][0]=True;self.store()
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)

    def test_jointly_duplicated_or_reordered_three_clock_arrays_rejected(self):
        originals={k:self.clocks[k].copy() for k in ('sensor_timestamp_s','sensor_last_update_s','expected_timestamp_s')}
        for kind in ('duplicate','reorder'):
            for key,value in originals.items():
                self.clocks[key]=value.copy()
                if kind=='duplicate':self.clocks[key][50]=self.clocks[key][49]
                else:self.clocks[key][[50,51]]=self.clocks[key][[51,50]]
            self.store()
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'eight actual float32 ticks'):
                host.validate_result(self.p,self.identity)

    def test_raw_rate_channel_and_schema_or_nonfinite_observation_reject(self):
        self.observation['raw_sdk_joint_velocity_rad_s'][0,0]+=.001;self.store()
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)
        self.observation['raw_sdk_joint_velocity_rad_s']=self.sample['joint_velocity_rad_s'].copy()
        self.observation['policy'][0,0]=np.nan;self.store()
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)

    def test_complete_substep_values_checked_including_interior_spike(self):
        with np.load(self.p/'physics_substeps.npz') as data:values={k:data[k] for k in data.files}
        values['computed_torque_nm'][1700,0,4]=1.7
        np.savez_compressed(self.p/'physics_substeps.npz',**values)
        with self.assertRaises(ValueError):host.validate_result(self.p,self.identity)

    def test_numeric_reader_rejects_shape_dtype_object_and_missing(self):
        path=self.p/'bad.npz'
        for value in (np.zeros((1,2)),np.array([object()],dtype=object),np.array([np.nan])):
            np.savez(path,a=value)
            with self.assertRaises(ValueError):numeric(path,'a',(1,))
        with self.assertRaises(ValueError):numeric(path,'missing',(1,))

    def test_command_has_only_declared_two_modes_and_readonly_inputs(self):
        a=SimpleNamespace(source=Path('/source'),run=Path('/qualified'),adapter=Path('/adapter'),observation=Path('/observation'),output=Path('/new'))
        for phase,n in host.PHASES.items():
            command=host.command(a,'owned',phase)
            self.assertEqual(command[command.index('--num-envs')+1],str(n))
            self.assertEqual(command[command.index('--controls')+1],'264')
            mounts=[command[i+1] for i,v in enumerate(command) if v=='-v']
            self.assertEqual(mounts,['/source:/workspace/hexapod:ro','/qualified:/qualified:ro','/adapter:/adapter:ro','/observation:/observation:ro','/new:/outputs:rw'])
            self.assertNotIn('--checkpoint',command);self.assertNotIn('--enable_cameras',command)
        with self.assertRaises(ValueError):host.command(a,'owned','train')

    def test_second_allocation_is_unreachable_after_first_rejection(self):
        calls=[];a=SimpleNamespace(source=Path('/source'),output=self.p)
        supervisor=SimpleNamespace(check_source=lambda _:None,run_owned=lambda a,p:calls.append(p))
        identities={k:{'replicas':n} for k,n in host.PHASES.items()}
        with patch.object(host,'validate_inputs',return_value=identities),patch.object(host,'validate_result',side_effect=ValueError('rejected')):
            with self.assertRaises(ValueError):host.run_phases(a,supervisor,identities,{})
        self.assertEqual(calls,['replicas_1'])

    def test_changed_input_stops_before_next_owned_job(self):
        calls=[];a=SimpleNamespace(source=Path('/source'),output=self.p)
        supervisor=SimpleNamespace(check_source=lambda _:None,run_owned=lambda a,p:calls.append(p))
        identities={k:{'replicas':n} for k,n in host.PHASES.items()}
        with patch.object(host,'validate_inputs',side_effect=[identities,{}]),patch.object(host,'validate_result',return_value={'passed':True}):
            with self.assertRaises(ValueError):host.run_phases(a,supervisor,identities,{})
        self.assertEqual(calls,['replicas_1'])

    def test_host_helper_and_initial_manifest_remain_bound(self):
        root=self.p/'host';root.mkdir();(root/'helper.py').write_text('immutable helper')
        manifest=root/'FREEZE_SHA256.json'
        manifest.write_text(json.dumps({'helper.py':host.sha(root/'helper.py')}))
        starting=host.sha(manifest)
        with patch.object(host,'__file__',str(root/'launcher.py')):
            self.assertEqual(host.verify_own_bundle(starting),starting)
            (root/'helper.py').write_text('changed helper')
            with self.assertRaises(ValueError):host.verify_own_bundle(starting)
            manifest.write_text(json.dumps({'helper.py':host.sha(root/'helper.py')}))
            with self.assertRaises(ValueError):host.verify_own_bundle(starting)

if __name__=='__main__':unittest.main()
