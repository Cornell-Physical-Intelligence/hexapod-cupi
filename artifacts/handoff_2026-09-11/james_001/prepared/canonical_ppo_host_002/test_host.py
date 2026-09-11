import copy, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'robot/active_model.json').is_file())
spec=importlib.util.spec_from_file_location('tested_canonical_ppo_host',HERE/'launch_ppo_spark.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
SOURCE=REPO/'tmp/canonical_ppo_integration_002'
class HostTests(unittest.TestCase):
    def args(self,root):
        return NS(**{k:root/k for k in h.INPUT_ARGS},output=root/'output',host_freeze_sha256='a'*64,isaaclab=root/'isaaclab')
    def identity(self,args):
        protocol=h.load_contract(SOURCE).protocol()
        return {'schema':h.SCHEMA,'phase':h.PHASE,'runtime_binding':{'runtime_tree_sha256':h.SOURCE_FREEZE,'scope':'fresh_canonical_405_408_two_update_smoke'},
            'fresh_neutral_controls':1000,'fresh_neutral_substeps':8000,'policy_controls':48,'policy_substeps':384,'total_controls':1048,'total_substeps':8384,
            'training_integration_only':True,'quality_admitted':False,'Stage2_complete':False,'bindings_sha256':'a'*64,'protocol':protocol,
            'native_identity':{'runtime_binding':{'runtime_tree_sha256':h.STANDING_FREEZE,'scope':'canonical_provisional_native_standing_only'},
                'num_envs':32,'controls':1000,'steps':8000,'dt':.0025,'control_dt':.02,'substeps_per_control':8,'settle_controls':200,'ground':True,'gravity':[0.,0.,-9.81],'actuation_state_sha256':h.ADMISSION_STATE,'physical_admission':False,'training_allowed':False},
            'policy_lineage':{'consumer_source_freeze_sha256':h.SOURCE_FREEZE,'standing_source_freeze_sha256':h.STANDING_FREEZE,'fresh_checkpoint_only':True,
                **{k:'a'*64 for k in ['standing1_state_sha256','standing32_state_sha256','servo_sha256','geometry_sha256','adapter_sha256','rsl_source_map_sha256']}}}
    def test_stdlib_import_no_gpu(self):
        subprocess.run([sys.executable,'-B','-S','-c',"import launch_ppo_spark,sys;assert not any(x in sys.modules for x in ('torch','numpy','isaaclab'))"],cwd=HERE,check=True,capture_output=True)
    def test_exact_owner_and_source_inventories(self):
        h.verify_tree(SOURCE,'FREEZE_SHA256.json',h.SOURCE_FREEZE)
        h.verify_tree(REPO/'tmp/updated_native_standing_003','FREEZE_SHA256.json',h.STANDING_FREEZE)
        self.assertEqual(h.sha(HERE/'inputs/source009_supervisor.py'),h.SUPERVISOR_CODE)
    def test_fixed_identity_and_missing_admissions_reject(self):
        args=self.args(Path('/a'));identity=self.identity(args)
        with patch.object(h,'sha',return_value='a'*64):
            self.assertEqual(h.validate_identity(identity,args),identity)
            for target,key,value in [('protocol','updates',50),('protocol','replicas',1024),('protocol','checkpoint_input','old.pt'),('protocol','auto_reset',True),('protocol','actor_width',315),('native_identity','num_envs',1),('native_identity','ground',False),('policy_lineage','standing32_state_sha256',None),('policy_lineage','fresh_checkpoint_only',False)]:
                bad=copy.deepcopy(identity);bad[target][key]=value
                with self.assertRaises(ValueError,msg=(target,key)):h.validate_identity(bad,args)
            for key,value in [('total_substeps',8000),('quality_admitted',True),('Stage2_complete',True),('schema','legacy')]:
                bad=copy.deepcopy(identity);bad[key]=value
                with self.assertRaises(ValueError):h.validate_identity(bad,args)
    def test_readonly_all_inputs_no_checkpoint_and_one_phase(self):
        a=self.args(Path('/a'));c=h.command(a,'exact-owned',h.PHASE)
        mounts=[c[i+1]for i,v in enumerate(c)if v=='-v']
        self.assertEqual(mounts,['/a/output:/output:rw','/a/source:/ppo:ro','/a/standing_source:/standing:ro','/a/asset:/asset:ro','/a/admission:/admission:ro','/a/standing_one:/standing_one:ro','/a/standing32:/standing32:ro','/a/bindings:/bindings.json:ro'])
        self.assertNotIn('--checkpoint',c);self.assertNotIn('--num-envs',c);self.assertNotIn('--steps',c);self.assertIn('/ppo/run_native_smoke.py',c)
        for phase in ('standing','wave','train_50','evaluate'):
            with self.assertRaises(ValueError):h.command(a,'owned',phase)
    def test_overlap_reused_and_dangling_output(self):
        with tempfile.TemporaryDirectory()as t:
            a=self.args(Path(t));a.output.mkdir()
            with self.assertRaises(ValueError):h.require_fresh_output(a)
            a.output=Path(t)/'dangling';a.output.symlink_to(Path(t)/'missing')
            with self.assertRaises(ValueError):h.require_fresh_output(a)
            for key in h.INPUT_ARGS:
                a.output=getattr(a,key)/'nested'
                with self.assertRaises(ValueError,msg=key):h.require_fresh_output(a)
    def test_foreign_cached_package_rejected_before_import(self):
        for name in ('canonical_direct_ppo','canonical_direct_ppo.native_contract','canonical_direct_ppo.runner'):
            with patch.dict(sys.modules,{name:NS(__file__='/foreign/'+name+'.py')}):
                with self.assertRaisesRegex(ValueError,'Foreign cached'):h.load_contract(SOURCE)
    def test_real_contract_template_remains_disabled_without_process(self):
        contract=h.load_contract(SOURCE)
        a=self.args(Path('/unused'));a.standing_source=REPO/'tmp/updated_native_standing_003';a.bindings=SOURCE/'BINDINGS.json'
        with patch.object(subprocess,'Popen',side_effect=AssertionError('No subprocess')),patch.object(subprocess,'run',side_effect=AssertionError('No subprocess')):
            with self.assertRaisesRegex(ValueError,'standing1_state_sha256'):contract.verify_inputs(h.native_args(a))
    def test_load_contract_restores_path_and_exact_origins(self):
        before=sys.path[:];contract=h.load_contract(SOURCE)
        self.assertEqual(before,sys.path);self.assertEqual(Path(contract.__file__).resolve(),SOURCE/'canonical_direct_ppo/native_contract.py')
    def test_no_changed_or_unlisted_input(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t);(p/'x').write_text('x');(p/'FREEZE_SHA256.json').write_text(json.dumps({'x':h.sha(p/'x')}));digest=h.sha(p/'FREEZE_SHA256.json')
            h.verify_tree(p,'FREEZE_SHA256.json',digest);(p/'extra').write_text('e')
            with self.assertRaises(ValueError):h.verify_tree(p,'FREEZE_SHA256.json',digest)
    def test_original_supervisor_codetypes_and_truthful_metadata(self):
        a=self.args(Path('/uncreated'));a.supervisor_source=REPO/'tmp/reference_physics_adapter_009/source_009';identity=self.identity(a)
        with patch.object(sys,'path',list(sys.path)):
            sys.path.insert(0,str(a.supervisor_source/'tools'))
            old=h.load_module('_original_code_test',a.supervisor_source/'tools/launch_reference_physics_spark.py')
            with patch.object(subprocess,'Popen',side_effect=AssertionError('No processes')),patch.object(subprocess,'run',side_effect=AssertionError('No processes')):
                new=h.load_supervisor(a,identity)
            for name in ('run_owned','owned_container'):
                self.assertEqual(getattr(old,name).__code__,getattr(new,name).__code__)
            with patch.object(h,'verify_inputs',return_value=identity)as verify:new.verified_source(a.source);verify.assert_called_once_with(a)
            with self.assertRaises(ValueError):new.verified_source(a.asset)
            with tempfile.TemporaryDirectory()as t:
                path=Path(t)/'jobs'/f'{h.PHASE}.json';path.parent.mkdir();new.save(path,{'no_policy_loaded':True})
                metadata=h.read(path);self.assertFalse(metadata['no_policy_loaded']);self.assertTrue(metadata['no_checkpoint_loaded']);self.assertIsNone(metadata['checkpoint_input']);self.assertTrue(metadata['training_integration_only']);self.assertFalse(metadata['quality_admitted']);self.assertFalse(metadata['physical_admission'])
    def run_fixture(self,root,*,failure=None,validation_failure=None,input_changed=False,mutate_late=False):
        a=self.args(root);a.output.mkdir();phase=a.output/h.PHASE;phase.mkdir();(phase/'native_errors.json').write_text('[]')
        identity=self.identity(a);accepted={'phase':h.PHASE,'updates_completed':2,'quality_admitted':False}
        verify=Mock(side_effect=[identity,{'changed':True}])if input_changed else Mock(return_value=identity)
        if mutate_late:
            calls=[]
            def inputs(_):
                calls.append(1)
                if len(calls)==2:(phase/'native_errors.json').write_text('["late"]')
                return identity
            verify=Mock(side_effect=inputs)
        with patch.object(h,'PARENT',NS(run_owned=Mock(side_effect=failure))),patch.object(h,'CONTRACT',NS(validate_result=Mock(return_value=accepted,side_effect=validation_failure))),patch.object(h,'verify_inputs',verify):
            if failure or validation_failure or input_changed or mutate_late:
                with self.assertRaises(Exception):h.run_campaign(a,identity)
            else:
                result=h.run_campaign(a,identity);self.assertEqual(result['accepted_phase'],accepted)
        return h.read(a.output/'campaign.json'),a
    def test_success_seals_post_exit_including_native_logs(self):
        with tempfile.TemporaryDirectory()as t:
            report,a=self.run_fixture(Path(t));self.assertEqual(report['status'],'completed');self.assertFalse(report['quality_admitted']);self.assertFalse(report['automatic_continuation'])
            self.assertIn('native_errors.json',h.read(a.output/(h.PHASE+'_immutable.sha256.json')))
    def test_native_or_reload_rejection_preserved(self):
        for native,validator in [(RuntimeError('native contact failure'),None),(None,ValueError('strict reload incomplete'))]:
            with tempfile.TemporaryDirectory()as t:
                report,_=self.run_fixture(Path(t),failure=native,validation_failure=validator)
                self.assertEqual(report['status'],'failed');self.assertTrue(report['terminal_inputs_unchanged']);self.assertNotIn('accepted_phase',report)
    def test_terminal_input_change_rejects_after_apparent_success(self):
        with tempfile.TemporaryDirectory()as t:
            report,_=self.run_fixture(Path(t),input_changed=True);self.assertEqual(report['status'],'failed');self.assertFalse(report['terminal_inputs_unchanged'])
    def test_late_native_output_change_rejects(self):
        with tempfile.TemporaryDirectory()as t:
            report,_=self.run_fixture(Path(t),mutate_late=True);self.assertEqual(report['status'],'failed');self.assertIn('Native payload changed',report['integrity_error'])
    def test_exact_deadlines_and_appready_remain(self):
        code=(HERE/'inputs/source009_supervisor.py').read_text()
        for required in ['deadline = time.monotonic() + 600','app_ready_deadline = time.monotonic() + 90','REFERENCE_SCREEN_APP_READY','/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock']:self.assertIn(required,code)
        self.assertNotIn('deadline_adapter',(HERE/'launch_ppo_spark.py').read_text())
if __name__=='__main__':unittest.main()
