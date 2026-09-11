import ast,copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'robot/active_model.json').is_file())
s=importlib.util.spec_from_file_location('ppo_guard_tests',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

class PPOAdmissionTests(unittest.TestCase):
    def test_pending_actual32_prevents_main_before_any_external_call(self):
        with patch.object(g,'STANDING32_STATE_SHA256',None),patch.object(g.subprocess,'run')as run,patch.object(g.subprocess,'check_output')as call,patch.object(g.os,'open')as opened,patch.object(g.importlib.util,'spec_from_file_location')as imported:
            with self.assertRaisesRegex(RuntimeError,'pending'):g.main()
            for operation in (run,call,opened,imported):operation.assert_not_called()

    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'bindings.json'
        binding={'standing_source_freeze_sha256':g.STANDING_SOURCE_SHA256,'standing1_state_sha256':g.STANDING_ONE_STATE_SHA256,
                 'standing32_state_sha256':'a'*64,'ready_for_native_dispatch':True}
        path.write_text(json.dumps(binding))
        identity={'schema':'canonical_direct_drive_ppo_smoke_v1','phase':'canonical_ppo_smoke',
          'runtime_binding':{'runtime_tree_sha256':g.SOURCE_SHA256,'scope':'fresh_canonical_405_408_two_update_smoke'},
          'training_integration_only':True,'quality_admitted':False,'Stage2_complete':False,
          'fresh_neutral_controls':1000,'fresh_neutral_substeps':8000,'policy_controls':48,'policy_substeps':384,'total_controls':1048,'total_substeps':8384,
          'policy_lineage':{'standing1_state_sha256':g.STANDING_ONE_STATE_SHA256,'standing32_state_sha256':'a'*64},
          'native_identity':{'standing_one_inventory_sha256':g.STANDING_ONE_INVENTORY_SHA256}}
        host=NS(SOURCE_FREEZE=g.SOURCE_SHA256,STANDING_FREEZE=g.STANDING_SOURCE_SHA256,SUPERVISOR_MAP=g.SUPERVISOR_SHA256,
          EXPECTED_COORDINATION='649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f',require_fresh_output=Mock(),verify_inputs=Mock(return_value=identity))
        return path,binding,identity,host

    def invoke(self,path,host):
        bound=g.sha(path)
        def hashes(p):return bound if p==path else g.HOST_SHA256
        with patch.multiple(g,BINDINGS=path,BINDINGS_SHA256=bound,STANDING32_STATE_SHA256='a'*64),patch.object(g,'require_final_bindings'),patch.object(g,'verify_reservation'),patch.object(g,'sha',side_effect=hashes),patch.object(g,'verify_manifest')as manifests:
            result=g.validate_ppo_inputs(host)
            self.assertEqual([c.args[0] for c in manifests.call_args_list],[g.HOST.parent,g.SOURCE,g.STANDING_SOURCE])
        return result

    def test_exact_ppohost_receives_all_immutable_input_paths(self):
        path,binding,identity,host=self.fixture();self.assertEqual(self.invoke(path,host),identity)
        args=host.verify_inputs.call_args.args[0]
        self.assertEqual(args.standing_source,g.STANDING_SOURCE);self.assertEqual(args.standing_one,g.STANDING_ONE)
        self.assertEqual(args.standing32,g.STANDING32);self.assertEqual(args.bindings,path)
        self.assertEqual(args.output,g.OUTPUT);self.assertFalse(hasattr(args,'checkpoint'))
        self.assertFalse(hasattr(args,'num_envs'));self.assertEqual(host.require_fresh_output.call_count,1)

    def test_disabled_wrong_pair_or_source_binding_rejects_before_host(self):
        for key,value in [('ready_for_native_dispatch',False),('standing32_state_sha256',None),('standing1_state_sha256','b'*64),('standing_source_freeze_sha256','b'*64)]:
            path,binding,identity,host=self.fixture();binding[key]=value;path.write_text(json.dumps(binding))
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.invoke(path,host)
            host.verify_inputs.assert_not_called()

    def test_rejected_native_host_and_wrong_fixed_ppo_scope_propagate(self):
        path,binding,identity,host=self.fixture();host.verify_inputs.side_effect=ValueError('standing32 physical gate rejected')
        with self.assertRaisesRegex(ValueError,'physical gate'):self.invoke(path,host)
        for key,value in [('policy_controls',480),('total_substeps',8000),('quality_admitted',True),('schema','old')]:
            path,binding,identity,host=self.fixture();identity[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.invoke(path,host)

    def test_actual_frozen_source_host_standing_inventories(self):
        for root,bound in [(REPO/'tmp/canonical_ppo_host_004',g.HOST_FREEZE_SHA256),(REPO/'tmp/canonical_ppo_integration_003',g.SOURCE_SHA256),(REPO/'tmp/updated_native_standing_005',g.STANDING_SOURCE_SHA256)]:
            g.verify_manifest(root,'FREEZE_SHA256.json',bound)

    def test_reservation_and_embedded_restorer_byte_identical_to_adopted_parent(self):
        def extract(path):
            tree=ast.parse(path.read_text());functions={n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
            embedded=[n.args[0].value for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='restorer'and n.func.attr=='write_text']
            return functions,embedded
        old,oldrest=extract(HERE/'inputs/guard_parent.py');new,newrest=extract(HERE/'launch_guarded_remote.py')
        mask,maskrest=extract(HERE/'inputs/mask_guard_parent.py');self.assertEqual(mask['verify_reservation'],new['verify_reservation']);self.assertEqual(oldrest,newrest);self.assertEqual(maskrest,newrest)
        for name in ['call','sha','valid_hash','verify_manifest']:self.assertEqual(old[name],new[name])

    def test_previous32_validation_preserves_parent_checks_with_declared_row_count(self):
        def function(path):
            text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='verify_previous_owner')
            return ''.join(text.splitlines(True)[node.lineno-1:node.end_lineno])
        old=function(HERE/'inputs/guard_parent.py');new=function(HERE/'launch_guarded_remote.py')
        new=''.join(line for line in new.splitlines(True)if not any(x in line for x in ["raise RuntimeError('Standing32 did not consume exact standing1')","standing_manifest=json.loads", "Actual standing32 complete inventory binding differs"]))
        new=new.replace('standing32','standing1').replace('STANDING32','STANDING_ONE').replace('!=STANDING_SOURCE_SHA256','!=SOURCE_SHA256').replace("identity.get('num_envs')!=32","identity.get('num_envs')!=1").replace("receipt.get('num_envs')!=32","receipt.get('num_envs')!=1").replace('pause018','pause011')
        self.assertEqual(new,old)

if __name__=='__main__':unittest.main()
