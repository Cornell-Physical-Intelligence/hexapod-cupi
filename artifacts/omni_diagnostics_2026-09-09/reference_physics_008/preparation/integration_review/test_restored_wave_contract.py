from pathlib import Path
import ast,copy,hashlib,json,sys,tempfile,unittest
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'reference_physics_adapter_007/source_007/tools')]
from test_screen_contract import source_fixture,freeze
from screen_contract import preflight
from screen_metrics import standing_quiet_review
from solver_comparison import PROTOCOL,validate_iteration_readback

class RestoredWaveContractTests(unittest.TestCase):
    def test_actual007_all32_quiet_pass_and005_bias_rejects_without_metric_change(self):
        for number,expected in [('005',False),('007',True)]:
            path=HERE.parent/f'reference_physics_results_{number}/run/standing/trace.npz'
            with np.load(path) as npz:data={key:npz[key] for key in npz.files}
            result=standing_quiet_review(data,data['joint_names'].tolist())
            self.assertEqual(result['passed'],expected)
            self.assertEqual(sum(row['pass'] for row in result['per_environment']),32 if expected else 30)
        with self.assertRaises(ValueError):standing_quiet_review({**data,'joint_position_rad':data['joint_position_rad'][:999]},data['joint_names'].tolist())

    def test_existing007_standing_protocol_or_missing_quiet_cannot_admit008wave(self):
        with tempfile.TemporaryDirectory() as td:
            source,args=source_fixture(Path(td));identity,_=preflight(args,source)
            args.mode='wave';args.num_envs=1;args.steps=2400;args.admission=Path(td)/'admission.json'
            for changed,gate in [(identity,{'passed':True}),({**identity,'solver_comparison':{'name':'reference007_standing_TGS_velocity_iterations_one_readback_correction'}},{'passed':True,'all_replica_quiet':{'passed':True}})]:
                args.admission.write_text(json.dumps({'identity':changed,'status':'completed','gate':gate}))
                with self.assertRaises(ValueError):preflight(args,source)

    def test_wave_dependency_hash_bound_independently_of_refrozen_source(self):
        with tempfile.TemporaryDirectory() as td:
            source,args=source_fixture(Path(td));p=source/'tools/wave_reference.py';p.write_text(p.read_text()+'\n# changed\n');freeze(source)
            with self.assertRaisesRegex(ValueError,'horizontal'):preflight(args,source)

    def test_single_wave_replica_uses_same_authored_root_readback(self):
        raw=HERE.parent/'reference_physics_adapter_007/inputs/actual006_failed_readback.json'
        rows=json.loads(raw.read_text())['actual_iteration_settings']['rows']
        one=[copy.deepcopy(r) for r in rows if r['path'].startswith('/World/envs/env_0/')]
        validate_iteration_readback(one,1)
        self.assertEqual(len([r for r in one if r['kind']=='body']),19)
        self.assertFalse(PROTOCOL['source007_standing_only_admission_reusable'])
        self.assertFalse(PROTOCOL['PPO_permitted'])
        self.assertEqual(PROTOCOL['wave_dependency']['horizontal_duration_fraction'],.8)

    def test_only_standing_then_wave_calls_and_preserved_progress_gate(self):
        tree=ast.parse((HERE/'launch_reference_physics_spark.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_owned']
        self.assertEqual([c.args[1].value for c in calls],['standing','wave'])
        old=ast.parse((HERE.parent/'reference_physics_adapter_007/source_007/tools/run_reference_physics.py').read_text())
        new=ast.parse((HERE/'run_reference_physics.py').read_text())
        def wave_admission(t):
            return [ast.dump(n) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='update' and any(k.arg=='kind' and isinstance(k.value,ast.Constant) and k.value.value=='single_leg_wave_bounded_physics_screen_not_stage2' for k in n.keywords)]
        self.assertEqual(wave_admission(old),wave_admission(new))
        self.assertEqual(len(wave_admission(new)),1)
if __name__=='__main__':unittest.main()
