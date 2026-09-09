"""Asset identity cannot silently change across audits, solver runs and PPO."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from mkii_asset_binding import verify_asset_binding, solver_runtime_equivalent


class AssetBindingTests(unittest.TestCase):
    def fixture(self):
        parent = 'robot/model_v4/'
        files = {parent+name: str(index)*64 for index, name in enumerate(
            ('model.usda', 'geometry.usdc', 'kinematics.json', 'manifest.json'), 1)}
        bundle = dict(model_id='mkii_fourbar_v4', usd_path_relative=parent+'model.usda',
            closure_constraint_variant='planar_d6_xy_v4', usd_root_sha256='1'*64,
            kinematics_sha256='3'*64, bundle_files_sha256=files)
        audit = {'pass': True, 'errors': [], 'usd_root_sha256': '1'*64,
            'kinematic_contract_sha256': '3'*64, 'closure_constraint_variant':'planar_d6_xy_v4',
            'dependencies':[{'path':path[len(parent):], 'sha256':sha} for path,sha in files.items()
                            if Path(path).suffix in ('.usda','.usdc')]}
        runtime = {key:bundle[key] for key in ('model_id','usd_root_sha256','kinematics_sha256','closure_constraint_variant')}
        runtime['asset_bundle'] = copy.deepcopy(bundle)
        return bundle, audit, copy.deepcopy(audit), runtime, {'files':copy.deepcopy(files)}

    def test_exact_asset_is_bound(self):
        self.assertTrue(verify_asset_binding(*self.fixture())['pass'])

    def test_different_asset_or_variant_at_any_stage_rejects(self):
        for stage in (1,2,3):
            for key in ('usd_root_sha256','closure_constraint_variant'):
                args=list(self.fixture());args[stage][key]='different'
                with self.subTest(stage=stage,key=key), self.assertRaises(ValueError):
                    verify_asset_binding(*args)

    def test_dependency_tampering_missing_extra_duplicate_and_escape_reject(self):
        for stage in (1,2):
            for mutation in ('tamper','missing','extra','duplicate','escape'):
                args=list(self.fixture());deps=args[stage]['dependencies']
                if mutation=='tamper':deps[1]['sha256']='a'*64
                if mutation=='missing':deps.pop()
                if mutation=='extra':deps.append({'path':'another.usdc','sha256':'a'*64})
                if mutation=='duplicate':deps.append(copy.deepcopy(deps[0]))
                if mutation=='escape':deps[1]['path']='../geometry.usdc'
                with self.subTest(stage=stage,mutation=mutation), self.assertRaises(ValueError):
                    verify_asset_binding(*args)

    def test_source_identity_and_live_full_bundle_are_required(self):
        args=list(self.fixture());args[4]['files'].pop('robot/model_v4/geometry.usdc')
        with self.assertRaises(ValueError):verify_asset_binding(*args)
        args=list(self.fixture());args[3]['asset_bundle']['bundle_files_sha256']['robot/model_v4/geometry.usdc']='f'*64
        with self.assertRaises(ValueError):verify_asset_binding(*args)

    def test_solver_runs_may_only_change_position_iterations(self):
        a=dict(self.fixture()[3],resolved_simulation={'solver_position_iterations':64,'solver_type':1},
               motor={'peak':5.5}, joint_names=['a','b'])
        b=copy.deepcopy(a);b['resolved_simulation']['solver_position_iterations']=128
        self.assertTrue(solver_runtime_equivalent(a,b))
        for key,bad in [('model_id','mkii_fourbar_v3'),('motor',{'peak':55}),('joint_names',['b','a'])]:
            changed=copy.deepcopy(b);changed[key]=bad
            self.assertFalse(solver_runtime_equivalent(a,changed))
        b['resolved_simulation']['solver_type']=True
        self.assertFalse(solver_runtime_equivalent(a,b))


if __name__=='__main__':unittest.main()
