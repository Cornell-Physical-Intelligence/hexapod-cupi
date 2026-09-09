"""Host/fixture contract integration against the actual selected-layout verifier."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import contact_audit
from test_contact_audit import KIN, ROOT

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

host=load('full_overlap_host',Path(__file__).with_name('host_run_full_overlap.py'))
layout=load('full_overlap_actual_layout',ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/environment_layout.py')
sys.path.insert(0, str(ROOT/'isaaclab/tests'))
from test_mkii_coincident_environment_layout import complete_evidence


def report(directory):
    bindings=contact_audit.expected_bindings(KIN['body_paths'],['/World/envs/env_0','/World/envs/env_1'])
    (directory/'trace.npz').write_bytes(b'fixture-bytes-covered-by-sha')
    value={'schema':'hexapod.live_full_body_collision_fixture.v1','case':'filtered','pass':False,
        'fixture_complete':True,'control_expectation_met':True,'simulation_training_admission':False,'hardware_admission':False,
        'physics_steps_requested':256,'errors':[],'source_identity':{'sha256':'source'},'fixture_source_sha256':'fixture',
        'native_pair_bindings':bindings,'native_rigid_path_readbacks':{row['source_body']:[row['source_body']] for row in bindings},
        'native_pair_count_dtypes':[{'counts':'torch.uint32','start_indices':'torch.uint32'} for _ in bindings],
        'pair_count_storage_dtype':'int64','environment_layout':complete_evidence(2)[0].layout_report,
        'runtime_manifest':{'resolved_environment_layout':layout.layout_runtime_descriptor()},
        'fixture_dependency_sha256':hashlib.sha256(Path(__file__).with_name('contact_audit.py').read_bytes()).hexdigest(),
        'metrics':{'samples':256,'finite':True,'max_pair_contact_count':0,'max_pair_force_n':0.,
            'contact_buffer_capacity_reached':False,'ground_support_observed_per_robot':[True,True],
            'source_body_excitation':[dict({key:row[key] for key in ('source_body','source_name','source_environment')},
                contact_count_observed=False,force_above_1mN_observed=False) for row in bindings]},
        'trace':{'file':'trace.npz','samples':256,'sha256':hashlib.sha256((directory/'trace.npz').read_bytes()).hexdigest()}}
    return value


class FullOverlapHostTests(unittest.TestCase):
    def check(self,directory,value):
        path=directory/'report.json';path.write_text(json.dumps(value))
        return host.require_report(path,case='filtered',steps=256,fixture_hash='fixture',contract={'sha256':'source'},
                                   solver_multiplier=2,contract_source_dir=ROOT)

    def test_actual_layout_descriptor_and62_exact_body_bindings_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);value=report(directory)
            self.assertTrue(self.check(directory,value)['fixture_complete'])

    def test_complete_runtime_reset_readback_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);baseline=report(directory)
            for corrupt in ('missing', 'partial', 'wrong_native_pose'):
                value=copy.deepcopy(baseline)
                evidence=value['environment_layout']
                if corrupt=='missing':
                    evidence.pop('latest_reset')
                elif corrupt=='partial':
                    evidence['latest_reset']['environment_row_indices']=[0]
                else:
                    evidence['latest_reset']['native_root_poses_xyzw'][1][0]=0.01
                with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, 'reset'):
                    self.check(directory,value)

    def test_missing_or_foreign_native_bindings_and_wrong_layout_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);baseline=report(directory)
            mutations=[lambda r:r['native_pair_bindings'][17]['target_filters'].__setitem__(5,r['native_pair_bindings'][17]['source_body']),
                lambda r:r['native_rigid_path_readbacks'].pop(next(iter(r['native_rigid_path_readbacks']))),
                lambda r:r['native_pair_count_dtypes'][0].__setitem__('counts','torch.float32'),
                lambda r:r['runtime_manifest']['resolved_environment_layout'].__setitem__('layout_id','grid_2m_v1'),
                lambda r:r['metrics'].__setitem__('contact_buffer_capacity_reached',True),
                lambda r:r['metrics'].__setitem__('max_pair_contact_count',1),
                lambda r:r.__setitem__('simulation_training_admission',True)]
            for mutate in mutations:
                value=copy.deepcopy(baseline);mutate(value)
                with self.assertRaises(ValueError):self.check(directory,value)

    def test_compose_retains_separate_readonly_mounts_owner_barrier_and_headless_sdk(self):
        actual=host.load_supervisor(ROOT)
        argv=host.compose_argv(actual,Path('/source'),Path('/fixture'),Path('/output'),'owned','nonce','filtered',256,2)
        self.assertIn('/source:/workspace/hexapod:ro',argv)
        self.assertIn('/fixture:/workspace/overlap_fixture:ro',argv)
        self.assertIn('/output:/workspace/validation_artifacts:rw',argv)
        self.assertIn(actual.OWNER_LABEL+'=nonce',argv)
        self.assertIn('exec "$@"',argv[argv.index('-c')+1])
        self.assertEqual(argv[argv.index('--case')+1],'filtered')
        self.assertTrue(any('live_full_body_overlap.py' in item for item in argv))
        self.assertNotIn('--checkpoint',argv)

    def test_source_and_canonical_control_are_checked_before_owned_admission_and_during_run(self):
        tree=ast.parse(Path(__file__).with_name('host_run_full_overlap.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
        unchanged=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='unchanged')
        text=ast.unparse(unchanged)
        self.assertIn('require_unchanged_source',text)
        self.assertIn('tree_identity(fixture)',text)
        self.assertIn('read_coordination_control',text)
        self.assertIn('require_coordination_none',text)
        self.assertNotIn('coordination_snapshot()',text)
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
        self.assertGreaterEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='unchanged' for n in calls),3)
        self.assertGreaterEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='resource_gate' for n in calls),4)


if __name__=='__main__':unittest.main()
