import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import query_contract as contract
import run_query


def dump(node):return ast.dump(node,include_attributes=False)
def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)


class SourceChecks(unittest.TestCase):
    def test_contract_file_loader_ignores_missing_path_and_unrelated_cached_module(self):
        code='''import importlib.util,sys,types,json
from pathlib import Path
path=Path(sys.argv[1]).resolve()
sys.path=[x for x in sys.path if Path(x or '.').resolve()!=path.parent]
sys.modules['inspection_contract']=types.SimpleNamespace(DT=999,STEPS=999)
spec=importlib.util.spec_from_file_location('_isolated_query_contract',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.DT==.0025 and module.STEPS==8
assert Path(module.parent.__file__).resolve()==path.with_name('inspection_contract.py')
print('PASS isolated file-relative contract')
'''
        result=subprocess.run([sys.executable,'-S','-B','-c',code,str(HERE/'query_contract.py')],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PASS isolated',result.stdout)

    def test_unchanged_helpers_and_exact_probe_inventory(self):
        origin=json.loads((HERE/'SOURCE_ORIGIN.json').read_text())
        for name,digest in origin['copied_helpers'].items():self.assertEqual(contract.sha(HERE/name),digest)
        self.assertEqual(contract.sha(HERE/'probe/FREEZE_SHA256.json'),contract.PROBE_FREEZE)
        contract.check_map(HERE/'probe',json.loads((HERE/'probe/FREEZE_SHA256.json').read_text()))

    def test_identical_physics_setup_and_eight_step_loop(self):
        a=function(HERE/'run_inspection.py','main');b=function(HERE/'run_query.py','main')
        loop=lambda t:next(n for n in ast.walk(t)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='n')
        self.assertEqual(dump(loop(a)),dump(loop(b)))
        cfg=lambda t:next(n for n in ast.walk(t)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='cfg'for x in n.targets))
        self.assertEqual(dump(cfg(a)),dump(cfg(b)))
        for name in ['tensor_provider','create_sdf_view','legacy_cooking_counter','sdk_readback','collect_native','snapshot']:
            self.assertEqual(dump(function(HERE/'run_inspection.py',name)),dump(function(HERE/'run_query.py',name)))
        step_count=lambda t:sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='step'for n in ast.walk(t))
        self.assertEqual(step_count(a),1);self.assertEqual(step_count(b),1)

    def test_actual_parent_admission_and_changed_missing_payload(self):
        original=Path('tmp/canonical_native_inspection_terminal_003/run/inspection').resolve()
        if not original.exists():self.skipTest('Actual003 admission is separately published; pass from repository root')
        result=contract.verify_admission(original)
        self.assertEqual(result['phase_a_state_sha256'],'987dd4d2a2314ec234e9629dc4c045f10d6230f3301b4d9594c0695052d58228')
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'admission';shutil.copytree(original,p)
            (p/'samples.json').write_text('[]')
            with self.assertRaises(ValueError):contract.verify_admission(p)
            (p/'samples.json').unlink()
            with self.assertRaises(ValueError):contract.verify_admission(p)

    def test_old_result_is_not_query_admission(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'state.json').write_text(json.dumps({'schema':'canonical_native_inspection_v1','identity':{}}))
            with self.assertRaisesRegex(ValueError,'identity'):contract.validate_result(p,{})

    def test_finish_seals_nested_query_state_and_preserves_late_error_file(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'query').mkdir();(p/'query/state.json').write_text('{"status":"completed"}')
            (p/'native_errors.json').write_text('[]');(p/'isaac_logs').mkdir();(p/'isaac_logs/live.log').write_text('initial')
            state={'status':'completed','errors':[]};identity={'fake':True}
            class App:
                def close(self):
                    (p/'native_errors.json').write_text('[{"late":true}]')
                    (p/'isaac_logs/live.log').write_text('late')
            with patch.object(run_query,'verify_inputs',return_value=identity):run_query.finish(p,state,None,identity,App())
            saved=json.loads((p/'state.json').read_text())
            self.assertIn('query/state.json',saved['outputs']);self.assertNotIn('native_errors.json',saved['outputs'])
            self.assertNotIn('isaac_logs/live.log',saved['outputs'])
            self.assertEqual(json.loads((p/'native_errors.json').read_text()),[{'late':True}])

    def test_late_native_error_rejected_before_diagnostic_acceptance(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);state={'schema':contract.SCHEMA,'identity':{},'status':'completed','inputs_unchanged':True,'errors':[],'native_error_events':[]}
            (p/'state.json').write_text(json.dumps(state));(p/'native_errors.json').write_text('[{"late":true}]')
            with self.assertRaisesRegex(ValueError,'late runtime'):contract.validate_result(p,{})

    def test_required_admission_argument_and_no_isaac_preflight_import(self):
        parser=function(HERE/'run_query.py','main')
        import_lines=[n.lineno for n in ast.walk(parser)if isinstance(n,ast.ImportFrom)and (n.module or '').startswith('isaac')]
        early_return=next(n.lineno for n in ast.walk(parser)if isinstance(n,ast.If)and isinstance(n.test,ast.Attribute)and n.test.attr=='preflight_only')
        self.assertTrue(all(x>early_return for x in import_lines))
        calls=[n for n in ast.walk(parser)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='add_argument'and any(isinstance(x,ast.Constant)and x.value=='--admission'for x in n.args)]
        self.assertEqual(len(calls),1);self.assertTrue(any(x.arg=='required'and x.value.value is True for x in calls[0].keywords))


if __name__=='__main__':unittest.main()
