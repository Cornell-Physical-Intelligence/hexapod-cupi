"""CPU-only actual import seam; no external subprocess, simulator or GPU calls."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,json,platform,subprocess,sys
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();native=Path(sys.argv[2]).resolve();here=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def forbid(*args,**kwargs):raise AssertionError('External processes are forbidden in this setup check')

with patch.object(subprocess,'run',side_effect=forbid),patch.object(subprocess,'Popen',side_effect=forbid),patch.object(subprocess,'check_output',side_effect=forbid):
 h=load('_actual_canonical_actuation_host',here/'launch_actuation_spark.py')
 supervisor=root/'tmp/reference_physics_adapter_009/source_009'
 sys.path.insert(0,str(supervisor/'tools'));old=load('_original_supervisor_for_review',supervisor/'tools/launch_reference_physics_spark.py')
 identity={'runtime_binding':{'runtime_tree_sha256':'a'*64,'scope':'canonical_native_coordinate_effort_only'}}
 args=SimpleNamespace(supervisor_source=supervisor,source=Path('/uncreated/canonical/source'),output=Path('/uncreated/canonical/output'),asset=Path('/uncreated/canonical/asset'),admission=Path('/uncreated/canonical/admission'))
 parent=h.load_supervisor(args,identity)
 for name in ('run_owned','owned_container'):assert getattr(parent,name).__code__==getattr(old,name).__code__,name
 assert parent.RUNTIME_TREE=='a'*64
 cmd=parent.command(args.source,args.output,'never-created','actuation')
 assert '/actuation/run_actuation.py' in cmd and '/uncreated/canonical/admission:/admission:ro' in cmd
 c=load('_actual_native_actuation_contract',native/'actuation_contract.py')
 assert (c.SCHEMA,c.STEPS,c.EXPERIMENT_STEPS,c.DT)==('canonical_native_actuation_v1',8,2056,.0025)
 assert Path(c._parent.__file__).resolve()==native/'inspection_contract.py'
 compile((native/'run_actuation.py').read_text(),str(native/'run_actuation.py'),'exec')
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
print(json.dumps({'passed':True,'python':platform.python_version(),'external_process_calls':0,'GPU_calls':0,'actual_native_contract_stdlib_import':True,'source_relative_parent_contract_origin':True,'native_entry_compiles':True,'unmodified_supervisor_code_objects':True,'no_deadline_adapter':not any('deadline_adapter' in k for k in sys.modules),'scope':'Local CPU setup only; runtime identity here is a declared synthetic injection, not a frozen native source or native admission'},indent=2))
