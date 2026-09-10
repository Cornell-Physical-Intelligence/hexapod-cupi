"""CPU-only actual import seam; no external subprocess, simulator or GPU calls."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,importlib.util,json,platform,subprocess,sys
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();here=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def forbid(*args,**kwargs):raise AssertionError('External processes are forbidden in this setup check')

with patch.object(subprocess,'run',side_effect=forbid),patch.object(subprocess,'Popen',side_effect=forbid),patch.object(subprocess,'check_output',side_effect=forbid):
 h=load('_actual_canonical_host',here/'launch_inspection_spark.py')
 supervisor=root/'tmp/reference_physics_adapter_009/source_009'
 sys.path.insert(0,str(supervisor/'tools'));old=load('_original_supervisor_for_review',supervisor/'tools/launch_reference_physics_spark.py')
 identity={'runtime_binding':{'runtime_tree_sha256':'a'*64,'scope':'canonical_native_inspection_only'}}
 args=SimpleNamespace(supervisor_source=supervisor,source=Path('/uncreated/canonical/source'),output=Path('/uncreated/canonical/output'),asset=Path('/uncreated/canonical/asset'))
 parent=h.load_supervisor(args,identity)
 for name in ('run_owned','owned_container'):assert getattr(parent,name).__code__==getattr(old,name).__code__,name
 assert parent.RUNTIME_TREE=='a'*64
 assert '/inspection/run_inspection.py' in parent.command(args.source,args.output,'never-created','inspection')
 native=root/'tmp/updated_native_inspection_002';c=load('_actual_native_inspection_contract',native/'inspection_contract.py')
 assert (c.SCHEMA,c.PHASE,c.STEPS,c.DT)==('canonical_native_inspection_v1','inspection',8,.0025)
 compile((native/'run_inspection.py').read_text(),str(native/'run_inspection.py'),'exec')
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
print(json.dumps({'passed':True,'python':platform.python_version(),'external_process_calls':0,'GPU_calls':0,'actual_native_contract_stdlib_import':True,'native_entry_compiles':True,'unmodified_supervisor_code_objects':True,'no_deadline_adapter':not any('deadline_adapter' in k for k in sys.modules),'scope':'Local CPU setup only; runtime identity here is a declared synthetic injection, not a frozen native source or native admission'},indent=2))
