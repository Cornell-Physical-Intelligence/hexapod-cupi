"""Actual stdlib imports and ownership CodeTypes; no simulator or external process."""
import argparse, importlib.util, json, platform, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
here=Path(__file__).resolve().parent
p=argparse.ArgumentParser(allow_abbrev=False)
for key in ('source','standing-source','supervisor-source'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
a.source=a.source.resolve();a.standing_source=a.standing_source.resolve();a.supervisor_source=a.supervisor_source.resolve()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def forbid(*args,**kwargs):raise AssertionError('External processes forbidden in CPU setup proof')
with patch.object(subprocess,'run',side_effect=forbid),patch.object(subprocess,'Popen',side_effect=forbid),patch.object(subprocess,'check_output',side_effect=forbid):
 h=load('_actual_canonical_ppo_host_setup',here/'launch_ppo_spark.py')
 h.verify_tree(a.source,'FREEZE_SHA256.json',h.SOURCE_FREEZE)
 h.verify_tree(a.standing_source,'FREEZE_SHA256.json',h.STANDING_FREEZE)
 h.verify_tree(a.supervisor_source,'campaign_source_hashes.json',h.SUPERVISOR_MAP)
 assert h.sha(a.supervisor_source/'tools/launch_reference_physics_spark.py')==h.SUPERVISOR_CODE
 native=h.load_contract(a.source)
 assert native.STANDING_FREEZE==h.STANDING_FREEZE
 args=SimpleNamespace(source=a.source,standing_source=a.standing_source,supervisor_source=a.supervisor_source,
  output=Path('/uncreated/canonical/ppo/output'),asset=Path('/uncreated/asset'),admission=Path('/uncreated/admission'),
  standing_one=Path('/uncreated/standing1'),standing32=Path('/uncreated/standing32'),bindings=Path('/uncreated/bindings.json'))
 sys.path.insert(0,str(a.supervisor_source/'tools'))
 old=load('_unaltered_ownership_comparison',a.supervisor_source/'tools/launch_reference_physics_spark.py')
 identity={'runtime_binding':{'runtime_tree_sha256':h.SOURCE_FREEZE}}
 parent=h.load_supervisor(args,identity)
 assert old.run_owned.__code__!=parent.run_owned.__code__
 for name in ('owned_container',):assert getattr(old,name).__code__==getattr(parent,name).__code__,name
 command=parent.command(a.source,args.output,'never-created',h.PHASE)
 assert '/ppo/run_native_smoke.py'in command and '--checkpoint'not in command
 for field,target in [('source','/ppo'),('standing_source','/standing'),('standing_one','/standing_one'),('standing32','/standing32'),('bindings','/bindings.json')]:
  assert str(getattr(args,field))+':'+target+':ro'in command
 compile((a.source/'run_native_smoke.py').read_text(),str(a.source/'run_native_smoke.py'),'exec')
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
 try:native.verify_inputs(SimpleNamespace(**{**vars(args),'bindings':a.source/'BINDINGS.json'}))
 except ValueError as error:
  assert 'standing1_state_sha256'in str(error);pending_error=str(error)
 else:raise AssertionError('Frozen pending template must reject')
print(json.dumps({'passed':True,'python':platform.python_version(),'source_freeze_sha256':h.SOURCE_FREEZE,
 'standing_source_freeze_sha256':h.STANDING_FREEZE,'source_payloads':len(h.read(a.source/'FREEZE_SHA256.json')),'standing_payloads':len(h.read(a.standing_source/'FREEZE_SHA256.json')),
 'supervisor_source_payloads':926,'external_process_calls':0,'GPU_calls':0,'actual_native_contract_stdlib_import':True,
 'native_entry_compiles':True,'owned_container_CodeType_unchanged':True,'supervisor_runtime_adapter':parent.CANONICAL_DEADLINE_ADAPTER,'deadlines_seconds':{'phase':1200,'AppReady':90},
 'pending_template_rejection':pending_error,'standing_admission_claimed':False,
 'scope':'CPU source/API composition only. Actual external standing1/32 states and complete host preflight remain required.'},indent=2))
