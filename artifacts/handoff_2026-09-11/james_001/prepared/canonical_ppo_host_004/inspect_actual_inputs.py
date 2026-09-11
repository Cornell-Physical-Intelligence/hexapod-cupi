"""Exact local canonical assets and actual actuation composition, with no process/GPU.

This checks the preparatory native1 input contract and the disabled PPO template.
It deliberately cannot substitute completed standing1/32 admission evidence.
"""
import argparse,importlib.util,json,platform,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(allow_abbrev=False)
for k in ('source','standing-source','supervisor-source','asset','admission'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args()
for k in vars(a):setattr(a,k,getattr(a,k).resolve())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def forbid(*a,**k):raise AssertionError('External processes forbidden in local canonical input proof')
with patch.object(subprocess,'run',side_effect=forbid),patch.object(subprocess,'Popen',side_effect=forbid),patch.object(subprocess,'check_output',side_effect=forbid):
 h=load('_canonical_ppo_actual_input_setup',root/'launch_ppo_spark.py')
 h.verify_tree(a.source,'FREEZE_SHA256.json',h.SOURCE_FREEZE)
 h.verify_tree(a.standing_source,'FREEZE_SHA256.json',h.STANDING_FREEZE)
 h.verify_tree(a.supervisor_source,'campaign_source_hashes.json',h.SUPERVISOR_MAP)
 assert h.sha(root/'inputs/active_model.json')==h.SELECTOR_SHA256
 selector=h.read(root/'inputs/active_model.json')
 for path,bound in [('robot.usda',selector['usd_sha256']),('source/model.json',selector['model']['sha256']),('source/source.urdf',selector['urdf']['sha256'])]:assert h.sha(a.asset/path)==bound
 standing=load('_standing_native_input_composition',a.standing_source/'standing_contract.py')
 native=standing.verify_inputs(SimpleNamespace(asset=a.asset,admission=a.admission,output=Path('/uncreated/native-input-proof'),num_envs=1,standing_one=None))
 assert native['actuation_state_sha256']==h.ADMISSION_STATE and native['inspector_freeze_sha256']==h.STANDING_FREEZE
 assert native['training_allowed']is False and native['physical_admission']is False
 contract=h.load_contract(a.source)
 try:contract.verify_inputs(SimpleNamespace(asset=a.asset,admission=a.admission,standing_source=a.standing_source,standing_one=Path('/uncreated/standing1'),standing32=Path('/uncreated/standing32'),bindings=a.source/'BINDINGS.json',output=Path('/uncreated/ppo-input-proof')))
 except ValueError as e:
  assert 'standing32_state_sha256'in str(e);pending=str(e)
 else:raise AssertionError('Disabled pending binding template must reject')
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
print(json.dumps({'passed':True,'python':platform.python_version(),'source_payloads':len(h.read(a.source/'FREEZE_SHA256.json')),'standing_source_payloads':len(h.read(a.standing_source/'FREEZE_SHA256.json')),'source_freeze_sha256':h.SOURCE_FREEZE,'standing_freeze_sha256':h.STANDING_FREEZE,'asset_manifest_sha256':native['asset_manifest_sha256'],'actual_actuation_state_sha256':native['actuation_state_sha256'],'native_input_contract':native,'PPO_template_rejection':pending,'external_process_calls':0,'GPU_calls':0,'standing1_or32_admission_claimed':False,'full_PPO_host_dispatch_preflight_passed':False,'scope':'Exact source/assets/actual-actuation CPU composition; actual completed same-source standing1/32 and enabled external bindings still required'},indent=2))
