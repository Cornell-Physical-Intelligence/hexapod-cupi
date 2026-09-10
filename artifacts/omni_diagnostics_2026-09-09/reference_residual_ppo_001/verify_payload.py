"""Portable read-only terminal payload verifier; no Isaac, Torch, SSH or dispatch."""
from pathlib import Path
import ast,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def verify_tree(path,manifest,bound=None):
 p=path/manifest
 if bound is not None and sha(p)!=bound:raise ValueError('Wrong frozen manifest '+str(p))
 if any(f.is_symlink() for f in path.rglob('*')):raise ValueError('Symbolic payload substitution')
 actual={str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file() and f!=p}
 if actual!=read(p):raise ValueError('Changed/missing/extra payload '+str(path))
 return len(actual)
def main():
 total=verify_tree(ROOT,'BUNDLE_SHA256.json');r=read(ROOT/'RECONSTRUCTION.json');frozen={}
 for name,spec in r['copied_frozen_bundles'].items():
  frozen[name]=verify_tree(ROOT/name,spec['manifest'],spec['sha256']);assert frozen[name]==spec['payload_files']
 for name,spec in r['referenced_maps'].items():
  p=ROOT/'references'/name;assert sha(p)==spec['sha256'] and len(read(p))==spec['payload_count']
 assert r['referenced_maps']['preparation001_manifest.json']['payload_count']==69
 a=read(ROOT/'actual/remote_audit.json');assert len(a['raw_payloads'])==25
 for name,h in a['raw_payloads'].items():assert sha(ROOT/'actual'/name)==h,name
 runmap={n[4:]:h for n,h in a['raw_payloads'].items() if n.startswith('run/')}
 assert len(runmap)==19 and runmap==read(ROOT/'compact_failure_and_fix/raw/complete_run_sha256.json')
 for name,h in runmap.items():
  p=ROOT/'compact_failure_and_fix/raw/run'/name
  if p.exists():assert sha(p)==h
 for key,count,ref in [('source009',926,'source009_manifest.json'),('bridge001',18,'bridge001_manifest.json'),('observation005',160,'observation005_manifest.json')]:
  assert a[key]['payload_files']==count and a[key]['unchanged'] and a[key]['manifest_sha256']==r['referenced_maps'][ref]['sha256']
 assert a['consumer001']['payload_files']==19 and a['consumer001']['unchanged'] and a['consumer001']['manifest_sha256']=='258c304f409373e49667bc7fb653026fcb0754c2c92d6dfd70eb04769c3e754b'
 assert a['host001']['payload_files']==3 and a['host001']['unchanged']
 assert a['admitted_assets_unchanged'] and a['admitted_asset_files']==550 and len(read(ROOT/'actual/run/inputs/study_before.sha256.json'))==550
 jobs=a['actual_jobs'];assert [j['phase'] for j in jobs]==['standing','smoke'] and all(j['cleanup_checked'] for j in jobs)
 assert jobs[0]['container_id'] and jobs[1]['container_id'] is None
 expected_names={j['container_name'] for j in jobs}|{jobs[0]['container_id']}
 assert set(a['owned_absence'])==expected_names
 for name,result in a['owned_absence'].items():assert result['returncode']==1 and 'no such object: '+name in result['stderr'].lower()
 pause=read(ROOT/'actual/forecast_pause/restored.json');assert pause==a['pause_restoration'] and pause['restored_unix']==1789030414.2352366
 assert sha(ROOT/'actual/forecast_pause/restored.json')==sha(ROOT/'compact_failure_and_fix/raw/forecast_pause_042/restored.json')
 tree=ast.parse((ROOT/'guard001/launch_guarded_remote.py').read_text());scripts=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 assert len(scripts)==1 and scripts[0].encode()==(ROOT/'actual/forecast_pause/resume_forecasting.py').read_bytes()
 campaign=read(ROOT/'actual/run/campaign.json');assert campaign['status']=='failed' and campaign['PPO_updates_completed']==0 and set(campaign['accepted_phases'])=={'standing'}
 state=read(ROOT/'actual/run/standing/state.json');assert state['status']=='completed' and state['control_steps']==1000 and state['gate']['passed']
 assert sha(ROOT/'actual/run/standing/state.json')==sha(ROOT/'actual/run/standing/admission.json')==campaign['accepted_phases']['standing']['admission_sha256']
 log=(ROOT/'actual/run/logs/smoke.log').read_text();assert '/outputs/cuda:0/campaign.json' in log and 'REFERENCE_SCREEN_APP_START' not in log and 'REFERENCE_SCREEN_APP_READY' not in log
 assert not any('/smoke/' in n or 'evaluate_' in n or n.endswith('.pt') for n in a['raw_payloads'])
 review=read(ROOT/'actual/review.json');assert review['PPO_updates']==0 and review['standing_physical_pass'] and review['standing_quiet_pass'] and review['quiet_passed_replicas']==32
 pref=read(ROOT/'actual/preflight002.json');assert pref['status']=='passed'
 assert pref['inputs']['consumer_freeze_sha256']==r['copied_frozen_bundles']['correction002/consumer']['sha256']
 assert pref['host_freeze_sha256']==r['copied_frozen_bundles']['correction002/host']['sha256'] and pref['guard_freeze_sha256']==r['copied_frozen_bundles']['correction002/guard']['sha256']
 assert pref['pause042_restoration']==pause and r['physical002_result_included'] is False
 print(json.dumps({'bundle_payloads_verified':total,'frozen_payloads':frozen,'actual_remote_payloads':25,'complete001_run_payloads':19,'standing32_physical_and_quiet_recomputed_pass':True,'original_parser_failure_preserved':True,'PPO_updates001':0,'owned_names_absent':2,'recorded_owned_IDs_absent':1,'smoke_ID_unknown':True,'pause042_restored_unix':pause['restored_unix'],'correction002_preparation_only':True,'preparation001_referenced_payloads':69,'read_only':True,'GPU_launches':0,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
