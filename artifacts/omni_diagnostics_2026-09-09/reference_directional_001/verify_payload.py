"""Portable read-only payload/provenance verifier; standard library, no GPU or dispatch."""
from pathlib import Path
import ast,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def verify_tree(path,manifest,expected=None):
 p=path/manifest
 if expected is not None and sha(p)!=expected:raise ValueError('Wrong bound manifest '+str(p))
 if any(f.is_symlink() for f in path.rglob('*')):raise ValueError('Symbolic payload substitution')
 actual={str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file() and f!=p}
 if actual!=read(p):raise ValueError('Changed/missing/extra payload '+str(path))
 return len(actual)
def main():
 total=verify_tree(ROOT,'BUNDLE_SHA256.json');r=read(ROOT/'RECONSTRUCTION.json');frozen={}
 for name,spec in r['copied_frozen_bundles'].items():
  frozen[name]=verify_tree(ROOT/name,spec['manifest'],spec['sha256']);assert frozen[name]==spec['payload_files']
 for name,h in r['copied_root_actual_review_sha256'].items():assert sha(ROOT/'root_actual_review'/name)==h
 audit=read(ROOT/'remote_audit.json');raw=ROOT/'raw'
 assert len(audit['raw_payloads'])==33
 for name,h in audit['raw_payloads'].items():assert sha(raw/name)==h,name
 assert audit['source_unchanged'] and audit['source_files']==930 and audit['source_manifest_sha256']==r['new_source_manifest_sha256']
 assert audit['admitted_assets_unchanged'] and audit['admitted_asset_files']==550
 assert sha(ROOT/r['admitted_asset_map'])==r['admitted_asset_map_sha256'] and len(read(ROOT/r['admitted_asset_map']))==550
 assert len(audit['owned_containers_absent'])==4
 for name,result in audit['owned_containers_absent'].items():assert result['returncode']==1 and 'no such object: '+name in result['stderr'].lower()
 restored=read(raw/'forecast_pause/restored.json');assert restored==audit['pause_restoration']
 assert restored['restored_unix']==1789030025.5451584
 assert set(restored['timers'])=={'stormscope-dispatch.timer','stormscope-scout.timer'}
 tree=ast.parse((ROOT/'guard/launch_guarded_remote.py').read_text())
 scripts=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 assert len(scripts)==1 and scripts[0].encode()==(raw/'forecast_pause/resume_forecasting.py').read_bytes()
 campaign=read(raw/'run/campaign.json');assert campaign['status']=='failed' and campaign['policy_training_started'] is False and campaign['stage2_complete'] is False
 assert set((raw/'run/jobs').glob('*.json'))=={raw/'run/jobs'/n for n in ('standing.json','reverse.json','standing_contact_data_audit.json','reverse_contact_data_audit.json')}
 state=read(raw/'run/reverse/state.json');assert state['status']=='running' and state['control_steps']==2400
 root=read(ROOT/'root_actual_review/report.json');replay=read(ROOT/'independent_replay.json')
 assert replay['original_gate_passed'] is False and root['original_gate_posthoc_replay']['passed'] is False
 assert replay['original50Hz_gap_m']==0.005875744391232729 and replay['original50Hz_gap_m']>.005
 assert replay['measured_qualified_landings']==11 and replay['quiet_duration_s']==12.46
 assert replay['later_cases_unrun']==['left_strafe','left_turn','forward_right_arc']
 print(json.dumps({'bundle_payloads_verified':total,'copied_frozen_payloads':frozen,'raw_payloads_verified':33,'root_audit_source_payloads':930,'root_audit_assets':550,'root_audit_owned_containers_absent':2,'restored_unix':restored['restored_unix'],'original_failed_runtime_preserved':True,'physical_reverse_admitted':False,'read_only':True,'GPU_launches':0,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
