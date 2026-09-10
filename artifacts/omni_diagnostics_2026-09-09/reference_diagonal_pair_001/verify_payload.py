"""Portable strict payload and historical evidence verification; no GPU or mutation."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,manifest,expected=None):
 p=root/manifest;m=json.loads(p.read_text())
 if expected is not None:assert sha(p)==expected
 actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f!=p}
 assert actual==set(m) and not any(f.is_symlink() for f in root.rglob('*'))
 for f,h in m.items():assert sha(root/f)==h,f
 return len(m)
def main():
 count=verify(R,'BUNDLE_SHA256.json');r=json.loads((R/'RECONSTRUCTION.json').read_text())
 for name,spec in r['copied_frozen_bundles'].items():assert verify(R/name,spec['manifest'],spec['sha256'])==spec['payloads']
 a=json.loads((R/'remote_audit.json').read_text());assert len(a['raw_payloads'])==56
 for f,h in a['raw_payloads'].items():assert sha(R/'raw'/f)==h,f
 assert a['source_manifest_sha256']==r['source_manifest_sha256'] and a['source_files']==934 and a['admitted_asset_files']==550
 assert a['source_unchanged'] and a['admitted_assets_unchanged'] and len(a['owned_containers_absent'])==6
 for token,value in a['owned_containers_absent'].items():assert value['returncode']!=0 and ('no such object' in value['stderr'].lower() or 'no such container' in value['stderr'].lower())
 assert json.loads((R/'raw/forecast_pause/restored.json').read_text())==a['pause_restoration']
 tree=ast.parse((R/'guard/launch_guarded_remote.py').read_text());scripts=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 assert len(scripts)==1 and scripts[0].encode()==(R/'raw/forecast_pause/resume_forecasting.py').read_bytes()
 review=json.loads((R/'actual_review/report.json').read_text())
 for case in ('lf_rr','lr_rf'):
  state=json.loads((R/'raw/run'/case/'state.json').read_text());v=review['cases'][case]
  assert state['status']=='completed' and state['control_steps']==1300 and state['diagnostic_controls']==1100
  assert state['diagnostic_result']==v['original_score']==v['independent_score'] and not v['numeric_replay_differences']
  assert state['proposed_diagnostic_criteria_met'] and state['diagnostic_result']['proposed_criteria_met']
  assert v['independent_post_measurement_replay']['rows_passed']==1100 and v['independent_post_measurement_replay']['first_error'] is None
  assert v['diagnostic_observer']['samples']==8801 and v['diagnostic_observer']['exact8_substeps_counters_timestamps_and_endpoint_parity']
  json.dumps(state,allow_nan=False)
 campaign=json.loads((R/'raw/run/campaign.json').read_text())
 assert campaign['status']=='completed' and set(campaign['cases'])=={'lf_rr','lr_rf'}
 for case,v in campaign['cases'].items():assert v['state_sha256']==sha(R/'raw/run'/case/'state.json') and v['proposed_criteria_met']
 assert not (R/'raw/run/pair').exists()
 print(json.dumps({'payloads_verified':count,'raw_payloads_verified':56,'source934_assets550_remote_proof_verified':True,'both_declared_static_diagonal_criteria_met':True,'walking_or_PPO_admission':False,'read_only':True},indent=2))
if __name__=='__main__':main()
