"""Portable strict hashes and actual fixture result verification; no GPU or writes."""
from pathlib import Path
import ast,hashlib,json,math
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,name,expected=None):
 p=root/name;m=json.loads(p.read_text())
 if expected is not None:assert sha(p)==expected
 assert {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f!=p}==set(m)
 assert not any(f.is_symlink() for f in root.rglob('*'))
 for f,h in m.items():assert sha(root/f)==h,f
 return len(m)
def main():
 count=verify(R,'BUNDLE_SHA256.json');r=json.loads((R/'RECONSTRUCTION.json').read_text())
 for n,s in r['copied_frozen_bundles'].items():assert verify(R/n,s['manifest'],s['sha256'])==s['payloads']
 a=json.loads((R/'remote_audit.json').read_text());assert len(a['raw_payloads'])==11
 for f,h in a['raw_payloads'].items():assert sha(R/'raw'/f)==h,f
 m=R/'preparation/source_overlay/campaign_source_hashes.json';assert sha(m)==a['source_manifest_sha256']==r['source_manifest_sha256']
 assert len(json.loads(m.read_text()))==a['source_files']==105 and a['source_unchanged']
 geometry=json.loads((R/'preparation/GEOMETRY_SHA256.json').read_text())
 assert len(geometry)==a['fixture_geometry_files']==97 and a['fixture_geometry_unchanged']
 assert geometry==json.loads((R/'raw/run/inputs/fixture_before.sha256.json').read_text())
 assert len(a['owned_containers_absent'])==2
 for v in a['owned_containers_absent'].values():assert v['returncode']!=0 and ('no such object' in v['stderr'].lower() or 'no such container' in v['stderr'].lower())
 assert json.loads((R/'raw/forecast_pause/restored.json').read_text())==a['pause_restoration']
 assert 'Result=success' in a['unit'] and 'ActiveState=inactive' in a['unit']
 t=ast.parse((R/'guard/launch_guarded_remote.py').read_text());scripts=[n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 assert len(scripts)==1 and scripts[0].encode()==(R/'raw/forecast_pause/resume_forecasting.py').read_bytes()
 c=json.loads((R/'raw/run/campaign.json').read_text());s=json.loads((R/'raw/run/fixtures/validation.json').read_text());v=json.loads((R/'actual_review/report.json').read_text())
 assert c['status']=='completed' and c['all48_fixture_smokes_passed'] and c['gate']==a['fixture_gate']
 assert c['fixture_validation_sha256']==sha(R/'raw/run/fixtures/validation.json')
 assert s['status']=='completed' and s['cpu_only'] is False and s['all_fixture_smokes_passed'] and s['simulated_seconds']==2.5
 assert s['robot_validation_performed'] is False and s['ready_for_terrain_training'] is False
 ids=c['protocol']['fixture_ids'];assert len(ids)==len(set(ids))==48
 assert [x['id'] for x in s['cpu_rows']]==ids==[x['id'] for x in s['runtime_rows']]
 for cpu,row in zip(s['cpu_rows'],s['runtime_rows']):
  assert cpu['cpu_passed'] and cpu['collision_approximation']=='none' and cpu['meters_per_unit']==1 and cpu['up_axis']=='Z'
  assert row['passed'] and row['ray_caster_passed'] and row['ray_caster_rays']==81 and 0<=row['ray_caster_max_height_error_m']<2e-4
  assert len(row['physx_queries'])==4 and len(row['probes'])==3
  for q in row['physx_queries'][:3]:assert q['passed'] and math.isfinite(q['height_m']) and abs(q['height_m']-q['expected_height_m'])<2e-4
  q=row['physx_queries'][-1];assert q['passed'] and q['height_m'] is None and q['expected_height_m'] is None
  for p in row['probes']:assert p['passed'] and abs(p['sphere_bottom_gap_m'])<=.003 and .8<=p['contact_fraction_last_40_steps']<=1
 assert v['actual48_fixture_smokes_passed'] and v['exact_host_contract_replay_equal'] and not v['Stage3_complete']
 assert c['terminal_contact_log_audit']['passed'] and c['terminal_contact_log_audit']['incomplete_data_warning_count']==0
 assert sha(R/'raw/run/logs/fixtures.log')==c['terminal_contact_log_audit']['log_sha256']
 print(json.dumps({'payloads_verified':count,'raw_payloads_verified':11,'source105_geometry97_remote_proof_verified':True,'actual48_fixture_results_passed':True,'robot_or_terrain_policy_admission':False,'read_only':True},indent=2))
if __name__=='__main__':main()
