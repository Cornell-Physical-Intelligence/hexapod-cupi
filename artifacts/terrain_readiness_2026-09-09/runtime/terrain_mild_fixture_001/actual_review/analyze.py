"""Read-only actual48-fixture result/provenance replay; no robot gait claim."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;RAW=HERE.parent;SOURCE=RAW.parent/'terrain_mild_fixture_adapter_001/source_mild_fixtures_001_final';sys.path.insert(0,str(SOURCE/'tools'))
from launch_mild_fixtures_spark import check_source
from mild_fixture_contract import catalog_identity,validate_result
from terrain_contact_evidence import audit_contact_log
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((RAW/'remote_audit.json').read_text())
for f,h in a['raw_payloads'].items():assert sha(RAW/f)==h,f
identity=check_source(SOURCE);assert identity['source_manifest_sha256']==a['source_manifest_sha256'];entries,geometry=catalog_identity(SOURCE)
state=json.loads((RAW/'run/fixtures/validation.json').read_text());gate=validate_result(state,SOURCE)
assert gate==a['fixture_gate'] and gate==json.loads((RAW/'run/campaign.json').read_text())['gate']
assert geometry==json.loads((RAW/'run/inputs/fixture_before.sha256.json').read_text())
log=audit_contact_log(RAW/'run/logs/fixtures.log');assert log['passed']
rows=state['runtime_rows'];family={}
for e,r in zip(entries,rows):
 key=e['family'];f=family.setdefault(key,{'fixtures':0,'train':0,'heldout':0,'max_ray_height_error_m':0.,'max_abs_sphere_bottom_gap_m':0.,'min_contact_fraction':1.})
 f['fixtures']+=1;f[e['split']]+=1;f['max_ray_height_error_m']=max(f['max_ray_height_error_m'],r['ray_caster_max_height_error_m'])
 f['max_abs_sphere_bottom_gap_m']=max(f['max_abs_sphere_bottom_gap_m'],max(abs(p['sphere_bottom_gap_m']) for p in r['probes']))
 f['min_contact_fraction']=min(f['min_contact_fraction'],min(p['contact_fraction_last_40_steps'] for p in r['probes']))
report={'scope':'Actual fixture geometry/ray/PhysX/probe import only; no robot, actor, perception or terrain walking',
 'raw_payloads_verified':len(a['raw_payloads']),'source_files_verified':identity['source_files'],'source_manifest_sha256':a['source_manifest_sha256'],
 'geometry_payloads_verified':len(geometry),'exact_host_contract_replay_equal':True,'actual48_fixture_smokes_passed':True,
 'ray_caster_rays':sum(r['ray_caster_rays'] for r in rows),'physx_queries':sum(len(r['physx_queries']) for r in rows),'physical_probes':sum(len(r['probes']) for r in rows),
 'outside_mesh_PhysX_misses':sum(r['physx_queries'][-1]['height_m'] is None for r in rows),
 'max_ray_caster_height_error_m':max(r['ray_caster_max_height_error_m'] for r in rows),
 'max_PhysX_height_error_m':max(abs(q['height_m']-q['expected_height_m']) for r in rows for q in r['physx_queries'] if q['height_m'] is not None),
 'max_abs_sphere_bottom_gap_m':max(abs(p['sphere_bottom_gap_m']) for r in rows for p in r['probes']),
 'minimum_probe_contact_fraction_last40steps':min(p['contact_fraction_last_40_steps'] for r in rows for p in r['probes']),
 'contact_data_log_audit':log,'families':family,'probe_integration_s':2.5,'initial_query_step_s':.005,'material':'nominal friction1,restitution0; no hardware calibration',
 'remote105source97geometry_owned_absence_and_pause045_restoration_verified':True,'new_live_lock_claim':False,
 'robot_validation_performed':False,'PPO_started':False,'perception_qualified':False,'ready_for_terrain_training':False,'Stage3_complete':False}
if (HERE/'report.json').exists():raise FileExistsError('Preserve existing review')
(HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
