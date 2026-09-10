"""Read-only offline replay of actual pair001; writes only a fresh local review report."""
from pathlib import Path
import sys,json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent;RESULT=HERE.parent;SOURCE=RESULT.parent/'reference_pair_physics_adapter_001/source_pair_001'
sys.path.insert(0,str(SOURCE/'tools'))
from score_transfer import score_transfer
from screen_metrics import standing_screen,standing_quiet_review
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((RESULT/'remote_audit.json').read_text())
for f,h in audit['raw_payloads'].items():assert sha(RESULT/f)==h,f
sm=json.loads((SOURCE/'campaign_source_hashes.json').read_text())
for f,h in sm.items():assert sha(SOURCE/f)==h,f
assert len(sm)==934 and sha(SOURCE/'campaign_source_hashes.json')==audit['source_manifest_sha256']
with np.load(RESULT/'run/pair/trace.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
with np.load(RESULT/'run/pair/physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k] for k in z.files}
refs=json.loads((RESULT/'run/pair/reference_states.json').read_text());post=json.loads((RESULT/'run/pair/post_step_measurements.json').read_text())
state=json.loads((RESULT/'run/pair/state.json').read_text())
score=score_transfer(d,refs,substeps=sub)
assert score['proposed_criteria_met'] and state['diagnostic_result']['proposed_criteria_met']
assert len(d['time_s'])==1100 and len(post)==1100 and len(sub['time_s'])==8801
assert np.allclose(d['time_s'][:,0],np.arange(201,1301)*.02,atol=1e-7,rtol=0)
for k,row in enumerate(post):assert row['all_existing_proposed_measurement_bounds_met'] and abs(row['time_s']-d['time_s'][k,0])<1e-7
for raw,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[raw][8::8],d[control])
corner=[0,2,3,5];contacts=d['distal_contact'][:,0]&d['contact_point_valid'][:,0]
assert contacts[:,corner].all() and contacts[-3:].all()
assert not any(d[k].any() for k in ['terminated','truncated','shaft_contact','coxa_contact','femur_contact','base_contact'])
assert max(np.abs(sub['computed_torque_nm']).max(),np.abs(sub['applied_torque_nm']).max())<=1.6
pair=[1,4];hold=np.array([r['state']['mode']=='unloaded_hold' for r in refs]);mask=hold&(~contacts[:,pair]).all(1)
force=d['reaction_force_world_n'][mask,0].astype(np.float64);mean=force.mean(0)
maximum=tuple(int(v) for v in np.unravel_index(np.abs(sub['computed_torque_nm']).argmax(),sub['computed_torque_nm'].shape))
with np.load(RESULT/'run/standing/trace.npz',allow_pickle=False) as z:
 names=z['joint_names'].tolist();standing={k:z[k] for k in z.files if k not in ('joint_names','legs')}
sg=standing_screen(standing);sq=standing_quiet_review(standing,names)
assert sg['passed'] and sq['passed']
measured_extrema={k:float((min if k=='measured_support_margin_m' else max)(p[k] for p in post)) for k in ['measured_support_margin_m','body_displacement_m','body_rotation_rad','corner_drift_m','corner_max_slip_mps']}
# Preserve arithmetic differences; do not falsely claim JSON byte equality.
differences=[]
def compare(a,b,path='score'):
 if isinstance(a,dict) and isinstance(b,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'.'+k)
 elif isinstance(a,list) and isinstance(b,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
 elif isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)):
  if a!=b:differences.append({'field':path,'local':a,'recorded':b,'absolute_difference':abs(a-b)})
 else:assert a==b,(path,a,b)
compare(score,state['diagnostic_result'])
report={'scope':'Independent offline replay of the separate proposed four-support load-transfer diagnostic; no faster gait/PPO qualification',
 'raw_payloads_verified':len(audit['raw_payloads']),'source_files_verified':len(sm),'source_manifest_sha256':audit['source_manifest_sha256'],
 'standing32_physical_and_quiet_replay_passed':sg['passed'] and sq['passed'],'pair_proposed_criteria_replay_passed':score['proposed_criteria_met'],
 'complete_substep_and_control_endpoint_equality':True,'all1100post_step_bounds_present':True,'score_replay':score,'numeric_replay_differences':differences,
 'measured_bounds_extrema':measured_extrema,'steady_pair_unload_mean_each_foot_reaction_world_n_float64':mean.tolist(),
 'steady_pair_unload_total_reaction_n_float64':mean.sum(0).tolist(),
 'max_actual_substep_requested':{'index':maximum[0],'relative_pair_time_s':float(sub['time_s'][maximum[0]]),'physical_time_s':float(sub['time_s'][maximum[0]])+4.,'runtime_joint':str(sub['joint_names'][maximum[2]]),'torque_nm':float(sub['computed_torque_nm'][maximum])},
 'no_new_live_cleanup_claim':'Remote audit separately records both owned IDs/names absent and pause040 restored; this replay only verifies fetched bytes.',
 'practical_conclusion':'Both middle feet can be unloaded together under measured four-corner support and existing torque bounds; diagonal loads are unequal. New support pairs and actual translation still need separate measured proof.',
 'next_priority':'Bounded reverse/strafe/turn/combined-twist reference screen at proven low speeds; faster paired support sequence remains a separate follow-on.',
 'physics_qualification':False,'stage2_complete':False,'velocity_fidelity_qualified':False}
output=HERE/'report.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['raw_payloads_verified','source_files_verified','standing32_physical_and_quiet_replay_passed','pair_proposed_criteria_replay_passed','measured_bounds_extrema','max_actual_substep_requested','numeric_replay_differences']},indent=2))
