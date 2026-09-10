"""Independent exact-source completed paired001 raw replay and preserved rejection."""
from pathlib import Path
import json,hashlib,sys,traceback
import numpy as np
H=Path(__file__).resolve().parent;SOURCE=H/'oracle'
sys.path.insert(0,str(SOURCE))
from screen_metrics import standing_quiet_review,measured_flight_touchdowns,measured_progress
from pair_motion_metrics import score_pair_motion,substep_evidence,replay_pair_reference,numeric_tree_difference
from physics_substeps import displacement_check
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files if k not in ('joint_names','legs')}
def differences(a,b,path='root'):
 if isinstance(a,dict) and isinstance(b,dict):
  return sum([differences(a.get(k),b.get(k),path+'.'+k) for k in sorted(set(a)|set(b))],[])
 if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):return sum([differences(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))],[])
 if a==b:return []
 return [{'path':path,'local':a,'remote':b}]
run=H.parent/'run';state=json.loads((run/'paired_forward/state.json').read_text());file=run/'paired_forward/trace.npz'
if not file.exists():file=run/'paired_forward/partial_trace.npz'
data=arrays(file);refs=json.loads((run/'paired_forward/reference_states.json').read_text());physics=arrays(run/'paired_forward/physics_substeps.npz');checks=json.loads((run/'paired_forward/paired_poststep_checks.json').read_text())
with np.load(run/'standing/trace.npz',allow_pickle=False) as z:names=z['joint_names'].tolist()
report={'scope':'Actual paired001 independent frozen-source replay; no walking or policy pass inferred from partial data','inputs_sha256':{},'remote_status':state['status'],'failure':state.get('failure'),'error':state.get('error'),'controls':len(data['time_s']),'source_manifest_sha256':json.loads((H/'ORACLE_SOURCE_MAP.json').read_text())['source_manifest_sha256']}
for p in [file,run/'paired_forward/state.json',run/'paired_forward/reference_states.json',run/'paired_forward/physics_substeps.npz',run/'paired_forward/paired_poststep_checks.json',run/'standing/trace.npz']:
 report['inputs_sha256'][str(p.relative_to(H.parent))]=sha(p)
standing=standing_quiet_review(arrays(run/'standing/trace.npz'),names)
remote_standing=json.loads((run/'standing/admission.json').read_text())['gate']['all_replica_quiet']
report['standing_quiet_replay']={'passed':standing['passed'],'replicas':standing['num_envs'],'numeric_differences':differences(standing,remote_standing),'full_rows_file':'standing_quiet.json'}
(H/'standing_quiet.json').write_text(json.dumps(standing,indent=2,allow_nan=False)+'\n')
try:
 gate=score_pair_motion(data,refs,physics,checks,joint_names=names,failure=state.get('failure'))
 report['independent_gate']=gate;report['gate_differences']=differences(gate,state.get('gate'))
except Exception as error:
 report['independent_gate_error']=repr(error);report['independent_gate_traceback']=traceback.format_exc()
results=[r['result'] for r in refs if 'result' in r]
last=results[-1] if results else {};report['last_reference']=last
report['flight_events']=measured_flight_touchdowns(data,start_step=200)
report['400Hz']=substep_evidence(data,physics)
n=len(data['time_s'])
if n>300:
 report['actual_motion_prefix50Hz']=measured_progress(data,start_step=299,end_step=min(1499,n-1))
 report['motion_prefix400Hz']=displacement_check(physics,300,min(1500,n))
report['observer_export_interval_note']='Inherited generic observer moving_window_or_prefix uses200..1400controls; new pairedmotion begins300. Use explicitly labeled300..1500paired report above, preserving original raw observer bytes.'
selected=physics['computed_torque_nm'][1601:];flat=int(abs(selected).argmax());idx=np.unravel_index(flat,selected.shape)
report['poststartup_torque_peak']={'abs_nm':float(abs(selected).max()),'runtime_joint':names[idx[-1]],'physics_index':int(idx[0]+1601),'time_s':float(physics['time_s'][idx[0]+1601]),'applied_nm':float(physics['applied_torque_nm'][idx[0]+1601,idx[1],idx[2]])}
report['full_initial_torque_peak_nm']=float(abs(physics['computed_torque_nm']).max());report['hardware_startup_qualified']=False
report['last_contact_state']={'time_s':float(data['time_s'][-1,0]),'distal_contact':data['distal_contact'][-1,0].tolist(),'valid_points':data['contact_point_valid'][-1,0].tolist(),'normal_forces_world_n':data['normal_force_world_n'][-1,0].tolist()}
report['native_rate_fidelity_or_PPO_admitted']=False
axis=-data['rotation_world_from_body'][299,0,:,1].copy();axis[2]=0.;axis/=np.linalg.norm(axis)
report['measured_motion_position_average_forward_mps']=report['actual_motion_prefix50Hz']['measured_forward_displacement_m']/24
report['measured_stop_displacement_after_zero_request_m']=float(np.sum((data['position_world_m'][-1,0]-data['position_world_m'][1499,0])*axis))
report['reference_stop_latency_s']=last['state']['reference_quiet_time_s']-last['state']['stop_requested_time_s']
report['synthetic_first_phase_not_used_for_result']=True
(H/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['remote_status','failure','controls','poststartup_torque_peak','last_contact_state']},indent=2))
