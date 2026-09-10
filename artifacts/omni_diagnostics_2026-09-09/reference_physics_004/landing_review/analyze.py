from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/omni_reference_wave_003'))
from wave_math import Swing
from serial_geometry import SerialGeometry,tensor
p=ROOT/'tmp/reference_physics_results_004/run/wave';r=json.loads((p/'reference_states.json').read_text());d=np.load(p/'trace.npz');last=r[-1]['result'];state=last['state'];i=3;t=state['desired_time_s'];spec=state['swing']
s=Swing(spec['start_s'],spec['duration_s'],spec['coefficients'][0],spec['endpoint_world_m'],spec['lift_m']);s.coeff=np.array(spec['coefficients'])
p0,v0,a0=s.sample(t);toe=d['reference_point_world_m'][-1,0,i];preload=np.array(state['reference_minus_measured_preload_world_m'][i]);endpoint=p0.copy();endpoint[2]=toe[2]+preload[2]
original=s.end-toe-preload;correction=endpoint-s.end
contacts=d['distal_contact'][:,0,i];end=len(contacts)-1;begin=end-1
while not contacts[begin-1]:begin-=1
z=d['reference_point_world_m'][:,0,i,2];peak=begin+z[begin:end].argmax()
g=SerialGeometry();names=d['joint_names'].tolist() if 'joint_names' in d else last['joint_names_runtime'];toleg=[names.index(n) for leg in g.names for n in leg]
q=d['joint_target_rad'][-1,0,toleg].reshape(6,3);fk=g.fk(tensor(q))[0].numpy();command_world=d['rotation_world_from_body'][-1,0]@fk[i]+d['position_world_m'][-1,0]
actualR=d['rotation_world_from_body'][-1,0];desiredR=np.array(state['desired_rotation_world_from_body']);dp=d['position_world_m'][-1,0]-np.array(state['desired_position_world_m']);rotation_offset=(actualR-desiredR)@fk[i]
report=dict(scope='Actual source004 first-failure diagnosis; no successor physics result',trace_sha256=hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest(),refstates_sha256=hashlib.sha256((p/'reference_states.json').read_bytes()).hexdigest(),failure=last['failure_reason'],confirmed_touchdowns_before_failure=state['confirmed_touchdowns'],leg='rf',time_s=t,swing_start_s=s.t0,phase=(t-s.t0)/s.duration,flight=dict(begin_index=int(begin),end_index=int(end-1),samples=int(end-begin),last_preflight_toe_z_m=float(z[begin-1]),peak_index=int(peak),peak_z_m=float(z[peak]),lift_m=float(z[peak]-z[begin-1]),returned_normal_force_n=float(d['normal_force_world_n'][-1,0,i,2]),actual_toe_speed_mps=float(np.linalg.norm(d['reference_point_velocity_world_mps'][-1,0,i]))),vectors=dict(actual_toe_world_m=toe.tolist(),original_endpoint_world_m=s.end.tolist(),current_virtual_world_m=p0.tolist(),initial_preload_world_m=preload.tolist(),candidate_landing_endpoint_world_m=endpoint.tolist(),original_error_vector_m=original.tolist(),candidate_correction_vector_m=correction.tolist(),virtual_minus_measured_m=(p0-toe).tolist(),body_translation_error_m=dp.tolist(),body_rotation_point_offset_m=rotation_offset.tolist(),command_fk_under_actual_pose_m=command_world.tolist(),actual_joint_deflection_point_error_m=(toe-command_world).tolist()),norms=dict(original_endpoint_preload_error_m=float(np.linalg.norm(original)),candidate_endpoint_correction_m=float(np.linalg.norm(correction)),virtual_endpoint_remaining_xy_m=float(np.linalg.norm((s.end-p0)[:2])),actual_endpoint_remaining_xy_m=float(np.linalg.norm((s.end-toe)[:2])),body_translation_error_m=float(np.linalg.norm(dp))),current_reference_velocity_world_mps=v0.tolist(),current_reference_acceleration_world_mps2=a0.tolist(),initial_stride_m=float(np.linalg.norm(s.end-s.coeff[0])),max_reference_velocity_rad_s=max(x['result']['diagnostics']['max_reference_velocity_rad_s'] for x in r if 'result' in x),max_reference_acceleration_rad_s2=max(x['result']['diagnostics']['max_reference_acceleration_rad_s2'] for x in r if 'result' in x))
(Path(__file__).parent/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
