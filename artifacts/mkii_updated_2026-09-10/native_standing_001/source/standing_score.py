"""Recompute unchanged quiet bounds plus explicit full400Hz contact/motor evidence."""
from pathlib import Path
import numpy as np
from standing_math import servo
from quiet_metrics import quiet_metrics,QUIET_GATES
from standing_contract import read,STEPS,CONTROLS,SETTLE,DT

def score(directory):
 d=Path(directory);s=read(d/'session.json');names=s['joint_names'];n=len(s['root_paths'])
 with np.load(d/'control_trace.npz')as z:data={k:z[k]for k in z.files}
 if len(data['sequence'])!=CONTROLS:raise ValueError('Incomplete control trace')
 data['position_world_m']=data['root_pose_xyzw'][...,:3]
 data['quaternion_world_wxyz']=data['root_pose_xyzw'][...,[6,3,4,5]]
 result={'num_envs':n,'controls':len(data['sequence']),'substeps':0,'gates':QUIET_GATES,'replicas':[],'scope':'Provisional simulation standing screen; no hardware calibration or walking admission.'}
 maxap=np.zeros(n);rawmax=np.zeros(n);sat=np.zeros((n,18),int);postcount=0;badcontact=np.zeros(n,int);nonfoot=np.zeros(n,int);clear=np.full(n,np.inf);height=np.full(n,np.inf);qdelta=np.zeros(n);counter=int(read(d/'initial_reset.json')['counter_after']);previous=None;previous_dq=None;control_end=[]
 native=read(d/'native_readback.json');limits=np.asarray(native['limits']);maxvel=np.asarray(native['native_max_velocity']);g=read(Path(__file__).parent/'servo_candidate.json');order=[g['joint_names'].index(x)for x in names];kp=np.asarray(g['stiffness_nm_per_rad'],np.float32)[order];kd=np.asarray(g['damping_nm_s_per_rad'],np.float32)[order]
 for f in s['substep_files']:
  with np.load(d/f)as z:r={k:z[k]for k in z.files}
  m=len(r['sequence']);start=result['substeps']
  if not np.array_equal(r['sequence'],np.arange(start,start+m)):raise ValueError('Raw sequence mismatch')
  if not np.array_equal(r['control_index'],np.arange(start,start+m)//8)or not np.array_equal(r['substep_index'],np.arange(start,start+m)%8):raise ValueError('Raw8-substep ordering mismatch')
  if counter is not None and r['explicit_counter'][0]!=counter+1 or np.any(np.diff(r['explicit_counter'])!=1):raise ValueError('Native counter recurrence mismatch')
  counter=int(r['explicit_counter'][-1]);result['substeps']+=m
  if not np.allclose(r['time_s'],(r['sequence']+1)*DT,atol=1e-12,rtol=0):raise ValueError('Raw timestamp differs')
  if not r['contact_valid'].all()or not r['interval_valid'].all():raise ValueError('Invalid raw observation history/contact')
  if not all(np.isfinite(v).all()for v in r.values()):raise ValueError('Nonfinite raw state')
  if np.any((r['joint_position_rad']<limits[None,:,:,0]-2e-6)|(r['joint_position_rad']>limits[None,:,:,1]+2e-6))or np.any(abs(r['joint_velocity_rad_s'])>maxvel[None]+2e-6):raise ValueError('Actual measured joint/rate bound exceeded')
  if not np.array_equal(r['native_input_pre_nm'],r['applied_torque_nm']):raise ValueError('Raw native external input mismatch')
  if np.any(r['joint_target_rad']!=0):raise ValueError('Standing neutral target changed')
  q=r['joint_position_rad'].astype(float)
  if previous is None:
   reset=read(d/'initial_reset.json')['post_reset'];previous=np.asarray(reset['joint_position_rad']);previous_dq=np.asarray(reset['joint_velocity_rad_s'])
  if not np.array_equal(r['pre_joint_position_rad'],np.concatenate([previous[None],q[:-1]]))or not np.array_equal(r['pre_joint_velocity_rad_s'],np.concatenate([previous_dq[None],r['joint_velocity_rad_s'][:-1]])):raise ValueError('Servo input not previous native post-step state')
  a,b,c=servo(r['pre_joint_position_rad'].reshape(-1,18),r['pre_joint_velocity_rad_s'].reshape(-1,18),r['joint_target_rad'].reshape(-1,18),kp,kd)
  for key,want in [('computed_torque_nm',a),('applied_torque_nm',b),('effort_ceiling_nm',c)]:
   if not np.array_equal(r[key],want.reshape(m,n,18)):raise ValueError('Raw provisional servo recurrence differs:'+key)
  dq=np.diff(np.concatenate([previous[None],q]),axis=0)/DT
  if not np.allclose(dq,r['interval_angle_rate_rad_s'],atol=1e-8,rtol=0):raise ValueError('Interval angle channel differs')
  previous=q[-1];previous_dq=r['joint_velocity_rad_s'][-1];qdelta=np.maximum(qdelta,np.max(abs(dq),axis=(0,2)))
  maxap=np.maximum(maxap,np.max(abs(r['applied_torque_nm']),axis=(0,2)));rawmax=np.maximum(rawmax,np.max(abs(r['computed_torque_nm']),axis=(0,2)))
  height=np.minimum(height,np.min(r['root_pose_xyzw'][...,2],axis=0))
  nonfoot+=r['nonfoot_contact'].sum(0);clear=np.minimum(clear,np.min(r['minimum_non_toe_floor_m'],axis=0))
  post=r['control_index']>=SETTLE;postcount+=int(post.sum())
  if post.any():
   sat+=(abs(r['computed_torque_nm'][post])>1.6).sum(0)
   badcontact+=(~r['distal_contact'][post].all(-1)).sum(0)
  end=r['substep_index']==7
  for index in np.flatnonzero(end):
   control=int(r['control_index'][index])
   for key in data:
    if key in r and not np.array_equal(data[key][control],r[key][index]):raise ValueError('Control endpoint differs:'+key)
 if result['substeps']!=STEPS or postcount!=(CONTROLS-SETTLE)*8:raise ValueError('Incomplete400Hz acquisition')
 for e in range(n):
  q=quiet_metrics(data,e,SETTLE,names,.02)
  physical={'max_applied_all_substeps_nm':float(maxap[e]),'max_requested_all_substeps_nm':float(rawmax[e]),'max_requested_saturation_fraction_400hz':float((sat[e]/postcount).max()),'mean_requested_saturation_fraction_400hz':float(sat[e].sum()/(postcount*18)),
   'post_settle_missing_six_toe_substeps':int(badcontact[e]),'all_controlled_nonfoot_substeps':int(nonfoot[e]),'minimum_non_toe_mesh_floor_m':float(clear[e]),'minimum_plate_height_m':float(height[e]),'interval_angle_rate_abs_max_rad_s':float(qdelta[e])}
  failed=[]
  if maxap[e]>1.60001:failed.append('applied_cap')
  if (sat[e]/postcount).max()>.005 or sat[e].sum()/(postcount*18)>.005:failed.append('requested_saturation_400hz')
  if badcontact[e]:failed.append('six_toe_support')
  if nonfoot[e]:failed.append('nonfoot_contact')
  if clear[e]<-.001:failed.append('non_toe_floor_clearance')
  if height[e]<.055:failed.append('plate_height')
  if q['window_duration_s']<10:failed.append('quiet_duration')
  result['replicas'].append({'env':e,'quiet':q,'physical':physical,'failed_physical_bounds':failed,'pass':q['pass']and not failed})
 result['all_pass']=all(x['pass']for x in result['replicas']);return result
