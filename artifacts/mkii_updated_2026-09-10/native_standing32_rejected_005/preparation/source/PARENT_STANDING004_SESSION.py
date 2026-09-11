"""Raw native held-target 400Hz external PD session; standalone standing has one reset.
No actor, DirectRLEnv, automatic reset, motor adapter or implicit drive is installed.
"""
from pathlib import Path
import json,time
import numpy as np
from standing_math import servo,classify_contacts
DT=.0025

def memory():
 result={}
 for f,keys in [('/proc/self/status',['VmRSS','VmHWM','VmSize']),('/proc/meminfo',['MemTotal','MemFree','MemAvailable','Cached','SReclaimable'])]:
  try:result[f]={k:v.strip()for line in Path(f).read_text().splitlines()if ':'in line for k,v in[line.split(':',1)]if k in keys}
  except OSError as e:result[f]=repr(e)
 try:
  import torch
  result['cuda_free_total']=list(torch.cuda.mem_get_info());result['torch_reserved']=torch.cuda.memory_reserved()
 except Exception as e:result['cuda_unavailable']=repr(e)
 return result

class NativeStandingSession:
 def __init__(self,view,contact_view,sim,model,native,geometry,output,save,device='cuda:0'):
  import warp as wp
  self.wp=wp;self.device=device;self.view=view;self.contact=contact_view;self.sim=sim;self.model=model;self.native=native;self.geometry=geometry;self.output=Path(output);self.save=save
  self.n=view.count;self.names=native['joint_names'];self.body_names=native['body_names'];self.roots=[str(x).rsplit('/',1)[0]for x in view.prim_paths]
  self.ids=wp.array(np.arange(self.n),dtype=wp.uint32,device=device);self.zero=np.zeros((self.n,18),np.float32)
  gains=json.loads((Path(__file__).parent/'servo_candidate.json').read_text());order=[gains['joint_names'].index(x)for x in self.names]
  self.kp=np.asarray(gains['stiffness_nm_per_rad'],np.float32)[order];self.kd=np.asarray(gains['damping_nm_s_per_rad'],np.float32)[order]
  paths=list(contact_view.sensor_paths);expected={r+'/'+b for r in self.roots for b in self.body_names}
  if len(paths)!=19*self.n or set(paths)!=expected or contact_view.filter_count!=1:raise ValueError('Contact sensor paths/filter count differ')
  self.sensor_map=[]
  for p in paths:
   root,body=p.rsplit('/',1);self.sensor_map.append((self.roots.index(root),body))
  self.save(self.output/'contact_view.json',{'sensor_paths':paths,'filter_paths':list(contact_view.filter_paths),'capacity':contact_view.max_contact_data_count,'scope':'Native raw view freshly read after each explicit step; no fabricated per-sensor clock age.'})
  self.count=0;self.captured_count=0;self.initial_counter=sim.get_physics_step_count();self.control=0;self.reset_count=0;self.counter=sim.get_physics_step_count();self.rows=[];self.controls=[];self.files=[];self.failure=None
  self.contact_stream=(self.output/'contacts.jsonl').open('w');self.start=time.monotonic();self.memory_start=memory()
  self.current=None;self.last_control_rows=[]
 def array(self,x):return self.wp.array(np.asarray(x,dtype=np.float32),dtype=self.wp.float32,device=self.device)
 def get(self,name):return np.array(getattr(self.view,name)().numpy(),copy=True)
 def snap(self,row=None):
  row={}if row is None else row
  row['active_getter']='update_articulations_kinematic';self.sim.physics_sim_view.update_articulations_kinematic()
  for key,getter in [('joint_position_rad','get_dof_positions'),('joint_velocity_rad_s','get_dof_velocities'),('link_pose_xyzw','get_link_transforms'),('root_pose_xyzw','get_root_transforms'),('root_com_velocity','get_root_velocities')]:
   row['active_getter']=getter;row[key]=self.get(getter)
  row.pop('active_getter');return row
 def set_force(self,x):
  self.view.set_dof_actuation_forces(self.array(x),self.ids)
  actual=self.get('get_dof_actuation_forces')
  if not np.array_equal(actual,x):
   self.save(self.output/'failed_force_input.json',{'count':self.count,'command':x.tolist(),'actual':actual.tolist()});raise ValueError('Native external input mismatch')
  return actual
 def reset_canonical(self):
  if self.reset_count or self.count:raise ValueError('Standing permits only one initial reset before controlled physics')
  pre=self.snap();root=pre['root_pose_xyzw'].copy();root[:,:3]=0.;root[:,3:]=[0,0,0,1]
  # Authored root is the bottom-plate frame. Grid placement is declared, no yaw changes.
  for e,path in enumerate(self.roots):
   index=0 if path=='/Robot'else int(path.rsplit('_',1)[1]);root[e,:3]=[(index%8)*2.,(index//8)*2.,self.geometry.meta['reset_plate_m']]
  self.set_force(self.zero);self.view.set_root_transforms(self.array(root),self.ids);self.view.set_root_velocities(self.array(np.zeros((self.n,6))),self.ids)
  self.view.set_dof_positions(self.array(self.zero),self.ids);self.view.set_dof_velocities(self.array(self.zero),self.ids)
  self.view.set_dof_position_targets(self.array(self.zero),self.ids);self.view.set_dof_velocity_targets(self.array(self.zero),self.ids)
  row=self.snap();targets=self.get('get_dof_position_targets');veltarget=self.get('get_dof_velocity_targets');external=self.get('get_dof_actuation_forces')
  self.save(self.output/'initial_reset.json',{'scope':'Single generalized state reset after SDK warmup; solver warmstart cache is not claimed cold.', 'counter_before':self.counter,'counter_after':self.sim.get_physics_step_count(),'pre_reset':{k:v.tolist()for k,v in pre.items()},'post_reset':{k:v.tolist()for k,v in row.items()},'position_targets':targets.tolist(),'velocity_targets':veltarget.tolist(),'native_external':external.tolist(),'requested_root':root.tolist()})
  if self.sim.get_physics_step_count()!=self.counter:raise ValueError('Initial reset advanced physics')
  for key,want in [('joint_position_rad',self.zero),('joint_velocity_rad_s',self.zero),('root_pose_xyzw',root),('root_com_velocity',np.zeros((self.n,6)))]:np.testing.assert_allclose(row[key],want,atol=2e-6,rtol=0)
  for a in [targets,veltarget,external]:np.testing.assert_allclose(a,self.zero,atol=2e-6,rtol=0)
  self.previous_q=row['joint_position_rad'].copy();self.reset_count=1;self.current=row
  self.current.update(joint_target_rad=self.zero.copy(),computed_torque_nm=self.zero.copy(),applied_torque_nm=self.zero.copy(),distal_contact=np.zeros((self.n,6),bool),nonfoot_contact=np.zeros(self.n,bool),contact_valid=np.zeros(self.n,bool),terminated=np.zeros(self.n,bool),truncated=np.zeros(self.n,bool),interval_angle_rate_rad_s=self.zero.copy(),interval_valid=np.zeros(self.n,bool))
  return self.observe()
 def observe(self):
  if self.current is None:raise ValueError('Initial reset required')
  # All buffers are private copies; no caller may mutate the next servo input.
  return {k:v.copy()if isinstance(v,np.ndarray)else v for k,v in self.current.items()}
 def step_control(self,target):
  if self.failure:raise RuntimeError('First failure latched: '+self.failure)
  if self.reset_count!=1:raise ValueError('Initial reset required')
  target=np.asarray(target,dtype=np.float32).copy()
  if target.shape!=self.zero.shape or not np.isfinite(target).all():raise ValueError('Invalid held target')
  if not np.all(target>=np.asarray(self.native['limits'])[:,:,0])or not np.all(target<=np.asarray(self.native['limits'])[:,:,1]):raise ValueError('Held target outside native limits')
  # Future learner adds its versioned target governor; this session does not reinterpret actions.
  start=time.monotonic();row=None;control_rows=[]
  try:
   for sub in range(8):
    row={'control_index':np.array(self.control),'substep_index':np.array(sub),'sequence':np.array(self.count),'explicit_counter':np.array(self.sim.get_physics_step_count()),'active_getter':'get_dof_positions'}
    q=self.get('get_dof_positions');row['pre_joint_position_rad']=q;row['active_getter']='get_dof_velocities';dq=self.get('get_dof_velocities');row['pre_joint_velocity_rad_s']=dq
    raw,applied,ceiling=servo(q,dq,target,self.kp,self.kd);row['computed_torque_nm']=raw;row['applied_torque_nm']=applied;row['active_getter']='set_force';native_pre=self.set_force(applied)
    row={'pre_joint_position_rad':q,'pre_joint_velocity_rad_s':dq,'computed_torque_nm':raw,'applied_torque_nm':applied,'joint_target_rad':target.copy(),'native_input_pre_nm':native_pre,'sequence':np.array(self.count),'active_getter':'sim.step'}
    self.sim.step(render=False);self.count=self.sim.get_physics_step_count()-self.initial_counter
    row['explicit_counter']=np.array(self.sim.get_physics_step_count());row=self.snap(row)
    row['active_getter']='get_dof_actuation_forces';post=self.get('get_dof_actuation_forces');row['native_input_post_nm']=post
    row['active_getter']='get_dof_projected_joint_forces';reaction=self.get('get_dof_projected_joint_forces');row['projected_joint_reaction_nm']=reaction
    row['active_getter']='get_contact_data'
    data=[np.array(x.numpy(),copy=True)for x in self.contact.get_contact_data(DT)];row.pop('active_getter')
    try:contact=classify_contacts(data,self.sensor_map,row['link_pose_xyzw'],self.geometry,self.n)
    except Exception:
     np.savez_compressed(self.output/'failed_contact_buffer.npz',**{k:v for k,v in zip(['force','point','normal','separation','counts','starts'],data)});raise
    minimum,non_toe=self.geometry.clearance(row['link_pose_xyzw']);counter=self.sim.get_physics_step_count()
    row.update(pre_joint_position_rad=q,pre_joint_velocity_rad_s=dq,computed_torque_nm=raw,applied_torque_nm=applied,effort_ceiling_nm=ceiling,native_input_pre_nm=native_pre,native_input_post_nm=post,projected_joint_reaction_nm=reaction,
      joint_target_rad=target.copy(),interval_angle_rate_rad_s=(row['joint_position_rad'].astype(float)-self.previous_q)/DT,interval_valid=np.ones(self.n,bool),contact_valid=np.ones(self.n,bool),
      distal_contact=contact['distal_contact'],distal_force_world_n=contact['distal_force_world'],nonfoot_contact=contact['nonfoot_contact'],nonfoot_force_world_n=contact['nonfoot_force_world'],minimum_mesh_floor_m=minimum,minimum_non_toe_floor_m=non_toe,
      explicit_counter=np.array(counter),sequence=np.array(self.count-1),control_index=np.array(self.control),substep_index=np.array(sub),time_s=np.array(self.count*DT),
      terminated=np.zeros(self.n,bool),truncated=np.zeros(self.n,bool))
    limits=np.asarray(self.native['limits']);maxvel=np.asarray(self.native['native_max_velocity'])
    row['joint_limit_violation']=np.any((row['joint_position_rad']<limits[:,:,0]-2e-6)|(row['joint_position_rad']>limits[:,:,1]+2e-6),axis=1)
    row['joint_speed_violation']=np.any(abs(row['joint_velocity_rad_s'])>maxvel+2e-6,axis=1)
    row['terminated']=(row['root_pose_xyzw'][:,2]<.055)|row['joint_limit_violation']|row['joint_speed_violation']|row['nonfoot_contact']|(non_toe<-.001)
    self.rows.append({k:np.asarray(v).copy()for k,v in row.items()});control_rows.append(self.rows[-1]);self.captured_count+=1
    self.contact_stream.write(json.dumps({'sequence':self.count-1,'explicit_counter':counter,'patches':contact['patches']},allow_nan=False)+'\n')
    self.current=row;self.previous_q=row['joint_position_rad'].astype(float).copy()
    if counter!=self.counter+1:raise ValueError('Native physics counter mismatch')
    self.counter=counter
    if any(not np.isfinite(v).all()for v in row.values()):raise ValueError('Nonfinite native standing sample')
    if np.max(abs(applied))>1.60001:raise ValueError('Applied software cap exceeded')
    if row['terminated'].any():raise ValueError('Measured plate/joint/clearance/nonfoot gate failed; actual terminal sample retained')
   self.last_control_rows=[{k:v.copy()for k,v in r.items()}for r in control_rows]
   self.control+=1;self.controls.append({k:np.asarray(v).copy()for k,v in self.current.items()})
   if len(self.rows)>=800:self.flush()
   return self.observe()
  except BaseException as e:
   self.failure=repr(e);self.count=self.sim.get_physics_step_count()-self.initial_counter
   if row is not None:
    partial={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in row.items()};partial.update(actual_explicit_counter=self.sim.get_physics_step_count(),actual_controlled_steps=self.count,error=self.failure)
    self.save(self.output/'failed_partial_step.json',partial)
   self.save(self.output/'session_failure.json',{'error':self.failure,'steps':self.count,'controls':self.control});self.flush();raise
 def observe_control_substeps(self):
  if len(self.last_control_rows)!=8:raise ValueError('No complete eight-substep control yet')
  return [{k:v.copy()for k,v in r.items()}for r in self.last_control_rows]
 def flush(self):
  if self.rows:
   name=f'substeps_{len(self.files):03d}.npz';np.savez_compressed(self.output/name,**{k:np.stack([r[k]for r in self.rows])for k in self.rows[0]});self.files.append(name);self.rows=[]
  self.contact_stream.flush()
 def close(self):
  try:self.set_force(self.zero)
  finally:
   self.flush();self.contact_stream.close()
   if self.controls:np.savez_compressed(self.output/'control_trace.npz',**{k:np.stack([r[k]for r in self.controls])for k in self.controls[0]})
   self.save(self.output/'session.json',{'steps':self.count,'controls':self.control,'reset_count':self.reset_count,'failure':self.failure,'substep_files':self.files,'body_names':self.body_names,'joint_names':self.names,'root_paths':self.roots,'memory_start':self.memory_start,'memory_end':memory(),'wall_s':time.monotonic()-self.start,'captured_steps':self.captured_count,'all_rows_recorded':self.captured_count==self.count})
