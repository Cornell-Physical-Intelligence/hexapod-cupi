"""Bounded raw-view state-reset and external effort experiment; no servo or learning."""
import json,time
from pathlib import Path
import numpy as np
from actuation_contract import DT,BASELINE_STEPS,PULSE_STEPS,COAST_STEPS,PULSE_NM,BOUND_NM,BOUND_Q,BOUND_DQ,EXPERIMENT_STEPS
from floating_pulse import floating_mass
from inspect_core import validate_native_frames

def evidence_json(row):
 def safe(x):
  if isinstance(x,float)and not np.isfinite(x):return {'nonfinite':repr(x)}
  if isinstance(x,dict):return {k:safe(v)for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [safe(v)for v in x]
  return x
 return json.dumps(safe(row),allow_nan=False)+'\n'

def numeric_memory():
 result={'scope':'process/host/CUDA observations, not attribution to meshSDF'}
 try:
  result['proc_status']={k.strip():v.strip()for line in Path('/proc/self/status').read_text().splitlines()if ':'in line for k,v in [line.split(':',1)]if k in ['VmRSS','VmHWM','VmSize']}
  result['host_meminfo']={k.strip():v.strip()for line in Path('/proc/meminfo').read_text().splitlines()if ':'in line for k,v in [line.split(':',1)]if k in ['MemTotal','MemFree','MemAvailable','Cached','SReclaimable']}
 except OSError as e:result['host_unavailable']=repr(e)
 try:
  import torch
  result['cuda_free_total']=list(torch.cuda.mem_get_info());result['torch_reserved']=torch.cuda.memory_reserved()
 except Exception as e:result['cuda_unavailable']=repr(e)
 return result

class Experiment:
 def __init__(self,view,sim,model,native,output,snapshot,save,to_list,errors,device='cuda:0'):
  import warp as wp
  self.wp=wp;self.device=device;self.view=view;self.sim=sim;self.model=model;self.native=native
  self.output=Path(output);self.snapshot=snapshot;self.save=save;self.to_list=to_list;self.errors=errors
  self.names=list(native['joint_names']);self.zero=np.zeros((1,18),dtype=np.float32)
  self.root=np.asarray(to_list(view.get_root_transforms()),dtype=np.float32).copy()
  self.ids=wp.array([0],dtype=wp.uint32,device=device)
  self.count=0;self.reset_count=0;self.coordinate_count=0;self.case_count=0
  self.initial_counter=sim.get_physics_step_count();self.last_counter=self.initial_counter
  self.case_results=[];self.coordinate_results=[];self.max_excursion=0.;self.max_sdk_rate=0.;self.max_interval_rate=0.
  self.summary={'status':'running','errors':[],'memory_start':numeric_memory(),'scope':'State-reset coordinate/external-force diagnostic; no servo/standing/policy qualification',
   'reset_scope':'Explicit generalized state/input reset; not a fresh solver world or proof that warmstart caches are cleared',
   'native_input_semantics':'Pre-step native external actuation input readback; post-step buffer and projected/full reactions retained separately',
   'warmup_and_prefix_counter':self.initial_counter}
  self.stream=(self.output/'experiment.jsonl').open('w');self.reset_stream=(self.output/'reset_readbacks.jsonl').open('w')
  self.previous_q=None

 def arr(self,a):return self.wp.array(np.asarray(a,dtype=np.float32),dtype=self.wp.float32,device=self.device)

 def failure_record(self,kind,row):
  # Append rather than overwrite: final zero-force cleanup may also fail.
  with(self.output/(kind+'.jsonl')).open('a')as stream:
   stream.write(evidence_json(row))

 def set_force(self,force):
  x=np.asarray(force,dtype=np.float32)
  if x.shape!=(1,18)or not np.isfinite(x).all()or np.max(abs(x))>BOUND_NM:raise ValueError('Unsafe external command')
  record={'step_count':self.count,'physics_counter':self.sim.get_physics_step_count(),'requested_external_nm':x.tolist(),'operation':'set_dof_actuation_forces'}
  try:
   self.view.set_dof_actuation_forces(self.arr(x),self.ids)
   record['operation']='get_dof_actuation_forces'
   pre=np.asarray(self.to_list(self.view.get_dof_actuation_forces()),dtype=np.float32)
   record['actual_native_input']=pre.tolist()
   if not np.array_equal(pre,x):raise ValueError('Native pre-step actuation input differs from software command')
  except Exception as e:
   record['error']=repr(e);self.failure_record('failed_force_input',record);raise
  return pre

 def reset(self,label,q=None):
  q=self.zero.copy()if q is None else np.asarray(q,dtype=np.float32).reshape(1,18).copy()
  self.set_force(self.zero)
  self.view.set_root_transforms(self.arr(self.root),self.ids)
  self.view.set_root_velocities(self.arr(np.zeros((1,6))),self.ids)
  self.view.set_dof_positions(self.arr(q),self.ids)
  self.view.set_dof_velocities(self.arr(self.zero),self.ids)
  # Synchronize unused target buffers explicitly; zero gains remain unchanged.
  self.view.set_dof_position_targets(self.arr(q),self.ids)
  self.view.set_dof_velocity_targets(self.arr(self.zero),self.ids)
  row={'kind':'reset','label':label,'reset_index':self.reset_count,'requested_q':q.tolist()}
  try:
   row.update(self.snapshot(self.view,self.sim,self.count))
   for key,getter in [('root_pose_xyzw','get_root_transforms'),('root_com_velocity','get_root_velocities'),('position_target','get_dof_position_targets'),
                      ('velocity_target','get_dof_velocity_targets'),('native_external_input','get_dof_actuation_forces')]:
    row['active_getter']=getter;row[key]=self.to_list(getattr(self.view,getter)())
   row.pop('active_getter')
  except Exception as e:
   row['error']=repr(e);self.failure_record('failed_reset_readback',row);raise
  self.reset_stream.write(evidence_json(row));self.reset_stream.flush()
  if self.sim.get_physics_step_count()!=self.last_counter:raise ValueError('Reset secretly stepped physics')
  for k,expected in [('joint_position',q),('joint_velocity_sdk',self.zero),('position_target',q),('velocity_target',self.zero),('native_external_input',self.zero)]:
   np.testing.assert_allclose(row[k],expected,atol=2e-6,rtol=0,err_msg=k)
  np.testing.assert_allclose(row['root_pose_xyzw'],self.root,atol=2e-6,rtol=0)
  np.testing.assert_allclose(row['root_com_velocity'],0,atol=2e-6,rtol=0)
  validate_native_frames(row,self.native['body_names'],self.names,self.model)
  self.previous_q=q.copy();self.reset_count+=1
  return row

 def step(self,label,phase,force):
  command=np.asarray(force,dtype=np.float32).copy();pre=self.set_force(command)
  self.sim.step(render=False)
  row=self.snapshot(self.view,self.sim,self.count+1)
  row.update(kind='step',label=label,phase=phase,sequence=self.count,
   requested_external_nm=command.tolist(),software_applied_nm=command.tolist(),native_input_pre=pre.tolist())
  for key,getter in [('native_input_post','get_dof_actuation_forces'),('projected_joint_reaction_nm','get_dof_projected_joint_forces'),
                     ('incoming_joint_wrench_child_frame','get_link_incoming_joint_force'),('root_com_velocity','get_root_velocities')]:
   try:row[key]=self.to_list(getattr(self.view,getter)())
   except Exception as e:
    row['failed_getter']={'name':getter,'error':repr(e)}
    self.save(self.output/'failed_partial_step.json',row);raise
  q=np.asarray(row['joint_position']);dq=np.asarray(row['joint_velocity_sdk']);angle=(q-self.previous_q)/DT
  row['interval_angle_rate_rad_s']=angle.tolist();row['dt_s']=DT
  # Write the offending raw sample before a gate can raise.
  self.stream.write(evidence_json(row));self.stream.flush()
  self.count+=1;self.summary['step_count']=self.count
  if row['explicit_step_counter']!=self.last_counter+1:raise ValueError('Physics counter did not advance exactly once')
  self.last_counter=row['explicit_step_counter'];self.previous_q=q.copy()
  for k in ['joint_position','joint_velocity_sdk','link_pose_xyzw','link_com_velocity','native_input_pre','native_input_post','projected_joint_reaction_nm','incoming_joint_wrench_child_frame','root_com_velocity','interval_angle_rate_rad_s']:
   if not np.isfinite(np.asarray(row[k],dtype=float)).all():raise ValueError('Nonfinite '+k)
  self.max_excursion=max(self.max_excursion,float(abs(q).max()));self.max_sdk_rate=max(self.max_sdk_rate,float(abs(dq).max()));self.max_interval_rate=max(self.max_interval_rate,float(abs(angle).max()))
  if self.max_excursion>BOUND_Q or self.max_sdk_rate>BOUND_DQ or self.max_interval_rate>BOUND_DQ:raise ValueError('Declared diagnostic motion bound exceeded')
  validate_native_frames(row,self.native['body_names'],self.names,self.model)
  if self.errors:raise RuntimeError('Native PhysX error event')
  return row

 def run(self):
  try:
   self.reset('initial_baseline')
   for _ in range(BASELINE_STEPS):last=self.step('initial_baseline','baseline',self.zero)
   baseline_q=np.asarray(last['joint_position']);baseline_dq=np.asarray(last['joint_velocity_sdk'])
   if np.max(abs(baseline_q))>1e-4 or np.max(abs(baseline_dq))>1e-3:raise ValueError('Zero-input baseline is not suitable for this pulse experiment')
   for j,name in enumerate(self.names):
    for sign in [-1,1]:
     q=self.zero.copy();q[0,j]=sign*.01
     row=self.reset(f'coordinate:{name}:{sign}',q)
     self.coordinate_results.append({'joint':name,'sign':sign,'reset_index':row['reset_index'],'passed':True});self.coordinate_count+=1
   self.reset('mass_matrix_origin')
   M,r=floating_mass(self.model,self.names)
   native_M=np.asarray(self.to_list(self.view.get_generalized_mass_matrices()),dtype=float)
   # Root getters use COM velocity. Preserve both coordinate representations;
   # do not guess a native matrix convention from array shape alone.
   rootcom=r['details']['body']['com'];from floating_pulse import skew
   C=np.eye(24);C[:3,3:6]=skew(rootcom) # origin velocity = COM velocity + r_com x omega
   M_com=C.T@M@C
   origin_match=native_M.shape==(1,24,24)and np.allclose(native_M[0],M,rtol=2e-4,atol=2e-6)
   com_match=native_M.shape==(1,24,24)and np.allclose(native_M[0],M_com,rtol=2e-4,atol=2e-6)
   self.save(self.output/'mass_matrix_readback.json',{'native':native_M.tolist(),'cpu_root_origin':M.tolist(),'cpu_root_com_velocity':M_com.tolist(),
    'origin_match':bool(origin_match),'com_match':bool(com_match),'scope':'Root-frame convention comparison; no full nonlinear contact dynamics claim'})
   if not(origin_match or com_match):raise ValueError('Native floating mass matrix differs from both explicitly reconstructed root conventions')
   for j,name in enumerate(self.names):
    for sign in [-1,1]:
     label=f'pulse:{name}:{sign}';self.reset(label)
     for _ in range(BASELINE_STEPS):baseline=self.step(label,'baseline',self.zero)
     q0=np.asarray(baseline['joint_position'])[0].copy()
     if max(abs(q0))>1e-4 or np.max(np.abs(baseline['joint_velocity_sdk']))>1e-3:raise ValueError('State-reset baseline did not settle before pulse')
     command=self.zero.copy();command[0,j]=sign*PULSE_NM
     for _ in range(PULSE_STEPS):pulse=self.step(label,'pulse',command)
     for _ in range(COAST_STEPS):coast=self.step(label,'coast',self.zero)
     u=np.zeros(24);u[6+j]=sign*PULSE_NM;a=np.linalg.solve(M,u)
     predicted=a[6:]*DT*DT*(PULSE_STEPS*(PULSE_STEPS+1)/2+PULSE_STEPS*COAST_STEPS)
     measured=np.asarray(coast['joint_position'])[0]-q0
     ratio=float(measured[j]/predicted[j])
     result={'joint':name,'sign':sign,'predicted_all_joint_displacement':predicted.tolist(),'actual_all_joint_displacement':measured.tolist(),
      'actuated_joint_response_ratio':ratio,'ratio_scope':'Local constant rigid-link inertia prediction; 0.5..1.5 diagnostic units/sign discriminator, not control qualification',
      'passed':bool(.5<=ratio<=1.5)}
     self.case_results.append(result);self.save(self.output/'case_results.json',self.case_results)
     if not result['passed']:raise ValueError('Effort-response units/sign discriminator failed')
     self.case_count+=1
   self.set_force(self.zero)
   if self.count!=EXPERIMENT_STEPS:raise ValueError('Wrong completed step count')
   self.summary['status']='completed'
  except BaseException as e:
   self.summary['status']='failed';self.summary['errors'].append(repr(e));raise
  finally:
   try:self.set_force(self.zero)
   except Exception as e:self.summary['status']='failed';self.summary['errors'].append('clear_force:'+repr(e))
   actual_steps=self.sim.get_physics_step_count()-self.initial_counter
   if actual_steps!=self.count:
    self.summary['status']='failed';self.summary['errors'].append('observed rows versus actual physics step count mismatch')
   self.summary.update(case_count=self.case_count,coordinate_count=self.coordinate_count,step_count=self.count,actual_physics_steps=actual_steps,reset_count=self.reset_count,
    all_native_inputs_match=self.summary['status']=='completed',all_coordinate_checks=self.coordinate_count==36,
    maximum_joint_excursion_rad=self.max_excursion,maximum_sdk_rate_rad_s=self.max_sdk_rate,maximum_interval_rate_rad_s=self.max_interval_rate,
    memory_end=numeric_memory(),final_counter=self.sim.get_physics_step_count(),joint_names=self.names)
   self.save(self.output/'coordinate_results.json',self.coordinate_results);self.save(self.output/'experiment_summary.json',self.summary)
   self.stream.close();self.reset_stream.close()
  if self.summary['status']!='completed':raise RuntimeError('Experiment finalizer failed')
  return self.summary
