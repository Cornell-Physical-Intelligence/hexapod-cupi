"""New paired-controller state fragment/history only; not an846/849 actor packet."""
from pathlib import Path
import hashlib,json
import numpy as np
VERSION='pair_contact_motion_fragment_v001'
GLOBAL_MODES=('hold','contact_hold','paired_unloading','paired_motion','stopping_reference_motion','reference_quiet_hold')
FOOT_MODES=('unloading','swing','landing_blend','awaiting_landing_support','awaiting_contact','contact_hold')
LEGS=('lf','lm','lr','rf','rm','rr')
PAIR_ORDER=(('lm','rm'),('lf','rr'),('lr','rf'))

def pack(output):
 s=output['state']
 if s['version']!='pair_contact_motion_v001':raise ValueError('Exact new paired state version required; scalarwave846/849 is incompatible')
 names=[];values=[]
 def add(name,value,shape=()):
  a=np.asarray(value,dtype=np.float64)
  if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite fixed-shape state required:'+name)
  labels=[name] if not shape else [name+'['+','.join(map(str,i))+']' for i in np.ndindex(shape)]
  names.extend(labels);values.extend(a.reshape(-1).tolist())
 def optional(name,value,shape=()):add(name+'.valid',value is not None);add(name+'.value',np.zeros(shape) if value is None else value,shape)
 def enum(name,value,choices):
  if value not in choices:raise ValueError('Unknown mode:'+str(value))
  add(name,[value==c for c in choices],(len(choices),))
 def curve(prefix,c):
  add(prefix+'.valid',c is not None);c=c or {}
  kind=c.get('type','swing');enum(prefix+'.type',kind,('swing','advanced'))
  for key in ('t0','duration','lift'):add(prefix+'.'+key,c.get(key,0.))
  add(prefix+'.end',c.get('end',np.zeros(3)),(3,));add(prefix+'.coeff',c.get('coeff',np.zeros((6,3))),(6,3))
  add(prefix+'.horizontal_fraction',c.get('horizontal_fraction',0.));add(prefix+'.horizontal_coeff',c.get('horizontal_coeff',np.zeros((6,3))),(6,3))
 add('valid',bool(output['valid'][0]));enum('mode',s['mode'],GLOBAL_MODES)
 pair=None if s['active_pair'] is None else tuple(s['active_pair'])
 if pair is not None and pair not in PAIR_ORDER:raise ValueError('Unknown active pair')
 add('active_pair',[pair==p for p in PAIR_ORDER],(3,))
 for k in ('time_s','desired_yaw_delta_rad','derating_factor','hold_until_s','liftoffs','confirmed_touchdowns','next_pair_index','completed_pairs'):add(k,s[k])
 for k in ('desired_position_world_m','command_filter_velocity','command_filter_rate','requested','command_target'):add(k,s[k],(3,))
 add('initial_rotation_world_from_body',s['initial_rotation_world_from_body'],(3,3))
 for k in ('reference_anchors_world_m','measured_anchors_world_m','preload_world_m','initial_joint_preload_rad','neutral_reference_toes_body_m','joint_lower_rad','joint_upper_rad','q_leg_major','v_leg_major'):add(k,s[k],(6,3))
 for k in ('stop_requested_time_s','reference_quiet_time_s'):optional(k,s[k])
 add('failure_latched',s['failure'] is not None)
 add('declared_max_forward_mps',output['diagnostics']['configuration']['max_translation_mps'])
 for i,entry in enumerate(s['foot_cycles']):
  prefix='foot.'+LEGS[i];add(prefix+'.exists',entry is not None);f=entry or {}
  if f and (f['leg']!=i or f['current_leg'] not in (None,i)):raise ValueError('Foot slot/name mismatch')
  add(prefix+'.active',f.get('current_leg') is not None);enum(prefix+'.mode',f.get('mode','unloading'),FOOT_MODES)
  for k in ('flight_seen','flight_count','contact_count','raw_force_free_runs','raw_force_free_samples','unqualified_contact_returns','last_unqualified_run_samples','descent_seen','landing_contact_gap_steps','hold_until'):add(prefix+'.'+k,f.get(k,0))
  for k in ('last_unqualified_return_time','last_unqualified_lift_m','flight_baseline_z','flight_peak_z','landing_trigger_s','landing_endpoint_correction_m','landing_original_contact_error_m','landing_target_excursion_m'):optional(prefix+'.'+k,f.get(k))
  for k in ('landing_contact_origin','landing_preload_world_m'):optional(prefix+'.'+k,f.get(k),(3,))
  for k in ('swing','landing'):curve(prefix+'.'+k,f.get(k))
 vector=np.asarray(values,dtype=np.float64);spec={'version':VERSION,'fields':names,'features':len(names),'leg_order':LEGS,'joint_state_order':'named LEGS-major, 3joints per leg','dtype':'float64 CPU prototype','scope':'controller-state fragment only; no actor/critic composition or history adoption'}
 signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
 return {'version':VERSION,'schema_sha256':signature,'values':vector,'time_s':float(s['time_s']),'schema':spec}

class FragmentHistory:
 """One replica; explicit reset epochs prevent stale state across interruptions."""
 def __init__(self,length=3):
  if not isinstance(length,int) or isinstance(length,bool) or length<1:raise ValueError('Positive history length')
  self.length=length;self.frames=None
 def reset(self,packet,reset_id):
  self.schema=packet['schema_sha256'];self.epoch=reset_id;self.time=packet['time_s'];self.steps=0
  self.frames=np.repeat(np.asarray(packet['values'])[None],self.length,axis=0).copy();return self.frames.copy()
 def append(self,packet,reset_id):
  if self.frames is None or reset_id!=self.epoch:raise ValueError('Explicit history reset required for new episode/continuation')
  if packet['schema_sha256']!=self.schema or abs(packet['time_s']-self.time-.02)>1e-7:raise ValueError('Schema or contiguous20ms time mismatch')
  row=np.asarray(packet['values']);
  if row.shape!=self.frames.shape[1:] or not np.isfinite(row).all():raise ValueError('History frame mismatch')
  self.frames=np.concatenate((self.frames[1:],row[None]),axis=0);self.time=packet['time_s'];self.steps+=1;return self.frames.copy()
 def checkpoint(self):return {'schema':self.schema,'epoch':self.epoch,'time':self.time,'steps':self.steps,'length':self.length,'frames':self.frames.tolist()}
 def restore(self,d,packet,reset_id):
  if d['length']!=self.length or d['schema']!=packet['schema_sha256'] or d['epoch']!=reset_id or abs(d['time']-packet['time_s'])>1e-7:raise ValueError('Exact history boundary/schema/epoch required')
  frames=np.asarray(d['frames'],dtype=float)
  if frames.shape!=(self.length,len(packet['values'])) or not np.isfinite(frames).all() or not np.array_equal(frames[-1],packet['values']):raise ValueError('Last history state differs from restored controller')
  self.schema=d['schema'];self.epoch=d['epoch'];self.time=d['time'];self.steps=d['steps'];self.frames=frames.copy()
