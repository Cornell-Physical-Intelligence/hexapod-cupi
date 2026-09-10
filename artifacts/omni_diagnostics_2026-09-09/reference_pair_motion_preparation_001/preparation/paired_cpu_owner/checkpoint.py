"""Explicit JSON controller checkpoint at a matched measured boundary; no simulator state writes."""
from pathlib import Path
import copy,hashlib,json
from dataclasses import asdict
import numpy as np
from pair_motion import PairContactReference,FootCycle,AdvancedHorizontalSwing,Swing
H=Path(__file__).resolve().parent

def serial(value):
 if isinstance(value,np.ndarray):return serial(value.tolist())
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [serial(v) for v in value]
 return value

def digest(value):return hashlib.sha256(json.dumps(serial(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def source_hashes():
 paths=['pair_motion.py','state_packet.py','checkpoint.py']+sorted(json.loads((H/'SOURCE_INPUTS.json').read_text())['oracle_files'])
 return {f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in paths}

def export_controller(g,snapshot):
 m=g._read(snapshot)
 if abs(float(m['time_s'])-g.time)>1e-7 or np.max(abs(m['joint_target_rad']-g._runtime(g.q)))>2e-7:raise ValueError('Checkpoint requires matched measured time/executed target')
 output=g._output(g.q,g.v,np.zeros_like(g.q),m)
 payload=serial({'version':'pair_contact_motion_checkpoint_v001','source_hashes':source_hashes(),'runtime_names':g.names,'configuration':asdict(g.cfg),'initial_snapshot':g._initial_snapshot,'state':output['state'],'last_diagnostics':g.last_diagnostics})
 return {'payload':payload,'sha256':digest(payload)}

def restore_curve(d):
 if d is None:return None
 coeff=np.asarray(d['coeff'],dtype=float);end=np.asarray(d['end'],dtype=float);duration=d['duration']
 if coeff.shape!=(6,3) or end.shape!=(3,) or not np.isfinite(coeff).all() or not np.isfinite(end).all() or duration<=0:raise ValueError('Invalid curve checkpoint')
 if d['type']=='advanced':
  c=AdvancedHorizontalSwing(d['t0'],duration,coeff[0],end,d['lift'],d['horizontal_fraction']);c.horizontal.coeff=np.asarray(d['horizontal_coeff'],dtype=float)
 elif d['type']=='swing':c=Swing(d['t0'],duration,coeff[0],end,d['lift'])
 else:raise ValueError('Unknown curve type')
 c.coeff=coeff.copy();return c

def restore_controller(g,checkpoint,snapshot):
 p=checkpoint['payload']
 if digest(p)!=checkpoint['sha256'] or p['version']!='pair_contact_motion_checkpoint_v001' or p['source_hashes']!=source_hashes() or tuple(p['runtime_names'])!=g.names or digest(p['configuration'])!=digest(asdict(g.cfg)):raise ValueError('Checkpoint identity/bytes/configuration mismatch')
 g.reset(p['initial_snapshot']);s=p['state']
 if s['version']!=g.STATE_VERSION or s['failure'] is not None:raise ValueError('Only this valid controller state can continue')
 fields={'time':'time_s','position':'desired_position_world_m','yaw':'desired_yaw_delta_rad','R0':'initial_rotation_world_from_body','command':'command_filter_velocity','command_rate':'command_filter_rate','requested':'requested','command_target':'command_target','factor':'derating_factor','anchors':'reference_anchors_world_m','measured_anchors':'measured_anchors_world_m','preload_world':'preload_world_m','preload_q':'initial_joint_preload_rad','neutral':'neutral_reference_toes_body_m','lower':'joint_lower_rad','upper':'joint_upper_rad','q':'q_leg_major','v':'v_leg_major','hold_until':'hold_until_s','liftoffs':'liftoffs','touchdowns':'confirmed_touchdowns','stop_requested_time':'stop_requested_time_s','reference_quiet_time':'reference_quiet_time_s','pair_index':'next_pair_index','completed_pairs':'completed_pairs','mode':'mode'}
 arrays={'position','R0','command','command_rate','requested','command_target','anchors','measured_anchors','preload_world','preload_q','neutral','lower','upper','q','v'}
 for dest,key in fields.items():setattr(g,dest,np.asarray(s[key],dtype=float) if dest in arrays else copy.deepcopy(s[key]))
 legs=('lf','lm','lr','rf','rm','rr');g.active_pair=None if s['active_pair'] is None else tuple(legs.index(n) for n in s['active_pair']);g.foot_cycles=[]
 for entry in s['foot_cycles']:
  if entry is None:g.foot_cycles.append(None);continue
  d=copy.deepcopy(entry)
  for k in ('swing','landing'):d[k]=restore_curve(d[k])
  for k in ('landing_contact_origin','landing_preload_world_m'):
   if d[k] is not None:d[k]=np.asarray(d[k],dtype=float)
  g.foot_cycles.append(FootCycle(**d))
 g.last_diagnostics=copy.deepcopy(p['last_diagnostics']);g._bounds(g.q,g.v,np.zeros_like(g.q));m=g._read(snapshot)
 if abs(float(m['time_s'])-g.time)>1e-7 or np.max(abs(m['joint_target_rad']-g._runtime(g.q)))>2e-7:raise ValueError('Live snapshot does not match checkpoint boundary; no state overwrite')
 return g._output(g.q,g.v,np.zeros_like(g.q),m)

CORE_FIELDS=('reference_position','reference_velocity','residual_position','residual_velocity','initialized')
def export_motor_target(core):
 payload={'version':'pair_motion_motor_target_state_v001','core_source_sha256':source_hashes()['oracle/reference_residual.py'],'runtime_names':list(core.joint_names),'configuration':core.config.contract(),'lower':core.lower.tolist(),'upper':core.upper.tolist(),'state':{k:getattr(core,k).tolist() for k in CORE_FIELDS}}
 return {'payload':payload,'sha256':digest(payload)}

def restore_motor_target(core,checkpoint,g):
 import torch
 p=checkpoint['payload']
 if (digest(p)!=checkpoint['sha256'] or p['version']!='pair_motion_motor_target_state_v001' or p['core_source_sha256']!=source_hashes()['oracle/reference_residual.py']
     or tuple(p['runtime_names'])!=core.joint_names or p['configuration']!=core.config.contract() or p['lower']!=core.lower.tolist() or p['upper']!=core.upper.tolist()):raise ValueError('Motor-target checkpoint identity differs')
 values={}
 for k in CORE_FIELDS:
  old=getattr(core,k);v=torch.as_tensor(p['state'][k],dtype=old.dtype,device=old.device)
  if v.shape!=old.shape or not torch.isfinite(v).all():raise ValueError('Motor-target checkpoint shape/finite error')
  values[k]=v
 if not values['initialized'].all():raise ValueError('Initialized target state required')
 if not np.array_equal(values['reference_position'].cpu().numpy(),g._runtime(g.q)[None]) or not np.array_equal(values['reference_velocity'].cpu().numpy(),g._runtime(g.v)[None]):raise ValueError('Motor target reference P/V differs from paired controller')
 core._reference_bounds(values['reference_position'])
 if values['residual_position'].abs().max()>core.config.residual_radius_rad or values['residual_velocity'].abs().max()>core.config.residual_velocity_rad_s:raise ValueError('Residual checkpoint exceeds preserved bounds')
 with torch.inference_mode(False):
  for k,v in values.items():setattr(core,k,v.clone())
