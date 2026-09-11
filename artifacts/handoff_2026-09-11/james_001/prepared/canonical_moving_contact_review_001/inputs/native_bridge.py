"""New canonical PPO/native-standing boundary. No simulator startup or reset policy."""
import numpy as np
import torch
from .adapter import ObservationScales,ObservationHistory,ActorFrameInputs
from .frames import world_to_native_body,native_to_navigation,body_velocity_at_root_origin
from .smoke_config import NUM_ENVS,CONTROL_DT,TARGET_SCALE_RAD,SLEW_RAD_PER_CONTROL

LIMIT_NUMERIC_TOL_RAD=2e-6 # same native generalized-state readback tolerance; no widened travel
REWARD_SCHEMA='canonical_quiet_smoke_reward_v1'


def checked_q(q,lower,upper):
 q=np.asarray(q,dtype=np.float64)
 if q.shape!=(NUM_ENVS,18)or not np.isfinite(q).all():raise ValueError('Malformed measured joint positions')
 if np.any(q<lower-LIMIT_NUMERIC_TOL_RAD)or np.any(q>upper+LIMIT_NUMERIC_TOL_RAD):raise ValueError('Measured joint exceeded admitted native limits')
 return q


def govern(action,held,lower,upper,neutral):
 """Enforce the actual float32 emitted-target budget around actual previous hold."""
 raw=np.asarray(action,dtype=np.float64)
 if raw.shape!=(NUM_ENVS,18)or not np.isfinite(raw).all():raise ValueError('Malformed policy action')
 held=np.asarray(held,dtype=np.float64);clipped=np.clip(raw,-1.,1.)
 requested=np.clip(neutral+TARGET_SCALE_RAD*clipped,lower,upper)
 lo=np.maximum(lower,held-SLEW_RAD_PER_CONTROL);hi=np.minimum(upper,held+SLEW_RAD_PER_CONTROL)
 target=np.clip(requested,lo,hi).astype(np.float32)
 # Round inward by one float32 ULP if the initial cast crossed an exact bound.
 target=np.where(target.astype(float)>hi,np.nextafter(target,np.float32(-np.inf)),target)
 target=np.where(target.astype(float)<lo,np.nextafter(target,np.float32(np.inf)),target)
 if np.any(target.astype(float)<lo)or np.any(target.astype(float)>hi):raise ValueError('No representable float32 held target inside bounds')
 return {'raw_action':raw.copy(),'clipped_action':clipped,'requested_target_rad':requested,'emitted_target_rad':target}


def quiet_reward(before,after,target_delta,neutral):
 """Smoke stabilization objective only; formal SDK quiet gates remain separate."""
 q=np.asarray(after['joint_position_rad'],float)
 rate=(q-np.asarray(before['joint_position_rad'],float))/CONTROL_DT
 _,gyro=body_velocity_at_root_origin(after['root_com_velocity'],after['root_pose_xyzw'],np.zeros(3))
 components={'position_cost':-np.mean(((q-neutral)/.1)**2,axis=1),
             'control_interval_rate_cost':-np.mean(rate**2,axis=1),
             'body_angular_cost':-np.mean((gyro/.25)**2,axis=1),
             'executed_target_delta_cost':-np.mean((target_delta/.04)**2,axis=1)}
 reward=sum(components.values())
 if not np.isfinite(reward).all():raise ValueError('Nonfinite smoke reward')
 return reward,components


class CanonicalNativeBridge:
 """Consume an already initialized, admitted neutral-hold session; never reset it."""
 num_envs=NUM_ENVS;num_actions=18
 def __init__(self,session,model,*,device,native_errors=None):
  self.session=session;self.device=device;self.model=model;self.rows=[];self.failure=None;self.count=0;self.native_errors=[]if native_errors is None else native_errors
  if self.native_errors:raise ValueError('Native error event before actor boundary')
  if session.n!=NUM_ENVS or session.reset_count!=1:raise ValueError('Expected32 rows and one explicit canonical reset')
  named={j['name']:j for j in model['joints']}
  if len(named)!=18 or set(session.names)!=set(named)or len(session.names)!=18:raise ValueError('Native joint names differ from exact model')
  self.names=list(session.names);self.native=session.native
  bounds=np.asarray(self.native['limits'],float)
  if bounds.shape!=(NUM_ENVS,18,2)or not np.array_equal(bounds,np.broadcast_to(bounds[:1],bounds.shape)):raise ValueError('Native joint limits differ across replicas')
  self.lower=bounds[0,:,0];self.upper=bounds[0,:,1]
  wanted=np.asarray([[named[n]['lower'],named[n]['upper']]for n in self.names])
  if not np.allclose(bounds[0],wanted,atol=2e-6,rtol=0):raise ValueError('Native limits differ from canonical model')
  self.neutral=np.array([named[n]['default_value']for n in self.names],float)
  self.body_index=session.body_names.index('body');self.root_com_local=np.asarray(self.native['coms'],float)[:,self.body_index,:3]
  self.command=np.zeros((NUM_ENVS,3));self.previous_clipped=np.zeros((NUM_ENVS,18))
  self.current=session.observe();self._validate(self.current,require_valid=True)
  if not np.asarray(self.current['distal_contact']).all():raise ValueError('Initial native boundary lacks six-toe support')
  if not np.array_equal(self.current['joint_target_rad'],np.broadcast_to(self.neutral,(NUM_ENVS,18))):raise ValueError('PPO must start at actual admitted neutral hold')
  self.held=np.asarray(self.current['joint_target_rad'],np.float32).copy()
  scales=ObservationScales(*[np.ones(n)for n in [3,3,3,18,18,18,18,3]])
  self.history=ObservationHistory(self.neutral,scales,NUM_ENVS)
  self.history.reset_rows(np.arange(NUM_ENVS),self._inputs(self.current))
  self.initial_counter=int(self.current['explicit_counter'])
  self.requested_saturation_counts=np.zeros((NUM_ENVS,18),dtype=np.int64);self.substeps_verified=0
 def _validate(self,row,require_valid):
  checked_q(row['joint_position_rad'],self.lower,self.upper)
  for key in ['joint_velocity_rad_s','root_pose_xyzw','root_com_velocity','joint_target_rad','computed_torque_nm','applied_torque_nm','minimum_non_toe_floor_m']:
   if not np.isfinite(row[key]).all():raise ValueError('Nonfinite native field:'+key)
  if require_valid and (not np.asarray(row['contact_valid']).all()or not np.asarray(row['interval_valid']).all()):raise ValueError('Native observation boundary is not valid yet')
  if np.asarray(row['terminated']).any()or np.asarray(row['truncated']).any():raise ValueError('Native termination; no automatic learner reset')
  if np.asarray(row['nonfoot_contact']).any():raise ValueError('Native nonfoot contact; smoke aborted')
  if np.min(row['minimum_non_toe_floor_m'])<-.001:raise ValueError('Native non-toe mesh floor clearance violated')
  if np.min(np.asarray(row['root_pose_xyzw'])[:,2])<.055:raise ValueError('Native plate height violated')
  if np.max(abs(row['applied_torque_nm']))>1.60001:raise ValueError('Native applied cap violated')
 def _inputs(self,row):
  pose=np.asarray(row['root_pose_xyzw']);linear,angular=body_velocity_at_root_origin(row['root_com_velocity'],pose,self.root_com_local)
  gravity=world_to_native_body(np.tile([0.,0.,-1.],(NUM_ENVS,1)),pose[:,3:])
  return ActorFrameInputs(angular,gravity,self.command,row['joint_position_rad'],row['joint_velocity_rad_s'],self.previous_clipped,self.held)
 def get_observations(self):
  actor=self.history.actor_observation()
  linear,_=body_velocity_at_root_origin(self.current['root_com_velocity'],self.current['root_pose_xyzw'],self.root_com_local)
  critic=self.history.critic_observation(linear)
  return {'policy':torch.as_tensor(actor,dtype=torch.float32,device=self.device),'critic':torch.as_tensor(critic,dtype=torch.float32,device=self.device)}
 def step(self,action):
  if self.failure:raise RuntimeError('First bridge failure latched:'+self.failure)
  if np.any(self.command!=0):raise ValueError('This bounded integration smoke requires zero commands')
  before=self.current
  proposal=govern(action.detach().cpu().numpy(),self.held,self.lower,self.upper,self.neutral)
  record={'control':self.count,'command':self.command.copy(),**{k:v.copy()for k,v in proposal.items()},'status':'attempted'}
  self.rows.append(record)
  try:
   if self.native_errors:raise ValueError('Native error event before policy control')
   after=self.session.step_control(proposal['emitted_target_rad'])
   record['native_endpoint']={k:np.array(v,copy=True)for k,v in after.items()}
   if self.native_errors:raise ValueError('Native error event during policy control')
   substeps=self.session.observe_control_substeps()
   if len(substeps)!=8:raise ValueError('Missing native control substeps')
   for sub,row in enumerate(substeps):
    self._validate(row,require_valid=True)
    if int(row['explicit_counter'])!=self.initial_counter+8*self.count+sub+1:raise ValueError('Wrong native substep counter')
    if not np.array_equal(row['joint_target_rad'],proposal['emitted_target_rad']):raise ValueError('Substep held target differs')
    if not np.asarray(row['distal_contact']).all():raise ValueError('Zero-command smoke lost six-toe support')
    self.requested_saturation_counts+=(np.abs(row['computed_torque_nm'])>1.6)
    self.substeps_verified+=1
    # Reject once the full declared384-sample .5% allowance becomes impossible.
    if np.any(self.requested_saturation_counts>48*8*.005):raise ValueError('Full400Hz requested-saturation smoke budget exceeded')
   self._validate(after,require_valid=True)
   if not np.array_equal(after['joint_target_rad'],proposal['emitted_target_rad']):raise ValueError('Native held target differs from emitted target')
   if int(after['explicit_counter'])!=self.initial_counter+8*(self.count+1):raise ValueError('Wrong native8-substep control boundary')
   delta=np.asarray(after['joint_target_rad'],float)-self.held
   if np.max(abs(delta))>SLEW_RAD_PER_CONTROL:raise ValueError('Actual emitted target slew exceeds formal budget')
   reward,components=quiet_reward(before,after,delta,self.neutral)
   record['reward']=reward.copy();record['reward_components']={k:v.copy()for k,v in components.items()}
   self.held=np.asarray(after['joint_target_rad'],np.float32).copy();self.previous_clipped=proposal['clipped_action'].copy();self.current=after
   self.history.push(self._inputs(after));self.count+=1;record['status']='completed'
   return self.get_observations(),torch.as_tensor(reward,dtype=torch.float32,device=self.device),torch.zeros(NUM_ENVS,dtype=torch.bool,device=self.device),{}
  except BaseException as error:
   self.failure=repr(error);record['status']='failed';record['error']=self.failure;raise
 def export_runtime_state(self):
  return {'schema':'canonical_native_bridge_v1','joint_names':self.names,'completed_policy_controls':self.count,
          'initial_native_counter':self.initial_counter,'history':self.history.history.copy(),
          'previous_clipped_action':self.previous_clipped.copy(),'actual_held_target':self.held.copy(),'command':self.command.copy()}
 def export_audit(self,output):
  # Native session owns full400Hz/contact records, even if step_control raised.
  self.session.flush()
  path=output/'policy_control_trace.pt';torch.save(self.rows,path)
  return {'native_session_steps':self.session.count,'native_session_controls':self.session.control,
          'policy_controls':self.count,'policy_attempts':len(self.rows),'failure':self.failure,
          'policy_substeps_verified':self.substeps_verified,'requested_saturation_counts':self.requested_saturation_counts.tolist(),
          'raw400Hz_owner':'matching native standing session','reward_schema':REWARD_SCHEMA,
          'command_scope':'zero-command integration only','trace_file':path.name,'quality_admitted':False}
