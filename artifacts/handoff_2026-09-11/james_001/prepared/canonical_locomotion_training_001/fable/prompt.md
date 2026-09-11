Act as an independent implementation reviewer. Respond in <=700 words of final findings only; no tools, no edits, no authority to launch. Exact model Claude Fable5.1, MAX. User requires central docs/PROJECT_SITE.md records for later repo adoption; this is isolated CPU proposal only.
Canonical detailed direct-drive 7.466088235kg,19 bodies18 joints. Native source005 uses32 position/0 velocity solver,50Hz holds/400Hz PD,1.6Nm and <=0.5% requested saturation,0.040rad/20ms actual target limiter. No geometry/physics changes. Fresh1 standing passed; matching32 pending. Frozen PPO003 is32x24x2 zero-command no-auto-reset infrastructure,405 actor/408 critic,5x81 history, rawdq and actual heldtarget-q, previous clipped action. MLP512/256/128 ELU, empiricalnormalizers, action targetscale.10rad/std.02; no gaitclock/filter/fakevelocity. Six-toe standing bridge rejects all lifts and nonzero commands. It cannot run walking unchanged.
Existing reviewed objective001 proposes level0 requests.02-.04m/s/.08-.16rad/s, allbearings/yaw/arcs/stops/reversals;32x24x50 noautoreset is unapproved proposal. Tracks all8 body-root samples, tilt/effort/actual target slew/acceleration/slip; no q-neutral posture reward. Commands fixed12s per-row programs, explicit program restart with NO physical/history reset. Reward must consume c[t], posthold history includes c[t+1] and heldtarget[t]-q[t+1]. Initial actor history repeats firstvalidframe (declared initialization, not five measured previous commands).
New bounded implementation CPU seams: (1) transactional command/history sequencing with no next actor on incomplete hold; (2) reward packet receives exact8 native states and typed contact evidence, maps world->nativebody->(-y,x,z), root COM velocity corrected to rootorigin. Finite material-point slip uses same current patch material point transformed into previous link pose; no patch-position drift. Native classifier already accepts rawnormals within1e-3 and exact f=0,n=0,sep=0 inactive rows. Reward normalizes ONLY previously accepted active normals. Current source005 exposes8numeric rows but NOT rawpatches via accessor (only fullJSONstream). New read-only complete8patch accessor needed; preserve everyzero/failed row.
No moving min support count or gait schedule adopted. Suggest explicit separate proposed moving rule pending real singlefootlift/return and review; never sixsupport whenever c==0 (feet can stillreturn). Empty valid contacts can'tbe admitted as locomotion. A Boolean force-off->on event alone is not qualifiedflight. Keep standingprelude/formalquiet scorer unchanged. Safeinitial noautoreset rollout should retain failureprefix+abort, not quietly drop transition or bootstrapfall.
Throughput: actualprior32 neutral1000controls8000steps585s session,600s hosttimeout distinct;contactsJSON4.620GB andprefixcopieswholeactivefile. Local losslesscodec parity exists but notadopted; no measurednative speedup. Formalacceptance raw/evidence versus futuretrainingtelemetry require explicit recordingcontract not silentdownsampling. AfterCPUchanges avoid rerunningnativejustforpackaging. Evaluate exact checkpoint withfulloriginalformalquiet10s andallbearingheldouts, separate from2updateintegration.
Please identify up to4 concrete implementation bugs/redflags, smallest useful next native job aftersmoke, and explicitly resolve movingcontactpredicate vs trainingtelemetry design. No invented thresholds or nativeadmissions. Are our proposed boundedCPU seams enough to make nextpilot honest; what missingmeasurements block it? Avoid genericRLsurvey. Below exact selected code for causal/reward/command semantics and inputhashes.

FILE tmp/canonical_walk_objective_001/commands.py
"""Proposed command-only curriculum; no physics, reset, actor or automatic promotion."""
from dataclasses import dataclass
import numpy as np

SCHEMA='canonical_omni_command_curriculum_v1'
DT=.02
LEVELS={0:((.02,.04),(.08,.16)),1:((.04,.08),(.16,.32)),2:((.08,.16),(.32,.50))}


@dataclass(frozen=True)
class Segment:
 controls:int
 command:tuple


def program(env_id,episode,level,seed=20260910):
 """25 equal-duration strata:1 quiet,24 motion; exactly20% total zero time."""
 if any(type(v)is not int or v<0 for v in [env_id,episode,seed])or level not in LEVELS:raise ValueError('Invalid explicit curriculum identity')
 slot=(env_id+7*episode)%25
 rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(env_id,episode,level)))
 linear,yaw=LEVELS[level];v=float(rng.uniform(*linear));w=float(rng.uniform(*yaw))
 if slot==0:return {'schema':SCHEMA,'kind':'quiet','slot':slot,'segments':[Segment(600,(0.,0.,0.))],'total_controls':600}
 if slot<=8:kind='axis';angle=((slot-1+episode)%4)*np.pi/2;command=np.array([v*np.cos(angle),v*np.sin(angle),0.])
 elif slot<=14:kind='diagonal';angle=np.pi/4+((slot-9+episode)%4)*np.pi/2;command=np.array([v*np.cos(angle),v*np.sin(angle),0.])
 elif slot<=18:kind='pure_yaw';command=np.array([0.,0.,w*(1 if (slot+episode)%2 else -1)])
 else:
  kind='arc';angle=((slot-19+episode)%8)*np.pi/4;command=np.array([v*np.cos(angle),v*np.sin(angle),w*(1 if rng.integers(2)else -1)])
 command[abs(command)<1e-14]=0.;a=tuple(command);b=tuple(-command);zero=(0.,0.,0.)
 if episode%2:segments=[Segment(250,a),Segment(250,b),Segment(100,zero)] # direct reversal, then stop
 else:segments=[Segment(250,a),Segment(50,zero),Segment(250,b),Segment(50,zero)]
 return {'schema':SCHEMA,'kind':kind,'slot':slot,'segments':segments,'total_controls':600}


def command_at(plan,control):
 if type(control)is not int or not 0<=control<plan['total_controls']:raise ValueError('Command clock outside declared program')
 for segment in plan['segments']:
  if control<segment.controls:return np.asarray(segment.command,dtype=np.float64).copy()
  control-=segment.controls
 raise ValueError('Incomplete command program')


class CommandBank:
 """Independent command-program epochs/clocks; never resets physical or policy history."""
 def __init__(self,n,level,seed=20260910):
  if type(n)is not int or n<=0 or level not in LEVELS:raise ValueError('Invalid explicit allocation')
  self.n=n;self.level=level;self.seed=seed;self.epochs=np.zeros(n,np.int64);self.clocks=np.zeros(n,np.int64)
  self.plans=[program(i,0,level,seed)for i in range(n)]
 def observe(self):return np.stack([command_at(p,int(t))for p,t in zip(self.plans,self.clocks)])
 def advance(self):
  if np.any(self.clocks>=600):raise ValueError('Program boundary requires explicit caller restart; no physical reset implied')
  self.clocks+=1
  return (self.clocks==600).copy()
 def reset_rows(self,rows):
  rows=np.asarray(rows)
  if rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer)or len(set(rows.tolist()))!=len(rows)or np.any(rows<0)or np.any(rows>=self.n):raise ValueError('Invalid selected reset rows')
  for i in rows:
   self.epochs[i]+=1;self.clocks[i]=0;self.plans[i]=program(int(i),int(self.epochs[i]),self.level,self.seed)
 def state(self):return {'schema':SCHEMA,'level':self.level,'seed':self.seed,'epochs':self.epochs.copy(),'clocks':self.clocks.copy()}
 @classmethod
 def from_state(cls,state):
  epochs=np.asarray(state['epochs']);clocks=np.asarray(state['clocks'])
  if state.get('schema')!=SCHEMA or epochs.ndim!=1 or clocks.shape!=epochs.shape or not np.issubdtype(epochs.dtype,np.integer)or not np.issubdtype(clocks.dtype,np.integer)or np.any(epochs<0)or np.any(clocks<0)or np.any(clocks>600):raise ValueError('Invalid command-program checkpoint')
  bank=cls(len(epochs),state['level'],state['seed']);bank.epochs=epochs.copy();bank.clocks=clocks.copy()
  bank.plans=[program(i,int(e),bank.level,bank.seed)for i,e in enumerate(epochs)]
  return bank


def held_out_cases(level):
 """Fixed new-model diagnostics, not substituted historical admission labels."""
 if level not in LEVELS:raise ValueError('Unknown level')
 lin,yaw=LEVELS[level];v=sum(lin)/2;w=sum(yaw)/2;rows=[]
 for i in range(8):
  angle=i*np.pi/4;rows.append((f'translation_{i}',(v*np.cos(angle),v*np.sin(angle),0.)))
 rows.extend([('yaw_positive',(0.,0.,w)),('yaw_negative',(0.,0.,-w))])
 for i in range(8):
  angle=np.pi/8+i*np.pi/4
  for sign in [-1,1]:rows.append((f'arc_{i}_yaw_{sign:+d}',(v*np.cos(angle),v*np.sin(angle),w*sign)))
 rows.extend((f'quiet_{i}',(0.,0.,0.))for i in range(6))
 result=[]
 for name,c in rows:
  c=np.asarray(c);c[abs(c)<1e-14]=0.
  result.append({'name':name,'segments':[Segment(100,(0.,0.,0.)),Segment(400,tuple(c)),Segment(600,(0.,0.,0.))],
          'total_controls':1100,'motion_controls':400 if c.any()else 0,'stop_controls':600,'quiet_last_controls':500,
          'training_distribution_admission':False,'requires_complete_reset_free_trace':True})
 return result

FILE tmp/canonical_walk_objective_001/causal_command.py
"""Command attribution at a policy boundary, independent of actor/physics implementations."""
import numpy as np


def begin_control(requested_command,actor_frame81,actual_held_target,current_q,control_index):
 c=np.asarray(requested_command,float);frame=np.asarray(actor_frame81,float);held=np.asarray(actual_held_target,float);q=np.asarray(current_q,float)
 n=len(c)
 if c.shape!=(n,3)or frame.shape!=(n,81)or held.shape!=(n,18)or q.shape!=(n,18):raise ValueError('Malformed policy boundary')
 if not all(np.isfinite(a).all()for a in [c,frame,held,q])or type(control_index)is not int or control_index<0:raise ValueError('Invalid policy boundary clock/value')
 # Compare the explicitly declared float32 policy packet, not host float64 spelling.
 if not np.array_equal(frame[:,6:9].astype(np.float32),c.astype(np.float32)):raise ValueError('New requested command must be visible before its first action')
 if not np.array_equal(frame[:,63:81].astype(np.float32),(held-q).astype(np.float32)):raise ValueError('Actor frame substituted a future/unexecuted target')
 return {'control_index':control_index,'reward_requested_command':c.copy(),'previous_actual_held_target':held.copy(),
         'new_command_precedes_action':True,'history_advanced_by_checker':False}

FILE tmp/canonical_walk_objective_001/objective.py
"""Proposed reward mathematics only. Native safety/quiet gates are independent."""
from dataclasses import dataclass
import numpy as np

SCHEMA='canonical_omni_reward_proposal_v1'


@dataclass(frozen=True)
class Weights:
 planar_tracking:float=5.
 yaw_tracking:float=2.
 tilt:float=.2
 roll_pitch_rate:float=.05
 vertical_rate:float=.05
 applied_effort:float=.02
 requested_excess:float=.1
 target_delta:float=.02
 target_second_difference:float=.01
 requested_target_excess:float=.05
 contact_slip:float=.05
 zero_command_angle_rate:float=.05
 zero_command_target_delta:float=.05
 terminal:float=20.


def bounded_square(x):
 """Smooth bounded cost with no hard reward clipping and nonzero finite tails."""
 with np.errstate(over='ignore'):return 1.-1./(1.+np.asarray(x,float)**2)


def terminal_margin(weights=Weights(),dt=.02,gamma=.99):
 costs=sum(v for k,v in vars(weights).items()if k not in ['planar_tracking','yaw_tracking','terminal'])
 # Worst infinite discounted negative cost plus largest terminal-step positive credit.
 required=dt*costs/(1-gamma)+dt*(weights.planar_tracking+weights.yaw_tracking)
 return {'maximum_cost_rate':costs,'minimum_discounted_continuing_return':-dt*costs/(1-gamma),
         'strict_terminal_penalty_lower_bound':required,'declared_terminal_penalty':weights.terminal,'sufficient':weights.terminal>required}


def reward(command,linear_body,angular_body,gravity_body,computed_torque,applied_torque,target_delta,target_previous_delta,requested_minus_executed,
           interval_angle_rate,contact_tangent_speed,contact_active,valid_interval,terminated,truncated,*,dt=.02,weights=Weights()):
 """All velocities alreadyforward/left/up; command is operator/requested, never actor-reduced."""
 c=np.asarray(command,float)
 if any(not np.isfinite(v)or v<0 for v in vars(weights).values()):raise ValueError('Reward cost/benefit weights must be finite nonnegative')
 if c.ndim!=2 or c.shape[1]!=3 or not np.isfinite(c).all():raise ValueError('Malformed requested command')
 n=len(c)
 def array(x,shape,name):
  a=np.asarray(x,float)
  if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Malformed/nonfinite '+name)
  return a
 v=array(linear_body,(n,8,3),'eight body-root linear velocities');w=array(angular_body,(n,8,3),'eight body angular velocities');g=array(gravity_body,(n,8,3),'eight projected gravities')
 if not np.allclose(np.linalg.norm(g,axis=2),1.,atol=1e-6,rtol=0):raise ValueError('Projected gravity must be normalized')
 raw=array(computed_torque,(n,8,18),'all eight computed torque samples');applied=array(applied_torque,(n,8,18),'all eight applied torque samples')
 delta=array(target_delta,(n,18),'actual target delta');previous=array(target_previous_delta,(n,18),'previous actual delta');dq=array(interval_angle_rate,(n,18),'actual interval angle rate')
 excess=array(requested_minus_executed,(n,18),'requested minus executed target')
 slip=array(contact_tangent_speed,(n,8,6),'all eight material point slip samples')
 if np.any(slip<0):raise ValueError('Slip speed cannot be negative')
 active=np.asarray(contact_active);valid=np.asarray(valid_interval);term=np.asarray(terminated);timeout=np.asarray(truncated)
 for a,shape in [(active,(n,8,6)),(valid,(n,)),(term,(n,)),(timeout,(n,))]:
  if a.shape!=shape or a.dtype!=bool:raise ValueError('Explicit boolean mask required')
 if not valid.all():raise ValueError('Invalid interval cannot be replaced by zero-rate reward')
 if dt!=.02:raise ValueError('This proposal is explicitly50Hz; reversion required for another clock')
 if not terminal_margin(weights,dt)['sufficient']:raise ValueError('Terminal penalty does not dominate bounded negative continuation')
 sigma_v=np.maximum(.02,.5*np.linalg.norm(c[:,:2],axis=1));sigma_w=np.maximum(.05,.5*abs(c[:,2]))
 zero=np.all(c==0,axis=1)
 components={
  'planar_tracking':weights.planar_tracking*np.exp(-np.mean(np.sum((v[:,:,:2]-c[:,None,:2])**2,axis=2),axis=1)/sigma_v**2),
  'yaw_tracking':weights.yaw_tracking*np.exp(-np.mean((w[:,:,2]-c[:,None,2])**2,axis=1)/sigma_w**2),
  # g_xy^2 alone aliases an upside-down body to upright. This is |g-[0,0,-1]|^2.
  'tilt':-weights.tilt*np.mean(bounded_square(np.sqrt(np.maximum(2*(1+g[:,:,2]),0))/.15),axis=1),
  'roll_pitch_rate':-weights.roll_pitch_rate*np.mean(bounded_square(w[:,:,:2]/.25),axis=(1,2)),
  'vertical_rate':-weights.vertical_rate*np.mean(bounded_square(v[:,:,2]/.1),axis=1),
  'applied_effort':-weights.applied_effort*np.mean(bounded_square(applied/1.6),axis=(1,2)),
  'requested_excess':-weights.requested_excess*np.mean(bounded_square(np.maximum(abs(raw)-1.6,0)/1.6),axis=(1,2)),
  'target_delta':-weights.target_delta*np.mean(bounded_square(delta/.04),axis=1),
  'target_second_difference':-weights.target_second_difference*np.mean(bounded_square((delta-previous)/.04),axis=1),
  'requested_target_excess':-weights.requested_target_excess*np.mean(bounded_square(excess/.04),axis=1),
  'contact_slip':-weights.contact_slip*np.sum(bounded_square(slip/.05)*active,axis=(1,2))/(8*6),
  'zero_command_angle_rate':-weights.zero_command_angle_rate*np.mean(bounded_square(dq/.05),axis=1)*zero,
  'zero_command_target_delta':-weights.zero_command_target_delta*np.mean(bounded_square(delta/.002),axis=1)*zero}
 rates=sum(components.values());value=dt*rates-weights.terminal*term
 if not np.isfinite(value).all():raise ValueError('Nonfinite proposed reward')
 return {'reward':value,'component_rates':components,'termination_penalty':-weights.terminal*term,
  'bootstrap_allowed':~term,'timeout':timeout.copy(),'requested_command':c.copy(),'schema':SCHEMA,
  'physics_admission':False,'quality_admission':False}


def material_point_tangent_velocity(point_world,normal_world,previous_pose,current_pose,dt):
 """Finite interval velocity of the currentcontact MATERIAL point, not patch-location drift.

 Poses (...,4,4) map native linklocal coordinates to world. Normals areunitworldvectors.
 This finite interval is a diagnostic approximation, not instantaneous frictional power.
 """
 p=np.asarray(point_world,float);normal=np.asarray(normal_world,float);old=np.asarray(previous_pose,float);new=np.asarray(current_pose,float)
 if p.shape!=normal.shape or p.shape[-1]!=3 or old.shape!=p.shape[:-1]+(4,4)or new.shape!=old.shape:raise ValueError('Invalid material point frames')
 if not all(np.isfinite(a).all()for a in [p,normal,old,new])or not np.isfinite(dt)or dt<=0:raise ValueError('Invalid material point values/clock')
 if not np.allclose(np.linalg.norm(normal,axis=-1),1.,atol=1e-6,rtol=0):raise ValueError('Contact normal is not active/unit')
 for pose in [old,new]:
  rot=pose[...,:3,:3]
  if not np.allclose(np.swapaxes(rot,-1,-2)@rot,np.eye(3),atol=1e-6,rtol=0)or not np.allclose(np.linalg.det(rot),1.,atol=1e-6,rtol=0)or not np.allclose(pose[...,3,:],[0,0,0,1],atol=1e-12,rtol=0):raise ValueError('Pose is not a rigid transform')
 local=np.einsum('...ji,...j->...i',new[...,:3,:3],p-new[...,:3,3])
 old_world=np.einsum('...ij,...j->...i',old[...,:3,:3],local)+old[...,:3,3]
 velocity=(p-old_world)/dt
 return velocity-np.sum(velocity*normal,axis=-1,keepdims=True)*normal

INPUT_HASHES
{"tmp/canonical_ppo_integration_003/canonical_direct_ppo/adapter.py": {"sha256": "57f4089cdb19632025d26306decc0465958e928f9bbab09dee0e6aba6482bc06", "selected_copy": "oracles/adapter.py"}, "tmp/canonical_ppo_integration_003/canonical_direct_ppo/frames.py": {"sha256": "e81f7d40d3fcbe2ec505a524f826aff0b44627f6df4c9e96b809617423be215b", "selected_copy": "oracles/frames.py"}, "tmp/canonical_walk_objective_001/commands.py": {"sha256": "46699dca2d8048bd4cc9339238e3696454ffddea99e08f703b516ebee236190d", "selected_copy": "oracles/commands.py"}, "tmp/canonical_walk_objective_001/objective.py": {"sha256": "e7ccfa2e92c5525d79812943efc99c4b0dc5885558894816df49ff190714f40b", "selected_copy": "oracles/objective.py"}, "tmp/canonical_walk_objective_001/causal_command.py": {"sha256": "933b86df0b6074d4597da782311e7689cf8356c74a4fa460383e6229d20476b8", "selected_copy": "oracles/causal_command.py"}, "tmp/canonical_ppo_integration_003/FREEZE_SHA256.json": {"sha256": "970cd1766366c9bdba3239205695e998da80819d2283ce0e261a3b265ca48183", "copy": false}, "tmp/updated_native_standing_005/FREEZE_SHA256.json": {"sha256": "c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131", "copy": false}, "tmp/canonical_walk_objective_001/FREEZE_SHA256.json": {"sha256": "22bfab99db8cfa18656546a0e94f592676310ac983d31b5815ef5cef15ea12f4", "copy": false}, "tmp/canonical_ppo_throughput_design_001/BENCHMARK.json": {"sha256": "65c10914568bf14b7a538cd347a05777b4a2ba1dada5809c5eb8f9e9cdfee167", "copy": false}, "tmp/canonical_ppo_throughput_design_001/EVIDENCE.json": {"sha256": "1f09d809002ab3e4c5d3979992531113e7d7bf4aaad60d0f39fcf71135a04216", "copy": false}}