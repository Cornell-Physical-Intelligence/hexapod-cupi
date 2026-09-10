Continue as our independent engineering partner at MAX reasoning. The user explicitly says not to defer to Claude, but to use useful findings, and asks you to look for implementation bugs and red flags. Reconsider your prior high-effort advice using actual code below. Correct factual assumptions and separate proposed diagnostic thresholds from unchanged acceptance gates. Your previous response says16/1 physics; the actual direct baseline uses solver16/4. Your description of CAPS action units/gradient size is a hypothesis until checked against actual implementation. Existing fresh standing passes and separate complete static support-set transfers already exist; do not manufacture blockers or repeat completed evidence. Source003 fix only changes inference-created registered buffers during checkpoint verification; the corrected smoke is running now. We will not change active-source bytes. Real hardware identification is essential before transfer but cannot be an unexplained stop on authorized simulation preparation. The250ms lease issue has an existing planted-contact versus new optical-support design; validity at swing start alone does not prove safety at landing across an unknown cell. Please evaluate this carefully.

Review the actual code. Return prioritized concrete bugs/red flags with locations and a minimal discriminating experiment for each. Then revise architectural recommendation: what to adopt, what to reject, what is unknown; whether50-update matched pilots remain useful; whether any proven bug must be fixed first. Do not impose unvalidated3x/10x improvement as acceptance gates or universally say frozen stance failure proves every controller impossible. No model authority; evidence decides. No tools or real experiment claims.

FILE direct_config.py
```python
"""Explicit launch selection for one immutable direct315 experiment source."""
import copy

SCHEMA = 'direct315_stand_stop_caps_native_v2'
CHECKPOINT = '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
ALLOCATIONS = {'smoke': {'replicas': 32, 'controls_per_update': 24, 'updates': 2},
               'pilot': {'replicas': 1024, 'controls_per_update': 24, 'updates': 50}}

def protocol():
    return {'schema': SCHEMA, 'allocations': copy.deepcopy(ALLOCATIONS),
            'branches': ['curriculum', 'caps'], 'checkpoint_sha256': CHECKPOINT,
            'command_schedule': '25percent_quiet_other_rows_8s_motion_8s_stop',
            'evaluations': ['constant', 'stop'], 'automatic_continuation': False}

def selection(mode, allocation, branch, evaluation, iterations):
    if mode == 'train':
        if allocation not in ALLOCATIONS or branch not in ('curriculum', 'caps') or evaluation is not None:
            raise ValueError('Training requires explicit allocation and branch only')
        result = {'schema': SCHEMA, 'allocation': allocation, 'branch': branch, **ALLOCATIONS[allocation]}
        if iterations is not None and (type(iterations) is not int or iterations != result['updates']):
            raise ValueError('Iteration override differs from bounded allocation')
        result['caps'] = {'temporal_weight': .1 if branch == 'caps' else 0.,
                          'spatial_weight': .1 if branch == 'caps' else 0., 'noise_scale': 1., 'noise_seed': 1157}
        return result
    if allocation is not None or branch is not None or iterations is not None:
        raise ValueError('Nontraining phase cannot request allocation/branch/iterations')
    if mode == 'validate' and evaluation is None:
        return {'schema': SCHEMA, 'evaluation': None, 'replicas': 32}
    if mode == 'evaluate' and evaluation in ('constant', 'stop'):
        return {'schema': SCHEMA, 'evaluation': evaluation, 'replicas': 48}
    raise ValueError('Only explicit standing, train, constant or stop diagnostic phases exist')

def configure(cfg, selected):
    from caps import validated_options
    expected = selection('train', selected.get('allocation'), selected.get('branch'), None, None)
    if selected != expected:
        raise ValueError('Training selection changed')
    result = copy.deepcopy(cfg)
    if result['obs_groups'] != {'actor': ['policy'], 'critic': ['critic']} or result.get('clip_actions') is not None:
        raise ValueError('Exact 315/318 observation groups and unclipped wrapper required')
    result.update(num_steps_per_env=24, save_interval=1, max_iterations=selected['updates'])
    result['algorithm'].update(class_name='caps_ppo:CapsPPO', caps_options=validated_options(selected['caps']))
    return result

```

FILE direct_contract.py
```python
"""Host-safe standard-library admission/result contract; no simulator or NumPy imports."""
import hashlib, json, math
from pathlib import Path
from direct_config import ALLOCATIONS, CHECKPOINT, SCHEMA, protocol, selection

PLAN='robot/hexapod_mkii_length_study/training_plan.json'
PARENT='4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
URDF='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c'
PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
EXPECTED_OVERRIDES={'target_slew_rad_per_20ms':.04,'reward_weights':{'stand_joint_velocity':-.5,'stand_target_velocity':-.15,'stand_posture':-2.,'action_rate':-.075,'saturation':-.75,'torque_excess':-.6,'worst_torque_excess':-.2,'stand_raw_action':0.},'observation_noise_scale':1.,'target_filter_time_constant_s':0.}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def finite(x):
    if type(x) not in (int,float) or not math.isfinite(x): raise ValueError('Expected finite numeric evidence')
    return x
def check_hash(p,bound):
    if not isinstance(bound,str) or len(bound)!=64 or sha(p)!=bound: raise ValueError('Missing/changed evidence '+str(p))

def verify_smoke_campaign(smoke, identity):
    smoke=Path(smoke);campaign=read(smoke/'campaign.json')
    required=('standing','train','final_constant','final_stop')
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('allocation')!='smoke' or campaign.get('branch')!='caps' or campaign.get('identity')!=identity:
        raise ValueError('Smoke campaign did not complete same-source terminal integrity')
    if set(campaign.get('accepted_phases',{}))!=set(required) or campaign.get('planned_phases')!=list(required):
        raise ValueError('Smoke phase inventory differs')
    train=validate_result(smoke/'train','train',identity)
    for phase in required:
        current=validate_result(smoke/phase,phase,identity,train['checkpoint_sha256'])
        if campaign['accepted_phases'][phase]!=current: raise ValueError('Smoke accepted receipt changed')
    return sha(smoke/'campaign.json')

def verify_inputs(args):
    source=Path(args.source).resolve(); manifest=read(source/'campaign_source_hashes.json')
    actual={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file() and p.name!='campaign_source_hashes.json'}
    if actual!=manifest or any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Changed/unlisted source input')
    origin=read(source/'source_origin.json');plan=read(source/PLAN)
    if origin.get('schema')!=SCHEMA or origin.get('cold_parent_source_sha256')!=PARENT: raise ValueError('Wrong shared direct training lineage')
    if plan['omni']['direct_recovery_training']!=protocol() or plan['omni']['overrides']!=EXPECTED_OVERRIDES: raise ValueError('Wrong behavior/physical/reward profile')
    if (plan['validation_num_envs'],plan['validation_control_steps'],plan['evaluation_num_envs'],plan['training_num_envs'],plan['training_iterations'],plan['physics_dt_s'],plan['decimation'])!=(32,1000,48,1024,50,.0025,8): raise ValueError('Wrong declared allocation or timestep')
    if plan['omni']['diagnostics']!={'duration_s':12,'settle_s':2,'seed':7057,'trace_envs_per_scenario':1,'controller':'policy'}: raise ValueError('Historical constant diagnostic changed')
    repair=plan['omni']['repair_training']
    if repair!={'checkpoint_sha256':CHECKPOINT,'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}: raise ValueError('Checkpoint/optimizer/exploration initialization changed')
    check_hash(args.checkpoint,CHECKPOINT)
    selected=selection('train',args.allocation,args.branch,None,None)
    if args.allocation=='smoke' and args.branch!='caps': raise ValueError('Integration smoke exercises positive CAPS only')
    identity={'schema':SCHEMA,'source_manifest_sha256':sha(source/'campaign_source_hashes.json'),'plan_sha256':sha(source/PLAN),
              'checkpoint_sha256':CHECKPOINT,'actor_width':315,'critic_width':318,'selection':selected,
              'overrides':EXPECTED_OVERRIDES,'diagnostic_options':plan['omni']['diagnostics'],'Stage2_complete':False}
    if args.allocation=='pilot':
        smoke=getattr(args,'smoke',None)
        if smoke is None: raise ValueError('Pilot requires completed same-source actual CAPS smoke')
        smoke_identity={**identity,'selection':selection('train','smoke','caps',None,None)}
        identity['smoke_campaign_sha256']=verify_smoke_campaign(smoke,smoke_identity)
        identity['smoke_state_sha256']=sha(Path(smoke)/'train/state.json')
        identity['smoke_receipt_sha256']=sha(Path(smoke)/'train/training_receipt.json')
    return identity

validate_inputs=verify_inputs

def validate_result(directory,phase,identity,expected_checkpoint_sha256=None):
    directory=Path(directory)
    if phase not in PHASES: raise ValueError('Undeclared phase')
    state=read(directory/'state.json')
    if state.get('status')!='completed' or any(state.get(k)!=v for k,v in {'variant':'f050_t060','urdf_sha256':URDF,'plan_sha256':identity['plan_sha256'],'stance_index':0}.items()): raise ValueError('Incomplete/wrong-source phase')
    expected_mode='validate' if phase=='standing' else ('train' if phase=='train' else 'evaluate')
    expected_selection=identity['selection'] if phase=='train' else selection(expected_mode,None,None,None if phase=='standing' else phase.split('_')[1],None)
    if state.get('mode')!=expected_mode or state.get('direct_selection')!=expected_selection: raise ValueError('Wrong phase selection')
    result={'phase':phase,'state_sha256':sha(directory/'state.json'),'source_manifest_sha256':identity['source_manifest_sha256'],'Stage2_complete':False}
    if phase=='standing':
        a=read(directory/'admission.json')
        if not a.get('gate',{}).get('passed') or any(a.get(k)!=state.get(k) for k in ('variant','urdf_sha256','plan_sha256','stance_index')): raise ValueError('Matching standing admission absent')
        result.update(passed=True,admission_sha256=sha(directory/'admission.json'));return result
    if phase=='train':
        receipt=read(directory/'training_receipt.json');sel=identity['selection']
        if receipt!=state.get('training_receipt') or receipt.get('selection')!=sel or receipt.get('complete') is not True or receipt.get('updates_completed')!=sel['updates'] or state.get('iterations')!=sel['updates']: raise ValueError('Incomplete/mismatched bounded learning receipt')
        reload=receipt.get('reload',{})
        if not all(reload.get(k) is True for k in ('passed','exact_actor_critic_normalizer_optimizer','exact_deterministic_action')) or reload.get('optimizer_entries',0)<=0: raise ValueError('Actual strict checkpoint reload unproved')
        bound=receipt['final_checkpoint_sha256'];check_hash(directory/'policy/final.pt',bound)
        if bound!=state.get('checkpoint_sha256') or bound!=reload.get('checkpoint_sha256'): raise ValueError('Checkpoint receipt mismatch')
        init=read(directory/'repair_initialization.json')
        if init.get('checkpoint_sha256')!=CHECKPOINT or not init.get('actor_and_critic_preserved_except_std') or not init.get('observation_normalizers_preserved') or init.get('optimizer_state_entries')!=0: raise ValueError('Wrong warm start')
        audit=receipt['audit']
        if audit.get('controls')!=sel['updates']*24 or audit.get('replicas')!=sel['replicas']: raise ValueError('Wrong actual control count')
        for key in ('terminations_per_row','truncations_per_row','requested_torque_max_per_row_nm','applied_torque_max_per_row_nm','requested_saturation_fraction_per_row'):
            if len(audit.get(key,[]))!=sel['replicas']: raise ValueError('Missing per-replica audit')
            for value in audit[key]:
                if finite(value)<0: raise ValueError('Negative raw metric')
        if max(audit['applied_torque_max_per_row_nm'])>1.60001: raise ValueError('Applied cap exceeded')
        for filename,key in [('training_trace.npz','trace_sha256'),('training_joint_trace.npz','joint_trace_sha256'),('training_events.json','event_ledger_sha256')]: check_hash(directory/filename,audit[key])
        expected_updates=[2] if sel['allocation']=='smoke' else [10,25,50]
        if set(receipt['decision_checkpoints'])!={str(x) for x in expected_updates}: raise ValueError('Decision checkpoint inventory differs')
        for update in expected_updates:
            row=receipt['decision_checkpoints'][str(update)]
            if row.get('file')!='decision_'+str(update).zfill(3)+'.pt' or row.get('completed_updates')!=update: raise ValueError('Wrong decision checkpoint label')
            check_hash(directory/'policy'/row['file'],row['sha256'])
        finite(receipt['wall_seconds'])
        result.update(complete=True,updates_completed=sel['updates'],checkpoint_sha256=bound,receipt_sha256=sha(directory/'training_receipt.json'),scope='Bounded optimization/integrity, not physical acceptance');return result
    expected=CHECKPOINT if phase.startswith('initial_') else expected_checkpoint_sha256
    if expected is None or state.get('checkpoint_sha256')!=expected: raise ValueError('Evaluation checkpoint unbound')
    filename='diagnostics.json' if phase.endswith('_constant') else 'stop_diagnostics.json'
    report=read(directory/filename)
    if not report.get('complete') or report.get('checkpoint_sha256')!=expected or report.get('overrides')!=EXPECTED_OVERRIDES or len(report.get('scenarios',[]))!=12: raise ValueError('Missing/mismatched directional diagnostic')
    obs=report['observation_audit']
    if obs.get('actor_width')!=315 or obs.get('critic_width')!=318 or any(obs.get(k)!=0 for k in ('max_command_slice_difference','max_same_step_repeat_difference','max_history_shift_difference')): raise ValueError('Observation history/command audit failed')
    if phase.endswith('_constant'):
        if report.get('kind')!='diagnostic_not_qualification' or report.get('options')!=identity['diagnostic_options']: raise ValueError('Wrong unchanged constant diagnostic')
        for row in report['scenarios']:
            if finite(row['windows']['all']['applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded')
        trace='diagnostic_trace.npz'
    else:
        if report.get('kind')!='direct315_move_to_zero_diagnostic_v1' or report.get('controls')!=1600 or report.get('replicas')!=48: raise ValueError('Wrong stop protocol')
        if finite(report['all_control_applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded outside scored windows')
        failed=[]
        for row in report['scenarios']:
            if len(row.get('replicas',[]))!=4: raise ValueError('Missing stop replicas')
            for replica in row['replicas']:
                quiet=replica['quiet']
                if quiet.get('window_samples')!=500 or quiet.get('window_duration_s')!=10.: raise ValueError('Incomplete quiet window')
                if finite(quiet['max_applied_torque_nm'])>1.60001 or finite(replica['motion']['applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded')
                if not quiet['pass']: failed.append(replica['env_id'])
        result['quiet_failed_env_ids']=failed;trace='stop_trace.npz'
    result.update(complete=True,checkpoint_sha256=expected,report_sha256=sha(directory/filename),trace_sha256=sha(directory/trace),scope='Acquisition completion only; all acceptance failures retained')
    return result

def runtime_arguments(phase,allocation,branch):
    if phase not in PHASES: raise ValueError('Undeclared phase')
    selected=selection('train',allocation,branch,None,None)
    if allocation=='smoke' and (branch!='caps' or phase not in ('standing','train','final_constant','final_stop')): raise ValueError('Smoke scope is standing, CAPS two updates and matched final diagnostics only')
    mode='validate' if phase=='standing' else ('train' if phase=='train' else 'evaluate')
    argv=['/source/tools/train_length_study.py','--package','/source/robot/hexapod_mkii_length_study','--output','/output/'+phase,'--variant','f050_t060','--stance-index','0','--mode',mode,
          '--headless','--device','cuda:0','--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']
    if phase!='standing': argv+=['--admission','/admission/admission.json','--checkpoint','/checkpoint/original.pt' if phase=='train' else '/checkpoint/evaluated.pt']
    if phase=='train': argv+=['--direct-allocation',allocation,'--direct-branch',branch,'--iterations',str(selected['updates'])]
    elif phase!='standing': argv+=['--direct-evaluation',phase.split('_')[1]]
    return argv

```

FILE curriculum.py
```python
"""Explicit 25% quiet rows; others alternate8s motion /8s stop targets.

No new observation field or hidden motor state: targets ramp through the exact
old slew_commands function into the command already observed by the actor.
"""
import math
import torch
class StandStopSchedule:
 def __init__(self,count,device,seed=157,dt=.02):
  if count<4 or dt!=.02:raise ValueError('At least4 rows at exact50Hz required')
  self.n=count;self.device=device;self.dt=dt
  self.generator=torch.Generator(device=device);self.generator.manual_seed(seed)
  ids=torch.arange(count,device=device)
  self.quiet=ids%4==0;self.offset=(ids%4)*200
  self.age=torch.zeros(count,device=device,dtype=torch.long);self.episode=torch.zeros_like(self.age)
  self.motion=torch.zeros(count,3,device=device);self.reset(ids)
 def reset(self,ids):
  ids=ids.long();u=torch.rand(len(ids),4,device=self.device,generator=self.generator)
  a=2*math.pi*u[:,0];speed=.06+.14*u[:,1];yaw=(.12+.28*u[:,2])*torch.where(u[:,3]<.5,-1.,1.)
  # A separate draw prevents correlation of yaw sign and motion category.
  category=torch.rand(len(ids),device=self.device,generator=self.generator)
  translating=category>=.25;turning=(category<.25)|(category>=.75)
  self.motion[ids]=torch.stack((speed*a.cos()*translating,speed*a.sin()*translating,yaw*turning),-1)
  self.age[ids]=0;self.episode[ids]+=1
 def advance(self):
  self.age+=1;moving=((self.age+self.offset)%800<400)&~self.quiet
  return torch.where(moving[:,None],self.motion,0.)
 def state(self):return {'age_controls':self.age.clone(),'episode':self.episode.clone(),'quiet_rows':self.quiet.clone(),'motion_twist':self.motion.clone(),'phase_offset_controls':self.offset.clone()}

def make_environment(parent):
 class BalancedOmniEnv(parent):
  def __init__(self,cfg,render_mode=None,*,evaluation=False,**kw):
   self.direct_schedule=None
   # The parent's evaluation flag only disables random command sampling;
   # its physical stepping, reward timing and command slew remain exact.
   super().__init__(cfg,render_mode,evaluation=True,**kw)
   if not evaluation:self.direct_schedule=StandStopSchedule(self.num_envs,self.device,cfg.seed,self.step_dt)
  def _sample_commands(self,ids):
   self._commands[ids]=0
   if hasattr(self,'omni_targets'):self.omni_targets[ids]=0
   if self.direct_schedule is not None:self.direct_schedule.reset(ids)
  def _get_rewards(self):
   if self.direct_schedule is not None:self.omni_targets.copy_(self.direct_schedule.advance())
   return super()._get_rewards()
 return BalancedOmniEnv

```

FILE caps.py
```python
"""CAPS-style policy-mean regularization; preserves the 315/318 model interface.

Only auxiliary rollout storage gains keys. Command/action history is never
perturbed. Temporal pairs exclude resets and command changes. Normalizers are
not updated by synthetic neighbors. This is a proposal, not admitted physics.
"""
import math
import torch
from tensordict import TensorDict
WIDTH=315

def validated_options(o):
 if not isinstance(o,dict) or set(o)!={'temporal_weight','spatial_weight','noise_scale','noise_seed'}:raise ValueError('Exact CAPS options required')
 for k in ['temporal_weight','spatial_weight','noise_scale']:
  if isinstance(o[k],bool) or not isinstance(o[k],(float,int)) or not math.isfinite(o[k]) or not 0<=o[k]<=1:raise ValueError('Invalid CAPS coefficient')
 if not isinstance(o['noise_seed'],int) or isinstance(o['noise_seed'],bool):raise ValueError('Integer independent noise seed required')
 return dict(o)

def noise_scales(device,dtype):
 s=torch.zeros(5,63,device=device,dtype=dtype)
 s[:,:3]=.00375;s[:,3:6]=.01;s[:,9:27]=.005;s[:,27:45]=.0025
 return s.flatten()

def latest_commands(obs):return obs[:,4*63+6:4*63+9]

class PairState:
 def __init__(self,policy):
  self._check(policy);self.current=policy.detach().clone();self.previous=torch.zeros_like(policy);self.valid=torch.zeros(len(policy),1,device=policy.device,dtype=torch.bool)
 @staticmethod
 def _check(p):
  if p.ndim!=2 or p.shape[1]!=WIDTH or not torch.isfinite(p).all():raise ValueError('Finite Nx315 actor observations required')
 def advance(self,next_policy,dones):
  self._check(next_policy)
  if next_policy.shape!=self.current.shape or dones.shape!=(len(next_policy),):raise ValueError('Pair shape mismatch')
  unchanged=(latest_commands(next_policy)==latest_commands(self.current)).all(-1)
  self.valid=(~dones.bool() & unchanged)[:,None]
  self.previous=torch.where(self.valid,self.current,0.).detach().clone()
  self.current=next_policy.detach().clone()
 def decorate(self,obs):
  if not torch.equal(obs['policy'],self.current):raise ValueError('Cached observation changed without a step')
  out=obs.clone();out['caps_previous_policy']=self.previous.clone();out['caps_pair_valid']=self.valid.clone();return out

class CapsPairWrapper:
 def __init__(self,env):
  self.env=env;self.cached=env.get_observations().clone();self.pairs=PairState(self.cached['policy']);self.failure=None
 def __getattr__(self,key):return getattr(self.env,key)
 @property
 def episode_length_buf(self):return self.env.episode_length_buf
 @episode_length_buf.setter
 def episode_length_buf(self,value):self.env.episode_length_buf=value
 def get_observations(self):
  if self.failure is not None:raise RuntimeError(self.failure)
  return self.pairs.decorate(self.cached)
 def step(self,action):
  if self.failure is not None:raise RuntimeError(self.failure)
  try:
   obs,reward,done,extras=self.env.step(action)
   self.pairs.advance(obs['policy'],done);self.cached=obs.clone()
   return self.get_observations(),reward,done,extras
  except BaseException as exc:self.failure=repr(exc);raise

class MeanRegularizer:
 def __init__(self,options,device):
  self.options=validated_options(options);self.generator=torch.Generator(device=device);self.generator.manual_seed(options['noise_seed']);self.stats=[]
 def __call__(self,actor,obs):
  o=self.options
  if actor.obs_groups!=['policy'] or actor.is_recurrent:raise ValueError('Exact feedforward policy-only315 actor required')
  if not o['temporal_weight'] and not o['spatial_weight']:
   # No actor calls or random draws: the zero branch is an exact PPO ablation.
   return obs['policy'].new_zeros(())
  p=obs['policy'];previous=obs['caps_previous_policy'];valid=obs['caps_pair_valid']
  PairState._check(p);PairState._check(previous)
  if previous.shape!=p.shape or valid.shape!=(len(p),1) or not torch.isfinite(valid).all() or not ((valid==0)|(valid==1)).all():raise ValueError('Malformed temporal pairing evidence')
  # RSL5.0.1 RolloutStorage stores observation keys as float tensors.
  # Require exact binary values before interpreting this auxiliary mask.
  valid=valid.bool()
  mu=actor(obs)
  prev=obs.clone();prev['policy']=previous
  temporal=((mu-actor(prev)).square().mean(-1)*valid[:,0]).sum()/valid.sum().clamp_min(1)
  neighbor=obs.clone();noise=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=self.generator).clamp(-3,3)*noise_scales(p.device,p.dtype)*o['noise_scale'];neighbor['policy']=p+noise
  spatial=(mu-actor(neighbor)).square().mean()
  loss=o['temporal_weight']*temporal+o['spatial_weight']*spatial
  if not torch.isfinite(loss):raise ValueError('Nonfinite regularization loss')
  self.stats.append({'temporal':float(temporal.detach()),'spatial':float(spatial.detach()),'weighted':float(loss.detach()),'valid_pair_fraction':float(valid.float().mean())})
  return loss

```

FILE caps_ppo.py
```python
# Derived from RSL-RL5.0.1 PPO.update, BSD-3-Clause.
# Copyright (c)2021-2026 ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# Full license preserved in inputs/RSL_LICENSE.
import hashlib, inspect
from pathlib import Path
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from caps import MeanRegularizer
BASE_SHA='a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c'
class CapsPPO(PPO):
    def __init__(self,*args,caps_options,**kwargs):
        if hashlib.sha256(Path(inspect.getfile(PPO)).read_bytes()).hexdigest()!=BASE_SHA:
            raise RuntimeError('Installed RSL PPO differs from reviewed source')
        super().__init__(*args,**kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('This bounded CAPS consumer excludes RND/symmetry/distributed/recurrent modes')
        self.regularizer=MeanRegularizer(caps_options,self.device)
    def update(self):
        self.regularizer.stats=[]
        result=self._caps_update()
        for key in ['temporal','spatial','weighted','valid_pair_fraction']:
            rows=self.regularizer.stats
            result['caps_'+key]=sum(r[key] for r in rows)/len(rows) if rows else 0.
        return result
    def _caps_update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = loss + self.regularizer(self.actor, batch.observations)

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

```

FILE direct_training.py
```python
"""Bounded training evidence and real save/reload check; legacy physics is unchanged."""
import copy, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, d): Path(p).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')

def audited_environment(parent):
    class AuditedDirectEnv(parent):
        def __init__(self, *a, **kw):
            self.direct_audit_rows=[]; self.direct_audit_events=[]; self.direct_audit_samples=[]
            super().__init__(*a, **kw)
        def _get_rewards(self):
            command=self._commands.clone()
            reward=super()._get_rewards()
            d=self._robot.data
            distal, shaft, _=self._get_foot_contact_state()
            coxa=self._coxa_contact_sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(1)>1
            femur=torch.cat([s.data.net_forces_w_history.torch.norm(dim=-1).amax(1)>1 for s in self._femur_contact_sensors],1)
            nonfoot=torch.cat((coxa,femur,shaft),1).any(-1)
            dq=d.joint_vel.torch; target_delta=self._processed_actions-self.omni_previous_target
            v=self._vector_in_command_frame(d.root_lin_vel_b.torch)
            w=self._vector_in_command_frame(d.root_ang_vel_b.torch)
            row={'command':command,'reward':reward, 'age_s':self._episode_elapsed_s,
                 'requested_torque_abs_max_nm':d.computed_torque.torch.abs().amax(-1),
                 'applied_torque_abs_max_nm':d.applied_torque.torch.abs().amax(-1),
                 'requested_saturation_fraction':(d.computed_torque.torch.abs()>1.6).float().mean(-1),
                 'reported_joint_velocity_rms_rad_s':dq.square().mean(-1).sqrt(),
                 'target_delta_rms_rad':target_delta.square().mean(-1).sqrt(),
                 'velocity_navigation_mps':v,'gyro_navigation_rad_s':w,
                 'position_world_m':d.root_pos_w.torch,'quaternion_world_xyzw':d.root_quat_w.torch,
                 'terminated':self.reset_terminated,'truncated':self.reset_time_outs,
                 'nonfoot':nonfoot,'support_count':distal.sum(-1)}
            self.direct_audit_rows.append({k:x.detach().clone() for k,x in row.items()})
            detail={'joint_position_rad':d.joint_pos.torch,'joint_velocity_rad_s':dq,
                    'joint_target_rad':self._processed_actions,'computed_torque_nm':d.computed_torque.torch,
                    'applied_torque_nm':d.applied_torque.torch,'raw_policy_action':self.omni_raw_policy_action}
            self.direct_audit_samples.append({k:x[:8].detach().clone() for k,x in detail.items()})
            ids=torch.nonzero(self.reset_terminated|self.reset_time_outs).flatten()
            if len(ids):
                self.direct_audit_events.append({'control':len(self.direct_audit_rows),'ids':ids.detach().clone(),
                                                'fields':{k:x[ids].detach().clone() for k,x in {**row,**detail}.items()}})
            finite=torch.stack([torch.isfinite(x).all() for x in [*row.values(),*detail.values()]]).all()
            if not bool(finite) or bool((row['applied_torque_abs_max_nm']>1.60001).any()):
                raise RuntimeError('Nonfinite direct training evidence or applied cap exceeded; pre-reset row preserved')
            return reward
    return AuditedDirectEnv

def export_audit(env, output):
    output=Path(output)
    if not env.direct_audit_rows:
        return {'controls':0,'scope':'No physical learning samples collected'}
    raw={k:torch.stack([r[k] for r in env.direct_audit_rows]).cpu().numpy() for k in env.direct_audit_rows[0]}
    raw['time_s']=(np.arange(len(env.direct_audit_rows))+1)*env.step_dt
    raw['joint_names']=np.array(env._robot.joint_names)
    np.savez_compressed(output/'training_trace.npz',**raw)
    sample={k:torch.stack([r[k] for r in env.direct_audit_samples]).cpu().numpy() for k in env.direct_audit_samples[0]}
    np.savez_compressed(output/'training_joint_trace.npz',**sample,env_ids=np.arange(min(8,env.num_envs)),time_s=raw['time_s'],joint_names=raw['joint_names'])
    events=[{'control':e['control'],'ids':e['ids'].cpu().tolist(),'fields':{k:v.cpu().tolist() for k,v in e['fields'].items()}} for e in env.direct_audit_events]
    save(output/'training_events.json',events)
    return {'controls':len(env.direct_audit_rows),'replicas':env.num_envs,
            'terminations_per_row':raw['terminated'].sum(0).astype(int).tolist(),
            'truncations_per_row':raw['truncated'].sum(0).astype(int).tolist(),
            'requested_torque_max_per_row_nm':raw['requested_torque_abs_max_nm'].max(0).tolist(),
            'applied_torque_max_per_row_nm':raw['applied_torque_abs_max_nm'].max(0).tolist(),
            'requested_saturation_fraction_per_row':raw['requested_saturation_fraction'].mean(0).tolist(),
            'trace_sha256':sha(output/'training_trace.npz'),'joint_trace_sha256':sha(output/'training_joint_trace.npz'),
            'event_ledger_sha256':sha(output/'training_events.json'),
            'scope':'50 Hz pre-reset endpoint evidence; no 400 Hz motor qualification or SDK-rate fidelity claim'}

def equal_tree(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def verify_reload(runner, wrapped, checkpoint):
    saved=torch.load(checkpoint,map_location=runner.device,weights_only=False)
    before=copy.deepcopy(runner.alg.save())
    obs=wrapped.get_observations()
    with torch.inference_mode(): action=runner.alg.actor(obs).clone()
    # RSL updates normalization during inference-mode rollout; update() replaces
    # _std with an inference tensor. Strict load_state_dict copies into buffers
    # outside that context. Normalize only such buffers after learning, preserving
    # all values and parameter/optimizer objects; never alter the checkpoint.
    normalized_buffers=[]
    with torch.inference_mode(False), torch.no_grad():
        for model_name in ('actor','critic'):
            for module_name,module in getattr(runner.alg,model_name).named_modules():
                for name,value in tuple(module.named_buffers(recurse=False)):
                    if torch.is_inference(value):
                        ordinary=value.detach().clone()
                        if torch.is_inference(ordinary) or not torch.equal(value,ordinary):
                            raise RuntimeError('Inference buffer value-preserving clone failed')
                        setattr(module,name,ordinary)
                        normalized_buffers.append('.'.join(x for x in (model_name,module_name,name) if x))
        runner.load(str(checkpoint),strict=True,map_location=runner.device)
    after=runner.alg.save()
    if not equal_tree(before,after) or not equal_tree(saved['actor_state_dict'],after['actor_state_dict']) or not equal_tree(saved['critic_state_dict'],after['critic_state_dict']):
        raise RuntimeError('Strict actor/critic/normalizer/optimizer reload differs')
    with torch.inference_mode(): exact=torch.equal(action,runner.alg.actor(obs))
    if not exact: raise RuntimeError('Deterministic action changed after strict reload')
    return {'passed':True,'exact_actor_critic_normalizer_optimizer':True,'exact_deterministic_action':True,
            'optimizer_entries':len(runner.alg.optimizer.state),'checkpoint_sha256':sha(checkpoint),
            'inference_buffers_cloned_after_learning':normalized_buffers,
            'reload_method':'Strict same-runner load after value-preserving inference-buffer normalization'}

def learn_and_verify(env, wrapped, runner, selected, output):
    output=Path(output); start=time.perf_counter(); start_iteration=runner.current_learning_iteration
    receipt={'selection':selected,'initial_runner_iteration':start_iteration,'complete':False,'Stage2_complete':False,
             'updates_completed':0,'optimizer_updates':[]}
    original_update=runner.alg.update
    def observed_update():
        tick=time.perf_counter();losses=original_update()
        receipt['updates_completed']+=1
        receipt['optimizer_updates'].append({'completed_update':receipt['updates_completed'],
            'optimizer_wall_seconds':time.perf_counter()-tick,'losses':{k:float(v) for k,v in losses.items()},
            'learning_rate':float(runner.alg.learning_rate)})
        return losses
    runner.alg.update=observed_update
    try:
        # Same native20 s episodes and randomized initial episode ages as old direct PPO.
        runner.learn(num_learning_iterations=selected['updates'],init_at_random_ep_len=True)
        checkpoint=output/'policy/final.pt'
        if checkpoint.exists(): raise FileExistsError(checkpoint)
        runner.save(str(checkpoint))
        receipt['reload']=verify_reload(runner,wrapped,checkpoint)
        expected=selected['updates']*selected['controls_per_update']
        if len(env.direct_audit_rows)!=expected or receipt['updates_completed']!=selected['updates']: raise RuntimeError('Actual collected controls or optimizer updates differ from bounded allocation')
        receipts={}
        for update in ([2] if selected['allocation']=='smoke' else [10,25,50]):
            native=output/'policy'/('model_'+str(start_iteration+update-1)+'.pt')
            target=output/'policy'/('decision_'+str(update).zfill(3)+'.pt')
            if not native.is_file() or target.exists(): raise RuntimeError('Missing or existing immutable decision checkpoint')
            shutil.copyfile(native,target)
            receipts[str(update)]={'file':target.name,'sha256':sha(target),'native_iteration':start_iteration+update-1,'completed_updates':update}
        receipt.update(complete=True,updates_completed=selected['updates'],decision_checkpoints=receipts,
                       final_checkpoint_sha256=sha(checkpoint),final_runner_iteration=runner.current_learning_iteration,
                       learned_std=runner.alg.actor.distribution.std_param.detach().cpu().tolist())
        return receipt
    except BaseException as exc:
        receipt['error']=repr(exc);raise
    finally:
        runner.alg.update=original_update
        receipt['wall_seconds']=time.perf_counter()-start
        try: receipt['audit']=export_audit(env,output)
        except BaseException as exc:
            receipt.update(complete=False,audit_error=repr(exc));save(output/'training_receipt.json',receipt);raise
        if torch.cuda.is_available():
            receipt['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(env.device)
            receipt['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved(env.device)
        save(output/'training_receipt.json',receipt)

```

FILE direct_stop_evaluation.py
```python
"""Real movement-to-zero commands, full per-replica traces and unchanged quiet bounds."""
import json
from pathlib import Path
import numpy as np
import torch
from direct_quiet_metrics import QUIET_GATES, quiet_metrics

def target_at(control, moving_targets):
    if type(control) is not int or not 0 <= control < 1600:
        raise ValueError('Exact 32 s stop diagnostic control range required')
    return moving_targets if 200 <= control < 800 else torch.zeros_like(moving_targets)

def summarize(data, scenarios, names, dt):
    from omni_diagnostics import _summary
    if dt != .02 or data['command'].shape != (1600,48,3):
        raise ValueError('Complete32 s /48-replica stop diagnostic required')
    rows=[]
    for i, scenario in enumerate(scenarios):
        replicas=[]
        for replica in range(4):
            index=i*4+replica
            quiet=quiet_metrics(data,index,1100,names,dt)
            q=data['joint_position_rad'][1100:,index]
            quiet['interval_angle_velocity_rms_rad_s']=np.sqrt(np.mean((np.diff(q,axis=0)/dt)**2,axis=0)).tolist()
            quiet['interval_angle_note']='50Hz adjacent-angle interval average; separate from raw SDK joint velocity and unchanged gates'
            subset={k:v[:,index:index+1] for k,v in data.items()}
            mask=np.zeros((1600,1),dtype=bool);mask[300:800]=True
            replicas.append({'env_id':index,'motion':_summary(subset,mask,names),'quiet':quiet,
                             'requested_target_nonzero_controls':int(np.any(data['target_command'][:,index]!=0,axis=-1).sum())})
        rows.append({**scenario,'replicas':replicas})
    return rows

@torch.inference_mode()
def evaluate_stop(env, runner, plan, output, checkpoint_sha):
    from tensordict import TensorDict
    from omni_diagnostics import diagnostic_scenarios
    from omni_flat_evaluation import save
    if env.num_envs!=48 or env.step_dt!=.02: raise ValueError('Stop diagnostic requires48 environments at50Hz')
    output=Path(output);scenarios=diagnostic_scenarios()
    targets=torch.tensor([s['command'] for s in scenarios],device=env.device).repeat_interleave(4,0)
    policy=runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled=True; snapshots=[]
    observation_audit={'actor_width':315,'critic_width':318,'max_command_slice_difference':0.,'max_same_step_repeat_difference':0.,'max_history_shift_difference':0.}
    previous=None; previous_done=None; complete=False
    try:
        env.reset(seed=27057);env.episode_length_buf.zero_()
        for control in range(1600):
            env.set_evaluation_targets(target_at(control,targets))
            obs=env._get_observations();actor=obs['policy'];repeat=env._get_observations()
            observation_audit['max_same_step_repeat_difference']=max(observation_audit['max_same_step_repeat_difference'],float((actor-repeat['policy']).abs().max()))
            command=env._commands*torch.tensor([5.,5.,2.5],device=env.device)
            observation_audit['max_command_slice_difference']=max(observation_audit['max_command_slice_difference'],float((actor[:,-57:-54]-command).abs().max()))
            if previous is not None and (~previous_done).any():
                valid=~previous_done
                observation_audit['max_history_shift_difference']=max(observation_audit['max_history_shift_difference'],float((actor[valid,:-63]-previous[valid,63:]).abs().max()))
            previous=actor.clone()
            _,_,term,trunc,_=env.step(policy(TensorDict(obs,batch_size=[48])))
            previous_done=term|trunc
            sample={k:v.copy() for k,v in env.omni_diagnostic_sample.items()}
            # Exact bound legacy capture stores Isaac XYZW under its old WXYZ label.
            # Preserve raw pre-reset values, then explicitly convert this NEW trace.
            sample['quaternion_world_xyzw']=sample['quaternion_world_wxyz'].copy()
            sample['quaternion_world_wxyz']=sample['quaternion_world_xyzw'][...,[3,0,1,2]].copy()
            snapshots.append(sample)
            if not all(np.isfinite(v).all() for v in sample.values()): raise RuntimeError('Nonfinite stop diagnostic')
            if np.abs(sample['applied_torque_nm']).max()>1.60001: raise RuntimeError('Applied cap exceeded in preserved stop trace')
            if not np.array_equal(sample['terminated'],term.cpu().numpy()) or not np.array_equal(sample['truncated'],trunc.cpu().numpy()): raise RuntimeError('Pre-reset stop evidence mismatch')
            if (control+1)%100==0: print('DIRECT_STOP '+str(control+1)+'/1600',flush=True)
        complete=True
    finally:
        env.omni_diagnostic_enabled=False
        if snapshots:
            data={k:np.stack([s[k] for s in snapshots]) for k in snapshots[0]}
            names=list(env._robot.joint_names)
            np.savez_compressed(output/'stop_trace.npz',**data,joint_names=np.array(names),time_s=(np.arange(len(snapshots))+1)*.02)
            report={'complete':complete,'kind':'direct315_move_to_zero_diagnostic_v1','checkpoint_sha256':checkpoint_sha,
                    'controls':len(snapshots),'replicas':48,'scenarios':summarize(data,scenarios,names,.02) if complete else [],
                    'all_control_applied_torque_abs_max_nm':float(np.abs(data['applied_torque_nm']).max()),
                    'quiet_gates':QUIET_GATES,'observation_audit':observation_audit,
                    'schedule_s':{'initial_zero':[0,4],'motion':[4,16],'stop':[16,32],'scored_quiet':[22,32]},
                    'overrides':plan['omni']['overrides'],'Stage2_complete':False,
                    'scope':'Complete is acquisition only, failed quiet/torque/termination bounds remain failures; 50Hz not substep qualification'}
            save(output/'stop_diagnostics.json',report)
    return report

```

FILE direct_quiet_metrics.py
```python
"""Byte-preserved quiet scorer bodies from velocity003; new caller supplies explicit WXYZ.
Source SHA256 63c55e8c480e5bdb319dae2ecb498f9752b9529f2dc5e7b2abc05513e8f6757d
"""
import numpy as np

QUIET_GATES = {
    "max_planar_excursion_m": .01,
    "max_heading_excursion_deg": 2.,
    "max_joint_velocity_rms_rad_s": .03,
    "max_joint_position_range_rad": .02,
    "max_target_step_abs_p95_rad_per_20ms": .002,
    "max_requested_torque_saturation_fraction": .005,
    "max_applied_torque_nm": 1.60001,
}

def quiet_metrics(data, env_index, start_step, joint_names, dt):
    """Score one contiguous window without deleting failures or restarting time."""
    take = lambda key: data[key][start_step:, env_index]
    q = take("joint_position_rad")
    target = take("joint_target_rad")
    velocity = take("joint_velocity_rad_s")
    position = take("position_world_m")
    quat = take("quaternion_world_wxyz")
    if len(q) < 2:
        raise ValueError("Quiet window needs at least two samples")
    w, x, y, z = quat.T
    heading = np.unwrap(np.arctan2(-1 + 2*(x*x+z*z), 2*(w*z-x*y)))
    joint_rms = np.sqrt(np.mean(velocity**2, axis=0))
    qrange = np.ptp(q, axis=0)
    target_p95 = np.quantile(np.abs(np.diff(target, axis=0)) * .02 / dt, .95, axis=0)
    requested = take("computed_torque_nm")
    if not all(np.isfinite(value).all() for value in (q, target, velocity, position, quat, requested, take("applied_torque_nm"))):
        raise ValueError("Nonfinite quiet-review state")
    row = {
        "window_samples": len(q), "window_duration_s": len(q) * dt,
        "max_planar_excursion_m": float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max()),
        "max_heading_excursion_deg": float(np.degrees(np.abs(heading - heading[0]).max())),
        "max_joint_velocity_rms_rad_s": float(joint_rms.max()),
        "max_joint_position_range_rad": float(qrange.max()),
        "max_target_step_abs_p95_rad_per_20ms": float(target_p95.max()),
        "max_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean(0).max()),
        "mean_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean()),
        "max_applied_torque_nm": float(np.abs(take("applied_torque_nm")).max()),
        "requested_torque_abs_max_nm": float(np.abs(requested).max()),
        # Any failure anywhere in the trial invalidates the result, even if the
        # robot is quiet after an automatic reset inside/before the scored window.
        "terminations": int(data["terminated"][:, env_index].sum()),
        "truncations": int(data["truncated"][:, env_index].sum()),
        "joints": {name: {"velocity_rms_rad_s": float(joint_rms[j]),
                          "position_range_rad": float(qrange[j]),
                          "target_step_abs_p95_rad_per_20ms": float(target_p95[j]),
                          "saturation_fraction": float((np.abs(requested[:, j]) > 1.6).mean())}
                   for j, name in enumerate(joint_names)},
    }
    row["failed_bounds"] = [key for key, bound in QUIET_GATES.items() if row[key] > bound]
    row["pass"] = not row["failed_bounds"] and row["terminations"] == 0 and row["truncations"] == 0
    return row

```
