"""CPU-only schedule, integrity checks and descriptive policy metrics."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SCHEMA = 'hexapod.deterministic_flat_policy_evaluation.v1'
SOURCE_SHA256 = 'c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc'
CAPTURE_HELPER_SHA256 = 'ff60b5227ea8fbc2f3fc3ffe5091f0691ccff13cc1dab8e85063bb48d24df993'
DT = .02
PHASES = (('settle',50),('motion',150),('stop',100),('reverse',150))
CASES = (('stand',(0.,0.,0.)),('forward',(.15,0.,0.)),('backward',(-.15,0.,0.)),
    ('left',(0.,.15,0.)),('right',(0.,-.15,0.)),
    ('forward_left',(.10,.10,0.)),('forward_right',(.10,-.10,0.)),
    ('backward_left',(-.10,.10,0.)),('backward_right',(-.10,-.10,0.)),
    ('yaw_left',(0.,0.,.30)),('yaw_right',(0.,0.,-.30)),('combined',(.10,.05,.20)))
TOOL_FILES = ('evaluation_common.py','evaluate_policy.py','run_evaluation.py')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def helpers(directory):
    path=Path(directory).resolve(strict=True)/'capture_common.py'
    if digest(path)!=CAPTURE_HELPER_SHA256:raise ValueError('Frozen capture helper hash differs')
    name='deterministic_eval_frozen_capture_common'
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path:raise ValueError('Capture helper path changed')
        return module
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def tool_identity(directory):return {name:digest(Path(directory)/name) for name in TOOL_FILES}

def verify_inputs(common,source,checkpoint,admission,training):
    result=common.verify_inputs(source,checkpoint,admission,training)
    if result['contract']['sha256']!=SOURCE_SHA256:raise ValueError('Evaluation requires frozen 1600 Hz source')
    return result

def schedule(step):
    import numpy as np
    if type(step) is not int or not 0<=step<450:raise ValueError('Schedule index outside 450-control evaluation')
    offset=0
    for name,count in PHASES:
        if step<offset+count:
            commands=np.asarray([value for _,value in CASES],dtype=np.float32)
            commands*=1. if name=='motion' else -1. if name=='reverse' else 0.
            return name,commands
        offset+=count
    raise AssertionError('Unreachable')

def schedule_descriptor():
    return dict(control_dt_s=DT,controls=450,seconds=9.,cases=[dict(name=n,motion_command=list(v)) for n,v in CASES],
        phases=[dict(name=n,controls=c,seconds=c*DT) for n,c in PHASES],
        command_override=True,policy_deterministic=True,reverse_semantics='negative of each motion command')

def require_request(request,verified,tools):
    if (request.get('schema')!=SCHEMA or request.get('contract')!=verified['contract']
        or request.get('input_sha256')!=verified['input_sha256'] or request.get('evaluation_tools_sha256')!=tools
        or request.get('frozen_capture_helper_sha256')!=CAPTURE_HELPER_SHA256
        or request.get('schedule')!=schedule_descriptor()):raise ValueError('Guarded evaluation request mismatch')

def assert_command_observation(command,observation):
    import numpy as np
    if command.shape!=(12,3) or observation.shape!=(12,84) or not np.array_equal(command,observation[:,9:12]):
        raise ValueError('Scheduled command differs from actual consumed policy observation')

def body_to_navigation(value):
    import numpy as np
    value=np.asarray(value)
    if value.shape[-1]!=3:raise ValueError('Body vector must end with three coordinates')
    return np.stack((-value[...,1],value[...,0],value[...,2]),axis=-1)


def summarize(states):
    import numpy as np
    cmd=np.asarray(states['command_navigation']);vel=np.asarray(states['end_com_velocity_navigation'])
    done=np.asarray(states['done']);pos=np.asarray(states['pre_plate_pos_w_m']);end=np.asarray(states['end_plate_pos_w_m'])
    if (cmd.shape!=(450,12,3) or vel.shape!=cmd.shape or done.shape!=(450,12)
        or pos.shape!=cmd.shape or end.shape!=cmd.shape or not np.isin(done,[0,1]).all()
        or any(not np.isfinite(v).all() for v in (cmd,vel,pos,end))):raise ValueError('Invalid evaluation trace shapes or values')
    if not np.array_equal(cmd,np.stack([schedule(i)[1] for i in range(450)])):raise ValueError('Trace did not follow frozen command schedule')
    obs=np.asarray(states['observation_policy'])
    if obs.shape!=(450,12,84) or not np.array_equal(obs[:,:,9:12],cmd):raise ValueError('Trace contains stale command observations')
    for key,width in (('end_foot_pad',6),('end_foot_slip_speed_mps',6),('end_applied_motor_torque_nm',18),
            ('end_raw_motor_demand_nm',18),('end_motor_clipping_nm',18),('end_burst_headroom',18),('end_joint_acc_rad_s2',18)):
        value=np.asarray(states[key])
        if value.shape!=(450,12,width) or not np.isfinite(value).all():raise ValueError('Invalid evaluation trace field: '+key)
    result=[]
    for world,(case,_) in enumerate(CASES):
        row={'case':case,'world':world,'reset_transition_indices':np.flatnonzero(done[:,world]).tolist(),'phases':{}}
        start=0
        for phase,count in PHASES:
            sl=slice(start,start+count);keep=~done[sl,world].astype(bool);start+=count
            actual=vel[sl,world][keep];target=cmd[sl,world][keep]
            block={'transitions':count,'transitions_used':int(keep.sum()),'reset_count':int((~keep).sum())}
            if keep.any():
                error=actual-target
                delta=end[sl,world][keep]-pos[sl,world][keep]
                block.update(command=target[0].tolist(),mean_com_velocity=actual.mean(0).tolist(),
                    tracking_rmse=np.sqrt(np.mean(error**2,0)).tolist(),tracking_mae=np.mean(np.abs(error),0).tolist(),
                    absolute_error_p95=np.quantile(np.abs(error),.95,axis=0).tolist(),
                    plate_planar_path_length_m=float(np.linalg.norm(delta[:,:2],axis=1).sum()),
                    valid_plate_displacement_world_m=delta.sum(0).tolist(),
                    planar_com_speed_mean_mps=float(np.linalg.norm(actual[:,:2],axis=1).mean()),
                    absolute_yaw_rate_mean_rad_s=float(np.abs(actual[:,2]).mean()),
                    near_stationary_fraction=float((np.linalg.norm(actual[:,:2],axis=1)<.01).mean()),
                    minimum_foot_support=int(np.asarray(states['end_foot_pad'])[sl,world][keep].sum(-1).min()),
                    loaded_pad_slip_rms_mps=float(np.sqrt((np.asarray(states['end_foot_slip_speed_mps'])[sl,world][keep]**2
                        *np.asarray(states['end_foot_pad'])[sl,world][keep]).sum()/max(1,np.asarray(states['end_foot_pad'])[sl,world][keep].sum()))))
                block.update(max_applied_torque_nm=float(np.abs(np.asarray(states['end_applied_motor_torque_nm'])[sl,world][keep]).max()),
                    max_raw_motor_demand_nm=float(np.abs(np.asarray(states['end_raw_motor_demand_nm'])[sl,world][keep]).max()),
                    minimum_burst_headroom=float(np.asarray(states['end_burst_headroom'])[sl,world][keep].min()),
                    motor_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.asarray(states['end_joint_acc_rad_s2'])[sl,world][keep]**2))),
                    motor_clipping_fraction=float((np.asarray(states['end_motor_clipping_nm'])[sl,world][keep]>1e-6).mean()))
                if phase=='stop':block['stop_path_distance_over_valid_transitions_m']=block['plate_planar_path_length_m']
            row['phases'][phase]=block
        result.append(row)
    return {'schema':SCHEMA,'cases':result,'policy_skill_pass':None,'hardware_admission':False,
        'navigation_or_terrain_qualification':False,
        'velocity_semantics':'Actual final-physics-substep root COM linear velocity and body angular velocity in anatomical navigation coordinates, before automatic reset; this is the frozen reward velocity source.',
        'reset_semantics':'Done transitions excluded from tracking/path summaries; pre-reset terminal states remain in raw traces. Later post-reset transitions remain and resets are counted, so reset-free survival must be assessed separately.',
        'near_stationary_threshold_semantics':'0.01 m/s is descriptive only; pure-yaw and stand cases intentionally can have zero translation.',
        'scope':'Descriptive fixed-command flat-model evaluation, not reward-derived walking success or a new admission gate.'}

def validate_report(path,verified,tools):
    path=Path(path);result=json.loads(path.read_text())
    if (result.get('schema')!=SCHEMA or result.get('pass') is not True or result.get('errors')!=[]
        or result.get('controls_completed')!=450 or result.get('physics_substeps')!=14400
        or result.get('num_envs')!=12 or result.get('policy_dt_s')!=DT or result.get('schedule')!=schedule_descriptor()
        or result.get('runtime_manifest')!=verified['runtime_manifest']
        or result.get('admitted_runtime_comparison',{}).get('pass') is not True
        or result.get('contract')!=verified['contract']
        or result.get('input_sha256')!=verified['input_sha256'] or result.get('evaluation_tools_sha256')!=tools
        or result.get('policy_skill_pass') is not None
        or any(result.get(k) is not True for k in ('physics_coverage_pass','policy_state_unchanged','checkpoint_unchanged'))
        or result.get('hardware_admission') is not False or result.get('navigation_or_terrain_qualification') is not False):
        raise ValueError('Evaluation execution/integrity report mismatch')
    for name in ('states.npz','metadata.json','metrics.json'):
        data=(path.parent/name).read_bytes();record=result['artifacts'][name]
        if hashlib.sha256(data).hexdigest()!=record['sha256'] or len(data)!=record['bytes']:raise ValueError('Evaluation artifact hash mismatch')
    return result


class ScheduledCommands:
    """Temporary external command override; frozen reset/physics methods still run."""
    def __init__(self,raw):self.raw=raw;self.current=None;self.calls=0
    def __enter__(self):
        self.had='_sample_commands' in vars(self.raw);self.previous=vars(self.raw).get('_sample_commands')
        self.original=self.raw._sample_commands
        def sample(env_ids):
            if self.current is None:raise ValueError('Reset command requested before schedule selected')
            self.raw._commands[env_ids]=self.current[env_ids]
            self.raw._command_time_left_s[env_ids]=self.raw.cfg.episode_length_s+1.
            self.calls+=1
        self.raw._sample_commands=sample
        return self
    def prepare(self,step,wrapped):
        import torch
        phase,cmd=schedule(step)
        self.current=torch.as_tensor(cmd,device=self.raw._commands.device,dtype=self.raw._commands.dtype)
        self.raw._commands.copy_(self.current)
        self.raw._command_time_left_s.fill_(self.raw.cfg.episode_length_s+1.)
        # DirectRLEnv's source getter refreshes obs using this command. Its
        # previous-action copy repeats the already recorded last clipped action.
        obs=wrapped.get_observations()
        assert_command_observation(self.raw._commands.detach().cpu().numpy(),obs['policy'].detach().cpu().numpy())
        return phase,obs
    def __exit__(self,*_):
        if self.had:self.raw._sample_commands=self.previous
        else:del self.raw._sample_commands


class EndStateCapture:
    """Read final-step physical state while source done computation precedes reset."""
    def __init__(self,raw,sample):self.raw=raw;self.sample=sample;self.latest=None;self.calls=0
    def __enter__(self):
        self.had='_get_dones' in vars(self.raw);self.previous=vars(self.raw).get('_get_dones');self.original=self.raw._get_dones
        def dones():
            result=self.original()
            self.latest=self.sample()
            self.calls+=1
            return result
        self.raw._get_dones=dones
        return self
    def __exit__(self,*_):
        if self.had:self.raw._get_dones=self.previous
        else:del self.raw._get_dones
