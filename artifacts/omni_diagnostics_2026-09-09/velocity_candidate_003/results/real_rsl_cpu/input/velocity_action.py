"""New serial C-study action semantics; CPU/Torch only, never imports a runtime.

Discrete controller: acceleration-bounded target velocity, then semi-implicit
position integration. Bounds concern executable 20 ms target knots, not motor
speed capability or a claim of continuous-jerk/torque feasibility.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import torch

PROFILES = {'diagnostic_003': .03, 'formal_004': .04}
LINEAGE = 'c_serial_omni_target_velocity_v1'

@dataclass(frozen=True)
class VelocityActionConfig:
    profile: str
    max_acceleration_rad_s2: float = 8.
    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError('Choose diagnostic_003 or formal_004; never pool profiles')
        value=self.max_acceleration_rad_s2
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not .1<=value<=20:
            raise ValueError('Explicit finite acceleration in [0.1,20] rad/s^2 required')
    @property
    def max_velocity_rad_s(self): return PROFILES[self.profile]/.02
    def contract(self):
        return {'lineage':LINEAGE,'profile':self.profile,'max_position_step_rad_per_20ms':PROFILES[self.profile],
                'max_target_velocity_rad_s':self.max_velocity_rad_s,'max_target_acceleration_rad_s2':self.max_acceleration_rad_s2,
                'integration':'semi_implicit_discrete','actor_frame_width':99,'actor_width':495,'critic_width':498,
                'physics_admitted':False,'motor_speed_limit_claim':False}

@dataclass
class ExecutableTarget:
    target_position_rad: torch.Tensor
    target_velocity_rad_s: torch.Tensor
    target_acceleration_rad_s2: torch.Tensor
    desired_velocity_rad_s: torch.Tensor
    clipped_action: torch.Tensor
    action_clipped: torch.Tensor
    acceleration_limited: torch.Tensor
    joint_braking_active: torch.Tensor
    valid: torch.Tensor

class JointTargetVelocity:
    """Stateful named-joint action path with anticipatory limit braking.

    The full state needed for the next step is target position and velocity.
    No command-dependent branch can disable balance feedback. Zero action asks
    velocity to settle to zero and then holds the reached target; it does not
    move grounded feet back to a preferred neutral posture.
    """
    def __init__(self, joint_names: Sequence[str], lower_by_name: Mapping[str,float],
                 upper_by_name: Mapping[str,float], num_envs: int, config: VelocityActionConfig,
                 *, device='cpu', dtype=torch.float64):
        names=tuple(joint_names)
        if len(names)!=18 or len(set(names))!=18 or set(names)!=set(lower_by_name) or set(names)!=set(upper_by_name):
            raise ValueError('Exactly 18 unique runtime names with complete named limits required')
        if type(num_envs) is not int or num_envs<=0: raise ValueError('Positive num_envs required')
        self.joint_names=names;self.config=config
        self.lower=torch.tensor([lower_by_name[n] for n in names],device=device,dtype=dtype)
        self.upper=torch.tensor([upper_by_name[n] for n in names],device=device,dtype=dtype)
        if not torch.is_floating_point(self.lower) or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all() or not (self.lower<self.upper).all():
            raise ValueError('Finite ordered joint limits required')
        self.position=torch.zeros(num_envs,18,device=device,dtype=dtype)
        self.velocity=torch.zeros_like(self.position)
        self.initialized=torch.zeros(num_envs,device=device,dtype=torch.bool)

    def _checked_values(self, value, shape, name):
        x=torch.as_tensor(value,device=self.position.device,dtype=self.position.dtype)
        if x.shape!=shape or not torch.isfinite(x).all():raise ValueError(f'{name}: matched finite tensor required')
        return x

    def reset(self, measured_position, env_ids=None, *, target_velocity=None):
        """Anchor reset targets to actual reset joint positions, not stale actions.

        Default target velocity is zero. Optional nonzero target-state restoration
        must be position/rate/braking viable; this is not a policy checkpoint load.
        No state is changed if any requested reset row is invalid.
        """
        if env_ids is None:ids=torch.arange(len(self.position),device=self.position.device)
        else:
            ids=torch.as_tensor(env_ids,device=self.position.device)
            if ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64) or len(ids)!=len(torch.unique(ids)) or (ids<0).any() or (ids>=len(self.position)).any():
                raise ValueError('Unique in-range integer reset IDs required')
        q=self._checked_values(measured_position,(len(ids),18),'reset position')
        v=torch.zeros_like(q) if target_velocity is None else self._checked_values(target_velocity,q.shape,'reset velocity')
        a=self.config.max_acceleration_rad_s2
        if ((q<self.lower)|(q>self.upper)).any() or (v.abs()>self.config.max_velocity_rad_s).any():
            raise ValueError('Reset target outside declared position/rate bounds')
        if (q+v.clamp_min(0).square()/(2*a)>self.upper+1e-10).any() or (q-v.clamp_max(0).square()/(2*a)<self.lower-1e-10).any():
            raise ValueError('Reset velocity cannot brake before joint limit')
        self.position[ids]=q;self.velocity[ids]=v;self.initialized[ids]=True

    def step(self, raw_action, dt=.02):
        """Emit one executable knot; dt and every constraint are explicit.

        q_next=q+dt*v_next and |v_next-v|<=a*dt. The additional bound
        dt*|v_next| + v_next^2/(2*a) <= remaining joint distance guarantees
        a conservative braking reserve after this position increment.
        """
        if isinstance(dt,bool) or not isinstance(dt,(int,float)) or not math.isfinite(dt) or not 0<dt<=.02:
            raise ValueError('Finite 0<dt<=20ms required')
        if not self.initialized.all():raise RuntimeError('Reset every environment before stepping')
        raw=self._checked_values(raw_action,self.position.shape,'action')
        q=self.position;v=self.velocity;a=self.config.max_acceleration_rad_s2;vmax=self.config.max_velocity_rad_s
        clipped=raw.clamp(-1.,1.);desired=clipped*vmax
        up=(self.upper-q).clamp_min(0);down=(q-self.lower).clamp_min(0)
        # Stable positive root of dt*x+x^2/(2a)=distance. No cancellation near a limit.
        def brake_bound(distance):
            return (2*a*distance)/((a*a*dt*dt+2*a*distance).sqrt()+a*dt)
        upper_velocity=brake_bound(up).clamp_max(vmax)
        lower_velocity=-brake_bound(down).clamp_max(vmax)
        low=torch.maximum(v-a*dt,lower_velocity);high=torch.minimum(v+a*dt,upper_velocity)
        tolerance=32*torch.finfo(q.dtype).eps
        if (low>high+tolerance).any():raise RuntimeError('Controller state lost braking viability; no target emitted')
        # A sub-ULP interval mismatch can arise at a float32 boundary. This does
        # not authorize projection of an infeasible physical target.
        high=torch.maximum(high,low)
        nxt=desired.maximum(low).minimum(high)
        position=q+dt*nxt
        acceleration=(nxt-v)/dt
        if (position<self.lower-tolerance).any() or (position>self.upper+tolerance).any() or not torch.isfinite(position).all():
            raise RuntimeError('Target bound violation; no state committed')
        # Rollouts use inference_mode; persisted state must also reset safely
        # outside that context. No action gradient is part of this controller.
        with torch.inference_mode(False):
            result=ExecutableTarget(position.detach().clone(),nxt.detach().clone(),acceleration.detach().clone(),desired.detach().clone(),clipped.detach().clone(),
                (raw!=clipped).clone(),((desired<v-a*dt)|(desired>v+a*dt)).clone(),
                ((desired<lower_velocity)|(desired>upper_velocity)).clone(),torch.ones(len(q),device=q.device,dtype=torch.bool))
            self.position=position.detach().clone();self.velocity=nxt.detach().clone()
        return result

    def observable_state(self, default_position):
        """36 noiseless, controller-known values appended to each 63-wide frame."""
        default=self._checked_values(default_position,self.position.shape,'default position')
        if not self.initialized.all():raise RuntimeError('Uninitialized controller observation')
        return torch.cat((self.position-default,self.velocity/self.config.max_velocity_rad_s),-1).clone()


def append_executable_state(base_frame, controller:JointTargetVelocity, default_position):
    """Current 63-wide sensor/command/action frame plus target offset/velocity."""
    frame=torch.as_tensor(base_frame,device=controller.position.device,dtype=controller.position.dtype)
    if frame.shape!=(len(controller.position),63) or not torch.isfinite(frame).all():
        raise ValueError('Finite 63-wide base frame required')
    return torch.cat((frame,controller.observable_state(default_position)),-1)


def verify_new_lineage_checkpoint(metadata,config,expected_sha256,actual_sha256):
    """Fail closed before a runner load. Old 315/318 actors are incompatible."""
    expected={'lineage':LINEAGE,'profile':config.profile,'actor_width':495,'critic_width':498,
              'action_semantics':'normalized_joint_target_velocity','integration':'semi_implicit_discrete',
              'max_acceleration_rad_s2':config.max_acceleration_rad_s2}
    if not isinstance(metadata,dict) or any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Checkpoint action/observation/controller lineage mismatch')
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256) or expected_sha256!=actual_sha256:
        raise ValueError('Checkpoint SHA256 mismatch')
    return True
