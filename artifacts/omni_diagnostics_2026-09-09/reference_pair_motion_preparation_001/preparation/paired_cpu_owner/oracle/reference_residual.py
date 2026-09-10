"""CPU/Torch finite position residual around an independently executable reference.

This is a new controller contract, not a conversion of old PPO checkpoints.
Target limits are engineering experiment parameters, not motor-speed claims.
"""
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import torch


LINEAGE = 'c_serial_omni_reference_position_residual_v1'
PROFILES = {'formal_004': .04, 'diagnostic_003': .03}


@dataclass(frozen=True)
class ResidualConfig:
    profile: str
    residual_radius_rad: float
    residual_velocity_rad_s: float
    residual_acceleration_rad_s2: float
    total_acceleration_rad_s2: float

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError('Declare formal_004 or diagnostic_003 separately')
        for key in ('residual_radius_rad', 'residual_velocity_rad_s',
                    'residual_acceleration_rad_s2', 'total_acceleration_rad_s2'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Explicit positive finite residual/controller limits required')
        if (self.residual_radius_rad > .1 or self.residual_velocity_rad_s >= self.total_velocity_rad_s
                or self.residual_acceleration_rad_s2 >= self.total_acceleration_rad_s2):
            raise ValueError('Residual needs a finite radius and positive remaining reference P/V/A budget')

    @property
    def total_velocity_rad_s(self):
        return PROFILES[self.profile] / .02

    @property
    def reference_velocity_rad_s(self):
        return self.total_velocity_rad_s - self.residual_velocity_rad_s

    @property
    def reference_acceleration_rad_s2(self):
        return self.total_acceleration_rad_s2 - self.residual_acceleration_rad_s2

    def contract(self):
        return {'lineage': LINEAGE, **self.__dict__,
                'total_position_step_rad_per_20ms': PROFILES[self.profile],
                'total_velocity_rad_s': self.total_velocity_rad_s,
                'reference_velocity_rad_s': self.reference_velocity_rad_s,
                'reference_acceleration_rad_s2': self.reference_acceleration_rad_s2,
                'action_semantics': 'bounded_position_residual_goal_tanh',
                'integration': 'separate_reference_knots_plus_bounded_residual_servo',
                'controller_observation_width': 72,
                'zero_command_feedback_disabled': False,
                'old_checkpoint_transfer_supported': False,
                'physics_admitted': False, 'motor_speed_limit_claim': False}


class ReferenceResidualTarget:
    """The residual is bounded in position, velocity and acceleration.

    A constant raw action requests a finite residual goal. It cannot wind up an
    unbounded velocity integrator. Zero raw action returns the residual toward
    zero; zero body command never suppresses the actor's balancing feedback.

    The reference receives a separate explicit rate/acceleration budget and a
    joint-limit margin equal to the full residual radius. Invalid references
    fail before state mutation. They are never silently clipped or slowed.
    """
    def __init__(self, joint_names: Sequence[str], lower_by_name: Mapping[str, float],
                 upper_by_name: Mapping[str, float], num_envs: int, config: ResidualConfig,
                 *, device='cpu', dtype=torch.float64):
        names = tuple(joint_names)
        if len(names) != 18 or len(set(names)) != 18 or set(names) != set(lower_by_name) or set(names) != set(upper_by_name):
            raise ValueError('Exactly18 unique runtime joint names and complete named limits required')
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('Positive environment count required')
        self.joint_names = names
        self.config = config
        self.lower = torch.tensor([lower_by_name[n] for n in names], device=device, dtype=dtype)
        self.upper = torch.tensor([upper_by_name[n] for n in names], device=device, dtype=dtype)
        if (not self.lower.is_floating_point() or not torch.isfinite(self.lower).all()
                or not torch.isfinite(self.upper).all()
                or not (self.upper-self.lower > 2*config.residual_radius_rad).all()):
            raise ValueError('Finite floating joint limits must contain the full residual margin')
        self.reference_position = torch.zeros((num_envs, 18), device=device, dtype=dtype)
        self.reference_velocity = torch.zeros_like(self.reference_position)
        self.residual_position = torch.zeros_like(self.reference_position)
        self.residual_velocity = torch.zeros_like(self.reference_position)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def _tensor(self, value, shape, name):
        value = torch.as_tensor(value, device=self.lower.device, dtype=self.lower.dtype)
        if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
            raise ValueError(name + ': finite exact-shape tensor required')
        return value

    def reset(self, measured_position, reference_position, env_ids=None):
        ids = torch.arange(len(self.initialized), device=self.lower.device) if env_ids is None else torch.as_tensor(env_ids, device=self.lower.device)
        if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64)
                or len(torch.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(self.initialized)).any()):
            raise ValueError('Unique in-range integer reset IDs required')
        shape = (len(ids), 18)
        measured = self._tensor(measured_position, shape, 'measured reset position')
        reference = self._tensor(reference_position, shape, 'reference reset position')
        self._reference_bounds(reference)
        if not torch.equal(measured, reference):
            raise ValueError('Reset reference must equal actual canonical reset joint targets; no pose jump')
        self.reference_position[ids] = reference
        self.reference_velocity[ids] = 0
        self.residual_position[ids] = 0
        self.residual_velocity[ids] = 0
        self.initialized[ids] = True

    def _reference_bounds(self, reference):
        radius = self.config.residual_radius_rad
        if ((reference < self.lower+radius) | (reference > self.upper-radius)).any():
            raise ValueError('Reference joint position lacks the declared full residual margin')

    def step(self, reference_position, raw_residual_action, *, reference_valid, dt=.02,
             analytic_reference_velocity=None, analytic_reference_acceleration=None):
        """Return executable P/V/A knots and all controller-known state.

        P/V/A bounds are checked on discrete executable knots. Optional analytic
        derivatives are retained separately, never substituted for measured
        control-step differences. The first runtime proof must use raw zeros.
        """
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not 0 < dt <= .02:
            raise ValueError('Explicit finite control dt in (0,.02] required')
        if not self.initialized.all():
            raise RuntimeError('Reset every environment before emitting targets')
        valid = torch.as_tensor(reference_valid, device=self.lower.device)
        if valid.dtype != torch.bool or valid.shape != self.initialized.shape or not valid.all():
            raise ValueError('Every reference must explicitly pass geometry/contact-transition validity')
        shape = self.reference_position.shape
        reference = self._tensor(reference_position, shape, 'reference position')
        raw = self._tensor(raw_residual_action, shape, 'residual action')
        self._reference_bounds(reference)
        cfg = self.config
        ref_v = (reference-self.reference_position)/dt
        ref_a = (ref_v-self.reference_velocity)/dt
        tolerance = 1e-5
        if (ref_v.abs() > cfg.reference_velocity_rad_s+tolerance).any() or (ref_a.abs() > cfg.reference_acceleration_rad_s2+tolerance).any():
            raise ValueError('Reference exceeds its executable rate/acceleration budget; no target emitted')
        analytic = {}
        for name, value in (('analytic_reference_velocity_rad_s', analytic_reference_velocity),
                            ('analytic_reference_acceleration_rad_s2', analytic_reference_acceleration)):
            if value is not None:
                analytic[name] = self._tensor(value, shape, name)
        goal = cfg.residual_radius_rad*torch.tanh(raw)
        r = self.residual_position
        v = self.residual_velocity
        a = cfg.residual_acceleration_rad_s2
        vmax = cfg.residual_velocity_rad_s

        def brake(distance):
            # Stable positive root of dt*u + u²/(2a) = distance.
            return (2*a*distance)/((a*a*dt*dt+2*a*distance).sqrt()+a*dt)

        error = goal-r
        desired = error.sign()*brake(error.abs()).clamp_max(vmax)
        lower_v = -brake((r+cfg.residual_radius_rad).clamp_min(0)).clamp_max(vmax)
        upper_v = brake((cfg.residual_radius_rad-r).clamp_min(0)).clamp_max(vmax)
        low = torch.maximum(v-a*dt, lower_v)
        high = torch.minimum(v+a*dt, upper_v)
        if (low > high+32*torch.finfo(r.dtype).eps).any():
            raise RuntimeError('Residual state lost braking viability; no target emitted')
        next_v = desired.maximum(low).minimum(torch.maximum(high, low))
        next_r = r+dt*next_v
        residual_a = (next_v-v)/dt
        q = reference+next_r
        total_v = ref_v+next_v
        total_a = ref_a+residual_a
        if ((next_r.abs() > cfg.residual_radius_rad+1e-6).any()
                or (q < self.lower-1e-6).any() or (q > self.upper+1e-6).any()
                or (total_v.abs() > cfg.total_velocity_rad_s+tolerance).any()
                or (total_a.abs() > cfg.total_acceleration_rad_s2+tolerance).any()):
            raise RuntimeError('Combined target violated declared P/V/A contract')
        # No tensor stored for later reset may depend on inference-only storage.
        with torch.inference_mode(False):
            clone = lambda value: value.detach().clone()
            result = {name:clone(value) for name,value in {
                'target_position_rad':q, 'target_velocity_rad_s':total_v,
                'target_acceleration_rad_s2':total_a,
                'reference_position_rad':reference, 'reference_velocity_rad_s':ref_v,
                'reference_acceleration_rad_s2':ref_a,
                'residual_position_rad':next_r, 'residual_velocity_rad_s':next_v,
                'residual_acceleration_rad_s2':residual_a,
                'residual_goal_rad':goal, 'raw_residual_action':raw, **analytic}.items()}
            self.reference_position = clone(reference)
            self.reference_velocity = clone(ref_v)
            self.residual_position = clone(next_r)
            self.residual_velocity = clone(next_v)
        return result

    def observable_state(self, default_position):
        """72 noiseless values make all persistent controller state executable.

        Ordered: total target−default, total target velocity/total limit,
        residual position/radius, residual velocity/residual limit. Reference
        P/V are recoverable by subtraction. Gait phase/contact/command state
        belongs to the separate reference generator and must also be observed.
        """
        default = self._tensor(default_position, self.reference_position.shape, 'default position')
        if not self.initialized.all():
            raise RuntimeError('Controller not initialized')
        return torch.cat((self.reference_position+self.residual_position-default,
                          (self.reference_velocity+self.residual_velocity)/self.config.total_velocity_rad_s,
                          self.residual_position/self.config.residual_radius_rad,
                          self.residual_velocity/self.config.residual_velocity_rad_s), dim=-1).clone()


def reject_old_checkpoint(metadata):
    """No policy training contract has been admitted for this prototype yet."""
    raise ValueError('Reference-residual physics prototype does not load PPO checkpoints; new actor contract required')
