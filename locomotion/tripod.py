"""Prescribed tripod targets with measured touchdown, without physics writes.

Zhang et al. (2024), section 4.2, equations (1)-(3) define the waveforms.
docs/TRAINING.md declares the geometry, contact and transition adaptations.
"""
import math
import numpy as np

from .env_config import JOINT_NAMES, KD, LEGS
from .tripod_config import TripodConfig

TRIPOD_A = np.array([True, False, True, False, True, False])
PHASE = np.where(TRIPOD_A, 0., math.pi)
FORWARD_SIGN = np.array([-1., -1., -1., 1., 1., 1.])
DT = .02


def wave(phase, lift_power=1.):
    """Evaluate the paper waveform or the declared cosine-lift adaptation."""
    if lift_power not in (.75, 1.):
        raise ValueError('Lift power must be .75 or 1')
    phase = np.asarray(phase, float)
    lift = np.where(np.cos(phase) > 0., .5*(1.+np.cos(2.*phase)), 0.)
    return np.sin(phase), lift if lift_power == 1. else lift**lift_power


def supported(command):
    value = np.asarray(command, float)
    if value.shape != (3,) or not np.isfinite(value).all():
        return False
    forward, left, yaw = value
    return bool(left == 0 and ((yaw == 0 and 0 <= forward <= .10)
                              or (forward == 0 and abs(yaw) <= .20)))


def joint_order(names):
    """Map a complete external joint list into the approved canonical order."""
    names = tuple(names)
    if len(names) != 18 or set(names) != set(JOINT_NAMES):
        raise ValueError('Expected the 18 distinct approved joint names')
    return np.array([names.index(name) for name in JOINT_NAMES])


def damping_target(reference, previous_reference):
    """Cancel the approved servo's damping term at nominal joint velocity."""
    reference = np.asarray(reference, float).reshape(18)
    previous_reference = np.asarray(previous_reference, float).reshape(18)
    return reference+np.asarray(KD)*(reference-previous_reference)/(12.*DT)


def displacement_clock(progress, ramp):
    """Integrate cosine endpoint ramps and a constant-speed middle interval."""
    if not 0 < ramp <= .20 or not 0 <= progress <= 1.:
        raise ValueError('Phase clock requires bounded progress and endpoint ramps')
    v = 1./(1.-ramp)
    if progress < ramp:
        return .5*v*(progress-ramp/math.pi*math.sin(math.pi*progress/ramp))
    if progress > 1.-ramp:
        return 1.-displacement_clock(1.-progress, ramp)
    return v*(progress-ramp/2.)


def feedback_blend(state, elapsed, duration):
    if state == 'walk':
        return 1.
    if state in ('start', 'settle'):
        blend = .5*(1.-math.cos(math.pi*min(1., elapsed/duration)))
        return blend if state == 'start' else 1.-blend
    return 0.


def overlap_sweep(progress, forward=False):
    """Keep swing feet in support motion outside the declared return window."""
    if not 0 <= progress <= 1.:
        raise ValueError('Support overlap requires half-cycle progress in [0, 1]')
    u = np.clip((progress-.15)/.65, 0., 1.)
    returned = displacement_clock(float(u), .10) if forward else u*u*(3.-2.*u)
    return -1.-2.*progress+4.*returned, 1.-2.*progress


def feedback_target(reference, servo_target, measured, neutral, lower, upper, gain, blend):
    """Bound encoder-error feedback through the existing motor-target envelope."""
    measured = np.asarray(measured, float)
    if measured.shape != (18,) or not np.isfinite(measured).all():
        raise ValueError('Joint feedback requires 18 finite measured positions')
    offset = np.clip(blend*gain*(reference-measured), -.070, .070)
    requested = servo_target+offset
    bounded = np.clip(requested, np.maximum(lower, neutral-.35), np.minimum(upper, neutral+.35))
    return bounded, offset, bool(np.any(bounded != requested))


def velocity_feedback_target(servo_target, reference_velocity, measured_velocity,
                             neutral, lower, upper, gain, blend):
    """Damp joint-velocity error through the existing motor-target envelope."""
    reference_velocity, measured_velocity = [np.asarray(v, float)
                                            for v in (reference_velocity, measured_velocity)]
    if any(v.shape != (18,) or not np.isfinite(v).all()
           for v in (reference_velocity, measured_velocity)):
        raise ValueError('Velocity feedback requires 18 finite joint velocities')
    offset = np.clip(blend*gain*np.asarray(KD)*(reference_velocity-measured_velocity)/12., -.070, .070)
    requested = servo_target+offset
    bounded = np.clip(requested, np.maximum(lower, neutral-.35), np.minimum(upper, neutral+.35))
    return bounded, offset, bool(np.any(bounded != requested))


class TripodController:
    def __init__(self, neutral, lower, upper, config=None, *, model=None, toe_local_points=None):
        self.cfg = config or TripodConfig()
        self.neutral, self.lower, self.upper = [np.asarray(v, float).reshape(6, 3).copy()
                                               for v in (neutral, lower, upper)]
        if not all(np.isfinite(v).all() for v in (self.neutral, self.lower, self.upper)):
            raise ValueError('Joint geometry must be finite')
        if not np.allclose(self.neutral, [0., -.30, .40], rtol=0, atol=1e-7):
            raise ValueError('This adaptation requires the approved walking stance')
        if np.any(self.lower >= self.upper):
            raise ValueError('Invalid joint bounds')
        self.geometry = {}
        if self.cfg.geometry_sweep_gain:
            from .tripod_kinematics import stance_geometry
            if model is None or toe_local_points is None:
                raise ValueError('Geometry variant requires the admitted model and toe points')
            for mode, height in zip(('low', 'raised'), self.cfg.geometry_lift_m):
                points, jacobians = stance_geometry(model, toe_local_points, self.stance(mode))
                lift = np.array([np.linalg.solve(j, [0., 0., height]) for j in jacobians])
                self.geometry[mode] = (points, jacobians, lift[:, 1:])
        # Check combinations, including the maximum permitted recovery extension.
        for mode in ('low', 'raised'):
            base = self.stance(mode)
            for lift in (np.zeros(2), self.lift(mode), [-.04, 0.]):
                for hip in (-.30, .30):
                    pose = base.copy(); pose[:, 0] += hip; pose[:, 1:] += lift
                    self._validate_target(pose)
        self.reset()

    def stance(self, mode):
        if mode not in ('low', 'raised'):
            raise ValueError('Unknown clearance mode')
        value = self.neutral.copy()
        if mode == 'raised':
            value[:, 1:] += self.cfg.raised_offset_rad
        return value

    def lift(self, mode, *, forward=False):
        if self.cfg.forward_support_overlap and forward:
            return np.array(self.cfg.forward_low_lift_rad if mode == 'low'
                            else self.cfg.forward_raised_lift_rad)
        if self.geometry:
            return self.geometry[mode][2].copy()
        return np.array(self.cfg.low_lift_rad if mode == 'low' else self.cfg.raised_lift_rad)

    def reset(self):
        self.mode = 'low'
        self.state = 'idle'
        self.fault = None
        self.target = self.neutral.copy()
        self.command = np.zeros(3)
        self.contacts = np.zeros(6, dtype=bool)
        self.on_count = np.zeros(6, dtype=int)
        self.off_count = np.zeros(6, dtype=int)
        self.half = 0
        self.progress = 0.
        self.elapsed = 0.
        self.stride_elapsed = 0.
        self.ramping_stride = False
        self.wait_s = 0.
        self.support_loss_s = 0.
        self.seen_off = np.zeros(6, dtype=bool)
        self.landed = np.zeros(6, dtype=bool)
        self.pitch_hold = np.zeros((6, 2))
        self.swing_start = np.zeros((6, 2))
        self.from_target = self.target.copy()
        self.to_target = self.target.copy()
        self.transition_mode = 'low'
        self.stopping = False
        self.support_from_touchdown = np.zeros(6, dtype=bool)

    def _validate_target(self, target):
        if (not np.isfinite(target).all() or np.any(target < self.lower)
                or np.any(target > self.upper) or np.max(abs(target-self.neutral)) > .35000001):
            raise ValueError('Controller target exceeds the unchanged joint/action envelope')

    def _emit(self, desired):
        self._validate_target(desired)
        self.target += np.clip(desired-self.target, -.040, .040)
        return self.target.reshape(18).copy()

    def _fault(self, reason):
        self.fault = reason
        self.state = 'fault'
        return self.target.reshape(18).copy()

    def _contacts(self, force):
        force = np.asarray(force, float)
        if force.shape != (6,) or not np.isfinite(force).all() or np.any(force < 0):
            raise ValueError('Contact feedback requires six finite nonnegative measured forces')
        self.on_count = np.where(force >= self.cfg.contact_on_n, self.on_count+1, 0)
        self.off_count = np.where(force <= self.cfg.contact_off_n, self.off_count+1, 0)
        self.contacts[self.on_count >= self.cfg.debounce_controls] = True
        self.contacts[self.off_count >= self.cfg.debounce_controls] = False

    @property
    def swing(self):
        return TRIPOD_A.copy() if self.half % 2 == 0 else ~TRIPOD_A

    @property
    def stopped(self):
        return self.state == 'idle' and bool(self.contacts.all())

    def _begin_blend(self, state, target):
        self.state, self.elapsed = state, 0.
        self.from_target, self.to_target = self.target.copy(), target.copy()

    def _hip(self):
        if self.command[0] > 0:
            return FORWARD_SIGN*min(.30, self.cfg.hip_amplitude_rad*self.command[0]/.05)
        return np.full(6, self.cfg.hip_amplitude_rad*self.command[2]/.20)

    def _sweep(self):
        if not self.geometry:
            result = np.zeros((6, 3)); result[:, 0] = self._hip()
            return result
        points, jacobians, _ = self.geometry[self.mode]
        velocity = np.tile([0., -self.command[0], 0.], (6, 1))
        velocity[:, 0] -= self.command[2]*points[:, 1]
        velocity[:, 1] += self.command[2]*points[:, 0]
        displacement = velocity*self.cfg.period_s*self.cfg.geometry_sweep_gain/4.
        return np.array([np.linalg.solve(j, d) for j, d in zip(jacobians, displacement)])

    def step(self, command, foot_force_n, mode='low'):
        """Advance one 20 ms control using measured distal-force magnitudes."""
        if not supported(command):
            raise ValueError('Unsupported command: forward, pure yaw or zero required')
        if mode not in ('low', 'raised'):
            raise ValueError('Unknown clearance mode')
        self._contacts(foot_force_n)
        if self.fault:
            return self.target.reshape(18).copy()
        requested = np.asarray(command, float)
        if self.state == 'idle':
            if not self.contacts.all():
                self.wait_s += DT
                if self.wait_s >= self.cfg.initial_support_s-1e-9:
                    return self._fault('initial_support_timeout')
                return self.target.reshape(18).copy()
            self.wait_s = 0.
            if mode != self.mode:
                self.transition_mode = mode
                self._begin_blend('clearance', self.stance(mode))
            elif requested.any():
                self.command = requested.copy()
                self.half, self.progress = 0, 0.
                self.stride_elapsed = 0.
                self.ramping_stride = bool(self.cfg.startup_stride_ramp and requested[0] > 0.)
                self.pitch_hold.fill(0); self.swing_start.fill(0)
                self.support_from_touchdown.fill(False)
                endpoint = self.stance(self.mode)
                if not self.ramping_stride:
                    endpoint += self._sweep()*np.sin(-math.pi/2+PHASE)[:, None]
                self._begin_blend('start', endpoint)
            else:
                return self.target.reshape(18).copy()

        if self.state in ('start', 'settle', 'clearance'):
            if self.ramping_stride and self.state == 'start':
                if mode != self.mode or requested[0] <= 0.:
                    self._begin_blend('settle', self.stance(self.mode))
                else:
                    self.command = requested.copy()
            self.elapsed = min(self.elapsed+DT, self.cfg.transition_s)
            h = .5*(1.-math.cos(math.pi*self.elapsed/self.cfg.transition_s))
            result = self._emit(self.from_target+h*(self.to_target-self.from_target))
            self.support_loss_s = 0. if self.contacts.all() else self.support_loss_s+DT
            if self.support_loss_s > self.cfg.recovery_s+1e-9:
                return self._fault('support_loss_during_'+self.state)
            if self.elapsed >= self.cfg.transition_s-1e-9:
                if self.state == 'clearance':
                    self.mode = self.transition_mode
                    self.state = 'idle'
                elif self.state == 'settle':
                    self.state = 'idle'; self.command.fill(0)
                    self.ramping_stride = False
                else:
                    self.state = 'walk'
                    self.seen_off.fill(False); self.landed.fill(False)
                    self.stopping = False
            return result

        self.stopping |= mode != self.mode or not np.array_equal(requested, self.command)
        swing = self.swing
        self.seen_off |= swing & ~self.contacts
        self.support_loss_s = 0. if self.contacts[~swing].all() else self.support_loss_s+DT
        if self.support_loss_s > self.cfg.recovery_s+1e-9:
            return self._fault('stance_support_loss')
        base = self.stance(self.mode)
        coefficients = self._sweep()
        previous_coefficients = coefficients
        if self.ramping_stride:
            old_scale = .5*(1.-math.cos(math.pi*self.stride_elapsed/self.cfg.period_s))
            self.stride_elapsed = min(self.cfg.period_s, self.stride_elapsed+DT)
            scale = .5*(1.-math.cos(math.pi*self.stride_elapsed/self.cfg.period_s))
            previous_coefficients = coefficients*old_scale
            coefficients = coefficients*scale
        touchdown = swing & self.seen_off & self.contacts & ~self.landed & (self.progress >= .5)
        touchdown_base = base
        forward_overlap = self.cfg.forward_support_overlap and self.command[0] > 0.
        overlap = self.cfg.support_overlap and (self.command[2] != 0. or forward_overlap)
        if overlap:
            swing_shape, support_shape = overlap_sweep(self.progress, forward_overlap)
            shape = np.where(swing, swing_shape, support_shape)
            touchdown_base = base+previous_coefficients*shape[:, None]
        self.pitch_hold[touchdown] = (self.target-touchdown_base)[touchdown, 1:]
        self.landed |= touchdown
        previous_progress = self.progress
        self.progress = min(1., self.progress+2.*DT/self.cfg.period_s)
        if self.cfg.startup_stride_ramp and overlap:
            remaining = (0. if self.progress >= 1.-1e-9 else
                         (1.-self.progress)/max(1e-12, 1.-previous_progress))
            self.pitch_hold[self.landed] *= remaining
        phase_advance = self.progress*math.pi
        stance_return = .5*(1.+math.cos(math.pi*self.progress))
        if self.cfg.phase_ramp_fraction:
            displacement = displacement_clock(self.progress, self.cfg.phase_ramp_fraction)
            phase_advance = math.acos(np.clip(1.-2.*displacement, -1., 1.))
            stance_return = 1.-displacement
        phase = -math.pi/2+self.half*math.pi+phase_advance+PHASE
        hip, lift = wave(phase, self.cfg.swing_lift_power)
        if overlap:
            swing_shape, support_shape = overlap_sweep(self.progress, forward_overlap)
            hip = np.where(swing, swing_shape, support_shape)
            stance_return = 1.-self.progress
        baseline = base+coefficients*hip[:, None]
        desired = baseline.copy()
        hold_base = baseline if overlap else base
        desired[self.landed, 1:] = hold_base[self.landed, 1:]+self.pitch_hold[self.landed]
        # Return a landed leg to the paper's zero-lift stance over its support half-cycle.
        support = ~swing & self.support_from_touchdown
        start_sine = np.sin(-math.pi/2+self.half*math.pi+PHASE)
        residual = (self.pitch_hold if overlap else
                    self.pitch_hold-coefficients[:, 1:]*start_sine[:, None])
        desired[support, 1:] += stance_return*residual[support]
        moving = swing & ~self.landed
        pitches = lift[:, None]*self.lift(self.mode, forward=forward_overlap)
        if self.progress < .5:
            pitches += (1.-lift[:, None])*self.swing_start
        desired[moving, 1:] = baseline[moving, 1:]+pitches[moving]

        if self.progress >= 1.-1e-9:
            if not self.seen_off[swing].all():
                return self._fault('swing_failed_to_lift')
            if self.contacts[swing].all() and self.landed[swing].all():
                result = self._emit(desired)
                self.wait_s = 0.
                if self.stopping:
                    self._begin_blend('settle', base)
                else:
                    self.pitch_hold[~swing] = 0.
                    self.support_from_touchdown = swing.copy()
                    self.half += 1; self.progress = 0.
                    self.swing_start = self.pitch_hold.copy()
                    self.seen_off.fill(False); self.landed.fill(False)
                return result
            self.wait_s += DT
            missing = swing & ~self.contacts
            desired[missing, 1] = baseline[missing, 1]-min(.04, self.cfg.recovery_rate_rad_s*self.wait_s)
            if self.wait_s > self.cfg.recovery_s+1e-9:
                return self._fault('touchdown_timeout')
        return self._emit(desired)

    def snapshot(self):
        return {'state': self.state, 'mode': self.mode, 'fault': self.fault,
                'half': self.half, 'progress': self.progress, 'stopped': self.stopped,
                'ramping_stride': self.ramping_stride, 'stride_elapsed_s': self.stride_elapsed,
                'contacts': self.contacts.tolist(), 'landed': self.landed.tolist(),
                'active_command': self.command.tolist(), 'target_rad': self.target.reshape(18).tolist()}
