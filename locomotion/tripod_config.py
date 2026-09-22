"""Declared geometry adaptation and finite Zhang et al. controller sweep."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class TripodConfig:
    period_s: float = 1.2
    hip_amplitude_rad: float = .12
    low_lift_rad: tuple = (.35, -.30)
    raised_lift_rad: tuple = (.45, -.35)
    raised_offset_rad: tuple = (-.15, .15)
    transition_s: float = 1.
    contact_on_n: float = 2.
    contact_off_n: float = 1.
    debounce_controls: int = 2
    recovery_s: float = .40
    recovery_rate_rad_s: float = .10
    initial_support_s: float = 2.
    geometry_sweep_gain: float = 0.
    geometry_lift_m: tuple = (.022, .028)
    damping_compensation: bool = False
    phase_ramp_fraction: float = 0.
    joint_feedback_gain: float = 0.
    swing_lift_power: float = 1.
    joint_velocity_feedback_gain: float = 0.
    joint_velocity_filter_hz: float = 0.
    support_overlap: bool = False
    startup_stride_ramp: bool = False
    forward_support_overlap: bool = False
    forward_low_lift_rad: tuple = (.25, 0.)
    forward_raised_lift_rad: tuple = (.28, -.10)

    def __post_init__(self):
        values = (self.period_s, self.hip_amplitude_rad, self.transition_s,
                  self.contact_on_n, self.contact_off_n, self.recovery_s,
                  self.recovery_rate_rad_s, self.initial_support_s,
                  self.geometry_sweep_gain, self.phase_ramp_fraction, self.joint_feedback_gain, self.swing_lift_power,
                  self.joint_velocity_feedback_gain,
                  self.joint_velocity_filter_hz,
                  *self.geometry_lift_m,
                  *self.forward_low_lift_rad, *self.forward_raised_lift_rad,
                  *self.low_lift_rad, *self.raised_lift_rad, *self.raised_offset_rad)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Controller parameters must be finite')
        if (self.period_s < 1.2 or not 0 < self.hip_amplitude_rad <= .30
                or self.transition_s < 1. or not 0 <= self.contact_off_n < self.contact_on_n
                or type(self.debounce_controls) is not int or self.debounce_controls < 1
                or not 0 < self.recovery_s <= .40 or not 0 < self.recovery_rate_rad_s <= .10
                or self.initial_support_s < self.recovery_s
                or not 0 <= self.geometry_sweep_gain <= 1.1
                or len(self.geometry_lift_m) != 2 or any(h <= 0 for h in self.geometry_lift_m)
                or type(self.damping_compensation) is not bool
                or (self.damping_compensation and not self.geometry_sweep_gain)
                or not 0 <= self.phase_ramp_fraction <= .20
                or (self.phase_ramp_fraction and not self.damping_compensation)
                or not 0 <= self.joint_feedback_gain <= 1.
                or (self.joint_feedback_gain and not self.phase_ramp_fraction)
                or self.swing_lift_power not in (.75, 1.)
                or (self.swing_lift_power != 1. and (not self.phase_ramp_fraction or self.joint_feedback_gain))
                or self.joint_velocity_feedback_gain not in (0., 2.)
                or (self.joint_velocity_feedback_gain and
                    (not self.phase_ramp_fraction or self.swing_lift_power != 1. or
                     (self.joint_feedback_gain and
                      (self.joint_feedback_gain != .5 or self.joint_velocity_filter_hz != 5.))))
                or self.joint_velocity_filter_hz not in (0., 5.)
                or (self.joint_velocity_filter_hz and not self.joint_velocity_feedback_gain)
                or type(self.support_overlap) is not bool
                or (self.support_overlap and (self.joint_feedback_gain != .5
                    or self.joint_velocity_feedback_gain != 2. or self.joint_velocity_filter_hz != 5.))
                or type(self.startup_stride_ramp) is not bool
                or (self.startup_stride_ramp and not self.support_overlap)
                or type(self.forward_support_overlap) is not bool
                or (self.forward_support_overlap and not self.startup_stride_ramp)
                or self.forward_low_lift_rad != (.25, 0.)
                or self.forward_raised_lift_rad != (.28, -.10)
                or any(len(v) != 2 for v in (self.low_lift_rad, self.raised_lift_rad, self.raised_offset_rad))):
            raise ValueError('Controller parameters exceed the declared envelope')

    def declaration(self):
        return {'schema': 'zhang_tripod_geometry_adaptation_v1', **asdict(self),
                'trajectory_variant': ('forward_overlap' if self.forward_support_overlap else
                                       'startup' if self.startup_stride_ramp else
                                       'overlap' if self.support_overlap else
                                       'pd_filtered' if self.joint_feedback_gain and self.joint_velocity_feedback_gain else
                                       'velocity_filtered' if self.joint_velocity_filter_hz else
                                       'velocity' if self.joint_velocity_feedback_gain else
                                       'liftoff' if self.swing_lift_power != 1. else
                                       'feedback' if self.joint_feedback_gain else
                                       'retimed' if self.phase_ramp_fraction else
                                       'damping' if self.damping_compensation else
                                       'geometry' if self.geometry_sweep_gain else 'paper'),
                'control_dt_s': .02, 'target_slew_rad': .040,
                'low_root_target_m': .0978,
                'raised_root_target_m': .1093 if self.geometry_sweep_gain else .1088,
                'minimum_root_increase_m': .008,
                'minimum_toe_lift_m': {'low': .012, 'raised': .016},
                'paper_doi': '10.3389/frobt.2024.1426269', 'stage2_complete': False}


SWEEP = tuple(replace(TripodConfig(), period_s=period, hip_amplitude_rad=amplitude)
              for period in (1.2, 1.6) for amplitude in (.12, .18))

GEOMETRY_SWEEP = tuple(replace(TripodConfig(), geometry_sweep_gain=gain,
                              raised_offset_rad=(-.15, .10)) for gain in (1., 1.1))
SWEEPS = {'paper': SWEEP, 'geometry': GEOMETRY_SWEEP,
          'damping': (replace(GEOMETRY_SWEEP[0], damping_compensation=True),),
          'retimed': (replace(GEOMETRY_SWEEP[0], damping_compensation=True, phase_ramp_fraction=.10),)}
SWEEPS['feedback'] = tuple(replace(SWEEPS['retimed'][0], joint_feedback_gain=gain) for gain in (.5, 1.))
SWEEPS['liftoff'] = (replace(SWEEPS['retimed'][0], swing_lift_power=.75),)
SWEEPS['velocity'] = (replace(SWEEPS['retimed'][0], joint_velocity_feedback_gain=2.),)
SWEEPS['velocity_filtered'] = (replace(SWEEPS['velocity'][0], joint_velocity_filter_hz=5.),)
SWEEPS['pd_filtered'] = (replace(SWEEPS['velocity_filtered'][0], joint_feedback_gain=.5),)
SWEEPS['overlap'] = (replace(SWEEPS['pd_filtered'][0], support_overlap=True),)
SWEEPS['startup'] = (replace(SWEEPS['overlap'][0], startup_stride_ramp=True),)
SWEEPS['forward_overlap'] = (replace(SWEEPS['startup'][0], forward_support_overlap=True),)
