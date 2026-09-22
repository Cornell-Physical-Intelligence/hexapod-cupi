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

    def __post_init__(self):
        values = (self.period_s, self.hip_amplitude_rad, self.transition_s,
                  self.contact_on_n, self.contact_off_n, self.recovery_s,
                  self.recovery_rate_rad_s, self.initial_support_s,
                  *self.low_lift_rad, *self.raised_lift_rad, *self.raised_offset_rad)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Controller parameters must be finite')
        if (self.period_s < 1.2 or not 0 < self.hip_amplitude_rad <= .30
                or self.transition_s < 1. or not 0 <= self.contact_off_n < self.contact_on_n
                or type(self.debounce_controls) is not int or self.debounce_controls < 1
                or not 0 < self.recovery_s <= .40 or not 0 < self.recovery_rate_rad_s <= .10
                or self.initial_support_s < self.recovery_s
                or any(len(v) != 2 for v in (self.low_lift_rad, self.raised_lift_rad, self.raised_offset_rad))):
            raise ValueError('Controller parameters exceed the declared envelope')

    def declaration(self):
        return {'schema': 'zhang_tripod_geometry_adaptation_v1', **asdict(self),
                'control_dt_s': .02, 'target_slew_rad': .040,
                'low_root_target_m': .0978, 'raised_root_target_m': .1088,
                'minimum_root_increase_m': .008,
                'minimum_toe_lift_m': {'low': .012, 'raised': .016},
                'paper_doi': '10.3389/frobt.2024.1426269', 'stage2_complete': False}


SWEEP = tuple(replace(TripodConfig(), period_s=period, hip_amplitude_rad=amplitude)
              for period in (1.2, 1.6) for amplitude in (.12, .18))
