"""Declare the omnidirectional command bank, its hold times and the gait-clock period."""
from dataclasses import dataclass
import math

# One gait cycle in 50 Hz controls; the contact-schedule reward and the policy clock inputs share it.
GAIT_CLOCK_CONTROLS = 60


def default_gait_clock(mode, networks):
    """The clock period a run uses when the command line names none: MLP training needs it, other modes run without it."""
    return GAIT_CLOCK_CONTROLS if mode == 'train' and networks == 'mlp' else 0


@dataclass(frozen=True)
class TaskConfig:
    seed: int = 20260914
    hold_min_seconds: float = 2.
    hold_max_seconds: float = 5.
    zero_command_fraction: float = .15
    speed_levels_mps: tuple = (.025, .05)
    maximum_speed_mps: float = .05
    yaw_rate_rad_s: float = .2
    arc_speed_mps: float = .04
    arc_yaw_rate_rad_s: float = .15

    def validate(self):
        if (self.hold_min_seconds, self.hold_max_seconds, self.zero_command_fraction) != (2., 5., .15):
            raise ValueError("Command holds and zero quota differ from the recorded task")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if (self.speed_levels_mps != (.025, .05) or self.maximum_speed_mps not in self.speed_levels_mps
                or self.yaw_rate_rad_s != .2 or self.arc_speed_mps != .04 or self.arc_yaw_rate_rad_s != .15):
            raise ValueError("Command bank differs from the recorded task")


def command_bank(config):
    """Nonzero navigation-frame commands; zero has a separate reserved quota."""
    config.validate()
    result = []
    for speed in config.speed_levels_mps:
        if speed <= config.maximum_speed_mps:
            for bearing in range(8):
                angle = bearing * math.pi / 4
                result.append([speed * math.cos(angle), speed * math.sin(angle), 0.])
    result.extend([[0., 0., -config.yaw_rate_rad_s], [0., 0., config.yaw_rate_rad_s]])
    if config.arc_speed_mps <= config.maximum_speed_mps:
        result.extend([[config.arc_speed_mps, 0., -config.arc_yaw_rate_rad_s],
                       [config.arc_speed_mps, 0., config.arc_yaw_rate_rad_s]])
    return result
