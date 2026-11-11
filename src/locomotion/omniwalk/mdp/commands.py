"""Declare the omnidirectional command bank, its hold times and the gait-clock period."""

import math
from dataclasses import dataclass

# One gait cycle in 50 Hz controls; the contact-schedule reward and the policy clock inputs share it.
GAIT_CLOCK_CONTROLS = 60
# Speed-scaled gait clocks by name: walking speeds (m/s) and, at each, the cycle in controls and the share
# of it each tripod swings, linear between the speeds and held beyond them. speed_v1 keeps strides near
# 55 mm, inside the +-0.35 rad action range and the 0.04 rad target slew; up to 0.075 m/s it is the fixed clock.
GAIT_CLOCK_SCALINGS = {
    "speed_v1": ((0.075, 0.1, 0.125, 0.15), (60.0, 52.0, 44.0, 40.0), (0.4, 0.48, 0.5, 0.5)),
}


def default_gait_clock(mode, networks):
    """The clock period a run uses when the command line names none: MLP training needs it, other modes run without it."""
    return GAIT_CLOCK_CONTROLS if mode == "train" and networks == "mlp" else 0


@dataclass(frozen=True)
class TaskConfig:
    seed: int = 20260914
    hold_min_seconds: float = 2.0
    hold_max_seconds: float = 5.0
    zero_command_fraction: float = 0.15
    speed_levels_mps: tuple = (0.025, 0.05)
    maximum_speed_mps: float = 0.05
    yaw_rate_rad_s: float = 0.2
    arc_speed_mps: float = 0.04
    arc_yaw_rate_rad_s: float = 0.15

    def validate(self):
        if (self.hold_min_seconds, self.hold_max_seconds, self.zero_command_fraction) != (
            2.0,
            5.0,
            0.15,
        ):
            raise ValueError("Command holds and zero quota differ from the recorded task")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if (
            self.speed_levels_mps != (0.025, 0.05)
            or self.maximum_speed_mps not in self.speed_levels_mps
            or self.yaw_rate_rad_s != 0.2
            or self.arc_speed_mps != 0.04
            or self.arc_yaw_rate_rad_s != 0.15
        ):
            raise ValueError("Command bank differs from the recorded task")


def command_bank(config):
    """Nonzero navigation-frame commands; zero has a separate reserved quota."""
    config.validate()
    result = []
    for speed in config.speed_levels_mps:
        if speed <= config.maximum_speed_mps:
            for bearing in range(8):
                angle = bearing * math.pi / 4
                result.append([speed * math.cos(angle), speed * math.sin(angle), 0.0])
    result.extend([[0.0, 0.0, -config.yaw_rate_rad_s], [0.0, 0.0, config.yaw_rate_rad_s]])
    if config.arc_speed_mps <= config.maximum_speed_mps:
        result.extend(
            [
                [config.arc_speed_mps, 0.0, -config.arc_yaw_rate_rad_s],
                [config.arc_speed_mps, 0.0, config.arc_yaw_rate_rad_s],
            ]
        )
    return result
