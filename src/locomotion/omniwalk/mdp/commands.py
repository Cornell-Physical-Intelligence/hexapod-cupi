"""Declare the omnidirectional command bank, its hold times and the gait-clock period."""

from dataclasses import dataclass

from contracts.command import CommandSet, load_requirement_set

_DEFAULT_RULES, _ = load_requirement_set()
COMMAND_SET = CommandSet.from_mapping(_DEFAULT_RULES["commands"])

# One gait cycle in 50 Hz controls; the contact-schedule reward and the policy clock inputs share it.
GAIT_CLOCK_CONTROLS = 60


def default_gait_clock(mode, networks):
    """The clock period a run uses when the command line names none: MLP training needs it, other modes run without it."""
    return GAIT_CLOCK_CONTROLS if mode == "train" and networks == "mlp" else 0


@dataclass(frozen=True)
class TaskConfig:
    seed: int = 20260914
    hold_min_seconds: float = _DEFAULT_RULES["hold_min_s"]
    hold_max_seconds: float = 5.0
    zero_command_fraction: float = 0.15
    speed_levels_mps: tuple = COMMAND_SET.translate_speeds_mps
    maximum_speed_mps: float = max(COMMAND_SET.translate_speeds_mps)
    yaw_rate_rad_s: float = COMMAND_SET.turn_rate_rad_s
    arc_speed_mps: float = COMMAND_SET.arc_speed_mps
    arc_yaw_rate_rad_s: float = COMMAND_SET.arc_turn_rate_rad_s

    def validate(self):
        if (self.hold_min_seconds, self.hold_max_seconds, self.zero_command_fraction) != (
            _DEFAULT_RULES["hold_min_s"],
            5.0,
            0.15,
        ):
            raise ValueError("Command holds and zero quota differ from the recorded task")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if (
            self.speed_levels_mps != COMMAND_SET.translate_speeds_mps
            or self.maximum_speed_mps not in self.speed_levels_mps
            or self.yaw_rate_rad_s != COMMAND_SET.turn_rate_rad_s
            or self.arc_speed_mps != COMMAND_SET.arc_speed_mps
            or self.arc_yaw_rate_rad_s != COMMAND_SET.arc_turn_rate_rad_s
            or not COMMAND_SET.stand
        ):
            raise ValueError("Command bank differs from the recorded task")


def command_bank(config):
    """Nonzero navigation-frame commands; zero has a separate reserved quota."""
    config.validate()
    return [list(command.as_tuple()) for command in COMMAND_SET.nonzero(config.maximum_speed_mps)]
