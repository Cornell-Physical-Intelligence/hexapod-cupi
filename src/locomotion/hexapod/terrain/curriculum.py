"""Torch-only pieces of the terrain task: episode outcome, swing clearance and the command-speed curriculum.

Kept free of Isaac Lab imports so the CPU tests can load this file directly.
"""

from __future__ import annotations

import torch


def episode_outcome(
    v_along_cmd_sum: torch.Tensor,
    steps: torch.Tensor,
    cmd_speed: torch.Tensor,
    fell: torch.Tensor,
    timed_out: torch.Tensor,
    lin_vel_max: float,
    speed_tol_ratio: float = 0.25,
    min_cmd_ratio: float = 0.1,
    move_down_speed_ratio: float = 0.5,
):
    """Outcome of the episodes that just ended.

    Success: the episode reached its time-out without a fall and the episode-mean speed along the command direction
    is within ``speed_tol_ratio`` of the commanded speed. For commands below ``min_cmd_ratio * lin_vel_max`` the
    tolerance is computed against that floor, so a near-zero command does not make the test degenerate. ``slow``
    marks episodes below ``move_down_speed_ratio`` of the command (a terrain-curriculum demotion). Envs with no steps
    count as not run (never a success).
    """
    mean_v = v_along_cmd_sum / steps.clamp_min(1.0)
    speed_err = torch.abs(mean_v - cmd_speed)
    tol_base = torch.clamp_min(cmd_speed, min_cmd_ratio * lin_vel_max)
    success = timed_out & ~fell & (speed_err < speed_tol_ratio * tol_base) & (steps > 0)
    slow = mean_v < move_down_speed_ratio * cmd_speed
    return success, slow, speed_err, mean_v


def terrain_moves(success: torch.Tensor, fell: torch.Tensor, slow: torch.Tensor, ran: torch.Tensor):
    """Terrain curriculum: up a level on a success, down on a fall or a slow episode."""
    move_up = success & ran
    move_down = (fell | slow) & ran & ~move_up
    return move_up, move_down


def swing_clearance(toe_z: torch.Tensor, in_contact: torch.Tensor, target: float) -> torch.Tensor:
    """Per-env sum over swing feet of min(toe height above the mean stance-toe height, target) / target.

    ``toe_z`` and ``in_contact`` are (num_envs, num_feet). With no foot in contact the lowest toe is the reference.
    """
    n_contact = in_contact.float().sum(dim=1).clamp_min(1.0)
    ref = torch.where(in_contact.any(dim=1), (toe_z * in_contact).sum(dim=1) / n_contact, toe_z.min(dim=1).values)
    lift = (toe_z - ref[:, None]).clamp(0.0, target) / target
    return (lift * ~in_contact).sum(dim=1)


class CommandCurriculum:
    """Raise the largest commanded speed once the robot tracks the current top speed.

    Starts at ``start``. Over windows of ``window_episodes`` finished episodes whose command was in the top bucket
    [max - step, max], when the success rate is >= ``success``, the max goes up by ``step`` (capped at ``cap``).
    """

    def __init__(self, start: float, step: float, cap: float, success: float, window_episodes: int):
        self.value = float(start)
        self.step, self.cap, self.success, self.window_episodes = float(step), float(cap), float(success), int(window_episodes)
        self._n = 0
        self._wins = 0

    def update(self, cmd_speed: torch.Tensor, success: torch.Tensor, ran: torch.Tensor) -> float:
        top = ran & (cmd_speed >= self.value - self.step - 1e-6)
        self._n += int(top.sum().item())
        self._wins += int((top & success).sum().item())
        if self._n >= self.window_episodes:
            if self._wins / self._n >= self.success and self.value < self.cap - 1e-6:
                self.value = min(self.value + self.step, self.cap)
            self._n = 0
            self._wins = 0
        return self.value
