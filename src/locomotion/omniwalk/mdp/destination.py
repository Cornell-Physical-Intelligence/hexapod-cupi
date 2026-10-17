"""Follow each command's destination, the pose the command would have reached, and describe the gap to it.

The destination starts at the body's planar pose when a command begins and moves control by control
along the command: translation in the turning body axes and the yaw rate. The task pays for staying
near it (``destination_weight``) and the wrapper's ``destination_input`` gives the actor and critic
five values that describe the gap: the planar gap in the body's navigation axes over 0.1 m, the sine
and cosine of the heading gap, and the controls since the command began over 250.
"""

import torch

CONTROL_DT_S = 0.02
DESTINATION_FEATURES = 5
DISTANCE_SCALE_M = 0.1
LONGEST_COMMAND_CONTROLS = 250
NEUTRAL_FEATURES = (0.0, 0.0, 0.0, 1.0, 0.0)


def planar_pose(pose):
    """World x, y and the body's rotation about world z of root pose rows (position, then XYZW)."""
    x, y, z, w = pose[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((pose[:, 0], pose[:, 1], yaw), -1)


def advance_destination(destination, commands, dt=CONTROL_DT_S):
    """Move each planar destination one control along its command; navigation (x, y) is body (-y, x)."""
    body = torch.stack((commands[:, 1], -commands[:, 0]), -1).to(destination)
    middle = destination[:, 2] + commands[:, 2].to(destination) * dt / 2
    c, s = torch.cos(middle), torch.sin(middle)
    step = torch.stack((c * body[:, 0] - s * body[:, 1], s * body[:, 0] + c * body[:, 1]), -1) * dt
    turn = destination[:, 2:] + commands[:, 2:].to(destination) * dt
    return torch.cat((destination[:, :2] + step, turn), -1)


def destination_distance(pose, destination):
    """Planar distance from each root pose to its destination."""
    actual = planar_pose(pose).to(destination)
    return torch.linalg.vector_norm(actual[:, :2] - destination[:, :2], dim=-1)


def destination_features(pose, destination, elapsed):
    """The five inputs for root poses, destinations ``(x, y, heading)`` and controls since the command began."""
    actual = planar_pose(pose).to(destination)
    gap = destination[:, :2] - actual[:, :2]
    c, s = torch.cos(actual[:, 2]), torch.sin(actual[:, 2])
    body_x, body_y = c * gap[:, 0] + s * gap[:, 1], -s * gap[:, 0] + c * gap[:, 1]
    turn = destination[:, 2] - actual[:, 2]
    return torch.stack(
        (
            -body_y / DISTANCE_SCALE_M,
            body_x / DISTANCE_SCALE_M,
            torch.sin(turn),
            torch.cos(turn),
            elapsed.to(destination) / LONGEST_COMMAND_CONTROLS,
        ),
        -1,
    )


class DestinationTracker:
    """Each replica's destination from the poses and commands seen before each control, for evaluation."""

    def __init__(self, num_envs, device):
        self.destination = torch.zeros(num_envs, 3, device=device)
        self.elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.command = torch.zeros(num_envs, 3, device=device)

    def inputs(self, pose, command, episode_steps):
        """Advance one control and return the five inputs; an episode start or a new command restarts."""
        command = command.to(self.command)
        fresh = (episode_steps.to(self.command.device) == 0) | (command != self.command).any(-1)
        moved = advance_destination(self.destination, self.command)
        start = planar_pose(pose).to(self.destination)
        self.destination = torch.where(fresh[:, None], start, moved)
        self.elapsed = torch.where(fresh, torch.zeros_like(self.elapsed), self.elapsed + 1)
        self.command = command.clone()
        return destination_features(pose, self.destination, self.elapsed)
