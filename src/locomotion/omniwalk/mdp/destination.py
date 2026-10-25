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


# ``ghost_v10`` inputs: what a robot can still act on when the ghost walks at its top speed. Seven values: the
# planar gap with its part along the ghost's path removed (the sideways distance; the whole gap under a
# zero command), in the body's navigation axes over 0.1 m, then the sine and cosine of the heading gap and
# the controls since the command began over 250 as in ``destination_features``, then the body's planar
# velocity over the last control less the ghost's, in navigation axes over 0.1 m/s. The first five fill
# the columns of the five ``ghost`` inputs.
GHOST_V10_FEATURES = 7
VELOCITY_SCALE_MPS = 0.1


def ghost_v10_features(pose, previous_pose, destination, elapsed, command):
    """The seven ``ghost_v10`` inputs for root poses now and one control earlier, destinations, the controls
    since each command began and the commands (forward, left, yaw rate)."""
    actual = planar_pose(pose).to(destination)
    before = planar_pose(previous_pose).to(destination)
    c, s = torch.cos(actual[:, 2]), torch.sin(actual[:, 2])

    def navigation(vector):
        body_x, body_y = c * vector[:, 0] + s * vector[:, 1], -s * vector[:, 0] + c * vector[:, 1]
        return torch.stack((-body_y, body_x), -1)

    gap = navigation(destination[:, :2] - actual[:, :2])
    turn = destination[:, 2] - actual[:, 2]
    ct, st = torch.cos(turn), torch.sin(turn)
    planar = command[:, :2].to(destination)
    ghost = torch.stack(
        (ct * planar[:, 0] - st * planar[:, 1], st * planar[:, 0] + ct * planar[:, 1]), -1
    )
    speed = torch.linalg.vector_norm(ghost, dim=-1, keepdim=True)
    direction = ghost / speed.clamp_min(1e-9)
    side = torch.where(speed > 0, gap - (gap * direction).sum(-1, keepdim=True) * direction, gap)
    velocity = navigation((actual[:, :2] - before[:, :2]) / CONTROL_DT_S)
    return torch.cat(
        (
            side / DISTANCE_SCALE_M,
            torch.stack((torch.sin(turn), torch.cos(turn)), -1),
            elapsed.to(destination)[:, None] / LONGEST_COMMAND_CONTROLS,
            (velocity - ghost) / VELOCITY_SCALE_MPS,
        ),
        -1,
    )


class GhostV10Tracker(DestinationTracker):
    """``DestinationTracker`` that returns the seven ``ghost_v10`` inputs; the first control of an episode
    reads the body at rest."""

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.pose = None

    def inputs(self, pose, command, episode_steps):
        super().inputs(pose, command, episode_steps)
        fresh = (episode_steps.to(pose.device) == 0)[:, None]
        previous = pose if self.pose is None else torch.where(fresh, pose, self.pose)
        self.pose = pose.clone()
        return ghost_v10_features(pose, previous, self.destination, self.elapsed, self.command)
