"""Locate the toe points on each tibia and turn raw articulation readings into the measured state."""

import numpy as np
import torch

from locomotion.omniwalk.mdp.frames import inverse_rotate, rotate
from locomotion.robots.hexapod import LEGS


def toe_points(geometry_meta, reference_metadata):
    """Six tibia-frame toe points: the reference stance's, else the far end of each toe cap on its axis."""
    default = []
    for leg in LEGS:
        shape = next(x for x in geometry_meta["shapes"] if x["body"] == leg + "_tibia")
        transform = np.asarray(shape["shape_to_link"])
        bounds = np.asarray(shape["cap_bounds_m"])
        default.append((transform @ np.array([bounds[1, 0], 0.0, 0.0, 1.0]))[:3])
    points = np.asarray(reference_metadata.get("toe_local_points_m", default))
    # Both environments hold the points in float32, so check the range there.
    if points.shape != (6, 3) or not np.isfinite(points.astype(np.float32)).all():
        raise ValueError("Invalid reference toe points")
    return points


def measured_state(q, dq, root, link, velocity, root_com_local, toe_index, toe_local):
    """Body-frame rates, gravity and toe positions from canonical-order joint readings and native link poses.

    ``velocity`` holds the root's center-of-mass linear velocity and its angular velocity in world axes;
    ``linear`` is the velocity of the root frame's origin in body axes.
    """
    for key, value in (
        ("q", q),
        ("dq", dq),
        ("root", root),
        ("link", link),
        ("velocity", velocity),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite native " + key)
    quaternion = root[:, 3:]
    angular = inverse_rotate(quaternion, velocity[:, 3:])
    com_offset_world = rotate(quaternion, root_com_local)
    origin_velocity = velocity[:, :3] - torch.cross(velocity[:, 3:], com_offset_world, dim=-1)
    linear = inverse_rotate(quaternion, origin_velocity)
    down = torch.as_tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=root.device)
    gravity = inverse_rotate(quaternion, down.expand(len(root), -1))
    toe_pose = link[:, toe_index]
    toe_world = toe_pose[:, :, :3] + rotate(toe_pose[:, :, 3:], toe_local.expand(len(root), -1, -1))
    toe_body = inverse_rotate(quaternion[:, None].expand(-1, 6, -1), toe_world - root[:, None, :3])
    return {
        "q": q,
        "dq": dq,
        "root": root,
        "link": link,
        "root_velocity": velocity,
        "linear": linear,
        "angular": angular,
        "gravity": gravity,
        "toe_world": toe_world,
        "toe_body": toe_body,
    }
