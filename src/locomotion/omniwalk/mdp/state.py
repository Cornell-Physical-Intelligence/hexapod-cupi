"""Locate the toe points on each tibia."""

import numpy as np

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
    if points.shape != (6, 3) or not np.isfinite(points).all():
        raise ValueError("Invalid reference toe points")
    return points
