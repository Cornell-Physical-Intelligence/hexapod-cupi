"""Finite planar poses in the named navigation frame."""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Pose2D:
    """Planar pose in the world/navigation frame.

    ``heading_rad`` is the robot's forward direction measured counterclockwise
    from world +x, so a robot at ``heading_rad = 0`` moving at ``vx_mps > 0``
    travels toward increasing ``x_m``. Forward is the anatomical forward of
    :mod:`contracts.frames`, not the imported body +X.
    """

    x_m: float
    y_m: float
    heading_rad: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "heading_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
