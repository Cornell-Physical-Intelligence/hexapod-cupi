"""Isolated Phase 3 perception configuration for the RobStride hexapod.

This package is intentionally not imported by :mod:`hexapod_rl` and does not
register a Gym task.  Importing it therefore cannot change the active Phase 1
environment or training run.
"""

from .sensor_model import (
    DEFAULT_PHASE3_SENSOR_MODEL_CFG,
    Phase3SensorFrame,
    Phase3SensorModel,
    Phase3SensorModelCfg,
)

__all__ = [
    "DEFAULT_PHASE3_SENSOR_MODEL_CFG",
    "Phase3SensorFrame",
    "Phase3SensorModel",
    "Phase3SensorModelCfg",
]
