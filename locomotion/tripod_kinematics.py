"""Compatibility import for tripod_kinematics."""
import sys
from .core import tripod_kinematics as _implementation

sys.modules[__name__] = _implementation
