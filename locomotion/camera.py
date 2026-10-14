"""Compatibility import for camera."""
import sys
from .evaluation import camera as _implementation

sys.modules[__name__] = _implementation
