"""Compatibility import for tripod_config."""
import sys
from .tasks import tripod_config as _implementation

sys.modules[__name__] = _implementation
