"""Compatibility import for tripod."""
import sys
from .tasks import tripod as _implementation

sys.modules[__name__] = _implementation
