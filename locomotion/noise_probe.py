"""Compatibility import for noise_probe."""
import sys
from .evaluation import noise_probe as _implementation

sys.modules[__name__] = _implementation
