"""Compatibility import for motion features."""
import sys
from .core import features as _implementation
sys.modules[__name__] = _implementation
