"""Compatibility import for the native Isaac environment."""
import sys
from .backends.isaac import env as _implementation
sys.modules[__name__] = _implementation
