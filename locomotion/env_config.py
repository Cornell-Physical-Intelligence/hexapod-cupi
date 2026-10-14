"""Compatibility import for the native environment configuration."""
import sys
from .backends.isaac import config as _implementation
sys.modules[__name__] = _implementation
