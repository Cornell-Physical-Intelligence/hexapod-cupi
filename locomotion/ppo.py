"""Compatibility import for ppo."""
import sys
from .learning import ppo as _implementation
sys.modules[__name__] = _implementation
