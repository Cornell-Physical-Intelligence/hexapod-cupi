"""Compatibility import for amp_ppo."""
import sys
from .learning import amp_ppo as _implementation
sys.modules[__name__] = _implementation
