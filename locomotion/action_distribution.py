"""Compatibility import for action_distribution."""
import sys
from .learning import action_distribution as _implementation
sys.modules[__name__] = _implementation
