"""Compatibility import for rate_schedule."""
import sys
from .learning import rate_schedule as _implementation
sys.modules[__name__] = _implementation
