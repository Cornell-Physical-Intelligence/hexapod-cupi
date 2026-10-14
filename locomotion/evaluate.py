"""Compatibility import for native evaluation."""
import sys
from .evaluation import runner as _implementation
sys.modules[__name__] = _implementation
