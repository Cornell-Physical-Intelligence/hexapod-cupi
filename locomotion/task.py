"""Compatibility import for task."""
import sys
from .tasks import task as _implementation
sys.modules[__name__] = _implementation
