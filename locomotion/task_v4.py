"""Compatibility import for task_v4."""
import sys
from .tasks import task_v4 as _implementation
sys.modules[__name__] = _implementation
