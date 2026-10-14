"""Compatibility import for task_v2."""
import sys
from .tasks import task_v2 as _implementation
sys.modules[__name__] = _implementation
