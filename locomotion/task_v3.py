"""Compatibility import for task_v3."""
import sys
from .tasks import task_v3 as _implementation
sys.modules[__name__] = _implementation
