"""Compatibility import for task_v4_yaw_gate."""
import sys
from .tasks import task_v4_yaw_gate as _implementation
sys.modules[__name__] = _implementation
