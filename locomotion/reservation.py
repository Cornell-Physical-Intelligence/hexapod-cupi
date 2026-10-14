"""Compatibility entry point for reservation."""
import sys
from .runs import reservation as _implementation

sys.modules[__name__] = _implementation
