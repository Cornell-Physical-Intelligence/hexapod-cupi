"""Compatibility entry point for spark_paths."""
import sys
from .runs import spark_paths as _implementation

sys.modules[__name__] = _implementation
