"""Compatibility import for evaluation_config."""
import sys
from .evaluation import evaluation_config as _implementation

sys.modules[__name__] = _implementation
