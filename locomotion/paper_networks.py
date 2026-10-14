"""Compatibility import for paper_networks."""
import sys
from .learning import paper_networks as _implementation
sys.modules[__name__] = _implementation
