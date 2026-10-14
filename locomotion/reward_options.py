"""Compatibility import for reward_options."""
import sys
from .tasks import reward_options as _implementation

sys.modules[__name__] = _implementation
