"""Compatibility import for amp_discriminator."""
import sys
from .learning import amp_discriminator as _implementation
sys.modules[__name__] = _implementation
