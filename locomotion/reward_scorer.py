"""Compatibility import for reward_scorer."""
import sys
from .evaluation import reward_scorer as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
