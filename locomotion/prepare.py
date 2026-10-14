"""Compatibility entry point for prepare."""
import sys
from .runs import prepare as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
