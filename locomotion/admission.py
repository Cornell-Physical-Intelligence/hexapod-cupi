"""Compatibility entry point for admission."""
import sys
from .runs import admission as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
