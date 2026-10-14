"""Compatibility import for tripod_evaluate."""
import sys
from .evaluation import tripod_evaluate as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
