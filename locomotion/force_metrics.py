"""Compatibility import for force_metrics."""
import sys
from .evaluation import force_metrics as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
