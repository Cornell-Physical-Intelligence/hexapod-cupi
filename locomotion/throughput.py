"""Compatibility import for throughput."""
import sys
from .evaluation import throughput as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
