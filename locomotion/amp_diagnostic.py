"""Compatibility import for amp_diagnostic."""
import sys
from .evaluation import amp_diagnostic as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
