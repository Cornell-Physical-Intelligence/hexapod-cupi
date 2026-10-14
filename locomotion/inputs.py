"""Compatibility entry point for inputs."""
import sys
from .runs import inputs as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
