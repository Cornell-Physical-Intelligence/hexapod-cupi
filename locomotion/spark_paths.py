"""Bound CUPI allocations and retain the historical Spark path contract."""
from pathlib import Path

LEGACY_ROOT = Path('/home/orionh/HEXAPOD_runs/restart_20260914')
CUPI_ROOT = Path('/srv/cupi/hexapod')
RUN_ROOTS = (LEGACY_ROOT, CUPI_ROOT / 'runs')
INPUT_ROOTS = (LEGACY_ROOT, CUPI_ROOT / 'inputs')


def within_roots(value, roots):
    """Check remote path syntax; the host launcher also rejects symlinks."""
    path = Path(value)
    return path.is_absolute() and '..' not in path.parts and any(root in path.parents for root in roots)
