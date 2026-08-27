"""Operator-side run composition, run contract, and startup supervision.

The bash launchers under ``isaaclab/deploy`` remain the executors: they hold the
shared GPU flock, gate on Docker and the systemd service, verify the immutable
parent, write atomic artifacts, and clean up by exact container ID. This package
wraps them. It composes their argv from the named configs under ``configs/``,
supervises one launcher invocation as a child process, times the startup
milestones the bash cannot see, captures diagnostics on a stall, and makes the
bounded retry decision specified in ``docs/OPERATIONS.md`` §6.

Nothing here imports torch, Isaac Lab, or ``hexapod_env``; the whole package is
standard library only so it runs under any interpreter.

Implemented to spec; not yet validated in vivo on the Spark.
"""

from __future__ import annotations

__all__ = [
    "compose",
    "configio",
    "contract",
    "gates",
    "labels",
    "supervisor",
    "system",
]

__version__ = "0.1.0"
