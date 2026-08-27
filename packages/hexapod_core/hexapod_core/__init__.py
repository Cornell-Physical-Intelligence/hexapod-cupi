"""Frozen, versioned interface contracts for the hexapod stack.

Everything the simulator, the runtime and the planner must agree on lives here
and nowhere else: the 66-dimensional observation layout, the 18-dimensional
action interface and its timing, the velocity-command boundary, the runtime
joint order, the RS05 actuator numbers, and the coordinate contract.

This package is stdlib-only on purpose -- no torch, no numpy, no Isaac -- so a
microcontroller-side tool, an operator laptop and the training container all
read the same numbers. Every module carries ``SCHEMA_VERSION = 1``. A v1 value
is never edited: a changed interface is a new module or a new constant beside
the old one, because the deployed checkpoint's meaning is frozen with it.
"""

from __future__ import annotations

from . import action, actuator, command, frames, joints, observation


SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "action",
    "actuator",
    "command",
    "frames",
    "joints",
    "observation",
]
