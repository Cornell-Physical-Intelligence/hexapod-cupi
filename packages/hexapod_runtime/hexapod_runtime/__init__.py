"""The deployable policy runtime: one observation builder, one action pipeline.

Both halves are pure Python over ``hexapod_core``'s frozen contracts, so the
same code path runs in simulation and on the robot. That is the point: the
runtime is not a re-derivation of the training environment's behavior, it is
the same behavior, and ``isaaclab/tests/test_runtime_parity.py`` fails if the
two ever disagree.
"""

from __future__ import annotations

from . import action_pipeline, observation_builder


__all__ = ["action_pipeline", "observation_builder"]
