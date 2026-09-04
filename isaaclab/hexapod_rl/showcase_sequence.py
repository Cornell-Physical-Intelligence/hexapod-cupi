"""Compatibility shim for ``hexapod_rl.showcase_sequence``.

The implementation now lives in ``hexapod_env.showcase_sequence``
(``packages/hexapod_env/hexapod_env/showcase_sequence.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.showcase_sequence import *  # noqa: F401,F403
from hexapod_env.showcase_sequence import (
    CommandSegment,
    ScheduledCommandSegment,
    default_showcase_segments,
    schedule_segments,
    acceptance_thresholds,
    evaluate_segment_acceptance,
    annotate_showcase_frame,
)

__all__ = [
    "CommandSegment",
    "ScheduledCommandSegment",
    "acceptance_thresholds",
    "annotate_showcase_frame",
    "default_showcase_segments",
    "evaluate_segment_acceptance",
    "schedule_segments",
]
