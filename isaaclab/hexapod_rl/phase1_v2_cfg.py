"""Compatibility shim for ``hexapod_rl.phase1_v2_cfg``.

The implementation now lives in ``hexapod_env.phase1_v2_cfg``
(``packages/hexapod_env/hexapod_env/phase1_v2_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.phase1_v2_cfg import *  # noqa: F401,F403
from hexapod_env.phase1_v2_cfg import (
    HexapodPhase1V2EnvCfg,
    HexapodPhase1V2PPORunnerCfg,
)

__all__ = [
    "HexapodPhase1V2EnvCfg",
    "HexapodPhase1V2PPORunnerCfg",
]
