"""Compatibility shim for ``hexapod_rl.phase2g_cfg``.

The implementation now lives in ``hexapod_env.phase2g_cfg``
(``packages/hexapod_env/hexapod_env/phase2g_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.phase2g_cfg import *  # noqa: F401,F403
from hexapod_env.phase2g_cfg import (
    HexapodStage2GInsectGaitEnvCfg,
    HexapodStage2GInsectGaitAdaptEnvCfg,
    HexapodStage2GInsectGaitPPORunnerCfg,
    HexapodStage2GInsectGaitAdaptPPORunnerCfg,
)

__all__ = [
    "HexapodStage2GInsectGaitEnvCfg",
    "HexapodStage2GInsectGaitAdaptEnvCfg",
    "HexapodStage2GInsectGaitPPORunnerCfg",
    "HexapodStage2GInsectGaitAdaptPPORunnerCfg",
]
