"""Compatibility shim for ``hexapod_rl.phase2e_cfg``.

The implementation now lives in ``hexapod_env.phase2e_cfg``
(``packages/hexapod_env/hexapod_env/phase2e_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.phase2e_cfg import *  # noqa: F401,F403
from hexapod_env.phase2e_cfg import (
    HexapodStage2EE0CommandCfg,
    HexapodStage2EE1CommandCfg,
    HexapodStage2EE2CommandCfg,
    HexapodPhase2RecoveryStage2EE0EnvCfg,
    HexapodPhase2RecoveryStage2EE1EnvCfg,
    HexapodPhase2RecoveryStage2EE2EnvCfg,
    HexapodPhase2RecoveryStage2EE0PPORunnerCfg,
    HexapodPhase2RecoveryStage2EE1PPORunnerCfg,
    HexapodPhase2RecoveryStage2EE2PPORunnerCfg,
)

__all__ = [
    "HexapodPhase2RecoveryStage2EE0EnvCfg",
    "HexapodPhase2RecoveryStage2EE0PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EE1EnvCfg",
    "HexapodPhase2RecoveryStage2EE1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EE2EnvCfg",
    "HexapodPhase2RecoveryStage2EE2PPORunnerCfg",
    "HexapodStage2EE0CommandCfg",
    "HexapodStage2EE1CommandCfg",
    "HexapodStage2EE2CommandCfg",
]
