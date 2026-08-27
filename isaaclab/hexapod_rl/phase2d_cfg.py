"""Compatibility shim for ``hexapod_rl.phase2d_cfg``.

The implementation now lives in ``hexapod_env.phase2d_cfg``
(``packages/hexapod_env/hexapod_env/phase2d_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.phase2d_cfg import *  # noqa: F401,F403
from hexapod_env.phase2d_cfg import (
    HexapodStage2DC0CommandCfg,
    HexapodStage2DC1CommandCfg,
    HexapodStage2DC2CommandCfg,
    HexapodStage2DC3CommandCfg,
    HexapodStage2DC4CommandCfg,
    HexapodStage2DC5CommandCfg,
    HexapodPhase2RecoveryStage2DC0EnvCfg,
    HexapodPhase2RecoveryStage2DC1EnvCfg,
    HexapodPhase2RecoveryStage2DC2EnvCfg,
    HexapodPhase2RecoveryStage2DC3EnvCfg,
    HexapodPhase2RecoveryStage2DC4EnvCfg,
    HexapodPhase2RecoveryStage2DC5EnvCfg,
    HexapodPhase2RecoveryStage2DC0PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC1PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC2PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC3PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC4PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC5PPORunnerCfg,
)

__all__ = [
    "HexapodPhase2RecoveryStage2DC0EnvCfg",
    "HexapodPhase2RecoveryStage2DC0PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC1EnvCfg",
    "HexapodPhase2RecoveryStage2DC1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC2EnvCfg",
    "HexapodPhase2RecoveryStage2DC2PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC3EnvCfg",
    "HexapodPhase2RecoveryStage2DC3PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC4EnvCfg",
    "HexapodPhase2RecoveryStage2DC4PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC5EnvCfg",
    "HexapodPhase2RecoveryStage2DC5PPORunnerCfg",
    "HexapodStage2DC0CommandCfg",
    "HexapodStage2DC1CommandCfg",
    "HexapodStage2DC2CommandCfg",
    "HexapodStage2DC3CommandCfg",
    "HexapodStage2DC4CommandCfg",
    "HexapodStage2DC5CommandCfg",
]
