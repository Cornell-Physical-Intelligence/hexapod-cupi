"""Compatibility shim for ``hexapod_rl.phase2_cfg``.

The implementation now lives in ``hexapod_env.phase2_cfg``
(``packages/hexapod_env/hexapod_env/phase2_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.phase2_cfg import *  # noqa: F401,F403
from hexapod_env.phase2_cfg import (
    HexapodVelocityCommandCfg,
    HexapodStage1RecoveryCommandCfg,
    HexapodStage2RecoveryCommandCfg,
    HexapodStage2BLateralAcquisitionCommandCfg,
    HexapodStage2CStabilizedForwardCommandCfg,
    HexapodPhase2RecoveryStage1EnvCfg,
    HexapodPhase2RecoveryStage1PPORunnerCfg,
    HexapodPhase2RecoveryStage2EnvCfg,
    HexapodPhase2RecoveryStage2PPORunnerCfg,
    HexapodPhase2RecoveryStage2BLateralEnvCfg,
    HexapodPhase2RecoveryStage2BLateralPPORunnerCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg,
    HexapodPhase2WarmupEnvCfg,
    HexapodPhase2FinalEnvCfg,
    HexapodPhase2WarmupPPORunnerCfg,
    HexapodPhase2FinalPPORunnerCfg,
)

__all__ = [
    "HexapodPhase2RecoveryStage1EnvCfg",
    "HexapodPhase2RecoveryStage1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EnvCfg",
    "HexapodPhase2RecoveryStage2PPORunnerCfg",
    "HexapodPhase2RecoveryStage2BLateralEnvCfg",
    "HexapodPhase2RecoveryStage2BLateralPPORunnerCfg",
    "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg",
    "HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg",
    "HexapodPhase2FinalEnvCfg",
    "HexapodPhase2FinalPPORunnerCfg",
    "HexapodPhase2WarmupEnvCfg",
    "HexapodPhase2WarmupPPORunnerCfg",
    "HexapodStage1RecoveryCommandCfg",
    "HexapodStage2RecoveryCommandCfg",
    "HexapodStage2BLateralAcquisitionCommandCfg",
    "HexapodStage2CStabilizedForwardCommandCfg",
    "HexapodVelocityCommandCfg",
]
