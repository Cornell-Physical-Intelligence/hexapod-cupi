"""Gym registration callback used by Isaac Lab's training and play scripts."""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "Isaac-Velocity-Flat-Hexapod-RobStride-Direct-v0"
PHASE1_V2_TASK_ID = "Isaac-Velocity-Flat-Hexapod-RobStride-Phase1-V2-Direct-v0"
PHASE1_V3_TASK_ID = "Isaac-Velocity-Flat-Hexapod-RobStride-Phase1-V3-Direct-v0"
PHASE1_V4_TASK_ID = "Isaac-Velocity-Flat-Hexapod-RobStride-Phase1-V4-Direct-v0"
PHASE1_V5_TASK_ID = (
    "Isaac-Velocity-Flat-Hexapod-RobStride-Phase1-Anatomical-V5-Direct-v0"
)
PHASE2_WARMUP_TASK_ID = "Isaac-Velocity-Omni-Warmup-Hexapod-RobStride-Direct-v0"
PHASE2_RECOVERY_STAGE1_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage1-Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2-Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2B-Lateral-Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-"
    "Hexapod-RobStride-Direct-v0"
)
STAGE2G_INSECT_GAIT_TASK_ID = (
    "Isaac-Velocity-Omni-Stage2G-Insect-Gait-"
    "Hexapod-RobStride-Direct-v0"
)
STAGE2G_INSECT_GAIT_ADAPT_TASK_ID = (
    "Isaac-Velocity-Omni-Stage2G-Insect-Gait-Adapt-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C0_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C0-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C1_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C1-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C2_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C2-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C3_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C3-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C4_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C4-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2D_C5_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-C5-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2E_E0_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E0-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2E_E1_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E1-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_RECOVERY_STAGE2E_E2_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E2-"
    "Hexapod-RobStride-Direct-v0"
)
PHASE2_FINAL_TASK_ID = "Isaac-Velocity-Omni-Hexapod-RobStride-Direct-v0"
# Stable shorthand for callers that only need the completed Phase 2 profile.
PHASE2_TASK_ID = PHASE2_FINAL_TASK_ID
# Asset v1 (ADR-0001): the CAD assembly under its own ID. Every ID above keeps
# loading the Phase-0 mock it was trained on.
MKII_V1_FLAT_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0"


def register_envs() -> list[str]:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.env_cfg:HexapodFlatEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.ppo_cfg:HexapodPPORunnerCfg",
            },
        )
    if MKII_V1_FLAT_TASK_ID not in gym.registry:
        gym.register(
            id=MKII_V1_FLAT_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.env_cfg:HexapodMkiiV1FlatEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.ppo_cfg:HexapodMkiiV1PPORunnerCfg",
            },
        )
    if PHASE1_V2_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE1_V2_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase1_v2_cfg:HexapodPhase1V2EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase1_v2_cfg:HexapodPhase1V2PPORunnerCfg"
                ),
            },
        )
    if PHASE1_V3_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE1_V3_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase1_v3_cfg:HexapodPhase1V3EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase1_v3_cfg:HexapodPhase1V3PPORunnerCfg"
                ),
            },
        )
    if PHASE1_V4_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE1_V4_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase1_v4_cfg:HexapodPhase1V4EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase1_v4_cfg:HexapodPhase1V4PPORunnerCfg"
                ),
            },
        )
    if PHASE1_V5_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE1_V5_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase1_v5_cfg:HexapodPhase1V5EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase1_v5_cfg:HexapodPhase1V5PPORunnerCfg"
                ),
            },
        )
    if PHASE2_WARMUP_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_WARMUP_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase2_cfg:HexapodPhase2WarmupEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.phase2_cfg:HexapodPhase2WarmupPPORunnerCfg",
            },
        )
    if PHASE2_RECOVERY_STAGE1_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_RECOVERY_STAGE1_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:HexapodPhase2RecoveryStage1EnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage1PPORunnerCfg"
                ),
            },
        )
    if PHASE2_RECOVERY_STAGE2_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_RECOVERY_STAGE2_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:HexapodPhase2RecoveryStage2EnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage2PPORunnerCfg"
                ),
            },
        )
    if PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage2BLateralEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage2BLateralPPORunnerCfg"
                ),
            },
        )
    if PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2_cfg:"
                    "HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg"
                ),
            },
        )
    if STAGE2G_INSECT_GAIT_TASK_ID not in gym.registry:
        gym.register(
            id=STAGE2G_INSECT_GAIT_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2g_cfg:HexapodStage2GInsectGaitEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2g_cfg:HexapodStage2GInsectGaitPPORunnerCfg"
                ),
            },
        )
    if STAGE2G_INSECT_GAIT_ADAPT_TASK_ID not in gym.registry:
        gym.register(
            id=STAGE2G_INSECT_GAIT_ADAPT_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hexapod_rl.phase2g_cfg:HexapodStage2GInsectGaitAdaptEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hexapod_rl.phase2g_cfg:"
                    "HexapodStage2GInsectGaitAdaptPPORunnerCfg"
                ),
            },
        )
    stage2d_registrations = (
        (
            PHASE2_RECOVERY_STAGE2D_C0_TASK_ID,
            "HexapodPhase2RecoveryStage2DC0EnvCfg",
            "HexapodPhase2RecoveryStage2DC0PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2D_C1_TASK_ID,
            "HexapodPhase2RecoveryStage2DC1EnvCfg",
            "HexapodPhase2RecoveryStage2DC1PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2D_C2_TASK_ID,
            "HexapodPhase2RecoveryStage2DC2EnvCfg",
            "HexapodPhase2RecoveryStage2DC2PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2D_C3_TASK_ID,
            "HexapodPhase2RecoveryStage2DC3EnvCfg",
            "HexapodPhase2RecoveryStage2DC3PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2D_C4_TASK_ID,
            "HexapodPhase2RecoveryStage2DC4EnvCfg",
            "HexapodPhase2RecoveryStage2DC4PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2D_C5_TASK_ID,
            "HexapodPhase2RecoveryStage2DC5EnvCfg",
            "HexapodPhase2RecoveryStage2DC5PPORunnerCfg",
        ),
    )
    for task_id, env_cfg_name, runner_cfg_name in stage2d_registrations:
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="hexapod_rl.env:HexapodEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": (
                        f"hexapod_rl.phase2d_cfg:{env_cfg_name}"
                    ),
                    "rsl_rl_cfg_entry_point": (
                        f"hexapod_rl.phase2d_cfg:{runner_cfg_name}"
                    ),
                },
            )
    stage2e_registrations = (
        (
            PHASE2_RECOVERY_STAGE2E_E0_TASK_ID,
            "HexapodPhase2RecoveryStage2EE0EnvCfg",
            "HexapodPhase2RecoveryStage2EE0PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2E_E1_TASK_ID,
            "HexapodPhase2RecoveryStage2EE1EnvCfg",
            "HexapodPhase2RecoveryStage2EE1PPORunnerCfg",
        ),
        (
            PHASE2_RECOVERY_STAGE2E_E2_TASK_ID,
            "HexapodPhase2RecoveryStage2EE2EnvCfg",
            "HexapodPhase2RecoveryStage2EE2PPORunnerCfg",
        ),
    )
    for task_id, env_cfg_name, runner_cfg_name in stage2e_registrations:
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="hexapod_rl.env:HexapodEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": (
                        f"hexapod_rl.phase2e_cfg:{env_cfg_name}"
                    ),
                    "rsl_rl_cfg_entry_point": (
                        f"hexapod_rl.phase2e_cfg:{runner_cfg_name}"
                    ),
                },
            )
    if PHASE2_FINAL_TASK_ID not in gym.registry:
        gym.register(
            id=PHASE2_FINAL_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.phase2_cfg:HexapodPhase2FinalEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.phase2_cfg:HexapodPhase2FinalPPORunnerCfg",
            },
        )
    # Isaac Lab intersects this with Hydra's unconsumed arguments.
    return sys.argv[1:]
