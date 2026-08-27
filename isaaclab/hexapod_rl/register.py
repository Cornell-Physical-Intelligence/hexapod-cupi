"""Compatibility shim for ``hexapod_rl.register``.

The implementation now lives in ``hexapod_env.register``
(``packages/hexapod_env/hexapod_env/register.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.register import *  # noqa: F401,F403
from hexapod_env.register import (
    TASK_ID,
    PHASE1_V2_TASK_ID,
    PHASE1_V3_TASK_ID,
    PHASE1_V4_TASK_ID,
    PHASE1_V5_TASK_ID,
    PHASE2_WARMUP_TASK_ID,
    PHASE2_RECOVERY_STAGE1_TASK_ID,
    PHASE2_RECOVERY_STAGE2_TASK_ID,
    PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID,
    PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID,
    STAGE2G_INSECT_GAIT_TASK_ID,
    STAGE2G_INSECT_GAIT_ADAPT_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C0_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C1_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C2_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C3_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C4_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C5_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E0_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E1_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E2_TASK_ID,
    PHASE2_FINAL_TASK_ID,
    PHASE2_TASK_ID,
    register_envs,
)

__all__ = [
    "TASK_ID",
    "PHASE1_V2_TASK_ID",
    "PHASE1_V3_TASK_ID",
    "PHASE1_V4_TASK_ID",
    "PHASE1_V5_TASK_ID",
    "PHASE2_WARMUP_TASK_ID",
    "PHASE2_RECOVERY_STAGE1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2_TASK_ID",
    "PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID",
    "PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID",
    "STAGE2G_INSECT_GAIT_TASK_ID",
    "STAGE2G_INSECT_GAIT_ADAPT_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C0_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C2_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C3_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C4_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C5_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E0_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E2_TASK_ID",
    "PHASE2_FINAL_TASK_ID",
    "PHASE2_TASK_ID",
    "register_envs",
]
