"""Compatibility shim for the historical ``hexapod_rl`` package.

The implementation now lives in ``packages/hexapod_env/hexapod_env``. This
package stays behind so the Spark deployment (``PYTHONPATH`` pointed at
``isaaclab/``), the gym entry-point strings of the form
``hexapod_rl.<module>:<Class>``, and every external ``import hexapod_rl`` keep
working unchanged. Importing any ``hexapod_rl`` submodule runs this module
first, so the ``sys.path`` bootstrap below always executes before a submodule
shim body.
"""

import sys
from pathlib import Path

_PACKAGES_DIR = Path(__file__).resolve().parents[2] / "packages"
# ``packages/`` is the workspace package root; ``packages/hexapod_env`` is the
# import root of the hexapod_env distribution inside it. Both are added so this
# keeps working whichever convention a caller's PYTHONPATH already follows.
_SOURCE_ROOTS = [str(_PACKAGES_DIR), str(_PACKAGES_DIR / "hexapod_env")]
# Idempotent: each root is prepended at most once.
sys.path[:0] = [root for root in _SOURCE_ROOTS if root not in sys.path]

from hexapod_env import *  # noqa: E402,F401,F403
from hexapod_env import (  # noqa: E402
    PHASE1_V2_TASK_ID,
    PHASE1_V3_TASK_ID,
    PHASE1_V4_TASK_ID,
    PHASE1_V5_TASK_ID,
    PHASE2_FINAL_TASK_ID,
    PHASE2_RECOVERY_STAGE1_TASK_ID,
    PHASE2_RECOVERY_STAGE2_TASK_ID,
    PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID,
    PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C0_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C1_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C2_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C3_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C4_TASK_ID,
    PHASE2_RECOVERY_STAGE2D_C5_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E0_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E1_TASK_ID,
    PHASE2_RECOVERY_STAGE2E_E2_TASK_ID,
    PHASE2_TASK_ID,
    PHASE2_WARMUP_TASK_ID,
    TASK_ID,
    register_envs,
)

__all__ = [
    "PHASE1_V2_TASK_ID",
    "PHASE1_V3_TASK_ID",
    "PHASE1_V4_TASK_ID",
    "PHASE1_V5_TASK_ID",
    "PHASE2_FINAL_TASK_ID",
    "PHASE2_RECOVERY_STAGE1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2_TASK_ID",
    "PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID",
    "PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C0_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C2_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C3_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C4_TASK_ID",
    "PHASE2_RECOVERY_STAGE2D_C5_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E0_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E1_TASK_ID",
    "PHASE2_RECOVERY_STAGE2E_E2_TASK_ID",
    "PHASE2_TASK_ID",
    "PHASE2_WARMUP_TASK_ID",
    "TASK_ID",
    "register_envs",
]
