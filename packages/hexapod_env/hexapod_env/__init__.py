"""Isaac Lab locomotion task for the RobStride-powered hexapod."""

from __future__ import annotations

from typing import Any

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


def __getattr__(name: str) -> Any:
    """Resolve the task IDs and ``register_envs`` from :mod:`.register` on use.

    ``register`` imports ``gymnasium``, which only exists inside the Isaac Lab
    container, so the import is deferred. Every historical consumer still sees
    the same eager surface: the ``hexapod_rl`` compatibility shim imports these
    names by name at import time, which resolves them immediately. Deferring
    also keeps :mod:`hexapod_env.rewards` importable with ``torch`` alone.
    """

    if name in __all__:
        from . import register

        return getattr(register, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
