"""Compatibility shim for ``hexapod_rl.env_cfg``.

The implementation now lives in ``hexapod_env.env_cfg``
(``packages/hexapod_env/hexapod_env/env_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.env_cfg import *  # noqa: F401,F403
from hexapod_env.env_cfg import (
    LEG_LINK_NAMES,
    GROUND_PLANE_COLLISION_PATH,
    EventCfg,
    HexapodFlatEnvCfg,
    MkiiV1EventCfg,
    HexapodMkiiV1FlatEnvCfg,
)

__all__ = [
    "LEG_LINK_NAMES",
    "GROUND_PLANE_COLLISION_PATH",
    "EventCfg",
    "HexapodFlatEnvCfg",
    "MkiiV1EventCfg",
    "HexapodMkiiV1FlatEnvCfg",
]
