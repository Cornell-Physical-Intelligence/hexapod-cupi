"""Compatibility shim for ``hexapod_rl.asset_cfg``.

The implementation now lives in ``hexapod_env.asset_cfg``
(``packages/hexapod_env/hexapod_env/asset_cfg.py``). Names are re-exported
explicitly so ``importlib``-based entry-point resolution finds real attributes
here. Importing this module first imports the ``hexapod_rl`` package, whose
``__init__`` puts the workspace ``packages/`` directory on ``sys.path``; that
bootstrap is idempotent, so it is equally correct if it already ran.
"""

from hexapod_env.asset_cfg import *  # noqa: F401,F403
from hexapod_env.asset_cfg import (
    USD_PATH,
    ROOT_LINK_NAME,
    COXA_JOINTS,
    FEMUR_JOINTS,
    TIBIA_JOINTS,
    ROBSTRIDE_RS05_CFG,
    HEXAPOD_CFG,
)

__all__ = [
    "USD_PATH",
    "ROOT_LINK_NAME",
    "COXA_JOINTS",
    "FEMUR_JOINTS",
    "TIBIA_JOINTS",
    "ROBSTRIDE_RS05_CFG",
    "HEXAPOD_CFG",
]
