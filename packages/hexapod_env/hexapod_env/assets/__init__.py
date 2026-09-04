"""Robot-asset descriptions for the hexapod task.

:mod:`.spec` is stdlib-only and describes each robot model (USD, names,
limits, stance). :mod:`.articulation` turns a spec into an Isaac Lab
``ArticulationCfg`` and needs Isaac Lab; import it only inside the container.
This package ``__init__`` therefore exposes the specs alone.
"""

from .spec import (
    MKII_V1_ASSET,
    MOCK_ASSET,
    RUNTIME_ORDER_CONFIRMED,
    RUNTIME_ORDER_PROVISIONAL,
    HexapodAssetSpec,
)

__all__ = [
    "HexapodAssetSpec",
    "MKII_V1_ASSET",
    "MOCK_ASSET",
    "RUNTIME_ORDER_CONFIRMED",
    "RUNTIME_ORDER_PROVISIONAL",
]
