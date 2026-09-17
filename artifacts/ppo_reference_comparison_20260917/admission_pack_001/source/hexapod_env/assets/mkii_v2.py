"""Corrected simulation configuration of the existing serial CAD assembly.

V2 versions the import/reset contract, not the mechanical design. V1 is frozen
so archived checkpoints and their hash manifest retain their meaning. The
new USD must pass the URDF/USD integrity and geometric reset gates before use.
The runtime order is still checked against the articulation at construction;
a new PhysX import must confirm it before any new training is admitted.
"""
from dataclasses import replace

from .spec import MKII_V1_ASSET, RUNTIME_ORDER_PROVISIONAL

MKII_V2_ASSET = replace(
    MKII_V1_ASSET,
    name="mkii_v2",
    usd_path_container=(
        "/workspace/hexapod/robot/hexapod_mkii_assy/usd/"
        "hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda"
    ),
    usd_env_var="HEXAPOD_MKII_V2_USD_PATH",
    # Exact collision geometry gives 0.137963888698354 m. Round upward to
    # 1 micrometre, and release all six pads at least 5 mm above the plane.
    # This nominal height is geometric contact, NOT measured settling.
    nominal_height_m=0.137964,
    reset_root_height_m=0.142964,
    runtime_order_status=RUNTIME_ORDER_PROVISIONAL,
    mass_kg=8.26081134,
)

__all__ = ["MKII_V2_ASSET"]
