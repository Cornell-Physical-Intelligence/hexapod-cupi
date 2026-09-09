"""Compatibility entry point for the opt-in CAD v2 configuration."""
import sys
from pathlib import Path

# The historical shim bootstraps only hexapod_env. V2 also shares its pure
# action contract with the runtime package through hexapod_core.
_CORE_ROOT = str(Path(__file__).resolve().parents[3] / "packages" / "hexapod_core")
if _CORE_ROOT not in sys.path:
    sys.path.insert(0, _CORE_ROOT)

from hexapod_env.tasks.mkii_v2.config import (  # noqa: E402,F401
    HEXAPOD_MKII_V2_CFG,
    HexapodMkiiV2FlatEnvCfg,
    HexapodMkiiV2PPORunnerCfg,
)

__all__ = ["HEXAPOD_MKII_V2_CFG", "HexapodMkiiV2FlatEnvCfg", "HexapodMkiiV2PPORunnerCfg"]
