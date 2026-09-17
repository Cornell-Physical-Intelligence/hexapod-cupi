"""Deployment entry point for the approved RS05 direct task."""
import sys
from pathlib import Path

_PACKAGES = Path(__file__).resolve().parents[3] / "packages"
for _name in ("hexapod_core", "hexapod_env"):
    _root = str(_PACKAGES / _name)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from hexapod_env.tasks.mkii_rs05.config import (  # noqa: E402,F401
    HexapodMkiiRs05FlatEnvCfg,
    HexapodMkiiRs05PPORunnerCfg,
)


def __getattr__(name):
    # Hydra imports this module before Kit to resolve configuration. Importing
    # the environment here would load the simulator into that process.
    if name == "HexapodMkiiRs05Env":
        from hexapod_env.tasks.mkii_rs05.env import HexapodMkiiRs05Env

        return HexapodMkiiRs05Env
    raise AttributeError(name)


__all__ = ["HexapodMkiiRs05Env", "HexapodMkiiRs05FlatEnvCfg", "HexapodMkiiRs05PPORunnerCfg"]
