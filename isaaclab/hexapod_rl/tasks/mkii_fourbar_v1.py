"""Isaac deployment entry points for the new physical-linkage task."""
import sys
from pathlib import Path

_PACKAGES = Path(__file__).resolve().parents[3] / "packages"
for _name in ("hexapod_core", "hexapod_runtime"):
    _root = str(_PACKAGES / _name)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from hexapod_env.tasks.mkii_fourbar_v1.config import HexapodMkiiFourbarV1EnvCfg, HexapodMkiiFourbarV1PPORunnerCfg


def __getattr__(name):
    # Hydra imports this module before Kit to resolve configuration. Importing
    # Articulation/DirectRLEnv here would load standalone USD into that process.
    if name == "HexapodMkiiFourbarEnv":
        from hexapod_env.tasks.mkii_fourbar_v1.env import HexapodMkiiFourbarEnv
        return HexapodMkiiFourbarEnv
    raise AttributeError(name)
