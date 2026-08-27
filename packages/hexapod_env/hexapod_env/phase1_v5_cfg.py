"""Corrected Phase 1 task with anatomical nose-forward navigation axes.

The imported robot root uses body ``-Y`` as its physical nose direction.  This
profile exposes the conventional navigation command ``+X`` as anatomical
forward and body ``+X`` as navigation ``+Y`` (left), without rotating the
articulation or changing joint coordinates.
"""

from isaaclab.utils.configclass import configclass

from .phase1_v4_cfg import HexapodPhase1V4EnvCfg
from .phase1_v2_cfg import HexapodPhase1V2PPORunnerCfg


@configclass
class HexapodPhase1V5EnvCfg(HexapodPhase1V4EnvCfg):
    """Train nose-first forward motion in the anatomical navigation frame."""

    command_frame: str = "navigation"


@configclass
class HexapodPhase1V5PPORunnerCfg(HexapodPhase1V2PPORunnerCfg):
    """Scratch PPO run for the corrected long-axis forward gait."""

    max_iterations = 500
    save_interval = 25
    experiment_name = "hexapod_robstride_phase1_anatomical_forward_v5_direct"


__all__ = ["HexapodPhase1V5EnvCfg", "HexapodPhase1V5PPORunnerCfg"]
