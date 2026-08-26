#!/usr/bin/env python3
"""Evaluate an RSL-RL hexapod checkpoint with fixed velocity commands.

This follows Isaac Lab v3.0.0-beta2.patch1's RSL-RL play flow, but replaces
the task's randomized commands during every reset and records quantitative
pre-reset state.  Repeating ``--command`` evaluates several commands in
parallel, one command per environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation

from hexapod_rl.asset_cfg import FEMUR_JOINTS, TIBIA_JOINTS
from hexapod_rl.env_cfg import HexapodFlatEnvCfg
from hexapod_rl.phase1_v2_cfg import HexapodPhase1V2EnvCfg, HexapodPhase1V2PPORunnerCfg
from hexapod_rl.phase1_v3_cfg import HexapodPhase1V3EnvCfg, HexapodPhase1V3PPORunnerCfg
from hexapod_rl.phase1_v4_cfg import HexapodPhase1V4EnvCfg, HexapodPhase1V4PPORunnerCfg
from hexapod_rl.phase1_v5_cfg import HexapodPhase1V5EnvCfg, HexapodPhase1V5PPORunnerCfg
from hexapod_rl.phase2_cfg import (
    HexapodPhase2FinalEnvCfg,
    HexapodPhase2FinalPPORunnerCfg,
    HexapodPhase2RecoveryStage1EnvCfg,
    HexapodPhase2RecoveryStage1PPORunnerCfg,
    HexapodPhase2RecoveryStage2EnvCfg,
    HexapodPhase2RecoveryStage2PPORunnerCfg,
    HexapodPhase2RecoveryStage2BLateralEnvCfg,
    HexapodPhase2RecoveryStage2BLateralPPORunnerCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg,
    HexapodPhase2WarmupEnvCfg,
    HexapodPhase2WarmupPPORunnerCfg,
)
from hexapod_rl.phase2g_cfg import (
    HexapodStage2GInsectGaitAdaptEnvCfg,
    HexapodStage2GInsectGaitAdaptPPORunnerCfg,
    HexapodStage2GInsectGaitEnvCfg,
    HexapodStage2GInsectGaitPPORunnerCfg,
)
from hexapod_rl.phase2d_cfg import (
    HexapodPhase2RecoveryStage2DC0EnvCfg,
    HexapodPhase2RecoveryStage2DC0PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC1EnvCfg,
    HexapodPhase2RecoveryStage2DC1PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC2EnvCfg,
    HexapodPhase2RecoveryStage2DC2PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC3EnvCfg,
    HexapodPhase2RecoveryStage2DC3PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC4EnvCfg,
    HexapodPhase2RecoveryStage2DC4PPORunnerCfg,
    HexapodPhase2RecoveryStage2DC5EnvCfg,
    HexapodPhase2RecoveryStage2DC5PPORunnerCfg,
)
from hexapod_rl.phase2e_cfg import (
    HexapodPhase2RecoveryStage2EE0EnvCfg,
    HexapodPhase2RecoveryStage2EE0PPORunnerCfg,
    HexapodPhase2RecoveryStage2EE1EnvCfg,
    HexapodPhase2RecoveryStage2EE1PPORunnerCfg,
    HexapodPhase2RecoveryStage2EE2EnvCfg,
    HexapodPhase2RecoveryStage2EE2PPORunnerCfg,
)
from hexapod_rl.ppo_cfg import HexapodPPORunnerCfg
from hexapod_rl.register import (
    STAGE2G_INSECT_GAIT_TASK_ID,
    STAGE2G_INSECT_GAIT_ADAPT_TASK_ID,
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
    PHASE2_WARMUP_TASK_ID,
    TASK_ID,
    register_envs,
)


POLICY_COMMAND_SLICE = slice(9, 12)
EXPECTED_OBSERVATION_DIM = 66
TASK_CONFIGS = {
    TASK_ID: (HexapodFlatEnvCfg, HexapodPPORunnerCfg),
    PHASE1_V2_TASK_ID: (HexapodPhase1V2EnvCfg, HexapodPhase1V2PPORunnerCfg),
    PHASE1_V3_TASK_ID: (HexapodPhase1V3EnvCfg, HexapodPhase1V3PPORunnerCfg),
    PHASE1_V4_TASK_ID: (HexapodPhase1V4EnvCfg, HexapodPhase1V4PPORunnerCfg),
    PHASE1_V5_TASK_ID: (HexapodPhase1V5EnvCfg, HexapodPhase1V5PPORunnerCfg),
    PHASE2_WARMUP_TASK_ID: (
        HexapodPhase2WarmupEnvCfg,
        HexapodPhase2WarmupPPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE1_TASK_ID: (
        HexapodPhase2RecoveryStage1EnvCfg,
        HexapodPhase2RecoveryStage1PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2_TASK_ID: (
        HexapodPhase2RecoveryStage2EnvCfg,
        HexapodPhase2RecoveryStage2PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID: (
        HexapodPhase2RecoveryStage2BLateralEnvCfg,
        HexapodPhase2RecoveryStage2BLateralPPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID: (
        HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg,
        HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg,
    ),
    STAGE2G_INSECT_GAIT_TASK_ID: (
        HexapodStage2GInsectGaitEnvCfg,
        HexapodStage2GInsectGaitPPORunnerCfg,
    ),
    STAGE2G_INSECT_GAIT_ADAPT_TASK_ID: (
        HexapodStage2GInsectGaitAdaptEnvCfg,
        HexapodStage2GInsectGaitAdaptPPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C0_TASK_ID: (
        HexapodPhase2RecoveryStage2DC0EnvCfg,
        HexapodPhase2RecoveryStage2DC0PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C1_TASK_ID: (
        HexapodPhase2RecoveryStage2DC1EnvCfg,
        HexapodPhase2RecoveryStage2DC1PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C2_TASK_ID: (
        HexapodPhase2RecoveryStage2DC2EnvCfg,
        HexapodPhase2RecoveryStage2DC2PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C3_TASK_ID: (
        HexapodPhase2RecoveryStage2DC3EnvCfg,
        HexapodPhase2RecoveryStage2DC3PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C4_TASK_ID: (
        HexapodPhase2RecoveryStage2DC4EnvCfg,
        HexapodPhase2RecoveryStage2DC4PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2D_C5_TASK_ID: (
        HexapodPhase2RecoveryStage2DC5EnvCfg,
        HexapodPhase2RecoveryStage2DC5PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2E_E0_TASK_ID: (
        HexapodPhase2RecoveryStage2EE0EnvCfg,
        HexapodPhase2RecoveryStage2EE0PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2E_E1_TASK_ID: (
        HexapodPhase2RecoveryStage2EE1EnvCfg,
        HexapodPhase2RecoveryStage2EE1PPORunnerCfg,
    ),
    PHASE2_RECOVERY_STAGE2E_E2_TASK_ID: (
        HexapodPhase2RecoveryStage2EE2EnvCfg,
        HexapodPhase2RecoveryStage2EE2PPORunnerCfg,
    ),
    PHASE2_FINAL_TASK_ID: (HexapodPhase2FinalEnvCfg, HexapodPhase2FinalPPORunnerCfg),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RobStride hexapod policy with fixed commands."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        help="RSL-RL model_*.pt checkpoint path. Repeat to screen several in one Kit process.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_CONFIGS),
        default=TASK_ID,
        help="Registered task/config used to interpret the checkpoint.",
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        action="append",
        metavar=("VX", "VY", "YAW_RATE"),
        help=(
            "Fixed task-frame command in m/s, m/s, rad/s. For anatomical tasks "
            "these are forward, left, yaw. Repeat to evaluate "
            "multiple commands in parallel. Default: 0.30 0.00 0.00."
        ),
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help=(
            "Number of independently randomized environment copies per fixed command. "
            "Useful for parallel 'robot army' videos. Default: 1."
        ),
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        help="Override environment grid spacing in metres (for example 0.8 for army videos).",
    )
    parser.add_argument(
        "--processed-joint-target-slew-limit-rad-per-20ms",
        type=float,
        help=(
            "Override the task's final processed joint-target slew limit in "
            "radians per 20 ms policy step. Use the exact value from training "
            "when screening a checkpoint trained with a command-line env override."
        ),
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, help="Number of 50 Hz policy steps.")
    duration.add_argument("--seconds", type=float, default=10.0, help="Simulated duration. Default: 10 s.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=25,
        help="Steps excluded from reported metrics. Default: 25 (0.5 s).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed. Default: 42.")
    parser.add_argument(
        "--disable-randomization",
        action="store_true",
        help="Disable startup friction and base-mass randomization for a nominal-model test.",
    )
    parser.add_argument(
        "--root-height-m",
        type=float,
        help="Override the reset root height for a controlled stance A/B test.",
    )
    parser.add_argument(
        "--femur-angle-rad",
        type=float,
        help="Override all six reset/default femur angles for a stance A/B test.",
    )
    parser.add_argument(
        "--tibia-angle-rad",
        type=float,
        help="Override all six reset/default tibia angles for a stance A/B test.",
    )
    parser.add_argument("--video", action="store_true", help="Record the complete evaluation.")
    parser.add_argument(
        "--video-dir",
        help="Video output directory. Defaults to <checkpoint-dir>/videos/eval.",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Recording resolution in pixels. Isaac Lab's default is 1280 720.",
    )
    parser.add_argument(
        "--follow-camera",
        action="store_true",
        help=(
            "Smoothly translate the recording camera with one robot. Camera X/Y values "
            "become offsets from the robot while Z remains a fixed world height."
        ),
    )
    parser.add_argument(
        "--follow-env-index",
        type=int,
        default=0,
        help="Parallel environment to frame when --follow-camera is active. Default: 0.",
    )
    parser.add_argument(
        "--camera-follow-time-constant",
        type=float,
        default=0.20,
        metavar="SECONDS",
        help=(
            "Exponential X/Y camera smoothing time constant. Zero disables smoothing. "
            "Default: 0.20 s."
        ),
    )
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(1.4, 1.4, 0.8),
        metavar=("X", "Y", "Z"),
        help=(
            "Static world-space eye, or X/Y follow offsets plus fixed world Z with "
            "--follow-camera."
        ),
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.15),
        metavar=("X", "Y", "Z"),
        help=(
            "Static world-space target, or X/Y follow offsets plus fixed world Z with "
            "--follow-camera."
        ),
    )
    parser.add_argument(
        "--camera-horizontal-fov-deg",
        type=float,
        metavar="DEGREES",
        help=(
            "Pin the Kit recording camera's horizontal field of view. This makes "
            "wide army-shot framing reproducible instead of inheriting mutable viewport intrinsics."
        ),
    )
    parser.add_argument("--json", dest="json_path", help="Also write the JSON report to this path.")
    parser.add_argument(
        "--fail-on-fall",
        action="store_true",
        help="Exit with status 2 when any post-warmup fall is observed.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seconds is not None and args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.copies < 1:
        parser.error("--copies must be at least 1")
    if args.env_spacing is not None and args.env_spacing <= 0.0:
        parser.error("--env-spacing must be positive")
    if args.processed_joint_target_slew_limit_rad_per_20ms is not None and not (
        math.isfinite(args.processed_joint_target_slew_limit_rad_per_20ms)
        and args.processed_joint_target_slew_limit_rad_per_20ms > 0.0
    ):
        parser.error(
            "--processed-joint-target-slew-limit-rad-per-20ms must be finite and positive"
        )
    if args.video_dir and not args.video:
        parser.error("--video-dir requires --video")
    if args.video_resolution and not args.video:
        parser.error("--video-resolution requires --video")
    if args.video_resolution and any(value <= 0 for value in args.video_resolution):
        parser.error("--video-resolution values must be positive")
    if args.follow_camera and not args.video:
        parser.error("--follow-camera requires --video")
    if args.follow_env_index < 0:
        parser.error("--follow-env-index cannot be negative")
    if args.camera_follow_time_constant < 0.0:
        parser.error("--camera-follow-time-constant cannot be negative")
    if args.camera_horizontal_fov_deg is not None and not args.video:
        parser.error("--camera-horizontal-fov-deg requires --video")
    if args.camera_horizontal_fov_deg is not None and not (
        0.0 < args.camera_horizontal_fov_deg < 179.0
    ):
        parser.error("--camera-horizontal-fov-deg must be between 0 and 179 degrees")
    if args.root_height_m is not None and not (
        math.isfinite(args.root_height_m) and args.root_height_m > 0.0
    ):
        parser.error("--root-height-m must be finite and positive")
    if args.femur_angle_rad is not None and not (
        math.isfinite(args.femur_angle_rad) and 0.0 <= args.femur_angle_rad <= 1.74533
    ):
        parser.error("--femur-angle-rad must lie within the URDF range [0, 1.74533]")
    if args.tibia_angle_rad is not None and not (
        math.isfinite(args.tibia_angle_rad) and 0.0 <= args.tibia_angle_rad <= 2.53073
    ):
        parser.error("--tibia-angle-rad must lie within the URDF range [0, 2.53073]")
    return args


def _apply_processed_joint_target_slew_override(
    env_cfg: Any,
    override_rad_per_20ms: float | None,
) -> dict[str, float | str | None]:
    """Resolve the playback slew limit and return machine-readable provenance."""

    if override_rad_per_20ms is None:
        resolved = getattr(
            env_cfg, "processed_joint_target_slew_limit_rad_per_20ms", None
        )
        source = "task_default"
    else:
        resolved = float(override_rad_per_20ms)
        env_cfg.processed_joint_target_slew_limit_rad_per_20ms = resolved
        source = "cli_override"

    if resolved is not None and (
        not math.isfinite(float(resolved)) or float(resolved) <= 0.0
    ):
        raise ValueError(
            "resolved processed joint-target slew limit must be None or finite "
            f"and positive, got {resolved!r}"
        )
    return {
        "processed_joint_target_slew_limit_rad_per_20ms": (
            None if resolved is None else float(resolved)
        ),
        "source": source,
    }


def _apply_stance_override(env_cfg: Any, args: argparse.Namespace) -> dict[str, float] | None:
    """Apply a symmetric reset/default pose without mutating the shared asset config."""

    requested = (
        args.root_height_m,
        args.femur_angle_rad,
        args.tibia_angle_rad,
    )
    if all(value is None for value in requested):
        return None

    root_height = float(
        args.root_height_m
        if args.root_height_m is not None
        else env_cfg.robot.init_state.pos[2]
    )
    femur_angle = float(
        args.femur_angle_rad
        if args.femur_angle_rad is not None
        else env_cfg.robot.init_state.joint_pos[FEMUR_JOINTS[0]]
    )
    tibia_angle = float(
        args.tibia_angle_rad
        if args.tibia_angle_rad is not None
        else env_cfg.robot.init_state.joint_pos[TIBIA_JOINTS[0]]
    )

    root_x, root_y, _ = env_cfg.robot.init_state.pos
    env_cfg.robot.init_state.pos = (root_x, root_y, root_height)
    env_cfg.robot.init_state.joint_pos.update(
        {name: femur_angle for name in FEMUR_JOINTS}
    )
    env_cfg.robot.init_state.joint_pos.update(
        {name: tibia_angle for name in TIBIA_JOINTS}
    )
    # The height reward is not used during deterministic playback, but keeping
    # the nominal target aligned makes this config safe for short diagnostic
    # rollouts that inspect reward components.
    env_cfg.nominal_height_m = root_height
    return {
        "root_height_m": root_height,
        "femur_angle_rad": femur_angle,
        "tibia_angle_rad": tibia_angle,
    }


def _force_commands(raw_env: Any, observations: Any, commands: torch.Tensor) -> None:
    """Replace both the environment command and the already-built policy observation."""
    policy_observation = observations["policy"]
    if policy_observation.shape != (raw_env.num_envs, EXPECTED_OBSERVATION_DIM):
        raise RuntimeError(
            "The evaluator's command slice no longer matches the environment observation: "
            f"expected {(raw_env.num_envs, EXPECTED_OBSERVATION_DIM)}, got "
            f"{tuple(policy_observation.shape)}."
        )
    # Isaac Lab's observation manager can return a PyTorch inference tensor.
    # PyTorch 2.8 only permits in-place updates to such tensors while inference
    # mode is active, including the initial command injection before rollout.
    with torch.inference_mode():
        raw_env._commands.copy_(commands)
        policy_observation[:, POLICY_COMMAND_SLICE].copy_(commands)
        # Phase 2 normally resamples commands inside _get_rewards(), before the
        # evaluator's pre-reset metric hook runs.  Hold its timer at infinity so
        # this fixed-command evaluator never attributes a transition to a random
        # replacement command. Resets may recreate finite timers, and this helper
        # is deliberately called again after every step/reset.
        command_timer = getattr(raw_env, "_command_resampling_time_left_s", None)
        if command_timer is not None:
            command_timer.fill_(torch.inf)


def _set_recording_camera_intrinsics(
    camera_prim_path: str,
    horizontal_fov_deg: float,
    resolution: tuple[int, int],
) -> None:
    """Pin perspective intrinsics after Kit has created its recording camera."""

    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(camera_prim_path) if stage is not None else None
    if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f"Recording camera {camera_prim_path!r} is not a valid USD Camera prim"
        )

    width, height = resolution
    # Use Isaac Lab's standard 35 mm spherical-projector aperture and explicitly
    # match the render-product aspect ratio so both horizontal and vertical FOV
    # are deterministic.
    horizontal_aperture_mm = 20.955
    vertical_aperture_mm = horizontal_aperture_mm * float(height) / float(width)
    focal_length_mm = horizontal_aperture_mm / (
        2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0)
    )
    camera = UsdGeom.Camera(prim)
    # Kit's built-in perspective cameras live on the session layer. Author
    # intrinsics there as well so a weaker root-layer opinion cannot be ignored.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera.GetHorizontalApertureAttr().Set(horizontal_aperture_mm)
        camera.GetVerticalApertureAttr().Set(vertical_aperture_mm)
        camera.GetFocalLengthAttr().Set(focal_length_mm)

    actual_focal_length_mm = float(camera.GetFocalLengthAttr().Get())
    if not math.isclose(actual_focal_length_mm, focal_length_mm, abs_tol=1.0e-4):
        raise RuntimeError(
            "Kit recording camera rejected the requested field of view: "
            f"expected focal length {focal_length_mm:.6f} mm, got "
            f"{actual_focal_length_mm:.6f} mm"
        )


class SmoothFollowCameraWrapper(gym.Wrapper):
    """Translate the Kit recording camera with a robot without touching physics state.

    This wrapper must sit *inside* Gymnasium's :class:`RecordVideo` wrapper.  Its
    post-step camera update then happens after physics/reset handling but before
    ``RecordVideo`` asks the environment to render that step's frame.

    Only the tracked root's world X/Y position is followed.  Camera and target Z
    remain fixed in world coordinates, avoiding visible bobbing from the gait.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        env_index: int,
        eye: tuple[float, float, float],
        lookat: tuple[float, float, float],
        policy_dt: float,
        time_constant: float,
    ):
        super().__init__(env)
        raw_env = env.unwrapped
        if env_index >= raw_env.num_envs:
            raise ValueError(
                f"--follow-env-index ({env_index}) must be smaller than the number "
                f"of evaluation environments ({raw_env.num_envs})"
            )

        # Import after AppLauncher starts Kit.  This official helper targets the
        # same /OmniverseKit_Persp prim used by Isaac Lab's RGB video recorder.
        from isaaclab_physx.renderers.kit_viewport_utils import (
            set_kit_renderer_camera_view,
        )

        self._raw_env = raw_env
        self._env_index = env_index
        self._eye = tuple(float(value) for value in eye)
        self._lookat = tuple(float(value) for value in lookat)
        self._set_camera_view = set_kit_renderer_camera_view
        self._anchor_xy: list[float] | None = None
        self._smoothing_alpha = (
            1.0
            if time_constant == 0.0
            else -math.expm1(-float(policy_dt) / float(time_constant))
        )
        self._update_camera(snap=True)

    def _update_camera(self, *, snap: bool) -> None:
        root_xy = (
            self._raw_env._robot.data.root_pos_w.torch[self._env_index, :2]
            .detach()
            .cpu()
            .tolist()
        )
        desired_x, desired_y = (float(value) for value in root_xy)
        if snap or self._anchor_xy is None:
            self._anchor_xy = [desired_x, desired_y]
        else:
            alpha = self._smoothing_alpha
            self._anchor_xy[0] += alpha * (desired_x - self._anchor_xy[0])
            self._anchor_xy[1] += alpha * (desired_y - self._anchor_xy[1])

        anchor_x, anchor_y = self._anchor_xy
        eye = (anchor_x + self._eye[0], anchor_y + self._eye[1], self._eye[2])
        target = (
            anchor_x + self._lookat[0],
            anchor_y + self._lookat[1],
            self._lookat[2],
        )
        self._set_camera_view(eye=eye, target=target)

    def _tracked_env_done(self, terminated: Any, truncated: Any) -> bool:
        return bool(
            terminated[self._env_index].item() or truncated[self._env_index].item()
        )

    def reset(self, **kwargs: Any) -> Any:
        result = self.env.reset(**kwargs)
        self._update_camera(snap=True)
        return result

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self._update_camera(snap=self._tracked_env_done(result[2], result[3]))
        return result


class MetricAccumulator:
    """Accumulate one row of metrics per parallel fixed-command environment."""

    def __init__(self, raw_env: Any, warmup_steps: int):
        self.raw_env = raw_env
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.num_joints = raw_env._robot.num_joints
        self.warmup_steps = warmup_steps
        self.step_index = -1

        vector3 = lambda: torch.zeros((self.num_envs, 3), device=self.device)
        scalar = lambda: torch.zeros(self.num_envs, device=self.device)

        self.sample_count = scalar()
        self.sum_reward = scalar()
        self.sum_body_linear_velocity = vector3()
        self.sum_command_frame_linear_velocity = vector3()
        self.sum_world_linear_velocity = vector3()
        self.sum_body_angular_velocity = vector3()
        self.sum_command_frame_angular_velocity = vector3()
        self.sum_command_error = vector3()
        self.sum_abs_command_error = vector3()
        self.sum_squared_command_error = vector3()

        self.sum_abs_applied_torque = scalar()
        self.sum_squared_applied_torque = scalar()
        self.sum_abs_mechanical_power = scalar()
        self.applied_saturation_count = scalar()
        self.computed_over_rating_count = scalar()
        self.peak_abs_applied_torque = scalar()
        self.peak_abs_computed_torque = scalar()
        joint_matrix = lambda: torch.zeros(
            (self.num_envs, self.num_joints), device=self.device
        )
        self.sum_squared_applied_torque_by_joint = joint_matrix()
        self.applied_saturation_count_by_joint = joint_matrix()
        self.computed_over_rating_count_by_joint = joint_matrix()
        self.peak_abs_applied_torque_by_joint = joint_matrix()
        self.peak_abs_computed_torque_by_joint = joint_matrix()
        self.current_computed_over_run = torch.zeros(
            (self.num_envs, self.num_joints), dtype=torch.long, device=self.device
        )
        self.max_computed_over_run = torch.zeros_like(self.current_computed_over_run)
        self.computed_abs_samples: list[torch.Tensor] = []

        self.falls = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.timeouts = torch.zeros_like(self.falls)
        self.sum_base_height = scalar()
        self.sum_squared_base_height = scalar()
        self.min_base_height = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.max_base_height = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self.sum_squared_vertical_velocity = scalar()
        self.sum_squared_roll_pitch_angular_velocity = scalar()
        self.sum_squared_tilt_radians = scalar()
        # Retain the scalar stability traces so the report can include robust
        # tail metrics in addition to means/RMS. Evaluation episodes are short,
        # and the torque evaluator already keeps a larger per-joint trace.
        self.stability_samples: list[torch.Tensor] = []
        self.max_tilt_radians = scalar()

        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_root_position = vector3()
        self.cumulative_displacement = vector3()
        self.horizontal_path_length = scalar()

    @property
    def collecting(self) -> bool:
        return self.step_index >= self.warmup_steps

    def capture_pre_reset(self, reward: torch.Tensor) -> None:
        """Capture state while DirectRLEnv.step still contains the terminal pose."""
        if not self.collecting:
            return

        robot_data = self.raw_env._robot.data
        root_position = robot_data.root_pos_w.torch.clone()
        new_series = ~self.initialized
        if torch.any(new_series):
            self.last_root_position[new_series] = root_position[new_series]
            self.initialized[new_series] = True

        position_delta = root_position - self.last_root_position
        self.cumulative_displacement += position_delta
        self.horizontal_path_length += torch.linalg.norm(position_delta[:, :2], dim=1)
        self.last_root_position.copy_(root_position)

        body_linear_velocity = robot_data.root_lin_vel_b.torch
        world_linear_velocity = robot_data.root_lin_vel_w.torch
        body_angular_velocity = robot_data.root_ang_vel_b.torch
        command_frame_linear_velocity = self.raw_env._vector_in_command_frame(
            body_linear_velocity
        )
        command_frame_angular_velocity = self.raw_env._vector_in_command_frame(
            body_angular_velocity
        )
        command_error = torch.cat(
            (
                command_frame_linear_velocity[:, :2] - self.raw_env._commands[:, :2],
                command_frame_angular_velocity[:, 2:3] - self.raw_env._commands[:, 2:3],
            ),
            dim=1,
        )

        applied_torque = robot_data.applied_torque.torch
        computed_torque = robot_data.computed_torque.torch
        joint_velocity = robot_data.joint_vel.torch
        rated_torque = float(self.raw_env.cfg.rated_torque_nm)
        saturation_threshold = rated_torque - 1.0e-3

        self.sample_count += 1.0
        self.sum_reward += reward
        self.sum_body_linear_velocity += body_linear_velocity
        self.sum_command_frame_linear_velocity += command_frame_linear_velocity
        self.sum_world_linear_velocity += world_linear_velocity
        self.sum_body_angular_velocity += body_angular_velocity
        self.sum_command_frame_angular_velocity += command_frame_angular_velocity
        self.sum_command_error += command_error
        self.sum_abs_command_error += torch.abs(command_error)
        self.sum_squared_command_error += torch.square(command_error)

        self.sum_abs_applied_torque += torch.sum(torch.abs(applied_torque), dim=1)
        self.sum_squared_applied_torque += torch.sum(torch.square(applied_torque), dim=1)
        self.sum_squared_applied_torque_by_joint += torch.square(applied_torque)
        self.sum_abs_mechanical_power += torch.sum(
            torch.abs(applied_torque * joint_velocity), dim=1
        )
        applied_saturation = torch.abs(applied_torque) >= saturation_threshold
        computed_over_rating = torch.abs(computed_torque) > rated_torque
        self.applied_saturation_count += torch.sum(applied_saturation, dim=1)
        self.computed_over_rating_count += torch.sum(computed_over_rating, dim=1)
        self.applied_saturation_count_by_joint += applied_saturation
        self.computed_over_rating_count_by_joint += computed_over_rating
        self.current_computed_over_run.copy_(
            torch.where(
                computed_over_rating,
                self.current_computed_over_run + 1,
                torch.zeros_like(self.current_computed_over_run),
            )
        )
        self.max_computed_over_run.copy_(
            torch.maximum(self.max_computed_over_run, self.current_computed_over_run)
        )
        self.computed_abs_samples.append(torch.abs(computed_torque).detach().clone())
        self.peak_abs_applied_torque = torch.maximum(
            self.peak_abs_applied_torque, torch.amax(torch.abs(applied_torque), dim=1)
        )
        self.peak_abs_computed_torque = torch.maximum(
            self.peak_abs_computed_torque, torch.amax(torch.abs(computed_torque), dim=1)
        )
        self.peak_abs_applied_torque_by_joint = torch.maximum(
            self.peak_abs_applied_torque_by_joint, torch.abs(applied_torque)
        )
        self.peak_abs_computed_torque_by_joint = torch.maximum(
            self.peak_abs_computed_torque_by_joint, torch.abs(computed_torque)
        )

        self.falls += self.raw_env.reset_terminated.to(dtype=torch.long)
        self.timeouts += self.raw_env.reset_time_outs.to(dtype=torch.long)
        base_height = root_position[:, 2]
        vertical_velocity = world_linear_velocity[:, 2]
        roll_pitch_angular_velocity = torch.linalg.norm(
            body_angular_velocity[:, :2], dim=1
        )
        upright_cosine = torch.clamp(-robot_data.projected_gravity_b.torch[:, 2], -1.0, 1.0)
        tilt_radians = torch.acos(upright_cosine)

        self.sum_base_height += base_height
        self.sum_squared_base_height += torch.square(base_height)
        self.min_base_height = torch.minimum(self.min_base_height, base_height)
        self.max_base_height = torch.maximum(self.max_base_height, base_height)
        self.sum_squared_vertical_velocity += torch.square(vertical_velocity)
        self.sum_squared_roll_pitch_angular_velocity += torch.square(
            roll_pitch_angular_velocity
        )
        self.sum_squared_tilt_radians += torch.square(tilt_radians)
        self.stability_samples.append(
            torch.stack(
                (
                    torch.abs(vertical_velocity),
                    roll_pitch_angular_velocity,
                    tilt_radians,
                ),
                dim=1,
            ).detach().clone()
        )
        self.max_tilt_radians = torch.maximum(self.max_tilt_radians, tilt_radians)

    def begin_displacement_measurement(self) -> None:
        """Set the baseline immediately before the first measured policy step."""
        root_position = self.raw_env._robot.data.root_pos_w.torch
        self.last_root_position.copy_(root_position)
        self.initialized.fill_(True)

    def account_for_resets(self, dones: torch.Tensor) -> None:
        """Remove simulator reset jumps from cumulative world displacement."""
        reset_mask = dones.to(dtype=torch.bool)
        if self.collecting and torch.any(reset_mask):
            post_reset_position = self.raw_env._robot.data.root_pos_w.torch
            self.last_root_position[reset_mask] = post_reset_position[reset_mask]
            self.current_computed_over_run[reset_mask] = 0

    def report(
        self,
        commands: torch.Tensor,
        checkpoint: str,
        task_id: str,
        policy_dt: float,
        total_steps: int,
    ) -> dict[str, Any]:
        count = torch.clamp(self.sample_count, min=1.0)
        torque_count = count * self.num_joints
        measured_seconds = self.sample_count * policy_dt

        mean_body_linear = self.sum_body_linear_velocity / count[:, None]
        mean_command_frame_linear = self.sum_command_frame_linear_velocity / count[:, None]
        mean_world_linear = self.sum_world_linear_velocity / count[:, None]
        mean_body_angular = self.sum_body_angular_velocity / count[:, None]
        mean_command_frame_angular = self.sum_command_frame_angular_velocity / count[:, None]
        mean_error = self.sum_command_error / count[:, None]
        mean_abs_error = self.sum_abs_command_error / count[:, None]
        rmse_error = torch.sqrt(self.sum_squared_command_error / count[:, None])
        displacement_rate = self.cumulative_displacement / torch.clamp(
            measured_seconds[:, None], min=policy_dt
        )
        per_joint_rms_applied = torch.sqrt(
            self.sum_squared_applied_torque_by_joint / count[:, None]
        )
        computed_abs_samples = torch.stack(self.computed_abs_samples, dim=0)
        stability_samples = torch.stack(self.stability_samples, dim=0)
        mean_base_height = self.sum_base_height / count
        base_height_std = torch.sqrt(
            torch.clamp(
                self.sum_squared_base_height / count - torch.square(mean_base_height),
                min=0.0,
            )
        )
        base_height_peak_to_peak = self.max_base_height - self.min_base_height
        vertical_velocity_rms = torch.sqrt(
            self.sum_squared_vertical_velocity / count
        )
        roll_pitch_angular_velocity_rms = torch.sqrt(
            self.sum_squared_roll_pitch_angular_velocity / count
        )
        tilt_rms_radians = torch.sqrt(self.sum_squared_tilt_radians / count)
        joint_names = list(getattr(self.raw_env._robot, "joint_names", ()))
        if len(joint_names) != self.num_joints:
            joint_names = [f"joint_{index}" for index in range(self.num_joints)]

        rows: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            displacement = self.cumulative_displacement[index]
            computed_abs_flat = computed_abs_samples[:, index, :].flatten()
            stability_trace = stability_samples[:, index, :]
            per_joint_torque = [
                {
                    "name": joint_names[joint_index],
                    "rms_applied_nm": float(
                        per_joint_rms_applied[index, joint_index].item()
                    ),
                    "peak_abs_applied_nm": float(
                        self.peak_abs_applied_torque_by_joint[index, joint_index].item()
                    ),
                    "peak_abs_computed_nm": float(
                        self.peak_abs_computed_torque_by_joint[index, joint_index].item()
                    ),
                    "applied_at_rating_fraction": float(
                        (
                            self.applied_saturation_count_by_joint[index, joint_index]
                            / count[index]
                        ).item()
                    ),
                    "computed_demand_over_rating_fraction": float(
                        (
                            self.computed_over_rating_count_by_joint[index, joint_index]
                            / count[index]
                        ).item()
                    ),
                    "maximum_computed_over_rating_burst_s": float(
                        self.max_computed_over_run[index, joint_index].item() * policy_dt
                    ),
                }
                for joint_index in range(self.num_joints)
            ]
            row = {
                "index": index,
                "command": {
                    "frame": self.raw_env._command_frame,
                    "body_vx_mps": float(commands[index, 0].item()),
                    "body_vy_mps": float(commands[index, 1].item()),
                    "yaw_rate_radps": float(commands[index, 2].item()),
                },
                "samples": int(self.sample_count[index].item()),
                "measured_seconds": float(measured_seconds[index].item()),
                "displacement_world_m": [float(value) for value in displacement.tolist()],
                "horizontal_net_displacement_m": float(torch.linalg.norm(displacement[:2]).item()),
                "horizontal_path_length_m": float(self.horizontal_path_length[index].item()),
                "displacement_rate_world_mps": [
                    float(value) for value in displacement_rate[index].tolist()
                ],
                "mean_body_linear_velocity_mps": [
                    float(value) for value in mean_body_linear[index].tolist()
                ],
                "mean_command_frame_linear_velocity_mps": [
                    float(value) for value in mean_command_frame_linear[index].tolist()
                ],
                "mean_world_linear_velocity_mps": [
                    float(value) for value in mean_world_linear[index].tolist()
                ],
                "mean_body_angular_velocity_radps": [
                    float(value) for value in mean_body_angular[index].tolist()
                ],
                "mean_command_frame_angular_velocity_radps": [
                    float(value) for value in mean_command_frame_angular[index].tolist()
                ],
                "mean_command_error": [float(value) for value in mean_error[index].tolist()],
                "mean_abs_command_error": [
                    float(value) for value in mean_abs_error[index].tolist()
                ],
                "rmse_command_error": [float(value) for value in rmse_error[index].tolist()],
                "planar_velocity_rmse_mps": float(
                    torch.linalg.norm(rmse_error[index, :2]).item()
                ),
                "yaw_rate_rmse_radps": float(rmse_error[index, 2].item()),
                "falls": int(self.falls[index].item()),
                "timeouts": int(self.timeouts[index].item()),
                "fall_free": bool(self.falls[index].item() == 0),
                "mean_base_height_m": float(mean_base_height[index].item()),
                "base_height_std_m": float(base_height_std[index].item()),
                "base_height_peak_to_peak_m": float(
                    base_height_peak_to_peak[index].item()
                ),
                "minimum_base_height_m": float(self.min_base_height[index].item()),
                "maximum_base_height_m": float(self.max_base_height[index].item()),
                "vertical_velocity_rms_mps": float(
                    vertical_velocity_rms[index].item()
                ),
                "vertical_velocity_abs_p95_mps": float(
                    torch.quantile(stability_trace[:, 0], 0.95).item()
                ),
                "roll_pitch_angular_velocity_rms_radps": float(
                    roll_pitch_angular_velocity_rms[index].item()
                ),
                "roll_pitch_angular_velocity_p95_radps": float(
                    torch.quantile(stability_trace[:, 1], 0.95).item()
                ),
                "tilt_rms_degrees": float(
                    torch.rad2deg(tilt_rms_radians[index]).item()
                ),
                "tilt_p95_degrees": float(
                    torch.rad2deg(torch.quantile(stability_trace[:, 2], 0.95)).item()
                ),
                "maximum_tilt_degrees": float(
                    torch.rad2deg(self.max_tilt_radians[index]).item()
                ),
                "mean_reward_per_step": float((self.sum_reward[index] / count[index]).item()),
                "torque": {
                    "rated_continuous_nm": float(self.raw_env.cfg.rated_torque_nm),
                    "mean_abs_applied_nm": float(
                        (self.sum_abs_applied_torque[index] / torque_count[index]).item()
                    ),
                    "rms_applied_nm": float(
                        torch.sqrt(self.sum_squared_applied_torque[index] / torque_count[index]).item()
                    ),
                    "peak_abs_applied_nm": float(self.peak_abs_applied_torque[index].item()),
                    "peak_abs_computed_nm": float(self.peak_abs_computed_torque[index].item()),
                    "computed_abs_p99_nm": float(
                        torch.quantile(computed_abs_flat, 0.99).item()
                    ),
                    "computed_abs_p999_nm": float(
                        torch.quantile(computed_abs_flat, 0.999).item()
                    ),
                    "max_per_joint_rms_applied_nm": float(
                        torch.amax(per_joint_rms_applied[index]).item()
                    ),
                    "maximum_computed_over_rating_burst_s": float(
                        torch.amax(self.max_computed_over_run[index]).item() * policy_dt
                    ),
                    "applied_at_rating_fraction": float(
                        (self.applied_saturation_count[index] / torque_count[index]).item()
                    ),
                    "computed_demand_over_rating_fraction": float(
                        (self.computed_over_rating_count[index] / torque_count[index]).item()
                    ),
                    "mean_total_abs_mechanical_power_w": float(
                        (self.sum_abs_mechanical_power[index] / count[index]).item()
                    ),
                    "per_joint": per_joint_torque,
                },
            }
            rows.append(row)

        return {
            "checkpoint": checkpoint,
            "task": task_id,
            "command_frame": self.raw_env._command_frame,
            "seed": int(self.raw_env.cfg.seed),
            "deterministic_policy": True,
            "policy_step_seconds": policy_dt,
            "requested_steps": total_steps,
            "warmup_steps": self.warmup_steps,
            "commands_evaluated_in_parallel": self.num_envs,
            "results": rows,
        }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"checkpoint={report['checkpoint']}")
    print(
        f"task={report['task']} policy_dt_s={report['policy_step_seconds']:.6f} "
        f"steps={report['requested_steps']} warmup_steps={report['warmup_steps']}"
    )
    for row in report["results"]:
        command = row["command"]
        torque = row["torque"]
        print(
            f"command[{row['index']}]="
            f"({command['body_vx_mps']:.3f},{command['body_vy_mps']:.3f},"
            f"{command['yaw_rate_radps']:.3f}) "
            f"command_frame={command['frame']} "
            f"task_v=({row['mean_command_frame_linear_velocity_mps'][0]:.3f},"
            f"{row['mean_command_frame_linear_velocity_mps'][1]:.3f}) "
            f"world_displacement=({row['displacement_world_m'][0]:.3f},"
            f"{row['displacement_world_m'][1]:.3f})m "
            f"planar_rmse={row['planar_velocity_rmse_mps']:.3f}m/s "
            f"falls={row['falls']} timeouts={row['timeouts']} "
            f"torque_mean={torque['mean_abs_applied_nm']:.3f}Nm "
            f"torque_peak={torque['peak_abs_applied_nm']:.3f}Nm "
            f"at_rating={100.0 * torque['applied_at_rating_fraction']:.2f}% "
            f"demand_over_rating={100.0 * torque['computed_demand_over_rating_fraction']:.2f}%"
        )
    print("EVALUATION_JSON_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("EVALUATION_JSON_END")


def main() -> int:
    args = _parse_args()
    register_envs()

    base_command_values = args.command or [[0.30, 0.0, 0.0]]
    command_values = [
        command
        for command in base_command_values
        for _ in range(args.copies)
    ]
    if args.follow_camera and args.follow_env_index >= len(command_values):
        raise ValueError(
            f"--follow-env-index ({args.follow_env_index}) must be smaller than the "
            f"number of --command environments ({len(command_values)})"
        )
    checkpoints = [retrieve_file_path(value) for value in args.checkpoint]
    if args.video and len(checkpoints) != 1:
        raise ValueError("Video capture accepts exactly one --checkpoint")
    checkpoint = checkpoints[0]

    env_cfg_type, agent_cfg_type = TASK_CONFIGS[args.task]
    env_cfg = env_cfg_type()
    agent_cfg = agent_cfg_type()
    action_processing = _apply_processed_joint_target_slew_override(
        env_cfg,
        args.processed_joint_target_slew_limit_rad_per_20ms,
    )
    stance_override = _apply_stance_override(env_cfg, args)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = len(command_values)
    if args.env_spacing is not None:
        env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    if args.video_resolution is not None:
        video_resolution = tuple(args.video_resolution)
        env_cfg.viewer.resolution = video_resolution
        if env_cfg.video_recorder is None:
            raise RuntimeError("The selected task has no video recorder configuration")
        env_cfg.video_recorder.window_width = video_resolution[0]
        env_cfg.video_recorder.window_height = video_resolution[1]
    env_cfg.log_dir = os.path.dirname(checkpoint)
    if args.disable_randomization:
        env_cfg.events = None
    startup_randomization_enabled = env_cfg.events is not None
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    policy_dt = float(env_cfg.decimation * env_cfg.sim.dt)
    total_steps = args.steps if args.steps is not None else int(math.ceil(args.seconds / policy_dt))
    if args.warmup_steps >= total_steps:
        raise ValueError(
            f"--warmup-steps ({args.warmup_steps}) must be smaller than evaluation steps ({total_steps})"
        )

    if args.video:
        args.enable_cameras = True
    video_dir = args.video_dir or os.path.join(os.path.dirname(checkpoint), "videos", "eval")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    report: dict[str, Any]

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video and (
            args.follow_camera or args.camera_horizontal_fov_deg is not None
        ):
            # Initialize the recorder's render product before moving its camera.
            # Its first render otherwise reapplies the static ViewerCfg pose and
            # would override the first follow-camera update.
            # Bypass Gym's OrderEnforcing wrapper here because RSL-RL performs
            # the public reset a few lines later in its wrapper constructor.
            if gym_env.unwrapped.render() is None:
                raise RuntimeError("The Isaac Lab RGB video backend did not return a frame")
        if args.video and args.camera_horizontal_fov_deg is not None:
            if env_cfg.video_recorder is None:
                raise RuntimeError("The selected task has no video recorder configuration")
            resolution = (
                tuple(args.video_resolution)
                if args.video_resolution is not None
                else (
                    env_cfg.video_recorder.window_width,
                    env_cfg.video_recorder.window_height,
                )
            )
            _set_recording_camera_intrinsics(
                env_cfg.viewer.cam_prim_path,
                args.camera_horizontal_fov_deg,
                resolution,
            )
        if args.video and args.follow_camera:
            gym_env = SmoothFollowCameraWrapper(
                gym_env,
                env_index=args.follow_env_index,
                eye=tuple(args.camera_eye),
                lookat=tuple(args.camera_lookat),
                policy_dt=policy_dt,
                time_constant=args.camera_follow_time_constant,
            )
        if args.video:
            gym_env = gym.wrappers.RecordVideo(
                gym_env,
                video_folder=video_dir,
                step_trigger=lambda step: step == 0,
                video_length=total_steps,
                disable_logger=True,
            )

        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            raw_env = env.unwrapped
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            configure_seed(args.seed, True)
            commands = torch.tensor(command_values, dtype=torch.float32, device=raw_env.device)
            reports: list[dict[str, Any]] = []

            for checkpoint in checkpoints:
                runner.load(checkpoint)
                policy = runner.get_inference_policy(device=raw_env.device)

                # Reinitialize all physics and task buffers between policies while
                # keeping Kit/PhysX alive. Reseed immediately before every reset:
                # reset joint perturbations and any other startup draws must be
                # identical across policies for command-local seed comparisons.
                configure_seed(args.seed, True)
                with torch.inference_mode():
                    observations, _ = env.reset()
                    raw_env.episode_length_buf.zero_()
                _force_commands(raw_env, observations, commands)

                accumulator = MetricAccumulator(raw_env, args.warmup_steps)
                original_get_rewards = raw_env._get_rewards

                def measured_get_rewards() -> torch.Tensor:
                    reward = original_get_rewards()
                    accumulator.capture_pre_reset(reward)
                    return reward

                raw_env._get_rewards = measured_get_rewards
                try:
                    for step in range(total_steps):
                        accumulator.step_index = step
                        if step == args.warmup_steps:
                            accumulator.begin_displacement_measurement()
                        with torch.inference_mode():
                            actions = policy(observations, stochastic_output=False)
                            observations, _, dones, _ = env.step(actions)
                            policy.reset(dones)

                        accumulator.account_for_resets(dones)
                        reset_mask = dones.to(dtype=torch.bool)
                        if torch.any(reset_mask):
                            # HexapodEnv randomizes full-reset episode ages for PPO training.
                            # Evaluation episodes always restart at age zero.
                            raw_env.episode_length_buf[reset_mask] = 0
                        _force_commands(raw_env, observations, commands)
                finally:
                    raw_env._get_rewards = original_get_rewards

                reports.append(
                    accumulator.report(commands, checkpoint, args.task, policy_dt, total_steps)
                )
                reports[-1]["reset_stance_override"] = stance_override
                # Keep the playback provenance machine-readable.  In
                # particular, robust screens must not be able to mistake a
                # deterministic, randomization-disabled replay for a
                # randomized startup trial.
                reports[-1]["startup_randomization_enabled"] = (
                    startup_randomization_enabled
                )
                reports[-1]["action_processing"] = dict(action_processing)
        finally:
            env.close()

        # AppLauncher closes Kit while leaving this context.  Emit and flush the
        # report before that shutdown because some Kit builds do not return to
        # Python after ``app.close()``.
        for report in reports:
            _print_summary(report)
        if args.json_path:
            json_path = Path(args.json_path).expanduser().resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_payload: dict[str, Any] = reports[0] if len(reports) == 1 else {
                "action_processing": dict(action_processing),
                "evaluations": reports,
            }
            json_path.write_text(
                json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"json_report={json_path}")
        if args.video:
            print(f"video={Path(video_dir).expanduser().resolve() / 'rl-video-step-0.mp4'}")
        sys.stdout.flush()

        if args.fail_on_fall and any(
            row["falls"] > 0 for report in reports for row in report["results"]
        ):
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
