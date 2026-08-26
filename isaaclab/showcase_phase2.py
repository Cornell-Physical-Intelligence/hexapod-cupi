#!/usr/bin/env python3
"""Record and score a deterministic timed Phase 2 joystick-policy showcase.

The policy receives one anatomical-navigation command at a time: stand,
forward, left/right strafe, left/right yaw, diagonal, and backward.  The output
JSON keeps independent steady-state metrics and acceptance decisions for every
segment while the optional video shows the complete command transitions.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation

from evaluate_checkpoint import (
    MetricAccumulator,
    SmoothFollowCameraWrapper,
    TASK_CONFIGS,
    _force_commands,
    _set_recording_camera_intrinsics,
)
from hexapod_rl.register import (
    PHASE2_FINAL_TASK_ID,
    PHASE2_WARMUP_TASK_ID,
    register_envs,
)
from hexapod_rl.showcase_sequence import (
    ScheduledCommandSegment,
    acceptance_thresholds,
    annotate_showcase_frame,
    default_showcase_segments,
    evaluate_segment_acceptance,
    schedule_segments,
)


PHASE2_TASK_IDS = (PHASE2_WARMUP_TASK_ID, PHASE2_FINAL_TASK_ID)


class CommandOverlayWrapper(gym.Wrapper):
    """Add the active navigation command to RGB frames without changing physics."""

    def __init__(self, env: gym.Env, schedule: tuple[ScheduledCommandSegment, ...]):
        super().__init__(env)
        self._schedule = schedule
        self._active = schedule[0]
        self._progress = 0.0

    def set_segment(self, segment: ScheduledCommandSegment, progress: float) -> None:
        self._active = segment
        self._progress = float(progress)

    def render(self) -> Any:
        frame = self.env.render()
        if frame is None:
            return None
        return annotate_showcase_frame(
            frame,
            label=self._active.segment.label,
            command=self._active.segment.command,
            segment_index=self._active.index,
            segment_count=len(self._schedule),
            progress=self._progress,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record and score a fixed Phase 2 navigation-command sequence using "
            "one deterministic RSL-RL checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Accepted model_*.pt path.")
    parser.add_argument(
        "--task",
        choices=PHASE2_TASK_IDS,
        default=PHASE2_WARMUP_TASK_ID,
        help="Phase 2 task/config used to interpret the checkpoint.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Independent copies receiving the same timed command. Default: 1.",
    )
    parser.add_argument(
        "--env-spacing", type=float, help="Override cloned-environment spacing in metres."
    )
    parser.add_argument(
        "--stand-seconds", type=float, default=2.0, help="Stand-segment duration. Default: 2 s."
    )
    parser.add_argument(
        "--motion-seconds",
        type=float,
        default=3.0,
        help="Duration of each movement segment. Default: 3 s.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Start of each segment excluded from that segment's metrics. Default: 0.5 s.",
    )
    parser.add_argument("--seed", type=int, default=54, help="Evaluation seed. Default: 54.")
    parser.add_argument(
        "--disable-randomization",
        action="store_true",
        help="Disable startup friction and base-mass randomization for nominal acceptance.",
    )
    parser.add_argument("--video", action="store_true", help="Record the complete sequence.")
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Record clean simulator frames without command labels or arrows.",
    )
    parser.add_argument(
        "--video-dir",
        help="Output directory. Defaults to <checkpoint-dir>/videos/phase2_showcase.",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Recording resolution in pixels.",
    )
    parser.add_argument(
        "--follow-camera",
        action="store_true",
        help="Smoothly translate the recording camera with one robot.",
    )
    parser.add_argument(
        "--follow-env-index",
        type=int,
        default=0,
        help="Copy to follow when --follow-camera is active. Default: 0.",
    )
    parser.add_argument(
        "--camera-follow-time-constant",
        type=float,
        default=0.20,
        metavar="SECONDS",
        help="Follow-camera X/Y smoothing constant. Default: 0.20 s.",
    )
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(0.90, 1.05, 0.58),
        metavar=("X", "Y", "Z"),
        help="Static eye, or X/Y follow offsets plus fixed world Z.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.15),
        metavar=("X", "Y", "Z"),
        help="Static target, or X/Y follow offsets plus fixed world Z.",
    )
    parser.add_argument(
        "--camera-horizontal-fov-deg",
        type=float,
        metavar="DEGREES",
        help="Pin recording-camera horizontal field of view.",
    )
    parser.add_argument("--json", dest="json_path", help="Write the JSON report here.")
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Exit with status 2 unless every copy passes every segment gate.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    if args.copies < 1:
        parser.error("--copies must be at least 1")
    if args.env_spacing is not None and args.env_spacing <= 0.0:
        parser.error("--env-spacing must be positive")
    if not math.isfinite(args.stand_seconds) or args.stand_seconds <= 0.0:
        parser.error("--stand-seconds must be finite and positive")
    if not math.isfinite(args.motion_seconds) or args.motion_seconds <= 0.0:
        parser.error("--motion-seconds must be finite and positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0.0:
        parser.error("--settle-seconds must be finite and non-negative")
    if args.video_dir and not args.video:
        parser.error("--video-dir requires --video")
    if args.video_resolution and not args.video:
        parser.error("--video-resolution requires --video")
    if args.video_resolution and any(value <= 0 for value in args.video_resolution):
        parser.error("--video-resolution values must be positive")
    if args.follow_camera and not args.video:
        parser.error("--follow-camera requires --video")
    if args.follow_env_index < 0 or args.follow_env_index >= args.copies:
        parser.error("--follow-env-index must select one of the requested copies")
    if args.camera_follow_time_constant < 0.0:
        parser.error("--camera-follow-time-constant cannot be negative")
    if args.camera_horizontal_fov_deg is not None and not args.video:
        parser.error("--camera-horizontal-fov-deg requires --video")
    if args.camera_horizontal_fov_deg is not None and not (
        0.0 < args.camera_horizontal_fov_deg < 179.0
    ):
        parser.error("--camera-horizontal-fov-deg must be between 0 and 179 degrees")
    return args


def _navigation_command_tensor(
    raw_env: Any, segment: ScheduledCommandSegment
) -> torch.Tensor:
    return torch.tensor(
        [segment.segment.command] * raw_env.num_envs,
        dtype=torch.float32,
        device=raw_env.device,
    )


def _segment_payload(
    *,
    accumulator: MetricAccumulator,
    scheduled: ScheduledCommandSegment,
    commands: torch.Tensor,
    checkpoint: str,
    task_id: str,
    policy_dt_s: float,
) -> dict[str, Any]:
    raw_report = accumulator.report(
        commands, checkpoint, task_id, policy_dt_s, scheduled.steps
    )
    rows = raw_report["results"]
    accepted = True
    for row in rows:
        # The shared evaluator retains legacy body_* JSON keys. This showcase is
        # navigation-only, so expose unambiguous field names in its independent schema.
        command = row["command"]
        row["command"] = {
            "frame": "navigation",
            "vx_mps": command["body_vx_mps"],
            "vy_mps": command["body_vy_mps"],
            "yaw_rate_radps": command["yaw_rate_radps"],
        }
        row["acceptance"] = evaluate_segment_acceptance(
            row, scheduled.segment.command
        )
        accepted = accepted and bool(row["acceptance"]["accepted"])

    return {
        "index": scheduled.index,
        "key": scheduled.segment.key,
        "label": scheduled.segment.label,
        "command": {
            "frame": "navigation",
            "vx_mps": scheduled.segment.command[0],
            "vy_mps": scheduled.segment.command[1],
            "yaw_rate_radps": scheduled.segment.command[2],
        },
        "start_s": scheduled.start_s,
        "stop_s": scheduled.stop_s,
        "duration_s": scheduled.steps * policy_dt_s,
        "steps": scheduled.steps,
        "settle_steps_excluded": scheduled.settle_steps,
        "settle_seconds_excluded": scheduled.settle_steps * policy_dt_s,
        "measured_steps": scheduled.measured_steps,
        "accepted": accepted,
        "results": rows,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"checkpoint={report['checkpoint']}")
    print(
        f"task={report['task']} command_frame=navigation "
        f"duration_s={report['rendered_duration_s']:.2f} "
        f"copies={report['copies']} accepted={report['accepted']}"
    )
    for segment in report["segments"]:
        for row in segment["results"]:
            velocity = row["mean_command_frame_linear_velocity_mps"]
            angular = row["mean_command_frame_angular_velocity_radps"]
            acceptance = row["acceptance"]
            print(
                f"segment[{segment['index']}]={segment['key']} copy={row['index']} "
                f"nav_v=({velocity[0]:+.3f},{velocity[1]:+.3f})m/s "
                f"nav_yaw={angular[2]:+.3f}rad/s "
                f"planar_rmse={row['planar_velocity_rmse_mps']:.3f}m/s "
                f"yaw_rmse={row['yaw_rate_rmse_radps']:.3f}rad/s "
                f"falls={row['falls']} accepted={acceptance['accepted']}"
            )
            for reason in acceptance["rejection_reasons"]:
                print(f"  reject: {reason}")
    print("PHASE2_SHOWCASE_JSON_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PHASE2_SHOWCASE_JSON_END")


def main() -> int:
    args = _parse_args()
    register_envs()
    checkpoint = retrieve_file_path(args.checkpoint)

    env_cfg_type, agent_cfg_type = TASK_CONFIGS[args.task]
    env_cfg = env_cfg_type()
    agent_cfg = agent_cfg_type()
    if env_cfg.command_frame != "navigation":
        raise RuntimeError(
            f"Phase 2 showcase requires navigation commands, got {env_cfg.command_frame!r}"
        )
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.copies
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
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    policy_dt_s = float(env_cfg.decimation * env_cfg.sim.dt)
    schedule = schedule_segments(
        default_showcase_segments(
            stand_seconds=args.stand_seconds, motion_seconds=args.motion_seconds
        ),
        policy_dt_s=policy_dt_s,
        settle_seconds=args.settle_seconds,
    )
    total_steps = schedule[-1].stop_step
    # The stock task times out at 20 s. A full showcase is longer, so give the
    # same physical rollout one uninterrupted episode unless it genuinely falls.
    env_cfg.episode_length_s = total_steps * policy_dt_s + 5.0

    if args.video:
        args.enable_cameras = True
    video_dir = args.video_dir or os.path.join(
        os.path.dirname(checkpoint), "videos", "phase2_showcase"
    )
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    segment_reports: list[dict[str, Any]] = []

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(
            args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None
        )
        if args.video and (
            args.follow_camera or args.camera_horizontal_fov_deg is not None
        ):
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
                policy_dt=policy_dt_s,
                time_constant=args.camera_follow_time_constant,
            )

        overlay: CommandOverlayWrapper | None = None
        if args.video and not args.no_overlay:
            overlay = CommandOverlayWrapper(gym_env, schedule)
            gym_env = overlay
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
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
            configure_seed(args.seed, True)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=raw_env.device)

            with torch.inference_mode():
                observations, _ = env.reset()
                raw_env.episode_length_buf.zero_()

            active_accumulator: MetricAccumulator | None = None
            original_get_rewards = raw_env._get_rewards

            def measured_get_rewards() -> torch.Tensor:
                reward = original_get_rewards()
                if active_accumulator is None:
                    raise RuntimeError("showcase metric accumulator is not active")
                active_accumulator.capture_pre_reset(reward)
                return reward

            raw_env._get_rewards = measured_get_rewards
            try:
                for scheduled in schedule:
                    commands = _navigation_command_tensor(raw_env, scheduled)
                    _force_commands(raw_env, observations, commands)
                    active_accumulator = MetricAccumulator(
                        raw_env, scheduled.settle_steps
                    )
                    if overlay is not None:
                        overlay.set_segment(scheduled, 0.0)

                    for local_step in range(scheduled.steps):
                        active_accumulator.step_index = local_step
                        if local_step == scheduled.settle_steps:
                            active_accumulator.begin_displacement_measurement()
                        if overlay is not None:
                            overlay.set_segment(
                                scheduled, (local_step + 1) / scheduled.steps
                            )
                        with torch.inference_mode():
                            actions = policy(observations, stochastic_output=False)
                            observations, _, dones, _ = env.step(actions)
                            policy.reset(dones)

                        active_accumulator.account_for_resets(dones)
                        reset_mask = dones.to(dtype=torch.bool)
                        if torch.any(reset_mask):
                            raw_env.episode_length_buf[reset_mask] = 0
                        _force_commands(raw_env, observations, commands)

                    segment_reports.append(
                        _segment_payload(
                            accumulator=active_accumulator,
                            scheduled=scheduled,
                            commands=commands,
                            checkpoint=checkpoint,
                            task_id=args.task,
                            policy_dt_s=policy_dt_s,
                        )
                    )
            finally:
                raw_env._get_rewards = original_get_rewards
        finally:
            env.close()

        accepted = all(segment["accepted"] for segment in segment_reports)
        report = {
            "schema": "hexapod.phase2_joystick_showcase.v1",
            "checkpoint": checkpoint,
            "task": args.task,
            "command_frame": "navigation",
            "seed": args.seed,
            "deterministic_policy": True,
            "startup_randomization_enabled": not args.disable_randomization,
            "copies": args.copies,
            "policy_step_seconds": policy_dt_s,
            "rendered_steps": total_steps,
            "rendered_duration_s": total_steps * policy_dt_s,
            "acceptance_thresholds": acceptance_thresholds(),
            "accepted": accepted,
            "segments": segment_reports,
        }
        _print_summary(report)
        if args.json_path:
            json_path = Path(args.json_path).expanduser().resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"json_report={json_path}")
        if args.video:
            print(
                f"video={Path(video_dir).expanduser().resolve() / 'rl-video-step-0.mp4'}"
            )
        sys.stdout.flush()

        if args.fail_on_rejection and not accepted:
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
