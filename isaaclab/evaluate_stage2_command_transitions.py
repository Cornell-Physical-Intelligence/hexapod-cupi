#!/usr/bin/env python3
"""Evaluate one policy through an uninterrupted Stage2 joystick schedule.

Unlike the fixed-command batch evaluator, this runner changes the navigation
command in-place while preserving policy, actuator-filter, and simulator state.
Each segment reports a one-second transient window and a disjoint two-second
steady window for every randomized environment copy.
"""

from __future__ import annotations

import argparse
import hashlib
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

from evaluate_checkpoint import MetricAccumulator, TASK_CONFIGS, _force_commands
from hexapod_rl.asset_cfg import FEMUR_JOINTS, TIBIA_JOINTS
from hexapod_rl.register import PHASE2_RECOVERY_STAGE2E_E2_TASK_ID, register_envs
from stage2_command_transition_contract import (
    COPIES,
    EVALUATION_SEED,
    POLICY_STEP_SECONDS,
    SCHEMA_ID,
    SCHEDULE_SHA256,
    SEGMENT_STEPS,
    SETTLING_DWELL_STEPS,
    STEADY_STEPS,
    TOTAL_DURATION_SECONDS,
    TOTAL_STEPS,
    TRANSIENT_STEPS,
    schedule_payload,
    scheduled_segments,
    settling_completion_step,
    tracking_within_settling_band,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpoint through the audited continuous Stage2 "
            "joystick-command transition schedule."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Exact model_*.pt path.")
    parser.add_argument(
        "--task",
        required=True,
        help="Terminal Stage2E E2 navigation task ID.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=COPIES,
        help=f"Parallel randomized copies. Audited contract: {COPIES}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EVALUATION_SEED,
        help=f"Startup-randomization seed. Audited contract: {EVALUATION_SEED}.",
    )
    parser.add_argument(
        "--disable-randomization",
        action="store_true",
        help="Diagnostic only; audited transition grading requires randomization.",
    )
    parser.add_argument("--json", dest="json_path", required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.copies < 1:
        parser.error("--copies must be at least 1")
    return args


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _command_tensor(raw_env: Any, command: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(
        [command] * raw_env.num_envs,
        dtype=torch.float32,
        device=raw_env.device,
    )


class TransientTraceAccumulator:
    """Capture transient peaks and a dwell-based tracking-settling time."""

    def __init__(
        self,
        raw_env: Any,
        command: tuple[float, float, float],
    ) -> None:
        self.raw_env = raw_env
        self.command = command
        self.num_envs = raw_env.num_envs
        self.linear_samples: list[torch.Tensor] = []
        self.angular_samples: list[torch.Tensor] = []
        self.base_height_samples: list[torch.Tensor] = []
        self.vertical_velocity_samples: list[torch.Tensor] = []
        self.roll_pitch_rate_samples: list[torch.Tensor] = []
        self.tilt_samples: list[torch.Tensor] = []
        self.initial_base_height: torch.Tensor | None = None

    def begin_before_first_action(self) -> None:
        self.initial_base_height = (
            self.raw_env._robot.data.root_pos_w.torch[:, 2].detach().clone()
        )

    def capture_pre_reset(self) -> None:
        robot_data = self.raw_env._robot.data
        body_linear = robot_data.root_lin_vel_b.torch
        body_angular = robot_data.root_ang_vel_b.torch
        command_linear = self.raw_env._vector_in_command_frame(body_linear)
        command_angular = self.raw_env._vector_in_command_frame(body_angular)
        upright_cosine = torch.clamp(
            -robot_data.projected_gravity_b.torch[:, 2], -1.0, 1.0
        )
        self.linear_samples.append(command_linear.detach().clone())
        self.angular_samples.append(command_angular.detach().clone())
        self.base_height_samples.append(
            robot_data.root_pos_w.torch[:, 2].detach().clone()
        )
        self.vertical_velocity_samples.append(
            torch.abs(robot_data.root_lin_vel_w.torch[:, 2]).detach().clone()
        )
        self.roll_pitch_rate_samples.append(
            torch.linalg.norm(body_angular[:, :2], dim=1).detach().clone()
        )
        self.tilt_samples.append(torch.acos(upright_cosine).detach().clone())

    def report(self, policy_dt_s: float) -> list[dict[str, Any]]:
        if self.initial_base_height is None:
            raise RuntimeError("transient baseline was not captured")
        if len(self.linear_samples) != TRANSIENT_STEPS:
            raise RuntimeError(
                "transient trace must contain exactly "
                f"{TRANSIENT_STEPS} samples, got {len(self.linear_samples)}"
            )
        linear = torch.stack(self.linear_samples).cpu()
        angular = torch.stack(self.angular_samples).cpu()
        heights = torch.stack(
            [self.initial_base_height, *self.base_height_samples]
        ).cpu()
        vertical = torch.stack(self.vertical_velocity_samples).cpu()
        roll_pitch = torch.stack(self.roll_pitch_rate_samples).cpu()
        tilt = torch.rad2deg(torch.stack(self.tilt_samples)).cpu()
        command_tensor = torch.tensor(self.command, dtype=linear.dtype)

        rows: list[dict[str, Any]] = []
        for copy_index in range(self.num_envs):
            settled_flags = [
                tracking_within_settling_band(
                    self.command,
                    tuple(float(value) for value in linear[step, copy_index].tolist()),
                    tuple(float(value) for value in angular[step, copy_index].tolist()),
                )
                for step in range(TRANSIENT_STEPS)
            ]
            completion_step = settling_completion_step(
                settled_flags, dwell_steps=SETTLING_DWELL_STEPS
            )
            planar_error = torch.linalg.norm(
                linear[:, copy_index, :2] - command_tensor[:2], dim=1
            )
            x_error = torch.abs(
                linear[:, copy_index, 0] - command_tensor[0]
            )
            y_error = torch.abs(
                linear[:, copy_index, 1] - command_tensor[1]
            )
            yaw_error = torch.abs(
                angular[:, copy_index, 2] - command_tensor[2]
            )
            rows.append(
                {
                    "copy_index": copy_index,
                    "samples": TRANSIENT_STEPS,
                    "measured_seconds": TRANSIENT_STEPS * policy_dt_s,
                    "baseline_captured_before_first_action": True,
                    "settling_dwell_steps": SETTLING_DWELL_STEPS,
                    "settled_sample_count": sum(settled_flags),
                    "settling_completed": completion_step is not None,
                    "settling_completion_step": completion_step,
                    "settling_time_s": (
                        None
                        if completion_step is None
                        else completion_step * policy_dt_s
                    ),
                    # Preserve the full finite tracking trace so the grader can
                    # independently recompute signed dwell and axis-local
                    # transient error instead of trusting summary claims.
                    "command_frame_linear_velocity_mps_samples": [
                        [float(value) for value in sample.tolist()]
                        for sample in linear[:, copy_index, :]
                    ],
                    "command_frame_yaw_rate_radps_samples": [
                        float(value)
                        for value in angular[:, copy_index, 2].tolist()
                    ],
                    "maximum_planar_tracking_error_mps": float(
                        torch.amax(planar_error).item()
                    ),
                    "maximum_x_tracking_error_mps": float(
                        torch.amax(x_error).item()
                    ),
                    "maximum_y_tracking_error_mps": float(
                        torch.amax(y_error).item()
                    ),
                    "maximum_yaw_tracking_error_radps": float(
                        torch.amax(yaw_error).item()
                    ),
                    "base_height_peak_to_peak_m": float(
                        (torch.amax(heights[:, copy_index]) - torch.amin(heights[:, copy_index])).item()
                    ),
                    "maximum_abs_vertical_velocity_mps": float(
                        torch.amax(vertical[:, copy_index]).item()
                    ),
                    "maximum_roll_pitch_angular_velocity_radps": float(
                        torch.amax(roll_pitch[:, copy_index]).item()
                    ),
                    "maximum_tilt_degrees": float(
                        torch.amax(tilt[:, copy_index]).item()
                    ),
                }
            )
        return rows


def _segment_payload(
    *,
    scheduled: Any,
    command_tensor: torch.Tensor,
    transient_accumulator: MetricAccumulator,
    steady_accumulator: MetricAccumulator,
    trace_accumulator: TransientTraceAccumulator,
    checkpoint: str,
    task_id: str,
    policy_dt_s: float,
) -> dict[str, Any]:
    transient_rows = transient_accumulator.report(
        command_tensor, checkpoint, task_id, policy_dt_s, TRANSIENT_STEPS
    )["results"]
    steady_rows = steady_accumulator.report(
        command_tensor, checkpoint, task_id, policy_dt_s, SEGMENT_STEPS
    )["results"]
    trace_rows = trace_accumulator.report(policy_dt_s)
    result_count = len(transient_rows)
    if result_count < 1 or not (
        len(steady_rows) == result_count and len(trace_rows) == result_count
    ):
        raise RuntimeError("transition segment result counts are inconsistent")
    results = [
        {
            "copy_index": copy_index,
            "transient_trace": trace_rows[copy_index],
            "transient_metrics": transient_rows[copy_index],
            "steady_state_metrics": steady_rows[copy_index],
        }
        for copy_index in range(result_count)
    ]
    expected = schedule_payload()[scheduled.index]
    return {**expected, "results": results}


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"checkpoint={report['checkpoint']} task={report['task']} "
        f"schedule_sha256={report['schedule_sha256']} copies={report['copies']}"
    )
    for segment in report["segments"]:
        for result in segment["results"]:
            trace = result["transient_trace"]
            steady = result["steady_state_metrics"]
            print(
                f"segment[{segment['index']}]={segment['key']} "
                f"copy={result['copy_index']} settled={trace['settling_completed']} "
                f"settling_s={trace['settling_time_s']} "
                f"steady_rmse={steady['planar_velocity_rmse_mps']:.3f} "
                f"tilt_p95={steady['tilt_p95_degrees']:.3f}deg "
                f"falls={steady['falls'] + result['transient_metrics']['falls']}"
            )


def main() -> int:
    args = _parse_args()
    register_envs()
    if args.task != PHASE2_RECOVERY_STAGE2E_E2_TASK_ID:
        raise ValueError(
            "audited command-transition admission requires the terminal Stage2E "
            f"E2 task {PHASE2_RECOVERY_STAGE2E_E2_TASK_ID!r}"
        )
    if args.task not in TASK_CONFIGS:
        raise ValueError(f"task is not mapped by evaluate_checkpoint.py: {args.task!r}")
    checkpoint = retrieve_file_path(args.checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    env_cfg_type, agent_cfg_type = TASK_CONFIGS[args.task]
    env_cfg = env_cfg_type()
    agent_cfg = agent_cfg_type()
    if env_cfg.command_frame != "navigation":
        raise RuntimeError(
            "command-transition evaluation requires navigation frame, got "
            f"{env_cfg.command_frame!r}"
        )
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.copies
    env_cfg.log_dir = os.path.dirname(checkpoint)
    configured_events = env_cfg.events
    if args.disable_randomization:
        env_cfg.events = None
    startup_randomization_enabled = env_cfg.events is not None
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    policy_dt_s = float(env_cfg.decimation * env_cfg.sim.dt)
    if not math.isclose(policy_dt_s, POLICY_STEP_SECONDS, abs_tol=1.0e-9):
        raise RuntimeError(
            f"transition contract requires {POLICY_STEP_SECONDS}s policy steps, "
            f"got {policy_dt_s}"
        )
    env_cfg.episode_length_s = TOTAL_DURATION_SECONDS + 5.0
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )
    segment_reports: list[dict[str, Any]] = []
    rollout_results: list[dict[str, Any]] = []
    runtime_joint_names_sorted: list[str] = []

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            raw_env = env.unwrapped
            runtime_joint_names_sorted = sorted(
                str(name) for name in raw_env._robot.joint_names
            )
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
            configure_seed(args.seed, True)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=raw_env.device)
            with torch.inference_mode():
                observations, _ = env.reset()
                raw_env.episode_length_buf.zero_()

            active_steady: MetricAccumulator | None = None
            active_transient: MetricAccumulator | None = None
            active_trace: TransientTraceAccumulator | None = None
            active_local_step = -1
            rollout_accumulator = MetricAccumulator(raw_env, 0)
            rollout_accumulator.begin_displacement_measurement()
            original_get_rewards = raw_env._get_rewards

            def measured_get_rewards() -> torch.Tensor:
                reward = original_get_rewards()
                if active_steady is None or active_transient is None or active_trace is None:
                    raise RuntimeError("transition accumulators are not active")
                rollout_accumulator.capture_pre_reset(reward)
                active_steady.capture_pre_reset(reward)
                if active_local_step < TRANSIENT_STEPS:
                    active_transient.capture_pre_reset(reward)
                    active_trace.capture_pre_reset()
                return reward

            raw_env._get_rewards = measured_get_rewards
            try:
                for scheduled in scheduled_segments():
                    command = scheduled.segment.command
                    commands = _command_tensor(raw_env, command)
                    # This updates both the environment command and the already
                    # built observation before the first action of the segment.
                    # No environment or policy reset occurs at this boundary.
                    _force_commands(raw_env, observations, commands)
                    active_steady = MetricAccumulator(raw_env, TRANSIENT_STEPS)
                    active_transient = MetricAccumulator(raw_env, 0)
                    active_trace = TransientTraceAccumulator(raw_env, command)
                    active_transient.begin_displacement_measurement()
                    active_trace.begin_before_first_action()

                    for local_step in range(SEGMENT_STEPS):
                        active_local_step = local_step
                        rollout_accumulator.step_index = scheduled.start_step + local_step
                        active_steady.step_index = local_step
                        active_transient.step_index = local_step
                        if local_step == TRANSIENT_STEPS:
                            active_steady.begin_displacement_measurement()
                        with torch.inference_mode():
                            actions = policy(observations, stochastic_output=False)
                            observations, _, dones, _ = env.step(actions)
                            # Recurrent state is reset only for a genuine
                            # simulator termination, never for a command change.
                            policy.reset(dones)
                        rollout_accumulator.account_for_resets(dones)
                        active_steady.account_for_resets(dones)
                        if local_step < TRANSIENT_STEPS:
                            active_transient.account_for_resets(dones)
                        reset_mask = dones.to(dtype=torch.bool)
                        if torch.any(reset_mask):
                            raw_env.episode_length_buf[reset_mask] = 0
                        # Full resets can rebuild commands/timers. Reapply the
                        # active schedule command before the next policy action.
                        _force_commands(raw_env, observations, commands)

                    segment_reports.append(
                        _segment_payload(
                            scheduled=scheduled,
                            command_tensor=commands,
                            transient_accumulator=active_transient,
                            steady_accumulator=active_steady,
                            trace_accumulator=active_trace,
                            checkpoint=checkpoint,
                            task_id=args.task,
                            policy_dt_s=policy_dt_s,
                        )
                    )
            finally:
                raw_env._get_rewards = original_get_rewards
            zero_commands = torch.zeros(
                (raw_env.num_envs, 3), dtype=torch.float32, device=raw_env.device
            )
            raw_rollout_results = rollout_accumulator.report(
                zero_commands,
                checkpoint,
                args.task,
                policy_dt_s,
                TOTAL_STEPS,
            )["results"]
            rollout_results = [
                {
                    "index": row["index"],
                    "samples": row["samples"],
                    "measured_seconds": row["measured_seconds"],
                    "falls": row["falls"],
                    "timeouts": row["timeouts"],
                    "fall_free": row["fall_free"],
                    "torque": row["torque"],
                }
                for row in raw_rollout_results
            ]
        finally:
            env.close()

    total_falls = sum(int(row["falls"]) for row in rollout_results)
    total_timeouts = sum(int(row["timeouts"]) for row in rollout_results)
    initial_joint_positions = env_cfg.robot.init_state.joint_pos
    command_cfg = env_cfg.velocity_command
    material_params = configured_events.physics_material.params
    base_mass_params = configured_events.base_mass.params
    actuator_cfg = env_cfg.robot.actuators["legs"]
    robot_usd_path = str(env_cfg.robot.spawn.usd_path)
    report = {
        "schema": SCHEMA_ID,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "task": args.task,
        "task_config_class": (
            f"{env_cfg_type.__module__}.{env_cfg_type.__qualname__}"
        ),
        "agent_config_class": (
            f"{agent_cfg_type.__module__}.{agent_cfg_type.__qualname__}"
        ),
        "command_frame": "navigation",
        "seed": args.seed,
        "deterministic_policy": True,
        "startup_randomization_enabled": startup_randomization_enabled,
        "copies": args.copies,
        "policy_step_seconds": policy_dt_s,
        "rendered_steps": TOTAL_STEPS,
        "rendered_duration_s": TOTAL_DURATION_SECONDS,
        "episode_length_seconds": env_cfg.episode_length_s,
        "reset_stance_override": None,
        "config_snapshot": {
            "action_scale": float(env_cfg.action_scale),
            "rated_torque_nm": float(env_cfg.rated_torque_nm),
            "nominal_height_m": float(env_cfg.nominal_height_m),
            "stand_nominal_height_m": float(env_cfg.stand_nominal_height_m),
            "moving_command_threshold_mps": float(
                env_cfg.moving_command_threshold_mps
            ),
            "axis_command_active_threshold": float(
                env_cfg.axis_command_active_threshold
            ),
            "reset_root_height_m": float(env_cfg.robot.init_state.pos[2]),
            "reset_femur_angle_rad": float(initial_joint_positions[FEMUR_JOINTS[0]]),
            "reset_tibia_angle_rad": float(initial_joint_positions[TIBIA_JOINTS[0]]),
            "processed_joint_target_slew_limit_rad_per_20ms": float(
                env_cfg.processed_joint_target_slew_limit_rad_per_20ms
            ),
            "velocity_command_sampling_mode": env_cfg.velocity_command.sampling_mode,
            "velocity_command_resampling_time_range_s": list(
                env_cfg.velocity_command.resampling_time_range_s
            ),
            "command_lin_vel_x_range_mps": list(env_cfg.command_lin_vel_x_range_mps),
            "command_lin_vel_y_range_mps": list(env_cfg.command_lin_vel_y_range_mps),
            "command_yaw_rate_range_rad_s": list(env_cfg.command_yaw_rate_range_rad_s),
            "terminate_on_computed_torque_demand_nm": float(
                env_cfg.terminate_on_computed_torque_demand_nm
            ),
            "terminate_on_computed_torque_demand_duration_s": float(
                env_cfg.terminate_on_computed_torque_demand_duration_s
            ),
            "torque_demand_termination_grace_s": float(
                env_cfg.torque_demand_termination_grace_s
            ),
            "velocity_command_bucket_counts": list(command_cfg.bucket_counts),
            "velocity_command_bucket_stride": int(command_cfg.bucket_stride),
            "velocity_command_joystick_forward_range_mps": list(
                command_cfg.joystick_forward_range
            ),
            "velocity_command_joystick_reverse_abs_range_mps": list(
                command_cfg.joystick_reverse_abs_range
            ),
            "velocity_command_joystick_lateral_abs_range_mps": list(
                command_cfg.joystick_lateral_abs_range
            ),
            "velocity_command_joystick_yaw_abs_range_rad_s": list(
                command_cfg.joystick_yaw_abs_range
            ),
            "velocity_command_yaw_anchor_forward_range_mps": list(
                command_cfg.yaw_anchor_forward_range
            ),
            "startup_static_friction_range": list(
                material_params["static_friction_range"]
            ),
            "startup_dynamic_friction_range": list(
                material_params["dynamic_friction_range"]
            ),
            "startup_restitution_range": list(
                material_params["restitution_range"]
            ),
            "startup_material_num_buckets": int(material_params["num_buckets"]),
            "startup_material_make_consistent": bool(
                material_params["make_consistent"]
            ),
            "startup_base_mass_additive_range_kg": list(
                base_mass_params["mass_distribution_params"]
            ),
            "robot_usd_path": robot_usd_path,
            "robot_usd_sha256": _sha256_file(robot_usd_path),
            "runtime_joint_names_sorted": runtime_joint_names_sorted,
            "actuator_saturation_effort_nm": float(
                actuator_cfg.saturation_effort
            ),
            "actuator_effort_limit_nm": float(actuator_cfg.effort_limit),
            "actuator_effort_limit_sim_nm": float(actuator_cfg.effort_limit_sim),
            "actuator_velocity_limit_rad_s": float(actuator_cfg.velocity_limit),
            "actuator_velocity_limit_sim_rad_s": float(
                actuator_cfg.velocity_limit_sim
            ),
            "actuator_stiffness_nm_rad": float(actuator_cfg.stiffness),
            "actuator_damping_nm_s_rad": float(actuator_cfg.damping),
            "actuator_armature_kg_m2": float(actuator_cfg.armature),
            "actuator_friction_nm": float(actuator_cfg.friction),
            "actuator_dynamic_friction_nm": float(
                actuator_cfg.dynamic_friction
            ),
            "actuator_viscous_friction_nm_s_rad": float(
                actuator_cfg.viscous_friction
            ),
        },
        "continuous_episode_requested": True,
        "continuous_episode_achieved": total_falls == 0 and total_timeouts == 0,
        "total_falls": total_falls,
        "total_timeouts": total_timeouts,
        "command_resampling_suppressed": True,
        "environment_reset_at_command_boundaries": False,
        "policy_state_reset_at_command_boundaries": False,
        "schedule_sha256": SCHEDULE_SHA256,
        "schedule": schedule_payload(),
        "segments": segment_reports,
        # This full-run accumulator prevents a motor over-rating burst that
        # crosses a transient/steady or command boundary from being split into
        # individually passing windows.
        "rollout_safety_metrics": rollout_results,
    }
    _print_summary(report)
    json_path = Path(args.json_path).expanduser().resolve()
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("x", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
    except FileExistsError:
        print(f"refusing to overwrite transition report: {json_path}", file=sys.stderr)
        return 73
    print(f"json_report={json_path}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
