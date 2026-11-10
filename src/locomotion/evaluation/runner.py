"""Native policy evaluation: record each case of the walking requirements and judge the traces.

``run_cases`` runs the cases of one requirement set, one native trial each, through ``run_batch``,
which records every 400 Hz sample and every 50 Hz control into a batch directory. The judge in
``requirements.py`` then reads the traces and writes ``judgement.json``. Every trial has one
initial reset and uninterrupted physics; a fall preserves its prefix and the next case still runs.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import locomotion.evaluation.requirements as requirements
from locomotion.evaluation.admission import verify_native_recipe
from locomotion.evaluation.capture import (
    DiagnosticGeometry,
    classify_patches,
    diagnostic_rotation,
    diagnostic_servo,
)
from locomotion.evaluation.force_metrics import write_summary
from locomotion.robots.hexapod import JOINT_NAMES, KD, MODEL_SHA256, URDF_SHA256, USD_SHA256
from locomotion.robots.rs05 import STIFFNESS_NM_PER_RAD


class _NativeTrialTermination(RuntimeError):
    """The recorded native endpoint terminated this trial without resetting it."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _cpu(value):
    return (
        value.detach().cpu().numpy().copy()
        if torch.is_tensor(value)
        else np.array(value, copy=True)
    )


def _deadline(seconds):
    if seconds is None:
        return math.inf
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Evaluation wall-clock allocation must be finite and positive")
    return time.monotonic() + seconds


class ExactEvaluationCapture:
    """Capture all eight native samples, including full unedited patch packets."""

    def __init__(self, env, output, geometry_extrema):
        if env.capture is not None:
            raise ValueError("Another owner already occupies the native capture hook")
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        with np.load(geometry_extrema, allow_pickle=False) as data:
            self.geometry = DiagnosticGeometry(
                env.geometry_meta, {k: data[k] for k in data.files}, env.native_body_names
            )
        self.counter = env.sim.get_physics_step_count()
        self.initial_counter = self.counter
        self.count = 0
        self.rows, self.last_control, self.files = [], [], []
        self.failure = None
        self.contacts = (self.output / "contacts.jsonl").open("w")
        self.maximum_applied = np.zeros(env.num_envs)
        self.maximum_requested = np.zeros(env.num_envs)
        self.joint_bound_violations = np.zeros(env.num_envs, dtype=int)
        self.speed_bound_violations = np.zeros(env.num_envs, dtype=int)
        self.nonfoot_steps = np.zeros(env.num_envs, dtype=int)
        self.minimum_clearance = np.full(env.num_envs, np.inf)
        self.minimum_plate = np.full(env.num_envs, np.inf)
        self.previous_q = _cpu(env.current["q"]).astype(float)
        self.native_limits = np.asarray(env.native_readback["limits"])
        self.native_speed = np.asarray(env.native_readback["native_max_velocity"])
        _save(
            self.output / "initial_state.json",
            {k: _cpu(v).tolist() for k, v in env.current.items()},
        )
        env.capture = self

    def __call__(
        self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre
    ):
        if substep == 0:
            self.last_control = []
        row = {
            "sequence": np.array(self.count),
            "control_index": np.array(self.count // 8),
            "substep_index": np.array(substep),
            "time_s": np.array((self.count + 1) * 0.0025),
            "explicit_counter": np.array(env.sim.get_physics_step_count()),
            "pre_joint_position_rad": _cpu(q),
            "pre_joint_velocity_rad_s": _cpu(dq),
            "joint_position_rad": _cpu(state["q"]),
            "joint_velocity_rad_s": _cpu(state["dq"]),
            "joint_target_rad": _cpu(target),
            "computed_torque_nm": _cpu(requested),
            "applied_torque_nm": _cpu(applied),
            "effort_ceiling_nm": _cpu(ceiling),
            "native_input_pre_nm": _cpu(native_pre),
            "root_pose_xyzw": _cpu(state["root"]),
            "root_com_velocity": _cpu(state["root_velocity"]),
            "link_pose_xyzw": _cpu(state["link"]),
            "command": _cpu(env.commands),
            "toe_xyz_world_m": _cpu(state["toe_world"]),
            "toe_xyz_body_m": _cpu(state["toe_body"]),
        }
        try:
            buffers = [np.array(x.numpy(), copy=True) for x in env.contact.get_contact_data(0.0025)]
            contacts = classify_patches(
                buffers, env.sensor_map, row["link_pose_xyzw"], self.geometry, env.num_envs
            )
            minimum, non_toe = self.geometry.clearance(row["link_pose_xyzw"])
            row.update(
                distal_contact=contacts["distal_contact"],
                distal_force_world_n=contacts["distal_force_world"],
                nonfoot_contact=contacts["nonfoot_contact"],
                nonfoot_force_world_n=contacts["nonfoot_force_world"],
                minimum_mesh_floor_m=minimum,
                minimum_non_toe_floor_m=non_toe,
                interval_angle_rate_rad_s=(
                    row["joint_position_rad"].astype(float) - self.previous_q
                )
                / 0.0025,
            )
            self.rows.append(row)
            self.last_control.append(row)
            self.contacts.write(
                json.dumps(
                    {
                        "sequence": self.count,
                        "explicit_counter": int(row["explicit_counter"]),
                        "patches": contacts["patches"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self.count += 1
            if int(row["explicit_counter"]) != self.counter + 1 or substep != (self.count - 1) % 8:
                raise RuntimeError("Native evaluation sample/counter continuity differs")
            self.counter += 1
            if not all(np.isfinite(v).all() for v in row.values()):
                raise FloatingPointError("Nonfinite native evaluation evidence")
            if not np.array_equal(row["native_input_pre_nm"], row["applied_torque_nm"]):
                raise RuntimeError("Actual native motor input differs")
            wanted = diagnostic_servo(
                row["pre_joint_position_rad"],
                row["pre_joint_velocity_rad_s"],
                row["joint_target_rad"],
                np.full(18, STIFFNESS_NM_PER_RAD, np.float32),
                np.asarray(KD, np.float32),
            )
            for name, expected in zip(
                ("computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm"), wanted
            ):
                if not np.array_equal(row[name], expected):
                    raise RuntimeError("Canonical motor recurrence differs: " + name)
            self.maximum_applied = np.maximum(
                self.maximum_applied, abs(row["applied_torque_nm"]).max(-1)
            )
            self.maximum_requested = np.maximum(
                self.maximum_requested, abs(row["computed_torque_nm"]).max(-1)
            )
            self.joint_bound_violations += (
                (row["joint_position_rad"] < self.native_limits[:, :, 0] - 2e-6)
                | (row["joint_position_rad"] > self.native_limits[:, :, 1] + 2e-6)
            ).any(-1)
            self.speed_bound_violations += (
                abs(row["joint_velocity_rad_s"]) > self.native_speed + 2e-6
            ).any(-1)
            self.nonfoot_steps += row["nonfoot_contact"]
            self.minimum_clearance = np.minimum(self.minimum_clearance, non_toe)
            self.minimum_plate = np.minimum(self.minimum_plate, row["root_pose_xyzw"][:, 2])
            self.previous_q = row["joint_position_rad"].astype(float)
            if self.maximum_applied.max() > 1.60001:
                raise RuntimeError("Native applied torque cap exceeded")
            if len(self.rows) >= 800:
                self.flush()
        except BaseException as error:
            self.failure = repr(error)
            _save(
                self.output / "failed_partial_step.json",
                {k: np.asarray(v).tolist() for k, v in row.items()},
            )
            self.flush()
            raise

    def control_record(self, result, requested_command, control):
        if len(self.last_control) != 8:
            raise RuntimeError("A complete eight-sample hold is required")
        raw = {key: np.stack([r[key] for r in self.last_control]) for key in self.last_control[0]}
        last = self.last_control[-1]
        world_velocity = last["root_com_velocity"][:, :3]
        rotations = np.stack([diagnostic_rotation(q) for q in last["root_pose_xyzw"][:, 3:]])
        body_velocity = np.einsum("nji,nj->ni", rotations, world_velocity)
        gyro = np.einsum("nji,nj->ni", rotations, last["root_com_velocity"][:, 3:])
        nav = np.stack((-body_velocity[:, 1], body_velocity[:, 0], body_velocity[:, 2]), -1)
        return {
            "time_s": np.array((control + 1) * 0.02),
            "root_pose_xyzw": last["root_pose_xyzw"],
            "velocity_world_mps": world_velocity,
            "velocity_navigation_mps": nav,
            "gyro_body_rad_s": gyro,
            "joint_position_rad": last["joint_position_rad"],
            "joint_velocity_rad_s": last["joint_velocity_rad_s"],
            "joint_target_rad": last["joint_target_rad"],
            "computed_torque_nm": last["computed_torque_nm"],
            "applied_torque_nm": last["applied_torque_nm"],
            "saturation_count_400hz": (abs(raw["computed_torque_nm"]) > 1.6).sum(0),
            "applied_torque_squared_sum_400hz": np.square(raw["applied_torque_nm"]).sum(0),
            "computed_torque_abs_max_400hz": abs(raw["computed_torque_nm"]).max((0, 2)),
            "applied_torque_abs_max_400hz": abs(raw["applied_torque_nm"]).max((0, 2)),
            "nonfoot_contact": last["nonfoot_contact"],
            "nonfoot_contact_count_400hz": raw["nonfoot_contact"].sum(0),
            "missing_six_toe_count_400hz": (~raw["distal_contact"].all(-1)).sum(0),
            "minimum_non_toe_floor_m": raw["minimum_non_toe_floor_m"].min(0),
            "command": last["command"],
            "requested_command": np.asarray(requested_command, np.float32),
            "terminated": _cpu(result["terminated"]),
            "truncated": _cpu(result["truncated"]),
            "reset": np.zeros(self.env.num_envs, bool),
        }

    def flush(self):
        if self.rows:
            name = f"substeps_{len(self.files):03d}.npz"
            np.savez_compressed(
                self.output / name, **{k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
            )
            self.files.append(name)
            self.rows = []
        self.contacts.flush()

    def close(self):
        self.env.capture = None
        self.flush()
        self.contacts.close()
        receipt = {
            "schema": "canonical_exact_evaluation_capture_v1",
            "steps": self.count,
            "initial_counter": self.initial_counter,
            "final_counter": self.counter,
            "failure": self.failure,
            "substep_files": self.files,
            "maximum_applied_nm": self.maximum_applied.tolist(),
            "maximum_requested_nm": self.maximum_requested.tolist(),
            "joint_bound_violation_steps": self.joint_bound_violations.tolist(),
            "speed_bound_violation_steps": self.speed_bound_violations.tolist(),
            "nonfoot_contact_steps_400hz": self.nonfoot_steps.tolist(),
            "minimum_non_toe_floor_m": [
                None if not np.isfinite(x) else float(x) for x in self.minimum_clearance
            ],
            "minimum_plate_height_m": [
                None if not np.isfinite(x) else float(x) for x in self.minimum_plate
            ],
            "body_names": self.env.native_body_names,
            "joint_names": list(JOINT_NAMES),
            "files": {p.name: _sha(p) for p in sorted(self.output.iterdir()) if p.is_file()},
            "contact_classification": "exact_distal_points",
        }
        _save(self.output / "capture.json", receipt)
        return receipt


def _set_command(env, commands):
    env.commands.copy_(torch.as_tensor(commands, dtype=torch.float32, device=env.device))
    # History contains proprio only; update command-bearing observations before
    # the actor, without inserting a history sample or native physics step.
    return env.observations(env.current)


@torch.inference_mode()
def run_batch(
    env,
    policy,
    cases,
    output,
    geometry_extrema,
    *,
    checkpoint_sha256,
    source_sha256,
    seed=27057,
    video_case_id=None,
    progress=None,
    max_wall_seconds=None,
    motion_features=None,
):
    """One initial reset, full fixed-duration trials; a failure preserves prefixes.

    Each case holds its command until its ``stop_control`` and commands zero after it; a case
    without a stop control holds its command throughout, and a route case asks the waypoint
    follower for its command from the body's true pose. ``motion_features`` maps the environment
    state to the motion features that the trace records before and after each control. The report
    holds recording facts only; the judge decides pass or fail from the trace.
    """
    if not cases or len(cases) > env.num_envs or len({x["controls"] for x in cases}) != 1:
        raise ValueError("Batch cases must fit the environment and have one fixed duration")
    cases = [{"stop_control": case["controls"], **case} for case in cases]
    producers = [
        requirements.producer(case) for case in cases + [cases[-1]] * (env.num_envs - len(cases))
    ]
    duration = cases[0]["controls"] * 0.02
    if env.cfg.episode_seconds <= duration:
        raise ValueError("Declared episode timeout would truncate the requested evaluation")
    deadline = _deadline(max_wall_seconds)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    assignments = cases + [cases[-1]] * (env.num_envs - len(cases))
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.commands.zero_()
    env.reset()
    capture = ExactEvaluationCapture(env, out / "native400hz", geometry_extrema)
    rows = []
    failure = None
    failure_kind = None
    writer = None
    frames = 0
    allocation_limit_reached = False
    video_index = next((i for i, c in enumerate(cases) if c["case_id"] == video_case_id), None)
    declaration = {
        "schema": "hexapod_native_batch_v1",
        "cases": cases,
        "assigned_case_ids": [c["case_id"] for c in assignments],
        "padding_replicas": env.num_envs - len(cases),
        "seed": seed,
        "episode_timeout_seconds": env.cfg.episode_seconds,
        "max_wall_seconds": max_wall_seconds,
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_sha256,
        "model_sha256": MODEL_SHA256,
        "urdf_sha256": URDF_SHA256,
        "usd_sha256": USD_SHA256,
        "geometry_extrema_sha256": _sha(geometry_extrema),
        "contact_classification": "exact_distal_points",
        "velocity_measurement_point": "native root center_of_mass; actual world velocity rotated into body/navigation",
        "initial_resets": 1,
        "resets_during_trial": 0,
        "pose_forcing": False,
    }
    _save(out / "declaration.json", declaration)
    try:
        if video_index is not None:
            if not env.cfg.render:
                raise ValueError(
                    "Native cameras must be enabled before constructing a video evaluation environment"
                )
            import imageio.v2 as imageio

            env.render(index=video_index)
            writer = imageio.get_writer(
                str(out / "rollout.mp4"), fps=25, codec="libx264", quality=8
            )
        for control in range(cases[0]["controls"]):
            if control % 100 == 0 and time.monotonic() >= deadline:
                allocation_limit_reached = True
                raise RuntimeError("Evaluation wall-clock allocation exhausted; preserved prefix")
            root = _cpu(env.current["root"])
            requested = [produce(control, root[i]) for i, produce in enumerate(producers)]
            observation = _set_command(env, requested)
            before = None if motion_features is None else motion_features(env.current)
            action = (
                policy.act(observation["obs"], deterministic=True)
                if hasattr(policy, "act")
                else policy(observation["obs"])
            )
            result = env.step(action)
            row = capture.control_record(result, requested, control)
            row["policy_action"] = _cpu(action)
            row["policy_observation"] = _cpu(observation["obs"])
            row["critic_observation"] = _cpu(observation["critic"])
            if motion_features is not None:
                row["amp_state_before"] = _cpu(before)
                row["amp_state_after"] = _cpu(motion_features(env.current))
            row["toe_xyz_world_m"] = _cpu(env.current["toe_world"])
            row["distal_contact"] = capture.last_control[-1]["distal_contact"].copy()
            if hasattr(policy, "model"):
                mean, estimate = policy.model.actor(observation["obs"])
                if not torch.equal(mean, action):
                    raise RuntimeError(
                        "Recorded deterministic policy differs from its actual actor"
                    )
                row["estimated_velocity_navigation_mps"] = _cpu(estimate)
            rows.append(row)
            if writer is not None and control % 2 == 1:
                frame = env.render(index=video_index)
                writer.append_data(frame)
                frames += 1
                if frames == 1:
                    imageio.imwrite(str(out / "first_frame.png"), frame)
            if progress is not None and (control + 1) % 100 == 0:
                progress(
                    {
                        "case_ids": [c["case_id"] for c in cases],
                        "controls": control + 1,
                        "requested_controls": cases[0]["controls"],
                    }
                )
            if bool(result["terminated"].any() or result["truncated"].any()):
                raise _NativeTrialTermination(
                    "Native terminal/timeout: preserved failed prefix; no automatic reset"
                )
        verify_native_recipe(env, "after_evaluation")
    except BaseException as error:
        failure = repr(error)
        failure_kind = (
            "native_terminal_prefix"
            if isinstance(error, _NativeTrialTermination)
            else "allocation_deadline"
            if allocation_limit_reached
            else "acquisition_error"
        )
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception as error:
            failure = (failure + "; " if failure else "") + "Video finalization: " + repr(error)
            failure_kind = "acquisition_error"
        receipt = capture.close()
        data = {k: np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
        if data:
            np.savez_compressed(out / "control_trace.npz", **data)
    complete = failure is None and receipt["steps"] == 8 * len(rows) and receipt["failure"] is None
    results = [
        {
            "case_id": case["case_id"],
            "controls": len(rows),
            "native_capture_complete": complete and len(rows) == case["controls"],
            "joint_bound_violation_steps": receipt["joint_bound_violation_steps"][index],
            "speed_bound_violation_steps": receipt["speed_bound_violation_steps"][index],
            "maximum_applied_nm": receipt["maximum_applied_nm"][index],
            "nonfoot_contact_steps_400hz": receipt["nonfoot_contact_steps_400hz"][index],
            "minimum_non_toe_floor_m": receipt["minimum_non_toe_floor_m"][index],
            "minimum_plate_height_m": receipt["minimum_plate_height_m"][index],
            "batch_failure": failure,
        }
        for index, case in enumerate(cases)
    ]
    report = {
        **declaration,
        "acquisition_complete": failure is None,
        "failure": failure,
        "failure_kind": failure_kind,
        "controls": len(rows),
        "allocation_limit_reached": allocation_limit_reached,
        "native_capture_failure": receipt["failure"],
        "recorded_physics_steps": receipt["steps"],
        "results": results,
        "video_frames": frames,
        "video_case_id": video_case_id if video_index is not None else None,
        "files": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    load_path = out / "force_metrics.json"
    loads = write_summary(out, load_path)
    report["force_metrics"] = {"path": load_path.name, "status": loads["status"]}
    report["files"][load_path.name] = _sha(load_path)
    _save(out / "report.json", report)
    return report


def run_cases(
    env,
    policy,
    output,
    geometry_extrema,
    checkpoint,
    *,
    requirements_path=requirements.REQUIREMENTS,
    requirements_set="stage2_flat",
    case_ids=None,
    seed=27057,
    video_case_id=None,
    progress=None,
    max_wall_seconds=None,
    motion_features=None,
):
    """Run the cases of one requirement set, one native trial each, then judge the traces.

    ``case_ids`` selects an ordered subset; the judgement then covers that subset only and says so
    in its allocation. A fall ends its case and the next case still runs. An acquisition error, a
    native physics error or the wall-clock deadline stops the run; recorded cases stay judged.
    """
    rules, requirements_sha = requirements.load_requirements(requirements_path, requirements_set)
    by_id = {case["case_id"]: case for case in requirements.cases(rules)}
    if case_ids is None:
        case_ids = list(by_id)
    if not case_ids or len(set(case_ids)) != len(case_ids) or set(case_ids) - set(by_id):
        raise ValueError("Case selection must be a nonempty ordered subset of the requirement set")
    selected = [by_id[case_id] for case_id in case_ids]
    if env.num_envs != 1:
        raise ValueError("Requirement cases run on one native robot")
    if env.cfg.episode_seconds <= max(case["controls"] for case in selected) * 0.02:
        raise ValueError("Episode timeout would truncate a requirement case")
    if video_case_id is not None and (video_case_id not in case_ids or not env.cfg.render):
        raise ValueError("A video case must be selected and needs native cameras")
    started = time.monotonic()
    deadline = _deadline(max_wall_seconds)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = _sha(checkpoint)
    source_sha = _sha(__file__)
    judge_sha = _sha(requirements.__file__)
    allocation = {
        "schema": "hexapod_requirements_allocation_v1",
        "requirements_set": requirements_set,
        "requirements_sha256": requirements_sha,
        "judge_sha256": judge_sha,
        "orchestrator_sha256": source_sha,
        "checkpoint_sha256": checkpoint_sha,
        "cases": selected,
        "seed": seed,
        "episode_timeout_seconds": env.cfg.episode_seconds,
        "max_wall_seconds": max_wall_seconds,
        "video_case_id": video_case_id,
    }
    _save(out / "allocation.json", allocation)
    failure = None
    reports = []
    for index, case in enumerate(selected):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = "Wall-clock allocation exhausted before " + case["case_id"]
            break
        if env.native_errors:
            failure = "Native physics errors prevent another case"
            break
        try:
            report = run_batch(
                env,
                policy,
                [case],
                out / f"batch_{index:03d}",
                geometry_extrema,
                checkpoint_sha256=checkpoint_sha,
                source_sha256=source_sha,
                seed=seed,
                video_case_id=video_case_id if case["case_id"] == video_case_id else None,
                progress=progress,
                max_wall_seconds=None if not math.isfinite(remaining) else remaining,
                motion_features=motion_features,
            )
        except Exception as error:
            failure = repr(error)
            _save(out / "acquisition_exception.json", {"case": case, "error": failure})
            break
        reports.append(report)
        if report["failure"] and report["failure_kind"] != "native_terminal_prefix":
            failure = report["failure"]
            break
    if (
        _sha(checkpoint) != checkpoint_sha
        or _sha(__file__) != source_sha
        or _sha(requirements.__file__) != judge_sha
    ):
        failure = "Evaluation input changed during collection"
    judgement = requirements.judge(out, requirements_path, requirements_set)
    judgement.update(
        allocation=allocation,
        failure=failure,
        wall_seconds=time.monotonic() - started,
        video_files=[
            str(Path(f"batch_{i:03d}") / "rollout.mp4")
            for i, r in enumerate(reports)
            if r["video_frames"]
        ],
    )
    judgement["pass"] = (
        bool(judgement["pass"]) and failure is None and len(judgement["cases"]) == len(selected)
    )
    _save(out / "judgement.json", judgement)
    return judgement
