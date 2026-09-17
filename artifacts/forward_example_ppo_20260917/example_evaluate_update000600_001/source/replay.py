"""Realize analytic joint targets through native PD; never force a replay pose.

Call ``realize_prior`` after the caller starts Isaac and creates PaperWalkEnv.
Only the initial reset writes state. Every subsequent sample comes from eight
native 400 Hz steps; the raw AMP61 feature path is exactly the policy env path.
This is a feasibility screen and reference recording, not Stage 2 admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayConfig:
    settle_seconds: float = 4.
    cycles: int = 3
    max_tracking_rms_rad: float = .05
    max_saturation_fraction: float = .02
    min_command_projection_fraction: float = .10
    applied_cap_nm: float = 1.60001

    def validate(self):
        if self.settle_seconds != 4. or self.cycles != 3:
            raise ValueError("Replay fixes 4 s settle and three complete cycles")
        if (self.max_tracking_rms_rad != .05 or self.max_saturation_fraction != .02
                or self.min_command_projection_fraction != .10 or self.applied_cap_nm != 1.60001):
            raise ValueError("Replay selection criteria must be declared before execution")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _stack(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}


def load_target_cycles(prior_path, metadata_path, env):
    """Validate named input identity, command blocks and the actual target range."""
    from .env_config import MODEL_SHA256, URDF_SHA256
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("model_sha256") != MODEL_SHA256 or metadata.get("urdf_sha256") != URDF_SHA256:
        raise ValueError("Replay reference model differs")
    if metadata.get("prior_sha256") != _sha(prior_path):
        raise ValueError("Replay reference bytes differ")
    if metadata.get("joint_names") != list(env.joint_names):
        raise ValueError("Replay joint order differs")
    if not np.allclose(metadata["nominal_joint_position_rad"], _cpu(env.neutral), atol=2e-8, rtol=0):
        raise ValueError("Replay neutral differs from native environment")
    if metadata.get("dt_s") != .02 or env.cfg.control_dt != .02 or env.cfg.decimation != 8:
        raise ValueError("Replay requires unchanged 50/400 Hz execution")
    commands = np.asarray(metadata["commands"], dtype=np.float32)
    count = round(metadata["period_s"] / .02)
    if commands.ndim != 2 or commands.shape[1] != 3 or count < 3 or not np.isclose(count*.02, metadata["period_s"]):
        raise ValueError("Invalid command or cycle declaration")
    with np.load(prior_path, allow_pickle=False) as data:
        states = data["states"].copy()
        nxt = data["next_states"].copy()
        recorded_commands = data["commands"].copy()
    if states.shape != (len(commands)*count, 61) or nxt.shape != states.shape:
        raise ValueError("Reference does not contain complete AMP61 command cycles")
    if not all(np.isfinite(x).all() for x in (states, nxt, commands, recorded_commands)):
        raise ValueError("Nonfinite reference")
    if not np.array_equal(recorded_commands, np.repeat(commands, count, axis=0)):
        raise ValueError("Reference command blocks differ")
    if not np.array_equal(nxt.reshape(-1, count, 61), np.roll(states.reshape(-1, count, 61), -1, axis=1)):
        raise ValueError("Reference transitions cross command boundaries")
    q = states[:, :18].reshape(len(commands), count, 18)
    if np.max(np.abs(np.roll(q, -1, axis=1).astype(float)-q)) > env.cfg.target_slew_rad + 1e-7:
        raise ValueError("Reference cyclic targets exceed declared slew")
    actions = (q - _cpu(env.neutral)) / env.cfg.action_scale_rad
    if np.max(np.abs(actions)) > 1.:
        raise ValueError("Reference target would be silently action-clamped")
    if env.num_envs < len(commands):
        raise ValueError("One reset-free batch must contain every reference command")
    total_seconds = 4. + 3*count*.02
    if env.cfg.episode_seconds <= total_seconds:
        raise ValueError("Episode timeout would interrupt the reset-free replay")
    return metadata, commands, q, actions


class NativeReplayCapture:
    """Actual 400 Hz torque and exact toe/shaft patch evidence, without gating stance."""
    def __init__(self, env):
        from .env import _DiagnosticGeometry
        if env.capture is not None:
            raise ValueError("Replay requires exclusive ownership of the native capture hook")
        self.env = env
        self.geometry = _DiagnosticGeometry(env.geometry_meta, {}, env.native_body_names)
        self.rows = []
        self.counter = env.sim.get_physics_step_count()
        env.capture = self

    def __call__(self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        from .env import _classify_patches
        data = [np.array(x.numpy(), copy=True) for x in env.contact.get_contact_data(env.cfg.physics_dt)]
        contact = _classify_patches(data, env.sensor_map, _cpu(state["link"]), self.geometry, env.num_envs)
        counter = env.sim.get_physics_step_count()
        row = {"physics_counter": np.array(counter), "control_index": np.array(len(self.rows)//8),
               "substep_index": np.array(substep),
               "joint_position_rad": _cpu(state["q"]), "joint_velocity_rad_s": _cpu(state["dq"]),
               "joint_target_rad": _cpu(target), "requested_torque_nm": _cpu(requested),
               "applied_torque_nm": _cpu(applied), "effort_ceiling_nm": _cpu(ceiling),
               "native_input_pre_nm": _cpu(native_pre), "root_pose_xyzw": _cpu(state["root"]),
               "distal_contact": contact["distal_contact"],
               "distal_force_world_n": contact["distal_force_world"],
               "nonfoot_contact": contact["nonfoot_contact"],
               "nonfoot_force_world_n": contact["nonfoot_force_world"]}
        joint_bad = ((state["q"] < env.lower-2e-6) | (state["q"] > env.upper+2e-6)).any(-1)
        row["fall_or_joint_violation"] = _cpu((state["root"][:, 2] < .045)
            | (state["gravity"][:, 2] > -np.cos(.85)) | joint_bad)
        self.rows.append(row)
        if counter != self.counter + 1 or len(self.rows) % 8 != (substep + 1) % 8:
            raise RuntimeError("Replay native substep recurrence differs")
        self.counter = counter
        if not all(np.isfinite(value).all() for value in row.values()):
            raise FloatingPointError("Nonfinite native replay evidence")
        if not np.array_equal(row["native_input_pre_nm"], row["applied_torque_nm"]):
            raise RuntimeError("Replay native torque input differs")
        if np.max(np.abs(row["applied_torque_nm"])) > 1.60001:
            raise RuntimeError("Replay breached unchanged actuator cap")

    def close(self):
        self.env.capture = None


def _selection(raw, native, row_commands, count, settle, config):
    """Selection never edits a recorded transition or replaces actual velocity."""
    summaries = []
    scored = slice(settle + count, settle + 3*count)
    native_scored = slice((settle + count)*8, (settle + 3*count)*8)
    for e, command in enumerate(row_commands):
        reason = []
        terminated = bool(raw["terminated"][:, e].any() or native["fall_or_joint_violation"][:, e].any())
        truncated = bool(raw["truncated"][:, e].any())
        nonfoot = bool(native["nonfoot_contact"][:, e].any())
        rms = float(np.sqrt(np.mean((native["joint_position_rad"][native_scored, e].astype(float)
                                    - native["joint_target_rad"][native_scored, e])**2)))
        saturation = (np.abs(native["requested_torque_nm"][native_scored, e]) > 1.6).mean(0)
        # Actual AMP native body velocity -> navigation forward/left; yaw is +Z.
        actual = raw["next_states"][scored, e]
        mean_nav = np.array([-actual[:, 37].mean(), actual[:, 36].mean(), actual[:, 41].mean()])
        speed = float(np.linalg.norm(command[:2]))
        linear_ratio = float(np.dot(mean_nav[:2], command[:2]) / speed**2) if speed > 1e-8 else None
        yaw_ratio = float(mean_nav[2] / command[2]) if abs(command[2]) > 1e-8 else None
        if terminated: reason.append("fall_or_joint_termination")
        if truncated: reason.append("timeout")
        if nonfoot: reason.append("native_nonfoot_contact")
        if rms >= config.max_tracking_rms_rad: reason.append("actual_target_joint_rms")
        if float(saturation.max()) >= config.max_saturation_fraction: reason.append("requested_saturation_fraction")
        if any(x is not None and x < config.min_command_projection_fraction for x in (linear_ratio, yaw_ratio)):
            reason.append("commanded_motion_not_realized")
        summaries.append({"env": e, "command": command.tolist(), "accepted": not reason,
            "rejected_reasons": reason, "joint_tracking_rms_rad": rms,
            "requested_target_tracking_rms_rad": float(np.sqrt(np.mean((
                raw["next_states"][scored, e, :18].astype(float)
                - raw["requested_joint_target_rad"][scored, e])**2))),
            "actual_target_step_max_rad": float(np.abs(np.diff(
                raw["applied_joint_target_rad"][:, e].astype(float), axis=0)).max()),
            "scored_actual_target_step_max_rad": float(np.abs(np.diff(
                raw["applied_joint_target_rad"][scored, e].astype(float), axis=0)).max()),
            "worst_joint_saturation_fraction": float(saturation.max()),
            "overall_saturation_fraction": float(saturation.mean()),
            "peak_femur_applied_torque_nm": float(np.abs(native["applied_torque_nm"][:, e, 1::3]).max()),
            "peak_applied_torque_nm": float(np.abs(native["applied_torque_nm"][:, e]).max()),
            "peak_requested_torque_nm": float(np.abs(native["requested_torque_nm"][:, e]).max()),
            "minimum_toe_support_count": int(native["distal_contact"][native_scored, e].sum(-1).min()),
            "mean_actual_navigation_velocity": mean_nav.tolist(),
            "linear_command_projection_fraction": linear_ratio, "yaw_command_projection_fraction": yaw_ratio,
            "terminated": terminated, "truncated": truncated, "nonfoot_contact": nonfoot})
    return summaries


@torch.inference_mode()
def realize_prior(env, prior_path, metadata_path, output_dir, *, config=None, progress=None, physics_identity=None):
    """Record one batch of native demonstrations and export selected cycle pairs.

    Integration: after env construction call this function, inspect report
    ``all_commands_accepted`` and consume ``realized_prior.npz`` only if true.
    ``heldout_prior.npz`` contains the third cycle, NOT an independent period
    variant. A separate 1.3 s replay is still required for Fable's generalisation
    test. Keep original metadata for model/neutral physics binding; bind the
    realized NPZ and replay report separately in the learner identity.
    """
    config = config or ReplayConfig()
    config.validate()
    metadata, commands, q, actions = load_target_cycles(prior_path, metadata_path, env)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    count = q.shape[1]
    settle = round(config.settle_seconds / env.cfg.control_dt)
    command_index = np.arange(env.num_envs) % len(commands)
    row_commands = commands[command_index]
    env.commands.zero_()
    observation = env.reset()
    start_counter = env.sim.get_physics_step_count()
    capture = NativeReplayCapture(env)
    rows, failure = [], None
    declaration = {"schema": "canonical_paper_native_prior_replay_v1", "config": asdict(config),
        "physics_identity": physics_identity,
        "source_sha256": _sha(__file__), "analytic_prior_sha256": _sha(prior_path),
        "analytic_metadata_sha256": _sha(metadata_path), "model_sha256": metadata["model_sha256"],
        "urdf_sha256": metadata["urdf_sha256"], "num_envs": env.num_envs,
        "command_index_by_env": command_index.tolist(), "command_order": commands.tolist(),
        "dt_s": .02, "physics_dt_s": .0025, "cycle_controls": count,
        "neutral_joint_position_rad": metadata["nominal_joint_position_rad"],
        "amp_order": metadata["amp_order"], "amp_feature_transform": "None; raw env AMP61 for both policy and native reference",
        "split": {"settle": "excluded", "cycle_1": "excluded warmup", "cycle_2": "train", "cycle_3": "heldout same trajectory"},
        "independent_heldout_variant": False, "pose_forcing_after_initial_reset": False,
        "optional_bc": "Actual pre-hold policy observations and normalized desired actions are exported; no BC runs automatically. The observation contains the previous actual post-slew target. This adaptation is not paper-exact.",
        "resets_after_initial": 0, "physical_admission": False, "stage2_complete": False,
        "selection_scope": "Replay feasibility only; standing saturation gate remains 0.005. No per-transition cherry-picking: reject a complete environment demonstration."}
    _json(output / "declaration.json", declaration)
    try:
        for step in range(settle + config.cycles*count):
            moving = step >= settle
            cycle = (step-settle)//count if moving else -1
            phase = (step-settle)%count if moving else -1
            if step == settle:
                env.commands.copy_(torch.as_tensor(row_commands, device=env.device))
                # Current history contains proprio only. Refresh command-bearing
                # obs/critic coherently, without resetting or advancing physics.
                observation = env._observations(env.current)
            requested_q = q[command_index, phase] if moving else np.broadcast_to(_cpu(env.neutral), (env.num_envs, 18))
            action = actions[command_index, phase] if moving else np.zeros((env.num_envs, 18), np.float32)
            before = _cpu(observation["amp"])
            before_obs = _cpu(observation["obs"])
            result = env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device))
            row = {"states": before, "next_states": _cpu(result["amp"]),
                "observations": before_obs,
                "commands": _cpu(env.commands), "requested_joint_target_rad": requested_q.copy(),
                "applied_joint_target_rad": _cpu(env.telemetry["joint_target_rad"]),
                "actions": action.copy(), "terminated": _cpu(result["terminated"]),
                "truncated": _cpu(result["truncated"]), "cycle_index": np.array(cycle),
                "phase_index": np.array(phase), "control_index": np.array(step),
                "physics_counter": np.array(env.sim.get_physics_step_count())}
            if (row["states"].shape != (env.num_envs, 61) or row["next_states"].shape != (env.num_envs, 61)
                    or row["observations"].shape != (env.num_envs, 231)):
                raise ValueError("Native AMP61 shape differs")
            if not np.array_equal(row["observations"][:, 210:213], row["commands"]):
                raise ValueError("BC pre-hold observations contain stale commands")
            if not all(np.isfinite(value).all() for value in row.values()):
                raise FloatingPointError("Nonfinite replay transition")
            rows.append(row)
            observation = result
            if progress is not None and (step+1 == settle or (moving and phase+1 == count)):
                progress({"controls": step+1, "cycle_completed": cycle+1, "any_termination": bool(row["terminated"].any())})
            # No auto-reset and no further unstable stepping after a fall. This
            # preserves the failed partial attempt and exports no expert set.
            if row["terminated"].any() or row["truncated"].any():
                raise RuntimeError("Native replay termination/timeout; whole attempt rejected")
    except BaseException as error:
        failure = repr(error)
    finally:
        capture.close()
        raw, native = _stack(rows), _stack(capture.rows)
        if raw: np.savez_compressed(output / "replay_controls.npz", **raw)
        if native: np.savez_compressed(output / "replay_substeps.npz", **native)
    complete = failure is None and len(rows) == settle + 3*count and len(capture.rows) == len(rows)*8
    summaries = _selection(raw, native, row_commands, count, settle, config) if complete else []
    accepted = np.array([row["accepted"] for row in summaries], bool)
    accepted_commands = sorted(set(command_index[accepted].tolist())) if len(accepted) else []
    all_accepted = complete and len(accepted_commands) == len(commands)
    datasets = {}
    # No reusable expert file is created for an incomplete or rejected command
    # bank. Full raw evidence above is retained even if every row was rejected.
    if all_accepted:
        for name, cycle in (("realized_prior.npz", 1), ("heldout_prior.npz", 2)):
            selection = raw["cycle_index"] == cycle
            dataset = {key: raw[key][selection][:, accepted].reshape(-1, raw[key].shape[-1])
                       for key in ("states", "next_states", "commands", "observations", "actions", "requested_joint_target_rad", "applied_joint_target_rad")}
            dataset["env_index"] = np.tile(np.flatnonzero(accepted), count)
            dataset["phase_index"] = np.repeat(np.arange(count), accepted.sum())
            dataset["cycle_index"] = np.full(count*accepted.sum(), cycle)
            np.savez_compressed(output / name, **dataset)
            datasets[name] = {"sha256": _sha(output/name), "transitions": len(dataset["states"])}
    settled_height = raw["next_states"][settle-1, :, 42].tolist() if len(rows) >= settle else None
    report = {**declaration, "complete": complete, "failure": failure, "controls": len(rows),
        "physics_steps": env.sim.get_physics_step_count()-start_counter, "recorded_substeps": len(capture.rows),
        "all_commands_accepted": all_accepted, "all_replicas_accepted": bool(len(accepted) and accepted.all()),
        "accepted_command_indices": accepted_commands, "replicas": summaries,
        "settled_root_height_by_env_m": settled_height, "datasets": datasets,
        "raw_files": {p.name: _sha(p) for p in sorted(output.glob("*.npz"))}}
    _json(output / "report.json", report)
    return report
