#!/usr/bin/env python3
"""Live standing characterization of the corrected serial CAD baseline.

Uses Isaac Lab 3's preset/launch helpers, not the retired standalone launcher.
No policy is trained. A successful report covers standing on a plane only;
the serial four-bar approximation and driven joint/direction checks remain
explicitly outside this gate. A short run is a probe, not full acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Sequence
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
EXPECTED_JOINTS = tuple(f"{leg}_{kind}" for kind in ("coxa_yaw", "femur_pitch", "tibia_pitch") for leg in LEGS)
PHYSICS_DT_S = 0.005
DECIMATION = 4


def build_parser(add_launcher_args=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report", type=Path, default=Path("validation_artifacts/mkii_v2_validation.json"))
    if add_launcher_args is not None:
        add_launcher_args(parser)
        parser.set_defaults(visualizer=[])
    return parser


def validate_arguments(args, overrides: Sequence[str] = ()) -> None:
    if args.num_envs < 1 or args.steps < 2:
        raise ValueError("--num_envs must be positive and --steps at least 2")
    if overrides:
        raise ValueError(f"Unknown/task/config overrides are forbidden in fixed CAD v2 validation: {list(overrides)}")
    if args.report.exists():
        raise ValueError("Report already exists; choose a new path to preserve validation evidence")


class Window:
    """Small CPU-testable reducer; runtime supplies tensor-reduced samples."""
    def __init__(self):
        self.steps = self.env_samples = self.joint_samples = self.saturated = 0
        self.height_sum = self.abs_computed_sum = 0.0
        self.min_height = self.max_height = None
        self.max_applied = self.max_computed = 0.0
        self.vertical_squared_sum = self.roll_pitch_squared_sum = self.tilt_squared_sum = 0.0
        self.nonfoot_env_steps = 0
        self.named_contacts = {}
        self.max_computed_by_joint = {}
        self.saturated_by_env_joint = None
        self.minimum_support_by_env = None

    def add(self, sample: dict) -> None:
        scalars = [value for value in sample.values() if isinstance(value, (float, int))]
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Nonfinite aggregate validation sample")
        saturation = sample["saturated_by_env_joint"]
        support = sample["support_count_by_env"]
        if (len(saturation) != sample["env_samples"] or len(support) != sample["env_samples"]
                or any(len(row) != 18 or any(value not in (0, 1) for value in row) for row in saturation)
                or any(not isinstance(value, int) or not 0 <= value <= 6 for value in support)
                or sum(map(sum, saturation)) != sample["saturated"]
                or sample["joint_samples"] != 18 * sample["env_samples"]):
            raise ValueError("Invalid per-environment motor saturation/support sample")
        if self.saturated_by_env_joint is None:
            self.saturated_by_env_joint = [[0] * 18 for _ in saturation]
            self.minimum_support_by_env = list(support)
        if len(saturation) != len(self.saturated_by_env_joint):
            raise ValueError("Environment count changed within a validation window")
        for env_index, row in enumerate(saturation):
            for joint_index, value in enumerate(row):
                self.saturated_by_env_joint[env_index][joint_index] += value
            self.minimum_support_by_env[env_index] = min(self.minimum_support_by_env[env_index], support[env_index])
        self.steps += 1
        self.env_samples += sample["env_samples"]
        self.joint_samples += sample["joint_samples"]
        self.saturated += sample["saturated"]
        self.height_sum += sample["height_sum"]
        self.abs_computed_sum += sample["abs_computed_sum"]
        self.min_height = sample["min_height"] if self.min_height is None else min(self.min_height, sample["min_height"])
        self.max_height = sample["max_height"] if self.max_height is None else max(self.max_height, sample["max_height"])
        self.max_applied = max(self.max_applied, sample["max_applied"])
        self.max_computed = max(self.max_computed, sample["max_computed"])
        self.vertical_squared_sum += sample["vertical_squared_sum"]
        self.roll_pitch_squared_sum += sample["roll_pitch_squared_sum"]
        self.tilt_squared_sum += sample["tilt_squared_sum"]
        self.nonfoot_env_steps += sample["nonfoot_env_steps"]
        for name, count in sample["named_contacts"].items():
            self.named_contacts[name] = self.named_contacts.get(name, 0) + count
        for name, value in sample["max_computed_by_joint"].items():
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite computed torque: {name}")
            self.max_computed_by_joint[name] = max(self.max_computed_by_joint.get(name, 0.0), value)

    def report(self) -> dict:
        fractions = [[value / self.steps for value in row] for row in self.saturated_by_env_joint] if self.steps else []
        return {
            "steps": self.steps, "env_samples": self.env_samples, "joint_samples": self.joint_samples,
            "sample_unit": "physics_substep",
            "min_base_height_m": self.min_height, "max_base_height_m": self.max_height,
            "mean_base_height_m": self.height_sum / self.env_samples if self.env_samples else None,
            "max_abs_applied_torque_nm": self.max_applied, "max_abs_computed_torque_nm": self.max_computed,
            "mean_abs_computed_torque_nm": self.abs_computed_sum / self.joint_samples if self.joint_samples else None,
            "saturated_joint_samples": self.saturated,
            "torque_saturation_fraction": self.saturated / self.joint_samples if self.joint_samples else None,
            "torque_saturation_fraction_by_env_joint": fractions,
            "worst_env_joint_torque_saturation_fraction": max(map(max, fractions)) if fractions else None,
            "minimum_support_count_by_env": self.minimum_support_by_env,
            "minimum_support_count": min(self.minimum_support_by_env) if self.minimum_support_by_env else None,
            "vertical_velocity_rms_mps": math.sqrt(self.vertical_squared_sum / self.env_samples) if self.env_samples else None,
            "roll_pitch_rate_rms_radps": math.sqrt(self.roll_pitch_squared_sum / self.env_samples) if self.env_samples else None,
            "tilt_rms_deg": math.degrees(math.sqrt(self.tilt_squared_sum / self.env_samples)) if self.env_samples else None,
            "non_foot_contact_env_steps": self.nonfoot_env_steps,
            "named_non_foot_contact_env_steps": self.named_contacts,
            "max_abs_computed_torque_by_joint_nm": self.max_computed_by_joint,
        }


def grade_report(report: dict) -> list[str]:
    """Standing criteria; never treats startup torque as settled saturation."""
    errors = []
    for key in ("cpu_asset_integrity_pass", "kit_asset_integrity_pass", "finite_states_and_observations", "commands_held_zero"):
        if report.get(key) is not True:
            errors.append(f"{key} did not pass")
    if tuple(report.get("joint_names", ())) != EXPECTED_JOINTS:
        errors.append("Imported joint order does not match the CAD v2 manifest")
    if report.get("body_count") != 19 or sorted(report.get("foot_names", [])) != sorted(f"{leg}_tibia" for leg in LEGS):
        errors.append("Expected 19 bodies and exactly the six named tibia contact sensors")
    if not report.get("anatomical_frame_wiring", {}).get("pass"):
        errors.append("Static anatomical command-frame wiring failed")
    clearance = report.get("reset_geometry", {}).get("minimum_collision_clearance_m")
    if clearance is None or not math.isfinite(clearance) or clearance < -1e-6:
        errors.append("Measured reset configuration penetrates the plane")
    if report.get("steps_completed") != report.get("steps_requested"):
        errors.append("Requested standing steps did not complete")
    if report.get("terminated_count", 0) or report.get("truncated_count", 0):
        errors.append("Unexpected termination or truncation")
    sampling = report.get("sampling", {})
    if (sampling.get("mode") != "every_physics_substep_after_scene_update"
            or sampling.get("physics_handles_decimation") is not False
            or sampling.get("physics_dt_s") != PHYSICS_DT_S
            or sampling.get("expected_substeps_per_policy_step") != DECIMATION
            or sampling.get("observed_substeps") != report.get("steps_requested", 0) * DECIMATION
            or sampling.get("all_policy_steps_have_expected_samples") is not True):
        errors.append("Complete 200 Hz physics-substep sampling was not established")
    if sum(report.get(name, {}).get("steps", 0) for name in ("startup", "settled")) != sampling.get("observed_substeps"):
        errors.append("Physics sample counts disagree with the reported standing windows")
    for name in ("startup", "settled"):
        window = report.get(name, {})
        height = window.get("min_base_height_m")
        if not window.get("steps") or height is None or not math.isfinite(height) or height < 0.055:
            errors.append(f"{name}: no valid standing samples or base fell below 0.055 m")
        torque = window.get("max_abs_applied_torque_nm")
        if torque is None or not math.isfinite(torque) or torque > 1.61:
            errors.append(f"{name}: applied torque exceeded 1.61 N*m")
    settled = report.get("settled", {})
    saturation = settled.get("torque_saturation_fraction")
    if saturation is None or not math.isfinite(saturation) or saturation >= 0.005:
        errors.append("Settled torque saturation must remain below 0.5% of joint samples")
    worst = settled.get("worst_env_joint_torque_saturation_fraction")
    if worst is None or not math.isfinite(worst) or worst >= 0.005:
        errors.append("Settled saturation must be below 0.5% for every environment and motor")
    computed = settled.get("max_abs_computed_torque_nm")
    if computed is None or not math.isfinite(computed) or computed > 1.60001:
        errors.append("Settled computed torque exceeded the 1.6 N*m rating (1e-5 tolerance)")
    if settled.get("minimum_support_count") is None or settled["minimum_support_count"] < 1:
        errors.append("Settled foot-contact stream lacks at least one loaded pad in every environment/substep")
    if settled.get("non_foot_contact_env_steps", 0):
        errors.append("Settled base/coxa/femur/tibia-shaft ground contact")
    return errors


def reset_geometry_from_state(urdf, joint_names, joint_positions, root_positions, root_quaternions, ground_heights):
    """URDF collision support bounds at measured reset joint/root states.

    This is FK at live state readback, not a PhysX penetration query. It avoids
    relying on articulation link-pose caches before the first physics step.
    """
    import numpy as np
    from tools.assets.audit_mkii_stance import forward_kinematics, primitive_bottom_z, _origin

    root = ET.parse(urdf).getroot()
    counts = {len(joint_positions), len(root_positions), len(root_quaternions), len(ground_heights)}
    if len(counts) != 1 or not joint_positions:
        raise ValueError("Reset state batches must have matching nonzero lengths")
    if tuple(joint_names) != EXPECTED_JOINTS or any(len(q) != 18 for q in joint_positions):
        raise ValueError("Reset state requires the observed 18 CAD joint names and positions")
    minima = []
    feet = {leg: [] for leg in LEGS}
    for q, xyz, quaternion, ground in zip(joint_positions, root_positions, root_quaternions, ground_heights):
        if len(xyz) != 3 or len(quaternion) != 4 or not all(math.isfinite(value) for value in [*q, *xyz, *quaternion, ground]):
            raise ValueError("Reset states must contain finite positions and quaternions")
        w, x, y, z = quaternion
        if not math.isclose(sum(value*value for value in quaternion), 1.0, abs_tol=1e-5):
            raise ValueError("Reset root quaternion is not unit length")
        transform = np.eye(4)
        transform[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        transform[:3, 3] = xyz
        poses = forward_kinematics(root, dict(zip(joint_names, q)))
        bottoms = []
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                shape = list(collision.find("geometry"))[0]
                bottom = primitive_bottom_z(transform @ poses[link.get("name")] @ _origin(collision), shape) - ground
                bottoms.append(bottom)
                if link.get("name").endswith("_tibia") and shape.tag == "sphere":
                    feet[link.get("name").split("_", 1)[0]].append(bottom)
        minima.append(min(bottoms))
    return {"method": "URDF collision FK at live reset joint/root readback",
            "minimum_collision_clearance_m": min(minima), "per_env_minimum_collision_clearance_m": minima,
            "minimum_pad_clearance_by_leg_m": {leg: min(values) for leg, values in feet.items()}}


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def contact_stream_faults(forces, points):
    """Return tensor flags, permitting undefined positions only when unloaded."""
    import torch
    if forces.shape != points.shape or forces.ndim != 3 or forces.shape[-2:] != (6, 3):
        raise ValueError("Expected one force/contact point for each of six feet per environment")
    loaded = torch.linalg.norm(forces, dim=-1) > 1.0
    return (~torch.isfinite(forces).all(),
            (loaded & ~torch.isfinite(points).all(dim=-1)).any())


class PhysicsSubstepHook:
    """Observe the explicit DirectRLEnv physics loop without changing that loop.

    Isaac Lab's PhysX path writes actions, steps physics, then updates the
    scene once per substep. Backends which hide decimation cannot be admitted
    by this observer. The original update always runs before measurement.
    """
    def __init__(self, raw, callback, metadata):
        self.raw, self.callback, self.metadata = raw, callback, metadata
        self.total = 0
        self.start = None

    def __enter__(self):
        if getattr(self.raw, "_physics_handles_decimation", None) is not False:
            raise ValueError("Cannot observe every substep when physics handles decimation internally")
        if self.raw.cfg.decimation != DECIMATION or not math.isclose(self.raw.cfg.sim.dt, PHYSICS_DT_S, abs_tol=1e-12):
            raise ValueError("Standing validator requires four 0.005 s physics substeps per policy step")
        self.scene = self.raw.scene
        self.original = self.scene.update
        self.prior_instance_update = vars(self.scene).get("update")
        self.had_instance_update = "update" in vars(self.scene)
        self.metadata.update(mode="every_physics_substep_after_scene_update", physics_dt_s=PHYSICS_DT_S,
            physics_handles_decimation=False, expected_substeps_per_policy_step=DECIMATION,
            observed_substeps=0, all_policy_steps_have_expected_samples=False,
            contact_force_threshold_n=1.0,
            contact_classification="per-tibia aggregate contact centroid; mixed shaft/pad contact may be obscured",
            coverage="state and actuator/contact readbacks after each physics substep; not solver-iteration impulses")
        for sensor in [*self.raw._feet_contact_sensors, self.raw._base_contact_sensor,
                       self.raw._coxa_contact_sensor, *self.raw._femur_contact_sensors]:
            if not 0 <= sensor.cfg.update_period <= PHYSICS_DT_S:
                raise ValueError("Contact sensor update period cannot cover every physics substep")

        def observed_update(*args, **kwargs):
            dt = kwargs.get("dt", args[0] if args else None)
            if self.start is None or dt is None or not math.isclose(dt, PHYSICS_DT_S, abs_tol=1e-12):
                raise ValueError("Unexpected scene update outside a measured 0.005 s physics substep")
            result = self.original(*args, **kwargs)
            self.callback()
            self.total += 1
            self.metadata["observed_substeps"] = self.total
            return result
        self.scene.update = observed_update
        return self

    def begin_policy_step(self):
        if self.start is not None:
            raise ValueError("Previous policy-step measurement is incomplete")
        self.start = self.total
        self.metadata["all_policy_steps_have_expected_samples"] = False

    def finish_policy_step(self):
        if self.start is None or self.total - self.start != DECIMATION:
            raise ValueError("Expected exactly four observed physics substeps per policy step")
        self.start = None
        self.metadata["all_policy_steps_have_expected_samples"] = True

    def __exit__(self, *_):
        if self.start is not None:
            self.metadata["all_policy_steps_have_expected_samples"] = False
        if self.had_instance_update:
            self.scene.update = self.prior_instance_update
        else:
            del self.scene.update


class PhysicsSamples:
    """Reduce on the device, transferring four compact samples once per step."""
    SCALARS = ("saturated", "height_sum", "min_height", "max_height", "max_applied", "max_computed",
               "abs_computed_sum", "vertical_squared_sum", "roll_pitch_squared_sum", "tilt_squared_sum",
               "nonfoot_env_steps")
    FAULTS = ("nonfinite_state", "nonfinite_contact_forces", "nonfinite_loaded_contact_position",
              "nonfinite_contact_sensor_pose", "nonzero_standing_commands")
    STATE_FIELDS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_b", "projected_gravity_b",
                    "joint_pos", "joint_vel", "applied_torque", "computed_torque")

    def __init__(self, raw, joint_names, foot_names):
        self.raw, self.joint_names, self.foot_names = raw, joint_names, foot_names
        self.groups = [(raw._base_contact_sensor, "base"), (raw._coxa_contact_sensor, "coxa")]
        self.groups += [(sensor, "femur") for sensor in raw._femur_contact_sensors]
        self.named_contacts = [f"{name}:{category}" for sensor, category in self.groups for name in sensor.body_names]
        self.named_contacts += [f"{name}:shaft" for name in foot_names]
        self.pending = []

    def capture(self):
        import torch
        raw = self.raw
        fields = {name: _tensor(getattr(raw._robot.data, name)) for name in self.STATE_FIELDS}
        if fields["computed_torque"].shape != (raw.num_envs, 18):
            raise ValueError("Invalid per-environment actuator readback shape")
        height = fields["root_pos_w"][:, 2] - _tensor(raw._terrain.env_origins)[:, 2]
        computed, applied = fields["computed_torque"].abs(), fields["applied_torque"].abs()
        contact_data = [sensor.data for sensor in raw._feet_contact_sensors]
        foot_forces = torch.stack([_tensor(data.force_matrix_w)[:, 0, 0] for data in contact_data], dim=1)
        points = torch.stack([_tensor(data.contact_pos_w)[:, 0, 0] for data in contact_data], dim=1)
        force_fault, point_fault = contact_stream_faults(foot_forces, points)
        pose_fault = ~torch.stack([torch.isfinite(_tensor(getattr(data, name))).all()
                                  for data in contact_data for name in ("pos_w", "quat_w")]).all()
        pad, shaft, _ = raw._get_foot_contact_state()
        if pad.shape != (raw.num_envs, 6) or shaft.shape != pad.shape:
            raise ValueError("Invalid six-foot contact classification shape")
        masks, named_counts = [], []
        for sensor, _ in self.groups:
            # Current force, not history: one observation for this exact step.
            forces = _tensor(sensor.data.net_forces_w)
            force_fault = force_fault | ~torch.isfinite(forces).all()
            mask = torch.linalg.norm(forces, dim=-1) > 1.0
            masks.append(mask)
            named_counts.append(mask.sum(dim=0))
        masks.append(shaft)
        named_counts.append(shaft.sum(dim=0))
        nonfoot = torch.cat(masks, dim=1).any(dim=1)
        tilt = torch.acos(torch.clamp(-fields["projected_gravity_b"][:, 2], -1., 1.))
        saturated = computed > 1.6
        scalars = torch.stack((saturated.sum(), height.sum(), height.min(), height.max(), applied.max(),
            computed.max(), computed.sum(), fields["root_lin_vel_w"][:, 2].square().sum(),
            fields["root_ang_vel_b"][:, :2].square().sum(), tilt.square().sum(), nonfoot.sum()))
        faults = torch.stack((~torch.stack([torch.isfinite(value).all() for value in fields.values()]).all(),
                              force_fault, point_fault, pose_fault, (raw._commands != 0).any()))
        # All entries are newly reduced/copied tensors, never views of mutable
        # PhysX buffers. Keeping them on-device avoids a sync per metric.
        self.pending.append(torch.cat((scalars, faults.to(scalars.dtype), saturated.flatten().to(scalars.dtype),
            pad.sum(dim=1).to(scalars.dtype), computed.max(dim=0)[0], torch.cat(named_counts).to(scalars.dtype))))

    def drain(self):
        import torch
        if len(self.pending) != DECIMATION:
            raise ValueError("Expected exactly four buffered physics samples")
        rows = torch.stack(self.pending).cpu().tolist()
        self.pending.clear()
        result = []
        for row in rows:
            offset = len(self.SCALARS)
            faults = [name for name, value in zip(self.FAULTS, row[offset:offset+len(self.FAULTS)]) if value]
            if faults:
                raise ValueError(f"Invalid physics-substep streams: {faults}")
            offset += len(self.FAULTS)
            count = self.raw.num_envs
            saturation = [[int(value) for value in row[offset+i*18:offset+(i+1)*18]] for i in range(count)]
            offset += count * 18
            support = [int(value) for value in row[offset:offset+count]]
            offset += count
            maxima = dict(zip(self.joint_names, row[offset:offset+18]))
            offset += 18
            sample = dict(zip(self.SCALARS, row[:len(self.SCALARS)]))
            sample.update(env_samples=count, joint_samples=count*18,
                saturated_by_env_joint=saturation, support_count_by_env=support,
                max_computed_by_joint=maxima,
                named_contacts={name: int(value) for name, value in zip(self.named_contacts, row[offset:])})
            result.append(sample)
        return result


def run_cpu_asset_gate(urdf: Path, usd: Path, *, python_executable=None, checker_script=None) -> dict:
    """Keep standalone OpenUSD's binary libraries out of the Kit process.

    Kit may bundle another OpenUSD ABI than the standalone SDK Python. A child
    process can use the latter for preflight without poisoning Kit's imports.
    """
    command = [str(python_executable or sys.executable),
               str(checker_script or ROOT / "tools/assets/prepare_mkii_usd.py"),
               str(urdf), str(usd), "--check"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"CPU asset checker returned no valid JSON (exit {completed.returncode}): {completed.stderr[-2000:]}") from error
    if not isinstance(result, dict) or result.get("pass") is not True or completed.returncode != 0:
        raise ValueError(f"CPU asset checker failed (exit {completed.returncode}): {result}")
    return result


def configure_nominal_dynamics(cfg) -> dict:
    """Keep nominal URDF masses and explicitly assign deterministic materials."""
    cfg.events.base_mass = None
    material = cfg.events.physics_material
    if material is None:
        raise ValueError("Expected the CAD material startup term for explicit nominal friction")
    material.params = {**material.params,
        "static_friction_range": (1.0, 1.0), "dynamic_friction_range": (1.0, 1.0),
        "restitution_range": (0.0, 0.0), "num_buckets": 1}
    return {"base_mass_event": "disabled; retain imported URDF masses",
            "material_startup_term": "fixed values, no random variation",
            "robot_static_friction": 1.0, "robot_dynamic_friction": 1.0, "robot_restitution": 0.0,
            "reset_joint_jitter_rad": [-0.03, 0.03], "seed": 0}


def finalize_report(report: dict, path: Path, error: BaseException | None = None) -> None:
    """Persist evidence before native Kit cleanup can terminate the process."""
    if report.get("_persisted"):
        return
    report.setdefault("errors", [])
    if error is not None:
        report["errors"].append(f"{type(error).__name__}: {error}")
    else:
        report["errors"].extend(grade_report(report))
    report["pass"] = not report["errors"]
    report["standing_gate_pass"] = report["pass"]
    startup_peak = report.get("startup", {}).get("max_abs_computed_torque_nm")
    report["startup_raw_rating_exceeded"] = startup_peak is not None and startup_peak > 1.6
    report["hardware_admission"] = "not_granted; startup demand and real dynamics require separate review"
    report["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    report["_persisted"] = True  # Private in-memory flag; not serialized.
    print(f"MKII_V2_VALIDATION_{'PASS' if report['pass'] else 'FAIL'} report={path}", flush=True)
    if report["errors"]:
        print(json.dumps(report["errors"]), flush=True)


def run_live(args, report, resolve_task_config, launch_simulation):
    import gymnasium as gym
    import torch
    from hexapod_core import cad_manifest_v2 as contract

    cfg, _ = resolve_task_config(TASK_ID, "")
    cfg.seed = 0
    cfg.scene.num_envs = args.num_envs
    report["nominal_dynamics_overrides"] = configure_nominal_dynamics(cfg)
    cfg.velocity_command = None
    cfg.command_lin_vel_x_range_mps = cfg.command_lin_vel_y_range_mps = cfg.command_yaw_rate_range_rad_s = (0.0, 0.0)
    cfg.stand_command_fraction = 1.0
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.decimation * cfg.sim.dt + 1.0)
    if cfg.command_frame != "navigation" or tuple(cfg.expected_runtime_joint_names) != contract.RUNTIME_JOINT_NAMES:
        raise ValueError("Resolved config is not the fixed CAD v2 navigation/joint contract")
    urdf = ROOT / contract.URDF_PATH
    usd = Path(cfg.robot.spawn.usd_path)
    asset_report = run_cpu_asset_gate(urdf, usd)
    report.update(cpu_asset_integrity_pass=asset_report["pass"], urdf_sha256=contract.verify_urdf_identity(urdf),
                  usd_path=str(usd), usd_root_sha256=asset_report["usd_root_sha256"],
                  asset_integrity_errors=asset_report["errors"], policy_step_dt_s=cfg.decimation * cfg.sim.dt,
                  cpu_asset_check_process="isolated_child", cpu_asset_check_usd_version=asset_report.get("usd_version"))
    if not asset_report["pass"]:
        raise ValueError(f"CPU asset integrity failed: {asset_report['errors']}")
    startup, settled = Window(), Window()
    settling_steps = min(max(1, args.steps // 5), args.steps - 1)
    report["settling_steps"] = settling_steps
    with launch_simulation(cfg, args):
        try:
            # Import pxr-dependent code only AFTER Kit has selected its own binary
            # libraries. Revalidate using that runtime before constructing the env.
            from tools.assets.prepare_mkii_usd import validate as validate_asset
            kit_asset_report = validate_asset(urdf, usd)
            report.update(kit_asset_integrity_pass=kit_asset_report["pass"],
                          kit_usd_version=kit_asset_report.get("usd_version"))
            if not kit_asset_report["pass"]:
                raise ValueError(f"Kit-native asset integrity failed: {kit_asset_report['errors']}")
            env = gym.make(TASK_ID, cfg=cfg)
        except BaseException as error:
            finalize_report(report, args.report, error)
            raise
        try:
            observations, _ = env.reset(seed=0)
            raw = env.unwrapped
            raw.episode_length_buf.zero_()
            raw._commands.zero_()
            robot = raw._robot
            joint_names = list(robot.joint_names)
            foot_names = [str(name) for sensor in raw._feet_contact_sensors for name in sensor.body_names]
            report.update(joint_names=joint_names, body_count=robot.num_bodies, foot_names=foot_names,
                          commands_held_zero=True, finite_states_and_observations=True)
            print(f"MKII_V2_JOINT_NAMES {json.dumps(joint_names)}", flush=True)
            if tuple(joint_names) != contract.RUNTIME_JOINT_NAMES or robot.num_joints != 18:
                raise ValueError("Observed articulation joint order/count mismatch")
            test_vectors = torch.tensor([[0,-1,0],[1,0,0],[0,0,1],[0,1,0],[-1,0,0],[0,0,-1]], dtype=torch.float32, device=raw.device)
            expected = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float32, device=raw.device)
            actual = raw._vector_in_command_frame(test_vectors)
            report["anatomical_frame_wiring"] = {"pass": bool(torch.equal(actual, expected)),
                "mode": "static_vector_wiring_only", "driven_physical_direction_test": "not_run"}
            data = robot.data
            report["reset_geometry"] = reset_geometry_from_state(urdf, joint_names,
                _tensor(data.joint_pos).cpu().tolist(), _tensor(data.root_pos_w).cpu().tolist(),
                _tensor(data.root_quat_w).cpu().tolist(), _tensor(raw._terrain.env_origins)[:, 2].cpu().tolist())
            report["reset_joint_positions_rad"] = _tensor(data.joint_pos).cpu().tolist()
            report["reset_root_positions_m"] = _tensor(data.root_pos_w).cpu().tolist()
            actions = torch.zeros(env.action_space.shape, device=raw.device)
            collector = PhysicsSamples(raw, joint_names, foot_names)
            report["sampling"] = {}
            with PhysicsSubstepHook(raw, collector.capture, report["sampling"]) as sampling:
                for step in range(args.steps):
                    raw._commands.zero_()
                    sampling.begin_policy_step()
                    with torch.inference_mode():
                        observations, rewards, terminated, truncated, _ = env.step(actions)
                    sampling.finish_policy_step()
                    report["steps_completed"] = step + 1
                    report["terminated_count"] += int(torch.count_nonzero(_tensor(terminated)).item())
                    report["truncated_count"] += int(torch.count_nonzero(_tensor(truncated)).item())
                    fields = {"observations": _tensor(observations["policy"]), "reward": _tensor(rewards)}
                    bad = [name for name, value in fields.items() if not bool(torch.isfinite(value).all())]
                    if bad:
                        report.update(finite_states_and_observations=False, nonfinite_fields=bad)
                        raise ValueError(f"Nonfinite live observation fields: {bad}")
                    if bool(torch.any(raw._commands != 0)):
                        report["commands_held_zero"] = False
                        raise ValueError("Standing commands changed during validation")
                    try:
                        samples = collector.drain()
                    except ValueError:
                        report["finite_states_and_observations"] = False
                        raise
                    for sample in samples:
                        (startup if step < settling_steps else settled).add(sample)
                    report["startup"], report["settled"] = startup.report(), settled.report()
                    if step == 0 or (step + 1) % 100 == 0:
                        minimum = min(sample["min_height"] for sample in samples)
                        maximum = max(sample["max_computed"] for sample in samples)
                        print(f"MKII_V2_PROGRESS step={step+1}/{args.steps} min_height={minimum:.6f} computed={maximum:.6f}", flush=True)
                    if report["terminated_count"] or report["truncated_count"]:
                        break
        finally:
            # Kit shutdown may call exit(0) rather than return to Python main.
            # Write both passing and failing evidence before env.close as well.
            finalize_report(report, args.report, sys.exc_info()[1])
            env.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Locate the output before optional SDK imports so import errors also leave
    # a machine-readable failure artifact. Unknown args are validated below.
    early, _ = build_parser().parse_known_args(argv)
    report = {"schema_version": 3, "task_id": TASK_ID, "mode": "corrected_serial_standing_baseline",
        "full_model_1to1_approved": False, "training_admission": "not_granted_by_this_report",
        "driven_joint_direction_check": "not_run", "steps_requested": early.steps,
        "num_envs": early.num_envs, "steps_completed": 0, "terminated_count": 0, "truncated_count": 0,
        "run_kind": "acceptance" if early.steps >= 1000 and early.num_envs >= 32 else "short_probe",
        "pass": False, "errors": []}
    paths = [ROOT / "isaaclab", ROOT / "packages/hexapod_env", ROOT / "packages/hexapod_core", ROOT / "tools"]
    sys.path[:0] = [str(path) for path in paths if str(path) not in sys.path]
    try:
        validate_arguments(early)
        from hexapod_env.tasks.mkii_v2.register import register_mkii_v2, MKII_V2_FLAT_TASK_ID
        if MKII_V2_FLAT_TASK_ID != TASK_ID:
            raise ValueError("Validator/registration task identity mismatch")
        register_mkii_v2()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        parser = build_parser(add_launcher_args)
        original_argv = sys.argv
        try:
            sys.argv = [str(Path(__file__)), *argv]
            args, overrides = setup_preset_cli(parser)
        finally:
            sys.argv = original_argv
        validate_arguments(args, overrides)
        # Hydra must see no unreviewed task/config overrides during resolution.
        original_argv = sys.argv
        try:
            sys.argv = [str(Path(__file__))]
            run_live(args, report, resolve_task_config, launch_simulation)
        finally:
            sys.argv = original_argv
    except Exception as error:
        if not report.get("_persisted"):
            finalize_report(report, early.report, error)
        else:
            # A cleanup failure must still return nonzero; the saved report
            # describes physical validation completed before cleanup.
            if report["pass"]:
                print(f"MKII_V2_CLEANUP_ERROR {type(error).__name__}: {error}", flush=True)
            report["pass"] = False
    if not report.get("_persisted"):
        finalize_report(report, early.report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
