#!/usr/bin/env python3
"""Bounded mechanism diagnostics with pre-force/post-physics traces; never admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("tools", "isaaclab", "packages/hexapod_core", "packages/hexapod_env")]
from mkii_training_contract import TASK_ID, digest, identity, write_json
from hexapod_core.fourbar_v1 import ACTIVE_JOINT_NAMES, PHYSICS_DT_S, DECIMATION
from validate_mkii_fourbar import PhysicalMetrics, SubstepHook, apply_numerical_recipe, grade, tensor


TRACE_COORDINATE_CONVENTIONS = {
    "sample_phase": "after scene.update at each physics substep",
    "world_frame": "right-handed world XYZ; +Z up",
    "body_link_pos_w": "link frame origin in world coordinates, metres; not centre of mass",
    "body_link_quat_w": "Isaac Lab 3 native XYZW; rotates link-local vectors into world coordinates",
    "terrain_origin_w": "per-environment terrain origin in world coordinates, metres",
    "terrain_support_plane": "world Z equals terrain_origin_w/z for this flat-ground diagnostic",
    "foot_force_w": "per-foot net contact force in world coordinates, newtons",
}


def body_pose_trace_fields(body_names, pos, quat, terrain_origin):
    """Preserve native link poses so offline collider support heights are recoverable."""
    envs, bodies = pos.shape[0], len(body_names)
    if (len(set(body_names)) != bodies or tuple(pos.shape) != (envs, bodies, 3)
            or tuple(quat.shape) != (envs, bodies, 4) or tuple(terrain_origin.shape) != (envs, 3)):
        raise ValueError("Diagnostic body pose layout does not match named bodies/environments")
    return [
        ("body_link_pos_w", pos.reshape(envs, -1), [f"{name}_{a}" for name in body_names for a in "xyz"]),
        ("body_link_quat_w", quat.reshape(envs, -1), [f"{name}_{a}" for name in body_names for a in "xyzw"]),
        ("terrain_origin_w", terrain_origin, list("xyz")),
    ]


def motions(kind):
    if kind == "individuals":
        groups, duration = [[name] for name in ACTIVE_JOINT_NAMES], 50
    elif kind == "groups":
        groups, duration = [list(ACTIVE_JOINT_NAMES[i:i+6]) for i in (0, 6, 12)], 100
    elif kind == "lf_tibia":
        groups, duration = [["lf_tibia_lever_pivot"]], 100
    else:
        raise ValueError("Unknown diagnostic motion")
    result = [{"motors": group, "offset_rad": sign*.04, "steps": duration}
              for group in groups for sign in (1., -1.)]
    result.append({"motors": [], "offset_rad": 0., "steps": 100})
    return result


def parser(add_launcher_args=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--solver-multiplier", type=int, choices=(1, 2), default=1)
    p.add_argument("--diagnostic-motion", choices=("individuals", "groups", "lf_tibia"), default="individuals")
    p.add_argument("--diagnostic-usd", choices=("revolute_v3", "planar_d6_v4"), default="revolute_v3")
    p.add_argument("--report", type=Path, required=True)
    if add_launcher_args:
        add_launcher_args(p)
        p.set_defaults(visualizer=[])
    return p


def persist_then_close(report, path, env):
    """Kit teardown can exit the process; durable evidence must precede it."""
    report["pass"] = False
    write_json(path, report)
    print("FOURBAR_DIAGNOSTIC_RESULT " + json.dumps({key: report.get(key) for key in
        ("diagnostic_complete", "pass", "errors", "physical_gate_errors", "trace_samples")}), flush=True)
    if env is not None:
        env.close()


class Trace:
    def __init__(self, raw, metrics, output):
        self.raw, self.metrics, self.output = raw, metrics, output
        self.pending, self.files, self.columns = [], [], []
        self.before = None
        self.force_writes = self.samples = 0

    def write_forces(self):
        raw = self.raw
        import warp as wp
        self.before = {"pre_q": tensor(raw._robot.data.joint_pos).clone(),
                       "pre_qd": tensor(raw._robot.data.joint_vel).clone(),
                       "target": raw.motor_state("joint_pos_target").clone(),
                       "processed_target": raw._processed_actions.clone(),
                       "velocity_target": raw.motor_state("joint_vel_target").clone(),
                       "feedforward": raw.motor_state("joint_effort_target").clone()}
        motor_q = raw.coordinates.gather(self.before["pre_q"])
        motor_qd = raw.coordinates.gather(self.before["pre_qd"])
        gains = {key: tensor(getattr(raw._motor_model, key)).index_select(-1, raw._motor_order)
                 for key in ("stiffness", "damping")}
        self.before["p_term"] = gains["stiffness"]*(self.before["target"]-motor_q)
        self.before["d_term"] = gains["damping"]*(self.before["velocity_target"]-motor_qd)
        self.original_write()
        self.before["demand"] = raw.motor_state("computed_torque").clone()
        self.before["applied"] = raw.motor_state("applied_torque").clone()
        # Query the backend only after PD has consumed its input. A getter may
        # refresh a shared buffer; diagnostics must not refresh it before PD.
        self.before["direct_pre_q"] = wp.to_torch(raw._robot.root_view.get_dof_positions()).clone()
        self.before["direct_pre_qd"] = wp.to_torch(raw._robot.root_view.get_dof_velocities()).clone()
        self.force_writes += 1

    def capture(self):
        import torch
        from isaaclab.utils.math import matrix_from_quat
        if self.before is None or self.force_writes != self.samples + 1:
            raise ValueError("Diagnostic requires one force write per physics observation")
        self.metrics.capture()
        raw, data = self.raw, self.raw._robot.data
        pos, quat = tensor(data.body_link_pos_w), tensor(data.body_link_quat_w)
        rot = matrix_from_quat(quat)
        lin, ang = tensor(data.body_link_lin_vel_w), tensor(data.body_link_ang_vel_w)
        gaps, axes, velocities = [], [], []
        for pair in self.metrics.frames:
            states = []
            for i, f in pair:
                offset = torch.einsum("nij,j->ni", rot[:, i], f[:3, 3])
                basis = rot[:, i] @ f[:3, :3]
                states.append((pos[:, i]+offset, basis, lin[:, i]+torch.cross(ang[:, i], offset, dim=-1)))
            a, b = states
            inverse = a[1].transpose(-1, -2)
            gaps.append(torch.einsum("nij,nj->ni", inverse, b[0]-a[0]))
            axes.append(torch.einsum("nij,nj->ni", inverse, b[1][:, :, 2]-a[1][:, :, 2]))
            velocities.append(torch.einsum("nij,nj->ni", inverse, b[2]-a[2]))
        fields = [(key, value, raw._robot.joint_names if key in ("pre_q", "pre_qd", "direct_pre_q", "direct_pre_qd") else raw.active_joint_names)
                  for key, value in self.before.items()]
        fields += [("post_q", tensor(data.joint_pos), raw._robot.joint_names),
                   ("post_qd", tensor(data.joint_vel), raw._robot.joint_names),
                   ("instantaneous_limit", raw.motor_telemetry("instantaneous_limit_nm"), raw.active_joint_names),
                   ("headroom", raw.motor_telemetry("burst_headroom"), raw.active_joint_names)]
        foot_names = [f"{leg}_{axis}" for leg in ("lf", "lm", "lr", "rf", "rm", "rr") for axis in "xyz"]
        forces = torch.stack([tensor(s.data.net_forces_w)[:, 0] for s in raw._feet_contact_sensors], 1)
        fields.append(("foot_force_w", forces.flatten(1), foot_names))
        body_axes = [f"{name}_{axis}" for name in raw._robot.body_names for axis in "xyz"]
        fields += [("body_link_linear_velocity_w", lin.flatten(1), body_axes),
                   ("body_link_angular_velocity_w", ang.flatten(1), body_axes)]
        fields += body_pose_trace_fields(raw._robot.body_names, pos, quat, tensor(raw._terrain.env_origins))
        for key, values in (("hinge_gap_local", gaps), ("hinge_axis_difference_local", axes),
                            ("hinge_relative_point_velocity_local", velocities)):
            names = [f"{leg}_{axis}" for leg in ("lf", "lm", "lr", "rf", "rm", "rr") for axis in "xyz"]
            fields.append((key, torch.stack(values, 1).flatten(1), names))
        columns = [f"{key}/{name}" for key, _, names in fields for name in names]
        if self.columns and self.columns != columns:
            raise ValueError("Diagnostic trace layout changed")
        self.columns = columns
        self.pending.append(torch.cat([value for _, value, _ in fields], -1).detach().cpu().numpy().copy())
        self.samples += 1
        self.before = None

    def flush(self, segment):
        import numpy as np
        if not self.pending:
            return
        path = self.output / f"trace_{len(self.files):03d}.npz"
        values = np.stack(self.pending)
        if not np.isfinite(values).all():
            raise ValueError("Diagnostic trace contains nonfinite values")
        with path.open("xb") as stream:
            np.savez_compressed(stream, values=values, columns=np.asarray(self.columns))
        self.files.append({"file": path.name, "sha256": digest(path), "segment": dict(segment),
                           "shape": list(values.shape), "first_physics_sample": self.samples-len(self.pending)})
        self.pending.clear()

    def __enter__(self):
        self.had_override = "write_data_to_sim" in vars(self.raw.scene)
        self.previous = vars(self.raw.scene).get("write_data_to_sim")
        self.original_write = self.raw.scene.write_data_to_sim
        self.raw.scene.write_data_to_sim = self.write_forces
        return self

    def __exit__(self, *_):
        if self.had_override:
            self.raw.scene.write_data_to_sim = self.previous
        else:
            del self.raw.scene.write_data_to_sim


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    early, _ = parser().parse_known_args(argv)
    if early.report.exists() or not 1 <= early.num_envs <= 8 or not 2 <= early.steps <= 1000:
        raise ValueError("Diagnostic needs new evidence, 1..8 environments and 2..1000 standing steps")
    early.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "hexapod.fourbar_diagnostic.v1", "mode": "diagnose", "task_id": TASK_ID,
        "pass": False, "simulation_training_admission": False, "hardware_admission": False,
        "diagnostic_complete": False, "diagnostic_motion": early.diagnostic_motion,
        "diagnostic_usd": early.diagnostic_usd,
        "num_envs": early.num_envs, "steps_requested": early.steps, "steps_completed": 0,
        "driven_steps": 0, "driven_steps_completed": 0,
        "driven_steps_requested": sum(row["steps"] for row in motions(early.diagnostic_motion)),
        "trace_coordinate_conventions": dict(TRACE_COORDINATE_CONVENTIONS),
        "physics_substeps": 0, "solver_multiplier": early.solver_multiplier,
        "terminated_count": 0, "truncated_count": 0, "errors": [], "trace_files": []}
    env = trace = None
    segment = {"phase": "setup"}
    try:
        report["contract"] = identity()
        kinematics = json.loads((ROOT / "configs/mkii_fourbar_v3_kinematics.json").read_text())
        usd = ROOT / kinematics["usd_path_relative"]
        if early.diagnostic_usd == "planar_d6_v4":
            usd = ROOT / "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v4/hexapod_mkii_fourbar_v4.usda"
        report["usd_path_relative"], report["usd_sha256"] = usd.relative_to(ROOT).as_posix(), digest(usd)
        result = subprocess.run([sys.executable, str(ROOT / "tools/audit_mkii_fourbar_usd.py"), str(usd)],
                                text=True, capture_output=True, timeout=180)
        report["cpu_asset_pass"] = result.returncode == 0 and json.loads(result.stdout).get("pass") is True
        if not report["cpu_asset_pass"]:
            raise ValueError("Diagnostic CPU asset audit failed")
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        saved = sys.argv
        try:
            sys.argv = [__file__, *argv]
            args, overrides = setup_preset_cli(parser(add_launcher_args))
            if overrides:
                raise ValueError(f"Unreviewed config overrides: {overrides}")
            sys.argv = [__file__]
            cfg, _ = resolve_task_config(TASK_ID, "")
        finally:
            sys.argv = saved
        plan = motions(args.diagnostic_motion)
        cfg.scene.num_envs, cfg.seed = args.num_envs, 0
        cfg.robot.spawn.usd_path = str(usd)
        cfg.standing_only, cfg.reset_joint_jitter_rad = True, 0.
        cfg.episode_length_s = (args.steps + sum(row["steps"] for row in plan) + 2)*.02 + 1.
        report["numerical_recipe"] = apply_numerical_recipe(cfg, args.solver_multiplier)
        report["solver_iterations"] = [cfg.robot.spawn.articulation_props.solver_position_iteration_count,
                                        cfg.robot.spawn.articulation_props.solver_velocity_iteration_count]
        if "pxr" in sys.modules:
            raise ValueError("Diagnostic configuration imported standalone USD before Kit")
        with launch_simulation(cfg, args):
            try:
                from audit_mkii_fourbar_usd import validate
                import mkii_fourbar_kinematics as kin
                import gymnasium as gym
                import torch
                import warp as wp
                report["kit_asset_pass"] = validate(kin.URDF, kin.PINS, usd)["pass"]
                if not report["kit_asset_pass"]:
                    raise ValueError("Diagnostic Kit asset audit failed")
                env = gym.make(TASK_ID, cfg=cfg)
                raw = env.unwrapped
                env.reset(seed=0)
                raw.episode_length_buf.zero_()
                report.update(body_count=raw._robot.num_bodies, joint_count=raw._robot.num_joints,
                    active_motor_count=len(raw.active_joint_names), joint_names=list(raw._robot.joint_names),
                    active_motor_names=list(raw.active_joint_names), runtime_manifest=raw.runtime_manifest)
                expected = torch.tensor([kinematics["default_joint_positions_rad"][n] for n in raw._robot.joint_names], device=raw.device)
                report["reset_max_joint_error_rad"] = float((tensor(raw._robot.data.joint_pos)-expected).abs().max())
                vectors = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=raw.device)
                report["anatomical_frame_pass"] = bool(torch.equal(raw._vector_in_command_frame(vectors), torch.eye(3, device=raw.device)))
                report["solver_readback"] = {key: wp.to_torch(getattr(raw._robot.root_view, method)()).clone().cpu().tolist()
                    for key, method in {"armatures": "get_dof_armatures", "stiffnesses": "get_dof_stiffnesses",
                                         "dampings": "get_dof_dampings"}.items()}
                report["actuator_backend_flags"] = {name: getattr(raw._robot, name, None)
                    for name in ("_has_implicit_actuators", "_has_newton_actuators")}
                from pxr import PhysxSchema
                from isaaclab.sim.utils.stage import get_current_stage
                stage = get_current_stage()
                report["scene_force_iteration_readback"] = [
                    {"prim": str(prim.GetPath()), "enabled": PhysxSchema.PhysxSceneAPI(prim).GetEnableExternalForcesEveryIterationAttr().Get()}
                    for prim in stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxSceneAPI)]
                report["motor_readback"] = {key: tensor(getattr(raw._motor_model, key)).clone().cpu().tolist()
                    for key in ("armature", "stiffness", "damping")}
                metrics = PhysicalMetrics(raw, kinematics)
                trace = Trace(raw, metrics, early.report.parent)
                actions = torch.zeros(raw.num_envs, 18, device=raw.device)
                segments = [{"phase": "standing", "motors": [], "offset_rad": 0., "steps": args.steps}]
                segments += [dict(row, phase="driven") for row in plan]
                with trace, SubstepHook(raw, trace.capture) as hook:
                    for segment in segments:
                        actions.zero_()
                        for motor in segment["motors"]:
                            actions[:, raw.active_joint_names.index(motor)] = segment["offset_rad"]/.30
                        for index in range(segment["steps"]):
                            metrics.window = ("startup" if index < max(1, args.steps//5) else "settled") if segment["phase"] == "standing" else "driven"
                            report["test_context"] = dict(segment, control_step=index)
                            _, _, terminated, truncated, _ = hook.step(env, actions)
                            metrics.drain()
                            report["windows"], report["physics_substeps"] = metrics.windows, hook.total
                            report["terminated_count"] += int(tensor(terminated).sum())
                            report["truncated_count"] += int(tensor(truncated).sum())
                            report["steps_completed" if segment["phase"] == "standing" else "driven_steps"] += 1
                            report["driven_steps_completed"] = report["driven_steps"]
                            if report["terminated_count"] or report["truncated_count"]:
                                report["termination_reasons"] = {name: int(tensor(value).sum()) for name, value in raw.last_termination_reasons.items()}
                                raise ValueError("Diagnostic observed reset; final trace precedes reset")
                        trace.flush(segment)
                        print("FOURBAR_DIAGNOSTIC_SEGMENT " + json.dumps({"segment": segment, "metrics": metrics.windows[metrics.window]}), flush=True)
                report["diagnostic_complete"] = True
                report["physical_gate_errors"] = grade(report)
            except BaseException as error:
                report["errors"].append(f"{type(error).__name__}: {error}")
                raise
            finally:
                try:
                    if trace is not None:
                        trace.flush(segment)
                        report["trace_files"] = trace.files
                        report["trace_samples"] = trace.samples
                        report["force_writes"] = trace.force_writes
                except BaseException as error:
                    report["errors"].append(f"Trace persistence {type(error).__name__}: {error}")
                    raise
                finally:
                    persist_then_close(report, early.report, env)
    except Exception as error:
        import traceback
        traceback.print_exc()
        message = f"{type(error).__name__}: {error}"
        if message not in report["errors"]:
            report["errors"].append(message)
    finally:
        if not early.report.exists():
            persist_then_close(report, early.report, None)
    return 0 if report["diagnostic_complete"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
