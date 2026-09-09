#!/usr/bin/env python3
"""Offline, nonadmitting reconstruction of the 32-environment individual prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def differences(a, b, path=""):
    if type(a) is not type(b):
        return [{"path": path, "replay": a, "campaign008": b}]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append({"path": path+"/"+key, "replay": a.get(key), "campaign008": b.get(key)})
            else:
                out += differences(a[key], b[key], path+"/"+key)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": path, "replay": a, "campaign008": b}]
        return [row for index, (x, y) in enumerate(zip(a, b)) for row in differences(x, y, path+f"/{index}")]
    return [] if a == b else [{"path": path, "replay": a, "campaign008": b}]


class TraceSet:
    """Verify sparse global indexing; never reinterpret recorded count as physics count."""
    def __init__(self, report_path, report, kind):
        self.directory = Path(report_path).resolve().parent
        self.report, self.kind = report, kind
        self.records = report["trace_files" if kind == "trace" else "control_trace_files"]
        self.index_field = "first_physics_sample" if kind == "trace" else "first_control_step"
        self.first, self.end = ((2200*report["numerical_recipe"]["decimation"], 2500*report["numerical_recipe"]["decimation"])
                                if kind == "trace" else (0, 2500))
        self.columns = None
        cursor, seen = self.first, set()
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("Missing trace records")
        for record in self.records:
            name = record["file"]
            if (not isinstance(name, str) or Path(name).name != name or not name.startswith(kind+"_")
                    or not name.endswith(".npz") or name in seen or (self.directory/name).is_symlink()):
                raise ValueError("Escaped, duplicate or invalid trace filename")
            seen.add(name)
            shape = record["shape"]
            if (not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
                    or shape[1] != report["num_envs"] or type(record[self.index_field]) is not int or record[self.index_field] != cursor):
                raise ValueError("Trace interval gap/overlap, shape or environment mismatch")
            cursor += shape[0]
        if cursor != self.end:
            raise ValueError("Trace does not cover the requested interval")

    def read(self, record):
        path = self.directory/record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"columns", "values"}:
                raise ValueError("Unexpected NPZ contents")
            values, columns = archive["values"], archive["columns"]
            if (columns.ndim != 1 or columns.dtype.kind not in "US" or len(set(columns.tolist())) != len(columns)
                    or any(not isinstance(s, str) or s.count("/") != 1 for s in columns.tolist())
                    or list(values.shape) != record["shape"] or values.shape[-1] != len(columns)
                    or values.dtype != np.float32 or not np.isfinite(values).all()):
                raise ValueError("Invalid trace dtype, names, shape or finite values")
            columns = columns.tolist()
            if self.columns is not None and columns != self.columns:
                raise ValueError("Trace columns changed between segments")
            self.columns = columns
        return values

    def block(self, values, field, names):
        return values[..., [self.columns.index(f"{field}/{name}") for name in names]]

    def vectors(self, values, field, names=LEGS):
        return self.block(values, field, [f"{name}_{axis}" for name in names for axis in "xyz"]).reshape(*values.shape[:2], len(names), 3)


def float32_phase_response(control_q, motor_index, phase_index):
    """Same five rows and float32 tensor mean/subtraction; CPU reduction is disclosed."""
    start = 1000 + phase_index*100
    means = [torch.from_numpy(control_q[start+sign*50+45:start+sign*50+50, :, motor_index].copy()).mean(0)
             for sign in (0, 1)]
    delta = means[0]-means[1]
    return [mean.tolist() for mean in means], delta.tolist(), float(delta.min()), int(delta.argmin())


def phase_analysis(control_q, report, baseline):
    names = report["active_motor_names"]
    recorded = report["individual_phase_means"]
    if len(recorded) != 30:
        raise ValueError("Expected all 30 individual sign records")
    rows = []
    for motor_index, name in enumerate(names[:15]):
        means, delta, minimum, worst = float32_phase_response(control_q, motor_index, motor_index)
        offsets = []
        for sign, mean in enumerate(means):
            phase = recorded[2*motor_index+sign]
            if (phase["motor"] != name or phase["first_control_step"] != 1000+motor_index*100+sign*50
                    or phase["offset_rad"] != (.04 if sign == 0 else -.04) or phase["control_steps"] != 50
                    or phase["mean_control_indices"] != [45, 46, 47, 48, 49] or phase["mean_dtype"] != "float32"
                    or len(phase["mean_joint_position_rad"]) != report["num_envs"]):
                raise ValueError("Recorded phase alignment/dtype differs from full validator")
            other = np.asarray(phase["mean_joint_position_rad"], dtype=np.float32)
            offsets.append(float(np.max(np.abs(np.asarray(mean, dtype=np.float64)-other.astype(np.float64)))))
        saved = report["individual_motor_response_by_env"][name]
        baseline_min = baseline["individual_motor_positive_minus_negative_rad"][name]
        rows.append({"motor": name, "positive_mean_by_env_rad": means[0], "negative_mean_by_env_rad": means[1],
                     "positive_minus_negative_by_env_rad": delta, "minimum_rad": minimum, "worst_env_index": worst,
                     "report_mean_max_abs_recompute_difference_rad": offsets,
                     "report_delta_max_abs_recompute_difference_rad": float(np.max(np.abs(np.array(delta)-saved["positive_minus_negative_rad"]))),
                     "campaign008_minimum_rad": baseline_min, "replay_minus_campaign008_minimum_rad": minimum-baseline_min,
                     "exact_campaign008_minimum_match": minimum == baseline_min})
    return rows


def motor_dynamics(speed, demand, applied, limit, no_load_rad_s):
    """Demand, clipping and braking intent are distinct from delivered torque."""
    return {"opposing_demand": speed*demand < 0,
            "overspeed_with_opposing_demand": (np.abs(speed) > no_load_rad_s) & (speed*demand < 0),
            "raw_minus_applied_abs_nm": np.abs(demand-applied),
            "applied_above_recorded_limit_nm": np.maximum(np.abs(applied)-limit, 0.)}


def passive_residual(values, names, kinematics, *, position=False):
    return np.stack([values[..., names.index(name)]-relation["multiplier"]*values[..., names.index(relation["source_joint"])]
                     -(relation["offset_rad"] if position else 0.) for name, relation in kinematics["passive_relations"].items()], -1)


def event_index(values, *, first=False):
    if first:
        indices = np.argwhere(values)
        return tuple(map(int, indices[0])) if len(indices) else None
    return tuple(map(int, np.unravel_index(np.abs(values).argmax(), values.shape)))


def control_events(traces, values, record, report, kin, events):
    """All 2500 boundaries; these endpoint speeds are not PD input velocities."""
    names = report["active_motor_names"]
    speed, demand, applied, limit = [traces.block(values, key, names) for key in
        ("joint_vel", "computed_torque", "applied_torque", "instantaneous_limit_nm")]
    no_load = report["runtime_manifest"]["motor_contract"]["configuration"]["vendor"]["no_load_rpm"]*math.tau/60
    peak_cap = report["runtime_manifest"]["motor_contract"]["configuration"]["vendor"]["peak_output_torque_nm"]
    feet = traces.vectors(values, "body_contact_force_w", [leg+"_tibia" for leg in LEGS])
    support = np.sum(np.linalg.norm(feet, axis=-1) > 1., axis=-1)
    passive = passive_residual(traces.block(values, "tree_joint_vel", report["joint_names"]), report["joint_names"], kin)
    for label, array, labels, first in (
        ("first_endpoint_speed_above_no_load", np.abs(speed) > no_load, names, True),
        ("first_raw_demand_above_vendor_peak", np.abs(demand) > peak_cap, names, True),
        ("first_raw_applied_difference_above_10uNm", np.abs(demand-applied) > 1e-5, names, True),
        ("first_zero_feet_above_1N", (support == 0)[..., None], ["all_feet"], True),
        ("peak_endpoint_speed_rad_s", speed, names, False), ("peak_raw_demand_nm", demand, names, False),
        ("peak_passive_endpoint_velocity_residual_rad_s", passive, list(kin["passive_relations"]), False)):
        index = event_index(array, first=first)
        if index is None: continue
        local, env, component = index
        control = record["first_control_step"]+local
        value = float(array[index]); previous = events.get(label)
        if previous is not None and ((first and previous["control_step"] <= control)
                or (not first and abs(previous["value"]) >= abs(value))): continue
        events[label] = {"event": label, "control_step": control, "environment": env, "name": labels[component],
            "value": value, "post_time_s": (control+1)*report["runtime_manifest"]["policy_dt_s"],
            "last_physics_sample": (control+1)*report["numerical_recipe"]["decimation"]-1,
            "supported_feet_above_1N": int(support[local, env]), "file": record["file"], "file_sha256": record["sha256"],
            "active_motor_endpoint_speed_rad_s": speed[local, env].tolist(),
            "active_motor_last_substep_raw_demand_nm": demand[local, env].tolist(),
            "active_motor_last_substep_applied_nm": applied[local, env].tolist(),
            "active_motor_recorded_limit_nm": limit[local, env].tolist()}


def detailed_analysis(traces, report, kin, worst_envs):
    names, joints = report["active_motor_names"], report["joint_names"]
    dt, decimation = report["numerical_recipe"]["physics_dt_s"], report["numerical_recipe"]["decimation"]
    no_load = report["runtime_manifest"]["motor_contract"]["configuration"]["vendor"]["no_load_rpm"]*math.tau/60
    peak_cap = report["runtime_manifest"]["motor_contract"]["configuration"]["vendor"]["peak_output_torque_nm"]
    events, segment_rows = {}, []
    def keep(label, array, record, values, metric_names, first=False):
        index = event_index(array, first=first)
        if index is None:
            return
        sample, env, component = index
        absolute_sample = record["first_physics_sample"]+sample
        value = float(array[index])
        previous = events.get(label)
        if previous is not None and ((first and previous["physics_sample"] <= absolute_sample)
                                     or (not first and abs(previous["value"]) >= abs(value))):
            return
        events[label] = {"event": label, "value": value, "name": metric_names[component], "environment": env,
            "physics_sample": absolute_sample, "control_step": absolute_sample//decimation,
            "substep_in_control": absolute_sample % decimation, "pre_time_s": absolute_sample*dt,
            "post_time_s": (absolute_sample+1)*dt, "file": record["file"], "file_sha256": record["sha256"],
            "segment": record["segment"], "nearby_same_environment_samples": [
                snapshot(values, s, env, record) for s in range(max(0, sample-2), min(len(values), sample+3))]}

    def snapshot(values, sample, env, record):
        def get(field, name):
            return float(values[sample, env, traces.columns.index(f"{field}/{name}")])
        motors = {}
        for leg in LEGS[:3]:
            name = f"{leg}_tibia_lever_pivot"
            motors[name] = {key: get(key, name) for key in ("pre_q", "post_q", "pre_qd", "post_qd", "direct_pre_qd",
                "target", "processed_target", "p_term", "d_term", "feedforward", "demand", "applied", "instantaneous_limit", "headroom")}
            motors[name]["q_finite_difference_interval_average_rad_s"] = (get("post_q", name)-get("pre_q", name))/dt
        return {"physics_sample": record["first_physics_sample"]+sample,
            "motors": motors, "C_pin_gap_local_m": {leg: [get("hinge_gap_local", leg+"_"+axis) for axis in "xyz"] for leg in LEGS},
            "C_pin_relative_velocity_local_m_s": {leg: [get("hinge_relative_point_velocity_local", leg+"_"+axis) for axis in "xyz"] for leg in LEGS},
            "foot_force_w_n": {leg: [get("foot_force_w", leg+"_"+axis) for axis in "xyz"] for leg in LEGS}}

    for record in traces.records:
        values = traces.read(record)
        demand, applied, speed, post_speed, limit = [traces.block(values, key, names) for key in ("demand", "applied", "pre_qd", "post_qd", "instantaneous_limit")]
        q0, q1 = [traces.block(values, key, joints).astype(np.float64) for key in ("pre_q", "post_q")]
        vel = traces.block(values, "post_qd", joints)
        residual = passive_residual(vel, joints, kin)
        gap = np.linalg.norm(traces.vectors(values, "hinge_gap_local"), axis=-1)
        pin_velocity = np.linalg.norm(traces.vectors(values, "hinge_relative_point_velocity_local"), axis=-1)
        foot_force = np.linalg.norm(traces.vectors(values, "foot_force_w"), axis=-1)
        dynamics = motor_dynamics(speed, demand, applied, limit, no_load)
        fd = (q1-q0)/dt
        pd_reconstruction = (traces.block(values, "p_term", names)+traces.block(values, "d_term", names)
                             +traces.block(values, "feedforward", names))
        for label, array, labels in (("peak_raw_demand_nm", demand, names), ("peak_applied_torque_nm", applied, names),
            ("peak_proportional_term_nm", traces.block(values, "p_term", names), names),
            ("peak_derivative_term_nm", traces.block(values, "d_term", names), names),
            ("peak_raw_minus_recorded_P_D_feedforward_nm", demand-pd_reconstruction, names),
            ("peak_cached_minus_native_pre_speed_rad_s", traces.block(values, "pre_qd", joints)-traces.block(values, "direct_pre_qd", joints), joints),
            ("peak_pre_step_speed_rad_s", speed, names), ("peak_post_step_speed_rad_s", post_speed, names),
            ("peak_raw_minus_applied_abs_nm", dynamics["raw_minus_applied_abs_nm"], names),
            ("peak_C_pin_gap_m", gap, LEGS), ("peak_C_pin_relative_speed_m_s", pin_velocity, LEGS),
            ("peak_passive_velocity_residual_rad_s", residual, list(kin["passive_relations"])),
            ("peak_interval_average_minus_post_instantaneous_speed_rad_s", fd-vel, joints)):
            keep(label, array, record, values, labels)
        for label, condition, labels in (("first_pre_speed_above_no_load_with_opposing_demand", dynamics["overspeed_with_opposing_demand"], names),
            ("first_raw_demand_above_vendor_peak", np.abs(demand) > peak_cap, names),
            ("first_raw_applied_difference_above_10uNm", dynamics["raw_minus_applied_abs_nm"] > 1e-5, names),
            ("first_zero_feet_above_1N", (np.sum(foot_force > 1., axis=-1) == 0)[..., None], ["all_feet"])):
            keep(label, condition, record, values, labels, first=True)
        for leg, env in worst_envs.items():
            name = leg+"_tibia_lever_pivot"; motor = names.index(name)
            peak = int(np.abs(demand[:, env, motor]).argmax())
            segment_rows.append({"motor": name, "environment": env, "segment": record["segment"],
                "physics_interval_half_open": [record["first_physics_sample"], record["first_physics_sample"]+len(values)],
                "max_abs_raw_demand_nm": float(np.abs(demand[:, env, motor]).max()),
                "max_abs_applied_nm": float(np.abs(applied[:, env, motor]).max()),
                "max_abs_pre_speed_rad_s": float(np.abs(speed[:, env, motor]).max()),
                "minimum_headroom": float(traces.block(values, "headroom", [name])[:, env, 0].min()),
                "minimum_supported_feet_above_1N": int((foot_force[:, env] > 1.).sum(-1).min()),
                "samples": [snapshot(values, s, env, record) for s in sorted({0, peak, len(values)-1})]})
    return sorted(events.values(), key=lambda row: (row["physics_sample"], row["event"])), segment_rows


def analyze(report_path, baseline_path, kinematics_path):
    paths = {"replay_report": Path(report_path).resolve(), "campaign008_report": Path(baseline_path).resolve(), "kinematics": Path(kinematics_path).resolve()}
    report, baseline, kin = [json.loads(paths[key].read_text()) for key in paths]
    if (report.get("diagnostic_motion") != "validation_prefix" or report.get("diagnostic_complete") is not True
            or report.get("pass") is not False or report.get("simulation_training_admission") is not False
            or report.get("hardware_admission") is not False or report.get("errors") != [] or report.get("num_envs") != 32
            or report.get("steps_completed") != 1000 or report.get("driven_steps_completed") != 1500):
        raise ValueError("Require complete nonadmitting 32-environment prefix report")
    recipe = report["numerical_recipe"]
    if (report["physics_substeps"] != 2500*recipe["decimation"] or report["force_writes"] != report["physics_substeps"]
            or report["physics_samples_observed"] != report["physics_substeps"]
            or report["trace_samples"] != 300*recipe["decimation"] or report["control_trace_samples"] != 2500):
        raise ValueError("Full-rate metrics and sparse trace counters disagree")
    if digest(paths["kinematics"]) != report["runtime_manifest"]["kinematics_sha256"]:
        raise ValueError("Kinematic relation file is not the executed model")
    if report["active_motor_names"] != kin["active_joint_names"] or len(set(report["joint_names"])) != 30:
        raise ValueError("Recorded motor/tree identities differ")
    runtime_diff = differences(report["runtime_manifest"], baseline["runtime_manifest"])
    placement_diff = differences(report["reset_root_positions_m"], baseline["reset_root_positions_m"])
    source_diff = differences(report["contract"]["files"], baseline["contract"]["files"])
    controls = TraceSet(report_path, report, "control")
    arrays, boundary_events = [], {}
    for record in controls.records:
        values = controls.read(record)
        arrays.append(controls.block(values, "joint_pos", report["active_motor_names"]))
        control_events(controls, values, record, report, kin, boundary_events)
    control_q = np.concatenate(arrays)
    phases = phase_analysis(control_q, report, baseline)
    worst_envs = {leg: phases[12+i]["worst_env_index"] for i, leg in enumerate(LEGS[:3])}
    traces = TraceSet(report_path, report, "trace")
    events, trajectories = detailed_analysis(traces, report, kin, worst_envs)
    return {"schema": "hexapod.motion_prefix_offline_analysis.v1", "analysis_complete": True,
        "simulation_training_admission": False, "hardware_admission": False,
        "inputs": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
        "analyzer_sha256": digest(__file__), "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
        "trace_files": report["trace_files"], "control_trace_files": report["control_trace_files"],
        "comparability": {"exact_runtime_match": not runtime_diff, "runtime_differences": runtime_diff,
            "exact_ordered_32_reset_positions_match": not placement_diff, "placement_differences": placement_diff,
            "source_file_differences": source_diff, "replay_source_sha256": report["contract"]["sha256"],
            "campaign008_source_sha256": baseline["contract"]["sha256"],
            "interpretation": "Source identities may differ for diagnostic instrumentation; every runtime/asset/placement field is compared without exclusions."},
        "phase_reduction": "Last five control endpoints (45..49), torch float32 mean and subtraction on CPU; any CPU/GPU reduction roundoff is reported explicitly.",
        "motor_responses": phases, "worst_lever_environments": {leg: {"environment": env, "reset_xyz_m": report["reset_root_positions_m"][env]} for leg, env in worst_envs.items()},
        "earliest_cross_run_control_divergence": {"available": False, "reason": "Campaign008 has only global response minima, not per-control traces. No exact earliest cross-run divergence can be recovered."},
        "chronological_control_boundary_events": sorted(boundary_events.values(), key=lambda row: (row["control_step"], row["event"])),
        "chronological_detailed_events": events, "worst_environment_segment_trajectories": trajectories,
        "limitations": ["Only controls2200..2499 have detailed800Hz traces; earlier detailed onset is unobserved.",
            "Events are timestamped independently; peaks do not imply coincidence or a unique cause.",
            "q finite difference is an interval-average velocity, not the reported instantaneous endpoint velocity.",
            "Control telemetry aligns post-control qdot with the preceding final-substep torque; only detailed pre_qd aligns exactly with the PD demand input.",
            "480rpm is the model's vendor no-load speed point, not an established hard braking-speed capability.",
            "Opposing raw demand is braking intent; applied torque and the provisional symmetric envelope determine delivered braking.",
            "1N foot support and10uNm clipping selectors label events only; this analyzer grants no physical admission."]}


def markdown(result):
    lines = ["# Individual-prefix replay analysis", "", "Offline diagnostic evidence; no training or hardware admission.", "",
        f"Exact runtime match: **{result['comparability']['exact_runtime_match']}**. Exact ordered32 reset positions: **{result['comparability']['exact_ordered_32_reset_positions_match']}**.", "",
        "| Motor | Replay minimum (rad) | Campaign008 minimum (rad) | Difference | Worst env |", "|---|---:|---:|---:|---:|"]
    for row in result["motor_responses"]:
        lines.append(f"| {row['motor']} | {row['minimum_rad']:.9g} | {row['campaign008_minimum_rad']:.9g} | {row['replay_minus_campaign008_minimum_rad']:.4g} | {row['worst_env_index']} |")
    lines += ["", "## Chronological detailed events", "", "| Physics sample | Control | Env | Event | Value |", "|---:|---:|---:|---|---:|"]
    for row in result["chronological_detailed_events"]:
        lines.append(f"| {row['physics_sample']} | {row['control_step']} | {row['environment']} | {row['event']} ({row['name']}) | {row['value']:.9g} |")
    lines += ["", "## Full-prefix control-boundary events", "", "| Control | Env | Event | Value |", "|---:|---:|---|---:|"]
    for row in result["chronological_control_boundary_events"]:
        lines.append(f"| {row['control_step']} | {row['environment']} | {row['event']} ({row['name']}) | {row['value']:.9g} |")
    lines += ["", "## Interpretation limits", "", result["earliest_cross_run_control_divergence"]["reason"], ""]
    lines += ["- "+item for item in result["limitations"]]
    lines += ["", "## Input identity", "", *[f"- {key}: `{row['sha256']}` — `{row['path']}`" for key, row in result["inputs"].items()], "",
              "Every NPZ name, hash, sample interval and column layout is verified; the JSON records every file hash and exact event neighborhood.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path); parser.add_argument("campaign008_report", type=Path)
    parser.add_argument("--kinematics", type=Path, default=ROOT/"configs/mkii_fourbar_v3_kinematics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise ValueError("Output directory must be new; preserve prior evidence")
    result = analyze(args.report, args.campaign008_report, args.kinematics)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.output_dir/"README.md").write_text(markdown(result))


if __name__ == "__main__": main()
