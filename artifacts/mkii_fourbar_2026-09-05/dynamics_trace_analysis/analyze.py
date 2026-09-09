"""Verify and summarize named pre-force/post-physics diagnostic traces offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TREE_FIELDS = ("pre_q", "pre_qd", "direct_pre_q", "direct_pre_qd", "post_q", "post_qd")
MOTOR_FIELDS = ("target", "processed_target", "velocity_target", "feedforward", "p_term", "d_term",
                "demand", "applied", "instantaneous_limit", "headroom")
HINGE_FIELDS = ("hinge_gap_local", "hinge_axis_difference_local", "hinge_relative_point_velocity_local")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {label}")
    return value


def unique_names(values, label):
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"Missing names: {label}")
    if len(values) != len(set(values)):
        raise ValueError(f"Repeated names: {label}")
    return values


class Traces:
    def __init__(self, report_path):
        self.path = Path(report_path).resolve()
        self.report_hash = digest(self.path)
        self.report = json.loads(self.path.read_text())
        r = self.report
        if r.get("schema") != "hexapod.fourbar_diagnostic.v1":
            raise ValueError("Unsupported diagnostic report schema")
        self.envs = integer(r["num_envs"], "num_envs", 1)
        self.joints = unique_names(r["joint_names"], "joint_names")
        self.motors = unique_names(r["active_motor_names"], "active_motor_names")
        if not set(self.motors) <= set(self.joints):
            raise ValueError("Active motor absent from tree joint names")
        recipe = r["numerical_recipe"]
        if type(recipe["physics_dt_s"]) not in (int, float):
            raise ValueError("Invalid physics timestep type")
        self.dt = float(recipe["physics_dt_s"])
        self.decimation = integer(recipe["decimation"], "decimation", 1)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid physics timestep")
        self.files = r["trace_files"]
        if not isinstance(self.files, list) or not self.files:
            raise ValueError("No trace files")
        self.columns = None
        self.fields = {}
        cursor, seen = 0, set()
        for index, record in enumerate(self.files):
            name = record["file"]
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("Trace path must be a local NPZ basename")
            path = (self.path.parent/name).resolve()
            if path.parent != self.path.parent or name in seen:
                raise ValueError("Repeated or escaped trace path")
            seen.add(name)
            if integer(record["first_physics_sample"], "first sample") != cursor:
                raise ValueError("Trace ranges have a gap, overlap, or wrong ordering")
            values, columns = self.read(index, check_columns=False)
            if self.columns is None:
                self.columns = columns
                for column, label in enumerate(columns):
                    if label.count("/") != 1 or not all(label.split("/")):
                        raise ValueError("Invalid key/name trace column")
                    key, field_name = label.split("/")
                    self.fields.setdefault(key, {})[field_name] = column
            elif columns != self.columns:
                raise ValueError("Trace column layout changed between segments")
            requested = integer(record["segment"]["steps"], "segment steps", 1)*self.decimation
            if values.shape[0] > requested:
                raise ValueError("Segment contains more physics samples than requested")
            if (r.get("diagnostic_complete") is True or index < len(self.files)-1) and values.shape[0] != requested:
                raise ValueError("A completed segment has missing physics samples")
            cursor += values.shape[0]
        self.samples = cursor
        if integer(r["trace_samples"], "trace_samples") != cursor:
            raise ValueError("Report sample total differs from trace ranges")
        writes = integer(r["force_writes"], "force_writes")
        physical = integer(r["physics_substeps"], "physics_substeps")
        if r.get("diagnostic_complete") is True:
            completed = integer(r["steps_completed"], "steps_completed") + integer(r["driven_steps_completed"], "driven_steps_completed")
            if writes != cursor or physical != cursor or completed*self.decimation != cursor:
                raise ValueError("Completed report has inconsistent force/control/physics coverage")
        elif not (cursor <= writes <= cursor+1 and physical <= cursor):
            raise ValueError("Partial report has inconsistent coverage")
        for key in TREE_FIELDS:
            self.require_names(key, self.joints)
        for key in MOTOR_FIELDS:
            self.require_names(key, self.motors)
        self.legs = self.xyz_groups("hinge_gap_local")
        for key in (*HINGE_FIELDS, "foot_force_w"):
            if self.xyz_groups(key) != self.legs:
                raise ValueError("Hinge/foot vector identities differ")
        if self.xyz_groups("body_link_linear_velocity_w") != self.xyz_groups("body_link_angular_velocity_w"):
            raise ValueError("Body velocity identities differ")

    def read(self, index, check_columns=True):
        record = self.files[index]
        path = self.path.parent/record["file"]
        if not isinstance(record["shape"], list) or len(record["shape"]) != 3:
            raise ValueError("Invalid trace shape metadata")
        for value in record["shape"]:
            integer(value, "shape dimension", 1)
        if digest(path) != record["sha256"]:
            raise ValueError(f"Trace SHA-256 mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"values", "columns"}:
                raise ValueError("Unexpected NPZ members")
            values, labels = data["values"], data["columns"]
            if labels.ndim != 1 or labels.dtype.kind not in "US":
                raise ValueError("Trace columns must be a string vector")
            columns = labels.tolist()
            unique_names(columns, "trace columns")
            if (values.ndim != 3 or values.dtype.kind != "f" or values.shape[0] < 1
                    or values.shape[1] != self.envs or values.shape[2] != len(columns)
                    or list(values.shape) != record["shape"] or not np.isfinite(values).all()):
                raise ValueError("Trace shape, dtype or finite-value check failed")
            if check_columns and columns != self.columns:
                raise ValueError("Trace layout changed")
            return values.astype(np.float64), columns

    def require_names(self, key, names):
        if set(self.fields.get(key, {})) != set(names):
            raise ValueError(f"Missing or unexpected names for {key}")

    def xyz_groups(self, key):
        names = self.fields.get(key, {})
        groups = sorted({name[:-2] for name in names if name.endswith(("_x", "_y", "_z"))})
        if not groups or set(names) != {name+"_"+axis for name in groups for axis in "xyz"}:
            raise ValueError(f"Incomplete XYZ vectors: {key}")
        return groups

    def block(self, values, key, names):
        return values[:, :, [self.fields[key][name] for name in names]]

    def vectors(self, values, key, names):
        return self.block(values, key, [name+"_"+axis for name in names for axis in "xyz"]).reshape(
            values.shape[0], self.envs, len(names), 3)

    def context(self, sample):
        for index, record in enumerate(self.files):
            first = record["first_physics_sample"]
            if first <= sample < first+record["shape"][0]:
                local = sample-first
                return {"segment_index": index, "segment": record["segment"],
                        "physics_sample": sample, "segment_physics_sample": local,
                        "segment_control_step": local//self.decimation,
                        "substep_in_control": local % self.decimation,
                        "pre_time_s": sample*self.dt, "post_time_s": (sample+1)*self.dt}
        raise ValueError("Event sample outside trace ranges")


def metric(array, names, first):
    index = np.unravel_index(int(np.argmax(np.abs(array))), array.shape)
    signed = float(array[index])
    return {"max_abs": abs(signed), "signed_value_at_max": signed,
            "name": names[index[2]], "physics_sample": int(first+index[0]), "environment": int(index[1]),
            "rms": float(np.sqrt(np.mean(array*array))), "count": array.size}


def summarize(t, values, record):
    first = record["first_physics_sample"]
    results = {}
    def add(key, array, names):
        results[key] = metric(array, names, first)
    for key in ("demand", "applied", "p_term", "d_term", "feedforward"):
        add(key, t.block(values, key, t.motors), t.motors)
    preq, postq = [t.block(values, key, t.joints) for key in ("pre_q", "post_q")]
    preqd, postqd = [t.block(values, key, t.joints) for key in ("pre_qd", "post_qd")]
    dqdt = (postq-preq)/t.dt
    for key, array in (("pre_qd", preqd), ("post_qd", postqd), ("finite_difference_dq_dt", dqdt),
                       ("post_qd_minus_dq_dt", postqd-dqdt), ("pre_qd_minus_dq_dt", preqd-dqdt),
                       ("cached_minus_direct_pre_q", preq-t.block(values, "direct_pre_q", t.joints)),
                       ("cached_minus_direct_pre_qd", preqd-t.block(values, "direct_pre_qd", t.joints))):
        add(key, array, t.joints)
    pieces = [t.block(values, key, t.motors) for key in ("p_term", "d_term", "feedforward", "demand")]
    residual = pieces[0]+pieces[1]+pieces[2]-pieces[3]
    add("p_plus_d_plus_ff_minus_demand", residual, t.motors)
    add("target_minus_processed_target", t.block(values, "target", t.motors)-t.block(values, "processed_target", t.motors), t.motors)
    rounding = 8*np.finfo(np.float32).eps*(1+sum(np.abs(x) for x in pieces))
    hinges = {}
    for key in HINGE_FIELDS:
        vectors = t.vectors(values, key, t.legs)
        norms = np.linalg.norm(vectors, axis=-1)
        add(key+"_norm", norms, t.legs)
        for axis_index, axis in enumerate("xyz"):
            add(key+"_"+axis, vectors[:, :, :, axis_index], t.legs)
        hinges[key] = {leg: {"max_norm": float(norms[:, :, i].max()),
                              "max_abs_xyz": np.max(np.abs(vectors[:, :, i]), axis=(0, 1)).tolist()}
                       for i, leg in enumerate(t.legs)}
    forces = t.vectors(values, "foot_force_w", t.legs)
    return {"segment": record["segment"], "first_physics_sample": first,
            "physics_samples": values.shape[0], "environment_count": t.envs,
            "metrics": results, "hinges": hinges,
            "p_d_ff_reconstruction_outside_float32_rounding_count": int(np.count_nonzero(np.abs(residual)>rounding)),
            "minimum_supported_feet_above_1n": int((np.linalg.norm(forces, axis=-1)>1.).sum(-1).min())}


def readbacks(t):
    r, warnings = t.report, []
    result = {"actuator_backend_flags": r.get("actuator_backend_flags"),
              "scene_force_iteration_readback": r.get("scene_force_iteration_readback")}
    model_names = r.get("runtime_manifest", {}).get("observed_motor_model_joint_names")
    for key, names in (("solver_readback", t.joints), ("motor_readback", model_names)):
        if not names:
            result[key] = {"available_by_name": False, "raw": r.get(key)}
            warnings.append(f"{key} joint order unavailable; do not assume canonical motor order")
            continue
        unique_names(names, key+" names")
        if key == "motor_readback" and set(names) != set(t.motors):
            raise ValueError("Motor readback names differ from active motors")
        rows = {}
        for field, raw in r.get(key, {}).items():
            array = np.asarray(raw, dtype=float)
            if array.shape == (len(names),):
                array = array[None]
            if array.ndim != 2 or array.shape[1] != len(names) or array.shape[0] not in (1, t.envs) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key}/{field} readback shape")
            rows[field] = {name: array[:, index].tolist() for index, name in enumerate(names)}
        result[key] = {"available_by_name": bool(rows), "joint_order": names, "values_by_name": rows}
    result["configured_motor_parameters"] = r.get("runtime_manifest", {}).get("resolved_motor_configuration")
    return result, warnings


def event(t, peak, window):
    sample, env = int(peak["physics_sample"]), int(peak["environment"])
    context = t.context(sample)
    values, _ = t.read(context["segment_index"])
    frame = values[context["segment_physics_sample"], env]
    named = {key: {name: float(frame[index]) for name, index in names.items()} for key, names in t.fields.items()}
    leg = peak["name"] if peak["name"] in t.legs else peak["name"].split("_")[0]
    anatomical = {"associated_leg": leg if leg in t.legs else None,
                  "associated_motor_names": [name for name in t.motors if name.startswith(leg+"_")],
                  "foot_net_forces_w_by_body": {name+"_tibia": [named["foot_force_w"][name+"_"+axis] for axis in "xyz"]
                                                for name in t.legs},
                  "association_is_anatomical_not_causal": True}
    start, end = max(0, sample-window), min(t.samples, sample+window+1)
    rows = []
    for index, record in enumerate(t.files):
        first, count = record["first_physics_sample"], record["shape"][0]
        low, high = max(first, start), min(first+count, end)
        if low >= high:
            continue
        block, _ = t.read(index)
        rows.extend({"physics_sample": current, "segment_index": index,
                     "values": block[current-first, env].tolist()} for current in range(low, high))
    return {"peak": peak, "context": context, "anatomical_context": anatomical, "named_state": named,
            "window": {"requested_samples_each_side": window, "environment": env,
                       "first_physics_sample": start, "end_physics_sample_exclusive": end,
                       "values_use_top_level_trace_columns": True, "rows": rows}}


def analyze(report_path, window=8):
    integer(window, "window", 0)
    if window > 100:
        raise ValueError("Window must be at most100 samples each side")
    t = Traces(report_path)
    summaries = [summarize(t, t.read(index)[0], record) for index, record in enumerate(t.files)]
    global_metrics = {}
    for key in summaries[0]["metrics"]:
        items = [row["metrics"][key] for row in summaries]
        worst = max(items, key=lambda x: x["max_abs"])
        count = sum(row["count"] for row in items)
        global_metrics[key] = dict(worst, rms=math.sqrt(sum(row["rms"]**2*row["count"] for row in items)/count), count=count)
    configured, warnings = readbacks(t)
    events = {key: event(t, global_metrics[key], window) for key in
              ("demand", "applied", "hinge_gap_local_norm", "hinge_axis_difference_local_norm")}
    if digest(t.path) != t.report_hash:
        raise ValueError("Diagnostic report changed during analysis")
    if any(digest(t.path.parent/record["file"]) != record["sha256"] for record in t.files):
        raise ValueError("Trace changed during analysis")
    return {"schema": "hexapod.fourbar_trace_analysis.v1", "integrity_verified": True,
            "scope": "Offline numerical evidence only; no causal, physical-admission or hardware claim",
            "report_path": str(t.path), "report_sha256": t.report_hash,
            "source_diagnostic_complete": t.report.get("diagnostic_complete"),
            "source_errors": t.report.get("errors"), "source_physical_gate_errors": t.report.get("physical_gate_errors"),
            "diagnostic_usd": t.report.get("diagnostic_usd"), "usd_sha256": t.report.get("usd_sha256"),
            "numerical_recipe": t.report["numerical_recipe"], "trace_files": t.files,
            "trace_samples": t.samples, "environment_count": t.envs, "trace_columns": t.columns,
            "global_metrics": global_metrics, "segments": summaries, "events": events,
            "readbacks": configured, "warnings": warnings,
            "interpretation_limits": [
                "P,D,FF,demand and applied correspond to pre-step inputs; post state is measured after physics.",
                "Reported velocity need not equal position finite difference because PhysX separates position and velocity corrections.",
                "Direct backend getters execute after PD reads cached inputs, before physics; compare them without changing pre-step timing.",
                "Only foot net forces are traced, not all-body forces or individual contact impulses; body fields are velocities.",
                "Solver drive stiffness/damping may correctly be zero when the explicit motor model supplies nonzeroPD gains."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, help="New output JSON; omit to print it")
    parser.add_argument("--window", type=int, default=8, help="Physics samples each side of each peak")
    args = parser.parse_args()
    result = analyze(args.report, args.window)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.out:
        with args.out.open("x") as stream:
            stream.write(payload)
        print(json.dumps({"analysis": str(args.out), "integrity_verified": True,
                          "trace_samples": result["trace_samples"],
                          "peak_demand_nm": result["global_metrics"]["demand"]["max_abs"],
                          "peak_closure_m": result["global_metrics"]["hinge_gap_local_norm"]["max_abs"]}))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
