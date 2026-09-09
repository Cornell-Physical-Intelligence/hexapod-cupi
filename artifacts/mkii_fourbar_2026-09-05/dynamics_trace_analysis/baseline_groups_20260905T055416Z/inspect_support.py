"""Reproduce the baseline reversal force/velocity supplement without Isaac or GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(report_path):
    report = json.loads(report_path.read_text())
    record = report["trace_files"][6]
    assert record["segment"]["offset_rad"] == -.04
    assert record["segment"]["motors"] == [f"{leg}_tibia_lever_pivot" for leg in ("lf", "lm", "lr", "rf", "rm", "rr")]
    trace_path = report_path.parent / record["file"]
    if sha(trace_path) != record["sha256"]:
        raise ValueError("Trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as data:
        values = data["values"].astype(np.float64)
        columns = data["columns"].tolist()
    assert list(values.shape) == record["shape"] and len(columns) == len(set(columns)) and np.isfinite(values).all()
    dt = report["numerical_recipe"]["physics_dt_s"]
    legs = ("lf", "lm", "lr", "rf", "rm", "rr")

    def field(key, names):
        return values[:, :, [columns.index(f"{key}/{name}") for name in names]]

    force = field("foot_force_w", [f"{leg}_{a}" for leg in legs for a in "xyz"]).reshape(len(values), -1, 6, 3)
    norm = np.linalg.norm(force, axis=-1)
    root_vel = field("body_link_linear_velocity_w", [f"body_{a}" for a in "xyz"])
    tibia_vel = field("body_link_linear_velocity_w", [f"{leg}_tibia_{a}" for leg in legs for a in "xyz"]).reshape(len(values), -1, 6, 3)
    events = []
    for local, env in np.argwhere((norm > 1.).sum(-1) == 0):
        events.append({"sample": int(record["first_physics_sample"]+local), "environment": int(env),
            "post_time_s": (record["first_physics_sample"]+int(local)+1)*dt,
            "foot_force_w_n": dict(zip(legs, force[local, env].tolist())),
            "body_link_linear_velocity_w_m_s": root_vel[local, env].tolist(),
            "tibia_link_origin_linear_velocity_w_m_s": dict(zip(legs, tibia_vel[local, env].tolist()))})
    # Save adjacent force/velocity observations; no position integration is asserted.
    window = []
    for local in range(12, 32):
        window.append({"sample": record["first_physics_sample"]+local,
            "foot_force_w_n_by_environment_leg_xyz": force[local].tolist(),
            "body_link_linear_velocity_w_m_s_by_environment_xyz": root_vel[local].tolist()})
    peak = np.unravel_index(np.argmax(norm), norm.shape)
    peak_force = force[peak]
    return {"schema": "hexapod.baseline_support_supplement.v1", "derived_evidence_only": True,
        "report_sha256": sha(report_path), "trace_file": record, "script_sha256": sha(Path(__file__)),
        "legs": legs, "threshold_loss_counts_entire_negative_lever_segment": {
            str(threshold): int(np.count_nonzero((norm > threshold).sum(-1) == 0)) for threshold in (0., .1, .5, 1., 2.)},
        "zero_support_events_above_1n": events,
        "peak_individual_foot_force": {"sample": int(record["first_physics_sample"]+peak[0]), "environment": int(peak[1]),
            "leg": legs[peak[2]], "force_w_n": peak_force.tolist(), "norm_n": float(norm[peak]),
            "horizontal_to_vertical_magnitude_ratio": float(np.linalg.norm(peak_force[:2])/abs(peak_force[2])),
            "force_times_dt_n_s": (peak_force*dt).tolist()},
        "adjacent_sample_window": window,
        "limitations": ["This baseline has no saved body pose, foot centre or ground clearance. Force-zero samples cannot prove geometric flight.",
            "Tibia link-origin velocity is not foot contact-point velocity.",
            "Contact force times dt is the reported force integrated over one simulation sample; it is not an independently measured physical impulse.",
            "Threshold variants are diagnostic counts only; the acceptance gate remains unchanged."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.report)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
