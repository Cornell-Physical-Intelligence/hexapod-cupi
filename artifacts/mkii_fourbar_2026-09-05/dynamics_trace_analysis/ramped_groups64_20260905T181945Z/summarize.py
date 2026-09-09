"""Verify actual substep target delivery and compact the ramped group diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ramp_metrics(delivered, endpoints, initial):
    """Compare recorded targets against a 16-sample linear delivery contract."""
    if (delivered.shape != endpoints.shape or delivered.ndim != 4
            or delivered.shape[1] != 16 or initial.shape != delivered.shape[2:]
            or not all(np.isfinite(x).all() for x in (delivered, endpoints, initial))):
        raise ValueError("Invalid named target trace layout")
    start = np.concatenate((initial[None], delivered[:-1, -1]), axis=0)
    end = endpoints[:, -1]
    fractions = (np.arange(16)+1)[None, :, None, None]/16.
    expected = start[:, None]+fractions*(end-start)[:, None]
    previous = np.concatenate((start[:, None], delivered[:, :-1]), axis=1)
    residual = np.abs(delivered-expected)
    statistics = {
        "target_values_checked": int(delivered.size),
        "control_environment_intervals": int(delivered.shape[0]*delivered.shape[2]),
        "max_linear_interpolation_residual_rad": float(residual.max()),
        "residuals_above_1e_7_rad": int(np.count_nonzero(residual>1e-7)),
        "max_endpoint_change_within_control_rad": float(np.abs(endpoints-end[:, None]).max()),
        "max_last_substep_minus_endpoint_rad": float(np.abs(delivered[:, -1]-end).max()),
        "max_delivered_substep_increment_rad": float(np.abs(delivered-previous).max()),
        "max_control_endpoint_increment_rad": float(np.abs(end-start).max()),
        "max_delivered_minus_endpoint_rad": float(np.abs(delivered-end[:, None]).max()),
        "moving_control_environment_motor_intervals": int(np.count_nonzero(np.abs(end-start)>1e-6)),
    }
    statistics["pass"] = bool(
        statistics["residuals_above_1e_7_rad"] == 0
        and statistics["max_endpoint_change_within_control_rad"] == 0
        and statistics["max_last_substep_minus_endpoint_rad"] == 0
        and statistics["max_delivered_substep_increment_rad"] <= .0025+1e-7
        and statistics["max_control_endpoint_increment_rad"] <= .04+1e-7)
    return statistics


def summarize(report_path, analysis_path, supervisor_path, source_manifest_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import analyze
    report = json.loads(report_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    supervisor = json.loads(supervisor_path.read_text())
    assert report["diagnostic_complete"] and report["diagnostic_motion"] == "groups"
    assert analysis["integrity_verified"] and analysis["report_sha256"] == sha(report_path)
    assert supervisor["contract"] == report["contract"]
    assert supervisor["source_manifest"]["sha256"] == sha(source_manifest_path)
    manifest = {}
    for line in source_manifest_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in manifest:
            raise ValueError("Duplicate source-manifest path")
        manifest[name] = digest
    assert len(manifest) == supervisor["source_manifest"]["files"]
    assert all(manifest[name] == digest for name, digest in report["contract"]["files"].items())
    traces = analyze.Traces(report_path)
    assert traces.decimation == 16 and traces.dt == .00125
    previous = None
    per_segment, examples, groups = [], [], []
    zero_velocity_max, feedforward_max = 0., 0.
    for index, record in enumerate(traces.files):
        values, _ = traces.read(index)
        target = traces.block(values, "target", traces.motors)
        endpoint = traces.block(values, "processed_target", traces.motors)
        if previous is None:
            previous = traces.block(values, "pre_q", traces.motors)[0]
        controls = len(values)//16
        target, endpoint = [x.reshape(controls, 16, traces.envs, 18) for x in (target, endpoint)]
        metrics = ramp_metrics(target, endpoint, previous)
        per_segment.append(dict(segment_index=index, first_physics_sample=record["first_physics_sample"], **metrics))
        start = np.concatenate((previous[None], target[:-1, -1]), axis=0)
        for control in range(min(2, controls)):
            delta = endpoint[control, -1, 0]-start[control, 0]
            motor = int(np.argmax(np.abs(delta)))
            if abs(delta[motor]) > 1e-6:
                examples.append({"segment_index": index, "segment_control_step": control,
                    "first_physics_sample": record["first_physics_sample"]+control*16,
                    "environment": 0, "motor": traces.motors[motor],
                    "start_rad": float(start[control, 0, motor]), "endpoint_rad": float(endpoint[control, -1, 0, motor]),
                    "delivered_target_each_substep_rad": target[control, :, 0, motor].tolist(),
                    "endpoint_each_substep_rad": endpoint[control, :, 0, motor].tolist()})
        previous = target[-1, -1]
        zero_velocity_max = max(zero_velocity_max, float(np.abs(traces.block(values, "velocity_target", traces.motors)).max()))
        feedforward_max = max(feedforward_max, float(np.abs(traces.block(values, "feedforward", traces.motors)).max()))
        if index in range(1, 7):
            assert record["segment"]["steps"] == 100
            names = record["segment"]["motors"]
            positions = traces.block(values, "post_q", names)
            groups.append((names, positions[np.arange(90,100)*16+15].mean(0)))
    responses = {}
    for (names, positive), (negative_names, negative) in zip(groups[::2], groups[1::2]):
        assert names == negative_names
        for index, name in enumerate(names):
            delta = positive[:, index]-negative[:, index]
            responses[name] = {"positive_minus_negative_rad_by_environment": delta.tolist(),
                "minimum_rad": float(delta.min()), "minimum_environment": int(delta.argmin()),
                "above_original_0_005_rad_direction_threshold": bool((delta>.005).all())}
    events = {}
    for key, event in analysis["events"].items():
        state = event["named_state"]
        events[key] = {name: event[name] for name in ("peak", "context", "anatomical_context")}
        events[key]["motor_state"] = {field: {name: state[field][name] for name in traces.motors}
            for field in ("pre_q", "pre_qd", "post_q", "post_qd", "target", "processed_target", "p_term", "d_term", "demand", "applied")}
    return {"schema": "hexapod.ramped_groups_trace_summary.v1", "derived_evidence_only": True,
        "report_sha256": sha(report_path), "analysis_sha256": sha(analysis_path), "analysis_path": str(analysis_path),
        "supervisor_sha256": sha(supervisor_path), "source_manifest_sha256": sha(source_manifest_path),
        "source_commit": supervisor["source_commit"], "source_contract_sha256": report["contract"]["sha256"],
        "source_manifest_files": len(manifest), "contract_files_matched_to_source_manifest": len(report["contract"]["files"]),
        "source_archive_metadata": supervisor["source_archive"],
        "summarizer_sha256": sha(Path(__file__)), "analyzer_sha256": sha(Path(analyze.__file__)),
        "integrity_verified": True, "numerical_recipe": analysis["numerical_recipe"],
        "trace_files": analysis["trace_files"], "global_metrics": analysis["global_metrics"],
        "segments": analysis["segments"], "readbacks": analysis["readbacks"], "events": events,
        "ramp_delivery_pass": all(x["pass"] for x in per_segment) and zero_velocity_max == 0 and feedforward_max == 0,
        "ramp_delivery_by_segment": per_segment, "ramp_delivery_examples": examples,
        "maximum_velocity_target_rad_s": zero_velocity_max, "maximum_feedforward_nm": feedforward_max,
        "group_response_measurement": "Mean post-q after each of control steps90..99 for each sign, then positive minus negative per environment",
        "group_responses": responses, "all_group_directions_above_0_005_rad": all(x["above_original_0_005_rad_direction_threshold"] for x in responses.values()),
        "limitations": ["This is an 8-environment group diagnostic, not the full qualification or a hardware test.",
            "The source archive is retained remotely; its metadata and source hash list are copied here, not the archive.",
            "Linear-delivery residual allowance 1e-7 rad covers float32 target rounding; exact endpoint equality and unchanged within-control endpoints are checked separately."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "analysis", "supervisor", "source_manifest", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    value = summarize(args.report, args.analysis, args.supervisor, args.source_manifest)
    with args.out.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"ramp_delivery_pass": value["ramp_delivery_pass"],
        "all_group_directions_above_0_005_rad": value["all_group_directions_above_0_005_rad"],
        "minimum_group_direction_rad": min(x["minimum_rad"] for x in value["group_responses"].values()),
        "maximum_linear_delivery_residual_rad": max(x["max_linear_interpolation_residual_rad"] for x in value["ramp_delivery_by_segment"])}))
