"""Compact the verified v5 group trace and reproduce control-boundary responses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(report_path, analysis_path, geometry_path):
    report = json.loads(report_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    assert report["diagnostic_motion"] == "groups" and report["diagnostic_complete"]
    assert analysis["integrity_verified"] and analysis["report_sha256"] == sha(report_path) == geometry["report_sha256"]
    assert geometry["zero_support_environment_samples"] == len(geometry["zero_support_events"])
    decimation = report["numerical_recipe"]["decimation"]
    responses = {}
    for positive_index in (1, 3, 5):
        measured = []
        for index, sign in ((positive_index, 1), (positive_index+1, -1)):
            record = report["trace_files"][index]
            assert record["segment"]["offset_rad"] == sign*.04 and record["segment"]["steps"] == 100
            trace_path = report_path.parent/record["file"]
            assert sha(trace_path) == record["sha256"]
            with np.load(trace_path, allow_pickle=False) as data:
                values, columns = data["values"], list(data["columns"])
                assert list(values.shape) == record["shape"]
                names = record["segment"]["motors"]
                # Match validator: q after each of control steps 90..99.
                indices = [columns.index("post_q/"+name) for name in names]
                final_physics_samples = np.arange(90, 100)*decimation+decimation-1
                measured.append(values[final_physics_samples][..., indices].astype(np.float64).mean(0))
        for column, name in enumerate(names):
            delta = measured[0][:, column]-measured[1][:, column]
            responses[name] = {"positive_minus_negative_rad_by_environment": delta.tolist(),
                "minimum_rad": float(delta.min()), "minimum_environment": int(delta.argmin()),
                "above_original_0_005_rad_direction_threshold": bool((delta>.005).all())}
    events = {}
    for key, event in analysis["events"].items():
        state = event["named_state"]
        events[key] = {name: event[name] for name in ("peak", "context", "anatomical_context")}
        events[key]["motor_state"] = {field: {name: state[field][name] for name in report["active_motor_names"]}
            for field in ("pre_q", "pre_qd", "post_q", "post_qd", "target", "p_term", "d_term", "demand", "applied", "instantaneous_limit")}
    return {"schema": "hexapod.v5_groups_compact_analysis.v1", "derived_evidence_only": True,
        "report_sha256": sha(report_path), "analysis_sha256": sha(analysis_path),
        "geometry_sha256": sha(geometry_path), "summarizer_sha256": sha(Path(__file__)),
        "analysis_path": str(analysis_path), "numerical_recipe": analysis["numerical_recipe"],
        "integrity_verified": analysis["integrity_verified"], "trace_files": analysis["trace_files"],
        "global_metrics": analysis["global_metrics"], "segments": analysis["segments"],
        "readbacks": analysis["readbacks"], "events": events,
        "group_response_measurement": "Mean post-q at the last physics sample of control steps90..99 for each sign, then positive minus negative per environment",
        "group_responses": responses, "all_group_directions_above_0_005_rad": all(x["above_original_0_005_rad_direction_threshold"] for x in responses.values()),
        "support_loss_environment_samples": geometry["zero_support_environment_samples"],
        "support_loss_duration_each_environment_s": report["numerical_recipe"]["physics_dt_s"],
        "support_loss_unique_samples": sorted({x["physics_sample"] for x in geometry["zero_support_events"]}),
        "limitations": ["No new admission; the unchanged support gate failed.",
            "This 8-environment group-only diagnostic omits the full qualification's preceding individual-motor history.",
            "Float64 reconstruction uses recorded float32 body poses and ideal primitive surfaces; micrometre-scale separation is not a hardware contact measurement."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "analysis", "geometry", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    value = summarize(args.report, args.analysis, args.geometry)
    with args.out.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"all_group_directions_above_0_005_rad": value["all_group_directions_above_0_005_rad"],
        "minimum_group_direction_rad": min(x["minimum_rad"] for x in value["group_responses"].values()),
        "support_loss_unique_samples": value["support_loss_unique_samples"]}))
