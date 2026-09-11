#!/usr/bin/env python3
"""Summarize completed matched evaluations without inventing a weighted score."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import json
import math
from pathlib import Path


METRICS = ("abs_forward_error_mps", "tilt_rms_deg", "distal_slip_speed_mps",
           "saturation_fraction", "positive_mechanical_power_w")


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/evaluate/evaluation.json")):
        data = json.loads(path.read_text())
        speeds = data.get("speeds", [])
        if not data.get("complete") or [v["command_mps"] for v in speeds] != [0.1, 0.2, 0.3]:
            continue
        if not all(math.isfinite(v[k]) for v in speeds for k in METRICS):
            continue
        row = {"variant": data["variant"], "evaluation":str(path),
               **{k:sum(v[k] for v in speeds)/len(speeds) for k in METRICS},
               "worst_speed_fall_fraction":max(v["fall_fraction"] for v in speeds),
               "worst_speed_nonfoot_contact_fraction":max(v["nonfoot_contact_fraction"] for v in speeds),
               "worst_speed_saturation_fraction":max(v["saturation_fraction"] for v in speeds),
               "tracks_all_test_speeds":all(v["abs_forward_error_mps"]<=max(0.03,0.25*v["command_mps"]) for v in speeds)}
        row["meets_screening_constraints"] = (row["worst_speed_fall_fraction"]==0
            and row["worst_speed_nonfoot_contact_fraction"]==0
            and row["worst_speed_saturation_fraction"]<=0.005
            and row["tracks_all_test_speeds"])
        rows.append(row)
    safe = [r for r in rows if r["meets_screening_constraints"]]
    for row in rows:
        row["pareto_candidate"] = row in safe and not any(
            all(other[k]<=row[k] for k in METRICS) and any(other[k]<row[k] for k in METRICS)
            for other in safe)
    rows.sort(key=lambda r:(not r["meets_screening_constraints"],not r["pareto_candidate"],
                            r["worst_speed_fall_fraction"],r["abs_forward_error_mps"]))
    result = {"kind":"provisional_single_seed_morphology_screen", "completed_evaluations":len(rows),
              "all_49_evaluated":len(rows)==49,
              "pareto_candidates":[r["variant"] for r in rows if r["pareto_candidate"]],"variants":rows}
    (root/"ranking.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    print(json.dumps(summarize(parser.parse_args().root),indent=2))
