#!/usr/bin/env python3
"""Find reset candidates under the same geometric/torque rules for every size.

This is a static prescreen, never a substitute for Isaac standing validation.
It writes a separate training input; the generated URDFs remain unchanged.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import generate_length_study as study


def main():
    package = study.DEFAULT_CONFIG.parent
    config = json.loads(study.DEFAULT_CONFIG.read_text())
    manifest = json.loads((package / "manifest.json").read_text())
    base, mapping, _, clouds = study.prepare(config)
    for name, points in list(clouds.items()):
        groups = [points]
        if name.startswith("tibia"):
            distal = points[:, 1] > config["distal_foot_fraction"] * points[:, 1].max()
            groups = [points[distal], points[~distal]]
        clouds[name] = np.concatenate([g[ConvexHull(g).vertices] for g in groups])
    result = {
        "method": "same static stance search for every morphology; dynamic admission still required",
        "physics_dt_s": 0.0025, "decimation": 8,
        "training_num_envs": 256, "training_iterations": 500,
        "training_seed": 57, "evaluation_seed": 7057,
        "evaluation_forward_speeds_mps": [0.10, 0.20, 0.30],
        "validation_num_envs": 32, "validation_control_steps": 1000,
        "evaluation_num_envs": 64, "evaluation_control_steps_per_speed": 1000,
        "ranking": "Provisional single-seed screen. Compare falls, contact, tracking, tilt, slip, torque duty and positive mechanical power; do not equate reward with hardware optimality.",
        "variants": {},
    }
    for record in manifest["variants"]:
        sf, st = record["femur_scale"], record["tibia_scale"]
        robot = study.make_variant(base, mapping, sf, st)
        candidates = []
        for f in range(15, 66, 5):
            for t in range(80, 126, 5):
                cfg = copy.deepcopy(config)
                cfg["screening_pose_mock_rad"].update(femur=math.radians(f), tibia=math.radians(t))
                screen = study.screen(robot, mapping, clouds, cfg, (sf, st))
                if (screen["six_foot_geometry_eligible"]
                    and screen["nonfoot_mesh_vertex_clearance_m"] >= 0.005
                    and screen["root_height_at_contact_m"] >= 0.070
                    and screen["max_abs_hold_torque_nm"] <= 1.3):
                    candidates.append({"femur_deg": f, "tibia_deg": t, **screen})
        candidates.sort(key=lambda c: (c["max_abs_hold_torque_nm"], abs(c["femur_deg"]-35)+abs(c["tibia_deg"]-125)))
        # Keep distinct alternatives rather than three nearly identical poses.
        picked = []
        for c in candidates:
            if all(abs(c["femur_deg"]-p["femur_deg"])+abs(c["tibia_deg"]-p["tibia_deg"]) >= 10 for p in picked):
                picked.append(c)
            if len(picked) == 3:
                break
        result["variants"][record["variant"]] = {"urdf_sha256": record["sha256"], "stances": picked}
        print(record["variant"], [(c["femur_deg"], c["tibia_deg"], round(c["max_abs_hold_torque_nm"],3)) for c in picked], flush=True)
    study.write_json(package / "training_plan.json", result)


if __name__ == "__main__":
    main()
