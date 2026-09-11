#!/usr/bin/env python3
"""Prepare separate single-C training inputs and a named-joint stepping reference."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from experiments.c_length_study.tools import screen_length_mechanics as s

out=s.ROOT/"artifacts/length_study_2026-09-09/candidate_c_training_inputs"
out.mkdir(exist_ok=True)
package=s.ROOT/"robot/hexapod_mkii_length_study"
manifest=json.loads((package/"manifest.json").read_text())
urdf=package/"urdf/f050_t060.urdf"
xml=ET.parse(urdf).getroot();robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,package))
poses=json.loads((out.parent/"mechanics_stage1_v3_static/f050_t060_poses.json").read_text())
pose=next(p for p in poses if p["femur_deg"]==40 and p["knee_deg"]==120)
t=s.trajectory(robot,pose,"tripod",.10,.02,256,speeds=(.2,))
assert t["workspace_pass"] and t["speed_results"][0]["minimum_required_peak_torque_nm"]<1.6
names=[manifest["link_joint_mapping"][leg]["joints"][kind] for leg in s.LEGS for kind in s.KINDS]
q=np.array([v["q"] for v in t["samples"]]).reshape(256,18)
reference={"variant":"f050_t060","urdf_sha256":hashlib.sha256(urdf.read_bytes()).hexdigest(),
           "joint_names":names,"phase_sample_convention":"(index+0.5)/count",
           "positions_rad":q.tolist(),"derivative_per_cycle":((np.roll(q,-1,axis=0)-np.roll(q,1,axis=0))*128).tolist(),
           "gait":"tripod","duty_factor":.65,"stance_travel_m":.10,"foot_lift_m":.02,
           "cycles_per_meter":6.5,"reference_speed_range_mps":[.10,.20],
           "stance":pose,"method":"position and velocity reference; actual base motion and forces come from torque-limited physics"}
(out/"candidate_c_reference.json").write_text(json.dumps(reference,indent=2)+"\n")
plan=json.loads((package/"training_plan.json").read_text())
stance={"femur_deg":40,"tibia_deg":120,"method":"user-selected lower C stance; mechanical audit, Isaac admission still required",
        "joint_positions_rad":dict(zip(names,np.array(pose["q"]).reshape(-1).tolist())),
        "root_height_at_contact_m":pose["root_height_m"],"suggested_reset_root_height_m":pose["root_height_m"]+.006,
        "belly_clearance_m":pose["belly_clearance_m"],"simulation_validated":False}
plan.update(method="single user-selected C; phase-guided residual PPO on full 19-body/18-joint robot",
            training_num_envs=1024,training_iterations=300,training_seed=57,
            evaluation_forward_speeds_mps=[.10,.20],evaluation_num_envs=32,
            reference_controller={"file":"candidate_c_reference.json","sha256":hashlib.sha256((out/"candidate_c_reference.json").read_bytes()).hexdigest(),
                                  "residual_scale_rad":.12,"warmup_s":2.0,"velocity_targets":True},
            variants={"f050_t060":{"urdf_sha256":reference["urdf_sha256"],"stances":[stance]}},
            ranking="One selected geometry. Completion requires actual walking metrics and verified video, not an update count.")
(out/"training_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
print(json.dumps({"out":str(out),"joints":len(names),"mass_kg":robot.mass,"root_height_m":pose["root_height_m"]}))
