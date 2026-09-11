#!/usr/bin/env python3
"""Audit C/D torque components and omitted lower postures; CPU only."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from experiments.c_length_study.tools import screen_length_mechanics as s
from experiments.c_length_study.tools.run_length_mechanical_paths import admissible

OUT=s.ROOT/"artifacts/length_study_2026-09-09/stance_audit_all_low_poses"
OUT.mkdir(exist_ok=True)
PKG=s.ROOT/"robot/hexapod_mkii_length_study"
BASE=OUT.parent
manifest=json.loads((PKG/"manifest.json").read_text())
review=json.loads((BASE/"stage1_candidate_review.json").read_text())
report={"status":"running","method":"C/D prescribed-motion torque-component and height-stratified audit",
        "stage2_started":False,"started_unix":time.time(),"variants":{},
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(s.__file__))}}
for p in (Path(__file__),Path(s.__file__)):(OUT/p.name).write_bytes(p.read_bytes())

def save():
    report["elapsed_s"]=time.time()-report["started_unix"]
    (OUT/"audit.json").write_text(json.dumps(report,indent=2))

for name in ("f050_t060","f050_t080"):
    xml=ET.parse(PKG/"urdf"/(name+".urdf")).getroot()
    robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,PKG))
    poses=json.loads((BASE/"mechanics_stage1_v3_static"/(name+"_poses.json")).read_text())
    selected=next(c["refined_0p20"] for c in review["candidates"] if c["variant"]==name)
    pose=next(p for p in poses if (p["femur_deg"],p["knee_deg"])==(selected["femur_deg"],selected["knee_deg"]))
    history=[];original_load=robot.load
    def trace(rows,com,support,*args,**kwargs):
        result=original_load(rows,com,support,*args,**kwargs)
        if kwargs.get("required_wrench") is not None:
            history.append((rows,support,result))
        return result
    robot.load=trace
    trial=s.trajectory(robot,pose,"tripod",.10,.02,256,speeds=(.2,))
    robot.load=original_load
    speed=trial["speed_results"][0];phase=speed["worst_phase_index"]
    rows,support,load=history[phase]
    torques=np.array(load["torques_nm"]);i,j=np.unravel_index(np.abs(torques).argmax(),torques.shape)
    forces=np.array(load["forces_n"]);row=rows[i];force=forces[i]
    static=robot.legs[i].fk(np.array(trial["samples"][phase]["q"])[i],np.array(pose["toes"])[i])
    parts={"gravity_nm":float(static["gravity"][j]),
           "motion_and_motor_nm":float(row["gravity"][j]-static["gravity"][j]),
           "vertical_contact_nm":float(-row["J"][2,j]*force[2]),
           "horizontal_contact_nm":float(-row["J"][:2,j]@force[:2])}
    assert abs(sum(parts.values())-torques[i,j])<1e-9
    nominal,_=robot.pose(np.array(pose["q"]),np.array(pose["toes"]))
    leg=robot.legs[0];nom=nominal[0]
    torque_history=np.array([x[2]["torques_nm"] for x in history])
    result={"original_selected_pose":{k:pose[k] for k in ("femur_deg","knee_deg","belly_clearance_m")},
            "original_peak_audit":{"minimum_required_peak_torque_nm":speed["minimum_required_peak_torque_nm"],
             "one_limiting_joint":leg.joints[j].get("name") if i==0 else robot.legs[i].joints[j].get("name"),
             "leg":s.LEGS[i],"joint_kind":s.KINDS[j],"in_stance":i in support,
             "signed_torque_nm":float(torques[i,j]),"components":parts,"contact_force_n":force.tolist(),
             "foot_minus_hip_radial_mm":float((nom["foot"]-nom["pivots"][1])@leg.radial*1000),
             "foot_minus_knee_radial_mm":float((nom["foot"]-nom["pivots"][2])@leg.radial*1000),
             "peak_by_joint_kind_nm":dict(zip(s.KINDS,np.abs(torque_history).max(axis=(0,1)).tolist()))},
            "lower_pose_trials":[]}
    report["variants"][name]=result;save()
    print(name,"original",json.dumps(result["original_peak_audit"]),flush=True)
    # Retain separate finite height bands. The old minimum-height filter could
    # choose a tall posture for every threshold and miss these lower alternatives.
    for gait in ("tripod","ripple","wave"):
        for lo,hi in ((.08,.10),(.10,.12),(.12,.14)):
            eligible=[p for p in poses if lo<=p["belly_clearance_m"]<hi and gait in p["gaits"]
                      and p["gaits"][gait]["minimum_support_margin_m"]>=.01]
            eligible.sort(key=lambda p:p["gaits"][gait]["worst_peak_torque_nm"])
            for pose in eligible:
                for stride in (.04,.06,.10):
                    t=s.trajectory(robot,pose,gait,stride,.02,64,speeds=(.05,.10,.20))
                    t.pop("samples",None)
                    t.update({k:pose[k] for k in ("femur_deg","knee_deg","belly_clearance_m","stance_width_m","stance_length_m")})
                    for v in t.get("speed_results",[]):v["screen_pass"]=bool(admissible(t,v))
                    result["lower_pose_trials"].append(t);save()
            print(name,gait,lo,hi,"checked all",len(eligible),"poses",flush=True)
report["status"]="completed";save()
