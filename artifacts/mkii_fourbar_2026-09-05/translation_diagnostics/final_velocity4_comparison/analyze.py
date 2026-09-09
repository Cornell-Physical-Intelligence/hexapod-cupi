#!/usr/bin/env python3
"""Matched 128/1 versus 128/4 standing comparison, without new admission bounds."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/"batch_analysis/analyze.py"
spec=importlib.util.spec_from_file_location("batch_reader",BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def differences(before,after,path=""):
    if isinstance(before,dict) and isinstance(after,dict):
        out={}
        for key in sorted(set(before)|set(after)):
            if key not in before or key not in after:
                out[(path+"."+key).lstrip(".")]={"before":before.get(key),"after":after.get(key)}
            else:out.update(differences(before[key],after[key],(path+"."+key).lstrip(".")))
        return out
    return {} if before==after else {path:{"before":before,"after":after}}


def metrics(t,data,kin):
    summaries=base.summarize(t,data,640,3200,kin)
    get=lambda key,names:t.block(data,key,names)
    velocities=get("post_qd",t.joints)
    errors=np.stack([velocities[...,t.joints.index(name)]
                    -rel["multiplier"]*velocities[...,t.joints.index(rel["source_joint"])]
                    for name,rel in kin["passive_relations"].items()],-1)
    pin_vel=t.vectors(data,"hinge_relative_point_velocity_local",t.legs)
    for env,row in enumerate(summaries):
        v=errors[640:,env];c=pin_vel[640:,env]
        row.update({"max_post_mimic_velocity_residual_rad_s":float(abs(v).max()),
            "rms_post_mimic_velocity_residual_rad_s":float(np.sqrt(np.mean(v*v))),
            "max_C_pin_relative_velocity_m_s":float(np.linalg.norm(c,axis=-1).max()),
            "rms_C_pin_relative_velocity_m_s":float(np.sqrt(np.mean(np.sum(c*c,axis=-1))))})
    return summaries


def analyze(before_path,after_path,kin_path):
    bt,b,bs=base.load(before_path);at,a,ass=base.load(after_path)
    if bt.envs!=8 or at.envs!=8 or bt.columns!=at.columns:
        raise ValueError("Expected two matching eight-world trace layouts")
    br,ar=bt.report,at.report
    delta=differences(br["runtime_manifest"],ar["runtime_manifest"])
    expected={
        "numerical_recipe_id":{"before":"mkii_fourbar_tgs_external_forces_800hz_v3",
                               "after":"mkii_fourbar_tgs_external_forces_800hz_final_velocity4_v4"},
        "resolved_simulation.solver_velocity_iterations":{"before":1,"after":4},
    }
    if delta!=expected:
        raise ValueError(f"Unexpected full runtime difference: {delta}")
    if br["solver_iterations"]!=[128,1] or ar["solver_iterations"]!=[128,4]:
        raise ValueError("Resolved iteration reports do not match the intended comparison")
    for key in ("motor_readback","solver_readback","actuator_backend_flags","scene_force_iteration_readback"):
        if br[key]!=ar[key]:raise ValueError(f"Changed backend readback: {key}")
    for key in ("actual_terrain_origins_m","default_root_positions_m","initial_reset_root_positions_m"):
        if br["placement"][key]!=ar["placement"][key]:raise ValueError(f"Changed placement: {key}")
    deltas={key:float(abs(bt.block(b,key,bt.motors)-at.block(a,key,at.motors)).max())
            for key in ("target","processed_target","velocity_target","feedforward")}
    if any(deltas.values()):raise ValueError("Changed physical control inputs")
    if not np.array_equal(bt.block(b,"pre_q",bt.joints)[0],at.block(a,"pre_q",at.joints)[0]):
        raise ValueError("Changed initial all30 joint coordinates")
    raw=Path(kin_path).read_bytes();kin=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=br["runtime_manifest"]["kinematics_sha256"]:
        raise ValueError("Kinematic mapping hash mismatch")
    old,new=metrics(bt,b,kin),metrics(at,a,kin)
    compare_keys=("peak_raw_nm","peak_applied_nm","peak_active_joint_speed_rad_s",
        "peak_closure_point_m","peak_passive_relation_error_rad","max_post_mimic_velocity_residual_rad_s",
        "rms_post_mimic_velocity_residual_rad_s","max_C_pin_relative_velocity_m_s","rms_C_pin_relative_velocity_m_s")
    rows=[]
    for left,right in zip(old,new):
        rows.append({"environment":left["environment"],"origin_xyz_m":left["actual_terrain_origin_xyz_m"],
            "before":left,"after":right,
            "after_over_before":{key:(right[key]/left[key] if left[key] else None) for key in compare_keys}})
    return {
        "schema":"hexapod.final_velocity_iteration_comparison.v1",
        "sources":{name:{"report_path":str(t.path),"report_sha256":t.report_hash,
                          "trace_sha256":t.files[0]["sha256"],"source_commit":s["source_commit"],
                          "source_manifest":s["source_manifest"],"functional_contract_sha256":t.report["contract"]["sha256"],
                          "source_unchanged_and_owned_cleanup_confirmed":True}
                   for name,t,s in (("before",bt,bs),("after",at,ass))},
        "runtime_differences":delta,
        "functional_file_differences":differences(br["contract"]["files"],ar["contract"]["files"]),
        "comparison_checks":{"full_runtime_diff_is_only_version_and_velocity_count":True,
            "actual_origins_and_initial_root_positions_equal":True,"initial_all30_joint_coordinates_equal":True,
            "motor_and_backend_readbacks_equal":True,"control_input_max_differences":deltas},
        "window":{"physics_samples_half_open":[640,3200],"time_s":[.8,4.]},
        "environments":rows,
        "aggregate_maxima":{label:{key:max(row[key] for row in values) for key in compare_keys}
                            for label,values in (("before",old),("after",new))},
        "primary_settled_windows":{"before":br["windows"]["settled"],"after":ar["windows"]["settled"]},
        "primary_physical_gate_errors":{"before":br["physical_gate_errors"],"after":ar["physical_gate_errors"]},
        "reader_sha256":base.digest(BASE),
        "scope":"Matched diagnostic only. No additional admission threshold, full standing/driven qualification, solver-convergence admission, or hardware claim.",
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("before",type=Path);p.add_argument("after",type=Path)
    p.add_argument("kinematics",type=Path);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.write_text(json.dumps(analyze(args.before,args.after,args.kinematics),indent=2,sort_keys=True)+"\n")
