#!/usr/bin/env python3
"""Compare 1/4/16 final velocity passes and audit the new reported telemetry."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

PREVIOUS=Path(__file__).resolve().parents[1]/"final_velocity4_comparison/analyze.py"
spec=importlib.util.spec_from_file_location("velocity_comparison_reader",PREVIOUS)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def verify_telemetry(t,data,kin):
    descriptor=t.report["velocity_constraint_telemetry"]
    if (descriptor["acceptance_use"]!="observational_only_no_velocity_thresholds"
            or descriptor["passive_relation_names"]!=list(kin["passive_relations"])
            or descriptor["closure_pin_names"]!=kin["closure_joint_names"]):
        raise ValueError("Wrong velocity telemetry semantics or relation ordering")
    qdot=t.block(data,"post_qd",t.joints)
    residual=np.stack([qdot[...,t.joints.index(name)]
                       -rel["multiplier"]*qdot[...,t.joints.index(rel["source_joint"])]
                       for name,rel in kin["passive_relations"].items()],-1)
    pin=t.vectors(data,"hinge_relative_point_velocity_local",t.legs)
    out={}
    for name,start,stop in (("startup",0,640),("settled",640,3200)):
        report=t.report["windows"][name];v=residual[start:stop];p=pin[start:stop]
        passive_count=v.size;pin_count=p.shape[0]*p.shape[1]*p.shape[2]
        if report["passive_velocity_relation_samples"]!=passive_count or report["closure_relative_point_velocity_samples"]!=pin_count:
            raise ValueError("Reported velocity RMS denominator differs from actual samples")
        computed={
            "max_passive_velocity_relation_error_rad_s":float(abs(v).max()),
            "passive_velocity_relation_squared_sum":float((v*v).sum()),
            "rms_passive_velocity_relation_error_rad_s":float(np.sqrt(np.mean(v*v))),
            "max_closure_relative_point_velocity_m_s":float(np.linalg.norm(p,axis=-1).max()),
            "closure_relative_point_velocity_squared_norm_sum":float((p*p).sum()),
            "rms_closure_relative_point_velocity_m_s":float(np.sqrt(np.sum(p*p)/pin_count)),
        }
        # The report accumulates float32 residual/squared terms in float64;
        # this independent derivation uses float64 and stored hinge-local vectors.
        # A float32 rotation also separates those vectors from reported world vectors.
        checks={key:{"reported":report[key],"trace_derived":value,
                     "absolute_difference":abs(report[key]-value),
                     "within_float32_roundoff_allowance":bool(np.isclose(report[key],value,
                         rtol=64*np.finfo(np.float32).eps,atol=1e-10))}
                for key,value in computed.items()}
        if not all(row["within_float32_roundoff_allowance"] for row in checks.values()):
            raise ValueError(f"Reported velocity telemetry differs from raw trace: {name}: {checks}")
        out[name]={"passive_samples":passive_count,"pin_samples":pin_count,"metrics":checks}
    return {"descriptor":descriptor,"windows":out,
        "scope":"Telemetry consistency check only; roundoff allowance is not a physical acceptance limit.",
        "roundoff_comparison":{"relative":64*float(np.finfo(np.float32).eps),"absolute":1e-10}}


def analyze(paths,kin_path):
    runs=[previous.base.load(p) for p in paths]
    iterations=(1,4,16)
    ids=("mkii_fourbar_tgs_external_forces_800hz_v3",
         "mkii_fourbar_tgs_external_forces_800hz_final_velocity4_v4",
         "mkii_fourbar_tgs_external_forces_800hz_final_velocity16_v5")
    common=None;sources={};metrics={};runtime_differences={};consistency={}
    kin=json.loads(Path(kin_path).read_text())
    for (t,data,supervisor),count,recipe in zip(runs,iterations,ids):
        r=t.report;runtime=copy.deepcopy(r["runtime_manifest"])
        if (t.envs!=8 or r["solver_iterations"]!=[128,count]
                or runtime["numerical_recipe_id"]!=recipe
                or runtime["resolved_simulation"]["solver_velocity_iterations"]!=count):
            raise ValueError("Unexpected solver recipe, batch size or count")
        if previous.base.digest(kin_path)!=runtime["kinematics_sha256"]:
            raise ValueError("Shared kinematic mapping differs")
        runtime.pop("numerical_recipe_id");runtime["resolved_simulation"].pop("solver_velocity_iterations")
        if common is None:common=runtime
        elif runtime!=common:raise ValueError("Physical runtime differs beyond recipe/velocity count")
        first,first_data,_=runs[0]
        if t.columns!=first.columns:raise ValueError("Trace names or field layout differ")
        for key in ("motor_readback","solver_readback","actuator_backend_flags","scene_force_iteration_readback"):
            if r[key]!=first.report[key]:raise ValueError(f"Backend readback changed: {key}")
        for key in ("actual_terrain_origins_m","default_root_positions_m","initial_reset_root_positions_m"):
            if r["placement"][key]!=first.report["placement"][key]:raise ValueError(f"Placement changed: {key}")
        for key in ("target","processed_target","velocity_target","feedforward"):
            if not np.array_equal(t.block(data,key,t.motors),first.block(first_data,key,first.motors)):
                raise ValueError(f"Input changed: {key}")
        if not np.array_equal(t.block(data,"pre_q",t.joints)[0],first.block(first_data,"pre_q",first.joints)[0]):
            raise ValueError("Initial all30 joint coordinates changed")
        label=str(count)
        sources[label]={"source_commit":supervisor["source_commit"],"functional_contract_sha256":r["contract"]["sha256"],
            "source_manifest":supervisor["source_manifest"],"report_path":str(t.path),"report_sha256":t.report_hash,
            "trace_sha256":t.files[0]["sha256"],"supervisor_completion_source_and_cleanup_verified":True}
        metrics[label]=previous.metrics(t,data,kin)
        consistency[label]={
            "cached_vs_direct_pre_q_max_difference_rad":float(abs(t.block(data,"pre_q",t.joints)-t.block(data,"direct_pre_q",t.joints)).max()),
            "cached_vs_direct_pre_qd_max_difference_rad_s":float(abs(t.block(data,"pre_qd",t.joints)-t.block(data,"direct_pre_qd",t.joints)).max()),
            "P_plus_D_plus_FF_residual_max_Nm":float(abs(t.block(data,"p_term",t.motors)+t.block(data,"d_term",t.motors)
                +t.block(data,"feedforward",t.motors)-t.block(data,"demand",t.motors)).max())}
        runtime_differences[label]=previous.differences(first.report["runtime_manifest"],r["runtime_manifest"])
    t16,d16,_=runs[-1]
    fields=("peak_raw_nm","peak_applied_nm","peak_active_joint_speed_rad_s","peak_closure_point_m",
        "peak_passive_relation_error_rad","max_post_mimic_velocity_residual_rad_s","rms_post_mimic_velocity_residual_rad_s",
        "max_C_pin_relative_velocity_m_s","rms_C_pin_relative_velocity_m_s")
    rows=[{"environment":e,"origin_xyz_m":metrics['1'][e]["actual_terrain_origin_xyz_m"],
           "by_velocity_iterations":{n:metrics[n][e] for n in ('1','4','16')},
           "sixteen_over_four":{key:(metrics['16'][e][key]/metrics['4'][e][key] if metrics['4'][e][key] else None)
                                for key in fields}} for e in range(8)]
    return {"schema":"hexapod.final_velocity_solve_sequence.v1","sources":sources,
        "comparison_checks":{"full_physical_runtime_equal_except_recipe_and_velocity_count":True,
            "initial_state_actual_origins_targets_and_backend_readbacks_equal":True},
        "common_physical_runtime":common,"runtime_differences":runtime_differences,
        "functional_changes_four_to_sixteen":previous.differences(runs[1][0].report["contract"]["files"],t16.report["contract"]["files"]),
        "trace_consistency":consistency,"environments":rows,
        "aggregate_maxima":{label:{key:max(row[key] for row in values) for key in fields} for label,values in metrics.items()},
        "primary_settled_windows":{str(n):t.report["windows"]["settled"] for (t,_,_),n in zip(runs,iterations)},
        "primary_physical_gate_errors":{str(n):t.report["physical_gate_errors"] for (t,_,_),n in zip(runs,iterations)},
        "velocity16_report_vs_raw_trace":verify_telemetry(t16,d16,kin),
        "measurement_window":{"physics_samples_half_open":[640,3200],"time_s":[.8,4.]},
        "dependency_sha256":previous.base.digest(PREVIOUS),
        "scope":"Diagnostic iteration study; no new physical gates, full qualification, PPO admission or hardware claim."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("velocity1_report",type=Path);p.add_argument("velocity4_report",type=Path)
    p.add_argument("velocity16_report",type=Path);p.add_argument("kinematics",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.write_text(json.dumps(analyze(
        [args.velocity1_report,args.velocity4_report,args.velocity16_report],args.kinematics),indent=2,sort_keys=True)+"\n")
