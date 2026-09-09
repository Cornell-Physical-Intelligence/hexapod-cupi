#!/usr/bin/env python3
"""Compare the same standing robot/location singly and as row zero of a batch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

GENERIC = Path(__file__).resolve().parents[2] / "dynamics_trace_analysis/analyze.py"
spec = importlib.util.spec_from_file_location("verified_trace_reader", GENERIC)
generic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generic)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    t = generic.Traces(path)
    r = t.report
    if (r["diagnostic_complete"] is not True or len(t.files) != 1
            or r["diagnostic_motion"] != "standing" or t.samples != 3200
            or t.dt != .00125 or t.decimation != 16):
        raise ValueError("Expected complete 200-control, 3200-substep standing diagnostic")
    supervisor_path = t.path.parent / "supervisor.json"
    supervisor = json.loads(supervisor_path.read_text())
    if (supervisor["execution"] != "diagnostic_complete"
            or supervisor["source_identity_unchanged_at_finish"] is not True
            or supervisor["cleanup"] != "removed_exact_id"
            or supervisor["supervisor_exit_code"] != 0):
        raise ValueError("Supervisor did not confirm complete, unchanged-source diagnostic and cleanup")
    manifest = t.path.parent / "source.SHA256SUMS"
    if digest(manifest) != supervisor["source_manifest"]["sha256"]:
        raise ValueError("Source manifest SHA differs from supervisor")
    data, _ = t.read(0)
    return t, data, supervisor


def field(t, data, key, names=None):
    return t.block(data, key, names or t.motors)


def summarize(t, data, first, stop, kinematics):
    result = []
    active_qd = field(t, data, "pre_qd")
    demand = field(t, data, "demand")
    applied = field(t, data, "applied")
    p_term, d_term = [field(t, data, key) for key in ("p_term", "d_term")]
    feet = t.vectors(data, "foot_force_w", t.legs)
    gaps = t.vectors(data, "hinge_gap_local", t.legs)
    axes = t.vectors(data, "hinge_axis_difference_local", t.legs)
    preq, postq = [field(t, data, key, t.joints) for key in ("pre_q", "post_q")]
    passive_errors = []
    for name, rel in kinematics["passive_relations"].items():
        passive_errors.append(postq[...,t.joints.index(name)]
            - rel["multiplier"]*postq[...,t.joints.index(rel["source_joint"])] - rel["offset_rad"])
    passive = np.stack(passive_errors, -1)
    root = field(t, data, "body_link_pos_w", ["body_"+a for a in "xyz"])
    for env in range(t.envs):
        values = demand[first:stop, env]
        at = np.unravel_index(np.argmax(abs(values)), values.shape)
        sample, motor = first+int(at[0]), int(at[1])
        j = t.joints.index(t.motors[motor])
        result.append({
            "environment": env,
            "actual_initial_root_xyz_m": t.report["placement"]["initial_reset_root_positions_m"][env],
            "actual_terrain_origin_xyz_m": t.report["placement"]["actual_terrain_origins_m"][env],
            "root_mean_xyz_m": root[first:stop,env].mean(0).tolist(),
            "peak_raw_nm": float(abs(values).max()),
            "peak_applied_nm": float(abs(applied[first:stop,env]).max()),
            "peak_active_joint_speed_rad_s": float(abs(active_qd[first:stop,env]).max()),
            "rms_active_joint_speed_rad_s": float(np.sqrt(np.mean(active_qd[first:stop,env]**2))),
            "rms_demeaned_demand_nm": float(np.sqrt(np.mean((values-values.mean(0))**2))),
            "peak_d_term_nm": float(abs(d_term[first:stop,env]).max()),
            "peak_closure_point_m": float(np.linalg.norm(gaps[first:stop,env],axis=-1).max()),
            "peak_axis_chord": float(np.linalg.norm(axes[first:stop,env],axis=-1).max()),
            "peak_passive_relation_error_rad": float(abs(passive[first:stop,env]).max()),
            "min_feet_with_force_gt1N": int((np.linalg.norm(feet[first:stop,env],axis=-1)>1).sum(-1).min()),
            "peak_raw_event": {
                "physics_sample":sample,"pre_time_s":sample*t.dt,"motor":t.motors[motor],
                "raw_nm":float(demand[sample,env,motor]),"applied_nm":float(applied[sample,env,motor]),
                "p_term_nm":float(p_term[sample,env,motor]),"d_term_nm":float(d_term[sample,env,motor]),
                "pre_joint_speed_rad_s":float(active_qd[sample,env,motor]),
                "post_joint_speed_rad_s":float(field(t,data,"post_qd")[sample,env,motor]),
                "joint_position_difference_rate_rad_s":float((postq[sample,env,j]-preq[sample,env,j])/t.dt),
                "foot_force_xyz_N": {leg: feet[sample,env,i].tolist() for i,leg in enumerate(t.legs)},
                "matching_sample_window": [
                    {"physics_sample":s,"demand_nm":float(demand[s,env,motor]),
                     "d_term_nm":float(d_term[s,env,motor]),"pre_qd_rad_s":float(active_qd[s,env,motor]),
                     "post_qd_rad_s":float(field(t,data,"post_qd")[s,env,motor]),
                     "dq_dt_rad_s":float((postq[s,env,j]-preq[s,env,j])/t.dt),
                     "total_vertical_foot_force_N":float(feet[s,env,:,2].sum())}
                    for s in range(max(first,sample-3),min(stop,sample+4))],
            },
        })
    return result


def analyze(single_path, batch_path, repo_root, origin_path=None):
    st, sd, ss = load(single_path)
    bt, bd, bs = load(batch_path)
    if st.envs != 1 or bt.envs != 8:
        raise ValueError("Expected one robot versus eight robots")
    if st.columns != bt.columns or st.report["runtime_manifest"] != bt.report["runtime_manifest"]:
        raise ValueError("Trace columns or full runtime manifest differ")
    if (st.report["contract"] != bt.report["contract"] or ss["source_commit"] != bs["source_commit"]
            or ss["source_manifest"] != bs["source_manifest"]):
        raise ValueError("Full functional contract or source manifest differs")
    sr, br = st.report, bt.report
    kin_path = repo_root / "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/kinematics.json"
    if digest(kin_path) != sr["runtime_manifest"]["kinematics_sha256"]:
        raise ValueError("Local kinematic contract differs from the running asset")
    kin = json.loads(kin_path.read_text())
    usd_path = repo_root / sr["usd_path_relative"]
    if digest(usd_path) != sr["usd_sha256"]:
        raise ValueError("Local USD differs from the running asset")
    source_names = {row["source_joint"] for row in kin["passive_relations"].values()}
    if len(kin["passive_relations"]) != 12 or len(source_names) != 6:
        raise ValueError("Unexpected passive mapping")
    comparisons = {
        "runtime_manifest_equal": True, "functional_contract_equal": True,
        "full_source_manifest_equal": True, "trace_column_names_equal": True,
        "matching_initial_root_xyz_equal": sr["placement"]["initial_reset_root_positions_m"][0]
                                         == br["placement"]["initial_reset_root_positions_m"][0],
        "matching_terrain_origin_equal": sr["placement"]["actual_terrain_origins_m"][0]
                                        == br["placement"]["actual_terrain_origins_m"][0],
        "initial_all30_joint_max_difference_rad":float(abs(field(st,sd,"pre_q",st.joints)[0,0]
                                             -field(bt,bd,"pre_q",bt.joints)[0]).max()),
        "all_delivered_target_max_difference_rad":float(abs(field(st,sd,"target")-field(bt,bd,"target")).max()),
        "all_processed_target_max_difference_rad":float(abs(field(st,sd,"processed_target")-field(bt,bd,"processed_target")).max()),
        "motor_readback_all_rows_equal":all(np.array_equal(np.asarray(br["motor_readback"][key]),
            np.repeat(np.asarray(sr["motor_readback"][key]),bt.envs,axis=0)) for key in sr["motor_readback"]),
        "solver_readback_all_rows_equal":all(np.array_equal(np.asarray(br["solver_readback"][key]),
            np.repeat(np.asarray(sr["solver_readback"][key]),bt.envs,axis=0)) for key in sr["solver_readback"]),
        "native_clone_reference_relationships_recorded": False,
    }
    for key,value in comparisons.items():
        if key.endswith("equal") and value is not True:
            raise ValueError(f"Matched comparison prerequisite failed: {key}")
    for key in ("initial_all30_joint_max_difference_rad", "all_delivered_target_max_difference_rad",
                "all_processed_target_max_difference_rad"):
        if comparisons[key] != 0:
            raise ValueError(f"Matched initial state/target differs: {key}")
    checks = {}
    for label,t,data in (("single",st,sd),("batch",bt,bd)):
        checks[label] = {
            "pre_q_cached_direct_max_difference_rad":float(abs(field(t,data,"pre_q",t.joints)-field(t,data,"direct_pre_q",t.joints)).max()),
            "pre_qd_cached_direct_max_difference_rad_s":float(abs(field(t,data,"pre_qd",t.joints)-field(t,data,"direct_pre_qd",t.joints)).max()),
            "p_plus_d_plus_ff_residual_max_nm":float(abs(field(t,data,"p_term")+field(t,data,"d_term")+field(t,data,"feedforward")-field(t,data,"demand")).max()),
            "velocity_target_max_rad_s":float(abs(field(t,data,"velocity_target")).max()),
            "feedforward_max_nm":float(abs(field(t,data,"feedforward")).max()),
            "delivered_target_time_variation_rad":float(abs(field(t,data,"target")-field(t,data,"target")[0]).max()),
        }
    result = {
        "schema":"hexapod.matched_batch_standing_comparison.v1",
        "sources": {name:{"report_relative":str(t.path.relative_to(repo_root)),
                           "report_sha256":t.report_hash,"trace_sha256":t.files[0]["sha256"],
                           "source_commit":s["source_commit"],"source_manifest_sha256":s["source_manifest"]["sha256"],
                           "functional_contract_sha256":t.report["contract"]["sha256"],
                           "source_unchanged_and_cleanup_confirmed":True}
                    for name,t,s in (("single",st,ss),("batch",bt,bs))},
        "common_runtime_manifest":sr["runtime_manifest"],
        "comparison_checks":comparisons,"trace_consistency":checks,
        "matching_row0_field_max_absolute_differences": {
            key:float(abs(sd[:,0,[st.fields[key][name] for name in st.fields[key]]]
                    -bd[:,0,[bt.fields[key][name] for name in st.fields[key]]]).max())
            for key in st.fields},
        "kinematic_passive_mapping":kin["passive_relations"],
        "kinematic_mapping_limitation":"Authored mapping and observed relative coordinates are available; native per-clone reference relationship paths were not captured by these runs.",
        "windows":{},
        "support_metric":"Number of six named foot-force vectors whose magnitude exceeds1N; primary report retains full pad/ground classification.",
        "proves_a_particular_native_solver_cause":False,
        "analysis_dependency_sha256":digest(GENERIC),
    }
    for name,first,stop in (("matched_settled_0p8_to4s",640,3200),("matched_late_2p4_to4s",1920,3200)):
        result["windows"][name]={"physics_sample_range_half_open":[first,stop],
            "pre_time_range_s":[first*st.dt,stop*st.dt],
            "single":summarize(st,sd,first,stop,kin),"batch":summarize(bt,bd,first,stop,kin)}
    if origin_path is not None:
        ot, od, os = load(origin_path)
        if (ot.envs != 1 or ot.columns != bt.columns or ot.report["runtime_manifest"] != br["runtime_manifest"]
                or os["source_manifest"] != bs["source_manifest"]):
            raise ValueError("Origin control does not match batch source/runtime")
        origin = ot.report["placement"]["actual_terrain_origins_m"][0]
        matches=[i for i,xyz in enumerate(br["placement"]["actual_terrain_origins_m"]) if xyz==origin]
        if len(matches)!=1:
            raise ValueError("Origin control must match exactly one batch row")
        row=matches[0]
        result["secondary_origin_control"]={
            "report_relative":str(ot.path.relative_to(repo_root)),"report_sha256":ot.report_hash,
            "trace_sha256":ot.files[0]["sha256"],"matching_batch_row":row,
            "source_and_runtime_match":True,
            "all3200_samples_field_max_absolute_differences":{
                key:float(abs(od[:,0,[ot.fields[key][name] for name in ot.fields[key]]]
                    -bd[:,row,[bt.fields[key][name] for name in ot.fields[key]]]).max())
                for key in ot.fields},
        }
    def grid(count):
        rows=math.ceil(count/math.sqrt(count)); cols=math.ceil(count/rows)
        return [[-(i//cols-(rows-1)/2)*2,(i%cols-(cols-1)/2)*2,0.] for i in range(count)]
    result["derived_grid_layouts"]={
        "source":"/home/orionh/IsaacLab/source/isaaclab/isaaclab/cloner/cloner_utils.py:481-530",
        "source_sha256":"80cdeda2fe6dc10391ac799cde36dc4eb9167e83613758bc497b3f08f148607a",
        "method":"Exact transcription of installed grid_transforms formula, spacing2m; 32env positions derived, not a live32env readback.",
        "eight_env_positions_m":grid(8),"thirtytwo_env_positions_m":grid(32),
        "eight_matches_actual_readback":grid(8)==br["placement"]["actual_terrain_origins_m"],
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_report",type=Path)
    parser.add_argument("batch_report",type=Path)
    parser.add_argument("--repo-root",type=Path,required=True)
    parser.add_argument("--origin-report",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.single_report,args.batch_report,args.repo_root.resolve(),args.origin_report)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
