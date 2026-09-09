"""Compare matched standing traces; placement correlation is not causal proof."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multiply(a, b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3], b[:3]), a[3]*b[3]-a[:3]@b[:3]]


def conjugate(q):
    return q*np.array([-1, -1, -1, 1])


def rotate(q, vector):
    q = q/np.linalg.norm(q)
    return vector+2*np.cross(q[:3], np.cross(q[:3], vector)+q[3]*vector)


def read_run(path, kinematics_path):
    report = json.loads(path.read_text())
    record = report["trace_files"][0]
    trace_path = path.parent/record["file"]
    if (not report["diagnostic_complete"] or report["errors"] or record["first_physics_sample"] != 0
            or record["segment"] != {"phase": "standing", "motors": [], "offset_rad": 0., "steps": 200}
            or sha(trace_path) != record["sha256"]):
        raise ValueError("Unqualified standing report or trace identity")
    recipe = report["numerical_recipe"]
    if recipe["physics_dt_s"] != .00125 or recipe["decimation"] != 16:
        raise ValueError("This matched audit requires the 800 Hz / 50 Hz timing")
    with np.load(trace_path, allow_pickle=False) as data:
        values, columns = data["values"].astype(np.float64), data["columns"].tolist()
    if (list(values.shape) != record["shape"] or values.shape[:2] != (3200, 8)
            or len(columns) != len(set(columns)) or not np.isfinite(values).all()):
        raise ValueError("Invalid eight-environment standing trace")
    get = lambda key, names: values[..., [columns.index(key+"/"+name) for name in names]]
    names = report["active_motor_names"]
    arrays = {key: get(key, names) for key in ("pre_q", "post_q", "pre_qd", "post_qd", "demand", "applied", "p_term", "d_term", "target", "processed_target", "velocity_target", "feedforward")}
    origin = get("terrain_origin_w", list("xyz"))
    root = get("body_link_pos_w", ["body_"+a for a in "xyz"])
    body_names = [name[:-2] for key, name in (column.split("/") for column in columns) if key == "body_link_pos_w" and name.endswith("_x")]
    body_positions = get("body_link_pos_w", [name+"_"+a for name in body_names for a in "xyz"]).reshape(3200,8,len(body_names),3)
    per_env = []
    for env in range(8):
        block = {key: value[640:3200, env] for key, value in arrays.items()}
        qd, torque, dterm = block["pre_qd"], block["demand"], block["d_term"]
        peak_index = np.unravel_index(np.argmax(abs(dterm)), dterm.shape)
        sample, motor = 640+int(peak_index[0]), int(peak_index[1])
        per_env.append({"environment": env, "terrain_origin_m": origin[0,env].tolist(),
            "radius_xy_m": float(np.linalg.norm(origin[0,env,:2])),
            "root_mean_xy_m": root[640:,env,:2].mean(0).tolist(),
            "root_max_xyz_float32_ulp_m": np.spacing(abs(root[640:,env]).astype(np.float32)).max(0).tolist(),
            "all_body_max_xyz_float32_ulp_m": np.spacing(abs(body_positions[640:,env]).astype(np.float32)).max((0,1)).tolist(),
            "peak_raw_nm": float(abs(torque).max()), "peak_applied_nm": float(abs(block["applied"]).max()),
            "raw_demeaned_rms_nm": float(np.sqrt(np.mean((torque-torque.mean(0))**2))),
            "peak_qd_rad_s": float(abs(qd).max()), "rms_qd_rad_s": float(np.sqrt(np.mean(qd*qd))),
            "peak_d_nm": float(abs(dterm).max()), "rms_d_nm": float(np.sqrt(np.mean(dterm*dterm))),
            "max_motor_q_range_rad": float(np.ptp(block["pre_q"],axis=0).max()),
            "peak_d_event": {"sample": sample, "motor": names[motor],
                **{key: float(arrays[key][sample,env,motor]) for key in ("pre_q", "post_q", "pre_qd", "post_qd", "p_term", "d_term", "demand", "applied")},
                "dq_dt_rad_s": float((arrays["post_q"][sample,env,motor]-arrays["pre_q"][sample,env,motor])/.00125)}})
    radius = np.array([row["radius_xy_m"] for row in per_env])
    correlations = {key: float(np.corrcoef(radius, [row[key] for row in per_env])[0,1])
        for key in ("peak_raw_nm", "raw_demeaned_rms_nm", "peak_qd_rad_s", "rms_qd_rad_s")}
    peak_env = max(per_env, key=lambda row: row["peak_d_nm"])
    event, env = peak_env["peak_d_event"], peak_env["environment"]
    if sha(kinematics_path) != report["contract"]["files"]["configs/mkii_fourbar_v3_kinematics.json"]:
        raise ValueError("Kinematics identity mismatch")
    kin = json.loads(kinematics_path.read_text())
    frame = kin["joint_frames"][event["motor"]]
    parent, child = frame["body0"], frame["body1"]
    local_axis = np.array(frame["body0_from_hinge_matrix"])[:3,2]
    leg = event["motor"].split("_")[0]
    poses = {name: get("body_link_quat_w", [name+"_"+a for a in "xyzw"]) for name in (parent,child)}
    angles = {name: get("body_link_angular_velocity_w", [name+"_"+a for a in "xyz"]) for name in (parent,child)}
    forces = get("foot_force_w", [leg+"_"+a for a in "xyz"])
    gap = get("hinge_gap_local", [leg+"_"+a for a in "xyz"])
    def relative_quat(i):
        a,b = [poses[name][i,env]/np.linalg.norm(poses[name][i,env]) for name in (parent,child)]
        return multiply(conjugate(a),b)
    motor = names.index(event["motor"])
    window = []
    for i in range(event["sample"]-4,event["sample"]+4):
        delta = multiply(relative_quat(i),conjugate(relative_quat(i-1)))
        if delta[3]<0: delta *= -1
        window.append({"physics_sample": i, "pre_time_s": i*.00125,
            **{key: float(arrays[key][i,env,motor]) for key in ("pre_q", "post_q", "pre_qd", "post_qd", "p_term", "d_term", "demand", "applied")},
            "joint_position_difference_rate_rad_s": float((arrays["post_q"][i,env,motor]-arrays["pre_q"][i,env,motor])/.00125),
            "body_pose_relative_rotation_rate_rad_s": float(2*np.arctan2(delta[:3]@local_axis,delta[3])/.00125),
            "body_relative_angular_velocity_projected_on_hinge_rad_s": float((angles[child][i,env]-angles[parent][i,env])@rotate(poses[parent][i,env],local_axis)),
            "associated_foot_net_force_w_n": forces[i,env].tolist(), "associated_hinge_gap_local_m": gap[i,env].tolist()})
    result = {"report_path": str(path), "report_sha256": sha(path), "standing_trace_sha256": record["sha256"],
        "standing_trace_shape": list(values.shape), "source_contract_sha256": report["contract"]["sha256"],
        "usd_sha256": report["usd_sha256"], "kinematics_sha256": sha(kinematics_path), "numerical_recipe": recipe,
        "matched_sample_range_half_open": [640,3200], "matched_physics_time_interval_s": [.8,4.],
        "per_environment": per_env, "radius_correlations_n8_only": correlations,
        "maximum_target_change_rad": float(np.ptp(arrays["target"],axis=0).max()),
        "maximum_target_endpoint_difference_rad": float(abs(arrays["target"]-arrays["processed_target"]).max()),
        "maximum_velocity_target_rad_s": float(abs(arrays["velocity_target"]).max()), "maximum_feedforward_nm": float(abs(arrays["feedforward"]).max()),
        "peak_derivative_event_context": {"environment": env, "motor": event["motor"], "parent_body":parent,"child_body":child,"rows":window}}
    return result, report["runtime_manifest"], (arrays["target"],origin,arrays["pre_q"][0])


def differences(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b: result.append({"path":path+key,"refined":a.get(key),"nominal":b.get(key)})
            else: result.extend(differences(a[key],b[key],path+key+"."))
        return result
    return [] if a==b else [{"path":path.rstrip("."),"refined":a,"nominal":b}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("refined_report","nominal_report","kinematics","out"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    refined,rm,ra=read_run(args.refined_report,args.kinematics)
    nominal,nm,na=read_run(args.nominal_report,args.kinematics)
    result={"schema":"hexapod.standing_placement_comparison.v1","derived_evidence_only":True,
        "script_sha256":sha(Path(__file__)),"refined":refined,"nominal":nominal,
        "runtime_differences":differences(rm,nm),
        "maximum_actual_target_difference_between_runs_rad":float(abs(ra[0]-na[0]).max()),
        "maximum_terrain_origin_difference_between_runs_m":float(abs(ra[1]-na[1]).max()),
        "maximum_initial_active_joint_difference_between_runs_rad":float(abs(ra[2]-na[2]).max()),
        "limits":["One run per setting, eight correlated placements with only three radii; correlations do not establish a cause.",
            "TGS internal state is not recorded; the trace sees only completed1.25ms physics steps.",
            "Different target-scheduler metadata/source code remains a recorded confound, although actual standing commands are identical.",
            "The longer32-environment reports cannot supply a matched per-environment trace comparison."]}
    with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"output":str(args.out),"refined_peak_nm":max(row["peak_raw_nm"] for row in refined["per_environment"]),"nominal_peak_nm":max(row["peak_raw_nm"] for row in nominal["per_environment"])}))


if __name__=="__main__":main()
