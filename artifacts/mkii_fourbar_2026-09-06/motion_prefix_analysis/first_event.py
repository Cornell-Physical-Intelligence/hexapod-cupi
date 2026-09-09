#!/usr/bin/env python3
"""CPU-only exact-sample reconstruction around the first observed clipping episode."""
import argparse
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


def rotation_xyzw(q):
    q = q.astype(np.float64)/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),
        2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(*q.shape[:-1],3,3)


def minimum_pad_height(values,traces,urdf,legs):
    heights=[]
    for leg in legs:
        name=leg+"_tibia"
        pos=traces.block(values,"body_link_pos_w",[name+"_"+a for a in "xyz"])
        rot=rotation_xyzw(traces.block(values,"body_link_quat_w",[name+"_"+a for a in "xyzw"]))
        bottoms=[]
        for collision in urdf.find(f"link[@name='{name}']").findall("collision"):
            sphere=collision.find("geometry/sphere")
            if sphere is None:continue
            origin=collision.find("origin")
            local=np.array([float(v) for v in origin.get("xyz","0 0 0").split()])
            bottoms.append(pos[...,2]+np.einsum("...j,j->...",rot[...,2,:],local)-float(sphere.get("radius")))
        if not bottoms:raise ValueError("No URDF foot sphere found")
        heights.append(np.min(bottoms,axis=0))
    return np.stack(heights,axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True);p.add_argument("--reader",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("Preserve previous output; require new directory")
    spec=importlib.util.spec_from_file_location("prefix_reader",args.reader)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    r=json.loads(args.report.read_text());kin_path=args.source/"configs/mkii_fourbar_v3_kinematics.json"
    urdf_relative="robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf";urdf_path=args.source/urdf_relative
    if (reader.digest(kin_path)!=r["runtime_manifest"]["kinematics_sha256"]
            or reader.digest(urdf_path)!=r["contract"]["files"][urdf_relative]):raise ValueError("Unmatched kinematic/URDF source")
    kin=json.loads(kin_path.read_text());urdf=ET.parse(urdf_path).getroot()
    t=reader.TraceSet(args.report,r,"trace");names,joints=r["active_motor_names"],r["joint_names"]
    dt=r["numerical_recipe"]["physics_dt_s"];dec=r["numerical_recipe"]["decimation"]
    per_env=[{"environment":env,"reset_xyz_m":xyz,"max_raw_nm":0.,"max_gap_m":0.,"max_passive_velocity_rad_s":0.,
        "first_clip_sample":None,"first_raw_above_5p5_sample":None,"first_speed_above_480rpm_sample":None} for env,xyz in enumerate(r["reset_root_positions_m"])]
    detailed=[];peak_preclip=None
    args.out.mkdir(parents=True)
    csv_path=args.out/"first_event_all_envs.csv.gz"
    saved_rows=0
    with gzip.open(csv_path,"wt",newline="") as file:
        writer=csv.writer(file);header_written=False
        for record in t.records:
            v=t.read(record);first=record["first_physics_sample"]
            raw,applied,limit,headroom=[t.block(v,key,names) for key in ("demand","applied","instantaneous_limit","headroom")]
            q0,q1,qd0,qd1=[t.block(v,key,joints).astype(np.float64) for key in ("pre_q","post_q","pre_qd","post_qd")]
            posres=reader.passive_residual(q1,joints,kin,position=True);velres=reader.passive_residual(qd1,joints,kin)
            gap=np.linalg.norm(t.vectors(v,"hinge_gap_local"),axis=-1)
            feet=t.vectors(v,"foot_force_w");pad_height=minimum_pad_height(v,t,urdf,reader.LEGS)
            speed=t.block(v,"pre_qd",names)
            for env,row in enumerate(per_env):
                for field,array in (("max_raw_nm",raw),("max_gap_m",gap),("max_passive_velocity_rad_s",velres)):
                    at=np.unravel_index(np.abs(array[:,env]).argmax(),array[:,env].shape)
                    value=float(abs(array[at[0],env,at[1]]))
                    if value>row[field]:row[field]=value;row[field+"_sample"]=first+int(at[0])
                for field,condition in (("first_clip_sample",np.abs(raw[:,env]-applied[:,env])>1e-5),
                    ("first_raw_above_5p5_sample",np.abs(raw[:,env])>5.5),
                    ("first_speed_above_480rpm_sample",np.abs(speed[:,env])>480*np.pi/30)):
                    where=np.argwhere(condition)
                    if len(where) and row[field] is None:
                        row[field]=first+int(where[0,0]);row[field+"_motor"]=names[int(where[0,1])]
            for local in range(max(0,35984-first),min(len(v),36112-first)):
                sample=first+local
                if not header_written:
                    writer.writerow(["physics_sample","environment","pre_time_s","post_time_s",*t.columns]);header_written=True
                for env in range(r["num_envs"]):
                    writer.writerow([sample,env,format(sample*dt,".17g"),format((sample+1)*dt,".17g"),*[format(float(value),".9g") for value in v[local,env]]]);saved_rows+=1
                if 36000<=sample<=36078:
                    dv=qd1[local]-qd0[local];at=np.unravel_index(np.abs(dv).argmax(),dv.shape)
                    if peak_preclip is None or abs(float(dv[at]))>peak_preclip["abs_velocity_change_rad_s"]:
                        peak_preclip={"sample":sample,"environment":int(at[0]),"joint":joints[int(at[1])],
                            "abs_velocity_change_rad_s":abs(float(dv[at])),"pre_qd":float(qd0[local][at]),"post_qd":float(qd1[local][at])}
                if 36048<=sample<=36096:
                    env=24;motor=names.index("lf_femur_pitch")
                    row={"sample":sample,"control_step":sample//dec,"substep":sample%dec,"environment":env,
                        "LF_femur":{key:float(t.block(v,key,["lf_femur_pitch"])[local,env,0]) for key in
                            ("pre_q","post_q","pre_qd","post_qd","direct_pre_qd","target","processed_target","p_term","d_term","demand","applied","instantaneous_limit","headroom")},
                        "LF_leg_post_q_rad":{name:float(q1[local,env,joints.index(name)]) for name in joints if name.startswith("lf_")},
                        "LF_leg_post_qd_rad_s":{name:float(qd1[local,env,joints.index(name)]) for name in joints if name.startswith("lf_")},
                        "foot_force_xyz_N":dict(zip(reader.LEGS,feet[local,env].tolist())),
                        "reported_net_foot_impulse_Ns":dict(zip(reader.LEGS,(feet[local,env]*dt).tolist())),
                        "lowest_foot_sphere_bottom_world_z_m":dict(zip(reader.LEGS,pad_height[local,env].tolist())),
                        "C_gap_norm_m":dict(zip(reader.LEGS,gap[local,env].tolist())),
                        "passive_position_error_rad":dict(zip(kin["passive_relations"],posres[local,env].tolist())),
                        "passive_velocity_error_rad_s":dict(zip(kin["passive_relations"],velres[local,env].tolist()))}
                    j=joints.index("lf_femur_pitch");row["LF_femur"]["interval_average_qdot_rad_s"]=(q1[local,env,j]-q0[local,env,j])/dt
                    row["LF_femur"]["raw_minus_applied_nm"]=float(raw[local,env,motor]-applied[local,env,motor])
                    detailed.append(row)
    summary={"schema":"hexapod.prefix_first_event.v1","simulation_training_admission":False,"hardware_admission":False,
        "inputs":{str(path):reader.digest(path) for path in (args.report,args.reader,Path(__file__),kin_path,urdf_path)},
        "source_sha256":r["contract"]["sha256"],"runtime_manifest":r["runtime_manifest"],"trace_files":r["trace_files"],
        "csv":{"file":csv_path.name,"sha256":reader.digest(csv_path),"samples_half_open":[35984,36112],"environments":32,"rows":saved_rows,
               "numeric_format":"9 significant decimal digits preserve every recorded float32 value; times17digits"},
        "per_environment":per_env,"earliest_clipping_environments":sorted([row for row in per_env if row["first_clip_sample"] is not None],key=lambda row:row["first_clip_sample"]),
        "largest_velocity_change_after_reversal_before_first_clip":peak_preclip,"env24_exact_samples":detailed,
        "interpretation_limits":["The selected sequence reconstructs temporal order; it is not a counterfactual proving an amplification factor.",
            "Foot impulse is reported net force multiplied by1.25ms, not individual native contact impulses or effective mass.",
            "The velocity jump and contact impulse occur inside the same solved step; their internal solver ordering cannot be recovered.",
            "Finite-difference qdot is an interval average; SDK endpoint qdot can differ after constraint/contact velocity corrections.",
            "Raw demand equals applied torque before the first clip; later speed-envelope clipping removes some requested braking but cannot retroactively initiate that first event."]}
    (args.out/"first_event.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"csv_rows":saved_rows,"earliest_clip":summary["earliest_clipping_environments"][0],"preclip_jump":peak_preclip}))


if __name__=="__main__":main()
