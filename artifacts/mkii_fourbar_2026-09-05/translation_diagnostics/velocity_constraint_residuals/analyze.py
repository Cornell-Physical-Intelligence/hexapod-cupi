#!/usr/bin/env python3
"""Measure native-mimic velocity residuals in a completed standing trace."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

READER=Path(__file__).resolve().parents[2]/"dynamics_trace_analysis/analyze.py"
spec=importlib.util.spec_from_file_location("trace_reader",READER)
reader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def analyze(report_path,kinematics_path):
    t=reader.Traces(report_path)
    if len(t.files)!=1 or t.report["diagnostic_motion"]!="standing" or not t.report["diagnostic_complete"]:
        raise ValueError("Expected one complete standing trace")
    data,_=t.read(0)
    kin_bytes=Path(kinematics_path).read_bytes()
    if hashlib.sha256(kin_bytes).hexdigest()!=t.report["runtime_manifest"]["kinematics_sha256"]:
        raise ValueError("Kinematic mapping hash mismatch")
    kin=json.loads(kin_bytes)
    names=list(kin["passive_relations"])
    def get(key,n):return t.block(data,key,n)
    q0,q1,v0,v1=[get(k,t.joints) for k in ("pre_q","post_q","pre_qd","post_qd")]
    def residual(value,offset):
        return np.stack([value[...,t.joints.index(n)]-rel["multiplier"]*value[...,t.joints.index(rel["source_joint"])]
                         -(rel["offset_rad"] if offset else 0.) for n,rel in kin["passive_relations"].items()],-1)
    pos=residual(q1,True)
    vel=residual(v1,False)
    dq=residual((q1-q0)/t.dt,False)
    first=640
    index=np.unravel_index(abs(vel[first:]).argmax(),vel[first:].shape)
    sample,env,j=first+int(index[0]),int(index[1]),int(index[2])
    joint=names[j];relation=kin["passive_relations"][joint];leg=joint.split('_')[0]
    hinge=t.vectors(data,"hinge_relative_point_velocity_local",t.legs)
    gap=t.vectors(data,"hinge_gap_local",t.legs)
    feet=t.vectors(data,"foot_force_w",t.legs)
    result={
        "schema":"hexapod.native_mimic_velocity_residual.v1",
        "report_path":str(t.path),"report_sha256":t.report_hash,
        "trace_sha256":t.files[0]["sha256"],"reader_sha256":reader.digest(READER),
        "functional_source_sha256":t.report["contract"]["sha256"],
        "runtime_manifest":t.report["runtime_manifest"],
        "window":{"samples_half_open":[first,t.samples],"time_s":[first*t.dt,t.samples*t.dt]},
        "formula":"qdot_passive - multiplier*qdot_source; zero offset derivative",
        "per_environment":[{"environment":e,
            "max_post_mimic_velocity_residual_rad_s":float(abs(vel[first:,e]).max()),
            "rms_post_mimic_velocity_residual_rad_s":float(np.sqrt(np.mean(vel[first:,e]**2))),
            "max_finite_difference_mimic_velocity_residual_rad_s":float(abs(dq[first:,e]).max()),
            "max_position_relation_error_rad":float(abs(pos[first:,e]).max()),
            "max_C_pin_relative_velocity_m_s":float(np.linalg.norm(hinge[first:,e],axis=-1).max())}
            for e in range(t.envs)],
        "peak_event":{"sample":sample,"pre_time_s":sample*t.dt,"environment":env,"joint":joint,
            "relation":relation,"post_velocity_residual_rad_s":float(vel[sample,env,j]),
            "position_relation_error_rad":float(pos[sample,env,j]),
            "finite_difference_relation_velocity_rad_s":float(dq[sample,env,j]),
            "C_pin_relative_velocity_local_m_s":hinge[sample,env,t.legs.index(leg)].tolist(),
            "C_pin_gap_local_m":gap[sample,env,t.legs.index(leg)].tolist(),
            "foot_force_xyz_N":feet[sample,env,t.legs.index(leg)].tolist(),
            "post_qd_rad_s":{n:float(v1[sample,env,t.joints.index(n)]) for n in (joint,relation["source_joint"])},
            "next_pre_cached_and_direct_qd_match":bool(np.array_equal(v1[sample,env],get("direct_pre_qd",t.joints)[sample+1,env])),
            "window":[{"sample":s,"post_velocity_residual_rad_s":float(vel[s,env,j]),
                "position_residual_rad":float(pos[s,env,j]),"finite_difference_residual_rad_s":float(dq[s,env,j]),
                "C_pin_relative_velocity_local_m_s":hinge[s,env,t.legs.index(leg)].tolist(),
                "foot_force_xyz_N":feet[s,env,t.legs.index(leg)].tolist()}
                for s in range(max(first,sample-3),min(t.samples,sample+4))]},
        "claimed_cause":"No unique cause established; measures a velocity-level constraint residual and motivates one targeted solver experiment.",
    }
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path);p.add_argument("kinematics",type=Path);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.write_text(json.dumps(analyze(args.report,args.kinematics),indent=2,sort_keys=True)+"\n")
