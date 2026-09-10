#!/usr/bin/env python3
"""Equal-budget inverse-dynamics path screen after all-size static search."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import screen_length_mechanics as s


def choose_poses(poses,gait):
    eligible=[p for p in poses if gait in p["gaits"] and p["gaits"][gait]["minimum_support_margin_m"]>=.01]
    selected={}
    # Include multiple knee openings and multiple ground-clearance regimes.
    # The same selection rule applies to every geometry, including controls.
    for knee in (100,110,120,130,140):
        choices=[p for p in eligible if p["knee_deg"]==knee]
        if choices:
            p=min(choices,key=lambda p:p["gaits"][gait]["worst_peak_torque_nm"])
            selected[(p["femur_deg"],p["knee_deg"])]=p
    for clearance in (.09,.12,.15):
        choices=[p for p in eligible if p["belly_clearance_m"]>=clearance]
        if choices:
            p=min(choices,key=lambda p:p["gaits"][gait]["worst_peak_torque_nm"])
            selected[(p["femur_deg"],p["knee_deg"])]=p
    return list(selected.values())


def admissible(row,speed):
    if not row.get("workspace_pass") or not row.get("force_balance_pass"):return False
    return (row["minimum_support_margin_m"]>=.01 and row["minimum_nonfoot_clearance_m"]>=.005
            and row["minimum_stance_pad_gap_m"]>=-.002 and speed.get("force_balance_pass")
            and speed["minimum_required_peak_torque_nm"]<=1.6
            and speed.get("motor_speed_below_full_continuous_torque_region",False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--package",type=Path,default=s.ROOT/"robot/hexapod_mkii_length_study")
    p.add_argument("--variants",nargs="*");p.add_argument("--phase-count",type=int,default=64)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    static=json.loads((args.static/"mechanical_screen.json").read_text())
    if static["status"]!="completed":raise RuntimeError("Static screen must finish before path screening")
    manifest=json.loads((args.package/"manifest.json").read_text());start=time.time()
    # Capture exact source files before computation begins.
    for path in (Path(__file__),Path(s.__file__)):
        (args.output/path.name).write_bytes(path.read_bytes())
    report={"status":"running","method":"sampled_inverse_dynamics_with_minimax_contact_force_distribution",
            "phase_count":args.phase_count,"stride_lengths_m":[.06,.10],"foot_lift_m":.02,
            "speeds_mps":[.05,.10,.20,.30],"variants":{},"started_unix":start,
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(s.__file__))}}
    records=manifest["variants"]
    for record in records:
        name=record["variant"]
        if args.variants and name not in args.variants:continue
        xml=ET.parse(args.package/record["urdf"]).getroot()
        assert hashlib.sha256((args.package/record["urdf"]).read_bytes()).hexdigest()==record["sha256"]
        robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,args.package))
        poses=json.loads((args.static/(name+"_poses.json")).read_text());trials=[];best={}
        for gait in ("tripod","ripple","wave"):
            for pose in choose_poses(poses,gait):
                for stride in (.06,.10):
                    row=s.trajectory(robot,pose,gait,stride,.02,args.phase_count)
                    row.pop("samples",None)
                    row.update(femur_deg=pose["femur_deg"],knee_deg=pose["knee_deg"],belly_clearance_m=pose["belly_clearance_m"],
                               stance_width_m=pose["stance_width_m"],stance_length_m=pose["stance_length_m"])
                    for speed in row.get("speed_results",[]):
                        speed["screen_pass"]=bool(admissible(row,speed))
                        if speed["screen_pass"]:
                            for clearance in (.06,.09,.12,.15):
                                if row["belly_clearance_m"]+1e-9<clearance:continue
                                key=f"v{speed['speed_mps']:.2f}_h{clearance:.2f}"
                                value={k:v for k,v in row.items() if k!="speed_results"}
                                value.update(speed)
                                if key not in best or speed["minimum_required_peak_torque_nm"]<best[key]["minimum_required_peak_torque_nm"]:
                                    best[key]=value
                    trials.append(row)
        entry={"femur_length_m":record["femur_length_m"],"tibia_length_m":record["tibia_length_m"],
               "urdf_sha256":record["sha256"],"total_mass_kg":robot.mass,"trials":len(trials),"best":best}
        (args.output/(name+"_trials.json")).write_text(json.dumps(trials,indent=2))
        report["variants"][name]=entry
        report["elapsed_s"]=time.time()-start
        (args.output/"path_screen.json").write_text(json.dumps(report,indent=2))
        print(name,len(trials),{k:round(v["minimum_required_peak_torque_nm"],3) for k,v in best.items() if k.endswith('h0.09')},flush=True)
    report.update(status="completed",elapsed_s=time.time()-start)
    (args.output/"path_screen.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
