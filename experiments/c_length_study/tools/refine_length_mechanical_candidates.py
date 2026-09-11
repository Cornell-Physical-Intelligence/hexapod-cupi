#!/usr/bin/env python3
"""Check candidate path loads at finer temporal resolution before review."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from experiments.c_length_study.tools import screen_length_mechanics as s
from experiments.c_length_study.tools.run_length_mechanical_paths import admissible

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--screen",type=Path,required=True);p.add_argument("--static",type=Path,required=True)
p.add_argument("--package",type=Path,default=s.ROOT/"robot/hexapod_mkii_length_study")
p.add_argument("--output",type=Path,required=True);p.add_argument("--variants",nargs="+",required=True)
args=p.parse_args();report=json.loads((args.screen/"path_screen.json").read_text())
manifest=json.loads((args.package/"manifest.json").read_text());results={}
for name in args.variants:
    entry=report["variants"][name];xml=ET.parse(args.package/"urdf"/(name+".urdf")).getroot()
    robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,args.package))
    poses=json.loads((args.static/(name+"_poses.json")).read_text());results[name]={}
    for key,old in entry["best"].items():
        if key not in ("v0.05_h0.09","v0.10_h0.09","v0.20_h0.09"):continue
        pose=next(p for p in poses if p["femur_deg"]==old["femur_deg"] and p["knee_deg"]==old["knee_deg"])
        row=s.trajectory(robot,pose,old["gait"],old["stride_m"],old["lift_m"],256,speeds=(old["speed_mps"],))
        row.pop("samples",None)
        if row.get("force_balance_pass"):
            speed=row.pop("speed_results")[0];row.update(speed)
            row["screen_pass"]=bool(admissible(row,speed))
            if speed["force_balance_pass"]:
                row["peak_change_fraction_vs_64"]=row["minimum_required_peak_torque_nm"]/old["minimum_required_peak_torque_nm"]-1
        row.update(femur_deg=old["femur_deg"],knee_deg=old["knee_deg"],belly_clearance_m=old["belly_clearance_m"],
                   stance_width_m=old["stance_width_m"],stance_length_m=old["stance_length_m"])
        results[name][key]=row
        args.output.write_text(json.dumps(results,indent=2))
        print(name,key,row.get("minimum_required_peak_torque_nm"),row.get("screen_pass"),flush=True)
