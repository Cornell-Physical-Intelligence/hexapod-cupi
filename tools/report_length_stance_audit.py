#!/usr/bin/env python3
"""Refine and illustrate the C/D lower-stance audit without changing URDFs."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import screen_length_mechanics as s
from run_length_mechanical_paths import admissible
sys.path.insert(0,str(s.ROOT/"robot/tools"))
from pack_urdf_viewer import load_stl

BASE=s.ROOT/"artifacts/length_study_2026-09-09"
OUT=BASE/"stance_audit_all_low_poses"
audit=json.loads((OUT/"audit.json").read_text());assert audit["status"]=="completed"
review=json.loads((BASE/"stage1_candidate_review.json").read_text())
PKG=s.ROOT/"robot/hexapod_mkii_length_study"
manifest=json.loads((PKG/"manifest.json").read_text())
models={};results={};panels=[]
for name in ("f050_t060","f050_t080"):
    xml=ET.parse(PKG/"urdf"/(name+".urdf")).getroot()
    robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,PKG))
    poses=json.loads((BASE/"mechanics_stage1_v3_static"/(name+"_poses.json")).read_text())
    models[name]=(xml,robot,poses)
    high=next(c["refined_0p20"] for c in review["candidates"] if c["variant"]==name)
    options=[]
    for t in audit["variants"][name]["lower_pose_trials"]:
        if not .095<=t["belly_clearance_m"]<=.12 or not t.get("force_balance_pass"):continue
        if t["minimum_support_margin_m"]<.01 or t["minimum_nonfoot_clearance_m"]<.005 or t["minimum_stance_pad_gap_m"]<-.002:continue
        for v in t["speed_results"]:
            if v["speed_mps"]==.2 and v.get("force_balance_pass") and v.get("motor_speed_below_full_continuous_torque_region"):
                options.append((t,v))
    old,oldv=min(options,key=lambda pair:pair[1]["minimum_required_peak_torque_nm"])
    pose=next(p for p in poses if (p["femur_deg"],p["knee_deg"])==(old["femur_deg"],old["knee_deg"]))
    t=s.trajectory(robot,pose,old["gait"],old["stride_m"],old["lift_m"],256,speeds=(.1,.2))
    t.pop("samples",None)
    for v in t["speed_results"]:v["screen_pass"]=bool(admissible(t,v))
    t.update({k:pose[k] for k in ("femur_deg","knee_deg","belly_clearance_m","stance_width_m","stance_length_m")})
    results[name]=t
    panels.extend([(name,"Earlier selected stance",high),(name,"Lower stance audit",{**t,**t["speed_results"][1]})])
(OUT/"refined_lower_stances.json").write_text(json.dumps(results,indent=2))

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig,axes=plt.subplots(2,2,figsize=(10,8.5),layout="constrained")
colors={"coxa":"#8b9197","femur":"#167f83","tibia":"#d58533"};cache={}
for ax,(name,label,value) in zip(axes.flat,panels):
    xml,robot,poses=models[name]
    pose=next(p for p in poses if (p["femur_deg"],p["knee_deg"])==(value["femur_deg"],value["knee_deg"]))
    rows,_=robot.pose(np.array(pose["q"]));row=rows[0];leg=robot.legs[0]
    height=pose["root_height_m"]
    def project(points):return np.stack(((points-leg.yaw_origin)@leg.radial,points[...,2]+height),axis=-1)*1000
    for i,kind in enumerate(s.KINDS):
        link=xml.find(f"link[@name='{manifest['link_joint_mapping']['lf']['links'][kind]}']")
        for visual in link.findall("visual"):
            mesh=visual.find("geometry/mesh");file=mesh.get("filename").split("/")[-1]
            if file not in cache:cache[file]=load_stl(PKG/"meshes"/file)
            triangles=cache[file]*s.study.vec(mesh.get("scale","1 1 1"));T=row["transforms"][i]@s.study.origin(visual)
            world=np.einsum("ntj,ij->nti",triangles,T[:3,:3])+T[:3,3]
            ax.add_collection(PolyCollection(project(world),facecolors=colors[kind],edgecolors="none",rasterized=True))
    joints=project(np.array(row["pivots"]))
    ax.plot(joints[:,0],joints[:,1],"o",ms=4,mec="#25282b",mfc="white")
    belly=pose["belly_clearance_m"]*1000
    ax.add_patch(Rectangle((-55,belly),55,23,facecolor="#6e747a"))
    ax.axhline(0,color="#555555",lw=1)
    ax.set(xlim=(-60,225),ylim=(-5,215),aspect="equal",xlabel="Outward from coxa axis (mm)",ylabel="Height above ground (mm)")
    letter="C" if name=="f050_t060" else "D"
    peak=value["minimum_required_peak_torque_nm"]
    ax.set_title(f"{letter}: {label}\n{belly:.0f} mm belly clearance · {peak:.2f} N·m peak",fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Same leg lengths, different postures",fontsize=18)
fig.text(.02,-.03,"Exact mock leg meshes; body block schematic. Teal = femur, orange = tibia.\nTorque estimates at prescribed 0.20 m/s, 100 mm stride, 20 mm foot lift; 256 phase samples. No actual walking trial.",fontsize=10)
fig.savefig(OUT/"same_lengths_lower_stance.png",dpi=170,bbox_inches="tight");plt.close(fig)

lines=["# Stance audit: why tibia length and body height changed the shortlist","",
"The original Stage 1 shortlist did not establish the best body posture for steady walking. C remains a provisional geometry candidate; the earlier tall poses are examples, not required geometry or validated walking stances. A remains excluded by the user for motor packaging. Stage 2 is still held.","",
"## Torque explanation","",
"C and D used the same nominal femur/knee angles (10/110 degrees), but extending a tibia that points inward brought the foot closer to the hip. At the nominal left-front pose, the radial foot-to-hip offset fell from 61.45 mm for C to 53.66 mm for D; the knee-to-foot offset magnitude increased from 9.95 to 17.74 mm. Longer links do not increase every joint's perpendicular force lever arm together.","",
"The full load calculation balances link gravity, motion, motor terms and 3D contact forces. The contact solver minimizes the largest motor torque at every phase and can change horizontal force sharing. It does not prescribe a realizable feedback controller or smooth force transitions. The 1.381 versus 1.349 N·m difference is about 2.3%, not decisive evidence for the longer tibia. Force-to-joint torque follows the Jacobian-transpose relationship explained in [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/).","",
"## Selection gap and additional checks","",
"The original pose selection retained minimum-static-torque seeds per knee angle and minimum-clearance threshold. A minimum-height threshold can repeatedly select a taller posture. Lower poses with slightly higher static load but better motion headroom could be discarded before the path screen. The screen also prescribed a level, constant-velocity body and therefore did not measure roll, pitch, heave or disturbance recovery. The previous recommendation overstated how much those results supported overall steady walking.","",
f"The additional CPU audit evaluated all retained original static-grid poses between 80 and 140 mm belly clearance for C and D: {sum(len(v['lower_pose_trials']) for v in audit['variants'].values())} trajectory configurations, with tripod/ripple/wave schedules, 40/60/100 mm strides, 20 mm lift and 0.05/0.10/0.20 m/s targets. This is an expanded finite search, not an exhaustive workspace or gait proof. The 40 mm stride extends the original trajectory range.","",
"For each geometry, the lowest estimated peak at 0.20 m/s within 95–120 mm belly clearance was rechecked at 256 phase samples. The heights differ, so this is a low-posture comparison rather than an exactly matched-height experiment.","",
"| Geometry | Belly clearance | Femur / knee angles | Peak at 0.10 m/s | Peak at 0.20 m/s | 0.20 m/s screen |",
"|---|---:|---:|---:|---:|---|"]
for name,t in results.items():
    v1,v2=t["speed_results"]
    lines.append(f"| {name} | {t['belly_clearance_m']*1000:.1f} mm | {t['femur_deg']} / {t['knee_deg']}° | {v1['minimum_required_peak_torque_nm']:.3f} N·m | {v2['minimum_required_peak_torque_nm']:.3f} N·m | {'Pass' if v2['screen_pass'] else 'Fail'} |")
lines += ["",f"![Same geometry in taller and lower poses]({OUT/'same_lengths_lower_stance.png'})","",
"Lower body height can reduce the overturning moment from horizontal acceleration at a given support footprint. The actual benefit depends on whole-robot COM height, foot placement, contact forces, available joint travel and ground clearance. A visually crouched pose alone is not a stability or efficiency certificate.","",
"Next mechanical comparison should explicitly vary body height and stance width, preserve motion headroom before pruning poses, and present the torque/work/workspace tradeoffs at common absolute heights. Extend that comparison to the remaining geometries before using the original shortlist as a final ranking. Keep production hardware fit and real joint/linkage limits separate from the mock sensitivity study. No GPU trial has started.",""]
(OUT/"STANCE_AUDIT.md").write_text("\n".join(lines))
print(json.dumps({n:{"belly_mm":t["belly_clearance_m"]*1000,"speeds":t["speed_results"]} for n,t in results.items()},indent=2))
