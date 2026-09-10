#!/usr/bin/env python3
"""Stage-one review figures from exact URDF meshes and completed CPU results."""
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
sys.path.insert(0,str(s.ROOT/"robot/tools"))
from pack_urdf_viewer import load_stl

OUT=s.ROOT/"artifacts/length_study_2026-09-09"
main=json.loads((OUT/"mechanics_stage1_motor_complete/path_screen.json").read_text())
boundary=json.loads((OUT/"mechanics_stage1_boundary_paths/path_screen.json").read_text())
assert main["status"]==boundary["status"]=="completed"
refined={**json.loads((OUT/"stage1_refined_motor_candidates.json").read_text()),
         **json.loads((OUT/"stage1_refined_boundary_candidates.json").read_text())}
entries={**main["variants"],**boundary["variants"]}
choices=[("A","f040_t050","Boundary probe: motor fit unresolved"),
         ("B","f050_t050","Compact candidate"),
         ("C","f050_t060","Balanced candidate"),
         ("D","f050_t080","More ground clearance"),
         ("E","f050_t100","Long-tibia energy comparison"),
         ("F","f055_t060","More femur room; less torque margin")]
decisions_path=OUT/"stage1_user_review.json"
decisions=json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
rejected=decisions.get("rejected_candidates",[])
rejected_names={r["variant"] for r in rejected}
choices=[c for c in choices if c[1] not in rejected_names]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.titlesize":15,"axes.labelsize":12})
colors={"coxa":"#8b9197","femur":"#167f83","tibia":"#d58533"}
meshcache={}


def model(name,pose):
    package=s.ROOT/"robot"/("hexapod_mkii_length_study_boundary" if name in boundary["variants"] else "hexapod_mkii_length_study")
    xml=ET.parse(package/"urdf"/(name+".urdf")).getroot();manifest=json.loads((package/"manifest.json").read_text())
    robot=s.Robot(xml,manifest["link_joint_mapping"],s.mesh_clouds(xml,package))
    q=np.tile(np.radians([0,pose["femur_deg"],pose["knee_deg"]]),(6,1))
    rows,_=robot.pose(q);contacts,_,_=robot.geometry(rows);height=-contacts[:,2].min()
    return package,xml,manifest,robot,rows,height


def draw_leg(ax,name,pose,ghost=False):
    package,xml,manifest,robot,rows,height=model(name,pose);leg=robot.legs[0];row=rows[0]
    def project(points):return np.stack(((points-leg.yaw_origin)@leg.radial,points[...,2]+height),axis=-1)*1000
    joints=project(np.array(row["pivots"]+[row["foot"]]))
    if ghost:
        ax.plot(joints[:,0],joints[:,1],"--",color="#b4b9be",lw=1.8,zorder=0)
        return
    for index,kind in enumerate(s.KINDS):
        link=xml.find(f"link[@name='{manifest['link_joint_mapping']['lf']['links'][kind]}']")
        for visual in link.findall("visual"):
            mesh=visual.find("geometry/mesh");file=mesh.get("filename").split("/")[-1]
            if file not in meshcache:meshcache[file]=load_stl(package/"meshes"/file)
            triangles=meshcache[file]*s.study.vec(mesh.get("scale","1 1 1"))
            T=row["transforms"][index]@s.study.origin(visual)
            world=np.einsum("ntj,ij->nti",triangles,T[:3,:3])+T[:3,3]
            ax.add_collection(PolyCollection(project(world),facecolors=colors[kind],edgecolors="none",rasterized=True,zorder=2))
    ax.add_patch(Rectangle((-55,(height-.023)*1000),55,23,facecolor="#6e747a",zorder=1))
    ax.plot(joints[:3,0],joints[:3,1],"o",ms=4,mec="#25282b",mfc="white",zorder=4)


baseline=refined["f100_t100"]["v0.05_h0.09"]
fig,axes=plt.subplots(3,2,figsize=(10,12),layout="constrained")
fig.suptitle("Stage 1 candidates — same scale, exact mock mesh profiles",fontsize=18)
review=[]
for ax,(letter,name,note) in zip(axes.flat,choices):
    value=refined[name]["v0.20_h0.09"]
    assert value["screen_pass"],name
    entry=entries[name];review.append({"label":letter,"variant":name,"note":note,**entry,"refined_0p20":value})
    draw_leg(ax,"f100_t100",baseline,True);draw_leg(ax,name,value)
    ax.axhline(0,color="#41464b",lw=1)
    ax.set(xlim=(-60,225),ylim=(-8,285),aspect="equal",xlabel="Outward from coxa axis (mm)",ylabel="Height above ground (mm)")
    ax.set_title(f"{letter} · {entry['femur_length_m']*1000:g} / {entry['tibia_length_m']*1000:.0f} mm\n{note}",fontsize=13)
    ax.text(.04,.96,f"Peak estimate {value['minimum_required_peak_torque_nm']:.2f} N·m",transform=ax.transAxes,va="top",fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10)
for ax in list(axes.flat)[len(choices):]:
    ax.axis("off")
if rejected:
    axes.flat[-1].text(.08,.6,"A excluded by user review\n58 / 105 mm\nFemur too short for motor packaging",transform=axes.flat[-1].transAxes,fontsize=13,va="center",color="#555555",linespacing=1.6)
fig.text(.02,-.025,"Femur / tibia lengths. Coxa unchanged. Teal = femur; orange = tibia. Dashed = original mock baseline.\nEach solid profile uses its screened stance. These are geometry/load candidates; no walking trial or production CAD fit is certified.",fontsize=12)
fig.savefig(OUT/"stage1_candidate_profiles.png",dpi=170,bbox_inches="tight");plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout="constrained")
sizes=[50,60,70,80,90,100,110]
for ax,speed in zip(axes,(.1,.2)):
    a=np.full((7,7),np.nan)
    for fi,f in enumerate(sizes):
        for ti,t in enumerate(sizes):
            value=main['variants'][f'f{f:03d}_t{t:03d}']['best'].get(f'v{speed:.2f}_h0.09')
            if value:a[fi,ti]=value['minimum_required_peak_torque_nm']
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#e9e9e9')
    im=ax.imshow(a,vmin=.6,vmax=1.6,cmap=cmap)
    for i in range(7):
        for j in range(7):
            ax.text(j,i,f'{a[i,j]:.2f}' if np.isfinite(a[i,j]) else '—',ha='center',va='center',fontsize=10,color='white' if a[i,j]>1.25 else '#222222')
    ax.set(xticks=range(7),yticks=range(7),xticklabels=sizes,yticklabels=sizes,xlabel='Tibia (% of 210 mm)',ylabel='Femur (% of 145 mm)',title=f'{speed:.2f} m/s target')
fig.colorbar(im,ax=axes,label='Minimum estimated peak joint torque (N·m)',shrink=.8)
fig.suptitle('Original 49-size screen: 20 mm foot lift, at least 90 mm belly clearance',fontsize=15)
fig.text(.02,-.04,'— = no sampled motion passed; not proof that the geometry cannot walk. Values use 64 phase samples; shortlisted cases are rechecked at 256.',fontsize=11)
fig.savefig(OUT/"stage1_torque_map.png",dpi=180,bbox_inches="tight");plt.close(fig)

(OUT/"stage1_candidate_review.json").write_text(json.dumps({"status":decisions.get("status","awaiting_user_candidate_confirmation"),"candidates":review,"user_review":decisions},indent=2))
text=['# Stage 1: candidates for your smell test','',
      '**The user selected C and authorized a full-robot walking policy. Manufacturing fit and hardware performance remain unverified.**' if decisions.get('stage2_authorized') else '**Stage 2 is held pending your confirmation. No hardware length has been selected.**','',
      '**Stance audit update:** The original pose selection missed lower alternatives. C also passes a refined prescribed-motion check at about 108 mm belly clearance with 1.387 N·m peak at 0.20 m/s. The tall poses below are examples, not required postures or a steady-walking ranking. See [the stance audit](stance_audit_all_low_poses/STANCE_AUDIT.md) for the corrected interpretation and images.','',
      'User review: A (58 / 105 mm) is excluded because its femur is too short to fit the motors. B–F remain under consideration. C is the recommended lead for further testing; this recommendation is not user approval to start Stage 2.' if rejected else 'Candidate review is pending.','',
      f'Screened 49 original sizes plus 9 boundary/intermediate probes at the same 8.2608 kg mass and 1.6 N·m continuous torque limit. Coxa remained fixed. The full path search evaluated {sum(v["trials"] for v in entries.values())} sampled trajectory configurations across four requested speeds.','',
      f'![Candidate profiles]({OUT / "stage1_candidate_profiles.png"})','',
      '| Candidate | Femur / tibia (mm) | Peak at 0.20 m/s (N·m) | Margin below 1.6 | Belly clearance (mm) | Positive mechanical work (J/m) | Role |',
      '|---|---:|---:|---:|---:|---:|---|']
for r in review:
    v=r['refined_0p20'];peak=v['minimum_required_peak_torque_nm']
    text.append(f"| {r['label']} · {r['variant']} | {r['femur_length_m']*1000:g} / {r['tibia_length_m']*1000:.0f} | {peak:.2f} | {100*(1-peak/1.6):.0f}% | {v['belly_clearance_m']*1000:.0f} | {v['positive_mechanical_work_per_m_j']:.1f} | {r['note']} |")
text+=['','These figures are inverse-dynamics estimates for prescribed motion with optimized contact-force sharing. They are not measured motor current, battery energy, validated walking speed or a stability guarantee.','',
       'C (72.5 / 126 mm) is the recommended lead for the next test. Compared with B, it provides about 23 mm more nominal belly clearance and 16% less estimated positive mechanical work per distance, with nearly identical worst-motor RMS torque. D is the closest competitor: about 41 mm more clearance than C, slightly lower peak torque, and similar work demand. The screen has not measured whether either gives steadier real body motion. E retains the long-tibia comparison; F is a longer-femur packaging fallback to check. A is excluded by the user for motor packaging. Rigid motor, shaft, bearing and linkage fit remains unverified for the remaining candidates. Do not manufacture from these stretched meshes.','',
       f'![Full torque map]({OUT / "stage1_torque_map.png"})','',
       '## What was checked','',
       '- All original 49 URDFs and 9 generated probe URDFs retain their exact recorded hashes, transferred mass properties, fixed coxa and original mock joint conventions.',
       '- A common grid of femur/knee postures, six/three/four/five-foot support, 3D force/moment balance, unilateral contact, a conservative friction pyramid (coefficient 0.8), joint range and actual mesh extrema.',
       '- Tripod/ripple/wave schedules; duty factors 0.65/0.80/0.90; 60 and 100 mm stance travel; 20 mm foot lift; 0.05/0.10/0.20/0.30 m/s targets. Each geometry uses the same pose-selection rules.',
       '- Periodic leg linear/angular acceleration, transformed inertia tensors, 0.0007 kg·m² joint armature, 0.01 N·m Coulomb friction and 0.002 N·m·s viscous friction. Candidate speed demand stays within the motor model’s full continuous-torque region.',
       '- At least 10 mm COM-to-support-edge margin, 5 mm nonfoot clearance, and no more than 2 mm point-contact/pad discrepancy. Candidates were rechecked at 256 phase samples after the 64-sample screen.',
       '- Independent tests check inverse kinematics against exact forward kinematics for all six legs, Jacobians against finite differences, gravity torque against potential-energy derivatives, and force/moment/friction constraints. All 304 CPU tests passed.','',
       '## What the screen does not establish','',
       'The body trajectory is prescribed for inverse dynamics; a controller has not yet demonstrated that motion. Contact force transitions, impacts, slip, disturbance recovery, feedback tracking, inter-leg/self-collision and thermal endurance remain for Isaac and hardware checks. Mock joint stops and the production four-bar transmission also need reconciliation.','',
       'Mass and transferred link inertia were held fixed across lengths. This separates a length sensitivity study from a true CAD redesign; final masses, COMs and inertias must be regenerated from rigid parts. Mechanical work omits motor/drive heating and other electrical losses.','',
       'No sampled motion passed at 0.30 m/s. The original 145/210 mm mock baseline had a passing sampled case only at 0.05 m/s. These are findings about the tested motion family, not upper bounds on either design’s capability. The actual production CAD URDF is a separate control and is not equivalent to the mock baseline.','',
       'The search still benefits from shorter femurs near its lower boundary. The minimum manufacturable femur is unresolved; the boundary probes must not be mistaken for a certified optimum. Different duty factors, longer strides, alternative body motion and additional postures may change the ranking.','',
       '## Review decision','',
       'Please identify the candidate dimensions that look mechanically plausible and any you want ruled out. Stage 2 controlled walking trials will wait for that confirmation. All GPU training remains held.','']
(OUT/"STAGE1_CANDIDATE_REVIEW.md").write_text('\n'.join(text))
print(json.dumps({r['label']:[r['variant'],round(r['refined_0p20']['minimum_required_peak_torque_nm'],4)] for r in review},indent=2))
