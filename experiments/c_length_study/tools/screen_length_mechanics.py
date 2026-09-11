#!/usr/bin/env python3
"""Quasi-static, point-contact geometry screen. No learned policy or GPU.

Loads solve full force/moment balance and an inscribed Coulomb friction pyramid.
Reported torque is a lower bound on the worst motor load at a sampled pose;
it excludes inertia/acceleration, gearbox losses and active balance transients.
Failure of this finite pose/trajectory search is not geometric impossibility.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog, least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"robot/tools"))
import generate_length_study as study

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
KINDS = ("coxa", "femur", "tibia")
SUPPORTS = {
    "six": [tuple(range(6))],
    "tripod": [(0, 2, 4), (1, 3, 5)],
    "ripple": [tuple(i for i in range(6) if i not in pair) for pair in ((0,5),(1,3),(2,4))],
    "wave": [tuple(i for i in range(6) if i != lifted) for lifted in range(6)],
}
PHASE = {"tripod": np.array([0,.5,0,.5,0,.5]),
         "ripple": np.array([0,1/3,2/3,1/3,2/3,0]),
         "wave": np.array([0,2/6,4/6,3/6,5/6,1/6])}
DUTY = {"tripod":.65,"ripple":.80,"wave":.90}


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])


class Leg:
    def __init__(self, xml, mapping, clouds):
        self.joints = [xml.find(f"joint[@name='{mapping['joints'][k]}']") for k in KINDS]
        self.links = [xml.find(f"link[@name='{mapping['links'][k]}']") for k in KINDS]
        self.origins = [study.origin(j) for j in self.joints]
        self.axes = [study.vec(j.find("axis").get("xyz")) for j in self.joints]
        self.properties = [study.inertial(link)[:2] for link in self.links]
        self.inertias = [study.inertial(link)[2] for link in self.links]
        limits = np.array([[float(j.find("limit").get(k)) for k in ("lower","upper")] for j in self.joints])
        self.lower = limits[:,0]+.025*(limits[:,1]-limits[:,0])
        self.upper = limits[:,1]-.025*(limits[:,1]-limits[:,0])
        self.clouds = [clouds[mapping['links'][k]] for k in KINDS]
        # A point at the extreme longitudinal toe, with transverse coordinates
        # taken from that actual mesh section. Exact pad geometry is audited too.
        tip = self.clouds[2]
        self.toe = tip[tip[:,1] > tip[:,1].max()-1e-5].mean(axis=0)
        self.distal = tip[:,1] > .93*tip[:,1].max()
        zero=self.fk(np.zeros(3))
        T0,T1,T2=zero["transforms"]
        self.yaw_origin=T0[:3,3]
        self.radial=T1[:3,:3]@np.array([0.,1.,0.])
        self.tangent=np.cross([0.,0.,1.],self.radial)
        self.hip_offset=T1[:3,3]-self.yaw_origin
        self.femur_length=float((T2[:3,3]-T1[:3,3])@self.radial)
        self.toe_rotation=T2[:3,:3]
        self.radial_angle=math.atan2(self.radial[1],self.radial[0])

    def fk(self, q, toe=None):
        T=np.eye(4); transforms=[]; pivots=[]; axes=[]; com=[]
        for origin, axis, angle, prop in zip(self.origins,self.axes,q,self.properties):
            T=T@origin; pivots.append(T[:3,3].copy()); axes.append(T[:3,:3]@axis)
            R=np.eye(4);R[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix();T=T@R
            transforms.append(T.copy());com.append(T[:3,:3]@prop[1]+T[:3,3])
        foot=T[:3,:3]@(self.toe if toe is None else toe)+T[:3,3]
        J=np.stack([np.cross(a,foot-p) for a,p in zip(axes,pivots)],axis=1)
        gravity=np.array([-sum(float(axes[i]@np.cross(com[j]-pivots[i],np.array([0,0,-9.81*self.properties[j][0]])))
                               for j in range(i,3)) for i in range(3)])
        return {"foot":foot,"J":J,"gravity":gravity,"transforms":transforms,"com":com,"axes":axes,"pivots":pivots}

    def ik(self,target,initial,toe):
        # Closed-form planar 3R solution, verified against the exact URDF FK.
        # The small off-plane toe coordinate is retained in the yaw solution.
        d=target-self.yaw_origin;tip=self.toe_rotation@toe
        lateral=float((self.hip_offset+tip)@self.tangent)
        rho=float(np.linalg.norm(d[:2]));radial2=rho*rho-lateral*lateral
        if radial2>0:
            radius=math.sqrt(radial2)-float(self.hip_offset@self.radial)
            height=float(d[2]-self.hip_offset[2]);L=self.femur_length
            ty=float(tip@self.radial);tz=float(tip[2]);K=math.hypot(ty,tz)
            cosine=(radius*radius+height*height-L*L-K*K)/(2*L*K)
            delta=math.acos(float(np.clip(cosine,-1,1)))
            yaw=math.atan2(d[1],d[0])-math.asin(lateral/rho)-self.radial_angle
            q=np.array([(yaw+math.pi)%(2*math.pi)-math.pi,
                        math.atan2(height,radius)+math.atan2(K*math.sin(delta),L+K*math.cos(delta)),
                        delta+math.atan2(tz,ty)])
            clipped=np.clip(q,self.lower,self.upper)
            error=float(np.linalg.norm(self.fk(clipped,toe)["foot"]-target))
            if error<1e-5 or (np.any(q<self.lower-1e-5) or np.any(q>self.upper+1e-5)) or abs(cosine)>1:
                return clipped,error
        def fun(q):return self.fk(q,toe)["foot"]-target
        def jac(q):return self.fk(q,toe)["J"]
        opt=least_squares(fun,np.clip(initial,self.lower+1e-8,self.upper-1e-8),jac=jac,
                          bounds=(self.lower,self.upper),gtol=1e-9,ftol=1e-9,xtol=1e-9,max_nfev=25)
        return opt.x,float(np.linalg.norm(fun(opt.x)))


class Robot:
    def __init__(self, xml, mapping, clouds):
        self.xml=xml
        self.legs=[Leg(xml,mapping[name],clouds) for name in LEGS]
        self.body=study.inertial(xml.find("link[@name='body_mock']"))[:2]
        self.body_cloud=clouds["body_mock"]
        self.mass=self.body[0]+sum(m for leg in self.legs for m,_ in leg.properties)
        self.armature=.0007
        self.dynamic_friction=.01
        self.viscous_friction=.002

    def pose(self,q,toes=None):
        rows=[leg.fk(angles,None if toes is None else toes[i]) for i,(leg,angles) in enumerate(zip(self.legs,q))]
        com=(self.body[0]*self.body[1]+sum((mass*p for leg,row in zip(self.legs,rows)
             for (mass,_),p in zip(leg.properties,row["com"])),np.zeros(3)))/self.mass
        return rows,com

    def geometry(self,rows):
        low=float(self.body_cloud[:,2].min()); contacts=[]; toes=[]
        for leg,row in zip(self.legs,rows):
            for i,(cloud,T) in enumerate(zip(leg.clouds,row["transforms"])):
                points=np.einsum("nj,ij->ni",cloud,T[:3,:3])+T[:3,3]
                if i==2:
                    candidates=np.flatnonzero(leg.distal)
                    ix=candidates[np.argmin(points[candidates,2])]
                    contacts.append(points[ix]);toes.append(cloud[ix])
                    low=min(low,float(points[~leg.distal,2].min()))
                else:low=min(low,float(points[:,2].min()))
        return np.array(contacts),np.array(toes),low

    def load(self,rows,com,support,friction=.8,required_wrench=None):
        """Minimize worst absolute torque, allowing unequal 3D contact forces."""
        n=len(support);nv=3*n+1;eq=np.zeros((6,nv));ub=[];bub=[]
        for slot,i in enumerate(support):
            eq[:3,3*slot:3*slot+3]=np.eye(3)
            eq[3:,3*slot:3*slot+3]=skew(rows[i]["foot"])
            for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
                a=np.zeros(nv);a[3*slot:3*slot+3]=[sx,sy,-friction];ub.append(a);bub.append(0.)
        for i,row in enumerate(rows):
            for j in range(3):
                c=np.zeros(nv)
                if i in support:
                    slot=support.index(i);c[3*slot:3*slot+3]=-row["J"][:,j]
                for sign in (-1,1):
                    a=sign*c;a[-1]=-1;ub.append(a);bub.append(-sign*row["gravity"][j])
        weight=np.array([0,0,self.mass*9.81])
        b=np.r_[weight,np.cross(com,weight)] if required_wrench is None else required_wrench
        normal=float(b[2])
        if normal<0 or abs(b[0])+abs(b[1])>friction*normal+1e-8:return None
        objective=np.zeros(nv);objective[-1]=1
        # Finite bounds follow directly from nonnegative normal forces and
        # their fixed sum. They remove unconstrained columns near singular poses.
        force_bounds=((-friction*normal,friction*normal),(-friction*normal,friction*normal),(0,normal))
        arguments=dict(A_ub=np.array(ub),b_ub=bub,A_eq=eq,b_eq=b,
                    bounds=[bound for _ in support for bound in force_bounds]+[(0,None)])
        opt=linprog(objective,**arguments,method="highs")
        if opt.status not in (0,2):
            opt=linprog(objective,**arguments,method="highs-ds",options={"presolve":False})
        if opt.status not in (0,2):
            rounded={**arguments,"A_eq":np.round(eq,10),"A_ub":np.round(arguments["A_ub"],10)}
            opt=linprog(objective,**rounded,method="highs-ipm",options={"presolve":False})
        if opt.status==2:return None
        if not opt.success:raise RuntimeError("Contact solver numerical failure: "+opt.message)
        forces=np.zeros((6,3));forces[list(support)]=opt.x[:-1].reshape(n,3)
        torques=np.array([r["gravity"]-r["J"].T@f for r,f in zip(rows,forces)])
        hull=ConvexHull(np.array([rows[i]["foot"][:2] for i in support]))
        margin=float((-hull.equations[:,:2]@com[:2]-hull.equations[:,2]).min())
        residual=float(np.max(np.abs(eq@opt.x-b)))
        if residual>1e-6:raise RuntimeError("Invalid contact-force balance")
        return {"peak_torque_nm":float(np.abs(torques).max()),"support_margin_m":margin,
                "forces_n":forces.tolist(),"torques_nm":torques.tolist(),"balance_residual":residual}


def mesh_clouds(xml,package):
    result={};cache={}
    for link in xml.findall("link"):
        points=[]
        for visual in link.findall("visual"):
            mesh=visual.find("geometry/mesh")
            file=mesh.get("filename").split("/")[-1]
            if file not in cache:
                vertices=study.stl_vertices(package/"meshes"/file)
                # Preserve distal-foot and nonfoot hulls separately.
                parts=[vertices]
                if file=="Tibia.stl":
                    distal=vertices[:,1]>.93*vertices[:,1].max();parts=[vertices[distal],vertices[~distal]]
                cache[file]=np.concatenate([part[ConvexHull(part).vertices] for part in parts])
            cloud=cache[file]*study.vec(mesh.get("scale","1 1 1"));T=study.origin(visual)
            transformed=np.einsum("nj,ij->ni",cloud,T[:3,:3])+T[:3,3]
            if not np.isfinite(transformed).all():raise RuntimeError("Nonfinite mesh transform")
            points.append(transformed)
        result[link.get("name")]=np.concatenate(points)
    return result


def scan(robot):
    results=[]
    for f in range(10,81,10):
        for t in range(80,141,10):
            q=np.tile(np.radians([0,f,t]),(6,1));rows,com=robot.pose(q)
            contacts,toes,nonfoot=robot.geometry(rows)
            ground=float(contacts[:,2].min());height=-ground
            belly=height+float(robot.body_cloud[:,2].min())
            if belly<.06 or nonfoot-ground<.005 or np.ptp(contacts[:,2])>.001:continue
            rows,com=robot.pose(q,toes)
            data={"femur_deg":f,"knee_deg":t,"root_height_m":height,"belly_clearance_m":belly,
                  "nonfoot_clearance_m":nonfoot-ground,"q":q.tolist(),"toes":toes.tolist(),
                  "stance_width_m":float(np.ptp(contacts[:,0])),"stance_length_m":float(np.ptp(contacts[:,1])),"gaits":{}}
            for gait,sets in SUPPORTS.items():
                loads=[robot.load(rows,com,support) for support in sets]
                if all(load is not None for load in loads):
                    data["gaits"][gait]={"worst_peak_torque_nm":max(l["peak_torque_nm"] for l in loads),
                         "minimum_support_margin_m":min(l["support_margin_m"] for l in loads)}
            results.append(data)
    return results


def trajectory(robot,pose,gait,stride,lift,phase_count=64,speeds=(.05,.10,.20,.30)):
    """Sample point-foot step workspace and quasistatic loads, not dynamics."""
    baseq=np.array(pose["q"]);toes=np.array(pose["toes"])
    initial,_=robot.pose(baseq,toes);feet=np.array([r["foot"] for r in initial]);q=baseq.copy()
    maxerror=0.;peak=0.;minmargin=1.;minclear=1.;mingap=1.;samples=[];all_rows=[];all_com=[];supports=[]
    for index in range(phase_count):
        phase=((index+.5)/phase_count+PHASE[gait])%1
        beta=DUTY[gait];stance=phase<beta
        target=feet.copy()
        u=np.where(stance,phase/beta,(phase-beta)/(1-beta))
        blend=10*u**3-15*u**4+6*u**5
        ratio=(1-beta)/beta
        target[:,1]+=np.where(stance,stride*(u-.5),stride*(.5+ratio*u-(1+ratio)*blend))
        target[:,2]+=np.where(stance,0,lift*64*u**3*(1-u)**3)
        for i,leg in enumerate(robot.legs):
            q[i],error=leg.ik(target[i],q[i],toes[i]);maxerror=max(error,maxerror)
        if maxerror>.001:
            return {"workspace_pass":False,"max_ik_error_m":maxerror,"gait":gait,"stride_m":stride,"lift_m":lift}
        rows,com=robot.pose(q,toes);contacts,_,nonfoot=robot.geometry(rows)
        minclear=min(minclear,nonfoot+pose["root_height_m"])
        mingap=min(mingap,float((contacts[stance,2]+pose["root_height_m"]).min()))
        load=robot.load(rows,com,tuple(np.flatnonzero(stance)))
        if load is None:
            return {"workspace_pass":True,"force_balance_pass":False,"gait":gait,"stride_m":stride,"lift_m":lift}
        peak=max(peak,load["peak_torque_nm"]);minmargin=min(minmargin,load["support_margin_m"])
        samples.append({"phase":float((index+.5)/phase_count),"q":q.tolist(),"load":load})
        all_rows.append(rows);all_com.append(com);supports.append(tuple(np.flatnonzero(stance)))
    # Central differences give a kinematic rate estimate at commanded 0.20m/s.
    positions=np.array([s["q"] for s in samples]);period=stride/(.2*DUTY[gait])
    velocity=(np.roll(positions,-1,axis=0)-np.roll(positions,1,axis=0))/(2*period/phase_count)
    joint_acceleration=(np.roll(positions,-1,axis=0)-2*positions+np.roll(positions,1,axis=0))*phase_count**2
    # Inverse dynamics of the prescribed periodic motion. Body has constant
    # world velocity and fixed orientation; legs create inertial reaction loads.
    # This is a feasibility calculation, not evidence that feedback can track it.
    centers=np.array([[r["com"] for r in rows] for rows in all_rows])
    acc=(np.roll(centers,-1,axis=0)-2*centers+np.roll(centers,1,axis=0))*phase_count**2
    rotations=np.array([[[T[:3,:3] for T in r["transforms"]] for r in rows] for rows in all_rows])
    relative=np.roll(rotations,-1,axis=0)@np.swapaxes(np.roll(rotations,1,axis=0),-1,-2)
    omega=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(phase_count,6,3,3)*(phase_count/2)
    alpha=(np.roll(omega,-1,axis=0)-np.roll(omega,1,axis=0))*(phase_count/2)
    dynamics=[]
    for n,rows in enumerate(all_rows):
        wrench=np.zeros(6);extra=np.zeros((6,3))
        for i,(leg,row) in enumerate(zip(robot.legs,rows)):
            moments=[];forces=[]
            for j,(mass,_) in enumerate(leg.properties):
                R=rotations[n,i,j];I=R@leg.inertias[j]@R.T
                force=mass*acc[n,i,j];moment=I@alpha[n,i,j]+np.cross(omega[n,i,j],I@omega[n,i,j])
                forces.append(force);moments.append(moment)
                wrench[:3]+=force;wrench[3:]+=np.cross(row["com"][j],force)+moment
            for j in range(3):
                extra[i,j]=sum(row["axes"][j]@(np.cross(row["com"][k]-row["pivots"][j],forces[k])+moments[k]) for k in range(j,3))
        dynamics.append((wrench,extra))
    speed_results=[]
    for speed in speeds:
        duration=stride/(speed*DUTY[gait]);factor=1/duration**2;worst=0.;valid=True;worst_phase=None;torque_history=[]
        for n,(rows,com,support,(wrench,extra)) in enumerate(zip(all_rows,all_com,supports,dynamics)):
            weight=np.array([0.,0.,robot.mass*9.81]);required=np.r_[weight,np.cross(com,weight)]+factor*wrench
            qd=velocity[n]*speed/.2
            motor=robot.armature*joint_acceleration[n]*factor+robot.dynamic_friction*np.sign(qd)+robot.viscous_friction*qd
            moving_rows=[{**row,"gravity":row["gravity"]+factor*extra[i]+motor[i]} for i,row in enumerate(rows)]
            load=robot.load(moving_rows,com,support,required_wrench=required)
            if load is None:valid=False;break
            torque_history.append(load["torques_nm"])
            if load["peak_torque_nm"]>worst:worst=load["peak_torque_nm"];worst_phase=n
        metrics={}
        if valid:
            torque=np.array(torque_history);qd=velocity*speed/.2
            metrics={"worst_motor_rms_torque_nm":float(np.sqrt(np.mean(torque**2,axis=0)).max()),
                     "sum_mean_squared_torque_nm2":float(np.mean(np.sum(torque**2,axis=(1,2)))),
                     "positive_mechanical_work_per_m_j":float(np.mean(np.maximum(torque*qd,0).sum(axis=(1,2)))/speed),
                     "motor_speed_below_full_continuous_torque_region":bool(np.abs(qd).max()<50.265482457*(1-1.6/5.5))}
        speed_results.append({"speed_mps":speed,"force_balance_pass":valid,**metrics,
                  "minimum_required_peak_torque_nm":worst if valid else None,"worst_phase_index":worst_phase,
                  "estimated_max_joint_speed_rad_s":float(np.abs(velocity).max()*speed/.2)})
    return {"workspace_pass":True,"force_balance_pass":True,"gait":gait,"stride_m":stride,"lift_m":lift,
            "worst_peak_torque_nm":peak,"minimum_support_margin_m":minmargin,
            "minimum_nonfoot_clearance_m":minclear,"minimum_stance_pad_gap_m":mingap,
            "max_ik_error_m":maxerror,"estimated_max_joint_speed_at_0p2_mps":float(np.abs(velocity).max()),
            "phase_count":phase_count,"speed_results":speed_results,"samples":samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,default=ROOT/"robot/hexapod_mkii_length_study")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--variants",nargs="*")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.package/"manifest.json").read_text());start=time.time()
    report={"status":"running","method":"quasistatic_friction_constrained_minimax_contact_force_balance",
            "limits":{"continuous_torque_nm":1.6,"preferred_quasistatic_peak_nm":1.2,"friction":.8,
                      "minimum_belly_clearance_m":.06,"minimum_nonfoot_clearance_m":.005},
            "limitations":["No acceleration or impact loads; no battery-energy prediction",
              "Fixed transferred inertias and stretched mock meshes, not production CAD",
              "Finite pose/path samples; no proof of global feasibility or impossibility",
              "Point contacts; actual pad gaps and self-collision need dynamic/CAD validation"],
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"variants":{}}
    for record in manifest["variants"]:
        name=record["variant"]
        if args.variants and name not in args.variants:continue
        raw=(args.package/record["urdf"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record["sha256"]:raise RuntimeError("URDF hash mismatch")
        xml=ET.fromstring(raw);robot=Robot(xml,manifest["link_joint_mapping"],mesh_clouds(xml,args.package))
        poses=scan(robot);best={}
        for gait in ("tripod","ripple","wave"):
            for clearance in (.06,.09,.12,.15):
                choices=[p for p in poses if p["belly_clearance_m"]>=clearance and gait in p["gaits"] and p["gaits"][gait]["minimum_support_margin_m"]>=.01]
                choices.sort(key=lambda p:p["gaits"][gait]["worst_peak_torque_nm"])
                if choices:best[f"{gait}_{round(clearance*1000)}"]=choices[0]
        entry={"urdf_sha256":record["sha256"],"femur_length_m":record["femur_length_m"],"tibia_length_m":record["tibia_length_m"],
               "total_mass_kg":robot.mass,"poses_screened":len(poses),"best_static":best}
        (args.output/(name+"_poses.json")).write_text(json.dumps(poses,indent=2))
        report["variants"][name]=entry
        (args.output/"mechanical_screen.json").write_text(json.dumps(report,indent=2))
        print(name,len(poses),{g:round(best[g+"_90"]["gaits"][g]["worst_peak_torque_nm"],3) if g+"_90" in best else None for g in ("tripod","ripple","wave")},flush=True)
    report.update(status="completed",elapsed_s=time.time()-start)
    (args.output/"mechanical_screen.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
