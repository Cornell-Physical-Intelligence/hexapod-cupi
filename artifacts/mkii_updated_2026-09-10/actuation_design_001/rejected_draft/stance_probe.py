"""Exact mesh-vertex ground extrema and ideal static vertical-force feasibility.
Uses convex-hull VERTICES only to accelerate linear minima; never writes/replaces colliders.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics
from inertia_probe import calculate,MODEL_SHA


def collision_points(stage):
 clouds={};sources=[]
 for p in stage.Traverse():
  if not p.HasAPI(UsdPhysics.CollisionAPI):continue
  a=p
  while a and not a.HasAPI(UsdPhysics.RigidBodyAPI):a=a.GetParent()
  if not a:raise ValueError('Unowned collider')
  P=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
  M=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
  L=np.array(UsdGeom.Xformable(a).ComputeLocalToWorldTransform(0)).T
  T=np.linalg.inv(L)@M;local=P@T[:3,:3].T+T[:3,3]
  clouds.setdefault(a.GetName(),[]).append(local)
  sources.append({'collider':str(p.GetPath()),'owner':a.GetName(),'vertex_count':len(P)})
 # Linear extrema of original mesh vertices are exact at convex hull vertices.
 result={}
 for n,arrays in clouds.items():
  points=np.vstack(arrays);h=ConvexHull(points);result[n]=points[h.vertices]
 return result,sources


def evaluate(model,clouds,q=None):
 r=calculate(model,q);T=r['frames'];world={n:p@T[n][:3,:3].T+T[n][:3,3]for n,p in clouds.items()}
 feet=[]
 for leg in ['lf','lm','lr','rf','rm','rr']:
  n=leg+'_tibia';i=np.argmin(world[n][:,2]);feet.append({'leg':leg,'position':world[n][i].copy(),'local_contact':clouds[n][i].copy()})
 floor=min(f['position'][2]for f in feet);root_height=-floor
 foot_heights=[float(f['position'][2]-floor)for f in feet]
 min_non_tibia=min(float(p[:,2].min()-floor)for n,p in world.items()if not n.endswith('_tibia'))
 mass=sum(x['mass']for x in model['links']);com=sum(x['mass']*r['details'][x['name']]['com']for x in model['links'])/mass
 # All six vertical reaction forces and minimax joint effort. No horizontal/friction advantage assumed.
 Aeq=np.zeros((3,7));Aeq[0,:6]=1.;Aeq[1,:6]=[f['position'][0]for f in feet];Aeq[2,:6]=[f['position'][1]for f in feet]
 beq=np.array([mass*9.81,mass*9.81*com[0],mass*9.81*com[1]])
 J=np.zeros((18,6));_,axes,origins,ancestors=__import__('inertia_probe').frames(model,np.zeros(18)if q is None else q,r['order'])
 for i,f in enumerate(feet):
  for j in ancestors[f['leg']+'_tibia']:
   J[r['order'].index(j),i]=np.cross(axes[j],f['position']-origins[j])[2]
 # Actuator holding torque = gravity-compensation minus external support generalized force.
 A=np.vstack([np.c_[-J,-np.ones(18)],np.c_[J,-np.ones(18)]])
 b=np.r_[-r['hold'],r['hold']]
 result=linprog(np.r_[np.zeros(6),1.],A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=[(1.,None)]*6+[(0.,None)],method='highs')
 out={'q_rad':(np.zeros(18)if q is None else np.asarray(q)).tolist(),'joint_names':r['order'],
      'root_plate_height_touch_m':root_height,'root_height_5mm_above_touch_m':root_height+.005,
      'feet':[{'leg':f['leg'],'body_frame_position':f['position'].tolist(),'tibia_local_vertex':f['local_contact'].tolist()}for f in feet],
      'foot_height_spread_m':max(foot_heights)-min(foot_heights),'non_tibia_collision_clearance_m':min_non_tibia,
      'ideal_vertical_static_lp_success':bool(result.success),'lp_message':result.message}
 if result.success:
  F=result.x[:6];torque=r['hold']-J@F
  out.update(normal_forces_n=F.tolist(),joint_hold_torque_nm=torque.tolist(),minimax_abs_joint_torque_nm=float(result.x[-1]),
             static_force_moment_residual= (Aeq@result.x-beq).tolist())
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('Preserve previous report')
 b=(a.asset/'source/model.json').read_bytes();assert hashlib.sha256(b).hexdigest()==MODEL_SHA;model=json.loads(b)
 stage=Usd.Stage.Open(str(a.asset/'robot.usda'));clouds,sources=collision_points(stage)
 result={'model_sha256':MODEL_SHA,'scope':'CPU geometry plus ideal frictionless vertical static LP. No native contacts, dynamics, stiffness, thermal or collision-free combined-pose certification.',
         'root_fixed_inertia_not_used_as_support_model':True,'collision_sources':sources,
         'neutral_candidate':evaluate(model,clouds)}
 a.output.write_text(json.dumps(result,indent=2)+'\n')
