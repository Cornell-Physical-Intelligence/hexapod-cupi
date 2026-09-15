"""Named provisional servo and exact authored-mesh contact/clearance interpretation."""
import numpy as np
from inspect_core import rotation
RPM=np.array([0.,70.,275.,340.,450.,477.,480.]);NM=np.array([5.5,5.5,4.,3.,1.6,.5,0.])
LEGS=['lf','lm','lr','rf','rm','rr']

def servo(q,dq,target,kp,kd):
 q,dq,target=[np.asarray(x,dtype=np.float32)for x in(q,dq,target)]
 if q.ndim!=2 or q.shape[1]!=18 or dq.shape!=q.shape or target.shape!=q.shape:raise ValueError('Wrong named batched servo shape')
 if any(not np.isfinite(x).all()for x in(q,dq,target,kp,kd)):raise ValueError('Nonfinite servo input')
 raw=kp*(target-q)-kd*dq
 ceiling=np.minimum(1.6,np.interp(abs(dq.astype(float))*60/(2*np.pi),RPM,NM,right=0.)).astype(np.float32)
 # Float32 representation of 1.6 is accepted only to the explicit1.60001 numeric gate.
 return raw.astype(np.float32),np.clip(raw,-ceiling,ceiling).astype(np.float32),ceiling

class Geometry:
 def __init__(self,meta,clouds,body_names):
  self.meta=meta;self.clouds={k:np.asarray(v)for k,v in clouds.items()};self.names=body_names
  if set(body_names)!=set(meta['body_names']):raise ValueError('Geometry body mapping differs')
  self.shapes={x['body']:x for x in meta['shapes']}
 def cap(self,body,point_world,pose):
  s=self.shapes[body];T=np.asarray(s['shape_to_link']);R=rotation(np.asarray(pose)[3:])
  local=(np.asarray(point_world)-pose[:3])@R
  point=(local-T[:3,3])@T[:3,:3]
  b=np.asarray(s['cap_bounds_m']);skin=s['contact_offset_m']
  ok=point[0]>=s['cap_lower_x_m']-1e-12 and point[0]<=b[1,0]+skin and np.all(point[1:]>=b[0,1:]-skin)and np.all(point[1:]<=b[1,1:]+skin)
  return bool(ok),point
 def clearance(self,poses):
  n=len(poses);allmin=np.full(n,np.inf);nonmin=np.full(n,np.inf)
  for e in range(n):
   for j,name in enumerate(self.names):
    p=poses[e,j];z=rotation(p[3:])[2]
    allmin[e]=min(allmin[e],float((self.clouds['all__'+name]@z+p[2]).min()))
    nonmin[e]=min(nonmin[e],float((self.clouds['non_toe__'+name]@z+p[2]).min()))
  return allmin,nonmin

def classify_contacts(data,sensor_map,poses,geometry,num_envs):
 """Classify each actual native patch before force aggregation. No centroid proxy."""
 force,point,normal,sep,counts,starts=[np.asarray(x)for x in data]
 if counts.shape!=(19*num_envs,1)or starts.shape!=counts.shape:raise ValueError('Incomplete sensor/filter contact layout')
 if np.any(counts<0)or np.any(starts<0):raise ValueError('Negative contact indexing')
 feet=np.zeros((num_envs,6,3));other=np.zeros((num_envs,4,3));patches=[];used=set();cap_cache={}
 for i,(e,body)in enumerate(sensor_map):
  start,count=int(starts[i,0]),int(counts[i,0])
  if count==0:continue
  if start+count>len(force):raise ValueError('Contact buffer overflow/range')
  for k in range(start,start+count):
   if k in used:raise ValueError('Overlapping contact-buffer ranges')
   used.add(k);f=float(force[k,0]);p=point[k];n=normal[k];d=float(sep[k,0])
   if not np.isfinite(np.r_[f,p,n,d]).all():raise ValueError('Nonfinite used contact patch')
   inactive_zero_normal=bool(f==0. and np.all(n==0.) and d==0.)
   if abs(np.linalg.norm(n)-1)>1e-3 and not inactive_zero_normal:raise ValueError('Invalid patch normal')
   local=None
   if body.endswith('_tibia'):
    key=(e,body)
    if key not in cap_cache:
     s=geometry.shapes[body];pose=poses[e,geometry.names.index(body)]
     cap_cache[key]=(pose,np.asarray(s['shape_to_link']),rotation(np.asarray(pose)[3:]),np.asarray(s['cap_bounds_m']),s['contact_offset_m'],s['cap_lower_x_m'])
    pose,T,R,b,skin,lower=cap_cache[key]
    local=(np.asarray(p)-pose[:3])@R
    local=(local-T[:3,3])@T[:3,:3]
    cap=bool(local[0]>=lower-1e-12 and local[0]<=b[1,0]+skin and np.all(local[1:]>=b[0,1:]-skin)and np.all(local[1:]<=b[1,1:]+skin))
    category='toe'if cap else'shaft'
   else:category='coxa'if body.endswith('_coxa')else'femur'if body.endswith('_femur')else'body'
   if category=='toe':feet[e,LEGS.index(body[:2])]+=f*n
   else:other[e,['body','coxa','femur','shaft'].index(category)]+=f*n
   patches.append({'buffer_index':k,'env':e,'body':body,'category':category,'normal_force_n':f,'point_world_m':p.tolist(),'normal_world':n.tolist(),'separation_m':d,'inactive_zero_normal':inactive_zero_normal,'shape_point_m':None if local is None else local.tolist()})
 if len(used)>=len(force):raise ValueError('Contact capacity exhausted; completeness is unknown')
 # Any nonfoot patch with >1N is retained even when vector cancellation would hide it.
 nonfoot=np.zeros(num_envs,dtype=bool);body_forces={}
 for p in patches:
  if p['category']!='toe':
   if abs(p['normal_force_n'])>1.:nonfoot[p['env']]=True
   key=(p['env'],p['body'])
   if key not in body_forces:body_forces[key]=np.zeros(3)
   body_forces[key]=body_forces[key]+p['normal_force_n']*np.asarray(p['normal_world'])
 for i,(e,body)in enumerate(sensor_map):
  body_force=body_forces.get((e,body))
  if body_force is not None and np.linalg.norm(body_force)>1.:nonfoot[e]=True
 return {'distal_contact':np.linalg.norm(feet,axis=-1)>1.,'distal_force_world':feet,'nonfoot_contact':nonfoot,'nonfoot_force_world':other,'patches':patches,'used_patch_count':len(used)}
