"""Recover flat-ground sphere-pad heights from genuine post-physics pose traces."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('urdf',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
r=json.loads(args.report.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert r['diagnostic_complete'] and r['errors']==[]
assert sha(args.urdf)==r['contract']['files']['robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf']
assert 'XYZW' in r['trace_coordinate_conventions']['body_link_quat_w']
root=ET.parse(args.urdf).getroot();legs=['lf','lm','lr','rf','rm','rr'];events=[];peak=None;first_failure=None;all_loss_count=0
for record in r['trace_files']:
 path=args.report.parent/record['file'];assert sha(path)==record['sha256']
 with np.load(path,allow_pickle=False) as n:
  v=n['values'];c=list(n['columns']);assert list(v.shape)==record['shape']
  get=lambda key,names: v[..., [c.index(key+'/'+name) for name in names]].astype(np.float64)
  origin=get('terrain_origin_w',['x','y','z'])
  forces=get('foot_force_w',[leg+'_'+a for leg in legs for a in 'xyz']).reshape(*v.shape[:2],6,3)
  gaps=get('hinge_gap_local',[leg+'_'+a for leg in legs for a in 'xyz']).reshape(*v.shape[:2],6,3)
  qerr=np.linalg.norm(gaps,axis=-1).max(-1)
  bottoms=[]
  for leg in legs:
   name=leg+'_tibia';pos=get('body_link_pos_w',[name+'_'+a for a in 'xyz']);q=get('body_link_quat_w',[name+'_'+a for a in 'xyzw'])
   q=q/np.linalg.norm(q,axis=-1,keepdims=True)
   # Local-to-world rotation row Z for native XYZW quaternions.
   x,y,z,w=np.moveaxis(q,-1,0);rz=np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1)
   link=next(x for x in root.findall('link') if x.get('name')==name);bottom=[]
   for collision in link.findall('collision'):
    sphere=collision.find('geometry/sphere')
    if sphere is None:continue
    xyz=np.fromstring(collision.find('origin').get('xyz'),sep=' ')
    bottom.append(pos[...,2]+np.einsum('...i,i->...',rz,xyz)-float(sphere.get('radius'))-origin[...,2])
   bottoms.append(np.min(bottom,axis=0))
  bottoms=np.stack(bottoms,axis=-1);support=(np.linalg.norm(forces,axis=-1)>1).sum(-1)
  def row(i,e):
   return dict(physics_sample=record['first_physics_sample']+int(i),environment=int(e),
     segment=record['segment'],foot_sphere_bottom_m=dict(zip(legs,bottoms[i,e].tolist())),
     foot_force_w=dict(zip(legs,forces[i,e].tolist())),max_pin_gap_m=float(qerr[i,e]),
     body_velocity_w=get('body_link_linear_velocity_w',['body_'+a for a in 'xyz'])[i,e].tolist())
  if record['segment']['phase']!='standing':
   lost=np.argwhere(support==0);all_loss_count+=len(lost)
   for i,e in lost[:40]:events.append(row(i,e))
   failed=np.argwhere(qerr>.0001)
   if first_failure is None and len(failed):
    i,e=failed[0];first_failure=[row(j,e) for j in range(max(0,i-4),min(len(v),i+5))]
  i,e=np.unravel_index(np.argmax(qerr),qerr.shape)
  if peak is None or qerr[i,e]>peak['max_pin_gap_m']:peak=row(i,e)
value=dict(schema='hexapod.foot_geometry_trace_inspection.v1',scope='Post-physics poses and forces; no physical admission',
 report_sha256=sha(args.report),urdf_sha256=sha(args.urdf),zero_support_environment_samples=all_loss_count,
 zero_support_events=events,first_pin_failure_window=first_failure,peak_pin_gap=peak)
with args.out.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
print(json.dumps({'zero_support_environment_samples':all_loss_count,'peak_pin_gap':peak}))
