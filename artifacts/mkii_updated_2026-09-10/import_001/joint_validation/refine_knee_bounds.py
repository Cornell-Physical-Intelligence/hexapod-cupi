import sys,json,itertools
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/tmp/hexapod-urdf-joint-validation-20260910/deps')
import fcl
root=Path('/tmp/hexapod-urdf-build-20260910');model=json.load(open(root/'model.json'));parts=model['parts'];joints=model['joints']
def Tof(p):
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();T[:3,3]=p['xyz'];return T
jt={j['name']:Tof(j)for j in joints};pt={p['id']:Tof(p)for p in parts}
def fk(q):
 w={'body':np.eye(4)};todo=list(joints)
 while todo:
  nxt=[]
  for j in todo:
   if j['parent']not in w:nxt.append(j);continue
   r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*q.get(j['name'],0)).as_matrix();w[j['child']]=w[j['parent']]@jt[j['name']]@r
  todo=nxt
 return w
names={p['id']:p['name']for p in parts}
# Actual mesh names safe retain source tokens.
structural=['first_joint_top','first_joint_bottom_plate','first_joint_spacer','femur_first_stage','tibia_attatchment_plate','tibia.stl','top_enclosure','bottom_enclosure','bottom_plate']
selected=[p for p in parts if any(x in p['mesh']for x in structural)]
# Canonical structure-only pairs include adjacent and nonadjacent links.
geom={};obj={}
for p in selected:
 if p['mesh']not in geom:
  m=trimesh.load(root/'meshes'/p['mesh'],process=True);g=fcl.BVHModel();g.beginModel(len(m.vertices),len(m.faces));g.addSubModel(np.asarray(m.vertices,float),np.asarray(m.faces,np.int32));g.endModel();geom[p['mesh']]=g
 obj[p['id']]=fcl.CollisionObject(geom[p['mesh']])
rows=[]
for kd in np.arange(-20,20.01,1):
 for fd in [-20,0,20]:
  q={j['name']:np.radians(kd if 'tibia' in j['name']else fd)for j in joints if 'coxa'not in j['name']};w=fk(q)
  for p in selected:
   T=w[p['link']]@pt[p['id']];obj[p['id']].setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
  hits=[];near=[]
  for a,b in itertools.combinations(selected,2):
   if a['link']==b['link']:continue
   if 'tibia'not in a['link']and'tibia'not in b['link']:continue
   r=fcl.CollisionResult();n=fcl.collide(obj[a['id']],obj[b['id']],fcl.CollisionRequest(num_max_contacts=1,enable_contact=True),r)
   if n:hits.append({'parts':[a['id'],b['id']],'meshes':[a['mesh'],b['mesh']],'bodies':[a['link'],b['link']]})
   else:
    d=fcl.distance(obj[a['id']],obj[b['id']],fcl.DistanceRequest(),fcl.DistanceResult())
    near.append((float(d),a['id'],b['id']))
  rows.append({'knee_offset_deg':float(kd),'femur_offset_deg':fd,'intersections':hits,'minimum_nonintersecting_separation':min(near)if near else None})
  print(json.dumps({'knee':kd,'femur':fd,'hits':len(hits),'min_separation_mm':min(near)[0]*1000 if near else None}),flush=True)
Path('/tmp/hexapod-urdf-joint-validation-20260910/knee_bound_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
