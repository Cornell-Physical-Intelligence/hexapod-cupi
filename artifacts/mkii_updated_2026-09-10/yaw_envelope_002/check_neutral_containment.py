#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,trimesh
from derive_yaw_envelope import matrix,rz
root=Path('/tmp/hexapod-urdf-build-20260910');out=Path('/tmp/hexapod-urdf-yaw-envelope-20260910');model=json.load(open(root/'model.json'));limits=json.load(open(out/'yaw_published_joint_bounds.json'));pmap={p['id']:p for p in model['parts']};jmap={j['name']:j for j in model['joints']};plates=[p for p in pmap.values()if p['link']=='body'and'standoff_plate'in p['mesh']];coxae=[p for p in pmap.values()if p['link'].endswith('_coxa')];mesh={};representative={};component_counts={}
for p in plates+coxae:
 f=p['mesh']
 if f not in mesh:
  m=trimesh.load(root/'meshes'/f,process=True);mesh[f]=m;groups=trimesh.graph.connected_components(m.face_adjacency,nodes=np.arange(len(m.faces)));representative[f]=np.array([m.vertices[m.faces[ids[0],0]]for ids in groups]);component_counts[f]=len(groups)
report={'schema':1,'pose':'each coxa at published free-interval midpoint; no femur/tibia groups','method':'One actual surface vertex per face-connected component, transformed into each opposite solid frame. Test both coxa-in-plate and plate-in-coxa. AABB exclusion first; ray containment only when representative point lies in bounding box. Target meshes must be watertight and consistently oriented.','all_target_meshes_watertight':all(m.is_watertight for m in mesh.values()),'all_target_meshes_consistent_winding':all(m.is_winding_consistent for m in mesh.values()),'non_watertight_meshes':[f for f,m in mesh.items()if not m.is_watertight],'mesh_component_counts':component_counts,'contained_representatives':[],'unresolved_nonwatertight':[],'representative_point_solid_pairs_tested':0,'aabb_excluded':0,'ray_queries':0}
T={}
for p in plates:T[p['id']]=matrix(p)
for p in coxae:T[p['id']]=matrix(jmap[p['link']+'_yaw'])@rz(limits['joints'][p['link']+'_yaw']['zero_shift_rad'])@matrix(p)
def check(source,target):
 a,b=source['id'],target['id'];rel=np.linalg.inv(T[b])@T[a];points=representative[source['mesh']]@rel[:3,:3].T+rel[:3,3];m=mesh[target['mesh']];insidebox=np.all(points>=m.bounds[0]-1e-10,axis=1)&np.all(points<=m.bounds[1]+1e-10,axis=1);report['representative_point_solid_pairs_tested']+=len(points);report['aabb_excluded']+=int((~insidebox).sum())
 if insidebox.any():
  if not m.is_watertight or not m.is_winding_consistent:report['unresolved_nonwatertight'].append({'source_part':a,'target_part':b});return
  contained=m.contains(points[insidebox]);report['ray_queries']+=int(insidebox.sum())
  if contained.any():report['contained_representatives'].append({'source_part':a,'target_part':b,'source_link':source['link'],'target_link':target['link'],'source_component_indices':np.flatnonzero(insidebox)[contained].tolist()})
for c in coxae:
 for p in plates:check(c,p);check(p,c)
report['all_closed_solid_qualification']=report['all_target_meshes_watertight'] and report['all_target_meshes_consistent_winding'];report['no_component_containment_detected']=not report['contained_representatives']and not report['unresolved_nonwatertight'];report['scope']='With separately proved positive surface separation throughout the connected1Dinterval, no representative containment at this pose excludes persistent nested-solid intersection for checked closed components. Does not validate chassis clearances outside the six standoff solids or coupled leg configurations.'
(out/'neutral_component_containment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='mesh_component_counts'},indent=2))
