"""Derive contact mask and exact plane-extrema accelerators from immutable USD meshes.
Convex-hull vertex selection preserves a linear plane minimum; never changes colliders.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics

CAP_X=.115

def prepare(asset,output):
 stage=Usd.Stage.Open(str(asset/'robot.usda'));clouds={};non_toe={};shapes=[];bounds=None
 for p in stage.Traverse():
  if not p.HasAPI(UsdPhysics.CollisionAPI):continue
  body=p
  while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
  if not body:raise ValueError('Unowned collision mesh')
  points=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=np.float64)
  T=np.linalg.inv(np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T)@np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
  local=points@T[:3,:3].T+T[:3,3];name=body.GetName();clouds.setdefault(name,[]).append(local)
  # The exact shared tibia mesh is the only cap-eligible collider on each tibia.
  refs=p.GetMetadata('references');is_tibia=name.endswith('_tibia') and len(points)==11523
  source={'path':str(p.GetPath()),'body':name,'vertices':len(points),'point_sha256':hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest(),
          'contact_offset_m':float(p.GetAttribute('physxCollision:contactOffset').Get()),'rest_offset_m':float(p.GetAttribute('physxCollision:restOffset').Get())}
  if abs(source['contact_offset_m']-.001)>1e-9 or source['rest_offset_m']!=0:raise ValueError('Changed authored collision skin')
  if is_tibia:
   # Bind exact mesh array, not solely vertex count or a link-name guess.
   if source['point_sha256']!='b35cef34ab924764378bbdb5b3e29756d29df70182c8b3dec2e226bfc4d972de':raise ValueError('Wrong distal source mesh')
   cap=points[points[:,0]>=CAP_X]
   b=np.stack([cap.min(0),cap.max(0)])
   if bounds is not None and not np.array_equal(bounds,b):raise ValueError('Inconsistent shared toe mesh')
   bounds=b;source.update(shape_to_link=T.tolist(),cap_lower_x_m=CAP_X,cap_bounds_m=b.tolist())
   shapes.append(source)
   face_counts=np.asarray(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get());indices=np.asarray(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get())
   if not np.all(face_counts==3):raise ValueError('Expected exact triangular tibia mesh')
   triangles=indices.reshape(-1,3);edges=np.vstack([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]);edges=np.unique(np.sort(edges,axis=1),axis=0)
   a,b=points[edges[:,0]],points[edges[:,1]];cross=(a[:,0]<CAP_X)&(b[:,0]>CAP_X)|(b[:,0]<CAP_X)&(a[:,0]>CAP_X)
   a,b=a[cross],b[cross];cut=a+(b-a)*((CAP_X-a[:,0])/(b[:,0]-a[:,0]))[:,None]
   non=np.vstack([points[points[:,0]<=CAP_X],cut]);non_toe.setdefault(name,[]).append(non@T[:3,:3].T+T[:3,3])
   source['non_toe_plane_intersections']=len(cut);source['non_toe_extrema_scope']='Exact clipped original triangle surface X<=0.115, including all crossing-edge intersections.'
  else:non_toe.setdefault(name,[]).append(local)
 arrays={}
 for kind,groups in [('all',clouds),('non_toe',non_toe)]:
  for name,parts in groups.items():
   p=np.vstack(parts);arrays[kind+'__'+name]=p[ConvexHull(p).vertices]
 if len(clouds)!=19 or len(shapes)!=6:raise ValueError('Wrong body/cap inventory')
 output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output/'geometry_extrema.npz',**arrays)
 result={'schema':'canonical_mesh_contact_classification_v1','asset_robot_sha256':hashlib.sha256((asset/'robot.usda').read_bytes()).hexdigest(),
  'scope':'Authored diagnostic classification, not measured hardware foot or replacement collision geometry.',
  'cap_rule':'Native floor-contact point in exact source shape coordinates: X>=0.115; upper X and both Y/Z bounds from cap mesh vertices plus authored contactOffset.',
  'clearance_min_m':-.001,'plate_min_m':.055,'floor_z_m':0.,'neutral_plate_touch_m':.07661109101311193,'reset_plate_m':.08161109101311194,
  'shapes':shapes,'body_names':sorted(clouds),'extrema_sha256':hashlib.sha256((output/'geometry_extrema.npz').read_bytes()).hexdigest()}
 (output/'geometry.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.asset,a.output)
