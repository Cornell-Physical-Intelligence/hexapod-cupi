exec(open('/tmp/hexapod-urdf-joint-validation-20260910/refine_knee_bounds.py').read().split('names=')[0])
from trimesh.proximity import signed_distance
pmap={p['id']:p for p in parts};w=fk({});results=[]
for tib,screw in [(6,243),(6,244),(6,235),(6,9)]:
 a=pmap[tib];b=pmap[screw];ma=trimesh.load(root/'meshes'/a['mesh'],process=True);mb=trimesh.load(root/'meshes'/b['mesh'],process=True);Ta=w[a['link']]@pt[tib];Tb=w[b['link']]@pt[screw];rel=np.linalg.inv(Ta)@Tb;v=mb.vertices@rel[:3,:3].T+rel[:3,3]
 # Mesh signed distances inspect all unique screw vertices in exact tibia coordinates.
 d=signed_distance(ma,v);r={'tibia':tib,'screw':screw,'tibia_watertight':ma.is_watertight,'tibia_winding_consistent':ma.is_winding_consistent,'screw_vertices':len(v),'max_inside_distance_m':float(d.max()),'min_signed_distance_m':float(d.min()),'vertices_inside_gt1um':int((d>1e-6).sum()),'vertices_inside_gt0p1mm':int((d>1e-4).sum()),'deepest_point_tibia_local_m':v[d.argmax()].tolist()};print(json.dumps(r),flush=True);results.append(r)
Path('/tmp/hexapod-urdf-joint-validation-20260910/neutral_screw_penetration.json').write_text(json.dumps(results,indent=2)+'\n')
