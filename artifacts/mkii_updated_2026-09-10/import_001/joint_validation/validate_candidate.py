#!/usr/bin/env python3
"""Independent CAD-pose roundtrip and finite exact-mesh collision audit."""
from __future__ import annotations
import argparse,collections,hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import fcl

def out(value):print(json.dumps(value),flush=True)
def to_T(xyz,quat):
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(quat).as_matrix();T[:3,3]=xyz;return T
def jdump(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.floating):return float(x)
 if isinstance(x,np.integer):return int(x)
 raise TypeError(type(x))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--skip-collision',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 model=json.loads((args.model_dir/'model.json').read_text());cad=json.loads((args.model_dir/'cad_pose.json').read_text());parts=model['parts'];part_by_id={p['id']:p for p in parts};joints=model['joints'];jnames=[j['name']for j in joints];parents={j['child']:j for j in joints};linknames={j['parent']for j in joints}|set(parents);roots=linknames-set(parents)
 assert len(joints)==18 and len(linknames)==19 and roots=={'body'}
 assert len(parents)==18 and len(jnames)==len(set(jnames))
 assert len(parts)==1753 and {p['id']for p in parts}==set(range(1753))
 assert set(cad)==set(jnames)
 for j in joints:assert math.isfinite(j['lower']) and math.isfinite(j['upper']) and j['lower']<0<j['upper']
 jT={j['name']:to_T(j['xyz'],j['quaternion_xyzw'])for j in joints};pT={p['id']:to_T(p['xyz'],p['quaternion_xyzw'])for p in parts}
 def fk(q):
  T={'body':np.eye(4)};todo=list(joints)
  for _ in range(19):
   if not todo:break
   nxt=[]
   for j in todo:
    if j['parent']not in T:nxt.append(j);continue
    rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(np.asarray(j['axis'])*q.get(j['name'],0.)).as_matrix();T[j['child']]=T[j['parent']]@jT[j['name']]@rot
   todo=nxt
  assert not todo and len(T)==19
  return T
 source_root=ET.parse(args.source/'robot.urdf').getroot();source=source_root.findall('link/visual');assert len(source)==1753;meshes={};sourceT={};sourcefile={}
 for i,v in enumerate(source):
  f=v.find('geometry/mesh').get('filename').split('/')[-1];o=v.find('origin');T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');sourceT[i]=T;sourcefile[i]=f
  if f not in meshes:meshes[f]=trimesh.load(args.source/'assets'/f,process=True)
 # Check renamed output mesh files are byte-identical and correctly mapped.
 mesh_checks={}
 for p in parts:
  f=sourcefile[p['id']];output_file=args.model_dir/'meshes'/p['mesh']
  if f not in mesh_checks:
   sha_source=hashlib.sha256((args.source/'assets'/f).read_bytes()).hexdigest();sha_output=hashlib.sha256(output_file.read_bytes()).hexdigest();assert sha_source==sha_output;mesh_checks[f]={'output_mesh':p['mesh'],'sha256':sha_source}
  else:assert mesh_checks[f]['output_mesh']==p['mesh']
 Tcad=fk(cad);errors=[];uniquevertexevaluations=0
 for p in parts:
  i=p['id'];target=Tcad[p['link']]@pT[i];delta=target-sourceT[i];vertices=meshes[sourcefile[i]].vertices;err=np.linalg.norm(vertices@delta[:3,:3].T+delta[:3,3],axis=1);uniquevertexevaluations+=len(vertices)
  errors.append({'source_index':i,'source_mesh':sourcefile[i],'body':p['link'],'max_vertex_error_m':float(err.max()),'rms_vertex_error_m':float(np.sqrt(np.mean(err**2))),'origin_error_m':float(np.linalg.norm(delta[:3,3]))})
 report={'schema':1,'model_sha256':hashlib.sha256((args.model_dir/'model.json').read_bytes()).hexdigest(),'source_urdf_sha256':hashlib.sha256((args.source/'robot.urdf').read_bytes()).hexdigest(),'mesh_copy_checks':{'unique_meshes':len(mesh_checks),'all_byte_identical':True},'graph':{'links':19,'moving_joints':18,'root':'body','connected_acyclic':True,'parts_counted_exactly_once':1753},'roundtrip':{'coordinate_contract':'candidate cad_pose.json reproduces original supplied pose; comparison directly uses source XML/STL, independent of builder pickle transforms','unique_vertices_evaluated_over_all_instances':uniquevertexevaluations,'max_vertex_error_m':max(e['max_vertex_error_m']for e in errors),'parts_above_10um':sum(e['max_vertex_error_m']>1e-5 for e in errors),'largest_20':sorted(errors,key=lambda e:e['max_vertex_error_m'],reverse=True)[:20]}}
 (args.out/'part_roundtrip_errors.json').write_text(json.dumps(errors,indent=2)+'\n');out({'roundtrip':report['roundtrip']})
 # Independently parse the written URDF inertial blocks.
 urdf_path=args.model_dir/'urdf'/'hexapod_updated_inspection.urdf';u=ET.parse(urdf_path).getroot();inertials=[]
 assert len(u.findall('link'))==19 and len(u.findall('joint'))==18
 for l in u.findall('link'):
  inertia=l.find('inertial');m=float(inertia.find('mass').get('value'));a=inertia.find('inertia').attrib;I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]]);e=np.linalg.eigvalsh(I);assert m>0 and e.min()>0 and e[2]<=e[0]+e[1]+1e-12;inertials.append({'body':l.get('name'),'mass_kg':m,'principal_inertia_kg_m2':e})
 # Parse the emitted XML independently, including serialized RPY near gimbal lock.
 def xmlT(origin):
  T=np.eye(4)
  if origin is not None:T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
  return T
 xmlworld={'body':np.eye(4)};todo=list(u.findall('joint'))
 while todo:
  nxt=[]
  for j in todo:
   parent=j.find('parent').get('link');child=j.find('child').get('link')
   if parent not in xmlworld:nxt.append(j);continue
   rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*cad[j.get('name')]).as_matrix();xmlworld[child]=xmlworld[parent]@xmlT(j.find('origin'))@rot
  assert len(nxt)<len(todo);todo=nxt
 xmlerr=[];xmlpartids=[];xmlvsmodel=[]
 for link in u.findall('link'):
  for vis in link.findall('visual'):
   i=int(vis.get('name').split('_')[-1]);assert Path(vis.find('geometry/mesh').get('filename')).name==part_by_id[i]['mesh'];xmlpartids.append(i);target=xmlworld[link.get('name')]@xmlT(vis.find('origin'));delta=target-sourceT[i];v=meshes[sourcefile[i]].vertices;err=np.linalg.norm(v@delta[:3,:3].T+delta[:3,3],axis=1);xmlerr.append(float(err.max()));p=part_by_id[i];modeltarget=Tcad[p['link']]@pT[i];xmlvsmodel.append(float(np.max(np.abs(target-modeltarget))))
 assert sorted(xmlpartids)==list(range(1753))
 report['xml_roundtrip']={'urdf_sha256':hashlib.sha256(urdf_path.read_bytes()).hexdigest(),'serialized_visuals':len(xmlpartids),'max_original_vertex_error_m':max(xmlerr),'max_transform_entry_difference_from_model':max(xmlvsmodel),'parts_above_10um':sum(e>1e-5 for e in xmlerr),'serialized_RPY_independently_verified':max(xmlvsmodel)<1e-10,'roundtrip_within_source_precision':max(xmlerr)<1e-5}
 out({'xml_roundtrip':report['xml_roundtrip']})
 report['inertial_checks']={'positive_mass_spd_triangle_inequality':True,'mass_kg':sum(i['mass_kg']for i in inertials),'bodies':inertials}
 # Finite ranges are visual/inspection envelopes, never hard stops.
 poses=[('neutral',{})]
 for j in joints:
  for side in ['lower','upper']:poses.append((j['name']+'_'+side,{j['name']:j[side]}))
 legs=['lf','lm','lr','rf','rm','rr'];by_leg={leg:[j for j in joints if j['name'].startswith(leg+'_')]for leg in legs}
 for leg in legs:
  for bits in itertools.product([0,1],repeat=3):poses.append((leg+'_corners_'+''.join(map(str,bits)),{j['name']:j['upper' if b else 'lower']for j,b in zip(by_leg[leg],bits)}))
 for bits in itertools.product([0,1],repeat=3):
  for alternating in [False,True]:
   q={}
   for li,leg in enumerate(legs):
    for j,b in zip(by_leg[leg],bits):q[j['name']]=j['upper' if (b^int(alternating and li%2))else'lower']
   poses.append(('all_corners_'+''.join(map(str,bits))+('_alternating' if alternating else ''),q))
 lower=np.array([j['lower']for j in joints]);upper=np.array([j['upper']for j in joints]);samples=qmc.Sobol(d=18,scramble=True,seed=20260910).random_base2(5)
 for k,s in enumerate(samples):poses.append(('sobol_%02d'%k,dict(zip(jnames,lower+s*(upper-lower)))))
 # Structural/housing mesh minimum heights at fixed deck height; ground below
 # baseline sole support is a kinematic diagnostic, not a free-standing simulation.
 neutral=fk({});sole_zero=[]
 for p in parts:
  if sourcefile[p['id']]=='tibia.stl':
   T=neutral[p['link']]@pT[p['id']];z=(meshes['tibia.stl'].vertices@T[:3,:3].T+T[:3,3])[:,2];sole_zero.append({'body':p['link'],'minimum_z_m':float(z.min())})
 floor=min(s['minimum_z_m']for s in sole_zero);ground=[]
 support_vertices={f:m.convex_hull.vertices for f,m in meshes.items()}
 for label,q in poses:
  wt=fk(q);feet=[];other_min=(float('inf'),None)
  for p in parts:
   i=p['id'];T=wt[p['link']]@pT[i];v=support_vertices[sourcefile[i]];minimum=float((v@T[2,:3]+T[2,3]).min()-floor)
   if sourcefile[i]=='tibia.stl':feet.append({'body':p['link'],'sole_clearance_m':minimum})
   elif minimum<other_min[0]:other_min=(minimum,i)
  ground.append({'pose':label,'min_sole_clearance_m':min(f['sole_clearance_m']for f in feet),'max_sole_clearance_m':max(f['sole_clearance_m']for f in feet),'min_non_tibia_part_clearance_m':other_min[0],'min_non_tibia_part_index':other_min[1],'feet':feet})
 report['finite_envelope']={'pose_count':len(poses),'samples':'neutral; each of 18 joints individually at two endpoints; each leg eight coupled corners; 16 synchronized/alternating full-robot corner poses;32 fixed-seed18D Sobol samples','joint_bounds':{j['name']:[j['lower'],j['upper']]for j in joints},'limits_are_hardware_stops':False,'continuous_motion_checked':False,'full_cartesian_limit_box_certified':False}
 report['ground']={'fixed_base_floor_z_m':floor,'body_z_translation_for_neutral_floor_contact_m':-floor,'neutral_feet':sole_zero,'neutral_foot_height_spread_m':max(s['minimum_z_m']for s in sole_zero)-floor,'minimum_non_tibia_part_clearance_any_pose_m':min(g['min_non_tibia_part_clearance_m']for g in ground),'minimum_sole_clearance_any_pose_m':min(g['min_sole_clearance_m']for g in ground),'below_floor_pose_count':sum(g['min_sole_clearance_m']< -1e-5 for g in ground),'interpretation':'Joint sweeps move feet below the neutral fixed-base support plane; they require lifting the deck or coordinating contacts. This is not a support, actuator, stability or simulation qualification.'}
 (args.out/'finite_pose_ground_checks.json').write_text(json.dumps(ground,indent=2)+'\n');out({'ground':report['ground'],'pose_count':len(poses)})
 if not args.skip_collision:
  # Actual original triangle BVHs: one shared geometry per mesh, one object per
  # instance, broadphase separated by physical body. No convex hulls or spheres.
  geometry={}
  for f,m in meshes.items():
   g=fcl.BVHModel();g.beginModel(len(m.vertices),len(m.faces));g.addSubModel(np.asarray(m.vertices,dtype=np.float64),np.asarray(m.faces,dtype=np.int32));g.endModel();geometry[f]=g
  parts_by_body=collections.defaultdict(list)
  for p in parts:parts_by_body[p['link']].append(p)
  objects={p['id']:fcl.CollisionObject(geometry[sourcefile[p['id']]])for p in parts};managers={}
  for link,pp in parts_by_body.items():
   manager=fcl.DynamicAABBTreeCollisionManager();manager.registerObjects([objects[p['id']]for p in pp]);manager.setup();managers[link]=manager
  geom_to_file={id(g):f for f,g in geometry.items()};byfileid=collections.defaultdict(list)
  for p in parts:byfileid[sourcefile[p['id']]].append(p['id'])
  # Contact records identify shared mesh plus exact current placement. Rounding
  # only indexes object identities; collision queries use full double transforms.
  tf_key=lambda T:tuple(np.round(T[:3,:].reshape(-1),11))
  static_external={'top_enclosure.stl','bottom_enclosure.stl','bottom_plate.stl','standoff_plate.stl','standoff_plate__2.stl','first_joint_top.stl','first_joint_bottom_plate.stl','first_joint_spacer.stl','femur_first_stage.stl','tibia_attatchment_plate.stl','tibia.stl'}
  collision_rows=[];baseline_pairs=None
  for count,(label,q)in enumerate(poses):
   wt=fk(q);lookup={};seen={};queries=[0]
   for p in parts:
    T=wt[p['link']]@pT[p['id']];objects[p['id']].setTransform(fcl.Transform(T[:3,:3],T[:3,3]));lookup[(sourcefile[p['id']],tf_key(T))]=p['id']
   for manager in managers.values():manager.update()
   def cb(a,b,data):
    queries[0]+=1;result=fcl.CollisionResult();request=fcl.CollisionRequest(num_max_contacts=4,enable_contact=True);n=fcl.collide(a,b,request,result)
    if n:
     c=result.contacts[0];fa=geom_to_file[id(c.o1)];fb=geom_to_file[id(c.o2)];Ta=np.eye(4);Ta[:3,:3]=a.getRotation();Ta[:3,3]=a.getTranslation();Tb=np.eye(4);Tb[:3,:3]=b.getRotation();Tb[:3,3]=b.getTranslation()
     # FCL contact geometry order can differ from callback object order.
     ka=(fa,tf_key(Ta));kb=(fb,tf_key(Tb))
     if ka not in lookup or kb not in lookup:ka=(fa,tf_key(Tb));kb=(fb,tf_key(Ta))
     ia=lookup.get(ka);ib=lookup.get(kb)
     if ia is None or ib is None:raise RuntimeError('cannot resolve collision instance IDs')
     key=tuple(sorted([ia,ib]));seen[key]={'part_indices':key,'meshes':[sourcefile[key[0]],sourcefile[key[1]]],'bodies':[part_by_id[key[0]]['link'],part_by_id[key[1]]['link']],'position_m':np.asarray(c.pos),'both_primary_structure':fa in static_external and fb in static_external,'has_primary_structure':fa in static_external or fb in static_external,'triangle_contact_count_capped_at_4':n}
    return False
   for la,lb in itertools.combinations(sorted(managers),2):managers[la].collide(managers[lb],None,cb)
   if baseline_pairs is None:baseline_pairs=set(seen)
   for key,row in seen.items():row['present_at_neutral']=key in baseline_pairs
   collision_rows.append({'pose':label,'cross_body_broadphase_candidate_queries':queries[0],'intersecting_part_pairs':len(seen),'new_pairs_vs_neutral':len(set(seen)-baseline_pairs),'primary_structure_pairs':sum(r['both_primary_structure']for r in seen.values()),'new_primary_structure_pairs':sum(r['both_primary_structure']and key not in baseline_pairs for key,r in seen.items()),'pairs':list(seen.values())})
   if count%10==0:out({'collision_pose':count,'name':label,'pairs':len(seen),'new':len(set(seen)-baseline_pairs),'primary_structure_pairs':collision_rows[-1]['primary_structure_pairs']})
  (args.out/'exact_mesh_collision_samples.json').write_text(json.dumps(collision_rows,indent=2,default=jdump)+'\n')
  report['collision']={'backend':'python-fcl0.7.0.11','geometry':'actual original STL triangle meshes; 59 shared BVHs and 1753 positioned objects, cross-body queries only','poses':len(poses),'triangle_surface_queries_only':True,'containment_without_surface_crossing_checked':False,'neutral_intersecting_pairs':collision_rows[0]['intersecting_part_pairs'],'neutral_primary_structure_pairs':collision_rows[0]['primary_structure_pairs'],'max_new_primary_structure_pairs':max(r['new_primary_structure_pairs']for r in collision_rows),'poses_with_new_primary_structure_pairs':[r['pose']for r in collision_rows if r['new_primary_structure_pairs']],'max_new_part_pairs':max(r['new_pairs_vs_neutral']for r in collision_rows),'notes':'Exact tessellation can intersect at intended mating/contact interfaces. Neutral contacts preserved; new contact pairs flagged rather than deleted. Finite pose sampling does not certify swept-volume clearance or all joint combinations.'}
 geometry_contract={'parts':[{k:p[k]for k in ['id','link','mesh','xyz','quaternion_xyzw']}for p in parts],'joints':[{k:j[k]for k in ['name','parent','child','xyz','quaternion_xyzw','axis','lower','upper']}for j in joints]}
 report['geometry_and_bounds_sha256']=hashlib.sha256(json.dumps(geometry_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 report['qualified_for_sim2real']=False
 (args.out/'validation_report.json').write_text(json.dumps(report,indent=2,default=jdump)+'\n');out({'done':str(args.out/'validation_report.json')})
if __name__=='__main__':main()
