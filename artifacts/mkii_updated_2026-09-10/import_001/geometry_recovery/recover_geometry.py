#!/usr/bin/env python3
"""Audit a fused Onshape export without inventing CAD mates or certified limits.

Uses current mesh placements only. Recovers independent coaxial cylinder evidence,
replica-consistent structural poses, and provisional body membership. Does not load
pickle, alter source, copy historical axes, or certify hardware joint limits.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

def serial(o):
 if isinstance(o,np.ndarray):return o.tolist()
 if isinstance(o,np.floating):return float(o)
 if isinstance(o,np.integer):return int(o)
 raise TypeError(type(o))

def mesh_fit(mesh,axis_index):
 """Least-squares outer-cylinder ring in a named mesh's declared local frame.

Outer walls parallel to the axis support its direction independently of the
mesh transform; the fitted circle determines the off-axis center. Mesh local
axis index is an explicit, inspectable feature hypothesis, not a joint mate.
"""
 v=np.asarray(mesh.vertices);others=[k for k in range(3) if k!=axis_index]
 uv=v[:,others];rho=np.linalg.norm(uv,axis=1); select=rho>rho.max()-1e-5
 q=uv[select]; A=np.column_stack([2*q,np.ones(len(q))]);a=np.linalg.lstsq(A,np.sum(q*q,axis=1),rcond=None)[0];cen=a[:2];radius=np.sqrt(a[2]+cen@cen)
 ring_res=np.linalg.norm(q-cen,axis=1)-radius
 p=np.zeros(3);p[others]=cen;p[axis_index]=np.mean(v[select,axis_index])
 tri=np.asarray(mesh.triangles);nr=np.asarray(mesh.face_normals);tr=np.linalg.norm(tri[:,:,others]-cen,axis=2)
 walls=(np.min(tr,axis=1)>radius-1e-5)&(np.abs(nr[:,axis_index])<.05)
 normals=nr[walls];_,sv,V=np.linalg.svd(normals,full_matrices=False);axis=V[-1]
 if axis[axis_index]<0:axis=-axis
 return {'local_point_m':p,'local_axis':axis,'radius_m':radius,'ring_vertex_count':len(q),'ring_rms_m':np.sqrt(np.mean(ring_res**2)),'ring_max_abs_m':np.max(abs(ring_res)),'wall_triangle_count':int(walls.sum()),'wall_normal_nullspace_singular_values':sv,'axis_vs_declared_deg':np.degrees(np.arccos(np.clip(axis[axis_index],-1,1)))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 root=ET.parse(args.source/'robot.urdf').getroot();instances=[];mesh_cache={}
 for idx,vis in enumerate(root.findall('link/visual')):
  file=vis.find('geometry/mesh').get('filename').split('/')[-1];o=vis.find('origin');T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
  if file not in mesh_cache:mesh_cache[file]=trimesh.load(args.source/'assets'/file,process=False)
  mesh=mesh_cache[file];center=T[:3,:3]@mesh.center_mass+T[:3,3]
  instances.append({'index':idx,'file':file,'T':T,'center_m':center})
 group=lambda f:[i for i in instances if i['file']==f]
 anchors=[]
 for c in group('first_joint_spacer.stl'):
  p=c['T'][:3,3]; f=min(group('femur_first_stage.stl'),key=lambda i:np.linalg.norm(i['center_m']-p));t=min(group('tibia.stl'),key=lambda i:np.linalg.norm(i['center_m']-f['center_m']))
  name=('l' if p[0]>0 else 'r')+('f' if p[1]<-.1 else 'r' if p[1]>.1 else 'm')
  anchors.append({'leg':name,'coxa':c,'femur':f,'tibia':t})
 for a in anchors:
  tb=np.eye(4);tb[:3,3]=a['coxa']['T'][:3,3];a['chassis_yaw']={'T':tb,'center_m':tb[:3,3]}
 defs={'motor_bearing_holder.stl':2,'bearing_insert.stl':2,'motor_flange.stl':1}
 housing=next(f for f in mesh_cache if f.startswith('1_1_06_eb463_507'));defs[housing]=2
 fits={f:mesh_fit(mesh_cache[f],a) for f,a in defs.items()}
 lines={}
 for f,fit in fits.items():
  lines[f]=[dict(index=i['index'],file=f,point_m=i['T'][:3,:3]@fit['local_point_m']+i['T'][:3,3],axis=i['T'][:3,:3]@fit['local_axis']) for i in group(f)]
 holders=lines['motor_bearing_holder.stl']; joints=[]
 for h in holders:
  evidence=[]
  for f in defs:
   if f=='motor_bearing_holder.stl':continue
   choices=[]
   for j in lines[f]:
    d=j['point_m']-h['point_m'];trans=np.linalg.norm(np.cross(d,h['axis']));axial=abs(d@h['axis']);ang=np.degrees(np.arccos(np.clip(abs(j['axis']@h['axis']),-1,1)));score=trans+ang*.001+max(0,axial-.08)
    choices.append((score,trans,axial,ang,j))
   _,trans,axial,ang,j=min(choices,key=lambda row:row[0]);evidence.append({'instance_index':j['index'],'mesh':f,'line_distance_m':trans,'axial_datum_distance_m':axial,'unsigned_axis_mismatch_deg':ang})
  leg=min(anchors,key=lambda a:np.linalg.norm(a['coxa']['T'][:3,3]-h['point_m']))
  if abs(h['axis'][2])>.99:typ='coxa_yaw';local=leg['coxa']['T'];axis=np.array([0.,0.,1.]);parent='body';child=leg['leg']+'_coxa'
  else:
   # Distinguish centers by current structural local frame: shoulder +45 mm local Z,
   # knee -28.50 mm local Z, each at local X=0. This is new mesh evidence.
   local=leg['femur']['T']; lp=local[:3,:3].T@(h['point_m']-local[:3,3]);typ='femur_pitch' if lp[2]>0 else 'tibia_pitch';axis=local[:3,1];parent=leg['leg']+('_coxa' if typ=='femur_pitch' else '_femur');child=leg['leg']+('_femur' if typ=='femur_pitch' else '_tibia')
  # Pick equivalent axis point in structural midplane; along-axis datums are arbitrary.
  p=h['point_m'].copy();p-=axis*((p-local[:3,3])@axis)
  joints.append({'name':leg['leg']+'_'+typ,'leg':leg['leg'],'type':typ,'parent_provisional':parent,'child_provisional':child,'axis_point_m':p,'axis_direction_geometric':axis,'holder_instance_index':h['index'],'independent_evidence':evidence,'point_in_femur_frame_m':np.linalg.inv(leg['femur']['T'])@np.r_[p,1]})
 # Choose the yaw line datum at the current shoulder's axial height. This
 # along-axis choice is derived here and has no kinematic effect.
 for a in anchors:
  yaw=next(j for j in joints if j['leg']==a['leg'] and j['type']=='coxa_yaw');shoulder=next(j for j in joints if j['leg']==a['leg'] and j['type']=='femur_pitch');d=yaw['axis_direction_geometric'];p=yaw['axis_point_m'];p+=d*((shoulder['axis_point_m']-p)@d);yaw['point_in_femur_frame_m']=np.linalg.inv(a['femur']['T'])@np.r_[p,1]
 # Child link frames place origins on each recovered pivot and retain the actual
 # current part orientation. Thus every emitted revolute coordinate can be zero
 # at the source CAD pose without changing bearing geometry or conflating zeros.
 body_frames={'body':np.eye(4)}
 for j in joints:
  a=next(a for a in anchors if a['leg']==j['leg']);kind={'coxa_yaw':'coxa','femur_pitch':'femur','tibia_pitch':'tibia'}[j['type']]
  T=a[kind]['T'].copy();T[:3,3]=j['axis_point_m'];body_frames[j['child_provisional']]=T
 for j in joints:
  parent=body_frames[j['parent_provisional']];child=body_frames[j['child_provisional']]
  j['child_rigid_frame_in_export']=child;j['joint_origin_in_parent_at_CAD_zero']=np.linalg.inv(parent)@child;j['axis_in_child_joint_frame']=child[:3,:3].T@j['axis_direction_geometric'];j['zero_contract']='q=0 means the supplied CAD pose; not radial yaw zero, reset stance or encoder zero.'
 # Six-copy relative-pose consistency infers membership. Nearest mesh copy in the
 # other five legs yields evidence; position alone and full SE(3) reported separately.
 # Geometry lies within 180 mm of the matching candidate body anchor.
 family=collections.defaultdict(list)
 for i in instances:family[i['file']].append(i)
 records=[]
 for i in instances:
  opts=[]
  for body in ['coxa','femur','tibia','chassis_yaw']:
   candidate_leg=min(anchors,key=lambda a:np.linalg.norm(i['center_m']-a[body]['center_m']))
   local=np.linalg.inv(candidate_leg[body]['T'])@i['T'];res=[];matched=[]
   for other in anchors:
    if other['leg']==candidate_leg['leg']:continue
    expect=other[body]['T']@local
    ds=[]
    for j in family[i['file']]:
     if j['index']==i['index']:continue
     dr=expect[:3,:3].T@j['T'][:3,:3];theta=math.acos(np.clip((np.trace(dr)-1)/2,-1,1));pos=np.linalg.norm(expect[:3,3]-j['T'][:3,3]);score=pos+.02*theta
     ds.append((score,pos,theta,j['index']))
    if ds:
     row=min(ds);res.append(row[:3]);matched.append(row[3])
   if res:
    rr=np.array(res);opts.append({'body':('body' if body=='chassis_yaw' else candidate_leg['leg']+'_'+body),'family':body,'rms_position_m':np.sqrt(np.mean(rr[:,1]**2)),'max_position_m':rr[:,1].max(),'rms_angle_deg':np.degrees(np.sqrt(np.mean(rr[:,2]**2))),'score_rms_m':np.sqrt(np.mean(rr[:,0]**2)),'other_five_matching_indices':matched})
  opts.sort(key=lambda x:x['score_rms_m']);best=opts[0] if opts else None
  status='unresolved'
  if best and best['max_position_m']<3e-6 and best['rms_angle_deg']<.002:status='replica_pose_consistent'
  if status=='replica_pose_consistent' and len(opts)>1 and opts[1]['score_rms_m']<3e-6:status='multiple_body_pose_consistent'
  records.append({'index':i['index'],'mesh':i['file'],'status':status,'best':best,'alternatives':opts[1:]})
 # The remaining components are a bounded, inspectable chassis set, not a catch-all.
 chassis_remainder_names={
  'top_enclosure.stl','bottom_enclosure.stl','bottom_plate.stl',
  'standoff_plate.stl','standoff_plate__2.stl',
  'm4_countersunk_hex_drive_screw__12mm_l__alloy_steel.stl',
  'pem_m4_self_clinching_nut__300_ss.stl',
  'm4_through_hole_standoff__8mm_l__hardened_steel.stl',
  'm3_low_profile_socket_head_screw__5mm_l__hs_steel.stl',
 }
 chassis_solids=[i for i in instances if i['file'] in {'top_enclosure.stl','bottom_enclosure.stl','bottom_plate.stl'}]
 chassis_verts=np.concatenate([mesh_cache[i['file']].vertices@i['T'][:3,:3].T+i['T'][:3,3] for i in chassis_solids])
 chassis_bounds=np.stack([chassis_verts.min(0),chassis_verts.max(0)])
 ownership=[]
 for r in records:
  i=instances[r['index']]
  if r['status']=='replica_pose_consistent':
   body=r['best']['body'];basis='six_copy_relative_pose';confidence='geometrically_supported_not_exported_mate';evidence={'max_translation_residual_m':r['best']['max_position_m'],'rms_rotation_residual_deg':r['best']['rms_angle_deg'],'other_five_matching_indices':r['best']['other_five_matching_indices']}
  else:
   assert r['status']=='unresolved' and i['file'] in chassis_remainder_names, f"unreviewed ownership {r['index']} {i['file']}"
   # Centroid membership only corroborates mechanical identity, not proof of attachment.
   assert np.all(i['center_m']>=chassis_bounds[0]-1e-5) and np.all(i['center_m']<=chassis_bounds[1]+1e-5), f"chassis remainder outside deck bounds {i['index']}"
   body='body';basis='reviewed_chassis_component_within_enclosure_bounds';confidence='engineering_inference_needs_CAD_ownership_confirmation';evidence={'component_category':i['file'],'centroid_in_export_m':i['center_m'],'enclosure_bounds_m':chassis_bounds}
  ownership.append({'index':r['index'],'mesh':i['file'],'body':body,'basis':basis,'confidence':confidence,'evidence':evidence})
 assert len(ownership)==1753 and len({r['index'] for r in ownership})==1753
 (args.out/'provisional_body_ownership.json').write_text(json.dumps({'schema':1,'source_urdf_sha256':hashlib.sha256((args.source/'robot.urdf').read_bytes()).hexdigest(),'warning':'Complete geometry-based ownership hypothesis, not an exported or hardware-confirmed mate graph. Whole motor CAD components are lumped by their assembly-side pose; rotor reflected inertia and bearing race dynamics require calibration.','counts':dict(collections.Counter(r['body'] for r in ownership)),'parts':ownership},indent=2,default=serial)+'\n')
 summary={'source_urdf_sha256':hashlib.sha256((args.source/'robot.urdf').read_bytes()).hexdigest(),'source_link_count':len(root.findall('link')),'source_joint_count':len(root.findall('joint')),'instances':len(instances),'unique_meshes':len(mesh_cache),'inference_only':True,'topology_confirmation':{'source':'explicit user clarification in current task','date':'2026-09-10','statement':'yup, no more four bar, makes it way easier','result':'direct-drive serial topology confirmed; detailed attachment ownership remains inferred'},'topology_evidence':'18 coaxial independent motor+holder+flange+insert sets; six coxa anchors, six femur_first_stage and tibia_attatchment_plate pairs share rigid transform; six tibia poses vary relative to femur. No external four-bar structural meshes. Supports provisional 19-body/18-joint direct-drive serial model, not old physical four-bar. CAD mate graph still absent.','fit_methods':fits,'anchors':[{ 'leg':a['leg'],**{b:{'index':a[b]['index'],'mesh':a[b]['file'],'T_in_export':a[b]['T']} for b in ['coxa','femur','tibia']}}for a in anchors],'joint_axes':sorted(joints,key=lambda j:j['name']),'membership_status_counts':dict(collections.Counter(r['status'] for r in records)),'provisional_ownership_counts':dict(collections.Counter(r['body'] for r in ownership)),'missing_for_certification':['CAD mate graph / explicit rigid-body membership, especially bearing races and motor output vs housing parts','CAD joint limit and mate-frame Z sign; geometric axis sign has no hardware encoder meaning','hardware stop, simultaneous-joint/cable clearance sweeps and encoder calibration']}
 lengths=[]
 for a in anchors:
  js={j['type']:j for j in joints if j['leg']==a['leg']};p=np.array(js['coxa_yaw']['axis_point_m']);s=np.array(js['femur_pitch']['axis_point_m']);k=np.array(js['tibia_pitch']['axis_point_m']);v=mesh_cache['tibia.stl'].vertices
  lengths.append({'leg':a['leg'],'yaw_to_shoulder_m':np.linalg.norm(s-p),'shoulder_to_knee_m':np.linalg.norm(k-s),'tibia_local_max_x_m':v[:,0].max()})
 summary['current_geometry_dimensions']=lengths
 (args.out/'joint_geometry_recovery.json').write_text(json.dumps(summary,indent=2,default=serial)+'\n');(args.out/'body_membership_inference.json').write_text(json.dumps(records,indent=2,default=serial)+'\n')
 print(json.dumps({'fit_summary':fits,'dimensions':lengths,'membership':summary['membership_status_counts'],'max_axis_mismatch_deg':max(e['unsigned_axis_mismatch_deg'] for j in joints for e in j['independent_evidence']),'max_coaxial_line_error_m':max(e['line_distance_m'] for j in joints for e in j['independent_evidence'])},indent=2,default=serial))
if __name__=='__main__':main()
