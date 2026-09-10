#!/usr/bin/env python3
"""Independent proof/mapping check against the current integrated model.

Recomputes radii from unprocessed STL vertices and reruns FCL midpoint distances
using current rebased yaw joints; never mutates the robot model or source.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,zipfile
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

def Tof(p):
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();T[:3,3]=p['xyz'];return T
def spin(q):
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,0,q]).as_matrix();return T
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def angle_near(theta,reference):return reference+(theta-reference+180)%360-180

def main():
 here=Path(__file__).resolve().parent;repo=here.parents[2];ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,default=repo/'robot/hexapod_mkii_updated_v1/model.json');ap.add_argument('--bundle',type=Path,default=here);ap.add_argument('--source-zip',type=Path,default=here.parent/'import_001/source/HexapodLegUpdatedV2.zip');ap.add_argument('--fcl-path',type=Path);args=ap.parse_args()
 if args.fcl_path:sys.path.insert(0,str(args.fcl_path))
 import fcl
 bundle=args.bundle;model=json.load(open(args.model));pmap={p['id']:p for p in model['parts']};jmap={j['name']:j for j in model['joints']};data=json.load(open(bundle/'yaw_geometry_limits.json'));published=json.load(open(bundle/'yaw_published_joint_bounds.json'));cert=json.load(open(bundle/'continuous_clearance_certificate.json'));partsT={i:Tof(p)for i,p in pmap.items()}
 with zipfile.ZipFile(args.source_zip)as z:
  names=[n for n in z.namelist()if Path(n).name=='robot.urdf'];assert len(names)==1;source=ET.fromstring(z.read(names[0])).findall('link/visual')
 sourceT={}
 for i,v in enumerate(source):
  o=v.find('origin');T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz'),sep=' ');sourceT[i]=T
 plates=[p for p in pmap.values()if p['link']=='body'and'standoff_plate'in p['mesh']];coxae=[p for p in pmap.values()if p['link'].endswith('_coxa')];assert len(plates)==6 and len(coxae)==846
 meshes={};geoms={};objects={}
 for p in plates+coxae:
  f=p['mesh']
  if f not in meshes:
   m=trimesh.load(args.model.parent/'meshes'/f,process=False);meshes[f]=m;g=fcl.BVHModel();g.beginModel(len(m.vertices),len(m.faces));g.addSubModel(np.asarray(m.vertices,dtype=float),np.asarray(m.faces,dtype=np.int32));g.endModel();geoms[f]=g
  objects[p['id']]=fcl.CollisionObject(geoms[f])
 def manager(parts):
  mm=fcl.DynamicAABBTreeCollisionManager();mm.registerObjects([objects[p['id']]for p in parts]);mm.setup();return mm
 for p in plates:
  T=partsT[p['id']];objects[p['id']].setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
 fixed=manager(plates);results=[];new_leaves={}
 for record in data['legs']:
  leg=record['leg'];name=leg+'_coxa_yaw';j=jmap[name];shoulder=jmap[leg+'_femur_pitch'];pj=published['joints'][name];joint=data['joints'][name];assert all(pj[k]==joint[k]for k in ['lower_rad','upper_rad','zero_shift_rad']);shift=pj['zero_shift_rad'];assert abs(j['zero_shift_rad']-shift)<1e-14;assert abs(j['lower']-(pj['lower_rad']-shift))<1e-14;assert abs(j['upper']-(pj['upper_rad']-shift))<1e-14;assert j['default_value']==0;assert np.max(abs(np.array(j['axis'])-[0,0,1]))<1e-14
  tt=Tof(j);expected=[pj['provenance']['absolute_body_azimuth_lower_deg'],pj['provenance']['absolute_body_azimuth_midpoint_deg'],pj['provenance']['absolute_body_azimuth_upper_deg']];angles=[]
  for q,reference in zip([j['lower'],0,j['upper']],expected):
   d=(tt@spin(q))[:3,:3]@np.array(shoulder['xyz']);angle=angle_near(math.degrees(math.atan2(d[1],d[0])),reference);angles.append(angle);assert abs(angle-reference)<1e-9
  previous=tt@spin(-shift);d=previous[:3,:3]@np.array(shoulder['xyz']);az=angle_near(math.degrees(math.atan2(d[1],d[0])),record['old_radial_zero_azimuth_deg']);assert abs(az-record['old_radial_zero_azimuth_deg'])<1e-9
  group=[p for p in coxae if p['link']==leg+'_coxa'];assert len(group)==141;moving=manager(group);radius=0.;cad_error=0.;vertex_count=0;Tc=tt@spin(j['cad_value'])
  for p in group:
   vertices=meshes[p['mesh']].vertices;local=vertices@partsT[p['id']][:3,:3].T+partsT[p['id']][:3,3];radius=max(radius,float(np.linalg.norm(local[:,:2],axis=1).max()));delta=Tc@partsT[p['id']]-sourceT[p['id']];cad_error=max(cad_error,float(np.linalg.norm(vertices@delta[:3,:3].T+delta[:3,3],axis=1).max()));vertex_count+=len(vertices)
  assert cad_error<1e-5;assert abs(radius-record['moving_geometry_max_radius_about_yaw_m'])<2e-8
  leaves=sorted(cert[leg]['certified_leaves'],key=lambda x:x['lower_offset_rad']);assert not cert[leg]['unresolved'];assert abs(leaves[0]['lower_offset_rad']-pj['lower_rad'])<1e-14 and abs(leaves[-1]['upper_offset_rad']-pj['upper_rad'])<1e-14
  for a,b in zip(leaves,leaves[1:]):assert abs(a['upper_offset_rad']-b['lower_offset_rad'])<1e-14
  checked=[]
  for leaf in leaves:
   a,b=leaf['lower_offset_rad'],leaf['upper_offset_rad'];assert a<b;mid=(a+b)/2;assert abs(mid-leaf['midpoint_offset_rad'])<1e-14;stored=leaf['midpoint_distance_m']-2*record['moving_geometry_max_radius_about_yaw_m']*math.sin((b-a)/4);assert abs(stored-leaf['continuous_distance_lower_bound_m'])<1e-14 and stored>=.0005
   # Independently query unprocessed source triangles in the CURRENT rebased model.
   Tjoint=tt@spin(mid-shift)
   for p in group:
    T=Tjoint@partsT[p['id']];objects[p['id']].setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
   moving.update();dd=fcl.DistanceData(request=fcl.DistanceRequest());moving.distance(fixed,dd,fcl.defaultDistanceCallback);distance=float(dd.result.min_distance);bound=distance-2*(radius+1e-12)*math.sin((b-a)/4);assert bound>=.0005, (leg,mid,bound);assert abs(distance-leaf['midpoint_distance_m'])<2e-8
   checked.append({'previous_zero_lower_rad':a,'previous_zero_upper_rad':b,'current_zero_midpoint_rad':mid-shift,'rechecked_midpoint_distance_m':distance,'recorded_midpoint_distance_m':leaf['midpoint_distance_m'],'independently_recomputed_continuous_gap_lower_bound_m':bound})
  row={'leg':leg,'coxa_parts':len(group),'all_current_yaw_defaults_zero':True,'absolute_azimuth_deg_at_lower_zero_upper':angles,'expected_absolute_azimuth_deg_at_lower_zero_upper':expected,'old_radial_reference_measured_deg':az,'zero_shift_rad':shift,'rebased_joint_interval_rad':[j['lower'],j['upper']],'raw_vertex_radius_m':radius,'certificate_radius_m':record['moving_geometry_max_radius_about_yaw_m'],'unprocessed_vertices_evaluated':vertex_count,'max_source_CAD_vertex_error_m':cad_error,'continuous_interval_coverage_verified':True,'leaf_count':len(checked),'all_leaf_original_formulas_verified':True,'all_midpoint_distances_requeried':True,'max_requeried_distance_difference_m':max(abs(x['rechecked_midpoint_distance_m']-x['recorded_midpoint_distance_m'])for x in checked),'minimum_rechecked_continuous_gap_lower_bound_m':min(x['independently_recomputed_continuous_gap_lower_bound_m']for x in checked)};results.append(row);new_leaves[leg]=checked;print(json.dumps(row),flush=True)
 report={'schema':1,'model_path_relative_to_repository':str(args.model.relative_to(repo)),'model_sha256':sha(args.model),'input_hashes':{f:sha(bundle/f)for f in ['yaw_geometry_limits.json','yaw_geometry_limits_used_by_builder.json','yaw_published_joint_bounds.json','continuous_clearance_certificate.json']},'source_zip_sha256':sha(args.source_zip),'method':'Independent endpoint FK in current model, current/default and former-zero mapping, actual unprocessed-vertex radii and CAD-pose reproduction, complete leaf partition and formula checks; reruns every midpoint FCLdistance using current rebased joint and unprocessedSTL triangles.','passed':True,'standoff_clearance_only':True,'continuous_minimum_gap_m':.0005,'all_configuration_collision_certificate':False,'joint_review_pitch_ranges_not_tested_here':True,'legs':results}
 (bundle/'integrated_model_verification.json').write_text(json.dumps(report,indent=2)+'\n');(bundle/'independent_midpoint_distance_checks.json').write_text(json.dumps(new_leaves,indent=2)+'\n');print(json.dumps({'passed':True,'leaves_rechecked':sum(r['leaf_count']for r in results),'minimum_gap_bound_m':min(r['minimum_rechecked_continuous_gap_lower_bound_m']for r in results)}),flush=True)
if __name__=='__main__':main()
