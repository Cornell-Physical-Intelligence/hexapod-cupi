#!/usr/bin/env python3
"""Intrinsic coxa rotation envelope against actual chassis structural triangles."""
import argparse,collections,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/tmp/hexapod-urdf-joint-validation-20260910/deps')
import fcl

def jdefault(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.floating):return float(x)
 if isinstance(x,np.integer):return int(x)
 raise TypeError(type(x))
def matrix(p):
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();T[:3,3]=p['xyz'];return T
def rz(q):
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,0,q]).as_matrix();return T
def wrapped(deg):return float((deg+180)%360-180)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,default=Path('/tmp/hexapod-urdf-build-20260910'));ap.add_argument('--out',type=Path,default=Path('/tmp/hexapod-urdf-yaw-envelope-20260910'));ap.add_argument('--buffer-mm',type=float,default=.5);ap.add_argument('--certify-existing',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 model=json.load(open(args.model_dir/'model.json'));pmap={p['id']:p for p in model['parts']};jmap={j['name']:j for j in model['joints']};legs=['lf','lm','lr','rf','rm','rr'];plateids=[p['id']for p in pmap.values()if p['link']=='body' and 'standoff_plate'in p['mesh']];shellids=[p['id']for p in pmap.values()if p['link']=='body' and any(f in p['mesh']for f in ['top_enclosure','bottom_enclosure','bottom_plate'])];assert len(plateids)==6 and len(shellids)==3
 coxaids={leg:[p['id']for p in pmap.values()if p['link']==leg+'_coxa']for leg in legs};assert all(len(ids)==141 for ids in coxaids.values())
 ids=sorted(set(plateids+shellids+sum(coxaids.values(),[])));mesh={};geoms={};objects={};partT={i:matrix(pmap[i])for i in ids};geomfile={}
 for i in ids:
  f=pmap[i]['mesh']
  if f not in mesh:
   m=trimesh.load(args.model_dir/'meshes'/f,process=True);g=fcl.BVHModel();g.beginModel(len(m.vertices),len(m.faces));g.addSubModel(np.asarray(m.vertices,float),np.asarray(m.faces,np.int32));g.endModel();mesh[f]=m;geoms[f]=g;geomfile[id(g)]=f
  objects[i]=fcl.CollisionObject(geoms[f])
 keyT=lambda T:tuple(np.round(T[:3,:].reshape(-1),11))
 fixedlookup={}
 for i in plateids+shellids:
  T=partT[i];objects[i].setTransform(fcl.Transform(T[:3,:3],T[:3,3]));fixedlookup[(pmap[i]['mesh'],keyT(T))]=i
 def manager(inds):
  m=fcl.DynamicAABBTreeCollisionManager();m.registerObjects([objects[i]for i in inds]);m.setup();return m
 fixedmanagers={'standoff':manager(plateids),'other_structure':manager(shellids),'all_structure':manager(plateids+shellids)};movingmanagers={leg:manager(coxaids[leg])for leg in legs};cache={};calls=0
 def probe(leg,q,target='standoff'):
  nonlocal calls
  k=(leg,round(float(q),13),target)
  if k in cache:return cache[k]
  calls+=1;coxaT=matrix(jmap[leg+'_coxa_yaw'])@rz(q);lookup=dict(fixedlookup)
  for i in coxaids[leg]:
   T=coxaT@partT[i];objects[i].setTransform(fcl.Transform(T[:3,:3],T[:3,3]));lookup[(pmap[i]['mesh'],keyT(T))]=i
  movingmanagers[leg].update();best={'distance_m':float('inf')};queries=0
  def callback(a,b,data):
   nonlocal best,queries
   queries+=1;r=fcl.DistanceResult();d=fcl.distance(a,b,fcl.DistanceRequest(enable_nearest_points=True),r)
   if d<best['distance_m']:
    fa=geomfile[id(r.o1)];fb=geomfile[id(r.o2)];Ta=np.eye(4);Ta[:3,:3]=a.getRotation();Ta[:3,3]=a.getTranslation();Tb=np.eye(4);Tb[:3,:3]=b.getRotation();Tb[:3,3]=b.getTranslation();ka=(fa,keyT(Ta));kb=(fb,keyT(Tb))
    if ka not in lookup or kb not in lookup:ka=(fa,keyT(Tb));kb=(fb,keyT(Ta))
    ia=lookup[ka];ib=lookup[kb];pair=sorted([ia,ib],key=lambda i:pmap[i]['link']=='body');best={'distance_m':float(d),'coxa_part_id':pair[0],'chassis_part_id':pair[1],'coxa_mesh':pmap[pair[0]]['mesh'],'chassis_mesh':pmap[pair[1]]['mesh'],'nearest_points_m':r.nearest_points,'triangle_indices':[r.b1,r.b2]}
   return best['distance_m']<=0,best['distance_m']
  movingmanagers[leg].distance(fixedmanagers[target],None,callback);best['narrowphase_distance_queries']=queries;best['offset_rad']=float(q);best['offset_deg']=float(np.degrees(q));cache[k]=best;return best
 if args.certify_existing:
  data=json.load(open(args.out/'yaw_geometry_limits.json'));original=json.dumps(data,indent=2,default=jdefault)+'\n'
  if not (args.out/'raw_threshold_yaw_limits.json').exists():(args.out/'raw_threshold_yaw_limits.json').write_text(original)
  buffer=args.buffer_mm/1000.;all_leaves={}
  for row in data['legs']:
   leg=row['leg'];az0=row['old_radial_zero_azimuth_deg'];rawlo=row['buffered_standoff_boundaries']['lower']['absolute_free_azimuth_deg'];rawhi=row['buffered_standoff_boundaries']['upper']['absolute_free_azimuth_deg'];loabs=math.ceil(rawlo*100)/100;hiabs=math.floor(rawhi*100)/100;lo=math.radians(loabs-az0);hi=math.radians(hiabs-az0);mid=(lo+hi)/2;radius=row['moving_geometry_max_radius_about_yaw_m'];todo=[(lo,hi,0)];leaves=[];unresolved=[];start_calls=calls;guard=1e-10
   while todo:
    a,b,depth=todo.pop();middle=(a+b)/2;obs=probe(leg,middle);bound=obs['distance_m']-2*radius*math.sin((b-a)/4);record={'lower_offset_rad':a,'upper_offset_rad':b,'midpoint_offset_rad':middle,'midpoint_distance_m':obs['distance_m'],'continuous_distance_lower_bound_m':bound,'depth':depth}
    if bound>=buffer+guard:leaves.append(record)
    elif depth>=35 or calls-start_calls>=3000:unresolved.append(record)
    else:todo.extend([(a,middle,depth+1),(middle,b,depth+1)])
   ends={'lower':probe(leg,lo),'upper':probe(leg,hi)};certified=not unresolved and all(e['distance_m']>=buffer+guard for e in ends.values())
   cert={'certified':certified,'minimum_distance_m':buffer,'numerical_acceptance_guard_m':guard,'method':'For interval[a,b], allmovingvertices rotate aroundthe fixedZaxis withradius<=R. The originaltriangle-set distance atmidpoint minus2Rsin((b-a)/4) lowerboundsevery angleinthatinterval. Adaptive subintervalleaves cover the wholepublished interval.','maximum_vertex_radius_m':radius,'pose_distance_queries':calls-start_calls,'certified_leaf_intervals':len(leaves),'unresolved_intervals':len(unresolved),'maximum_subdivision_depth':max(r['depth']for r in leaves+unresolved),'minimum_leaf_distance_lower_bound_m':min(r['continuous_distance_lower_bound_m']for r in leaves)if leaves else None,'only_one_dimensional_coxa_to_standoff_geometry':True,'covers_other_body_parts_or_femur_tibia_configurations':False}
   row['published_interval']={'absolute_lower_azimuth_deg':loabs,'absolute_upper_azimuth_deg':hiabs,'absolute_midpoint_azimuth_deg':(loabs+hiabs)/2,'absolute_lower_wrapped_deg':wrapped(loabs),'absolute_upper_wrapped_deg':wrapped(hiabs),'absolute_midpoint_wrapped_deg':wrapped((loabs+hiabs)/2),'old_zero_lower_rad':lo,'old_zero_upper_rad':hi,'old_zero_shift_rad':mid,'old_radial_zero_shift_deg':math.degrees(mid),'angular_half_width_deg':(hiabs-loabs)/2,'endpoint_distances':ends,'rounding':'absoluteendpoints roundedinward to0.01deg'};row['continuous_standoff_clearance_certificate']=cert;row['continuous_requested_buffer_certified']=certified
   joint=data['joints'][leg+'_coxa_yaw'];joint['lower_rad']=lo;joint['upper_rad']=hi;joint['zero_shift_rad']=mid;joint['provenance'].update({'absolute_body_azimuth_lower_deg':loabs,'absolute_body_azimuth_upper_deg':hiabs,'absolute_body_azimuth_midpoint_deg':(loabs+hiabs)/2,'published_endpoints_rounded_inward_deg':.01,'continuous_standoff_minimum_0p5mm_certified':certified,'certificate_leaf_count':len(leaves)})
   all_leaves[leg]={'certified_leaves':leaves,'unresolved':unresolved};print(json.dumps({'certified_leg':leg,'published_absolute_deg':[loabs,hiabs],'midpoint_deg':(loabs+hiabs)/2,'certified':certified,'leaves':len(leaves),'queries':calls-start_calls,'minimum_lower_bound_mm':cert['minimum_leaf_distance_lower_bound_m']*1000}),flush=True)
  data['published_angle_resolution_deg']=.01;data['continuous_intrinsic_standoff_clearance_certified']=all(r['continuous_standoff_clearance_certificate']['certified']for r in data['legs']);data['certificate_queries']=calls;data['notes'].append('Publishedbounds are roundedinwardto0.01degrees; continuous0.5mm certification appliesonlytorotationofthewholecoxa groupagainstallsixstandoffs, nottootherbodyclearances oruserexpandedpitchcombinations.')
  (args.out/'yaw_geometry_limits.json').write_text(json.dumps(data,indent=2,default=jdefault)+'\n');(args.out/'continuous_clearance_certificate.json').write_text(json.dumps(all_leaves,indent=2,default=jdefault)+'\n');return
 step=np.radians(2.0);grid=np.arange(-90,91)*step;buffer=args.buffer_mm/1000.;tol=np.radians(1e-5);summary={'schema':1,'source_model_sha256':hashlib.sha256((args.model_dir/'model.json').read_bytes()).hexdigest(),'source_grouping':'unchanged 141-part coxa group for each leg','chassis_standoff_part_ids':plateids,'other_structure_part_ids':shellids,'azimuth_convention':'+X=0deg, +Y=90deg, positive counterclockwise viewed from +Z; forward(-Y)=-90deg','minimum_requested_buffer_m':buffer,'grid_step_deg':2,'boundary_bracket_tolerance_deg':1e-5,'only_intrinsic_coxa_motion':True,'femur_tibia_groups_included':False,'joints':{},'legs':[]};scan={}
 def interval(leg,target,threshold,values):
  center=len(grid)//2
  if values[center]['distance_m']<=threshold:return {'contains_old_zero':False,'old_zero':values[center]}
  bounds=[]
  for direction in [-1,1]:
   k=center
   while 0<=k+direction<len(grid) and values[k+direction]['distance_m']>threshold:k+=direction
   if not 0<=k+direction<len(grid):bounds.append({'unbounded_within_full_halfturn':True});continue
   free=float(grid[k]);blocked=float(grid[k+direction]);iterations=0
   while abs(free-blocked)>tol:
    q=(free+blocked)/2;r=probe(leg,q,target);iterations+=1
    if r['distance_m']>threshold:free=q
    else:blocked=q
   # Always return the just-free endpoint so a0.5mm buffer is not rounded away.
   bounds.append({'free_offset_rad':free,'blocked_offset_rad':blocked,'free':probe(leg,free,target),'blocked':probe(leg,blocked,target),'iterations':iterations})
  return {'contains_old_zero':True,'lower':bounds[0],'upper':bounds[1]}
 for leg in legs:
  yaw=jmap[leg+'_coxa_yaw'];shoulder=jmap[leg+'_femur_pitch'];T=matrix(yaw);d=T[:3,:3]@np.asarray(shoulder['xyz']);az0=math.atan2(d[1],d[0]);values=[probe(leg,q)for q in grid];other=[probe(leg,q,'other_structure')for q in grid];scan[leg]={'standoff':values,'other_structure':other};raw=interval(leg,'standoff',0.,values);buff=interval(leg,'standoff',buffer,values);otherraw=interval(leg,'other_structure',0.,other);otherbuff=interval(leg,'other_structure',buffer,other)
  assert buff['contains_old_zero'] and 'free_offset_rad'in buff['lower'] and 'free_offset_rad'in buff['upper'],(leg,buff)
  lo=buff['lower']['free_offset_rad'];hi=buff['upper']['free_offset_rad'];mid=(lo+hi)/2
  for rr in [raw,buff,otherraw,otherbuff]:
   if rr.get('contains_old_zero'):
    for side in ['lower','upper']:
     if 'free_offset_rad'in rr[side]:
      x=rr[side];x['absolute_free_azimuth_deg']=float(np.degrees(az0+x['free_offset_rad']));x['absolute_blocked_azimuth_deg']=float(np.degrees(az0+x['blocked_offset_rad']));x['absolute_free_azimuth_wrapped_deg']=wrapped(x['absolute_free_azimuth_deg'])
  # Additional interior 0.5-degree sampling guards against holes missed by coarse
  # interval scan; it remains finite evidence, not all-continuous clearance proof.
  densegrid=np.linspace(lo,hi,int(np.ceil((hi-lo)/np.radians(.5)))+1);dense=[probe(leg,q)for q in densegrid];minimum=min(r['distance_m']for r in dense);assert minimum>=buffer-1e-9
  endpoints_other={'lower':probe(leg,lo,'other_structure'),'upper':probe(leg,hi,'other_structure'),'midpoint':probe(leg,mid,'other_structure')}
  maxradius=0.
  for i in coxaids[leg]:
   vv=mesh[pmap[i]['mesh']].vertices@partT[i][:3,:3].T+partT[i][:3,3];maxradius=max(maxradius,float(np.linalg.norm(vv[:,:2],axis=1).max()))
  result={'leg':leg,'old_radial_zero_azimuth_deg':float(np.degrees(az0)),'raw_standoff_collision_boundaries':raw,'buffered_standoff_boundaries':buff,'other_structure_raw_boundaries':otherraw,'other_structure_buffered_boundaries':otherbuff,'recommended_midpoint_azimuth_deg':float(np.degrees(az0+mid)),'recommended_midpoint_wrapped_deg':wrapped(float(np.degrees(az0+mid))),'recommended_zero_shift_from_old_radial_deg':float(np.degrees(mid)),'recommended_neutral_standoff_distance':probe(leg,mid),'other_structure_at_standoff_interval':endpoints_other,'interior_scan_sample_count':len(dense),'interior_scan_max_step_deg':float(np.degrees(np.max(np.diff(densegrid)))),'interior_scan_min_distance_m':minimum,'moving_geometry_max_radius_about_yaw_m':maxradius,'continuous_min_distance_lower_bound_from_grid_m':minimum-2*maxradius*math.sin(float(np.max(np.diff(densegrid)))/4),'continuous_requested_buffer_certified':False}
  # Endpoint guards, not a claim that user-widened pitch combinations cannot collide.
  summary['legs'].append(result);summary['joints'][leg+'_coxa_yaw']={'lower_rad':lo,'upper_rad':hi,'bounds_frame':'previous_viewer_zero','zero_shift_rad':mid,'provenance':{'method':'all141coxa instances vs six exact chassis standoff triangle meshes; full360deg coarse scan, bisection to1e-5deg and <=0.5deg interior sample','minimum_endpoint_buffer_m':buffer,'absolute_body_azimuth_lower_deg':float(np.degrees(az0+lo)),'absolute_body_azimuth_upper_deg':float(np.degrees(az0+hi)),'absolute_body_azimuth_midpoint_deg':float(np.degrees(az0+mid)),'limiting_lower_part_ids':[buff['lower']['free']['coxa_part_id'],buff['lower']['free']['chassis_part_id']],'limiting_upper_part_ids':[buff['upper']['free']['coxa_part_id'],buff['upper']['free']['chassis_part_id']],'not_a_combined_femur_tibia_clearance_gate':True}}
  print(json.dumps({'leg':leg,'absolute_deg':[float(np.degrees(az0+lo)),float(np.degrees(az0+hi))],'midpoint_deg':float(np.degrees(az0+mid)),'radial_shift_deg':float(np.degrees(mid)),'lower_pair':summary['joints'][leg+'_coxa_yaw']['provenance']['limiting_lower_part_ids'],'upper_pair':summary['joints'][leg+'_coxa_yaw']['provenance']['limiting_upper_part_ids'],'other_structure_endpoint_distances_mm':[endpoints_other[s]['distance_m']*1000 for s in ['lower','upper','midpoint']]}),flush=True)
  (args.out/'yaw_geometry_limits.json').write_text(json.dumps(summary,indent=2,default=jdefault)+'\n')
 summary['total_pose_queries']=calls;summary['qualified_hardware_stops']=False;summary['notes']=['Only the connected free yaw interval containing current radial inspection stance is selected. Other free pockets may exist beyond obstacles.','Distances use original tessellated surfaces; no convex hulls/spheres and no bearing relocation. Triangle-mesh containment without crossing is not independently tested.','The0.5mm buffer is a proposed CAD inspection margin, not hardware manufacturing/cable/load allowance.','Only coxa rotation changes. User-specified widened femur/tibia motion can still cause coupled collisions.']
 (args.out/'yaw_geometry_limits.json').write_text(json.dumps(summary,indent=2,default=jdefault)+'\n');(args.out/'full_rotation_scan.json').write_text(json.dumps(scan,indent=2,default=jdefault)+'\n');print(json.dumps({'done':True,'queries':calls}),flush=True)
if __name__=='__main__':main()
