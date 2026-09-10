#!/usr/bin/env python3
"""Apply reviewed joint travel and yaw-zero shifts without changing CAD geometry.

--joint-review contains joints{name:{lower_rad,upper_rad,
 bounds_frame:'previous_viewer_zero',zero_shift_rad,provenance}}.
Pitch zero shifts must be zero. Yaw bounds are supplied about the previous zero;
the output origin rotates by zero_shift_rad and its limits/CAD value subtract it.
All model part placements, link masses/COM/tensors, and mesh bytes stay unchanged.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pose(p):
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();T[:3,3]=p['xyz'];return T

def rz(q):
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0.,0.,q]).as_matrix();return T

def fk(model,q=None):
 q={} if q is None else q;out={'body':np.eye(4)};remaining=list(model['joints'])
 for _ in range(len(remaining)+1):
  pending=[]
  for j in remaining:
   if j['parent'] not in out:pending.append(j);continue
   if j['child'] in out:raise ValueError('Duplicate child/cycle')
   if np.max(np.abs(np.asarray(j['axis'])-[0,0,1]))>1e-12:raise ValueError('Review requires established joint-local +Z axis')
   out[j['child']]=out[j['parent']]@pose(j)@rz(q.get(j['name'],0.))
  if not pending:return out
  if len(pending)==len(remaining):raise ValueError('Disconnected or cyclic graph')
  remaining=pending
 raise ValueError('Graph traversal failed')

def wrap(a):return math.atan2(math.sin(a),math.cos(a))

def stable_origin(el,T):
 R=T[:3,:3];horizontal=math.hypot(R[0,0],R[1,0]);pitch=math.atan2(-R[2,0],horizontal)
 if horizontal>1e-14:
  yaw=math.atan2(R[1,0],R[0,0]);sy,cy=math.sin(yaw),math.cos(yaw)
  roll=math.atan2(sy*R[0,2]-cy*R[1,2],-sy*R[0,1]+cy*R[1,1])
 else:roll=math.atan2(-R[1,2],R[1,1]);yaw=0.
 el.set('xyz',' '.join(format(float(x),'.17g') for x in T[:3,3]))
 el.set('rpy',' '.join(format(float(x),'.17g') for x in [roll,pitch,yaw]))

def apply_review(original,review):
 if review.get('schema')!=1 or not isinstance(review.get('revision'),str):raise ValueError('Joint review requires schema1 and revision')
 entries=review['joints'];names={j['name'] for j in original['joints']}
 if set(entries)!=names:raise ValueError('Review must cover exactly every joint')
 model=copy.deepcopy(original);oldzero=fk(original);changes=[]
 for j in model['joints']:
  e=entries[j['name']]
  if e.get('bounds_frame')!='previous_viewer_zero':raise ValueError('Input limits must use the previous viewer zero')
  lo,hi,shift=[float(e[k]) for k in ('lower_rad','upper_rad','zero_shift_rad')]
  if not np.isfinite([lo,hi,shift]).all() or lo>=hi:raise ValueError('Invalid finite ordered limits')
  is_yaw=j['name'].endswith('_coxa_yaw')
  if not is_yaw and shift!=0.:raise ValueError('User explicitly retained all pitch zeros')
  if not is_yaw:
   requested=(-120.,80.) if j['name'].endswith('_femur_pitch') else (-5.,180.)
   if max(abs(lo-math.radians(requested[0])),abs(hi-math.radians(requested[1])))>1e-12:raise ValueError('Pitch bounds differ from this confirmed user revision')
  if is_yaw:
   T=pose(j)@rz(shift);j['quaternion_xyzw']=Rotation.from_matrix(T[:3,:3]).as_quat().tolist()
   # Rotation about the joint's own +Z never moves its origin.
   direction=oldzero[j['child']][:3,0];az=math.degrees(math.atan2(direction[1],direction[0]))
   previous_az=float(e.get('absolute_previous_zero_azimuth_deg',az))
   if abs(wrap(math.radians(previous_az-az)))>1e-8:raise ValueError('Provided absolute azimuth disagrees with source frame')
   j.update({'absolute_previous_zero_azimuth_deg':previous_az,'absolute_zero_azimuth_deg':previous_az+math.degrees(shift),
    'absolute_lower_azimuth_deg':previous_az+math.degrees(lo),'absolute_upper_azimuth_deg':previous_az+math.degrees(hi),
    'absolute_azimuth_convention':'body +X(left)=0deg,+Y=90deg,forward(-Y)=-90deg;CCW about+Z;unwrapped interval',
    'limit_provenance':'geometry_coxa_vs_standoff_endpoint_clearance'})
  else:j['limit_provenance']='user_current_viewer_zero'
  j.update({'lower':lo-shift,'upper':hi-shift,'cad_value':j['cad_value']-shift,'default_value':0.,
   'zero_shift_rad':shift,'previous_zero_lower_rad':lo,'previous_zero_upper_rad':hi,
   'limits_frame':'current_joint_zero','limits_are_collision_free_envelope':False,'limit_evidence':copy.deepcopy(e.get('provenance',{}))})
  if not j['lower']<=0<=j['upper']:raise ValueError('Recommended neutral must lie inside reviewed travel')
  model['cad_pose'][j['name']]=j['cad_value'];changes.append({'joint':j['name'],**{k:j[k] for k in ('lower','upper','cad_value','zero_shift_rad','limit_provenance')}})
 model['joint_review_revision']=review['revision'];model['joint_review_provenance']=copy.deepcopy(review.get('provenance',{}))
 model['historical_inspection_envelope']={'status':'historical only;133-pose evidence does not qualify revised travel',
  'joint_limits_previous_zero':{j['name']:{'lower_rad':j['lower'],'upper_rad':j['upper']} for j in original['joints']}}
 model['inspection_stance']={'yaw':'midpoint of each reviewed coxa-vs-standoff interval','femur_elevation_deg':20,'relative_knee_deg':-110,'tibia_elevation_deg':-90,'pitch_zeros_changed':False}
 model['limits_semantics']='Femur[-120,+80]deg and tibia[-5,+180]deg are user-confirmed travel relative to unchanged prior viewer pitch zeros. Coxa zero is each buffered intrinsic coxa/standoff interval midpoint. These are travel limits, not a collision-free full-robot envelope.'
 model['warnings']=[w for w in model.get('warnings',[]) if 'finite sweep' not in w.lower() and 'finite joint' not in w.lower()]
 model['warnings']+=['The user-specified wide femur/tibia travel includes self-colliding and ground-intersecting combinations.','Coxa limits concern coxa versus standoff geometry at 0.5 mm endpoint clearance; they do not qualify femur/tibia/body clearance.','The previous 133-pose collision report applies only to the earlier narrow inspection envelope.']
 model['isaac_admitted']=False
 return model,changes

def aggregate_in_cad(model):
 frames=fk(model,model['cad_pose']);parts=[]
 for l in model['links']:
  T=frames[l['name']];R=T[:3,:3];parts.append((l['mass'],R@np.asarray(l['com'])+T[:3,3],R@np.asarray(l['inertia'])@R.T))
 mass=math.fsum(p[0] for p in parts);com=sum(m*c for m,c,I in parts)/mass;inertia=np.zeros((3,3))
 for m,c,I in parts:
  d=c-com;inertia+=I+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
 return {'mass_kg':mass,'com_export_m':com.tolist(),'inertia_about_com_export_kg_m2':inertia.tolist()}

def compare_revision(old,new):
 A=fk(old,old['cad_pose']);B=fk(new,new['cad_pose']);errors=[]
 for p in old['parts']:
  errors.append({'id':p['id'],'transform_max_abs_error':float(np.max(np.abs(A[p['link']]@pose(p)-B[p['link']]@pose(p))))})
 before=aggregate_in_cad(old);after=aggregate_in_cad(new)
 pitch_preserved=all(j['xyz']==n['xyz'] and j['quaternion_xyzw']==n['quaternion_xyzw'] and j['cad_value']==n['cad_value'] for j,n in zip(old['joints'],new['joints']) if not j['name'].endswith('_coxa_yaw'))
 check={'all_parts_and_link_inertials_unchanged':old['parts']==new['parts'] and old['links']==new['links'],
  'motor_mass_additions_unchanged':old.get('mass_additions')==new.get('mass_additions'),
  'all_pitch_origins_and_cad_values_unchanged':pitch_preserved,'graph_acyclic_connected':len(B)==len(new['links']),
  'all_part_pose_max_abs_error':max(e['transform_max_abs_error'] for e in errors),
  'CAD_mass_error_kg':abs(before['mass_kg']-after['mass_kg']),
  'CAD_com_error_m':float(np.max(np.abs(np.asarray(before['com_export_m'])-after['com_export_m']))),
  'CAD_inertia_error_kg_m2':float(np.max(np.abs(np.asarray(before['inertia_about_com_export_kg_m2'])-after['inertia_about_com_export_kg_m2']))),
  'aggregate_before':before,'aggregate_after':after}
 check['passes']=bool(check['all_parts_and_link_inertials_unchanged'] and check['motor_mass_additions_unchanged'] and pitch_preserved and check['graph_acyclic_connected'] and all(check[k]<1e-12 for k in ['all_part_pose_max_abs_error','CAD_mass_error_kg','CAD_com_error_m','CAD_inertia_error_kg_m2']))
 return check,errors

def parse_origin(el):
 T=np.eye(4);T[:3,3]=np.fromstring(el.get('xyz'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(el.get('rpy'),sep=' ')).as_matrix();return T

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-model-dir',type=Path,required=True);ap.add_argument('--joint-review',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 src=a.source_model_dir.resolve();out=a.out.resolve()
 if src==out or src in out.parents:raise ValueError('Successor output must be separate from preserved source model')
 out.mkdir(parents=True,exist_ok=True);(out/'urdf').mkdir(exist_ok=True);(out/'meshes').mkdir(exist_ok=True)
 review=read(a.joint_review);oldraw=read(src/'model.json');raw,changes=apply_review(oldraw,review);checks,part_errors=compare_revision(oldraw,raw)
 if not checks['passes']:raise ValueError('Raw-model CAD invariance failure')
 variants={'model.json':raw};variant_checks={'model.json':checks}
 if (src/'model_rs05_mass_corrected.json').exists():
  original=read(src/'model_rs05_mass_corrected.json');corrected,_=apply_review(original,review);check,_=compare_revision(original,corrected)
  if not check['passes']:raise ValueError('Corrected-model CAD invariance failure')
  variants['model_rs05_mass_corrected.json']=corrected;variant_checks['model_rs05_mass_corrected.json']=check
 mapping=read(src/'mesh_mapping.json');vertices={}
 for item in mapping:
  source=src/'meshes'/item['mesh'];dest=out/'meshes'/item['mesh']
  if sha(source)!=item['sha256']:raise ValueError('Source mesh changed')
  if not dest.exists():shutil.copyfile(source,dest)
  if sha(dest)!=item['sha256']:raise ValueError('Successor mesh mismatch')
  vertices[item['mesh']]=np.asarray(trimesh.load_mesh(dest,process=False).vertices)
 baseline=fk(raw);minima=[];tibia_meshes={item['mesh'] for item in mapping if Path(item['source_mesh']).name=='tibia.stl'}
 for p in raw['parts']:
  T=baseline[p['link']]@pose(p);v=vertices[p['mesh']];minimum=float(np.min(v@T[2,:3]+T[2,3]));minima.append({'id':p['id'],'name':p['name'],'link':p['link'],'mesh':p['mesh'],'min_z_before_root_m':minimum})
 tibia_min=min(p['min_z_before_root_m'] for p in minima if p['mesh'] in tibia_meshes);root_height=.005-tibia_min
 for p in minima:p['ground_clearance_m']=p['min_z_before_root_m']+root_height
 lowest=min(minima,key=lambda p:p['ground_clearance_m'])
 for name,model in variants.items():
  model['root_height_m']=root_height;model['root_height_basis']='actual lowest tibia mesh vertex at revised default pose +5mm ground clearance';write(out/name,model)
 write(out/'cad_pose.json',raw['cad_pose']);write(out/'mesh_mapping.json',mapping);write(out/'joint_review.json',review)
 write(out/'part_clearances_default.json',minima);write(out/'part_pose_revision_errors.json',part_errors)
 write(out/'previous_joint_limits.json',{j['name']:{'lower':j['lower'],'upper':j['upper'],'cad_value':j['cad_value'],'xyz':j['xyz'],'quaternion_xyzw':j['quaternion_xyzw']} for j in oldraw['joints']})
 new_by_name={j['name']:j for j in raw['joints']};xml_checks={}
 for source in sorted((src/'urdf').glob('*.urdf')):
  tree=ET.parse(source);robot=tree.getroot();robot.insert(0,ET.Comment('REVIEWED JOINT TRAVEL SUCCESSOR. Pitch zeros preserved; coxa neutral rebased to buffered interval midpoint. Wide user pitch travel is not a collision-free envelope. No Isaac admission.'))
  if {j.get('name') for j in robot.findall('joint')}!=set(new_by_name):raise ValueError('URDF graph mismatch')
  for el in robot.findall('joint'):
   j=new_by_name[el.get('name')]
   if j['name'].endswith('_coxa_yaw'):stable_origin(el.find('origin'),pose(j))
   el.find('limit').set('lower',format(j['lower'],'.17g'));el.find('limit').set('upper',format(j['upper'],'.17g'))
  ET.indent(robot,space='  ');dest=out/'urdf'/source.name;tree.write(dest,encoding='utf-8',xml_declaration=True)
  # Independently parse the actual serialized artifact, not in-memory transforms.
  robot=ET.parse(dest).getroot();X={'body':np.eye(4)}
  for j in robot.findall('joint'):X[j.find('child').get('link')]=X[j.find('parent').get('link')]@parse_origin(j.find('origin'))@rz(raw['cad_pose'][j.get('name')])
  previous_fk=fk(oldraw,oldraw['cad_pose']);byid={p['id']:p for p in oldraw['parts']};errs=[]
  for link in robot.findall('link'):
   for vis in link.findall('visual'):
    idx=int(vis.get('name').split('_')[1]);p=byid[idx];errs.append(float(np.max(np.abs(X[link.get('name')]@parse_origin(vis.find('origin'))-previous_fk[p['link']]@pose(p)))))
  xml_checks[source.name]={'parts_checked':len(errs),'max_CAD_transform_error':max(errs),'passes':len(errs)==len(raw['parts']) and max(errs)<1e-12,'sha256':sha(dest)}
  if not xml_checks[source.name]['passes']:raise ValueError('Serialized successor CAD invariance failed')
 for name in ['rs05_mass_correction.json','RS05_VARIANTS.md','build_report.json']:
  if (src/name).exists():shutil.copyfile(src/name,out/('previous_'+name))
 if (src/'rs05_mass_correction.json').exists():
  prior=read(src/'rs05_mass_correction.json');current=copy.deepcopy(prior)
  current['previous_source_raw_files_sha256']=copy.deepcopy(prior['source_raw_files_sha256'])
  current['source_raw_files_sha256']={'model.json':sha(out/'model.json'),'hexapod_updated_inspection.urdf':sha(out/'urdf/hexapod_updated_inspection.urdf')}
  current['source_raw_files_sha256_semantics']='Corresponding raw assets for this coordinate revision; correction calculations and old source identities retained under previous_source_raw_files_sha256 and previous_correction_provenance.'
  current['previous_correction_provenance']={'source_report_sha256':sha(src/'rs05_mass_correction.json'),'source_report_filename':'previous_rs05_mass_correction.json','original_builder_sha256':prior.get('builder_sha256'),'mass_additions_recomputed':False,'all_link_frame_mass_COM_tensor_and_addition_arrays_unchanged':True}
  current['joint_review_revision']=review['revision'];current['historical_correction_checks']=copy.deepcopy(prior.get('checks',{}))
  current['checks']={'same_mass_additions_and_link_inertials':all(v['all_parts_and_link_inertials_unchanged'] and v['motor_mass_additions_unchanged'] for v in variant_checks.values()),'both_CAD_pose_tensor_roundtrips_pass':all(v['passes'] for v in variant_checks.values()),'serialized_rebased_URDFs_pass':all(v['passes'] for v in xml_checks.values())}
  current['current_successor_builder_sha256']=sha(Path(__file__));write(out/'rs05_mass_correction.json',current)
  (out/'RS05_VARIANTS.md').write_text('# RS05 mass variants — reviewed joint-travel successor\n\nThe raw CAD and nominal RS05 mass lineages retain exactly the preceding per-link mass, COM, inertia and 18 housing additions. Raw mass is 5.147603654203134 kg; nominal-motor-mass total is 7.466088235225788 kg. No new mass allocation was introduced.\n\nAll three URDF variants now use the current joint review: femur [-120, +80] degrees and tibia [-5, +180] degrees about unchanged pitch zeros; coxa zero and bounds use each reviewed standoff interval midpoint. These are travel limits, not a collision-free motion envelope. See BUILD_REPORT.md and joint_review.json for current coordinate definitions.\n\nThe RS05 URDF variants retain sourced 5.5 N·m peak effort and 50.26548245743669 rad/s maximum speed; the inspection-only file retains disabled physical actuation. Runtime torque-speed, continuous-duty and hardware protection requirements remain.\n\nrs05_mass_correction.json identifies current corresponding raw-file hashes and separately preserves the original correction calculation hashes. Its full per-link before/after tables and additions are unchanged. previous_RS05_VARIANTS.md is frozen prior provenance, including its old narrow envelope; it is not current travel documentation. No Isaac admission is claimed.\n')
 report={'revision':review['revision'],'status':'reviewed joint travel successor; geometry/mass preserved; full motion combinations unqualified',
  'review_sha256':sha(a.joint_review),'current_cad_pose_sha256':sha(out/'cad_pose.json'),'current_model_files_sha256':{n:sha(out/n) for n in variants},'builder_sha256':sha(Path(__file__)),'source_model_files_sha256':{n:sha(src/n) for n in variants},
  'source_model_dir':str(src),'counts':{'parts':len(raw['parts']),'links':len(raw['links']),'joints':len(raw['joints']),'meshes':len(mapping)},
  'variant_CAD_invariance':variant_checks,'serialized_URDF_checks':xml_checks,'pitch_zeros_changed':False,
  'joint_changes':changes,'yaw_absolute_metadata':{j['name']:{k:v for k,v in j.items() if k.startswith('absolute_') or k in ['lower','upper','zero_shift_rad','limit_evidence']} for j in raw['joints'] if j['name'].endswith('_coxa_yaw')},
  'user_travel_limits_about_unchanged_pitch_zero_deg':{'femur':[-120,80],'tibia':[-5,180]},
  'root_height_m':root_height,'lowest_default_part':lowest,'default_parts_below_ground':[p for p in minima if p['ground_clearance_m']<0],
  'previous133pose_collision_report_applicability':'historical narrow envelope only; not a check of these revised bounds',
  'coxa_geometry_scope':'coxa-group vs standoff-plate intrinsic interval;0.5mm endpoint clearance. Femur/tibia/body combinations excluded.',
  'bounds_are_collision_free':False,'isaac_admitted':False,'frozen_previous_evidence_modified':False}
 write(out/'build_report.json',report);write(out/'joint_review_report.json',report)
 lines=['# Reviewed joint travel and coxa neutral successor','',
  f'Revision: `{review["revision"]}`. All **{len(raw["parts"])} parts, {len(raw["links"])} bodies, {len(raw["joints"])} joints**, mesh bytes and exact link mass/COM/inertia records are preserved. The prior evidence bundle remains frozen.','',
  '## User-specified pitch travel','',
  'Femur travel is **−120° to +80°** and tibia travel is **−5° to +180°**, both relative to the **unchanged previous viewer pitch zero**. No pitch origin, CAD coordinate, motor sign or link frame was changed. These are user-confirmed travel limits, **not a collision-free envelope**; some combinations intersect other robot parts or the ground.','',
  '## Coxa coordinate rebase','',
  'Each coxa zero is the midpoint of its buffered intrinsic coxa-versus-standoff interval. Endpoints use the independent geometry review’s **0.5 mm separation buffer**. This check excludes femur/tibia geometry and does not certify full-robot motion. Body azimuth uses +X(left)=0°, +Y=90°, forward(−Y)=−90° and positive counterclockwise about +Z. Absolute endpoints below form continuous, unwrapped intervals.','',
  '| Joint | Previous-zero shift (°) | Absolute neutral (°) | Absolute bounds (°) | New offsets (°) |','|---|---:|---:|---:|---:|']
 for j in raw['joints']:
  if j['name'].endswith('_coxa_yaw'):lines.append(f'| {j["name"]} | {math.degrees(j["zero_shift_rad"]):.6f} | {j["absolute_zero_azimuth_deg"]:.6f} | {j["absolute_lower_azimuth_deg"]:.6f} to {j["absolute_upper_azimuth_deg"]:.6f} | {math.degrees(j["lower"]):.6f} to {math.degrees(j["upper"]):.6f} |')
 lines+=['','For each yaw joint, `J_new = J_old Rz(shift)` and `q_CAD,new = q_CAD,old − shift`; reported yaw limits subtract that same shift. Link frames remain the original CAD anchor frames, so part placements and link inertias remain unchanged. Hardware encoder zeros remain uncalibrated.','',
  '## Verification','',
  f'Raw-model maximum CAD part-transform change: **{checks["all_part_pose_max_abs_error"]:.3g}**. All serialized URDF variants independently reconstruct every part below 1e−12 maximum matrix-entry change. Both raw and nominal-motor-mass-corrected aggregate mass, COM and full inertia remain invariant at the source CAD pose.','',
  f'The revised default root height is **{root_height:.12f} m**, calculated from the actual tibia mesh extrema. The lowest default part has **{lowest["ground_clearance_m"]*1000:.6f} mm** ground clearance. This is a default-pose geometry check, not planted-foot support or a wide-travel contact test.','',
  '**The previous 133-pose collision result is historical and applies only to the earlier narrow inspection envelope.** It is not reused as validation of this wider user travel. New dynamics preparation must use this revision and retain its distinct checks. No Isaac admission or physical-stop calibration is claimed.','',
  '## Reproduce','', '```sh','python apply_joint_review.py --source-model-dir /path/to/previous/model --joint-review /path/to/joint_review.json --out /path/to/successor','```','',
  '`joint_review.json` records provenance; `joint_review_report.json` records source hashes, all coordinate changes and invariance checks. `previous_*` files preserve historical inputs and must not be read as current travel qualification.']
 (out/'BUILD_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'revision':review['revision'],'passes':all(c['passes'] for c in variant_checks.values()) and all(c['passes'] for c in xml_checks.values()),'root_height_m':root_height,'max_raw_CAD_pose_error':checks['all_part_pose_max_abs_error'],'outputs':str(out)},indent=2))

if __name__=='__main__':main()
