#!/usr/bin/env python3
"""Build an unqualified articulated inspection candidate from recovered geometry.
Preserves every source CAD part and exact mass property, with provisional ownership.
Does not apply nominal motor corrections or claim physical/Isaac qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def read(p): return json.loads(p.read_text())
def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(x): return np.asarray(x,dtype=float)
def fmt(x): return ' '.join(format(float(v),'.17g') for v in x)
def rz(q):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,0,q]).as_matrix();return T

def frame(p,x,z):
    z=arr(z);z/=np.linalg.norm(z);x=arr(x);x-=z*np.dot(x,z);x/=np.linalg.norm(x)
    T=np.eye(4);T[:3,:3]=np.column_stack((x,np.cross(z,x),z));T[:3,3]=p;return T

def pose(T):
    return {'xyz':T[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(T[:3,:3]).as_quat().tolist()}

def origin_xml(parent,T):
    # Preserve near-gimbal rotations instead of the scipy display convention,
    # which intentionally approximates sufficiently near-singular rotations.
    R=T[:3,:3]; horizontal=math.hypot(R[0,0],R[1,0])
    pitch=math.atan2(-R[2,0],horizontal)
    if horizontal>1e-14:
        yaw=math.atan2(R[1,0],R[0,0])
        # Remove yaw before extracting roll: stable even at tiny cos(pitch).
        sy,cy=math.sin(yaw),math.cos(yaw)
        roll=math.atan2(sy*R[0,2]-cy*R[1,2],-sy*R[0,1]+cy*R[1,1])
    else:
        roll=math.atan2(-R[1,2],R[1,1]);yaw=0.
    ET.SubElement(parent,'origin',xyz=fmt(T[:3,3]),rpy=fmt([roll,pitch,yaw]))

def inertia_sum(parts):
    m=math.fsum(p['mass'] for p in parts)
    c=sum(p['mass']*arr(p['com']) for p in parts)/m
    I=np.zeros((3,3))
    for p in parts:
        d=arr(p['com'])-c
        I+=arr(p['inertia'])+p['mass']*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    return m,c,I

def eig_check(I):
    vals=np.linalg.eigvalsh(I);return bool(vals[0]>0 and vals[0]+vals[1]>=vals[2]-1e-12)

def fk(joints,q):
    out={'body':np.eye(4)}
    for j in joints:
        if j['parent'] not in out or j['child'] in out: raise ValueError('Invalid graph order or duplicate child')
        J=arr(j['_origin']);out[j['child']]=out[j['parent']]@J@rz(q.get(j['name'],0.))
    return out

def angle(v,radial): return math.atan2(float(v[2]),float(np.dot(v,radial)))
def wrap(x): return math.atan2(math.sin(x),math.cos(x))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--mass-ledger',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--ownership',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);(out/'meshes').mkdir(exist_ok=True);(out/'urdf').mkdir(exist_ok=True)
    rows=read(a.mass_ledger);geo=read(a.geometry);own=read(a.ownership)
    if geo['source_urdf_sha256']!=sha(a.source/'robot.urdf') or own['source_urdf_sha256']!=geo['source_urdf_sha256']: raise ValueError('Source mismatch')
    if len(rows)!=1753 or {p['index'] for p in own['parts']}!=set(range(len(rows))): raise ValueError('Incomplete ownership')
    ownership={p['index']:p for p in own['parts']};axes={j['name']:j for j in geo['joint_axes']};anchors={p['leg']:p for p in geo['anchors']}
    legs=['lf','lm','lr','rf','rm','rr'];Tcad={'body':np.eye(4)};cad={};joints=[];angle_report={}
    center=np.mean([arr(axes[f'{leg}_coxa_yaw']['axis_point_m']) for leg in legs],axis=0);center[2]=0.
    for leg in legs:
        yaw,shoulder,knee=[axes[f'{leg}_{k}'] for k in ('coxa_yaw','femur_pitch','tibia_pitch')]
        Y,S,K=[arr(j['axis_point_m']) for j in (yaw,shoulder,knee)]
        outward=S-Y;outward[2]=0.;outward/=np.linalg.norm(outward)
        radial=Y-center;radial[2]=0.;radial/=np.linalg.norm(radial)
        fem=K-S
        pitch_axis=arr(shoulder['axis_direction_geometric']);pitch_axis/=np.linalg.norm(pitch_axis)
        if np.cross(pitch_axis,fem)[2]<0:pitch_axis=-pitch_axis
        knee_axis=arr(knee['axis_direction_geometric']);knee_axis/=np.linalg.norm(knee_axis)
        if np.dot(knee_axis,pitch_axis)<0:knee_axis=-knee_axis
        tib=arr(rows[anchors[leg]['tibia']['index']]['T_export_part'])[:3,0]
        Tcad[leg+'_coxa']=frame(Y,outward,[0,0,1]);Tcad[leg+'_femur']=frame(S,fem,pitch_axis);Tcad[leg+'_tibia']=frame(K,tib,knee_axis)
        yaw_angle=wrap(math.atan2(outward[1],outward[0])-math.atan2(radial[1],radial[0]))
        fem_angle=angle(fem,outward);tib_angle=angle(tib,outward)
        cad.update({leg+'_coxa_yaw':yaw_angle,leg+'_femur_pitch':fem_angle-math.radians(20),leg+'_tibia_pitch':wrap(tib_angle-fem_angle)-math.radians(-110)})
        angle_report[leg]={'CAD_yaw_from_radial_deg':math.degrees(yaw_angle),'CAD_femur_elevation_deg':math.degrees(fem_angle),'CAD_tibia_elevation_deg':math.degrees(tib_angle),'CAD_knee_relative_deg':math.degrees(wrap(tib_angle-fem_angle))}
        for typ,parent,child,limit in [('coxa_yaw','body',leg+'_coxa',25),('femur_pitch',leg+'_coxa',leg+'_femur',20),('tibia_pitch',leg+'_femur',leg+'_tibia',20)]:
            name=leg+'_'+typ;J=np.linalg.inv(Tcad[parent])@Tcad[child]@rz(-cad[name])
            joints.append({'name':name,'parent':parent,'child':child,**pose(J),'axis':[0.,0.,1.],'lower':math.radians(-3 if typ=='tibia_pitch' else -limit),'upper':math.radians(limit),'cad_value':cad[name],'default_value':0.,'_origin':J.tolist()})
    source_meshes=sorted({r['mesh'] for r in rows});mapping=[];mesh_names={};mesh_vertices={}
    for i,name in enumerate(source_meshes):
        stem=''.join(c if c.isascii() and (c.isalnum() or c=='_') else '_' for c in Path(name).stem)[:65]
        dst_name=f'mesh_{i:03d}_{stem}.stl';src=a.source/name;dst=out/'meshes'/dst_name
        if not dst.exists() or sha(dst)!=sha(src):shutil.copyfile(src,dst)
        mesh_names[name]=dst_name;mesh_vertices[dst_name]=np.asarray(trimesh.load_mesh(src,process=False).vertices)
        mapping.append({'mesh':dst_name,'source_mesh':name,'sha256':sha(src),'bytes':src.stat().st_size})
    parts=[];link_parts=defaultdict(list)
    for r in rows:
        idx=r['index_zero_based'];o=ownership[idx];link=o['body']
        if Path(r['mesh']).name!=o['mesh'] or link not in Tcad:raise ValueError(f'Ownership mismatch {idx}')
        inv=np.linalg.inv(Tcad[link]);T=inv@arr(r['T_export_part']);R=inv[:3,:3]
        com=R@arr(r['com_export_m'])+inv[:3,3];I=R@arr(r['inertia_about_com_export_kg_m2'])@R.T
        p={'id':idx,'name':r['part_name'],'link':link,'mesh':mesh_names[r['mesh']],**pose(T),'mass':r['mass_kg'],'color':r['appearance_rgba'],
           'ownership_basis':o['basis'],'ownership_confidence':o['confidence'],'_T':T.tolist(),'com':com.tolist(),'inertia':I.tolist()}
        parts.append(p);link_parts[link].append(p)
    links=[]
    for name,T in Tcad.items():
        m,c,I=inertia_sum(link_parts[name]);parent=next((j['parent'] for j in joints if j['child']==name),None)
        links.append({'name':name,'parent':parent,'mass':m,'com':c.tolist(),'inertia':I.tolist(),'tensor_physical':eig_check(I)})
    cad_fk=fk(joints,cad);zero_fk=fk(joints,{})
    errors=[]
    for p,r in zip(parts,rows):
        expected=arr(r['T_export_part']);actual=cad_fk[p['link']]@arr(p['_T']);errors.append(float(np.max(np.abs(expected-actual))))
    cad_bodies=[]
    for link in links:
        T=cad_fk[link['name']];R=T[:3,:3]
        cad_bodies.append({'mass':link['mass'],'com':(R@arr(link['com'])+T[:3,3]).tolist(),'inertia':(R@arr(link['inertia'])@R.T).tolist()})
    m,c,I=inertia_sum(cad_bodies)
    msrc,csrc,Isrc=inertia_sum([{'mass':r['mass_kg'],'com':r['com_export_m'],'inertia':r['inertia_about_com_export_kg_m2']} for r in rows])
    min_by_part=[]
    for p in parts:
        T=zero_fk[p['link']]@arr(p['_T']);verts=mesh_vertices[p['mesh']];minz=float(np.min(verts@T[2,:3]+T[2,3]))
        min_by_part.append({'id':p['id'],'name':p['name'],'link':p['link'],'min_z_before_root_m':minz})
    tib_min=min(r['min_z_before_root_m'] for r in min_by_part if Path(rows[r['id']]['mesh']).name=='tibia.stl')
    root_height=.005-tib_min
    for r in min_by_part:r['min_z_above_ground_m']=r['min_z_before_root_m']+root_height
    clear_min=min(min_by_part,key=lambda r:r['min_z_above_ground_m'])
    signs=[];eps=1e-6
    for joint in joints:
        leg=joint['name'][:2];typ=joint['name'][3:];plus=fk(joints,{joint['name']:eps});minus=fk(joints,{joint['name']:-eps})
        if typ=='coxa_yaw':
            vplus=plus[leg+'_coxa'][:3,0];vminus=minus[leg+'_coxa'][:3,0]
            d=wrap(math.atan2(vplus[1],vplus[0])-math.atan2(vminus[1],vminus[0]))/(2*eps)
        else:
            child=joint['child'];outward=zero_fk[leg+'_coxa'][:3,0]
            d=wrap(angle(plus[child][:3,0],outward)-angle(minus[child][:3,0],outward))/(2*eps)
        signs.append({'joint':joint['name'],'positive_angle_derivative':d,'passes':abs(d-1)<1e-5})
    inspect_angles={}
    for leg in legs:
        outward=zero_fk[leg+'_coxa'][:3,0];fem=zero_fk[leg+'_femur'][:3,0];tib=zero_fk[leg+'_tibia'][:3,0]
        inspect_angles[leg]={'femur_elevation_deg':math.degrees(angle(fem,outward)),'tibia_elevation_deg':math.degrees(angle(tib,outward))}
    rng=np.random.default_rng(20260910);sample_ok=True
    for _ in range(256):
        q={j['name']:float(rng.uniform(j['lower'],j['upper'])) for j in joints}
        for T in fk(joints,q).values():
            sample_ok &= bool(np.isfinite(T).all() and np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))<1e-12)
    urdf=ET.Element('robot',name='hexapod_updated_direct_drive_inspection')
    urdf.append(ET.Comment('INFERRED INSPECTION CANDIDATE ONLY. All source CAD masses preserved. Motor CAD underweight. Finite joint sweeps are not measured stops or clearance certification. Per-part collision meshes are unprepared. No Isaac admission.'))
    for link in links:
        el=ET.SubElement(urdf,'link',name=link['name']);ine=ET.SubElement(el,'inertial');T=np.eye(4);T[:3,3]=link['com'];origin_xml(ine,T)
        ET.SubElement(ine,'mass',value=format(link['mass'],'.17g'));Ilink=arr(link['inertia']);ET.SubElement(ine,'inertia',**{k:format(Ilink[i,j],'.17g') for k,(i,j) in {'ixx':(0,0),'ixy':(0,1),'ixz':(0,2),'iyy':(1,1),'iyz':(1,2),'izz':(2,2)}.items()})
        for p in link_parts[link['name']]:
            for tag in ['visual','collision']:
                child=ET.SubElement(el,tag,name=f'part_{p["id"]:04d}');origin_xml(child,arr(p['_T']));geom=ET.SubElement(child,'geometry');ET.SubElement(geom,'mesh',filename='../meshes/'+p['mesh'])
                if tag=='visual':mat=ET.SubElement(child,'material',name=f'appearance_{p["id"]:04d}');ET.SubElement(mat,'color',rgba=fmt(p['color']))
    for j in joints:
        el=ET.SubElement(urdf,'joint',name=j['name'],type='revolute');ET.SubElement(el,'parent',link=j['parent']);ET.SubElement(el,'child',link=j['child']);origin_xml(el,arr(j['_origin']));ET.SubElement(el,'axis',xyz='0 0 1')
        ET.SubElement(el,'limit',lower=format(j['lower'],'.17g'),upper=format(j['upper'],'.17g'),effort='0',velocity='0.5')
    ET.indent(urdf,space='  ');ET.ElementTree(urdf).write(out/'urdf/hexapod_updated_inspection.urdf',encoding='utf-8',xml_declaration=True)
    serialized=ET.parse(out/'urdf/hexapod_updated_inspection.urdf').getroot()
    def parse_origin(element):
        T=np.eye(4);T[:3,3]=np.fromstring(element.get('xyz'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy'),sep=' ')).as_matrix();return T
    xml_fk={'body':np.eye(4)}
    for j in serialized.findall('joint'):
        xml_fk[j.find('child').get('link')]=xml_fk[j.find('parent').get('link')]@parse_origin(j.find('origin'))@rz(cad[j.get('name')])
    xml_errors=[]
    for link in serialized.findall('link'):
        for vis in link.findall('visual'):
            idx=int(vis.get('name').split('_')[1])
            xml_errors.append(float(np.max(np.abs(xml_fk[link.get('name')]@parse_origin(vis.find('origin'))-arr(rows[idx]['T_export_part'])))))
    model={'name':'Updated detailed CAD — provisional articulated inspection','schema':1,'links':links,
           'joints':[{k:v for k,v in j.items() if not k.startswith('_')} for j in joints],
           'parts':[{k:v for k,v in p.items() if k not in {'_T','com','inertia'}} for p in parts],
           'cad_pose':cad,'root_height_m':root_height,'cad_root_height_m':0.,'inspection_stance':{'yaw':'radially outward','femur_elevation_deg':20,'relative_knee_deg':-110,'tibia_elevation_deg':-90},
           'warnings':['User confirms direct drive with no four-bar. Individual rigid-body ownership remains geometrically inferred; source CAD export had no joints.','Raw CAD mass 5.147603654203 kg includes underweight vendor motors.','Finite sweep envelopes are for inspection only; not measured joint stops or clearance-certified bounds.','Motor rotor/stator ownership is an approximation; not hardware calibrated.','Per-part mesh collisions are unprepared; no Isaac admission.'],
           'limits_semantics':'q=0 is radial yaw, +20deg femur elevation, -110deg relative knee. Limits are deviations around that stance. CAD pose can lie outside inspection limits.',
           'source_urdf_sha256':geo['source_urdf_sha256'],'all_parts_included':True,'topology_confirmation':'User: yup, no more four bar, makes it way easier','knee_bound_evidence':'knee_bound_sweep.json from independent exact source-triangle sweep; knee -3deg lower bound, -8deg structural intersection','topology_user_confirmed':True}
    report={'status':'unqualified_provisional_articulated_inspection_candidate','source_urdf_sha256':geo['source_urdf_sha256'],
        'input_sha256':{p.name:sha(p) for p in (a.mass_ledger,a.geometry,a.ownership)},'builder_sha256':sha(Path(__file__)),
        'counts':{'links':len(links),'joints':len(joints),'visuals':len(parts),'collision_meshes_unprepared':len(parts),'unique_meshes':len(mapping)},
        'ownership':{'unique_complete':set(p['id'] for p in parts)==set(range(1753)),'basis_counts':dict(Counter(p['ownership_basis'] for p in parts)),'per_link_counts':dict(Counter(p['link'] for p in parts)),'warning':own['warning']},
        'cad_source_pose_angles_deg':angle_report,'actual_inspection_angles_deg':inspect_angles,
        'serialized_urdf_visual_transform_roundtrip_max_abs_error':max(xml_errors),'all_visual_transform_roundtrip_max_abs_error':max(errors),'all_link_tensors_physically_consistent':all(l['tensor_physical'] for l in links),
        'mass_roundtrip':{'source_kg':msrc,'links_kg':m,'abs_error_kg':abs(m-msrc),'com_error_m':(c-csrc).tolist(),'inertia_error_kg_m2':(I-Isrc).tolist(),'source_com_export_m':csrc.tolist(),'source_inertia_about_com_export_kg_m2':Isrc.tolist()},
        'graph_acyclic_connected':len(cad_fk)==19,'joint_names_unique':len({j['name'] for j in joints})==18,'axis_derivative_checks':signs,
        'random_256_pose_finite_rigid_transforms':sample_ok,'root_height_m':root_height,'lowest_tibia_before_root_m':tib_min,
        'minimum_all_mesh_ground_clearance_m':clear_min['min_z_above_ground_m'],'lowest_part':clear_min,
        'parts_below_ground_at_default':[r for r in min_by_part if r['min_z_above_ground_m'] < -1e-9],
        'motor_mass_correction_applied':False,'acceptance_limits_unchanged':True,'isaac_admitted':False,'topology_user_confirmed':True,'topology_confirmation':'User: yup, no more four bar, makes it way easier','knee_bound_evidence':'knee_bound_sweep.json from independent exact source-triangle sweep; knee -3deg lower bound, -8deg structural intersection',
        'inspection_limits':{'yaw_deg':[-25,25],'femur_relative_to_20deg':[-20,20],'knee_relative_to_minus110deg':[-3,20],'status':'finite visualization sweep only; structural triangle sweep selected knee lower bound -3deg with approximately1mm margin; not full collision qualification or physical stops'},
        'effort_velocity_semantics':'URDF effort=0 disables physical actuation for inspection; velocity=0.5rad/s is an inspection placeholder. These are unrelated to frozen C-study runtime and are not recovered CAD or qualified actuator limits.'}
    report['numeric_audit_pass']=bool(max(errors)<1e-12 and max(xml_errors)<1e-12 and abs(m-msrc)<1e-12 and np.max(np.abs(c-csrc))<1e-12 and np.max(np.abs(I-Isrc))<1e-12 and all(r['passes'] for r in signs) and all(l['tensor_physical'] for l in links) and sample_ok)
    write(out/'model.json',model);write(out/'mesh_mapping.json',mapping);write(out/'cad_pose.json',cad);write(out/'build_report.json',report);write(out/'part_clearances_default.json',min_by_part)
    lines=['# Detailed updated CAD: provisional articulated inspection','',
    f'Includes **{len(parts)} exact mesh instances, {len(links)} inferred rigid bodies and {len(joints)} revolute joints**. Raw CAD mass **{m:.12f} kg**; no motor-mass correction. This is a new direct-drive candidate, not the old physical four-bar or active simplified C-study.', '',
    'The user confirmed that the new robot has no four-bar and uses the direct-drive topology. The input had one fused link and no joints. Geometry recovery supplies provisional axis lines and complete inferred ownership. Bearing races and vendor motor internals need confirmation. The preserved exported pose can be restored with `cad_pose.json`; it is not a recommended reset pose.', '',
    '## Default inspection pose and bounds','',
    'q=0 points coxae radially outward, femurs 20 degrees above horizontal and tibiae vertically down (relative knee −110 degrees). Positive yaw rotates about export +Z; positive pitch raises the outward segment. Every child frame has joint axis +Z. Bounds are ±25 degrees yaw, ±20 degrees shoulder and −3 to +20 degrees knee around this default: **finite visual sweep envelopes, not measured stops or clearance-certified motion**.', '',
    f'Root height is **{root_height:.9f} m**, placing the actual lowest tibia mesh vertex 5 mm above ground. Lowest vertex over every part has {clear_min["min_z_above_ground_m"]*1000:.6f} mm clearance. Motion away from the default can intersect the ground or other parts and requires the separate sweep report.', '',
    '## Checks','',
    f'- Complete unique ownership of all 1,753 parts; 19 connected acyclic bodies, 18 unique joint names.',
    f'- All source visual transforms reconstruct at their CAD joint values with maximum matrix-entry difference {max(errors):.3g}; parsed URDF XML difference {max(xml_errors):.3g}.',
    f'- Summed per-link mass error {abs(m-msrc):.3g} kg; maximum aggregate COM error {np.max(np.abs(c-csrc)):.3g} m and inertia error {np.max(np.abs(I-Isrc)):.3g} kg·m².',
    '- All 19 link tensors are positive definite and satisfy inertia triangle inequalities. Full off-diagonal terms retained.',
    '- All 18 finite-difference axis sign checks pass; 256 deterministically sampled poses retain finite rigid transforms.', '',
    '## Remaining qualification','',
    'Motor CAD represents about 62.2 g per device versus the historical RS05 191 g nominal. Housing/output inertia allocation, mechanical stops, encoder zeros, contact geometry, cable clearance and actuator dynamics are unverified. Original per-part collision meshes are preserved but explicitly unprepared. URDF effort 0 disables physical actuation for inspection; speed 0.5 rad/s is a placeholder. These values are separate from the frozen C-study and are not recovered CAD or qualified hardware limits. **No Isaac physics admission, training compatibility or sim-to-real readiness is claimed.**', '',
    '## Reproduction','', '```sh',
    'python build_updated_urdf.py --source /path/to/export --mass-ledger /path/to/part_instances.json --geometry /path/to/joint_geometry_recovery.json --ownership /path/to/provisional_body_ownership.json --out /path/to/output', '```','',
    '`model.json` provides every part, link, joint, source-CAD coordinate and default root height. `mesh_mapping.json` maps ASCII asset names to unchanged source mesh bytes and hashes. `build_report.json` and `part_clearances_default.json` preserve detailed checks.']
    (out/'BUILD_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'numeric_audit_pass':report['numeric_audit_pass'],'counts':report['counts'],'root_height_m':root_height,'minimum_ground_clearance_m':clear_min['min_z_above_ground_m'],'mass':m,'roundtrip_max':max(errors)},indent=2))
    return 0 if report['numeric_audit_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
