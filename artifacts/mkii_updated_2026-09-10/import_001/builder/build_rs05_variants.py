#!/usr/bin/env python3
"""Create sourced RS05 dynamics candidates without modifying raw CAD evidence.

Adds missing nominal motor mass only to each identified housing's inferred link.
Mass allocation uses a CAD housing-bounds cylinder and is explicitly provisional.
The preserved raw-CAD/inspection assets remain byte-identical.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from build_updated_urdf import inertia_sum, origin_xml, eig_check


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pose(p):
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix();T[:3,3]=p['xyz'];return T

def dump_xml(root,path):ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def fk(model):
    out={'body':np.eye(4)}
    for j in model['joints']:
        R=np.eye(4);R[:3,:3]=Rotation.from_rotvec([0,0,j['cad_value']]).as_matrix()
        out[j['child']]=out[j['parent']]@pose(j)@R
    return out

def sum_in_cad(model):
    frames=fk(model);parts=[]
    for link in model['links']:
        T=frames[link['name']];R=T[:3,:3]
        parts.append({'mass':link['mass'],'com':(R@np.asarray(link['com'])+T[:3,3]).tolist(),'inertia':(R@np.asarray(link['inertia'])@R.T).tolist()})
    m,c,I=inertia_sum(parts);return {'mass_kg':m,'com_export_m':c.tolist(),'inertia_about_com_export_kg_m2':I.tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--mass-audit',type=Path,required=True);ap.add_argument('--part-types',type=Path,required=True);ap.add_argument('--spec-review',type=Path,required=True)
    a=ap.parse_args();out=a.raw_dir;raw_json=out/'model.json';raw_urdf=out/'urdf/hexapod_updated_inspection.urdf';before={p.name:sha(p) for p in [raw_json,raw_urdf]}
    raw=read(raw_json);audit=read(a.mass_audit);types=read(a.part_types);motor=audit['motor_audit'];mapping={r['mesh']:r['source_mesh'] for r in read(out/'mesh_mapping.json')}
    anchors=motor['anchor_part_indices'];nominal=.191;count=len(anchors)
    if count!=18 or motor['anchor_unique_com_count_1um']!=18 or not motor['all_vendor_type_counts_divisible_by_motor_count']:raise ValueError('18 repeated motor sets not established')
    delta=nominal-motor['vendor_mass_total_kg']/count
    if delta<=0:raise ValueError('No positive missing motor mass')
    housing=next(g for g in types if g['mesh']==motor['anchor_mesh']);bounds=np.asarray(housing['mesh_geometry']['bounds_mesh_frame_m']);center=bounds.mean(axis=0)
    radius=.25*((bounds[1,0]-bounds[0,0])+(bounds[1,1]-bounds[0,1]));height=bounds[1,2]-bounds[0,2]
    cylinder=np.diag([delta*(3*radius**2+height**2)/12,delta*(3*radius**2+height**2)/12,delta*radius**2/2])
    corrected=copy.deepcopy(raw);corrected['name']='Updated detailed CAD — RS05 nominal-mass candidate';corrected['mass_basis']='All exported CAD parts unchanged, plus 18 provisional housing cylinder additions to reach 191g nominal motor mass each'
    additions=[];bylink={l['name']:[] for l in raw['links']}
    parts={p['id']:p for p in raw['parts']}
    for idx in anchors:
        p=parts[idx]
        if mapping[p['mesh']]!=motor['anchor_mesh']:raise ValueError('Motor housing mapping mismatch')
        T=pose(p);R=T[:3,:3];com=R@center+T[:3,3];I=R@cylinder@R.T
        addition={'id':f'rs05_nominal_mass_addition_{idx}','housing_source_part_id':idx,'link':p['link'],'mass_kg':delta,'com_link_m':com.tolist(),'inertia_about_com_link_kg_m2':I.tolist(),
            'housing_mesh':p['mesh'],'housing_center_mesh_m':center.tolist(),'cylinder_radius_m':float(radius),'cylinder_height_m':float(height),
            'status':'provisional missing-mass spatial allocation; not measured motor COM, rotor inertia or calibrated housing/output split'}
        additions.append(addition);bylink[p['link']].append({'mass':delta,'com':com.tolist(),'inertia':I.tolist()})
    perlink=[]
    for link in corrected['links']:
        original=next(l for l in raw['links'] if l['name']==link['name']);m,c,I=inertia_sum([original]+bylink[link['name']]);link.update({'mass':m,'com':c.tolist(),'inertia':I.tolist(),'tensor_physical':eig_check(I)})
        perlink.append({'link':link['name'],'raw_cad_mass_kg':original['mass'],'added_mass_kg':math.fsum(x['mass'] for x in bylink[link['name']]),'rs05_corrected_mass_kg':m,
            'raw_cad_com_link_m':original['com'],'corrected_com_link_m':c.tolist(),'raw_cad_inertia_about_com_link_kg_m2':original['inertia'],'corrected_inertia_about_com_link_kg_m2':I.tolist(),
            'motor_housing_additions':len(bylink[link['name']])})
    actuator={'model':'RobStride RS05 (repository established actuator identity)','nominal_motor_mass_kg':nominal,'urdf_peak_effort_nm':5.5,'urdf_max_velocity_rad_s':16*math.pi,
        'visualization_sweep_velocity_rad_s':.5,'source':'docs/RS05_SPEC_REVIEW.md','source_sha256':sha(a.spec_review),
        'warning':'URDF peak torque and maximum speed are not simultaneously available operating points. Runtime must separately enforce torque-speed limits, continuous duty, current/thermal/voltage derating, hardware protection and identified actuator dynamics.'}
    corrected['actuator_specification']=actuator;corrected['mass_additions']=additions;corrected['raw_cad_total_mass_kg']=audit['aggregate_source_cad']['mass_kg'];corrected['total_mass_kg']=sum(l['mass'] for l in corrected['links'])
    corrected['warnings']=[w for w in corrected['warnings'] if 'underweight vendor motors' not in w]
    corrected['warnings']+=['191g per-motor nominal mass restored with provisional housing cylinder allocation. Real motor COM, rotor/output inertia and housing ownership remain uncalibrated.','Peak effort/speed come from established RS05 spec; runtime torque-speed and continuous-duty enforcement remains required.']
    corrected['isaac_admitted']=False
    raw_spec_root=ET.parse(raw_urdf).getroot();raw_spec_root.set('name','hexapod_updated_rs05_raw_cad')
    raw_spec_root.insert(0,ET.Comment('RS05 peak specification variant; raw CAD masses unchanged and underweight. 5.5Nm and 16*pi rad/s are separate envelope limits; enforce actual torque-speed and duty limits in runtime. Collision geometry remains unprepared.'))
    for j in raw_spec_root.findall('joint'):
        j.find('limit').set('effort','5.5');j.find('limit').set('velocity',format(16*math.pi,'.17g'))
    # The inherited inspection comment is retained as provenance, while this
    # variant explicitly provides sourced actuator limits and its own warning.
    raw_spec=out/'urdf/hexapod_updated_rs05_raw_cad.urdf';dump_xml(raw_spec_root,raw_spec)
    corrected_root=copy.deepcopy(raw_spec_root);corrected_root.set('name','hexapod_updated_rs05_mass_corrected')
    corrected_root.insert(0,ET.Comment('NOMINAL MOTOR MASS CANDIDATE. Source CAD mass properties unchanged; missing mass added at 18 identified housing centers as CAD-sized solid cylinders. Allocation is provisional; no physical calibration or Isaac admission.'))
    for el in corrected_root.findall('link'):
        link=next(l for l in corrected['links'] if l['name']==el.get('name'));old=el.find('inertial');pos=list(el).index(old);el.remove(old);ine=ET.Element('inertial');el.insert(pos,ine)
        T=np.eye(4);T[:3,3]=link['com'];origin_xml(ine,T);ET.SubElement(ine,'mass',value=format(link['mass'],'.17g'));I=np.asarray(link['inertia'])
        ET.SubElement(ine,'inertia',**{k:format(I[i,j],'.17g') for k,(i,j) in {'ixx':(0,0),'ixy':(0,1),'ixz':(0,2),'iyy':(1,1),'iyz':(1,2),'izz':(2,2)}.items()})
    corrected_path=out/'urdf/hexapod_updated_rs05_mass_corrected.urdf';dump_xml(corrected_root,corrected_path)
    rawsum=sum_in_cad(raw);correctedsum=sum_in_cad(corrected)
    invariants={'raw_json_and_urdf_preserved':before=={p.name:sha(p) for p in [raw_json,raw_urdf]},'all_source_part_records_unchanged':raw['parts']==corrected['parts'],
        'joints_and_visual_geometry_unchanged':raw['joints']==corrected['joints'], '18_unique_housing_additions':len({a['housing_source_part_id'] for a in additions})==18,
        'all_corrected_link_tensors_physically_consistent':all(l['tensor_physical'] for l in corrected['links']),
        'expected_total_mass_matches':abs(correctedsum['mass_kg']-(rawsum['mass_kg']+count*delta))<1e-12,
        'total_vendor_plus_added_mass_equals18nominal':abs(motor['vendor_mass_total_kg']+sum(x['mass_kg'] for x in additions)-18*nominal)<1e-12}
    report={'status':'RS05 nominal mass and sourced actuator envelope candidate; spatial mass allocation provisional', 'source_raw_files_sha256':before,
        'builder_sha256':sha(Path(__file__)),'motor_count':count,'raw_vendor_mass_per_motor_kg':motor['vendor_mass_total_kg']/count,'nominal_mass_per_motor_kg':nominal,
        'missing_mass_per_motor_kg':delta,'total_added_mass_kg':count*delta,'housing_geometry':{'bounds_mesh_m':bounds.tolist(),'center_mesh_m':center.tolist(),'radius_m':float(radius),'height_m':float(height),'mesh_sha256':housing['mesh_geometry']['sha256']},
        'raw_cad_total':rawsum,'corrected_total':correctedsum,'actuator_specification':actuator,'per_link_before_after':perlink,'additions':additions,'checks':invariants,
        'mass_allocation_physically_calibrated':False,'isaac_admitted':False,'collision_geometry':'unprepared source per-part meshes; collision preparation and physics admission remain separate',
        'confidence_note':'Motor count and housing dimensions are measured from this CAD export. 191 g mass and peak actuator limits are sourced from established repository/manufacturer RS05 evidence. Missing mass center and solid cylinder inertia allocation are explicitly approximate; original CAD part masses and all unrelated material densities are unchanged.'}
    write(out/'model_rs05_mass_corrected.json',corrected);write(out/'rs05_mass_correction.json',report)
    lines=['# RS05 dynamics variants for the updated direct-drive robot','',
        '**Raw CAD stays unchanged.** Three URDFs are provided:', '',
        '- `hexapod_updated_inspection.urdf`: raw CAD mass ledger and finite inspection bounds; physical actuation disabled in this visualization-only version.',
        '- `hexapod_updated_rs05_raw_cad.urdf`: identical raw mass/geometry, with sourced RS05 peak torque and speed fields. The motors remain underweight.',
        '- `hexapod_updated_rs05_mass_corrected.urdf`: RS05 peak torque/speed plus missing nominal motor mass restored on the corresponding housing links. This is the dynamics candidate; missing-mass spatial allocation remains provisional.', '',
        f'All variants retain the same 1,753 meshes, 19 rigid bodies, 18 joints and tightened finite knee inspection interval [−3, +20] degrees. Raw mass **{rawsum["mass_kg"]:.12f} kg**; corrected mass **{correctedsum["mass_kg"]:.12f} kg**.', '',
        '## Motor correction','',
        f'The 18 unique housing instances and repeated vendor component counts establish 18 CAD motor sets, each contributing {1000*motor["vendor_mass_total_kg"]/count:.9f}g. The established RS05 nominal is 191 g. Each housing link receives **{1000*delta:.9f}g**, totaling **{count*delta:.12f} kg**.', '',
        f'The added mass is centered at the actual housing mesh bounding-box center {center.tolist()} m in its mesh frame. Its approximate solid cylinder inertia uses **{radius*2000:.6f} mm diameter ×{height*1000:.6f} mm height**, measured from this export. This reproduces the existing repository mass top-up method, without reusing old robot total masses or changing unrelated material density.', '',
        '**This is not measured rotor/stator mass distribution.** Motor COM, internal rotor inertia, output-side inertia, housing ownership and real manufactured mass remain calibration tasks. All original exported part masses, COMs, full tensors and meshes are unchanged; 18 separate additions are recorded in the model and correction JSON.', '',
        '## Actuator fields','',
        f'Sourced from `docs/RS05_SPEC_REVIEW.md`: **5.5 N·m peak effort** and **{16*math.pi:.12f} rad/s (480rpm) maximum speed**. They describe separate envelope limits, not a simultaneous operating point. Runtime must enforce torque-speed behavior, continuous duty, thermal/current/voltage protections and identified actuator dynamics. Viewer sweep speed remains 0.5 rad/s and has no hardware-limit meaning.', '',
        '## Per-link mass before and after','', '| Link | Raw CAD (kg) | Addition (kg) | Corrected (kg) | Motors |','|---|---:|---:|---:|---:|']
    for row in perlink:lines.append(f'|{row["link"]}|{row["raw_cad_mass_kg"]:.9f}|{row["added_mass_kg"]:.9f}|{row["rs05_corrected_mass_kg"]:.9f}|{row["motor_housing_additions"]}|')
    lines+=['','Full before/after COM and inertia matrices, addition centers/tensors, source hashes and all checks are in `rs05_mass_correction.json`. Every corrected link tensor remains positive definite and physically consistent. Raw files were verified byte-identical after variant creation.','',
        '**Isaac qualification remains pending:** collision meshes need preparation; source-export topology was absent and per-part ownership remains geometrically inferred; simultaneous clearance, hardware stops, reset/contact stability and bounded physics probes must pass. Sourced peak fields plus corrected total mass do not by themselves establish sim-to-real readiness.','']
    (out/'RS05_VARIANTS.md').write_text('\n'.join(lines))
    print(json.dumps({'checks':invariants,'raw_kg':rawsum['mass_kg'],'corrected_kg':correctedsum['mass_kg'],'delta_each_kg':delta,'radius_m':float(radius),'height_m':float(height)},indent=2))
    return 0 if all(invariants.values()) else 1

if __name__=='__main__':raise SystemExit(main())
