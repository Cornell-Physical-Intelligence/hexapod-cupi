#!/usr/bin/env python3
"""Prepare deterministic terrain fixtures, mount sweep and explicit readiness gates."""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from experiments.terrain.tools.terrain_readiness import terrain_mesh, validate_mesh, mesh_usda, optical_rotation_body

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/terrain_readiness_2026-09-09')
    p.add_argument('--flat-evaluation',type=Path)
    args=p.parse_args();out=args.out.resolve()
    if out.exists():p.error('Use a fresh output directory; preparation outputs are immutable snapshots')
    out.mkdir(parents=True)
    fixtures=[]
    for split,seeds in [('train',(1103,2207,3301,4409)),('heldout',(7103,8209))]:
        for family in ('smooth_rough','ramp','step','ridge','pit'):
            for seed in seeds:
                ident=f'{split}_{family}_{seed}'
                v,f,meta=terrain_mesh(family,seed)
                checks=validate_mesh(v,f)
                path=out/'terrains'/f'{ident}.usda';path.parent.mkdir(exist_ok=True)
                path.write_text(mesh_usda(v,f))
                np.savez_compressed(path.with_suffix('.npz'),vertices=v,faces=f)
                fixtures.append(dict(id=ident,split=split,**meta,**checks,
                                     usda=str(path.relative_to(out)),sha256=sha(path)))
    save(out/'terrain_catalog.json',{'version':1,'status':'cpu_geometry_checked_isaac_import_pending',
         'train_seeds':[1103,2207,3301,4409],'heldout_seeds':[7103,8209],
         'heldout_use':'regression fixtures only; final qualification requires unseen layouts/sites',
         'import_requirements':['No global ground plane under pits','No convex hull approximation',
             'Assign measured friction/contact parameters','Validate sensor rays and physics contacts agree',
             'No extra fixture objects in standalone flat admission','Validate start stance for actual robot'],
         'fixtures':fixtures})

    urdf=ROOT/'robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf'
    xml=ET.parse(urdf).getroot()
    masses={x.attrib['name']:float(x.find('inertial/mass').attrib['value']) for x in xml.findall('link')}
    candidates=[]
    for leg in ('lf','lm','lr','rf','rm','rr'):
        joint=next(j for j in xml.findall('joint') if j.attrib['name']==f'{leg}_coxa_yaw')
        pivot=np.fromstring(joint.find('origin').attrib['xyz'],sep=' ')
        for model in ('D405','D435'):
            for z in (.04,.07,.10):
                for pitch in (35,45,60):
                    candidates.append({'id':f'{leg}_{model}_z{round(z*1000)}_p{pitch}',
                        'leg_sector':leg,'model':model,
                        'body_position_m':[float(pivot[0]),float(pivot[1]),z],
                        'optical_rotation_body':optical_rotation_body(pivot[:2],pitch).tolist(),
                        'downward_pitch_deg':pitch,'cad_fit_checked':False,'moving_leg_occlusion_checked':False})
    save(out/'mount_candidates.json',{'version':1,'source_urdf':str(urdf.relative_to(ROOT)),
         'source_urdf_sha256':sha(urdf),'body_axes':'forward=-Y,left=+X,up=+Z',
         'optical_axes':'+Z forward,+X image right,+Y image down',
         'geometry_scope':'Production CAD hip anchors; proposed camera positions, not fitted brackets',
         'candidates':candidates,'rig_counts_to_compare':[2,4,6],
         'evaluate':['actual C detailed CAD and study model separately','recorded gait and full joint sweeps',
              'roll/pitch and plate height','all travel/turn bearings','survey payload occlusion',
              'per-leg support-region visibility/age/error','stopping swept envelope','loaded USB timing']})
    save(out/'payload_ledger.json',{'version':1,'source_urdf_sha256':sha(urdf),
         'current_link_masses_kg':masses,'current_total_kg':sum(masses.values()),
         'available_sensors_confirmed_by_user':['Mid-360','D455'],
         'purchase_restriction':False,
         'components_to_reconcile':[{'name':n,'measured_mass_kg':None,'included_in_current_urdf':None,
             'body_com_m':None,'inertia_at_com_kg_m2':None} for n in
             ('battery','Jetson_carrier_cooling','power_CAN_wiring','Mid-360','D455','near_depth_rig',
              'all_sensor_brackets_cables','survey_payload')],
         'warning':'Do not add nominal component masses until inclusion in current assembly is reconciled'})
    save(out/'observation_contract.json',{'version':'terrain_map_v1_proposed',
         'deployment_approved':False,'base_contract':'omni_history_direct_v1 retained as baseline',
         'commands':['v_forward_mps','v_left_mps','yaw_rate_rad_s'],
         'body_to_navigation':[[0,-1,0],[1,0,0],[0,0,1]],
         'joint_order':'runtime joint_names; name lookup only',
         'terrain':{'frame':'gravity-aligned, body-centred navigation frame',
             'extent_m':[2,2],'resolution_m':.02,
             'channels':['relative_height_scaled','usable_mask','age_scaled','sigma_scaled'],
             'timebase':'same monotonic acquisition timebase as observation assembly',
             'max_age_s':.25,'max_std_m':.015,'height_scale_m':.20,
             'threshold_status':'proposed pilot settings; calibrate against real sensor/pose errors',
             'missing':'zero height padding requires zero usable mask; never infer support',
             'support_rule':'all required cells must be usable; material/support feasibility checked separately'},
         'actor_forbidden':['true_contact_forces','true_friction','perfect_root_velocity',
                            'true_terrain_without_sensor_model'],
         'teacher_only':['terrain_truth','contact_truth','randomized_dynamics_parameters'],
         'reuse':'Existing phase3 sensor transport code after compatibility/clock tests; do not reuse old mass/mount defaults'})
    blockers=['Full commanded-motion gates not yet independently passed',
              'Detailed C production fit/linkage validation',
              'Payload mass/COM/inertia reconciliation',
              'Calibrated actuator/control model and limits',
              'Isaac terrain import/contact/sensor smoke test',
              'Terrain environment/reward adapter implementation',
              'Standing admission on exact payload asset']
    evidence=None
    if args.flat_evaluation:
        evaluation=json.loads(args.flat_evaluation.read_text())
        evidence={'path':str(args.flat_evaluation.resolve()),'sha256':sha(args.flat_evaluation),
                  'static_passed':sum(bool(r['pass']) for r in evaluation['static']),
                  'static_total':len(evaluation['static']),
                  'transition_passed':sum(bool(r['pass']) for r in evaluation['transitions']),
                  'all_scenarios_pass':evaluation['all_scenarios_pass']}
        if evidence['all_scenarios_pass']:blockers.pop(0)
    save(out/'readiness.json',{'cpu_preparation_complete':True,'ready_for_terrain_training':False,
         'terrain_count':len(fixtures),'mount_candidate_count':len(candidates),
         'blocking_items':blockers,'flat_evaluation':evidence,
         'next_gpu_task':'short imported-terrain geometry/contact/sensor validation when scheduling permits',
         'long_training_started':False,
         'sources':{str(x.relative_to(ROOT)):sha(x) for x in
             (Path(__file__),ROOT/'experiments/terrain/tools/terrain_readiness.py',urdf)}})
    print(json.dumps({'out':str(out),'terrains':len(fixtures),'mount_candidates':len(candidates),
                      'launchable':False,'blocking_items':len(blockers)}))


if __name__=='__main__':main()
