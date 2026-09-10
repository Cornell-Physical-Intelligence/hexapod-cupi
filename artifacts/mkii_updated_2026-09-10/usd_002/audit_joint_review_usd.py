#!/usr/bin/env python3
"""Check the user-requested angular limits and endpoint kinematics, without physics.
The +180-degree scalar bound is checked directly; rotation matrices alone cannot
separate physically equivalent +180/-180 endpoint orientations. This is an
encoding/kinematic audit, never a certificate of collision-free joint travel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from prepare_updated_usd import audit, origin

SOURCES = {
    'usd_degree_units': 'https://openusd.org/release/api/usd_physics_page_front.html',
    'physx_revolute_coordinate_wrap': 'https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html',
}


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def joint_frame(api, side):
    pos = getattr(api, f'GetLocalPos{side}Attr')().Get()
    quat = getattr(api, f'GetLocalRot{side}Attr')().Get()
    rot = Gf.Rotation(Gf.Quatd(quat).GetNormalized())
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        rot.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)
    ]).T
    transform[:3, 3] = pos
    return transform


def evaluate_graph(root_transform, edges, coordinates):
    frames = {'body': root_transform.copy()}
    pending = list(edges.items())
    while pending:
        changed = False
        for name, edge in pending[:]:
            if edge['parent'] not in frames:
                continue
            if edge['child'] in frames:
                raise ValueError('Duplicate incoming edge or cycle')
            frames[edge['child']] = (
                frames[edge['parent']]
                @ edge['frame0']
                @ rotation_z(coordinates.get(name, 0.))
                @ np.linalg.inv(edge['frame1'])
            )
            pending.remove((name, edge))
            changed = True
        if not changed:
            raise ValueError('Disconnected joint graph')
    return frames


def audit_successor(bundle):
    bundle = bundle.resolve()
    baseline = audit(bundle)
    errors = list(baseline['errors'])
    stage = Usd.Stage.Open(str(bundle / 'robot.usda'))
    urdf = ET.parse(bundle / 'source/source.urdf').getroot()
    model = json.loads((bundle / 'source/model.json').read_text())
    model_joints = {j['name']: j for j in model['joints']}
    expected_edges, actual_edges, joint_rows = {}, {}, []
    for node in urdf.findall('joint'):
        name = node.get('name')
        parent, child = node.find('parent').get('link'), node.find('child').get('link')
        prim = stage.GetPrimAtPath('/Robot/joints/' + name)
        api = UsdPhysics.RevoluteJoint(prim)
        body0, body1 = api.GetBody0Rel().GetTargets(), api.GetBody1Rel().GetTargets()
        if list(map(str, body0)) != ['/Robot/' + parent] or list(map(str, body1)) != ['/Robot/' + child]:
            errors.append('USD joint relationship mismatch: ' + name)
            continue
        lo, hi = api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()
        expected = node.find('limit')
        expected_lo, expected_hi = [math.degrees(float(expected.get(k))) for k in ('lower', 'upper')]
        semantic_expected = None
        if name.endswith('_femur_pitch'):
            semantic_expected = [-120., 80.]
        elif name.endswith('_tibia_pitch'):
            semantic_expected = [-5., 180.]
        elif name.endswith('_coxa_yaw'):
            # The new coordinate zero is the midpoint of the per-leg allowed arc.
            if abs(expected_lo + expected_hi) > 1e-6:
                errors.append('Yaw limit interval is not centered at new zero: ' + name)
        else:
            errors.append('Unexpected joint naming: ' + name)
        ordered = all(math.isfinite(v) for v in (lo, hi)) and -360 < lo <= 0 <= hi < 360 and lo < hi
        if not ordered:
            errors.append('Invalid ordered/angular range: ' + name)
        semantic_ok = semantic_expected is None or np.allclose([lo, hi], semantic_expected, rtol=0, atol=1e-5)
        if not semantic_ok:
            errors.append('User-requested scalar degree limits changed: ' + name)
        if max(abs(lo - expected_lo), abs(hi - expected_hi)) > 1e-5:
            errors.append('Scalar limit differs from source URDF: ' + name)
        m = model_joints[name]
        if max(abs(expected_lo - math.degrees(m['lower'])), abs(expected_hi - math.degrees(m['upper']))) > 1e-8:
            errors.append('Model/URDF degree limits differ: ' + name)
        if name.endswith('_tibia_pitch') and hi != 180.:
            errors.append('Positive180 endpoint was wrapped or changed: ' + name)
        expected_edges[name] = {'parent': parent, 'child': child, 'frame0': origin(node), 'frame1': np.eye(4)}
        actual_edges[name] = {'parent': body0[0].name, 'child': body1[0].name, 'frame0': joint_frame(api, 0), 'frame1': joint_frame(api, 1)}
        joint_rows.append({'joint': name, 'lower_deg': lo, 'upper_deg': hi,
                           'range_deg': hi - lo, 'scalar_limits_match_user': bool(semantic_ok),
                           'within_documented_revolute_wrap_interval': bool(ordered)})
    root_t = np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/Robot/body')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    samples = [('all_zero', {})]
    for row in joint_rows:
        name = row['joint']
        for side in ('lower', 'upper'):
            samples.append((name + '_' + side, {name: math.radians(row[side + '_deg'])}))
        if name.endswith('_tibia_pitch'):
            samples.append((name + '_179p999deg', {name: math.radians(179.999)}))
    samples.extend([
        ('all_lower_bounds_kinematics_only', {r['joint']: math.radians(r['lower_deg']) for r in joint_rows}),
        ('all_upper_bounds_kinematics_only', {r['joint']: math.radians(r['upper_deg']) for r in joint_rows}),
    ])
    sample_rows = []
    if len(expected_edges) == len(actual_edges) == 18:
        for name, coordinates in samples:
            expected_frames = evaluate_graph(root_t, expected_edges, coordinates)
            actual_frames = evaluate_graph(root_t, actual_edges, coordinates)
            rotation_error = max(float(abs(expected_frames[k][:3, :3] - actual_frames[k][:3, :3]).max()) for k in expected_frames)
            position_error = max(float(abs(expected_frames[k][:3, 3] - actual_frames[k][:3, 3]).max()) for k in expected_frames)
            passed = rotation_error < 3e-6 and position_error < 3e-7
            sample_rows.append({'pose': name, 'coordinates_rad': coordinates, 'max_rotation_matrix_error': rotation_error,
                                'max_position_error_m': position_error, 'pass': bool(passed)})
            if not passed:
                errors.append('Endpoint kinematics mismatch: ' + name)
    else:
        errors.append('Cannot test endpoint kinematics with missing joint relationships')
    return {
        'status': 'offline successor encoding and kinematics verified; native physics pending',
        'pass': not errors, 'errors': errors,
        'usd_sha256': hashlib.sha256((bundle / 'robot.usda').read_bytes()).hexdigest(),
        'source_model_sha256': hashlib.sha256((bundle / 'source/model.json').read_bytes()).hexdigest(),
        'source_urdf_sha256': hashlib.sha256((bundle / 'source/source.urdf').read_bytes()).hexdigest(),
        'baseline_membership_geometry_inertia_audit_pass': baseline['pass'],
        'joint_limits': joint_rows, 'kinematic_samples': sample_rows,
        'kinematic_sample_count': len(sample_rows),
        'scalar_angle_convention': 'Limits remain scalar degrees; +180 is not normalized to -180. Quaternions encode frame rotations only.',
        'collision_clearance_certificate': False,
        'native_cooking_or_dynamics_run': False,
        'caveat': 'The sampled wide-range poses may intersect the body, other links, or ground. These tests only compare coordinate/frame math, with no contact-free claim.',
        'official_sources': SOURCES,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = audit_successor(args.bundle)
    if args.output:
        if args.output.exists():
            raise ValueError('Choose a new audit filename; do not replace evidence')
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'pass', 'errors', 'kinematic_sample_count')}, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
