#!/usr/bin/env python3
"""Dedicated CPU gate for the 31-body / 30-tree / 6-closure physical MKII asset.

Checks authored geometry and a prescribed kinematic branch. It does not claim
that PhysX maintains closure or that the actuator has been hardware calibrated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from audit_mkii_stance import _origin
import mkii_fourbar_kinematics as kin
from prepare_mkii_usd import check_stage_units_and_scale, dependencies, mass_comparison, read_inertials, rigid_bodies, sha256

FRAME_ATOL = 2e-7
CLOSURE_POINT_ATOL = 2e-7
CLOSURE_AXIS_ATOL = 2e-6
from mkii_fourbar_kinematics import (
    CONTRACT_NUMERIC_ATOL, CONTRACT_ANGLE_DIAGNOSTIC_ATOL, compare_kinematic_contract,
)

def transform(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def joint_frame(joint, side):
    result = np.eye(4)
    result[:3, 3] = getattr(joint, f'GetLocalPos{side}Attr')().Get()
    rotation = Gf.Rotation(Gf.Quatd(getattr(joint, f'GetLocalRot{side}Attr')().Get()))
    result[:3, :3] = np.array([rotation.TransformDir(Gf.Vec3d(*a)) for a in np.eye(3)]).T
    return result


def validate(urdf, pins, usd, contract_path=None):
    from prepare_mkii_fourbar_usd import mesh_sources, read_stl

    urdf, pins, usd = Path(urdf), Path(pins), Path(usd)
    contract_path = Path(contract_path) if contract_path else usd.parent/'kinematics.json'
    contract = json.loads(contract_path.read_text())
    expected = kin.make_contract(urdf, pins)
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    comparison = compare_kinematic_contract(contract, expected)
    errors.extend('Kinematic contract differs: '+message for message in comparison['errors'])
    root, _, _ = kin.load_model(urdf, pins)
    stage = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadAll)
    if not stage:
        raise ValueError(f'Cannot open USD {usd}')
    check_stage_units_and_scale(stage)
    check(stage.GetDefaultPrim().GetPath() == '/Robot', 'Expected /Robot default prim')
    check(stage.GetRootLayer().customLayerData.get('kinematic_contract_sha256') == sha256(contract_path), 'Authored kinematic contract hash mismatch')
    bodies = rigid_bodies(stage)
    check(set(bodies) == set(contract['body_paths']), '31-body membership mismatch')
    mass = mass_comparison(stage, read_inertials(urdf))
    errors.extend(mass['errors'])
    source_frames = contract['joint_frames']
    usd_joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
    check(set(usd_joints) == set(source_frames), '36 physical revolute joint identities mismatch')
    check(len([prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]) == 36, 'Unexpected non-revolute joint')
    observed_frames, active, closures = {}, [], []
    for name in sorted(set(usd_joints) & set(source_frames)):
        joint, reference = usd_joints[name], source_frames[name]
        check(joint.GetAxisAttr().Get() == 'Z', f'{name}: joint axis is not Z')
        check(joint.GetJointEnabledAttr().Get() is True, f'{name}: joint disabled')
        excluded = joint.GetExcludeFromArticulationAttr().Get()
        check(excluded is reference['exclude_from_articulation'], f'{name}: articulation exclusion differs')
        if excluded:
            closures.append(name)
        row = dict(reference)
        for side in [0, 1]:
            expected_path = '/Robot/'+contract['body_paths'][reference[f'body{side}']]
            check([str(x) for x in getattr(joint, f'GetBody{side}Rel')().GetTargets()] == [expected_path], f'{name}: body{side} target differs')
            matrix = joint_frame(joint, side)
            check(np.allclose(matrix, reference[f'body{side}_from_hinge_matrix'], atol=FRAME_ATOL, rtol=0), f'{name}: body{side} hinge frame differs')
            row[f'body{side}_from_hinge_matrix'] = matrix.tolist()
        observed_frames[name] = row
        drive = joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, 'angular')
        check(drive == reference['actuated'], f'{name}: active/passive drive mismatch')
        attrs = [a.GetName().lower() for a in joint.GetPrim().GetAttributes() if a.HasAuthoredValue()]
        schemas = str(joint.GetPrim().GetMetadata('apiSchemas')).lower()
        check(not any('mimic' in a for a in attrs) and 'mimic' not in schemas, f'{name}: mimic remains')
        if reference['actuated']:
            active.append(name)
            api = UsdPhysics.DriveAPI(joint.GetPrim(), 'angular')
            check(api.GetStiffnessAttr().Get() == 0 and api.GetDampingAttr().Get() == 0, f'{name}: hidden USD PD controller')
            check(api.GetMaxForceAttr().Get() == 5.5, f'{name}: unexpected source peak solver allowance')
        else:
            check(not any(('drive:' in a or 'armature' in a or 'maxforce' in a or 'jointfriction' in a) for a in attrs), f'{name}: passive actuator property')
        if name in contract['joint_limits_rad']:
            actual = np.radians([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
            check(np.allclose(actual, contract['joint_limits_rad'][name], atol=2e-7, rtol=0), f'{name}: transformed limits differ')
        else:
            check(not joint.GetLowerLimitAttr().HasAuthoredValue() and not joint.GetUpperLimitAttr().HasAuthoredValue(), f'{name}: unexpected closure limit')
    check(set(active) == set(contract['active_joint_names']) and len(active) == 18, '18 active motors required')
    check(set(closures) == set(contract['closure_joint_names']) and len(closures) == 6, 'Six external closures required')
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    check(len(roots) == 1 and roots[0].GetPath() == '/Robot/Geometry/body', 'One floating-body articulation root required')
    source_zero = kin.forward_kinematics(source_frames, {name: 0. for name in contract['tree_joint_names']})
    visual_count = collision_count = 0
    meshes = mesh_sources(root, urdf)
    verified_meshes = set()
    for link in root.findall('link'):
        name = link.get('name')
        body = bodies.get(name)
        if not body:
            continue
        check(str(body.GetPath()) == '/Robot/'+contract['body_paths'][name], f'{name}: body hierarchy differs')
        body_pose = transform(body)
        check(np.allclose(body_pose, source_zero[name], atol=FRAME_ATOL, rtol=0), f'{name}: zero pose differs')
        check('PhysxContactReportAPI' in str(body.GetMetadata('apiSchemas')), f'{name}: contact reporting absent')
        visual_scope = stage.GetPrimAtPath(body.GetPath().AppendPath('visuals'))
        collision_scope = stage.GetPrimAtPath(body.GetPath().AppendPath('collisions'))
        check(len(visual_scope.GetChildren()) == len(link.findall('visual')), f'{name}: visual count differs')
        check(len(collision_scope.GetChildren()) == len(link.findall('collision')), f'{name}: collision count differs')
        for index, visual in enumerate(link.findall('visual')):
            prim = stage.GetPrimAtPath(body.GetPath().AppendPath(f'visuals/visual_{index:03d}'))
            check(bool(prim) and prim.IsA(UsdGeom.Mesh), f'{name}: missing CAD mesh {index}')
            if not prim:
                continue
            visual_count += 1
            source_mesh = visual.find('./geometry/mesh')
            expected_transform = _origin(visual)
            expected_transform[:3, :3] = expected_transform[:3, :3] @ np.diag([float(v) for v in source_mesh.get('scale', '1 1 1').split()])
            actual_transform = np.linalg.inv(body_pose) @ transform(prim)
            check(np.allclose(actual_transform, expected_transform, atol=FRAME_ATOL, rtol=0), f'{name}: visual {index} placement differs')
            uri = source_mesh.get('filename')
            row = meshes[uri]
            check(prim.GetAttribute('hexapod:sourceStlSha256').Get() == row['sha256'], f'{name}: visual {index} mesh identity differs')
            if uri not in verified_meshes:
                verified_meshes.add(uri)
                points, normals = read_stl(row['path'])
                mesh = UsdGeom.Mesh(prim)
                check(np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), points), f'{uri}: STL vertices differ')
                check(np.array_equal(np.asarray(mesh.GetNormalsAttr().Get()), normals), f'{uri}: STL normals differ')
                check(np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), np.arange(len(points))), f'{uri}: triangle indices differ')
                check(np.array_equal(np.asarray(mesh.GetFaceVertexCountsAttr().Get()), np.full(len(normals), 3)), f'{uri}: triangle counts differ')
        for index, collision in enumerate(link.findall('collision')):
            prim = stage.GetPrimAtPath(body.GetPath().AppendPath(f'collisions/collision_{index:03d}'))
            collision_count += 1
            check(bool(prim) and prim.HasAPI(UsdPhysics.CollisionAPI), f'{name}: collision {index} missing')
            if not prim:
                continue
            check(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True, f'{name}: disabled collision {index}')
            shape = list(collision.find('geometry'))[0]
            expected_transform = _origin(collision)
            if shape.tag == 'box':
                check(prim.IsA(UsdGeom.Cube) and UsdGeom.Cube(prim).GetSizeAttr().Get() == 1., f'{name}: box size/type differs')
                expected_transform[:3, :3] = expected_transform[:3, :3] @ np.diag([float(v) for v in shape.get('size').split()])
            elif shape.tag == 'sphere':
                check(prim.IsA(UsdGeom.Sphere) and abs(UsdGeom.Sphere(prim).GetRadiusAttr().Get()-float(shape.get('radius'))) < 1e-9, f'{name}: sphere differs')
            elif shape.tag == 'cylinder':
                check(prim.IsA(UsdGeom.Cylinder) and abs(UsdGeom.Cylinder(prim).GetRadiusAttr().Get()-float(shape.get('radius'))) < 1e-9 and abs(UsdGeom.Cylinder(prim).GetHeightAttr().Get()-float(shape.get('length'))) < 1e-9 and UsdGeom.Cylinder(prim).GetAxisAttr().Get() == 'Z', f'{name}: cylinder differs')
            check(np.allclose(np.linalg.inv(body_pose) @ transform(prim), expected_transform, atol=FRAME_ATOL, rtol=0), f'{name}: collision {index} placement differs')
    maxima = {'point_error_m': 0., 'axis_error_rad': 0.}
    sample_count = 0
    if set(observed_frames) == set(source_frames):
        # Sweep all six local intervals simultaneously; the legs are independent
        # for closure. This does not prove inter-leg clearance or solver behavior.
        for fraction in np.linspace(0., 1., 201):
            positions = kin.expand_active({name: (1-fraction)*contract['joint_limits_rad'][name][0]+fraction*contract['joint_limits_rad'][name][1] for name in contract['active_joint_names']})
            for row in kin.closure_errors(observed_frames, positions).values():
                sample_count += 1
                for key in maxima:
                    maxima[key] = max(maxima[key], row[key])
                check(row['axis_dot'] > 0, 'Closure hinge axes reversed')
        for row in kin.closure_errors(observed_frames, contract['default_joint_positions_rad']).values():
            for key in maxima:
                maxima[key] = max(maxima[key], row[key])
        check(maxima['point_error_m'] <= CLOSURE_POINT_ATOL, 'Authored hinge closure position sweep exceeds tolerance')
        check(maxima['axis_error_rad'] <= CLOSURE_AXIS_ATOL, 'Authored hinge closure axis sweep exceeds tolerance')
        reset_poses = kin.forward_kinematics(observed_frames, contract['default_joint_positions_rad'])
        rows = kin.collision_heights(root, reset_poses, contract['reset_root_height_m'])
        check(min(r['bottom_z_m'] for r in rows if r['foot']) >= .005-2e-7, 'Reset feet lack explicit 5mm clearance')
        check(min(r['bottom_z_m'] for r in rows if not r['foot']) > 0, 'Reset non-foot penetration')
    else:
        rows = []
    deps = dependencies(usd, usd.parent)
    return {'pass': not errors, 'errors': errors, 'schema': 'hexapod.mkii_fourbar_v3.cpu_audit.v1',
            'rigid_bodies': len(bodies), 'tree_joints': len(usd_joints)-len(closures), 'closure_joints': len(closures),
            'active_joints': len(active), 'mass_kg': sum(x.mass for x in read_inertials(urdf).values()),
            'mass_properties': mass, 'visual_instances': visual_count, 'collision_primitives': collision_count,
            'unique_source_meshes': len(verified_meshes), 'prescribed_closure_samples': sample_count,
            'prescribed_closure_maxima': maxima, 'reset_primitive_heights': rows,
            'source_sha256': contract['source_sha256'], 'kinematic_contract_sha256': sha256(contract_path),
            'kinematic_contract_comparison': comparison,
            'usd_root_sha256': sha256(usd), 'dependencies': deps,
            'tolerances': {'frame_absolute': FRAME_ATOL, 'closure_point_m': CLOSURE_POINT_ATOL, 'closure_axis_rad': CLOSURE_AXIS_ATOL},
            'physical_validation': 'not_performed',
            'scope': 'Authored geometry, source properties and sampled prescribed kinematics; no PhysX, continuous self-collision, torque-speed or thermal qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usd', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = validate(kin.URDF, kin.PINS, args.usd)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.report:
        if args.report.exists():
            raise SystemExit('Refusing to overwrite an audit report')
        args.report.write_text(text)
    print(json.dumps({key: result[key] for key in ['pass', 'errors', 'rigid_bodies', 'tree_joints', 'closure_joints', 'prescribed_closure_maxima']}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
