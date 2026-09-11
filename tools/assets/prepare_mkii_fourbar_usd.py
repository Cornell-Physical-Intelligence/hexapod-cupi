#!/usr/bin/env python3
"""Build a NEW 31-body physical four-bar USD directly from measured CAD frames.

CPU/OpenUSD only: no Isaac application, importer defaults, or GPU. The 12 URDF
kinematic mimics are replaced by six ordinary excluded revolutes by default.
Explicit candidates use D6 transverse rows or native bilateral PhysX joint
couplings. No variant receives live-physics admission from this tool.
The output directory must be new. URDFs and historical USDs are never modified.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, Vt

from tools.assets.audit_mkii_stance import _origin
from tools.assets import mkii_fourbar_kinematics as kin
from tools.assets.prepare_mkii_usd import dependencies, principal_axes, read_inertials, sha256

REVOLUTE_CLOSURE = 'revolute_5row_v3'
PLANAR_D6_CLOSURE = 'planar_d6_xy_v4'
PHYSICAL_MIMIC_CLOSURE = 'physical_mimic_v5'
CLOSURE_VARIANTS = (REVOLUTE_CLOSURE, PLANAR_D6_CLOSURE, PHYSICAL_MIMIC_CLOSURE)


def check_closure_variant(value):
    if value not in CLOSURE_VARIANTS:
        raise ValueError(f'Unknown closure constraint variant: {value!r}')
    return value


def set_transform(prim, matrix):
    """Column-vector NumPy matrix to row-vector Gf matrix, without inversion."""
    UsdGeom.Xformable(prim).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Matrix4d(*map(float, np.asarray(matrix).T.flat)))


def quaternion(matrix):
    return Gf.Quatf(Gf.Matrix3d(*map(float, np.asarray(matrix)[:3, :3].T.flat)).ExtractRotation().GetQuat()).GetNormalized()


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84+count*50 and count:
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)
        points = triangles['vertices'].reshape(-1, 3).copy()
        normals = triangles['normal'].copy()
    else:
        vertices, normals = [], []
        for line in data.decode('ascii').splitlines():
            words = line.split()
            if words[:1] == ['vertex']:
                vertices.append([float(v) for v in words[1:]])
            elif words[:2] == ['facet', 'normal']:
                normals.append([float(v) for v in words[2:]])
        points, normals = np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) % 3
            or normals.shape != (len(points)//3, 3)
            or not np.isfinite(points).all() or not np.isfinite(normals).all()):
        raise ValueError(f'Invalid STL triangle data: {path}')
    return points, normals


def mesh_sources(root, urdf):
    result = {}
    package = Path(urdf).resolve().parent.parent
    for visual in root.findall('./link/visual'):
        mesh = visual.find('./geometry/mesh')
        if mesh is None:
            raise ValueError('Expected CAD mesh visual')
        filename = mesh.get('filename')
        prefix = 'package://hexapod_mkii_assy/'
        if not filename.startswith(prefix):
            raise ValueError(f'Unsupported mesh URI {filename}')
        path = (package / filename[len(prefix):]).resolve()
        if not path.is_file() or not path.is_relative_to(package):
            raise ValueError(f'Missing or escaped mesh: {filename}')
        key = 'mesh_'+sha256(path)[:24]
        result[filename] = {'path': path, 'prim_path': '/Meshes/'+key, 'sha256': sha256(path)}
    return result


def create_mesh_layer(path, meshes):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, '/Meshes')
    seen = set()
    for row in meshes.values():
        if row['prim_path'] in seen:
            continue
        seen.add(row['prim_path'])
        points, normals = read_stl(row['path'])
        mesh = UsdGeom.Mesh.Define(stage, row['prim_path'])
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(normals), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(points), dtype=np.int32)))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
        mesh.GetPrim().CreateAttribute('hexapod:sourceStlSha256', Sdf.ValueTypeNames.String).Set(row['sha256'])
    stage.GetRootLayer().Save()


def author_visual(stage, path, visual, materials, meshes):
    uri = visual.find('./geometry/mesh').get('filename')
    prim = stage.DefinePrim(path, 'Mesh')
    prim.GetReferences().AddReference('./geometry.usdc', meshes[uri]['prim_path'])
    transform = _origin(visual)
    scale = np.array([float(x) for x in visual.find('./geometry/mesh').get('scale', '1 1 1').split()])
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        raise ValueError('Invalid visual scale')
    transform[:3, :3] = transform[:3, :3] @ np.diag(scale)
    set_transform(prim, transform)
    name = visual.find('material').get('name') if visual.find('material') is not None else None
    color = materials.get(name, [0.5, 0.5, 0.5, 1.])
    UsdGeom.Mesh(prim).CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
    UsdGeom.Mesh(prim).CreateDisplayOpacityAttr([color[3]])
    prim.SetInstanceable(True)


def author_collision(stage, path, collision):
    shape = list(collision.find('geometry'))[0]
    transform = _origin(collision)
    if shape.tag == 'sphere':
        geom = UsdGeom.Sphere.Define(stage, path)
        geom.CreateRadiusAttr(float(shape.get('radius')))
    elif shape.tag == 'cylinder':
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(float(shape.get('radius')))
        geom.CreateHeightAttr(float(shape.get('length')))
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
    elif shape.tag == 'box':
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.)
        transform[:3, :3] = transform[:3, :3] @ np.diag([float(x) for x in shape.get('size').split()])
    else:
        raise ValueError(f'Unsupported collision shape {shape.tag}')
    set_transform(geom.GetPrim(), transform)
    geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    return geom


def author_stage(output, root, contract, meshes, *, closure_variant=REVOLUTE_CLOSURE):
    check_closure_variant(closure_variant)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    robot = UsdGeom.Xform.Define(stage, '/Robot')
    stage.SetDefaultPrim(robot.GetPrim())
    stage.GetRootLayer().customLayerData = {
        'hexapod_physical_fourbar_version': 3,
        'kinematic_contract_sha256': sha256(output.parent/'kinematics.json'),
        'source_linkage_urdf_sha256': contract['source_sha256']['linkage_urdf'],
        'source_cad_pin_frames_sha256': contract['source_sha256']['cad_pin_frames'],
        'physical_validation': 'not_performed',
    }
    if closure_variant in (PLANAR_D6_CLOSURE, PHYSICAL_MIMIC_CLOSURE):
        metadata = dict(stage.GetRootLayer().customLayerData)
        metadata.update(hexapod_physical_fourbar_version=5 if closure_variant == PHYSICAL_MIMIC_CLOSURE else 4,
                        closure_constraint_variant=closure_variant)
        stage.GetRootLayer().customLayerData = metadata
    UsdGeom.Scope.Define(stage, '/Robot/Geometry')
    UsdGeom.Scope.Define(stage, '/Robot/Physics')
    frames = contract['joint_frames']
    zero = {name: 0. for name in contract['tree_joint_names']}
    poses = kin.forward_kinematics(frames, zero)
    tree_by_child = {frame['body1']: name for name, frame in frames.items() if not frame['exclude_from_articulation']}
    inertials = read_inertials(kin.URDF)
    materials = {m.get('name'): [float(x) for x in m.find('color').get('rgba').split()] for m in root.findall('material')}
    for link in root.findall('link'):
        name = link.get('name')
        path = '/Robot/'+contract['body_paths'][name]
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        matrix = np.eye(4) if name == 'body' else kin.relative_pose(frames[tree_by_child[name]], 0.)
        set_transform(prim, matrix)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        prim.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float).Set(0.)
        prim.AddAppliedSchema('PhysxContactReportAPI')
        prim.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float).Set(0.)
        mass = UsdPhysics.MassAPI.Apply(prim)
        data = inertials[name]
        moments, axes = principal_axes(data.tensor)
        mass.CreateMassAttr(data.mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, data.com)))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
        mass.CreatePrincipalAxesAttr(axes)
        if name == 'body':
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            prim.AddAppliedSchema('PhysxArticulationAPI')
            prim.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(False)
        UsdGeom.Scope.Define(stage, path+'/visuals')
        UsdGeom.Scope.Define(stage, path+'/collisions')
        for index, visual in enumerate(link.findall('visual')):
            author_visual(stage, path+f'/visuals/visual_{index:03d}', visual, materials, meshes)
        for index, collision in enumerate(link.findall('collision')):
            author_collision(stage, path+f'/collisions/collision_{index:03d}', collision)
    for name, frame in frames.items():
        if closure_variant == PHYSICAL_MIMIC_CLOSURE and frame['exclude_from_articulation']:
            # The equivalent passive-coordinate constraints below replace the
            # external loop joint. CAD endpoints remain in the hashed contract.
            continue
        planar_closure = (closure_variant == PLANAR_D6_CLOSURE
                          and frame['exclude_from_articulation'])
        if planar_closure:
            joint = UsdPhysics.Joint.Define(stage, '/Robot/Physics/'+name)
            # The tree already enforces hinge-axis alignment and zero axial
            # separation. The excluded D6 adds only in-plane point closure.
            # USD axes without a LimitAPI or DriveAPI remain free.
            for axis in ('transX', 'transY'):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1.)
                limit.CreateHighAttr(-1.)
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, '/Robot/Physics/'+name)
            joint.CreateAxisAttr('Z')
        joint.CreateBody0Rel().SetTargets(['/Robot/'+contract['body_paths'][frame['body0']]])
        joint.CreateBody1Rel().SetTargets(['/Robot/'+contract['body_paths'][frame['body1']]])
        for side in [0, 1]:
            matrix = np.array(frame[f'body{side}_from_hinge_matrix'])
            getattr(joint, f'CreateLocalPos{side}Attr')(Gf.Vec3f(*map(float, matrix[:3, 3])))
            getattr(joint, f'CreateLocalRot{side}Attr')(quaternion(matrix))
        joint.CreateExcludeFromArticulationAttr(frame['exclude_from_articulation'])
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateJointEnabledAttr(True)
        if name in contract['joint_limits_rad']:
            lower, upper = contract['joint_limits_rad'][name]
            joint.CreateLowerLimitAttr(float(np.degrees(lower)))
            joint.CreateUpperLimitAttr(float(np.degrees(upper)))
        if frame['actuated']:
            # Explicit motor model supplies actual effort. These zero gains must
            # not introduce a second PD controller or give passive joints drives.
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(0.)
            drive.CreateMaxForceAttr(5.5)
            drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
    if closure_variant == PHYSICAL_MIMIC_CLOSURE:
        for leg in kin.LEGS:
            reference = '/Robot/Physics/'+leg+'_tibia_lever_pivot'
            for suffix, gearing in (('tibia_pitch', -1.), ('tibia_rod_pivot', 1.)):
                prim = stage.GetPrimAtPath('/Robot/Physics/'+leg+'_'+suffix)
                # Native PhysX bilateral impulse constraint:
                # q_target + gearing*q_reference + offset = 0.
                # This is not a kinematic copy or a passive position drive.
                prim.AddAppliedSchema('PhysxMimicJointAPI:rotZ')
                prefix = 'physxMimicJoint:rotZ:'
                for name, value in (('gearing', gearing), ('offset', 0.),
                                    ('naturalFrequency', 0.), ('dampingRatio', 0.)):
                    prim.CreateAttribute(prefix+name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
                prim.CreateAttribute(prefix+'referenceJointAxis', Sdf.ValueTypeNames.Token,
                                     custom=False, variability=Sdf.VariabilityUniform).Set('rotZ')
                prim.CreateRelationship(prefix+'referenceJoint', custom=False).SetTargets([reference])
    stage.GetRootLayer().Save()


def prepare(output, urdf=kin.URDF, pins=kin.PINS, contract_path=kin.CONTRACT,
            *, closure_variant=REVOLUTE_CLOSURE):
    check_closure_variant(closure_variant)
    output = Path(output).resolve()
    if output.suffix not in ('.usd', '.usda') or output.exists():
        raise ValueError('Choose a new .usd/.usda output; existing artifacts are immutable')
    if output.parent.exists() and any(output.parent.iterdir()):
        raise ValueError('Output bundle must be absent or empty')
    root, _, _ = kin.load_model(urdf, pins)
    contract = kin.make_contract(urdf, pins)
    supplied = json.loads(Path(contract_path).read_text())
    comparison = kin.compare_kinematic_contract(supplied, contract)
    if not comparison['pass']:
        raise ValueError('Supplied runtime kinematic contract differs from derived geometry: '+str(comparison['errors']))
    # The supplied portable contract remains the exact runtime identity; tiny
    # platform math differences do not rewrite its bytes or authored defaults.
    contract = supplied
    meshes = mesh_sources(root, urdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fourbar_build_', dir=output.parent.parent) as tmp:
        work = Path(tmp)
        local_output = work/output.name
        shutil.copyfile(contract_path, work/'kinematics.json')
        create_mesh_layer(work/'geometry.usdc', meshes)
        author_stage(local_output, root, contract, meshes, closure_variant=closure_variant)
        from tools.assets.audit_mkii_fourbar_usd import validate
        report = validate(urdf, pins, local_output, work/'kinematics.json', closure_variant=closure_variant)
        if not report['pass']:
            raise ValueError('Physical four-bar CPU validation failed: '+str(report['errors']))
        report['source_meshes'] = {uri: {'sha256': row['sha256'], 'usd_prim_path': row['prim_path']} for uri, row in sorted(meshes.items())}
        report['builder_sha256'] = sha256(Path(__file__))
        report['kinematics_tool_sha256'] = sha256(Path(kin.__file__))
        report['physical_validation'] = 'not_performed'
        (work/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        for path in work.iterdir():
            shutil.move(str(path), output.parent/path.name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=kin.ROOT/kin.USD_RELATIVE)
    parser.add_argument('--closure-variant', choices=CLOSURE_VARIANTS, default=REVOLUTE_CLOSURE)
    args = parser.parse_args()
    if args.closure_variant != REVOLUTE_CLOSURE and args.output == kin.ROOT/kin.USD_RELATIVE:
        parser.error('A diagnostic closure variant requires an explicit new --output path')
    report = prepare(args.output, closure_variant=args.closure_variant)
    print(json.dumps({'pass': report['pass'], 'output': str(args.output), 'bodies': report['rigid_bodies'], 'tree_joints': report['tree_joints'], 'closures': report['closure_joints']}))


if __name__ == '__main__':
    main()
