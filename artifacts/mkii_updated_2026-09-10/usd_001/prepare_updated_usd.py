#!/usr/bin/env python3
"""Portable, offline USD preparation; no Isaac/PhysX engine is imported.
Requires numpy and OpenUSD. This authors a recipe, never an admission report.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, shutil, struct, xml.etree.ElementTree as ET
import importlib.util
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt
SDF_SELECTION = {'mesh_003_1_1_02_eb461_502_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__.stl': (256, None, 'exposed actuator end housing; interior hardware omitted separately'), 'mesh_004_1_1_06_eb463_507_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__.stl': (256, None, 'exposed actuator cylindrical housing'), 'mesh_005_1_1_06_eb463_509_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__.stl': (256, None, 'exposed actuator end housing'), 'mesh_006_1_1_31_eb463_510_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__.stl': (256, None, 'exposed actuator end housing'), 'mesh_016_bottom_enclosure.stl': (900, 0.005, 'chassis outer enclosure, measured5mm overall thickness'), 'mesh_017_bottom_plate.stl': (900, 0.0016, 'chassis plate, measured1.6mm plate thickness'), 'mesh_031_femur_first_stage.stl': (320, 0.002032, 'U bracket, measured2.032mm side walls'), 'mesh_032_first_joint_bottom_plate.stl': (512, 0.0016, 'U bracket and exposed bearing apertures, measured1.6mm wall'), 'mesh_033_first_joint_spacer.stl': (256, None, 'structural joint spacer'), 'mesh_034_first_joint_top.stl': (512, 0.0016, 'coxa top plate, measured1.6mm plate'), 'mesh_045_motor_bearing_holder.stl': (320, None, 'exposed bearing support ring; preserve large bore'), 'mesh_046_motor_flange.stl': (256, None, 'exposed output flange'), 'mesh_054_standoff_plate.stl': (640, None, 'chassis structural standoff plate'), 'mesh_055_standoff_plate__2.stl': (512, None, 'chassis structural standoff plate'), 'mesh_056_tibia.stl': (1024, None, 'entire fork and foot surface; source cap is nonspherical'), 'mesh_057_tibia_attatchment_plate.stl': (320, 0.002032, 'U bracket with exposed bearing apertures, measured2.032mm wall'), 'mesh_058_top_enclosure.stl': (900, 0.005, 'chassis outer enclosure, measured5mm overall thickness')}
SOURCES = {'sdf_schema': 'https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html', 'sdf_thin_walls': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/guides/collision_guide.html', 'collider_compatibility': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html', 'joint_velocity_units': 'https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_joint_a_p_i.html', 'mass_axes': 'https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html', 'adjacent_collision_filter': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/articulation_stability_guide.html'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, d):
    Path(p).write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')

def vec(s):
    return np.array([float(v) for v in s.split()])

def quat_matrix(xyzw):
    x, y, z, w = np.asarray(xyzw, dtype=float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

def transform(xyz, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def from_record(r):
    return transform(r['xyz'], quat_matrix(r['quaternion_xyzw']))

def origin(element):
    o = element.find('origin')
    return np.eye(4) if o is None else transform(vec(o.get('xyz', '0 0 0')), rpy_matrix(vec(o.get('rpy', '0 0 0'))))

def gf_matrix(T):
    return Gf.Matrix4d(*map(float, T.T.flat))

def gf_quat(R, precision='float'):
    q = Gf.Matrix3d(*map(float, R.T.flat)).ExtractRotation().GetQuat()
    q = q if q.GetReal() >= 0 else -q
    return Gf.Quatf(q).GetNormalized() if precision == 'float' else q.GetNormalized()

def set_transform(prim, T):
    UsdGeom.Xformable(prim).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(gf_matrix(T))

def attr(p, n, t, value, uniform=False):
    p.CreateAttribute(n, t, custom=False, variability=Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying).Set(value)

def schema(p, name):
    p.AddAppliedSchema(name)

def read_stl(p):
    data = p.read_bytes()
    n = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f'Expected exact binary STL: {p}')
    tri = np.frombuffer(data, offset=84, count=n, dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))['vertices'].copy()
    if not np.isfinite(tri).all():
        raise ValueError('Nonfinite STL vertices')
    vertices, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    if np.any(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) <= 1e-15):
        raise ValueError(f'Degenerate source triangle: {p.name}')
    normals = np.frombuffer(data, offset=84, count=n, dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))['normal'].copy()
    return (vertices, faces, normals)

def principal(tensor):
    d, C = np.linalg.eigh(tensor)
    if d[0] <= 0 or d[-1] > d[0] + d[1] + 1e-12:
        raise ValueError('Unphysical principal moments')
    for i in range(3):
        if C[np.argmax(abs(C[:, i])), i] < 0:
            C[:, i] *= -1
    if np.linalg.det(C) < 0:
        C[:, -1] *= -1
    # NumPy eigenvectors are columns. gf_quat transposes into Gf row convention.
    q = gf_quat(C)
    rotation = Gf.Rotation(Gf.Quatd(q))
    R = np.array([rotation.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)]).T
    if not np.allclose(R @ np.diag(d) @ R.T, tensor, rtol=1e-05, atol=1e-08):
        raise ValueError('Principal-axis roundtrip failed')
    return (d, q)

def fk_model(d):
    frames = {'body': np.eye(4)}
    remaining = list(d['joints'])
    while remaining:
        progressed = False
        for j in remaining[:]:
            if j['parent'] not in frames:
                continue
            if j.get('default_value', 0) != 0:
                raise ValueError('Preparation expects the URDF default pose at zero')
            frames[j['child']] = frames[j['parent']] @ from_record(j)
            remaining.remove(j)
            progressed = True
        if not progressed:
            raise ValueError('Cyclic or disconnected body graph')
    return frames

def collision_metadata(meshes, parts):
    counts = collections.Counter((p['mesh'] for p in parts))
    rows = []
    for m in meshes:
        name = m['mesh']
        selected = name in SDF_SELECTION
        row = {'mesh': name, 'source_mesh': m['source_mesh'], 'instances': counts[name], 'selected_for_collision': selected}
        if selected:
            resolution, wall, reason = SDF_SELECTION[name]
            spacing = max(np.asarray(m['bounds_m'])[1] - np.asarray(m['bounds_m'])[0]) / resolution
            row.update(approximation='sdf', resolution=resolution, spacing_m=float(spacing), measured_wall_m=wall, wall_samples=None if wall is None else wall / spacing, reason=reason)
            if wall is not None and wall < 4 * spacing:
                raise ValueError('Requested SDF resolution loses measured thin wall')
        else:
            row['reason'] = 'Excluded from environment collision: fastener, internal bearing/gear/rotor, or small internal actuator component. Exact visual and assigned mass/inertia are preserved; exposed fastener snag/contact remains an explicit modeling omission.'
        rows.append(row)
    return rows

def prepare(source, urdf, output, model_path=None):
    effort, velocity = (5.5, 50.26548245743669)
    source, urdf, output = (source.resolve(), urdf.resolve(), output.resolve())
    if output.exists():
        raise ValueError('Output must be a new directory; never overwrite prepared evidence')
    model_path = (model_path or source / 'model.json').resolve()
    d = json.loads(model_path.read_text())
    mapping = json.loads((source / 'mesh_mapping.json').read_text())
    if len(d['links']) != 19 or len(d['joints']) != 18 or len(d['parts']) != 1753:
        raise ValueError('Expected19 bodies18 joints1753 parts')
    if not d.get('topology_user_confirmed'):
        raise ValueError('User confirmation of serial topology required')
    initial_hashes = {str(p): sha(p) for p in [model_path, source / 'mesh_mapping.json', urdf]}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    shutil.copy2(model_path, output / 'source/model.json')
    shutil.copy2(source / 'mesh_mapping.json', output / 'source/mesh_mapping.json')
    shutil.copy2(urdf, output / 'source/source.urdf')
    library = Usd.Stage.CreateNew(str(output / 'geometry.usdc'))
    groot = UsdGeom.Scope.Define(library, '/Geometry')
    library.SetDefaultPrim(groot.GetPrim())
    UsdGeom.SetStageMetersPerUnit(library, 1)
    UsdPhysics.SetStageKilogramsPerUnit(library, 1)
    UsdGeom.SetStageUpAxis(library, UsdGeom.Tokens.z)
    meshes = []
    for i, item in enumerate(mapping):
        p = source / 'meshes' / item['mesh']
        if sha(p) != item['sha256']:
            raise ValueError('Source STL hash differs from map')
        vertices, faces, normals = read_stl(p)
        mesh = UsdGeom.Mesh.Define(library, f'/Geometry/mesh_{i:03d}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreateDoubleSidedAttr(False)
        mesh.CreateExtentAttr([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))])
        record = {**item, 'prim_path': str(mesh.GetPath()), 'vertices': len(vertices), 'triangles': len(faces), 'normal_array_sha256': hashlib.sha256(normals.astype('<f4').tobytes()).hexdigest(), 'vertex_array_sha256': hashlib.sha256(vertices.astype('<f4').tobytes()).hexdigest(), 'triangle_index_sha256': hashlib.sha256(faces.astype('<i4').tobytes()).hexdigest(), 'bounds_m': [vertices.min(0).tolist(), vertices.max(0).tolist()]}
        meshes.append(record)
    library.GetRootLayer().Save()
    library = None
    meshinfo = {x['mesh']: x for x in meshes}
    recipe = collision_metadata(meshes, d['parts'])
    selected = {x['mesh']: x for x in recipe if x['selected_for_collision']}
    save(output / 'geometry_manifest.json', meshes)
    save(output / 'collision_recipe.json', recipe)
    stage = Usd.Stage.CreateNew(str(output / 'robot.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Robot')
    stage.SetDefaultPrim(root.GetPrim())
    set_transform(root.GetPrim(), transform([0, 0, d['root_height_m']], np.eye(3)))
    stage.GetRootLayer().customLayerData = {'preparation_status': 'offline prepared, native cooking/physics pending', 'model_sha256': sha(model_path), 'urdf_sha256': sha(urdf), 'floating_base': True, 'material_contact_properties': 'unmeasured; no physics material authored', 'mass_lineage': str(d.get('mass_basis', 'raw CAD part mass/inertia; see builder provenance')), 'coordinate_convention': 'metres kilograms Z-up; joints local+Z; joint limits stored degrees in USD'}
    frames = fk_model(d)
    for link in d['links']:
        body = UsdGeom.Xform.Define(stage, '/Robot/' + link['name']).GetPrim()
        set_transform(body, frames[link['name']])
        rb = UsdPhysics.RigidBodyAPI.Apply(body)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(body)
        moments, q = principal(np.asarray(link['inertia']))
        mass.CreateMassAttr(link['mass'])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*link['com']))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
        mass.CreatePrincipalAxesAttr(q)
        if link['name'] == 'body':
            UsdPhysics.ArticulationRootAPI.Apply(body)
            schema(body, 'PhysxArticulationAPI')
            attr(body, 'physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, True)
    for part in d['parts']:
        mesh = meshinfo[part['mesh']]
        base = f"/Robot/{part['link']}"
        vis = UsdGeom.Mesh.Define(stage, base + f"/visuals/part_{part['id']:04d}")
        vis.GetPrim().GetReferences().AddReference('./geometry.usdc', mesh['prim_path'])
        set_transform(vis.GetPrim(), from_record(part))
        vis.CreatePurposeAttr(UsdGeom.Tokens.render)
        vis.CreateDisplayColorAttr([Gf.Vec3f(*part['color'][:3])])
        vis.CreateDisplayOpacityAttr([float(part['color'][3])])
        attr(vis.GetPrim(), 'hexapod:sourcePartId', Sdf.ValueTypeNames.Int, part['id'])
        attr(vis.GetPrim(), 'hexapod:sourceMesh', Sdf.ValueTypeNames.String, part['mesh'])
        if part['mesh'] not in selected:
            continue
        coll = UsdGeom.Mesh.Define(stage, base + f"/collisions/part_{part['id']:04d}")
        coll.GetPrim().GetReferences().AddReference('./geometry.usdc', mesh['prim_path'])
        set_transform(coll.GetPrim(), from_record(part))
        coll.CreatePurposeAttr(UsdGeom.Tokens.guide)
        coll.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim = coll.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
        schema(prim, 'PhysxSDFMeshCollisionAPI')
        for name, t, value in [('sdfResolution', Sdf.ValueTypeNames.Int, selected[part['mesh']]['resolution']), ('sdfSubgridResolution', Sdf.ValueTypeNames.Int, 6), ('sdfBitsPerSubgridPixel', Sdf.ValueTypeNames.Token, 'BitsPerPixel32'), ('sdfNarrowBandThickness', Sdf.ValueTypeNames.Float, 0.01), ('sdfMargin', Sdf.ValueTypeNames.Float, 0.01), ('sdfEnableRemeshing', Sdf.ValueTypeNames.Bool, False), ('sdfTriangleCountReductionFactor', Sdf.ValueTypeNames.Float, 1.0)]:
            attr(prim, 'physxSDFMeshCollision:' + name, t, value, True)
        schema(prim, 'PhysxCollisionAPI')
        attr(prim, 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, 0.0)
        attr(prim, 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, 0.001)
        attr(prim, 'hexapod:sourcePartId', Sdf.ValueTypeNames.Int, part['id'])
        attr(prim, 'hexapod:sourceMesh', Sdf.ValueTypeNames.String, part['mesh'])
    for j in d['joints']:
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/Robot/joints/' + j['name'])
        prim = joint.GetPrim()
        joint.CreateBody0Rel().SetTargets(['/Robot/' + j['parent']])
        joint.CreateBody1Rel().SetTargets(['/Robot/' + j['child']])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*j['xyz']))
        joint.CreateLocalRot0Attr(gf_quat(quat_matrix(j['quaternion_xyzw'])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateAxisAttr(UsdPhysics.Tokens.z)
        joint.CreateLowerLimitAttr(math.degrees(j['lower']))
        joint.CreateUpperLimitAttr(math.degrees(j['upper']))
        joint.CreateJointEnabledAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        schema(prim, 'PhysxJointAPI')
        attr(prim, 'physxJoint:maxJointVelocity', Sdf.ValueTypeNames.Float, math.degrees(velocity))
        attr(prim, 'hexapod:effortLimitNm', Sdf.ValueTypeNames.Double, effort)
        attr(prim, 'hexapod:velocityLimitRadPerSec', Sdf.ValueTypeNames.Double, velocity)
        attr(prim, 'hexapod:driveStatus', Sdf.ValueTypeNames.String, 'No drive gains or DriveAPI authored. External calibrated actuator runtime must enforce torque-speed/thermal limits; effort metadata is not an applied effort cap.')
    stage.GetRootLayer().Save()
    stage = None
    if any((sha(Path(p)) != h for p, h in initial_hashes.items())):
        raise ValueError('Source changed during preparation; output is not qualified and must not be adopted')
    report = audit(output)
    save(output / 'offline_audit.json', report)
    save(output / 'preparation_manifest.json', {'status': 'offline prepared, native cooking/physics pending', 'pass_offline_audit': report['pass'], 'native_PhysxSchema_available': bool(importlib.util.find_spec('pxr.PhysxSchema')), 'source_hashes': {'model.json': sha(model_path), 'source.urdf': sha(urdf), 'mesh_mapping.json': sha(source / 'mesh_mapping.json')}, 'files': {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}, 'official_sources': SOURCES, 'selected_collision_instances': sum((x['instances'] for x in recipe if x['selected_for_collision'])), 'omitted_collision_instances': sum((x['instances'] for x in recipe if not x['selected_for_collision'])), 'nominal_motor_effort_metadata_nm': effort, 'nominal_motor_velocity_limit_rad_s': velocity, 'actuator_validation': 'No gains, armature, friction, drive damping or torque-speed/thermal envelope identified or qualified. Effort metadata does not constrain arbitrary applied torques.', 'collision_status': 'Exact source meshes retained; SDF is an uncooked GPU recipe; no CPU plain-trimesh fallback; no duplicate cap collider.'})
    return report

def audit(bundle):
    """Read the saved USD and independently compare against URDF/RPY and hashes."""
    stage = Usd.Stage.Open(str(bundle / 'robot.usda'))
    d = json.loads((bundle / 'source/model.json').read_text())
    root = ET.parse(bundle / 'source/source.urdf').getroot()
    manifest = json.loads((bundle / 'geometry_manifest.json').read_text())
    meshinfo = {x['mesh']: x for x in manifest}
    errors = []
    checks = {}
    links = {l.get('name'): l for l in root.findall('link')}
    joints = {j.get('name'): j for j in root.findall('joint')}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    usd_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    visuals = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath())]
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    checks['counts'] = {'bodies': len(bodies), 'revolute_joints': len(usd_joints), 'visuals': len(visuals), 'collisions': len(collisions), 'unique_meshes': len(manifest)}
    if len(bodies) != 19 or len(usd_joints) != 18 or len(visuals) != 1753 or (len(manifest) != 59):
        errors.append('Count mismatch')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        errors.append('Stage unit mismatch')
    expected_frames = {'body': transform([0, 0, d['root_height_m']], np.eye(3))}
    pending = list(joints.values())
    while pending:
        progress = False
        for j in pending[:]:
            par = j.find('parent').get('link')
            child = j.find('child').get('link')
            if par not in expected_frames:
                continue
            expected_frames[child] = expected_frames[par] @ origin(j)
            pending.remove(j)
            progress = True
        if not progress:
            raise ValueError('URDF has disconnected joint graph')
    actual_frames = {p.GetName(): np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T for p in bodies}
    massrows = []
    jointrows = []
    for name, l in links.items():
        p = stage.GetPrimAtPath('/Robot/' + name)
        api = UsdPhysics.MassAPI(p)
        it = l.find('inertial')
        o = origin(it)
        ie = it.find('inertia')
        a = {k: float(v) for k, v in ie.attrib.items()}
        I = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
        I = o[:3, :3] @ I @ o[:3, :3].T
        q = Gf.Quatd(api.GetPrincipalAxesAttr().Get())
        rot = Gf.Rotation(q)
        R = np.array([rot.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)]).T
        reconstructed = R @ np.diag(np.array(api.GetDiagonalInertiaAttr().Get())) @ R.T
        err = float(abs(I - reconstructed).max())
        merr = abs(api.GetMassAttr().Get() - float(it.find('mass').get('value')))
        cerr = float(abs(np.array(api.GetCenterOfMassAttr().Get()) - o[:3, 3]).max())
        ferr = float(abs(actual_frames[name] - expected_frames[name]).max())
        okay = np.allclose(I, reconstructed, rtol=1e-05, atol=1e-08) and merr < 1e-06 and (cerr < 1e-07) and (ferr < 1e-07)
        massrows.append({'link': name, 'tensor_error_kg_m2': err, 'mass_error_kg': merr, 'com_error_m': cerr, 'world_frame_max_error': ferr, 'pass': bool(okay)})
        if not okay:
            errors.append('Body roundtrip ' + name)
    for p in usd_joints:
        j = joints[p.GetName()]
        api = UsdPhysics.RevoluteJoint(p)
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        w = []
        for bi in [0, 1]:
            pos = getattr(api, f'GetLocalPos{bi}Attr')().Get()
            q = getattr(api, f'GetLocalRot{bi}Attr')().Get()
            rot = Gf.Rotation(Gf.Quatd(q))
            R = np.array([rot.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)]).T
            w.append(actual_frames[parent if bi == 0 else child] @ transform(pos, R))
        mismatch = float(abs(w[0] - w[1]).max())
        limit = j.find('limit')
        limiterror = max(abs(api.GetLowerLimitAttr().Get() - math.degrees(float(limit.get('lower')))), abs(api.GetUpperLimitAttr().Get() - math.degrees(float(limit.get('upper')))))
        axis_ok = api.GetAxisAttr().Get() == 'Z' and np.allclose(vec(j.find('axis').get('xyz')), [0, 0, 1])
        okay = mismatch < 1e-06 and limiterror < 1e-05 and axis_ok
        jointrows.append({'joint': p.GetName(), 'world_joint_frame_error': mismatch, 'limit_error_degrees': limiterror, 'axis_Z': bool(axis_ok), 'pass': bool(okay)})
        if not okay:
            errors.append('Joint frames/limits ' + p.GetName())
    library = Usd.Stage.Open(str(bundle / 'geometry.usdc'))
    vertices_cache = {}
    for item in manifest:
        m = UsdGeom.Mesh.Get(library, item['prim_path'])
        v = np.array(m.GetPointsAttr().Get(), dtype='<f4')
        f = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype='<i4')
        counts = np.array(m.GetFaceVertexCountsAttr().Get())
        normals = np.array(m.GetNormalsAttr().Get(), dtype='<f4')
        vertices_cache[item['mesh']] = v
        if hashlib.sha256(v.tobytes()).hexdigest() != item['vertex_array_sha256'] or hashlib.sha256(f.tobytes()).hexdigest() != item['triangle_index_sha256'] or hashlib.sha256(normals.tobytes()).hexdigest() != item['normal_array_sha256'] or (not np.all(counts == 3)):
            errors.append('Source geometry array mismatch ' + item['mesh'])
    parts_by_name = {f"part_{p['id']:04d}": p for p in d['parts']}
    bounds = []
    foot_bounds = []
    max_vis_transform = 0
    urdf_visuals = {(l.get('name'), v.get('name')): v for l in links.values() for v in l.findall('visual')}
    for prim in visuals:
        name = prim.GetName()
        part = parts_by_name[name]
        l = part['link']
        u = urdf_visuals[l, name]
        T = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        expected = expected_frames[l] @ origin(u)
        err = float(abs(T - expected).max())
        max_vis_transform = max(max_vis_transform, err)
        if err > 1e-07:
            errors.append('Visual placement mismatch ' + name)
        mesh = prim.GetAttribute('hexapod:sourceMesh').Get()
        expected_mesh = Path(u.find('geometry/mesh').get('filename')).name
        refs = prim.GetMetadata('references').GetAddedOrExplicitItems()
        if mesh != expected_mesh or len(refs) != 1 or refs[0].assetPath != './geometry.usdc' or (str(refs[0].primPath) != meshinfo[expected_mesh]['prim_path']):
            errors.append('Visual reference mismatch ' + name)
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype='<f4')
        indices = np.array(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype='<i4')
        if hashlib.sha256(points.tobytes()).hexdigest() != meshinfo[expected_mesh]['vertex_array_sha256'] or hashlib.sha256(indices.tobytes()).hexdigest() != meshinfo[expected_mesh]['triangle_index_sha256']:
            errors.append('Composed visual geometry mismatch ' + name)
        verts = vertices_cache[mesh]
        world = verts @ T[:3, :3].T + T[:3, 3]
        bounds.append([world.min(0), world.max(0)])
        if mesh == 'mesh_056_tibia.stl':
            foot_bounds.append({'link': l, 'min_z_m': float(world[:, 2].min()), 'max_z_m': float(world[:, 2].max())})
    expected_collider_ids = {p['id'] for p in d['parts'] if p['mesh'] in SDF_SELECTION}
    actual_collider_ids = {p.GetAttribute('hexapod:sourcePartId').Get() for p in collisions}
    if actual_collider_ids != expected_collider_ids or len(collisions) != len(expected_collider_ids):
        errors.append('Selected collision membership differs')
    for p in collisions:
        if UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != 'sdf' or 'PhysxSDFMeshCollisionAPI' not in p.GetMetadata('apiSchemas').GetAddedOrExplicitItems():
            errors.append('Missing native SDF recipe ' + str(p.GetPath()))
        mesh = p.GetAttribute('hexapod:sourceMesh').Get()
        part = next((x for x in d['parts'] if x['id'] == p.GetAttribute('hexapod:sourcePartId').Get()))
        expected = expected_frames[part['link']] @ origin(urdf_visuals[part['link'], f"part_{part['id']:04d}"])
        actual = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        if mesh != part['mesh'] or not np.allclose(expected, actual, atol=1e-07, rtol=0) or p.GetAttribute('physxSDFMeshCollision:sdfResolution').Get() != SDF_SELECTION[mesh][0]:
            errors.append('Collider geometry/recipe mismatch ' + str(p.GetPath()))
        points = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype='<f4')
        indices = np.array(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get(), dtype='<i4')
        if hashlib.sha256(points.tobytes()).hexdigest() != meshinfo[mesh]['vertex_array_sha256'] or hashlib.sha256(indices.tobytes()).hexdigest() != meshinfo[mesh]['triangle_index_sha256']:
            errors.append('Composed collision geometry mismatch ' + str(p.GetPath()))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(bundle / 'robot.usda'))
    dependencies = [Path(l.realPath).resolve() for l in layers if l.realPath] + [Path(a).resolve() for a in assets]
    if unresolved or any((not p.is_relative_to(bundle.resolve()) for p in dependencies)):
        errors.append('Nonportable/unresolved dependency')
    articulation = [p for p in bodies if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation) != 1 or articulation[0].GetName() != 'body' or any((p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())):
        errors.append('Floating articulation root incorrect')
    arr = np.array(bounds)
    checks.update(body_properties=massrows, joint_frames=jointrows, visual_max_transform_error=max_vis_transform, world_visual_bounds_m=[arr[:, 0].min(0).tolist(), arr[:, 1].max(0).tolist()], foot_geometry_bounds=foot_bounds, root_placement_z_m=d['root_height_m'], source_geometry_hashes_match=not any(('geometry array' in e for e in errors)), dependencies=[p.relative_to(bundle.resolve()).as_posix() for p in sorted(set(dependencies))], floating_base=True)
    return {'status': 'offline prepared, native cooking/physics pending', 'pass': not errors, 'errors': errors, 'checks': checks, 'numpy_version': np.__version__, 'openusd_version': list(Usd.GetVersion()), 'qualifies_isaac_runtime': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--source', type=Path, required=True)
    create.add_argument('--urdf', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--model', type=Path)
    validate = sub.add_parser('audit')
    validate.add_argument('bundle', type=Path)
    a = p.parse_args()
    r = prepare(a.source, a.urdf, a.output, a.model) if a.mode == 'prepare' else audit(a.bundle.resolve())
    print(json.dumps({'status': r['status'], 'pass': r['pass'], 'errors': r['errors'], 'counts': r['checks']['counts'], 'world_visual_bounds_m': r['checks']['world_visual_bounds_m']}, indent=2))
    return 0 if r['pass'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
