#!/usr/bin/env python3
"""Repair and verify a CAD USD on the CPU, without starting Isaac Sim.

The output is a new portable bundle: byte-preserved imported layers under
``source/`` and one stronger layer containing URDF-derived inertia and limits.
No installed converter, input USD, URDF, or archived checkpoint is modified.

Requires NumPy and OpenUSD (``uv run python tools/prepare_mkii_usd.py --help``).
The principal-axis quaternion is regenerated from the URDF tensor; an existing
quaternion is never blindly inverted. See OpenUSD MassAPI and GfMatrix3d docs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

INERTIA_RTOL = 1e-5
INERTIA_ATOL = 1e-8
IGNORED_REPAIR_ATTRIBUTES = {"physics:diagonalInertia", "physics:principalAxes",
                             "physics:lowerLimit", "physics:upperLimit"}


@dataclass(frozen=True)
class Inertial:
    mass: float
    com: np.ndarray
    tensor: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw; columns act on vectors."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def vector(text: str) -> np.ndarray:
    result = np.array([float(value) for value in text.split()])
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite three-vector, got {text!r}")
    return result


def read_inertials(urdf: Path) -> dict[str, Inertial]:
    result = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        name = link.get("name")
        if not name or name in result:
            raise ValueError(f"Missing or duplicate URDF link name: {name!r}")
        inertial = link.find("inertial")
        if inertial is None or len(link.findall('inertial')) != 1:
            raise ValueError(f"{name}: explicit URDF inertial is required")
        mass_element, tensor_element = inertial.find('mass'), inertial.find('inertia')
        if mass_element is None or mass_element.get('value') is None or tensor_element is None:
            raise ValueError(f"{name}: inertial requires explicit mass and inertia children")
        if not {'ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'}.issubset(tensor_element.attrib):
            raise ValueError(f"{name}: inertia requires all six tensor components")
        mass = float(mass_element.get("value"))
        values = {key: float(value) for key, value in tensor_element.attrib.items()}
        tensor = np.array([[values['ixx'], values['ixy'], values['ixz']],
                           [values['ixy'], values['iyy'], values['iyz']],
                           [values['ixz'], values['iyz'], values['izz']]])
        if not math.isfinite(mass) or mass <= 0 or not np.isfinite(tensor).all():
            raise ValueError(f"{name}: nonpositive mass or nonfinite inertial")
        moments = np.linalg.eigvalsh(tensor)
        if moments[0] <= 0 or moments[2] > moments[0] + moments[1] + 1e-12:
            raise ValueError(f"{name}: inertia must be positive definite and satisfy the moment triangle inequality")
        origin = inertial.find("origin")
        com = vector(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = vector(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        rotation = rpy_matrix(rpy)
        # Both tensors are at the COM. Translating COM is not a parallel-axis shift.
        result[name] = Inertial(mass, com, rotation @ tensor @ rotation.T)
    if not result:
        raise ValueError("URDF has no links")
    return result


def principal_axes(tensor: np.ndarray) -> tuple[np.ndarray, Gf.Quatf]:
    moments, columns = np.linalg.eigh(tensor)
    # Stabilize sign choices, then make the eigenbasis a proper rotation.
    for index in range(3):
        if columns[np.argmax(np.abs(columns[:, index])), index] < 0:
            columns[:, index] *= -1
    if np.linalg.det(columns) < 0:
        columns[:, -1] *= -1
    # NumPy eigenvectors are columns; Gf transformation matrices act on rows.
    rotation = Gf.Matrix3d(*map(float, columns.T.flat)).ExtractRotation().GetQuat()
    if rotation.GetReal() < 0:
        rotation = -rotation
    quaternion = Gf.Quatf(rotation).GetNormalized()
    reconstructed = tensor_from_usd(moments, quaternion)
    if not np.allclose(reconstructed, tensor, rtol=INERTIA_RTOL, atol=INERTIA_ATOL):
        raise ValueError("Principal-axis conversion failed its numerical round trip")
    return moments, quaternion


def tensor_from_usd(moments, quaternion) -> np.ndarray:
    """Independent Gf vector transform checks the authoring matrix convention."""
    rotation = Gf.Rotation(Gf.Quatd(quaternion))
    axes = np.array([rotation.TransformDir(Gf.Vec3d(*axis)) for axis in np.eye(3)]).T
    return axes @ np.diag(np.asarray(moments, dtype=float)) @ axes.T


def rigid_bodies(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    result = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetName() in result:
                raise ValueError(f"Duplicate USD rigid-body name: {prim.GetName()}")
            result[prim.GetName()] = prim
    return result


def mass_comparison(stage: Usd.Stage, inertials: dict[str, Inertial]) -> dict:
    bodies = rigid_bodies(stage)
    errors = []
    if set(bodies) != set(inertials):
        errors.append(f"Rigid-body membership differs: missing={sorted(set(inertials)-set(bodies))}, unexpected={sorted(set(bodies)-set(inertials))}")
    rows = []
    for name in sorted(set(bodies) & set(inertials)):
        body, expected = bodies[name], inertials[name]
        api = UsdPhysics.MassAPI(body)
        required = (api.GetMassAttr(), api.GetCenterOfMassAttr(), api.GetDiagonalInertiaAttr(), api.GetPrincipalAxesAttr())
        if not body.HasAPI(UsdPhysics.MassAPI) or any(not attr or attr.Get() is None for attr in required):
            errors.append(f"{name}: explicit MassAPI and mass/COM/inertia attributes are required")
            rows.append({"link": name, "mass_ok": False, "com_ok": False, "inertia_ok": False,
                         "max_abs_tensor_error_kg_m2": None})
            continue
        mass = api.GetMassAttr().Get()
        com = np.array(api.GetCenterOfMassAttr().Get(), dtype=float)
        moments = np.array(api.GetDiagonalInertiaAttr().Get(), dtype=float)
        q = api.GetPrincipalAxesAttr().Get()
        mass_ok = mass is not None and math.isfinite(mass) and math.isclose(mass, expected.mass, rel_tol=1e-6, abs_tol=1e-7)
        com_ok = np.isfinite(com).all() and np.allclose(com, expected.com, rtol=1e-6, atol=1e-7)
        q_values = np.array([q.GetReal(), *q.GetImaginary()])
        axes_ok = np.isfinite(q_values).all() and abs(np.linalg.norm(q_values)-1) < 1e-5
        moments_ok = np.isfinite(moments).all() and bool((moments > 0).all())
        error = None
        inertia_ok = False
        if axes_ok and moments_ok:
            actual = tensor_from_usd(moments, q)
            error = float(np.max(np.abs(actual-expected.tensor)))
            inertia_ok = bool(np.allclose(actual, expected.tensor, rtol=INERTIA_RTOL, atol=INERTIA_ATOL))
        row = {"link": name, "mass_ok": bool(mass_ok), "com_ok": bool(com_ok),
               "inertia_ok": inertia_ok, "max_abs_tensor_error_kg_m2": error}
        rows.append(row)
        if not all((mass_ok, com_ok, inertia_ok)):
            errors.append(f"{name}: mass={mass_ok}, COM={com_ok}, inertia={inertia_ok}")
    return {"errors": errors, "links": rows}


def check_stage_units_and_scale(stage: Usd.Stage) -> None:
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError("CAD preparation requires metres, kilograms and Z-up")
    for name, prim in rigid_bodies(stage).items():
        matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
        linear = matrix[:3, :3]
        if (not np.isfinite(matrix).all() or not np.allclose(linear @ linear.T, np.eye(3), atol=1e-7, rtol=0)
                or not math.isclose(float(np.linalg.det(linear)), 1.0, abs_tol=1e-7)):
            raise ValueError(f"{name}: scaled, reflected, sheared or nonfinite rigid-body frame is unsupported")


def dependencies(path: Path, bundle_root: Path | None = None) -> list[dict]:
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    if unresolved:
        raise ValueError(f"Unresolved USD assets: {list(unresolved)}")
    files = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    files.update(Path(asset).resolve() for asset in assets)
    result = []
    for file in sorted(files):
        if not file.is_file():
            raise ValueError(f"Missing USD dependency: {file}")
        if bundle_root is not None and not file.is_relative_to(bundle_root.resolve()):
            raise ValueError(f"USD dependency escapes the portable bundle: {file}")
        relative = str(file.relative_to(bundle_root.resolve())) if bundle_root else file.name
        result.append({"path": relative, "sha256": sha256(file)})
    return result


def non_inertia_digest(stage: Usd.Stage) -> str:
    """Compare composed transforms, joint properties, collision data and schemas.

    Byte-preserved dependency hashes cover shared mesh prototypes as well.
    Asset values use the authored path, not machine-specific resolved paths.
    """
    digest = hashlib.sha256()
    for prim in stage.Traverse():
        digest.update(f"{prim.GetPath()}:{prim.GetTypeName()}:{prim.GetMetadata('apiSchemas')}".encode())
        for attribute in sorted(prim.GetAttributes(), key=lambda value: value.GetName()):
            if attribute.GetName() in IGNORED_REPAIR_ATTRIBUTES:
                continue
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath):
                value = value.path
            digest.update(f"{attribute.GetName()}:{value}".encode())
        for rel in sorted(prim.GetRelationships(), key=lambda value: value.GetName()):
            digest.update(f"{rel.GetName()}:{rel.GetTargets()}".encode())
    return digest.hexdigest()


def validate(urdf: Path, usd: Path, *, portable: bool = True) -> dict:
    from usd_geometry_audit import validate_geometry

    stage = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadAll)
    if not stage:
        raise ValueError(f"Cannot open USD: {usd}")
    check_stage_units_and_scale(stage)
    inertials = read_inertials(urdf)
    geometry = validate_geometry(urdf, stage)
    mass = mass_comparison(stage, inertials)
    deps = dependencies(usd, usd.parent if portable else None)
    errors = geometry['errors'] + mass['errors']
    return {"pass": not errors, "errors": errors, "urdf_sha256": sha256(urdf),
            "usd_root_sha256": sha256(usd), "mass_properties": mass,
            "geometry": geometry, "dependencies": deps,
            "usd_version": list(Usd.GetVersion()), "numpy_version": np.__version__,
            "inertia_tolerance": {"relative": INERTIA_RTOL, "absolute_kg_m2": INERTIA_ATOL}}


def prepare(urdf: Path, source: Path, output: Path) -> dict:
    from usd_geometry_audit import validate_geometry

    urdf, source, output = urdf.resolve(), source.resolve(), output.resolve()
    if output.suffix not in ('.usd', '.usda') or output.exists():
        raise ValueError("Choose a new .usd/.usda output; existing artifacts are never overwritten")
    if output.parent.exists() and any(output.parent.iterdir()):
        raise ValueError("Output bundle directory must be absent or empty")
    if output.parent == source.parent or output.parent.is_relative_to(source.parent):
        raise ValueError("Output must be outside the imported source directory")
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if not source_stage:
        raise ValueError(f"Cannot open source USD: {source}")
    if source_stage.GetRootLayer().customLayerData.get('hexapod_inertia_repair_version'):
        raise ValueError("Source is already prepared; validate it instead of nesting repairs")
    check_stage_units_and_scale(source_stage)
    inertials = read_inertials(urdf)
    geometry = validate_geometry(urdf, source_stage)
    if geometry['errors']:
        raise ValueError(f"Source geometry does not match URDF: {geometry['errors']}")
    before = mass_comparison(source_stage, inertials)
    if len(before['links']) != len(inertials) or any(not (row['mass_ok'] and row['com_ok']) for row in before['links']):
        raise ValueError("Source body membership, mass or COM mismatch; inertia repair cannot hide a frame/import defect")
    source_deps = dependencies(source, source.parent)
    expected_digest = non_inertia_digest(source_stage)
    output.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mkii-usd-', dir=output.parent.parent) as temporary:
        work = Path(temporary)
        raw = work / 'source'
        raw.mkdir()
        # Copy the whole self-contained import, including inactive physics variants.
        # A symlink would make source provenance and portability ambiguous.
        for file in source.parent.rglob('*'):
            if file.is_symlink():
                raise ValueError(f"Source bundle contains a symlink: {file}")
        shutil.copytree(source.parent, raw, dirs_exist_ok=True)
        for record in source_deps:
            if sha256(raw / record['path']) != record['sha256']:
                raise ValueError("Source changed while copying")
        target = work / output.name
        layer = Sdf.Layer.CreateNew(str(target))
        layer.subLayerPaths = [(Path('source') / source.name).as_posix()]
        layer.customLayerData = {'hexapod_inertia_repair_version': 2,
                                 'urdf_sha256': sha256(urdf),
                                 'source_usd_sha256': sha256(source)}
        layer.Save()
        stage = Usd.Stage.Open(str(target), load=Usd.Stage.LoadAll)
        root = source_stage.GetDefaultPrim()
        if not root:
            raise ValueError("Source USD must have a valid default prim")
        stage.SetDefaultPrim(stage.GetPrimAtPath(root.GetPath()))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
        for name, prim in rigid_bodies(stage).items():
            moments, quaternion = principal_axes(inertials[name].tensor)
            api = UsdPhysics.MassAPI(prim)
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
            api.CreatePrincipalAxesAttr(quaternion)
        limits = {joint.get('name'): joint.find('limit') for joint in ET.parse(urdf).getroot().findall('joint')}
        # USD stores revolute limits in degrees. Re-author from the URDF to
        # remove converter decimal truncation before deriving runtime soft limits.
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                limit = limits[prim.GetName()]
                joint = UsdPhysics.RevoluteJoint(prim)
                joint.CreateLowerLimitAttr(math.degrees(float(limit.get('lower'))))
                joint.CreateUpperLimitAttr(math.degrees(float(limit.get('upper'))))
        stage.GetRootLayer().Save()
        if non_inertia_digest(stage) != expected_digest:
            raise ValueError("Repair changed a composed non-inertia property")
        # Reopen after serialization: float rounding and layer composition are tested.
        del stage
        result = validate(urdf, target)
        if not result['pass']:
            raise ValueError(f"Corrected USD did not pass: {result['errors']}")
        result['source_inertia_failures'] = [row['link'] for row in before['links'] if not row['inertia_ok']]
        result['source_dependencies'] = source_deps
        result['non_inertia_semantic_sha256'] = expected_digest
        result['tools_sha256'] = {name: sha256(Path(__file__).with_name(name))
                                  for name in ('prepare_mkii_usd.py', 'usd_geometry_audit.py')}
        result['live_isaac_validation'] = 'not_run'
        (work / 'asset_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        files = sorted(file for file in work.rglob('*') if file.is_file())
        (work / 'SHA256SUMS').write_text(''.join(f"{sha256(file)}  {file.relative_to(work).as_posix()}\n" for file in files))
        if output.parent.exists():
            output.parent.rmdir()  # only an already-checked empty destination
        work.rename(output.parent)
    relocated = validate(urdf, output)
    if not relocated['pass'] or relocated['dependencies'] != result['dependencies']:
        raise ValueError("Relocated USD bundle failed validation or dependency hashes changed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdf', type=Path)
    parser.add_argument('usd', type=Path, help='source bundle root for prepare, corrected root for --check')
    parser.add_argument('output', nargs='?', type=Path, help='new root USD in an absent/empty bundle directory')
    parser.add_argument('--check', action='store_true', help='validate without writing')
    args = parser.parse_args()
    try:
        if args.check:
            if args.output is not None:
                parser.error('--check does not accept output')
            result = validate(args.urdf, args.usd)
        else:
            if args.output is None:
                parser.error('prepare requires output')
            result = prepare(args.urdf, args.usd, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['pass'] else 1
    except (ValueError, RuntimeError) as error:
        print(json.dumps({'pass': False, 'errors': [str(error)]}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
