"""Independent geometry checks shared by CPU and Isaac terrain admission.

The stored triangle meshes are authoritative; no height-field interpolation or
convex hull is substituted. USD imports are lazy so most checks run without USD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from terrain_readiness import validate_mesh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path, fixture_ids=None):
    path = Path(path).resolve()
    catalog = json.loads(path.read_text())
    entries = catalog["fixtures"]
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate fixture IDs")
    if fixture_ids is not None:
        requested = set(fixture_ids)
        missing = requested - {e["id"] for e in entries}
        if missing:
            raise ValueError(f"Unknown fixture IDs: {sorted(missing)}")
        entries = [e for e in entries if e["id"] in requested]
    if not entries:
        raise ValueError("Select at least one fixture")
    result = []
    for entry in entries:
        usd = (path.parent / entry["usda"]).resolve()
        if not usd.is_relative_to(path.parent) or digest(usd) != entry["sha256"]:
            raise ValueError(f"Fixture path/hash mismatch: {entry['id']}")
        with np.load(usd.with_suffix(".npz"), allow_pickle=False) as data:
            vertices, faces = data["vertices"].copy(), data["faces"].copy()
        summary = validate_mesh(vertices, faces)
        if any(summary[k] != entry[k] for k in ("vertices", "triangles")):
            raise ValueError(f"Fixture count mismatch: {entry['id']}")
        result.append((entry, usd, vertices, faces))
    return result


def vertical_surface_heights(vertices, faces, xy, *, origin_z=2.0):
    """Highest downward triangle hit, including negative heights; miss is NaN.

    Barycentric intersection with the XY projection deliberately excludes
    vertical faces. It is independent of the fixture generator and USD parser.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite Nx2 sample locations")
    tri = np.asarray(vertices, dtype=float)[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u, v = b - a, c - a
    denominator = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    eligible = np.abs(denominator) > 1e-12
    a, u, v, denominator = a[eligible], u[eligible], v[eligible], denominator[eligible]
    heights = np.full(len(xy), np.nan)
    for i, location in enumerate(xy):
        q = location - a[:, :2]
        beta = (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]) / denominator
        gamma = (u[:, 0] * q[:, 1] - u[:, 1] * q[:, 0]) / denominator
        z = a[:, 2] + beta * u[:, 2] + gamma * v[:, 2]
        inside = (beta >= -1e-8) & (gamma >= -1e-8) & (beta + gamma <= 1 + 1e-8) & (z <= origin_z)
        if inside.any():
            heights[i] = z[inside].max()
    return heights


def collision_meshes(stage, root_path):
    """Return exact static triangle collision prims or reject unsafe import."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Missing terrain prim {root_path}")
    meshes = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Plane) or prim.GetTypeName() == "Plane":
            raise ValueError("Terrain import contains a ground plane")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Fixture terrain must remain static")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError("Unexpected non-mesh terrain collider")
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise ValueError("Disabled terrain collider")
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                raise ValueError("Terrain lacks explicit triangle collision schema")
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none":
                raise ValueError("Terrain convexification would change support geometry")
            meshes.append(prim)
    if len(meshes) != 1:
        raise ValueError(f"Expected one authoritative terrain mesh, got {len(meshes)}")
    return meshes


def audit_usd(usd, vertices, faces):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        raise ValueError(f"USD could not open: {usd}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Terrain must be Z up with metre units")
    prim = collision_meshes(stage, stage.GetDefaultPrim().GetPath())[0]
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3) or not np.array_equal(indices, faces.reshape(-1)):
        raise ValueError("USD triangle topology differs from NPZ")
    if points.shape != vertices.shape or not np.allclose(points, vertices, atol=1e-7, rtol=0):
        raise ValueError("USD geometry differs from NPZ")
    if UsdGeom.Xformable(prim).GetOrderedXformOps():
        raise ValueError("Source fixture has an unexpected transform")
    return {"mesh_path": str(prim.GetPath()), "max_vertex_error_m": float(np.max(np.abs(points - vertices))),
            "usd_sha256": digest(usd), "npz_sha256": digest(Path(usd).with_suffix(".npz")),
            "collision_approximation": "none", "meters_per_unit": 1, "up_axis": "Z"}


def bind_fixture_material(stage, prim, *, friction=1.0, contact_offset_m=0.001):
    """Author runtime overrides only; TerrainImporter ignores USD material cfg."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    if not np.isfinite([friction, contact_offset_m]).all() or friction <= 0 or not 0 < contact_offset_m <= .002:
        raise ValueError("Explicit positive friction and <=2 mm contact offset required")
    material_path = str(prim.GetPath()) + "/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(friction))
    physics.CreateDynamicFrictionAttr(float(friction))
    physics.CreateRestitutionAttr(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr("multiply")
    physx_material.CreateRestitutionCombineModeAttr("multiply")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(float(contact_offset_m))
    collision.CreateRestOffsetAttr(0.0)


def probe_locations(entry):
    """One common flat-start probe and two terrain feature probes."""
    if entry["family"] == "pit":
        return np.array([[-1., -.65], [0., 0.], [.7, .65]])
    if entry["family"] == "ridge":
        return np.array([[-1., -.65], [0., 0.], [.7, .65]])
    return np.array([[-1., -.65], [.5, 0.], [.9, .65]])
