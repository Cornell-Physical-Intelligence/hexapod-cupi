"""CPU-only URDF/USD articulation and collision-geometry regression checks.

``validate_geometry(urdf_path, stage)`` accepts an already opened OpenUSD stage.
It checks the exact named graph from the URDF, zero-pose frames, positive joint
axes, anchors, limits, and the physical shapes of all collision primitives.
No Isaac Sim application is started. Visual mesh fidelity is outside this gate.
Rigid bodies must have unit scale; primitive scale is included in dimensions.
The 0.1 mrad body/joint and 0.2 mrad collision angular tolerances accommodate
documented converter small-angle quantization. Every collider also has a 0.1 mm
bound on surface displacement, matching the CAD registration criterion. These
are explicit import prequalification tolerances, not claims of exact geometry.
"""

from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


TOLERANCES = {
    "body_translation_m": 1e-4,
    "body_rotation_rad": 1e-4,
    "joint_anchor_m": 1e-5,
    "joint_axis_rad": 1e-4,
    "joint_frame_rad": 1e-4,
    "joint_limit_rad": 1e-6,
    "collision_translation_m": 1e-5,
    "collision_rotation_rad": 2e-4,
    "collision_dimension_m": 1e-6,
    "collision_surface_displacement_bound_m": 1e-4,
    "collision_world_surface_displacement_bound_m": 1e-4,
    "rotation_orthogonality": 1e-6,
}


def _vec(text, default="0 0 0"):
    result = np.array([float(x) for x in (text or default).split()])
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("expected three finite numbers")
    return result


def _origin(element):
    transform = np.eye(4)
    if element is None:
        return transform
    x, y, z = _vec(element.get("rpy"))
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    transform[:3, :3] = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    transform[:3, 3] = _vec(element.get("xyz"))
    return transform


def _matrix(matrix):
    # Gf uses row vectors; all NumPy transforms here act on column vectors.
    return np.asarray(matrix, dtype=float).T


def _rotation(quaternion):
    components = np.array([quaternion.GetReal(), *quaternion.GetImaginary()])
    if not np.all(np.isfinite(components)) or not np.isclose(np.linalg.norm(components), 1, rtol=1e-6, atol=1e-9):
        raise ValueError("joint rotation quaternion must be finite and normalized")
    rotation = Gf.Rotation(quaternion)
    return np.column_stack([
        np.array(rotation.TransformDir(Gf.Vec3d(*axis))) for axis in np.eye(3)
    ])


def _angle(rotation):
    # atan2 retains precision at the small angles checked by this audit.
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return float(np.arctan2(np.linalg.norm(skew) / 2,
                           np.clip((np.trace(rotation) - 1) / 2, -1, 1)))


def _axis_angle(first, second, symmetric=False):
    dot = float(np.dot(first, second))
    if symmetric:
        dot = abs(dot)
    return float(np.arccos(np.clip(dot, -1, 1)))


def _rigid(rotation):
    return (np.all(np.isfinite(rotation))
            and np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            <= TOLERANCES["rotation_orthogonality"]
            and abs(np.linalg.det(rotation) - 1) <= TOLERANCES["rotation_orthogonality"])


def _urdf_shape(collision):
    geometry = collision.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError("collision requires exactly one geometry")
    shape = geometry[0]
    transform = _origin(collision.find("origin"))
    kind = shape.tag
    if kind == "box":
        dimensions = _vec(shape.get("size"))
    elif kind == "cylinder":
        dimensions = np.array([float(shape.get("radius")), float(shape.get("length"))])
    elif kind == "sphere":
        dimensions = np.array([float(shape.get("radius"))])
    else:
        raise ValueError(f"unsupported collision geometry {kind!r}")
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("collision dimensions must be finite and positive")
    return {"kind": kind, "dimensions": dimensions,
            "center": transform[:3, 3], "rotation": transform[:3, :3]}


def _usd_shape(prim, relative_transform):
    basis = relative_transform[:3, :3]
    scales = np.linalg.norm(basis, axis=0)
    if not np.all(np.isfinite(relative_transform)) or np.any(scales <= 0):
        raise ValueError("nonfinite or degenerate collision transform")
    rotation = basis / scales
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > TOLERANCES["rotation_orthogonality"]:
        raise ValueError("sheared collision transform")
    # Reflection has no effect on these centrally symmetric primitive shapes.
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    if prim.IsA(UsdGeom.Cube):
        kind = "box"
        dimensions = scales * float(UsdGeom.Cube(prim).GetSizeAttr().Get())
    elif prim.IsA(UsdGeom.Sphere):
        kind = "sphere"
        if not np.allclose(scales, scales[0], rtol=1e-6, atol=1e-9):
            raise ValueError("nonuniform scale turns sphere into unsupported ellipsoid")
        dimensions = np.array([float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scales[0]])
    elif prim.IsA(UsdGeom.Cylinder):
        kind = "cylinder"
        shape = UsdGeom.Cylinder(prim)
        axis = {"X": 0, "Y": 1, "Z": 2}[str(shape.GetAxisAttr().Get())]
        radial = [i for i in range(3) if i != axis]
        if not np.isclose(scales[radial[0]], scales[radial[1]], rtol=1e-6, atol=1e-9):
            raise ValueError("nonuniform radial scale turns cylinder into unsupported ellipse")
        dimensions = np.array([float(shape.GetRadiusAttr().Get()) * scales[radial[0]],
                               float(shape.GetHeightAttr().Get()) * scales[axis]])
        # Only the cylinder's axis direction affects its physical geometry.
        rotation = rotation[:, [*radial, axis]]
    else:
        raise ValueError(f"unsupported USD collision type {prim.GetTypeName()!r}")
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("collision dimensions must be finite and positive")
    return {"kind": kind, "dimensions": dimensions,
            "center": relative_transform[:3, 3], "rotation": rotation,
            "path": str(prim.GetPath())}


def _shape_errors(expected, actual):
    if expected["kind"] != actual["kind"]:
        return None
    position = float(np.linalg.norm(expected["center"] - actual["center"]))
    if expected["kind"] == "box":
        candidates = []
        # Box axes may be permuted or flipped without changing its shape.
        for permutation in permutations(range(3)):
            dimensions = float(np.max(np.abs(expected["dimensions"] - actual["dimensions"][list(permutation)])))
            for signs in product((-1, 1), repeat=3):
                rotation = actual["rotation"][:, permutation] * np.array(signs)
                if np.linalg.det(rotation) > 0:
                    angle = _angle(expected["rotation"].T @ rotation)
                    candidates.append((max(dimensions / TOLERANCES["collision_dimension_m"],
                                           angle / TOLERANCES["collision_rotation_rad"]), dimensions, angle))
        _, dimensions, angle = min(candidates)
    else:
        dimensions = float(np.max(np.abs(expected["dimensions"] - actual["dimensions"])))
        angle = (_axis_angle(expected["rotation"][:, 2], actual["rotation"][:, 2], True)
                 if expected["kind"] == "cylinder" else 0.0)
    if expected["kind"] == "box":
        # A box point lies no farther than half its diagonal from its center.
        radius = max(np.linalg.norm(expected["dimensions"]), np.linalg.norm(actual["dimensions"])) / 2
        size_bound = dimensions * np.sqrt(3) / 2
    elif expected["kind"] == "cylinder":
        radius = max(np.hypot(shape["dimensions"][0], shape["dimensions"][1] / 2)
                     for shape in (expected, actual))
        differences = np.abs(expected["dimensions"] - actual["dimensions"])
        size_bound = np.hypot(differences[0], differences[1] / 2)
    else:
        radius, size_bound = 0.0, dimensions
    # Triangle inequality gives a conservative correspondence/Hausdorff bound:
    # center displacement + size change + maximal rotational displacement.
    surface_bound = float(position + size_bound + 2 * radius * np.sin(angle / 2))
    return {"collision_translation_m": position, "collision_rotation_rad": angle,
            "collision_dimension_m": dimensions,
            "collision_surface_displacement_bound_m": surface_bound}


def _transform_shape(shape, transform):
    return {**shape, "center": transform[:3, :3] @ shape["center"] + transform[:3, 3],
            "rotation": transform[:3, :3] @ shape["rotation"]}


def validate_geometry(urdf_path: str | Path, stage) -> dict:
    """Return JSON-compatible errors, counts and maximum geometric discrepancies.

    Every collider is matched one-to-one within its owning rigid body, without
    depending on importer-generated primitive names or traversal order. This
    permits physically equivalent sphere/cylinder/box orientations only.
    """
    errors = []
    maxima = {key: 0.0 for key in TOLERANCES if key != "rotation_orthogonality"}
    result = {"errors": errors, "counts": {}, "max_errors": maxima,
              "tolerances": dict(TOLERANCES)}

    def check(metric, value, context):
        if not np.isfinite(value):
            errors.append(f"{context}: nonfinite {metric}")
            return
        maxima[metric] = max(maxima[metric], float(value))
        if value > TOLERANCES[metric]:
            errors.append(f"{context}: {metric}={value:.9g} exceeds {TOLERANCES[metric]:.9g}")

    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    if len(links) != len(root.findall("link")) or None in links:
        errors.append("URDF has duplicate or missing link names")
    if len(joints) != len(root.findall("joint")) or None in joints:
        errors.append("URDF has duplicate or missing joint names")
    parents = {}
    for name, joint in joints.items():
        if joint.get("type") != "revolute" or joint.find("mimic") is not None:
            errors.append(f"{name}: expected independent revolute URDF joint")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if child in parents or parent not in links or child not in links:
            errors.append(f"{name}: invalid URDF parent/child graph")
        parents[child] = (parent, joint)
    roots = set(links) - set(parents)
    if len(roots) != 1:
        errors.append("URDF does not have exactly one root")
        return result
    root_name = next(iter(roots))
    expected_world = {root_name: np.eye(4)}
    for _ in links:
        for child, (parent, joint) in parents.items():
            if parent in expected_world:
                expected_world[child] = expected_world[parent] @ _origin(joint.find("origin"))
    if set(expected_world) != set(links):
        errors.append("URDF graph is disconnected or cyclic")
        return result
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        errors.append("geometry gate requires metresPerUnit=1")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("geometry gate requires Z-up")
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    usd_joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    by_name = {prim.GetName(): prim for prim in bodies}
    by_joint_name = {prim.GetName(): prim for prim in usd_joints}
    result["counts"].update(expected_bodies=len(links), bodies=len(bodies),
                            expected_joints=len(joints), joints=len(usd_joints))
    if len(by_name) != len(bodies) or set(by_name) != set(links):
        errors.append(f"rigid-body names differ: missing={sorted(set(links) - set(by_name))}, extra={sorted(set(by_name) - set(links))}, duplicates={len(bodies)-len(by_name)}")
    if len(by_joint_name) != len(usd_joints) or set(by_joint_name) != set(joints):
        errors.append(f"joint names differ: missing={sorted(set(joints) - set(by_joint_name))}, extra={sorted(set(by_joint_name) - set(joints))}, duplicates={len(usd_joints)-len(by_joint_name)}")
    if root_name not in by_name:
        return result
    articulation_roots = [prim for prim in stage.Traverse()
                          if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    result["counts"]["articulation_roots"] = len(articulation_roots)
    if len(articulation_roots) != 1 or articulation_roots[0] != by_name[root_name]:
        errors.append(f"expected one floating articulation root on {by_name[root_name].GetPath()}")
    for prim in bodies:
        body_api = UsdPhysics.RigidBodyAPI(prim)
        if not body_api.GetRigidBodyEnabledAttr().Get() or body_api.GetKinematicEnabledAttr().Get():
            errors.append(f"{prim.GetPath()}: body must be enabled and dynamic")
    cache = UsdGeom.XformCache()
    world = {name: _matrix(cache.GetLocalToWorldTransform(prim)) for name, prim in by_name.items()}
    checked_ancestors = set()
    for prim in bodies:
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            path = str(ancestor.GetPath())
            if path not in checked_ancestors and UsdGeom.Xformable(ancestor):
                checked_ancestors.add(path)
                xform = UsdGeom.Xformable(ancestor)
                if not _rigid(_matrix(xform.GetLocalTransformation())[:3, :3]):
                    errors.append(f"{path}: body ancestors have scale, reflection, or shear")
                if xform.GetTimeSamples():
                    errors.append(f"{path}: animated body transform is unsupported in static asset gate")
            ancestor = ancestor.GetParent()
    for name, transform in world.items():
        if not np.all(np.isfinite(transform)) or not _rigid(transform[:3, :3]):
            errors.append(f"{name}: rigid body or its ancestors have scale, reflection, shear, or nonfinite transform")
    if any(not _rigid(value[:3, :3]) or not np.all(np.isfinite(value)) for value in world.values()):
        return result
    inverse_root = np.linalg.inv(world[root_name])
    for name in set(links) & set(by_name):
        actual = inverse_root @ world[name]
        check("body_translation_m", np.linalg.norm(actual[:3, 3] - expected_world[name][:3, 3]), name)
        check("body_rotation_rad", _angle(expected_world[name][:3, :3].T @ actual[:3, :3]), name)
    by_path = {str(prim.GetPath()): name for name, prim in by_name.items()}
    for name in set(joints) & set(by_joint_name):
        prim = by_joint_name[name]
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            errors.append(f"{name}: USD joint is not revolute")
            continue
        joint = joints[name]
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        relation_names = []
        for index in (0, 1):
            targets = prim.GetRelationship(f"physics:body{index}").GetTargets()
            relation_names.append(by_path.get(str(targets[0])) if len(targets) == 1 else None)
        if relation_names != [parent, child]:
            errors.append(f"{name}: body0/body1 {relation_names} != URDF {[parent, child]}")
            continue
        if not UsdPhysics.Joint(prim).GetJointEnabledAttr().Get():
            errors.append(f"{name}: USD joint disabled")
        if UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            errors.append(f"{name}: USD joint excluded from articulation")
        frames = []
        for index, body in enumerate((parent, child)):
            transform = np.eye(4)
            transform[:3, 3] = np.array(prim.GetAttribute(f"physics:localPos{index}").Get())
            try:
                transform[:3, :3] = _rotation(prim.GetAttribute(f"physics:localRot{index}").Get())
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
                break
            frames.append(inverse_root @ world[body] @ transform)
        if len(frames) != 2:
            continue
        axis = _vec(joint.find("axis").get("xyz") if joint.find("axis") is not None else "1 0 0")
        if not np.isclose(np.linalg.norm(axis), 1.0, rtol=1e-6, atol=1e-9):
            errors.append(f"{name}: URDF axis is not normalized")
            continue
        expected_axis = expected_world[child][:3, :3] @ axis
        usd_axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())]]
        for index, frame in enumerate(frames):
            check("joint_anchor_m", np.linalg.norm(frame[:3, 3] - expected_world[child][:3, 3]), f"{name} body{index}")
            check("joint_axis_rad", _axis_angle(frame[:3, :3] @ usd_axis, expected_axis), f"{name} body{index} positive axis")
        check("joint_frame_rad", _angle(frames[0][:3, :3].T @ frames[1][:3, :3]), f"{name} zero-pose frames")
        limit = joint.find("limit")
        for side in ("lower", "upper"):
            check("joint_limit_rad", abs(np.deg2rad(float(prim.GetAttribute(f"physics:{side}Limit").Get())) - float(limit.get(side))), f"{name} {side}")
    expected_shapes = {name: [_urdf_shape(collision) for collision in link.findall("collision")]
                       for name, link in links.items()}
    actual_shapes = {name: [] for name in links}
    colliders = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    result["counts"].update(expected_collisions=sum(map(len, expected_shapes.values())), collisions=len(colliders))
    for prim in colliders:
        owner = prim
        while owner and str(owner.GetPath()) not in by_path:
            owner = owner.GetParent()
        name = by_path.get(str(owner.GetPath())) if owner else None
        if name not in links:
            errors.append(f"{prim.GetPath()}: collision has no expected rigid-body owner")
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            errors.append(f"{prim.GetPath()}: collision is disabled")
        try:
            transform = np.linalg.inv(world[name]) @ _matrix(cache.GetLocalToWorldTransform(prim))
            actual_shapes[name].append(_usd_shape(prim, transform))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{prim.GetPath()}: {exc}")
    for name, expected in expected_shapes.items():
        actual = actual_shapes[name]
        if len(expected) != len(actual):
            errors.append(f"{name}: collision count {len(actual)} != URDF {len(expected)}")
        comparisons = [[_shape_errors(first, second) for second in actual] for first in expected]
        matched = {}

        def assign(index, seen):
            for candidate, measures in enumerate(comparisons[index]):
                if candidate in seen or measures is None or any(value > TOLERANCES[key] for key, value in measures.items()):
                    continue
                seen.add(candidate)
                if candidate not in matched or assign(matched[candidate], seen):
                    matched[candidate] = index
                    return True
            return False

        for index in range(len(expected)):
            if not assign(index, set()):
                same_kind = [measures for measures in comparisons[index] if measures is not None]
                closest = min(same_kind, key=lambda entry: max(value / TOLERANCES[key] for key, value in entry.items()), default=None)
                if closest:
                    for metric, value in closest.items():
                        maxima[metric] = max(maxima[metric], float(value))
                errors.append(f"{name}: no one-to-one collision match for URDF {expected[index]['kind']} #{index}; closest_errors={closest}")
        for candidate, index in matched.items():
            for metric, value in comparisons[index][candidate].items():
                check(metric, value, actual[candidate]["path"])
            world_measures = _shape_errors(_transform_shape(expected[index], expected_world[name]),
                                           _transform_shape(actual[candidate], inverse_root @ world[name]))
            check("collision_world_surface_displacement_bound_m",
                  world_measures["collision_surface_displacement_bound_m"], actual[candidate]["path"])
    return result
