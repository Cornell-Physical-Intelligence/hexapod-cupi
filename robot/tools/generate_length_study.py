#!/usr/bin/env python3
"""Generate mock-geometry URDFs carrying the CAD serial model's inertials.

Requires NumPy and SciPy, but no Isaac Sim. This is a controlled geometry
experiment, not a CAD mass-property recomputation or a training acceptance gate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "hexapod_mkii_length_study"
DEFAULT_CONFIG = ROOT / "robot" / PACKAGE / "study.json"
KINDS = ("coxa", "femur", "tibia")


def vec(value):
    result = np.fromstring(value, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid vector: {value}")
    return result


def fmt(value):
    return f"{float(value):.12g}"


def xyz(value):
    return " ".join(map(fmt, value))


def origin(element):
    result = np.eye(4)
    element = element.find("origin")
    if element is not None:
        result[:3, 3] = vec(element.get("xyz", "0 0 0"))
        result[:3, :3] = Rotation.from_euler("xyz", vec(element.get("rpy", "0 0 0"))).as_matrix()
    return result


def inertial(link):
    element = link.find("inertial")
    a = {key: float(value) for key, value in element.find("inertia").attrib.items()}
    tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                       [a["ixy"], a["iyy"], a["iyz"]],
                       [a["ixz"], a["iyz"], a["izz"]]])
    transform = origin(element)
    rotation = transform[:3, :3]
    return float(element.find("mass").get("value")), transform[:3, 3], rotation @ tensor @ rotation.T


def set_inertial(link, mass, com, tensor):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    element = ET.Element("inertial")
    ET.SubElement(element, "origin", xyz=xyz(com), rpy="0 0 0")
    ET.SubElement(element, "mass", value=fmt(mass))
    ET.SubElement(element, "inertia", **{
        key: fmt(tensor[i, j]) for key, i, j in
        (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
         ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))})
    link.insert(0, element)


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("Cannot define anatomical frame from parallel/zero vectors")
    return vector / norm


def pitch_basis(raising_axis, segment):
    """Columns: raising axis, segment direction, cross product; right handed."""
    x = unit(raising_axis)
    y = unit(segment - x * np.dot(x, segment))
    return np.column_stack((x, y, np.cross(x, y)))


def yaw_basis(positive_yaw_axis, hip_offset):
    z = unit(positive_yaw_axis)
    y = unit(hip_offset - z * np.dot(z, hip_offset))
    return np.column_stack((np.cross(y, z), y, z))


def stl_vertices(path):
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) != 84 + 50 * n or not n:
        raise ValueError(f"Expected nonempty binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    points = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).astype(float)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite STL: {path}")
    return np.unique(points, axis=0)


def mesh_path(uri):
    if not uri.startswith("package://"):
        raise ValueError(f"Expected a package mesh URI: {uri}")
    return ROOT / "robot" / uri.removeprefix("package://")


def visual_vertices(link):
    clouds = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        points = stl_vertices(mesh_path(mesh.get("filename"))) * vec(mesh.get("scale", "1 1 1"))
        transform = origin(visual)
        clouds.append(np.einsum("nj,ij->ni", points, transform[:3, :3]) + transform[:3, 3])
    return np.concatenate(clouds)


def actuator_snapshot(path):
    """Read the existing motor kwargs without importing Isaac or executing code."""
    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr == "pi":
            return math.pi
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError(f"Unsupported actuator expression: {ast.dump(node)}")

    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROBSTRIDE_RS05_CFG" for t in statement.targets):
            if not isinstance(statement.value, ast.Call) or not isinstance(statement.value.func, ast.Name) or statement.value.func.id != "DCMotorCfg":
                raise ValueError("Expected ROBSTRIDE_RS05_CFG = DCMotorCfg(...)")
            return {item.arg: evaluate(item.value) for item in statement.value.keywords}
    raise ValueError("Missing ROBSTRIDE_RS05_CFG")


def fk(robot, positions):
    transforms = {"body_mock": np.eye(4)}
    axes = {}
    pending = list(robot.findall("joint"))
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            transform = transforms[parent] @ origin(joint)
            axis = vec(joint.find("axis").get("xyz"))
            name = joint.get("name")
            axes[name] = (transform[:3, 3].copy(), transform[:3, :3] @ axis)
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * positions.get(name, 0.0)).as_matrix()
            transforms[joint.find("child").get("link")] = transform @ motion
            pending.remove(joint)
        if len(pending) == count:
            raise ValueError("Disconnected or cyclic model")
    return transforms, axes


def prepare(config):
    mock = ET.parse(ROOT / config["geometry_urdf"]).getroot()
    source = ET.parse(ROOT / config["inertia_urdf"]).getroot()
    source_links = {e.get("name"): e for e in source.findall("link")}
    source_joints = {e.get("name"): e for e in source.findall("joint")}
    # Remove only the seven empty export scaffolding links. No physical link,
    # geometry, transform, or inertia is merged into a different moving body.
    for link in list(mock.findall("link")):
        if len(link) == 0:
            mock.remove(link)
    for joint in list(mock.findall("joint")):
        if joint.get("type") == "fixed":
            if joint.find("child").get("link") == "body_mock" and not np.allclose(origin(joint), np.eye(4), atol=1e-12):
                raise ValueError("Mock root is no longer an identity transform")
            mock.remove(joint)
    links = {e.get("name"): e for e in mock.findall("link")}
    by_child = {e.find("child").get("link"): e for e in mock.findall("joint")}
    clouds = {name: visual_vertices(link) for name, link in links.items()}
    mapping = {}
    transfer = {}
    body_offset = np.array([0, 0, clouds["body_mock"][:, 2].min()])
    mass, com, tensor = inertial(source_links["body"])
    set_inertial(links["body_mock"], mass, com + body_offset, tensor)
    transfer["body_mock"] = {"source_link": "body", "source_to_mock_rotation": np.eye(3).tolist(),
                             "translation_m": body_offset.tolist()}
    for leg, suffix in config["leg_to_mock_suffix"].items():
        names = {kind: kind + suffix for kind in KINDS}
        joints = {kind: by_child[names[kind]] for kind in KINDS}
        mapping[leg] = {"links": names, "joints": {kind: joints[kind].get("name") for kind in KINDS}}
        mount = origin(joints["coxa"])[:3, 3]
        if (mount[0] > 0) != leg.startswith("l") or (leg.endswith("f") and mount[1] >= -0.1) or (leg.endswith("r") and mount[1] <= 0.1) or (leg.endswith("m") and abs(mount[1]) > 0.01):
            raise ValueError(f"Anatomical leg mapping is wrong for {leg}")
        for kind in KINDS:
            source_name = f"{leg}_{kind}"
            if kind == "coxa":
                source_basis = yaw_basis(vec(source_joints[f"{leg}_coxa_yaw"].find("axis").get("xyz")),
                                         origin(source_joints[f"{leg}_femur_pitch"])[:3, 3])
                mock_basis = yaw_basis(vec(joints[kind].find("axis").get("xyz")), origin(joints["femur"])[:3, 3])
            else:
                if kind == "femur":
                    source_segment = origin(source_joints[f"{leg}_tibia_pitch"])[:3, 3]
                    mock_segment = origin(joints["tibia"])[:3, 3]
                    mock_axis = vec(joints[kind].find("axis").get("xyz"))
                else:
                    pads = [origin(c)[:3, 3] for c in source_links[source_name].findall("collision") if c.find("geometry/sphere") is not None]
                    if len(pads) != 2:
                        raise ValueError(f"Expected two CAD foot-pad spheres on {source_name}")
                    source_segment = max(pads, key=lambda p: np.linalg.norm(p))
                    mock_segment = np.array([0, clouds[names[kind]][:, 1].max(), 0])
                    # CAD positive knee opens; mock positive knee closes.
                    mock_axis = -vec(joints[kind].find("axis").get("xyz"))
                source_basis = pitch_basis(vec(source_joints[f"{leg}_{kind}_pitch"].find("axis").get("xyz")), source_segment)
                mock_basis = pitch_basis(mock_axis, mock_segment)
            rotation = mock_basis @ source_basis.T
            mass, com, tensor = inertial(source_links[source_name])
            set_inertial(links[names[kind]], mass, rotation @ com, rotation @ tensor @ rotation.T)
            transfer[names[kind]] = {"source_link": source_name, "source_to_mock_rotation": rotation.tolist(), "translation_m": [0, 0, 0]}
            current_joint = source_joints[f"{leg}_{kind}_{'yaw' if kind == 'coxa' else 'pitch'}"]
            for key in ("effort", "velocity"):
                joints[kind].find("limit").set(key, current_joint.find("limit").get(key))
            old_dynamics = joints[kind].find("dynamics")
            if old_dynamics is not None:
                joints[kind].remove(old_dynamics)
            joints[kind].append(copy.deepcopy(current_joint.find("dynamics")))
    if len(links) != 19 or len(by_child) != 18:
        raise ValueError("Expected 19 physical links and 18 joints")
    return mock, mapping, transfer, clouds


def make_variant(base, mapping, femur_scale, tibia_scale):
    robot = copy.deepcopy(base)
    links = {e.get("name"): e for e in robot.findall("link")}
    joints = {e.get("name"): e for e in robot.findall("joint")}
    for leg in mapping.values():
        knee = joints[leg["joints"]["tibia"]].find("origin")
        offset = vec(knee.get("xyz"))
        offset[1] *= femur_scale
        knee.set("xyz", xyz(offset))
        for kind, scale in (("femur", femur_scale), ("tibia", tibia_scale)):
            link = links[leg["links"][kind]]
            mass, com, tensor = inertial(link)
            com[1] *= scale
            # Fixed mass and COM-centred tensor is the explicitly chosen ablation.
            # Do not silently scale motors or claim new manufacturing inertials.
            set_inertial(link, mass, com, tensor)
            for geometry in link.findall("visual") + link.findall("collision"):
                transform = origin(geometry)
                if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-10):
                    raise ValueError("Longitudinal mesh scaling requires aligned mock mesh frames")
                offset = transform[:3, 3]
                offset[1] *= scale
                geometry.find("origin").set("xyz", xyz(offset))
                mesh = geometry.find("geometry/mesh")
                scales = vec(mesh.get("scale", "1 1 1"))
                scales[1] *= scale
                mesh.set("scale", xyz(scales))
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", f"package://{PACKAGE}/meshes/{Path(mesh.get('filename')).name}")
    return robot


def validate(robot, package):
    links = {e.get("name"): e for e in robot.findall("link")}
    joints = robot.findall("joint")
    if len(links) != len(robot.findall("link")) or len(links) != 19 or len(joints) != 18 or len({j.get("name") for j in joints}) != 18:
        raise ValueError("Wrong/duplicate links or joints")
    children = [j.find("child").get("link") for j in joints]
    if len(set(children)) != 18 or set(links) - set(children) != {"body_mock"}:
        raise ValueError("Invalid articulation tree")
    fk(robot, {})
    for joint in joints:
        limits = {key: float(value) for key, value in joint.find("limit").attrib.items()}
        if joint.get("type") != "revolute" or joint.find("mimic") is not None or not math.isclose(np.linalg.norm(vec(joint.find("axis").get("xyz"))), 1.0, abs_tol=1e-9):
            raise ValueError("Invalid revolute joint")
        if not all(math.isfinite(v) for v in limits.values()) or not limits["lower"] < limits["upper"] or min(limits["effort"], limits["velocity"]) <= 0:
            raise ValueError("Invalid joint limits")
    masses, eigmins = [], []
    for link in links.values():
        mass, com, tensor = inertial(link)
        eigenvalues = np.linalg.eigvalsh(tensor)
        if not np.isfinite(tensor).all() or not np.isfinite(com).all() or not math.isfinite(mass) or mass <= 0 or eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[:2].sum() + 1e-10:
            raise ValueError(f"Nonphysical mass/inertia: {link.get('name')}")
        if not link.findall("visual") or not link.findall("collision"):
            raise ValueError("Missing visual/collision")
        for mesh in link.findall(".//mesh"):
            relative = mesh.get("filename").removeprefix(f"package://{PACKAGE}/")
            if not (package / relative).is_file() or np.any(vec(mesh.get("scale", "1 1 1")) <= 0):
                raise ValueError(f"Missing/invalid mesh: {relative}")
        masses.append(mass)
        eigmins.append(float(eigenvalues[0]))
    return {"physical_links": 19, "revolute_joints": 18, "total_mass_kg": sum(masses),
            "minimum_principal_inertia_kg_m2": min(eigmins), "static_structure_pass": True}


def screen(robot, mapping, clouds, config, scales):
    """Level-body, frictionless six-foot gravity balance at a shared test pose.

    No dynamics, PD controller, friction, material compliance or collision engine.
    Normal reactions solve force and roll/pitch moment balance; negative reactions
    or differing foot heights make the six-foot approximation ineligible.
    """
    positions = {leg["joints"][kind]: config["screening_pose_mock_rad"][kind] for leg in mapping.values() for kind in KINDS}
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        if not float(limit.get("lower")) <= positions[joint.get("name")] <= float(limit.get("upper")):
            raise ValueError("Screening pose outside mock limits")
    transforms, axes = fk(robot, positions)
    links = {link.get("name"): link for link in robot.findall("link")}
    mass_com = {}
    transformed = {}
    for name, link in links.items():
        mass, com, _ = inertial(link)
        t = transforms[name]
        mass_com[name] = (mass, t[:3, :3] @ com + t[:3, 3])
        points = clouds[name].copy()
        if name.startswith("femur"):
            points[:, 1] *= scales[0]
        if name.startswith("tibia"):
            points[:, 1] *= scales[1]
        transformed[name] = (points, np.einsum("nj,ij->ni", points, t[:3, :3]) + t[:3, 3])
    contacts, nonfoot_z = [], []
    for leg in mapping.values():
        local, world = transformed[leg["links"]["tibia"]]
        distal = local[:, 1] > config["distal_foot_fraction"] * local[:, 1].max()
        pad = world[distal]
        contacts.append(pad[np.argmin(pad[:, 2])])
        nonfoot_z.append(float(world[~distal, 2].min()))
    for name, (_, world) in transformed.items():
        if not name.startswith("tibia"):
            nonfoot_z.append(float(world[:, 2].min()))
    contacts = np.array(contacts)
    total_mass = sum(item[0] for item in mass_com.values())
    total_com = sum(mass * com for mass, com in mass_com.values()) / total_mass
    a = np.vstack((np.ones(6), contacts[:, 0], contacts[:, 1]))
    b = total_mass * 9.81 * np.array([1.0, total_com[0], total_com[1]])
    reactions = np.linalg.lstsq(a, b, rcond=None)[0]
    residual = float(np.max(np.abs(a @ reactions - b)))
    torques = {}
    for index, leg in enumerate(mapping.values()):
        force = np.array([0.0, 0.0, reactions[index]])
        for depth, kind in enumerate(KINDS):
            name = leg["joints"][kind]
            pivot, axis = axes[name]
            moment = np.cross(contacts[index] - pivot, force)
            for child_kind in KINDS[depth:]:
                mass, com = mass_com[leg["links"][child_kind]]
                moment += np.cross(com - pivot, np.array([0.0, 0.0, -mass * 9.81]))
            torques[name] = -float(axis @ moment)
    ground_z = float(contacts[:, 2].min())
    foot_spread = float(np.ptp(contacts[:, 2]))
    clearance = min(nonfoot_z) - ground_z
    return {"method": "six_foot_static_gravity_balance_at_common_mock_pose",
            "joint_positions_rad": positions, "signed_hold_torques_nm": torques,
            "max_abs_hold_torque_nm": max(map(abs, torques.values())),
            "normal_reactions_n": reactions.tolist(), "balance_residual": residual,
            "foot_height_spread_m": foot_spread,
            "nonfoot_mesh_vertex_clearance_m": clearance,
            "root_height_at_contact_m": -ground_z,
            "suggested_reset_root_height_m": -min(ground_z, min(nonfoot_z)) + config["reset_clearance_m"],
            "six_foot_geometry_eligible": bool(reactions.min() > 0 and residual < 1e-8 and foot_spread < 1e-4 and clearance > 0),
            "simulation_validated": False, "training_ready": False}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate(config_path=DEFAULT_CONFIG, output=None):
    config = json.loads(config_path.read_text())
    # Preserve historical input manifests while binding new generation to the
    # same verified motor source after the production package refactor.
    if config["actuator_config"] == "isaaclab/hexapod_rl/asset_cfg.py":
        sys.path.insert(0, str(ROOT / "tools"))
        from c_study_runtime import bootstrap_c_study_runtime
        binding = bootstrap_c_study_runtime(repo_root=ROOT)
        config["actuator_config"] = str((Path(binding["package_directory"]) / "asset_cfg.py").relative_to(ROOT))
    output = output or DEFAULT_CONFIG.parent
    if output.resolve() in {(ROOT / "robot/hexapod_mkii_assy").resolve(), (ROOT / "robot/hexapod_mkii_mock_assy").resolve()}:
        raise ValueError("Output must be a separate study package")
    if config["inertia_policy"] != "fixed_transferred_tensor" or config["com_policy"] != "scale_longitudinal_offset_about_proximal_joint" or config["body_alignment"] != "align_bottom_planes_keep_xyz_directions":
        raise ValueError("Unsupported study physics policy")
    for key in ("femur_scales", "tibia_scales"):
        values = config[key]
        if not values or len(set(values)) != len(values) or any(not math.isfinite(s) or not 0 < s <= 1.1 or abs(100 * s - round(100 * s)) > 1e-8 for s in values):
            raise ValueError(f"{key} must contain unique positive integer percentages <= 110%")
    if set(config["leg_to_mock_suffix"]) != {"lf", "lm", "lr", "rf", "rm", "rr"}:
        raise ValueError("Expected all six anatomical legs")
    base, mapping, transfer, clouds = prepare(config)
    for directory in ("meshes", "urdf", "metadata"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    mesh_sources = sorted({mesh_path(mesh.get("filename")) for mesh in base.findall(".//mesh")})
    for path in mesh_sources:
        shutil.copyfile(path, output / "meshes" / path.name)
    # Convex-hull vertices preserve directional extrema; retain the distal split
    # on tibiae before taking either hull for the contact/shaft classification.
    for name, points in list(clouds.items()):
        groups = [points]
        if name.startswith("tibia"):
            distal = points[:, 1] > config["distal_foot_fraction"] * points[:, 1].max()
            groups = [points[distal], points[~distal]]
        clouds[name] = np.concatenate([group[ConvexHull(group).vertices] for group in groups])
    actuators = actuator_snapshot(ROOT / config["actuator_config"])
    base_links = {e.get("name"): e for e in base.findall("link")}
    base_joints = {e.get("name"): e for e in base.findall("joint")}
    reference = mapping["lf"]
    femur_length = float(origin(base_joints[reference["joints"]["tibia"]])[1, 3])
    tibia_length = float(clouds[reference["links"]["tibia"]][:, 1].max())
    coxa_offset = origin(base_joints[reference["joints"]["femur"]])[:3, 3]
    rows = []
    records = []
    for sf in config["femur_scales"]:
        for st in config["tibia_scales"]:
            name = f"f{round(sf * 100):03d}_t{round(st * 100):03d}"
            robot = make_variant(base, mapping, sf, st)
            robot.set("name", f"{PACKAGE}_{name}")
            robot.insert(0, ET.Comment(" GENERATED by robot/tools/generate_length_study.py. Mock joint zero/sign/limits; CAD inertials. See metadata and README. "))
            urdf = output / "urdf" / f"{name}.urdf"
            tree = ET.ElementTree(robot)
            ET.indent(tree, space="  ")
            tree.write(urdf, encoding="utf-8", xml_declaration=True)
            # Verify serialized values as well as the in-memory model.
            robot = ET.parse(urdf).getroot()
            audit = validate(robot, output)
            screening = screen(robot, mapping, clouds, config, (sf, st))
            rated = actuators["effort_limit"]
            screening["below_continuous_limit_at_screening_pose"] = bool(screening["max_abs_hold_torque_nm"] <= rated)
            row = {"variant": name, "femur_scale": sf, "tibia_scale": st,
                   "femur_length_m": femur_length * sf, "tibia_length_m": tibia_length * st,
                   "max_abs_static_hold_torque_nm": screening["max_abs_hold_torque_nm"],
                   "static_torque_margin_nm": rated - screening["max_abs_hold_torque_nm"],
                   "six_foot_geometry_eligible": screening["six_foot_geometry_eligible"],
                   "reset_root_height_m": screening["suggested_reset_root_height_m"],
                   "simulation_validated": False}
            rows.append(row)
            record = {**row, "urdf": f"urdf/{name}.urdf", "sha256": digest(urdf),
                      "is_unchanged_length_baseline": sf == st == 1,
                      "audit": audit, "screening": screening,
                      "distal_foot_min_y_m": config["distal_foot_fraction"] * tibia_length * st,
                      "foot_tip_offset_y_m": tibia_length * st}
            write_json(output / "metadata" / f"{name}.json", record)
            records.append(record)
    provenance = {relative: digest(ROOT / relative) for relative in (config["geometry_urdf"], config["inertia_urdf"], config["actuator_config"], "robot/tools/generate_length_study.py")}
    provenance["study_config_sha256"] = digest(config_path)
    provenance.update({str(path.relative_to(ROOT)): digest(path) for path in mesh_sources})
    manifest = {"schema_version": 1, "study": config, "source_sha256": provenance,
                "variant_count": len(records), "link_joint_mapping": mapping,
                "runtime_joint_order": "UNOBSERVED: resolve by joint name after import",
                "joint_convention": "ARCHIVED MOCK: femur zero horizontal/positive raises; tibia zero straight/positive closes; body root is TOP of mock plate",
                "body_bottom_in_root_m": float(clouds["body_mock"][:, 2].min()),
                "baseline_lengths_m": {"femur_hip_to_knee": femur_length, "tibia_knee_to_distal_mesh_tip": tibia_length,
                                       "coxa_yaw_to_hip_3d": float(np.linalg.norm(coxa_offset)),
                                       "coxa_yaw_to_hip_horizontal": float(np.linalg.norm(coxa_offset[:2]))},
                "inertial_transfer": transfer, "actuator_config_snapshot": actuators,
                "urdf_effort_is_peak_not_continuous": True,
                "training_ready": False, "variants": records}
    write_json(output / "manifest.json", manifest)
    with (output / "variants.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG.parent)
    args = parser.parse_args()
    manifest = generate(args.config, args.output)
    print(f"Generated {manifest['variant_count']} URDFs in {args.output / 'urdf'}")
    print(json.dumps(manifest["baseline_lengths_m"], indent=2))
    print("Static structural checks passed; simulation validation and training have not run.")


if __name__ == "__main__":
    main()
