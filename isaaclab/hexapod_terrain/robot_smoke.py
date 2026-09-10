"""Read-only asset/footprint checks for the selected C terrain standing smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from terrain_fixture_checks import digest, vertical_surface_heights
from .fixture_adapter import require_matching_admission

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")


def load_admitted_study(package, variant, stance_index, admission_path):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    plan = json.loads((package / "training_plan.json").read_text())
    record = next(r for r in manifest["variants"] if r["variant"] == variant)
    if not (np.isclose(record["femur_length_m"], .0725) and np.isclose(record["tibia_length_m"], .126)):
        raise ValueError("This runner is scoped to the user-selected full C study geometry")
    if not plan.get("omni") or plan.get("reference_controller"):
        raise ValueError("The exact current omnidirectional study plan is required")
    if manifest["actuator_config_snapshot"].get("effort_limit") != 1.6:
        raise ValueError("The current RS05 1.6 N·m continuous actuator cap must be retained")
    urdf = package / record["urdf"]
    if digest(urdf) != record["sha256"]:
        raise ValueError("Study URDF hash mismatch")
    xml = ET.parse(urdf).getroot()
    if len(xml.findall("link")) != 19 or len(xml.findall("joint")) != 18:
        raise ValueError("Expected full 19-link/18-joint study robot")
    stance = plan["variants"][variant]["stances"][stance_index]
    if {j.get("name") for j in xml.findall("joint")} != set(stance["joint_positions_rad"]):
        raise ValueError("Stance must specify every joint by name")
    identity = dict(variant=variant, urdf_sha256=digest(urdf),
                    plan_sha256=digest(package / "training_plan.json"), stance_index=stance_index)
    admission = json.loads(Path(admission_path).read_text())
    require_matching_admission(admission, identity)
    gate = admission["gate"]
    if gate.get("num_envs", 0) < 32 or gate.get("control_steps", 0) < 1000:
        raise ValueError("Full 32-environment/1000-step flat admission is required first")
    return manifest, plan, record, urdf, xml, stance, identity, admission


def audit_robot_usd(urdf, usd):
    """Verify imported inertials without calling the writer/repair helper."""
    from pxr import Gf, Usd, UsdPhysics, PhysxSchema

    xml = ET.parse(urdf).getroot()
    expected = {link.get("name"): link for link in xml.findall("link")}
    stage = Usd.Stage.Open(str(usd))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(bodies) != 19 or len(joints) != 18 or {b.GetName() for b in bodies} != set(expected):
        raise ValueError("Imported robot body/joint layout mismatch")
    errors, masses = [], []
    for body in bodies:
        inertial = expected[body.GetName()].find("inertial")
        origin = inertial.find("origin")
        if any(abs(float(v)) > 1e-12 for v in origin.get("rpy", "0 0 0").split()):
            raise ValueError("Study inertial tensor must use the link frame")
        com = np.fromstring(origin.get("xyz"), sep=" ")
        mass = float(inertial.find("mass").get("value"))
        a = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                           [a["ixz"], a["iyz"], a["izz"]]])
        api = UsdPhysics.MassAPI(body)
        rotation = Gf.Rotation(Gf.Quatd(api.GetPrincipalAxesAttr().Get()))
        basis = np.column_stack([rotation.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)])
        actual = basis @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ basis.T
        if (not np.allclose(actual, tensor, rtol=1e-5, atol=1e-8)
                or not np.allclose(api.GetCenterOfMassAttr().Get(), com, atol=1e-7)
                or not np.isclose(api.GetMassAttr().Get(), mass, rtol=1e-6)
                or np.linalg.eigvalsh(tensor).min() <= 0):
            raise ValueError(f"Inertia/mass/COM mismatch: {body.GetName()}")
        if not body.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise ValueError(f"Missing contact reports: {body.GetName()}")
        errors.append(float(np.max(np.abs(actual - tensor))))
        masses.append(mass)
    return dict(passed=True, body_count=19, joint_count=18, total_mass_kg=sum(masses),
                max_tensor_error_kg_m2=max(errors), usd_sha256=digest(usd), read_only=True)


def check_start_footprint(package, xml, manifest, record, stance, spec, terrain):
    """Check each exact distal pad's XY bounding box stays on the flat start pad."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "robot/tools"))
    import generate_length_study as geometry

    entry, _, vertices, faces = terrain
    transforms, _ = geometry.fk(xml, stance["joint_positions_rad"])
    c, s = np.cos(spec.start_body_yaw_rad), np.sin(spec.start_body_yaw_rad)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    rows = []
    for leg in LEGS:
        name = manifest["link_joint_mapping"][leg]["links"]["tibia"]
        link = xml.find(f"link[@name='{name}']")
        points = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected the study's exact collision mesh")
            path = Path(package) / "meshes" / Path(mesh.get("filename")).name
            cloud = geometry.stl_vertices(path) * geometry.vec(mesh.get("scale", "1 1 1"))
            transform = geometry.origin(collision)
            points.append(np.einsum("nj,ij->ni", cloud, transform[:3, :3]) + transform[:3, 3])
        cloud = np.concatenate(points)
        distal = cloud[cloud[:, 1] > record["tibia_length_m"] * manifest["study"]["distal_foot_fraction"]]
        if len(distal) == 0:
            raise ValueError(f"No distal foot collision vertices for {leg}")
        transform = transforms[name]
        body = np.einsum("nj,ij->ni", distal, transform[:3, :3]) + transform[:3, 3]
        world = np.einsum("nj,ij->ni", body, rotation)
        world[:, :2] += entry["start_xy_m"]
        lo, hi = world[:, :2].min(0), world[:, :2].max(0)
        samples = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 3)
                            for y in np.linspace(lo[1], hi[1], 3)])
        heights = vertical_surface_heights(vertices, faces, samples)
        passed = bool(np.isfinite(heights).all() and np.max(np.abs(heights)) <= 1e-6
                      and lo[0] >= -1.49 and hi[0] <= -.66 and lo[1] >= -1.49 and hi[1] <= 1.49)
        rows.append(dict(leg=leg, link=name, xy_bounds_m=[lo.tolist(), hi.tolist()], passed=passed))
    if not all(row["passed"] for row in rows):
        raise ValueError("Full C distal footprint does not fit the fixture flat start pad: " + json.dumps(rows))
    return rows
