"""Read-only imported C asset audit, called after AppLauncher.

The audit body is reused unchanged from hexapod_terrain.robot_smoke's
runtime-tested audit_robot_usd (original fixture/standing integration). It opens
USD for inspection and never authors or saves a layer. No terrain imports.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_candidate_usd(urdf, usd):
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
