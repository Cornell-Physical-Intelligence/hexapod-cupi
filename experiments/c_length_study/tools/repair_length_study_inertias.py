"""Author and numerically verify source URDF inertials on a NEW study USD.

Run with Isaac Sim's Python. This repairs only the explicitly supplied study
asset; it does not patch the installed importer or any archived USD.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def repair_and_verify(urdf_path, usd_path, *, enable_contact_reports=True):
    from pxr import Gf, Usd, UsdPhysics
    if enable_contact_reports:
        from pxr import PhysxSchema

    stage = Usd.Stage.Open(str(usd_path))
    links = {e.get("name"): e for e in ET.parse(urdf_path).getroot().findall("link")}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 19 or {p.GetName() for p in bodies} != set(links):
        raise ValueError("Imported rigid body names/count do not match the URDF")
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(joints) != 18:
        raise ValueError("Expected 18 imported joints")
    expected = {}
    for prim in bodies:
        element = links[prim.GetName()].find("inertial")
        a = {key: float(value) for key, value in element.find("inertia").attrib.items()}
        tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
        origin = element.find("origin")
        if any(abs(float(v)) > 1e-12 for v in origin.get("rpy", "0 0 0").split()):
            raise ValueError("Study generator must emit link-frame tensors (zero inertial rpy)")
        com = np.array([float(v) for v in origin.get("xyz").split()])
        mass = float(element.find("mass").get("value"))
        diagonal, columns = np.linalg.eigh(tensor)
        if np.linalg.det(columns) < 0:
            columns[:, 0] *= -1
        # Gf uses row vectors. NumPy eigenvectors are columns; transpose once.
        quaternion = Gf.Matrix3d(*map(float, columns.T.flatten())).ExtractRotation().GetQuat()
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr().Set(mass)
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com)))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, diagonal)))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(quaternion))
        if enable_contact_reports:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        expected[str(prim.GetPath())] = (mass, com, tensor)
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(usd_path))
    errors = []
    for path, (mass, com, tensor) in expected.items():
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        rotation = Gf.Rotation(Gf.Quatd(api.GetPrincipalAxesAttr().Get()))
        # Verify using Gf's own vector transform, independently of the matrix
        # construction above and independently of SciPy quaternion conventions.
        basis = np.column_stack([rotation.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)])
        actual = basis @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ basis.T
        if not np.allclose(actual, tensor, rtol=1e-5, atol=1e-8) or not np.allclose(api.GetCenterOfMassAttr().Get(), com, atol=1e-7) or not np.isclose(api.GetMassAttr().Get(), mass, rtol=1e-6):
            raise ValueError(f"Inertia/mass/COM round trip failed: {path}")
        errors.append(float(np.max(np.abs(actual - tensor))))
    return {"links_checked": len(expected), "max_abs_tensor_error_kg_m2": max(errors), "pass": True}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("usd", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair_and_verify(args.urdf, args.usd), indent=2))
