"""CPU-only test using the installed OpenUSD library, without SimulationApp."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from repair_length_study_inertias import repair_and_verify

package = Path(__file__).resolve().parents[1] / "robot/hexapod_mkii_length_study"
with tempfile.TemporaryDirectory() as temp:
    max_error = 0.0
    count = 0
    for urdf in sorted((package / "urdf").glob("*.urdf")):
        tree = ET.parse(urdf).getroot()
        usd = Path(temp) / f"{urdf.stem}.usda"
        stage = Usd.Stage.CreateNew(str(usd))
        UsdGeom.Xform.Define(stage, "/Robot")
        for link in tree.findall("link"):
            prim = UsdGeom.Xform.Define(stage, f"/Robot/{link.get('name')}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr().Set(123.0)
            api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1, 2, 3))
        for joint in tree.findall("joint"):
            UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{joint.get('name')}")
        stage.GetRootLayer().Save()
        result = repair_and_verify(urdf, usd, enable_contact_reports=False)
        assert result["links_checked"] == 19 and result["pass"]
        max_error = max(max_error, result["max_abs_tensor_error_kg_m2"])
        reopened = Usd.Stage.Open(str(usd))
        for link in tree.findall("link"):
            api = UsdPhysics.MassAPI(reopened.GetPrimAtPath(f"/Robot/{link.get('name')}"))
            expected = float(link.find("inertial/mass").get("value"))
            assert np.isclose(api.GetMassAttr().Get(), expected, rtol=1e-6)
        count += 1
    assert count == 49
    print(f"USD_CPU_ROUNDTRIP_PASS variants={count} bodies={count * 19} max_tensor_error={max_error:.3g}")
