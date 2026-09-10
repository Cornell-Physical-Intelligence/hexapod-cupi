"""OpenUSD CPU reproduction of the installed generic-spawner schema override."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
sys.path.insert(0, str(ROOT / "tools"))
from hexapod_terrain.fixture_adapter import reference_fixture_mesh
from terrain_fixture_checks import collision_meshes, load_catalog

@unittest.skipIf(Usd is None, "OpenUSD is required for actual composition checks")

class FixtureReferenceTests(unittest.TestCase):
    def test_xform_override_reproduces_failure_and_mesh_reference_preserves_geometry(self):
        catalog = ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json"
        for entry, usd, vertices, faces in load_catalog(catalog):
            before = hashlib.sha256(usd.read_bytes()).hexdigest()
            broken = Usd.Stage.CreateInMemory()
            prim = broken.DefinePrim("/World/ground/terrain", "Xform")
            prim.GetReferences().AddReference(str(usd))
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.IsA(UsdGeom.Mesh))
            with self.assertRaisesRegex(ValueError, "non-mesh terrain collider"):
                collision_meshes(broken, "/World/ground")
            stage = Usd.Stage.CreateInMemory()
            actual = reference_fixture_mesh(stage, "/World/ground/terrain", usd)
            self.assertTrue(actual.IsA(UsdGeom.Mesh), entry["id"])
            self.assertEqual(str(actual.GetPath()), "/World/ground/terrain")
            mesh = UsdGeom.Mesh(actual)
            np.testing.assert_allclose(np.array(mesh.GetPointsAttr().Get()), vertices, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces)
            self.assertEqual(UsdPhysics.MeshCollisionAPI(actual).GetApproximationAttr().Get(), "none")
            self.assertEqual(hashlib.sha256(usd.read_bytes()).hexdigest(), before)
            with self.assertRaisesRegex(ValueError, "already exists"):
                reference_fixture_mesh(stage, "/World/ground/terrain", usd)

    def test_unsafe_collision_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "convex.usda"
            source = Usd.Stage.CreateNew(str(p))
            mesh = UsdGeom.Mesh.Define(source, "/Terrain")
            source.SetDefaultPrim(mesh.GetPrim())
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")
            source.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "convexification"):
                reference_fixture_mesh(Usd.Stage.CreateInMemory(), "/World/Terrain", p)


if __name__ == "__main__":
    unittest.main()
