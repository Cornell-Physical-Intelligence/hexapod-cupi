"""CPU USD regression tests for the simplified-collision converter (PR #87).

Each test converts a disposable copy of the approved bundle in a subprocess, so the checks cover the command line,
optimized Python (``-O`` strips ``assert``) and reopening the output in a fresh process.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "src/locomotion/hexapod/tools/make_simple_collision.py"
BUNDLE = ROOT / "robot/hexapod_mkii_updated_v1/usd"
APPROVED_SHA256 = "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
PUBLISHED_SIMPLE_SHA256 = "0372cd514bbb5fdf2bdde802d152407cb5c732b2cc6cece6213aa2ca7e300bba"
VISUAL_MESHES = 1753
COLLISION_SHAPES = 19

REOPEN = """
import sys
from pxr import Usd, UsdGeom, UsdPhysics
stage = Usd.Stage.Open(sys.argv[1])
prims = list(Usd.PrimRange(stage.GetPrimAtPath("/Robot")))
meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
with_points = sum(1 for p in meshes if len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) > 0)
colliders = sum(1 for p in prims if p.HasAPI(UsdPhysics.CollisionAPI))
print(len(meshes), with_points, colliders)
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(directory):
    return {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def run_tool(*args, optimize=False):
    command = [sys.executable, *(["-O"] if optimize else []), str(TOOL), *map(str, args)]
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT)


class SimpleCollisionConverterTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sha256(BUNDLE / "robot.usda"), APPROVED_SHA256)
        self._tmp = tempfile.TemporaryDirectory(prefix="simple-collision-")
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.copy = self.work / "usd"
        shutil.copytree(BUNDLE, self.copy)
        self.src = self.copy / "robot.usda"
        self.before = tree_hashes(self.work)

    def reopen(self, path):
        result = subprocess.run(
            [sys.executable, "-c", REOPEN, str(path)], capture_output=True, text=True, check=True
        )
        return tuple(int(v) for v in result.stdout.split())

    def assert_refused(self, *args):
        for optimize in (False, True):
            with self.subTest(args=args, optimize=optimize):
                result = run_tool("--src", self.src, *args, optimize=optimize)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("refused, nothing written", result.stderr)
                self.assertEqual(tree_hashes(self.work), self.before)

    def test_default_conversion_reproduces_the_published_asset(self):
        result = run_tool("--src", self.src)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.copy / "robot_simple.usda"
        self.assertEqual(sha256(out), PUBLISHED_SIMPLE_SHA256)
        meshes, with_points, colliders = self.reopen(out)
        self.assertEqual(
            (meshes, with_points, colliders), (VISUAL_MESHES, VISUAL_MESHES, COLLISION_SHAPES)
        )
        after = tree_hashes(self.work)
        del after["usd/robot_simple.usda"]
        self.assertEqual(after, self.before)

    def test_regenerating_over_its_own_output_is_allowed(self):
        self.assertEqual(run_tool("--src", self.src).returncode, 0)
        result = run_tool("--src", self.src, optimize=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sha256(self.copy / "robot_simple.usda"), PUBLISHED_SIMPLE_SHA256)

    def test_output_aliasing_the_source_is_refused(self):
        self.assert_refused("--out", self.src)
        self.assert_refused("--out", self.copy / ".." / "usd" / "robot.usda")

    def test_output_aliasing_a_referenced_layer_is_refused(self):
        self.assert_refused("--out", self.copy / "geometry.usdc")

    def test_output_through_a_link_to_an_input_is_refused(self):
        os.symlink(self.src, self.copy / "link.usda")
        os.link(self.copy / "geometry.usdc", self.copy / "hard.usdc")
        self.before = tree_hashes(self.work)
        self.assert_refused("--out", self.copy / "link.usda")
        self.assert_refused("--out", self.copy / "hard.usdc")

    def test_output_over_another_bundle_file_is_refused(self):
        self.assert_refused("--out", self.copy / "offline_audit.json")
        self.assert_refused("--out", self.copy / "collision_recipe.json")

    def test_output_outside_the_source_directory_is_refused(self):
        (self.work / "elsewhere").mkdir()
        self.before = tree_hashes(self.work)
        self.assert_refused("--out", self.work / "elsewhere" / "robot_simple.usda")
        self.assert_refused("--out", self.work / "robot_simple.usda")

    def test_invalid_radius_settings_are_refused(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            self.assert_refused("--out", self.copy / "bad.usda", f"--radius_scale={value}")
        for value in ("0", "-0.5", "1.5", "nan"):
            self.assert_refused("--out", self.copy / "bad.usda", f"--radius_quantile={value}")


if __name__ == "__main__":
    unittest.main()
