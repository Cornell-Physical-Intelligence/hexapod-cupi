"""CPU orchestration tests: no test imports or starts Isaac Sim."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("verified_import_subject", ROOT / "tools/assets/import_urdf_to_usd.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
from tools.assets.enable_nested_contact_reports import enable_contact_reports  # noqa: E402

URDF = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf"


class ImportPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.output = self.base / "new" / "robot.usda"
        self.source = self.base / "source" / "original.usda"
        self.source.parent.mkdir()
        self.source.write_text("original bytes\n")
        self.events = []
        self.raw_paths = []

    def dependencies(self, path):
        return [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]

    def contacts(self, path):
        self.events.append("contacts")
        self.raw_paths.append(path)
        self.assertNotEqual(path, self.source)
        path.write_text(path.read_text() + "contacts authored\n")
        return 19

    def prepare(self, urdf, source, output):
        self.events.append("prepare")
        self.assertEqual(urdf, URDF)
        self.assertEqual(output, self.output)
        self.assertIn("contacts authored", source.read_text())
        record = json.loads((source.parent / "import_origin.json").read_text())
        self.assertEqual(record["contact_report_bodies"], 19)
        self.assertEqual(record["urdf_sha256"], hashlib.sha256(URDF.read_bytes()).hexdigest())
        return {"pass": True, "errors": []}

    @contextmanager
    def importer(self, urdf, raw, stiffness, damping):
        self.events.append("import")
        self.assertEqual(urdf, URDF)
        self.assertEqual((stiffness, damping), (30.0, 0.6))
        self.assertFalse(self.output.exists())
        raw.write_text("fresh raw import\n")
        try:
            yield
        finally:
            self.events.append("app_closed")

    def mocks(self):
        patcher = patch.multiple(subject, _dependencies=self.dependencies,
                                 _enable_contacts=self.contacts, _prepare=self.prepare)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cpu_source_path_never_calls_isaac_and_preserves_original(self):
        original = self.source.read_bytes()
        self.mocks()
        with patch.object(subject, "_isaac_import", side_effect=AssertionError("Isaac must not be called")) as gpu:
            report = subject.build_verified_asset(URDF, self.output, source_usd=self.source)
        self.assertTrue(report["pass"])
        gpu.assert_not_called()
        self.assertEqual(self.events, ["contacts", "prepare"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.raw_paths[0].parent.exists())
        self.assertEqual(list(self.base.glob(".mkii-import-*")), [])

    def test_fresh_import_is_verified_before_app_closes(self):
        self.mocks()
        with patch.object(subject, "_isaac_import", self.importer):
            self.assertTrue(subject.build_verified_asset(URDF, self.output)["pass"])
        self.assertEqual(self.events, ["import", "contacts", "prepare", "app_closed"])
        self.assertFalse(self.raw_paths[0].parent.exists())

    def test_failed_verification_closes_app_and_removes_owned_temporary_directory(self):
        self.mocks()
        with patch.object(subject, "_isaac_import", self.importer), patch.object(subject, "_prepare", side_effect=ValueError("bad inertia")):
            with self.assertRaisesRegex(ValueError, "bad inertia"):
                subject.build_verified_asset(URDF, self.output)
        self.assertEqual(self.events, ["import", "contacts", "app_closed"])
        self.assertFalse(self.raw_paths[0].parent.exists())
        self.assertFalse(self.output.exists())

    def test_failed_contacts_prevent_preparation(self):
        self.mocks()
        with patch.object(subject, "_enable_contacts", side_effect=RuntimeError("missing body")), patch.object(subject, "_prepare") as prepare:
            with self.assertRaisesRegex(RuntimeError, "missing body"):
                subject.build_verified_asset(URDF, self.output, source_usd=self.source)
            prepare.assert_not_called()
        self.assertEqual(list(self.base.glob(".mkii-import-*")), [])

    def test_nonpassing_preparation_report_is_not_success(self):
        self.mocks()
        with patch.object(subject, "_prepare", return_value={"pass": False, "errors": ["regression"]}):
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                subject.build_verified_asset(URDF, self.output, source_usd=self.source)

    def test_existing_output_is_rejected_before_isaac(self):
        self.output.parent.mkdir()
        self.output.write_text("preserve this")
        with patch.object(subject, "_isaac_import") as gpu:
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                subject.build_verified_asset(URDF, self.output)
            gpu.assert_not_called()
        self.assertEqual(self.output.read_text(), "preserve this")

    def test_nonempty_destination_directory_is_rejected_before_isaac(self):
        self.output.parent.mkdir()
        preserved = self.output.parent / "someone_else.txt"
        preserved.write_text("preserve this")
        with patch.object(subject, "_isaac_import") as gpu:
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                subject.build_verified_asset(URDF, self.output)
            gpu.assert_not_called()
        self.assertEqual(preserved.read_text(), "preserve this")

    def test_fixed_base_and_invalid_gains_are_rejected_before_isaac(self):
        for kwargs in ({"fix_base": True}, {"drive_stiffness": -1}, {"drive_damping": float("nan")}):
            with self.subTest(kwargs=kwargs), patch.object(subject, "_isaac_import") as gpu:
                with self.assertRaises(ValueError):
                    subject.build_verified_asset(URDF, self.output, **kwargs)
                gpu.assert_not_called()

    def test_linkage_asset_is_rejected_before_isaac(self):
        with patch.object(subject, "_isaac_import") as gpu:
            with self.assertRaisesRegex(ValueError, "serial MKII"):
                subject.build_verified_asset(URDF.with_name("hexapod_mkii_linkage.urdf"), self.output)
            gpu.assert_not_called()

    def test_output_nested_in_source_bundle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            subject.build_verified_asset(URDF, self.source.parent / "new" / "robot.usda", source_usd=self.source)

    def test_cpu_source_does_not_silently_ignore_requested_drive_override(self):
        with self.assertRaisesRegex(ValueError, "Drive overrides apply only to fresh imports"):
            subject.build_verified_asset(URDF, self.output, source_usd=self.source, drive_stiffness=5)

    def test_unresolved_source_dependency_is_rejected_without_copy_or_gpu(self):
        self.mocks()
        with patch.object(subject, "_dependencies", side_effect=ValueError("unresolved USD reference")), patch.object(subject, "_isaac_import") as gpu:
            with self.assertRaisesRegex(ValueError, "unresolved"):
                subject.build_verified_asset(URDF, self.output, source_usd=self.source)
            gpu.assert_not_called()
        self.assertEqual(list(self.base.glob(".mkii-import-*")), [])

    def test_source_symlink_is_rejected(self):
        self.mocks()
        (self.source.parent / "alias.usda").symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            subject.build_verified_asset(URDF, self.output, source_usd=self.source)

    def test_source_changing_during_copy_is_rejected(self):
        self.mocks()
        with patch.object(subject, "_dependencies", return_value=[{"path": self.source.name, "sha256": "wrong"}]):
            with self.assertRaisesRegex(ValueError, "changed while copying"):
                subject.build_verified_asset(URDF, self.output, source_usd=self.source)


class ContactAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "raw.usda"
        stage = Usd.Stage.CreateNew(str(self.path))
        for path in ("/Robot/body", "/Robot/body/leg"):
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
        stage.GetRootLayer().Save()

    def test_cpu_contact_tokens_persist_and_second_pass_is_idempotent(self):
        self.assertEqual(enable_contact_reports(self.path, 2), 2)
        first = self.path.read_bytes()
        self.assertEqual(enable_contact_reports(self.path, 2), 2)
        self.assertEqual(self.path.read_bytes(), first)
        stage = Usd.Stage.Open(str(self.path))
        for path in ("/Robot/body", "/Robot/body/leg"):
            prim = stage.GetPrimAtPath(path)
            self.assertIn("PhysxContactReportAPI", prim.GetMetadata("apiSchemas").GetAppliedItems())
            self.assertEqual(prim.GetAttribute("physxContactReport:threshold").Get(), 0)

    def test_wrong_body_count_is_rejected_before_writing(self):
        original = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Expected 19"):
            enable_contact_reports(self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_prepared_bundle_cannot_be_modified(self):
        stage = Usd.Stage.Open(str(self.path))
        stage.GetRootLayer().customLayerData = {"hexapod_inertia_repair_version": 2}
        stage.GetRootLayer().Save()
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable"):
            enable_contact_reports(self.path, 2)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
