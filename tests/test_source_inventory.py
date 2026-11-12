"""Owned source stays discoverable; relocated commands work outside the checkout."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "source_inventory", ROOT / "tools/source_inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class SourceInventoryTests(unittest.TestCase):
    def fixture(self, root):
        (root / "tools").mkdir()
        (root / "configs").mkdir()
        (root / "tools/example.py").write_text("x = 1\n")
        data = {
            "tools": [
                dict(
                    path="tools/example.py",
                    classification="current",
                    owner="lead",
                    purpose="fixture",
                )
            ]
        }
        (root / inventory.INVENTORY).write_text(json.dumps(data))
        return data

    def test_repository_inventory_covers_all_tools(self):
        self.assertGreater(len(inventory.check()), 0)

    def test_unowned_source_cannot_hide_outside_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "tools/extra.py").write_text("x = 2\n")
            with self.assertRaisesRegex(ValueError, "unowned"):
                inventory.check(root)

    def test_missing_source_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "tools/example.py").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                inventory.check(root)

    def test_production_cannot_import_a_frozen_evidence_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            (root / "src/locomotion").mkdir(parents=True)
            (root / "src/locomotion/consumer.py").write_text("from artifacts.run import model\n")
            data["tools"].append(
                dict(
                    path="src/locomotion/consumer.py",
                    classification="current",
                    owner="lead",
                    purpose="fixture",
                )
            )
            (root / inventory.INVENTORY).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "imports evidence"):
                inventory.check(root)

    def test_import_audit_resolves_absolute_relative_and_package_imports(self):
        path = "src/locomotion/rl/wrapper.py"
        for statement in (
            "from locomotion.omniwalk.mdp.destination import destination_features",
            "from ..omniwalk.mdp.destination import destination_features",
            "from locomotion import omniwalk as task",
            "import locomotion.omniwalk as task",
            "from locomotion import new_package",
        ):
            with (
                self.subTest(statement=statement),
                self.assertRaisesRegex(ValueError, "Forbidden import"),
            ):
                inventory.check_imports(path, statement)
        inventory.check_imports(path, "from .wrapper import OmniVecEnv")
        inventory.check_imports("src/locomotion/omniwalk/task.py", "from ..robots import hexapod")

    def test_control_packages_cannot_import_simulation_or_retained_experiment(self):
        for path in (
            "src/navigation/waypoint.py",
            "src/mission/controller.py",
            "src/contracts/command.py",
        ):
            for statement in ("from locomotion.omniwalk import task", "import locomotion"):
                with (
                    self.subTest(path=path, statement=statement),
                    self.assertRaisesRegex(ValueError, "Forbidden import"),
                ):
                    inventory.check_imports(path, statement)
        for statement in ("from locomotion import hexapod", "from .. import hexapod"):
            with (
                self.subTest(statement=statement),
                self.assertRaisesRegex(ValueError, "retained experiment"),
            ):
                inventory.check_imports("src/locomotion/rl/wrapper.py", statement)

    def test_relocated_cli_help_works_in_a_fresh_process_from_another_directory(self):
        paths = [
            "tools/archive.py",
            "tools/assets/prepare_updated_usd.py",
            "tools/assets/prepare_geometry.py",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                with self.subTest(path=path):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / path), "--help"],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
