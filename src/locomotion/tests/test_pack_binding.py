"""Check the prepared pack entry point, modes, reward record and source tampering without Isaac."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from locomotion.spark.prepare import prepare
from locomotion.spark.sources import REQUIREMENTS_INPUT, allocation_manifest, verify_runtime

ROOT = Path(__file__).resolve().parents[3]
REMOTE = "/srv/cupi/hexapod/runs/jcc463/pack_binding_fixture"


class PackBindingTests(unittest.TestCase):
    def test_frozen_evaluation_loads_its_bound_requirements_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(
                root / "pack",
                REMOTE,
                mode="evaluate",
                num_envs=1,
                checkpoint=REMOTE + "/checkpoint.pt",
                checkpoint_sha="a" * 64,
                checkpoint_declaration_sha="b" * 64,
            )
            source = root / "pack/source"
            manifest = verify_runtime(source)
            self.assertEqual(manifest, allocation_manifest(ROOT))
            self.assertIn(REQUIREMENTS_INPUT, manifest["files"])
            args = binding["command_args"]
            self.assertEqual(
                args[args.index("--requirements") + 1], "/source/" + REQUIREMENTS_INPUT
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; "
                    "from locomotion.evaluation import requirements; "
                    "assert Path(requirements.__file__).is_relative_to(Path(sys.argv[1])); "
                    "rules, digest = requirements.load_requirements(Path(sys.argv[1]) / sys.argv[2]); "
                    "assert len(requirements.cases(rules)) == 26; print(digest)",
                    str(source),
                    REQUIREMENTS_INPUT,
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(source)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), manifest["files"][REQUIREMENTS_INPUT])
            (source / REQUIREMENTS_INPUT).write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_runtime(source)

    def test_pack_uses_one_backend_and_keeps_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(root / "pack", REMOTE, mode="diagnostic")
            self.assertNotIn("--backend", binding["command_args"])
            self.assertFalse(binding["root_review_complete"])
            self.assertEqual(binding["module"], "locomotion.scripts.diagnose")
            self.assertNotIn("--noise-probe", binding["command_args"])
            manifest = verify_runtime(root / "pack/source")
            self.assertFalse(any("/tripod/" in name for name in manifest["files"]))
            self.assertFalse((root / "pack/source/locomotion/hexapod").exists())
            nested = root / "pack/source/locomotion/robots/rs05.py"
            nested.write_text(nested.read_text() + "\n# tamper\n")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_runtime(root / "pack/source")

    def test_evaluation_pack_ships_the_waypoint_follower_without_shapely_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            from locomotion.evaluation.checkpoint import CHECKPOINT_SHA, DECLARATION_SHA

            root = Path(directory)
            prepare(
                root / "pack",
                REMOTE,
                mode="evaluate",
                num_envs=1,
                checkpoint="/srv/cupi/hexapod/runs/james/checkpoint.pt",
                checkpoint_sha=CHECKPOINT_SHA,
                checkpoint_declaration_sha=DECLARATION_SHA,
            )
            files = set(verify_runtime(root / "pack/source")["files"])
            self.assertLessEqual(
                {"navigation/waypoint.py", "navigation/producer.py", "contracts/pose.py"}, files
            )
            self.assertFalse({"navigation/grid.py", "navigation/region.py"} & files)
            init = (root / "pack/source/navigation/__init__.py").read_text()
            self.assertNotIn("from . import", init)

    def test_retired_modes_fail_before_creating_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("tripod", "throughput"):
                output = Path(directory) / mode
                with (
                    self.subTest(mode=mode),
                    self.assertRaisesRegex(ValueError, "Invalid native allocation"),
                ):
                    prepare(output, REMOTE, mode=mode)
                self.assertFalse(output.exists())

    def test_active_reward_retires_version_switches_and_requires_matching_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            for option, value in (
                ("reward_version", "2"),
                ("reward_task", "v4"),
                ("reward_options", "yaw_gate_floor=0"),
            ):
                with self.subTest(option=option), self.assertRaises(TypeError):
                    prepare(output, REMOTE, **{option: value})
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "60-control"):
                prepare(output, REMOTE, gait_clock=40)
            self.assertFalse(output.exists())
            binding = prepare(output, REMOTE)
            self.assertFalse(
                any(value.startswith("--reward-") for value in binding["command_args"])
            )
            record = json.loads((output / "PACK.json").read_text())
            self.assertEqual(
                (record["reward_version"], record["reward_task"]), ("4", "v4_yaw_gate")
            )

    def test_pack_rejects_the_removed_hybrid_bc_option(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected"
            with self.assertRaises(TypeError):
                prepare(output, REMOTE, bc_warmstart=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
