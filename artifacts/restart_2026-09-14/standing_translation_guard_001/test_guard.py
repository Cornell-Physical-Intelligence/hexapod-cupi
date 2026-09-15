"""CPU-only rejection and ownership checks; never contact Spark or Docker."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("diagnostic_guard", Path(__file__).with_name("launch_guarded_diagnostic.py"))
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def fixture():
    root = "/home/orionh/HEXAPOD_runs/restart_20260914/"
    return {"schema": GUARD.SCHEMA, "root_review_complete": True, "placement": "origin",
            "source": root + "source", "host": root + "host", "native_bindings": root + "origin.json",
            "output": root + "standing_translation_origin_001", "asset": "/home/orionh/asset_001",
            "admission": "/home/orionh/admission_001", "supervisor_source": "/home/orionh/supervisor_009",
            "isaaclab": "/home/orionh/IsaacLab", "num_envs": 1, "diagnostic_only": True,
            "max_seconds": 1200, "source_freeze_sha256": "a" * 64, "host_freeze_sha256": "b" * 64,
            "native_bindings_sha256": "c" * 64, "reservation_root": str(GUARD.RESERVATION_ROOT),
            "policy_sha256": GUARD.POLICY_SHA256, "coordination_sha256": GUARD.COORDINATION_SHA256,
            **{key: False for key in GUARD.FALSE_FIELDS}}


def identity():
    return {"schema": "canonical_single_placement_diagnostic_v1", "num_envs": 1,
            "runtime_binding": {"runtime_tree_sha256": "a" * 64, "scope": "canonical_single_placement_diagnostic_only"},
            "placement": {"placement": "origin", "xy_m": [0.0, 0.0], "root": "/Robot"},
            "steps": 8000, "controls": 1000, "settle_controls": 200, "dt": .0025,
            "control_dt": .02, "substeps_per_control": 8, **{key: False for key in GUARD.FALSE_FIELDS}}


class BindingsTest(TestCase):
    def setUp(self):
        # The declared remote /home tree is regular on Spark; macOS /home is
        # itself a symlink. Preserve real checks for all local temporary trees.
        original = Path.is_symlink
        override = patch.object(Path, "is_symlink", lambda path:
                                False if Path("/home") in (path, *path.parents) else original(path))
        override.start()
        self.addCleanup(override.stop)

    def test_complete_diagnostic_fixture_is_accepted(self):
        value = fixture()
        GUARD.check_binding(value, "origin", Path(value["output"]))
        GUARD.check_identity(identity(), "origin", "a" * 64)

    def test_unfilled_and_unreviewed_binding_cannot_dispatch(self):
        for key, replacement in (("root_review_complete", False), ("source_freeze_sha256", None),
                                 ("host_freeze_sha256", "ROOT_PENDING"), ("native_bindings_sha256", "")):
            value = fixture()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                GUARD.check_binding(value, "origin", Path(value["output"]))

    def test_historical_source_and_reservation_rejected(self):
        for key, replacement in (("source_freeze_sha256", GUARD.OLD_SOURCE_FREEZE),
                                 ("policy_sha256", "dcb623201a9ba63d50865ce1bcd527eb7844c0b65b629683582c747516e7d607"),
                                 ("reservation_root", "/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001"),
                                 ("coordination_sha256", "649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f")):
            value = fixture()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                GUARD.check_binding(value, "origin", Path(value["output"]))

    def test_arm_output_and_input_overlap_rejected(self):
        value = fixture()
        for placement, output in (("xy14_4", value["output"]), ("origin", value["output"] + "_retry"),
                                  ("standing32", value["output"])):
            with self.subTest(placement=placement, output=output), self.assertRaises(ValueError):
                GUARD.check_binding(value, placement, Path(output))
        for bad_path in (value["source"] + "/results", value["host"], "/home/orionh/old_output"):
            changed = deepcopy(value)
            changed["output"] = bad_path
            with self.subTest(path=bad_path), self.assertRaises(ValueError):
                GUARD.check_binding(changed, "origin", Path(bad_path))

    def test_batch_admission_and_changed_time_rejected(self):
        changes = [(field, True) for field in GUARD.FALSE_FIELDS] + [("num_envs", 32), ("max_seconds", 1201)]
        for field, replacement in changes:
            value = fixture()
            value[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                GUARD.check_binding(value, "origin", Path(value["output"]))

    def test_native_admission_missing_false_field_and_placement_mismatch_rejected(self):
        for field in GUARD.FALSE_FIELDS:
            value = identity()
            value[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                GUARD.check_identity(value, "origin", "a" * 64)
            del value[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                GUARD.check_identity(value, "origin", "a" * 64)
        value = identity()
        value["placement"]["xy_m"] = [14.0, 4.0]
        with self.assertRaises(ValueError):
            GUARD.check_identity(value, "origin", "a" * 64)
        value = identity()
        value["schema"] = "canonical_native_standing_v1"
        with self.assertRaises(ValueError):
            GUARD.check_identity(value, "origin", "a" * 64)

    def test_changed_source_and_added_file_fail_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native.py"
            source.write_text("pass\n")
            manifest = root / "FREEZE_SHA256.json"
            manifest.write_text(json.dumps({"native.py": GUARD.sha(source)}))
            pinned = GUARD.sha(manifest)
            GUARD.verify_tree(root, pinned)
            source.write_text("raise RuntimeError('changed')\n")
            with self.assertRaises(ValueError):
                GUARD.verify_tree(root, pinned)
            source.write_text("pass\n")
            (root / "extra.py").write_text("pass\n")
            with self.assertRaises(ValueError):
                GUARD.verify_tree(root, pinned)
            with self.assertRaises(ValueError):
                GUARD.pinned_file(manifest, "d" * 64)


class OwnershipTest(TestCase):
    def test_both_locks_exclude_competing_acquisitions_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory) / name) for name in ("a", "b")]
            for name in paths:
                Path(name).touch()
            with patch.object(GUARD, "LOCKS", paths):
                with GUARD.both_locks():
                    for name in paths:
                        with open(name) as stream, self.assertRaises(BlockingIOError):
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name in paths:
                    with open(name) as stream:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_callback_install_preserves_lifecycle_and_rejects_runtime_policy_change(self):
        lifecycle = lambda *_: "unchanged owned lifecycle"
        parent = SimpleNamespace(preflight=lambda: {"preflight": True}, resources=lambda: ([], 123), run_owned=lifecycle)
        GUARD.install_reservation_checks(parent)
        self.assertIs(parent.run_owned, lifecycle)
        with patch.object(GUARD, "verify_reservation", return_value={"current": True}), \
             patch.object(GUARD, "no_live_compute", return_value={}):
            self.assertEqual(parent.preflight()["current_reservation"], {"current": True})
        with patch.object(GUARD, "verify_policy_bytes", side_effect=ValueError("policy changed")), self.assertRaises(ValueError):
            parent.resources()

    def test_unrecognized_container_cannot_be_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs").mkdir()
            name = "hexapod-reference-physics-" + "a" * 32
            (root / "jobs/standing.json").write_text(json.dumps({"container_name": name, "container_id": "b" * 64}))
            inspect = subprocess.CompletedProcess([], 0, "c" * 64 + " /" + name + " true\n", "")
            with patch.object(GUARD.subprocess, "run", return_value=inspect) as calls, self.assertRaises(ValueError):
                GUARD.cleanup_owned(root)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(calls.call_args.args[0][:2], ["docker", "inspect"])

    def test_exact_container_stop_is_followed_by_verified_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs").mkdir()
            name, identifier = "hexapod-reference-physics-" + "a" * 32, "b" * 64
            (root / "jobs/standing.json").write_text(json.dumps({"container_name": name, "container_id": identifier}))
            replies = [subprocess.CompletedProcess([], 0, identifier + " /" + name + " true\n", ""),
                       subprocess.CompletedProcess([], 0, "", ""),
                       subprocess.CompletedProcess([], 0, identifier + " /" + name + " false\n", "")]
            with patch.object(GUARD.subprocess, "run", side_effect=replies) as calls:
                report = GUARD.cleanup_owned(root)
            self.assertTrue(report["cleanup_checked"])
            self.assertEqual(calls.call_args_list[1].args[0], ["docker", "stop", "--time", "20", identifier])
            self.assertEqual(calls.call_args_list[2].args[0][-1], identifier)


if __name__ == "__main__":
    main()
