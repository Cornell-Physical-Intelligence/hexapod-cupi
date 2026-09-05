"""CPU-only admission, identity, memory and checkpoint supervisor regressions."""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tarfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "isaaclab/deploy/run-mkii-fourbar"
loader = importlib.machinery.SourceFileLoader("_fourbar_supervisor_tests", str(LAUNCHER))
spec = importlib.util.spec_from_loader(loader.name, loader)
supervisor = importlib.util.module_from_spec(spec)
loader.exec_module(supervisor)
from qualify_mkii_fourbar import qualify


CONTRACT = {"task_id": "Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0", "sha256": "fixture"}


def validation_report(multiplier=1):
    return {"pass": True, "errors": [], "contract": CONTRACT, "task_id": CONTRACT["task_id"],
            "solver_multiplier": multiplier, "solver_iterations": [32*multiplier, 4*multiplier],
            "num_envs": 32, "steps_requested": 1000, "steps_completed": 1000,
            "driven_steps": 2400, "driven_coordinate_pass": True,
            "windows": {window: {"mean_height_m": .138, "max_applied_nm": 1.2}
                        for window in ("settled", "driven")}}


class FourbarSupervisorTests(unittest.TestCase):
    def archive_fixture(self, directory):
        directory = Path(directory)
        source = directory / "source"
        source.mkdir()
        (source / "nested").mkdir()
        (source / "nested/program.py").write_bytes(b"print('exact source')\n")
        (source / "nested/program.py").chmod(0o755)
        (source / "ignored log with spaces.bin").write_bytes(bytes(range(256))*100)
        (source / ".git").mkdir()
        (source / ".git/private").write_text("must stay outside archive")
        manifest, archive = directory / "source.SHA256SUMS", directory / "source.tar.gz"
        report = supervisor.snapshot_source(source, manifest)
        return source, manifest, archive, report["sha256"]

    def test_source_archive_roundtrip_contains_every_manifest_file_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, expected = self.archive_fixture(directory)
            result = supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            expected_rows = dict(row.split("  ", 1)[::-1] for row in manifest.read_text().splitlines())
            with tarfile.open(archive, "r:gz") as bundle:
                self.assertEqual(set(bundle.getnames()), set(expected_rows))
                for member in bundle.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertFalse(Path(member.name).is_absolute())
                    content = bundle.extractfile(member).read()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected_rows[member.name])
                    self.assertEqual(content, (source / member.name).read_bytes())
                self.assertEqual(bundle.getmember("nested/program.py").mode, 0o755)
            self.assertEqual(result["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(result["files"], len(expected_rows))
            self.assertEqual(result["manifest_sha256"], expected)
            self.assertEqual(result["compressed_bytes"], archive.stat().st_size)
            # Archival remains useful after the isolated checkout is overwritten.
            (source / "nested/program.py").write_text("new probe")
            with tarfile.open(archive, "r:gz") as bundle:
                self.assertEqual(bundle.extractfile("nested/program.py").read(), b"print('exact source')\n")

    def test_source_archive_rejects_changed_bytes_and_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, expected = self.archive_fixture(directory)
            (source / "nested/program.py").write_text("different source bytes")
            with self.assertRaisesRegex(supervisor.Blocked, "changed between snapshot and archival"):
                supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".gz.partial").exists())

    def test_source_archive_refuses_manifest_mutation_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, expected = self.archive_fixture(directory)
            archive.write_bytes(b"previous evidence")
            with self.assertRaisesRegex(supervisor.Blocked, "overwrite"):
                supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            self.assertEqual(archive.read_bytes(), b"previous evidence")
            manifest.write_text(manifest.read_text()+"changed\n")
            with self.assertRaisesRegex(supervisor.Blocked, "manifest changed"):
                supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)

    def test_source_archive_rejects_relative_escape_absolute_git_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, _ = self.archive_fixture(directory)
            for relative in ("../outside", "/absolute", "nested/../../outside", ".git/private", "nested//program.py"):
                with self.subTest(relative=relative):
                    manifest.write_text(f"{'a'*64}  {relative}\n")
                    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(supervisor.Blocked, "invalid or duplicate"):
                        supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            manifest.write_text(f"{'a'*64}  nested/program.py\n"*2)
            with self.assertRaisesRegex(supervisor.Blocked, "duplicate"):
                supervisor.archive_source(source, manifest, archive,
                    expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_source_archive_rejects_file_replaced_by_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, expected = self.archive_fixture(directory)
            outside = Path(directory) / "outside"
            outside.write_bytes((source / "nested/program.py").read_bytes())
            (source / "nested/program.py").unlink()
            (source / "nested/program.py").symlink_to(outside)
            with self.assertRaisesRegex(supervisor.Blocked, "symlink escapes"):
                supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            self.assertFalse(archive.exists())

    def test_source_archive_rejects_file_replaced_by_fifo_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, expected = self.archive_fixture(directory)
            path = source / "nested/program.py"
            path.unlink()
            os.mkfifo(path)
            with self.assertRaisesRegex(supervisor.Blocked, "not a regular file"):
                supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            self.assertFalse(archive.exists())

    def test_source_archive_dereferences_internal_file_symlinks_as_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, manifest, archive, _ = self.archive_fixture(directory)
            (source / "alias.py").symlink_to("nested/program.py")
            expected = supervisor.snapshot_source(source, manifest)["sha256"]
            supervisor.archive_source(source, manifest, archive, expected_manifest_sha256=expected)
            with tarfile.open(archive, "r:gz") as bundle:
                self.assertTrue(bundle.getmember("alias.py").isfile())
                self.assertEqual(bundle.extractfile("alias.py").read(), (source / "nested/program.py").read_bytes())

    def test_source_snapshot_rejects_linebreak_paths_instead_of_ambiguous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "line\nbreak.py").write_text("ambiguous")
            with self.assertRaisesRegex(supervisor.Blocked, "line breaks"):
                supervisor.snapshot_source(source, Path(directory) / "manifest")

    def test_memory_boundary_uses_available_not_total_or_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            threshold_kib = supervisor.MIN_AVAILABLE_MEMORY_BYTES // 1024
            path.write_text(f"MemTotal: 999999999 kB\nMemFree: 1 kB\nMemAvailable: {threshold_kib} kB\n")
            self.assertEqual(supervisor.available_memory_bytes(path), 16*1024**3)
            path.write_text(f"MemTotal: 999999999 kB\nMemAvailable: {threshold_kib-1} kB\n")
            with self.assertRaisesRegex(supervisor.Blocked, "below the required 16 GiB"):
                supervisor.available_memory_bytes(path)

    def test_memory_cannot_be_missing_ambiguous_or_wrong_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            for value in ("MemFree: 999999999 kB\n", "MemAvailable: invalid kB\n",
                          "MemAvailable: 999999999 MB\n", "MemAvailable: -1 kB\n",
                          "MemAvailable: 999999999 kB\nMemAvailable: 999999999 kB\n"):
                with self.subTest(value=value):
                    path.write_text(value)
                    with self.assertRaises(supervisor.Blocked):
                        supervisor.available_memory_bytes(path)

    def test_memory_failure_prevents_any_docker_or_gpu_inspection(self):
        with patch.object(supervisor, "available_memory_bytes", side_effect=supervisor.Blocked("low memory")), \
             patch.object(supervisor, "command") as command:
            with self.assertRaisesRegex(supervisor.Blocked, "low memory"):
                supervisor.resource_gate()
            command.assert_not_called()

    def test_source_identity_is_rechecked_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda",
                "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf",
                "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json",
                "packages/robot/config.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            contract = supervisor.identity(root)
            self.assertTrue(supervisor.require_unchanged_source(root, contract, "before admission"))
            (root / "packages/robot/config.py").write_text("changed dynamics\n")
            for stage in ("before admission", "during execution"):
                with self.assertRaisesRegex(supervisor.Blocked, "identity changed"):
                    supervisor.require_unchanged_source(root, contract, stage)

    def test_report_requires_exact_workload_solver_task_and_finite_object(self):
        args = SimpleNamespace(mode="validate", steps=1000, num_envs=32, solver_multiplier=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(supervisor.Blocked):
                supervisor.validate_written_report(path, args=args, contract=CONTRACT)
            path.write_text(json.dumps(validation_report()))
            self.assertTrue(supervisor.validate_written_report(path, args=args, contract=CONTRACT)["pass"])
            for change in ({"steps_completed": 999}, {"steps_requested": 999}, {"num_envs": 1},
                           {"solver_multiplier": 2}, {"solver_iterations": [64, 8]},
                           {"task_id": "old-task"}, {"contract": {}}, {"pass": 1}, {"errors": ["failed"]}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(validation_report(), **change)))
                    with self.assertRaises(supervisor.Blocked):
                        supervisor.validate_written_report(path, args=args, contract=CONTRACT)
            for raw in ("[]", "null", "not json", '{"pass": NaN}'):
                path.write_text(raw)
                with self.assertRaises(supervisor.Blocked):
                    supervisor.validate_written_report(path, args=args, contract=CONTRACT)

    def test_training_requires_checkpoint_bytes_and_explicit_partial_pause(self):
        args = SimpleNamespace(mode="train", iterations=10, num_envs=32)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "report.json"
            checkpoint = directory / "checkpoint.pt"
            checkpoint.write_bytes(b"test checkpoint bytes, not loaded by CPU supervisor")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            sidecar = {"contract": CONTRACT, "checkpoint_sha256": digest, "next_iteration": 4}
            checkpoint.with_suffix(".pt.json").write_text(json.dumps(sidecar))
            base = {"pass": True, "errors": [], "task_id": CONTRACT["task_id"], "contract": CONTRACT,
                    "iterations_completed": 4, "iterations_requested": 10, "num_envs": 32,
                    "checkpoint_verified": True, "checkpoint_roundtrip_pass": True, "paused": True,
                    "checkpoint_sha256": digest, "next_iteration": 4}
            path.write_text(json.dumps(base))
            self.assertTrue(supervisor.validate_written_report(path, args=args, contract=CONTRACT)["paused"])
            for change in ({"paused": False}, {"iterations_completed": 0}, {"iterations_completed": True},
                           {"iterations_completed": 11}, {"iterations_requested": 100}, {"num_envs": 1},
                           {"checkpoint_roundtrip_pass": False}, {"checkpoint_sha256": "stale"},
                           {"next_iteration": 5}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(base, **change)))
                    with self.assertRaises((supervisor.Blocked, ValueError)):
                        supervisor.validate_written_report(path, args=args, contract=CONTRACT)
            path.write_text(json.dumps(base))
            checkpoint.write_bytes(b"silently replaced checkpoint")
            with self.assertRaises(ValueError):
                supervisor.validate_written_report(path, args=args, contract=CONTRACT)

    def test_compose_training_mounts_exact_checkpoint_sidecar_and_waits_at_barrier(self):
        args = SimpleNamespace(mode="train", admission=Path("/admission with spaces.json"),
                               checkpoint=Path("/previous/checkpoint.pt"), iterations=10, num_envs=32)
        argv = supervisor.compose_argv(Path("/source"), Path("/output"), "owned", "nonce", args)
        self.assertIn("/admission with spaces.json:/workspace/admission.json:ro", argv)
        self.assertIn("/previous/checkpoint.pt.json:/workspace/resume.pt.json:ro", argv)
        self.assertIn('exec "$@"', argv[argv.index("-c")+1])
        self.assertEqual(argv[argv.index("--checkpoint")+1], "/workspace/resume.pt")
        self.assertNotIn("--rm", argv)
        self.assertEqual(len([value for value in argv if value.startswith("--kit_args=")]), 1)


class FourbarQualificationTests(unittest.TestCase):
    def test_matching_complete_runs_qualify(self):
        result = qualify(validation_report(), validation_report(2), CONTRACT)
        self.assertTrue(result["pass"])
        self.assertTrue(result["simulation_training_admission"])
        self.assertEqual(result["errors"], [])
        self.assertIn("+/-0.04 rad driven qualification only", result["admission_scope"])
        self.assertIn("Self-collision is off", result["admission_scope"])
        self.assertNotIn("pass_", result)

    def test_solver_and_workload_must_match_actual_requested_configuration(self):
        for change in ({"num_envs": 64}, {"steps_completed": 1001, "steps_requested": 1001},
                       {"steps_requested": 2000}, {"solver_iterations": [32, 4]},
                       {"task_id": "serial-v2"}, {"driven_steps": 600}, {"driven_coordinate_pass": False}):
            with self.subTest(change=change):
                result = qualify(validation_report(), dict(validation_report(2), **change), CONTRACT)
                self.assertFalse(result["simulation_training_admission"])

    def test_nan_infinite_bool_and_string_metrics_cannot_evade_delta_gate(self):
        for window in ("settled", "driven"):
            for key in ("mean_height_m", "max_applied_nm"):
                for value in (float("nan"), float("inf"), True, "0.138"):
                    with self.subTest(window=window, key=key, value=value):
                        refined = validation_report(2)
                        refined["windows"][window][key] = value
                        result = qualify(validation_report(), refined, CONTRACT)
                        self.assertFalse(result["pass"])

    def test_convergence_rejects_height_or_torque_outside_explicit_bounds(self):
        for key, value in (("mean_height_m", .140), ("max_applied_nm", 1.4)):
            refined = validation_report(2)
            refined["windows"]["driven"][key] = value
            result = qualify(validation_report(), refined, CONTRACT)
            self.assertFalse(result["pass"])
            self.assertFalse(result["convergence"]["comparisons"][f"driven.{key}"]["pass"])


if __name__ == "__main__":
    unittest.main()
