"""CPU-only admission, identity, memory and checkpoint supervisor regressions."""
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "isaaclab/deploy/run-mkii-fourbar"
loader = importlib.machinery.SourceFileLoader("_fourbar_supervisor_tests", str(LAUNCHER))
spec = importlib.util.spec_from_loader(loader.name, loader)
supervisor = importlib.util.module_from_spec(spec)
loader.exec_module(supervisor)
from tools.assets.qualify_mkii_fourbar import qualify
from hexapod_core.fourbar_v1 import numerical_recipe


CONTRACT = {"task_id": "Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0", "sha256": "fixture",
            "files": {path: "a"*64 for path in supervisor.DIAGNOSTIC_USD_PATHS.values()}}


def validation_report(multiplier=1):
    recipe = numerical_recipe(multiplier)
    return {"pass": True, "errors": [], "asset_binding": {"pass": True}, "contract": CONTRACT, "task_id": CONTRACT["task_id"],
            "solver_multiplier": multiplier, "solver_iterations": [recipe["solver_position_iterations"], recipe["solver_velocity_iterations"]],
            "numerical_recipe": recipe,
            "runtime_manifest": {"resolved_simulation": {key: value for key, value in recipe.items() if key != "recipe_id"}},
            "num_envs": 32, "steps_requested": 1000, "steps_completed": 1000,
            "reset_root_positions_m": [[0., 0., .14297]]*32,
            "driven_steps": 2400, "driven_coordinate_pass": True,
            "windows": {window: {"mean_height_m": .138, "max_applied_nm": 1.2, "max_demand_nm": 1.3}
                        for window in ("settled", "driven")}}


def diagnostic_report(motion="individuals", multiplier=1):
    result = validation_report(multiplier)
    driven_steps = supervisor.DIAGNOSTIC_MOTION_STEPS[motion]
    result.update(schema="hexapod.fourbar_diagnostic.v1", mode="diagnose", num_envs=1)
    result.update({"pass": False, "diagnostic_complete": True,
                   "simulation_training_admission": False, "hardware_admission": False,
                   "diagnostic_usd": "revolute_v3", "usd_sha256": "a"*64,
                   "usd_path_relative": supervisor.DIAGNOSTIC_USD_PATHS["revolute_v3"],
                   "diagnostic_motion": motion, "driven_steps_requested": driven_steps,
                   "driven_steps_completed": driven_steps, "driven_steps": driven_steps,
                   "physics_substeps": (1000 + driven_steps) * result["numerical_recipe"]["decimation"]})
    result.update(trace_samples=result["physics_substeps"], force_writes=result["physics_substeps"])
    result.update(diagnostic_xy_offset_m=[0., 0.], placement={
        "requested_xy_offset_m": [0., 0.], "position_tolerance_m": 1e-5,
        "translation_verified": True, "reset_pose_verified": True,
        "original_terrain_origins_m": [[0., 0., 0.]], "actual_terrain_origins_m": [[0., 0., 0.]],
        "default_root_positions_m": [[0., 0., .14297]], "initial_reset_root_positions_m": [[0., 0., .14297]]})
    return result


def write_trace_fixture(directory, report):
    shape = (report["physics_substeps"], report["num_envs"], 2)
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape}).encode() + b"\n"
    data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + bytes(shape[0]*shape[1]*shape[2]*4)
    path = directory / "trace_000.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("values.npy", data)
    report["trace_files"] = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "shape": list(shape), "first_physics_sample": 0}]


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

    def test_diagnostic_routes_to_its_script_and_retains_owned_barrier(self):
        for motion in supervisor.DIAGNOSTIC_MOTION_STEPS:
            args = SimpleNamespace(mode="diagnose", steps=200, num_envs=1, solver_multiplier=2,
                                   diagnostic_motion=motion, diagnostic_usd="planar_d6_v4")
            argv = supervisor.compose_argv(Path("/source"), Path("/output"), "owned", "nonce", args)
            self.assertIn("/workspace/hexapod/isaaclab/diagnose_mkii_fourbar.py", argv)
            for flag, value in (("--steps", "200"), ("--num_envs", "1"),
                                ("--solver-multiplier", "2"), ("--diagnostic-motion", motion),
                                ("--diagnostic-usd", "planar_d6_v4")):
                self.assertEqual(argv[argv.index(flag)+1], value)
            self.assertIn('exec "$@"', argv[argv.index("-c")+1])
            self.assertIn("/source:/workspace/hexapod:ro", argv)
            self.assertIn(f"{supervisor.OWNER_LABEL}=nonce", argv)
            self.assertNotIn("--admission", argv)
            self.assertNotIn("--checkpoint", argv)
            self.assertEqual(len([value for value in argv if value.startswith("--kit_args=")]), 1)

    def test_diagnostic_reports_require_exact_complete_nonadmitting_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            # The strict 32-env sparse prefix has its own complete host fixtures.
            for motion in set(supervisor.DIAGNOSTIC_MOTION_STEPS) - {"validation_prefix"}:
                args = SimpleNamespace(mode="diagnose", steps=1000, num_envs=1, solver_multiplier=1,
                                       diagnostic_motion=motion, diagnostic_usd="revolute_v3")
                baseline = diagnostic_report(motion)
                write_trace_fixture(path.parent, baseline)
                path.write_text(json.dumps(baseline))
                result = supervisor.validate_written_report(path, args=args, contract=CONTRACT)
                self.assertTrue(result["diagnostic_complete"])
                self.assertFalse(result["pass"])
                self.assertFalse(result["simulation_training_admission"])
                for change in ({"pass": True}, {"pass": 0}, {"diagnostic_complete": 1},
                               {"schema": "hexapod.fourbar_validation.v1"}, {"mode": "validate"},
                               {"simulation_training_admission": True}, {"hardware_admission": True},
                               {"diagnostic_usd": "planar_d6_v4"}, {"usd_sha256": "b"*64},
                               {"usd_path_relative": "/outside.usda"},
                               {"diagnostic_motion": "other"}, {"steps_completed": 999},
                               {"num_envs": True}, {"solver_multiplier": True},
                               {"driven_steps_requested": 1}, {"driven_steps_completed": 1},
                               {"physics_substeps": baseline["physics_substeps"]-1},
                               {"trace_samples": baseline["physics_substeps"]-1},
                               {"force_writes": baseline["physics_substeps"]-1},
                               {"driven_steps": 1}, {"trace_files": []},
                               {"errors": ["trace failed"]}, {"contract": {}}):
                    with self.subTest(motion=motion, change=change):
                        path.write_text(json.dumps(dict(baseline, **change)))
                        with self.assertRaises(supervisor.Blocked):
                            supervisor.validate_written_report(path, args=args, contract=CONTRACT)

    def test_diagnostic_cannot_be_used_as_validation_training_or_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for mode in ("validate", "train"):
                args = SimpleNamespace(mode=mode, steps=1000, num_envs=32, solver_multiplier=1,
                                       iterations=10)
                for claimed_pass in (False, True):
                    report = dict(diagnostic_report(), **{"pass": claimed_pass})
                    path.write_text(json.dumps(report))
                    with self.assertRaises(supervisor.Blocked):
                        supervisor.validate_written_report(path, args=args, contract=CONTRACT)
            result = qualify(diagnostic_report(), diagnostic_report(multiplier=2), CONTRACT)
            self.assertFalse(result["simulation_training_admission"])

    def test_diagnostic_cli_bounds_and_default_motion_are_checked_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "isaaclab").mkdir(parents=True)
            for filename in ("validate_mkii_fourbar.py", "diagnose_mkii_fourbar.py"):
                (source / "isaaclab" / filename).write_text("# test fixture\n")
            argv = ["diagnose", "--source-dir", str(source), "--dry-run"]
            with patch.object(supervisor, "identity", return_value=CONTRACT), \
                 patch.object(supervisor, "command") as command, \
                 patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(supervisor.main(argv), 0)
                result = json.loads(stdout.getvalue())
                self.assertEqual(result["execution"], "not_started")
                launched = result["argv"]
                self.assertEqual(launched[launched.index("--diagnostic-motion")+1], "individuals")
                self.assertEqual(launched[launched.index("--steps")+1], "1000")
                self.assertEqual(launched[launched.index("--num_envs")+1], "1")
                self.assertEqual(launched[launched.index("--diagnostic-usd")+1], "revolute_v3")
                command.assert_not_called()
            for extra in (["--timeout-seconds", "7201"], ["--timeout-seconds", "29"],
                          ["--diagnostic-motion", "unbounded"], ["--solver-multiplier", "3"],
                          ["--diagnostic-usd", "other"], ["--num-envs", "9"], ["--steps", "1001"],
                          ["--checkpoint", "/checkpoint.pt"], ["--admission", "/admission.json"]):
                with self.subTest(extra=extra), patch.object(supervisor, "identity", return_value=CONTRACT), \
                     patch.object(supervisor, "command") as command, patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        supervisor.main(argv + extra)
                    command.assert_not_called()
            for mode in ("validate", "train"):
                with patch("sys.stderr", new_callable=io.StringIO), patch.object(supervisor, "command") as command:
                    with self.assertRaises(SystemExit):
                        supervisor.main([mode, "--source-dir", str(source), "--diagnostic-usd", "revolute_v3", "--dry-run"])
                    command.assert_not_called()

    def test_diagnostic_trace_checks_reject_tampering_shape_and_path_escapes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            baseline = diagnostic_report("lf_tibia")
            write_trace_fixture(directory, baseline)
            kwargs = dict(num_envs=1, samples=baseline["physics_substeps"])
            supervisor.validate_diagnostic_traces(directory, baseline, **kwargs)
            for changes in ({"file": "../trace_000.npz"}, {"sha256": "wrong"},
                            {"first_physics_sample": 1}, {"shape": [baseline["physics_substeps"], 1, 3]},
                            {"shape": [baseline["physics_substeps"]-1, 1, 2]}):
                report = dict(baseline, trace_files=[dict(baseline["trace_files"][0], **changes)])
                with self.subTest(changes=changes), self.assertRaises(supervisor.Blocked):
                    supervisor.validate_diagnostic_traces(directory, report, **kwargs)
            (directory / "trace_000.npz").write_bytes(b"replaced bytes")
            with self.assertRaisesRegex(supervisor.Blocked, "hash mismatch"):
                supervisor.validate_diagnostic_traces(directory, baseline, **kwargs)


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


class BlockedFlockWaiterTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        proc, executable, lock = root / "proc", root / "flock", root / "gpu.lock"
        executable.write_bytes(b"test trusted flock executable")
        lock.write_bytes(b"")
        pid = 1234
        process = proc / str(pid)
        (process / "fd").mkdir(parents=True)
        (process / "task" / str(pid)).mkdir(parents=True)
        (process / "exe").symlink_to(executable)
        (process / "fd/3").symlink_to(lock)
        (process / "wchan").write_text("locks_lock_inode_wait")
        (process / "stat").write_text(f"{pid} (flock) S " + "0 "*18 + "999\n")
        (process / "task" / str(pid) / "children").write_text("")
        info = lock.stat()
        key = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
        (proc / "locks").write_text(f"7: FLOCK ADVISORY WRITE 456 {key} 0 EOF\n"
                                   f"7: -> FLOCK ADVISORY WRITE {pid} {key} 0 EOF\n")
        processes = {pid: (1, "/usr/bin/flock /opt/wx/gpu.lock python /opt/wx/nowcast_run.py")}
        return pid, processes, dict(proc_root=proc, lock_path=lock, flock_executable=executable)

    def test_proven_real_childless_shared_lock_waiter_is_not_a_running_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            pid, processes, paths = self.fixture(directory)
            evidence = supervisor.blocked_flock_waiter(pid, processes, **paths)
            self.assertEqual(evidence["pid"], pid)
            self.assertEqual(evidence["holder_pid"], 456)
            self.assertEqual(evidence["lock_inode"], paths["lock_path"].stat().st_ino)
            original = supervisor.blocked_flock_waiter
            with patch.object(supervisor, "blocked_flock_waiter", side_effect=lambda p, ps: original(p, ps, **paths)):
                records = []
                self.assertEqual(supervisor.blocking_producers(processes, set(), waiter_evidence=records), [])
                self.assertEqual(records, [evidence])

    def test_acquired_flock_and_live_children_are_never_exempted(self):
        with tempfile.TemporaryDirectory() as directory:
            pid, processes, paths = self.fixture(directory)
            child = 2345
            processes[child] = (pid, "python /opt/wx/nowcast_run.py")
            self.assertIsNone(supervisor.blocked_flock_waiter(pid, processes, **paths))
            del processes[child]
            children = paths["proc_root"] / str(pid) / "task" / str(pid) / "children"
            children.write_text(str(child))
            self.assertIsNone(supervisor.blocked_flock_waiter(pid, processes, **paths))
            children.write_text("")
            # Owning the lock is not waiting for it, even with no current child.
            locks = paths["proc_root"] / "locks"
            locks.write_text(locks.read_text().splitlines()[0].replace("456", str(pid))+"\n")
            self.assertIsNone(supervisor.blocked_flock_waiter(pid, processes, **paths))

    def test_argv_masquerade_wrong_fd_nonwaiting_and_missing_proof_fail_closed(self):
        for failure in ("exe", "fd", "wchan", "locks", "children", "stat"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(failure=failure):
                pid, processes, paths = self.fixture(directory)
                process = paths["proc_root"] / str(pid)
                if failure == "exe":
                    other = Path(directory) / "bash"
                    other.write_bytes(b"a shell masquerading as flock in argv")
                    (process / "exe").unlink()
                    (process / "exe").symlink_to(other)
                elif failure == "fd":
                    other = Path(directory) / "other.lock"
                    other.write_bytes(b"")
                    (process / "fd/3").unlink()
                    (process / "fd/3").symlink_to(other)
                elif failure == "wchan":
                    (process / "wchan").write_text("do_wait")
                elif failure == "locks":
                    (paths["proc_root"] / "locks").unlink()
                elif failure == "children":
                    (process / "task" / str(pid) / "children").unlink()
                elif failure == "stat":
                    (process / "stat").write_text("malformed")
                original = supervisor.blocked_flock_waiter
                with patch.object(supervisor, "blocked_flock_waiter", side_effect=lambda p, ps: original(p, ps, **paths)):
                    self.assertEqual(supervisor.blocking_producers(processes, set()), [pid])

    def test_waiting_on_different_inode_or_missing_holder_is_not_enough(self):
        with tempfile.TemporaryDirectory() as directory:
            pid, processes, paths = self.fixture(directory)
            locks = paths["proc_root"] / "locks"
            rows = locks.read_text().splitlines()
            locks.write_text(rows[1]+"\n")
            self.assertIsNone(supervisor.blocked_flock_waiter(pid, processes, **paths))
            rows[1] = rows[1].replace(f":{paths['lock_path'].stat().st_ino} ", ":999999999 ")
            locks.write_text("\n".join(rows)+"\n")
            self.assertIsNone(supervisor.blocked_flock_waiter(pid, processes, **paths))

    def test_independent_gpu_pid_veto_still_applies_even_to_proven_waiter(self):
        pid = 1234
        def command(argv, **kwargs):
            if argv[0] == "ps":
                return SimpleNamespace(stdout=f"{pid} 1 /usr/bin/flock /opt/wx/gpu.lock python nowcast_run.py\n", returncode=0)
            if argv[0] == "systemctl":
                return SimpleNamespace(stdout="inactive\n", returncode=3)
            if argv[:2] == ["docker", "ps"]:
                return SimpleNamespace(stdout="", returncode=0)
            if argv[0] == "nvidia-smi" and "--query-compute-apps=pid,process_name" in argv:
                return SimpleNamespace(stdout=f"{pid}, unexpected_gpu_process\n", returncode=0)
            raise AssertionError(f"Unexpected command: {argv}")
        with patch.object(supervisor, "available_memory_bytes", return_value=32*1024**3), \
             patch.object(supervisor, "blocked_flock_waiter", return_value={"pid": pid}), \
             patch.object(supervisor, "command", side_effect=command):
            with self.assertRaisesRegex(supervisor.Blocked, "Unrelated or premature GPU process"):
                supervisor.resource_gate()


class ExplicitAssetSelectionTests(unittest.TestCase):
    def runtime(self, model):
        descriptor = dict(supervisor.ASSET_BUNDLES[model])
        root = Path(descriptor["usd_path_relative"])
        files = {path: "a"*64 for path in (str(root), *(str(root.parent/name) for name in ("geometry.usdc", "kinematics.json", "manifest.json")))}
        bundle = dict(descriptor, usd_root_sha256="a"*64, kinematics_sha256="a"*64, bundle_files_sha256=files)
        return dict(descriptor, usd_root_sha256="a"*64, kinematics_sha256="a"*64, asset_bundle=bundle)

    def test_selected_model_overrides_both_environment_names_with_container_path(self):
        for mode in ("validate", "train"):
            for model in supervisor.ASSET_BUNDLES:
                args = SimpleNamespace(mode=mode, asset_model=model, num_envs=32, steps=1000,
                    solver_multiplier=1, admission=Path("/admission.json"), checkpoint=None, iterations=3)
                with patch.dict(os.environ, {"HEXAPOD_USD_PATH": "/host/arbitrary.usda",
                    "HEXAPOD_MKII_FOURBAR_USD_PATH": "/host/other.usda"}):
                    argv = supervisor.compose_argv(Path("/source"), Path("/output"), "owned", "nonce", args)
                expected = "/workspace/hexapod/" + supervisor.ASSET_BUNDLES[model]["usd_path_relative"]
                for key in ("HEXAPOD_USD_PATH", "HEXAPOD_MKII_FOURBAR_USD_PATH"):
                    self.assertEqual([value for value in argv if value.startswith(key+"=")], [key+"="+expected])
                self.assertFalse(any("/host/" in value for value in argv))
        del args.asset_model
        argv = supervisor.compose_argv(Path("/source"), Path("/output"), "owned", "nonce", args)
        self.assertIn("HEXAPOD_USD_PATH=/workspace/hexapod/"+supervisor.ASSET_BUNDLES["mkii_fourbar_v3"]["usd_path_relative"], argv)

    def test_actual_model_and_all_bundle_hashes_must_match_requested_model(self):
        runtimes = {model: self.runtime(model) for model in supervisor.ASSET_BUNDLES}
        contract = dict(CONTRACT, files={path: checksum for runtime in runtimes.values()
            for path, checksum in runtime["asset_bundle"]["bundle_files_sha256"].items()})
        for model, runtime in runtimes.items():
            self.assertEqual(supervisor.require_requested_asset({"runtime_manifest": runtime}, model, contract)["model_id"], model)
            other = next(name for name in runtimes if name != model)
            with self.assertRaisesRegex(supervisor.Blocked, "differs from the requested"):
                supervisor.require_requested_asset({"runtime_manifest": runtimes[other]}, model, contract)
            changed = json.loads(json.dumps(runtime))
            changed["usd_root_sha256"] = "b"*64
            with self.assertRaises(supervisor.Blocked):
                supervisor.require_requested_asset({"runtime_manifest": changed}, model, contract)
            changed_contract = json.loads(json.dumps(contract))
            dependency = str(Path(runtime["usd_path_relative"]).parent / "geometry.usdc")
            changed_contract["files"][dependency] = "c"*64
            with self.assertRaises(supervisor.Blocked):
                supervisor.require_requested_asset({"runtime_manifest": runtime}, model, changed_contract)

    def test_final_validation_report_cannot_ignore_host_asset_selection(self):
        model = "mkii_fourbar_v4"
        runtime = self.runtime(model)
        contract = dict(CONTRACT, files=runtime["asset_bundle"]["bundle_files_sha256"])
        result = validation_report()
        result["contract"] = contract
        result["runtime_manifest"].update(runtime)
        args = SimpleNamespace(mode="validate", steps=1000, num_envs=32, solver_multiplier=1, asset_model=model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(result))
            self.assertTrue(supervisor.validate_written_report(path, args=args, contract=contract)["pass"])
            args.asset_model = "mkii_fourbar_v3"
            with self.assertRaises(supervisor.Blocked):
                supervisor.validate_written_report(path, args=args, contract=contract)

    def test_asset_selector_is_recorded_and_diagnostic_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "isaaclab").mkdir(parents=True)
            (source / "isaaclab/validate_mkii_fourbar.py").write_text("fixture")
            argv = ["validate", "--source-dir", str(source), "--asset-model", "mkii_fourbar_v4", "--dry-run"]
            with patch.object(supervisor, "identity", return_value=CONTRACT), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(supervisor.main(argv), 0)
                self.assertEqual(json.loads(stdout.getvalue())["asset_model"], "mkii_fourbar_v4")
            with patch("sys.stderr", new_callable=io.StringIO), patch.object(supervisor, "command") as command:
                with self.assertRaises(SystemExit):
                    supervisor.main(["diagnose", "--source-dir", str(source), "--asset-model", "mkii_fourbar_v4"])
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
