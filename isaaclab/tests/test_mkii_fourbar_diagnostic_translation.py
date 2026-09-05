"""Diagnostic-only placement and zero-motion tracing, without Isaac or GPU use."""
import argparse
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab/tests"))
from test_run_mkii_fourbar_supervisor import supervisor, CONTRACT, diagnostic_report, write_trace_fixture

spec = importlib.util.spec_from_file_location("_diagnostic_translation_tests", ROOT / "isaaclab/diagnose_mkii_fourbar.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticTranslationTests(unittest.TestCase):
    def raw(self):
        origins = torch.tensor([[0., 0., 0.], [2., 4., .5]])
        default = torch.tensor([[0., 0., .14297, 0., 0., 0., 1.]]).repeat(2, 1)
        data = SimpleNamespace(default_root_pose=default, root_pos_w=default[:, :3]+origins)
        robot = SimpleNamespace(data=data)
        robot.write_root_pose_to_sim_index = lambda root_pose, env_ids: data.root_pos_w.index_copy_(0, env_ids.long(), root_pose[:, :3])
        return SimpleNamespace(num_envs=2, _terrain=SimpleNamespace(env_origins=SimpleNamespace(torch=origins)),
                               _robot=robot, scene=SimpleNamespace(env_origins=origins.clone()))

    def actual_reset_pose(self, raw):
        path = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(path.read_text())
        reset = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
        selected = []
        for node in reset.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pose" for target in node.targets):
                selected.append(node)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name) and node.target.value.id == "pose":
                selected.append(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "write_root_pose_to_sim_index":
                selected.append(node)
        self.assertEqual(len(selected), 3)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
             {"self": raw, "env_ids": torch.arange(raw.num_envs, dtype=torch.int32), "tensor": diagnostic.tensor})

    def test_actual_terrain_translation_is_consumed_by_real_reset_code(self):
        for offset in ((0., 0.), (6., 6.), (-20., 20.)):
            raw = self.raw()
            original = raw._terrain.env_origins.torch.clone()
            placement = diagnostic.translate_reset_origins(raw, offset)
            self.assertTrue(torch.equal(raw.scene.env_origins, original))
            self.assertTrue(torch.equal(raw._terrain.env_origins.torch[:, 2], original[:, 2]))
            self.actual_reset_pose(raw)
            diagnostic.verify_reset_placement(raw, placement)
            report = {"placement": placement, "diagnostic_xy_offset_m": list(offset)}
            supervisor.validate_diagnostic_placement(report, xy_offset=offset, num_envs=2)
            self.assertTrue(placement["translation_verified"] and placement["reset_pose_verified"])
            self.assertEqual(placement["initial_reset_root_positions_m"], raw._robot.data.root_pos_w.tolist())

    def test_scene_only_or_detached_origin_mutation_cannot_pass(self):
        raw = self.raw()
        placement = diagnostic.translate_reset_origins(raw, (6., 6.))
        with self.assertRaisesRegex(ValueError, "Initial reset root"):
            diagnostic.verify_reset_placement(raw, placement)  # Root was never moved.
        self.actual_reset_pose(raw)
        raw._robot.data.root_pos_w[:, 2] += .001
        with self.assertRaises(ValueError):
            diagnostic.verify_reset_placement(raw, placement)
        class Detached:
            @property
            def torch(self):
                return torch.zeros(2, 3)
        raw._terrain.env_origins = Detached()
        with self.assertRaisesRegex(ValueError, "Actual reset origins"):
            diagnostic.translate_reset_origins(raw, (6., 6.))

    def test_standing_has_no_driven_segments_and_cli_offsets_are_bounded(self):
        self.assertEqual(diagnostic.motions("standing"), [])
        self.assertEqual(supervisor.DIAGNOSTIC_MOTION_STEPS["standing"], 0)
        args = diagnostic.parser().parse_args(["--report", "/unused/report.json", "--diagnostic-motion", "standing",
                                               "--diagnostic-xy-offset", "6", "-6"])
        self.assertEqual(args.diagnostic_xy_offset, [6., -6.])
        for value in ("nan", "inf", "-inf", "20.001", "-20.001"):
            for function in (diagnostic.bounded_xy_coordinate, supervisor.bounded_xy_coordinate):
                with self.assertRaises(argparse.ArgumentTypeError):
                    function(value)

    def test_main_translates_before_reset_and_checks_placement_before_tracing(self):
        tree = ast.parse((ROOT / "isaaclab/diagnose_mkii_fourbar.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        line = lambda name: next(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == name)
        resets = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "env" and node.func.attr == "reset"]
        self.assertEqual(len(resets), 1)
        self.assertLess(line("translate_reset_origins"), resets[0])
        self.assertLess(resets[0], line("verify_reset_placement"))
        self.assertLess(line("verify_reset_placement"), line("Trace"))

    def test_host_rejects_missing_wrong_or_forged_placement(self):
        raw = self.raw()
        placement = diagnostic.translate_reset_origins(raw, (6., 6.))
        self.actual_reset_pose(raw)
        diagnostic.verify_reset_placement(raw, placement)
        baseline = {"placement": placement, "diagnostic_xy_offset_m": [6., 6.]}
        for field, value in (("translation_verified", False), ("reset_pose_verified", 1),
                ("requested_xy_offset_m", [0., 0.]), ("position_tolerance_m", .1),
                ("original_terrain_origins_m", [[0., 0., 0.]]),
                ("initial_reset_root_positions_m", [[6., 6., 0.], [8., 10., 0.]]),
                ("actual_terrain_origins_m", [[6., 6., .001], [8., 10., .5]]),
                ("default_root_positions_m", [[float("nan"), 0., .14297], [0., 0., .14297]])):
            result = copy.deepcopy(baseline)
            result["placement"][field] = value
            with self.subTest(field=field), self.assertRaises(supervisor.Blocked):
                supervisor.validate_diagnostic_placement(result, xy_offset=(6., 6.), num_envs=2)
        with self.assertRaises(supervisor.Blocked):
            supervisor.validate_diagnostic_placement({}, xy_offset=(6., 6.), num_envs=2)

    def test_host_accepts_complete_standing_trace_and_rejects_nonzero_driven_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = diagnostic_report("standing")
            report.update(steps_requested=2, steps_completed=2)
            samples = 2 * report["numerical_recipe"]["decimation"]
            report.update(physics_substeps=samples, trace_samples=samples, force_writes=samples)
            write_trace_fixture(path.parent, report)
            args = SimpleNamespace(mode="diagnose", steps=2, num_envs=1, solver_multiplier=1,
                diagnostic_motion="standing", diagnostic_usd="revolute_v3", diagnostic_xy_offset=(0., 0.))
            path.write_text(json.dumps(report))
            result = supervisor.validate_written_report(path, args=args, contract=CONTRACT)
            self.assertTrue(result["diagnostic_complete"])
            self.assertIs(result["pass"], False)
            self.assertIs(result["simulation_training_admission"], False)
            for key in ("driven_steps_requested", "driven_steps_completed", "driven_steps"):
                bad = dict(report, **{key: 1})
                path.write_text(json.dumps(bad))
                with self.assertRaises(supervisor.Blocked):
                    supervisor.validate_written_report(path, args=args, contract=CONTRACT)

    def test_host_cli_routes_offsets_only_to_diagnosis_and_preserves_headless_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "isaaclab").mkdir(parents=True)
            for name in ("validate_mkii_fourbar.py", "diagnose_mkii_fourbar.py"):
                (source / "isaaclab" / name).write_text("# fixture")
            base = ["--source-dir", str(source), "--dry-run", "--diagnostic-xy-offset", "6", "6"]
            with patch.object(supervisor, "identity", return_value=CONTRACT), patch.object(supervisor, "command") as command, \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(supervisor.main(["diagnose", *base, "--diagnostic-motion", "standing", "--steps", "2"]), 0)
                argv = json.loads(output.getvalue())["argv"]
                index = argv.index("--diagnostic-xy-offset")
                self.assertEqual(argv[index+1:index+3], ["6.0", "6.0"])
                self.assertEqual(argv[argv.index("--viz")+1], "none")
                command.assert_not_called()
            for mode in ("validate", "train"):
                with patch.object(supervisor, "command") as command, patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        supervisor.main([mode, *base])
                    command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
