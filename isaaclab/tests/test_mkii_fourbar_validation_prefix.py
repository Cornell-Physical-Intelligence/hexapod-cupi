"""Exact individual replay history, sparse full-rate traces and nonadmitting host checks."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_diagnose_mkii_fourbar import module as diagnostic, ROOT
from test_run_mkii_fourbar_supervisor import supervisor, CONTRACT, diagnostic_report


def prefix_args():
    return SimpleNamespace(mode="diagnose", diagnostic_motion="validation_prefix", num_envs=32,
                           steps=1000, diagnostic_xy_offset=(0., 0.), diagnostic_usd="physical_mimic_v5", solver_multiplier=1)


def prefix_report():
    report = diagnostic_report("validation_prefix")
    report.update(num_envs=32, diagnostic_usd="physical_mimic_v5",
                  usd_path_relative=supervisor.DIAGNOSTIC_USD_PATHS["physical_mimic_v5"],
                  active_motor_names=list(diagnostic.ACTIVE_JOINT_NAMES), episode_length_s=69.02)
    for key in ("original_terrain_origins_m", "actual_terrain_origins_m", "default_root_positions_m", "initial_reset_root_positions_m"):
        report["placement"][key] *= 32
    report["reset_root_positions_m"] = copy.deepcopy(report["placement"]["initial_reset_root_positions_m"])
    decimation = diagnostic.DECIMATION
    report.update(trace_samples=300*decimation, physics_samples_observed=2500*decimation, control_trace_samples=2500,
                  trace_coverage={"mode": "validation_prefix_sparse_v1", "control_interval_half_open": [2200, 2500],
                    "physics_interval_half_open": [2200*decimation, 2500*decimation],
                    "physical_metrics": "all physics substeps", "control_telemetry": "every control boundary",
                    "environment_indices": list(range(32))})
    for index, segment in enumerate(diagnostic.motions("validation_prefix")):
        # Include one bad response: diagnostic collection must still complete.
        values = torch.full((32,), segment["offset_rad"])
        values[11] = -segment["offset_rad"]
        diagnostic.record_phase_mean(report, segment, [values.clone() for _ in range(5)], 1000+index*50)
    return report


def traces(directory, report):
    for prefix, count, start, index_field in (("trace", report["trace_samples"], 2200*diagnostic.DECIMATION, "first_physics_sample"),
                                              ("control", 2500, 0, "first_control_step")):
        path = directory/f"{prefix}_000.npz"
        values = np.zeros((count, 32, 2), np.float32)
        np.savez_compressed(path, values=values, columns=np.array(["q/a", "qd/a"]))
        report["trace_files" if prefix == "trace" else "control_trace_files"] = [
            {"file": path.name, "sha256": diagnostic.digest(path), "shape": list(values.shape), index_field: start}]


def fake_raw():
    names = list(diagnostic.ACTIVE_JOINT_NAMES)
    body_names = [f"body_{i}" for i in range(31)]
    data = SimpleNamespace(joint_pos=torch.zeros(2, 30), joint_vel=torch.zeros(2, 30),
        body_link_pos_w=torch.zeros(2, 31, 3), body_link_quat_w=torch.zeros(2, 31, 4),
        body_link_lin_vel_w=torch.zeros(2, 31, 3), body_link_ang_vel_w=torch.zeros(2, 31, 3), root_pos_w=torch.zeros(2, 3))
    data.body_link_quat_w[..., 3] = 1.
    states = {key: torch.zeros(2, 18) for key in ("joint_pos", "joint_vel", "joint_pos_target", "joint_vel_target",
                "joint_effort_target", "computed_torque", "applied_torque")}
    sensors = {name: SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 1, 3))) for name in body_names}
    events = []
    def backend(key):
        events.append("backend")
        return getattr(data, key)
    raw = SimpleNamespace(_robot=SimpleNamespace(data=data, body_names=body_names, joint_names=names+[f"passive{i}" for i in range(12)],
            root_view=SimpleNamespace(get_dof_positions=lambda: backend("joint_pos"), get_dof_velocities=lambda: backend("joint_vel"))),
        _terrain=SimpleNamespace(env_origins=torch.zeros(2, 3)), _processed_actions=torch.zeros(2, 18),
        active_joint_names=names, _body_contact_sensors=sensors, _feet_contact_sensors=list(sensors.values())[:6],
        _motor_model=SimpleNamespace(stiffness=torch.ones(2, 18), damping=torch.ones(2, 18)), _motor_order=torch.arange(18),
        motor_state=lambda key: states[key], motor_telemetry=lambda key: torch.ones(2, 18),
        coordinates=SimpleNamespace(gather=lambda value: value[:, :18]),
        scene=SimpleNamespace(write_data_to_sim=lambda: events.append("write")))
    return raw, states, events


class ValidationPrefixTests(unittest.TestCase):
    def test_exact_validator_prefix_and_no_recovery_or_state_reset(self):
        rows = diagnostic.motions("validation_prefix")
        self.assertEqual(rows, diagnostic.motions("individuals")[:30])
        self.assertEqual(sum(row["steps"] for row in rows), 1500)
        self.assertEqual(rows[-1]["motors"], ["lr_tibia_lever_pivot"])
        self.assertEqual([(1000+i*50, row["motors"][0]) for i, row in enumerate(rows) if i >= 24],
            [(2200, "lf_tibia_lever_pivot"), (2250, "lf_tibia_lever_pivot"), (2300, "lm_tibia_lever_pivot"),
             (2350, "lm_tibia_lever_pivot"), (2400, "lr_tibia_lever_pivot"), (2450, "lr_tibia_lever_pivot")])
        tree = ast.parse((ROOT/"isaaclab/diagnose_mkii_fourbar.py").read_text())
        segment_loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == "segment")
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("reset", "reset_idx") for n in ast.walk(segment_loop)))
        self.assertIn("index >= 45", ast.unparse(segment_loop))

    def test_exact_bounds_preserve_old_diagnostics(self):
        args = prefix_args()
        diagnostic.check_workload(args)
        for key, value in (("num_envs", 8), ("num_envs", 33), ("steps", 999), ("diagnostic_xy_offset", (1., 0.)), ("diagnostic_usd", "revolute_v3")):
            bad = copy.copy(args); setattr(bad, key, value)
            with self.assertRaises(ValueError): diagnostic.check_workload(bad)
        args.diagnostic_motion = "groups"
        with self.assertRaises(ValueError): diagnostic.check_workload(args)

    def test_end_hold_means_keep_bad_environment_and_reject_incomplete_history(self):
        report = prefix_report()
        response = report["individual_motor_response_by_env"]["lm_tibia_lever_pivot"]
        self.assertEqual(response["worst_env_index"], 11)
        self.assertLess(response["minimum_rad"], 0.)
        supervisor.validate_prefix_evidence(report, args=prefix_args())
        segment = diagnostic.motions("validation_prefix")[0]
        with self.assertRaises(ValueError): diagnostic.record_phase_mean({}, segment, [torch.zeros(32)]*4, 1000)
        with self.assertRaises(ValueError): diagnostic.record_phase_mean({}, segment, [torch.full((32,), float("nan"))]*5, 1000)

    def test_phase_delta_matches_validator_float32_arithmetic_exactly(self):
        report = {"num_envs": 32}
        positive = [torch.linspace(-.887, .731, 32, dtype=torch.float32)+i*.00731 for i in range(5)]
        negative = [torch.linspace(.1983, -.3671, 32, dtype=torch.float32)-i*.002913 for i in range(5)]
        phases = diagnostic.motions("validation_prefix")[:2]
        diagnostic.record_phase_mean(report, phases[0], positive, 1000)
        diagnostic.record_phase_mean(report, phases[1], negative, 1050)
        expected = torch.stack(positive).mean(0)-torch.stack(negative).mean(0)
        result = report["individual_motor_response_by_env"][diagnostic.ACTIVE_JOINT_NAMES[0]]
        self.assertEqual(result["positive_minus_negative_rad"], expected.tolist())
        self.assertEqual(result["minimum_rad"], expected.min().item())
        self.assertEqual(result["worst_env_index"], expected.argmin().item())
        python_delta = [a-b for a, b in zip(torch.stack(positive).mean(0).tolist(), torch.stack(negative).mean(0).tolist())]
        self.assertNotEqual(python_delta, expected.tolist())  # Regression genuinely distinguishes the bug.
        bad = {"num_envs": 32}
        diagnostic.record_phase_mean(bad, phases[0], positive, 1000)
        bad["individual_phase_means"][0]["mean_joint_position_rad"].pop()
        with self.assertRaises(ValueError): diagnostic.record_phase_mean(bad, phases[1], negative, 1050)
        with self.assertRaises(ValueError): diagnostic.record_phase_mean({"num_envs": 32}, phases[0], [value[:31] for value in positive], 1000)

    def test_sparse_hook_still_observes_all_steps_and_clones_selected_buffers(self):
        raw, states, events = fake_raw()
        frames = [[(0, torch.eye(4)), (1, torch.eye(4))] for _ in range(6)]
        observed = []
        metrics = SimpleNamespace(capture=lambda: observed.append(raw._robot.data.joint_pos[0, 0].item()), frames=frames)
        replacements = {"warp": SimpleNamespace(to_torch=lambda value: value), "isaaclab.utils.math": ModuleType("isaaclab.utils.math")}
        replacements["isaaclab.utils.math"].matrix_from_quat = lambda q: torch.eye(3).expand(*q.shape[:-1], 3, 3)
        saved = {key: sys.modules.get(key) for key in replacements}
        try:
            sys.modules.update(replacements)
            with tempfile.TemporaryDirectory() as directory:
                trace = diagnostic.Trace(raw, metrics, Path(directory), (1, 3))
                with trace:
                    for index in range(4*diagnostic.DECIMATION):
                        raw._robot.data.joint_pos.fill_(index)
                        raw.scene.write_data_to_sim()
                        raw._robot.data.joint_pos.add_(.5)
                        trace.capture()
                        if (index+1) % diagnostic.DECIMATION == 0: trace.flush({"control": index//diagnostic.DECIMATION})
                self.assertEqual(len(observed), 4*diagnostic.DECIMATION)
                self.assertEqual(trace.force_writes, trace.samples)
                self.assertEqual(trace.recorded_samples, 2*diagnostic.DECIMATION)
                self.assertEqual([row["first_physics_sample"] for row in trace.files], [diagnostic.DECIMATION, 2*diagnostic.DECIMATION])
                with np.load(Path(directory)/trace.files[0]["file"]) as archive:
                    columns = archive["columns"].tolist()
                    self.assertEqual(archive["values"][0, 0, columns.index("pre_q/"+raw._robot.joint_names[0])], diagnostic.DECIMATION)
                    self.assertEqual(archive["values"][0, 0, columns.index("post_q/"+raw._robot.joint_names[0])], diagnostic.DECIMATION+.5)
                # Native reads occur only in selected samples, always after force write.
                self.assertEqual(events.count("backend"), 4*diagnostic.DECIMATION)
                for index, event in enumerate(events):
                    if event == "backend" and events[index-1] != "backend": self.assertEqual(events[index-1], "write")
        finally:
            for key, value in saved.items():
                if value is None: sys.modules.pop(key, None)
                else: sys.modules[key] = value

    def test_control_rows_keep_all_motors_contacts_and_mutated_buffer_history(self):
        raw, states, _ = fake_raw()
        with tempfile.TemporaryDirectory() as directory:
            trace = diagnostic.ControlTrace(raw, None, Path(directory))
            states["joint_pos"][1, 13] = -.2
            raw._body_contact_sensors["body_30"].data.net_forces_w[1, 0, 2] = 8.
            trace.capture()
            states["joint_pos"].fill_(99.)
            trace.capture(); trace.flush({})
            with np.load(Path(directory)/trace.files[0]["file"]) as archive:
                columns, values = archive["columns"].tolist(), archive["values"]
                self.assertAlmostEqual(float(values[0, 1, columns.index("joint_pos/lm_tibia_lever_pivot")]), -.2)
                self.assertEqual(values[1, 1, columns.index("joint_pos/lm_tibia_lever_pivot")], 99.)
                self.assertEqual(values[0, 1, columns.index("body_contact_force_w/body_30_z")], 8.)
            self.assertEqual(trace.files[0]["first_control_step"], 0)

    def test_actual_host_accepts_nonadmitting_sparse_report_and_rejects_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); report = prefix_report(); traces(directory, report)
            path = directory/"report.json"
            path.write_text(json.dumps(report))
            result = supervisor.validate_written_report(path, args=prefix_args(), contract=CONTRACT)
            self.assertFalse(result["pass"])
            changes = [("trace_samples", 40000), ("physics_samples_observed", 39999), ("control_trace_samples", 2499),
                       ("episode_length_s", 51.02), ("individual_phase_means", report["individual_phase_means"][:-1]),
                       ("simulation_training_admission", True), ("reset_root_positions_m", [[0., 1., .14297]]*32)]
            for key, value in changes:
                bad = copy.deepcopy(report); bad[key] = value; path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(supervisor.Blocked):
                    supervisor.validate_written_report(path, args=prefix_args(), contract=CONTRACT)
            for field, index_field in (("trace_files", "first_physics_sample"), ("control_trace_files", "first_control_step")):
                bad = copy.deepcopy(report); bad[field][0][index_field] += 1; path.write_text(json.dumps(bad))
                with self.assertRaises(supervisor.Blocked): supervisor.validate_written_report(path, args=prefix_args(), contract=CONTRACT)

    def test_host_cli_prefix_is_diagnostic_only_and_keeps_owned_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); (source/"isaaclab").mkdir()
            for name in ("validate_mkii_fourbar.py", "diagnose_mkii_fourbar.py"): (source/"isaaclab"/name).write_text("# fixture")
            argv = ["diagnose", "--source-dir", str(source), "--dry-run", "--diagnostic-motion", "validation_prefix",
                    "--num-envs", "32", "--steps", "1000", "--diagnostic-usd", "physical_mimic_v5"]
            with patch.object(supervisor, "identity", return_value=CONTRACT), patch.object(supervisor, "command") as command, patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(supervisor.main(argv), 0)
                self.assertIn("validation_prefix", json.loads(output.getvalue())["argv"])
                command.assert_not_called()
            for extra in (["--steps", "999"], ["--num-envs", "8"], ["--diagnostic-xy-offset", "6", "0"]):
                with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit): supervisor.main(argv+extra)


if __name__ == "__main__": unittest.main()
