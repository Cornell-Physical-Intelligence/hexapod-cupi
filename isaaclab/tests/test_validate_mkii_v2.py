"""CPU-only parser, standing gates, metrics and measured-reset geometry tests."""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packages/hexapod_core"))
spec = importlib.util.spec_from_file_location("validate_mkii_v2_subject", ROOT / "isaaclab/validate_mkii_v2.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
from hexapod_core import cad_manifest_v2 as contract


@contextlib.contextmanager
def temporary_modules(replacements):
    """Restore only mocked SDK names; never clear newly imported native modules.

    patch.dict(sys.modules) restores the *entire* dictionary on exit, which can
    remove NumPy/Torch/pxr modules while their native libraries remain loaded.
    A later import can then crash instead of producing a Python test failure.
    """
    missing = object()
    original = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in original.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def sample(**overrides):
    value = {"env_samples": 2, "joint_samples": 36, "saturated": 0,
        "height_sum": 0.276, "min_height": 0.137, "max_height": 0.139,
        "max_applied": 0.9, "max_computed": 0.9, "abs_computed_sum": 32.4,
        "vertical_squared_sum": 0.0002, "roll_pitch_squared_sum": 0.0008,
        "tilt_squared_sum": 0.0002, "nonfoot_env_steps": 0,
        "named_contacts": {"lf_tibia:shaft": 0},
        "max_computed_by_joint": {"lf_tibia_pitch": 0.9}}
    value.update(overrides)
    count = value["env_samples"]
    value.setdefault("saturated_by_env_joint", [[int(env_index*18+joint_index < value["saturated"])
        for joint_index in range(18)] for env_index in range(count)])
    value.setdefault("support_count_by_env", [6] * count)
    return value


def passing_report():
    startup = validator.Window()
    settled = validator.Window()
    for _ in range(4):
        startup.add(sample(max_computed=2.8, max_applied=1.6, saturated=12))
        settled.add(sample())
    return {"cpu_asset_integrity_pass": True, "kit_asset_integrity_pass": True, "finite_states_and_observations": True,
        "commands_held_zero": True, "joint_names": list(contract.RUNTIME_JOINT_NAMES),
        "body_count": 19, "foot_names": [f"{leg}_tibia" for leg in contract.LEG_NAMES],
        "anatomical_frame_wiring": {"pass": True, "mode": "static_vector_wiring_only"},
        "reset_geometry": {"minimum_collision_clearance_m": 0.0012},
        "steps_completed": 2, "steps_requested": 2, "terminated_count": 0, "truncated_count": 0,
        "sampling": {"mode": "every_physics_substep_after_scene_update", "physics_handles_decimation": False,
            "physics_dt_s": .005, "expected_substeps_per_policy_step": 4, "observed_substeps": 8,
            "all_policy_steps_have_expected_samples": True},
        "startup": startup.report(), "settled": settled.report()}


class ValidatorParserTests(unittest.TestCase):
    def test_defaults_and_short_probe_options(self):
        args = validator.build_parser().parse_args([])
        self.assertEqual((args.num_envs, args.steps), (32, 1000))
        validator.validate_arguments(args)
        probe = validator.build_parser().parse_args(["--num_envs", "1", "--steps", "100"])
        validator.validate_arguments(probe)
        self.assertEqual((probe.num_envs, probe.steps), (1, 100))

    def test_wrong_task_abbreviated_task_and_hydra_overrides_are_rejected(self):
        for argv in (["--task", "wrong"], ["--tas=wrong"], ["task=wrong"], ["++env.task=wrong"], ["--asset", "mock"]):
            args, rest = validator.build_parser().parse_known_args(argv)
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                validator.validate_arguments(args, rest)

    def test_invalid_lengths_and_existing_report_are_rejected(self):
        for argv in (["--steps", "1"], ["--steps", "0"], ["--num_envs", "0"]):
            with self.assertRaises(ValueError):
                validator.validate_arguments(validator.build_parser().parse_args(argv))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text("preserved\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                validator.validate_arguments(validator.build_parser().parse_args(["--report", str(path)]))
            self.assertEqual(path.read_text(), "preserved\n")

    def test_joint_identity_is_bound_to_core_contract(self):
        self.assertEqual(validator.EXPECTED_JOINTS, contract.RUNTIME_JOINT_NAMES)


class StandingMetricTests(unittest.TestCase):
    def test_accumulates_real_denominators_and_named_contact_counts(self):
        window = validator.Window()
        window.add(sample(saturated=1, nonfoot_env_steps=1, named_contacts={"lf_tibia:shaft": 1}))
        window.add(sample(height_sum=0.280, min_height=0.14, max_height=0.14,
                          max_computed_by_joint={"lf_tibia_pitch": 1.1}))
        report = window.report()
        self.assertEqual(report["steps"], 2)
        self.assertAlmostEqual(report["torque_saturation_fraction"], 1 / 72)
        self.assertAlmostEqual(report["mean_base_height_m"], 0.139)
        self.assertEqual(report["named_non_foot_contact_env_steps"]["lf_tibia:shaft"], 1)
        self.assertEqual(report["max_abs_computed_torque_by_joint_nm"]["lf_tibia_pitch"], 1.1)
        json.dumps(report, allow_nan=False)

    def test_empty_and_nonfinite_samples_cannot_masquerade_as_a_pass(self):
        empty = validator.Window().report()
        json.dumps(empty, allow_nan=False)
        report = passing_report()
        report["settled"] = empty
        self.assertTrue(validator.grade_report(report))
        with self.assertRaises(ValueError):
            validator.Window().add(sample(max_computed=math.nan))

    def test_startup_demand_is_distinct_from_settled_saturation(self):
        report = passing_report()
        self.assertEqual(validator.grade_report(report), [])
        self.assertGreater(report["startup"]["max_abs_computed_torque_nm"], 1.6)
        report["settled"]["torque_saturation_fraction"] = 0.004999
        self.assertEqual(validator.grade_report(report), [])
        report["settled"]["torque_saturation_fraction"] = 0.005
        self.assertTrue(validator.grade_report(report))

    def test_one_persistently_saturated_motor_cannot_hide_in_576_motor_average(self):
        window = validator.Window()
        for _ in range(4):
            window.add(sample(env_samples=32, joint_samples=576, saturated=1, max_computed=1.600001))
        reduced = window.report()
        self.assertLess(reduced["torque_saturation_fraction"], .005)
        self.assertEqual(reduced["torque_saturation_fraction_by_env_joint"][0][0], 1.)
        self.assertEqual(reduced["worst_env_joint_torque_saturation_fraction"], 1.)
        report = passing_report()
        report["settled"] = reduced
        self.assertTrue(any("every environment and motor" in error for error in validator.grade_report(report)))

    def test_settled_peak_is_checked_independently_of_saturation_duty(self):
        report = passing_report()
        report["settled"]["max_abs_computed_torque_nm"] = 1.60001
        self.assertEqual(validator.grade_report(report), [])
        report["settled"]["max_abs_computed_torque_nm"] = 1.600011
        self.assertTrue(any("computed torque" in error for error in validator.grade_report(report)))

    def test_zero_support_in_one_environment_at_one_substep_fails(self):
        window = validator.Window()
        for support in ([6, 6], [6, 0], [6, 6], [6, 6]):
            window.add(sample(support_count_by_env=support))
        self.assertEqual(window.report()["minimum_support_count_by_env"], [6, 0])
        report = passing_report()
        report["settled"] = window.report()
        self.assertTrue(any("loaded pad" in error for error in validator.grade_report(report)))

    def test_missing_or_policy_only_sampling_cannot_pass(self):
        for key, value in (("mode", "policy_step_only"), ("physics_handles_decimation", True),
                           ("physics_dt_s", .02), ("observed_substeps", 2),
                           ("all_policy_steps_have_expected_samples", False)):
            report = passing_report()
            report["sampling"][key] = value
            with self.subTest(key=key):
                self.assertTrue(any("200 Hz" in error for error in validator.grade_report(report)))

    def test_malformed_per_motor_and_support_samples_fail_closed(self):
        for values in ({"saturated_by_env_joint": [[0] * 17] * 2}, {"support_count_by_env": [6]},
                       {"saturated_by_env_joint": [[math.nan] * 18] * 2}, {"support_count_by_env": [6, -1]},
                       {"saturated": 1, "saturated_by_env_joint": [[0] * 18] * 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validator.Window().add(sample(**values))

    def test_every_admission_failure_remains_visible(self):
        faults = (
            ("cpu_asset_integrity_pass", False), ("kit_asset_integrity_pass", False), ("finite_states_and_observations", False),
            ("commands_held_zero", False), ("joint_names", list(reversed(contract.RUNTIME_JOINT_NAMES))),
            ("body_count", 18), ("foot_names", ["lf_tibia"] * 6),
            ("anatomical_frame_wiring", {"pass": False}),
            ("reset_geometry", {"minimum_collision_clearance_m": -0.0079}),
            ("steps_completed", 1), ("terminated_count", 1), ("truncated_count", 1),
        )
        for key, value in faults:
            report = passing_report()
            report[key] = value
            with self.subTest(key=key):
                self.assertTrue(validator.grade_report(report))
        for key, value in (("min_base_height_m", 0.05), ("max_abs_applied_torque_nm", 1.611),
                           ("torque_saturation_fraction", math.nan), ("non_foot_contact_env_steps", 1)):
            report = passing_report()
            report["settled"][key] = value
            with self.subTest(key=key):
                self.assertTrue(validator.grade_report(report))


class PhysicsSubstepTests(unittest.TestCase):
    @staticmethod
    def environment():
        import torch
        def sensor(names, foot=False):
            count = len(names)
            data = types.SimpleNamespace(net_forces_w=torch.zeros(2, count, 3))
            if foot:
                data.force_matrix_w = torch.zeros(2, 1, 1, 3)
                data.force_matrix_w[..., 2] = 10.
                data.contact_pos_w = torch.zeros(2, 1, 1, 3)
                data.pos_w = torch.zeros(2, 1, 3)
                data.quat_w = torch.tensor([1., 0., 0., 0.]).repeat(2, 1, 1)
            return types.SimpleNamespace(body_names=names, data=data,
                                         cfg=types.SimpleNamespace(update_period=.005))
        data = types.SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .138]]).repeat(2, 1),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            root_lin_vel_w=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
            joint_pos=torch.zeros(2, 18), joint_vel=torch.zeros(2, 18),
            applied_torque=torch.full((2, 18), .8), computed_torque=torch.full((2, 18), .8))
        raw = types.SimpleNamespace(num_envs=2, _physics_handles_decimation=False,
            cfg=types.SimpleNamespace(decimation=4, sim=types.SimpleNamespace(dt=.005)),
            _robot=types.SimpleNamespace(data=data), _commands=torch.zeros(2, 3),
            _terrain=types.SimpleNamespace(env_origins=torch.zeros(2, 3)),
            _base_contact_sensor=sensor(["body"]),
            _coxa_contact_sensor=sensor([f"{leg}_coxa" for leg in validator.LEGS]),
            _femur_contact_sensors=[sensor([f"{leg}_femur"]) for leg in validator.LEGS],
            _feet_contact_sensors=[sensor([f"{leg}_tibia"], True) for leg in validator.LEGS],
            shaft=torch.zeros(2, 6, dtype=torch.bool))
        def contacts():
            loaded = torch.stack([torch.linalg.norm(s.data.force_matrix_w[:, 0, 0], dim=-1) > 1.
                                  for s in raw._feet_contact_sensors], dim=1)
            return loaded & ~raw.shaft, raw.shaft.clone(), torch.zeros(2, 6)
        raw._get_foot_contact_state = contacts
        class Scene:
            def __init__(self):
                self.updates = 0
                self.after_update = lambda: None
            def update(self, dt):
                self.updates += 1
                self.after_update()
                return "updated"
        raw.scene = Scene()
        return raw

    def collector(self, raw):
        return validator.PhysicsSamples(raw, list(contract.RUNTIME_JOINT_NAMES),
                                       [f"{leg}_tibia" for leg in validator.LEGS])

    def test_hook_observes_after_each_physics_update_and_restores_method(self):
        raw = self.environment()
        observed, metadata = [], {}
        original = raw.scene.update
        with validator.PhysicsSubstepHook(raw, lambda: observed.append(raw.scene.updates), metadata) as hook:
            hook.begin_policy_step()
            for _ in range(4):
                self.assertEqual(raw.scene.update(dt=.005), "updated")
            hook.finish_policy_step()
        self.assertEqual(raw.scene.update, original)
        self.assertNotIn("update", vars(raw.scene))
        self.assertEqual(observed, [1, 2, 3, 4])
        self.assertEqual(metadata["observed_substeps"], 4)
        self.assertTrue(metadata["all_policy_steps_have_expected_samples"])

    def test_hidden_decimation_wrong_dt_missing_extra_samples_and_errors_fail_closed(self):
        for backend in (True, None):
            raw = self.environment()
            raw._physics_handles_decimation = backend
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "internally"):
                with validator.PhysicsSubstepHook(raw, lambda: None, {}):
                    self.fail("backend must not be admitted")
        for count in (0, 1, 3, 5):
            raw, metadata = self.environment(), {}
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "exactly four"):
                with validator.PhysicsSubstepHook(raw, lambda: None, metadata) as hook:
                    hook.begin_policy_step()
                    for _ in range(count):
                        raw.scene.update(.005)
                    hook.finish_policy_step()
            self.assertNotIn("update", vars(raw.scene))
            self.assertFalse(metadata["all_policy_steps_have_expected_samples"])
        raw = self.environment()
        with self.assertRaisesRegex(ValueError, "Unexpected scene update"):
            with validator.PhysicsSubstepHook(raw, lambda: None, {}) as hook:
                hook.begin_policy_step()
                raw.scene.update(.02)
        self.assertEqual(raw.scene.updates, 0)
        self.assertNotIn("update", vars(raw.scene))
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            with validator.PhysicsSubstepHook(raw, mock.Mock(side_effect=RuntimeError("observer failed")), {}) as hook:
                hook.begin_policy_step()
                raw.scene.update(.005)
        self.assertNotIn("update", vars(raw.scene))
        raw._feet_contact_sensors[0].cfg.update_period = .02
        with self.assertRaisesRegex(ValueError, "sensor update period"):
            with validator.PhysicsSubstepHook(raw, lambda: None, {}):
                pass
        self.assertNotIn("update", vars(raw.scene))

    def test_first_substep_motor_spike_and_shaft_touch_survive_quiet_policy_endpoint(self):
        raw = self.environment()
        collector = self.collector(raw)
        def dynamics():
            raw._robot.data.computed_torque[0, 0] = 1.8 if raw.scene.updates == 1 else .8
            raw.shaft[0, 0] = raw.scene.updates == 1
        raw.scene.after_update = dynamics
        with validator.PhysicsSubstepHook(raw, collector.capture, {}) as hook:
            hook.begin_policy_step()
            for _ in range(4):
                raw.scene.update(.005)
            hook.finish_policy_step()
        self.assertAlmostEqual(raw._robot.data.computed_torque.max().item(), .8, places=6)
        self.assertFalse(raw.shaft.any())
        window = validator.Window()
        for value in collector.drain():
            window.add(value)
        report = window.report()
        self.assertAlmostEqual(report["max_abs_computed_torque_nm"], 1.8, places=6)
        self.assertEqual(report["worst_env_joint_torque_saturation_fraction"], .25)
        self.assertEqual(report["non_foot_contact_env_steps"], 1)
        self.assertEqual(report["named_non_foot_contact_env_steps"]["lf_tibia:shaft"], 1)
        self.assertEqual(report["minimum_support_count_by_env"], [5, 6])

    def test_loaded_nan_contact_is_rejected_but_unloaded_nan_is_not_a_support(self):
        raw = self.environment()
        collector = self.collector(raw)
        for index in range(4):
            raw._feet_contact_sensors[0].data.contact_pos_w[0, 0, 0, 1] = math.nan if index == 1 else 0.
            collector.capture()
        self.assertEqual(raw._feet_contact_sensors[0].data.contact_pos_w[0, 0, 0, 1].item(), 0.)
        with self.assertRaisesRegex(ValueError, "nonfinite_loaded_contact_position"):
            collector.drain()
        for sensor in raw._feet_contact_sensors:
            sensor.data.force_matrix_w.zero_()
            sensor.data.contact_pos_w.fill_(math.nan)
        for _ in range(4):
            collector.capture()
        window = validator.Window()
        for value in collector.drain():
            window.add(value)
        self.assertEqual(window.report()["minimum_support_count"], 0)
        report = passing_report()
        report["settled"] = window.report()
        self.assertTrue(any("loaded pad" in error for error in validator.grade_report(report)))

    def test_nonfinite_force_sensor_pose_or_state_is_rejected_at_substep(self):
        for kind in ("force", "pose", "state"):
            raw = self.environment()
            collector = self.collector(raw)
            if kind == "force":
                raw._feet_contact_sensors[0].data.force_matrix_w[0, 0, 0, 0] = math.nan
            elif kind == "pose":
                raw._feet_contact_sensors[0].data.quat_w[0, 0, 0] = math.nan
            else:
                raw._robot.data.joint_pos[0, 0] = math.nan
            for _ in range(4):
                collector.capture()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Invalid physics-substep"):
                collector.drain()


class MeasuredResetGeometryTests(unittest.TestCase):
    def geometry(self, *, height=0.142964, positions=None):
        positions = positions or [contract.DEFAULT_JOINT_POSITIONS_BY_NAME[n] for n in contract.RUNTIME_JOINT_NAMES]
        return validator.reset_geometry_from_state(ROOT / contract.URDF_PATH,
            list(contract.RUNTIME_JOINT_NAMES), [positions], [[0., 0., height]], [[1., 0., 0., 0.]], [0.])

    def test_live_state_geometry_distinguishes_old_penetration_and_corrected_reset(self):
        corrected = self.geometry()
        self.assertAlmostEqual(corrected["minimum_collision_clearance_m"], 0.0050001113, places=9)
        self.assertLess(self.geometry(height=0.130)["minimum_collision_clearance_m"], -0.0079)
        self.assertEqual(set(corrected["minimum_pad_clearance_by_leg_m"]), set(contract.LEG_NAMES))

    def test_measured_joint_jitter_changes_the_result(self):
        defaults = [contract.DEFAULT_JOINT_POSITIONS_BY_NAME[n] for n in contract.RUNTIME_JOINT_NAMES]
        changed = list(defaults)
        changed[contract.RUNTIME_JOINT_NAMES.index("lf_femur_pitch")] += 0.03
        self.assertNotEqual(self.geometry(positions=changed)["minimum_pad_clearance_by_leg_m"]["lf"],
                            self.geometry()["minimum_pad_clearance_by_leg_m"]["lf"])

    def test_nonfinite_or_missing_measured_states_fail(self):
        with self.assertRaises(ValueError):
            self.geometry(height=math.nan)
        with self.assertRaises(ValueError):
            validator.reset_geometry_from_state(ROOT / contract.URDF_PATH, contract.RUNTIME_JOINT_NAMES,
                                                 [], [], [], [])


class OpenUsdProcessBoundaryTests(unittest.TestCase):
    def test_real_cpu_preflight_does_not_load_pxr_into_a_fresh_parent(self):
        # The parent deliberately has no site packages. Its child uses the
        # same Python binary normally, with the workspace's standalone pxr.
        program = """import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('validator', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name == 'pxr' or name.startswith('pxr.') for name in sys.modules)
report = module.run_cpu_asset_gate(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
assert not any(name == 'pxr' or name.startswith('pxr.') for name in sys.modules)
print(json.dumps({'pass': report['pass'], 'usd_version': report['usd_version']}))
"""
        usd = ROOT / "robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda"
        result = subprocess.run([sys.executable, "-S", "-c", program,
            str(ROOT / "isaaclab/validate_mkii_v2.py"), str(ROOT / contract.URDF_PATH), str(usd)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["pass"])

    def test_child_errors_and_invalid_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = Path(directory) / "checker.py"
            for program in ("print('not JSON')", "print('{\"pass\":false,\"errors\":[\"bad tensor\"]}'); raise SystemExit(1)",
                            "print('{\"pass\":true}'); raise SystemExit(1)"):
                checker.write_text(program)
                with self.subTest(program=program), self.assertRaises((ValueError, RuntimeError)):
                    validator.run_cpu_asset_gate(Path("unused.urdf"), Path("unused.usda"), checker_script=checker)

    @staticmethod
    def configuration():
        return types.SimpleNamespace(seed=0, scene=types.SimpleNamespace(num_envs=1), episode_length_s=20.,
            events=types.SimpleNamespace(base_mass=object(), physics_material=types.SimpleNamespace(params={})),
            command_frame="navigation", expected_runtime_joint_names=contract.RUNTIME_JOINT_NAMES,
            decimation=4, sim=types.SimpleNamespace(dt=.005),
            robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path="unused.usda")))

    def test_kit_asset_import_and_validation_follow_launcher_entry(self):
        import builtins
        import torch  # Load native extensions before guarding import ordering.
        events = []
        fake_gym = types.ModuleType("gymnasium")
        fake_prepare = types.ModuleType("prepare_mkii_usd")
        asset = {"pass": True, "errors": [], "usd_root_sha256": "fixture", "usd_version": [0, 26, 8]}
        fake_prepare.validate = lambda *_: events.append("kit_asset_gate") or asset
        def make(*_, **__):
            events.append("make_environment")
            raise RuntimeError("stop before simulation")
        fake_gym.make = make
        @contextlib.contextmanager
        def launch(*_):
            events.append("launcher_entered")
            try:
                yield
            finally:
                self.assertTrue(report_path.is_file(), "Report must precede non-returning Kit shutdown")
                raise SystemExit(0)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "prepare_mkii_usd" or name == "pxr" or name.startswith("pxr."):
                self.assertIn("launcher_entered", events, "pxr imported before Kit selected its ABI")
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                temporary_modules({"gymnasium": fake_gym, "prepare_mkii_usd": fake_prepare}), \
                mock.patch.object(validator, "run_cpu_asset_gate", side_effect=lambda *_: events.append("child_gate") or asset), \
                mock.patch("builtins.__import__", side_effect=guarded_import), contextlib.redirect_stdout(io.StringIO()):
            report_path = Path(directory) / "failed_startup.json"
            with self.assertRaises(SystemExit) as exit_context:
                validator.run_live(types.SimpleNamespace(num_envs=1, steps=100, report=report_path), {},
                                   lambda *_: (self.configuration(), None), launch)
            self.assertEqual(exit_context.exception.code, 0)
            saved = json.loads(report_path.read_text())
            self.assertFalse(saved["pass"])
            self.assertIn("stop before simulation", saved["errors"][0])
        self.assertEqual(events, ["child_gate", "launcher_entered", "kit_asset_gate", "make_environment"])

    def test_cpu_child_failure_is_written_as_json_without_launching_kit(self):
        import torch  # Keep native extension imports outside the temporary module map.
        fake_registration = types.ModuleType("hexapod_env.tasks.mkii_v2.register")
        fake_registration.register_mkii_v2 = lambda: None
        fake_registration.MKII_V2_FLAT_TASK_ID = validator.TASK_ID
        fake_sdk = types.ModuleType("isaaclab_tasks.utils")
        fake_sdk.add_launcher_args = lambda parser: None
        fake_sdk.setup_preset_cli = lambda parser: (parser.parse_args(), [])
        fake_sdk.resolve_task_config = lambda *_: (self.configuration(), None)
        fake_sdk.launch_simulation = mock.Mock(side_effect=AssertionError("Kit must not start after a failed preflight"))
        fake_gym = types.ModuleType("gymnasium")
        with tempfile.TemporaryDirectory() as directory, \
                temporary_modules({"hexapod_env.tasks.mkii_v2.register": fake_registration,
                                   "isaaclab_tasks.utils": fake_sdk, "gymnasium": fake_gym}), \
                mock.patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(validator, "run_cpu_asset_gate", side_effect=ValueError("CPU child rejected bad tensor")):
            report_path = Path(directory) / "failure.json"
            result = validator.main(["--num_envs", "1", "--steps", "100", "--report", str(report_path)])
            report = json.loads(report_path.read_text())
        self.assertEqual(result, 1)
        self.assertFalse(report["pass"])
        self.assertIn("CPU child rejected bad tensor", report["errors"][0])
        fake_sdk.launch_simulation.assert_not_called()

    def test_success_report_survives_exit_zero_and_is_not_written_twice(self):
        report = passing_report()
        @contextlib.contextmanager
        def nonreturning_cleanup():
            try:
                yield
            finally:
                raise SystemExit(0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "passed.json"
            with self.assertRaises(SystemExit):
                with nonreturning_cleanup():
                    validator.finalize_report(report, path)
            original = path.read_bytes()
            saved = json.loads(original)
            self.assertTrue(saved["pass"])
            self.assertTrue(saved["standing_gate_pass"])
            self.assertTrue(saved["startup_raw_rating_exceeded"])
            self.assertIn("not_granted", saved["hardware_admission"])
            self.assertNotIn("_persisted", saved)
            validator.finalize_report(report, path)
            self.assertEqual(path.read_bytes(), original)

    def test_nominal_validation_disables_mass_and_material_variation(self):
        cfg = self.configuration()
        cfg.events.physics_material.params = {"asset_cfg": "all_robot_bodies", "static_friction_range": (.7, 1.2)}
        overrides = validator.configure_nominal_dynamics(cfg)
        self.assertIsNone(cfg.events.base_mass)
        self.assertEqual(cfg.events.physics_material.params["static_friction_range"], (1., 1.))
        self.assertEqual(cfg.events.physics_material.params["dynamic_friction_range"], (1., 1.))
        self.assertEqual(cfg.events.physics_material.params["restitution_range"], (0., 0.))
        self.assertEqual(cfg.events.physics_material.params["num_buckets"], 1)
        self.assertEqual(cfg.events.physics_material.params["asset_cfg"], "all_robot_bodies")
        self.assertEqual(overrides["seed"], 0)


if __name__ == "__main__":
    unittest.main()
